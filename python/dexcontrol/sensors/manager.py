# Copyright (C) 2026 Dexmate Inc.
# SPDX-License-Identifier: Apache-2.0

"""Every sensor on the robot, reached through ``robot.sensors``.

Returns native handles and manages shared camera subscription lifetimes.
The Rust core exposes the same grouping as ``robot.sensors()``.
"""

from __future__ import annotations

import time
import math
import threading
from contextlib import contextmanager
from typing import Any, TYPE_CHECKING

from dexcontrol.exceptions import SensorNotAvailableError
if TYPE_CHECKING:
    from dexcontrol._native import (ImuSensor, Lidar2dSensor, Lidar3dSensor,
                                    UltrasonicSensor, CameraSensor, WrenchSensor,
                                    TemperatureSensor, Sensor)


class Sensors:
    """Typed and by-name access to a robot's sensors."""

    def __init__(self, robot: Any) -> None:
        self._robot = robot
        self._leases = {}
        self._lease_lock = threading.RLock()

    def names(self) -> list[str]:
        """Enabled sensor names."""
        return list(self._robot.sensor_names)

    def has_sensor(self, name: str) -> bool:
        return name in self.names()

    @staticmethod
    def _duration(value, name):
        if not math.isfinite(value) or value < 0:
            raise ValueError(f"{name} must be finite and non-negative")
        return value

    def _channel_active(self, name, channel, window):
        capabilities = set(self._robot._sensor_info(name)["capabilities"])
        state_accessors = {"imu": self.get_imu, "lidar_2d": self.get_lidar_2d,
                           "lidar_3d": self.get_lidar_3d, "ultrasonic": self.get_ultrasonic}
        if channel == "raw" and not capabilities & {"imu", "lidar_2d", "lidar_3d", "ultrasonic", "camera_stream"}:
            age = self.get_raw(name).get_age()
            return age is not None and (window is None or age <= window)
        if channel in state_accessors and channel in capabilities:
            age = state_accessors[channel](name).get_age()
            return age is not None and (window is None or age <= window)
        if "camera_stream" in capabilities:
            camera = self.get_camera(name)
            if channel not in camera.streams:
                raise ValueError(f"Unknown data channel {name}/{channel}; available: {self.channels(name)}")
            try:
                age = camera.get_age(channel)
            except SensorNotAvailableError:
                return False  # A configured, unsubscribed image stream.
            return age is not None and (window is None or age <= window)
        raise ValueError(f"Unknown data channel {name}/{channel}; available: {self.channels(name)}")

    def channels(self, name: str) -> list[str]:
        """Configured data channels of a physical device, including embedded IMUs."""
        caps = set(self._robot._sensor_info(name)["capabilities"])
        channels = sorted(caps & {"imu", "lidar_2d", "lidar_3d", "ultrasonic"})
        if "camera_stream" in caps:
            channels += list(self.get_camera(name).streams)
        return channels or ["raw"]

    def get_active_sensors(self, *, window: float | None = 1.0) -> list[str]:
        """Devices with at least one fresh data channel; None accepts any age."""
        if window is not None:
            self._duration(window, "window")
        return [name for name in self.names()
                if any(self._channel_active(name, channel, window) for channel in self.channels(name))]

    def wait_for_sensors(self, timeout: float = 5.0, *, window: float | None = 1.0) -> bool:
        """Wait for any enabled device to report a fresh data channel."""
        return self._wait(lambda: bool(self.get_active_sensors(window=window)), timeout)

    def wait_for_all_active(self, timeout: float = 5.0, *, window: float | None = 1.0) -> bool:
        """Wait for at least one fresh channel on every enabled device."""
        names = self.names()
        return self._wait(lambda: bool(names) and len(self.get_active_sensors(window=window)) == len(names), timeout)

    def wait_for_channels(self, channels: dict[str, list[str]], timeout: float = 5.0,
                          *, window: float | None = 1.0) -> bool:
        """Wait for every selected channel, e.g. {'head_camera': ['imu', 'left_rgb']}."""
        if window is not None:
            self._duration(window, "window")
        selected = [(name, channel) for name, values in channels.items() for channel in values]
        if not selected:
            raise ValueError("select at least one data channel")
        # Validate all selections before waiting, without short-circuiting on inactivity.
        for name, channel in selected:
            self._channel_active(name, channel, window)
        return self._wait(lambda: all(self._channel_active(n, c, window) for n, c in selected), timeout)

    def _wait(self, ready, timeout):
        deadline = time.monotonic() + self._duration(timeout, "timeout")
        while True:
            if ready():
                return True
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return False
            time.sleep(min(0.05, remaining))

    @contextmanager
    def subscribe_camera(self, name: str, stream: str, *, policy="latest", history=None):
        """Own a shared subscription within this manager. The last context releases it.

        Concurrent modern owners must use the same policy. Legacy camera
        getters share these subscriptions without changing their policy.
        Do not mix owned contexts with manual subscribe/unsubscribe on the
        same stream.
        """
        with self._subscribe_camera(name, stream, (policy, history)) as camera:
            yield camera

    @contextmanager
    def _subscribe_camera(self, name, stream, requested=None):
        """Share a stream; legacy readers accept its existing buffering policy."""
        key = (name, stream)
        with self._lease_lock:
            lease = self._leases.get(key)
            if lease is None:
                camera = self.get_camera(name)
                owned = True
                if requested is None:
                    try:
                        camera.stats(stream)
                    except SensorNotAvailableError:
                        requested = ("latest", None)
                    else:
                        owned = False  # A manual subscription belongs to its caller.
                if owned:
                    camera.subscribe(stream, policy=requested[0], history=requested[1])
                lease = self._leases[key] = [camera, requested, 0, owned]
            elif requested is not None and lease[1] is not None and lease[1] != requested:
                raise ValueError(f"{name}/{stream} already has an owner with a different policy")
            # A lease without a recorded policy wraps a manual subscribe():
            # its buffering belongs to that caller, and owned contexts over it
            # share whatever it is rather than claiming a policy of their own.
            lease[2] += 1
        try:
            yield lease[0]
        finally:
            with self._lease_lock:
                lease[2] -= 1
                if lease[2] == 0:
                    del self._leases[key]
                    if lease[3]:
                        lease[0].unsubscribe(stream)

    # ------------------------------------------------------------------
    # Typed access to the native handles.
    #
    # Prefixed with ``get_`` where the Rust API is simply
    # ``sensors().imu(name)``, because this object also resolves sensors *by
    # name* -- and a bare ``ultrasonic()`` method would shadow the sensor
    # actually named "ultrasonic". Rust has no such clash: there, methods and
    # sensor names do not share a namespace.
    def get_imu(self, name: str) -> ImuSensor:
        """IMU handle for a standalone sensor or an embedded camera/LiDAR IMU."""
        return self._robot._imu(name)

    def get_lidar_2d(self, name: str) -> Lidar2dSensor:
        """Native 2D lidar handle."""
        return self._robot._lidar_2d(name)

    def get_lidar_3d(self, name: str) -> Lidar3dSensor:
        """Native 3D lidar handle."""
        return self._robot._lidar_3d(name)

    def get_ultrasonic(self, name: str) -> UltrasonicSensor:
        """Native ultrasonic handle."""
        return self._robot._ultrasonic(name)

    def get_camera(self, name: str) -> CameraSensor:
        """Native camera handle."""
        return self._robot._camera(name)

    def get_wrench(self, component: str) -> WrenchSensor:
        """Native force-torque handle for the component it is mounted on."""
        return self._robot._wrench(component)

    def get_temperature(self, component: str) -> TemperatureSensor:
        """Native temperature handle for a component (``left_arm``, ``head``)
        or for a camera that declares the ``temperature`` capability
        (``head_camera``, the wrist cameras)."""
        return self._robot._temperature(component)

    def get_raw(self, name: str) -> Sensor:
        """Capability-neutral handle for a sensor with no typed accessor."""
        return self._robot._raw_sensor(name)

    # ------------------------------------------------------------------
    _LEGACY_SENSOR_HINTS = {
        "head_imu": "the head IMU belongs to the head camera: enable 'head_camera' and use "
        "robot.sensors.get_imu('head_camera')",
        "base_imu": "the chassis IMU is named 'chassis_imu'",
        "lidar": "2D lidars are named by position, e.g. 'lidar_2d_front'",
    }

    def __getitem__(self, name: str) -> Any:
        """The typed handle for one sensor, chosen from its capabilities."""
        if name not in self.names():
            raise SensorNotAvailableError(name, hint=self._unavailable_hint(name))
        capabilities = set(self._robot._sensor_info(name)["capabilities"])
        for capability, accessor in (
            ("lidar_2d", self.get_lidar_2d),
            ("lidar_3d", self.get_lidar_3d),
            ("camera_stream", self.get_camera),
            ("imu", self.get_imu),
            ("ultrasonic", self.get_ultrasonic),
        ):
            if capability in capabilities:
                return accessor(name)
        return self.get_raw(name)

    def _unavailable_hint(self, name: str) -> str:
        """Why a sensor name is unavailable: undeclared, declared but not enabled, or renamed."""
        legacy = self._LEGACY_SENSOR_HINTS.get(name)
        if legacy is not None:
            return f"Renamed since dexcontrol 0.5: {legacy}."
        try:
            info = self._robot._sensor_info(name)
        except Exception:  # noqa: BLE001 - unknown to the profile
            return (
                f"The profile does not declare a sensor named {name!r}; declared sensors are "
                f"listed by this connection's resolved_config()['sensors']."
            )
        if not info.get("enabled", False):
            return (
                f"The profile declares {name!r} but it was not enabled; pass "
                f"enable_sensors=[{name!r}, ...] to Robot()."
            )
        return ""

    def __getattr__(self, name: str) -> Any:
        if name.startswith("_"):
            raise AttributeError(name)
        try:
            return self[name]
        except SensorNotAvailableError:
            raise
        except Exception as error:  # noqa: BLE001
            raise SensorNotAvailableError(name) from error

    def __iter__(self):
        return iter(self.names())

    def __repr__(self) -> str:
        return f"Sensors({self.names()!r})"

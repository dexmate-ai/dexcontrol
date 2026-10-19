# Copyright (C) 2026 Dexmate Inc.
#
# This software is dual-licensed:
#
# 1. GNU Affero General Public License v3.0 (AGPL-3.0)
#    See LICENSE-AGPL for details
#
# 2. Commercial License
#    For commercial licensing terms, contact: contact@dexmate.ai

"""Every sensor on the robot, reached through ``robot.sensors``.

Returns the native handles directly; this holds no sensor logic of its own.
The Rust core exposes the same grouping as ``robot.sensors()``.
"""

from __future__ import annotations

import time
from typing import Any

from dexcontrol.exceptions import SensorNotAvailableError


class Sensors:
    """Typed and by-name access to a robot's sensors."""

    def __init__(self, robot: Any) -> None:
        self._robot = robot

    def names(self) -> list[str]:
        """Enabled sensor names."""
        return list(self._robot.sensor_names)

    def has_sensor(self, name: str) -> bool:
        return name in self.names()

    def get_active_sensors(self) -> list[str]:
        """Names of the sensors currently reporting data."""
        active = []
        for name in self.names():
            try:
                if self[name].is_active():
                    active.append(name)
            except Exception:  # noqa: BLE001 - inactive on any failure
                continue
        return active

    def wait_for_sensors(self, timeout: float = 5.0) -> bool:
        """Waits until at least one enabled sensor is active."""
        return self._wait(lambda active, total: bool(active), timeout)

    def wait_for_all_active(self, timeout: float = 5.0) -> bool:
        """Waits until every enabled sensor is active."""
        return self._wait(lambda active, total: total > 0 and active == total, timeout)

    def _wait(self, ready: Any, timeout: float) -> bool:
        """Polls until ``ready(active_count, total_count)`` holds.

        Unlike a single sensor's ``wait_for_active`` this cannot delegate to
        the native waiter: readiness means something different per sensor kind
        (a subscribed camera stream, a decoded IMU state), so the aggregate is
        computed here.
        """
        deadline = time.monotonic() + max(timeout, 0.0)
        total = len(self.names())
        while True:
            if ready(len(self.get_active_sensors()), total):
                return True
            if time.monotonic() >= deadline:
                return False
            time.sleep(0.05)

    # ------------------------------------------------------------------
    # Typed access to the native handles.
    #
    # Prefixed with ``get_`` where the Rust API is simply
    # ``sensors().imu(name)``, because this object also resolves sensors *by
    # name* -- and a bare ``ultrasonic()`` method would shadow the sensor
    # actually named "ultrasonic". Rust has no such clash: there, methods and
    # sensor names do not share a namespace.
    def get_imu(self, name: str):
        """IMU handle for a standalone sensor or an embedded camera/LiDAR IMU."""
        return self._robot._imu(name)

    def get_lidar_2d(self, name: str):
        """Native 2D lidar handle."""
        return self._robot._lidar_2d(name)

    def get_lidar_3d(self, name: str):
        """Native 3D lidar handle."""
        return self._robot._lidar_3d(name)

    def get_ultrasonic(self, name: str):
        """Native ultrasonic handle."""
        return self._robot._ultrasonic(name)

    def get_camera(self, name: str):
        """Native camera handle."""
        return self._robot._camera(name)

    def get_wrench(self, component: str):
        """Native force-torque handle for the component it is mounted on."""
        return self._robot._wrench(component)

    def get_temperature(self, component: str):
        """Native temperature handle for a component."""
        return self._robot._temperature(component)

    def get_raw(self, name: str):
        """Capability-neutral handle for a sensor with no typed accessor."""
        return self._robot._raw_sensor(name)

    # ------------------------------------------------------------------
    def __getitem__(self, name: str) -> Any:
        """The typed handle for one sensor, chosen from its capabilities."""
        if name not in self.names():
            raise SensorNotAvailableError(name)
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

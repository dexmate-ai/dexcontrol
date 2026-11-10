# Copyright (C) 2026 Dexmate Inc.
# SPDX-License-Identifier: Apache-2.0

"""The DexControl 0.5.x Python API, served by the native runtime.

Import ``Robot`` from here instead of ``dexcontrol`` to keep code written
against the pre-0.6 pure-Python package running::

    from dexcontrol.compat.v05 import Robot, get_robot_config

    configs = get_robot_config()
    configs.enable_sensor("head_camera")
    with Robot(configs=configs) as robot:
        robot.left_arm.set_joint_pos(target, wait_time=3.0)
        frame = robot.sensors.head_camera.get_left_rgb()

Everything here is a thin Python wrapper over the handles of one
:class:`dexcontrol.Robot`: one connection, one watchdog, one set of safety
guards. Equivalent spellings also exist as deprecated modern aliases;
legacy semantics belong to this adapter. The supported scope is listed in
``docs/MIGRATING-FROM-0.5.md``. In short:

* Reproduced: constructor and sensor-enable configuration, ``wait_time`` /
  ``exit_on_reach`` waiting (including the arm's host-side ramp),
  ``move_joint_pos``, scalar joint getters, the 0.5 state dictionary keys,
  0.5 motion-state names, sensor attribute names (``head_imu``, ``lidar``),
  camera convenience getters, wrench and temperature sub-sensors, the
  ``supported_models`` decorator.
* Refused with an error that names the replacement: ``Robot(configs=...)``
  fields other than sensor enables, ``query_hand_type``,
  ``release_brake(joints=None)``, subscription
  policies, old managed-trajectory signatures, and ``execute_trajectory(relative=True)``.
* Not restored under any flag: joint-limit clipping, automatic head homing,
  implicit all-joint brake release, or any bypass of the E-stop, freshness,
  step and limit guards. Commands that 0.5 accepted may be rejected now.
"""

from __future__ import annotations

from collections.abc import Mapping
import functools
import math
import signal
import sys
import threading
import time
import warnings
from typing import Any, Callable, Dict, List, Literal, Optional, Sequence, Union

import numpy as np

import dexcontrol
from dexcontrol import _native
from dexcontrol.exceptions import (
    ComponentError,
    ComponentNotAvailableError,
    ConfigurationError,
    DexcontrolError,
    ModelNotSupportedError,
    PluginNotAvailableError,
    RobotConnectionError,
    SensorNotAvailableError,
    ServiceUnavailableError,
)

__all__ = [
    "Robot",
    "RobotConfig",
    "get_robot_config",
    "supported_models",
    "resolve_robot_model",
    "Arm",
    "Head",
    "Torso",
    "Hand",
    "Chassis",
    "Sensors",
    "MotionHandle",
    "MultiMotionHandle",
    "MotionState",
    "DISABLE_HEARTBEAT_ENV_VAR",
    "COMM_CFG_PATH_ENV_VAR",
    # Exceptions, re-exported so `from dexcontrol.compat.v05 import X` works.
    "DexcontrolError",
    "ComponentError",
    "ComponentNotAvailableError",
    "ConfigurationError",
    "ModelNotSupportedError",
    "PluginNotAvailableError",
    "RobotConnectionError",
    "SensorNotAvailableError",
    "ServiceUnavailableError",
]

MIGRATION_GUIDE = "docs/MIGRATING-FROM-0.5.md"

#: ``dexcontrol.utils.constants.DISABLE_HEARTBEAT_ENV_VAR`` in 0.5.
DISABLE_HEARTBEAT_ENV_VAR = "DEXCONTROL_DISABLE_HEARTBEAT"
#: ``dexcontrol.utils.constants.COMM_CFG_PATH_ENV_VAR`` in 0.5. The native
#: runtime reads the DexComm configuration itself; the name is kept so old
#: scripts that import it still import.
COMM_CFG_PATH_ENV_VAR = "DEXCOMM_CONFIG_PATH"

#: The 0.5 motion-state vocabulary. The modern names are mapped onto it.
MotionState = Literal["accepted", "working", "finished", "cancelled", "error"]

_STATE_TO_V05: Dict[str, str] = {
    "pending": "accepted",
    "running": "working",
    "succeeded": "finished",
    "cancelled": "cancelled",
    "superseded": "cancelled",
    "failed": "error",
}

_ARM_MAX_VEL_CEILING = 2.8  # rad/s; 0.5 clamped configured limits to this
_ARM_DEFAULT_CONTROL_HZ = 100.0
_CHASSIS_DEFAULT_CONTROL_HZ = 50.0
_REACH_TOLERANCE = 0.05


def _unsupported(what: str, instead: str) -> NotImplementedError:
    return NotImplementedError(
        f"{what} is not available in dexcontrol >= 0.7 and is not reproduced by "
        f"dexcontrol.compat.v05: {instead}. See {MIGRATION_GUIDE}."
    )


def _wait_seconds(value: float) -> float:
    value = float(value)
    if not math.isfinite(value) or value < 0:
        raise ValueError("wait_time must be finite and non-negative")
    return value


def _control_rate(value: float) -> float:
    value = float(value)
    if not math.isfinite(value) or value <= 0:
        raise ValueError("control_hz must be a positive finite number")
    return value


def _as_array(value: Any) -> np.ndarray:
    return np.asarray(value, dtype=np.float32)


def _select(values: np.ndarray, joint_id: Union[int, Sequence[int], None]) -> Any:
    """0.5 ``_extract_joint_info``: scalar (or one row) for an int, rows for a list."""
    if joint_id is None:
        return values
    if isinstance(joint_id, (int, np.integer)):
        return values[int(joint_id)]
    return values[list(joint_id)]


def _wait_for_position(
    component: Any,
    target: np.ndarray,
    wait_time: float,
    exit_on_reach: bool,
    exit_on_reach_kwargs: Optional[Dict[str, Any]],
) -> None:
    """0.5 ``RobotJointComponent._wait_for_position``: sleep, or poll until reached."""
    if not exit_on_reach:
        time.sleep(wait_time)
        return
    kwargs = dict(exit_on_reach_kwargs or {})
    kwargs.setdefault("tolerance", _REACH_TOLERANCE)
    deadline = time.monotonic() + wait_time
    while time.monotonic() < deadline:
        if component.is_joint_pos_reached(target, **kwargs):
            return
        time.sleep(0.01)


# --------------------------------------------------------------------------
# Configuration shim
# --------------------------------------------------------------------------


class _SensorConfig:
    """One entry of ``RobotConfig.sensors``: only ``enabled`` is writable."""

    __slots__ = ("name", "_enabled", "_transport", "_alias_of")

    def __init__(self, name: str, alias_of: Optional[str] = None) -> None:
        object.__setattr__(self, "name", name)
        object.__setattr__(self, "_enabled", False)
        object.__setattr__(self, "_transport", "zenoh")
        object.__setattr__(self, "_alias_of", alias_of)

    @property
    def enabled(self) -> bool:
        return self._enabled

    @enabled.setter
    def enabled(self, value: bool) -> None:
        object.__setattr__(self, "_enabled", bool(value))

    @property
    def transport(self) -> str:
        return self._transport

    @transport.setter
    def transport(self, value: str) -> None:
        if value != "zenoh":
            raise ConfigurationError(
                f'sensors[{self.name!r}].transport = {value!r}: only "zenoh" streams are '
                f"available in the default dexcontrol build; RTC streams are not. "
                f"Remove the assignment. See {MIGRATION_GUIDE}."
            )

    def __setattr__(self, key: str, value: Any) -> None:
        if key in ("enabled", "transport"):
            object.__setattr__(self, key, value)  # routes through the properties
            return
        raise ConfigurationError(
            f"sensors[{self.name!r}].{key} cannot be set through dexcontrol.compat.v05; "
            f"only `enabled` is honoured. Sensor parameters come from the robot profile "
            f"(dexcontrol.robot_config()) or a `config_file`. See {MIGRATION_GUIDE}."
        )

    def __repr__(self) -> str:
        return f"_SensorConfig(name={self.name!r}, enabled={self._enabled})"


class _ComponentConfig:
    """One entry of ``RobotConfig.components``: read-only."""

    __slots__ = ("name",)

    def __init__(self, name: str) -> None:
        object.__setattr__(self, "name", name)

    def __setattr__(self, key: str, value: Any) -> None:
        hint = ""
        if self.name == "estop" and key == "monitoring":
            hint = (
                " For a read-only connection that performs no safety setup use "
                "dexcontrol.DiagnosticClient."
            )
        raise ConfigurationError(
            f"components[{self.name!r}].{key} cannot be set through dexcontrol.compat.v05; "
            f"component parameters come from the robot profile.{hint} See {MIGRATION_GUIDE}."
        )

    def __repr__(self) -> str:
        return f"_ComponentConfig(name={self.name!r})"


class _FrozenKeys(Mapping):
    """A dict whose key set cannot change (0.5 examples assigned new sensors)."""

    def __init__(self, kind: str, items: Dict[str, Any]) -> None:
        self._items = dict(items)
        self._kind = kind

    def __getitem__(self, key: str) -> Any:
        return self._items[key]

    def __iter__(self):
        return iter(self._items)

    def __len__(self) -> int:
        return len(self._items)

    def _reject_mutation(self, *args: Any, **kwargs: Any) -> None:
        raise ConfigurationError(f"{self._kind} entries cannot be changed. See {MIGRATION_GUIDE}.")

    clear = update = pop = popitem = setdefault = __ior__ = _reject_mutation

    def __setitem__(self, key: str, value: Any) -> None:
        raise ConfigurationError(
            f"{self._kind}[{key!r}] = ...: dexcontrol.compat.v05 cannot add or replace "
            f"{self._kind} entries; the robot profile declares them. Declared "
            f"{self._kind}: {sorted(self)}. See {MIGRATION_GUIDE}."
        )

    def __delitem__(self, key: str) -> None:
        raise ConfigurationError(f"{self._kind} entries cannot be removed. See {MIGRATION_GUIDE}.")


# 0.5 sensor names that the current profiles spell differently.
_SENSOR_ALIASES: Dict[str, str] = {"head_imu": "head_camera"}


class RobotConfig:
    """What 0.5's ``get_robot_config()`` returned, reduced to what dexcontrol used.

    Supported: ``robot_model``, ``sensors[name].enabled`` (read/write),
    ``has_sensor``, ``enable_sensor``, ``components`` (read-only names).
    Anything else raises :class:`ConfigurationError` at assignment time, before
    a connection is attempted.
    """

    __slots__ = ("profile", "robot_model", "sensors", "components")

    def __setattr__(self, key: str, value: Any) -> None:
        raise ConfigurationError(
            f"RobotConfig.{key} cannot be assigned; only sensors[name].enabled is writable. "
            f"See {MIGRATION_GUIDE}."
        )

    def __delattr__(self, key: str) -> None:
        raise ConfigurationError(f"RobotConfig.{key} cannot be removed. See {MIGRATION_GUIDE}.")

    def __init__(self, profile: str) -> None:
        resolved = _native.robot_config(profile)
        object.__setattr__(self, "profile", resolved["profile_name"])
        object.__setattr__(self, "robot_model", resolved["robot"]["model"])
        sensors: Dict[str, _SensorConfig] = {
            name: _SensorConfig(name) for name in resolved["sensors"]
        }
        for alias, target in _SENSOR_ALIASES.items():
            if target in sensors and alias not in sensors:
                sensors[alias] = _SensorConfig(alias, alias_of=target)
        object.__setattr__(self, "sensors", _FrozenKeys("sensors", sensors))
        object.__setattr__(self, "components", _FrozenKeys(
            "components", {name: _ComponentConfig(name) for name in resolved["components"]}
        ))

    def has_sensor(self, name: str) -> bool:
        return name in self.sensors

    def enable_sensor(self, name: str) -> None:
        if name not in self.sensors:
            raise ConfigurationError(
                f"unknown sensor {name!r} for profile {self.profile!r}; declared sensors: "
                f"{sorted(self.sensors)}"
            )
        self.sensors[name].enabled = True

    def enabled_sensors(self) -> List[str]:
        """Profile sensor names to pass as ``enable_sensors`` (aliases resolved)."""
        names: List[str] = []
        for name, cfg in self.sensors.items():
            if cfg.enabled:
                target = cfg._alias_of or name
                if target not in names:
                    names.append(target)
        return names

    def __repr__(self) -> str:
        return f"RobotConfig(profile={self.profile!r}, enabled_sensors={self.enabled_sensors()!r})"


def get_robot_config(variant: Optional[str] = None) -> RobotConfig:
    """0.5 ``dexcontrol.core.config.get_robot_config``.

    Resolves the profile the way :class:`dexcontrol.Robot` does
    (``ROBOT_CONFIG``, then ``ROBOT_NAME``) unless ``variant`` names one.
    """
    return RobotConfig(variant or _native.profile_from_environment())


def resolve_robot_model() -> str:
    """0.5 ``dexcontrol.utils.compat.resolve_robot_model``."""
    return _native.robot_config(_native.profile_from_environment())["robot"]["model"]


def supported_models(*models: str) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """0.5 ``dexcontrol.utils.compat.supported_models``.

    Checks the robot model resolved from the environment before the decorated
    function runs and exits with status 1 when it is not listed, as 0.5 did.
    """
    if not models:
        raise ValueError("supported_models() requires at least one model name")

    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            try:
                model = resolve_robot_model()
            except Exception as error:  # noqa: BLE001 - mirrors 0.5's exit path
                print(f"Cannot determine robot model: {error}", file=sys.stderr)
                sys.exit(1)
            if model not in models:
                print(
                    f"This script is only supported on ({', '.join(models)}). "
                    f"Detected robot model from environment: {model}",
                    file=sys.stderr,
                )
                sys.exit(1)
            return func(*args, **kwargs)

        wrapper.__supported_models__ = models  # type: ignore[attr-defined]
        return wrapper

    return decorator


# --------------------------------------------------------------------------
# Motion handles
# --------------------------------------------------------------------------


class MotionHandle:
    """0.5 ``MotionHandle`` names over a native handle."""

    def __init__(self, native: Any) -> None:
        self._native = native

    @property
    def motion_id(self) -> int:
        return self._native.motion_id

    @property
    def is_done(self) -> bool:
        return self._native.is_done

    @property
    def state(self) -> str:
        return _STATE_TO_V05.get(self._native.state, self._native.state)

    @property
    def message(self) -> str:
        return self._native.message

    def wait(self, timeout: Optional[float] = None) -> str:
        state = self._native.wait(timeout)
        return _STATE_TO_V05.get(state, state)

    def cancel(self) -> None:
        self._native.cancel()

    @property
    def native(self) -> Any:
        """The underlying :class:`dexcontrol.MotionHandle`."""
        return self._native

    def __repr__(self) -> str:
        return f"MotionHandle(motion_id={self.motion_id}, state={self.state!r})"


class MultiMotionHandle(MotionHandle):
    """0.5 ``MultiMotionHandle`` names over a native group handle."""

    @property
    def motion_id(self) -> int:  # the 0.5 group handle had no id
        raise AttributeError("MultiMotionHandle has no motion_id; see .native.handles")

    def __repr__(self) -> str:
        return f"MultiMotionHandle(state={self.state!r})"


def _fallthrough(wrapper: Any, name: str) -> Any:
    """``__getattr__`` body for the wrappers: re-run a class property so its own
    exception surfaces (``SensorNotAvailableError`` is an ``AttributeError``,
    which would otherwise land here), else defer to the native handle."""
    if name.startswith("_"):
        raise AttributeError(name)
    descriptor = getattr(type(wrapper), name, None)
    if isinstance(descriptor, property) and descriptor.fget is not None:
        return descriptor.fget(wrapper)
    return getattr(wrapper._native, name)


def _temperature_handle(robot: "Robot", component: str) -> Any:
    """The native temperature handle, or ``SensorNotAvailableError`` when the
    profile declares no temperature endpoint for that component."""
    try:
        return robot._native.sensors.get_temperature(component)
    except (ConfigurationError, ValueError) as error:
        raise SensorNotAvailableError(
            f"{component} temperature",
            hint=f"The robot profile declares no temperature endpoint for {component!r}.",
        ) from error


# --------------------------------------------------------------------------
# Joint components
# --------------------------------------------------------------------------


class _JointComponent:
    """Shared 0.5 ``RobotJointComponent`` surface over a native ``JointComponent``.

    Attributes not redefined here fall through to the native handle.
    """

    def __init__(self, native: Any, robot: "Robot") -> None:
        self._native = native
        self._robot = robot

    def __getattr__(self, name: str) -> Any:
        return _fallthrough(self, name)

    @property
    def native(self) -> Any:
        """The underlying :class:`dexcontrol.JointComponent`."""
        return self._native

    # -- getters -------------------------------------------------------------

    def get_joint_pos(self, joint_id: Union[int, Sequence[int], None] = None) -> Any:
        return _select(_as_array(self._native.get_joint_pos()), joint_id)

    def get_joint_vel(self, joint_id: Union[int, Sequence[int], None] = None) -> Any:
        return _select(_as_array(self._native.get_joint_vel()), joint_id)

    def get_joint_current(self, joint_id: Union[int, Sequence[int], None] = None) -> Any:
        return _select(_as_array(self._native.get_joint_current()), joint_id)

    def get_joint_torque(self, joint_id: Union[int, Sequence[int], None] = None) -> Any:
        return _select(_as_array(self._native.get_joint_torque()), joint_id)

    def get_joint_err(self, joint_id: Union[int, Sequence[int], None] = None) -> Any:
        return _select(np.asarray(self._native.get_joint_err(), dtype=np.uint32), joint_id)

    def get_joint_state(self, joint_id: Union[int, Sequence[int], None] = None) -> Any:
        """``(N, 3)`` of position, velocity, current (torque when no current)."""
        state = self._native.get_state()
        if "position" not in state or "velocity" not in state:
            raise ValueError("Joint positions and velocities are not available for this component.")
        third = state.get("current")
        if third is None or len(third) == 0:
            third = state.get("torque")
        if third is None or len(third) == 0:
            raise ValueError(
                f"Either current or torque should be available for this {type(self).__name__}."
            )
        stacked = np.stack(
            [_as_array(state["position"]), _as_array(state["velocity"]), _as_array(third)], axis=1
        )
        return _select(stacked, joint_id)

    def get_joint_state_dict(self, joint_id: Union[int, Sequence[int], None] = None) -> Dict[str, Any]:
        state = self.get_joint_state()
        names = list(self._native.joint_names)
        indices = np.atleast_1d(_select(np.arange(len(names)), joint_id))
        return {names[int(i)]: state[int(i)] for i in indices}

    def get_state(self) -> Dict[str, Any]:
        """The 0.5 raw state keys: ``pos``, ``vel``, ``cur``, ``torque``, ``error``."""
        modern = self._native.get_state()
        out: Dict[str, Any] = {}
        for old, new in (("pos", "position"), ("vel", "velocity"), ("cur", "current"), ("torque", "torque")):
            if new in modern and modern[new] is not None and len(modern[new]):
                out[old] = _as_array(modern[new])
        out["error"] = np.asarray(self._native.get_joint_err(), dtype=np.uint32)
        for key in ("timestamp_ns", "frame_id"):
            if key in modern:
                out[key] = modern[key]
        return out

    # -- commands ------------------------------------------------------------

    def _joint_values(self, value: Any, label: str, scalar: bool = False) -> np.ndarray:
        names = list(self._native.joint_names)
        array = _as_array(value)
        if scalar and array.ndim == 0:
            array = np.full(len(names), float(array), dtype=np.float32)
        if array.shape != (len(names),):
            raise ValueError(f"{label} requires {len(names)} joint values, got shape {array.shape}")
        if not np.all(np.isfinite(array)):
            raise ValueError(f"{label} must contain only finite values")
        return array

    def _target(self, joint_pos: Any, relative: bool) -> np.ndarray:
        names = list(self._native.joint_names)
        if isinstance(joint_pos, dict):
            unknown = sorted(set(joint_pos) - set(names))
            if unknown:
                raise ValueError(f"Unknown joint name(s) {unknown}; joints are {names}")
            base = self._joint_values(self._native.get_joint_pos(), "current position")
            target = base.copy()
            for name, value in joint_pos.items():
                if np.ndim(value) != 0 or not math.isfinite(float(value)):
                    raise ValueError("joint_pos must contain only finite scalar values")
                index = names.index(name)
                target[index] = base[index] + value if relative else value
        else:
            target = self._joint_values(joint_pos, "joint_pos")
            if relative:
                target = self._joint_values(self._native.get_joint_pos(), "current position") + target
        return self._joint_values(target, "joint_pos")

    def _velocity(self, joint_vel: Any) -> np.ndarray:
        if isinstance(joint_vel, dict):
            names = list(self._native.joint_names)
            unknown = sorted(set(joint_vel) - set(names))
            if unknown:
                raise ValueError(f"Unknown joint name(s) {unknown}; joints are {names}")
            values = np.zeros(len(names), dtype=np.float32)
            for name, value in joint_vel.items():
                if np.ndim(value) != 0 or not math.isfinite(float(value)):
                    raise ValueError("joint_vel must contain only finite scalar values")
                values[names.index(name)] = value
            return self._joint_values(values, "joint_vel")
        return self._joint_values(joint_vel, "joint_vel", scalar=True)

    def set_joint_pos(
        self,
        joint_pos: Any,
        relative: bool = False,
        wait_time: float = 0.0,
        wait_kwargs: Optional[Dict[str, float]] = None,
        exit_on_reach: bool = False,
        exit_on_reach_kwargs: Optional[Dict[str, Any]] = None,
    ) -> None:
        """0.5 ``set_joint_pos``: one setpoint, then an optional bounded wait.

        ``wait_kwargs`` was reserved in 0.5 for every component but the arm
        and is accepted and ignored here. 0.5 clipped the target to the joint
        limits; this does not, so an out-of-range target raises.
        """
        del wait_kwargs
        wait_time = _wait_seconds(wait_time)
        target = self._target(joint_pos, relative)
        self._native.set_joint_pos(target)
        if wait_time > 0:
            _wait_for_position(self._native, target, wait_time, exit_on_reach, exit_on_reach_kwargs)

    def set_joint_pos_vel(
        self,
        joint_pos: Any,
        joint_vel: Any = None,
        relative: bool = False,
        wait_time: float = 0.0,
        exit_on_reach: bool = False,
        exit_on_reach_kwargs: Optional[Dict[str, Any]] = None,
    ) -> None:
        """0.5 head/torso/hand ``set_joint_pos_vel``."""
        wait_time = _wait_seconds(wait_time)
        target = self._target(joint_pos, relative)
        if joint_vel is None:
            self._native.set_joint_pos(target)
        else:
            self._native.set_joint_pos(target, joint_vel=self._velocity(joint_vel))
        if wait_time > 0:
            _wait_for_position(self._native, target, wait_time, exit_on_reach, exit_on_reach_kwargs)

    def get_predefined_pose(self, pose_name: str) -> np.ndarray:
        return _as_array(self._native.predefined_pose(pose_name))

    @property
    def temperature_sensor(self) -> Any:
        """0.5 ``component.temperature_sensor``."""
        return _temperature_handle(self._robot, self._native.name)

    def shutdown(self) -> None:
        """Components are owned by the robot; nothing to release here."""


class _ManagedJointComponent(_JointComponent):
    """0.5 ``ManagedJointComponent``: the motion-plugin surface."""

    def move_joint_pos(
        self,
        pos: Any,
        *,
        relative: bool = False,
        velocity_scale: Any = None,
    ) -> None:
        """Fire-and-forget target: streams an untracked goal to the motion plugin, as 0.5 did."""
        self._native.stream_joint_pos(pos, relative=relative, velocity_scale=velocity_scale)

    def move_to_joint_pos(
        self,
        pos: Any,
        *,
        relative: bool = False,
        velocity_scale: Any = None,
    ) -> MotionHandle:
        return MotionHandle(
            self._native.move_to_joint_pos(pos, relative=relative, velocity_scale=velocity_scale, wait=False)
        )

    def move_joint_trajectory(self, *args: Any, **kwargs: Any) -> MotionHandle:
        raise _unsupported(
            "move_joint_trajectory(positions, time_from_start, ...)",
            "convert to (positions, time_from_start) point pairs and call component.native.move_joint_trajectory(points); "
            "velocity, acceleration, scale and tracking arguments require explicit migration",
        )

    def go_to_pose(self, pose_name: str, timeout: Optional[float] = None) -> MotionHandle:
        """Tracked move to a named pose; waits for convergence as 0.5 did."""
        return MotionHandle(self._native.go_to_pose(pose_name, wait=True, timeout=timeout))


class Arm(_ManagedJointComponent):
    """0.5 ``dexcontrol.core.arm.Arm``."""

    def set_joint_pos(
        self,
        joint_pos: Any,
        relative: bool = False,
        wait_time: float = 0.0,
        wait_kwargs: Optional[Dict[str, float]] = None,
        exit_on_reach: bool = False,
        exit_on_reach_kwargs: Optional[Dict[str, Any]] = None,
    ) -> None:
        """0.5 arm ``set_joint_pos``.

        With ``wait_time == 0`` this is one setpoint. With ``wait_time > 0`` it
        reproduces 0.5's host-side ramp: a linear trajectory from the measured
        position at ``wait_kwargs["control_hz"]`` (default 100) and
        ``wait_kwargs["max_vel"]`` (default: the joint velocity limits, at most
        2.8 rad/s), streamed through ``set_joint_pos``, then the target is
        re-published until ``wait_time`` elapses or, with ``exit_on_reach``, the
        arm is within tolerance. Every current guard applies to every step.
        """
        wait_time = _wait_seconds(wait_time)
        target = self._target(joint_pos, relative)
        if wait_time == 0:
            self._native.set_joint_pos(target)
            return
        kwargs = dict(wait_kwargs or {})
        control_hz = _control_rate(kwargs.get("control_hz", _ARM_DEFAULT_CONTROL_HZ))
        max_vel = kwargs.get("max_vel")
        if max_vel is None:
            limit = self._native.joint_vel_limit
            max_vel = (
                np.clip(_as_array(limit), 0.0, _ARM_MAX_VEL_CEILING)
                if limit is not None
                else _ARM_MAX_VEL_CEILING
            )
        max_vel = self._joint_values(max_vel, "max_vel", scalar=True)
        if np.any(max_vel <= 0):
            raise ValueError("max_vel must be positive and finite")
        reach = dict(exit_on_reach_kwargs or {})
        reach.setdefault("tolerance", _REACH_TOLERANCE)
        limits = self._native.joint_pos_limit
        if limits is not None:
            limits = np.asarray(limits)
            if np.any(target < limits[:, 0]) or np.any(target > limits[:, 1]):
                raise ValueError("joint_pos exceeds joint position limits")
        current = self._joint_values(self._native.get_joint_pos(), "current position")
        delta = target.astype(np.float64) - current
        duration = float(np.max(np.abs(delta) / max_vel))
        step_count = duration * control_hz
        if not math.isfinite(step_count):
            raise ValueError("max_vel and control_hz produce a non-finite ramp duration")
        steps = max(1, int(math.ceil(step_count)))

        limiter = dexcontrol.RateLimiter(control_hz)
        start = time.monotonic()
        # Generate only the points used within the time budget; tiny velocities
        # must not allocate an unbounded trajectory before the first command.
        for index in range(1, steps + 1):
            if time.monotonic() - start > wait_time:
                break
            self._native.set_joint_pos(current + delta * (index / steps))
            limiter.sleep()
        while time.monotonic() - start < wait_time:
            self._native.set_joint_pos(target)
            limiter.sleep()
            if exit_on_reach and self._native.is_joint_pos_reached(target, **reach):
                break

    def set_joint_pos_vel(self, joint_pos: Any, joint_vel: Any, relative: bool = False) -> None:  # type: ignore[override]
        """0.5 arm ``set_joint_pos_vel`` (no waiting arguments)."""
        target = self._target(joint_pos, relative)
        self._native.set_joint_pos(target, joint_vel=self._velocity(joint_vel))

    def release_brake(self, enable: bool, joints: Optional[Sequence[int]] = None) -> Dict[str, Any]:
        """``joints`` is required. 0.5 treated ``None`` as every joint; that is
        not reproduced (an arm can drop). Use ``release_all_brakes(enable)``."""
        if joints is None:
            raise ValueError(
                "release_brake(enable, joints=None): omitting `joints` no longer releases "
                "every brake. Pass the joint indices, or call release_all_brakes(enable) "
                f"deliberately. See {MIGRATION_GUIDE}."
            )
        return self._native.release_brake(enable, list(joints))

    @property
    def wrench_sensor(self) -> "_WrenchSensor":
        """0.5 ``arm.wrench_sensor``."""
        return _WrenchSensor(self._robot.sensors.get_wrench(self._native.name), self._native)


class Head(_ManagedJointComponent):
    """0.5 ``dexcontrol.core.head.Head``."""


class Torso(_ManagedJointComponent):
    """0.5 ``dexcontrol.core.torso.Torso``."""


class Hand(_JointComponent):
    """0.5 ``dexcontrol.core.hand.Hand`` and its subclasses (``HandF5D6``, ``DexGripper``).

    ``open_hand``/``close_hand`` keep their 0.5 arguments on the native handle
    and fall through unchanged.
    """


class Chassis:
    """0.5 ``dexcontrol.core.chassis.Chassis``."""

    def __init__(self, native: Any, robot: "Robot") -> None:
        self._native = native
        self._robot = robot

    def __getattr__(self, name: str) -> Any:
        return _fallthrough(self, name)

    @property
    def native(self) -> Any:
        return self._native

    def _hold(self, publish: Callable[[], Any], wait_time: float, wait_kwargs: Optional[Dict[str, float]]) -> None:
        """0.5 ``_execute_timed_command``: re-publish the command for ``wait_time``."""
        wait_time = _wait_seconds(wait_time)
        control_hz = _control_rate((wait_kwargs or {}).get("control_hz", _CHASSIS_DEFAULT_CONTROL_HZ))
        limiter = dexcontrol.RateLimiter(control_hz)
        publish()
        if wait_time == 0:
            return
        deadline = time.monotonic() + wait_time
        while time.monotonic() < deadline:
            publish()
            limiter.sleep()

    def set_velocity(
        self,
        vx: float,
        vy: float,
        wz: float,
        wait_time: float = 0.0,
        wait_kwargs: Optional[Dict[str, float]] = None,
    ) -> None:
        self._hold(lambda: self._native.set_velocity(vx, vy, wz), wait_time, wait_kwargs)

    def move_straight(self, speed: float, wait_time: float = 0.0, wait_kwargs: Optional[Dict[str, float]] = None) -> None:
        self.set_velocity(speed, 0.0, 0.0, wait_time, wait_kwargs)

    def move_sideways(self, speed: float, wait_time: float = 0.0, wait_kwargs: Optional[Dict[str, float]] = None) -> None:
        self.set_velocity(0.0, speed, 0.0, wait_time, wait_kwargs)

    def turn(self, angular_speed: float, wait_time: float = 0.0, wait_kwargs: Optional[Dict[str, float]] = None) -> None:
        self.set_velocity(0.0, 0.0, angular_speed, wait_time, wait_kwargs)

    def set_motion_state(
        self,
        steering_angle: Any,
        wheel_velocity: Any,
        wait_time: float = 0.0,
        wait_kwargs: Optional[Dict[str, float]] = None,
    ) -> None:
        self._hold(lambda: self._native.set_motion_state(steering_angle, wheel_velocity), wait_time, wait_kwargs)

    def set_steering_angle(self, steering_angle: Any, wait_time: float = 0.0, wait_kwargs: Optional[Dict[str, float]] = None) -> None:
        self._hold(lambda: self._native.set_steering_angle(steering_angle), wait_time, wait_kwargs)

    def set_wheel_velocity(self, wheel_velocity: Any, wait_time: float = 0.0, wait_kwargs: Optional[Dict[str, float]] = None) -> None:
        self._hold(lambda: self._native.set_wheel_velocity(wheel_velocity), wait_time, wait_kwargs)

    def get_joint_pos(self, joint_id: Union[int, Sequence[int], None] = None) -> Any:
        return _select(_as_array(self._native.get_joint_pos()), joint_id)

    def _group_temperature(self, group: str) -> Any:
        # Not reproduced: 0.5 read one temperature topic per wheel group
        # (``{group}_temperature_sub_topic``); the native temperature reader
        # exposes a single sensor per component, so there is no per-group
        # handle to return. Raising is more honest than returning the same
        # sensor for both groups.
        raise SensorNotAvailableError(
            f"chassis {group} temperature",
            hint=(
                "The profile declares the chassis temperature topics per wheel group "
                f"({group}_temperature_sub_topic) and the native temperature reader does "
                "not expose them; use robot.sensors.get_temperature(component) for the "
                "components that report one."
            ),
        )

    @property
    def steer_temperature_sensor(self) -> Any:
        """0.5 ``chassis.steer_temperature_sensor``."""
        return self._group_temperature("steer")

    @property
    def drive_temperature_sensor(self) -> Any:
        """0.5 ``chassis.drive_temperature_sensor``."""
        return self._group_temperature("drive")

    def shutdown(self) -> None:
        """Owned by the robot."""


# --------------------------------------------------------------------------
# Sensors
# --------------------------------------------------------------------------


class _WrenchSensor:
    """0.5 ``ArmWrenchSensor``: wrench plus the arm's two buttons."""

    def __init__(self, native: Any, arm: Any) -> None:
        self._native = native
        self._arm = arm

    def get_wrench_state(self) -> np.ndarray:
        return _as_array(self._native.get_wrench_state())

    def get_button_state(self) -> Dict[str, bool]:
        state = self._arm.get_button_state()
        return {"blue_button": bool(state.get("blue_button", False)), "green_button": bool(state.get("green_button", False))}

    def get_state(self) -> Dict[str, Any]:
        return dict(wrench=self.get_wrench_state(), **self.get_button_state())

    def get_blue_button_state(self) -> bool:
        return self.get_button_state()["blue_button"]

    def get_green_button_state(self) -> bool:
        return self.get_button_state()["green_button"]

    def is_active(self) -> bool:
        return self._native.is_active()

    def wait_for_active(self, timeout: float = 5.0) -> bool:
        return self._native.wait_for_active(timeout)

    def shutdown(self) -> None:
        """Owned by the robot."""


class _Imu:
    """0.5 IMU surface. ``legacy_keys`` selects the ZED spelling (``ang_vel``)."""

    def __init__(self, native: Any, legacy_keys: bool) -> None:
        self._native = native
        self._legacy = legacy_keys

    def __getattr__(self, name: str) -> Any:
        if name.startswith("_"):
            raise AttributeError(name)
        return getattr(self._native, name)

    def get_obs(self, obs_keys: Optional[List[str]] = None) -> Optional[Dict[str, Any]]:
        if obs_keys is None:
            obs_keys = ["ang_vel", "acc", "quat"] if self._legacy else ["gyro", "acc", "quat"]
        modern = self._native.get_obs()
        if modern is None:
            return None
        out: Dict[str, Any] = {"timestamp_ns": modern.get("timestamp_ns")}
        for key in obs_keys:
            source = "gyro" if key in ("ang_vel", "gyro") else key
            if source in modern:
                out[key] = modern[source]
        return out

    def shutdown(self) -> None:
        """Owned by the robot."""


class _Camera:
    """0.5 ``ZedCameraSensor`` / ``ZedXOneCameraSensor`` / ``USBCameraSensor`` getters.

    A getter acquires a shared subscription on its first read. Closing this
    view releases its ownership; other legacy views and modern subscription
    contexts keep the stream alive. Manually subscribed streams remain owned
    by their caller.
    Getters return NumPy arrays, or ``None`` until the first frame arrives.
    ``include_timestamp=True`` returns ``{"data": array, "timestamp": ns}``
    using the native frame's timestamp_ns, as in the 0.5 Zenoh form.
    """

    def __init__(self, native: Any, manager: Any, name: str) -> None:
        self._native = native
        self._manager = manager
        self._name = name
        self._subscriptions: Dict[str, Any] = {}
        self._camera_info_cache: Optional[Dict[str, Any]] = None
        self._closed = False
        self._lock = threading.Lock()

    def __getattr__(self, name: str) -> Any:
        if name.startswith("_"):
            raise AttributeError(name)
        if self._closed:
            raise RuntimeError("Compatibility camera is closed")
        return getattr(self._native, name)

    @property
    def native(self) -> Any:
        return self._native

    @property
    def available_streams(self) -> List[str]:
        return list(self._native.streams)

    @property
    def active_streams(self) -> List[str]:
        return [s for s in self.available_streams if self._subscribed(s)]

    def _subscribed(self, stream: str) -> bool:
        try:
            self._native.stats(stream)
        except SensorNotAvailableError:  # stats raises until subscribed
            return False
        return True

    def _ensure(self, stream: str) -> None:
        with self._lock:
            if self._closed:
                raise RuntimeError("Compatibility camera is closed")
            if stream not in self._subscriptions:
                subscription = self._manager._subscribe_camera(self._name, stream)
                subscription.__enter__()
                self._subscriptions[stream] = subscription

    def _frame(self, stream: str, include_timestamp: bool) -> Any:
        if stream not in self.available_streams:
            return None
        self._ensure(stream)
        frame = self._native.latest_frame(stream)
        if frame is None:
            return None
        return {"data": frame["data"], "timestamp": frame["timestamp_ns"]} if include_timestamp else frame["data"]

    def get_obs(self, obs_keys: Optional[List[str]] = None, include_timestamp: bool = False) -> Any:
        streams = self.available_streams
        if streams == ["rgb"]:
            # Mono cameras accepted include_timestamp as the first positional argument.
            if isinstance(obs_keys, bool):
                include_timestamp, obs_keys = obs_keys, None
            if obs_keys is None:
                return self._frame("rgb", include_timestamp)
        return {key: self._frame(key, include_timestamp) for key in (obs_keys or streams)}

    def get_rgb(self) -> Any:
        return self._frame("rgb", False)

    def get_left_rgb(self) -> Any:
        return self._frame("left_rgb", False)

    def get_right_rgb(self) -> Any:
        return self._frame("right_rgb", False)

    def get_depth(self) -> Any:
        return self._frame("depth", False)

    def _dims(self) -> Any:
        shapes = {}
        for stream in self.active_streams:
            shape = self._native.frame_shape(stream)
            shapes[stream] = None if shape is None else (int(shape[0]), int(shape[1]))
        if self.available_streams == ["rgb"]:
            return shapes.get("rgb")
        return shapes

    @property
    def resolution(self) -> Any:
        """``(height, width)`` per stream once a frame has arrived; ``None`` before.

        This property uses received frame dimensions. ``get_camera_info``
        queries server metadata and calibration separately.
        """
        return self._dims()

    @property
    def height(self) -> Any:
        dims = self._dims()
        if isinstance(dims, dict):
            return {k: (v[0] if v else None) for k, v in dims.items()}
        return dims[0] if dims else None

    @property
    def width(self) -> Any:
        dims = self._dims()
        if isinstance(dims, dict):
            return {k: (v[1] if v else None) for k, v in dims.items()}
        return dims[1] if dims else None

    def is_active(self) -> bool:
        """0.5: any subscribed stream is receiving frames."""
        return any(self._native.is_active(s) for s in self.active_streams)

    def is_stream_active(self, stream_name: str) -> bool:
        return self._subscribed(stream_name) and bool(self._native.is_active(stream_name))

    def wait_for_active(self, timeout: float = 5.0, require_all: bool = False) -> bool:
        """0.5 ``BaseCameraSensor.wait_for_active``: waits on the streams this
        wrapper (or the caller) has subscribed; subscribes every stream first
        when none is, as 0.5's enabled streams were all live."""
        streams = self.active_streams
        if not streams:
            for stream in self.available_streams:
                self._ensure(stream)
            streams = self.active_streams
        if not streams:
            return True
        return bool(self._native.wait_for_active(streams, timeout=timeout, require_all=require_all))

    def get_camera_info(self, force_refresh: bool = False) -> Optional[Dict[str, Any]]:
        """0.5 cached server metadata; None when the information is unavailable."""
        with self._lock:
            if self._closed:
                raise RuntimeError("Compatibility camera is closed")
            if not force_refresh and self._camera_info_cache is not None:
                return self._camera_info_cache
            try:
                info = self._native.get_camera_info(timeout=2.0)
            except DexcontrolError:
                return None
            self._camera_info_cache = info
            return info

    def shutdown(self) -> None:
        """Release this view's camera leases without disrupting other owners."""
        with self._lock:
            self._closed = True
            subscriptions, self._subscriptions = self._subscriptions, {}
        for subscription in subscriptions.values():
            try:
                subscription.__exit__(None, None, None)
            except Exception:  # noqa: BLE001 - robot may already be closed
                pass


class Sensors:
    """0.5 ``robot.sensors``: attribute access by the 0.5 names."""

    def __init__(self, robot: "Robot") -> None:
        self._robot = robot
        self._cache: Dict[str, Any] = {}
        self._closed = False

    def _require_open(self) -> None:
        if self._closed or self._robot.is_shutdown():
            raise RuntimeError("Compatibility sensor view is closed")

    def _modern(self) -> Any:
        self._require_open()
        return self._robot._native.sensors

    def _names(self) -> List[str]:
        self._require_open()
        return list(self._robot._native.sensor_names)

    def _by_capability(self, capability: str, old_name: str) -> str:
        matches = [
            n for n in self._names() if capability in self._robot._native._sensor_info(n)["capabilities"]
        ]
        if len(matches) == 1:
            return matches[0]
        if not matches:
            raise SensorNotAvailableError(old_name)
        raise SensorNotAvailableError(
            f"{old_name} (ambiguous: {matches}; use robot.sensors.<name> with one of them)"
        )

    def _resolve(self, name: str) -> Any:
        modern = self._modern()
        if name == "head_imu":
            return _Imu(modern.get_imu(_SENSOR_ALIASES["head_imu"]), legacy_keys=True)
        if name == "lidar":
            return modern.get_lidar_2d(self._by_capability("lidar_2d", name))
        if name == "base_imu":
            if "chassis_imu" not in self._names():
                raise SensorNotAvailableError(name, hint="Enable 'chassis_imu'; camera and LiDAR IMUs are not base IMUs.")
            return _Imu(modern.get_imu("chassis_imu"), legacy_keys=False)
        if name not in self._names():
            raise SensorNotAvailableError(name)
        capabilities = set(self._robot._native._sensor_info(name)["capabilities"])
        if "camera_stream" in capabilities:
            return _Camera(modern.get_camera(name), modern, name)
        if "imu" in capabilities:
            return _Imu(modern.get_imu(name), legacy_keys=False)
        return modern[name]

    def __getattr__(self, name: str) -> Any:
        if name.startswith("_"):
            raise AttributeError(name)
        self._require_open()
        if name not in self._cache:
            self._cache[name] = self._resolve(name)
        return self._cache[name]

    def __getitem__(self, name: str) -> Any:
        return self.__getattr__(name)

    def has_sensor(self, name: str) -> bool:
        if name == "head_imu":
            return _SENSOR_ALIASES["head_imu"] in self._names()
        if name == "base_imu":
            return "chassis_imu" in self._names()
        if name == "lidar":
            try:
                self._by_capability("lidar_2d", name)
            except SensorNotAvailableError:
                return False
            return True
        return name in self._names()

    def get_active_sensors(self) -> List[str]:
        return self._modern().get_active_sensors()

    def wait_for_sensors(self, timeout: float = 5.0) -> bool:
        return self._modern().wait_for_sensors(timeout)

    def wait_for_all_active(self, timeout: float = 5.0) -> bool:
        return self._modern().wait_for_all_active(timeout)

    def get_sensor_count(self) -> int:
        return len(self._names())

    # Typed accessors of the modern manager, so mixed code keeps working.
    def get_imu(self, name: str) -> Any:
        return self._modern().get_imu(name)

    def get_camera(self, name: str) -> Any:
        return self._modern().get_camera(name)

    def get_lidar_2d(self, name: str) -> Any:
        return self._modern().get_lidar_2d(name)

    def get_lidar_3d(self, name: str) -> Any:
        return self._modern().get_lidar_3d(name)

    def get_ultrasonic(self, name: str) -> Any:
        return self._modern().get_ultrasonic(name)

    def get_wrench(self, component: str) -> Any:
        return self._modern().get_wrench(component)

    def get_temperature(self, component: str) -> Any:
        return self._modern().get_temperature(component)

    def shutdown(self) -> None:
        """Releases the camera subscriptions this wrapper created."""
        self._closed = True
        for handle in list(self._cache.values()):
            if isinstance(handle, _Camera):
                handle.shutdown()
        self._cache.clear()

    def __iter__(self):
        return iter(self._names())

    def __repr__(self) -> str:
        return f"Sensors({self._names()!r})"


# --------------------------------------------------------------------------
# Robot
# --------------------------------------------------------------------------

_signal_handlers_installed = False
_signal_lock = threading.Lock()


def _install_default_signal_handlers() -> None:
    """``auto_shutdown=True``: handle SIGTERM/SIGHUP only where nothing else does."""
    global _signal_handlers_installed
    with _signal_lock:
        if _signal_handlers_installed:
            return
        if threading.current_thread() is not threading.main_thread():
            warnings.warn(
                "Robot(auto_shutdown=True) from a worker thread: signal handlers can only be "
                "installed from the main thread. Call dexcontrol.install_signal_handlers() "
                "there, or pass auto_shutdown=False.",
                RuntimeWarning,
                stacklevel=3,
            )
            return
        eligible = tuple(
            getattr(signal, name)
            for name in ("SIGTERM", "SIGHUP")
            if hasattr(signal, name) and signal.getsignal(getattr(signal, name)) == signal.SIG_DFL
        )
        if eligible:
            dexcontrol.install_signal_handlers(eligible)
        _signal_handlers_installed = True


class Robot:
    """0.5 ``dexcontrol.robot.Robot`` over one :class:`dexcontrol.Robot`.

    Args:
        configs: A :class:`RobotConfig` from :func:`get_robot_config`, or
            ``None`` to resolve the profile from the environment.
        auto_shutdown: Install SIGTERM/SIGHUP handlers that turn the signal
            into a clean interpreter exit, as 0.5 did, but only for signals
            that still have their default action. Handlers an application
            installed are left alone.
        **native_kwargs: Passed to :class:`dexcontrol.Robot` (for example
            ``simulation=True``). ``profile`` and ``enable_sensors`` are
            derived from ``configs`` and cannot be given together with it.

    Use :meth:`Robot.wrap` to put this surface over a robot you already
    created; that view never shuts the shared robot down.
    """

    def __init__(
        self,
        configs: Optional[RobotConfig] = None,
        auto_shutdown: bool = True,
        **native_kwargs: Any,
    ) -> None:
        if configs is not None:
            if not isinstance(configs, RobotConfig):
                raise ConfigurationError(
                    "Robot(configs=...) accepts the object returned by "
                    "dexcontrol.compat.v05.get_robot_config(); the 0.5 dexbot_utils "
                    f"configuration classes are no longer available. See {MIGRATION_GUIDE}."
                )
            for key in ("profile", "enable_sensors"):
                if key in native_kwargs:
                    raise ConfigurationError(f"pass either configs= or {key}=, not both")
            native_kwargs["profile"] = configs.profile
            native_kwargs["enable_sensors"] = configs.enabled_sensors()
        if auto_shutdown:
            _install_default_signal_handlers()
        self._native = dexcontrol.Robot(**native_kwargs)
        self._owns_native = True
        self._init_views()

    @classmethod
    def wrap(cls, robot: Any) -> "Robot":
        """A 0.5 view over an existing :class:`dexcontrol.Robot` (not owned)."""
        view = cls.__new__(cls)
        view._native = robot
        view._owns_native = False
        view._init_views()
        return view

    def _init_views(self) -> None:
        self._components: Dict[str, Any] = {}
        self._sensors = Sensors(self)
        self._closed = False

    @property
    def native(self) -> Any:
        """The underlying :class:`dexcontrol.Robot`."""
        return self._native

    # -- components ----------------------------------------------------------

    _WRAPPERS = {
        "left_arm": Arm,
        "right_arm": Arm,
        "head": Head,
        "torso": Torso,
        "left_hand": Hand,
        "right_hand": Hand,
        "chassis": Chassis,
    }

    def _component(self, name: str) -> Any:
        if name not in self._components:
            if not self._native.has_component(name):
                raise ComponentNotAvailableError(name)
            if name == "chassis":
                handle = self._native.chassis
            elif name in ("battery", "estop", "heartbeat"):
                handle = getattr(self._native, name)
            else:
                handle = self._native.joints(name)
            wrapper = self._WRAPPERS.get(name)
            self._components[name] = wrapper(handle, self) if wrapper else handle
        return self._components[name]

    def __getattr__(self, name: str) -> Any:
        if name.startswith("_"):
            raise AttributeError(name)
        descriptor = getattr(type(self), name, None)
        if isinstance(descriptor, property) and descriptor.fget is not None:
            return descriptor.fget(self)
        if name in self._WRAPPERS or name in ("battery", "estop", "heartbeat"):
            return self._component(name)
        if self._native.has_component(name):
            return self._component(name)
        return getattr(self._native, name)

    @property
    def sensors(self) -> Sensors:
        return self._sensors

    # ``robot_model``, ``robot_name``, ``has_component``, ``have_hand``,
    # ``validate_component_names``, ``clear_error`` and
    # ``compensate_torso_pitch`` are unchanged since 0.5 and reach the modern
    # robot through ``__getattr__``.

    def reboot_component(self, board: str) -> None:
        """0.5 name of :meth:`dexcontrol.Robot.reboot`; no deprecation warning here."""
        self._native.reboot(board)

    def has_sensor(self, sensor: str) -> bool:
        return self._sensors.has_sensor(sensor)

    def get_controllable_component_map(self) -> Dict[str, Any]:
        return {name: self._component(name) for name in self._native.get_controllable_component_map()}

    # -- queries -------------------------------------------------------------

    def get_version_info(self, show: bool = True) -> Dict[str, Any]:
        info = self._native.version_info
        if not info:
            raise ServiceUnavailableError("Version info service not responding.\n")
        if show:
            print("Version information:")
            for key, value in info.items():
                print(f"  {key}: {value}")
        return info

    def get_component_status(self, show: bool = True) -> Dict[str, Any]:
        try:
            status = self._native.component_status()
        except Exception as error:  # noqa: BLE001 - 0.5 wrapped every failure
            raise ServiceUnavailableError(f"Failed to retrieve component status: {error}") from error
        if show:
            print("Component status:")
            for key, value in status.items():
                print(f"  {key}: {value}")
        return status

    def query_hand_type(self, max_attempts: int = 5) -> None:
        del max_attempts
        raise _unsupported(
            "query_hand_type()",
            "the hand is selected by the robot profile, not detected; read robot.robot_model, "
            "robot.has_component('left_hand') or robot.left_hand.joint_names",
        )

    def query_ntp(
        self,
        sample_count: int = 30,
        show: bool = False,
        device: Literal["soc", "jetson"] = "soc",
    ) -> Dict[str, Any]:
        if device != "soc":
            raise _unsupported(f"query_ntp(device={device!r})", "only the SOC clock is queried")
        result = self._native.query_ntp(sample_count)
        if show:
            print("NTP:")
            for key, value in result.items():
                print(f"  {key}: {value}")
        return result

    # -- motion --------------------------------------------------------------

    def set_joint_pos(self, joint_pos: Dict[str, Any], relative: bool = False) -> None:
        self._native.set_joint_pos(joint_pos, relative=relative)

    def move_joint_pos(self, joint_pos: Dict[str, Any], *, relative: bool = False, velocity_scale: Any = None) -> None:
        self._native.stream_joint_pos(joint_pos, relative=relative, velocity_scale=velocity_scale)

    def move_to_joint_pos(
        self, joint_pos: Dict[str, Any], *, relative: bool = False, velocity_scale: Any = None
    ) -> MultiMotionHandle:
        return MultiMotionHandle(
            self._native.move_to_joint_pos(joint_pos, relative=relative, velocity_scale=velocity_scale, wait=False)
        )

    @property
    def default_velocity_scale(self) -> Optional[float]:
        return self._native.default_velocity_scale

    @default_velocity_scale.setter
    def default_velocity_scale(self, value: Optional[float]) -> None:
        self._native.default_velocity_scale = value

    def execute_trajectory(self, trajectory: Dict[str, Any], control_hz: float = 100, relative: bool = False) -> None:
        """0.5 ``execute_trajectory`` with absolute waypoints.

        ``relative=True`` is refused: 0.5 added each tick's offset to that
        tick's measured position, so a constant offset compounded and the arm
        walked. Convert the offsets to absolute waypoints from one start
        position first.
        """
        if not trajectory:
            raise ValueError("Trajectory must be a non-empty dictionary")
        if relative:
            raise ValueError(
                "execute_trajectory(relative=True) is not reproduced: dexcontrol 0.5 resolved "
                "every waypoint against the position measured at that tick, which compounds "
                "offsets. Add the offsets to one starting position and pass absolute "
                f"waypoints. See {MIGRATION_GUIDE}."
            )
        self._native.execute_trajectory(trajectory, control_hz)

    def get_joint_pos_dict(self, component: Union[str, List[str]]) -> Dict[str, float]:
        """Flat ``{joint_name: position}`` across the requested components."""
        names = [component] if isinstance(component, str) else list(component)
        if not isinstance(component, (str, list)):
            raise ValueError("Component must be a string or list of strings")
        out: Dict[str, float] = {}
        for name in names:
            if not self._native.has_component(name):
                raise KeyError(f"Invalid component name: {name}")
            out.update(self._native.get_joint_pos_dict(name))
        return out

    # -- removed subscription controls ---------------------------------------

    def set_idle_timeout(self, seconds: float) -> None:
        del seconds
        raise _unsupported("set_idle_timeout()", "native state subscriptions are always on")

    def set_subscription_policy(self, policy: str, recursive: bool = True) -> None:
        del policy, recursive
        raise _unsupported(
            "Robot.set_subscription_policy()",
            "native state subscriptions are always on; camera streams use subscribe()/unsubscribe()",
        )

    def get_subscription_status(self) -> None:
        raise _unsupported("get_subscription_status()", "native state subscriptions are always on")

    # -- lifecycle -----------------------------------------------------------

    def shutdown(self) -> None:
        """Releases this view's subscriptions and, if it created the robot, the robot."""
        if self._closed:
            return
        self._closed = True
        self._sensors.shutdown()
        if self._owns_native:
            self._native.shutdown()

    close = shutdown

    def is_shutdown(self) -> bool:
        return self._closed or bool(self._native.is_shutdown())

    def __enter__(self) -> "Robot":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.shutdown()

    def __repr__(self) -> str:
        owner = "owned" if self._owns_native else "view"
        return f"compat.v05.Robot({self._native!r}, {owner})"

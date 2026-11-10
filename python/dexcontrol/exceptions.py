# Copyright (C) 2026 Dexmate Inc.
# SPDX-License-Identifier: Apache-2.0

"""Custom exceptions for dexcontrol.

This module defines a hierarchy of exceptions for better error handling
and user experience when connection or configuration issues occur. Every
native (Rust) failure raised by dexcontrol is an instance of one of these
classes, so ``except dexcontrol.DexcontrolError`` catches them all; pure
argument validation raises the built-in ``ValueError``/``TypeError``.
"""

from __future__ import annotations


class DexcontrolError(Exception):
    """Base exception for all dexcontrol errors.

    All custom dexcontrol exceptions inherit from this class,
    making it easy to catch all dexcontrol-specific errors.

    Attributes:
        code: Stable machine-readable identifier: the name of the native
            error variant that produced this exception (for example
            ``"Timeout"``, ``"StaleState"``, ``"WriteBlocked"``). ``None``
            for exceptions raised by the pure-Python layer. Exceptions the
            binding raises as built-in ``ValueError`` carry the same
            attribute. Match on ``code`` rather than on message text.
    """

    code: str | None = None


class ConfigurationError(DexcontrolError):
    """Raised when there is a configuration problem.

    This includes missing or invalid environment variables,
    missing config files, or invalid configuration content.
    """


class RobotConnectionError(DexcontrolError):
    """Raised when the robot cannot be reached.

    This typically indicates network issues, robot not powered on,
    or communication routing problems.
    """


class ServiceUnavailableError(DexcontrolError):
    """Raised when a specific service is not responding.

    This indicates the robot is likely connected but a specific
    service (e.g., hand type query) is not available. This could
    happen during robot initialization or if a component is disabled.
    """


class ComponentError(DexcontrolError):
    """Raised when a component fails to initialize or activate.

    This indicates that one or more robot components could not be
    started, activated, or are in an invalid state for operation.
    """


class ComponentNotAvailableError(ComponentError, AttributeError):
    """Raised when accessing a component that is not available on this robot.

    This typically means the component is either not present on this robot
    model or has been disabled in the configuration.
    """

    def __init__(self, component: str, robot_model: str = "unknown") -> None:
        self.component = component
        self.robot_model = robot_model
        super().__init__(
            f"Component '{component}' is not available on this robot "
            f"(model: {robot_model}). "
            f"Use robot.has_component('{component}') to check availability "
            f"before access."
        )


class SensorNotAvailableError(ComponentError, AttributeError):
    """Raised when accessing a sensor that is not available or not initialized.

    This typically means the sensor is either not present on this robot model,
    not enabled in the configuration, or failed to initialize.
    """

    def __init__(self, sensor: str, hint: str = "") -> None:
        self.sensor = sensor
        self.hint = hint
        message = (
            f"Sensor '{sensor}' is not available or not initialized. "
            f"Use robot.has_sensor('{sensor}') to check availability "
            f"before access."
        )
        if hint:
            message = f"{message} {hint}"
        super().__init__(message)


class ModelNotSupportedError(ComponentError):
    """Raised when a method is called on an unsupported robot model.

    This indicates that the called method or feature is not available
    on the current robot model.
    """

    def __init__(
        self,
        method: str,
        robot_model: str = "unknown",
        supported_models: tuple[str, ...] = (),
    ) -> None:
        self.method = method
        self.robot_model = robot_model
        self.supported_models = supported_models
        super().__init__(
            f"'{method}' is not supported on model '{robot_model}'. "
            f"Supported models: {', '.join(supported_models)}"
        )


class PluginNotAvailableError(DexcontrolError):
    """Raised when a required server-side plugin is not available.

    This indicates that a plugin (e.g., the motion plugin) is not running
    on the robot-server, does not support the requested component, or that
    an optional feature is not built into this runtime.
    """


class DexTimeoutError(DexcontrolError, TimeoutError):
    """Base class for every native timeout.

    Raised directly for timeouts that are neither a motion wait nor a state
    wait: a service request that got no reply, a reliable publish whose
    delivery could not be confirmed. Also a built-in ``TimeoutError``.
    """


class MotionTimeoutError(DexTimeoutError):
    """Raised when a wait on a managed motion (or convergence wait) times out.

    For managed motions the still-live handle is exposed through the
    ``motion_handle`` attribute so the caller can keep waiting, inspect, or
    cancel it (the motion keeps running server-side).
    """

    motion_handle = None


class StateTimeoutError(DexTimeoutError):
    """Raised when waiting for a state sample times out.

    Typical cause: a component never published its first state sample
    within the connect deadline, or a fresh-state wait saw no update.
    Before 0.7 these surfaced as ``MotionTimeoutError``; both remain
    catchable as :class:`DexTimeoutError` and ``TimeoutError``.
    """


class StaleStateError(ComponentError):
    """Raised when measured state is older than the freshness limit.

    The command was well formed; the robot stopped telling us where it is,
    so a command depending on the measured state was refused. Subclass of
    :class:`ComponentError`, which is what earlier releases raised.
    """


class EStopActiveError(ComponentError, ValueError):
    """Raised when a write is blocked by the E-stop gate.

    Either the E-stop is engaged (``estop_active`` is true) or its state
    could not be read and the write was refused fail-closed
    (``estop_active`` is false).

    Migration note: earlier releases raised a plain ``ValueError`` here. For
    ONE release this class also inherits from ``ValueError`` so existing
    ``except ValueError`` handlers keep working; that base will be removed,
    so catch ``EStopActiveError`` (or ``ComponentError``) instead.

    Attributes:
        component: Component whose write was blocked.
        operation: Operation that was refused.
        estop_active: True when the E-stop is known to be engaged; false
            when its state was unreadable.
    """

    component: str | None = None
    operation: str | None = None
    estop_active: bool = False


class MotionStoppedError(DexcontrolError):
    """Raised when an operation was interrupted by a stop or safety action.

    Covers a waited move that ended cancelled or superseded, a host
    trajectory, reconnect or connect interrupted by ``stop()``, a safety
    action, or shutdown. The motion did NOT reach its target.

    Attributes:
        motion_handle: The motion handle (or group handle) when the
            interrupted operation had one, else ``None``.
        members: For a multi-component move, the names of the components
            whose motion was cancelled or superseded; else an empty tuple.
    """

    motion_handle = None
    members: tuple[str, ...] = ()


class ServiceRejectedError(ComponentError):
    """A delivered firmware request was explicitly refused by the server."""

    component: str
    operation: str
    server_code: str | None


__all__ = [
    "DexcontrolError",
    "ConfigurationError",
    "RobotConnectionError",
    "ServiceUnavailableError",
    "ComponentError",
    "ComponentNotAvailableError",
    "SensorNotAvailableError",
    "ModelNotSupportedError",
    "PluginNotAvailableError",
    "DexTimeoutError",
    "MotionTimeoutError",
    "StateTimeoutError",
    "StaleStateError",
    "EStopActiveError",
    "MotionStoppedError",
    "ServiceRejectedError",
]

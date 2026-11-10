# Copyright (C) 2026 Dexmate Inc.
# SPDX-License-Identifier: Apache-2.0

"""DexControl: Robot Control Interface Library.

Thin Python interface over the native (Rust) DexControl runtime. It
serves as the primary API for interacting with Dexmate robots.
"""

from __future__ import annotations

import atexit
import signal
import sys
from pathlib import Path

# Imported eagerly, on the importing (normally main) thread: the native log
# bridge resolves ``_logsink.emit`` on its drain thread, and an import cost or
# failure must not first surface there.
from dexcontrol import _logsink, _native  # noqa: F401 - eager import, see above

# Development checkout checks are excluded from customer wheels.
import importlib.util as _import_util

if _import_util.find_spec("dexcontrol._devtree") is not None:
    from dexcontrol import _devtree
else:
    _devtree = None

# A source checkout whose Rust sources are newer than the built extension
# would run a script against yesterday's engine without saying so.
_stale = _devtree.import_time_check() if _devtree is not None else None
if _stale is not None:
    import sys as _sys

    print(f"\n*** {_stale}\n", file=_sys.stderr, flush=True)
del _stale
from dexcontrol._logsink import configure_logging
from dexcontrol._prompt import ask, confirm
from dexcontrol._native import (
    MIN_SOC_SOFTWARE_VERSION,
    Battery,
    Chassis,
    ControlContext,
    EStop,
    Odometry,
    Pose2d,
    Heartbeat,
    JointComponent,
    MotionHandle,
    MotionStatus,
    MultiMotionHandle,
    RateLimiter,
    Sensor,
    _shutdown_all_robots,
    available_profiles,
    profile_for_robot_name,
    profile_from_environment,
    robot_config,
)
from dexcontrol.exceptions import (
    ComponentError,
    ComponentNotAvailableError,
    ConfigurationError,
    DexcontrolError,
    DexTimeoutError,
    EStopActiveError,
    ModelNotSupportedError,
    MotionStoppedError,
    MotionTimeoutError,
    PluginNotAvailableError,
    RobotConnectionError,
    SensorNotAvailableError,
    ServiceRejectedError,
    ServiceUnavailableError,
    StaleStateError,
    StateTimeoutError,
)
from dexcontrol.robot import Robot
from dexcontrol.diagnostics import DiagnosticClient
from dexcontrol.types import MotionState

try:
    from importlib.metadata import PackageNotFoundError, version

    __version__ = version("dexcontrol")
except PackageNotFoundError:  # pragma: no cover - not installed as a wheel
    __version__ = _native.__version__


def get_comm_cfg_path() -> str | None:
    """Get the communication (Zenoh) config path.

    Priority:
    1. ``.dzcfg`` files in ``~/.dexmate/comm/zenoh/`` or subdirectories
    2. ``zenoh*config*.json5`` files under ``~/.dexmate/comm/zenoh/``
    """
    base_dir = Path("~/.dexmate/comm/zenoh/").expanduser()
    dzcfg_files = sorted(base_dir.glob("**/*.dzcfg"))
    if dzcfg_files:
        return dzcfg_files[0].as_posix()
    json5_files = sorted(base_dir.glob("**/zenoh*config*.json5"))
    if json5_files:
        return json5_files[0].as_posix()
    return None


# Multi-robot shutdown at interpreter exit: a native shutdown of every
# still-live robot. Shutdown is idempotent, so robots closed explicitly (or
# via context manager) are unaffected. Registered after ``_native`` registered
# its log-bridge teardown, so (atexit being LIFO) robots are shut down first
# and their warnings are still delivered.
#
# ``atexit`` does NOT run when the process dies from a signal's default
# action (SIGTERM, SIGHUP); see :func:`install_signal_handlers`.
atexit.register(_shutdown_all_robots)


def install_signal_handlers(signals: tuple[int, ...] | None = None) -> None:
    """Opt in: turn SIGTERM / SIGHUP into a clean interpreter exit.

    By default Python dies immediately on SIGTERM and SIGHUP without running
    ``atexit`` hooks, so ``systemctl stop``, ``docker stop``, ``kill`` or a
    closed terminal ends the process WITHOUT the robot shutdown (and its
    stop request) that a normal exit performs. The pre-0.6 package installed
    such handlers implicitly at import; 0.7 installs none, because a library
    must not take over process-wide signal handling behind the
    application's back. Call this once, from the main thread, if nothing
    else in your application handles these signals::

        import dexcontrol
        dexcontrol.install_signal_handlers()

    The handler raises ``SystemExit(128 + signum)`` in the main thread, so
    ``finally`` blocks, context managers and ``atexit`` (including
    dexcontrol's robot shutdown) run. SIGINT is left alone: Python already
    turns it into ``KeyboardInterrupt``.

    Args:
        signals: Signal numbers to handle. Defaults to SIGTERM and, where it
            exists, SIGHUP.

    Raises:
        ValueError: when called from a thread other than the main thread
            (a ``signal.signal`` restriction).
    """
    if signals is None:
        signals = tuple(
            getattr(signal, name) for name in ("SIGTERM", "SIGHUP") if hasattr(signal, name)
        )

    def _exit_cleanly(signum, _frame):
        sys.exit(128 + signum)

    for signum in signals:
        signal.signal(signum, _exit_cleanly)


def build_info() -> dict[str, object]:
    """Identify the loaded native build; use content_id to distinguish source revisions."""
    info = _native.build_info()
    info["module_path"] = _native.__file__
    return info


__all__ = [
    "MIN_SOC_SOFTWARE_VERSION",
    "Battery",
    "Chassis",
    "ComponentError",
    "ComponentNotAvailableError",
    "ConfigurationError",
    "ControlContext",
    "DexTimeoutError",
    "DexcontrolError",
    "DiagnosticClient",
    "EStop",
    "Odometry",
    "Pose2d",
    "EStopActiveError",
    "Heartbeat",
    "JointComponent",
    "ModelNotSupportedError",
    "MotionHandle",
    "MotionState",
    "MotionStatus",
    "MotionStoppedError",
    "MotionTimeoutError",
    "MultiMotionHandle",
    "PluginNotAvailableError",
    "RateLimiter",
    "Robot",
    "RobotConnectionError",
    "Sensor",
    "SensorNotAvailableError",
    "ServiceRejectedError",
    "ServiceUnavailableError",
    "StaleStateError",
    "StateTimeoutError",
    "available_profiles",
    "build_info",
    "ask",
    "configure_logging",
    "confirm",
    "get_comm_cfg_path",
    "install_signal_handlers",
    "profile_for_robot_name",
    "profile_from_environment",
    "robot_config",
    "__version__",
]

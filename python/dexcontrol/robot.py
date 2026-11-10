# Copyright (C) 2026 Dexmate Inc.
# SPDX-License-Identifier: Apache-2.0

"""High-level Robot class: thin Python conveniences over the native runtime.

The native ``_native.Robot`` performs every connection, subscription,
decode, and control operation in Rust. This subclass adds only what reads
better in Python: the sensor accessor and a few small conveniences.
"""

from __future__ import annotations

import weakref

from dexcontrol.sensors.manager import Sensors
from typing import Any

from dexcontrol import _native


class Robot(_native.Robot):
    """Main robot interface (see the package README for examples).

    Always release the connection: use ``with Robot(...) as robot:`` or call
    :meth:`shutdown` (alias :meth:`close`). A robot that is garbage-collected
    without it still performs a bounded (2 s) native shutdown — stop request
    included — but emits a ``ResourceWarning``; do not rely on it. A robot
    object cannot be used in a forked child process (``RuntimeError``):
    create it in the child, or use the ``spawn`` start method.

    Args:
        profile: Built-in robot profile name. Defaults to the
            ``ROBOT_CONFIG`` environment variable, then to the profile the
            ``ROBOT_NAME`` environment variable maps to, then ``vega_1``
            (with a warning). A ``ROBOT_NAME`` that names no known robot
            raises :class:`~dexcontrol.ConfigurationError` instead of
            silently selecting ``vega_1``: the wrong profile moves real
            hardware with the wrong limits.
        config_file: Path to a custom robot YAML/JSON file (mutually
            exclusive with ``profile``).
        enable_sensors: Optional configured sensors to enable.
        simulation: When true, runs against the deterministic in-process
            loopback simulation instead of real hardware.
        watchdog: Controls the dedicated-process safety watchdog (one small
            extra Python process supervising heartbeat/E-stop on its own
            session, isolated from this one). ``None`` keeps the default:
            on for production robots, always off for simulation. An
            explicit bool overrides the production default.
        exit_on_safety_termination: When a safety monitor fires the
            configured ``request_process_termination`` action (e.g. a
            heartbeat timeout with that policy), the native layer makes the
            robot safe, emits the safety event, logs critically to stderr,
            and terminates this process with exit code 1 — immediately and
            without running Python cleanup (an unstable network must not
            leave a half-alive controller running). The exit uses the C
            ``_exit``: no ``atexit`` hooks, no ``finally`` blocks, no
            ``__del__``, no static destructors, and unflushed ``stdout``
            buffers are lost. ``None`` keeps the
            default: ``False`` in every language. Safety actions still stop
            the robot and emit events; pass ``True`` only when this application
            explicitly requires whole-process termination. Pausing heartbeat monitoring
            (``robot.heartbeat.pause()``) suspends the whole heartbeat
            reaction chain, including this one.
        context: Optional shared :class:`~dexcontrol.ControlContext`.
        require_version_check: When true, connecting fails closed with
            :class:`~dexcontrol.ConfigurationError` if the robot's versions
            service cannot be reached or verified. By default that only
            downgrades ``robot.health()["version_check"]``.
        initialize_arm_modes: When true (the default), connecting puts
            every arm into position mode once it is ready, and a required
            arm that cannot be enabled fails the connection with
            :class:`~dexcontrol.ComponentError`. Pass ``False`` to connect
            without touching the arms: for troubleshooting (clearing a
            component error on an arm whose motor will not enable) or for
            an application that sets modes itself. Nothing else changes;
            hands, head and chassis start as before.
        adopt_detected_hardware: When true (the default), connecting asks a
            Vega which hands it carries and adopts them: a known hand family
            replaces the profile's hands (the ``_gripper`` or ``_f5d6``
            variant's components are overlaid on the profile you chose, so
            ``Robot("vega_1p")`` on a robot with grippers gets ``left_hand``
            and ``right_hand``), a side the robot reports empty is
            disabled, and the connect log says what was adopted over what.
            Pass ``False`` to use the profile's hands exactly as configured;
            a disagreement is then only a warning. A ``config_file`` is
            never changed either way. Other models ignore this.

    Migration notes (from the pre-0.6 pure-Python package):
        * ``component.get_joint_pos(joint_id=3)`` (and the other
          ``get_joint_*`` getters) return a 1-element ``numpy.ndarray`` for an
          integer ``joint_id``; the old API returned a scalar. Index the
          result (``[0]``) or call ``float()`` on it.
        * No SIGTERM/SIGHUP handlers are installed. The old package
          converted them to ``sys.exit`` so ``atexit`` shut the robot down;
          call :func:`dexcontrol.install_signal_handlers` to opt in.
        * ``robot.health()`` additionally reports ``supervision_failed`` (a
          safety monitor task died; nothing is watching) and
          ``stop_undelivered`` (the last safety stop could not be delivered;
          the robot may still be moving).
    """

    # 0.5 ``Robot`` methods that no longer exist, plus the names migrators
    # most often guess at. The message names the replacement.
    _LEGACY_HINTS = {
        "get_version_info": "use the `version_info` property",
        "get_component_status": "use `component_status()`",
        "query_hand_type": "hands are adopted from the robot at connect; see `has_component('left_hand')` and the connect log's `end effectors:` line",
        "set_subscription_policy": "native state subscriptions are always on; camera streams use subscribe()/unsubscribe()",
        "get_subscription_status": "native state subscriptions are always on",
        "set_idle_timeout": "native state subscriptions are always on",
        "wrench_sensor": "use `robot.sensors.get_wrench(component)`",
        "temperature_sensor": "use `robot.sensors.get_temperature(component)`",
        "clear_component_errors": "use `clear_errors()` or `clear_error(component)`",
        "get_joint_pos": "joint getters live on components, e.g. `robot.left_arm.get_joint_pos()`",
    }
    _LEGACY_HAND_HINT = (
        "the robot reported no hand there at connect (see the connect log's "
        "`end effectors:` line); a hand the robot reports is adopted into any "
        "profile, or select the variant that configures one, e.g. 'vega_1_f5d6'"
    )

    def __getattr__(self, name: str):
        """Only reached after the native lookup (components, sensors) failed."""
        message = f'Robot has no component or sensor "{name}"'
        if name in ("left_hand", "right_hand"):
            message += f": {self._LEGACY_HAND_HINT}"
        elif name in self._LEGACY_HINTS:
            message += (
                f". Removed in dexcontrol 0.7: {self._LEGACY_HINTS[name]}. Legacy code can "
                "import `dexcontrol.compat.v05.Robot`; see docs/MIGRATING-FROM-0.5.md"
            )
        raise AttributeError(message)

    def __init__(
        self,
        profile=None,
        *,
        config_file=None,
        enable_sensors=(),
        simulation=False,
        watchdog=None,
        exit_on_safety_termination=None,
        context=None,
        require_version_check=False,
        initialize_arm_modes=True,
        adopt_detected_hardware=True,
        _sim_event_driven_estop=False,
    ):
        # The native __new__ has already connected using these arguments.
        del profile, config_file, enable_sensors, simulation, watchdog
        del exit_on_safety_termination, context, require_version_check
        del initialize_arm_modes, adopt_detected_hardware
        del _sim_event_driven_estop
        self._sensor_manager = None

    # ------------------------------------------------------------------
    # Sensors
    # ------------------------------------------------------------------
    @property
    def sensors(self):
        """Every sensor on this robot; see
        :class:`dexcontrol.sensors.manager.Sensors`.

        The manager refers back to this robot weakly (a strong reference
        would form a cycle that keeps an unclosed robot, its threads and its
        connection alive until the cycle collector runs), so keep the robot
        itself referenced while using it.
        """
        if self._sensor_manager is None:
            self._sensor_manager = Sensors(weakref.proxy(self))
        return self._sensor_manager

    # ------------------------------------------------------------------
    # Convenience helpers
    # ------------------------------------------------------------------
    def get_controllable_component_map(self) -> dict[str, Any]:
        """Maps every enabled joint-controllable component to its handle."""
        result: dict[str, Any] = {}
        for name in self.component_names:
            try:
                result[name] = self.joints(name)
            except Exception:  # noqa: BLE001 - non-joint components are skipped
                continue
        return result

    def validate_component_names(self, joint_pos: dict[str, Any]) -> None:
        """Raises ``ValueError`` when a command dictionary names an unknown
        or non-controllable component."""
        controllable = set(self.get_controllable_component_map())
        unknown = [name for name in joint_pos if name not in controllable]
        if unknown:
            raise ValueError(
                f"Unknown or non-controllable component(s): {sorted(unknown)}. "
                f"Controllable components: {sorted(controllable)}"
            )

    def have_hand(self, side: str) -> bool:
        """True when a hand component is configured on the given side."""
        if side not in ("left", "right"):
            raise ValueError("side must be 'left' or 'right'")
        return self.has_component(f"{side}_hand")

    def supports(self, component: str, capability: str) -> bool:
        """Whether an enabled component declares a capability in this connection."""
        info = self.component_info(component)
        return bool(info["enabled"]) and capability in {item["id"] for item in info["capabilities"]}

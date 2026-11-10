# Copyright (C) 2026 Dexmate Inc.
# SPDX-License-Identifier: Apache-2.0

"""Shared DualSense connection, button mapping, and cleanup helpers.

Imported by gamepad examples. Instantiation activates the controller and connects a Robot; L1 manages interactive movement state and touchpad toggles software E-stop. Running this module directly defines classes but starts no control loop."""

from __future__ import annotations

import threading
from abc import ABC, abstractmethod

from dexcontrol import RateLimiter, Robot

try:
    from dualsense_controller import DualSenseController
except ImportError as error:  # resolved when the optional example is instantiated
    DualSenseController = None
    _DUALSENSE_IMPORT_ERROR = error
else:
    _DUALSENSE_IMPORT_ERROR = None


class DualSenseTeleopBase(ABC):
    """Safety-gated base for real-robot DualSense teleoperation."""

    def __init__(
        self,
        control_hz: int = 200,
        button_update_hz: int = 20,
        device_index: int = 0,
    ) -> None:
        if DualSenseController is None:
            raise RuntimeError(
                "DualSense support is optional; install dexcontrol[examples]"
            ) from _DUALSENSE_IMPORT_ERROR
        self.is_running = True
        self.control_hz = control_hz
        self.button_update_hz = button_update_hz
        self.button_lock = threading.Lock()
        self.estop_lock = threading.Lock()
        self.active_buttons: set[str] = set()
        self.safe_pressed = False
        self.estop_on = False

        self.dualsense = DualSenseController(device_index_or_device_info=device_index)
        self.dualsense.activate()
        try:
            self.bot = Robot()
        except Exception:
            self.dualsense.deactivate()
            raise
        self._setup_button_mappings()

    def _setup_button_mappings(self) -> None:
        self.dualsense.btn_l1.on_down(self.safety_check)
        self.dualsense.btn_l1.on_up(self.safety_check_release)
        self.dualsense.btn_touchpad.on_down(self.toggle_estop)
        mapping = {
            "btn_up": "dpad_up",
            "btn_down": "dpad_down",
            "btn_right": "dpad_right",
            "btn_left": "dpad_left",
            "btn_circle": "circle",
            "btn_square": "square",
            "btn_triangle": "triangle",
            "btn_cross": "cross",
            "btn_r1": "r1",
            "btn_r2": "r2",
        }
        for button_name, action in mapping.items():
            button = getattr(self.dualsense, button_name)
            button.on_down(lambda name=action: self.add_button(name))
            button.on_up(lambda name=action: self.remove_button(name))
        self._setup_additional_mappings()

    def _setup_additional_mappings(self) -> None:
        """Hook for subclasses."""

    def add_button(self, button: str) -> None:
        if self.safe_pressed or button in {"l1", "touchpad"}:
            with self.button_lock:
                self.active_buttons.add(button)

    def remove_button(self, button: str) -> None:
        with self.button_lock:
            self.active_buttons.discard(button)

    def get_active_buttons(self) -> set[str]:
        with self.button_lock:
            return set(self.active_buttons)

    def safety_check(self) -> None:
        self.safe_pressed = True
        self.dualsense.left_rumble.set(50)
        self.dualsense.left_rumble.set(0)

    def safety_check_release(self) -> None:
        self.safe_pressed = False
        with self.button_lock:
            self.active_buttons.clear()
        self.stop_all_motion()

    def toggle_estop(self) -> None:
        with self.estop_lock:
            self.estop_on = not self.estop_on
            if self.estop_on:
                self.stop_all_motion()
                self.bot.estop.activate()
                self.dualsense.lightbar.set_color(255, 0, 0)
            else:
                self.bot.estop.deactivate()
                self.update_controller_feedback()

    def update_controller_feedback(self) -> None:
        self.dualsense.lightbar.set_color_white()

    def run_forever(self) -> None:
        limiter = RateLimiter(self.button_update_hz)
        try:
            while self.is_running and not self.bot.is_shutdown():
                self.update_motion()
                limiter.sleep()
        except KeyboardInterrupt:
            pass
        finally:
            self.cleanup()

    def cleanup(self) -> None:
        self.is_running = False
        try:
            self.stop_all_motion()
        finally:
            self.dualsense.lightbar.set_color_white()
            self.dualsense.deactivate()
            self.bot.shutdown()

    @abstractmethod
    def update_motion(self) -> None:
        """Update command targets from current controller state."""

    @abstractmethod
    def stop_all_motion(self) -> None:
        """Immediately stop/reset motion for the subclass."""

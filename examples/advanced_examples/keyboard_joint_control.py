# Copyright (C) 2025 Dexmate Inc.
#
# This software is dual-licensed:
#
# 1. GNU Affero General Public License v3.0 (AGPL-3.0)
#    See LICENSE-AGPL for details
#
# 2. Commercial License
#    For commercial licensing terms, contact: contact@dexmate.ai

"""
Keyboard Joint Control for Robot Components

Control individual joints of robot components (arms, torso, head) using keyboard input.
Hold w/s for smooth continuous motion — the robot moves while the key is held and stops
automatically on release. A background thread reads key events and detects release via
a timeout on terminal key-repeat events.

Keys:
    '0'-'9': Select joint by index
    'w': Hold to move joint in + direction
    's': Hold to move joint in - direction
    'q': Quit
    Ctrl+C: Quit (emergency exit)

Usage:
    python keyboard_joint_control.py --component left_arm
    python keyboard_joint_control.py --component left_arm --step-size 1.0 --control-rate 100
"""

import signal
import sys
import termios
import threading
import time
import tty
from dataclasses import dataclass
from types import FrameType
from typing import Literal

import numpy as np
import tyro
from loguru import logger

from dexcontrol.robot import Robot

# Time without a repeated key event before motion auto-stops (seconds).
# Terminal key-repeat is typically 30-50ms, so 150ms is a safe margin.
HOLD_TIMEOUT = 0.15

COMPONENT_CONFIG: dict[str, dict] = {
    "left_arm": {"num_joints": 7, "max_step": 10.0},
    "right_arm": {"num_joints": 7, "max_step": 10.0},
    "torso": {"num_joints": 3, "max_step": 5.0},
    "head": {"num_joints": 3, "max_step": 10.0},
}


@dataclass
class ControlState:
    """Mutable state for the control loop."""

    joint_idx: int
    motion_direction: int  # +1, -1, or 0
    last_move_key_time: float  # monotonic timestamp of last w/s event
    current_pos_deg: np.ndarray

    def status_line(self, max_joint_idx: int) -> str:
        state = (
            ">>>"
            if self.motion_direction == 1
            else "<<<"
            if self.motion_direction == -1
            else "---"
        )
        return f"\r[{state}] Joint {self.joint_idx}: {self.current_pos_deg[self.joint_idx]:.2f}° | hold w/s: move, 0-{max_joint_idx}: select, q: quit    "


class KeyboardReader:
    """Reads keypresses in a background thread with timestamps for hold detection.

    Usage::

        reader = KeyboardReader()
        reader.start()
        try:
            for key, t in reader.drain():
                ...
        finally:
            reader.stop()
    """

    def __init__(self):
        self._keys: list[tuple[str, float]] = []
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._old_settings: list | None = None

    def start(self):
        """Enter raw mode and start the background reader thread."""
        self._old_settings = termios.tcgetattr(sys.stdin.fileno())
        tty.setraw(sys.stdin.fileno())
        self._thread = threading.Thread(target=self._reader_loop, daemon=True)
        self._thread.start()

    def stop(self):
        """Restore terminal settings. The daemon thread exits with the process."""
        if self._old_settings is not None:
            termios.tcsetattr(sys.stdin.fileno(), termios.TCSADRAIN, self._old_settings)
            self._old_settings = None
        self._stop_event.set()

    def drain(self) -> list[tuple[str, float]]:
        """Return and clear all buffered (key, monotonic_timestamp) pairs."""
        with self._lock:
            keys = self._keys
            self._keys = []
        return keys

    def _reader_loop(self):
        while not self._stop_event.is_set():
            try:
                ch = sys.stdin.read(1)
                if ch:
                    with self._lock:
                        self._keys.append((ch, time.monotonic()))
            except Exception:
                break


def _get_component(robot: Robot, component: str):
    """Return the component object, enabling it if needed."""
    comp_obj = getattr(robot, component)
    if component == "head":
        comp_obj.set_mode("enable")
    return comp_obj


def _wait_for_keypress():
    """Block until a single key is pressed (enters/exits raw mode)."""
    fd = sys.stdin.fileno()
    old = termios.tcgetattr(fd)
    try:
        tty.setraw(fd)
        sys.stdin.read(1)
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old)


def _show(state: ControlState, max_joint_idx: int):
    print(state.status_line(max_joint_idx), end="", flush=True)


def control_joints(
    robot: Robot,
    component: Literal["left_arm", "right_arm", "torso", "head"],
    step_size: float,
    control_rate: float,
) -> None:
    """Interactive hold-to-move keyboard control of robot joints.

    Hold w/s for continuous motion at the configured rate. Release to stop.
    Number keys select the active joint.

    Args:
        robot: Robot instance
        component: Component to control
        step_size: Angle change in degrees per control command
        control_rate: Control command frequency in Hz
    """
    if control_rate <= 0.0:
        raise ValueError(f"control_rate must be positive, got {control_rate}")

    cfg = COMPONENT_CONFIG[component]
    num_joints: int = cfg["num_joints"]
    max_joint_idx = num_joints - 1

    # Safety-clamp step size
    max_step: float = cfg["max_step"]
    if step_size > max_step:
        logger.warning(f"Step size clamped to {max_step}° for {component} safety")
        step_size = max_step

    comp_obj = _get_component(robot, component)
    control_dt = 1.0 / control_rate

    # Print instructions
    effective_speed = step_size * control_rate
    print(f"\nControlling {component}, default joint index: {max_joint_idx}")
    print(f"Step size: {step_size}°/cmd × {control_rate} Hz = {effective_speed:.0f}°/s")
    print("Commands:")
    print("  w: Hold to move joint in + direction")
    print("  s: Hold to move joint in - direction")
    print(f"  0-{max_joint_idx}: Select joint by index")
    print("  q: Quit")
    print("\nPress any key to start...")

    _wait_for_keypress()

    # Ctrl+C handler — restore cursor then propagate
    original_sigint = signal.getsignal(signal.SIGINT)

    def _sigint_handler(_sig: int, _frame: FrameType | None):
        print("\033[?25h", end="", flush=True)
        print("\nEmergency stop triggered by Ctrl+C")
        raise KeyboardInterrupt

    signal.signal(signal.SIGINT, _sigint_handler)

    reader = KeyboardReader()
    reader.start()
    print("\033[?25l", end="", flush=True)  # Hide cursor

    try:
        state = ControlState(
            joint_idx=max_joint_idx,
            motion_direction=0,
            last_move_key_time=0.0,
            current_pos_deg=np.rad2deg(comp_obj.get_joint_pos()),
        )
        _show(state, max_joint_idx)

        last_cmd_time = time.monotonic()
        last_display_time = 0.0
        display_dt = (
            0.1  # Update display at 10 Hz (avoid costly get_joint_pos every cycle)
        )
        rel_joint_pos = np.zeros(num_joints)

        while True:
            loop_start = time.monotonic()

            # Process buffered keys
            quit_requested = False
            for key, key_time in reader.drain():
                if ord(key) == 3 or key == "q":  # Ctrl+C or q
                    quit_requested = True
                    break
                elif key == "w":
                    state.motion_direction = 1
                    state.last_move_key_time = key_time
                elif key == "s":
                    state.motion_direction = -1
                    state.last_move_key_time = key_time
                elif key.isdigit() and int(key) <= max_joint_idx:
                    state.joint_idx = int(key)
                    state.motion_direction = 0
                    state.current_pos_deg = np.rad2deg(comp_obj.get_joint_pos())
                    _show(state, max_joint_idx)

            if quit_requested:
                print("\nQuitting...")
                break

            # Auto-stop on key release (no repeat events within timeout)
            now = time.monotonic()
            if (
                state.motion_direction != 0
                and (now - state.last_move_key_time) > HOLD_TIMEOUT
            ):
                state.motion_direction = 0
                state.current_pos_deg = np.rad2deg(comp_obj.get_joint_pos())
                _show(state, max_joint_idx)

            # Send motion command at the control rate (no blocking I/O here)
            if state.motion_direction != 0 and (now - last_cmd_time) >= control_dt:
                rel_joint_pos[:] = 0.0
                rel_joint_pos[state.joint_idx] = np.deg2rad(
                    step_size * state.motion_direction
                )
                comp_obj.move_joint_pos(rel_joint_pos, relative=True)
                last_cmd_time = now

            # Update display at a lower rate to avoid get_joint_pos bottleneck
            if state.motion_direction != 0 and (now - last_display_time) >= display_dt:
                state.current_pos_deg = np.rad2deg(comp_obj.get_joint_pos())
                _show(state, max_joint_idx)
                last_display_time = now

            # Sleep remainder of control period
            elapsed = time.monotonic() - loop_start
            sleep_time = control_dt - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)

    except KeyboardInterrupt:
        print("\nControl interrupted by user")
    finally:
        reader.stop()
        signal.signal(signal.SIGINT, original_sigint)
        print("\033[?25h", end="", flush=True)  # Show cursor


def main(
    component: Literal["left_arm", "right_arm", "torso", "head"],
    step_size: float = 2.0,
    control_rate: float = 50.0,
) -> None:
    """Keyboard joint control for robot components.

    Args:
        component: Component to control ("left_arm", "right_arm", "torso", "head")
        step_size: Step size in degrees per control command (default: 2.0)
        control_rate: Control command frequency in Hz (default: 50.0)
    """
    logger.warning("Be ready to press e-stop if needed!")
    logger.warning("Please ensure the robot has enough space to move safely.")

    bot = Robot()
    try:
        control_joints(bot, component, step_size, control_rate)
    except Exception as e:
        logger.error(f"Error during joint control: {e}")
    finally:
        bot.shutdown()
        logger.info("Robot disconnected")


if __name__ == "__main__":
    tyro.cli(main)

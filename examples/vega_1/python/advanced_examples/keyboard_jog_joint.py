# Copyright (C) 2026 Dexmate Inc.
# SPDX-License-Identifier: Apache-2.0

"""Jog a selected joint using keyboard commands.

Default component is left_arm and selected joint is 0. w/s move in opposite directions, digits select a joint, and q exits. Interaction and command increments differ by language; see the language notes.

Raw-terminal hold/repeat w/s input at 100 Hz and 0.2 rad/s. A tap moves the joint by up to 0.05 rad; a hold keeps moving and stops within 0.05 rad of the release. Runs until q/Ctrl-C; no duration option.

Uses a Robot control connection: compatible arm modes may be enabled during startup, and shutdown requests a stop. Real hardware is selected unless --simulated is supplied."""

import select
import sys
import termios
import threading
import time
import tty
from dataclasses import dataclass
from typing import Optional

import numpy as np
import tyro
from dexcontrol import RateLimiter, Robot
from typing_extensions import Annotated

#: Terminal auto-repeat is typically 30-50 ms; this is a safe margin above
#: that, and the longest the joint keeps moving after the key is released.
HOLD_TIMEOUT = 0.15

#: How far the commanded setpoint may run ahead of the measured position
#: while a key is held.
MAX_LEAD = 0.2  # rad

#: Remaining travel kept when the key is released. Latching the setpoint to
#: the measured position instead cancelled a single tap outright: in the
#: 150 ms before the release is inferred the arm has barely started moving,
#: so the joint was pulled back to where it already was. This is also how
#: far a held joint can still travel after release.
RELEASE_LEAD = 0.05  # rad


class KeyboardReader:
    """Non-blocking single-keypress reader over a raw terminal."""

    def __init__(self) -> None:
        self._keys: list[tuple[str, float]] = []
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._saved: list | None = None

    def __enter__(self) -> "KeyboardReader":
        # Raw mode needs a terminal. Piped input (tests, scripted runs) is
        # read as-is, like the C++ example: a piped "q" still exits.
        if sys.stdin.isatty():
            self._saved = termios.tcgetattr(sys.stdin.fileno())
            tty.setraw(sys.stdin.fileno())
        threading.Thread(target=self._loop, daemon=True).start()
        return self

    def __exit__(self, *exc: object) -> None:
        self._stop.set()
        if self._saved is not None:
            termios.tcsetattr(sys.stdin.fileno(), termios.TCSADRAIN, self._saved)

    def drain(self) -> list[tuple[str, float]]:
        with self._lock:
            keys, self._keys = self._keys, []
        return keys

    def _loop(self) -> None:
        while not self._stop.is_set():
            # Poll so the thread notices _stop instead of parking forever in
            # read() and holding the terminal in raw mode after we exit.
            if not select.select([sys.stdin], [], [], 0.1)[0]:
                continue
            key = sys.stdin.read(1)
            if key:
                with self._lock:
                    self._keys.append((key, time.monotonic()))


@dataclass
class Args:
    """Command-line options."""

    component: str = "left_arm"
    joint: Annotated[int, tyro.conf.arg(help="initially selected joint")] = 0
    speed: Annotated[float, tyro.conf.arg(help="rad/s while held")] = 0.2
    control_hz: float = 100.0
    simulated: Annotated[bool, tyro.conf.arg(help="Use the in-process simulation")] = (
        False
    )
    profile: Annotated[Optional[str], tyro.conf.arg(help="Built-in robot profile")] = (
        None
    )
    config: Annotated[
        Optional[str], tyro.conf.arg(help="Custom robot configuration")
    ] = None


def main() -> None:
    args = tyro.cli(Args, description=__doc__, config=(tyro.conf.FlagCreatePairsOff,))
    if args.speed <= 0.0 or args.control_hz <= 0.0:
        raise SystemExit("--speed and --control-hz must be positive")

    with Robot(
        simulation=args.simulated, profile=args.profile, config_file=args.config
    ) as robot:
        component = robot.joints(args.component)
        names = list(component.joint_names)
        lower, upper = np.asarray(component.joint_pos_limit, dtype=float).T
        # Never jog faster than the joint itself allows.
        speed = np.minimum(args.speed, np.asarray(component.joint_vel_limit, float))
        if not 0 <= args.joint < len(names):
            raise SystemExit(f"--joint must be in [0, {len(names) - 1}]")

        target = np.asarray(component.get_joint_pos(), dtype=float)
        selected = args.joint
        direction = 0
        last_key_time = 0.0

        print(f"{args.component}: {len(names)} joints, {args.speed} rad/s")
        print("  w / s      hold to move joint +/-")
        print(f"  0-{len(names) - 1}        select joint")
        print("  q          quit")

        limiter = RateLimiter(args.control_hz)
        step = speed / args.control_hz
        with KeyboardReader() as keyboard:
            while True:
                for key, when in keyboard.drain():
                    if key == "q" or ord(key) == 3:  # q or Ctrl-C
                        # Raw mode: a bare newline drops a line without
                        # returning the carriage, so say \r\n explicitly.
                        print("\r\nstopping\r")
                        return
                    if key == "w":
                        direction, last_key_time = 1, when
                    elif key == "s":
                        direction, last_key_time = -1, when
                    elif key.isdigit() and int(key) < len(names):
                        selected, direction = int(key), 0
                        # Re-seed: the old setpoint belongs to the old joint.
                        target = np.asarray(component.get_joint_pos(), dtype=float)

                now = time.monotonic()
                measured = np.asarray(component.get_joint_pos(), dtype=float)
                if direction and now - last_key_time > HOLD_TIMEOUT:
                    direction = 0
                    # Cut the lead once on release, then hold that setpoint.
                    # Re-reading every idle tick would follow the joint's own
                    # sag under gravity and command it progressively downward.
                    np.clip(
                        target,
                        measured - RELEASE_LEAD,
                        measured + RELEASE_LEAD,
                        out=target,
                    )

                if direction:
                    target[selected] += direction * step[selected]
                np.clip(
                    target,
                    np.maximum(lower, measured - MAX_LEAD),
                    np.minimum(upper, measured + MAX_LEAD),
                    out=target,
                )
                component.set_joint_pos(target)

                arrow = {1: "+", -1: "-", 0: " "}[direction]
                print(
                    f"\r[{arrow}] {names[selected]:>10} "
                    f"{measured[selected]:+.3f} rad "
                    f"(limit {lower[selected]:+.2f} .. {upper[selected]:+.2f})   ",
                    end="",
                    flush=True,
                )
                limiter.sleep()


if __name__ == "__main__":
    main()

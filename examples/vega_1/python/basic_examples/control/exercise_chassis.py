# Copyright (C) 2026 Dexmate Inc.
# SPDX-License-Identifier: Apache-2.0

"""Drive the chassis in six directions, requesting a stop between them.

Command forward, backward, left, right, counter-clockwise, then clockwise for 3 s each, streaming commands at 50 Hz by default (--control-hz). Linear speed is 0.1 m/s and turn speed is 0.1 rad/s. Request stops between directions and on cleanup; requires a base capable of strafing.

Uses a Robot control connection: compatible arm modes may be enabled during startup, and shutdown requests a stop. Real hardware is selected unless --simulated is supplied."""

from dataclasses import dataclass
from typing import Optional

import tyro
from dexcontrol import Robot
from typing_extensions import Annotated


@dataclass
class Args:
    """Command-line options."""

    speed: float = 0.1
    duration: float = 3.0
    control_hz: float = 50.0
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
    if args.duration < 0:
        raise SystemExit("--duration must be non-negative")

    commands = (
        ("forward", (args.speed, 0.0, 0.0)),
        ("backward", (-args.speed, 0.0, 0.0)),
        ("left", (0.0, args.speed, 0.0)),
        ("right", (0.0, -args.speed, 0.0)),
        ("counter-clockwise", (0.0, 0.0, args.speed)),
        ("clockwise", (0.0, 0.0, -args.speed)),
    )
    with Robot(
        simulation=args.simulated, profile=args.profile, config_file=args.config
    ) as robot:
        if not robot.has_component("chassis"):
            raise SystemExit("this robot has no chassis")
        chassis = robot.chassis
        print(
            f"Chassis exercise | {len(commands)} phases | {args.control_hz:g} Hz\n"
            "Commands in robot frame: +vx forward, +vy left, +wz counter-clockwise.\n"
            "Each phase includes steering alignment if needed, then timed driving.",
            flush=True,
        )
        try:
            for index, (label, velocity) in enumerate(commands, 1):
                vx, vy, wz = velocity
                print(
                    f"[{index}/{len(commands)}] {label} | {args.duration:.2f} s | "
                    f"vx={vx:+.3f} m/s, vy={vy:+.3f} m/s, wz={wz:+.3f} rad/s",
                    flush=True,
                )
                chassis.drive_for(
                    *velocity, duration=args.duration, control_hz=args.control_hz
                )
                print("      Command stream complete; stop sent.", flush=True)
        finally:
            chassis.stop()


if __name__ == "__main__":
    main()

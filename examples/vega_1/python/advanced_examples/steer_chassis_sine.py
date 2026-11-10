# Copyright (C) 2026 Dexmate Inc.
# SPDX-License-Identifier: Apache-2.0

"""Drive forward while sinusoidally changing both steering angles.

For 6 s at 50 Hz, command both wheel speeds to 0.5 m/s and both steering angles to 0.6*sin(2*pi*0.5*t) rad. Request a stop on completion/cleanup. Requires a steer-drive base; it does not use generic chassis yaw velocity.

Uses a Robot control connection: compatible arm modes may be enabled during startup, and shutdown requests a stop. Real hardware is selected unless --simulated is supplied."""

import math
import time
from dataclasses import dataclass
from typing import Optional

import tyro
from dexcontrol import Robot
from typing_extensions import Annotated


@dataclass
class Args:
    """Command-line options."""

    amplitude: float = 0.6
    frequency: float = 0.5
    speed: Annotated[float, tyro.conf.arg(help="m/s")] = 0.5
    duration: float = 6.0
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

    with Robot(
        simulation=args.simulated, profile=args.profile, config_file=args.config
    ) as robot:
        if not robot.has_component("chassis"):
            raise SystemExit("this robot has no chassis")
        chassis = robot.chassis
        started = time.monotonic()
        period = 1.0 / args.control_hz
        try:
            while (elapsed := time.monotonic() - started) < args.duration:
                steering = args.amplitude * math.sin(
                    2.0 * math.pi * args.frequency * elapsed
                )
                chassis.set_motion_state([steering, steering], [args.speed, args.speed])
                time.sleep(period)
        finally:
            chassis.stop()


if __name__ == "__main__":
    main()

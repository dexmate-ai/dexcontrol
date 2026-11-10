# Copyright (C) 2026 Dexmate Inc.
# SPDX-License-Identifier: Apache-2.0

"""Read, enable, or disable arm force-torque sensor modes.

Require positional get, enable, or disable. Address both arms by default and print service replies; this does not zero the force sensors.

Uses a Robot control connection: compatible arm modes may be enabled during startup, and shutdown requests a stop. Real hardware is selected unless --simulated is supplied."""

import pprint
from dataclasses import dataclass
from typing import Literal, Optional

import tyro
from dexcontrol import Robot
from typing_extensions import Annotated


@dataclass
class Args:
    """Command-line options."""

    action: Annotated[Literal["get", "enable", "disable"], tyro.conf.Positional]
    side: Literal["left", "right", "both"] = "both"
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
    sides = ("left", "right") if args.side == "both" else (args.side,)
    with Robot(
        simulation=args.simulated, profile=args.profile, config_file=args.config
    ) as robot:
        for side in sides:
            arm = robot.joints(f"{side}_arm")
            result = (
                arm.get_force_torque_sensor_mode()
                if args.action == "get"
                else arm.activate_force_torque_sensor(args.action == "enable")
            )
            print(f"{side}_arm")
            pprint.pp(result, sort_dicts=False)


if __name__ == "__main__":
    main()

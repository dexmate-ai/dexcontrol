# Copyright (C) 2026 Dexmate Inc.
# SPDX-License-Identifier: Apache-2.0

"""Read or set end-effector RS485 baud rates.

Require positional get or set. Address both arm end-effector interfaces by default; set uses 115200 baud unless overridden. Print service replies.

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

    action: Annotated[Literal["get", "set"], tyro.conf.Positional]
    side: Literal["left", "right", "both"] = "both"
    baud_rate: int = 115200
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
                arm.get_ee_baud_rate()
                if args.action == "get"
                else arm.set_ee_baud_rate(args.baud_rate)
            )
            print(f"{side}_arm")
            pprint.pp(result, sort_dicts=False)


if __name__ == "__main__":
    main()

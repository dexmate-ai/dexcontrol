# Copyright (C) 2026 Dexmate Inc.
# SPDX-License-Identifier: Apache-2.0

"""Close one hand, then reopen it.

Command the right hand to its model-defined closed pose, wait 2 s, command its open pose, then wait 2 s. These delays are not tracked-motion completion checks.

Uses a Robot control connection: compatible arm modes may be enabled during startup, and shutdown requests a stop. Real hardware is selected unless --simulated is supplied."""

from dataclasses import dataclass
from typing import Literal, Optional

import tyro
from dexcontrol import Robot
from typing_extensions import Annotated


@dataclass
class Args:
    """Command-line options."""

    side: Literal["left", "right"] = "right"
    wait_time: float = 2.0
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
        hand = robot.joints(f"{args.side}_hand")
        hand.close_hand(wait_time=args.wait_time)
        hand.open_hand(wait_time=args.wait_time)


if __name__ == "__main__":
    main()

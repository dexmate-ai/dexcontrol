# Copyright (C) 2026 Dexmate Inc.
# SPDX-License-Identifier: Apache-2.0

"""Open one hand using its model-defined pose.

Command the right hand open and wait 2 s. Optional --grasp-torque changes the grip setting for a gripper; omitted values use model defaults. The delay is not a convergence check.

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
    grasp_torque: Annotated[
        Optional[float],
        tyro.conf.arg(
            help="normalized grip force for grippers; the model's value is used when omitted, and a stalled gripper can damage its motor above 0.5"
        ),
    ] = None
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
        # Grippers are torque-commanded; five-finger hands are not, and report
        # None here rather than accepting a grip force.
        if args.grasp_torque is not None and hand.gripper_torque is not None:
            print(f"grasp torque {hand.set_gripper_torque(args.grasp_torque)}")
        hand.open_hand(wait_time=args.wait_time)


if __name__ == "__main__":
    main()

# Copyright (C) 2026 Dexmate Inc.
# SPDX-License-Identifier: Apache-2.0

"""Move arms sequentially to a named pose without collision checking.

Move the left arm and then the right arm to L_shape at velocity scale 0.5, waiting up to 10 s each. --side can select one arm. Resolve model-declared pose frames, including torso compensation for L_shape. No homing or collision planning is performed.

Uses a Robot control connection: compatible arm modes may be enabled during startup, and shutdown requests a stop. Real hardware is selected unless --simulated is supplied."""

from dataclasses import dataclass
from typing import Literal, Optional

import tyro
from dexcontrol import Robot
from typing_extensions import Annotated


@dataclass
class Args:
    """Command-line options."""

    side: Literal["left", "right", "both"] = "both"
    pose: str = "L_shape"
    velocity_scale: float = 0.5
    timeout: float = 10.0
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
            target = arm.resolve_pose(args.pose)
            motion = arm.move_to_joint_pos(
                target, velocity_scale=args.velocity_scale, wait=False
            )
            print(f"{side}_arm: {motion.wait(timeout=args.timeout)}")


if __name__ == "__main__":
    main()

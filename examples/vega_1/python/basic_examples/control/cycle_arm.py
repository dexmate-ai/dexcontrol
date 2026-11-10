# Copyright (C) 2026 Dexmate Inc.
# SPDX-License-Identifier: Apache-2.0

"""Move one arm joint by an offset, then return to its starting position.

Save the right arm positions, move joint 0 by +0.2 rad, then return to the saved absolute start after outward success. By default compute an absolute outward target; --relative resolves the offset from fresh feedback. Both motions use velocity scale 0.2 and a 10 s wait deadline.

Uses a Robot control connection: compatible arm modes may be enabled during startup, and shutdown requests a stop. Real hardware is selected unless --simulated is supplied."""

from dataclasses import dataclass
from typing import Literal, Optional

import numpy as np
import tyro
from dexcontrol import Robot
from typing_extensions import Annotated


@dataclass
class Args:
    """Command-line options."""

    side: Literal["left", "right"] = "right"
    joint: int = 0
    delta: float = 0.2
    relative: bool = False
    velocity_scale: float = 0.2
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

    with Robot(
        simulation=args.simulated, profile=args.profile, config_file=args.config
    ) as robot:
        arm = robot.joints(f"{args.side}_arm")
        start = arm.get_joint_pos()
        target = np.zeros_like(start) if args.relative else start.copy()
        if not 0 <= args.joint < len(target):
            raise SystemExit(f"--joint must be in [0, {len(target) - 1}]")
        target[args.joint] += args.delta
        motion = arm.move_to_joint_pos(
            target, relative=args.relative, velocity_scale=args.velocity_scale, wait=False
        )
        motion.wait_success(timeout=args.timeout)
        arm.move_to_joint_pos(
            start, velocity_scale=args.velocity_scale, wait=False
        ).wait_success(timeout=args.timeout)


if __name__ == "__main__":
    main()

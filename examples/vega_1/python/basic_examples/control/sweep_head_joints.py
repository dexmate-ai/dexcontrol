# Copyright (C) 2026 Dexmate Inc.
# SPDX-License-Identifier: Apache-2.0

"""Sweep head joints through positive, negative, and zero targets.

First command head joint 0 to -pi/6 rad with other joints zero. Then command each joint to +0.5, -0.5, and zero radians, with all other target joints zero. Finish at all zeros; wait up to 10 s per motion.

The sequence assumes exactly three head joints.

Uses a Robot control connection: compatible arm modes may be enabled during startup, and shutdown requests a stop. Real hardware is selected unless --simulated is supplied."""

import math
from dataclasses import dataclass
from typing import Optional

import tyro
from dexcontrol import Robot
from typing_extensions import Annotated


def move_joint_sequence(head, joint: int, delta: float, timeout: float) -> None:
    """Move one head joint positive, negative, and back to zero."""
    positive = [0.0, 0.0, 0.0]
    negative = [0.0, 0.0, 0.0]
    positive[joint] = delta
    negative[joint] = -delta

    for target in (positive, negative, [0.0, 0.0, 0.0]):
        head.move_to_joint_pos(target, wait=False).wait(timeout=timeout)


@dataclass
class Args:
    """Command-line options."""

    delta: float = 0.5
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
        head = robot.joints("head")
        head.move_to_joint_pos([-math.pi / 6.0, 0.0, 0.0], wait=False).wait(
            timeout=args.timeout
        )
        for joint in range(3):
            move_joint_sequence(head, joint, args.delta, args.timeout)


if __name__ == "__main__":
    main()

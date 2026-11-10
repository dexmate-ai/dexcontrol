# Copyright (C) 2026 Dexmate Inc.
# SPDX-License-Identifier: Apache-2.0

"""Move the torso to an offset joint target, then return to its starting position.

Read the torso positions and move joint 0 by +0.1 rad with a planned tracked motion. By default compute an absolute target from the stored start; --relative lets the API resolve the offset against fresh feedback. Always return to the stored absolute start after outward success. Each motion wait has a 10 s deadline.

Uses a Robot control connection: compatible arm modes may be enabled during startup, and shutdown requests a stop. Real hardware is selected unless --simulated is supplied."""

from dataclasses import dataclass
from typing import Optional

import numpy as np
import tyro
from dexcontrol import Robot
from typing_extensions import Annotated


@dataclass
class Args:
    """Command-line options."""

    joint: int = 0
    delta: float = 0.1
    timeout: float = 10.0
    relative: bool = False
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
        torso = robot.joints("torso")
        start = torso.get_joint_pos()
        if not 0 <= args.joint < len(start):
            raise SystemExit(f"--joint must be in [0, {len(start) - 1}]")
        target = np.zeros_like(start) if args.relative else start.copy()
        target[args.joint] += args.delta
        torso.move_to_joint_pos(target, relative=args.relative, wait=True, timeout=args.timeout)
        torso.move_to_joint_pos(start, wait=True, timeout=args.timeout)


if __name__ == "__main__":
    main()

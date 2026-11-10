# Copyright (C) 2026 Dexmate Inc.
# SPDX-License-Identifier: Apache-2.0

"""Move the head to a named pose using its model-declared reference frame.

Move to home using model-defined torso-pitch compensation; tucked stays joint-relative. --passthrough uses raw stored values. Wait up to 10 s. No collision checking is performed.

Uses a Robot control connection: compatible arm modes may be enabled during startup, and shutdown requests a stop. Real hardware is selected unless --simulated is supplied."""

from dataclasses import dataclass
from typing import Optional

import tyro
from dexcontrol import Robot
from typing_extensions import Annotated


@dataclass
class Args:
    """Command-line options."""

    pose: str = "home"
    timeout: float = 10.0
    passthrough: bool = False
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
        target = head.get_pose(args.pose, passthrough=args.passthrough)
        print(head.move_to_joint_pos(target, wait=False).wait(timeout=args.timeout))


if __name__ == "__main__":
    main()

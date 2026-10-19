# Copyright (C) 2026 Dexmate Inc.
#
# This software is dual-licensed:
#
# 1. GNU Affero General Public License v3.0 (AGPL-3.0)
#    See LICENSE-AGPL for details
#
# 2. Commercial License
#    For commercial licensing terms, contact: contact@dexmate.ai

"""Move arms to a named pose using their model-declared reference frames.

Move the right arm to L_shape with model-defined torso-pitch compensation; wait up to 10 s. folded and folded_closed_hand stay joint-relative and receive no compensation. zero stores seven zeros relative to an upright torso and compensates only torso tilt from upright; --passthrough returns literal zeros. --passthrough uses raw stored values while retaining target validation. No collision checking is performed.

--side both resolves both arm targets before starting a group motion and waits up to --timeout seconds for the group. Starts are not synchronized.

Uses a Robot control connection: compatible arm modes may be enabled during startup, and shutdown requests a stop. Real hardware is selected unless --simulated is supplied."""

from dataclasses import dataclass
from typing import Literal, Optional

import tyro
from dexcontrol import Robot
from typing_extensions import Annotated


@dataclass
class Args:
    """Command-line options."""

    side: Literal["left", "right", "both"] = "right"
    pose: str = "L_shape"
    passthrough: bool = False
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
        sides = ("left", "right") if args.side == "both" else (args.side,)
        targets = {
            f"{side}_arm": robot.joints(f"{side}_arm").get_pose(
                args.pose, passthrough=args.passthrough
            )
            for side in sides
        }
        if args.side == "both":
            motion = robot.move_to_joint_pos(targets, wait=False)
        else:
            name = f"{args.side}_arm"
            motion = robot.joints(name).move_to_joint_pos(targets[name], wait=False)
        print(motion.wait(timeout=args.timeout))


if __name__ == "__main__":
    main()

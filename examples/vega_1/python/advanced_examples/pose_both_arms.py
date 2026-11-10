# Copyright (C) 2026 Dexmate Inc.
# SPDX-License-Identifier: Apache-2.0

"""Move both arms to a named pose using independent tracked motions.

Default to folded at velocity scale 0.5. Resolve each pose using its model-declared frame: folded poses are joint-relative; orientation reference poses use torso pitch. --passthrough uses raw stored values. Start both targets through best-effort fan-out, wait up to 10 s, and require aggregate success. No synchronized start or collision checking is provided.

Uses a Robot control connection: compatible arm modes may be enabled during startup, and shutdown requests a stop. Real hardware is selected unless --simulated is supplied."""

from dataclasses import dataclass
from typing import Optional

import tyro
from dexcontrol import Robot
from typing_extensions import Annotated


@dataclass
class Args:
    """Command-line options."""

    pose: str = "folded"
    velocity_scale: float = 0.5
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
        targets = {}
        for name in ("left_arm", "right_arm"):
            arm = robot.joints(name)
            target = arm.get_pose(args.pose, passthrough=args.passthrough)
            targets[name] = target
        motion = robot.move_to_joint_pos(
            targets, velocity_scale=args.velocity_scale, wait=False
        )
        state = motion.wait(timeout=args.timeout)
        # Reporting the state without checking it exits 0 on a failed fold,
        # which reads as success to anything scripting this.
        if state != "finished":
            raise SystemExit(f"fold reported {state!r} within {args.timeout:g}s")
        print(state)


if __name__ == "__main__":
    main()

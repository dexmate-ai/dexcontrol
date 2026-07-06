#!/usr/bin/env python3
# Copyright (C) 2025 Dexmate Inc.
#
# This software is dual-licensed:
#
# 1. GNU Affero General Public License v3.0 (AGPL-3.0)
#    See LICENSE-AGPL for details
#
# 2. Commercial License
#    For commercial licensing terms, contact: contact@dexmate.ai

"""Example script for moving a robot arm to a predefined pose.

This script demonstrates how to move either the left or right arm
to a predefined pose from the pose library, with an option to compensate
for torso pitch. It includes safety prompts and proper shutdown procedures.
"""

import tyro
from loguru import logger

from dexcontrol.robot import Robot


def main(
    pose: str = "L_shape",
    side: str = "left",
    comp_pitch: bool = False,
) -> None:
    """Moves the specified robot arm to a predefined pose.

    Args:
        pose: Name of the predefined pose to move to.
        side: Which arm to move ('left' or 'right').
        comp_pitch: Whether to compensate for torso pitch angle.

    Raises:
        ValueError: If side is not 'left' or 'right'.
    """
    # Validate input parameters
    if side not in ["left", "right"]:
        raise ValueError("side must be either 'left' or 'right'")

    # Safety confirmation
    logger.warning(
        "Warning: Be ready to press e-stop if needed! "
        "This example does not check for self-collisions."
    )
    logger.warning("Please ensure the arms and torso have sufficient space to move.")
    if input("Continue? [y/N]: ").lower() != "y":
        return

    # Initialize robot and get specified arm
    bot = Robot()
    arm = bot.left_arm if side == "left" else bot.right_arm

    try:
        logger.info(f"Moving {side} arm to {pose} position")
        arm_pose = arm.get_predefined_pose(pose)
        if comp_pitch:
            arm_pose = bot.compensate_torso_pitch(
                arm_pose, "left_arm" if side == "left" else "right_arm"
            )
        handle = arm.move_to_joint_pos(arm_pose)
        assert handle is not None
        result = handle.wait(timeout=6.0)
        if result != "finished":
            logger.error(
                f"Motion ended with state: {result}, message: {handle.message}"
            )
    finally:
        logger.info("Shutting down robot")
        bot.shutdown()


if __name__ == "__main__":
    tyro.cli(main)

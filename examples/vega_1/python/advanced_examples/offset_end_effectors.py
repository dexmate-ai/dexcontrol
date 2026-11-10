# Copyright (C) 2025-2026 Dexmate Inc.
# SPDX-License-Identifier: Apache-2.0

"""Execute four planned relative end-effector moves with Enter prompts.

Using dexmotion at 250 Hz: move the right end effector +0.1 m in z in frame R_arm_j5; move the left +0.1 m in y; move the left +0.1 m in x in frame L_arm_j3; rotate the right +pi/4 around x. Prompt before each segment and stream both arm joint arrays. Uses a real Robot control connection; no simulation flag."""

from typing import Optional

import numpy as np
import tyro
from dexmotion.motion_manager import MotionManager

from dexcontrol import RateLimiter, Robot, ask


def flat_joint_pos(bot: Robot, components: list[str]) -> dict[str, float]:
    """Return ``{joint_name: position}`` for the given components.

    ``Robot.get_joint_pos_dict(component=[...])`` nests positions by component
    (``{"left_arm": {"L_arm_j1": ...}}``); dexmotion wants one flat joint map.
    """
    nested = bot.get_joint_pos_dict(component=components)
    return {joint: pos for joints in nested.values() for joint, pos in joints.items()}


def move_real_robot(
    bot: Robot, qs_sample: list[list[float]], control_hz: float = 250
) -> None:
    """Move the real robot to a specific pose.

    Args:
        qs_sample: List of joint arrays to move
    """
    rate_limiter = RateLimiter(control_hz)
    for q in qs_sample:
        # Extract left and right arm joint arrays
        left_arm_joints = q[:7]
        right_arm_joints = q[7:14]

        # Set joint positions for both arms
        bot.left_arm.set_joint_pos(left_arm_joints)
        bot.right_arm.set_joint_pos(right_arm_joints)
        rate_limiter.sleep()


def main(profile: Optional[str] = None) -> None:
    """Execute four planned relative end-effector moves with Enter prompts.

    Using dexmotion at 250 Hz: move the right end effector +0.1 m in z in frame R_arm_j5; move the left +0.1 m in y; move the left +0.1 m in x in frame L_arm_j3; rotate the right +pi/4 around x. Prompt before each segment and stream both arm joint arrays. Uses a real Robot control connection; no simulation flag.

    Args:
        profile: Built-in robot profile (e.g. vega_1p_gripper). Defaults to the
            ROBOT_CONFIG environment variable. Also used as the dexmotion robot_type.
    """
    # Every other example uses the context manager; without it the session,
    # safety monitors and watchdog are never torn down on exit.
    with Robot(profile=profile) as bot:
        control_hz = 250
        components = ["left_arm", "right_arm", "head"]
        if bot.has_component("torso"):
            components.insert(2, "torso")
        initial_joint_pos = flat_joint_pos(bot, components)

        # Plan on the same robot model the connection resolved (flag or ROBOT_CONFIG).
        mm = MotionManager(
            robot_type=bot.profile_name,
            initial_joint_configuration_dict=initial_joint_pos,
        )

        moves = (
            (
                "move right end-effector by 0.1 meters in z direction",
                lambda: mm.right_arm.set_ee_pose(
                    pos=np.array([0.0, 0.0, 0.1]), relative=True, target_frame="R_arm_j5"
                ),
            ),
            (
                "move left end-effector by 0.1 meters in y direction",
                lambda: mm.left_arm.set_ee_pose(
                    pos=np.array([0.0, 0.1, 0.0]), relative=True
                ),
            ),
            (
                "move left end-effector by 0.1 meters in x direction",
                lambda: mm.left_arm.move_ee_xyz(
                    np.array([0.1, 0.0, 0.0]), target_frame="L_arm_j3"
                ),
            ),
            (
                "rotate right end-effector by 45 degrees around x-axis",
                lambda: mm.right_arm.move_ee_rpy(np.array([np.pi / 4, 0.0, 0.0])),
            ),
        )
        for description, plan in moves:
            qs_sample = plan()
            ask(f"Press Enter to {description}")
            move_real_robot(bot, qs_sample, control_hz)


if __name__ == "__main__":
    tyro.cli(main)

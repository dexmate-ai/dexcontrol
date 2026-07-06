# Copyright (C) 2025 Dexmate Inc.
#
# This software is dual-licensed:
#
# 1. GNU Affero General Public License v3.0 (AGPL-3.0)
#    See LICENSE-AGPL for details
#
# 2. Commercial License
#    For commercial licensing terms, contact: contact@dexmate.ai

"""Example script to control arm brake release.

This script demonstrates how to control brake release (over-limit drag)
for calibration or recovery purposes.

WARNING: Use with caution. When brake is released,
the arm may move unexpectedly if not properly supported.
"""

import logging
from typing import Literal

import tyro

from dexcontrol.robot import Robot

logger = logging.getLogger(__name__)


def brake(
    side: Literal["left", "right", "both"] = "left",
    enable: bool = True,
    joints: list[int] | None = None,
) -> None:
    """Control arm brake release (over-limit drag).

    When brake release is enabled, the arm can be manually moved beyond
    normal position limits for calibration or recovery purposes.

    Args:
        side: Which arm to control ('left', 'right', or 'both').
        enable: True to enable brake release, False to disable.
        joints: Optional list of joint indices (0-6) to operate on.
            If None, operates on all joints.
        show_status: Whether to show brake status after operation.
    """
    with Robot() as bot:
        if side in ("left", "both"):
            logger.info(
                f"{'Enabling' if enable else 'Disabling'} brake release for left arm..."
            )
            result = bot.left_arm.release_brake(enable, joints)
            logger.info(f"Left arm result: {result}")

        if side in ("right", "both"):
            logger.info(
                f"{'Enabling' if enable else 'Disabling'} brake release for right arm..."
            )
            result = bot.right_arm.release_brake(enable, joints)
            logger.info(f"Right arm result: {result}")


if __name__ == "__main__":
    tyro.cli(brake)

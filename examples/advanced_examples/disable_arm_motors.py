# Copyright (C) 2025 Dexmate Inc.
#
# This software is dual-licensed:
#
# 1. GNU Affero General Public License v3.0 (AGPL-3.0)
#    See LICENSE-AGPL for details
#
# 2. Commercial License
#    For commercial licensing terms, contact: contact@dexmate.ai

"""Example script to disable arm motors and control brake release.

This script demonstrates how to:
1. Disable arm motors with optional brake release mode
2. Control brake release (over-limit drag) for calibration or recovery

Recovery when a joint is stuck at limit (move arm back manually):
  1. Release brake on the stuck joint(s):
     python disable_arm_motors.py disable --side left --joint-idx 2 --release-brake
  2. Manually move the arm to a safe position (within limits).
  3. Re-engage brake:
     python disable_arm_motors.py brake --side left --enable False
  4. Re-enable motors if you used disable without --release-brake:
     (Robot reconnection or position-mode command will re-enable.)

WARNING: Use with caution. When motors are disabled or brake is released,
the arm may move unexpectedly if not properly supported.
"""

import logging
import sys
from typing import Literal

import tyro

from dexcontrol.exceptions import ServiceUnavailableError
from dexcontrol.robot import Robot

logger = logging.getLogger(__name__)


def _handle_brake_error(side: str, for_both: bool) -> None:
    """On brake service unavailable: log and exit or continue when --side both."""
    msg = (
        f"Brake service not available for {side} arm. "
        "This robot may have only one arm, or firmware may not expose the service."
    )
    if for_both:
        logger.warning(msg + " Skipping this arm.")
    else:
        logger.error(msg)
        logger.info("If this is a single-arm robot (e.g. vega_1p), try: --side left")
        sys.exit(1)


def disable(
    side: Literal["left", "right", "both"],
    joint_idx: list[int],
    release_brake: bool = False,
) -> None:
    """Disable arm motors with brake released or not.

    Args:
        side: Side of the arm to disable.
        joint_idx: Joint indices to disable.
        release_brake: Whether to release the brake (allows manual movement).
    """
    with Robot() as bot:
        if release_brake:
            # Use release_brake service to allow manual movement
            if side in ("left", "both"):
                try:
                    logger.info(f"Releasing brake for left arm joints: {joint_idx}")
                    result = bot.left_arm.release_brake(enable=True, joints=joint_idx)
                    logger.info(f"Left arm result: {result}")
                except ServiceUnavailableError:
                    _handle_brake_error("left", for_both=(side == "both"))
            if side in ("right", "both"):
                try:
                    logger.info(f"Releasing brake for right arm joints: {joint_idx}")
                    result = bot.right_arm.release_brake(enable=True, joints=joint_idx)
                    logger.info(f"Right arm result: {result}")
                except ServiceUnavailableError:
                    _handle_brake_error("right", for_both=(side == "both"))
        else:
            # Use set_modes to disable motors (keeps brake engaged)
            modes: list[Literal["position", "disable"]] = ["position"] * 7  # type: ignore[assignment]
            for idx in joint_idx:
                modes[idx] = "disable"

            if side in ("left", "both"):
                try:
                    logger.info(f"Disabling left arm joints: {joint_idx}")
                    bot.left_arm.set_modes(modes)
                except ServiceUnavailableError:
                    _handle_brake_error("left", for_both=(side == "both"))
            if side in ("right", "both"):
                try:
                    logger.info(f"Disabling right arm joints: {joint_idx}")
                    bot.right_arm.set_modes(modes)
                except ServiceUnavailableError:
                    _handle_brake_error("right", for_both=(side == "both"))


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
    """
    with Robot() as bot:
        if side in ("left", "both"):
            try:
                logger.info(
                    f"{'Enabling' if enable else 'Disabling'} brake release for left arm..."
                )
                result = bot.left_arm.release_brake(enable, joints)
                logger.info(f"Left arm result: {result}")
            except ServiceUnavailableError:
                _handle_brake_error("left", for_both=(side == "both"))

        if side in ("right", "both"):
            try:
                logger.info(
                    f"{'Enabling' if enable else 'Disabling'} brake release for right arm..."
                )
                result = bot.right_arm.release_brake(enable, joints)
                logger.info(f"Right arm result: {result}")
            except ServiceUnavailableError:
                _handle_brake_error("right", for_both=(side == "both"))


if __name__ == "__main__":
    tyro.extras.subcommand_cli_from_dict({"disable": disable, "brake": brake})

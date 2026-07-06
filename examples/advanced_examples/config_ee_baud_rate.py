# Copyright (C) 2025 Dexmate Inc.
#
# This software is dual-licensed:
#
# 1. GNU Affero General Public License v3.0 (AGPL-3.0)
#    See LICENSE-AGPL for details
#
# 2. Commercial License
#    For commercial licensing terms, contact: contact@dexmate.ai

"""End-effector baud rate configuration.

This script provides commands to query and set the RS485 baud rate
for the end-effector communication on the robot arms.

Usage:
    python config_ee_baud_rate.py get                          # Read baud rate for both arms
    python config_ee_baud_rate.py get --side right             # Read baud rate for right arm
    python config_ee_baud_rate.py set --baud-rate 921600       # Set left arm baud rate
    python config_ee_baud_rate.py set --side both --baud-rate 3000000  # Set both arms
"""

from typing import Literal

import tyro
from loguru import logger

from dexcontrol.robot import Robot

Side = Literal["left", "right", "both"]


def _selected_arms(bot: Robot, side: Side) -> list[tuple[str, object]]:
    """Return (label, arm) pairs for the selected side(s)."""
    arms = {
        "left": ("Left arm: ", bot.left_arm),
        "right": ("Right arm:", bot.right_arm),
    }
    sides = ("left", "right") if side == "both" else (side,)
    return [arms[s] for s in sides]


def _log_baud_rate(label: str, arm) -> None:
    """Query an arm's EE baud rate and log the result."""
    result = arm.get_ee_baud_rate()
    if result.get("success"):
        logger.info(f"{label} {result.get('baud_rate')}")
    else:
        logger.error(f"{label} Error - {result.get('message')}")


def get(side: Side = "both") -> None:
    """Get current end-effector RS485 baud rate.

    Args:
        side: Which arm to query ('left', 'right', or 'both').
    """
    with Robot() as bot:
        for label, arm in _selected_arms(bot, side):
            _log_baud_rate(label, arm)


def set_baud_rate(side: Side = "left", baud_rate: int = 115200) -> None:
    """Set end-effector RS485 baud rate.

    Args:
        side: Which arm to configure ('left', 'right', or 'both').
        baud_rate: Baud rate to set. Common values: 115200, 460800, 921600,
            1000000, 3000000.
    """
    with Robot() as bot:
        arms = _selected_arms(bot, side)

        for label, arm in arms:
            result = arm.set_ee_baud_rate(baud_rate)
            if not result.get("success"):
                logger.error(f"{label} Failed - {result.get('message')}")

        logger.info(
            "To make the new EE baud rate take effect, you need to "
            "reboot the robot after modification."
        )

        logger.info("Verifying new baud rate...")
        for label, arm in arms:
            _log_baud_rate(label, arm)


if __name__ == "__main__":
    tyro.extras.subcommand_cli_from_dict({"get": get, "set": set_baud_rate})

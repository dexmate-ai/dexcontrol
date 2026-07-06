# Copyright (C) 2025 Dexmate Inc.
#
# This software is dual-licensed:
#
# 1. GNU Affero General Public License v3.0 (AGPL-3.0)
#    See LICENSE-AGPL for details
#
# 2. Commercial License
#    For commercial licensing terms, contact: contact@dexmate.ai

"""Example script to get the full state of all robot components.

This script demonstrates how to retrieve the complete state dictionary from each
robot component (arms, hands, head, torso, chassis) using get_state().
"""

import pprint

import numpy as np
import tyro
from loguru import logger

from dexcontrol.core.component import RobotComponent
from dexcontrol.robot import Robot


def _round_floats(obj, precision=5):
    """Recursively round float values in nested dicts/lists."""
    if isinstance(obj, float):
        return round(obj, precision)
    if isinstance(obj, dict):
        return {k: _round_floats(v, precision) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_round_floats(v, precision) for v in obj]
    return obj


def main() -> None:
    """Gets and logs the full state from all robot components."""
    np.set_printoptions(precision=5)
    with Robot() as bot:
        components: dict[str, RobotComponent] = {
            "left_arm": bot.left_arm,
            "right_arm": bot.right_arm,
        }
        if bot.has_component("head"):
            components["head"] = bot.head
        if bot.has_component("torso"):
            components["torso"] = bot.torso
        if bot.has_component("chassis"):
            components["chassis_steer"] = bot.chassis.chassis_steer
            components["chassis_drive"] = bot.chassis.chassis_drive
        if bot.have_hand("left"):
            components["left_hand"] = bot.left_hand
        if bot.have_hand("right"):
            components["right_hand"] = bot.right_hand

        for name, component in components.items():
            state = _round_floats(component.get_state())
            logger.info(f"{name}:\n{pprint.pformat(state)}")


if __name__ == "__main__":
    tyro.cli(main)

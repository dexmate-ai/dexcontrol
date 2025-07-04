# Copyright (C) 2025 Dexmate Inc.
#
# This software is dual-licensed:
#
# 1. GNU Affero General Public License v3.0 (AGPL-3.0)
#    See LICENSE-AGPL for details
#
# 2. Commercial License
#    For commercial licensing terms, contact: contact@dexmate.ai

"""Example script to display robot system information.

This script demonstrates how to query and display various robot status information
including software version, component status, and battery level.
"""

from dexcontrol.robot import Robot


def main() -> None:
    """Display robot system information including version, status and battery.

    Queries the robot for software version, component status, and battery
    information and displays the results.
    """
    # Initialize robot with default configuration
    bot = Robot()

    # Display robot system information
    bot.get_software_version(show=True)
    bot.get_component_status(show=True)
    bot.battery.show()
    bot.estop.show()
    bot.heartbeat.show()
    bot.shutdown()


if __name__ == "__main__":
    main()

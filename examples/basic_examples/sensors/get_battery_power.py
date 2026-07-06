# Copyright (C) 2025 Dexmate Inc.
#
# This software is dual-licensed:
#
# 1. GNU Affero General Public License v3.0 (AGPL-3.0)
#    See LICENSE-AGPL for details
#
# 2. Commercial License
#    For commercial licensing terms, contact: contact@dexmate.ai

"""Example script to display real-time battery power consumption."""

import tyro
from dexcomm import RateLimiter

from dexcontrol.core.config import get_robot_config
from dexcontrol.robot import Robot


def main() -> None:
    """Continuously print battery power consumption until Ctrl+C."""
    configs = get_robot_config()
    with Robot(configs=configs) as bot:
        rate_limiter = RateLimiter(2.0)
        print("Streaming battery power (Ctrl+C to exit)")
        try:
            while True:
                status = bot.battery.get_status()
                print(
                    f"Power: {status['power']:6.2f} W  "
                    f"({status['current']:5.2f} A x {status['voltage']:5.2f} V)",
                    end="\r",
                    flush=True,
                )
                rate_limiter.sleep()
        except KeyboardInterrupt:
            print()


if __name__ == "__main__":
    tyro.cli(main)

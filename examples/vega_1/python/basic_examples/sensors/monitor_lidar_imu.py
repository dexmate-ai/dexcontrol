# Copyright (C) 2026 Dexmate Inc.
#
# This software is dual-licensed:
#
# 1. GNU Affero General Public License v3.0 (AGPL-3.0)
#    See LICENSE-AGPL for details
#
# 2. Commercial License
#    For commercial licensing terms, contact: contact@dexmate.ai

"""Monitor the IMU associated with a 3D LiDAR.

Enable lidar_3d_front, wait up to 10 s for its IMU stream, and print observations continuously at 0.05 s intervals until Ctrl+C. --samples N limits the number of polls. --sensor selects another sensor with an IMU capability.

Uses a Robot control connection: compatible arm modes may be enabled during startup, and shutdown requests a stop. Real hardware is selected unless --simulated is supplied."""

import pprint
import time
from dataclasses import dataclass
from typing import Optional

import tyro
from dexcontrol import Robot
from typing_extensions import Annotated


@dataclass
class Args:
    """Command-line options."""

    sensor: Annotated[
        str, tyro.conf.arg(help="Sensor name with an IMU stream declared by the robot profile")
    ] = "lidar_3d_front"
    samples: Annotated[
        Optional[int], tyro.conf.arg(help="Poll limit; omitted runs until Ctrl+C")
    ] = None
    period: float = 0.05
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
    if args.samples is not None and args.samples < 0:
        raise ValueError("--samples must be non-negative")
    with Robot(
        enable_sensors=[args.sensor],
        simulation=args.simulated,
        profile=args.profile,
        config_file=args.config,
    ) as robot:
        imu = robot.sensors.get_imu(args.sensor)
        if not imu.wait_for_active(timeout=10.0):
            raise SystemExit(f"{args.sensor} did not become active")
        try:
            index = 0
            while args.samples is None or index < args.samples:
                pprint.pp(imu.get_obs(), sort_dicts=False)
                index += 1
                if args.samples is None or index < args.samples:
                    time.sleep(args.period)
        except KeyboardInterrupt:
            pass


if __name__ == "__main__":
    main()

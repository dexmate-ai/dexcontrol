# Copyright (C) 2026 Dexmate Inc.
# SPDX-License-Identifier: Apache-2.0
"""Move the base forward 1 m, then turn 90 degrees, by measured amounts.

Drive straight by --distance metres (default 1.0) with chassis.move_by, then turn in place by --angle degrees (default 90) with chassis.turn_by. Each is a closed loop on the base odometry: the base drives while the odometry is watched, slows down near the target, and corrects only if the settled error is outside the tolerance; --speed overrides the cruise speed. Print the amount measured after each motion, then the odometry pose and its sample counters. The heading comes from an enabled base IMU where the model has one (vega_1: --enable-sensor chassis_imu, optionally --yaw-source chassis_imu), otherwise from the wheels (vega_1p). Stop the base on exit.

Uses a Robot control connection with startup arm initialization turned off; shutdown requests a stop. Real hardware is selected unless --simulated is supplied.
"""

import math
from dataclasses import dataclass, field
from typing import List, Optional

import tyro
from dexcontrol import Robot
from typing_extensions import Annotated


@dataclass
class Args:
    """Command-line options."""

    distance: float = 1.0
    angle: float = 90.0
    speed: Optional[float] = None
    yaw_source: str = "auto"
    enable_sensor: Annotated[
        List[str], tyro.conf.arg(help="Sensor to enable (repeatable)")
    ] = field(default_factory=list)
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
        simulation=args.simulated,
        profile=args.profile,
        config_file=args.config,
        enable_sensors=args.enable_sensor,
        initialize_arm_modes=False,
    ) as robot:
        if not robot.supports_odometry:
            raise SystemExit("this robot's chassis has no wheel model; odometry is unavailable")
        odometry = robot.start_odometry(yaw_source=args.yaw_source)
        print(f"heading from: {odometry.yaw_source}", flush=True)
        chassis = robot.chassis
        try:
            moved = chassis.move_by(args.distance, speed=args.speed)
            print(f"moved {moved:+.3f} m (asked {args.distance:+.3f} m)", flush=True)
            turned = chassis.turn_by(math.radians(args.angle), speed=args.speed)
            print(f"turned {math.degrees(turned):+.1f} deg (asked {args.angle:+.1f} deg)", flush=True)
            pose = odometry.pose()
            status = odometry.status()
            print(
                f"pose: x={pose.x:+.3f} m  y={pose.y:+.3f} m  yaw={math.degrees(pose.yaw):+.1f} deg | "
                f"samples={status['samples']} imu={status['imu_samples']} "
                f"slip={status['slip_samples']} gaps={status['gaps']}",
                flush=True,
            )
        finally:
            chassis.stop()


if __name__ == "__main__":
    main()

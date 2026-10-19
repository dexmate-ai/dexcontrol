# Copyright (C) 2026 Dexmate Inc.
#
# This software is dual-licensed:
#
# 1. GNU Affero General Public License v3.0 (AGPL-3.0)
#    See LICENSE-AGPL for details
#
# 2. Commercial License
#    For commercial licensing terms, contact: contact@dexmate.ai

"""Read and print observations from configured IMUs.

Read IMU streams from head_camera and chassis_imu (when declared) once by default, waiting up to 5 s for activity. --sensor changes the requested list; optional repeated reads use a 0.1 s interval.

Filters undeclared IMU names before connecting.

Uses a Robot control connection: compatible arm modes may be enabled during startup, and shutdown requests a stop. Real hardware is selected unless --simulated is supplied."""

import pprint
import time
from dataclasses import dataclass, field
from typing import List, Optional

import tyro
from dexcontrol import Robot, robot_config
from typing_extensions import Annotated


@dataclass
class Args:
    """Command-line options."""

    sensor: Annotated[List[str], tyro.conf.UseAppendAction] = field(
        default_factory=lambda: []
    )
    samples: int = 1
    period: float = 0.1
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
    requested = args.sensor or ["head_camera", "chassis_imu"]
    declared = robot_config(profile=args.profile, config_file=args.config)["sensors"]
    sensors = []
    for name in requested:
        if name in declared:
            sensors.append(name)
        else:
            print(f"{name}: not declared by this profile, skipping")
    if not sensors:
        raise SystemExit("none of the requested IMUs exist in this profile")
    with Robot(
        enable_sensors=sensors,
        simulation=args.simulated,
        profile=args.profile,
        config_file=args.config,
    ) as robot:
        available = [name for name in sensors if robot.has_sensor(name)]
        if not available:
            raise SystemExit("none of the requested IMUs exist in this profile")
        handles = {name: robot.sensors.get_imu(name) for name in available}
        for name, sensor in handles.items():
            if not sensor.wait_for_active(timeout=5.0):
                print(f"{name}: inactive")
        for index in range(args.samples):
            for name, sensor in handles.items():
                print(f"\n{name}")
                pprint.pp(sensor.get_obs(), sort_dicts=False)
            if index + 1 < args.samples:
                time.sleep(args.period)


if __name__ == "__main__":
    main()

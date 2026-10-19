# Copyright (C) 2026 Dexmate Inc.
#
# This software is dual-licensed:
#
# 1. GNU Affero General Public License v3.0 (AGPL-3.0)
#    See LICENSE-AGPL for details
#
# 2. Commercial License
#    For commercial licensing terms, contact: contact@dexmate.ai

"""Monitor live 2D LiDAR scans.

Enable lidar_2d_front, wait up to 10 s for activity, and poll continuously at 0.1 s intervals until Ctrl+C. --samples N limits the number of polls. Display XY points or range summaries; no scan file is saved.

Starts/connects a Rerun viewer by default; --no-display prints instead. Missing rerun-sdk falls back to printing.

Uses a Robot control connection: compatible arm modes may be enabled during startup, and shutdown requests a stop. Real hardware is selected unless --simulated is supplied."""

import time
from dataclasses import dataclass
from typing import Optional

import numpy as np
import tyro
from _viewer import ViewerOptions, set_sample_time, start
from dexcontrol import Robot
from read_2d_lidar_scan import scan_points
from typing_extensions import Annotated


@dataclass
class Args(ViewerOptions):
    """Command-line options."""

    sensor: str = "lidar_2d_front"
    samples: Annotated[
        Optional[int], tyro.conf.arg(help="Poll limit; omitted runs until Ctrl+C")
    ] = None
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
    if args.samples is not None and args.samples < 0:
        raise ValueError("--samples must be non-negative")

    with Robot(
        enable_sensors=[args.sensor],
        simulation=args.simulated,
        profile=args.profile,
        config_file=args.config,
    ) as robot:
        lidar = robot.sensors.get_lidar_2d(args.sensor)
        if not lidar.wait_for_active(timeout=10.0):
            raise SystemExit(f"{args.sensor} did not become active")

        visualize = start(args.sensor, args)
        started = time.monotonic()
        try:
            index = 0
            while args.samples is None or index < args.samples:
                scan = lidar.get_obs()
                if scan is not None:
                    points = scan_points(scan)
                    if visualize:
                        import rerun as rr

                        set_sample_time(index, time.monotonic() - started)
                        rr.log(args.sensor, rr.Points3D(points, radii=0.02))
                    else:
                        ranges = np.linalg.norm(points[:, :2], axis=1)
                        low = ranges.min() if len(ranges) else float("nan")
                        print(f"sample={index + 1} points={len(ranges)} min={low:.3f}")
                index += 1
                if args.samples is None or index < args.samples:
                    time.sleep(args.period)
        except KeyboardInterrupt:
            pass


if __name__ == "__main__":
    main()

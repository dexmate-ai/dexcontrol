# Copyright (C) 2026 Dexmate Inc.
#
# This software is dual-licensed:
#
# 1. GNU Affero General Public License v3.0 (AGPL-3.0)
#    See LICENSE-AGPL for details
#
# 2. Commercial License
#    For commercial licensing terms, contact: contact@dexmate.ai

"""Monitor point clouds from one 3D LiDAR.

Enable the front 3D LiDAR, wait up to 10 s for activity, and poll continuously at 0.1 s intervals until Ctrl+C. --samples N limits the number of polls. --position back selects the rear LiDAR; no cloud file is saved.

Starts/connects a Rerun viewer by default; --no-display prints instead. Missing rerun-sdk falls back to printing.

Uses a Robot control connection: compatible arm modes may be enabled during startup, and shutdown requests a stop. Real hardware is selected unless --simulated is supplied."""

import time
from dataclasses import dataclass
from typing import Literal, Optional

import numpy as np
import tyro
from _viewer import ViewerOptions, set_sample_time, start
from dexcontrol import Robot
from typing_extensions import Annotated


@dataclass
class Args(ViewerOptions):
    """Command-line options."""

    position: Literal["front", "back"] = "front"
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

    name = f"lidar_3d_{args.position}"
    with Robot(
        enable_sensors=[name],
        simulation=args.simulated,
        profile=args.profile,
        config_file=args.config,
    ) as robot:
        lidar = robot.sensors.get_lidar_3d(name)
        if not lidar.wait_for_active(timeout=10.0):
            raise SystemExit(f"{name} did not become active")

        visualize = start(name, args)
        started = time.monotonic()
        try:
            index = 0
            while args.samples is None or index < args.samples:
                points = lidar.get_points()
                if points is not None and len(points):
                    if visualize:
                        import rerun as rr

                        set_sample_time(index, time.monotonic() - started)
                        rr.log(name, rr.Points3D(points, colors=_height_colors(points)))
                    else:
                        print(f"sample={index + 1} points={len(points)}")
                index += 1
                if args.samples is None or index < args.samples:
                    time.sleep(args.period)
        except KeyboardInterrupt:
            pass


def _height_colors(points: np.ndarray) -> np.ndarray:
    """Maps z to a blue-to-red ramp, scaled to the cloud's own range.

    Scaled per cloud rather than to fixed limits: the useful contrast is
    between the floor and what is standing on it, and that range depends on
    where the sensor is mounted.
    """
    height = points[:, 2]
    low, high = float(height.min()), float(height.max())
    fraction = np.zeros_like(height) if high <= low else (height - low) / (high - low)
    colors = np.zeros((len(points), 3), dtype=np.uint8)
    colors[:, 0] = (fraction * 255).astype(np.uint8)
    colors[:, 2] = ((1.0 - fraction) * 255).astype(np.uint8)
    return colors


if __name__ == "__main__":
    main()

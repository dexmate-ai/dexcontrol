# Copyright (C) 2026 Dexmate Inc.
# SPDX-License-Identifier: Apache-2.0

"""Read and display one 2D LiDAR scan.

Enable lidar_2d_front, wait up to 10 s for activity, and read one cached scan. Python can visualize XY points; native examples print range statistics. No scan file is saved.

Starts/connects a Rerun viewer by default; --no-display prints instead. Missing rerun-sdk falls back to printing.

Uses a read-only DiagnosticClient connection; motors are not initialized. Real hardware is selected unless --simulated is supplied."""

from dataclasses import dataclass
from typing import Optional

import numpy as np
import tyro
from dexcontrol.viewer import ViewerOptions, set_sample_time, start
from dexcontrol import DiagnosticClient
from typing_extensions import Annotated


def scan_points(scan: dict) -> np.ndarray:
    """Polar ranges to Cartesian points on the sensor's z=0 plane."""
    ranges = np.asarray(scan["ranges"], dtype=np.float32)
    angles = np.asarray(scan["angles"], dtype=np.float32)
    # A scan reports its own out-of-range returns as non-finite; plotting them
    # stretches the view to infinity and hides the real geometry.
    finite = np.isfinite(ranges)
    ranges, angles = ranges[finite], angles[finite]
    return np.column_stack(
        [ranges * np.cos(angles), ranges * np.sin(angles), np.zeros_like(ranges)]
    )


@dataclass
class Args(ViewerOptions):
    """Command-line options."""

    sensor: str = "lidar_2d_front"
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

    with DiagnosticClient(
        enable_sensors=[args.sensor],
        simulation=args.simulated,
        profile=args.profile,
        config_file=args.config,
    ) as robot:
        lidar = robot.sensors.get_lidar_2d(args.sensor)
        if not lidar.wait_for_active(timeout=10.0):
            raise SystemExit(f"{args.sensor} did not become active")
        scan = lidar.get_obs()
        if scan is None:
            raise SystemExit("no scan available")

        points = scan_points(scan)
        if start(args.sensor, args):
            import rerun as rr

            set_sample_time(0, 0.0)
            rr.log(args.sensor, rr.Points3D(points, radii=0.02))
        else:
            ranges = np.linalg.norm(points[:, :2], axis=1)
            print(f"points={len(ranges)} min={ranges.min():.3f} max={ranges.max():.3f}")


if __name__ == "__main__":
    main()

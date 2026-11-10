# Copyright (C) 2026 Dexmate Inc.
# SPDX-License-Identifier: Apache-2.0

"""Measure full Robot connection time and query round-trip latency.

Create one control connection, then issue 10 version_info queries and print timing statistics. This includes normal Robot startup/cleanup side effects; it is not a read-only DiagnosticClient benchmark.

Also waits for active state (10 s default timeout) before finishing the connection timer.

Uses a Robot control connection: compatible arm modes may be enabled during startup, and shutdown requests a stop. Real hardware is selected unless --simulated is supplied."""

from __future__ import annotations

import statistics
import time
from dataclasses import dataclass
from typing import Optional

import tyro
from dexcontrol import Robot


def milliseconds(seconds: float) -> str:
    return f"{seconds * 1000:.1f} ms"


@dataclass
class Args:
    """Connection benchmark options."""

    profile: Optional[str] = None
    """Built-in robot profile (defaults to ROBOT_NAME)."""
    samples: int = 10
    """Number of query RTT samples."""
    timeout: float = 10.0
    """Active-state timeout."""
    simulated: bool = False
    config: Optional[str] = None


def main() -> None:
    args = tyro.cli(Args, description=__doc__, config=(tyro.conf.FlagCreatePairsOff,))
    if args.samples < 1:
        raise SystemExit("--samples must be at least 1")

    started = time.perf_counter()
    robot = Robot(
        profile=args.profile, config_file=args.config, simulation=args.simulated
    )
    try:
        if not robot.wait_for_active(timeout=args.timeout):
            raise RuntimeError(
                f"robot did not become active within {args.timeout:.1f} seconds"
            )
        connection_time = time.perf_counter() - started

        round_trips: list[float] = []
        for sample in range(args.samples):
            try:
                query_started = time.perf_counter()
                # The raw query, deliberately: `robot.version_info()` answers
                # from the connect-time response and would time a dictionary
                # lookup rather than a network round trip.
                robot.query("version_info")
                round_trips.append(time.perf_counter() - query_started)
            except Exception as error:
                print(f"query {sample + 1} failed: {error}")

        print("Connect Robot Latency Benchmark")
        print(f"  Init + discovery: {milliseconds(connection_time)}")
        if round_trips:
            deviation = statistics.stdev(round_trips) if len(round_trips) > 1 else 0.0
            print(f"  Query RTT samples: {len(round_trips)}")
            print(f"  Mean: {milliseconds(statistics.mean(round_trips))}")
            print(f"  Min:  {milliseconds(min(round_trips))}")
            print(f"  Max:  {milliseconds(max(round_trips))}")
            print(f"  Std:  {milliseconds(deviation)}")
        else:
            print("  Query RTT: no successful samples")
    finally:
        robot.shutdown()


if __name__ == "__main__":
    main()

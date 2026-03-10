# Copyright (C) 2025 Dexmate Inc.
#
# This software is dual-licensed:
#
# 1. GNU Affero General Public License v3.0 (AGPL-3.0)
#    See LICENSE-AGPL for details
#
# 2. Commercial License
#    For commercial licensing terms, contact: contact@dexmate.ai

"""Connect-robot latency benchmark.

Measures two phases of establishing a ready robot session:

  Phase 1 — Init + Discovery:
    Time from script start until Zenoh finds the robot's service
    (RobotQueryInterface creation + wait_for_service).

  Phase 2 — Query RTT:
    Round-trip time for each hand-type query call, repeated N times.

Usage::

    PYTHONPATH=src python examples/benchmark/connection/connect_robot_latency.py
    PYTHONPATH=src python examples/benchmark/connection/connect_robot_latency.py --n-runs 20
    PYTHONPATH=src python examples/benchmark/connection/connect_robot_latency.py --timeout 15.0
"""

import statistics
import sys
import time
from dataclasses import dataclass

import tyro
from loguru import logger
from rich.console import Console
from rich.table import Table

from dexcontrol.core.robot_query_interface import RobotQueryInterface


@dataclass
class Args:
    """Connect-robot latency benchmark."""

    n_runs: int = 10
    """Number of Phase 2 RTT repetitions."""

    timeout: float = 10.0
    """Discovery timeout in seconds (Phase 1 wait_for_service)."""


def _ms(seconds: float) -> str:
    """Format seconds as a milliseconds string."""
    return f"{seconds * 1000:.1f} ms"


def _print_results(
    phase1_s: float,
    rtt_times: list[float],
    console: Console,
) -> None:
    """Print benchmark results as a rich table.

    Args:
        phase1_s: Phase 1 duration in seconds (init + discovery).
        rtt_times: List of Phase 2 RTT measurements in seconds.
        console: Rich console instance.
    """
    table = Table(
        title="Connect Robot Latency Benchmark",
        show_header=False,
        show_edge=True,
        padding=(0, 1),
    )
    table.add_column("Metric", style="cyan", no_wrap=True)
    table.add_column("Value", style="green", justify="right")

    table.add_row("Phase 1 — Init + Discovery", _ms(phase1_s))

    if rtt_times:
        n = len(rtt_times)
        mean = statistics.mean(rtt_times)
        std = statistics.stdev(rtt_times) if n > 1 else 0.0

        table.add_row("", "")
        table.add_row(f"Phase 2 — Query RTT  (N={n})", "")
        table.add_row("  Mean", _ms(mean))
        table.add_row("  Min", _ms(min(rtt_times)))
        table.add_row("  Max", _ms(max(rtt_times)))
        table.add_row("  Std", _ms(std))
    else:
        table.add_row("Phase 2 — Query RTT", "no data (all runs failed)")

    console.print(table)


def main(args: Args) -> None:
    """Run the connect-robot latency benchmark."""
    console = Console()
    console.print("[bold cyan]Connect Robot Latency Benchmark[/bold cyan]")

    # ------------------------------------------------------------------
    # Phase 1: Init + Discovery
    # ------------------------------------------------------------------
    t0 = time.perf_counter()
    interface = RobotQueryInterface.create()

    if not interface._hand_querier.wait_for_service(timeout=args.timeout):  # type: ignore[attr-defined]
        console.print(
            f"[red]✗ Could not connect to robot within {args.timeout}s.[/red]\n"
            "  • Run 'dextop topic list' to verify you can receive topics from the robot.\n"
            "  • Check that the robot is powered on and the network is reachable."
        )
        interface.close()
        sys.exit(1)

    t1 = time.perf_counter()
    phase1_s = t1 - t0

    # ------------------------------------------------------------------
    # Phase 2: N RTT measurements
    # ------------------------------------------------------------------
    logger.info(f"Running {args.n_runs} RTT measurements…")
    rtt_times: list[float] = []

    for i in range(args.n_runs):
        try:
            t2 = time.perf_counter()
            interface._hand_querier.call(None)  # type: ignore[attr-defined]
            t3 = time.perf_counter()
            rtt_times.append(t3 - t2)
        except Exception as exc:  # catch all exceptions to preserve partial results
            console.print(f"[yellow]  Run {i + 1} failed: {exc}[/yellow]")

    interface.close()

    # ------------------------------------------------------------------
    # Results
    # ------------------------------------------------------------------
    console.print()
    _print_results(phase1_s, rtt_times, console)


if __name__ == "__main__":
    main(tyro.cli(Args))

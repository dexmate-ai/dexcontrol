# Copyright (C) 2025 Dexmate Inc.
#
# This software is dual-licensed:
#
# 1. GNU Affero General Public License v3.0 (AGPL-3.0)
#    See LICENSE-AGPL for details
#
# 2. Commercial License
#    For commercial licensing terms, contact: contact@dexmate.ai

"""DexControl connection benchmark.

Measures how long it takes to connect to the robot and receive the first
state update on every topic.  Useful for evaluating auto-router startup
overhead, Zenoh discovery latency, and per-component subscription time.

Phases timed:
  1. Session creation  (dexcomm.get_session — includes auto-router spawn)
  2. Robot() init       (config load + component creation + wait_for_active)
  3. Per-component      (individual subscriber → first message latency)

Usage:
    # Single run
    PYTHONPATH=src python examples/benchmark/connection_benchmark.py

    # Multiple runs for statistics
    PYTHONPATH=src python examples/benchmark/connection_benchmark.py --runs 5
"""

from __future__ import annotations

import statistics
import time

import tyro
from loguru import logger

# ---------------------------------------------------------------------------
# Timing helpers
# ---------------------------------------------------------------------------

_CONTROLLABLE_COMPONENTS = [
    "left_arm",
    "right_arm",
    "left_hand",
    "right_hand",
    "head",
    "torso",
    "chassis",
]

_MONITORING_COMPONENTS = ["battery", "estop", "heartbeat"]

_ALL_COMPONENTS = _CONTROLLABLE_COMPONENTS + _MONITORING_COMPONENTS


def _fmt_ms(seconds: float) -> str:
    """Format seconds as milliseconds string."""
    return f"{seconds * 1000:.1f} ms"


def _stats_line(values_s: list[float]) -> str:
    """Return mean / min / max / stdev for a list of seconds."""
    if not values_s:
        return "n/a"
    ms = [v * 1000 for v in values_s]
    mean = statistics.mean(ms)
    if len(ms) > 1:
        sd = statistics.stdev(ms)
        return f"mean={mean:.1f} ms  min={min(ms):.1f} ms  max={max(ms):.1f} ms  stdev={sd:.1f} ms"
    return f"{mean:.1f} ms"


# ---------------------------------------------------------------------------
# Single run
# ---------------------------------------------------------------------------


def _run_once() -> dict:
    """Execute one full connection cycle and return timing data."""
    from dexcontrol.robot import Robot

    result: dict = {
        "session_s": 0.0,
        "robot_total_s": 0.0,
        "components": {},
    }

    # --- Phase 1: Session creation (includes auto-router spawn) ------------
    import dexcomm

    t0 = time.perf_counter()
    dexcomm.get_session()
    t1 = time.perf_counter()
    result["session_s"] = t1 - t0

    # --- Phase 2: Robot initialization -------------------------------------
    t_robot_start = time.perf_counter()
    robot = Robot()
    t_robot_end = time.perf_counter()
    result["robot_total_s"] = t_robot_end - t_robot_start

    # --- Phase 3: Per-component timing (already connected via Robot()) -----
    for name in _ALL_COMPONENTS:
        comp = getattr(robot, name, None)
        if comp is None:
            continue
        active = comp.is_active()
        result["components"][name] = {
            "active": active,
        }

    # --- Cleanup -----------------------------------------------------------
    # robot.shutdown() calls cleanup_session() internally
    robot.shutdown()

    return result


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------


def _print_report(
    all_runs: list[dict],
) -> None:
    sep = "=" * 65
    logger.info(sep)
    logger.info("DEXCONTROL CONNECTION BENCHMARK RESULTS")
    logger.info(sep)
    n = len(all_runs)
    logger.info(f"Runs: {n}")
    logger.info("")

    # Session
    session_times = [r["session_s"] for r in all_runs]
    logger.info(f"Session creation:   {_stats_line(session_times)}")

    # Robot total
    robot_times = [r["robot_total_s"] for r in all_runs]
    logger.info(f"Robot() total:      {_stats_line(robot_times)}")

    # Total (session + robot)
    total_times = [r["session_s"] + r["robot_total_s"] for r in all_runs]
    logger.info(f"End-to-end total:   {_stats_line(total_times)}")

    logger.info("")
    logger.info("Per-component status (last run):")

    last = all_runs[-1]
    for name in _ALL_COMPONENTS:
        info = last["components"].get(name)
        if info is None:
            continue
        status = "active" if info["active"] else "INACTIVE"
        logger.info(f"  {name:<16s}  {status}")

    # If multiple runs, show per-component connection across runs
    if n > 1:
        logger.info("")
        logger.info("Component availability across runs:")
        all_names = set()
        for r in all_runs:
            all_names.update(r["components"].keys())
        for name in _ALL_COMPONENTS:
            if name not in all_names:
                continue
            active_count = sum(
                1
                for r in all_runs
                if r["components"].get(name, {}).get("active", False)
            )
            logger.info(f"  {name:<16s}  {active_count}/{n} runs active")

    logger.info(sep)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main(
    runs: int = 1,
) -> None:
    """Run the DexControl connection benchmark.

    Measures session creation time (including auto-router spawn),
    Robot() initialization time, and per-component connection status.

    Args:
        runs: Number of connection cycles to run (for statistics).
    """
    logger.info("=" * 65)
    logger.info("DEXCONTROL CONNECTION BENCHMARK")
    logger.info(f"Runs: {runs}")
    logger.info("=" * 65)

    all_runs: list[dict] = []
    for i in range(runs):
        if runs > 1:
            logger.info(f"\n--- Run {i + 1}/{runs} ---")

        result = _run_once()
        all_runs.append(result)

        logger.info(
            f"  Session: {_fmt_ms(result['session_s'])}  "
            f"Robot: {_fmt_ms(result['robot_total_s'])}  "
            f"Total: {_fmt_ms(result['session_s'] + result['robot_total_s'])}"
        )
        active = sum(1 for c in result["components"].values() if c["active"])
        logger.info(f"  Components: {active}/{len(result['components'])} active")

    logger.info("")
    _print_report(all_runs)


if __name__ == "__main__":
    tyro.cli(main)

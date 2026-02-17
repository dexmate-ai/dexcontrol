# Copyright (C) 2025 Dexmate Inc.
#
# This software is dual-licensed:
#
# 1. GNU Affero General Public License v3.0 (AGPL-3.0)
#    See LICENSE-AGPL for details
#
# 2. Commercial License
#    For commercial licensing terms, contact: contact@dexmate.ai

"""DexControl GIL availability benchmark.

Measures how much GIL time remains available for user programs (e.g. neural
network inference) when the full dexcontrol motor-control subscriber stack is
running against a real robot.

The benchmark runs in two phases:
  Phase 1 (Baseline): Pure-Python busy-loop with NO robot connected.
  Phase 2 (Loaded):   Same busy-loop with Robot() fully connected and all
                       motor-control subscribers actively receiving data.

Key metrics:
  - GIL availability %  = baseline_mean / loaded_mean * 100
  - Jitter (stdev)       = stdev of iteration times (callbacks cause GIL
                           interruptions → higher jitter)
  - State freshness      = age of latest component state after each GIL-hold
                           burst.  Rust-side storage updates continuously;
                           callback-based storage only updates at GIL switch
                           boundaries (~5 ms).

Usage:
    # Default (10 ms GIL hold, 10 s per phase)
    PYTHONPATH=src python examples/benchmark/gil_benchmark.py

    # Simulate heavier NN inference (50 ms per iteration)
    PYTHONPATH=src python examples/benchmark/gil_benchmark.py --gil-hold-ms 50

    # Shorter run for quick sanity check
    PYTHONPATH=src python examples/benchmark/gil_benchmark.py --duration 5
"""

from __future__ import annotations

import statistics
import time
from dataclasses import dataclass, field

import tyro
from loguru import logger

from dexcontrol.robot import Robot

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _hold_gil_busy(duration_ms: float) -> None:
    """Hold the GIL for *duration_ms* using a pure-Python busy loop."""
    end = time.perf_counter() + duration_ms / 1000.0
    while time.perf_counter() < end:
        _ = [i * i for i in range(100)]


def _compute_latency_stats(times_us: list[float]) -> dict[str, float]:
    """Return mean / median / p99 / max for a list of microsecond timings."""
    if not times_us:
        return {"mean": 0.0, "median": 0.0, "p99": 0.0, "max": 0.0}
    sorted_t = sorted(times_us)
    p99_idx = min(int(len(sorted_t) * 0.99), len(sorted_t) - 1)
    return {
        "mean": statistics.mean(times_us),
        "median": statistics.median(times_us),
        "p99": sorted_t[p99_idx],
        "max": sorted_t[-1],
    }


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class PhaseResult:
    """Timing results from a single benchmark phase."""

    iterations: int = 0
    wall_time_s: float = 0.0
    iter_times_ms: list[float] = field(default_factory=list)
    mean_ms: float = 0.0
    stdev_ms: float = 0.0
    throughput_hz: float = 0.0

    # State freshness (loaded phase only, milliseconds).
    # "How old is the latest component state right after a GIL-hold burst?"
    freshness_ms: list[float] = field(default_factory=list)
    freshness_mean_ms: float = 0.0
    freshness_stdev_ms: float = 0.0
    freshness_max_ms: float = 0.0


# ---------------------------------------------------------------------------
# Phase runner
# ---------------------------------------------------------------------------


def _run_phase(
    duration_s: float,
    gil_hold_ms: float,
    warmup_s: float,
    robot: Robot | None = None,
) -> PhaseResult:
    """Run a single measurement phase.

    When *robot* is provided the phase also records state freshness after
    every GIL-hold burst by comparing the latest component timestamp to
    the current wall clock.
    """
    # Pick a component for freshness probing.
    probe_component = None
    if robot is not None:
        for name in ("left_arm", "right_arm", "head", "torso"):
            comp = getattr(robot, name, None)
            if comp is not None and hasattr(comp, "get_timestamp_ns"):
                probe_component = comp
                break

    # Warmup.
    warmup_end = time.perf_counter() + warmup_s
    while time.perf_counter() < warmup_end:
        _hold_gil_busy(gil_hold_ms)

    # Measurement.
    times_ms: list[float] = []
    freshness_ms: list[float] = []

    phase_start = time.perf_counter()
    while time.perf_counter() - phase_start < duration_s:
        t0 = time.perf_counter()
        _hold_gil_busy(gil_hold_ms)
        t1 = time.perf_counter()
        times_ms.append((t1 - t0) * 1_000)

        # State freshness: how old is the latest state right now?
        if probe_component is not None:
            now_ns = time.time_ns()
            try:
                state_ns = probe_component.get_timestamp_ns()
                age_ms = (now_ns - state_ns) / 1_000_000
                if age_ms >= 0:
                    freshness_ms.append(age_ms)
            except RuntimeError:
                pass  # No state yet.

    result = PhaseResult()
    result.iterations = len(times_ms)
    result.wall_time_s = time.perf_counter() - phase_start
    result.iter_times_ms = times_ms
    if times_ms:
        result.mean_ms = statistics.mean(times_ms)
        result.stdev_ms = statistics.stdev(times_ms) if len(times_ms) > 1 else 0.0
        result.throughput_hz = len(times_ms) / result.wall_time_s

    if freshness_ms:
        result.freshness_ms = freshness_ms
        result.freshness_mean_ms = statistics.mean(freshness_ms)
        result.freshness_stdev_ms = (
            statistics.stdev(freshness_ms) if len(freshness_ms) > 1 else 0.0
        )
        result.freshness_max_ms = max(freshness_ms)

    return result


# ---------------------------------------------------------------------------
# Robot introspection
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


def _enumerate_active_components(robot: Robot) -> list[str]:
    """Return the names of all active motor-control + monitoring components."""
    active: list[str] = []
    for name in _CONTROLLABLE_COMPONENTS + _MONITORING_COMPONENTS:
        comp = getattr(robot, name, None)
        if comp is not None:
            active.append(name)
    return active


def _count_subscribers(robot: Robot) -> int:
    """Estimate total subscriber count across all active components."""
    count = 0
    for name in _CONTROLLABLE_COMPONENTS:
        comp = getattr(robot, name, None)
        if comp is None:
            continue
        if name == "chassis":
            count += 2  # steer + drive
        else:
            count += 1
    for name in _MONITORING_COMPONENTS:
        if getattr(robot, name, None) is not None:
            count += 1
    return count


# ---------------------------------------------------------------------------
# State-read latency measurement
# ---------------------------------------------------------------------------


def _measure_state_read_latency(robot: Robot, samples: int) -> list[float]:
    """Measure get_joint_pos() latency across all joint components.

    Returns a flat list of per-call timings in microseconds.
    """
    joint_components: list[str] = []
    for name in _CONTROLLABLE_COMPONENTS:
        comp = getattr(robot, name, None)
        if comp is not None and hasattr(comp, "get_joint_pos"):
            joint_components.append(name)

    if not joint_components:
        return []

    times_us: list[float] = []
    for _ in range(samples):
        for name in joint_components:
            comp = getattr(robot, name)
            t0 = time.perf_counter()
            comp.get_joint_pos()
            t1 = time.perf_counter()
            times_us.append((t1 - t0) * 1_000_000)
    return times_us


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------


def _log_report(
    *,
    gil_hold_ms: float,
    duration_s: float,
    warmup_s: float,
    active_components: list[str],
    subscriber_count: int,
    baseline: PhaseResult,
    loaded: PhaseResult,
    gil_availability_pct: float,
    state_read_stats: dict[str, float],
    state_read_samples: int,
) -> None:
    sep = "=" * 60
    logger.info(sep)
    logger.info("DEXCONTROL GIL AVAILABILITY BENCHMARK RESULTS")
    logger.info(sep)

    logger.info("Configuration:")
    logger.info(f"  GIL hold per iter: {gil_hold_ms} ms")
    logger.info(f"  Duration / phase:  {duration_s} s")
    logger.info(f"  Warmup / phase:    {warmup_s} s")

    logger.info("Robot:")
    logger.info(f"  Components:        {', '.join(active_components)}")
    logger.info(f"  Subscribers:       ~{subscriber_count}")

    logger.info("Phase 1 - Baseline (no robot):")
    logger.info(f"  Iterations:        {baseline.iterations}")
    logger.info(f"  Throughput:        {baseline.throughput_hz:.1f} Hz")
    logger.info(f"  Mean iter time:    {baseline.mean_ms:.3f} ms")
    logger.info(f"  Stdev iter time:   {baseline.stdev_ms:.3f} ms")

    logger.info("Phase 2 - Robot Loaded:")
    logger.info(f"  Iterations:        {loaded.iterations}")
    logger.info(f"  Throughput:        {loaded.throughput_hz:.1f} Hz")
    logger.info(f"  Mean iter time:    {loaded.mean_ms:.3f} ms")
    logger.info(f"  Stdev iter time:   {loaded.stdev_ms:.3f} ms")

    logger.info(f"GIL AVAILABILITY: {gil_availability_pct:.2f}%")

    if loaded.freshness_ms:
        logger.info("State Freshness (age of latest state after GIL-hold burst):")
        logger.info(f"  Mean:              {loaded.freshness_mean_ms:.3f} ms")
        logger.info(f"  Stdev:             {loaded.freshness_stdev_ms:.3f} ms")
        logger.info(f"  Max:               {loaded.freshness_max_ms:.3f} ms")

    if state_read_stats["mean"] > 0:
        logger.info(
            f"State Read Latency (get_joint_pos, {state_read_samples} samples/component):"
        )
        logger.info(f"  Mean:              {state_read_stats['mean']:.2f} us")
        logger.info(f"  Median:            {state_read_stats['median']:.2f} us")
        logger.info(f"  P99:               {state_read_stats['p99']:.2f} us")
        logger.info(f"  Max:               {state_read_stats['max']:.2f} us")

    logger.info(sep)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main(
    gil_hold_ms: float = 10.0,
    duration: float = 10.0,
    warmup_time: float = 2.0,
    state_read_samples: int = 1000,
) -> None:
    """Run the DexControl GIL availability benchmark.

    Connects to a real robot and measures how much GIL time is left for
    user programs when all motor-control subscribers are active.

    Args:
        gil_hold_ms: Simulated user-workload duration per iteration (ms).
        duration: Measurement duration per phase (seconds).
        warmup_time: Warmup before each phase (seconds).
        state_read_samples: Number of get_joint_pos() samples per component.
    """
    logger.info("=" * 60)
    logger.info("DEXCONTROL GIL AVAILABILITY BENCHMARK")
    logger.info("=" * 60)

    # ---- Phase 1: Baseline (no robot) ------------------------------------
    logger.info(
        f"Phase 1: Baseline | gil_hold={gil_hold_ms} ms, "
        f"duration={duration} s, warmup={warmup_time} s"
    )
    baseline = _run_phase(duration, gil_hold_ms, warmup_time)
    logger.info(
        f"Baseline done: {baseline.throughput_hz:.1f} Hz, "
        f"mean={baseline.mean_ms:.3f} ms, stdev={baseline.stdev_ms:.3f} ms"
    )

    # ---- Phase 2: Robot loaded -------------------------------------------
    logger.info("Phase 2: Connecting robot...")
    robot = Robot()
    active_components = _enumerate_active_components(robot)
    subscriber_count = _count_subscribers(robot)
    logger.info(f"Robot ready | components: {active_components}")
    logger.info(f"Estimated subscribers: ~{subscriber_count}")

    logger.info(
        f"Phase 2: Loaded | gil_hold={gil_hold_ms} ms, "
        f"duration={duration} s, warmup={warmup_time} s"
    )
    loaded = _run_phase(duration, gil_hold_ms, warmup_time, robot=robot)
    logger.info(
        f"Loaded done: {loaded.throughput_hz:.1f} Hz, "
        f"mean={loaded.mean_ms:.3f} ms, stdev={loaded.stdev_ms:.3f} ms"
    )

    # ---- State-read latency ----------------------------------------------
    logger.info(f"Measuring state-read latency ({state_read_samples} samples)...")
    read_times_us = _measure_state_read_latency(robot, state_read_samples)
    state_read_stats = _compute_latency_stats(read_times_us)

    # ---- Cleanup ---------------------------------------------------------
    robot.shutdown()

    # ---- Compute GIL availability ----------------------------------------
    if loaded.mean_ms > 0:
        gil_availability_pct = (baseline.mean_ms / loaded.mean_ms) * 100.0
    else:
        gil_availability_pct = 100.0

    # ---- Report ----------------------------------------------------------
    _log_report(
        gil_hold_ms=gil_hold_ms,
        duration_s=duration,
        warmup_s=warmup_time,
        active_components=active_components,
        subscriber_count=subscriber_count,
        baseline=baseline,
        loaded=loaded,
        gil_availability_pct=gil_availability_pct,
        state_read_stats=state_read_stats,
        state_read_samples=state_read_samples,
    )


if __name__ == "__main__":
    tyro.cli(main)

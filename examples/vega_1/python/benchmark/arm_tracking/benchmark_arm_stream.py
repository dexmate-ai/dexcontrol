# Copyright (C) 2026 Dexmate Inc.
# SPDX-License-Identifier: Apache-2.0

"""Benchmark each arm joint with a sine goal streamed through the motion plugin.

Prompt for confirmation unless --no-confirm; move the right seven-joint arm to [0,0,0,-0.5,0,0,0] rad. Test each joint for 10 s at 100 Hz with a 0.4 rad, 0.5 Hz sine streamed with stream_joint_pos (the untracked, plugin-smoothed path teleoperation uses); return to the reference between joints and at the end. Write measurement files and print the RMS and peak tracking error plus the estimated lag per joint. The amplitude is fitted to each joint's limits from the reference pose, reduced with a warning where the requested amplitude does not fit. No step guard applies on this path: the plugin bounds velocity, acceleration and jerk itself.

This is the hardware check for the streamed path: simulation follows the goal exactly and cannot show lag. Run it on the rig before any release that touches the motion path and compare the lag against the previous run.

Writes per-joint CSV, data.npz and summary.png under results beside this script; requires matplotlib.

Uses a Robot control connection: compatible arm modes may be enabled during startup, and shutdown requests a stop. Real hardware is selected unless --simulated is supplied."""

import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Literal, Optional

import matplotlib.pyplot as plt
import numpy as np
import tyro
from _tracking_helpers import NUM_JOINTS, ZERO_POS, fit_amplitude, verify_zero_position, validate_sample_count, write_tracking_csv
from loguru import logger
from matplotlib.axes import Axes

from dexcontrol import RateLimiter, Robot, confirm

SCRIPT_DIR = Path(__file__).resolve().parent


def plot_joint(ax: Axes, times: np.ndarray, cmd: np.ndarray, actual: np.ndarray, joint_idx: int, sin_frequency: float) -> None:
    """Plot the streamed goal against the measured position for one joint."""
    ax.plot(times, cmd[:, joint_idx], color="tab:blue", label="Streamed goal")
    ax.plot(times, actual[:, joint_idx], color="tab:orange", label="Measured")
    ax.set_title(f"Joint {joint_idx} - streamed sine {sin_frequency} Hz")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Position (rad)")
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3)


def tracking_metrics(times: np.ndarray, cmd: np.ndarray, actual: np.ndarray, joint_idx: int) -> tuple[float, float, float]:
    """RMS error, peak error (rad) and the lag (s) that best aligns measured with goal.

    The lag is the cross-correlation peak of the two zero-mean traces, resolved
    to the sample period, so it reads directly as the plugin's smoothing delay
    plus transport latency.
    """
    goal = cmd[:, joint_idx] - cmd[:, joint_idx].mean()
    measured = actual[:, joint_idx] - actual[:, joint_idx].mean()
    error = cmd[:, joint_idx] - actual[:, joint_idx]
    rms = float(np.sqrt(np.mean(error**2)))
    peak = float(np.max(np.abs(error)))
    if len(times) < 2 or not np.any(goal) or not np.any(measured):
        return rms, peak, 0.0
    correlation = np.correlate(measured, goal, mode="full")
    shift = int(np.argmax(correlation)) - (len(goal) - 1)
    period = float(np.median(np.diff(times)))
    return rms, peak, max(0.0, shift * period)


@dataclass
class Args:
    """Streamed-goal tracking benchmark for all 7 joints of an arm."""

    no_confirm: bool = False
    settle: float = 0.5
    simulated: bool = False
    profile: Optional[str] = None
    config: Optional[str] = None

    side: Literal["left", "right"] = "right"
    """Which arm to test (left or right)."""

    duration: float = 10.0
    """Duration of the sine test per joint in seconds."""

    control_hz: int = 100
    """Streaming rate in Hz (the rate a teleop loop would use)."""

    amplitude: float = 0.4
    """Amplitude of the sine wave in radians."""

    sin_frequency: float = 0.5
    """Frequency of the sine wave in Hz."""

    velocity_scale: Optional[float] = None
    """Optional per-joint velocity scale in (0, 1] passed to the plugin."""

    output_dir: str = "results"
    """Base output directory for results."""


def main(args: Args) -> None:
    """Stream a sine goal to every joint in turn and measure how the plugin follows it."""
    validate_sample_count(args.duration, args.control_hz)
    if args.duration * args.control_hz < 1:
        raise ValueError("benchmark duration must include at least one sample")
    validate_sample_count(args.settle, 1.0)

    logger.warning(
        "This benchmark moves the robot arm through sine trajectories. "
        "No collision checking is performed. Ensure the workspace is clear. "
        "Make sure the e-stop is accessible."
    )
    if not args.no_confirm:
        if not confirm("Continue?"):
            logger.info("Aborted by user.")
            return

    with Robot(simulation=args.simulated, profile=args.profile, config_file=args.config) as bot:
        arm = bot.left_arm if args.side == "left" else bot.right_arm
        arm_name = f"{args.side} arm"

        timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        result_dir = SCRIPT_DIR / args.output_dir / f"{timestamp_str}_{args.side}_stream"
        result_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Results will be saved to {result_dir}")

        logger.info(f"Moving {arm_name} to zero position")
        arm.move_to_joint_pos(ZERO_POS, wait=True, timeout=5.0)
        time.sleep(args.settle)
        verify_zero_position(arm)

        all_commands: list[np.ndarray] = []
        all_states: list[np.ndarray] = []
        all_times: list[np.ndarray] = []

        for joint_idx in range(NUM_JOINTS):
            logger.info(f"--- Testing joint {joint_idx} ---")
            if joint_idx > 0:
                # A tracked move settles the arm; streaming resumes once it
                # has finished, which wait=True guarantees.
                arm.move_to_joint_pos(ZERO_POS, wait=True, timeout=3.0)

            sample_times: list[float] = []
            commands: list[np.ndarray] = []
            states: list[np.ndarray] = []
            rate_limiter = RateLimiter(args.control_hz)
            omega = 2.0 * np.pi * args.sin_frequency
            amplitude = fit_amplitude(arm, joint_idx, args.amplitude, symmetric=True, omega=omega)
            start_time = time.monotonic()

            while time.monotonic() - start_time < args.duration:
                t = time.monotonic() - start_time
                joint_pos = ZERO_POS.copy()
                joint_pos[joint_idx] += amplitude * np.sin(omega * t)
                # One goal per tick; the plugin shapes the motion. Nothing is
                # returned to wait on.
                arm.stream_joint_pos(joint_pos, velocity_scale=args.velocity_scale)
                commands.append(joint_pos.copy())
                states.append(arm.get_joint_pos().copy())
                sample_times.append(time.monotonic() - start_time)
                rate_limiter.sleep()

            cmd_array = np.array(commands)
            state_array = np.array(states)
            time_array = np.array(sample_times)
            all_commands.append(cmd_array)
            all_states.append(state_array)
            all_times.append(time_array)
            write_tracking_csv(result_dir / f"joint_{joint_idx}.csv", sample_times, cmd_array, state_array)
            rms, peak, lag = tracking_metrics(time_array, cmd_array, state_array, joint_idx)
            logger.info(f"joint {joint_idx}: rms error {rms:.4f} rad, peak error {peak:.4f} rad, lag {lag * 1000:.0f} ms")

        fig, axes = plt.subplots(4, 2, figsize=(14, 16))
        axes_flat = axes.flatten()
        for joint_idx in range(NUM_JOINTS):
            plot_joint(axes_flat[joint_idx], all_times[joint_idx], all_commands[joint_idx], all_states[joint_idx], joint_idx, args.sin_frequency)
        axes_flat[NUM_JOINTS].set_visible(False)
        fig.suptitle(
            f"Streamed sine benchmark ({args.side} arm) | {args.sin_frequency} Hz, "
            f"amp={args.amplitude} rad, dur={args.duration} s, stream={args.control_hz} Hz",
            fontsize=14,
        )
        fig.tight_layout(rect=(0, 0, 1, 0.97))
        fig.savefig(result_dir / "summary.png", dpi=150)
        plt.close(fig)
        logger.info(f"Summary plot saved to {result_dir / 'summary.png'}")

        save_dict: dict[str, Any] = {
            "side": args.side,
            "duration": args.duration,
            "control_hz": args.control_hz,
            "amplitude": args.amplitude,
            "sin_frequency": args.sin_frequency,
            "zero_pos": ZERO_POS,
        }
        for joint_idx in range(NUM_JOINTS):
            save_dict[f"times_{joint_idx}"] = all_times[joint_idx]
            save_dict[f"commands_{joint_idx}"] = all_commands[joint_idx]
            save_dict[f"states_{joint_idx}"] = all_states[joint_idx]
        np.savez(result_dir / "data.npz", **save_dict)
        logger.info(f"Raw data saved to {result_dir / 'data.npz'}")

        logger.info("Returning to zero position")
        arm.move_to_joint_pos(ZERO_POS, wait=True, timeout=3.0)
    logger.info("Benchmark complete")


if __name__ == "__main__":
    main(tyro.cli(Args))

# Copyright (C) 2025-2026 Dexmate Inc.
# SPDX-License-Identifier: Apache-2.0

"""Benchmark each arm joint with a sine trajectory and velocity feed-forward.

Prompt for confirmation unless --no-confirm; move the right seven-joint arm to [0,0,0,-0.5,0,0,0] rad. Test each joint for 10 s at 200 Hz with a 0.4 rad, 1 Hz sine; return to the reference between joints and at the end. Write measurement files. This reference is not an all-zero pose. The amplitude is fitted to each joint's limits: the position limits from the reference pose and the velocity limit against the feed-forward peak (amplitude times angular frequency), reduced with a warning where the requested amplitude does not fit. Stream with the step guard disabled (tracking lag is what is measured).

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

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SCRIPT_DIR = Path(__file__).resolve().parent


def plot_joint(
    ax: Axes,
    times: np.ndarray,
    cmd: np.ndarray,
    actual: np.ndarray,
    joint_idx: int,
    sin_frequency: float,
) -> None:
    """Plot command vs actual for one joint on the given axes.

    Args:
        ax: Matplotlib axes to draw on.
        times: (N,) time array in seconds.
        cmd: (N, 7) command array.
        actual: (N, 7) state array.
        joint_idx: Index of the tested joint.
        sin_frequency: Sine frequency in Hz (for the title).
    """
    ax.plot(times, cmd[:, joint_idx], color="tab:blue", label="Command")
    ax.plot(times, actual[:, joint_idx], color="tab:orange", label="Actual")
    ax.set_title(f"Joint {joint_idx} - Sine {sin_frequency}Hz")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Position (rad)")
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


@dataclass
class Args:
    """Sine wave latency benchmark for all 7 joints of an arm."""

    no_confirm: bool = False
    settle: float = 0.5
    simulated: bool = False
    profile: Optional[str] = None
    config: Optional[str] = None

    side: Literal["left", "right"] = "right"
    """Which arm to test (left or right)."""

    duration: float = 10.0
    """Duration of the sine test per joint in seconds."""

    control_hz: int = 200
    """Control loop frequency in Hz."""

    amplitude: float = 0.4
    """Amplitude of the sine wave in radians."""

    sin_frequency: float = 1.0
    """Frequency of the sine wave in Hz."""

    output_dir: str = "results"
    """Base output directory for results."""


def main(args: Args) -> None:
    """Run the sine wave latency benchmark on all 7 joints."""
    validate_sample_count(args.duration, args.control_hz)
    if args.duration * args.control_hz < 1:
        raise ValueError("benchmark duration must include at least one sample")
    validate_sample_count(args.settle, 1.0)

    # ------------------------------------------------------------------
    # Safety warning
    # ------------------------------------------------------------------
    logger.warning(
        "This benchmark moves the robot arm through sine trajectories. "
        "No collision checking is performed. Ensure the workspace is clear. "
        "Make sure the e-stop is accessible."
    )

    if not args.no_confirm:
        if not confirm("Continue?"):
            logger.info("Aborted by user.")
            return

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------
    with Robot(simulation=args.simulated, profile=args.profile, config_file=args.config) as bot:
        arm = bot.left_arm if args.side == "left" else bot.right_arm
        arm_name = f"{args.side} arm"

        timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_base = SCRIPT_DIR / args.output_dir
        result_dir = output_base / f"{timestamp_str}_{args.side}_sin"
        result_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Results will be saved to {result_dir}")

        # ------------------------------------------------------------------
        # Move to zero position and verify
        # ------------------------------------------------------------------
        logger.info(f"Moving {arm_name} to zero position")
        arm.move_to_joint_pos(ZERO_POS, wait=True, timeout=5.0)
        time.sleep(args.settle)

        verify_zero_position(arm)

        # ------------------------------------------------------------------
        # Test each joint
        # ------------------------------------------------------------------
        all_commands: list[np.ndarray] = []
        all_states: list[np.ndarray] = []

        for joint_idx in range(NUM_JOINTS):
            logger.info(f"--- Testing joint {joint_idx} ---")

            # Return to zero between joints (skip before joint 0)
            if joint_idx > 0:
                arm.move_to_joint_pos(ZERO_POS, wait=True, timeout=3.0)

            sample_times = []
            commands: list[np.ndarray] = []
            states: list[np.ndarray] = []
            rate_limiter = RateLimiter(args.control_hz)
            # Fitted to the joint's position and velocity limits from the
            # reference pose (reduced with a warning where the requested
            # amplitude does not fit).
            omega = 2.0 * np.pi * args.sin_frequency
            amplitude = fit_amplitude(arm, joint_idx, args.amplitude, symmetric=True, omega=omega)
            start_time = time.monotonic()

            while time.monotonic() - start_time < args.duration:
                t = time.monotonic() - start_time

                # Sine position offset
                sine_value = amplitude * np.sin(omega * t)
                # Velocity feedforward (derivative of position)
                cos_value = amplitude * omega * np.cos(omega * t)

                joint_pos = ZERO_POS.copy()
                joint_pos[joint_idx] += sine_value

                joint_vel = np.zeros(NUM_JOINTS)
                joint_vel[joint_idx] = cos_value

                # Lag is what this benchmark measures: the step guard is off.
                arm.set_joint_pos(joint_pos, joint_vel=joint_vel, max_step=None)
                commands.append(joint_pos.copy())
                states.append(arm.get_joint_pos().copy())
                sample_times.append(time.monotonic() - start_time)
                rate_limiter.sleep()

            cmd_array = np.array(commands)
            state_array = np.array(states)
            all_commands.append(cmd_array)
            all_states.append(state_array)
            write_tracking_csv(result_dir / f"joint_{joint_idx}.csv", sample_times, cmd_array, state_array)

        # ------------------------------------------------------------------
        # Summary plot (4x2 grid, last cell off)
        # ------------------------------------------------------------------
        fig, axes = plt.subplots(4, 2, figsize=(14, 16))
        axes_flat = axes.flatten()

        for joint_idx in range(NUM_JOINTS):
            ax = axes_flat[joint_idx]
            cmd_array = all_commands[joint_idx]
            state_array = all_states[joint_idx]
            num_samples = len(cmd_array)
            times = np.arange(num_samples) / args.control_hz
            plot_joint(
                ax,
                times,
                cmd_array,
                state_array,
                joint_idx,
                args.sin_frequency,
            )

        # Turn off the unused 8th subplot
        axes_flat[NUM_JOINTS].set_visible(False)

        fig.suptitle(
            f"Sine Benchmark ({args.side} arm) | {args.sin_frequency} Hz, "
            f"amp={args.amplitude} rad, "
            f"dur={args.duration} s, "
            f"ctrl={args.control_hz} Hz",
            fontsize=14,
        )
        fig.tight_layout(rect=(0, 0, 1, 0.97))
        fig.savefig(result_dir / "summary.png", dpi=150)
        plt.close(fig)
        logger.info(f"Summary plot saved to {result_dir / 'summary.png'}")

        # ------------------------------------------------------------------
        # Save raw data
        # ------------------------------------------------------------------
        save_dict: dict[str, Any] = {
            "side": args.side,
            "duration": args.duration,
            "control_hz": args.control_hz,
            "amplitude": args.amplitude,
            "sin_frequency": args.sin_frequency,
            "zero_pos": ZERO_POS,
        }
        for joint_idx in range(NUM_JOINTS):
            save_dict[f"commands_{joint_idx}"] = all_commands[joint_idx]
            save_dict[f"states_{joint_idx}"] = all_states[joint_idx]

        data_path = result_dir / "data.npz"
        np.savez(data_path, **save_dict)
        logger.info(f"Raw data saved to {data_path}")

        # ------------------------------------------------------------------
        # Cleanup
        # ------------------------------------------------------------------
        logger.info("Returning to zero position")
        arm.move_to_joint_pos(ZERO_POS, wait=True, timeout=3.0)
    logger.info("Benchmark complete")


if __name__ == "__main__":
    main(tyro.cli(Args))

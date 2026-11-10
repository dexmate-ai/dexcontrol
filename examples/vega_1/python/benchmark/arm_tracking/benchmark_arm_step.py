# Copyright (C) 2025-2026 Dexmate Inc.
# SPDX-License-Identifier: Apache-2.0

"""Benchmark each arm joint with a ramped step trajectory.

Prompt unless --no-confirm; move the right seven-joint arm to [0,0,0,-0.5,0,0,0] rad. Test each joint for 5 s at 200 Hz: hold 0.3 s, ramp by pi/6 rad over 0.3 s, then hold. The step is fitted to each joint's limits from the reference pose: the other direction where the requested one does not fit, a smaller step where neither does, each logged. Stream positions without velocity feed-forward and with the step guard disabled (tracking lag is what is measured); return to the reference between joints and at the end. Write measurement files.

Writes per-joint CSV, data.npz and summary.png under results beside this script; requires matplotlib.

Uses a Robot control connection: compatible arm modes may be enabled during startup, and shutdown requests a stop. Real hardware is selected unless --simulated is supplied."""

import time
import warnings
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
    step_amplitude: float,
    step_time: float,
) -> None:
    """Plot command vs actual for one joint on the given axes.

    Args:
        ax: Matplotlib axes to draw on.
        times: (N,) time array in seconds.
        cmd: (N, 7) command array.
        actual: (N, 7) state array.
        joint_idx: Index of the tested joint.
        step_amplitude: Step amplitude in radians (for the title).
        step_time: Time at which the step occurs in seconds.
    """
    ax.plot(times, cmd[:, joint_idx], color="tab:blue", label="Command")
    ax.plot(times, actual[:, joint_idx], color="tab:orange", label="Actual")
    ax.axvline(
        step_time, color="tab:green", linestyle="--", alpha=0.7, label="Step time"
    )
    ax.set_title(f"Joint {joint_idx} - Step ({np.rad2deg(step_amplitude):.1f} deg)")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Position (rad)")
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


@dataclass
class Args:
    """Step response latency benchmark for all 7 joints of an arm."""

    settle: float = 0.5
    simulated: bool = False
    profile: Optional[str] = None
    config: Optional[str] = None

    side: Literal["left", "right"] = "right"
    """Which arm to test (left or right)."""

    duration: float = 5.0
    """Duration of the step test per joint in seconds."""

    control_hz: int = 200
    """Control loop frequency in Hz."""

    step_amplitude: float = np.deg2rad(30.0)
    """Amplitude of the step input in radians."""

    step_time: float = 0.3
    """Time at which the step transition starts in seconds."""

    transition_time: float = 0.3
    """Duration of the step transition in seconds."""

    max_vel: float | None = None
    """Maximum velocity in rad/s. If None, auto-calculated from transition_time."""

    output_dir: str = "results"
    """Base output directory for results."""

    no_confirm: bool = False
    """Skip the interactive safety confirmation prompt."""


def main(args: Args) -> None:
    """Run the step response latency benchmark on all 7 joints."""

    validate_sample_count(args.duration, args.control_hz)
    if args.duration * args.control_hz < 1:
        raise ValueError("benchmark duration must include at least one sample")
    validate_sample_count(args.settle, 1.0)
    if not np.isfinite(args.transition_time) or args.transition_time <= 0:
        raise ValueError("transition_time must be finite and positive")
    max_vel = args.max_vel if args.max_vel is not None else abs(args.step_amplitude) / args.transition_time
    if not np.isfinite(max_vel) or max_vel <= 0:
        raise ValueError("max_vel must be finite and positive")
    validate_sample_count(args.step_time, args.control_hz)
    validate_sample_count(abs(args.step_amplitude) / max_vel, args.control_hz)

    # ------------------------------------------------------------------
    # Safety warning
    # ------------------------------------------------------------------
    warnings.warn(
        "This benchmark moves the robot arm through step trajectories. "
        "No collision checking is performed. Ensure the workspace is clear.",
        stacklevel=1,
    )
    logger.warning(
        "The robot arm will move. Make sure the workspace is clear "
        "and the e-stop is accessible."
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
        result_dir = output_base / f"{timestamp_str}_{args.side}_step"
        result_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Results will be saved to {result_dir}")

        # ------------------------------------------------------------------
        # Pre-compute step trajectory (shared across all joints)
        # ------------------------------------------------------------------
        max_vel = args.max_vel
        if max_vel is None:
            max_vel = abs(args.step_amplitude) / args.transition_time

        def step_profile(step_amplitude: float) -> np.ndarray:
            """Hold, ramp to the step at max_vel, hold; one sample per tick."""
            # Ramp from 0 to the step amplitude at max_vel, excluding the start
            # point so the first streamed sample already advances the joint.
            ramp_steps = max(1, int(abs(step_amplitude) / max_vel * args.control_hz))
            ramp = np.linspace([0.0], [step_amplitude], ramp_steps + 1)
            transition_trajectory = ramp[1:]
            num_samples_total = int(args.duration * args.control_hz)
            num_samples_before = int(args.step_time * args.control_hz)
            pos_traj = np.concatenate(
                [
                    np.full(num_samples_before, 0.0),
                    transition_trajectory.flatten(),
                    np.full(
                        max(
                            0,
                            num_samples_total - num_samples_before - len(transition_trajectory),
                        ),
                        step_amplitude,
                    ),
                ]
            )
            # Truncate to exact duration
            return pos_traj[:num_samples_total]

        # The step is fitted to each joint's limits from the reference pose
        # (direction flipped, or size reduced, with a warning).
        step_amplitudes: list[float] = []

        # ------------------------------------------------------------------
        # Move to zero position and verify
        # ------------------------------------------------------------------
        logger.info(f"Moving {arm_name} to zero position")
        arm.move_to_joint_pos(ZERO_POS, wait=True, timeout=3.0)
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
                arm.move_to_joint_pos(ZERO_POS, wait=True, timeout=2.0)

            step_amplitude = fit_amplitude(arm, joint_idx, args.step_amplitude, symmetric=False)
            step_amplitudes.append(step_amplitude)
            pos_traj = step_profile(step_amplitude)

            sample_times = []
            commands: list[np.ndarray] = []
            states: list[np.ndarray] = []
            rate_limiter = RateLimiter(args.control_hz)
            start_time = time.monotonic()
            traj_idx = 0

            while traj_idx < len(pos_traj):
                step_value = pos_traj[traj_idx]

                joint_pos = ZERO_POS.copy()
                joint_pos[joint_idx] += step_value

                # The step guard refuses a setpoint more than 0.3 rad from the
                # measured position. Lag is what this benchmark measures, so
                # the guard is disabled for the stream (max_step=None).
                arm.set_joint_pos(joint_pos, max_step=None)
                commands.append(joint_pos.copy())
                states.append(arm.get_joint_pos().copy())
                sample_times.append(time.monotonic() - start_time)
                rate_limiter.sleep()
                traj_idx += 1

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
                step_amplitudes[joint_idx],
                args.step_time,
            )

        # Turn off the unused 8th subplot
        axes_flat[NUM_JOINTS].set_visible(False)

        fig.suptitle(
            f"Step Benchmark ({args.side} arm) | amp={np.rad2deg(args.step_amplitude):.1f} deg, "
            f"step_t={args.step_time} s, "
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
            "step_amplitude": args.step_amplitude,
            "step_amplitudes": np.array(step_amplitudes),
            "step_time": args.step_time,
            "transition_time": args.transition_time,
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

#!/usr/bin/env python3
# Copyright (C) 2026 Dexmate Inc.
# SPDX-License-Identifier: Apache-2.0

"""Prepare robot joint positions and replay a recorded trajectory.

Require a trajectory file and confirmation before motion. Move to preparation/start targets, then stream recorded joint positions with the configured tracking-error guard. Preparation motions and processing options differ by language; see the language notes.

CSV input (the format all three languages share; NPZ recordings are still accepted). Prompt, optionally smooth/resample/plot, then prompt before folding both arms, sending head home, crouching torso to crouch20_medium, re-resolving head home for the new torso posture, and closing hands. Press Enter before moving to the recorded first row and streaming. Defaults: 0.1 s position smoothing, speed factor 1, no velocity feed-forward; rate from file or 500 Hz.

Uses a Robot control connection: compatible arm modes may be enabled during startup, and shutdown requests a stop. Real hardware is selected unless --simulated is supplied."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import tyro
from dexcontrol import ask, confirm
from dexcontrol.robot import Robot
from loguru import logger
from typing_extensions import Annotated


def load_trajectory(filepath: Path) -> tuple[dict[str, np.ndarray], float]:
    """Load a trajectory recording: one component-by-tick array per component.

    The recording format shared by all three languages is CSV: an optional
    first line ``control_hz,<rate>``, a header of ``component:joint_index``
    columns grouped by component, then one row per tick. NPZ recordings (one
    tick-by-joint array per component, optional ``control_hz``) are still
    accepted for existing captures.
    """
    if not filepath.exists():
        raise FileNotFoundError(f"File not found: {filepath}")
    control_hz = 500.0  # default
    trajectory: dict[str, np.ndarray] = {}

    if filepath.suffix.lower() == ".npz":
        data = np.load(filepath, allow_pickle=False)
        for key in data.files:
            if key in ["control_hz", "control_frequency"]:
                control_hz = float(data[key].item())
            else:
                trajectory[key] = np.asarray(data[key], dtype=float)
    else:
        with filepath.open(encoding="utf-8") as file:
            line = file.readline().strip()
            if line.startswith("control_hz,"):
                control_hz = float(line.split(",", 1)[1])
                line = file.readline().strip()
            columns = []
            for column in line.split(","):
                part, _, joint_index_text = column.partition(":")
                if not joint_index_text.isdigit():
                    raise ValueError(f"bad header column {column!r} in {filepath}")
                columns.append((part, int(joint_index_text)))
            matrix = np.loadtxt(file, delimiter=",", ndmin=2)
        if matrix.shape[1] != len(columns):
            raise ValueError(f"{filepath}: {matrix.shape[1]} columns of data for {len(columns)} header columns")
        for index, (part, joint_index) in enumerate(columns):
            width = max(j for p, j in columns if p == part) + 1
            track = trajectory.setdefault(part, np.zeros((matrix.shape[0], width)))
            track[:, joint_index] = matrix[:, index]

    if control_hz != 500.0:
        logger.info(f"Found control frequency: {control_hz}Hz")
    for part, track in trajectory.items():
        logger.info(f"Loaded {part}: shape {track.shape}")
    return trajectory, control_hz


def resample_trajectory(
    trajectory: dict[str, np.ndarray], speed_factor: float
) -> dict[str, np.ndarray]:
    """Resample trajectory by speed factor (>1 = faster, <1 = slower)."""
    if speed_factor == 1.0:
        return trajectory

    resampled = {}
    for part, positions in trajectory.items():
        old_len = len(positions)
        new_len = int(np.ceil(old_len / speed_factor))
        old_indices = np.linspace(0, old_len - 1, old_len)
        new_indices = np.linspace(0, old_len - 1, new_len)

        # Interpolate each joint
        new_positions = np.zeros((new_len, positions.shape[1]))
        for j in range(positions.shape[1]):
            new_positions[:, j] = np.interp(new_indices, old_indices, positions[:, j])

        resampled[part] = new_positions

    return resampled


def smooth_trajectory(
    trajectory: dict[str, np.ndarray], sigma_time: float, hz: float
) -> dict[str, np.ndarray]:
    """Apply Gaussian smoothing to trajectory.

    Args:
        trajectory: Trajectory data
        sigma_time: Smoothing window in seconds (e.g., 0.1 for 100ms window)
        hz: Control frequency

    Returns:
        Smoothed trajectory
    """
    if sigma_time <= 0:
        return trajectory

    from scipy.ndimage import gaussian_filter1d

    # Convert the time-based sigma to samples. Sigma is a standard deviation,
    # so roughly 99.7% of the filter's weight falls within +/-3 sigma.
    sigma_samples = sigma_time * hz

    logger.info(
        f"Smoothing with {sigma_time:.3f}s window = {sigma_samples:.1f} samples at {hz}Hz"
    )

    return {
        part: gaussian_filter1d(positions, sigma=sigma_samples, axis=0, mode="nearest")
        for part, positions in trajectory.items()
    }


def compute_velocities(
    trajectory: dict[str, np.ndarray], hz: float, smooth_time: float = 0.01
) -> dict[str, np.ndarray]:
    """Compute velocities using finite differences.

    Args:
        trajectory: Position trajectory
        hz: Control frequency
        smooth_time: Smoothing window in seconds for velocity estimation

    Returns:
        Velocity trajectory
    """
    dt = 1.0 / hz
    velocities = {}

    for part, positions in trajectory.items():
        if len(positions) < 2:
            velocities[part] = np.zeros_like(positions)
            continue

        # Smooth before differentiation (convert time to samples)
        if smooth_time > 0:
            from scipy.ndimage import gaussian_filter1d

            sigma_samples = smooth_time * hz
            positions = gaussian_filter1d(
                positions, sigma=sigma_samples, axis=0, mode="nearest"
            )

        # Compute velocities
        vel = np.zeros_like(positions)
        vel[0] = (positions[1] - positions[0]) / dt
        vel[1:-1] = (positions[2:] - positions[:-2]) / (2 * dt)
        vel[-1] = (positions[-1] - positions[-2]) / dt

        velocities[part] = vel

    return velocities


def visualize_trajectory(
    trajectory: dict[str, np.ndarray],
    velocities: dict[str, np.ndarray] | None,
    hz: float,
) -> None:
    """Visualize trajectory with separate figures for each part."""
    import matplotlib.pyplot as plt

    # Set up time array
    num_frames = len(next(iter(trajectory.values())))
    time_array = np.arange(num_frames) / hz

    # Define colors for different joints
    colors = plt.get_cmap("tab10")(np.linspace(0, 1, 10))

    # Create figures for each part
    for part, positions in trajectory.items():
        # Create figure with two subplots always; hide velocity axis if not used
        if velocities and part in velocities:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
            fig.suptitle(
                f"{part.upper()} - Trajectory Commands", fontsize=18, fontweight="bold"
            )
        else:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
            fig.suptitle(
                f"{part.upper()} - Position Commands", fontsize=18, fontweight="bold"
            )
            ax2.axis("off")

        # Plot positions
        num_joints = positions.shape[1]
        for j in range(num_joints):
            ax1.plot(
                time_array,
                positions[:, j],
                color=colors[j % len(colors)],
                linewidth=2,
                label=f"Joint {j + 1}",
                alpha=0.8,
            )

        ax1.set_title("Position Commands", fontsize=14)
        ax1.set_xlabel("Time (s)", fontsize=12)
        ax1.set_ylabel("Position (rad)", fontsize=12)
        ax1.grid(True, alpha=0.3)
        ax1.legend(loc="best", frameon=True, fancybox=True, shadow=True)

        # Add min/max annotations
        for j in range(min(num_joints, 3)):  # Limit annotations to avoid clutter
            joint_data = positions[:, j]
            min_idx = np.argmin(joint_data)
            max_idx = np.argmax(joint_data)

            # Annotate min
            ax1.annotate(
                f"J{j + 1}: {joint_data[min_idx]:.3f}",
                xy=(time_array[min_idx], joint_data[min_idx]),
                xytext=(5, -15),
                textcoords="offset points",
                fontsize=8,
                alpha=0.7,
                arrowprops={"arrowstyle": "->", "alpha": 0.5},
            )

            # Annotate max
            ax1.annotate(
                f"J{j + 1}: {joint_data[max_idx]:.3f}",
                xy=(time_array[max_idx], joint_data[max_idx]),
                xytext=(5, 15),
                textcoords="offset points",
                fontsize=8,
                alpha=0.7,
                arrowprops={"arrowstyle": "->", "alpha": 0.5},
            )

        # Plot velocities if available
        if velocities and part in velocities:
            vels = velocities[part]
            for j in range(num_joints):
                ax2.plot(
                    time_array,
                    vels[:, j],
                    color=colors[j % len(colors)],
                    linewidth=2,
                    label=f"Joint {j + 1}",
                    alpha=0.8,
                )

            ax2.set_title("Velocity Commands", fontsize=14)
            ax2.set_xlabel("Time (s)", fontsize=12)
            ax2.set_ylabel("Velocity (rad/s)", fontsize=12)
            ax2.grid(True, alpha=0.3)
            ax2.legend(loc="best", frameon=True, fancybox=True, shadow=True)

            # Add zero line
            ax2.axhline(y=0, color="black", linestyle="--", alpha=0.3)

            # Add max velocity annotations
            for j in range(min(num_joints, 3)):  # Limit annotations
                vel_data = vels[:, j]
                max_vel_idx = np.argmax(np.abs(vel_data))
                ax2.annotate(
                    f"J{j + 1}: {vel_data[max_vel_idx]:.3f}",
                    xy=(time_array[max_vel_idx], vel_data[max_vel_idx]),
                    xytext=(5, 10),
                    textcoords="offset points",
                    fontsize=8,
                    alpha=0.7,
                    arrowprops={"arrowstyle": "->", "alpha": 0.5},
                )

        plt.tight_layout()

    # Show all figures at once
    plt.show()


def check_goal_difference(diff: dict[str, np.ndarray], max_goal_diff: float) -> bool:
    """Check maximum absolute joint difference and compare with threshold.

    Args:
        diff: Mapping from part name to per-joint delta array (frame - current).
        max_goal_diff: Maximum allowed absolute joint delta in radians.

    Returns:
        True if all absolute joint deltas are within the allowed threshold.
        False if any joint delta exceeds the threshold (also logs details).
    """
    max_diff_part: str | None = None
    max_joint_idx: int = -1
    max_diff: float = 0.0

    for part, delta in diff.items():
        abs_delta = np.abs(delta)
        part_max_idx = int(np.argmax(abs_delta))
        part_max_val = float(abs_delta[part_max_idx])
        if part_max_val > max_diff:
            max_diff = part_max_val
            max_diff_part = part
            max_joint_idx = part_max_idx
    if max_diff > max_goal_diff:
        logger.warning(
            f"Max |diff| between current and target position: {max_diff:.3f} rad in {max_diff_part} joint {max_joint_idx}"
        )
        logger.warning(
            f"This is greater than the allowed goal difference of {max_goal_diff:.3f} rad"
        )
        logger.warning("Exiting...")
        return False

    return True


def limit_violations(robot: Robot, trajectory: dict[str, np.ndarray]) -> dict:
    """Samples outside the robot model's joint limits, per joint.

    The native executor refuses a trajectory at the first sample outside the
    limits, which names one joint and one value. A recording made on a robot
    whose firmware allowed a little more travel than the model does fails
    that way thousands of samples later than the operator expects, so the
    whole file is checked here first and reported per joint.
    """
    report = {}
    for part, positions in trajectory.items():
        if not robot.has_component(part):
            continue
        joints = robot.joints(part)
        limits = joints.joint_pos_limit
        if limits is None:
            continue
        limits = np.asarray(limits, dtype=float)
        for index, name in enumerate(joints.joint_names):
            lower, upper = limits[index]
            column = positions[:, index]
            outside = int(((column < lower) | (column > upper)).sum())
            if outside:
                report[(part, name)] = (
                    outside, float(column.min()), float(column.max()), lower, upper
                )
    return report


def clamp_to_limits(robot: Robot, trajectory: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Clips every sample into the model's joint limits."""
    clamped = {}
    for part, positions in trajectory.items():
        limits = robot.joints(part).joint_pos_limit if robot.has_component(part) else None
        if limits is None:
            clamped[part] = positions
            continue
        limits = np.asarray(limits, dtype=float)
        clamped[part] = np.clip(positions, limits[:, 0], limits[:, 1])
    return clamped


def run_replay_loop(
    robot: Robot,
    trajectory: dict[str, np.ndarray],
    velocities: dict[str, np.ndarray] | None,
    hz: float,
    max_goal_diff: float | None = None,
) -> bool:
    """Validate, move to the first frame, and stream the trajectory in Rust.

    The native executor preflights every waypoint before publishing, releases
    the GIL for the complete stream, and checks cached joint tracking error on
    every tick when ``max_goal_diff`` is provided.
    """
    available = {
        part: positions
        for part, positions in trajectory.items()
        if robot.has_component(part)
    }
    if not available:
        raise ValueError("trajectory contains no components available on this robot")

    start = {part: positions[0] for part, positions in available.items()}
    robot.move_to_joint_pos(start, wait=False).wait(timeout=3.0)

    tracks: dict[str, object] = {}
    for part, positions in available.items():
        if velocities is not None and part in velocities:
            tracks[part] = {
                "positions": positions,
                "velocities": velocities[part],
            }
        else:
            tracks[part] = positions

    try:
        robot.execute_trajectory(
            tracks,
            control_hz=hz,
            max_tracking_error=max_goal_diff,
        )
    except KeyboardInterrupt:
        logger.info("Interrupted")
        return False
    return True


def replay_trajectory(
    filepath: Path,
    control_hz: float | None = None,
    gaussian_sigma: float = 0.0,
    use_velocity: bool = False,
    velocity_sigma: float = 1.0,
    speed_factor: float = 1.0,
    visualize: bool = False,
    max_goal_diff: float = 0.7,
    clamp_to_model_limits: bool = False,
    *,
    simulated: bool = False,
    profile: str | None = None,
    config: str | None = None,
):
    """Main replay function."""
    # Load trajectory
    trajectory, file_hz = load_trajectory(filepath)
    if not trajectory:
        logger.error("Empty trajectory")
        return

    # Set control frequency
    hz = control_hz or file_hz

    # Get original number of frames and duration
    original_frames = len(next(iter(trajectory.values())))
    original_duration = original_frames / hz

    # Apply speed factor by resampling
    if speed_factor != 1.0:
        logger.info(f"Applying speed factor {speed_factor}x")
        trajectory = resample_trajectory(trajectory, speed_factor)

    # Get actual number of frames after resampling
    num_frames = len(next(iter(trajectory.values())))
    duration = num_frames / hz

    logger.info(f"Original: {original_frames} frames, {original_duration:.1f}s")
    logger.info(f"Playback: {num_frames} frames @ {hz}Hz ({duration:.1f}s)")
    logger.info(f"Parts: {', '.join(trajectory.keys())}")

    # Apply smoothing
    if gaussian_sigma > 0:
        logger.info(f"Applying Gaussian smoothing (sigma={gaussian_sigma})")
        trajectory = smooth_trajectory(trajectory, gaussian_sigma, hz)

    # Compute velocities
    velocities = None
    if use_velocity:
        logger.info(f"Computing velocities (sigma={velocity_sigma})")
        velocities = compute_velocities(trajectory, hz, velocity_sigma)

    # Visualize if requested
    if visualize:
        logger.info("Visualizing trajectory...")
        visualize_trajectory(trajectory, velocities, hz)
        if not confirm("Continue with execution?"):
            logger.info("Execution cancelled")
            return

    with Robot(simulation=simulated, profile=profile, config_file=config) as robot:
        violations = limit_violations(robot, trajectory)
        if violations:
            for (part, name), (count, low, high, lower, upper) in violations.items():
                logger.warning(
                    f"{part}.{name}: {count} samples outside the model limits "
                    f"[{lower:.3f}, {upper:.3f}] (trajectory spans {low:.3f} .. {high:.3f})"
                )
            if not clamp_to_model_limits:
                logger.error(
                    "The trajectory exceeds the robot model's joint limits and would be "
                    "refused at the first such sample. Either the recording is out of "
                    "spec or the model limits are stale; re-run with --clamp-to-model-limits "
                    "to clip those samples, after checking the clipped motion is acceptable."
                )
                raise SystemExit(1)
            trajectory = clamp_to_limits(robot, trajectory)
            logger.warning("Clipped the trajectory into the model's joint limits.")
        logger.warning("Setting joint positions...")
        logger.warning("Press e-stop if needed!")
        logger.warning(
            "Please ensure the arms and the torso have sufficient space to move."
        )
        logger.warning(
            "If you have an end effector attached, some pre-existing trajectories may cause collisions with the robot."
        )
        logger.warning(
            "Will move the left arm to folded, the right arm to folded, and the head to home pose, then the torso to crouch20_medium and head home again."
        )
        if not confirm("Continue?"):
            logger.info("Execution cancelled")
            return

        # Build initial pose dict based on available components
        init_pose = {
            "left_arm": robot.left_arm.resolve_pose("folded"),
            "right_arm": robot.right_arm.resolve_pose("folded"),
        }
        if robot.has_component("head"):
            init_pose["head"] = robot.head.resolve_pose("home")

        handle = robot.move_to_joint_pos(init_pose)
        assert handle is not None
        handle.wait(timeout=5.0)

        if robot.has_component("torso"):
            robot.torso.go_to_pose("crouch20_medium", timeout=5.0)
            if robot.has_component("head"):
                robot.head.go_to_pose("home", timeout=5.0)

        if robot.have_hand("left"):
            robot.left_hand.close_hand()
        if robot.have_hand("right"):
            robot.right_hand.close_hand()

        ask("Press Enter to start the replay...")
        # Replay
        run_replay_loop(
            robot=robot,
            trajectory=trajectory,
            velocities=velocities if use_velocity else None,
            hz=hz,
            max_goal_diff=max_goal_diff,
        )

        logger.info("Done")


@dataclass
class Args:
    """Command-line options."""

    file: Annotated[
        Path, tyro.conf.Positional, tyro.conf.arg(help="trajectory recording (CSV; NPZ still accepted)")
    ]
    control_hz: Annotated[
        Optional[float],
        tyro.conf.arg(help="Control frequency (default: from file or 500Hz)"),
    ] = None
    smooth: Annotated[
        float,
        tyro.conf.arg(
            help="Position smoothing time window in seconds (e.g., 0.1 for 100ms)"
        ),
    ] = 0.1
    velocity_compensation: Annotated[
        bool, tyro.conf.arg(help="Send velocity feed-forward alongside positions")
    ] = False
    vel_smooth: Annotated[
        float,
        tyro.conf.arg(
            help="Velocity smoothing time window in seconds (default: 0.5s = 500ms)"
        ),
    ] = 0.5
    speed_factor: Annotated[
        float, tyro.conf.arg(help="Speed factor (2=2x faster, 0.5=2x slower)")
    ] = 1.0
    visualize: Annotated[
        bool, tyro.conf.arg(help="Visualize trajectory before execution")
    ] = False
    clamp_to_model_limits: Annotated[
        bool,
        tyro.conf.arg(
            help="Clip samples outside the robot model's joint limits instead of refusing the file"
        ),
    ] = False
    max_goal_diff: Annotated[
        float, tyro.conf.arg(help="Maximum goal difference in radians (default: 1.0)")
    ] = 1.0
    simulated: Annotated[bool, tyro.conf.arg(help="Use the in-process simulation")] = (
        False
    )
    profile: Annotated[Optional[str], tyro.conf.arg(help="Built-in robot profile")] = (
        None
    )
    config: Annotated[
        Optional[str], tyro.conf.arg(help="Custom robot configuration")
    ] = None


def main():
    args = tyro.cli(Args, description=__doc__, config=(tyro.conf.FlagCreatePairsOff,))
    logger.warning(
        "Warning: Be ready to press e-stop if needed! "
        "This example does not check for self-collisions."
    )
    logger.warning(
        "Please ensure the arms and the torso have sufficient space to move."
    )
    if not confirm("Continue?"):
        return

    if args.file.name.startswith("vega-1_dance"):
        speed = np.clip(args.speed_factor, 0.2, 3.0)
        if speed != args.speed_factor:
            logger.warning(f"Speed factor clamped to {speed} (valid range: 0.2-3.0)")
            args.speed_factor = speed
    else:
        speed = args.speed_factor

    if speed > 3.0:
        logger.warning("Speed factor is greater than 3.0!!! This can be dangerous!!!")
        if not confirm("Continue?"):
            return

    replay_trajectory(
        args.file,
        control_hz=args.control_hz,
        gaussian_sigma=args.smooth,
        use_velocity=args.velocity_compensation,
        velocity_sigma=args.vel_smooth,
        speed_factor=speed,
        visualize=args.visualize,
        max_goal_diff=args.max_goal_diff,
        clamp_to_model_limits=args.clamp_to_model_limits,
        simulated=args.simulated,
        profile=args.profile,
        config=args.config,
    )


if __name__ == "__main__":
    main()

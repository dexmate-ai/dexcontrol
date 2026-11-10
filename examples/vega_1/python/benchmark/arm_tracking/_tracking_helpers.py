# Copyright (C) 2025-2026 Dexmate Inc.
# SPDX-License-Identifier: Apache-2.0

"""Shared seven-joint tracking reference, CSV, and validation helpers.

Imported by arm benchmarks; the reference vector is [0,0,0,-0.5,0,0,0] rad. Contains no standalone benchmark or robot connection."""

import numpy as np
from loguru import logger

NUM_JOINTS = 7
ZERO_POS = np.array([0.0, 0.0, 0.0, -0.5, 0.0, 0.0, 0.0])
ZERO_TOLERANCE = 0.05  # rad
LIMIT_MARGIN = 0.02  # rad kept clear of a joint limit by test excursions
VELOCITY_MARGIN = 0.02  # fraction of the velocity limit kept clear by feed-forward


def fit_amplitude(arm, joint_idx: int, amplitude: float, *, symmetric: bool, omega: float = 0.0) -> float:
    """Amplitude of a test excursion from ZERO_POS that stays within the joint's limits.

    The client refuses a target outside the model's joint limits, and the
    reference pose sits close to a limit on some joints (R_arm_j2 has 0.45 rad
    of room upward for a pi/6 step). For a step (``symmetric=False``) the
    signed step is returned: the requested direction if it fits, the other
    direction if only that fits, otherwise the largest step that fits. For a
    sine (``symmetric=True``) the magnitude is returned, limited by the
    smaller of the two sides and, given the angular frequency ``omega``
    (rad/s), by the joint's velocity limit: the feed-forward peaks at
    ``amplitude * omega`` and the client refuses one above the limit (0.4 rad
    at 1 Hz is 2.51 rad/s against R_arm_j1's 2.4 rad/s). Every change is
    logged.
    """
    name = arm.joint_names[joint_idx]
    size = abs(float(amplitude))
    if symmetric and omega > 0.0 and arm.joint_vel_limit is not None:
        velocity_limit = float(arm.joint_vel_limit[joint_idx])
        room = velocity_limit * (1.0 - VELOCITY_MARGIN) / omega
        if size > room:
            logger.warning(
                f"{name}: sine amplitude reduced from {size:.3f} to {room:.3f} rad so the "
                f"velocity feed-forward stays within {velocity_limit:.2f} rad/s"
            )
            size = room
    limits = arm.joint_pos_limit
    if limits is None:
        return size if symmetric else float(amplitude)
    lower, upper = (float(v) for v in np.asarray(limits, dtype=float)[joint_idx])
    zero = float(ZERO_POS[joint_idx])
    room_up = max(0.0, upper - zero - LIMIT_MARGIN)
    room_down = max(0.0, zero - lower - LIMIT_MARGIN)
    if symmetric:
        room = min(room_up, room_down)
        if size <= room:
            return size
        logger.warning(
            f"{name}: sine amplitude reduced from {size:.3f} to {room:.3f} rad "
            f"to stay within [{lower:.3f}, {upper:.3f}]"
        )
        return room
    wanted_up = amplitude >= 0
    if size <= (room_up if wanted_up else room_down):
        return float(amplitude)
    if size <= (room_down if wanted_up else room_up):
        logger.warning(
            f"{name}: stepping {'negative' if wanted_up else 'positive'} instead; "
            f"{amplitude:+.3f} rad from the reference would leave [{lower:.3f}, {upper:.3f}]"
        )
        return -size if wanted_up else size
    best = room_up if room_up >= room_down else -room_down
    logger.warning(
        f"{name}: step reduced to {best:+.3f} rad to stay within [{lower:.3f}, {upper:.3f}]"
    )
    return best


def verify_zero_position(arm) -> None:
    """Read current joint positions and verify they match ZERO_POS.

    Args:
        arm: Arm component to read joint positions from.

    Raises:
        RuntimeError: If any joint deviates from ZERO_POS by more than
            ZERO_TOLERANCE.
    """
    actual = arm.get_joint_pos()
    errors = np.abs(actual - ZERO_POS)

    # Log actual positions so the user can see them
    pos_str = ", ".join(f"{v:.4f}" for v in actual)
    err_str = ", ".join(f"{v:.4f}" for v in errors)
    logger.info(f"Actual joint positions: [{pos_str}]")
    logger.info(f"Position errors:        [{err_str}]")

    bad = np.where(errors > ZERO_TOLERANCE)[0]
    if len(bad) > 0:
        details = ", ".join(
            f"joint {j}: actual={actual[j]:.4f}, err={errors[j]:.4f} rad" for j in bad
        )
        raise RuntimeError(
            f"Zero-position check failed (tolerance={ZERO_TOLERANCE} rad). "
            f"Failed joints: {details}"
        )
    logger.info("Zero position verified (all joints within tolerance)")


def validate_sample_count(seconds: float, hz: float) -> None:
    count = seconds * hz
    if not np.isfinite(seconds) or seconds < 0 or not np.isfinite(hz) or hz <= 0 or not np.isfinite(count) or count > 1_000_000:
        raise ValueError("benchmark requires non-negative duration, positive rate, and at most 1000000 samples")


def write_tracking_csv(path, times, commands, actual):
    header = "time_s," + ",".join(f"cmd_{j}" for j in range(7)) + "," + ",".join(f"actual_{j}" for j in range(7))
    data = np.column_stack((times, np.asarray(commands).reshape(-1, 7), np.asarray(actual).reshape(-1, 7)))
    np.savetxt(path, data, delimiter=",", header=header, comments="", fmt="%.9f")

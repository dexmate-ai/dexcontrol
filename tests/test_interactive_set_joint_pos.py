# Copyright (C) 2025 Dexmate Inc.
#
# This software is dual-licensed:
#
# 1. GNU Affero General Public License v3.0 (AGPL-3.0)
#    See LICENSE-AGPL for details
#
# 2. Commercial License
#    For commercial licensing terms, contact: contact@dexmate.ai

"""Interactive manual test for Robot.set_joint_pos().

This is NOT a pytest unit test -- it drives a real robot. It:

    1. Unfolds the robot to its operational (initial) position.
    2. Drops into an interactive prompt where you pick a component
       (left_arm / right_arm / head / torso) and type new joint
       positions. Each change is sent with ``bot.set_joint_pos(...)``.

``Robot.set_joint_pos`` is non-blocking: it fires a single setpoint and
returns immediately. After each command this script waits ``--settle``
seconds for the motion to finish, then reports the achieved delta.

Usage:
    python tests/test_interactive_set_joint_pos.py
    python tests/test_interactive_set_joint_pos.py --settle 3.0 --no-unfold
    python tests/test_interactive_set_joint_pos.py --degrees   # enter values in degrees

Keys at the component prompt:
    left_arm / right_arm / head / torso : select that component
    q                                   : quit

At the joint-value prompt:
    <space/comma separated floats>      : new absolute target for every joint
    <blank>                             : keep current position (no move)
    b                                   : back to component selection
"""

import time
from typing import Final

import numpy as np
import tyro
from loguru import logger

from dexcontrol.exceptions import DexcontrolError
from dexcontrol.robot import Robot

# Components this script can drive and their expected joint counts.
COMPONENTS: Final[dict[str, int]] = {
    "left_arm": 7,
    "right_arm": 7,
    "head": 3,
    "torso": 3,
}


def unfold_robot(bot: Robot, velocity_scale: float = 0.5) -> None:
    """Move the robot from folded to its operational (initial) position.

    Mirrors ``examples/advanced_examples/fold_robot.py:unfold_robot`` but
    reuses the already-open ``bot`` so the interactive session continues in
    the same connection.

    Args:
        bot: An open Robot instance.
        velocity_scale: Default velocity scale in (0, 1] for all joint motion.

    Raises:
        RuntimeError: If torso or arms fail to reach their targets.
    """
    torso_pose: Final[str] = "crouch45_high"
    arm_pose: Final[str] = "L_shape"
    arm_joints: Final[list[int]] = list(range(1, 7))

    bot.default_velocity_scale = velocity_scale

    # Torso first.
    if not bot.torso.is_pose_reached(torso_pose):
        logger.info("Moving torso to operational position")
        handle = bot.move_to_joint_pos(
            {"torso": bot.torso.get_predefined_pose(torso_pose)},
        )
        assert handle is not None
        handle.wait(timeout=10.0)
    if not bot.torso.is_pose_reached(torso_pose):
        raise RuntimeError("Torso did not reach the target position! Exiting...")

    # Then arms.
    left_ready = bot.left_arm.is_pose_reached(arm_pose, joint_id=arm_joints)
    right_ready = bot.right_arm.is_pose_reached(arm_pose, joint_id=arm_joints)
    if not (left_ready and right_ready):
        logger.info("Moving arms to operational position")
        left_arm_pose = bot.compensate_torso_pitch(
            bot.left_arm.get_predefined_pose(arm_pose), "left_arm"
        )
        right_arm_pose = bot.compensate_torso_pitch(
            bot.right_arm.get_predefined_pose(arm_pose), "right_arm"
        )
        handle = bot.move_to_joint_pos(
            {"left_arm": left_arm_pose, "right_arm": right_arm_pose},
        )
        assert handle is not None
        handle.wait(timeout=6.0)

    left_ready = bot.left_arm.is_pose_reached(arm_pose, joint_id=arm_joints)
    right_ready = bot.right_arm.is_pose_reached(arm_pose, joint_id=arm_joints)
    if not (left_ready and right_ready):
        raise RuntimeError("Arms did not reach the target position! Exiting...")

    # Head home.
    head_home = bot.compensate_torso_pitch(bot.head.get_predefined_pose("home"), "head")
    handle = bot.head.move_to_joint_pos(head_home)
    assert handle is not None
    handle.wait(timeout=2.0)
    logger.info("Robot is unfolded and at its initial position!")


def _component_map(bot: Robot) -> dict[str, object]:
    """Map the names this script supports to their component objects."""
    return {
        "left_arm": bot.left_arm,
        "right_arm": bot.right_arm,
        "head": bot.head,
        "torso": bot.torso,
    }


def _read_current(component) -> np.ndarray:
    """Read the current joint position of a component as a numpy array."""
    return np.asarray(component.get_joint_pos(), dtype=float)


def _read_limits(component) -> np.ndarray | None:
    """Return the component's (N, 2) joint position limits, or None if absent."""
    limit = getattr(component, "_joint_pos_limit", None)
    if limit is None:
        return None
    return np.asarray(limit, dtype=float)


def _parse_targets(raw: str, n: int) -> np.ndarray | None:
    """Parse user input into an array of ``n`` joint targets.

    Accepts space- or comma-separated floats, with optional surrounding
    brackets, e.g. ``0.1 0.2 0.3``, ``0.1, 0.2, 0.3`` or ``[0.1, 0.2, 0.3]``.
    Returns None on malformed input.
    """
    cleaned = raw.strip().strip("[]()")
    tokens = cleaned.replace(",", " ").split()
    if len(tokens) != n:
        logger.error(f"Expected {n} values, got {len(tokens)}.")
        return None
    try:
        return np.array([float(t) for t in tokens], dtype=float)
    except ValueError:
        logger.error("Could not parse all values as floats.")
        return None


def interactive_loop(bot: Robot, settle: float, degrees: bool) -> None:
    """Prompt the user to select a component and set its joint positions.

    Args:
        bot: An open Robot instance.
        settle: Seconds to wait after each (non-blocking) ``set_joint_pos``
            call before reading back the achieved position.
        degrees: If True, joint values are entered/displayed in degrees and
            converted to radians before sending.
    """
    comp_objs = _component_map(bot)
    unit = "deg" if degrees else "rad"

    while True:
        choice = (
            input(f"\nSelect component {list(COMPONENTS)} or 'q' to quit: ")
            .strip()
            .lower()
        )
        if choice == "q":
            logger.info("Exiting interactive loop.")
            return
        if choice not in COMPONENTS:
            logger.error(f"Unknown component: {choice!r}")
            continue

        component = comp_objs[choice]
        limits = _read_limits(component)  # radians, shape (N, 2) or None
        while True:
            current = _read_current(component)
            shown = np.degrees(current) if degrees else current
            logger.info(
                f"[{choice}] current ({unit}): [{', '.join(f'{v:.3f}' for v in shown)}]"
            )
            if limits is not None:
                lo = np.degrees(limits[:, 0]) if degrees else limits[:, 0]
                hi = np.degrees(limits[:, 1]) if degrees else limits[:, 1]
                logger.info(
                    f"[{choice}] limits ({unit}): "
                    + ", ".join(
                        f"j{i}=[{low:.3f},{high:.3f}]"
                        for i, (low, high) in enumerate(zip(lo, hi))
                    )
                )
            raw = input(
                f"Enter {len(current)} target values in {unit} "
                f"e.g. '0.1 0.2 ...' or '[0.1, 0.2, ...]' "
                f"(blank=keep, 'b'=back): "
            ).strip()
            if raw == "":
                continue
            if raw.lower() == "b":
                break

            targets = _parse_targets(raw, len(current))
            if targets is None:
                continue
            if degrees:
                targets = np.radians(targets)  # API works in radians

            # Warn about values the firmware will clip back into range -- a
            # common reason a joint appears not to move.
            if limits is not None:
                clipped = np.clip(targets, limits[:, 0], limits[:, 1])
                for i, (req, cl) in enumerate(zip(targets, clipped)):
                    if not np.isclose(req, cl):
                        logger.warning(
                            f"[{choice}] j{i} target {req:.3f} rad is OUT OF RANGE "
                            f"-> will be clipped to {cl:.3f} rad (likely no/!partial motion)"
                        )

            logger.info(f"Sending set_joint_pos for {choice}")
            try:
                bot.set_joint_pos({choice: targets})
            except DexcontrolError as e:
                # e.g. the large-jump safety guard rejected this command.
                logger.error(f"[{choice}] set_joint_pos rejected: {e}")
                continue
            # set_joint_pos is non-blocking; wait for the motion to settle
            # before reading back the achieved position.
            time.sleep(settle)
            after = _read_current(component)
            delta = after - current
            logger.info(
                f"[{choice}] move complete. delta (rad): "
                f"[{', '.join(f'{d:+.4f}' for d in delta)}]"
            )
            if np.allclose(delta, 0.0, atol=1e-3):
                logger.warning(
                    f"[{choice}] joints did NOT move (delta ~0). Check limits above, "
                    "idle-mode, or whether this component honors set_joint_pos."
                )


def main(
    settle: float = 3.0,
    velocity_scale: float = 0.5,
    unfold: bool = True,
    degrees: bool = False,
    verbose: bool = True,
) -> None:
    """Run the interactive set_joint_pos test.

    Args:
        settle: Seconds to wait after each (non-blocking) ``set_joint_pos``
            call before reading back the achieved position.
        velocity_scale: Velocity scale in (0, 1] used during unfolding.
        unfold: If True, unfold to the initial position before the prompt.
        degrees: If True, enter and display joint values in degrees.
        verbose: If True, force the loguru level to DEBUG so any internal
            debug logs from the robot are visible.
    """
    if not 0.0 < velocity_scale <= 1.0:
        raise ValueError(f"velocity_scale must be in (0, 1], got {velocity_scale}")

    if verbose:
        import sys

        logger.remove()
        logger.add(sys.stderr, level="DEBUG")

    logger.warning(
        "Be ready to press e-stop! This script moves the arms, torso and head "
        "and does NOT check for self-collisions. Ensure the robot has clear space."
    )
    if input("Continue? [y/N]: ").strip().lower() != "y":
        return

    with Robot() as bot:
        if unfold:
            unfold_robot(bot, velocity_scale=velocity_scale)
        interactive_loop(bot, settle=settle, degrees=degrees)


if __name__ == "__main__":
    tyro.cli(main)

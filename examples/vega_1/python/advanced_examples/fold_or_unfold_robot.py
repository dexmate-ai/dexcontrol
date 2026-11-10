# Copyright (C) 2026 Dexmate Inc.
# SPDX-License-Identifier: Apache-2.0

"""Fold or unfold the robot in ordered, verified stages.

Default: close available hands, move both arms to folded, then torso to folded and head to tucked. With --unfold: move torso to crouch45_high, then resolve/move head home, then resolve/move arms to L_shape. Skip absent components; verify arms are in position mode (re-initializing them if not) before commanding them; check motion success and measured positions within 0.1 rad before advancing. No collision checking is provided.

Wait 1 s after closing each available hand before the arm stage.

Uses a Robot control connection: compatible arm modes may be enabled during startup, and shutdown requests a stop. Real hardware is selected unless --simulated is supplied."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Optional

import numpy as np
import tyro
from dexcontrol import Robot
from typing_extensions import Annotated


def ensure_position_mode(robot: Robot, arms: list[str]) -> None:
    """Re-initialize arms that are not in position mode before commanding them.

    Connecting while the E-stop is engaged skips arm initialization ("E-stop
    prevents writes"), and nothing re-runs it once the E-stop is released. An
    arm left in another mode accepts a motion target and stays put, which
    looks like a lost command. Checking the modes here makes that visible and
    repairs it.
    """
    wrong = {}
    for name in arms:
        modes = robot.joints(name).get_modes()
        if any(mode != "position" for mode in modes):
            wrong[name] = modes
    if not wrong:
        return
    for name, modes in wrong.items():
        print(f"  {name} is not in position mode ({', '.join(modes)}); initializing arms")
    report = robot.initialize_arms()
    if report.get("skipped"):
        raise SystemExit(f"Stage stopped: arm initialization skipped: {report['skipped']}")
    for name in wrong:
        modes = robot.joints(name).get_modes()
        if any(mode != "position" for mode in modes):
            raise SystemExit(f"Stage stopped: {name} still not in position mode ({', '.join(modes)})")


def stage(
    robot: Robot, poses: dict[str, str], timeout: float, tolerance: float,
    number: int = 1, total: int = 1,
) -> None:
    """Report each motion and measured arrival independently before advancing."""
    if not math.isfinite(timeout) or timeout <= 0:
        raise ValueError("timeout must be finite and positive")
    if not math.isfinite(tolerance) or tolerance < 0:
        raise ValueError("tolerance must be finite and non-negative")
    # Native group members retain target insertion order.
    targets = {
        name: robot.joints(name).resolve_pose(poses[name])
        for name in sorted(poses) if robot.has_component(name)
    }
    if not targets:
        print(f"Stage {number}/{total}: skipped (components unavailable)")
        return
    described = ", ".join(f"{name} -> {poses[name]}" for name in targets)
    print(f"\nStage {number}/{total}: {described} (wait limit: {timeout:g} s)", flush=True)
    ensure_position_mode(robot, [name for name in targets if name.endswith("_arm")])
    try:
        group = robot.move_to_joint_pos(targets, wait=False)
    except Exception as error:
        raise SystemExit(f"Stage stopped: could not start the motions.\n  {error}") from None
    wait_error = None
    aggregate = None
    try:
        aggregate = group.wait(timeout=timeout)
    except Exception as error:
        wait_error = str(error)
    complete = aggregate == "finished" and wait_error is None
    handles = group.handles
    if len(handles) != len(targets):
        raise SystemExit("Stage stopped: motion member count does not match the submitted targets.")
    for (name, target), motion in zip(targets.items(), handles):
        print(f"  {name} -> {poses[name]}")
        try:
            status = motion.status
            print(f"    Motion: {status.state.value} (id: {motion.motion_id})")
            if status.message:
                print(f"    Reason: {status.message}")
            if status.error_code:
                print(f"    Error code: {status.error_code}")
            complete = complete and status.succeeded
        except Exception as error:
            complete = False
            print(f"    Motion: unavailable ({error})")
        try:
            joint = robot.joints(name)
            measured = np.asarray(joint.get_joint_pos(), dtype=float)
            target = np.asarray(target, dtype=float)
            if measured.shape != target.shape or not measured.size or not np.all(np.isfinite(measured)) or not np.all(np.isfinite(target)):
                raise ValueError("invalid joint feedback shape or values")
            errors = np.abs(measured - target)
            worst = int(np.argmax(errors))
            reached = errors[worst] <= tolerance
            complete = complete and bool(reached)
            print(f"    Target: {'reached' if reached else 'NOT reached'} "
                  f"(max error: {errors[worst]:.4f} rad; tolerance: {tolerance:g} rad)")
            if not reached:
                print(f"    Joint: {joint.joint_names[worst]}; measured: {measured[worst]:.4f} rad; "
                      f"target: {target[worst]:.4f} rad")
        except Exception as error:
            complete = False
            print(f"    Target: unverified ({error})")
    if wait_error:
        print(f"  Wait error: {wait_error}")
    if not complete:
        raise SystemExit("Stage stopped: motion success and target arrival were not both verified. "
                         "No later stage will run.")
    print("  Stage complete.")


@dataclass
class Args:
    """Command-line options."""

    unfold: bool = False
    timeout: float = 10.0
    tolerance: Annotated[
        float, tyro.conf.arg(help="per-joint arrival tolerance (rad)")
    ] = 0.1
    simulated: Annotated[bool, tyro.conf.arg(help="Use the in-process simulation")] = (
        False
    )
    profile: Annotated[Optional[str], tyro.conf.arg(help="Built-in robot profile")] = (
        None
    )
    config: Annotated[
        Optional[str], tyro.conf.arg(help="Custom robot configuration")
    ] = None


def main() -> None:
    args = tyro.cli(Args, description=__doc__, config=(tyro.conf.FlagCreatePairsOff,))

    if not math.isfinite(args.timeout) or args.timeout <= 0:
        raise SystemExit("timeout must be finite and positive")
    if not math.isfinite(args.tolerance) or args.tolerance < 0:
        raise SystemExit("tolerance must be finite and non-negative")
    print("Unfold robot" if args.unfold else "Fold robot")
    with Robot(
        simulation=args.simulated, profile=args.profile, config_file=args.config
    ) as robot:
        if args.unfold:
            # Resolve head/arm references after the torso reaches its new posture.
            stages = [
                {"torso": "crouch45_high"},
                {"head": "home"},
                {"left_arm": "L_shape", "right_arm": "L_shape"},
            ]
        else:
            for side in ("left", "right"):
                if robot.have_hand(side):
                    robot.joints(f"{side}_hand").close_hand(wait_time=1.0)
            # Arms come in first; the torso and head then tuck over them.
            stages = [
                {"left_arm": "folded", "right_arm": "folded"},
                {"torso": "folded", "head": "tucked"},
            ]

        for number, poses in enumerate(stages, 1):
            stage(robot, poses, args.timeout, args.tolerance, number, len(stages))
        print("\nUnfold complete." if args.unfold else "\nFold complete.")


if __name__ == "__main__":
    main()

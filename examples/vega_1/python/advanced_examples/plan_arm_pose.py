# Copyright (C) 2026 Dexmate Inc.
# SPDX-License-Identifier: Apache-2.0

"""Plan and execute a both-arm named-pose move with dexmotion.

Connect to the real robot and build a geometry-aware OMPL plan to L_shape with torso-pitch compensation, sampled at 250 Hz. If initially self-colliding, propose a separate escape interpolation and require confirmation. Require confirmation before executing the final path; the dexmotion viewer is on by default (--no-visualize turns it off). Collision checking covers the configured model, not a guarantee about the physical workspace. No simulation flag."""

from dataclasses import dataclass

from typing_extensions import Annotated

import numpy as np
import tyro
from dexcontrol import Robot, ask
from dexmotion.motion_manager import MotionManager
from dexmotion.tasks.move_out_of_self_collision_task import MoveOutOfSelfCollisionTask
from dexmotion.tasks.move_to_configuration_task import MoveToConfigurationTask
from dexmotion.utils import robot_utils
from loguru import logger

ARMS = ("left_arm", "right_arm")


def joint_positions(robot: Robot, components) -> dict[str, float]:
    """Flat `{joint_name: position}` across `components`.

    `get_joint_pos_dict` nests by component when given a list, and the planner
    wants one flat namespace, so the per-component dicts are merged here.
    """
    flat: dict[str, float] = {}
    for name in components:
        flat.update(robot.get_joint_pos_dict(name))
    return flat


def planned_components(robot: Robot) -> list[str]:
    """Components whose current pose the planner needs to reason about."""
    return [name for name in (*ARMS, "head", "torso") if robot.has_component(name)]


def goal_configuration(robot: Robot, pose: str) -> dict[str, float]:
    """The named pose for both arms, as `{joint_name: position}`.

    Torso pitch is compensated the way every other example does it; on a model
    without a torso the native helper substitutes pi/2.
    """
    goal: dict[str, float] = {}
    for name in ARMS:
        arm = robot.joints(name)
        target = arm.resolve_pose(pose)
        goal.update(dict(zip(arm.joint_names, np.asarray(target, dtype=float))))
    return goal


def as_tracks(robot: Robot, joint_names: list[str], qs: np.ndarray) -> dict:
    """Split a planner trajectory into one waypoint matrix per component.

    The planner uses its own joint ordering, so each component's columns are
    selected by name rather than assumed to be contiguous.
    """
    qs = np.asarray(qs, dtype=float)
    tracks = {}
    for name in ARMS:
        columns = [joint_names.index(joint) for joint in robot.joints(name).joint_names]
        tracks[name] = qs[:, columns]
    return tracks


def confirm(prompt: str) -> bool:
    return ask(f"{prompt} [Enter to run, anything else to abort]") == ""


def escape_self_collision(
    robot: Robot,
    manager: MotionManager,
    start: dict,
    control_hz: float,
    seconds: float,
) -> None:
    """Walk the arms out of a self-collision, if they are in one."""
    task = MoveOutOfSelfCollisionTask(
        initial_joint_configuration=start, motion_manager=manager, visualize=False
    )
    in_collision, resolved, free_configuration = task.run()
    if not in_collision:
        return
    if not resolved:
        raise SystemExit("the arms are self-colliding and no escape was found")

    logger.warning("arms are self-colliding; escaping first")
    joints = sorted(set(start) & set(free_configuration))
    steps = max(2, int(seconds * control_hz))
    blend = np.linspace(0.0, 1.0, steps)[:, None]
    begin = np.array([[start[joint] for joint in joints]])
    end = np.array([[free_configuration[joint] for joint in joints]])
    if not confirm("Move to the collision-free configuration?"):
        raise SystemExit("aborted")
    robot.execute_trajectory(
        as_tracks(robot, joints, begin + (end - begin) * blend), control_hz
    )


@dataclass
class Args:
    """Command-line options."""

    pose: str = "L_shape"
    control_hz: float = 250.0
    visualize: Annotated[
        bool,
        tyro.conf.arg(help="Show the robot and the planned path in the dexmotion viewer"),
    ] = True
    collision_escape_time: float = 2.0


def main() -> None:
    args = tyro.cli(Args, description=__doc__, config=(tyro.conf.FlagCreatePairsOff,))

    with Robot() as robot:
        start = joint_positions(robot, planned_components(robot))
        manager = MotionManager(
            initial_joint_configuration_dict=start,
            init_visualizer=args.visualize,
            init_local_ik=False,
        )
        escape_self_collision(
            robot, manager, start, args.control_hz, args.collision_escape_time
        )

        # Re-read: the escape above may have moved the arms.
        current = joint_positions(robot, ARMS)
        manager.set_joint_pos(current)
        planner = MoveToConfigurationTask(
            motion_manager=manager, planner_type="ompl", visualize=args.visualize
        )
        _, qs, _, _, duration = planner.run(
            goal_configuration_dict=goal_configuration(robot, args.pose),
            start_configuration_dict=current,
            control_frequency=args.control_hz,
            generate_trajectory=True,
            visualize_result=args.visualize,
        )
        if qs is None or len(qs) == 0:
            raise SystemExit(f"no collision-free path to {args.pose} was found")

        joint_names = robot_utils.get_joint_names(manager.pin_robot)
        logger.info(f"planned {len(qs)} waypoints over {duration:.1f}s")
        if not confirm(f"Run the planned path to {args.pose} on the real robot?"):
            raise SystemExit("aborted")
        robot.execute_trajectory(as_tracks(robot, joint_names, qs), args.control_hz)
        logger.success(f"reached {args.pose}")


if __name__ == "__main__":
    main()

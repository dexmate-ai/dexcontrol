"""Vega robot wrapper compatible with RobotEnv service."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Optional

import numpy as np
from scipy.spatial.transform import Rotation as R

# Ensure local sibling repositories are importable in workspace deployments.
# robot.py lives at: <repo>/src/dexcontrol/core/vega/robot.py
#   parents: [0]=vega, [1]=core, [2]=dexcontrol, [3]=src, [4]=<repo>
_REPO_ROOT = Path(__file__).resolve().parents[4]
_DEXCONTROL_SRC = _REPO_ROOT / "src"
_DEXCONTROL_TELEOP = _REPO_ROOT / "examples" / "teleop"
for _path in (_DEXCONTROL_SRC, _DEXCONTROL_TELEOP):
    if _path.exists() and str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from dexcontrol import Robot
from dexbot_utils.configs import get_robot_config

_base_arm_teleop_error = None
try:
    from base_arm_teleop import BaseIKController
except ImportError as e:
    BaseIKController = None
    _base_arm_teleop_error = e

try:
    from dexmotion.utils import robot_utils
except ImportError:
    robot_utils = None


SUPPORTED_ACTION_SPACES = (
    "joint_position",
    "joint_velocity",
    "joint_delta",
    "cartesian_velocity",
    "cartesian_delta",
)


class JointLimitExceededError(RuntimeError):
    """Raised when commanded joints exceed robot joint limits."""


class IKFailedError(RuntimeError):
    """Raised when IK cannot solve a requested Cartesian command."""


class CommunicationFailedError(RuntimeError):
    """Raised when arm/hand command transmission fails."""


class VegaRobot:
    """Thin wrapper around dexcontrol Robot for single-arm control."""

    def __init__(
        self,
        robot_model: str = "vega_1",
        arm_side: str = "left",
        control_hz: int = 20,
        gripper_type: str = "default",
        **kwargs,
    ):
        hand_type = kwargs.pop("hand_type", None)
        if kwargs:
            unexpected = ", ".join(sorted(kwargs.keys()))
            raise TypeError(f"Unexpected keyword arguments: {unexpected}")
        if hand_type is not None and gripper_type == "default":
            gripper_type = hand_type

        if arm_side not in ("left", "right"):
            raise ValueError(f"arm_side must be 'left' or 'right', got: {arm_side}")
        if BaseIKController is None:
            msg = (
                "BaseIKController not found. Ensure custom_dexcontrol/examples/teleop is available "
                "and dependencies are installed (e.g. pip install pytransform3d dexmotion dualsense_controller)."
            )
            if _base_arm_teleop_error is not None:
                msg += f" Original error: {_base_arm_teleop_error}"
            raise ImportError(msg) from _base_arm_teleop_error

        self.robot_model = robot_model
        self.arm_side = arm_side
        self.control_hz = int(control_hz)
        self.gripper_type = gripper_type

        configs = get_robot_config(robot_model)
        self.robot = Robot(configs=configs)
        self.arm = getattr(self.robot, f"{arm_side}_arm")
        hand_component = f"{arm_side}_hand"
        self.hand = getattr(self.robot, hand_component) if self.robot.has_component(hand_component) else None

        self.ik_controller = BaseIKController(bot=self.robot, visualize=False)
        self._arm_joint_names = [f"{arm_side[0].upper()}_arm_j{i + 1}" for i in range(7)]

        # Use Vega unfold pose (L_shape) as default reset, same as fold_robot.py unfold.
        self.reset_joints = np.asarray(
            self.arm.get_predefined_pose("L_shape"), dtype=np.float64
        )
        self.safe_transit_pose = self.reset_joints.copy()

        self._gripper_open_pos, self._gripper_close_pos = self._init_gripper_reference()
        self._prev_command_successful = True
        self._prev_gripper_command_successful = True
        self._prev_controller_latency_ms = 0.0

    def launch_robot(self) -> None:
        """Validate robot readiness and default control mode."""
        self.arm.set_modes(["position"] * 7)
        if self.arm.joint_pos_limit is None:
            raise RuntimeError("Arm joint position limits are unavailable")
        self.sync_motion_manager_with_arm(self.reset_joints)

    def update_command(
        self,
        command: np.ndarray,
        action_space: str,
        gripper_action_space: str = "position",
        blocking: bool = False,
    ) -> None:
        if action_space not in SUPPORTED_ACTION_SPACES:
            raise ValueError(f"Unsupported action_space '{action_space}'")
        if gripper_action_space not in ("position", "velocity"):
            raise ValueError(f"Unsupported gripper_action_space '{gripper_action_space}'")

        action = np.asarray(command, dtype=np.float64).reshape(-1)
        dt = 1.0 / max(1, self.control_hz)
        state_dict, _ = self.get_robot_state()
        current_joint_pos = np.asarray(state_dict["joint_positions"], dtype=np.float64)

        if action_space.startswith("joint"):
            if action.shape[0] != 8:
                raise ValueError(f"{action_space} expects 8 values, got {action.shape[0]}")
            arm_action = action[:7]
            gripper_action = float(action[7])
        else:
            if action.shape[0] != 7:
                raise ValueError(f"{action_space} expects 7 values, got {action.shape[0]}")
            arm_action = action[:6]
            gripper_action = float(action[6])

        if action_space == "joint_position":
            target_joint_pos = arm_action
        elif action_space == "joint_velocity":
            target_joint_pos = current_joint_pos + arm_action * dt
        elif action_space == "joint_delta":
            target_joint_pos = current_joint_pos + arm_action
        elif action_space == "cartesian_velocity":
            target_joint_pos = self._solve_cartesian_delta(arm_action[:3] * dt, arm_action[3:6] * dt)
        else:  # cartesian_delta
            target_joint_pos = self._solve_cartesian_delta(arm_action[:3], arm_action[3:6])

        try:
            self.update_joints(target_joint_pos, velocity=False, blocking=blocking)
            self.update_gripper(
                gripper_action,
                velocity=(gripper_action_space == "velocity"),
                blocking=blocking,
            )
            self._prev_command_successful = True
        except (JointLimitExceededError, IKFailedError):
            self._prev_command_successful = False
            raise
        except Exception as exc:
            self._prev_command_successful = False
            raise CommunicationFailedError(str(exc)) from exc

    def update_joints(
        self,
        joint_pos_command: np.ndarray,
        velocity: bool = False,
        blocking: bool = False,
    ) -> None:
        target_joint_pos = np.asarray(joint_pos_command, dtype=np.float64)
        if velocity:
            dt = 1.0 / max(1, self.control_hz)
            current_joint_pos = np.asarray(self.arm.get_joint_pos(), dtype=np.float64)
            target_joint_pos = current_joint_pos + target_joint_pos * dt

        self.validate_joint_limits(target_joint_pos)
        wait_time = (1.0 / max(1, self.control_hz)) if blocking else 0.0
        self.arm.set_joint_pos(target_joint_pos, wait_time=wait_time)
        self.sync_motion_manager_with_arm(target_joint_pos)

    def update_gripper(self, command: float, velocity: bool = True, blocking: bool = False) -> None:
        if self.hand is None:
            self._prev_gripper_command_successful = True
            return
        current = self._normalize_gripper_position(np.asarray(self.hand.get_joint_pos(), dtype=np.float64))
        if velocity:
            target = current + float(command) * (1.0 / max(1, self.control_hz))
        else:
            target = float(command)

        target = float(np.clip(target, 0.0, 1.0))
        wait_time = (1.0 / max(1, self.control_hz)) if blocking else 0.0

        try:
            if target <= 1e-3:
                self.hand.open_hand(wait_time=wait_time)
            elif target >= 1.0 - 1e-3:
                self.hand.close_hand(wait_time=wait_time)
            else:
                target_joint_pos = self._gripper_open_pos + target * (self._gripper_close_pos - self._gripper_open_pos)
                self.hand.set_joint_pos(target_joint_pos, wait_time=wait_time)
            self._prev_gripper_command_successful = True
        except Exception:
            self._prev_gripper_command_successful = False
            raise

    def get_robot_state(self) -> tuple[dict[str, Any], dict[str, int]]:
        joint_positions = np.asarray(self.arm.get_joint_pos(), dtype=np.float64)
        joint_velocities = np.asarray(self.arm.get_joint_vel(), dtype=np.float64)
        try:
            joint_torques = np.asarray(self.arm.get_joint_torque(), dtype=np.float64)
        except ValueError:
            joint_torques = np.zeros(7, dtype=np.float64)
        if self.hand is not None:
            hand_joint_pos = np.asarray(self.hand.get_joint_pos(), dtype=np.float64)
            gripper_position = float(self._normalize_gripper_position(hand_joint_pos))
        else:
            gripper_position = 0.0

        wrench_state = np.zeros(6, dtype=np.float64)
        if getattr(self.arm, "wrench_sensor", None) is not None:
            wrench_state = np.asarray(self.arm.wrench_sensor.get_wrench_state(), dtype=np.float64)

        cartesian_position = self._get_cartesian_pose(joint_positions=joint_positions)

        timestamp_ns = int(self.arm.get_timestamp_ns())
        timestamp_dict = {
            "robot_timestamp_seconds": int(timestamp_ns // 1_000_000_000),
            "robot_timestamp_nanos": int(timestamp_ns % 1_000_000_000),
            "robot_timestamp_us": int(timestamp_ns // 1_000),
        }

        state_dict = {
            "joint_positions": joint_positions,
            "joint_velocities": joint_velocities,
            "joint_torques_computed": joint_torques,
            "prev_joint_torques_computed": joint_torques.copy(),
            "prev_joint_torques_computed_safened": joint_torques.copy(),
            "motor_torques_measured": joint_torques.copy(),
            "gripper_position": gripper_position,
            "cartesian_position": cartesian_position,
            "wrench_state": wrench_state,
            "prev_controller_latency_ms": float(self._prev_controller_latency_ms),
            "prev_command_successful": bool(self._prev_command_successful),
            "prev_gripper_command_successful": bool(self._prev_gripper_command_successful),
        }
        return state_dict, timestamp_dict

    def validate_joint_limits(self, target_joint_pos: np.ndarray) -> None:
        limits = self.arm.joint_pos_limit
        if limits is None:
            return
        target_joint_pos = np.asarray(target_joint_pos, dtype=np.float64)
        low = limits[:, 0].astype(np.float64)
        high = limits[:, 1].astype(np.float64)
        violates = np.logical_or(target_joint_pos < low, target_joint_pos > high)
        if np.any(violates):
            raise JointLimitExceededError(
                f"Joint limit exceeded for indices {np.where(violates)[0].tolist()}"
            )

    def sync_motion_manager_with_arm(self, arm_joint_pos: np.ndarray) -> None:
        try:
            motion_manager = self.ik_controller.motion_manager
            qpos_dict = motion_manager.get_joint_pos_dict()
            for i, joint_name in enumerate(self._arm_joint_names):
                if joint_name in qpos_dict:
                    qpos_dict[joint_name] = float(arm_joint_pos[i])
            motion_manager.set_joint_pos(qpos_dict)
        except Exception:
            pass

    def _solve_cartesian_delta(self, delta_xyz: np.ndarray, delta_rpy: np.ndarray) -> np.ndarray:
        try:
            target_joint_pos = self.ik_controller.move_delta_cartesian(
                delta_xyz=np.asarray(delta_xyz, dtype=np.float64),
                delta_rpy=np.asarray(delta_rpy, dtype=np.float64),
                arm_side=self.arm_side,
            )
        except Exception as exc:
            raise IKFailedError(f"IK solver failed: {exc}") from exc

        target_joint_pos = np.asarray(target_joint_pos, dtype=np.float64)
        if target_joint_pos.shape[0] != 7:
            raise IKFailedError(f"IK returned invalid joint vector shape: {target_joint_pos.shape}")
        return target_joint_pos

    def _init_gripper_reference(self) -> tuple[np.ndarray, np.ndarray]:
        if self.hand is None:
            return np.array([0.0], dtype=np.float64), np.array([1.0], dtype=np.float64)
        try:
            open_pos = np.asarray(self.hand.get_predefined_pose("open"), dtype=np.float64)
        except Exception:
            open_pos = np.asarray(self.hand.get_joint_pos(), dtype=np.float64)

        try:
            close_pos = np.asarray(self.hand.get_predefined_pose("close"), dtype=np.float64)
        except Exception:
            close_pos = open_pos.copy()

        if open_pos.shape != close_pos.shape:
            close_pos = np.resize(close_pos, open_pos.shape)
        return open_pos, close_pos

    def _normalize_gripper_position(self, hand_joint_pos: np.ndarray) -> float:
        open_pos = np.asarray(self._gripper_open_pos, dtype=np.float64)
        close_pos = np.asarray(self._gripper_close_pos, dtype=np.float64)
        hand_joint_pos = np.asarray(hand_joint_pos, dtype=np.float64)
        if hand_joint_pos.shape != open_pos.shape:
            hand_joint_pos = np.resize(hand_joint_pos, open_pos.shape)

        denom = close_pos - open_pos
        valid = np.abs(denom) > 1e-6
        if not np.any(valid):
            return float(np.clip(np.mean(hand_joint_pos), 0.0, 1.0))

        ratio = np.zeros_like(hand_joint_pos, dtype=np.float64)
        ratio[valid] = (hand_joint_pos[valid] - open_pos[valid]) / denom[valid]
        return float(np.clip(np.mean(ratio[valid]), 0.0, 1.0))

    def _forward_kinematics(self, joint_positions: np.ndarray) -> np.ndarray:
        motion_manager = self.ik_controller.motion_manager
        qpos_dict = motion_manager.get_joint_pos_dict()
        for i, joint_name in enumerate(self._arm_joint_names):
            if joint_name in qpos_dict:
                qpos_dict[joint_name] = float(joint_positions[i])

        if robot_utils is not None:
            qpos = robot_utils.get_qpos_from_joint_dict(motion_manager.pin_robot, qpos_dict)
        else:
            self.sync_motion_manager_with_arm(np.asarray(joint_positions, dtype=np.float64))
            qpos = motion_manager.get_joint_pos()

        ee_pose_dict = motion_manager.fk(
            frame_names=motion_manager.target_frames,
            qpos=qpos,
            update_robot_state=False,
        )
        key = "L_ee" if self.arm_side == "left" else "R_ee"
        if key not in ee_pose_dict:
            key = sorted(ee_pose_dict.keys())[0]

        ee_pose = ee_pose_dict[key]
        pose_mat = ee_pose.np if hasattr(ee_pose, "np") else np.asarray(ee_pose)
        xyz = pose_mat[:3, 3]
        rpy = R.from_matrix(pose_mat[:3, :3]).as_euler("xyz", degrees=False)
        return np.concatenate([xyz, rpy]).astype(np.float64)

    def _get_cartesian_pose(self, joint_positions: Optional[np.ndarray] = None) -> np.ndarray:
        if joint_positions is None:
            joint_positions = np.asarray(self.arm.get_joint_pos(), dtype=np.float64)
        return self._forward_kinematics(joint_positions)

    def create_action_dict(
        self,
        action: np.ndarray,
        action_space: str,
        gripper_action_space: Optional[str] = None,
        robot_state: Optional[dict] = None,
    ) -> dict:
        """Convert action to comprehensive action dict with all representations.

        Used by policy_runner for action logging and data collection.
        """
        if action_space not in SUPPORTED_ACTION_SPACES:
            raise ValueError(f"Unsupported action_space: {action_space}")

        if robot_state is None:
            robot_state, _ = self.get_robot_state()

        action = np.asarray(action, dtype=np.float64)
        dt = 1.0 / max(1, self.control_hz)

        action_dict = {"robot_state": robot_state}

        # Determine gripper action space
        velocity = "velocity" in action_space
        if gripper_action_space is None:
            gripper_action_space = "velocity" if velocity else "position"

        # Extract gripper action
        if action_space.startswith("joint"):
            gripper_action = float(action[7])
        else:
            gripper_action = float(action[6])

        # Gripper conversions
        current_gripper = float(robot_state["gripper_position"])
        if gripper_action_space == "velocity":
            gripper_position = np.clip(current_gripper + gripper_action * dt, 0.0, 1.0)
        else:
            gripper_position = np.clip(gripper_action, 0.0, 1.0)

        action_dict["gripper_position"] = float(gripper_position)
        action_dict["gripper_delta"] = float(gripper_position - current_gripper)

        # Process action based on space
        current_joint_pos = np.asarray(robot_state["joint_positions"], dtype=np.float64)
        current_cart_pos = np.asarray(robot_state["cartesian_position"], dtype=np.float64)

        if action_space.startswith("cartesian"):
            cart_action = action[:6]

            if velocity:
                # cartesian_velocity
                action_dict["cartesian_velocity"] = cart_action.tolist()
                cartesian_delta = cart_action * dt
            else:
                # cartesian_delta
                cartesian_delta = cart_action
                cartesian_velocity = cart_action / dt
                action_dict["cartesian_velocity"] = cartesian_velocity.tolist()

            action_dict["delta_action"] = np.concatenate([cartesian_delta, [gripper_action]]).tolist()

            # Compute target cartesian position (using simple pose addition)
            target_cart = current_cart_pos.copy()
            target_cart[:3] += cartesian_delta[:3]  # xyz
            target_cart[3:6] += cartesian_delta[3:6]  # rpy
            action_dict["cartesian_position"] = target_cart.tolist()

            # Use IK to get joint positions
            try:
                target_joints = self._solve_cartesian_delta(cartesian_delta[:3], cartesian_delta[3:6])
                action_dict["joint_position"] = target_joints.tolist()
                joint_delta = target_joints - current_joint_pos
                action_dict["joint_velocity"] = (joint_delta / dt).tolist()
            except Exception:
                # IK failed, use current positions
                action_dict["joint_position"] = current_joint_pos.tolist()
                action_dict["joint_velocity"] = [0.0] * 7

        elif action_space.startswith("joint"):
            joint_action = action[:7]

            if action_space == "joint_position":
                target_joints = joint_action
                joint_delta = target_joints - current_joint_pos
                joint_velocity = joint_delta / dt
            elif action_space == "joint_velocity":
                joint_velocity = joint_action
                joint_delta = joint_velocity * dt
                target_joints = current_joint_pos + joint_delta
            else:  # joint_delta
                joint_delta = joint_action
                target_joints = current_joint_pos + joint_delta
                joint_velocity = joint_delta / dt

            action_dict["joint_position"] = target_joints.tolist()
            action_dict["joint_velocity"] = joint_velocity.tolist()
            action_dict["joint_delta"] = joint_delta.tolist()

            # Compute cartesian via FK
            try:
                target_cart = self._get_cartesian_pose(target_joints)
                action_dict["cartesian_position"] = target_cart.tolist()
            except Exception:
                action_dict["cartesian_position"] = current_cart_pos.tolist()

        return action_dict

    def stop(self) -> None:
        """Stop robot motion (emergency stop compatibility)."""
        try:
            # Stop arm motion by sending current position as target
            current_pos = np.asarray(self.arm.get_joint_pos(), dtype=np.float64)
            self.arm.set_joint_pos(current_pos, wait_time=0.0)
        except Exception:
            pass

    def close(self) -> None:
        try:
            self.robot.shutdown()
        except Exception:
            pass

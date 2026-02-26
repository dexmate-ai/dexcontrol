"""Vega robot wrapper compatible with RobotEnv service."""

from __future__ import annotations

import json as _json
import sys
import threading
import time as _time
from queue import Empty, Full, Queue
from pathlib import Path
from typing import Any, Optional

import numpy as np
from scipy.spatial.transform import Rotation as R

# #region agent log
_DBG_LOG = "/home/dexmate/.cursor/debug-2bec47.log"
def _dlog(loc, msg, data=None, hyp="", run=""):
    try:
        with open(_DBG_LOG, "a") as f:
            f.write(_json.dumps({"sessionId":"2bec47","location":loc,"message":msg,"data":{k:(v.tolist() if hasattr(v,'tolist') else v) for k,v in (data or {}).items()},"hypothesisId":hyp,"runId":run,"timestamp":int(_time.time()*1000)})+"\n")
    except Exception:
        pass
# #endregion
# #region agent log
_DBG_LOG2 = "/home/dexmate/.cursor/debug-ac3810.log"
_clip_step_counter = [0]
def _dlog2(loc, msg, data=None, hyp="", run=""):
    try:
        with open(_DBG_LOG2, "a") as f:
            f.write(_json.dumps({"sessionId":"ac3810","location":loc,"message":msg,"data":{k:(v.tolist() if hasattr(v,'tolist') else v) for k,v in (data or {}).items()},"hypothesisId":hyp,"runId":run,"timestamp":int(_time.time()*1000)})+"\n")
    except Exception:
        pass
# #endregion
# #region agent log
try:
    _dlog2("robot.py", "module_loaded", {"robot_module_path": str(Path(__file__).resolve())}, hyp="H_LOAD")
except Exception:
    pass
# #endregion

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
        ik_solver_type: str = "pink",
        use_velocity_feedforward: bool = False,
        robotiq_comport: str = "/dev/ttyUSB0",
        ema_alpha: float = 0.0,
        **kwargs,
    ):
        hand_type = kwargs.pop("hand_type", None)
        if kwargs:
            unexpected = ", ".join(sorted(kwargs.keys()))
            raise TypeError(f"Unexpected keyword arguments: {unexpected}")
        if hand_type is not None and gripper_type == "default":
            gripper_type = hand_type
        self._robotiq_comport = robotiq_comport
        self._ik_solver_type = ik_solver_type

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
        self.use_velocity_feedforward = bool(use_velocity_feedforward)

        configs = get_robot_config(robot_model)
        self.robot = Robot(configs=configs)
        self.arm = getattr(self.robot, f"{arm_side}_arm")
        hand_component = f"{arm_side}_hand"

        if gripper_type == "robotiq":
            from dexcontrol.core.robotiq_gripper import RobotiqGripper  # lazy import
            self.hand = RobotiqGripper(comport=self._robotiq_comport)
        else:
            self.hand = getattr(self.robot, hand_component) if self.robot.has_component(hand_component) else None

        custom_ik_cfg = self._build_ik_config()
        self.ik_controller = BaseIKController(
            bot=self.robot, visualize=False,
            ik_solver_type=self._ik_solver_type,
            custom_local_ik_config=custom_ik_cfg,
        )
        self._arm_joint_names = [f"{arm_side[0].upper()}_arm_j{i + 1}" for i in range(7)]

        # Use Vega unfold pose (L_shape) as default reset, same as fold_robot.py unfold.
        self.reset_joints = np.asarray(
            self.arm.get_predefined_pose("L_shape"), dtype=np.float64
        )
        self.safe_transit_pose = self.reset_joints.copy()

        self._gripper_open_pos, self._gripper_close_pos = self._init_gripper_reference()
        self._gripper_state_lock = threading.Lock()
        self._gripper_io_lock = threading.Lock()
        self._gripper_poll_interval_s = 1.0 / max(1, self.control_hz)
        self._gripper_position = 0.0
        self._gripper_joint_pos = np.asarray(self._gripper_open_pos, dtype=np.float64).copy()
        self._gripper_command_queue: Queue[float] | None = None
        self._gripper_stop_event = threading.Event()
        self._gripper_worker: threading.Thread | None = None
        self._prev_command_successful = True
        self._prev_gripper_command_successful = True
        self._prev_controller_latency_ms = 0.0
        self._prev_joint_vel: np.ndarray | None = None
        self._vel_smoothing_alpha = 0.3  # EMA factor: 0=fully smooth, 1=no smoothing

        # EMA filter for joint position commands (0.0 = disabled, 0.05~0.3 = typical range)
        self._ema_alpha = float(ema_alpha)
        self._ema_prev_qpos: np.ndarray | None = None

        if self.hand is not None:
            self._refresh_gripper_state()
            self._start_gripper_worker()

    def _build_ik_config(self):
        """Build an optimized IK config for real-time control."""
        if self._ik_solver_type != "pink":
            return None
        try:
            from dexmotion.configs.ik.ik_config import LocalPinkIKConfig, IKDampingWeightsConfig
            return LocalPinkIKConfig(
                solver_name="local_pink",
                target_frames=["L_ee", "R_ee"],
                dt=1.0 / max(1, self.control_hz),
                avoid_self_collisions=False,
                collision_pairs=0,
                collision_margin=0.02,
                qp_solver="proxqp",
                safety_buffer=0.2,
                damping_weights=IKDampingWeightsConfig(
                    default=1e-3,
                    override={
                        "torso_j1": 30000.0,
                        "torso_j2": 30000.0,
                        "torso_j3": 30000.0,
                        "R_arm_j2": 100,
                        "L_arm_j2": 100,
                        "R_arm_j3": 50.0,
                        "L_arm_j3": 50.0,
                    },
                ),
            )
        except Exception:
            return None

    def launch_robot(self) -> None:
        """Validate robot readiness and default control mode."""
        self.arm.set_modes(["position"] * 7)
        if self.arm.joint_pos_limit is None:
            raise RuntimeError("Arm joint position limits are unavailable")
        self._ema_prev_qpos = None  # Reset EMA state on launch
        self.sync_motion_manager_with_arm(self.reset_joints)

    def _start_gripper_worker(self) -> None:
        if self.hand is None:
            return
        if self._gripper_worker is not None and self._gripper_worker.is_alive():
            return
        self._gripper_command_queue = Queue(maxsize=1)
        self._gripper_stop_event.clear()
        self._gripper_worker = threading.Thread(
            target=self._gripper_worker_loop,
            name=f"vega-gripper-{self.arm_side}",
            daemon=True,
        )
        self._gripper_worker.start()

    def _stop_gripper_worker(self) -> None:
        self._gripper_stop_event.set()
        if self._gripper_worker is not None and self._gripper_worker.is_alive():
            self._gripper_worker.join(timeout=1.0)
        self._gripper_worker = None
        self._gripper_command_queue = None

    def _gripper_worker_loop(self) -> None:
        while not self._gripper_stop_event.is_set():
            latest_target: float | None = None
            if self._gripper_command_queue is not None:
                try:
                    latest_target = self._gripper_command_queue.get(
                        timeout=self._gripper_poll_interval_s
                    )
                    while True:
                        latest_target = self._gripper_command_queue.get_nowait()
                except Empty:
                    pass

            if latest_target is not None:
                self._execute_gripper_command(
                    latest_target, wait_time=0.0, raise_on_error=False
                )

            self._refresh_gripper_state()

    def _clear_gripper_command_queue(self) -> None:
        if self._gripper_command_queue is None:
            return
        while True:
            try:
                self._gripper_command_queue.get_nowait()
            except Empty:
                return

    def _gripper_target_to_joint_pos(self, target: float) -> np.ndarray:
        return self._gripper_open_pos + target * (self._gripper_close_pos - self._gripper_open_pos)

    def _refresh_gripper_state(self) -> bool:
        if self.hand is None:
            return False
        try:
            with self._gripper_io_lock:
                hand_joint_pos = np.asarray(self.hand.get_joint_pos(), dtype=np.float64)
            gripper_position = float(self._normalize_gripper_position(hand_joint_pos))
            with self._gripper_state_lock:
                self._gripper_joint_pos = hand_joint_pos.copy()
                self._gripper_position = gripper_position
            return True
        except Exception:
            self._prev_gripper_command_successful = False
            return False

    def _execute_gripper_command(
        self,
        target: float,
        wait_time: float,
        raise_on_error: bool,
    ) -> bool:
        if self.hand is None:
            self._prev_gripper_command_successful = True
            return True
        target = float(np.clip(target, 0.0, 1.0))
        target_joint_pos = self._gripper_target_to_joint_pos(target)
        try:
            with self._gripper_io_lock:
                if target <= 1e-3:
                    self.hand.open_hand(wait_time=wait_time)
                elif target >= 1.0 - 1e-3:
                    self.hand.close_hand(wait_time=wait_time)
                else:
                    self.hand.set_joint_pos(target_joint_pos, wait_time=wait_time)
            with self._gripper_state_lock:
                self._gripper_position = target
                self._gripper_joint_pos = np.asarray(target_joint_pos, dtype=np.float64).copy()
            self._prev_gripper_command_successful = True
            return True
        except Exception:
            self._prev_gripper_command_successful = False
            if raise_on_error:
                raise
            return False

    def _enqueue_gripper_target(self, target: float) -> None:
        target = float(np.clip(target, 0.0, 1.0))
        with self._gripper_state_lock:
            self._gripper_position = target
            self._gripper_joint_pos = self._gripper_target_to_joint_pos(target)
        if self._gripper_command_queue is None:
            self._execute_gripper_command(target, wait_time=0.0, raise_on_error=False)
            return
        try:
            self._gripper_command_queue.put_nowait(target)
        except Full:
            try:
                self._gripper_command_queue.get_nowait()
            except Empty:
                pass
            try:
                self._gripper_command_queue.put_nowait(target)
            except Full:
                pass

    def get_cached_gripper_position(self) -> float:
        if self.hand is None:
            return 0.0
        with self._gripper_state_lock:
            return float(self._gripper_position)

    def get_cached_gripper_joint_pos(self) -> np.ndarray:
        if self.hand is None:
            return np.zeros(1, dtype=np.float64)
        with self._gripper_state_lock:
            return np.asarray(self._gripper_joint_pos, dtype=np.float64).copy()

    def update_command(
        self,
        command: np.ndarray,
        action_space: str,
        gripper_action_space: str = "position",
        blocking: bool = False,
    ) -> None:
        # #region agent log
        _uc_t0 = _time.time()
        # #endregion
        if action_space not in SUPPORTED_ACTION_SPACES:
            raise ValueError(f"Unsupported action_space '{action_space}'")
        if gripper_action_space not in ("position", "velocity"):
            raise ValueError(f"Unsupported gripper_action_space '{gripper_action_space}'")

        action = np.asarray(command, dtype=np.float64).reshape(-1)
        dt = 1.0 / max(1, self.control_hz)
        state_dict, _ = self.get_robot_state()
        current_joint_pos = np.asarray(state_dict["joint_positions"], dtype=np.float64)
        # #region agent log
        _uc_t1 = _time.time()
        # #endregion

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

        # #region agent log
        _clip_step_counter[0] += 1
        _dlog2("robot.py:update_cmd", "action_input", {"step": _clip_step_counter[0], "action_space": action_space, "arm_action": np.asarray(arm_action), "current_joint_pos": current_joint_pos}, hyp="H1_CLIP_FREQ")
        # #endregion
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

        # Apply EMA smoothing to joint position command
        if self._ema_alpha > 0.0:
            if self._ema_prev_qpos is None:
                self._ema_prev_qpos = target_joint_pos.copy()
            else:
                target_joint_pos = (
                    self._ema_alpha * target_joint_pos
                    + (1.0 - self._ema_alpha) * self._ema_prev_qpos
                )
                self._ema_prev_qpos = target_joint_pos.copy()

        # #region agent log
        _uc_t2 = _time.time()
        # #endregion
        try:
            self.update_joints(target_joint_pos, velocity=False, blocking=blocking)
            # #region agent log
            _uc_t3 = _time.time()
            # #endregion
            self.update_gripper(
                gripper_action,
                velocity=(gripper_action_space == "velocity"),
                blocking=blocking,
            )
            # #region agent log
            _uc_t4 = _time.time()
            _dlog("robot.py:update_cmd","uc_timing",{"get_state_ms":(_uc_t1-_uc_t0)*1000,"ik_solve_ms":(_uc_t2-_uc_t1)*1000,"update_joints_ms":(_uc_t3-_uc_t2)*1000,"update_gripper_ms":(_uc_t4-_uc_t3)*1000,"total_ms":(_uc_t4-_uc_t0)*1000},hyp="H_UC_TIMING")
            # #endregion
            self._prev_command_successful = True
        except (JointLimitExceededError, IKFailedError):
            self._prev_command_successful = False
            raise
        except Exception as exc:
            self._prev_command_successful = False
            raise CommunicationFailedError(str(exc)) from exc

    _MOTOR_MAX_DELTA_RAD = 0.25

    _JOINT_LIMIT_TOLERANCE_RAD = 0.01  # ~0.57 deg tolerance for IK numerical precision

    def update_joints(
        self,
        joint_pos_command: np.ndarray,
        velocity: bool = False,
        blocking: bool = False,
    ) -> None:
        # #region agent log
        _uj_t0 = _time.time()
        # #endregion
        target_joint_pos = np.asarray(joint_pos_command, dtype=np.float64)
        if velocity:
            dt = 1.0 / max(1, self.control_hz)
            current_joint_pos = np.asarray(self.arm.get_joint_pos(), dtype=np.float64)
            target_joint_pos = current_joint_pos + target_joint_pos * dt

        # Clip small IK numerical errors within tolerance before hard validation
        limits = self.arm.joint_pos_limit
        if limits is not None:
            low = limits[:, 0].astype(np.float64)
            high = limits[:, 1].astype(np.float64)
            tol = self._JOINT_LIMIT_TOLERANCE_RAD
            target_joint_pos = np.where(
                (target_joint_pos < low) & (target_joint_pos >= low - tol),
                low, target_joint_pos,
            )
            target_joint_pos = np.where(
                (target_joint_pos > high) & (target_joint_pos <= high + tol),
                high, target_joint_pos,
            )

        self.validate_joint_limits(target_joint_pos)

        current = np.asarray(self.arm.get_joint_pos(), dtype=np.float64)
        diff = target_joint_pos - current
        max_diff = float(np.max(np.abs(diff)))
        was_clipped = max_diff > self._MOTOR_MAX_DELTA_RAD
        if was_clipped:
            clipped = np.clip(diff, -self._MOTOR_MAX_DELTA_RAD, self._MOTOR_MAX_DELTA_RAD)
            # #region agent log
            _truncated = diff - clipped
            _per_joint_clipped = np.abs(diff) > self._MOTOR_MAX_DELTA_RAD
            _dlog2("robot.py:update_joints", "clip_detail", {"step": _clip_step_counter[0], "max_diff_rad": max_diff, "max_diff_deg": float(np.rad2deg(max_diff)), "limit_rad": self._MOTOR_MAX_DELTA_RAD, "pct_of_motor_clip": round(100.0 * max_diff / self._MOTOR_MAX_DELTA_RAD, 2), "diff_before_rad": diff, "diff_after_rad": clipped, "truncated_rad": _truncated, "per_joint_clipped": _per_joint_clipped, "clipped_joint_indices": np.where(_per_joint_clipped)[0].tolist()}, hyp="H1_CLIP_FREQ,H2_JOINT_DIST,H3_DRIFT")
            # #endregion
            target_joint_pos = current + clipped
        else:
            # #region agent log
            _dlog2("robot.py:update_joints", "no_clip", {"step": _clip_step_counter[0], "max_diff_rad": max_diff, "max_diff_deg": float(np.rad2deg(max_diff)), "limit_rad": self._MOTOR_MAX_DELTA_RAD, "pct_of_motor_clip": round(100.0 * max_diff / self._MOTOR_MAX_DELTA_RAD, 2)}, hyp="H1_CLIP_FREQ")
            # #endregion
        # #region agent log
        _uj_t1 = _time.time()
        # #endregion

        if self.use_velocity_feedforward and not blocking:
            dt = 1.0 / max(1, self.control_hz)
            raw_vel = (target_joint_pos - current) / dt
            if self._prev_joint_vel is not None:
                a = self._vel_smoothing_alpha
                target_joint_vel = a * raw_vel + (1.0 - a) * self._prev_joint_vel
            else:
                target_joint_vel = raw_vel
            self._prev_joint_vel = target_joint_vel.copy()
            self.arm.set_joint_pos_vel(target_joint_pos, target_joint_vel, relative=False)
        elif blocking:
            wait_time = 1.0 / max(1, self.control_hz)
            self.arm.set_joint_pos(target_joint_pos, wait_time=wait_time)
        else:
            self.arm._send_position_command(target_joint_pos)
        # #region agent log
        _uj_t2 = _time.time()
        # #endregion
        self.sync_motion_manager_with_arm(target_joint_pos)
        # #region agent log
        _uj_t3 = _time.time()
        _dlog("robot.py:update_joints","uj_timing",{"validate_clip_ms":(_uj_t1-_uj_t0)*1000,"send_cmd_ms":(_uj_t2-_uj_t1)*1000,"sync_mm_ms":(_uj_t3-_uj_t2)*1000,"total_ms":(_uj_t3-_uj_t0)*1000,"was_clipped":was_clipped},hyp="H_UJ_TIMING")
        # #endregion

    def update_gripper(self, command: float, velocity: bool = True, blocking: bool = False) -> None:
        if self.hand is None:
            self._prev_gripper_command_successful = True
            return
        current = self.get_cached_gripper_position()
        if velocity:
            target = current + float(command) * (1.0 / max(1, self.control_hz))
        else:
            target = float(command)

        target = float(np.clip(target, 0.0, 1.0))
        wait_time = (1.0 / max(1, self.control_hz)) if blocking else 0.0

        if blocking:
            self._clear_gripper_command_queue()
            self._execute_gripper_command(target, wait_time=wait_time, raise_on_error=True)
            self._refresh_gripper_state()
            return

        self._enqueue_gripper_target(target)
        self._prev_gripper_command_successful = True

    def get_robot_state(self) -> tuple[dict[str, Any], dict[str, int]]:
        # #region agent log
        _gs_t0 = _time.time()
        # #endregion
        joint_positions = np.asarray(self.arm.get_joint_pos(), dtype=np.float64)
        joint_velocities = np.asarray(self.arm.get_joint_vel(), dtype=np.float64)
        try:
            joint_torques = np.asarray(self.arm.get_joint_torque(), dtype=np.float64)
        except ValueError:
            joint_torques = np.zeros(7, dtype=np.float64)
        # #region agent log
        _gs_t1 = _time.time()
        # #endregion
        gripper_position = self.get_cached_gripper_position() if self.hand is not None else 0.0
        # #region agent log
        _gs_t2 = _time.time()
        # #endregion

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
        # #region agent log
        _gs_t3 = _time.time()
        _dlog("robot.py:get_state","gs_timing",{"arm_queries_ms":(_gs_t1-_gs_t0)*1000,"hand_query_ms":(_gs_t2-_gs_t1)*1000,"fk_ts_ms":(_gs_t3-_gs_t2)*1000,"total_ms":(_gs_t3-_gs_t0)*1000},hyp="H_GS_TIMING")
        # #endregion

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

    def move_to_joints_planned(
        self,
        target_arm_joints: np.ndarray,
        control_frequency: float = 100.0,
    ) -> None:
        """Plan and execute a smooth trajectory from current to target joints."""
        from dexmotion.utils import robot_utils

        mm = self.ik_controller.motion_manager
        actual = np.asarray(self.arm.get_joint_pos(), dtype=np.float64)
        self.sync_motion_manager_with_arm(actual)

        start_dict = dict(mm.get_joint_pos_dict())
        goal_dict = dict(start_dict)
        for i, name in enumerate(self._arm_joint_names):
            if name in goal_dict:
                goal_dict[name] = float(target_arm_joints[i])

        waypoints = mm.plan_to_configuration(start_dict, goal_dict)
        if waypoints is None or len(waypoints) == 0:
            self.arm.set_joint_pos(target_arm_joints, wait_time=0.0)
            return

        wp_rows = []
        for wp in waypoints:
            if isinstance(wp, dict):
                wp_rows.append(robot_utils.get_qpos_from_joint_dict(mm.pin_robot, wp))
            else:
                wp_rows.append(np.asarray(wp, dtype=np.float64))
        wp_array = np.vstack(wp_rows)

        traj = mm.smooth_trajectory(
            wp_array, method="auto", control_frequency=control_frequency,
        )
        positions = traj.get("positions", wp_array)
        dt = 1.0 / control_frequency

        joint_names_full = list(mm.get_joint_pos_dict().keys())
        arm_indices = [joint_names_full.index(n) for n in self._arm_joint_names if n in joint_names_full]

        # #region agent log
        _dlog("robot.py:move_planned", "traj_info", {"wp_count": len(wp_rows), "traj_len": len(positions), "dt": dt, "arm_indices": arm_indices}, hyp="H1_plan")
        # #endregion

        for pos in positions:
            arm_pos = pos[arm_indices] if arm_indices else pos[:7]
            self.arm.set_joint_pos(np.asarray(arm_pos, dtype=np.float64), wait_time=0.0)
            _time.sleep(dt)

        mm.set_joint_pos(goal_dict)

    def sync_motion_manager_with_arm(self, arm_joint_pos: np.ndarray) -> None:
        try:
            motion_manager = self.ik_controller.motion_manager
            qpos_dict = motion_manager.get_joint_pos_dict()
            for i, joint_name in enumerate(self._arm_joint_names):
                if joint_name in qpos_dict:
                    qpos_dict[joint_name] = float(arm_joint_pos[i])
            # Clip to joint limits so motion manager accepts the configuration
            # even when the real arm is marginally outside limits (IK numerical error)
            if robot_utils is not None:
                qpos_dict = robot_utils.clip_joint_positions_to_limits(
                    motion_manager.pin_robot, qpos_dict
                )
            motion_manager.set_joint_pos(qpos_dict)
            # #region agent log
            mm_after_full = np.asarray(motion_manager.get_joint_pos(), dtype=np.float64)
            _dlog("robot.py:sync_mm","sync_done",{"target":np.asarray(arm_joint_pos),"mm_full_shape":mm_after_full.shape[0],"mm_after_full":mm_after_full},hyp="H2A")
            # #endregion
        except Exception:
            pass

    def _solve_cartesian_delta(self, delta_xyz: np.ndarray, delta_rpy: np.ndarray) -> np.ndarray:
        # #region agent log
        _scd_t0 = _time.time()
        # #endregion
        actual_joints = np.asarray(self.arm.get_joint_pos(), dtype=np.float64)
        # #region agent log
        _scd_t1 = _time.time()
        # #endregion
        self.sync_motion_manager_with_arm(actual_joints)
        # #region agent log
        _scd_t2 = _time.time()
        # #endregion
        # #region agent log
        mm_qpos_full = np.asarray(self.ik_controller.motion_manager.get_joint_pos(), dtype=np.float64)
        cart_before_fk = self._get_cartesian_pose(actual_joints)
        mm_arm_qpos = mm_qpos_full[:7] if self.arm_side == "left" else mm_qpos_full[7:14] if mm_qpos_full.shape[0] >= 14 else mm_qpos_full
        cart_before_mm = self._get_cartesian_pose(mm_arm_qpos)
        _dlog("robot.py:solve_cart","ik_input",{"delta_xyz":delta_xyz,"delta_rpy":delta_rpy,"mm_qpos_full_shape":mm_qpos_full.shape[0],"mm_arm_qpos":mm_arm_qpos,"actual_joints":actual_joints,"cart_fk_actual":cart_before_fk,"cart_fk_mm":cart_before_mm,"qpos_diff":(mm_arm_qpos-actual_joints)},hyp="H2A,H2B")
        # #endregion
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
        # #region agent log
        cart_after_ik = self._get_cartesian_pose(target_joint_pos)
        cart_expected = cart_before_mm.copy()
        cart_expected[:3] += np.asarray(delta_xyz, dtype=np.float64)
        _scd_t3 = _time.time()
        _dlog("robot.py:solve_cart","ik_output",{"target_joints":target_joint_pos,"cart_after_ik":cart_after_ik,"cart_expected":cart_expected,"cart_error":cart_after_ik-cart_expected,"joint_delta":target_joint_pos-mm_arm_qpos},hyp="H2A,H2B,H2C")
        _dlog("robot.py:solve_cart","scd_timing",{"get_joints_ms":(_scd_t1-_scd_t0)*1000,"sync_mm_ms":(_scd_t2-_scd_t1)*1000,"ik_total_ms":(_scd_t3-_scd_t2)*1000,"scd_total_ms":(_scd_t3-_scd_t0)*1000},hyp="H_SCD_TIMING")
        # #endregion
        # #region agent log
        _ik_joint_delta = target_joint_pos - actual_joints
        _dlog2("robot.py:solve_cart", "ik_result", {"step": _clip_step_counter[0], "delta_xyz": np.asarray(delta_xyz), "delta_rpy": np.asarray(delta_rpy), "delta_xyz_norm": float(np.linalg.norm(delta_xyz)), "delta_rpy_norm_deg": float(np.rad2deg(np.linalg.norm(delta_rpy))), "ik_joint_delta": _ik_joint_delta, "ik_joint_delta_abs_max": float(np.max(np.abs(_ik_joint_delta))), "ik_joint_delta_per_joint_deg": np.rad2deg(np.abs(_ik_joint_delta))}, hyp="H1_CLIP_FREQ,H2_JOINT_DIST")
        # #endregion
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
        self._stop_gripper_worker()
        if self.gripper_type == "robotiq" and self.hand is not None:
            try:
                self.hand.shutdown()
            except Exception:
                pass
        try:
            self.robot.shutdown()
        except Exception:
            pass

"""Vega RobotEnv gRPC Service."""

from __future__ import annotations

import argparse
import logging
import signal
import sys
import time
from concurrent import futures
from pathlib import Path
from typing import Any, Optional

import grpc
import numpy as np
from scipy.spatial.transform import Rotation as R

# Add package root for local imports.
# server.py lives at: <repo>/src/dexcontrol/core/robotenv_vega/server.py
#   parents: [0]=robotenv_vega, [1]=core, [2]=dexcontrol, [3]=src, [4]=<repo>
_this = Path(__file__).resolve()
sys.path.insert(0, str(_this.parents[2]))  # src/dexcontrol/ -> "from core.vega..."
sys.path.insert(0, str(_this.parents[4]))  # <repo>/          -> "from proto..."

from core.vega.robot import (  # noqa: E402
    CommunicationFailedError,
    IKFailedError,
    JointLimitExceededError,
    VegaRobot,
)
from proto import robotenv_pb2, robotenv_pb2_grpc  # noqa: E402


LOGGER = logging.getLogger("robotenv_vega")


class VegaRobotEnvService(robotenv_pb2_grpc.RobotEnvServicer):
    """RobotEnv service implementation for one Vega arm."""

    def __init__(
        self,
        robot_model: str = "vega_1",
        arm_side: str = "left",
        gripper_type: str = "default",
        frame_type: str = "vega_mobile_base",
        control_hz: int = 20,
        base_frame_rotation: Optional[list[float]] = None,
        **kwargs,
    ):
        hand_type = kwargs.pop("hand_type", None)
        if kwargs:
            unexpected = ", ".join(sorted(kwargs.keys()))
            raise TypeError(f"Unexpected keyword arguments: {unexpected}")
        if hand_type is not None and gripper_type == "default":
            gripper_type = hand_type

        self.robot_model = robot_model
        self.arm_side = arm_side
        self.gripper_type = gripper_type
        self.frame_type = frame_type
        self.control_hz = int(control_hz)
        self.base_frame_rotation = base_frame_rotation

        self._robot = VegaRobot(
            robot_model=robot_model,
            arm_side=arm_side,
            control_hz=control_hz,
            gripper_type=gripper_type,
        )
        self._robot.launch_robot()

        self.reset_joints = self._robot.reset_joints.copy()
        self.safe_transit_pose = self._robot.safe_transit_pose.copy()

        if base_frame_rotation is not None:
            rot = R.from_euler("XYZ", base_frame_rotation, degrees=True).as_matrix()
            self.R_world_to_robot = rot.T
            self.R_robot_to_world = rot
        else:
            self.R_world_to_robot = None
            self.R_robot_to_world = None

        LOGGER.info(
            "Initialized VegaRobotEnvService model=%s arm=%s gripper=%s frame=%s hz=%s",
            robot_model,
            arm_side,
            gripper_type,
            frame_type,
            control_hz,
        )

    def GetObservationSpec(self, request, context):
        del request, context
        spec = robotenv_pb2.ObservationSpec(
            num_arms=1,
            metadata={
                "robot_type": "vega",
                "robot_model": self.robot_model,
                "arm_side": self.arm_side,
                "gripper_type": self.gripper_type,
                "frame_type": self.frame_type,
                "control_hz": str(self.control_hz),
            },
        )

        spec.fields["joint_positions"].CopyFrom(
            robotenv_pb2.FieldSpec(
                dtype="float64",
                shape=[7],
                required=True,
                description="Joint positions in radians",
            )
        )
        spec.fields["joint_velocities"].CopyFrom(
            robotenv_pb2.FieldSpec(
                dtype="float64",
                shape=[7],
                required=False,
                description="Joint velocities in rad/s",
            )
        )
        spec.fields["joint_torques_computed"].CopyFrom(
            robotenv_pb2.FieldSpec(
                dtype="float64",
                shape=[7],
                required=False,
                description="Estimated joint torques in Nm",
            )
        )
        spec.fields["prev_joint_torques_computed"].CopyFrom(
            robotenv_pb2.FieldSpec(
                dtype="float64",
                shape=[7],
                required=False,
                description="Previous joint torques computed",
            )
        )
        spec.fields["prev_joint_torques_computed_safened"].CopyFrom(
            robotenv_pb2.FieldSpec(
                dtype="float64",
                shape=[7],
                required=False,
                description="Previous joint torques computed (safened)",
            )
        )
        spec.fields["motor_torques_measured"].CopyFrom(
            robotenv_pb2.FieldSpec(
                dtype="float64",
                shape=[7],
                required=False,
                description="Motor torques measured",
            )
        )
        spec.fields["gripper_position"].CopyFrom(
            robotenv_pb2.FieldSpec(
                dtype="float64",
                shape=[],
                required=True,
                description="Scalar gripper position in [0,1]",
            )
        )
        spec.fields["cartesian_position"].CopyFrom(
            robotenv_pb2.FieldSpec(
                dtype="float64",
                shape=[6],
                required=True,
                description="End-effector pose [x, y, z, roll, pitch, yaw]",
            )
        )
        spec.fields["wrench_state"].CopyFrom(
            robotenv_pb2.FieldSpec(
                dtype="float64",
                shape=[6],
                required=False,
                description="Wrench state [fx, fy, fz, tx, ty, tz]",
            )
        )
        spec.fields["prev_controller_latency_ms"].CopyFrom(
            robotenv_pb2.FieldSpec(
                dtype="float64",
                shape=[],
                required=False,
                description="Controller latency reported by robot stack",
            )
        )
        spec.fields["prev_command_successful"].CopyFrom(
            robotenv_pb2.FieldSpec(
                dtype="int64",
                shape=[],
                required=False,
                description="1 if previous command succeeded, otherwise 0",
            )
        )
        spec.fields["prev_gripper_command_successful"].CopyFrom(
            robotenv_pb2.FieldSpec(
                dtype="int64",
                shape=[],
                required=False,
                description="1 if previous gripper command succeeded, otherwise 0",
            )
        )
        spec.fields["timestamp"].CopyFrom(
            robotenv_pb2.FieldSpec(
                dtype="int64",
                shape=[],
                required=True,
                description="Robot timestamp in microseconds",
            )
        )
        return spec

    def GetConfig(self, request, context):
        del request, context
        return robotenv_pb2.RobotConfig(
            gripper_type=self.gripper_type,
            frame_type=self.frame_type,
            dof=[7],
            supported_action_spaces=[
                "joint_position",
                "joint_velocity",
                "joint_delta",
                "cartesian_velocity",
                "cartesian_delta",
            ],
            metadata={
                "robot_model": self.robot_model,
                "control_hz": str(self.control_hz),
                "arm_side": self.arm_side,
            },
        )

    def Reset(self, request, context):
        mode = request.mode or "home"
        LOGGER.info("Reset requested: mode=%s arm=%s", mode, self.arm_side)

        try:
            if mode == "home":
                target_joints = self.reset_joints
                self._execute_reset_sequence(target_joints)
            elif mode == "target":
                target_joints = self._extract_target_joints(request)
                self._execute_reset_sequence(target_joints)
            elif mode == "random":
                target_joints = self._sample_random_target()
                self._execute_reset_sequence(target_joints)
            elif mode == "current":
                pass
            else:
                context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
                context.set_details(f"Unknown reset mode: {mode}")
                return robotenv_pb2.ResetResponse()

            observation, timestamp_us = self._create_observation()
            return robotenv_pb2.ResetResponse(
                observation=observation,
                status="SUCCESS",
                message=f"Reset to {mode}",
                timestamp_us=timestamp_us,
            )
        except Exception as exc:
            LOGGER.exception("Reset failed (mode=%s): %s", mode, exc)
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(f"Reset failed: {exc}")
            return robotenv_pb2.ResetResponse()

    def Step(self, request, context):
        action_space = request.action_space
        action = np.asarray(request.action, dtype=np.float64)
        gripper_action_space = request.gripper_action_space
        if not gripper_action_space:
            gripper_action_space = "velocity" if "velocity" in action_space else "position"

        try:
            if self.R_world_to_robot is not None and "cartesian" in action_space:
                action = self._transform_action_to_robot_frame(action)

            self._robot.update_command(
                action,
                action_space=action_space,
                gripper_action_space=gripper_action_space,
                blocking=False,
            )
            observation, timestamp_us = self._create_observation()
            return robotenv_pb2.StepResponse(
                observation=observation,
                status="SUCCESS",
                message="",
                timestamp_us=timestamp_us,
            )
        except JointLimitExceededError as exc:
            observation, timestamp_us = self._safe_observation()
            return robotenv_pb2.StepResponse(
                observation=observation,
                status="JOINT_LIMIT_EXCEEDED",
                message=str(exc),
                timestamp_us=timestamp_us,
            )
        except IKFailedError as exc:
            observation, timestamp_us = self._safe_observation()
            return robotenv_pb2.StepResponse(
                observation=observation,
                status="IK_FAILED",
                message=str(exc),
                timestamp_us=timestamp_us,
            )
        except CommunicationFailedError as exc:
            observation, timestamp_us = self._safe_observation()
            return robotenv_pb2.StepResponse(
                observation=observation,
                status="COMMUNICATION_FAILED",
                message=str(exc),
                timestamp_us=timestamp_us,
            )
        except Exception as exc:
            LOGGER.exception("Step failed: %s", exc)
            observation, timestamp_us = self._safe_observation()
            return robotenv_pb2.StepResponse(
                observation=observation,
                status="ERROR",
                message=str(exc),
                timestamp_us=timestamp_us,
            )

    def HealthCheck(self, request, context):
        del request, context
        try:
            self._robot.get_robot_state()
            return robotenv_pb2.HealthCheckResponse(
                status="HEALTHY",
                message=f"Vega {self.arm_side} arm operational",
                version="1.0.0-vega",
            )
        except Exception as exc:
            return robotenv_pb2.HealthCheckResponse(
                status="UNHEALTHY",
                message=str(exc),
                version="1.0.0-vega",
            )

    def _create_observation(self) -> tuple[dict[str, Any], int]:
        state_dict, timestamp_dict = self._robot.get_robot_state()
        timestamp_us = self._timestamp_us(timestamp_dict)

        cartesian_position = np.asarray(state_dict["cartesian_position"], dtype=np.float64)
        if self.R_robot_to_world is not None:
            cartesian_position = self._transform_state_to_env_frame(cartesian_position)

        observation = {
            "joint_positions": robotenv_pb2.Value(
                float_array=robotenv_pb2.FloatArray(values=np.asarray(state_dict["joint_positions"]).tolist())
            ),
            "joint_velocities": robotenv_pb2.Value(
                float_array=robotenv_pb2.FloatArray(values=np.asarray(state_dict["joint_velocities"]).tolist())
            ),
            "joint_torques_computed": robotenv_pb2.Value(
                float_array=robotenv_pb2.FloatArray(values=np.asarray(state_dict["joint_torques_computed"]).tolist())
            ),
            "prev_joint_torques_computed": robotenv_pb2.Value(
                float_array=robotenv_pb2.FloatArray(values=np.asarray(state_dict.get("prev_joint_torques_computed", [0]*7)).tolist())
            ),
            "prev_joint_torques_computed_safened": robotenv_pb2.Value(
                float_array=robotenv_pb2.FloatArray(values=np.asarray(state_dict.get("prev_joint_torques_computed_safened", [0]*7)).tolist())
            ),
            "motor_torques_measured": robotenv_pb2.Value(
                float_array=robotenv_pb2.FloatArray(values=np.asarray(state_dict.get("motor_torques_measured", [0]*7)).tolist())
            ),
            "gripper_position": robotenv_pb2.Value(
                float_value=float(state_dict["gripper_position"])
            ),
            "cartesian_position": robotenv_pb2.Value(
                float_array=robotenv_pb2.FloatArray(values=cartesian_position.tolist())
            ),
            "wrench_state": robotenv_pb2.Value(
                float_array=robotenv_pb2.FloatArray(values=np.asarray(state_dict["wrench_state"]).tolist())
            ),
            "prev_controller_latency_ms": robotenv_pb2.Value(
                float_value=float(state_dict.get("prev_controller_latency_ms", 0.0))
            ),
            "prev_command_successful": robotenv_pb2.Value(
                int_value=int(bool(state_dict.get("prev_command_successful", True)))
            ),
            "prev_gripper_command_successful": robotenv_pb2.Value(
                int_value=int(bool(state_dict.get("prev_gripper_command_successful", True)))
            ),
            "timestamp": robotenv_pb2.Value(
                int_value=int(timestamp_us)
            ),
        }
        return observation, int(timestamp_us)

    def _safe_observation(self) -> tuple[dict[str, Any], int]:
        try:
            return self._create_observation()
        except Exception:
            now_us = int(time.time() * 1_000_000)
            return {}, now_us

    def _execute_reset_sequence(self, target_joints: np.ndarray) -> None:
        self._robot.update_gripper(0.0, velocity=False, blocking=True)
        self._robot.update_joints(self.safe_transit_pose, velocity=False, blocking=True)
        self._robot.update_joints(np.asarray(target_joints, dtype=np.float64), velocity=False, blocking=True)
        # Allow time for the arm to reach the target before we return the observation
        time.sleep(4.0)

    def _extract_target_joints(self, request) -> np.ndarray:
        if "joint_positions" not in request.params:
            raise ValueError("target reset requires params['joint_positions']")
        joint_values = request.params["joint_positions"].float_array.values
        if len(joint_values) != 7:
            raise ValueError(f"Expected 7 joint values, got {len(joint_values)}")
        target = np.asarray(joint_values, dtype=np.float64)
        limits = self._robot.arm.joint_pos_limit
        if limits is not None:
            low = limits[:, 0].astype(np.float64)
            high = limits[:, 1].astype(np.float64)
            violates = np.logical_or(target < low, target > high)
            if np.any(violates):
                target = np.clip(target, low, high)
                LOGGER.warning(
                    "Reset target joints clipped to limits (violated indices: %s)",
                    np.where(violates)[0].tolist(),
                )
        self._robot.validate_joint_limits(target)
        return target

    def _sample_random_target(self) -> np.ndarray:
        limits = self._robot.arm.joint_pos_limit
        if limits is None:
            return self.reset_joints.copy()

        center = self.reset_joints.astype(np.float64)
        noise = np.random.uniform(-0.15, 0.15, size=7)
        sampled = center + noise
        sampled = np.clip(sampled, limits[:, 0], limits[:, 1])
        return sampled

    def _timestamp_us(self, timestamp_dict: dict[str, Any]) -> int:
        if "robot_timestamp_us" in timestamp_dict:
            return int(timestamp_dict["robot_timestamp_us"])
        sec = int(timestamp_dict.get("robot_timestamp_seconds", int(time.time())))
        nsec = int(timestamp_dict.get("robot_timestamp_nanos", 0))
        return sec * 1_000_000 + (nsec // 1_000)

    def _transform_action_to_robot_frame(self, action: np.ndarray) -> np.ndarray:
        action = np.asarray(action, dtype=np.float64).copy()
        action[:3] = self.R_world_to_robot @ action[:3]
        action[3:6] = self.R_world_to_robot @ action[3:6]
        return action

    def _transform_state_to_env_frame(self, state: np.ndarray) -> np.ndarray:
        state = np.asarray(state, dtype=np.float64).copy()
        state[:3] = self.R_robot_to_world @ state[:3]
        state[3:6] = self.R_robot_to_world @ state[3:6]
        return state


def serve(
    grpc_port: int = 50061,
    robot_model: str = "vega_1",
    arm_side: str = "left",
    gripper_type: str = "default",
    frame_type: str = "vega_mobile_base",
    control_hz: int = 20,
    base_frame_rotation: Optional[list[float]] = None,
    **kwargs,
) -> None:
    """Start Vega RobotEnv gRPC server."""
    hand_type = kwargs.pop("hand_type", None)
    if kwargs:
        unexpected = ", ".join(sorted(kwargs.keys()))
        raise TypeError(f"Unexpected keyword arguments: {unexpected}")
    if hand_type is not None and gripper_type == "default":
        gripper_type = hand_type

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    service = VegaRobotEnvService(
        robot_model=robot_model,
        arm_side=arm_side,
        gripper_type=gripper_type,
        frame_type=frame_type,
        control_hz=control_hz,
        base_frame_rotation=base_frame_rotation,
    )

    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    robotenv_pb2_grpc.add_RobotEnvServicer_to_server(service, server)

    server_address = f"0.0.0.0:{grpc_port}"
    server.add_insecure_port(server_address)

    server.start()
    LOGGER.info(
        "Vega RobotEnv server started on %s (arm=%s, model=%s)",
        server_address,
        arm_side,
        robot_model,
    )

    def shutdown_handler(signum, frame):
        del signum, frame
        LOGGER.info("Shutting down Vega RobotEnv server")
        try:
            service._robot.close()
        except Exception:
            pass
        server.stop(grace=5)
        sys.exit(0)

    signal.signal(signal.SIGINT, shutdown_handler)
    signal.signal(signal.SIGTERM, shutdown_handler)
    server.wait_for_termination()


def main() -> None:
    parser = argparse.ArgumentParser(description="Vega RobotEnv gRPC Server")
    parser.add_argument("--grpc-port", type=int, default=50061, help="gRPC service port")
    parser.add_argument(
        "--robot-model",
        type=str,
        default="vega_1",
        help="Robot model (e.g. vega_1, vega_1_f5d6, vega_1u)",
    )
    parser.add_argument(
        "--arm-side",
        type=str,
        required=True,
        choices=["left", "right"],
        help="Which arm this server controls",
    )
    parser.add_argument(
        "--gripper-type",
        type=str,
        default="default",
        help="Gripper type metadata",
    )
    parser.add_argument(
        "--hand-type",
        dest="gripper_type",
        type=str,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--frame-type",
        type=str,
        default="vega_mobile_base",
        choices=["vega_mobile_base", "vega_table_mount", "vega_custom"],
        help="Robot mounting frame type",
    )
    parser.add_argument("--control-hz", type=int, default=20, help="Control frequency in Hz")
    parser.add_argument(
        "--base-frame-rotation",
        type=float,
        nargs=3,
        default=None,
        metavar=("ROLL", "PITCH", "YAW"),
        help="Custom base-frame rotation in degrees",
    )
    args = parser.parse_args()

    serve(
        grpc_port=args.grpc_port,
        robot_model=args.robot_model,
        arm_side=args.arm_side,
        gripper_type=args.gripper_type,
        frame_type=args.frame_type,
        control_hz=args.control_hz,
        base_frame_rotation=args.base_frame_rotation,
    )


if __name__ == "__main__":
    main()

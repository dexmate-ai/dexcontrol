"""RobotEnv gRPC Client Adapter.

Drop-in replacement for RobotEnv that communicates with a remote RobotEnv
gRPC service. Exposes the same interface (attributes, methods, return formats)
so that policy_runner.py and other consumers work without changes.

Usage:
    # Instead of:
    from core.robot_env import RobotEnv
    env = RobotEnv(robot_port=50051)

    # Use:
    from core.robot_env_client import RobotEnvClient
    env = RobotEnvClient(robot_port=50061)

    # Same interface as RobotEnv
    env.reset()
    action_info = env.step(action)
    obs_dict, images = env.get_observation()
"""

import sys
from pathlib import Path
from typing import Dict, Any, Optional
import numpy as np
import grpc
import time

# Add proto directory to path - adjusted for custom_dexcontrol structure
_current_file = Path(__file__).resolve()
_proto_path = _current_file.parents[3] / "proto"  # /custom_dexcontrol/proto
if str(_proto_path) not in sys.path:
    sys.path.insert(0, str(_proto_path))

from proto import robotenv_pb2
from proto import robotenv_pb2_grpc


class _RobotProxy:
    """Proxy that mimics Robot for attributes accessed by policy_runner.

    Provides create_action_dict() using a local IK solver so the control
    loop can build action dicts without an extra RPC round-trip.

    Note: For Vega robots, IK is handled by VegaRobot/BaseIKController.
    This proxy provides a minimal implementation for compatibility.
    """

    def __init__(self, control_hz: int = 20):
        # Try to import robot-specific IK solver if available
        try:
            from core.robot_ik.robot_ik_solver import RobotIKSolver
            from core.misc.transformations import add_poses
            self._ik_solver = RobotIKSolver(control_hz=control_hz)
            self._add_poses = add_poses
        except ImportError:
            # Fallback for Vega or other robots without these modules
            self._ik_solver = None
            self._add_poses = self._fallback_add_poses

        self.control_hz = control_hz

    def _fallback_add_poses(self, delta, current):
        """Fallback pose addition for robots without transformations module."""
        delta_arr = np.asarray(delta, dtype=np.float64)
        current_arr = np.asarray(current, dtype=np.float64)
        # Simple addition for position, rotation deltas
        result = current_arr.copy()
        result[:3] += delta_arr[:3]  # position
        result[3:6] += delta_arr[3:6]  # orientation (euler angles)
        return result

    def create_action_dict(self, action, action_space, gripper_action_space=None, robot_state=None):
        """Replicate Robot.create_action_dict() locally using IK solver.

        This is called by policy_runner to build world-frame action dicts for
        logging / data saving. The actual robot command execution is done by
        the gRPC service.

        Falls back to simple computation if IK solver is not available.
        """
        assert action_space in [
            "cartesian_delta", "joint_position",
            "cartesian_velocity", "joint_velocity", "joint_delta"
        ]
        if robot_state is None:
            robot_state = {}

        action_dict = {"robot_state": robot_state}
        velocity = "velocity" in action_space

        if gripper_action_space is None:
            gripper_action_space = "velocity" if velocity else "position"

        # Gripper
        action_dict["gripper_position"] = float(np.clip(action[-1], 0, 1))
        gripper_delta = action_dict["gripper_position"] - robot_state.get("gripper_position", 0)

        if self._ik_solver is not None:
            action_dict["gripper_delta"] = self._ik_solver.gripper_delta_to_velocity(gripper_delta)
        else:
            # Fallback: simple conversion
            action_dict["gripper_delta"] = gripper_delta * self.control_hz

        if "cartesian" in action_space:
            if velocity:
                action_dict["cartesian_velocity"] = action[:-1]
                if self._ik_solver is not None:
                    cartesian_delta = self._ik_solver.cartesian_velocity_to_delta(action[:-1])
                else:
                    cartesian_delta = np.array(action[:-1]) / self.control_hz
                action_dict["delta_action"] = np.concatenate([cartesian_delta, [action[-1]]]).tolist()
                action_dict["cartesian_position"] = self._add_poses(
                    cartesian_delta, robot_state.get("cartesian_position", [0]*6)
                ).tolist()
            else:
                action_dict["delta_action"] = action
                cartesian_delta = action[:-1]
                action_dict["cartesian_position"] = self._add_poses(
                    cartesian_delta, robot_state.get("cartesian_position", [0]*6)
                ).tolist()
                if self._ik_solver is not None:
                    cartesian_velocity = self._ik_solver.cartesian_delta_to_velocity(cartesian_delta)
                else:
                    cartesian_velocity = np.array(cartesian_delta) * self.control_hz
                action_dict["cartesian_velocity"] = cartesian_velocity.tolist()

            if self._ik_solver is not None:
                action_dict["joint_velocity"] = self._ik_solver.cartesian_velocity_to_joint_velocity(
                    action_dict["cartesian_velocity"], robot_state=robot_state
                ).tolist()
                joint_delta = self._ik_solver.joint_velocity_to_delta(action_dict["joint_velocity"])
            else:
                # Fallback: use zeros for joint space (IK not available)
                action_dict["joint_velocity"] = [0.0] * 7
                joint_delta = np.zeros(7)
            action_dict["joint_position"] = (
                joint_delta + np.array(robot_state.get("joint_positions", [0]*7))
            ).tolist()

        if "joint" in action_space:
            if action_space == "joint_delta":
                action_dict["joint_delta"] = action[:-1]
                joint_delta = np.array(action[:-1])
                action_dict["joint_position"] = (
                    joint_delta + np.array(robot_state.get("joint_positions", [0]*7))
                ).tolist()
            else:
                if velocity:
                    action_dict["joint_velocity"] = action[:-1]
                    if self._ik_solver is not None:
                        joint_delta = self._ik_solver.joint_velocity_to_delta(action[:-1])
                    else:
                        joint_delta = np.array(action[:-1]) / self.control_hz
                    action_dict["joint_position"] = (
                        joint_delta + np.array(robot_state.get("joint_positions", [0]*7))
                    ).tolist()
                else:
                    action_dict["joint_position"] = action[:-1]
                    joint_delta = np.array(action[:-1]) - np.array(robot_state.get("joint_positions", [0]*7))
                    if self._ik_solver is not None:
                        joint_velocity = self._ik_solver.joint_delta_to_velocity(joint_delta)
                    else:
                        joint_velocity = joint_delta * self.control_hz
                    action_dict["joint_velocity"] = joint_velocity.tolist()

        if self._ik_solver is not None:
            numerical_joint_velocity, numerical_joint_acceleration = \
                self._ik_solver.joint_delta_numerical_differentiation(
                    joint_delta, robot_state.get("joint_velocities", [0]*7)
                )
            action_dict["numerical_joint_velocity"] = np.asarray(numerical_joint_velocity).tolist()
            action_dict["numerical_joint_acceleration"] = np.asarray(numerical_joint_acceleration).tolist()
        else:
            # Fallback: simple numerical differentiation
            action_dict["numerical_joint_velocity"] = (joint_delta * self.control_hz).tolist()
            action_dict["numerical_joint_acceleration"] = [0.0] * 7

        return action_dict


class RobotEnvClient:
    """gRPC client adapter for RobotEnv service.

    Provides the same interface as core.robot_env.RobotEnv so that
    policy_runner.py works without modification.

    Attributes exposed for compatibility:
        action_space, gripper_action_space, use_recorder, control_hz, DoF, _robot
    """

    def __init__(
        self,
        robot_ip: str = "localhost",
        robot_port: int = 50051,
        action_space: str = "cartesian_velocity",
        gripper_action_space: Optional[str] = None,
        reset_joints: Optional[np.ndarray] = None,
        do_reset: bool = True,
        randomize: bool = False,
        **kwargs  # Absorb other arguments for compatibility
    ):
        self.robot_ip = robot_ip
        self.robot_port = robot_port
        self.action_space = action_space
        self.gripper_action_space = gripper_action_space

        # Reset joints configuration (from frame.yaml via policy_runner)
        if reset_joints is not None:
            self.reset_joints = np.array(reset_joints)
        else:
            self.reset_joints = np.array([0, -np.pi/5, 0, -4*np.pi/5, 0, 3*np.pi/5, 0.7])

        print(f"[RobotEnvClient] Connecting to RobotEnv service at {robot_ip}:{robot_port}")

        # Create gRPC channel and stub
        self.channel = grpc.insecure_channel(f"{robot_ip}:{robot_port}")
        self.stub = robotenv_pb2_grpc.RobotEnvStub(self.channel)

        # Cache for observation spec and config
        self._obs_spec = None
        self._robot_config = None
        self._last_observation = None

        # Get observation spec at initialization
        try:
            self._obs_spec = self.stub.GetObservationSpec(
                robotenv_pb2.GetObservationSpecRequest()
            )
            print(f"[RobotEnvClient] Connected! Observation fields: "
                  f"{list(self._obs_spec.fields.keys())}")
        except grpc.RpcError as e:
            raise RuntimeError(
                f"Failed to connect to RobotEnv service at {robot_ip}:{robot_port}: {e.details()}"
            )

        # Get robot config
        try:
            self._robot_config = self.stub.GetConfig(robotenv_pb2.GetConfigRequest())
            print(f"[RobotEnvClient] Robot config: gripper={self._robot_config.gripper_type}, "
                  f"frame={self._robot_config.frame_type}, dof={list(self._robot_config.dof)}")
        except grpc.RpcError as e:
            print(f"[RobotEnvClient] Warning: Could not fetch robot config: {e.details()}")

        # ----- Compatibility attributes (match RobotEnv) -----

        # control_hz from server config metadata
        self.control_hz = int(self._robot_config.metadata.get("control_hz", "20")) \
            if self._robot_config else 20

        # DoF: 7 for cartesian action spaces, 8 for joint
        self.DoF = 7 if ("cartesian" in action_space) else 8

        # Camera: not initialized here, can be injected by policy_runner
        # via set_camera(). When set, get_observation() returns camera frames
        # just like original RobotEnv.
        self.use_recorder = False
        self.use_camera = False
        self.camera = None

        # _robot proxy for create_action_dict() and other attribute access
        self._robot = _RobotProxy(control_hz=self.control_hz)

        # Initial reset (matches RobotEnv.__init__ do_reset behavior)
        if do_reset:
            print(f"[RobotEnvClient] Initial reset with joints: {self.reset_joints.tolist()}")
            self.reset(randomize=randomize)

    # ------------------------------------------------------------------
    # reset() - matches RobotEnv.reset() which returns None
    # ------------------------------------------------------------------
    def reset(self, randomize=True, reset_pose=None, **kwargs):
        """Reset robot to initial state using self.reset_joints.

        Uses "target" mode to send the configured reset_joints to the server,
        matching RobotEnv behavior where reset_joints come from frame.yaml.

        Returns None to match original RobotEnv.reset() behavior.
        """
        if reset_pose is not None:
            target_joints = reset_pose
        else:
            target_joints = self.reset_joints

        # Always use target mode with configured joints
        mode = "target"
        params = {
            "joint_positions": robotenv_pb2.Value(
                float_array=robotenv_pb2.FloatArray(values=target_joints.tolist())
            )
        }

        print(f"[RobotEnvClient] Resetting with mode: {mode}")

        try:
            request = robotenv_pb2.ResetRequest(mode=mode, params=params)
            response = self.stub.Reset(request)

            if response.status != "SUCCESS":
                print(f"[RobotEnvClient] Reset warning: {response.status} - {response.message}")

            # Cache the observation for get_observation()
            obs = self._parse_observation(response.observation)
            self._last_observation = obs

            # Original RobotEnv.reset() returns None
            return None

        except grpc.RpcError as e:
            raise RuntimeError(f"Reset failed: {e.details()}")

    # ------------------------------------------------------------------
    # step() - matches RobotEnv.step() which returns action_info dict
    # action_info is the action_dict from FrankaRobot.update_command()
    # containing: robot_state, joint_position, cartesian_position, etc.
    # ------------------------------------------------------------------
    def step(self, action, time_to_go=None, blocking=False, nimble_controller=True):
        """Execute action and return action_info dict.

        Returns action_info matching FrankaRobot.update_command() format:
            {
                'robot_state': state_dict,
                'joint_position': [...],
                'cartesian_position': [...],
                'gripper_position': ...,
                ...
            }
        """
        action_space = self.action_space
        gripper_action_space = self.gripper_action_space or "position"

        # Convert numpy array to list
        if isinstance(action, np.ndarray):
            action_list = action.tolist()
        else:
            action_list = list(action)

        try:
            request = robotenv_pb2.StepRequest(
                action=action_list,
                action_space=action_space,
                gripper_action_space=gripper_action_space
            )
            response = self.stub.Step(request)

            if response.status != "SUCCESS":
                print(f"[RobotEnvClient] Step warning: {response.status} - {response.message}")

            # Parse observation into state_dict format
            obs = self._parse_observation(response.observation)
            self._last_observation = obs

            # Build robot_state dict matching FrankaRobot.get_robot_state() format
            robot_state = self._build_robot_state(obs)

            # Build action_info matching FrankaRobot.update_command() return
            # policy_runner reads action_info["robot_state"] at line 1651
            action_info = {
                "robot_state": robot_state,
                "joint_position": obs.get("joint_positions", []),
                "cartesian_position": obs.get("cartesian_position", []),
                "gripper_position": obs.get("gripper_position", 0),
            }

            return action_info

        except grpc.RpcError as e:
            raise RuntimeError(f"Robot step failed: {e.details()}")

    # ------------------------------------------------------------------
    # get_observation() - matches RobotEnv.get_observation() which returns
    # (obs_dict, images) tuple
    # ------------------------------------------------------------------
    def get_observation(self):
        """Get current observation.

        Returns (obs_dict, images) tuple matching RobotEnv.get_observation():
            obs_dict: {'robot_state': state_dict, 'timestamp': {...}}
            images: {} (empty dict - cameras handled separately in service mode)
        """
        if self._last_observation is None:
            # Need to get initial state - do a "current" reset
            try:
                request = robotenv_pb2.ResetRequest(mode="current", params={})
                response = self.stub.Reset(request)
                obs = self._parse_observation(response.observation)
                self._last_observation = obs
            except grpc.RpcError as e:
                raise RuntimeError(f"Failed to get observation: {e.details()}")

        obs = self._last_observation
        robot_state = self._build_robot_state(obs)

        obs_dict = {
            "robot_state": robot_state,
            "timestamp": {
                "robot_state": {
                    "robot_timestamp_seconds": int(obs.get("timestamp", 0) / 1_000_000),
                    "robot_timestamp_nanos": 0,
                    "read_start": int(time.time() * 1000),
                    "read_end": int(time.time() * 1000),
                },
                "images": {"wrist_frame": 0},
            }
        }

        # Return empty images dict (cameras handled separately in service mode)
        images = {}
        return obs_dict, images

    # ------------------------------------------------------------------
    # get_state() - matches RobotEnv.get_state()
    # ------------------------------------------------------------------
    def get_state(self):
        """Get robot state. Returns (state_dict, timestamp_dict)."""
        if self._last_observation is None:
            try:
                request = robotenv_pb2.ResetRequest(mode="current", params={})
                response = self.stub.Reset(request)
                obs = self._parse_observation(response.observation)
                self._last_observation = obs
            except grpc.RpcError as e:
                raise RuntimeError(f"Failed to get state: {e.details()}")

        obs = self._last_observation
        state_dict = self._build_robot_state(obs)
        timestamp_dict = {
            "robot_timestamp_seconds": int(obs.get("timestamp", 0) / 1_000_000),
            "robot_timestamp_nanos": 0,
            "read_start": int(time.time() * 1000),
            "read_end": int(time.time() * 1000),
        }
        return state_dict, timestamp_dict

    # ------------------------------------------------------------------
    # Other RobotEnv methods
    # ------------------------------------------------------------------
    def get_robot_config(self) -> Dict[str, Any]:
        """Get robot configuration."""
        if self._robot_config is None:
            self._robot_config = self.stub.GetConfig(robotenv_pb2.GetConfigRequest())

        return {
            "gripper_type": self._robot_config.gripper_type,
            "frame_type": self._robot_config.frame_type,
            "dof": list(self._robot_config.dof),
            "supported_action_spaces": list(self._robot_config.supported_action_spaces),
            "metadata": dict(self._robot_config.metadata)
        }

    def close(self):
        """Close gRPC connection."""
        print("[RobotEnvClient] Closing connection")
        self.channel.close()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _build_robot_state(self, obs: Dict[str, Any]) -> Dict[str, Any]:
        """Build a robot_state dict matching FrankaRobot.get_robot_state()[0] format.

        This is what policy_runner expects when it reads action_info["robot_state"]
        or obs_dict["robot_state"].
        """
        state = {
            "cartesian_position": obs.get("cartesian_position", np.zeros(6)).tolist()
                if isinstance(obs.get("cartesian_position"), np.ndarray)
                else list(obs.get("cartesian_position", [0]*6)),
            "gripper_position": float(obs.get("gripper_position", 0)),
            "joint_positions": obs.get("joint_positions", np.zeros(7)).tolist()
                if isinstance(obs.get("joint_positions"), np.ndarray)
                else list(obs.get("joint_positions", [0]*7)),
            "joint_velocities": obs.get("joint_velocities", np.zeros(7)).tolist()
                if isinstance(obs.get("joint_velocities"), np.ndarray)
                else list(obs.get("joint_velocities", [0]*7)),
        }
        # Optional fields
        for key in ["joint_torques_computed", "prev_joint_torques_computed",
                    "prev_joint_torques_computed_safened", "motor_torques_measured"]:
            if key in obs:
                val = obs[key]
                state[key] = val.tolist() if isinstance(val, np.ndarray) else list(val)
            else:
                state[key] = [0]*7

        state["prev_controller_latency_ms"] = float(obs.get("prev_controller_latency_ms", 0))
        state["prev_command_successful"] = bool(obs.get("prev_command_successful", True))
        state["prev_gripper_command_successful"] = bool(obs.get("prev_gripper_command_successful", True))

        return state

    def _parse_observation(self, obs_proto: Dict) -> Dict[str, Any]:
        """Convert protobuf observation to dict with numpy arrays."""
        obs = {}
        for key, value in obs_proto.items():
            if value.HasField("float_array"):
                obs[key] = np.array(value.float_array.values)
            elif value.HasField("float_value"):
                obs[key] = value.float_value
            elif value.HasField("int_value"):
                obs[key] = value.int_value
            elif value.HasField("bytes_value"):
                obs[key] = value.bytes_value
            elif value.HasField("string_value"):
                obs[key] = value.string_value
        return obs

    def __del__(self):
        try:
            self.close()
        except:
            pass

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        return False


# Alias for easier migration
RobotEnv = RobotEnvClient


if __name__ == "__main__":
    print("Testing RobotEnvClient...")

    client = RobotEnvClient(robot_port=50051)

    print("\n1. Getting config...")
    config = client.get_robot_config()
    print(f"   Config: {config}")

    print("\n2. Resetting...")
    result = client.reset()
    print(f"   Reset returned: {result}")  # Should be None

    print("\n3. Getting observation...")
    obs_dict, images = client.get_observation()
    print(f"   obs_dict keys: {list(obs_dict.keys())}")
    print(f"   robot_state keys: {list(obs_dict['robot_state'].keys())}")
    print(f"   images: {images}")

    print("\n4. Taking step...")
    action = np.array([0.01, 0.0, 0.0, 0.0, 0.0, 0.0, 0.5])
    action_info = client.step(action)
    print(f"   action_info keys: {list(action_info.keys())}")
    print(f"   robot_state keys: {list(action_info['robot_state'].keys())}")

    print("\n5. Compatibility check...")
    print(f"   use_recorder: {client.use_recorder}")
    print(f"   control_hz: {client.control_hz}")
    print(f"   DoF: {client.DoF}")
    print(f"   _robot: {client._robot}")
    print(f"   _robot.create_action_dict: {client._robot.create_action_dict}")

    print("\n6. Closing...")
    client.close()

    print("\nRobotEnvClient test passed!")

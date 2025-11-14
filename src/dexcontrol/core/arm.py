# Copyright (C) 2025 Dexmate Inc.
#
# This software is dual-licensed:
#
# 1. GNU Affero General Public License v3.0 (AGPL-3.0)
#    See LICENSE-AGPL for details
#
# 2. Commercial License
#    For commercial licensing terms, contact: contact@dexmate.ai

"""Robot arm control module.

This module provides the Arm class for controlling a robot arm through Zenoh
communication and the ArmWrenchSensor class for reading wrench sensor data.
"""

import time
from typing import Any, Literal

import numpy as np
from dexcomm import Publisher, call_service
from dexcomm.serialization import serialize_protobuf
from dexcomm.serialization.protobuf import control_msg_pb2, control_query_pb2
from dexcomm.utils import RateLimiter
from jaxtyping import Float
from loguru import logger

from dexcontrol.config.core.arm import ArmConfig
from dexcontrol.core.component import RobotComponent, RobotJointComponent
from dexcontrol.utils.comm_helper import get_zenoh_config_path
from dexcontrol.utils.os_utils import resolve_key_name
from dexcontrol.utils.trajectory_utils import generate_linear_trajectory


class Arm(RobotJointComponent):
    """Robot arm control class.

    This class provides methods to control a robot arm by publishing commands and
    receiving state information through Zenoh communication.

    Attributes:
        mode_querier: Zenoh querier for setting arm mode.
        wrench_sensor: Optional ArmWrenchSensor instance for wrench sensor data.
    """

    def __init__(
        self,
        configs: ArmConfig,
    ) -> None:
        """Initialize the arm controller.

        Args:
            configs: Configuration parameters for the arm including communication topics.
        """
        super().__init__(
            state_sub_topic=configs.state_sub_topic,
            control_pub_topic=configs.control_pub_topic,
            state_message_type=control_msg_pb2.MotorStateWithCurrent,
            joint_name=configs.joint_name,
            joint_limit=configs.joint_limit
            if hasattr(configs, "joint_limit")
            else None,
            joint_vel_limit=configs.joint_vel_limit
            if hasattr(configs, "joint_vel_limit")
            else None,
            pose_pool=configs.pose_pool,
        )

        # Note: Mode querier functionality will need to be updated to use DexComm
        # For now, we'll store the query topic for later use
        self._mode_query_topic = resolve_key_name(configs.set_mode_query)

        # Initialize wrench sensor if configured
        self.wrench_sensor: ArmWrenchSensor | None = None
        if configs.wrench_sub_topic:
            self.wrench_sensor = ArmWrenchSensor(configs.wrench_sub_topic)

        # Initialize end effector pass through publisher using DexComm
        self._ee_pass_through_publisher = Publisher(
            topic=resolve_key_name(configs.ee_pass_through_pub_topic),
            serializer=serialize_protobuf,
            config=get_zenoh_config_path(),
        )

        self._default_control_hz = configs.default_control_hz
        if self._joint_vel_limit is not None:
            if np.any(self._joint_vel_limit > 2.8):
                logger.warning(
                    "Joint velocity limit is greater than 2.8. This is not recommended."
                )
                self._joint_vel_limit = np.clip(self._joint_vel_limit, 0, 2.8)
                logger.warning("Joint velocity limit is clamped to 2.8")

    def set_mode(self, mode: Literal["position", "disable"]) -> None:
        """Sets the operating mode of the arm.

        .. deprecated::
            Use set_modes() instead for setting arm modes.

        Args:
            mode: Operating mode for the arm. Must be either "position" or "disable".
                "position": Enable position control
                "disable": Disable control

        Raises:
            ValueError: If an invalid mode is specified.
        """
        logger.warning("arm.set_mode() is deprecated, use set_modes() instead")
        self.set_modes([mode] * 7)

    def set_modes(self, modes: list[Literal["position", "disable", "release"]]) -> None:
        """Sets the operating modes of the arm.

        Args:
            modes: List of operating modes for the arm. Each mode must be either "position", "disable", or "current".

        Raises:
            ValueError: If any mode in the list is invalid.
        """
        mode_map = {
            "position": control_query_pb2.SetArmMode.Mode.POSITION,
            "disable": control_query_pb2.SetArmMode.Mode.DISABLE,
            "release": control_query_pb2.SetArmMode.Mode.CURRENT,
        }

        for mode in modes:
            if mode not in mode_map:
                raise ValueError(
                    f"Invalid mode: {mode}. Must be one of {list(mode_map.keys())}"
                )

        if len(modes) != 7:
            raise ValueError("Arm modes length must match arm DoF (7).")

        converted_modes = [mode_map[mode] for mode in modes]
        query_msg = control_query_pb2.SetArmMode(modes=converted_modes)
        # Use DexComm's call_service for mode setting
        from dexcontrol.utils.comm_helper import get_zenoh_config_path

        call_service(
            self._mode_query_topic,
            request=query_msg,
            timeout=3.0,
            config=get_zenoh_config_path(),
            request_serializer=lambda x: x.SerializeToString(),
            response_deserializer=None,
        )

    def _send_position_command(self, joint_pos: np.ndarray) -> None:
        """Send joint position command.

        Args:
            joint_pos: Joint positions as numpy array.
        """
        control_msg = control_msg_pb2.MotorPosVelCurrentCommand()
        control_msg.pos.extend(joint_pos.tolist())
        self._publish_control(control_msg)

    def set_joint_pos(
        self,
        joint_pos: Float[np.ndarray, " N"] | list[float] | dict[str, float],
        relative: bool = False,
        wait_time: float = 0.0,
        wait_kwargs: dict[str, float] | None = None,
        exit_on_reach: bool = False,
        exit_on_reach_kwargs: dict[str, Any] | None = None,
    ) -> None:
        """Controls the arm in joint position mode.

        Args:
            joint_pos: Joint positions as either:
                - List of joint values [j1, j2, ..., j7]
                - Numpy array with shape (7,), in radians
                - Dictionary mapping joint names to position values
            relative: If True, the joint positions are relative to the current position.
            wait_time: Time to wait between movements in seconds. If wait_time is 0,
                the joint positions will be sent, and the function call will return
                immediately. If wait_time is greater than 0, the joint positions will
                be interpolated between the current position and the target position,
                and the function will wait for the specified time between each movement.

                **IMPORTANT: When wait_time is 0, you MUST call this function repeatedly
                in a high-frequency loop (e.g., 100 Hz). DO NOT call it just once!**
                The function is designed for continuous control when wait_time=0.
                The highest frequency that the user can call this function is 500 Hz.


            wait_kwargs: Keyword arguments for the interpolation (only used if
                wait_time > 0). Supported keys:
                - control_hz: Control frequency in Hz (default: 100).
                - max_vel: Maximum velocity in rad/s (default: 0.5).
            exit_on_reach: If True, the function will exit when the joint positions are reached.
            exit_on_reach_kwargs: Optional parameters for exit when the joint positions are reached.
        Raises:
            ValueError: If joint_pos dictionary contains invalid joint names.
        """
        if wait_kwargs is None:
            wait_kwargs = {}

        resolved_joint_pos = (
            self._resolve_relative_joint_cmd(joint_pos) if relative else joint_pos
        )
        resolved_joint_pos = self._convert_joint_cmd_to_array(resolved_joint_pos)
        if self._joint_limit is not None:
            resolved_joint_pos = np.clip(
                resolved_joint_pos, self._joint_limit[:, 0], self._joint_limit[:, 1]
            )

        if wait_time > 0.0:
            self._execute_trajectory_motion(
                resolved_joint_pos,
                wait_time,
                wait_kwargs,
                exit_on_reach,
                exit_on_reach_kwargs,
            )
        else:
            self._send_position_command(resolved_joint_pos)

    def _execute_trajectory_motion(
        self,
        target_joint_pos: Float[np.ndarray, " N"],
        wait_time: float,
        wait_kwargs: dict[str, float],
        exit_on_reach: bool = False,
        exit_on_reach_kwargs: dict[str, Any] | None = None,
    ) -> None:
        """Execute trajectory-based motion to target position.

        Args:
            target_joint_pos: Target joint positions as numpy array.
            wait_time: Total time for the motion.
            wait_kwargs: Parameters for trajectory generation.
            exit_on_reach: If True, the function will exit when the joint positions are reached.
            exit_on_reach_kwargs: Optional parameters for exit when the joint positions are reached.
        """
        # Set default parameters
        control_hz = wait_kwargs.get("control_hz", self._default_control_hz)
        max_vel = wait_kwargs.get("max_vel")
        if max_vel is None:
            max_vel = (
                self._joint_vel_limit if self._joint_vel_limit is not None else 2.8
            )
        exit_on_reach_kwargs = exit_on_reach_kwargs or {}
        exit_on_reach_kwargs.setdefault("tolerance", 0.05)

        # Create rate limiter and get current position
        rate_limiter = RateLimiter(control_hz)
        current_joint_pos = self.get_joint_pos().copy()

        # Generate trajectory using utility function
        trajectory, _ = generate_linear_trajectory(
            current_joint_pos, target_joint_pos, max_vel, control_hz
        )
        # Execute trajectory with time limit
        start_time = time.time()
        for pos in trajectory:
            if time.time() - start_time > wait_time:
                break
            self._send_position_command(pos)
            rate_limiter.sleep()

        # Hold final position for remaining time
        while time.time() - start_time < wait_time:
            self._send_position_command(target_joint_pos)
            rate_limiter.sleep()
            if exit_on_reach and self.is_joint_pos_reached(
                target_joint_pos, **exit_on_reach_kwargs
            ):
                break

    def set_joint_pos_vel(
        self,
        joint_pos: Float[np.ndarray, " N"] | list[float] | dict[str, float],
        joint_vel: Float[np.ndarray, " N"] | list[float] | dict[str, float],
        relative: bool = False,
    ) -> None:
        """Controls the arm in joint position mode with a velocity feedforward term.

        Warning:
            The joint_vel parameter should be well-planned (such as from a trajectory planner).
            Sending poorly planned or inappropriate joint velocity commands can cause the robot
            to behave unexpectedly or potentially get damaged. Ensure velocity commands are
            smooth, within safe limits, and properly coordinated across all joints.

            Additionally, this command MUST be called at high frequency (e.g., 100Hz) to take
            effect properly. DO NOT call this function just once or at low frequency, as this
            can lead to unpredictable robot behavior.

        Args:
            joint_pos: Joint positions as either:
                - List of joint values [j1, j2, ..., j7]
                - Numpy array with shape (7,), in radians
                - Dictionary of joint names and position values
            joint_vel: Joint velocities as either:
                - List of joint values [v1, v2, ..., v7]
                - Numpy array with shape (7,), in radians/sec
                - Dictionary of joint names and velocity values

            relative: If True, the joint positions are relative to the current position.

        Raises:
            ValueError: If joint_pos dictionary contains invalid joint names.
        """
        resolved_joint_pos = (
            self._resolve_relative_joint_cmd(joint_pos) if relative else joint_pos
        )
        resolved_joint_pos = self._convert_joint_cmd_to_array(resolved_joint_pos)
        if self._joint_limit is not None:
            resolved_joint_pos = np.clip(
                resolved_joint_pos, self._joint_limit[:, 0], self._joint_limit[:, 1]
            )
        target_pos = resolved_joint_pos
        target_vel = self._convert_joint_cmd_to_array(
            joint_vel, clip_value=self._joint_vel_limit
        )

        control_msg = control_msg_pb2.MotorPosVelCurrentCommand(
            pos=list(target_pos),
            vel=list(target_vel),
        )
        self._publish_control(control_msg)

    def shutdown(self) -> None:
        """Cleans up all Zenoh resources."""
        super().shutdown()
        try:
            # Mode querier cleanup no longer needed with DexComm
            pass
        except Exception as e:
            # Don't log "Undeclared querier" errors as warnings - they're expected during shutdown
            error_msg = str(e).lower()
            if not ("undeclared" in error_msg or "closed" in error_msg):
                logger.warning(
                    f"Error undeclaring mode querier for {self.__class__.__name__}: {e}"
                )

        if self.wrench_sensor:
            self.wrench_sensor.shutdown()

    def send_ee_pass_through_message(self, message: bytes):
        """Send an end effector pass through message to the robot arm.

        Args:
            message: The message to send to the robot arm.
        """
        control_msg = control_msg_pb2.EndEffectorPassThroughCommand(data=message)
        self._ee_pass_through_publisher.publish(control_msg)


class ArmWrenchSensor(RobotComponent):
    """Wrench sensor reader for the robot arm.

    This class provides methods to read wrench sensor data through Zenoh communication.
    """

    def __init__(self, state_sub_topic: str) -> None:
        """Initialize the wrench sensor reader.

        Args:
            state_sub_topic: Topic to subscribe to for wrench sensor data.
        """
        super().__init__(
            state_sub_topic=state_sub_topic,
            state_message_type=control_msg_pb2.WrenchState,
        )

    def get_wrench_state(self) -> Float[np.ndarray, "6"]:
        """Get the current wrench sensor reading.

        Returns:
            Array of wrench values [fx, fy, fz, tx, ty, tz].
        """
        state = self._get_state()
        return np.array(state.wrench, dtype=np.float32)

    def get_button_state(self) -> tuple[bool, bool]:
        """Get the state of the wrench sensor buttons.

        Returns:
            Tuple of (blue_button_state, green_button_state).
        """
        state = self._get_state()
        return state.blue_button, state.green_button

    def get_state(self) -> dict[str, Float[np.ndarray, "6"] | bool]:
        """Get the complete wrench sensor state.

        Returns:
            Dictionary containing wrench values and button states.
        """
        state = self._get_state()
        return {
            "wrench": np.array(state.wrench, dtype=np.float32),
            "blue_button": state.blue_button,
            "green_button": state.green_button,
        }

    def get_blue_button_state(self) -> bool:
        """Get the state of the blue button.

        Returns:
            True if the blue button is pressed, False otherwise.
        """
        state = self._get_state()
        return state.blue_button

    def get_green_button_state(self) -> bool:
        """Get the state of the green button.

        Returns:
            True if the green button is pressed, False otherwise.
        """
        state = self._get_state()
        return state.green_button

# Copyright (C) 2025 Dexmate Inc.
#
# This software is dual-licensed:
#
# 1. GNU Affero General Public License v3.0 (AGPL-3.0)
#    See LICENSE-AGPL for details
#
# 2. Commercial License
#    For commercial licensing terms, contact: contact@dexmate.ai

"""Robot head control module.

This module provides the Head class for controlling a robot head through Zenoh
communication. It handles joint position and velocity control, mode setting, and
state monitoring.
"""

from typing import Literal

import numpy as np
from dexcomm import call_service
from dexcomm.serialization.protobuf import control_msg_pb2, control_query_pb2
from jaxtyping import Float

from dexcontrol.config.core import HeadConfig
from dexcontrol.core.component import RobotJointComponent
from dexcontrol.utils.comm_helper import get_zenoh_config_path
from dexcontrol.utils.os_utils import resolve_key_name


class Head(RobotJointComponent):
    """Robot head control class.

    This class provides methods to control a robot head by publishing commands and
    receiving state information through Zenoh communication.

    Attributes:
        mode_querier: Zenoh querier for setting head mode.
        default_vel: Default joint velocities for all joints.
        max_vel: Maximum allowed joint velocities for all joints.
    """

    def __init__(
        self,
        configs: HeadConfig,
    ) -> None:
        """Initialize the head controller.

        Args:
            configs: Configuration parameters for the head including communication topics.
        """
        super().__init__(
            state_sub_topic=configs.state_sub_topic,
            control_pub_topic=configs.control_pub_topic,
            state_message_type=control_msg_pb2.MotorStateWithTorque,
            joint_name=configs.joint_name,
            joint_limit=configs.joint_limit
            if hasattr(configs, "joint_limit")
            else None,
            joint_vel_limit=configs.joint_vel_limit
            if hasattr(configs, "joint_vel_limit")
            else None,
            pose_pool=configs.pose_pool,
        )

        # Store the query topic for later use with DexComm
        self._mode_query_topic = resolve_key_name(configs.set_mode_query)
        assert self._joint_vel_limit is not None, "joint_vel_limit is not set"

    def set_joint_pos(
        self,
        joint_pos: Float[np.ndarray, "3"] | list[float] | dict[str, float],
        relative: bool = False,
        wait_time: float = 0.0,
        wait_kwargs: dict[str, float] | None = None,
        exit_on_reach: bool = False,
        exit_on_reach_kwargs: dict[str, float] | None = None,
    ) -> None:
        """Send joint position control commands to the head.

        Args:
            joint_pos: Joint positions as either:
                - List of joint values [j1, j2]
                - Numpy array with shape (3,), in radians
                - Dictionary mapping joint names to position values
            relative: If True, the joint positions are relative to the current position.
            wait_time: Time to wait after sending command in seconds. If 0, returns
                immediately after sending command.
            wait_kwargs: Optional parameters for trajectory generation (not used in Head).
            exit_on_reach: If True, the function will exit when the joint positions are reached.
            exit_on_reach_kwargs: Optional parameters for exit when the joint positions are reached.

        Raises:
            ValueError: If joint_pos dictionary contains invalid joint names.
        """
        self.set_joint_pos_vel(
            joint_pos,
            joint_vel=None,
            relative=relative,
            wait_time=wait_time,
            exit_on_reach=exit_on_reach,
            exit_on_reach_kwargs=exit_on_reach_kwargs,
        )

    def set_joint_pos_vel(
        self,
        joint_pos: Float[np.ndarray, "3"] | list[float] | dict[str, float],
        joint_vel: Float[np.ndarray, "3"]
        | list[float]
        | dict[str, float]
        | float
        | None = None,
        relative: bool = False,
        wait_time: float = 0.0,
        exit_on_reach: bool = False,
        exit_on_reach_kwargs: dict[str, float] | None = None,
    ) -> None:
        """Send control commands to the head.

        Args:
            joint_pos: Joint positions as either:
                - List of joint values [j1, j2]
                - Numpy array with shape (3,), in radians
                - Dictionary mapping joint names to position values
            joint_vel: Optional joint velocities as either:
                - List of joint values [v1, v2]
                - Numpy array with shape (3,), in rad/s
                - Dictionary mapping joint names to velocity values
                - Single float value to be applied to all joints
                If None, velocities are calculated based on default velocity setting.
            relative: If True, the joint positions are relative to the current position.
            wait_time: Time to wait after sending command in seconds. If 0, returns
                immediately after sending command.
            exit_on_reach: If True, the function will exit when the joint positions are reached.
            exit_on_reach_kwargs: Optional parameters for exit when the joint positions are reached.

        Raises:
            ValueError: If wait_time is negative or joint_pos dictionary contains
                invalid joint names.
        """
        if wait_time < 0.0:
            raise ValueError("wait_time must be greater than or equal to 0")

        # Handle relative positioning
        if relative:
            joint_pos = self._resolve_relative_joint_cmd(joint_pos)

        # Convert inputs to numpy arrays
        joint_pos = self._convert_joint_cmd_to_array(joint_pos)
        joint_vel = self._process_joint_velocities(joint_vel, joint_pos)

        if self._joint_limit is not None:
            joint_pos = np.clip(
                joint_pos, self._joint_limit[:, 0], self._joint_limit[:, 1]
            )
        if self._joint_vel_limit is not None:
            joint_vel = np.clip(
                joint_vel, -self._joint_vel_limit, self._joint_vel_limit
            )

        # Create and send control message
        control_msg = control_msg_pb2.MotorPosVelCommand()
        control_msg.pos.extend(joint_pos.tolist())
        control_msg.vel.extend(joint_vel.tolist())
        self._publish_control(control_msg)

        # Wait if specified
        self._wait_for_position(
            joint_pos=joint_pos,
            wait_time=wait_time,
            exit_on_reach=exit_on_reach,
            exit_on_reach_kwargs=exit_on_reach_kwargs,
        )

    def set_mode(self, mode: Literal["enable", "disable"]) -> None:
        """Set the operating mode of the head.

        Args:
            mode: Operating mode for the head. Must be either "enable" or "disable".

        Raises:
            ValueError: If an invalid mode is specified.
        """
        mode_map = {
            "enable": control_query_pb2.SetHeadMode.Mode.ENABLE,
            "disable": control_query_pb2.SetHeadMode.Mode.DISABLE,
        }

        if mode not in mode_map:
            raise ValueError(
                f"Invalid mode: {mode}. Must be one of {list(mode_map.keys())}"
            )

        query_msg = control_query_pb2.SetHeadMode()
        query_msg.mode = mode_map[mode]

        call_service(
            self._mode_query_topic,
            request=query_msg,
            timeout=5.0,
            config=get_zenoh_config_path(),
            request_serializer=lambda x: x.SerializeToString(),
            response_deserializer=None,
        )

    def get_joint_limit(self) -> Float[np.ndarray, "3 2"] | None:
        """Get the joint limits of the head.

        Returns:
            Array of joint limits with shape (3, 2), where the first column contains
            lower limits and the second column contains upper limits, or None if not configured.
        """
        return self._joint_limit

    def stop(self) -> None:
        """Stop the head by setting target position to current position with zero velocity."""
        current_pos = self.get_joint_pos()
        zero_vel = np.zeros(3, dtype=np.float32)
        self.set_joint_pos_vel(current_pos, zero_vel, relative=False, wait_time=0.0)

    def shutdown(self) -> None:
        """Clean up Zenoh resources for the head component."""
        self.stop()
        super().shutdown()
        # No need to undeclare queriers when using DexComm

    def _process_joint_velocities(
        self,
        joint_vel: Float[np.ndarray, "3"]
        | list[float]
        | dict[str, float]
        | float
        | None,
        joint_pos: np.ndarray,
    ) -> np.ndarray:
        """Process and validate joint velocities.

        Args:
            joint_vel: Joint velocities in various formats or None.
            joint_pos: Target joint positions for velocity calculation.

        Returns:
            Processed joint velocities as numpy array.
        """
        if joint_vel is None:
            # Calculate velocities based on motion direction and default velocity
            joint_motion = joint_pos - self.get_joint_pos()
            motion_norm = np.linalg.norm(joint_motion)

            if motion_norm < 1e-6:  # Avoid division by zero
                return np.zeros(3, dtype=np.float32)

            default_vel = (
                2.5 if self._joint_vel_limit is None else np.min(self._joint_vel_limit)
            )
            return (joint_motion / motion_norm) * default_vel

        if isinstance(joint_vel, (int, float)):
            # Single value - apply to all joints
            return np.full(3, joint_vel, dtype=np.float32)

        # Convert to array and clip to velocity limits
        return self._convert_joint_cmd_to_array(
            joint_vel, clip_value=self._joint_vel_limit
        )

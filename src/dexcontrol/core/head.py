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

from typing import Literal, cast

import numpy as np
from dexbot_utils import RobotInfo
from dexbot_utils.configs.components.vega_1 import Vega1HeadConfig
from dexcomm.codecs import (
    DictDataCodec,
    JointCmdCodec,
    JointModeCodec,
    JointModeEnum,
    JointStateCodec,
)
from jaxtyping import Float
from loguru import logger

from dexcontrol.core.component import RobotJointComponent


class Head(RobotJointComponent):
    """Robot head control class.

    Provides methods to control a 3-DOF robot head by publishing joint commands
    and receiving state information through Zenoh communication.  Supports
    position and velocity control, mode switching (enable/disable), and
    graceful stop/shutdown.
    """

    def __init__(
        self,
        name: str,
        robot_info: RobotInfo,
    ) -> None:
        """Initialize the head controller.

        Args:
            name: Component name used to look up joints and config from
                ``robot_info``.
            robot_info: RobotInfo instance that provides joint names, limits,
                and the component configuration.
        """
        joint_names = robot_info.get_component_joints(name)
        joint_pos_limits = robot_info.get_joint_pos_limits(joint_names)
        joint_vel_limits = robot_info.get_joint_vel_limits(joint_names)
        config = robot_info.get_component_config(name)
        config = cast(Vega1HeadConfig, config)
        super().__init__(
            name=name,
            state_sub_topic=config.state_sub_topic,
            control_pub_topic=config.control_pub_topic,
            state_decoder=JointStateCodec.decode,
            control_encoder=JointCmdCodec.encode,
            joint_name=joint_names,
            joint_pos_limit=joint_pos_limits,
            joint_vel_limit=joint_vel_limits,
            pose_pool=config.pose_pool,
        )

        # Store the query topic for later use with DexComm
        self._mode_querier = self._node.create_service_client(
            service_name=config.set_mode_query,
            request_encoder=JointModeCodec.encode,
            response_decoder=DictDataCodec.decode,
            timeout=5.0,
        )
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

        Velocities are computed automatically from the joint velocity limits.

        Args:
            joint_pos: Target joint positions as either:
                - A list of 3 floats ``[j1, j2, j3]``, in radians
                - A numpy array with shape ``(3,)``, in radians
                - A dict mapping joint names to position values
            relative: If ``True``, ``joint_pos`` is interpreted as an offset
                from the current joint positions.
            wait_time: Time to wait after sending the command, in seconds.
                Pass ``0.0`` to return immediately.
            wait_kwargs: Accepted for API compatibility; not used by ``Head``.
            exit_on_reach: If ``True``, return as soon as the target positions
                are reached (subject to ``exit_on_reach_kwargs`` tolerance),
                rather than waiting the full ``wait_time``.
            exit_on_reach_kwargs: Optional keyword arguments forwarded to
                ``is_joint_pos_reached`` (e.g. ``{"tolerance": 0.02}``).

        Raises:
            ValueError: If ``wait_time`` is negative.
            ValueError: If ``joint_pos`` is a dict with invalid joint names.
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
        """Send joint position and velocity control commands to the head.

        Args:
            joint_pos: Target joint positions as either:
                - A list of 3 floats ``[j1, j2, j3]``, in radians
                - A numpy array with shape ``(3,)``, in radians
                - A dict mapping joint names to position values
            joint_vel: Target joint velocities as either:
                - A list of 3 floats ``[v1, v2, v3]``, in rad/s
                - A numpy array with shape ``(3,)``, in rad/s
                - A dict mapping joint names to velocity values
                - A single float applied uniformly to all joints
                If ``None``, velocities are derived from the joint velocity limits
                and the direction of motion.
            relative: If ``True``, ``joint_pos`` is interpreted as an offset
                from the current joint positions.
            wait_time: Time to wait after sending the command, in seconds.
                Must be ``>= 0``. Pass ``0.0`` to return immediately.
            exit_on_reach: If ``True``, return as soon as the target positions
                are reached (subject to ``exit_on_reach_kwargs`` tolerance),
                rather than waiting the full ``wait_time``.
            exit_on_reach_kwargs: Optional keyword arguments forwarded to
                ``is_joint_pos_reached`` (e.g. ``{"tolerance": 0.02}``).

        Raises:
            ValueError: If ``wait_time`` is negative.
            ValueError: If ``joint_pos`` is a dict with invalid joint names.
        """
        if wait_time < 0.0:
            raise ValueError("wait_time must be greater than or equal to 0")

        # Handle relative positioning
        if relative:
            joint_pos = self._resolve_relative_joint_cmd(joint_pos)

        # Convert inputs to numpy arrays
        joint_pos = self._convert_joint_cmd_to_array(joint_pos)
        joint_vel = self._process_joint_velocities(joint_vel, joint_pos)

        if self._joint_pos_limit is not None:
            joint_pos = np.clip(
                joint_pos, self._joint_pos_limit[:, 0], self._joint_pos_limit[:, 1]
            )
        if self._joint_vel_limit is not None:
            joint_vel = np.clip(
                joint_vel, -self._joint_vel_limit, self._joint_vel_limit
            )

        # Create and send control message
        data = dict(pos=joint_pos, vel=joint_vel)
        self._publish_control(control_msg=data)

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
            "enable": JointModeEnum.ENABLE,
            "disable": JointModeEnum.DISABLE,
        }

        if mode not in mode_map:
            raise ValueError(
                f"Invalid mode: {mode}. Must be one of {list(mode_map.keys())}"
            )

        query_msg = {"mode": [mode_map[mode]] * 3}  # 3 head joints

        # Wait for service to be available before calling
        if not self._mode_querier.wait_for_service(timeout=5.0):
            logger.warning(
                f"{self._node.get_name()}: Mode service not available, command may fail"
            )

        self._mode_querier.call(query_msg)

    def get_joint_pos_limit(self) -> Float[np.ndarray, "3 2"] | None:
        """Get the joint position limits of the head.

        Returns:
            Array of joint position limits with shape (3, 2), where the first column contains
            lower limits and the second column contains upper limits, or None if not configured.
        """
        return self._joint_pos_limit

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

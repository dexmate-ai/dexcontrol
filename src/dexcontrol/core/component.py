# Copyright (C) 2025 Dexmate Inc.
#
# This software is dual-licensed:
#
# 1. GNU Affero General Public License v3.0 (AGPL-3.0)
#    See LICENSE-AGPL for details
#
# 2. Commercial License
#    For commercial licensing terms, contact: contact@dexmate.ai

"""Base module for robot components using DexComm communication.

This module provides base classes for robot components that use DexComm's
Raw API for communication. It includes RobotComponent for state-only components
and RobotJointComponent for components that also support control commands.
"""

import time
from typing import Any, Final, Mapping, TypeVar

import numpy as np
from dexcomm import Publisher, Subscriber
from dexcomm.serialization import deserialize_protobuf, serialize_protobuf
from google.protobuf.message import Message
from jaxtyping import Float
from loguru import logger

from dexcontrol.utils.comm_helper import get_zenoh_config_path
from dexcontrol.utils.os_utils import resolve_key_name

# Type variable for Message subclasses
M = TypeVar("M", bound=Message)


class RobotComponent:
    """Base class for robot components with state interface.

    A component represents a physical part of the robot that maintains state through
    Zenoh communication. It subscribes to state updates and provides methods to
    access the latest state data.

    Attributes:
        _state_message_type: Protobuf message class for component state.
        _zenoh_session: Active Zenoh session for communication.
        _subscriber: Zenoh subscriber for state updates.
    """

    def __init__(
        self,
        state_sub_topic: str,
        state_message_type: type[M],
    ) -> None:
        """Initializes RobotComponent.

        Args:
            state_sub_topic: Topic to subscribe to for state updates.
            state_message_type: Protobuf message class for component state.
        """
        self._state_message_type = state_message_type
        self._latest_state: M | None = None
        self._is_active = False
        self._init_subscriber(state_sub_topic, state_message_type)

    def _init_subscriber(
        self,
        state_sub_topic: str,
        state_message_type: type[M],
    ) -> None:
        """Initialize the subscriber for state updates using DexComm.

        Args:
            state_sub_topic: Topic to subscribe to for state updates.
            state_message_type: Protobuf message class for component state.
        """
        # Resolve topic with robot namespace
        full_topic = resolve_key_name(state_sub_topic)

        # Create DexComm subscriber with protobuf deserialization
        self._subscriber = Subscriber(
            topic=full_topic,
            callback=self._on_state_update,
            deserializer=lambda data: deserialize_protobuf(data, state_message_type),
            config=get_zenoh_config_path(),
        )

        logger.debug(
            f"Created subscriber for {self.__class__.__name__} on {full_topic}"
        )

    def _on_state_update(self, state: M) -> None:
        """Handle incoming state updates.

        Args:
            state: Deserialized protobuf state message.
        """
        self._latest_state = state
        if state:
            self._is_active = True

    def _get_state(self) -> M:
        """Gets the current state of the component.

        Returns:
            Parsed protobuf state message.

        Raises:
            None: If no state data is available.
        """
        if self._latest_state is None:
            logger.error(f"No state data available for {self.__class__.__name__}")
        return self._latest_state

    def wait_for_active(self, timeout: float = 5.0) -> bool:
        """Waits for the component to start receiving state updates.

        Args:
            timeout: Maximum time to wait in seconds.

        Returns:
            True if component becomes active, False if timeout is reached.
        """
        start_time = time.time()
        while time.time() - start_time < timeout:
            if self._is_active:
                return True
            time.sleep(0.1)
        return False

    def is_active(self) -> bool:
        """Check if component is receiving state updates.

        Returns:
            True if component is active, False otherwise.
        """
        return self._is_active

    def shutdown(self) -> None:
        """Cleans up Zenoh resources."""
        # Stop any ongoing operations if the component has a stop method
        if hasattr(self, "stop"):
            method = getattr(self, "stop")
            if callable(method):
                try:
                    method()
                except Exception as e:
                    # During shutdown, stop() methods may fail due to inactive subscribers
                    logger.debug(
                        f"Error during stop() for {self.__class__.__name__}: {e}"
                    )

        # Shutdown subscriber to release resources
        if hasattr(self, "_subscriber") and self._subscriber:
            self._subscriber.shutdown()

    def get_timestamp_ns(self) -> int:
        """Get the current timestamp (in nanoseconds) of the most recent state update.

        Returns:
            The current timestamp in nanoseconds when driver updated the state.
            We convert the time to client clock by adding the server time offset.
        """
        return self._get_state().timestamp_ns


class RobotJointComponent(RobotComponent):
    """Base class for robot components with both state and control interfaces.

    Extends RobotComponent to add APIs for interacting with joints.

    Attributes:
        _publisher: Publisher for control commands (Zenoh or dexcomm).
        _joint_name: List of joint names for this component.
        _pose_pool: Dictionary of predefined poses for this component.
    """

    @staticmethod
    def _convert_pose_pool_to_arrays(
        pose_pool: Mapping[str, list[float] | np.ndarray] | None = None,
    ) -> dict[str, np.ndarray] | None:
        """Convert pose pool values to numpy arrays.

        Args:
            pose_pool: Dictionary mapping pose names to lists or arrays of joint values.

        Returns:
            Dictionary mapping pose names to numpy arrays, or None if input is None.
        """
        if pose_pool is None:
            return None

        return {
            name: np.array(pose, dtype=np.float32) for name, pose in pose_pool.items()
        }

    def __init__(
        self,
        state_sub_topic: str,
        control_pub_topic: str,
        state_message_type: type[M],
        joint_name: list[str] | None = None,
        joint_limit: list[list[float]] | None = None,
        joint_vel_limit: list[float] | None = None,
        pose_pool: Mapping[str, list[float] | np.ndarray] | None = None,
    ) -> None:
        """Initializes RobotJointComponent.

        Args:
            state_sub_topic: Topic to subscribe to for state updates.
            control_pub_topic: Topic to publish control commands.
            state_message_type: Protobuf message class for component state.
            joint_name: List of joint names for this component.
            joint_limit: Joint position limits.
            joint_vel_limit: Joint velocity limits.
            pose_pool: Dictionary of predefined poses for this component.
        """
        super().__init__(state_sub_topic, state_message_type)

        # Resolve topic with robot namespace
        full_topic = resolve_key_name(control_pub_topic)

        # Create DexComm publisher with protobuf serialization
        self._publisher: Final[Publisher] = Publisher(
            topic=full_topic,
            serializer=serialize_protobuf,
            config=get_zenoh_config_path(),
        )

        logger.debug(f"Created publisher for {self.__class__.__name__} on {full_topic}")
        self._joint_name: list[str] | None = joint_name
        self._joint_limit: np.ndarray | None = (
            np.array(joint_limit) if joint_limit else None
        )
        self._joint_vel_limit: np.ndarray | None = (
            np.array(joint_vel_limit) if joint_vel_limit else None
        )

        self._pose_pool: dict[str, np.ndarray] | None = (
            self._convert_pose_pool_to_arrays(pose_pool)
        )

    def _publish_control(self, control_msg: Message) -> None:
        """Publishes a control command message.

        Args:
            control_msg: Protobuf control message to publish.
        """
        # DexComm publisher with protobuf serializer handles this
        self._publisher.publish(control_msg)

    def shutdown(self) -> None:
        """Cleans up all communication resources."""
        super().shutdown()
        try:
            if hasattr(self, "_publisher") and self._publisher:
                self._publisher.shutdown()
        except Exception as e:
            logger.warning(
                f"Error shutting down publisher for {self.__class__.__name__}: {e}"
            )

    @property
    def joint_name(self) -> list[str]:
        """Gets the joint names of the component.

        Returns:
            List of joint names.

        Raises:
            ValueError: If joint names are not available.
        """
        if self._joint_name is None:
            raise ValueError("Joint names not available for this component")
        return self._joint_name.copy()

    @property
    def joint_limit(self) -> np.ndarray | None:
        """Gets the joint limits of the component."""
        return self._joint_limit.copy() if self._joint_limit is not None else None

    @property
    def joint_vel_limit(self) -> np.ndarray | None:
        """Gets the joint velocity limits of the component."""
        return (
            self._joint_vel_limit.copy() if self._joint_vel_limit is not None else None
        )

    def get_predefined_pose(self, pose_name: str) -> np.ndarray:
        """Gets a predefined pose from the pose pool.

        Args:
            pose_name: Name of the pose to get.

        Returns:
            The joint positions for the requested pose.

        Raises:
            ValueError: If pose pool is not available or pose name is invalid.
        """
        if self._pose_pool is None:
            raise ValueError("Pose pool not available for this component.")
        if pose_name not in self._pose_pool:
            available_poses = list(self._pose_pool.keys())
            raise ValueError(
                f"Invalid pose name: {pose_name}. Available poses: {available_poses}"
            )
        return np.array(self._pose_pool[pose_name], dtype=float).copy()

    def get_joint_name(self) -> list[str]:
        """Gets the joint names of the component.

        Returns:
            List of joint names.

        Raises:
            ValueError: If joint names are not available.
        """
        return self.joint_name

    def get_joint_pos(
        self, joint_id: list[int] | int | None = None
    ) -> Float[np.ndarray, " N"]:
        """Gets the current positions of all joints in the component.

        The returned array contains joint positions in the same order as joint_id.

        Args:
            joint_id: Optional ID(s) of specific joints to query.

        Returns:
            Array of joint positions in component-specific units (radians for
            revolute joints and meters for prismatic joints).

        Raises:
            ValueError: If joint positions are not available for this component.
        """
        state = self._get_state()
        if not hasattr(state, "pos"):
            raise ValueError("Joint positions are not available for this component.")
        joint_pos = np.array(state.pos, dtype=np.float32)
        return self._extract_joint_info(joint_pos, joint_id=joint_id)

    def get_joint_pos_dict(
        self, joint_id: list[int] | int | None = None
    ) -> dict[str, float]:
        """Gets the current positions of all joints in the component as a dictionary.

        Args:
            joint_id: Optional ID(s) of specific joints to query.

        Returns:
            Dictionary mapping joint names to position values.

        Raises:
            ValueError: If joint positions are not available for this component.
        """
        values = self.get_joint_pos(joint_id)
        return self._convert_to_dict(values, joint_id)

    def get_joint_vel(
        self, joint_id: list[int] | int | None = None
    ) -> Float[np.ndarray, " N"]:
        """Gets the current velocities of all joints in the component.

        Args:
            joint_id: Optional ID(s) of specific joints to query.

        Returns:
            Array of joint velocities in component-specific units (radians/s for
            revolute joints and meters/s for prismatic joints).

        Raises:
            ValueError: If joint velocities are not available for this component.
        """
        state = self._get_state()
        if not hasattr(state, "vel"):
            raise ValueError("Joint velocities are not available for this component.")
        joint_vel = np.array(state.vel, dtype=np.float32)
        return self._extract_joint_info(joint_vel, joint_id=joint_id)

    def get_joint_vel_dict(
        self, joint_id: list[int] | int | None = None
    ) -> dict[str, float]:
        """Gets the current velocities of all joints in the component as a dictionary.

        Args:
            joint_id: Optional ID(s) of specific joints to query.

        Returns:
            Dictionary mapping joint names to velocity values.

        Raises:
            ValueError: If joint velocities are not available for this component.
        """
        values = self.get_joint_vel(joint_id)
        return self._convert_to_dict(values, joint_id)

    def get_joint_current(
        self, joint_id: list[int] | int | None = None
    ) -> Float[np.ndarray, " N"]:
        """Gets the current of all joints in the component.

        Args:
            joint_id: Optional ID(s) of specific joints to query.

        Returns:
            Array of joint currents in component-specific units (amperes).

        Raises:
            ValueError: If joint currents are not available for this component.
        """
        state = self._get_state()
        if not hasattr(state, "cur"):
            raise ValueError("Joint currents are not available for this component.")
        joint_cur = np.array(state.cur, dtype=np.float32)
        return self._extract_joint_info(joint_cur, joint_id=joint_id)

    def get_joint_torque(
        self, joint_id: list[int] | int | None = None
    ) -> Float[np.ndarray, " N"]:
        """Gets the torque of all joints in the component.

        Args:
            joint_id: Optional ID(s) of specific joints to query.

        Returns:
            Array of joint torques in component-specific units (Nm).

        Raises:
            ValueError: If joint torques are not available for this component.
        """
        state = self._get_state()
        if not hasattr(state, "torque"):
            raise ValueError("Joint torques are not available for this component.")
        joint_torque = np.array(state.torque, dtype=np.float32)
        return self._extract_joint_info(joint_torque, joint_id=joint_id)

    def get_joint_current_dict(
        self, joint_id: list[int] | int | None = None
    ) -> dict[str, float]:
        """Gets the current of all joints in the component as a dictionary.

        Args:
            joint_id: Optional ID(s) of specific joints to query.

        Returns:
            Dictionary mapping joint names to current values.

        Raises:
            ValueError: If joint currents are not available for this component.
        """
        values = self.get_joint_current(joint_id)
        return self._convert_to_dict(values, joint_id)

    def get_joint_err(self, joint_id: list[int] | int | None = None) -> np.ndarray:
        """Gets current joint error codes.

        Args:
            joint_id: Optional ID(s) of specific joints to query.

        Returns:
            Array of joint error codes.

        Raises:
            ValueError: If joint error codes are not available for this component.
        """
        state = self._get_state()
        if not hasattr(state, "error"):
            raise ValueError("Joint error codes are not available for this component.")
        joint_err = np.array(state.error, dtype=np.uint32)
        return self._extract_joint_info(joint_err, joint_id=joint_id)

    def get_joint_err_dict(
        self, joint_id: list[int] | int | None = None
    ) -> dict[str, int]:
        """Gets current joint error codes as a dictionary.

        Args:
            joint_id: Optional ID(s) of specific joints to query.

        Returns:
            Dictionary mapping joint names to error code values.

        Raises:
            ValueError: If joint error codes are not available for this component.
        """
        values = self.get_joint_err(joint_id)
        return self._convert_to_dict(values, joint_id)

    def get_joint_state(self, joint_id: list[int] | int | None = None) -> np.ndarray:
        """Gets current joint states including positions, velocities and currents.

        Args:
            joint_id: Optional ID(s) of specific joints to query.

        Returns:
            Array of joint positions, velocities, and currents.
            The last dimension corresponds to [positions, velocities, currents].

        Raises:
            ValueError: If joint positions or velocities are not available.
        """
        state = self._get_state()
        if not hasattr(state, "pos") or not hasattr(state, "vel"):
            raise ValueError(
                "Joint positions or velocities are not available for this component."
            )

        # Create initial state array with positions and velocities
        joint_pos = np.array(state.pos, dtype=np.float32)
        joint_vel = np.array(state.vel, dtype=np.float32)

        if hasattr(state, "cur"):
            # If currents are available, include them
            joint_cur = np.array(state.cur, dtype=np.float32)
            joint_state = np.stack([joint_pos, joint_vel, joint_cur], axis=1)
        elif hasattr(state, "torque"):
            # If torques are available, include them
            joint_torque = np.array(state.torque, dtype=np.float32)
            joint_state = np.stack([joint_pos, joint_vel, joint_torque], axis=1)
        else:
            raise ValueError(
                f"Either current or torque should be available for this {self.__class__.__name__}."
            )

        return self._extract_joint_info(joint_state, joint_id=joint_id)

    def get_joint_state_dict(
        self, joint_id: list[int] | int | None = None
    ) -> dict[str, Float[np.ndarray, "3"]]:
        """Gets current joint states including positions, velocities and currents as a dictionary.

        Args:
            joint_id: Optional ID(s) of specific joints to query.

        Returns:
            Dictionary mapping joint names to arrays of [position, velocity, current].

        Raises:
            ValueError: If joint positions or velocities are not available.
        """
        values = self.get_joint_state(joint_id)
        return self._convert_to_dict(values, joint_id)

    def _convert_joint_cmd_to_array(
        self,
        joint_cmd: Float[np.ndarray, " N"] | list[float] | dict[str, float],
        clip_value: float | np.ndarray | None = None,
    ) -> np.ndarray:
        """Convert joint command to numpy array format.

        Args:
            joint_cmd: Joint command as either:
                - List of joint values [j1, j2, ..., jN]
                - Numpy array with shape (N,)
                - Dictionary mapping joint names to values
            clip_value: Optional value to clip the output array. Can be:
                - float: symmetric clipping between [-clip_value, clip_value]
                - numpy array: element-wise clipping between [-clip_value, clip_value]

        Returns:
            Joint command as numpy array.
        """
        if isinstance(joint_cmd, dict):
            joint_cmd = self._convert_dict_to_array(joint_cmd)
        elif isinstance(joint_cmd, list):
            joint_cmd = np.array(joint_cmd, dtype=np.float32)
        else:
            joint_cmd = joint_cmd.astype(np.float32)

        if clip_value is not None:
            joint_cmd = np.clip(joint_cmd, -clip_value, clip_value)

        return joint_cmd

    def _resolve_relative_joint_cmd(
        self, joint_cmd: Float[np.ndarray, " N"] | list[float] | dict[str, float]
    ) -> Float[np.ndarray, " N"] | dict[str, float]:
        """Resolve relative joint command by adding current joint positions.

        Args:
            joint_cmd: Relative joint command as list, numpy array, or dictionary.

        Returns:
            Absolute joint command in the same format as input.
        """
        if isinstance(joint_cmd, dict):
            current_pos = self.get_joint_pos_dict()
            return {name: current_pos[name] + pos for name, pos in joint_cmd.items()}

        # Convert list to numpy array if needed
        joint_cmd = self._convert_joint_cmd_to_array(joint_cmd)
        return self.get_joint_pos() + joint_cmd

    @staticmethod
    def _extract_joint_info(
        joint_info: np.ndarray, joint_id: list[int] | int | None = None
    ) -> np.ndarray:
        """Extract the joint information of the component as a numpy array.

        Args:
            joint_info: Array of joint information.
            joint_id: Optional ID(s) of specific joints to extract.

        Returns:
            Array of joint information.

        Raises:
            ValueError: If an invalid joint ID is provided.
        """
        if joint_id is None:
            return joint_info

        if isinstance(joint_id, int):
            if joint_id >= len(joint_info):
                raise ValueError(
                    f"Invalid joint ID: {joint_id}. Must be less than {len(joint_info)}"
                )
            return joint_info[joint_id]

        # joint_id is a list
        if max(joint_id) >= len(joint_info):
            raise ValueError(
                f"Invalid joint ID in {joint_id}. Must be less than {len(joint_info)}"
            )
        return joint_info[joint_id]

    def _convert_to_dict(
        self, values: np.ndarray, joint_id: list[int] | int | None = None
    ) -> dict[str, Any]:
        """Convert a numpy array of joint values to a dictionary of joint names and values.

        Args:
            values: Array of joint values.
            joint_id: Optional ID(s) of specific joints for the output.

        Returns:
            Dictionary of joint names and values.

        Raises:
            ValueError: If joint names are not available for this component.
        """
        if self._joint_name is None:
            raise ValueError("Joint names not available for this component.")

        if joint_id is None:
            joint_id = list(range(len(self._joint_name)))
        elif isinstance(joint_id, int):
            joint_id = [joint_id]

        if len(values.shape) == 1:
            return {
                self._joint_name[id]: float(value)
                for id, value in zip(joint_id, values)
            }
        else:
            return {self._joint_name[id]: values[i] for i, id in enumerate(joint_id)}

    def _get_joint_index(self, joint_name: list[str] | str) -> list[int] | int:
        """Get the indices of the specified joints.

        Args:
            joint_name: Name(s) of the joints to get indices for.

        Returns:
            List of indices or single index corresponding to the requested joints.

        Raises:
            ValueError: If joint names are not available or if an invalid joint name is provided.
        """
        if self._joint_name is None:
            raise ValueError("Joint names not available for this component.")

        if isinstance(joint_name, str):
            try:
                return self._joint_name.index(joint_name)
            except ValueError:
                raise ValueError(
                    f"Invalid joint name: {joint_name}. Available joints: {self._joint_name}"
                )

        # joint_name is a list
        indices = []
        for name in joint_name:
            try:
                indices.append(self._joint_name.index(name))
            except ValueError:
                raise ValueError(
                    f"Invalid joint name: {name}. Available joints: {self._joint_name}"
                )
        return indices

    def _convert_dict_to_array(
        self, joint_pos_dict: dict[str, float]
    ) -> Float[np.ndarray, " N"]:
        """Convert joint position dictionary to array format.

        Args:
            joint_pos_dict: Dictionary mapping joint names to position values.

        Returns:
            Array of joint positions in the correct order.

        Raises:
            ValueError: If joint_pos_dict contains invalid joint names.
        """
        current_joint_pos = self.get_joint_pos().copy()
        target_joint_names = list(joint_pos_dict.keys())
        target_joint_indices = self._get_joint_index(target_joint_names)
        current_joint_pos[target_joint_indices] = list(joint_pos_dict.values())
        return current_joint_pos

    def set_joint_pos(
        self,
        joint_pos: Float[np.ndarray, " N"] | list[float] | dict[str, float],
        relative: bool = False,
        wait_time: float = 0.0,
        wait_kwargs: dict[str, float] | None = None,
        exit_on_reach: bool = False,
        exit_on_reach_kwargs: dict[str, Any] | None = None,
    ) -> None:
        """Send joint position control commands.

        Args:
            joint_pos: Joint positions as either:
                - List of joint values [j1, j2, ..., jN]
                - Numpy array with shape (N,)
                - Dictionary mapping joint names to position values
            relative: If True, the joint positions are relative to the current position.
            wait_time: Time to wait after sending command in seconds.
            wait_kwargs: Optional parameters for trajectory generation.
            exit_on_reach: If True, the function will exit when the joint positions are reached.
            exit_on_reach_kwargs: Optional parameters for exit when the joint positions are reached.

        Raises:
            ValueError: If joint_pos dictionary contains invalid joint names.
        """
        if relative:
            joint_pos = self._resolve_relative_joint_cmd(joint_pos)

        # Convert to array format
        if isinstance(joint_pos, (list, dict)):
            joint_pos = self._convert_joint_cmd_to_array(joint_pos)

        if self._joint_limit is not None:
            joint_pos = np.clip(
                joint_pos, self._joint_limit[:, 0], self._joint_limit[:, 1]
            )

        self._send_position_command(joint_pos)

        if wait_time > 0.0:
            self._wait_for_position(
                joint_pos, wait_time, exit_on_reach, exit_on_reach_kwargs
            )

    def _wait_for_position(
        self,
        joint_pos: Float[np.ndarray, " N"] | list[float] | dict[str, float],
        wait_time: float,
        exit_on_reach: bool = False,
        exit_on_reach_kwargs: dict[str, Any] | None = None,
    ) -> None:
        """Wait for a specified time with optional early exit when position is reached.

        Args:
            joint_pos: Target joint positions to check against.
            wait_time: Maximum time to wait in seconds.
            exit_on_reach: If True, exit early when joint positions are reached.
            exit_on_reach_kwargs: Optional parameters for position checking.
        """
        if exit_on_reach:
            # Set default tolerance if not provided
            exit_on_reach_kwargs = exit_on_reach_kwargs or {}
            exit_on_reach_kwargs.setdefault("tolerance", 0.05)

            # Convert to expected format for is_joint_pos_reached
            if isinstance(joint_pos, list):
                joint_pos = np.array(joint_pos, dtype=np.float32)

            # Wait until position is reached or timeout
            start_time = time.time()
            while time.time() - start_time < wait_time:
                if self.is_joint_pos_reached(joint_pos, **exit_on_reach_kwargs):
                    break
                time.sleep(0.01)
        else:
            time.sleep(wait_time)

    def _send_position_command(self, joint_pos: Float[np.ndarray, " N"]) -> None:
        """Send joint position command to the component.

        This method should be overridden by child classes to implement
        component-specific command message creation and publishing.

        Args:
            joint_pos: Joint positions as numpy array.

        Raises:
            NotImplementedError: If child class does not implement this method.
        """
        raise NotImplementedError("Child class must implement _send_position_command")

    def go_to_pose(
        self,
        pose_name: str,
        wait_time: float = 3.0,
        exit_on_reach: bool = False,
        exit_on_reach_kwargs: dict[str, float] | None = None,
    ) -> None:
        """Move the component to a predefined pose.

        Args:
            pose_name: Name of the pose to move to.
            wait_time: Time to wait for the component to reach the pose.
            exit_on_reach: If True, the function will exit when the joint positions are reached.
            exit_on_reach_kwargs: Optional parameters for exit when the joint positions are reached.

        Raises:
            ValueError: If pose pool is not available or if an invalid pose name is provided.
        """
        if self._pose_pool is None:
            raise ValueError("Pose pool not available for this component.")
        if pose_name not in self._pose_pool:
            raise ValueError(
                f"Invalid pose name: {pose_name}. Available poses: {list(self._pose_pool.keys())}"
            )
        pose = self._pose_pool[pose_name]
        self.set_joint_pos(
            pose,
            wait_time=wait_time,
            exit_on_reach=exit_on_reach,
            exit_on_reach_kwargs=exit_on_reach_kwargs,
        )

    def is_joint_pos_reached(
        self,
        joint_pos: np.ndarray | dict[str, float],
        tolerance: float = 0.05,
        joint_id: list[int] | int | None = None,
    ) -> bool:
        """Check if the robot's current joint positions are within a certain tolerance of the target positions.

        Args:
            joint_pos: Target joint positions.
            tolerance: Tolerance for joint position check.
            joint_id: Optional specific joint indices to check.

        Returns:
            True if all specified joint positions are within tolerance, False otherwise.
        """
        # Handle dictionary input
        if isinstance(joint_pos, dict):
            current_pos = self.get_joint_pos_dict()
            return self._check_dict_positions_reached(
                joint_pos, current_pos, tolerance, joint_id
            )

        # Handle numpy array input
        current_pos = self.get_joint_pos()
        return self._check_array_positions_reached(
            joint_pos, current_pos, tolerance, joint_id
        )

    def _check_dict_positions_reached(
        self,
        target_pos: dict[str, float],
        current_pos: dict[str, float],
        tolerance: float,
        joint_id: list[int] | int | None,
    ) -> bool:
        """Check if dictionary-based joint positions are reached.

        Args:
            target_pos: Target joint positions as dictionary.
            current_pos: Current joint positions as dictionary.
            tolerance: Tolerance for position check.
            joint_id: Optional specific joint indices to check.

        Returns:
            True if positions are within tolerance, False otherwise.
        """
        if joint_id is not None:
            # Get joint names for the specified indices
            if self._joint_name is None:
                raise ValueError("Joint names not available for this component")

            # Handle single index case
            if isinstance(joint_id, int):
                if joint_id >= len(self._joint_name):
                    return True  # Invalid index, consider it reached

                name = self._joint_name[joint_id]
                return (
                    name in target_pos
                    and abs(current_pos[name] - target_pos[name]) <= tolerance
                )

            # Handle list of indices - filter valid ones
            valid_names = []
            for idx in joint_id:
                if idx < len(self._joint_name):
                    name = self._joint_name[idx]
                    if name in target_pos:
                        valid_names.append(name)

            # Only check valid joints that are in the target position dictionary
            return all(
                abs(current_pos[name] - target_pos[name]) <= tolerance
                for name in valid_names
            )
        else:
            # Check all joints in the dictionary
            return all(
                abs(current_pos[name] - pos) <= tolerance
                for name, pos in target_pos.items()
            )

    def _check_array_positions_reached(
        self,
        target_pos: np.ndarray,
        current_pos: np.ndarray,
        tolerance: float,
        joint_id: list[int] | int | None,
    ) -> bool:
        """Check if array-based joint positions are reached.

        Args:
            target_pos: Target joint positions as numpy array.
            current_pos: Current joint positions as numpy array.
            tolerance: Tolerance for position check.
            joint_id: Optional specific joint indices to check.

        Returns:
            True if positions are within tolerance, False otherwise.
        """
        if joint_id is not None:
            if isinstance(joint_id, int):
                # Single index - simple and efficient
                if joint_id >= len(current_pos) or joint_id >= len(target_pos):
                    return True  # Invalid index, consider it reached
                return abs(current_pos[joint_id] - target_pos[joint_id]) <= tolerance
            else:
                # For multiple indices - process one by one
                # This avoids using list indexing with lists which ListConfig doesn't support
                if len(current_pos) == 0 or len(target_pos) == 0:
                    return True

                for idx in joint_id:
                    if idx < len(current_pos) and idx < len(target_pos):
                        if abs(current_pos[idx] - target_pos[idx]) > tolerance:
                            return False
                return True
        else:
            # Check all joints, ensuring arrays are same length
            min_len = min(len(current_pos), len(target_pos))
            is_reached = bool(
                np.all(
                    np.abs(current_pos[:min_len] - target_pos[:min_len]) <= tolerance
                )
            )
            return is_reached

    def is_pose_reached(
        self,
        pose_name: str,
        tolerance: float = 0.05,
        joint_id: list[int] | int | None = None,
    ) -> bool:
        """Check if the robot's current joint positions are within a certain tolerance of the target pose.

        Args:
            pose_name: Name of the pose to check against.
            tolerance: Tolerance for joint position check.
            joint_id: Optional specific joint indices to check.

        Returns:
            True if all specified joint positions are within tolerance, False otherwise.

        Raises:
            ValueError: If pose pool is not available or pose name is invalid.
        """
        if self._pose_pool is None:
            raise ValueError("Pose pool not available for this component.")
        if pose_name not in self._pose_pool:
            raise ValueError(
                f"Invalid pose name: {pose_name}. Available poses: {list(self._pose_pool.keys())}"
            )
        pose = self._pose_pool[pose_name]
        return self.is_joint_pos_reached(pose, tolerance=tolerance, joint_id=joint_id)

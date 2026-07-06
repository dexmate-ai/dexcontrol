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

import itertools
import json
import random
import threading
import time
import warnings
from collections.abc import Iterator
from typing import Any, Callable, Mapping, TypeVar

import numpy as np
from jaxtyping import Float
from loguru import logger

from dexcontrol.core.motion_handle import MotionHandle
from dexcontrol.core.shared_node import get_shared_node
from dexcontrol.core.subscription_policy import (
    SubscriptionPolicyManager,
    SubscriptionPolicyMixin,
)
from dexcontrol.exceptions import ServiceUnavailableError

# Type variable for Message subclasses
M = TypeVar("M")


class MotionPluginManaged:
    """Marker mixin for components that communicate with the motion plugin.

    The *motion plugin* is the robot's internal motion controller: a plugin
    that runs on the robot itself (inside the robot-server process), not in
    this client library. It receives the targets/trajectories published by
    these components, generates the actual motion (trajectory smoothing,
    gravity compensation, convergence detection) and streams the resulting
    commands to the hardware. The methods here only encode and publish
    requests over DexComm; all motion generation happens on the robot.

    Subclasses are responsible for owning their target publisher, status
    subscriber (if any), motion-id counter, and shutdown cleanup. This
    mixin exists for isinstance discrimination (e.g., Robot-level fan-outs
    and validation) and to document the relationship.

    The mixin has no ``__init__``, no instance state, and only a static
    helper for generating a per-component motion_id sequence.
    """

    @staticmethod
    def _new_motion_id_counter() -> Iterator[int]:
        """Return a motion_id sequence starting from a random base.

        A random start reduces the risk of colliding motion_ids when a
        new client process attaches to the same plugin session.
        """
        return itertools.count(start=random.randint(1, 2**32 - 1))


class RobotComponent(SubscriptionPolicyMixin):
    """Base class for robot components with state interface.

    A component represents a physical part of the robot that maintains state through
    Zenoh communication. It subscribes to state updates and provides methods to
    access the latest state data.

    Uses dexcomm's Rust-side storage for zero GIL contention - the background thread
    stores raw bytes without acquiring the GIL, and get_latest() decodes on-demand
    with smart caching (<1μs cache hit, ~10μs cache miss).

    Attributes:
        _node: DexComm node for communication management.
        _subscriber: DexComm subscriber with Rust-side state storage.
    """

    def __init__(
        self,
        name: str,
        state_sub_topic: str,
        state_decoder: Callable[[bytes], Any] | None = None,
    ) -> None:
        """Initializes RobotComponent.

        Args:
            name: Name of the component node.
            state_sub_topic: Topic to subscribe to for state updates.
            state_decoder: Decoder function for state messages.
        """
        super().__init__()
        self._node = get_shared_node()
        # No callback - use Rust-side storage for zero GIL contention
        self._subscriber = self._node.create_subscriber(
            topic=state_sub_topic,
            decoder=state_decoder,
        )
        self._policy_manager = SubscriptionPolicyManager(self._subscriber, name=name)
        self._subcomponents: dict[str, "RobotComponent"] = {}

    def get_state(self) -> Any:
        """Gets the current state of the component.

        Returns:
            Parsed state message from Rust-side storage with smart caching.

        Raises:
            ServiceUnavailableError: If no state data has been received yet.
        """
        msg = self._policy_manager.get_latest_managed()
        if msg is None:
            raise ServiceUnavailableError(
                f"No state data available for {self.__class__.__name__}"
            )
        return msg.data

    def wait_for_active(self, timeout: float = 5.0) -> bool:
        """Waits for the component to start receiving state updates.

        Args:
            timeout: Maximum time to wait in seconds.

        Returns:
            True if component becomes active, False if timeout is reached.
        """
        start_time = time.time()
        while time.time() - start_time < timeout:
            if self.is_active():
                return True
            time.sleep(0.1)
        return False

    def is_active(self) -> bool:
        """Check if component is receiving state updates.

        Returns:
            True if component is active, False otherwise.
        """
        return self._policy_manager.is_active()

    def shutdown(self) -> None:
        """Release communication resources (subscriber).

        This method only handles infrastructure cleanup. Application-level
        graceful stop (e.g. disabling motors) should be done by calling
        ``stop()`` separately before ``shutdown()``.  ``Robot.shutdown()``
        handles this — it calls ``stop()`` on active components, then
        ``shutdown()`` on all components.

        The shared Node is not shut down here — ``Robot.shutdown()`` calls
        ``shutdown_shared_node()`` as the final cleanup step.
        """
        if hasattr(self, "_subscriber") and self._subscriber:
            self._subscriber.shutdown()

    def get_timestamp_ns(self) -> int:
        """Get the timestamp (in nanoseconds) of the most recent state update.

        Returns:
            Timestamp in nanoseconds as recorded by the robot driver in the
            most recently received state message.

        Raises:
            ServiceUnavailableError: If no state data is available.
        """
        return self.get_state()["timestamp_ns"]


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
        name: str,
        state_sub_topic: str,
        control_pub_topic: str,
        control_encoder: Callable[[Any], bytes] | None = None,
        state_decoder: Callable[[bytes], Any] | None = None,
        joint_name: list[str] | None = None,
        joint_pos_limit: Float[np.ndarray, " N 2"] | None = None,
        joint_vel_limit: Float[np.ndarray, " N"] | None = None,
        pose_pool: Mapping[str, list[float] | np.ndarray] | None = None,
    ) -> None:
        """Initializes RobotJointComponent.

        Args:
            name: Name of the component node.
            state_sub_topic: Topic to subscribe to for state updates.
            control_pub_topic: Topic to publish control commands.
            control_encoder: Encoder function for control messages
            state_decoder: Decoder function for state messages
            joint_name: List of joint names for this component.
            joint_pos_limit: Joint position limits.
            joint_vel_limit: Joint velocity limits.
            pose_pool: Dictionary of predefined poses for this component.
        """
        super().__init__(name, state_sub_topic, state_decoder)

        self._control_pub_topic = control_pub_topic
        self._publisher = self._node.create_publisher(
            topic=control_pub_topic,
            encoder=control_encoder,
        )

        self._joint_name: list[str] | None = joint_name
        self._joint_pos_limit = joint_pos_limit
        self._joint_vel_limit = joint_vel_limit

        self._pose_pool: dict[str, np.ndarray] | None = (
            self._convert_pose_pool_to_arrays(pose_pool)
        )

    def _publish_control(self, control_msg: Any) -> None:
        """Publishes a control command message.

        Args:
            control_msg: Protobuf control message to publish.
        """
        # DexComm publisher with protobuf encoder handles this
        self._publisher.publish(control_msg)
        # Reset idle timer — controlling a component means we need its state feedback
        self._policy_manager.touch()

    def shutdown(self) -> None:
        """Cleans up the raw control publisher (and the inherited subscriber)."""
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
    def joint_pos_limit(self) -> np.ndarray | None:
        """Gets the joint position limits of the component.

        Returns:
            Array of shape (N, 2) where each row is [lower_limit, upper_limit]
            in radians (revolute) or meters (prismatic), or None if no limits
            were configured.
        """
        return (
            self._joint_pos_limit.copy() if self._joint_pos_limit is not None else None
        )

    @property
    def joint_vel_limit(self) -> np.ndarray | None:
        """Gets the joint velocity limits of the component.

        Returns:
            Array of shape (N,) containing the maximum speed for each joint
            in radians/s (revolute) or meters/s (prismatic), or None if no
            limits were configured.
        """
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
        state = self.get_state()
        if "pos" not in state:
            raise ValueError("Joint positions are not available for this component.")
        joint_pos = np.array(state["pos"], dtype=np.float32)
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
        state = self.get_state()
        if "vel" not in state:
            raise ValueError("Joint velocities are not available for this component.")
        joint_vel = np.array(state["vel"], dtype=np.float32)
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
        state = self.get_state()
        if "cur" not in state:
            raise ValueError("Joint currents are not available for this component.")
        joint_cur = np.array(state["cur"], dtype=np.float32)
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
        state = self.get_state()
        if "torque" not in state:
            raise ValueError("Joint torques are not available for this component.")
        joint_torque = np.array(state["torque"], dtype=np.float32)
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
        state = self.get_state()
        if not state.get("error"):
            raise ValueError("Joint error codes are not available for this component.")
        joint_err = np.array(state["error"], dtype=np.uint32)
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
            Array of shape (N, 3) where the last dimension is
            [position, velocity, current] when current data is available, or
            [position, velocity, torque] when only torque data is available.

        Raises:
            ValueError: If joint positions or velocities are not available.
        """
        state = self.get_state()
        if "pos" not in state or "vel" not in state:
            raise ValueError(
                "Joint positions or velocities are not available for this component."
            )

        # Create initial state array with positions and velocities
        joint_pos = np.array(state["pos"], dtype=np.float32)
        joint_vel = np.array(state["vel"], dtype=np.float32)

        if "cur" in state:
            # If currents are available, include them
            joint_cur = np.array(state["cur"], dtype=np.float32)
            joint_state = np.stack([joint_pos, joint_vel, joint_cur], axis=1)
        elif "torque" in state:
            # If torques are available, include them
            joint_torque = np.array(state["torque"], dtype=np.float32)
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
            Dictionary mapping joint names to arrays of [position, velocity, current]
            when current data is available, or [position, velocity, torque] when
            only torque data is available.

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
            wait_kwargs: Reserved for future use; currently not applied.
            exit_on_reach: If True, the function will exit when the joint positions are reached.
            exit_on_reach_kwargs: Optional parameters for exit when the joint positions are reached.

        Raises:
            ValueError: If joint_pos dictionary contains invalid joint names.
        """
        if wait_time > 0.0:
            warnings.warn(
                "wait_time in set_joint_pos() is deprecated and will be removed in "
                "dexcontrol 0.6.0. Use move_to_joint_pos() instead which relies on "
                "internal controller to handle motion generation, e.g. motion "
                "smoothing and gravity compensation.",
                DeprecationWarning,
                stacklevel=2,
            )

        if relative:
            joint_pos = self._resolve_relative_joint_cmd(joint_pos)

        # Convert to array format
        if isinstance(joint_pos, (list, dict)):
            joint_pos = self._convert_joint_cmd_to_array(joint_pos)

        if self._joint_pos_limit is not None:
            joint_pos = np.clip(
                joint_pos, self._joint_pos_limit[:, 0], self._joint_pos_limit[:, 1]
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


class ManagedJointComponent(RobotJointComponent, MotionPluginManaged):
    """Joint component managed by the motion plugin.

    The motion plugin is the robot's internal motion controller, running on
    the robot-server (see :class:`MotionPluginManaged`). This client class
    only publishes targets/trajectories to it and tracks their status; the
    motion itself (smoothing, gravity compensation, convergence) is computed
    on the robot.

    Adds the motion/target/ publisher, motion/status/ subscriber, motion-handle
    bookkeeping, ``move_joint_pos``, ``move_to_joint_pos``, ``go_to_pose``, and
    the per-component ``default_velocity_scale``. Used by ``Arm``, ``Head``,
    ``Torso``.

    Components that don't talk to the joint motion plugin (``Hand``,
    ``DexGripper``, ``ChassisSteer``, ``ChassisDrive``) inherit directly
    from ``RobotJointComponent`` and don't expose these methods.
    """

    def __init__(
        self,
        name: str,
        state_sub_topic: str,
        control_pub_topic: str,
        control_encoder: Callable[[Any], bytes] | None = None,
        state_decoder: Callable[[bytes], Any] | None = None,
        joint_name: list[str] | None = None,
        joint_pos_limit: Float[np.ndarray, " N 2"] | None = None,
        joint_vel_limit: Float[np.ndarray, " N"] | None = None,
        pose_pool: Mapping[str, list[float] | np.ndarray] | None = None,
    ) -> None:
        super().__init__(
            name=name,
            state_sub_topic=state_sub_topic,
            control_pub_topic=control_pub_topic,
            control_encoder=control_encoder,
            state_decoder=state_decoder,
            joint_name=joint_name,
            joint_pos_limit=joint_pos_limit,
            joint_vel_limit=joint_vel_limit,
            pose_pool=pose_pool,
        )

        # Target-control infrastructure (target publisher lazy-initialised on
        # first use).
        self._target_publisher: Any | None = None
        # Trajectory publisher for ``motion/trajectory/`` (lazy-initialised on
        # first ``move_joint_trajectory`` call).
        self._trajectory_publisher: Any | None = None

        self._default_velocity_scale: float | None = None

        # motion_id is generated by an iterator from the MotionPluginManaged
        # helper (random start, monotonic). The lock around _active_handles
        # also covers next() calls so the cancel-prev → new-id →
        # register-handle sequence is atomic.
        self._motion_id_counter: Iterator[int] = self._new_motion_id_counter()
        self._active_handles: dict[int, MotionHandle] = {}
        self._motion_status_lock = threading.Lock()

        # Eagerly create the status subscriber so zenoh has time to establish
        # the subscription before any tracked motion is attempted.
        self._motion_status_subscriber: Any | None = None
        self._ensure_status_subscriber()

    @property
    def default_velocity_scale(self) -> float | None:
        """Client-side default velocity scale for motion-plugin target calls.

        Returns the per-component default in (0, 1] applied when
        ``move_joint_pos``/``move_to_joint_pos`` is called without an explicit
        ``velocity_scale`` argument.
        ``None`` means no client-side default is set and the motion plugin's
        own default is used.
        """
        return self._default_velocity_scale

    @default_velocity_scale.setter
    def default_velocity_scale(self, value: float | None) -> None:
        if value is None:
            self._default_velocity_scale = None
            return
        v = float(value)
        if not np.isfinite(v):
            raise ValueError(f"default_velocity_scale must be finite, got {value}")
        if v <= 0.0 or v > 1.0:
            raise ValueError(f"default_velocity_scale must be in (0, 1], got {value}")
        self._default_velocity_scale = v

    def _publish_target(self, msg: Any) -> None:
        """Publishes a message to the motion-plugin target topic.

        Mirrors :meth:`_publish_control`: publishing to a component implies
        the caller will want state feedback, so we touch the policy manager
        to keep the state subscriber from being auto-idled while a motion
        is being streamed/executed.
        """
        self._target_publisher.publish(msg)
        self._policy_manager.touch()

    def _get_target_topic(self) -> str:
        """Derive the motion/target/ topic from the control/ topic.

        Uses the relative control_pub_topic (not the fully-qualified
        publisher.topic) to avoid double namespace prefixing.
        """
        return self._control_pub_topic.replace("control/", "motion/target/", 1)

    def _get_status_topic(self) -> str:
        """Derive the motion/status/ topic from the control/ topic."""
        return self._control_pub_topic.replace("control/", "motion/status/", 1)

    def _ensure_target_publisher(self) -> None:
        """Lazily create the target topic publisher on first use.

        Uses JointMotionCodec because the motion/target/ topic carries
        tracked motion primitives (with motion_id/cancel_motion_id) that
        JointCmd no longer supports. The streaming control/ topic still
        uses JointCmdCodec.
        """
        if self._target_publisher is not None:
            return
        from dexcomm.codecs import JointMotionCodec

        target_topic = self._get_target_topic()
        self._target_publisher = self._node.create_publisher(
            topic=target_topic,
            encoder=JointMotionCodec.encode,
        )

    def _ensure_trajectory_publisher(self) -> None:
        """Lazily create the motion/trajectory/ publisher on first use.

        Symmetric with :meth:`_ensure_target_publisher`. Derives the topic
        from the control_pub_topic so it stays consistent with the rest of
        the motion-plugin topic family.
        """
        if self._trajectory_publisher is not None:
            return
        from dexcomm.codecs import JointTrajectoryCodec

        trajectory_topic = self._get_target_topic().replace(
            "/target/", "/trajectory/", 1
        )
        self._trajectory_publisher = self._node.create_publisher(
            topic=trajectory_topic,
            encoder=JointTrajectoryCodec.encode,
        )

    def _ensure_status_subscriber(self) -> None:
        """Lazily create the status subscriber. Must be called BEFORE publishing
        a tracked target to avoid race condition."""
        if self._motion_status_subscriber is not None:
            return
        status_topic = self._get_status_topic()
        self._motion_status_subscriber = self._node.create_subscriber(
            topic=status_topic,
            callback=self._on_motion_status,
            decoder=None,  # Raw bytes; we JSON-decode inside _on_motion_status
        )
        logger.debug(f"Created status subscriber for {status_topic}")

    def _on_motion_status(self, raw: bytes) -> None:
        """Handle incoming motion status events from the plugin.

        The plugin publishes JSON bytes, e.g.:
        ``{"motion_id": 1, "state": "finished", "message": ""}``.
        """
        try:
            msg = json.loads(raw)
        except (json.JSONDecodeError, TypeError):
            logger.warning("Received malformed motion status message (not JSON)")
            return
        if not isinstance(msg, dict):
            logger.warning(
                "Received malformed motion status message (not a JSON object)"
            )
            return
        motion_id = msg.get("motion_id")
        state = msg.get("state")
        message = msg.get("message", "")
        if motion_id is None or state is None:
            return
        with self._motion_status_lock:
            handle = self._active_handles.get(motion_id)
            if handle is not None:
                handle._update_state(state, message)
                if handle.is_done:
                    self._active_handles.pop(motion_id, None)

    def _publish_cancel(self, motion_id: int) -> None:
        """Publish a cancel message to the target topic."""
        self._ensure_target_publisher()
        self._publish_target({"cancel_motion_id": motion_id})

    def _resolve_scale_arg(
        self,
        scale: float | np.ndarray | list[float] | None,
        n_joints: int,
    ) -> list[float] | None:
        """Resolve the ``scale`` argument to a per-joint list (or None).

        Precedence: explicit arg > component default > None (let plugin pick).
        A scalar is broadcast to all joints; an array/list is converted to a
        plain ``list[float]`` suitable for the JSON-encoded wire message.
        """
        effective = scale if scale is not None else self._default_velocity_scale
        if effective is None:
            return None
        if isinstance(effective, (int, float)):
            return [float(effective)] * n_joints
        return [float(v) for v in effective]

    def move_joint_pos(
        self,
        pos: np.ndarray | list[float] | dict[str, float],
        *,
        relative: bool = False,
        velocity_scale: float | np.ndarray | list[float] | None = None,
    ) -> None:
        """Send a target joint position to the motion plugin.

        The motion plugin is the robot's internal motion controller, running
        on the robot-server. It handles trajectory generation (smoothing,
        gravity compensation) and drives the hardware at its configured rate.
        This client method only publishes the target; it does not track
        convergence and returns immediately after publishing.

        Args:
            pos: Target joint positions. Accepts numpy array, list, or dict
                (keyed by joint name). Clipped to joint limits before sending.
            relative: If True, the positions are interpreted as offsets from
                the current joint positions.
            velocity_scale: Per-joint velocity scale in (0, 1]. Controls how
                fast the motion executes as a fraction of the hardware velocity
                ceiling. A scalar is broadcast to all joints. ``None`` falls
                back to ``self.default_velocity_scale`` if set, otherwise to
                the motion plugin's default (typically 0.5). Example: 0.25 =
                quarter of max speed.
        """
        self._send_motion_target(
            pos, scale=velocity_scale, relative=relative, return_handle=False
        )

    def move_to_joint_pos(
        self,
        pos: np.ndarray | list[float] | dict[str, float],
        *,
        relative: bool = False,
        velocity_scale: float | np.ndarray | list[float] | None = None,
    ) -> MotionHandle:
        """Send a target joint position to the motion plugin and return a handle.

        The motion plugin is the robot's internal motion controller, running
        on the robot-server. It handles trajectory generation (smoothing,
        gravity compensation) and drives the hardware at its configured rate.
        This client method returns a ``MotionHandle`` that can be used to
        monitor completion, cancellation, or plugin-side errors.

        Args:
            pos: Target joint positions. Accepts numpy array, list, or dict
                (keyed by joint name). Clipped to joint limits before sending.
            relative: If True, the positions are interpreted as offsets from
                the current joint positions.
            velocity_scale: Per-joint velocity scale in (0, 1]. Controls how
                fast the motion executes as a fraction of the hardware velocity
                ceiling. A scalar is broadcast to all joints. ``None`` falls
                back to ``self.default_velocity_scale`` if set, otherwise to
                the motion plugin's default (typically 0.5). Example: 0.25 =
                quarter of max speed.

        Returns:
            MotionHandle for monitoring completion, cancellation, or error.
        """
        handle = self._send_motion_target(
            pos, scale=velocity_scale, relative=relative, return_handle=True
        )
        assert handle is not None
        return handle

    def _send_motion_target(
        self,
        pos: np.ndarray | list[float] | dict[str, float],
        scale: float | np.ndarray | list[float] | None = None,
        relative: bool = False,
        return_handle: bool = False,
    ) -> MotionHandle | None:
        """Publish a target joint position to the motion plugin.

        Args mirror the public move methods; ``return_handle`` controls whether
        a client-generated motion_id and MotionHandle are created.
        """
        # Resolve relative positions
        if relative:
            pos = self._resolve_relative_joint_cmd(pos)

        # Resolve and clip position
        if isinstance(pos, (list, dict)):
            pos = self._convert_joint_cmd_to_array(pos)
        if self._joint_pos_limit is not None:
            pos = np.clip(pos, self._joint_pos_limit[:, 0], self._joint_pos_limit[:, 1])

        # Resolve velocity scale: explicit arg > component default > plugin default.
        # scalar → per-joint broadcast.
        scale_array = self._resolve_scale_arg(scale, len(pos))

        self._ensure_target_publisher()

        msg: dict = {"pos": pos}
        if scale_array is not None:
            msg["scale"] = scale_array

        if not return_handle:
            self._publish_target(msg)
            return None

        # Handle-returning mode: create the status subscriber before publishing.
        self._ensure_status_subscriber()

        # Atomically: cancel any in-flight handle-returning motions, allocate a new
        # motion_id, and register the new handle. Doing all three under a
        # single lock acquisition prevents concurrent handle-returning calls
        # from producing colliding motion_ids or out-of-order registration.
        with self._motion_status_lock:
            for existing_handle in self._active_handles.values():
                existing_handle._update_state("cancelled", "superseded by new target")
            self._active_handles.clear()

            motion_id = next(self._motion_id_counter)
            handle = MotionHandle(
                motion_id=motion_id,
                publish_cancel_fn=self._publish_cancel,
            )
            self._active_handles[motion_id] = handle

            # Publish while still holding the lock so a concurrent shutdown
            # cannot observe the registered motion_id and publish a cancel
            # before this target has actually reached the wire.
            msg["motion_id"] = motion_id
            self._publish_target(msg)

        return handle

    def _validate_trajectory_args(
        self,
        positions: np.ndarray | list,
        time_from_start: np.ndarray | list,
        velocities: np.ndarray | list | None,
        accelerations: np.ndarray | list | None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray | None, np.ndarray | None]:
        """Structural validation for ``move_joint_trajectory`` inputs.

        Only structural rules are checked here (shape, dtype-finite, length
        consistency). Semantic rules — scale bounds, time monotonicity,
        joint-limit bounds — are intentionally deferred to the motion
        plugin, which clips or rejects per its own validation posture
        (companion spec ``robot-server-plugin``
        ``docs/specs/2026-05-17-joint-trajectory-streaming-design.md`` §4.1).

        Returns the coerced numpy arrays so callers don't repeat the
        ``np.asarray`` work.
        """
        if self._joint_name is None:
            raise ValueError(
                "Cannot send a joint trajectory: component has no joint_name "
                "configured (dof is unknown)."
            )
        dof = len(self._joint_name)

        try:
            positions_arr = np.asarray(positions, dtype=float)
        except (TypeError, ValueError) as e:
            raise ValueError(f"positions cannot be converted to a float array: {e}")
        if positions_arr.ndim != 2:
            raise ValueError(
                f"positions must be a 2-D array with shape (N, dof); "
                f"got ndim={positions_arr.ndim}, shape={positions_arr.shape}."
            )
        n_points, dof_in = positions_arr.shape
        if dof_in != dof:
            raise ValueError(
                f"positions has dof={dof_in}, but component has dof={dof}."
            )
        if n_points < 2:
            raise ValueError(
                f"positions must have N >= 2 waypoints; got N={n_points}. "
                "For a single-point command use move_to_joint_pos()."
            )
        if not np.isfinite(positions_arr).all():
            raise ValueError("positions contains non-finite values (NaN or Inf).")

        try:
            time_arr = np.asarray(time_from_start, dtype=float)
        except (TypeError, ValueError) as e:
            raise ValueError(
                f"time_from_start cannot be converted to a float array: {e}"
            )
        if time_arr.shape != (n_points,):
            raise ValueError(
                f"time_from_start must have shape ({n_points},) to match "
                f"positions.shape[0]; got shape {time_arr.shape}."
            )
        if not np.isfinite(time_arr).all():
            raise ValueError("time_from_start contains non-finite values (NaN or Inf).")

        velocities_arr: np.ndarray | None = None
        if velocities is not None:
            try:
                velocities_arr = np.asarray(velocities, dtype=float)
            except (TypeError, ValueError) as e:
                raise ValueError(
                    f"velocities cannot be converted to a float array: {e}"
                )
            if velocities_arr.shape != positions_arr.shape:
                raise ValueError(
                    f"velocities must have the same shape as positions "
                    f"({positions_arr.shape}); got {velocities_arr.shape}."
                )
            if not np.isfinite(velocities_arr).all():
                raise ValueError("velocities contains non-finite values (NaN or Inf).")

        accelerations_arr: np.ndarray | None = None
        if accelerations is not None:
            try:
                accelerations_arr = np.asarray(accelerations, dtype=float)
            except (TypeError, ValueError) as e:
                raise ValueError(
                    f"accelerations cannot be converted to a float array: {e}"
                )
            if accelerations_arr.shape != positions_arr.shape:
                raise ValueError(
                    f"accelerations must have the same shape as positions "
                    f"({positions_arr.shape}); got {accelerations_arr.shape}."
                )
            if not np.isfinite(accelerations_arr).all():
                raise ValueError(
                    "accelerations contains non-finite values (NaN or Inf)."
                )

        return positions_arr, time_arr, velocities_arr, accelerations_arr

    def move_joint_trajectory(
        self,
        positions: np.ndarray,
        time_from_start: np.ndarray | list[float],
        velocities: np.ndarray | None = None,
        accelerations: np.ndarray | None = None,
        scale: float | np.ndarray | list[float] | None = None,
        tracked: bool = False,
    ) -> MotionHandle | None:
        """Send a timed joint trajectory to the motion plugin.

        Publishes a JointTrajectory message on motion/trajectory/{component}.
        The plugin tracks the trajectory using cubic-Hermite interpolation with
        Ruckig under hard vel/accel/jerk limits; see the motion plugin's
        docs/smoothing-algorithm.md for execution details.

        Args:
            positions: (N, dof) waypoint joint positions. N >= 2 required;
                for single-point use move_to_joint_pos().
            time_from_start: (N,) waypoint times in seconds. Must be
                monotonically increasing with t[0] >= 0 and t[-1] > 0.
            velocities: Optional (N, dof) per-waypoint velocity feedforward
                in rad/s. If None, the plugin uses centered finite differences
                between neighboring waypoints.
            accelerations: Optional (N, dof) per-waypoint acceleration in
                rad/s^2. Accepted on the wire for forward compatibility;
                ignored by the v1 plugin (it uses Hermite-derived acc instead).
            scale: Per-joint motion scale in (0, 1]. Scalar broadcasts to all
                joints. None lets the plugin pick its default. See the
                companion plugin spec §4.1.0 for clipping behavior on
                out-of-range values.
            tracked: If True, returns a MotionHandle for monitoring completion
                and cancellation. If False (default), fire-and-forget.

        Returns:
            MotionHandle if tracked=True, None otherwise.

        Raises:
            ValueError: structural input errors only (wrong shape, NaN/Inf,
                length mismatches). Semantic errors (monotone time, scale
                bounds, joint-limit overrun) are NOT validated here; the
                plugin handles them per its validation posture (companion
                spec §4.1).
        """
        positions_arr, time_arr, velocities_arr, accelerations_arr = (
            self._validate_trajectory_args(
                positions, time_from_start, velocities, accelerations
            )
        )

        scale_array = self._resolve_scale_arg(scale, positions_arr.shape[1])

        # Build dexcomm dict: every point carries time_from_start so the
        # plugin treats this as a timed trajectory.
        n_points = positions_arr.shape[0]
        points: list[dict] = []
        for i in range(n_points):
            point: dict = {
                "pos": positions_arr[i].tolist(),
                "time_from_start": float(time_arr[i]),
            }
            if velocities_arr is not None:
                point["vel"] = velocities_arr[i].tolist()
            if accelerations_arr is not None:
                point["accel"] = accelerations_arr[i].tolist()
            points.append(point)

        msg: dict = {"points": points}
        if scale_array is not None:
            msg["scale"] = scale_array

        if not tracked:
            self._ensure_trajectory_publisher()
            self._trajectory_publisher.publish(msg)
            self._policy_manager.touch()
            return None

        # Tracked mode: ensure status subscriber first to avoid races where
        # the plugin's first status event arrives before we're subscribed.
        self._ensure_status_subscriber()
        self._ensure_trajectory_publisher()

        # Atomically: cancel any in-flight tracked motions, allocate a new
        # motion_id, and register the new handle. Mirrors move_to_joint_pos
        # exactly so the two APIs are interchangeable from a tracking
        # standpoint.
        with self._motion_status_lock:
            for existing_handle in self._active_handles.values():
                existing_handle._update_state("cancelled", "superseded by new target")
            self._active_handles.clear()

            motion_id = next(self._motion_id_counter)
            handle = MotionHandle(
                motion_id=motion_id,
                publish_cancel_fn=self._publish_cancel,
            )
            self._active_handles[motion_id] = handle

            # Publish while still holding the lock so a concurrent shutdown
            # cannot observe the registered motion_id and publish a cancel
            # before this trajectory has actually reached the wire.
            msg["motion_id"] = motion_id
            self._trajectory_publisher.publish(msg)
            self._policy_manager.touch()
        return handle

    def go_to_pose(
        self,
        pose_name: str,
        timeout: float | None = None,
    ) -> MotionHandle:
        """Move the component to a predefined pose using the motion plugin.

        Sends the pose as a tracked target to the motion plugin, which handles
        trajectory smoothing, gravity compensation, and convergence detection.
        The plugin signals completion when the trajectory converges.

        Args:
            pose_name: Name of the pose to move to.
            timeout: Maximum time to wait for the motion to complete, in
                seconds. None means wait indefinitely until the plugin
                signals convergence.

        Returns:
            MotionHandle for the completed motion.

        Raises:
            ValueError: If pose pool is not available or if an invalid pose
                name is provided.
            TimeoutError: If the motion does not complete within timeout.
        """
        if self._pose_pool is None:
            raise ValueError("Pose pool not available for this component.")
        if pose_name not in self._pose_pool:
            raise ValueError(
                f"Invalid pose name: {pose_name}. "
                f"Available poses: {list(self._pose_pool.keys())}"
            )
        pose = self._pose_pool[pose_name]
        handle = self.move_to_joint_pos(pose)
        handle.wait(timeout=timeout)
        return handle

    def shutdown(self) -> None:
        """Cancel in-flight motions, then clean up motion-plugin resources."""
        # Cancel any in-flight tracked motions before destroying the target
        # publisher so the plugin stops its trajectory.
        try:
            handles_to_cancel: list[tuple[int, MotionHandle]] = []
            if hasattr(self, "_active_handles") and self._active_handles:
                with self._motion_status_lock:
                    handles_to_cancel = list(self._active_handles.items())
                    self._active_handles.clear()
            # Publish cancels OUTSIDE the lock: a status event arriving on the
            # receiver thread also takes _motion_status_lock, and we must not
            # block it behind a (potentially blocking) network publish while we
            # then tear down the very subscriber that feeds it.
            for motion_id, handle in handles_to_cancel:
                try:
                    self._publish_cancel(motion_id)
                except Exception:
                    pass  # Best-effort during shutdown
                handle._update_state("cancelled", "shutdown")
        except Exception as e:
            logger.debug(f"Error cancelling tracked motions during shutdown: {e}")

        # Clean up target-control resources.
        try:
            if hasattr(self, "_target_publisher") and self._target_publisher:
                self._target_publisher.shutdown()
        except Exception as e:
            logger.warning(
                f"Error shutting down target publisher for {self.__class__.__name__}: {e}"
            )
        try:
            if hasattr(self, "_trajectory_publisher") and self._trajectory_publisher:
                self._trajectory_publisher.shutdown()
        except Exception as e:
            logger.warning(
                f"Error shutting down trajectory publisher for {self.__class__.__name__}: {e}"
            )
        try:
            if (
                hasattr(self, "_motion_status_subscriber")
                and self._motion_status_subscriber
            ):
                self._motion_status_subscriber.shutdown()
        except Exception as e:
            logger.warning(
                f"Error shutting down status subscriber for {self.__class__.__name__}: {e}"
            )

        # Then run RobotJointComponent.shutdown (publisher + inherited subscriber).
        super().shutdown()

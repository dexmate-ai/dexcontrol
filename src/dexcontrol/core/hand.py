# Copyright (C) 2025 Dexmate Inc.
#
# This software is dual-licensed:
#
# 1. GNU Affero General Public License v3.0 (AGPL-3.0)
#    See LICENSE-AGPL for details
#
# 2. Commercial License
#    For commercial licensing terms, contact: contact@dexmate.ai

"""Robot hand control module.

This module provides the Hand class for controlling a robotic hand through Zenoh
communication. It handles joint position control and state monitoring.
"""

from enum import Enum
from typing import Any, cast

import numpy as np
from dexcomm.serialization.protobuf import control_msg_pb2
from jaxtyping import Float
from loguru import logger

from dexcontrol.config.core.hand import HandConfig
from dexcontrol.core.component import RobotComponent, RobotJointComponent


class HandType(Enum):
    UNKNOWN = "UNKNOWN"
    HandF5D6_V1 = "HandF5D6_V1"
    HandF5D6_V2 = "HandF5D6_V2"


class Hand(RobotJointComponent):
    """Robot hand control class.

    This class provides methods to control a robotic hand by publishing commands and
    receiving state information through Zenoh communication.
    """

    def __init__(
        self,
        configs: HandConfig,
        hand_type: HandType = HandType.HandF5D6_V1,
    ) -> None:
        """Initialize the hand controller.

        Args:
            configs: Hand configuration parameters containing communication topics
                and predefined hand positions.
        """
        super().__init__(
            state_sub_topic=configs.state_sub_topic,
            control_pub_topic=configs.control_pub_topic,
            state_message_type=control_msg_pb2.MotorStateWithCurrent,
            joint_name=configs.joint_name,
        )

        # Store predefined hand positions as private attributes
        self._joint_pos_open = np.array(configs.pose_pool["open"], dtype=np.float32)
        self._joint_pos_close = np.array(configs.pose_pool["close"], dtype=np.float32)

    def _send_position_command(
        self, joint_pos: Float[np.ndarray, " N"] | list[float]
    ) -> None:
        """Send joint position control commands to the hand.

        Args:
            joint_pos: Joint positions as list or numpy array.
        """
        control_msg = control_msg_pb2.MotorPosCommand()
        joint_pos_array = self._convert_joint_cmd_to_array(joint_pos)
        control_msg.pos.extend(joint_pos_array.tolist())
        self._publish_control(control_msg)

    def open_hand(
        self,
        wait_time: float = 0.0,
        exit_on_reach: bool = False,
        exit_on_reach_kwargs: dict[str, Any] | None = None,
    ) -> None:
        """Open the hand to the predefined open position.

        Args:
            wait_time: Time to wait after opening the hand.
            exit_on_reach: If True, the function will exit when the joint positions are reached.
            exit_on_reach_kwargs: Optional parameters for exit when the joint positions are reached.
        """
        self.set_joint_pos(
            self._joint_pos_open,
            wait_time=wait_time,
            exit_on_reach=exit_on_reach,
            exit_on_reach_kwargs=exit_on_reach_kwargs,
        )

    def close_hand(
        self,
        wait_time: float = 0.0,
        exit_on_reach: bool = False,
        exit_on_reach_kwargs: dict[str, Any] | None = None,
    ) -> None:
        """Close the hand to the predefined closed position.

        Args:
            wait_time: Time to wait after closing the hand.
            exit_on_reach: If True, the function will exit when the joint positions are reached.
            exit_on_reach_kwargs: Optional parameters for exit when the joint positions are reached.

        """
        self.set_joint_pos(
            self._joint_pos_close,
            wait_time=wait_time,
            exit_on_reach=exit_on_reach,
            exit_on_reach_kwargs=exit_on_reach_kwargs,
        )


class HandF5D6(Hand):
    """Specialized hand class for the F5D6 hand model.

    Extends the basic Hand class with additional control methods specific to
    the F5D6 hand model.
    """

    def __init__(
        self,
        configs: HandConfig,
        hand_type: HandType = HandType.HandF5D6_V1,
    ) -> None:
        super().__init__(configs)

        # Initialize touch sensor for F5D6_V2 hands
        self._hand_type = hand_type
        if self._hand_type == HandType.HandF5D6_V2:
            self._touch_sensor = HandF5D6TouchSensor(configs.touch_sensor_sub_topic)
        elif self._hand_type == HandType.HandF5D6_V1:
            self._touch_sensor = None
        else:
            raise ValueError(f"Invalid hand type: {self._hand_type}")

    def get_finger_tip_force(self) -> Float[np.ndarray, "5"]:
        """Get the force at the finger tips.

        Returns:
            Array of force values at the finger tips.
        """
        if self._touch_sensor is None:
            raise ValueError(
                f"Touch sensor not available for this hand type: {self._hand_type}"
            )
        return self._touch_sensor.get_fingertip_touch_net_force()

    def close_hand(
        self,
        wait_time: float = 0.0,
        exit_on_reach: bool = False,
        exit_on_reach_kwargs: dict[str, Any] | None = None,
    ) -> None:
        """Close the hand fully using a two-step approach for better control."""
        try:
            if self.is_joint_pos_reached(self._joint_pos_close, tolerance=0.1):
                return

            # First step: Move to intermediate position
            intermediate_pos = self._get_intermediate_close_position()
            first_step_wait_time = 0.8
            self.set_joint_pos(
                intermediate_pos,
                wait_time=first_step_wait_time,
                exit_on_reach=exit_on_reach,
                exit_on_reach_kwargs=exit_on_reach_kwargs,
            )

            # Second step: Move to final closed position
            remaining_wait_time = max(0.0, wait_time - first_step_wait_time)
            self.set_joint_pos(
                self._joint_pos_close,
                wait_time=remaining_wait_time,
                exit_on_reach=exit_on_reach,
                exit_on_reach_kwargs=exit_on_reach_kwargs,
            )
        except Exception as e:
            logger.warning(f"Failed to close hand: {e}")

    def open_hand(
        self,
        wait_time: float = 0.0,
        exit_on_reach: bool = False,
        exit_on_reach_kwargs: dict[str, Any] | None = None,
    ) -> None:
        """Open the hand fully using a two-step approach for better control."""
        try:
            if self.is_joint_pos_reached(self._joint_pos_open, tolerance=0.1):
                return

            # First step: Move to intermediate position
            intermediate_pos = self._get_intermediate_open_position()
            first_step_wait_time = 0.3
            self.set_joint_pos(
                intermediate_pos,
                wait_time=first_step_wait_time,
                exit_on_reach=exit_on_reach,
                exit_on_reach_kwargs=exit_on_reach_kwargs,
            )

            # Second step: Move to final open position
            remaining_wait_time = max(0.0, wait_time - first_step_wait_time)
            self.set_joint_pos(
                self._joint_pos_open,
                wait_time=remaining_wait_time,
                exit_on_reach=exit_on_reach,
                exit_on_reach_kwargs=exit_on_reach_kwargs,
            )
        except Exception as e:
            logger.warning(f"Failed to open hand: {e}")

    def _get_intermediate_close_position(self) -> np.ndarray:
        """Get intermediate position for closing hand.

        Returns:
            Intermediate joint positions for smooth closing motion.
        """
        intermediate_pos = self._joint_pos_close.copy()
        ratio = 0.2
        # Adjust thumb opposition joint (last joint)
        intermediate_pos[-1] = self._joint_pos_close[-1] * ratio + self._joint_pos_open[
            -1
        ] * (1 - ratio)
        return intermediate_pos

    def _get_intermediate_open_position(self) -> np.ndarray:
        """Get intermediate position for opening hand.

        Returns:
            Intermediate joint positions for smooth opening motion.
        """
        intermediate_pos = self._joint_pos_close.copy()
        ratio = 0.2
        # Adjust thumb opposition joint (last joint)
        intermediate_pos[-1] = self._joint_pos_close[-1] * ratio + self._joint_pos_open[
            -1
        ] * (1 - ratio)
        # Adjust thumb flexion joint (first joint)
        intermediate_pos[0] = self._joint_pos_close[0] * ratio + self._joint_pos_open[
            0
        ] * (1 - ratio)
        return intermediate_pos


class HandF5D6TouchSensor(RobotComponent):
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
            state_message_type=control_msg_pb2.HandTouchSensorState,
        )

    def get_fingertip_touch_net_force(self) -> Float[np.ndarray, "5"]:
        """Get the complete wrench sensor state.

        Returns:
            Dictionary containing wrench values and button states.
        """
        state = self._get_state()
        hand_touch_state = cast(control_msg_pb2.HandTouchSensorState, state)
        return np.array(hand_touch_state.force)

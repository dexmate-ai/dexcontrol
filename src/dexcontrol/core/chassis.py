# Copyright (C) 2025 Dexmate Inc.
#
# This software is dual-licensed:
#
# 1. GNU Affero General Public License v3.0 (AGPL-3.0)
#    See LICENSE-AGPL for details
#
# 2. Commercial License
#    For commercial licensing terms, contact: contact@dexmate.ai

"""Robot base control module.

This module provides the Chassis class for controlling a robot's wheeled base through
Zenoh communication. It handles steering and wheel velocity control.
"""

import time

import numpy as np
import zenoh

from dexcontrol.config.core import ChassisConfig
from dexcontrol.core.component import RobotJointComponent
from dexcontrol.proto import dexcontrol_msg_pb2
from dexcontrol.utils.rate_limiter import RateLimiter


class Chassis(RobotJointComponent):
    """Robot base control class.

    This class provides methods to control a robot's wheeled base by publishing commands
    and receiving state information through Zenoh communication.

    The chassis has 4 controllable joints:
    - Left steering angle
    - Left wheel velocity
    - Right steering angle
    - Right wheel velocity

    Attributes:
        max_vel: Maximum allowed wheel velocity in m/s.
    """

    def __init__(
        self,
        configs: ChassisConfig,
        zenoh_session: zenoh.Session,
    ) -> None:
        """Initialize the base controller.

        Args:
            configs: Base configuration parameters containing communication topics.
            zenoh_session: Active Zenoh communication session for message passing.
        """
        super().__init__(
            state_sub_topic=configs.state_sub_topic,
            control_pub_topic=configs.control_pub_topic,
            state_message_type=dexcontrol_msg_pb2.ChassisState,
            zenoh_session=zenoh_session,
            joint_name=configs.joint_name,
        )
        self.max_vel = configs.max_vel
        self._center_to_wheel_axis_dist = configs.center_to_wheel_axis_dist
        self._wheels_dist = configs.wheels_dist
        self._half_wheels_dist = self._wheels_dist / 2

        # Pre-compute geometry constants for efficiency
        self._center_to_wheel_dist = np.sqrt(
            self._center_to_wheel_axis_dist**2 + self._half_wheels_dist**2
        )
        self._left_wheel_ang_vel_vector = np.array(
            [
                -self._half_wheels_dist / self._center_to_wheel_dist,
                self._center_to_wheel_axis_dist / self._center_to_wheel_dist,
            ]
        )
        self._right_wheel_ang_vel_vector = np.array(
            [
                self._half_wheels_dist / self._center_to_wheel_dist,
                self._center_to_wheel_axis_dist / self._center_to_wheel_dist,
            ]
        )

        # Constants for steering angle constraints
        self._max_steering_angle = np.deg2rad(135)
        self._min_velocity_threshold = 1e-6
        self.max_lin_vel = self.max_vel
        self.max_ang_vel = self.max_vel / self._center_to_wheel_dist

    def stop(self) -> None:
        """Stop the base by setting all wheel velocities and steering to zero."""
        self.set_motion_state(
            np.zeros(2, dtype=np.float32), np.zeros(2, dtype=np.float32)
        )

    def shutdown(self) -> None:
        """Shutdown the base by stopping all motion."""
        self.stop()
        super().shutdown()

    @property
    def steering_angle(self) -> np.ndarray:
        """Get current steering angles.

        Returns:
            Numpy array of shape (2,) containing current steering angles in radians.
            Index 0 is left wheel, index 1 is right wheel.
        """
        state = self._get_state()
        return np.array([state.left.steering_pos, state.right.steering_pos])

    @property
    def wheel_velocity(self) -> np.ndarray:
        """Get current wheel velocities.

        Returns:
            Numpy array of shape (2,) containing current wheel velocities in m/s.
            Index 0 is left wheel, index 1 is right wheel.
        """
        state = self._get_state()
        return np.array([state.left.wheel_vel, state.right.wheel_vel])

    def set_velocity(
        self,
        vx: float,
        vy: float,
        wz: float,
        wait_time: float = 0.0,
        wait_kwargs: dict[str, float] | None = None,
        sequential_steering: bool = True,
        steering_wait_time: float = 1.0,
        steering_tolerance: float = 0.05,
    ) -> None:
        """Set the chassis velocity in the horizontal plane.

        This method takes the desired translational velocity (vx, vy) and rotational
        velocity (wz) of the chassis and computes the required steering angles
        and wheel velocities for each wheel.

        Coordinate System:
            - X-axis: Points forward (front of chassis)
            - Y-axis: Points left (left side of chassis)
            - Z-axis: Points up (vertical)
            - Rotation: Right-hand rule around Z-axis (positive = counter-clockwise)

        Args:
            vx: Translational velocity in x-direction in m/s.
                Positive = forward, Negative = backward
            vy: Translational velocity in y-direction in m/s.
                Positive = left, Negative = right
            wz: Rotational velocity around z-axis (yaw) in rad/s.
                Positive = counter-clockwise (left turn)
                Negative = clockwise (right turn)
            wait_time: Time to maintain the command in seconds.
            wait_kwargs: Additional parameters for wait behavior.
            sequential_steering: If True, first adjust steering angles, then set wheel
                velocities. If False, set both simultaneously.
            steering_wait_time: Time to wait for steering adjustment when
                sequential_steering=True.
            steering_tolerance: Angular tolerance in radians to consider steering
                "complete".

        Note:
            Steering angles are constrained to [-170°, 170°] ≈ [-2.967, 2.967] rad.
            If the required steering angle exceeds this range, the wheel velocity
            will be reversed and the steering angle adjusted accordingly.

        Examples:
            # Move forward at 0.5 m/s
            chassis.set_velocity(vx=0.5, vy=0.0, wz=0.0)

            # Move left at 0.3 m/s
            chassis.set_velocity(vx=0.0, vy=0.3, wz=0.0)

            # Turn left (counter-clockwise) at 0.5 rad/s
            chassis.set_velocity(vx=0.0, vy=0.0, wz=0.5)

            # Combined: forward + left + turn left
            chassis.set_velocity(vx=0.4, vy=0.2, wz=0.3)
        """
        vx = np.clip(vx, -self.max_lin_vel, self.max_lin_vel)
        vy = np.clip(vy, -self.max_lin_vel, self.max_lin_vel)
        wz = np.clip(wz, -self.max_ang_vel, self.max_ang_vel)
        wait_kwargs = wait_kwargs or {}

        # Convert to numpy array for vectorized computation
        linear_velocity = np.array([vx, vy])

        # Compute velocity contribution from rotation for each wheel
        left_wheel_rotation_vel = (
            wz * self._center_to_wheel_dist * self._left_wheel_ang_vel_vector
        )
        right_wheel_rotation_vel = (
            wz * self._center_to_wheel_dist * self._right_wheel_ang_vel_vector
        )

        # Total velocity for each wheel (translation + rotation)
        left_wheel_velocity = linear_velocity + left_wheel_rotation_vel
        right_wheel_velocity = linear_velocity + right_wheel_rotation_vel

        # Get current steering angles
        current_steering = self.steering_angle

        # Compute steering angles and wheel speeds
        left_steering_angle, left_wheel_speed = self._compute_wheel_control(
            left_wheel_velocity, current_steering[0]
        )
        right_steering_angle, right_wheel_speed = self._compute_wheel_control(
            right_wheel_velocity, current_steering[1]
        )

        target_steering = np.array([left_steering_angle, right_steering_angle])
        target_wheel_speeds = np.array([left_wheel_speed, right_wheel_speed])

        if sequential_steering:
            self._apply_sequential_steering(
                target_steering,
                target_wheel_speeds,
                steering_tolerance,
                steering_wait_time,
                wait_time,
                wait_kwargs,
            )
        else:
            # Apply both steering and velocity simultaneously
            self.set_motion_state(
                target_steering, target_wheel_speeds, wait_time, wait_kwargs
            )

    def move_straight(
        self,
        speed: float,
        wait_time: float = 0.0,
        wait_kwargs: dict[str, float] | None = None,
    ) -> None:
        """Move the chassis straight (forward/backward).

        Args:
            speed: Straight movement speed in m/s.
                   Positive/negative values move in opposite directions.
            wait_time: Time to maintain the movement in seconds.
            wait_kwargs: Additional parameters for wait behavior.
        """
        self.set_velocity(
            vx=speed,
            vy=0.0,
            wz=0.0,
            wait_time=wait_time,
            wait_kwargs=wait_kwargs,
            sequential_steering=True,
        )

    def move_sideways(
        self,
        speed: float,
        wait_time: float = 0.0,
        wait_kwargs: dict[str, float] | None = None,
    ) -> None:
        """Move the chassis sideways (strafe left/right).

        Args:
            speed: Sideways movement speed in m/s.
                   Positive: left, Negative: right
            wait_time: Time to maintain the movement in seconds.
            wait_kwargs: Additional parameters for wait behavior.
        """
        self.set_velocity(
            vx=0.0,
            vy=speed,
            wz=0.0,
            wait_time=wait_time,
            wait_kwargs=wait_kwargs,
            sequential_steering=True,
        )

    def turn(
        self,
        angular_speed: float,
        wait_time: float = 0.0,
        wait_kwargs: dict[str, float] | None = None,
        center: str = "base_center",
    ) -> None:
        """Turn the chassis in place.

        Args:
            angular_speed: Turning speed in rad/s.
                          Positive: counter-clockwise, Negative: clockwise
            wait_time: Time to maintain the turn in seconds.
            wait_kwargs: Additional parameters for wait behavior.
            center: Rotation center. Either "base_center" or "front_wheels_center".

        Raises:
            ValueError: If center is not a valid option.
        """
        if center == "base_center":
            self.set_velocity(
                vx=0.0,
                vy=0.0,
                wz=angular_speed,
                wait_time=wait_time,
                wait_kwargs=wait_kwargs,
                sequential_steering=True,
            )
        elif center == "front_wheels_center":
            speed = angular_speed * self._half_wheels_dist
            self.set_wheel_velocity(np.array([-speed, speed]), wait_time, wait_kwargs)
        else:
            raise ValueError(
                f"Invalid center: {center}. Must be 'base_center' or 'front_wheels_center'"
            )

    def set_steering_angle(
        self,
        steering_angle: float | np.ndarray,
        wait_time: float = 0.0,
        wait_kwargs: dict[str, float] | None = None,
    ) -> None:
        """Set the steering angles for both wheels.

        Args:
            steering_angle: Target steering angle(s) in radians. If float, same angle
                applied to both wheels. If array, [left_angle, right_angle].
            wait_time: Time to maintain the command in seconds.
            wait_kwargs: Additional parameters for wait behavior.
        """
        wait_kwargs = wait_kwargs or {}

        if isinstance(steering_angle, (int, float)):
            steering_angle = np.array([steering_angle, steering_angle])
        self.set_motion_state(steering_angle, np.zeros(2), wait_time, wait_kwargs)

    def set_wheel_velocity(
        self,
        wheel_velocity: float | np.ndarray,
        wait_time: float = 0.0,
        wait_kwargs: dict[str, float] | None = None,
    ) -> None:
        """Set the wheel velocities while maintaining current steering angles.

        Args:
            wheel_velocity: Target wheel velocity in m/s. If float, same velocity
                applied to both wheels. If array, [left_vel, right_vel].
            wait_time: Time to maintain the command in seconds.
            wait_kwargs: Additional parameters for wait behavior.
        """
        wait_kwargs = wait_kwargs or {}

        if isinstance(wheel_velocity, (int, float)):
            wheel_velocity = np.array([wheel_velocity, wheel_velocity])
        steering_angle = self.steering_angle
        self.set_motion_state(steering_angle, wheel_velocity, wait_time, wait_kwargs)

    def get_joint_pos(self, joint_id: list[int] | int | None = None) -> np.ndarray:
        """Get the joint positions of the chassis.

        Args:
            joint_id: Optional ID(s) of specific joints to query.
                0: left steering position
                1: left wheel position
                2: right steering position
                3: right wheel position

        Returns:
            Array of joint positions in the order:
            [left_steering_pos, left_wheel_pos, right_steering_pos, right_wheel_pos]
        """
        state = self._get_state()
        joint_pos = np.array(
            [
                state.left.steering_pos,
                state.left.wheel_pos,
                state.right.steering_pos,
                state.right.wheel_pos,
            ]
        )
        return self._extract_joint_info(joint_pos, joint_id=joint_id)

    def get_joint_pos_dict(
        self, joint_id: list[int] | int | None = None
    ) -> dict[str, float]:
        """Get the joint positions of the chassis as a dictionary.

        Args:
            joint_id: Optional ID(s) of specific joints to query.
                0: left steering position
                1: left wheel position
                2: right steering position
                3: right wheel position

        Returns:
            Dictionary mapping joint names to position values.
        """
        values = self.get_joint_pos(joint_id)
        return self._convert_to_dict(values, joint_id)

    def get_joint_state(self, joint_id: list[int] | int | None = None) -> np.ndarray:
        """Get current wheel states (left and right wheel).

        Args:
            joint_id: Optional ID(s) of specific joints to query.
                0: left wheel
                1: right wheel

        Returns:
            Numpy array containing:
                [:, 0]: Current steering positions in radians
                [:, 1]: Current wheel positions in radians
                [:, 2]: Current wheel velocities in m/s
        """
        state = self._get_state()
        wheel_states = np.array(
            [
                [state.left.steering_pos, state.left.wheel_pos, state.left.wheel_vel],
                [
                    state.right.steering_pos,
                    state.right.wheel_pos,
                    state.right.wheel_vel,
                ],
            ]
        )
        return self._extract_joint_info(wheel_states, joint_id=joint_id)

    def get_joint_state_dict(
        self, joint_id: list[int] | int | None = None
    ) -> dict[str, np.ndarray]:
        """Get current wheel states as a dictionary.

        Args:
            joint_id: Optional ID(s) of specific joints to query.
                0: left wheel
                1: right wheel

        Returns:
            Dictionary mapping joint names to arrays of [steering_pos, wheel_pos, wheel_vel]
        """
        values = self.get_joint_state(joint_id)
        return self._convert_to_dict(values, joint_id)

    def set_motion_state(
        self,
        steering_angle: float | np.ndarray,
        wheel_velocity: float | np.ndarray,
        wait_time: float = 0.0,
        wait_kwargs: dict[str, float] | None = None,
    ) -> None:
        """Send control commands to the chassis wheels.

        Sets steering positions and wheel velocities for both left and right wheels.
        Values can be provided either as numpy arrays or dictionaries mapping joint
        names to values. If using dictionaries, missing joints maintain current state.

        Args:
            steering_angle: Steering angle in radians. As array: [left_angle, right_angle]
                or dict mapping "L_wheel_j1"/"R_wheel_j1" to values.
            wheel_velocity: Wheel velocities in m/s. As array: [left_vel, right_vel]
                or dict mapping "L_wheel_j2"/"R_wheel_j2" to values.
            wait_time: Time to maintain the command in seconds.
            wait_kwargs: Additional parameters for wait behavior.
        """
        wait_kwargs = wait_kwargs or {}

        # Ensure wheel velocities are within limits
        wheel_velocity = np.clip(wheel_velocity, -self.max_vel, self.max_vel)

        if wait_time > 0.0:
            self._execute_timed_command(
                steering_angle, wheel_velocity, wait_time, wait_kwargs
            )
        else:
            self._send_single_command(steering_angle, wheel_velocity)

    def _apply_sequential_steering(
        self,
        target_steering: np.ndarray,
        target_wheel_speeds: np.ndarray,
        steering_tolerance: float,
        steering_wait_time: float,
        wait_time: float,
        wait_kwargs: dict[str, float],
    ) -> None:
        """Apply steering and velocity commands sequentially if needed.

        Args:
            target_steering: Target steering angles for both wheels.
            target_wheel_speeds: Target wheel speeds for both wheels.
            steering_tolerance: Angular tolerance to consider steering complete.
            steering_wait_time: Time to wait for steering adjustment.
            wait_time: Time to maintain final command.
            wait_kwargs: Additional parameters for wait behavior.
        """
        current_steering = self.steering_angle
        steering_error = np.linalg.norm(current_steering - target_steering)
        if steering_error > steering_tolerance:
            # Step 1: Adjust steering angles with zero wheel velocity
            self.set_motion_state(
                target_steering,
                np.zeros(2),
                wait_time=steering_wait_time,
                wait_kwargs=wait_kwargs,
            )

            # Step 2: Apply wheel velocities with the adjusted steering
            self.set_motion_state(
                target_steering, target_wheel_speeds, wait_time, wait_kwargs
            )
        else:
            # Steering is already close enough, apply both simultaneously
            self.set_motion_state(
                target_steering, target_wheel_speeds, wait_time, wait_kwargs
            )

    def _execute_timed_command(
        self,
        steering_angle: float | np.ndarray,
        wheel_velocity: float | np.ndarray,
        wait_time: float,
        wait_kwargs: dict[str, float],
    ) -> None:
        """Execute a command for a specified duration.

        Args:
            steering_angle: Target steering angles.
            wheel_velocity: Target wheel velocities.
            wait_time: Duration to maintain the command.
            wait_kwargs: Additional parameters including control frequency.
        """
        # Set default control frequency if not provided
        control_hz = wait_kwargs.get("control_hz", 50)
        rate_limiter = RateLimiter(control_hz)
        start_time = time.time()

        while time.time() - start_time < wait_time:
            self._send_single_command(steering_angle, wheel_velocity)
            rate_limiter.sleep()

    def _send_single_command(
        self,
        steering_angle: float | np.ndarray,
        wheel_velocity: float | np.ndarray,
    ) -> None:
        """Send a single control command to the chassis.

        Args:
            steering_angle: Target steering angles.
            wheel_velocity: Target wheel velocities.
        """
        control_msg = dexcontrol_msg_pb2.ChassisCommand()
        state = self._get_state()

        # Set steering positions and wheel velocities
        control_msg.left.steering_pos = _get_value(
            steering_angle, 0, "L_wheel_j1", state.left.steering_pos
        )
        control_msg.right.steering_pos = _get_value(
            steering_angle, 1, "R_wheel_j1", state.right.steering_pos
        )
        control_msg.left.wheel_vel = _get_value(
            wheel_velocity, 0, "L_wheel_j2", state.left.wheel_vel
        )
        control_msg.right.wheel_vel = _get_value(
            wheel_velocity, 1, "R_wheel_j2", state.right.wheel_vel
        )

        self._publish_control(control_msg)

    def _compute_wheel_control(
        self, wheel_velocity: np.ndarray, current_angle: float
    ) -> tuple[float, float]:
        """Compute steering angle and wheel speed for a given wheel velocity vector.

        Args:
            wheel_velocity: 2D velocity vector [vx, vy] for the wheel.
            current_angle: Current steering angle of the wheel in radians.

        Returns:
            Tuple of (steering_angle, wheel_speed) where:
            - steering_angle is in radians, constrained to [-170°, 170°]
            - wheel_speed is the magnitude of velocity (can be negative)
        """
        # Compute speed and direction
        wheel_speed = float(np.linalg.norm(wheel_velocity))

        # Handle zero velocity case
        if wheel_speed < self._min_velocity_threshold:
            return 0.0, 0.0

        # Compute base steering angle
        base_angle = float(-np.arctan2(wheel_velocity[1], wheel_velocity[0]))

        # Consider both possible solutions
        sol1 = (base_angle, wheel_speed)
        sol2 = (
            base_angle + np.pi if base_angle < 0 else base_angle - np.pi,
            -wheel_speed,
        )

        # Choose solution with angle closer to current angle
        angle1_diff = abs(sol1[0] - current_angle)
        angle2_diff = abs(sol2[0] - current_angle)

        # Select the better solution
        steering_angle, wheel_speed = sol1 if angle1_diff < angle2_diff else sol2

        # Ensure steering angle is within bounds
        steering_angle = float(
            np.clip(steering_angle, -self._max_steering_angle, self._max_steering_angle)
        )

        return steering_angle, wheel_speed


def _get_value(
    data: float | np.ndarray | dict[str, float],
    idx: int,
    key: str,
    fallback: float,
) -> float:
    """Helper function to extract values with fallback to current state.

    Args:
        data: Input scalar, array or dictionary containing values.
        idx: Index to access if data is array.
        key: Key to access if data is dictionary.
        fallback: Default value if key/index not found.

    Returns:
        Extracted float value.
    """
    if isinstance(data, dict):
        return float(data.get(key, fallback))
    if isinstance(data, (int, float)):
        return float(data)
    return float(data[idx])

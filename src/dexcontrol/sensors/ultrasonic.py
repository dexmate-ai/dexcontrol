# Copyright (C) 2025 Dexmate Inc.
#
# This software is dual-licensed:
#
# 1. GNU Affero General Public License v3.0 (AGPL-3.0)
#    See LICENSE-AGPL for details
#
# 2. Commercial License
#    For commercial licensing terms, contact: contact@dexmate.ai

"""Ultrasonic sensor implementations using DexComm subscribers.

This module provides ultrasonic sensor classes that use DexComm's
Raw API for distance measurements.
"""

import numpy as np
from dexcomm.codecs import UltrasonicStateCodec

from dexcontrol.core.shared_node import get_shared_node
from dexcontrol.core.subscription_policy import (
    SubscriptionPolicyManager,
    SubscriptionPolicyMixin,
)


class UltrasonicSensor(SubscriptionPolicyMixin):
    """Ultrasonic sensor using DexComm subscriber.

    This sensor provides distance measurements from ultrasonic sensors
    """

    def __init__(
        self,
        name,
        configs,
    ) -> None:
        """Initialize the ultrasonic sensor.

        Args:
            configs: Configuration for the ultrasonic sensor.
        """
        self._name = name
        self._node = get_shared_node()

        # Create the protobuf subscriber using our clean DexComm integration
        self._subscriber = self._node.create_subscriber(
            topic=configs.topic,
            callback=None,
            decoder=UltrasonicStateCodec.decode,
        )
        self._policy_manager = SubscriptionPolicyManager(
            self._subscriber, name=self._name
        )
        self._subcomponents: dict[str, object] = {}

    def shutdown(self) -> None:
        """Shutdown the ultrasonic sensor."""
        self._subscriber.shutdown()

    def is_active(self) -> bool:
        """Check if the ultrasonic sensor is actively receiving data.

        Returns:
            True if receiving data, False otherwise.
        """
        data = self._policy_manager.get_latest_managed()
        return data is not None

    def wait_for_active(self, timeout: float = 5.0) -> bool:
        """Wait for the ultrasonic sensor to start receiving data.

        Args:
            timeout: Maximum time to wait in seconds.

        Returns:
            True if sensor becomes active, False if timeout is reached.
        """
        msg = self._subscriber.wait_for_message(timeout)
        return msg is not None

    def get_obs(self) -> np.ndarray | None:
        """Get observation data for the ultrasonic sensor.

        This method provides a standardized observation format
        that can be used in robotics applications.

        Returns:
            Numpy array of distances in meters with shape (4,) in the order:
            [front_left, front_right, back_left, back_right].
        """
        msg = self._policy_manager.get_latest_managed()
        if msg is not None:
            data = msg.data
            obs = [
                data['front_left'],
                data['front_right'],
                data['back_left'],
                data['back_right'],
            ]
            return np.array(obs, dtype=np.float32)

        return None

    @property
    def name(self) -> str:
        """Get the sensor name.

        Returns:
            Sensor name string.
        """
        return self._name

# Copyright (C) 2025 Dexmate Inc.
#
# This software is dual-licensed:
#
# 1. GNU Affero General Public License v3.0 (AGPL-3.0)
#    See LICENSE-AGPL for details
#
# 2. Commercial License
#    For commercial licensing terms, contact: contact@dexmate.ai

"""Motor temperature sensor module.

Provides the TemperatureSensor class for reading motor/joint temperatures
from robot components via Zenoh communication.
"""

from typing import Any

from dexcomm.codecs import ComponentTemperatureCodec

from dexcontrol.core.component import RobotComponent


class TemperatureSensor(RobotComponent):
    """Motor temperature sensor for a robot component.

    Subscribes to a ComponentTemperature topic that publishes a nested map of
    joint names to temperature source readings in °C.

    Data format::

        {
            "L_arm_j1": {"motor": 35.0},
            "head_j1": {"motor": 30.0, "driver": 32.0},
            "torso_j1": {"motor": 27.0, "driver": 30.0},
        }

    Disabled by default (AUTO policy) — auto-idles after 5s of inactivity,
    auto-resumes on data access.
    """

    def __init__(self, name: str, state_sub_topic: str) -> None:
        """Initialize the temperature sensor.

        Args:
            name: Sensor name for identification and logging.
            state_sub_topic: Zenoh topic to subscribe to for temperature data.
        """
        super().__init__(
            name=name,
            state_sub_topic=state_sub_topic,
            state_decoder=ComponentTemperatureCodec.decode,
        )

    def get_temperatures(self) -> dict[str, dict[str, float]]:
        """Get all motor temperatures.

        Returns:
            Nested dictionary: joint name -> {source name -> temperature in °C}.
            Example::

                {
                    "L_arm_j1": {"motor": 35.0},
                    "head_j1": {"motor": 30.0, "driver": 32.0},
                }

        Raises:
            ServiceUnavailableError: If no temperature data is available.
        """
        state: dict[str, Any] = self.get_state()
        return dict(state.get("temperatures", {}))

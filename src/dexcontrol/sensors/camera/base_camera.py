# Copyright (C) 2025 Dexmate Inc.
#
# This software is dual-licensed:
#
# 1. GNU Affero General Public License v3.0 (AGPL-3.0)
#    See LICENSE-AGPL for details
#
# 2. Commercial License
#    For commercial licensing terms, contact: contact@dexmate.ai

"""Base camera sensor class with common functionality for all camera sensors."""

from abc import ABC, abstractmethod
from typing import Any

from loguru import logger

from dexcontrol.utils.comm_helper import query_json_service


class BaseCameraSensor(ABC):
    """Abstract base class for camera sensors.

    Provides common functionality for querying camera info, extracting RTC URLs,
    and managing camera metadata. Subclasses should implement the abstract methods
    for their specific camera type.
    """

    def __init__(self, name: str):
        """Initialize the base camera sensor.

        Args:
            name: Name of the camera sensor
        """
        self._name = name
        self._camera_info: dict[str, Any] | None = None

    @abstractmethod
    def shutdown(self) -> None:
        """Shutdown the camera sensor and release all resources."""
        pass

    @abstractmethod
    def is_active(self) -> bool:
        """Check if the camera sensor is actively receiving data."""
        pass

    @abstractmethod
    def wait_for_active(self, timeout: float = 5.0) -> bool:
        """Wait for the camera sensor to start receiving data."""
        pass

    def _query_camera_info(self, info_endpoint: str | None) -> None:
        """Query for camera metadata information.

        Args:
            info_endpoint: The endpoint to query for camera info
        """
        if not info_endpoint:
            logger.debug(
                f"'{self._name}': No info endpoint provided for camera info query."
            )
            return

        try:
            logger.debug(
                f"'{self._name}': Querying camera info from '{info_endpoint}'."
            )
            self._camera_info = query_json_service(info_endpoint, timeout=2.0)

            if self._camera_info:
                logger.info(f"'{self._name}': Successfully retrieved camera info.")
            else:
                logger.debug(
                    f"'{self._name}': No camera info available at '{info_endpoint}'."
                )

        except Exception as e:
            logger.debug(f"'{self._name}': Failed to query camera info: {e}")

    def _get_rtc_signaling_url(self, stream_name: str) -> str | None:
        """Extract RTC signaling URL from camera_info for a specific stream.

        Args:
            stream_name: Name of the stream (e.g., 'left_rgb', 'right_rgb', 'rgb')

        Returns:
            Signaling URL if available, None otherwise
        """
        if not self._camera_info:
            return None

        rtc_info = self._camera_info.get("rtc", {})
        streams = rtc_info.get("streams", {})
        stream_info = streams.get(stream_name, {})
        return stream_info.get("signaling_url")

    def _derive_info_endpoint_from_topic(self, topic: str) -> str | None:
        """Derive camera info endpoint from a topic.

        Args:
            topic: The camera topic, e.g.:
                - 'camera/head/left_rgb' -> 'camera/head/info'
                - 'camera/head/right_rgb' -> 'camera/head/info'
                - 'camera/head/depth' -> 'camera/head/info'
                - 'camera/base_left/rgb' -> 'camera/base_left/info'

        Returns:
            Derived info endpoint or None
        """
        if not topic:
            return None

        parts = topic.split("/")
        if len(parts) < 2:
            return None

        # Check if the last part is a stream identifier
        last_part = parts[-1].lower()
        stream_identifiers = ["left_rgb", "right_rgb", "depth", "rgb", "left", "right"]

        if last_part in stream_identifiers:
            # Remove the stream identifier and add 'info'
            # e.g., camera/head/left_rgb -> camera/head/info
            # e.g., camera/base_left/rgb -> camera/base_left/info
            return "/".join(parts[:-1]) + "/info"
        else:
            # The last part is not a stream identifier, just append '/info'
            # e.g., camera/head -> camera/head/info
            return topic + "/info"

    @property
    def name(self) -> str:
        """Get the sensor name.

        Returns:
            Sensor name string
        """
        return self._name

    @property
    def camera_info(self) -> dict[str, Any] | None:
        """Get the camera metadata information.

        Returns:
            Camera info dictionary if available, None otherwise.
            The dictionary typically contains intrinsic parameters,
            resolution, and other camera metadata.
        """
        return self._camera_info

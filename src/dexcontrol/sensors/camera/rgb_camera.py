# Copyright (C) 2025 Dexmate Inc.
#
# This software is dual-licensed:
#
# 1. GNU Affero General Public License v3.0 (AGPL-3.0)
#    See LICENSE-AGPL for details
#
# 2. Commercial License
#    For commercial licensing terms, contact: contact@dexmate.ai

"""RGB camera sensor implementation using RTC or DexComm subscriber."""

import numpy as np
from dexcomm import Subscriber
from loguru import logger

from dexcontrol.comm import create_camera_subscriber, create_rtc_camera_subscriber
from dexcontrol.comm.rtc import RTCSubscriber
from dexcontrol.config.sensors.cameras import RGBCameraConfig
from dexcontrol.sensors.camera.base_camera import BaseCameraSensor


class RGBCameraSensor(BaseCameraSensor):
    """RGB camera sensor that supports both RTC and standard DexComm subscribers.

    This sensor provides RGB image data from a camera. It can be configured to use
    either a high-performance RTC subscriber for real-time video streams or a
    standard DexComm subscriber for raw image topics. The mode is controlled by the
    `use_rtc` flag in the `RGBCameraConfig`.

    Both subscriber types provide the same API, making them interchangeable.
    """

    def __init__(
        self,
        configs: RGBCameraConfig,
    ) -> None:
        """Initialize the RGB camera sensor based on the provided configuration.

        Args:
            configs: Configuration object for the RGB camera sensor.
        """
        super().__init__(configs.name)
        self._configs = configs
        self._subscriber: RTCSubscriber | Subscriber | None = None

        subscriber_config = configs.subscriber_config.get("rgb", {})
        if not subscriber_config or not subscriber_config.get("enable", False):
            logger.info(f"RGBCameraSensor '{self._name}' is disabled in config.")
            return

        # Determine info endpoint and query camera info to potentially get RTC URLs
        subscriber_config = configs.subscriber_config.get("rgb", {})
        info_endpoint = self._determine_info_endpoint(subscriber_config)
        self._query_camera_info(info_endpoint)

        if configs.use_rtc:
            logger.info(f"'{self._name}': Using RTC subscriber.")

            # Try to get signaling URL from camera_info first
            rtc_url = self._get_rtc_signaling_url("rgb")
            if rtc_url:
                logger.info(f"'{self._name}': Creating RTC subscriber with direct URL.")
                self._subscriber = create_rtc_camera_subscriber(
                    signaling_url=rtc_url,
                )
            else:
                # Fallback to using info_key
                info_topic = subscriber_config.get("info_key") or subscriber_config.get("topic")
                if info_topic:
                    logger.info(f"'{self._name}': Creating RTC subscriber with info_key.")
                    self._subscriber = create_rtc_camera_subscriber(
                        info_topic=info_topic,
                    )
                else:
                    logger.warning(f"No RTC URL or info_key for '{self._name}' RTC mode.")
        else:
            logger.info(f"'{self._name}': Using standard subscriber.")
            topic = subscriber_config.get("topic")
            if topic:
                self._subscriber = create_camera_subscriber(
                    topic=topic,
                )
            else:
                logger.warning(
                    f"No 'topic' specified for '{self._name}' in non-RTC mode."
                )

        if self._subscriber is None:
            logger.warning(f"Failed to create subscriber for '{self._name}'.")

    def _determine_info_endpoint(self, subscriber_config: dict) -> str | None:
        """Determine the info endpoint for querying camera metadata.

        Args:
            subscriber_config: Subscriber configuration dict

        Returns:
            Info endpoint or None
        """
        if self._configs.use_rtc:
            return subscriber_config.get("info_key")
        else:
            topic = subscriber_config.get("topic")
            return self._derive_info_endpoint_from_topic(topic) if topic else None

    def shutdown(self) -> None:
        """Shutdown the camera sensor and release all resources."""
        if self._subscriber:
            self._subscriber.shutdown()
            logger.info(f"'{self._name}' sensor shut down.")

    def is_active(self) -> bool:
        """Check if the camera sensor is actively receiving data.

        Returns:
            True if the subscriber exists and is receiving data, False otherwise.
        """
        data = self._subscriber.get_latest()
        return data is not None

    def wait_for_active(self, timeout: float = 5.0) -> bool:
        """Wait for the camera sensor to start receiving data.

        Args:
            timeout: Maximum time to wait in seconds.

        Returns:
            True if the sensor becomes active within the timeout, False otherwise.
        """
        if not self._subscriber:
            logger.warning(f"'{self._name}': Cannot wait, no subscriber initialized.")
            return False
        msg = self._subscriber.wait_for_message(timeout)
        return msg is not None

    def get_obs(self) -> np.ndarray | None:
        """Get the latest observation (RGB image) from the sensor.

        Returns:
            The latest RGB image as a numpy array (HxWxC) if available, otherwise None.
        """
        data = self._subscriber.get_latest()
        return data if data is not None else None

    @property
    def height(self) -> int:
        """Get the height of the camera image.

        Returns:
            Height of the camera image.
        """
        image = self.get_obs()
        if image is None:
            return 0
        return image.shape[0]

    @property
    def width(self) -> int:
        """Get the width of the camera image.

        Returns:
            Width of the camera image.
        """
        image = self.get_obs()
        if image is None:
            return 0
        return image.shape[1]

    @property
    def resolution(self) -> tuple[int, int]:
        """Get the resolution of the camera image.

        Returns:
            Resolution of the camera image.
        """
        image = self.get_obs()
        if image is None:
            return 0, 0
        return image.shape[0], image.shape[1]

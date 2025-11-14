# Copyright (C) 2025 Dexmate Inc.
#
# This software is dual-licensed:
#
# 1. GNU Affero General Public License v3.0 (AGPL-3.0)
#    See LICENSE-AGPL for details
#
# 2. Commercial License
#    For commercial licensing terms, contact: contact@dexmate.ai

"""ZED camera sensor implementation using RTC or DexComm subscribers for RGB and depth."""

import time
from typing import Any

import numpy as np
from loguru import logger

from dexcontrol.comm import (
    create_camera_subscriber,
    create_depth_subscriber,
    create_rtc_camera_subscriber,
)
from dexcontrol.config.sensors.cameras import ZedCameraConfig
from dexcontrol.sensors.camera.base_camera import BaseCameraSensor


class ZedCameraSensor(BaseCameraSensor):
    """ZED camera sensor for multi-stream (RGB, Depth) data acquisition.

    This sensor manages left RGB, right RGB, and depth data streams from a ZED
    camera. It can be configured to use high-performance RTC subscribers for RGB
    streams (`use_rtc=True`) or standard DexComm subscribers. Both types provide
    the same API interface, making them interchangeable.
    """

    def __init__(
        self,
        configs: ZedCameraConfig,
    ) -> None:
        """Initialize the ZED camera sensor and its subscribers.

        Args:
            configs: Configuration object for the ZED camera.
        """
        super().__init__(configs.name)
        self._configs = configs
        self._subscribers: dict[
            str, Any | None
        ] = {}  # Will hold either RTCSubscriberWrapper or Subscriber

        self._create_subscribers()

    def _create_subscriber(
        self, stream_name: str, stream_config: dict[str, Any]
    ) -> Any | None:
        """Factory method to create a subscriber based on stream type and config."""
        try:
            if not stream_config.get("enable", False):
                logger.info(f"'{self._name}': Stream '{stream_name}' is disabled.")
                return None

            # Create Depth subscriber
            if stream_name == "depth":
                topic = stream_config.get("topic")
                if not topic:
                    logger.warning(f"'{self._name}': No 'topic' for depth stream.")
                    return None
                logger.info(f"'{self._name}': Creating depth subscriber.")
                # Use new DexComm integration
                return create_depth_subscriber(
                    topic=topic,
                )

            # Create RGB subscriber (RTC or DexComm)
            if self._configs.use_rtc:
                # Check if we have RTC info from camera_info
                rtc_url = self._get_rtc_signaling_url(stream_name)
                if rtc_url:
                    logger.info(
                        f"'{self._name}': Creating RTC subscriber for '{stream_name}' with direct URL."
                    )
                    return create_rtc_camera_subscriber(
                        signaling_url=rtc_url,
                    )
                else:
                    # Fallback to querying via info_key
                    info_key = stream_config.get("info_key")
                    if not info_key:
                        logger.warning(
                            f"'{self._name}': No RTC URL or info_key for stream '{stream_name}'."
                        )
                        return None
                    logger.info(
                        f"'{self._name}': Creating RTC subscriber for '{stream_name}' with info_key."
                    )
                    return create_rtc_camera_subscriber(
                        info_topic=info_key,
                    )
            else:
                topic = stream_config.get("topic")
                if not topic:
                    logger.warning(
                        f"'{self._name}': No 'topic' for Zenoh stream '{stream_name}'."
                    )
                    return None
                logger.info(
                    f"'{self._name}': Creating RGB subscriber for '{stream_name}'."
                )
                # Use new DexComm integration
                return create_camera_subscriber(
                    topic=topic,
                )

        except Exception as e:
            logger.error(
                f"Error creating subscriber for '{self._name}/{stream_name}': {e}"
            )
            return None

    def _create_subscribers(self) -> None:
        """Create subscribers for all configured camera streams."""
        subscriber_config = self._configs.subscriber_config

        # Determine info endpoint for camera metadata
        info_endpoint = self._determine_info_endpoint(subscriber_config)

        # Query camera info first to potentially get RTC URLs
        self._query_camera_info(info_endpoint)

        stream_definitions = {
            "left_rgb": subscriber_config.get("left_rgb", {}),
            "right_rgb": subscriber_config.get("right_rgb", {}),
            "depth": subscriber_config.get("depth", {}),
        }

        for name, config in stream_definitions.items():
            self._subscribers[name] = self._create_subscriber(name, config)

    def _determine_info_endpoint(self, subscriber_config: dict) -> str | None:
        """Determine the info endpoint for querying camera metadata.

        Args:
            subscriber_config: Subscriber configuration dict

        Returns:
            Info endpoint or None
        """
        # Try to find the first enabled RGB stream to get the base info endpoint
        for stream_name in ["left_rgb", "right_rgb"]:
            stream_config = subscriber_config.get(stream_name, {})
            if stream_config.get("enable", False):
                if self._configs.use_rtc:
                    info_key = stream_config.get("info_key")
                    if info_key:
                        return info_key
                else:
                    topic = stream_config.get("topic")
                    if topic:
                        return self._derive_info_endpoint_from_topic(topic)
        return None

    def shutdown(self) -> None:
        """Shutdown all active subscribers for the camera sensor."""
        logger.info(f"Shutting down all subscribers for '{self._name}'.")
        for stream_name, subscriber in self._subscribers.items():
            if subscriber:
                try:
                    subscriber.shutdown()
                    logger.debug(
                        f"'{self._name}': Subscriber '{stream_name}' shut down."
                    )
                except Exception as e:
                    logger.error(
                        f"Error shutting down '{stream_name}' subscriber for '{self._name}': {e}"
                    )
        logger.info(f"'{self._name}' sensor shut down.")

    def is_active(self) -> bool:
        """Check if any of the camera's subscribers are actively receiving data.

        Returns:
            True if at least one subscriber is active, False otherwise.
        """
        return any(
            sub.get_latest() is not None
            for sub in self._subscribers.values()
            if sub is not None
        )

    def is_stream_active(self, stream_name: str) -> bool:
        """Check if a specific camera stream is actively receiving data.

        Args:
            stream_name: The name of the stream (e.g., 'left_rgb', 'depth').

        Returns:
            True if the specified stream's subscriber is active, False otherwise.
        """
        subscriber = self._subscribers.get(stream_name)
        return subscriber.get_latest() is not None if subscriber else False

    def wait_for_active(self, timeout: float = 5.0, require_all: bool = False) -> bool:
        """Wait for camera streams to become active.

        Args:
            timeout: Maximum time to wait in seconds for each subscriber.
            require_all: If True, waits for all enabled streams to become active.
                         If False, waits for at least one stream to become active.

        Returns:
            True if the condition is met within the timeout, False otherwise.
        """
        enabled_subscribers = [s for s in self._subscribers.values() if s is not None]
        if not enabled_subscribers:
            logger.warning(f"'{self._name}': No subscribers enabled, cannot wait.")
            return True  # No subscribers to wait for

        if require_all:
            for sub in enabled_subscribers:
                if not sub.wait_for_message(timeout):
                    logger.warning(
                        f"'{self._name}': Timed out waiting for subscriber '{sub.name}'."
                    )
                    return False
            logger.info(f"'{self._name}': All enabled streams are active.")
            return True
        else:
            start_time = time.time()
            while time.time() - start_time < timeout:
                if self.is_active():
                    logger.info(f"'{self._name}': At least one stream is active.")
                    return True
                time.sleep(0.1)
            logger.warning(
                f"'{self._name}': Timed out waiting for any stream to become active."
            )
            return False

    def get_obs(
        self, obs_keys: list[str] | None = None, include_timestamp: bool = False
    ) -> dict[str, np.ndarray | None]:
        """Get the latest observation data from specified camera streams.

        Args:
            obs_keys: A list of stream names to retrieve data from (e.g.,
                      ['left_rgb', 'depth']). If None, retrieves data from all
                      enabled streams.
            include_timestamp: If True, includes the timestamp in the observation data.
                                The timestamp data is not available for RTC streams.

        Returns:
            A dictionary mapping stream names to their latest image data with timestamp. The
            image is a numpy array (HxWxC for RGB, HxW for depth) or None if
            no data is available for that stream. If include_timestamp is True,
            the value in the dictionary is a tuple with the image and timestamp.
        """
        keys_to_fetch = obs_keys or self.available_streams
        obs_out = {}
        for key in keys_to_fetch:
            subscriber = self._subscribers.get(key)
            data = subscriber.get_latest() if subscriber else None

            # DexComm returns dict with 'data' and 'timestamp' keys when timestamp is present
            has_timestamp = isinstance(data, dict) and "timestamp" in data

            if include_timestamp:
                # Always return a consistent structure
                if has_timestamp:
                    obs_out[key] = {
                        "data": data.get("data") if isinstance(data, dict) else None,
                        "timestamp": data.get("timestamp")
                        if isinstance(data, dict)
                        else None,
                    }
                else:
                    obs_out[key] = {"data": data, "timestamp": None}
            else:
                if has_timestamp:
                    # Extract payload when timestamp wrapper is present
                    obs_out[key] = data.get("data") if isinstance(data, dict) else None
                else:
                    obs_out[key] = data
        return obs_out

    def get_left_rgb(self) -> np.ndarray | None:
        """Get the latest image from the left RGB stream.

        Returns:
            The latest left RGB image as a numpy array, or None if not available.
        """
        subscriber = self._subscribers.get("left_rgb")
        return subscriber.get_latest() if subscriber else None

    def get_right_rgb(self) -> np.ndarray | None:
        """Get the latest image from the right RGB stream.

        Returns:
            The latest right RGB image as a numpy array, or None if not available.
        """
        subscriber = self._subscribers.get("right_rgb")
        return subscriber.get_latest() if subscriber else None

    def get_depth(self) -> np.ndarray | None:
        """Get the latest image from the depth stream.

        The depth data is returned as a numpy array with values in meters.

        Returns:
            The latest depth image as a numpy array, or None if not available.
        """
        subscriber = self._subscribers.get("depth")
        return subscriber.get_latest() if subscriber else None

    @property
    def available_streams(self) -> list:
        """Get list of available stream names.

        Returns:
            List of stream names that have active subscribers.
        """
        return [name for name, sub in self._subscribers.items() if sub is not None]

    @property
    def active_streams(self) -> list:
        """Get list of currently active stream names.

        Returns:
            List of stream names that are currently receiving data.
        """
        return [
            name
            for name, sub in self._subscribers.items()
            if sub and sub.get_latest() is not None
        ]

    @property
    def height(self) -> dict[str, int]:
        """Get the height of the camera image.

        Returns:
            Height of the camera image.
        """
        images = self.get_obs()
        return {
            name: image.shape[0] if image is not None else 0
            for name, image in images.items()
        }

    @property
    def width(self) -> dict[str, int]:
        """Get the width of the camera image.

        Returns:
            Width of the camera image.
        """
        images = self.get_obs()
        return {
            name: image.shape[1] if image is not None else 0
            for name, image in images.items()
        }

    @property
    def resolution(self) -> dict[str, tuple[int, int]]:
        """Get the resolution of the camera image.

        Returns:
            Resolution of the camera image.
        """
        images = self.get_obs()
        return {
            name: (image.shape[0], image.shape[1]) if image is not None else (0, 0)
            for name, image in images.items()
        }

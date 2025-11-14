# Copyright (C) 2025 Dexmate Inc.
#
# This software is dual-licensed:
#
# 1. GNU Affero General Public License v3.0 (AGPL-3.0)
#    See LICENSE-AGPL for details
#
# 2. Commercial License
#    For commercial licensing terms, contact: contact@dexmate.ai

"""Core subscriber utilities using DexComm's Raw API.

Simple factory functions for creating DexComm subscribers with
DexControl's default configurations.
"""

from pathlib import Path
from typing import Any, Callable

from dexcomm import BufferedSubscriber, Subscriber, wait_for_message
from dexcomm.serialization import deserialize_auto
from loguru import logger

from dexcontrol.utils.comm_helper import get_zenoh_config_path
from dexcontrol.utils.os_utils import resolve_key_name

# Import specialized deserializers
try:
    from dexcomm.serialization import deserialize_depth, deserialize_image

    IMAGE_DESERIALIZERS_AVAILABLE = True
except ImportError:
    logger.debug("DexComm image deserializers not available")
    deserialize_image = None
    deserialize_depth = None
    IMAGE_DESERIALIZERS_AVAILABLE = False

try:
    from dexcomm.serialization import deserialize_imu

    IMU_DESERIALIZER_AVAILABLE = True
except ImportError:
    logger.debug("DexComm IMU deserializer not available")
    deserialize_imu = None
    IMU_DESERIALIZER_AVAILABLE = False

try:
    from dexcomm.serialization.lidar import deserialize_lidar_2d

    LIDAR_DESERIALIZER_AVAILABLE = True
except ImportError:
    logger.debug("DexComm LiDAR deserializer not available")
    deserialize_lidar_2d = None
    LIDAR_DESERIALIZER_AVAILABLE = False


def create_subscriber(
    topic: str,
    callback: Callable[[Any], None] | None = None,
    buffer_size: int = 1,
    deserializer: Callable[[bytes], Any] | None = deserialize_auto,
    config: Path | None = None,
) -> Subscriber:
    """Create a DexComm Subscriber with DexControl defaults.

    Simple wrapper that handles topic resolution and default configuration.

    Args:
        topic: Topic to subscribe to (will be resolved with robot namespace)
        callback: Optional callback function for incoming messages
        buffer_size: Number of messages to buffer (default 1 for latest only)
        deserializer: Deserialization function (default: auto-detect)
        config: Optional config path (default: use robot's config)

    Returns:
        DexComm Subscriber instance

    Example:
        ```python
        # Simple subscription
        sub = create_subscriber("sensors/imu")
        data = sub.get_latest()

        # With callback
        sub = create_subscriber(
            "robot/state",
            callback=lambda msg: print(f"State: {msg}")
        )
        ```
    """
    full_topic = resolve_key_name(topic)
    config_path = config or get_zenoh_config_path()

    logger.debug(f"Creating subscriber for topic: {full_topic}")

    return Subscriber(
        topic=full_topic,
        callback=callback,
        deserializer=deserializer,
        buffer_size=buffer_size,
        config=config_path,
    )


def create_buffered_subscriber(
    topic: str,
    callback: Callable[[Any], None] | None = None,
    buffer_size: int = 100,
    deserializer: Callable[[bytes], Any] | None = deserialize_auto,
    config: Path | None = None,
) -> BufferedSubscriber:
    """Create a DexComm BufferedSubscriber for data collection.

    Args:
        topic: Topic to subscribe to
        callback: Optional callback function
        buffer_size: Maximum messages to buffer (default 100)
        deserializer: Deserialization function
        config: Optional config path

    Returns:
        DexComm BufferedSubscriber instance

    Example:
        ```python
        # Collect sensor data
        sub = create_buffered_subscriber("sensors/lidar", buffer_size=500)
        # ... wait for data ...
        all_scans = sub.get_buffer()
        ```
    """
    full_topic = resolve_key_name(topic)
    config_path = config or get_zenoh_config_path()

    logger.debug(f"Creating buffered subscriber for topic: {full_topic}")

    return BufferedSubscriber(
        topic=full_topic,
        callback=callback,
        deserializer=deserializer,
        buffer_size=buffer_size,
        config=config_path,
    )


def create_camera_subscriber(
    topic: str,
    callback: Callable[[Any], None] | None = None,
    buffer_size: int = 1,
) -> Subscriber:
    """Create a subscriber for RGB camera data.

    Uses DexComm's image deserializer for efficient image decompression.

    Args:
        topic: Camera topic
        callback: Optional callback for images
        buffer_size: Number of images to buffer

    Returns:
        Subscriber configured for RGB images
    """
    deserializer = (
        deserialize_image if IMAGE_DESERIALIZERS_AVAILABLE else deserialize_auto
    )

    return create_subscriber(
        topic=topic,
        callback=callback,
        buffer_size=buffer_size,
        deserializer=deserializer,
    )


def create_depth_subscriber(
    topic: str,
    callback: Callable[[Any], None] | None = None,
    buffer_size: int = 1,
) -> Subscriber:
    """Create a subscriber for depth camera data.

    Uses DexComm's depth deserializer for 16-bit depth images.

    Args:
        topic: Depth camera topic
        callback: Optional callback for depth images
        buffer_size: Number of images to buffer

    Returns:
        Subscriber configured for depth images
    """
    deserializer = (
        deserialize_depth if IMAGE_DESERIALIZERS_AVAILABLE else deserialize_auto
    )

    return create_subscriber(
        topic=topic,
        callback=callback,
        buffer_size=buffer_size,
        deserializer=deserializer,
    )


def create_imu_subscriber(
    topic: str,
    callback: Callable[[Any], None] | None = None,
    buffer_size: int = 1,
) -> Subscriber:
    """Create a subscriber for IMU data.

    Uses DexComm's IMU deserializer for 6-DOF/9-DOF IMU data.

    Args:
        topic: IMU topic
        callback: Optional callback for IMU data
        buffer_size: Number of samples to buffer

    Returns:
        Subscriber configured for IMU data
    """
    deserializer = deserialize_imu if IMU_DESERIALIZER_AVAILABLE else deserialize_auto

    return create_subscriber(
        topic=topic,
        callback=callback,
        buffer_size=buffer_size,
        deserializer=deserializer,
    )


def create_lidar_subscriber(
    topic: str,
    callback: Callable[[Any], None] | None = None,
    buffer_size: int = 1,
) -> Subscriber:
    """Create a subscriber for 2D LiDAR data.

    Uses DexComm's LiDAR deserializer for laser scan data.

    Args:
        topic: LiDAR topic
        callback: Optional callback for scan data
        buffer_size: Number of scans to buffer

    Returns:
        Subscriber configured for LiDAR data
    """
    deserializer = (
        deserialize_lidar_2d if LIDAR_DESERIALIZER_AVAILABLE else deserialize_auto
    )

    return create_subscriber(
        topic=topic,
        callback=callback,
        buffer_size=buffer_size,
        deserializer=deserializer,
    )


def create_generic_subscriber(
    topic: str,
    callback: Callable[[Any], None] | None = None,
    buffer_size: int = 1,
    raw_bytes: bool = False,
) -> Subscriber:
    """Create a generic subscriber with auto-detection or raw bytes.

    Args:
        topic: Topic to subscribe to
        callback: Optional callback
        buffer_size: Number of messages to buffer
        raw_bytes: If True, return raw bytes without deserialization

    Returns:
        Generic subscriber instance
    """
    deserializer = None if raw_bytes else deserialize_auto

    return create_subscriber(
        topic=topic,
        callback=callback,
        buffer_size=buffer_size,
        deserializer=deserializer,
    )


def quick_subscribe(topic: str, callback: Callable[[Any], None]) -> Subscriber:
    """Quick one-liner to subscribe with a callback.

    Args:
        topic: Topic to subscribe to
        callback: Callback function

    Returns:
        Subscriber instance

    Example:
        ```python
        sub = quick_subscribe("robot/state", lambda msg: print(msg))
        ```
    """
    return create_subscriber(topic, callback=callback)


def wait_for_any_message(topic: str, timeout: float = 5.0) -> Any | None:
    """Wait for a message on a topic.

    Convenience wrapper around DexComm's wait_for_message.

    Args:
        topic: Topic to wait on
        timeout: Maximum wait time in seconds

    Returns:
        Received message or None if timeout

    Example:
        ```python
        state = wait_for_any_message("robot/state", timeout=2.0)
        if state:
            print(f"Robot is in state: {state}")
        ```
    """
    full_topic = resolve_key_name(topic)
    return wait_for_message(
        topic=full_topic,
        timeout=timeout,
        config=get_zenoh_config_path(),
    )

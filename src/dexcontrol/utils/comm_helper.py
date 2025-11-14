# Copyright (C) 2025 Dexmate Inc.
#
# This software is dual-licensed:
#
# 1. GNU Affero General Public License v3.0 (AGPL-3.0)
#    See LICENSE-AGPL for details
#
# 2. Commercial License
#    For commercial licensing terms, contact: contact@dexmate.ai

"""Communication helper utilities for DexControl using DexComm.

This module provides simple helper functions for DexControl's communication
needs using the DexComm library's Raw API.
"""

import json
import time
from pathlib import Path

from dexcomm import ZenohConfig, call_service
from dexcomm.serialization import deserialize_json
from loguru import logger

import dexcontrol
from dexcontrol.utils.os_utils import resolve_key_name


def get_robot_config() -> ZenohConfig:
    """Get DexComm configuration for robot communication.

    This checks for the robot's Zenoh configuration file and creates
    appropriate DexComm config.

    Returns:
        ZenohConfig instance configured for the robot.
    """
    config_path = dexcontrol.COMM_CFG_PATH

    if config_path and config_path != Path("/tmp/no_config") and config_path.exists():
        logger.debug(f"Loading config from: {config_path}")
        return ZenohConfig.from_file(config_path)
    else:
        logger.debug("No config file found, using default peer mode")
        return ZenohConfig.default_peer()


def get_zenoh_config_path() -> Path | None:
    """Get robot config only if not using default SessionManager.

    DexComm's SessionManager will automatically use the config from
    environment variables if available, so we only need to provide
    config explicitly if we have a specific robot config file.

    Returns:
        path to the config file
    """
    config_path = dexcontrol.COMM_CFG_PATH
    return config_path


def query_json_service(
    topic: str,
    timeout: float = 2.0,
    max_retries: int = 1,
    retry_delay: float = 0.5,
) -> dict | None:
    """Query for JSON information using DexComm with retry logic.

    Args:
        topic: Topic to query (will be resolved with robot namespace).
        timeout: Maximum time to wait for a response in seconds.
        max_retries: Maximum number of retry attempts.
        retry_delay: Initial delay between retries (doubles each retry).

    Returns:
        Dictionary containing the parsed JSON response if successful, None otherwise.
    """
    resolved_topic = resolve_key_name(topic)
    logger.debug(f"Querying topic: {resolved_topic}")

    current_delay = retry_delay
    for attempt in range(max_retries + 1):
        try:
            data = call_service(
                resolved_topic,
                timeout=timeout,
                config=get_zenoh_config_path(),
                request_serializer=None,
                response_deserializer=deserialize_json,
            )

            if data:
                logger.debug(f"Successfully received JSON data from {resolved_topic}")
                return data

        except json.JSONDecodeError as e:
            logger.warning(f"Failed to parse JSON response from {resolved_topic}: {e}")
        except Exception as e:
            logger.warning(
                f"Query failed for {resolved_topic} (attempt {attempt + 1}/{max_retries + 1}): {e}"
            )

        if attempt < max_retries:
            logger.debug(f"Retrying in {current_delay:.1f} seconds...")
            time.sleep(current_delay)
            current_delay *= 2  # Exponential backoff

    logger.error(f"Failed to query {resolved_topic} after {max_retries + 1} attempts")
    return None

# Copyright (C) 2025 Dexmate Inc.
#
# This software is dual-licensed:
#
# 1. GNU Affero General Public License v3.0 (AGPL-3.0)
#    See LICENSE-AGPL for details
#
# 2. Commercial License
#    For commercial licensing terms, contact: contact@dexmate.ai

"""Miscellaneous robot components module.

This module provides classes for various auxiliary robot components such as Battery,
EStop (emergency stop), ServerLogSubscriber, and UltraSonicSensor.
"""

import json
import os
import threading
import time
from typing import Any, TypeVar, cast

from dexcomm import Subscriber, call_service
from dexcomm.serialization.protobuf import control_msg_pb2, control_query_pb2
from google.protobuf.message import Message
from loguru import logger
from rich.console import Console
from rich.table import Table

from dexcontrol.config.core import BatteryConfig, EStopConfig, HeartbeatConfig
from dexcontrol.core.component import RobotComponent
from dexcontrol.utils.comm_helper import get_zenoh_config_path
from dexcontrol.utils.os_utils import resolve_key_name

# Type variable for Message subclasses
M = TypeVar("M", bound=Message)


class Battery(RobotComponent):
    """Battery component that monitors and displays battery status information.

    This class provides methods to monitor battery state including voltage, current,
    temperature and power consumption. It can display the information in either a
    formatted rich table or plain text format.

    Attributes:
        _console: Rich console instance for formatted output.
        _monitor_thread: Background thread for battery monitoring.
        _shutdown_event: Event to signal thread shutdown.
    """

    def __init__(self, configs: BatteryConfig) -> None:
        """Initialize the Battery component.

        Args:
            configs: Battery configuration containing subscription topics.
        """
        super().__init__(configs.state_sub_topic, control_msg_pb2.BMSState)
        self._console = Console()
        self._shutdown_event = threading.Event()
        self._monitor_thread = threading.Thread(
            target=self._battery_monitor, daemon=True
        )
        self._monitor_thread.start()

    def _battery_monitor(self) -> None:
        """Background thread that periodically checks battery level and warns if low."""
        while not self._shutdown_event.is_set():
            try:
                if self.is_active():
                    battery_level = self.get_status()["percentage"]
                    if battery_level < 20:
                        logger.warning(
                            f"Battery level is low ({battery_level:.1f}%). "
                            "Please charge the battery."
                        )
            except Exception as e:
                logger.debug(f"Battery monitor error: {e}")

            # Check every 30 seconds (low frequency)
            self._shutdown_event.wait(30.0)

    def get_status(self) -> dict[str, float]:
        """Gets the current battery state information.

        Returns:
            Dictionary containing battery metrics including:
                - percentage: Battery charge level (0-100)
                - temperature: Battery temperature in Celsius
                - current: Current draw in Amperes
                - voltage: Battery voltage
                - power: Power consumption in Watts
        """
        state = self._get_state()
        if state is None:
            return {
                "percentage": 0.0,
                "temperature": 0.0,
                "current": 0.0,
                "voltage": 0.0,
                "power": 0.0,
            }
        return {
            "percentage": float(state.percentage),
            "temperature": float(state.temperature),
            "current": float(state.current),
            "voltage": float(state.voltage),
            "power": float(state.current * state.voltage),
        }

    def show(self) -> None:
        """Displays the current battery status as a formatted table with color indicators."""
        state = self._get_state()

        table = Table(title="Battery Status")
        table.add_column("Parameter", style="cyan")
        table.add_column("Value")

        if state is None:
            table.add_row("Status", "[red]No battery data available[/]")
            self._console.print(table)
            return

        battery_style = self._get_battery_level_style(state.percentage)
        table.add_row("Battery Level", f"[{battery_style}]{state.percentage:.1f}%[/]")

        temp_style = self._get_temperature_style(state.temperature)
        table.add_row("Temperature", f"[{temp_style}]{state.temperature:.1f}°C[/]")

        power = state.current * state.voltage
        power_style = self._get_power_style(power)
        table.add_row(
            "Power Consumption",
            f"[{power_style}]{power:.2f}W[/] ([blue]{state.current:.2f}A[/] "
            f"× [blue]{state.voltage:.2f}V[/])",
        )

        self._console.print(table)

    def shutdown(self) -> None:
        """Shuts down the battery component and stops monitoring thread."""
        self._shutdown_event.set()
        if self._monitor_thread and self._monitor_thread.is_alive():
            self._monitor_thread.join(timeout=2.0)  # Extended timeout
            if self._monitor_thread.is_alive():
                logger.warning("Battery monitor thread did not terminate cleanly")
        super().shutdown()

    @staticmethod
    def _get_battery_level_style(percentage: float) -> str:
        """Returns the appropriate style based on battery percentage.

        Args:
            percentage: Battery charge level (0-100).

        Returns:
            Rich text style string for color formatting.
        """
        if percentage < 30:
            return "bold red"
        elif percentage < 60:
            return "bold yellow"
        else:
            return "bold dark_green"

    @staticmethod
    def _get_temperature_style(temperature: float) -> str:
        """Returns the appropriate style based on temperature value.

        Args:
            temperature: Battery temperature in Celsius.

        Returns:
            Rich text style string for color formatting.
        """
        if temperature < -1:
            return "bold red"  # Too cold
        elif temperature <= 30:
            return "bold dark_green"  # Normal range
        elif temperature <= 38:
            return "bold orange"  # Getting warm
        else:
            return "bold red"  # Too hot

    @staticmethod
    def _get_power_style(power: float) -> str:
        """Returns the appropriate style based on power consumption.

        Args:
            power: Power consumption in Watts.

        Returns:
            Rich text style string for color formatting.
        """
        if power < 200:
            return "bold dark_green"
        elif power <= 500:
            return "bold orange"
        else:
            return "bold red"


class EStop(RobotComponent):
    """EStop component that monitors and controls emergency stop functionality.

    This class provides methods to monitor EStop state and activate/deactivate
    the software emergency stop.

    Attributes:
        _estop_query_name: Zenoh query name for setting EStop state.
        _monitor_thread: Background thread for EStop monitoring.
        _shutdown_event: Event to signal thread shutdown.
    """

    def __init__(
        self,
        configs: EStopConfig,
    ) -> None:
        """Initialize the EStop component.

        Args:
            configs: EStop configuration containing subscription topics.
        """
        self._enabled = configs.enabled
        super().__init__(configs.state_sub_topic, control_msg_pb2.EStopState)
        self._estop_query_name = configs.estop_query_name
        if not self._enabled:
            logger.warning("EStop monitoring is DISABLED via configuration")
            return
        self._shutdown_event = threading.Event()
        self._monitor_thread = threading.Thread(target=self._estop_monitor, daemon=True)
        self._monitor_thread.start()

    def _estop_monitor(self) -> None:
        """Background thread that continuously monitors EStop button state."""
        while not self._shutdown_event.is_set():
            try:
                if self.is_active() and self.is_button_pressed():
                    logger.critical(
                        "E-STOP BUTTON PRESSED! Exiting program immediately."
                    )
                    # Don't call self.shutdown() here as it would try to join the current thread
                    # os._exit(1) will terminate the entire process immediately
                    os._exit(1)
            except Exception as e:
                logger.debug(f"EStop monitor error: {e}")

            # Check every 100ms for responsive emergency stop
            self._shutdown_event.wait(0.1)

    def _set_estop(self, enable: bool) -> None:
        """Sets the software emergency stop (E-Stop) state of the robot.

        This controls the software E-Stop, which is separate from the physical button
        on the robot. The robot will stop if either the software or hardware E-Stop is
        activated.

        Args:
            enable: If True, activates the software E-Stop. If False, deactivates it.
        """
        query_msg = control_query_pb2.SetEstop(enable=enable)
        call_service(
            resolve_key_name(self._estop_query_name),
            request=query_msg,
            timeout=0.05,
            config=get_zenoh_config_path(),
            request_serializer=lambda x: x.SerializeToString(),
            response_deserializer=None,
        )
        logger.info(f"Set E-Stop to {enable}")

    def get_status(self) -> dict[str, bool]:
        """Gets the current EStop state information.

        Returns:
            Dictionary containing EStop metrics including:
                - button_pressed: EStop button pressed
                - software_estop_enabled: Software EStop enabled
        """
        state = self._get_state()
        state = cast(control_msg_pb2.EStopState, state)
        if state is None:
            return {
                "button_pressed": False,
                "software_estop_enabled": False,
            }
        button_pressed = (
            state.left_button_pressed
            or state.right_button_pressed
            or state.waist_button_pressed
            or state.wireless_button_pressed
        )
        return {
            "button_pressed": button_pressed,
            "software_estop_enabled": state.software_estop_enabled,
        }

    def is_button_pressed(self) -> bool:
        """Checks if the EStop button is pressed."""
        state = self._get_state()
        state = cast(control_msg_pb2.EStopState, state)
        button_pressed = (
            state.left_button_pressed
            or state.right_button_pressed
            or state.waist_button_pressed
            or state.wireless_button_pressed
        )
        return button_pressed

    def is_software_estop_enabled(self) -> bool:
        """Checks if the software EStop is enabled."""
        state = self._get_state()
        state = cast(control_msg_pb2.EStopState, state)
        return state.software_estop_enabled

    def activate(self) -> None:
        """Activates the software emergency stop (E-Stop)."""
        self._set_estop(True)

    def deactivate(self) -> None:
        """Deactivates the software emergency stop (E-Stop)."""
        self._set_estop(False)

    def toggle(self) -> None:
        """Toggles the software emergency stop (E-Stop) state of the robot."""
        self._set_estop(not self.is_software_estop_enabled())

    def shutdown(self) -> None:
        """Shuts down the EStop component and stops monitoring thread."""
        if self._enabled:
            self._shutdown_event.set()
            if self._monitor_thread and self._monitor_thread.is_alive():
                self._monitor_thread.join(timeout=2.0)  # Extended timeout
                if self._monitor_thread.is_alive():
                    logger.warning("EStop monitor thread did not terminate cleanly")
        super().shutdown()

    def show(self) -> None:
        """Displays the current EStop status as a formatted table with color indicators."""
        table = Table(title="E-Stop Status")
        table.add_column("Parameter", style="cyan")
        table.add_column("Value")

        button_pressed = self.is_button_pressed()
        button_style = "bold red" if button_pressed else "bold dark_green"
        table.add_row("Button Pressed", f"[{button_style}]{button_pressed}[/]")

        if_software_estop_enabled = self.is_software_estop_enabled()
        software_style = "bold red" if if_software_estop_enabled else "bold dark_green"
        table.add_row(
            "Software E-Stop Enabled",
            f"[{software_style}]{if_software_estop_enabled}[/]",
        )

        console = Console()
        console.print(table)


class Heartbeat:
    """Heartbeat monitor that ensures the low-level controller is functioning properly.

    This class monitors a heartbeat signal from the low-level controller and exits
    the program immediately if no heartbeat is received within the specified timeout.
    This provides a critical safety mechanism to prevent the robot from operating
    when the low-level controller is not functioning properly.

    Attributes:
        _subscriber: Zenoh subscriber for heartbeat data.
        _monitor_thread: Background thread for heartbeat monitoring.
        _shutdown_event: Event to signal thread shutdown.
        _timeout_seconds: Timeout in seconds before triggering emergency exit.
        _enabled: Whether heartbeat monitoring is enabled.
        _paused: Whether heartbeat monitoring is temporarily paused.
    """

    def __init__(
        self,
        configs: HeartbeatConfig,
    ) -> None:
        """Initialize the Heartbeat monitor.

        Args:
            configs: Heartbeat configuration containing topic and timeout settings.
        """
        self._timeout_seconds = configs.timeout_seconds
        self._enabled = configs.enabled
        self._paused = False
        self._paused_lock = threading.Lock()
        self._shutdown_event = threading.Event()

        # Always create the subscriber to monitor heartbeat, regardless of enabled state
        def heartbeat_callback(data):
            """Process heartbeat data and update internal state."""
            try:
                decoded_data = self._decode_heartbeat(data)
                # Store the decoded heartbeat data for monitoring
                self._latest_heartbeat_data = decoded_data
                self._last_heartbeat_time = time.time()
            except Exception as e:
                logger.debug(f"Failed to process heartbeat data: {e}")

        # Define a simple deserializer that just returns the raw data
        def heartbeat_deserializer(data: bytes) -> bytes:
            """Pass through deserializer for raw heartbeat data."""
            return data

        self._subscriber = Subscriber(
            topic=resolve_key_name(configs.heartbeat_topic),
            callback=heartbeat_callback,
            deserializer=heartbeat_deserializer,
            config=get_zenoh_config_path(),
        )

        # Initialize tracking variables
        self._latest_heartbeat_data = None
        self._last_heartbeat_time = None

        # Start monitoring thread
        self._monitor_thread = threading.Thread(
            target=self._heartbeat_monitor, daemon=True
        )
        self._monitor_thread.start()

        if not self._enabled:
            logger.info(
                "Heartbeat monitoring is DISABLED - will monitor but not exit on timeout"
            )
        else:
            logger.info(
                f"Heartbeat monitor started with {self._timeout_seconds}s timeout"
            )

    def _decode_heartbeat(self, data) -> float:
        """Decode heartbeat data from raw bytes.

        Args:
            data: Raw bytes containing heartbeat value.

        Returns:
            Decoded heartbeat timestamp value in seconds.
        """
        try:
            # Handle different data formats
            if isinstance(data, bytes):
                timestamp_str = data.decode("utf-8")
            elif isinstance(data, str):
                timestamp_str = data
            else:
                # If it's something else, try to convert to string
                timestamp_str = str(data)

            # Parse the timestamp (expected to be in milliseconds)
            timestamp_ms = float(timestamp_str)
            # Convert from milliseconds to seconds
            return timestamp_ms / 1000.0
        except (ValueError, AttributeError, UnicodeDecodeError) as e:
            logger.debug(
                f"Failed to decode heartbeat data: {e}, data type: {type(data)}"
            )
            raise

    def _is_subscriber_active(self) -> bool:
        """Check if the subscriber is active (has received data)."""
        return self._last_heartbeat_time is not None

    def _get_time_since_last_data(self) -> float | None:
        """Get time since last heartbeat data was received."""
        if self._last_heartbeat_time is None:
            return None
        return time.time() - self._last_heartbeat_time

    def _get_latest_heartbeat_data(self) -> float | None:
        """Get the latest heartbeat data."""
        return self._latest_heartbeat_data

    def _heartbeat_monitor(self) -> None:
        """Background thread that continuously monitors heartbeat signal."""
        if self._subscriber is None:
            return

        while not self._shutdown_event.is_set():
            try:
                # Skip if paused
                with self._paused_lock:
                    if self._paused:
                        self._shutdown_event.wait(0.1)
                        continue

                # Check timeout
                time_since_last = self._get_time_since_last_data()
                if time_since_last and time_since_last > self._timeout_seconds:
                    self._handle_timeout(time_since_last)

                # Check every 50ms for responsive monitoring
                self._shutdown_event.wait(0.05)

            except Exception as e:
                logger.error(f"Heartbeat monitor error: {e}")
                self._shutdown_event.wait(0.1)

    def _handle_timeout(self, time_since_last: float) -> None:
        """Handle heartbeat timeout based on enabled state."""
        if self._enabled:
            logger.critical(
                f"HEARTBEAT TIMEOUT! No fresh heartbeat data received for {time_since_last:.2f}s "
                f"(timeout: {self._timeout_seconds}s). Low-level controller may have failed. "
                "Exiting program immediately for safety."
            )
            os._exit(1)
        else:
            # Log warning only once per timeout period to avoid spam
            if (
                not hasattr(self, "_last_warning_time")
                or time.time() - self._last_warning_time > self._timeout_seconds
            ):
                logger.warning(
                    f"Heartbeat timeout detected ({time_since_last:.2f}s > {self._timeout_seconds}s) "
                    "but exit is disabled"
                )
                self._last_warning_time = time.time()

    def pause(self) -> None:
        """Pause heartbeat monitoring temporarily.

        When paused, the heartbeat monitor will not check for timeouts or exit
        the program. This is useful for scenarios where you need to temporarily
        disable safety monitoring (e.g., during system maintenance or testing).
        """
        with self._paused_lock:
            if self._paused:
                return
            self._paused = True

        if self._enabled:
            logger.warning(
                "Heartbeat monitoring PAUSED - safety mechanism temporarily disabled"
            )
        else:
            logger.info("Heartbeat monitoring paused (exit already disabled)")

    def resume(self) -> None:
        """Resume heartbeat monitoring after being paused."""
        with self._paused_lock:
            if not self._paused:
                return
            self._paused = False

        # Wait briefly to allow fresh heartbeat data
        time.sleep(0.1)

        if self._enabled:
            logger.info("Heartbeat monitoring RESUMED - safety mechanism re-enabled")
        else:
            logger.info("Heartbeat monitoring resumed (exit still disabled)")

    def is_paused(self) -> bool:
        """Check if heartbeat monitoring is currently paused.

        Returns:
            True if monitoring is paused, False if active or disabled.
        """
        with self._paused_lock:
            return self._paused

    def get_status(self) -> dict[str, bool | float | None]:
        """Gets the current heartbeat status information.

        Returns:
            Dictionary containing heartbeat metrics including:
                - is_active: Whether heartbeat signal is being received (bool)
                - last_value: Last received heartbeat value (float | None)
                - time_since_last: Time since last fresh data in seconds (float | None)
                - timeout_seconds: Configured timeout value (float)
                - enabled: Whether heartbeat monitoring is enabled (bool)
                - paused: Whether heartbeat monitoring is paused (bool)
        """
        if self._subscriber is None:
            return {
                "is_active": False,
                "last_value": None,
                "time_since_last": None,
                "timeout_seconds": self._timeout_seconds,
                "enabled": self._enabled,
                "paused": False,
            }

        last_value = self._get_latest_heartbeat_data()
        time_since_last = self._get_time_since_last_data()

        with self._paused_lock:
            paused = self._paused

        return {
            "is_active": self._is_subscriber_active(),
            "last_value": last_value,
            "time_since_last": time_since_last,
            "timeout_seconds": self._timeout_seconds,
            "enabled": self._enabled,
            "paused": paused,
        }

    def is_active(self) -> bool:
        """Check if heartbeat signal is being received.

        Returns:
            True if heartbeat is active, False otherwise.
        """
        if self._subscriber is None:
            return False
        return self._is_subscriber_active()

    @staticmethod
    def _format_uptime(seconds: float) -> str:
        """Convert seconds to human-readable uptime format with high resolution.

        Args:
            seconds: Total seconds of uptime.

        Returns:
            Human-readable string like "1mo 2d 3h 45m 12s 345ms".
        """
        # Calculate months (assuming 30 days per month)
        months = int(seconds // (86400 * 30))
        remaining = seconds % (86400 * 30)

        # Calculate days
        days = int(remaining // 86400)
        remaining = remaining % 86400

        # Calculate hours
        hours = int(remaining // 3600)
        remaining = remaining % 3600

        # Calculate minutes
        minutes = int(remaining // 60)
        remaining = remaining % 60

        # Calculate seconds and milliseconds
        secs = int(remaining)
        milliseconds = int((remaining - secs) * 1000)

        parts = []
        if months > 0:
            parts.append(f"{months}mo")
        if days > 0:
            parts.append(f"{days}d")
        if hours > 0:
            parts.append(f"{hours}h")
        if minutes > 0:
            parts.append(f"{minutes}m")
        if secs > 0:
            parts.append(f"{secs}s")
        if milliseconds > 0 or not parts:
            parts.append(f"{milliseconds}ms")

        return " ".join(parts)

    def shutdown(self) -> None:
        """Shuts down the heartbeat monitor and stops monitoring thread."""
        self._shutdown_event.set()
        if self._monitor_thread and self._monitor_thread.is_alive():
            self._monitor_thread.join(timeout=2.0)  # Extended timeout
            if self._monitor_thread.is_alive():
                logger.warning("Heartbeat monitor thread did not terminate cleanly")
        if self._subscriber:
            self._subscriber.shutdown()

    def show(self) -> None:
        """Displays the current heartbeat status as a formatted table with color indicators."""
        status = self.get_status()

        table = Table(title="Heartbeat Monitor Status")
        table.add_column("Parameter", style="cyan")
        table.add_column("Value")

        # Mode: Enabled/Disabled and Paused state
        mode_parts = []
        if not status["enabled"]:
            mode_parts.append("[yellow]Exit Disabled[/]")
        if status["paused"]:
            mode_parts.append("[yellow]Paused[/]")
        if not mode_parts:
            mode_parts.append("[green]Active[/]")
        table.add_row("Mode", " | ".join(mode_parts))

        # Signal status
        active_style = "green" if status["is_active"] else "red"
        table.add_row(
            "Signal",
            f"[{active_style}]{'Receiving' if status['is_active'] else 'No Signal'}[/]",
        )

        # Robot uptime
        if status["last_value"] is not None:
            uptime_str = self._format_uptime(status["last_value"])
            table.add_row("Robot Uptime", f"[blue]{uptime_str}[/]")

        # Time since last heartbeat
        if status["time_since_last"] is not None:
            time_since = float(status["time_since_last"])
            timeout = status["timeout_seconds"]
            timeout = float(timeout) if timeout is not None else 1.0
            time_style = (
                "red"
                if time_since > timeout
                else "yellow"
                if time_since > timeout * 0.5
                else "green"
            )
            table.add_row("Last Heartbeat", f"[{time_style}]{time_since:.1f}s ago[/]")

        # Timeout setting
        table.add_row("Timeout", f"[blue]{status['timeout_seconds']}s[/]")

        Console().print(table)


class ServerLogSubscriber:
    """Server log subscriber that monitors and displays server log messages.

    This class subscribes to the "logs" topic and handles incoming log messages
    from the robot server. It provides formatted display of server logs with
    proper error handling and validation.

    The server sends log information via the "logs" topic as JSON with format:
    {"timestamp": "ISO8601", "message": "text", "source": "robot_server"}

    Attributes:
        _zenoh_session: Zenoh session for communication.
        _log_subscriber: Zenoh subscriber for log messages.
    """

    def __init__(self) -> None:
        """Initialize the ServerLogSubscriber."""
        # DexComm will handle the communication
        self._log_subscriber = None
        self._initialize()

    def _initialize(self) -> None:
        """Initialize the log subscriber with error handling."""

        def log_handler(payload):
            """Handle incoming log messages from the server."""
            try:
                log_data = self._parse_log_payload(payload)
                if log_data:
                    self._display_server_log(log_data)
            except Exception as e:
                logger.warning(f"Failed to process server log: {e}")

        try:
            # Subscribe to server logs topic using DexComm
            self._log_subscriber = Subscriber(
                topic="logs", callback=log_handler, config=get_zenoh_config_path()
            )
            logger.debug("Server log subscriber initialized")
        except Exception as e:
            logger.warning(f"Failed to initialize server log subscriber: {e}")
            self._log_subscriber = None

    def _parse_log_payload(self, payload) -> dict[str, str] | None:
        """Parse log payload and return structured data.

        Args:
            payload: Raw payload from Zenoh sample.

        Returns:
            Parsed log data as dictionary or None if parsing fails.
        """
        try:
            if hasattr(payload, "to_bytes"):
                # Handle zenoh-style payload
                payload_str = payload.to_bytes().decode("utf-8")
            else:
                # Handle raw bytes from DexComm
                payload_str = (
                    payload.decode("utf-8")
                    if isinstance(payload, bytes)
                    else str(payload)
                )

            if not payload_str.strip():
                logger.debug("Received empty log payload")
                return None

            log_data = json.loads(payload_str)

            if not isinstance(log_data, dict):
                logger.warning(
                    f"Invalid log data format: expected dict, got {type(log_data)}"
                )
                return None

            return log_data
        except (json.JSONDecodeError, UnicodeDecodeError) as e:
            logger.warning(f"Failed to parse log payload: {e}")
            return None

    def _display_server_log(self, log_data: dict[str, str]) -> None:
        """Display formatted server log message.

        Args:
            log_data: Parsed log data dictionary.
        """
        # Extract log information with safe defaults
        timestamp = log_data.get("timestamp", "")
        message = log_data.get("message", "")
        source = log_data.get("source", "unknown")

        # Validate critical fields
        if not message:
            logger.debug("Received log with empty message")
            return

        # Log the server message with clear identification
        logger.info(f"[SERVER_LOG] [{timestamp}] [{source}] {message}")

    def is_active(self) -> bool:
        """Check if the log subscriber is active.

        Returns:
            True if subscriber is active, False otherwise.
        """
        return self._log_subscriber is not None

    def shutdown(self) -> None:
        """Clean up the log subscriber and release resources."""
        if self._log_subscriber is not None:
            try:
                self._log_subscriber.shutdown()
                self._log_subscriber = None
            except Exception as e:
                logger.error(f"Error cleaning up log subscriber: {e}")

    def get_status(self) -> dict[str, Any]:
        """Get the current status of the log subscriber.

        Returns:
            Dictionary containing status information:
                - is_active: Whether the subscriber is active
                - topic: The topic being subscribed to
        """
        return {
            "is_active": self.is_active(),
            "topic": "logs",
        }

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

import os
import threading
import time
from typing import cast

from dexbot_utils import RobotInfo
from dexbot_utils.configs.components.vega_1 import (
    BatteryConfig,
    EStopConfig,
    HeartbeatConfig,
)
from dexcomm import Node
from dexcomm.codecs import (
    BasicDataCodec,
    BMSStateCodec,
    EStopStateCodec,
    SoftwareEstopCodec,
)
from loguru import logger
from rich.console import Console
from rich.table import Table

from dexcontrol.core.component import RobotComponent


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

    def __init__(self, name: str, robot_info: RobotInfo) -> None:
        """Initialize the Battery component.

        Args:
            robot_info: RobotInfo instance.
        """
        config = robot_info.get_component_config(name)
        config = cast(BatteryConfig, config)
        super().__init__(
            name, config.state_sub_topic, state_decoder=BMSStateCodec.decode
        )
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
            "percentage": float(state["percentage"]),
            "temperature": float(state["temperature"]),
            "current": float(state["current"]),
            "voltage": float(state["voltage"]),
            "power": float(state["current"] * state["voltage"]),
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

        battery_style = self._get_battery_level_style(state["percentage"])
        table.add_row(
            "Battery Level", f"[{battery_style}]{state['percentage']:.1f}%[/]"
        )

        temp_style = self._get_temperature_style(state["temperature"])
        table.add_row("Temperature", f"[{temp_style}]{state['temperature']:.1f}°C[/]")

        power = state["current"] * state["voltage"]
        power_style = self._get_power_style(power)
        table.add_row(
            "Power Consumption",
            f"[{power_style}]{power:.2f}W[/] ([blue]{state['current']:.2f}A[/] "
            f"× [blue]{state['voltage']:.2f}V[/])",
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
        name: str,
        robot_info: RobotInfo,
    ) -> None:
        """Initialize the EStop component.

        Args:
            robot_info: RobotInfo instance.
        """
        config = robot_info.get_component_config(name)
        config = cast(EStopConfig, config)
        self._enabled = config.enabled
        super().__init__(
            name, config.state_sub_topic, state_decoder=EStopStateCodec.decode
        )
        self._estop_querier = self._node.create_service_client(
            service_name=config.estop_query_name,
            request_encoder=SoftwareEstopCodec.encode,
            response_decoder=None,
            timeout=0.05,
        )
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
        # Wait for service to be available
        if not self._estop_querier.wait_for_service(timeout=5.0):
            logger.warning(
                f"{self._node.get_name()}: E-Stop service not available, command may fail"
            )

        query_msg = {"enabled": enable}
        self._estop_querier.call(query_msg)
        logger.info(f"Set E-Stop to {enable}")

    def get_status(self) -> dict[str, bool]:
        """Gets the current EStop state information.

        Returns:
            Dictionary containing EStop metrics including:
                - button_pressed: EStop button pressed
                - software_estop_enabled: Software EStop enabled
        """
        state = self._get_state()
        if state is None:
            return {
                "button_pressed": False,
                "software_estop_enabled": False,
            }
        button_pressed = (
            state.get("left_base_estop_enabled", False)
            or state.get("right_base_estop_enabled", False)
            or state.get("torso_estop_enabled", False)
            or state.get("remote_estop_enabled", False)
        )
        return {
            "button_pressed": button_pressed,
            "software_estop_enabled": state["software_estop_enabled"],
        }

    def is_button_pressed(self) -> bool:
        """Checks if the EStop button is pressed."""
        state = self._get_state()
        button_pressed = (
            state.get("left_base_estop_enabled", False)
            or state.get("right_base_estop_enabled", False)
            or state.get("torso_estop_enabled", False)
            or state.get("remote_estop_enabled", False)
        )
        return button_pressed

    def is_software_estop_enabled(self) -> bool:
        """Checks if the software EStop is enabled."""
        state = self._get_state()
        return state["software_estop_enabled"]

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
        _paused: Atomic flag for pause state (no lock needed).
    """

    def __init__(
        self,
        name: str,
        robot_info: RobotInfo,
    ) -> None:
        """Initialize the Heartbeat monitor.

        Args:
            configs: Heartbeat configuration containing topic and timeout settings.
        """
        config = robot_info.get_component_config(name)
        config = cast(HeartbeatConfig, config)
        self._node = Node(name=name)
        self._timeout_seconds = config.timeout_seconds
        self._enabled = config.enabled
        self._paused = threading.Event()  # Set when paused (uses atomic ops internally)
        self._shutdown_event = threading.Event()

        # State tracking (minimal locking needed)
        # Callback only updates heartbeat value, monitor thread handles timing
        self._state_lock = threading.Lock()
        self._latest_heartbeat_ms = None  # Updated by callback
        self._last_seen_heartbeat_ms = None  # Last value seen by monitor
        self._last_heartbeat_time = None  # Time monitor last saw new data
        self._last_warning_time = 0.0

        # Create subscriber with optimized callback (no timing calls!)
        self._subscriber = self._node.create_subscriber(
            topic=config.heartbeat_topic,
            callback=self._on_heartbeat,
            decoder=BasicDataCodec.decode,
        )

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

    def _on_heartbeat(self, data: int) -> None:
        """Callback for heartbeat updates. Runs in subscriber thread - keep it FAST!

        Args:
            data: Heartbeat data in milliseconds.

        Note: No time.time() call here! Timing is handled by monitor thread.
        This reduces GIL hold time by ~50ns per heartbeat.
        """
        # Absolute minimum work - just store the value (no timing!)
        with self._state_lock:
            self._latest_heartbeat_ms = data

    def _heartbeat_monitor(self) -> None:
        """Background thread that continuously monitors heartbeat signal."""
        while not self._shutdown_event.is_set():
            try:
                # Early exit if paused (no lock needed - Event is thread-safe)
                if self._paused.is_set():
                    self._shutdown_event.wait(0.1)
                    continue

                # Single lock acquisition for all state operations
                now = time.time()  # Get current time once
                with self._state_lock:
                    current_value = self._latest_heartbeat_ms
                    last_seen = self._last_seen_heartbeat_ms

                    # Detect new heartbeat data
                    if current_value is not None and current_value != last_seen:
                        self._last_seen_heartbeat_ms = current_value
                        self._last_heartbeat_time = now

                    # Check timeout (use stored timestamp)
                    last_time = self._last_heartbeat_time

                # Check timeout outside lock
                if last_time is not None:
                    time_since_last = now - last_time
                    if time_since_last > self._timeout_seconds:
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
            now = time.time()
            if now - self._last_warning_time > self._timeout_seconds:
                logger.warning(
                    f"Heartbeat timeout detected ({time_since_last:.2f}s > {self._timeout_seconds}s) "
                    "but exit is disabled"
                )
                self._last_warning_time = now

    def pause(self) -> None:
        """Pause heartbeat monitoring temporarily.

        When paused, the heartbeat monitor will not check for timeouts or exit
        the program. This is useful for scenarios where you need to temporarily
        disable safety monitoring (e.g., during system maintenance or testing).
        """
        if self._paused.is_set():
            return

        self._paused.set()
        if self._enabled:
            logger.warning(
                "Heartbeat monitoring PAUSED - safety mechanism temporarily disabled"
            )
        else:
            logger.info("Heartbeat monitoring paused (exit already disabled)")

    def resume(self) -> None:
        """Resume heartbeat monitoring after being paused."""
        if not self._paused.is_set():
            return

        self._paused.clear()
        self._last_heartbeat_time = time.time()

        if self._enabled:
            logger.info("Heartbeat monitoring RESUMED - safety mechanism re-enabled")
        else:
            logger.info("Heartbeat monitoring resumed (exit still disabled)")

    def is_paused(self) -> bool:
        """Check if heartbeat monitoring is currently paused.

        Returns:
            True if monitoring is paused, False if active or disabled.
        """
        return self._paused.is_set()

    def get_status(self) -> dict[str, bool | float | None]:
        """Gets the current heartbeat status information.

        Returns:
            Dictionary containing heartbeat metrics including:
                - is_active: Whether heartbeat signal is being received (bool)
                - last_value: Last received heartbeat value in seconds (float | None)
                - time_since_last: Time since last fresh data in seconds (float | None)
                - timeout_seconds: Configured timeout value (float)
                - enabled: Whether heartbeat monitoring is enabled (bool)
                - paused: Whether heartbeat monitoring is paused (bool)
        """
        # Single lock acquisition for all state
        with self._state_lock:
            last_time = self._last_heartbeat_time
            last_ms = self._latest_heartbeat_ms

        # Compute derived values outside lock
        is_active = last_time is not None
        time_since_last = (time.time() - last_time) if last_time else None
        last_value = (last_ms / 1000.0) if last_ms else None

        return {
            "is_active": is_active,
            "last_value": last_value,
            "time_since_last": time_since_last,
            "timeout_seconds": self._timeout_seconds,
            "enabled": self._enabled,
            "paused": self._paused.is_set(),
        }

    def is_active(self) -> bool:
        """Check if heartbeat signal is being received.

        Returns:
            True if heartbeat is active, False otherwise.
        """
        with self._state_lock:
            return self._last_heartbeat_time is not None

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

        # Join monitor thread (reduced timeout - thread checks every 50ms)
        if self._monitor_thread.is_alive():
            self._monitor_thread.join(timeout=0.2)
            if self._monitor_thread.is_alive():
                logger.warning("Heartbeat monitor thread did not terminate cleanly")

        # Always shutdown subscriber
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

        # Server uptime
        if status["last_value"] is not None:
            uptime_str = self._format_uptime(status["last_value"])
            table.add_row("Server Uptime", f"[blue]{uptime_str}[/]")

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

# Copyright (C) 2025 Dexmate Inc.
#
# This software is dual-licensed:
#
# 1. GNU Affero General Public License v3.0 (AGPL-3.0)
#    See LICENSE-AGPL for details
#
# 2. Commercial License
#    For commercial licensing terms, contact: contact@dexmate.ai

"""Example script to display robot joint currents for any component in real-time.

Visualizes joint currents of any current-capable robot component (arms, head,
torso, chassis steer/drive) using matplotlib. Each selected component is drawn in
its own subplot. The chassis is split into its ``steer`` and ``drive`` motor
groups, which are plotted as two independent components.

Examples:
    # Plot every available current-capable component
    python display_component_current.py

    # Plot only the torso and head
    python display_component_current.py --components torso head

    # Plot the chassis steer and drive motors
    python display_component_current.py --components chassis
"""

import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Sequence

import matplotlib.pyplot as plt
import numpy as np
import tyro
from loguru import logger
from matplotlib.animation import FuncAnimation
from matplotlib.lines import Line2D

from dexcontrol.exceptions import ServiceUnavailableError
from dexcontrol.robot import Robot

# Component names (as exposed on Robot) that may provide joint currents. The
# chassis is handled specially because it is a container of two joint groups.
_CURRENT_COMPONENTS: tuple[str, ...] = ("left_arm", "right_arm", "head", "torso")


@dataclass
class PlotUnit:
    """A single plottable component with a current-reading interface.

    Attributes:
        label: Human-readable name shown in the subplot title.
        component: Object exposing ``get_joint_current()`` and ``joint_name``.
        currents: Per-joint history of current values keyed by joint index.
        lines: Matplotlib lines, one per joint.
        ax: The axis this unit is drawn on.
    """

    label: str
    component: Any
    currents: dict[int, deque] = field(default_factory=dict)
    lines: list[Line2D] = field(default_factory=list)
    ax: Any = None

    @property
    def joint_names(self) -> list[str]:
        """Joint names for this component."""
        return self.component.joint_name


def _resolve_units(bot: Robot, requested: Sequence[str] | None) -> list[PlotUnit]:
    """Resolve requested component names into plottable units.

    Components that are unavailable on the robot or that do not provide current
    readings are skipped with a warning. The ``chassis`` name expands into two
    units: ``chassis_steer`` and ``chassis_drive``.

    Args:
        bot: Connected robot instance.
        requested: Component names to plot. If None, all available current-capable
            components are used.

    Returns:
        List of resolved plot units.
    """
    # Build the catalog of candidate (label -> component) pairs available on this
    # robot, expanding the chassis into its steer/drive sub-components.
    catalog: dict[str, Any] = {}
    for name in _CURRENT_COMPONENTS:
        if bot.has_component(name):
            catalog[name] = getattr(bot, name)
    if bot.has_component("chassis"):
        catalog["chassis_steer"] = bot.chassis.chassis_steer
        catalog["chassis_drive"] = bot.chassis.chassis_drive

    # Determine which labels to plot.
    if requested:
        labels: list[str] = []
        for name in requested:
            if name == "chassis":
                labels.extend(["chassis_steer", "chassis_drive"])
            else:
                labels.append(name)
    else:
        labels = list(catalog)

    units: list[PlotUnit] = []
    for label in labels:
        component = catalog.get(label)
        if component is None:
            logger.warning(f"Component '{label}' is not available; skipping.")
            continue
        # Give the component a chance to start receiving state before probing.
        if not component.wait_for_active(timeout=5.0):
            logger.warning(
                f"Component '{label}' is not receiving state updates; skipping."
            )
            continue
        # Verify the component actually reports currents before adding it.
        try:
            component.get_joint_current()
        except (ValueError, ServiceUnavailableError):
            logger.warning(
                f"Component '{label}' does not provide joint currents; skipping."
            )
            continue
        units.append(PlotUnit(label=label, component=component))

    return units


class CurrentPlotter:
    """Real-time plotter for joint currents across arbitrary robot components.

    Draws one subplot per component, with one line per joint.

    Attributes:
        bot: Robot instance for getting joint states.
        max_points: Maximum number of data points to display.
        absolute: Whether to plot absolute values of current.
        units: Components being plotted.
        times: Deque storing timestamps.
        fig: Matplotlib figure.
        start_time: Start time of plotting.
    """

    def __init__(
        self,
        components: Sequence[str] | None = None,
        max_points: int = 100,
        absolute: bool = False,
    ) -> None:
        """Initialize the current plotter.

        Args:
            components: Component names to plot. If None, plots all available
                current-capable components. Use ``chassis`` to plot both the
                steer and drive motor groups.
            max_points: Maximum number of points in history.
            absolute: Whether to plot absolute values of current.

        Raises:
            RuntimeError: If no current-capable components could be resolved.
        """
        self.bot = Robot()
        self.max_points = max_points
        self.absolute = absolute
        self.times: deque = deque(maxlen=max_points)

        self.units = _resolve_units(self.bot, components)
        if not self.units:
            self.bot.shutdown()
            raise RuntimeError(
                "No current-capable components available to plot. "
                "Check the robot configuration and --components argument."
            )

        # Setup one subplot per component, stacked vertically.
        n = len(self.units)
        self.fig, axes = plt.subplots(n, 1, figsize=(10, max(4, 4 * n)), squeeze=False)
        for unit, ax in zip(self.units, axes[:, 0]):
            unit.ax = ax

        self._setup_axes()
        self._initialize_plot_lines()
        self.start_time = time.time()

    def _initialize_plot_lines(self) -> None:
        """Initialize plot lines and data storage for each joint of each unit."""
        for unit in self.units:
            unit.currents = {
                i: deque(maxlen=self.max_points) for i in range(len(unit.joint_names))
            }
            unit.lines = [
                unit.ax.plot([], [], label=f"{name} (A)", linewidth=2.0)[0]
                for name in unit.joint_names
            ]

    def _setup_axes(self) -> None:
        """Configure plot axes appearance."""
        for unit in self.units:
            ax = unit.ax
            ax.set_title(f"{unit.label} Joint Currents")
            ax.set_xlabel("Time (s)")
            ax.set_ylabel("Current (Amperes)")
            ax.grid(True)
            ax.set_xticks(np.linspace(0, 2, 11))
            ax.set_xlim(0, 2)

    def create_legends(self) -> None:
        """Create plot legends."""
        for unit in self.units:
            unit.ax.legend(loc="upper left", bbox_to_anchor=(1, 1))

    def update(self, _) -> list[Line2D]:
        """Update plot data.

        Args:
            _: Frame number (unused but required by FuncAnimation).

        Returns:
            List of updated plot lines.
        """
        current_time = time.time() - self.start_time
        self.times.append(current_time)
        shifted_times = np.array(list(self.times)) - (current_time - 2)

        updated_lines: list[Line2D] = []
        for unit in self.units:
            currents = unit.component.get_joint_current()
            for joint_idx, line in enumerate(unit.lines):
                val = currents[joint_idx]
                if self.absolute:
                    val = abs(val)
                unit.currents[joint_idx].append(val)
                line.set_data(shifted_times, list(unit.currents[joint_idx]))
            updated_lines.extend(unit.lines)

            # Update y-axis limits.
            unit.ax.relim()
            unit.ax.autoscale_view(scalex=False)

        return updated_lines


def main(
    components: list[str] | None = None,
    max_points: int = 100,
    absolute: bool = True,
) -> None:
    """Run the current plotting visualization.

    Args:
        components: Component names to plot (e.g. ``left_arm``, ``right_arm``,
            ``head``, ``torso``, ``chassis``, ``chassis_steer``,
            ``chassis_drive``). If omitted, all available current-capable
            components are plotted.
        max_points: Maximum number of points kept in history.
        absolute: Whether to plot absolute values of current.
    """
    plotter = CurrentPlotter(
        components=components, max_points=max_points, absolute=absolute
    )
    plotter.create_legends()

    try:
        _ = FuncAnimation(
            plotter.fig,
            plotter.update,
            interval=100,  # 10 Hz update rate
            blit=True,
            cache_frame_data=False,
            save_count=100,
        )
        plt.tight_layout()
        plt.show()
    except KeyboardInterrupt:
        pass
    finally:
        plotter.bot.shutdown()
        plt.close()


if __name__ == "__main__":
    tyro.cli(main)

# Copyright (C) 2025 Dexmate Inc.
#
# This software is dual-licensed:
#
# 1. GNU Affero General Public License v3.0 (AGPL-3.0)
#    See LICENSE-AGPL for details
#
# 2. Commercial License
#    For commercial licensing terms, contact: contact@dexmate.ai

"""Example script to show head ZED X Mini camera data live.

This script demonstrates how to retrieve head ZED X Mini camera data (RGB and depth
images) from the robot and display them live using matplotlib animation. It showcases
the simple API for getting camera data and provides live visualization.
"""

import os

# Fix Qt plugin issue by setting environment variables
os.environ["QT_QPA_PLATFORM_PLUGIN_PATH"] = ""

# Set matplotlib backend before importing pyplot
import matplotlib

try:
    # Try TkAgg backend first (more reliable for live display)
    matplotlib.use("TkAgg")
    print("Using TkAgg backend for display")
except ImportError:
    try:
        # Fallback to Qt5Agg
        matplotlib.use("Qt5Agg")
        print("Using Qt5Agg backend for display")
    except ImportError:
        # Last resort - use default
        print("Using default matplotlib backend")

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
import tyro

from dexcontrol.core.config import get_robot_config
from dexcontrol.robot import Robot


def get_camera_data(robot):
    """Simple function to get head ZED X Mini camera data from robot sensors.

    This demonstrates how easy it is to get camera data using our API.
    """
    return robot.sensors.head_camera.get_obs(
        obs_keys=["left_rgb", "right_rgb", "depth"], include_timestamp=True
    )


def print_camera_info(camera_info):
    """Nicely format and print any nested dictionary."""

    def print_dict(d, indent=0):
        """Recursively print dictionary with proper indentation."""
        for key, value in d.items():
            # Create indentation
            spaces = "  " * indent

            if isinstance(value, dict):
                print(f"{spaces}{key}:")
                print_dict(value, indent + 1)
            elif isinstance(value, (list, tuple)):
                print(f"{spaces}{key}: {value}")
            else:
                print(f"{spaces}{key}: {value}")

    if not camera_info:
        print("No camera information available")
        return

    print("\n" + "=" * 50)
    print("HEAD ZED X MINI CAMERA INFORMATION")
    print("=" * 50)
    print_dict(camera_info)
    print("=" * 50)
    print()


def visualize_camera_data(robot, fps: float = 30.0):
    """Visualize camera data using matplotlib."""
    from matplotlib import cm

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle("Live Head ZED X Mini Camera Feeds", fontsize=16)

    # Setup display
    displays = [ax.imshow(np.zeros((480, 640, 3))) for ax in axes]
    for ax, title in zip(axes, ["Left RGB", "Right RGB", "Depth"]):
        ax.set_title(title)
        ax.axis("off")

    plt.tight_layout()

    # Latency tracking
    stream_names = ["left_rgb", "right_rgb", "depth"]
    latency_history: dict[str, list[float]] = {k: [] for k in stream_names}
    LATENCY_WINDOW = 100  # rolling window size

    def update(frame):
        # Get camera data - simple API call
        camera_data = get_camera_data(robot)

        # Update displays
        titles = ["Left RGB", "Right RGB", "Depth"]
        latency_parts = []
        for i, key in enumerate(stream_names):
            if key in camera_data and camera_data[key] is not None:
                data = camera_data[key]

                # Extract image, publish timestamp, and receive timestamp
                if isinstance(data, dict):
                    img = data.get("data")
                    timestamp_ns = data.get("timestamp_ns")
                    receive_time_ns = data.get("receive_time_ns")
                else:
                    img = data
                    timestamp_ns = None
                    receive_time_ns = None

                # Skip if no image data
                if img is None:
                    continue

                # Compute latency (receive wallclock - publish wallclock)
                latency_ms = None
                if (
                    timestamp_ns is not None
                    and timestamp_ns > 0
                    and receive_time_ns is not None
                ):
                    latency_ms = (receive_time_ns - timestamp_ns) / 1e6
                    history = latency_history[key]
                    history.append(latency_ms)
                    if len(history) > LATENCY_WINDOW:
                        history.pop(0)
                    avg_ms = sum(history) / len(history)
                    min_ms = min(history)
                    max_ms = max(history)
                    latency_parts.append(
                        f"{titles[i]:>9s}: {latency_ms:6.1f}ms "
                        f"(avg {avg_ms:5.1f} | min {min_ms:5.1f} | max {max_ms:5.1f})"
                    )

                # Update title with shape info
                title = f"{titles[i]}\n{img.shape}"
                if latency_ms is not None:
                    title += f"\nlatency={latency_ms:.1f}ms"
                axes[i].set_title(title)

                # Process depth image for visualization
                if "depth" in key:
                    # Normalize depth to 0-255
                    img = (
                        (img - img.min()) / (img.max() - img.min() + 1e-8) * 255
                    ).astype(np.uint8)
                    # Apply colormap
                    img = (cm.viridis(img / 255.0)[:, :, :3] * 255).astype(np.uint8)

                displays[i].set_array(img)

        # Print live latency to terminal
        if latency_parts:
            print(
                "\033[2K\r"  # clear line
                + " | ".join(latency_parts),
                end="",
                flush=True,
            )

    # Start animation
    _ = animation.FuncAnimation(
        fig, update, interval=int(1000 / fps), blit=False, cache_frame_data=False
    )
    plt.show()
    print()  # newline after animation ends


def main(fps: float = 30.0, use_rtc: bool = False) -> None:
    """Main function to initialize robot and display head ZED X Mini camera feeds.

    Args:
        fps: Display refresh rate in Hz (default: 30.0)
        use_rtc: Use WebRTC for RGB streams if True (default: False)
    """
    configs = get_robot_config()
    configs.enable_sensor("head_camera")
    configs.sensors["head_camera"].transport = "rtc" if use_rtc else "zenoh"

    with Robot(configs=configs) as robot:
        # Wait for camera to become active
        print("Waiting for camera streams to become active...")
        if robot.sensors.head_camera.wait_for_active(timeout=5.0):
            print("Camera streams active!")
        else:
            print("Warning: Some camera streams may not be active")

        # Print camera information nicely
        camera_info = robot.sensors.head_camera.get_camera_info()
        print_camera_info(camera_info)

        # Start live camera visualization
        visualize_camera_data(robot, fps)


if __name__ == "__main__":
    tyro.cli(main)

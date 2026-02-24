#!/usr/bin/env python3
"""Test Cartesian (xyz + roll/pitch/yaw) motions via the Vega RobotEnv gRPC server.

Sends small delta motions along each axis and prints the observed cartesian_position
so you can verify the arm moves correctly in x, y, z, roll, pitch, yaw.

Coordinate frame (important):
  - cartesian_position and cartesian_delta are in the ROBOT BASE FRAME defined by
    the dexmotion/pinocchio model (Vega URDF). The directions of X, Y, Z depend on
    that URDF and how the robot is mounted.
  - In the IK (base_arm_teleop.move_delta_cartesian), delta_xyz is added to the
    end-effector position in this base frame: so +X = move in the model's X direction,
    etc. If +X makes the arm move "up", then the model's X axis is roughly vertical.
  - To align with a different world frame (e.g. X=forward, Z=up), use the server's
    --base-frame-rotation (roll, pitch, yaw in degrees) or combine axes in your client.
  - Exact convention: see the Vega URDF or robot documentation (e.g. dexmate-ai/vega).

Usage:
  # Server must be running first, e.g.:
  #   python src/dexcontrol/core/robotenv_vega/server.py --arm-side left --grpc-port 50061

  cd /path/to/custom_dexcontrol
  python examples/test_cartesian_axes.py

  # Optional: different host/port
  python examples/test_cartesian_axes.py --port 50062
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

# Repo root and src for imports when run as script
_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))
_SRC = _REPO / "src"
if _SRC.exists() and str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

import numpy as np

from dexcontrol.core.robot_env_client import RobotEnvClient


def format_pose(cart: list | np.ndarray) -> str:
    a = np.asarray(cart)
    if len(a) < 6:
        return str(cart)
    x, y, z = a[0], a[1], a[2]
    r, p, y = np.rad2deg(a[3]), np.rad2deg(a[4]), np.rad2deg(a[5])
    return f"xyz=({x:.3f}, {y:.3f}, {z:.3f}) m  rpy=({r:.1f}, {p:.1f}, {y:.1f}) deg"


def main() -> None:
    parser = argparse.ArgumentParser(description="Test Cartesian axes via RobotEnv gRPC")
    parser.add_argument("--host", default="localhost", help="RobotEnv server host")
    parser.add_argument("--port", type=int, default=50061, help="RobotEnv server port")
    parser.add_argument(
        "--action-space",
        default="cartesian_delta",
        choices=["cartesian_delta", "cartesian_velocity"],
        help="Action space for step",
    )
    parser.add_argument(
        "--delta-m",
        type=float,
        default=0.02,
        help="Position delta in meters (for x,y,z)",
    )
    parser.add_argument(
        "--delta-deg",
        type=float,
        default=5.0,
        help="Rotation delta in degrees (for roll, pitch, yaw)",
    )
    parser.add_argument(
        "--steps-per-axis",
        type=int,
        default=5,
        help="Number of steps per direction (more = larger motion)",
    )
    parser.add_argument("--no-reset", action="store_true", help="Skip initial reset")
    args = parser.parse_args()

    delta_rad = np.deg2rad(args.delta_deg)
    # Cartesian action: [dx, dy, dz, droll, dpitch, dyaw, gripper]
    gripper = 0.5

    print(f"Connecting to RobotEnv at {args.host}:{args.port} (action_space={args.action_space}) ...")
    print("Note: Cartesian axes are in the robot base frame (see script docstring).")
    client = RobotEnvClient(
        robot_ip=args.host,
        robot_port=args.port,
        action_space=args.action_space,
        do_reset=not args.no_reset,
    )

    if not args.no_reset:
        print("Reset done. Waiting 1s ...")
        time.sleep(1.0)

    def get_pose() -> np.ndarray:
        obs_dict, _ = client.get_observation()
        cart = obs_dict["robot_state"]["cartesian_position"]
        return np.asarray(cart, dtype=np.float64)

    def step_cartesian(dx: float, dy: float, dz: float, dr: float, dp: float, dyaw: float) -> None:
        action = [dx, dy, dz, dr, dp, dyaw, gripper]
        client.step(action)

    axes = [
        ("+X", args.delta_m, 0.0, 0.0, 0.0, 0.0, 0.0),
        ("-X", -args.delta_m, 0.0, 0.0, 0.0, 0.0, 0.0),
        ("+Y", 0.0, args.delta_m, 0.0, 0.0, 0.0, 0.0),
        ("-Y", 0.0, -args.delta_m, 0.0, 0.0, 0.0, 0.0),
        ("+Z", 0.0, 0.0, args.delta_m, 0.0, 0.0, 0.0),
        ("-Z", 0.0, 0.0, -args.delta_m, 0.0, 0.0, 0.0),
        ("+Roll", 0.0, 0.0, 0.0, delta_rad, 0.0, 0.0),
        ("-Roll", 0.0, 0.0, 0.0, -delta_rad, 0.0, 0.0),
        ("+Pitch", 0.0, 0.0, 0.0, 0.0, delta_rad, 0.0),
        ("-Pitch", 0.0, 0.0, 0.0, 0.0, -delta_rad, 0.0),
        ("+Yaw", 0.0, 0.0, 0.0, 0.0, 0.0, delta_rad),
        ("-Yaw", 0.0, 0.0, 0.0, 0.0, 0.0, -delta_rad),
    ]

    print("\nInitial pose:", format_pose(get_pose()))
    print("\nSending small motions per axis (watch the arm and reported pose).")
    print("Press Enter before each axis to start that motion.\n")

    for name, dx, dy, dz, dr, dp, dyaw in axes:
        input(f"  [{name}] Press Enter to run ... ")
        pose_before = get_pose()
        for _ in range(args.steps_per_axis):
            step_cartesian(dx, dy, dz, dr, dp, dyaw)
            time.sleep(0.05)
        pose_after = get_pose()
        delta = pose_after - pose_before
        print(f"  {name:8s}  before -> after   {format_pose(pose_before)}  ->  {format_pose(pose_after)}")
        print(f"           delta xyz(m)={delta[:3].round(4).tolist()}  rpy(deg)={np.rad2deg(delta[3:6]).round(2).tolist()}")
        time.sleep(0.2)

    print("\nDone. Final pose:", format_pose(get_pose()))
    client.close()
    print("Connection closed.")


if __name__ == "__main__":
    main()

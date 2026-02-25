#!/usr/bin/env python3
"""Quick 5-step test: send 0.04m X delta at 20Hz, print per-step results."""
from __future__ import annotations
import sys, time
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
for p in (_REPO, _REPO / "src"):
    if p.exists() and str(p) not in sys.path:
        sys.path.insert(0, str(p))

import numpy as np
from dexcontrol.core.robot_env_client import RobotEnvClient

def get_cart(c):
    obs, _ = c.get_observation()
    return np.asarray(obs["robot_state"]["cartesian_position"], dtype=np.float64)

def get_joints(c):
    obs, _ = c.get_observation()
    return np.asarray(obs["robot_state"]["joint_positions"], dtype=np.float64)

def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--port", type=int, default=50061)
    p.add_argument("--delta", type=float, default=0.04)
    p.add_argument("--steps", type=int, default=5)
    args = p.parse_args()

    client = RobotEnvClient(
        robot_ip="localhost", robot_port=args.port,
        action_space="cartesian_delta", do_reset=True,
    )
    time.sleep(0.5)

    print(f"Delta: {args.delta}m  Steps: {args.steps}  Port: {args.port}")
    print(f"Home cart: {np.array2string(get_cart(client), precision=4)}")
    print(f"Home joints: {np.array2string(get_joints(client), precision=4)}")
    print()

    for i in range(args.steps):
        cart_before = get_cart(client)
        joints_before = get_joints(client)
        t0 = time.time()
        client.step([args.delta, 0, 0, 0, 0, 0, 0.5])
        rpc_ms = (time.time() - t0) * 1000
        time.sleep(0.03)  # let motor settle a bit
        cart_after = get_cart(client)
        joints_after = get_joints(client)
        d_cart = cart_after - cart_before
        d_joints = joints_after - joints_before
        print(f"Step {i+1}: rpc={rpc_ms:.1f}ms")
        print(f"  Δcart xyz: {np.array2string(d_cart[:3]*1000, precision=1)}mm  "
              f"rpy: {np.array2string(np.rad2deg(d_cart[3:6]), precision=2)}°")
        print(f"  Δjoints(rad): {np.array2string(d_joints, precision=4)}")
        print(f"  max|Δjoint|: {np.max(np.abs(d_joints)):.4f} rad")
        print()

    client.close()

if __name__ == "__main__":
    main()

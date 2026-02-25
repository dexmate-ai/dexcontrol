#!/usr/bin/env python3
"""Diagnose arm joint dynamics by testing individual joint movements.

Tests small and large step sizes on each joint to understand:
- Does set_joint_pos respond to small deltas? Large deltas?
- What's the effective max velocity per joint?
- Does the motor ignore commands beyond a threshold?

Usage:
  python examples/test_arm_dynamics.py --arm-side right
  python examples/test_arm_dynamics.py --arm-side left --joint 2 --delta 0.1
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
for p in (_REPO, _REPO / "src"):
    if p.exists() and str(p) not in sys.path:
        sys.path.insert(0, str(p))

import numpy as np
from dexbot_utils.configs import get_robot_config
from dexcontrol.robot import Robot


def read_joints(arm) -> np.ndarray:
    return np.asarray(arm.get_joint_pos(), dtype=np.float64)


def send_target_and_wait(arm, target: np.ndarray, hz: float = 200, duration: float = 2.0):
    """Send set_joint_pos at hz for duration seconds, return final position."""
    dt = 1.0 / hz
    steps = int(duration * hz)
    for _ in range(steps):
        arm.set_joint_pos(target, wait_time=0.0)
        time.sleep(dt)
    return read_joints(arm)


def test_single_joint(arm, joint_idx: int, delta: float, hz: float, duration: float):
    """Move a single joint by delta while keeping others at current position."""
    current = read_joints(arm)
    target = current.copy()
    target[joint_idx] += delta

    print(f"    Target joint[{joint_idx}]: {current[joint_idx]:.4f} -> {target[joint_idx]:.4f} (delta={delta:+.4f} rad)")
    final = send_target_and_wait(arm, target, hz=hz, duration=duration)
    actual_delta = final[joint_idx] - current[joint_idx]
    other_deltas = np.abs(final - current)
    other_deltas[joint_idx] = 0.0
    print(f"    Result joint[{joint_idx}]: {final[joint_idx]:.4f} (moved {actual_delta:+.4f} rad, {abs(actual_delta/delta)*100:.0f}% of requested)")
    print(f"    Other joints max movement: {np.max(other_deltas):.4f} rad")
    return current, final


def main():
    parser = argparse.ArgumentParser(description="Test arm joint dynamics")
    parser.add_argument("--arm-side", default="right", choices=["left", "right"])
    parser.add_argument("--robot-model", default="vega_1")
    parser.add_argument("--joint", type=int, default=None, help="Test only this joint (0-6)")
    parser.add_argument("--delta", type=float, default=None, help="Custom delta in rad")
    parser.add_argument("--hz", type=float, default=200, help="Command frequency")
    parser.add_argument("--duration", type=float, default=2.0, help="Duration per test (seconds)")
    args = parser.parse_args()

    configs = get_robot_config(args.robot_model)
    robot = Robot(configs=configs)
    arm = getattr(robot, f"{args.arm_side}_arm")
    arm.set_modes(["position"] * 7)

    limits = arm.joint_pos_limit
    l_shape = np.asarray(arm.get_predefined_pose("L_shape"), dtype=np.float64)

    print(f"\n=== Arm Dynamics Test ({args.arm_side}) ===")
    print(f"Command freq: {args.hz} Hz, duration: {args.duration}s per test")
    current = read_joints(arm)
    print(f"\nCurrent:  {np.array2string(current, precision=3, separator=', ')}")
    print(f"L_shape:  {np.array2string(l_shape, precision=3, separator=', ')}")
    print(f"Distance: {np.array2string(l_shape - current, precision=3, separator=', ')}")
    print(f"\nJoint limits:")
    for i in range(7):
        dist = l_shape[i] - current[i]
        in_range = limits[i, 0] <= l_shape[i] <= limits[i, 1]
        print(f"  J{i}: [{limits[i,0]:+.3f}, {limits[i,1]:+.3f}]  current={current[i]:+.3f}  L_shape={l_shape[i]:+.3f}  delta={dist:+.3f}  {'OK' if in_range else 'OUT OF RANGE!'}")

    joints_to_test = [args.joint] if args.joint is not None else list(range(7))

    if args.delta is not None:
        deltas_to_test = [args.delta]
    else:
        deltas_to_test = [0.01, 0.05, 0.1, 0.3]

    for ji in joints_to_test:
        print(f"\n--- Joint {ji} ---")
        dist_to_l_shape = l_shape[ji] - read_joints(arm)[ji]
        sign = 1.0 if dist_to_l_shape > 0 else -1.0

        for delta in deltas_to_test:
            signed_delta = sign * delta
            input(f"  [J{ji} delta={signed_delta:+.3f}] Press Enter to run...")
            before, after = test_single_joint(arm, ji, signed_delta, args.hz, args.duration)
            time.sleep(0.5)

        input(f"\n  [J{ji} FULL L_shape delta={dist_to_l_shape:+.3f}] Press Enter to try full distance...")
        before, after = test_single_joint(arm, ji, dist_to_l_shape, args.hz, args.duration)
        time.sleep(0.5)

    print(f"\n=== All-joint simultaneous test ===")
    input("  Press Enter to send ALL joints to L_shape simultaneously...")
    current = read_joints(arm)
    print(f"  Before: {np.array2string(current, precision=3, separator=', ')}")
    final = send_target_and_wait(arm, l_shape, hz=args.hz, duration=5.0)
    print(f"  After:  {np.array2string(final, precision=3, separator=', ')}")
    err = np.abs(final - l_shape)
    print(f"  Error:  {np.array2string(err, precision=4, separator=', ')}")
    print(f"  Max error: {np.max(err):.4f} rad")

    print("\nDone.")


if __name__ == "__main__":
    main()

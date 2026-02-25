#!/usr/bin/env python3
"""Send 30 step commands and collect server-side timing breakdown."""
from __future__ import annotations
import sys, time
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
for p in (_REPO, _REPO / "src"):
    if p.exists() and str(p) not in sys.path:
        sys.path.insert(0, str(p))

import numpy as np
from dexcontrol.core.robot_env_client import RobotEnvClient

def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--port", type=int, default=50061)
    p.add_argument("--steps", type=int, default=30)
    p.add_argument("--hz", type=float, default=20)
    args = p.parse_args()

    client = RobotEnvClient(
        robot_ip="localhost", robot_port=args.port,
        action_space="cartesian_delta", do_reset=True,
    )
    time.sleep(0.5)

    dt = 1.0 / args.hz
    actions = [
        [0.02, 0, 0, 0, 0, 0, 0.5],
        [0, 0.02, 0, 0, 0, 0, 0.5],
        [0, 0, 0.02, 0, 0, 0, 0.5],
        [0, 0, 0, 0.05, 0, 0, 0.5],
        [0.01, 0.01, 0, 0, 0, 0.03, 0.5],
    ]

    print(f"Port: {args.port}  Steps: {args.steps}  Hz: {args.hz}")
    print("Sending steps — timing data goes to debug log...\n")

    rpc_times = []
    for i in range(args.steps):
        action = actions[i % len(actions)]
        t0 = time.time()
        client.step(action)
        elapsed = time.time() - t0
        rpc_times.append(elapsed * 1000)
        remaining = dt - elapsed
        if remaining > 0:
            time.sleep(remaining)

    rpc_arr = np.array(rpc_times)
    print(f"Client-side RPC latency (ms):")
    print(f"  mean={rpc_arr.mean():.1f}  std={rpc_arr.std():.1f}  "
          f"min={rpc_arr.min():.1f}  max={rpc_arr.max():.1f}  "
          f"p50={np.percentile(rpc_arr,50):.1f}  p95={np.percentile(rpc_arr,95):.1f}")
    print(f"\nDone. Check debug log for server-side breakdown.")
    client.close()

if __name__ == "__main__":
    main()

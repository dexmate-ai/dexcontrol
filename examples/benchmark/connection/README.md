# Connection Benchmark

Benchmarks for measuring robot connection and connect-robot latency. Use these to understand how long it takes to find the robot on the current network and complete the first RPC call.

> **Note:** Results are sensitive to network conditions (WiFi vs wired, network load, Zenoh peer discovery). Run multiple times and compare across environments.

## Scripts

| Script | Description |
|--------|-------------|
| `connect_robot_latency.py` | Measures Phase 1 (init + Zenoh discovery) and Phase 2 (query RTT × N runs) using a lightweight first RPC call |

## Prerequisites

Set the required environment variables before running:

```bash
export ROBOT_NAME=dm/<your-robot-id>
export ZENOH_CONFIG=~/.dexmate/comm/zenoh/<profile>/zenoh_peer_config.json5
```

See `CLAUDE.md` for the specific values for your robot.

## Usage

```bash
# Default: 10 RTT runs, 10s discovery timeout
PYTHONPATH=src python examples/benchmark/connection/connect_robot_latency.py

# More RTT runs for better statistics
PYTHONPATH=src python examples/benchmark/connection/connect_robot_latency.py --n-runs 30

# Longer discovery timeout for slow networks
PYTHONPATH=src python examples/benchmark/connection/connect_robot_latency.py --timeout 20.0
```

## Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--n-runs` | 10 | Number of Phase 2 RTT repetitions |
| `--timeout` | 10.0 | Discovery timeout in seconds |

## Output

```
Connect Robot Latency Benchmark
┌──────────────────────────────────────┬──────────┐
│ Phase 1 — Init + Discovery           │ 234.5 ms │
│                                      │          │
│ Phase 2 — Query RTT  (N=10)          │          │
│   Mean                               │  12.3 ms │
│   Min                                │  11.8 ms │
│   Max                                │  13.1 ms │
│   Std                                │   0.4 ms │
└──────────────────────────────────────┴──────────┘
```

## Connection Failure

If the robot is not reachable within the discovery timeout, the script exits with a clear error:

```
✗ Could not connect to robot within 10.0s.
  • Run 'dextop topic list' to verify you can receive topics from the robot.
  • Check that the robot is powered on and the network is reachable.
```

Use `--timeout` to increase the discovery window on slow or congested networks.

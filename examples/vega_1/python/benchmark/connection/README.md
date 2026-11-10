<!--
Copyright (C) 2026 Dexmate Inc.
SPDX-License-Identifier: Apache-2.0
-->

# Connection Benchmark

Benchmarks for measuring robot connection and query latency. Use these to understand how long it takes to find the robot on the current network and complete a round trip to it.

> **Note:** Results are sensitive to network conditions (WiFi vs wired, network load, Zenoh peer discovery). Run multiple times and compare across environments.

## Scripts

| Script | Description |
|--------|-------------|
| `benchmark_connection_latency.py` | Measures init + Zenoh discovery, then query round-trip time over N samples |
| `stress_heartbeat_gil.py` | Confirms heartbeat monitoring keeps running while Python holds the GIL |

Both scripts create a control connection: startup can enable compatible arm modes, and shutdown requests a stop. The latency benchmark supports `--simulated`; the heartbeat stress program does not.

## Prerequisites

Set the required environment variables before running:

```bash
export ROBOT_NAME=dm/<your-robot-id>
export ZENOH_CONFIG=~/.dexmate/comm/zenoh/<profile>/zenoh_peer_config.json5
```

`ROBOT_NAME` also selects the robot profile; pass `--profile` to override it.

## Usage

Install the package (`pip install .` or `maturin develop`) — the examples import
the installed `dexcontrol`, not a source tree.

```bash
# Default: 10 RTT samples, 10s discovery timeout
python examples/vega_1/python/benchmark/connection/benchmark_connection_latency.py

# More samples for better statistics
python examples/vega_1/python/benchmark/connection/benchmark_connection_latency.py --samples 30

# Longer discovery timeout for slow networks
python examples/vega_1/python/benchmark/connection/benchmark_connection_latency.py --timeout 20.0
```

## Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--profile` | `ROBOT_NAME` | Built-in robot profile |
| `--samples` | 10 | Number of query RTT samples |
| `--timeout` | 10.0 | Active-state timeout in seconds |

## Output

```
Connect Robot Latency Benchmark
  Init + discovery: 234.5 ms
  Query RTT samples: 10
  Mean: 12.3 ms
  Min:  11.8 ms
  Max:  13.1 ms
  Std:  0.4 ms
```

The benchmark issues a raw `version_info` query rather than calling
`robot.version_info()`, which answers from the cached connect-time response and
would time a dictionary lookup instead of a network round trip.

## Connection Failure

If the robot does not become active within the timeout, the script raises:

```
RuntimeError: robot did not become active within 10.0 seconds
```

Check that the robot is powered on and reachable, and use `--timeout` to widen
the discovery window on slow or congested networks.

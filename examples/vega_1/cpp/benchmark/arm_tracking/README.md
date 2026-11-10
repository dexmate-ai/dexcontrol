<!--
Copyright (C) 2026 Dexmate Inc.
SPDX-License-Identifier: Apache-2.0
-->

# Arm Tracking Benchmark (C++)

Per-joint tracking benchmarks for either arm, the C++ counterparts of
`examples/vega_1/python/benchmark/arm_tracking/`. Use them to compare tracking
before and after tuning PID multipliers (`advanced_examples/configure_arm_pid`).

> **Note:** From a remote machine the measured latency includes network
> round-trip time, especially over WiFi. Run on the robot computer or over
> wired Ethernet for control-loop measurements.
>
> Link the release library (`cargo build --release -p dexcontrol-capi`; the
> examples' CMake picks `target/release` when it exists). A debug build of
> the native runtime adds latency that is not the robot's.

| Binary | Test type | Control method |
| --- | --- | --- |
| `benchmark_arm_sine` | Sine wave tracking | `set_joint_pos` with velocities (position + velocity feed-forward) |
| `benchmark_arm_step` | Step response | `set_joint_pos` (position only) |

Both benchmarks:

1. warn and ask for confirmation (`--no-confirm` skips the prompt),
2. move the arm to the zero pose along a planned path and verify it arrived
   (every joint within 0.05 rad),
3. drive each of the seven joints in turn while the others hold zero, pacing
   the loop with the native `RateLimiter`,
4. print mean and peak absolute tracking error per joint, and
5. write one CSV per joint (`time_s, cmd_0..6, actual_0..6`) plus a
   `parameters.txt` into `results/<timestamp>_<side>_<kind>/`.

The CSVs replace the Python summary plots; open them with any plotting tool
to compare command against measured position.

```bash
build/bin/benchmark/arm_tracking/benchmark_arm_sine --side left
build/bin/benchmark/arm_tracking/benchmark_arm_step --side left --duration 3
# dry run against the in-process simulation
build/bin/benchmark/arm_tracking/benchmark_arm_sine --simulated --duration 1 --settle 0 --amplitude 0.3 --no-confirm
```

## Parameters

### benchmark_arm_sine

| Option | Default | Description |
| --- | --- | --- |
| `--side` | right | Which arm to test |
| `--duration` | 10.0 | Test duration per joint (s) |
| `--control-hz` | 200 | Control loop frequency (Hz) |
| `--amplitude` | 0.4 | Sine amplitude (rad) |
| `--sin-frequency` | 1.0 | Sine frequency (Hz) |
| `--output-dir` | results | Base output directory, relative to the working directory |
| `--settle` | 0.5 | Seconds to settle at zero before verifying (use 0 in simulation) |
| `--no-confirm` | off | Skip the safety confirmation |

The amplitude is fitted to each joint before its run and every change is
printed: it is kept inside the position limits from the reference pose, and
inside the velocity limit, because the feed-forward peaks at
`2π · amplitude · frequency` and the native limit check rejects a command
above a joint's limit (2.4 rad/s on the arm's first joint caps a 1 Hz sine at
about 0.37 rad). The step benchmark fits its step the same way: the other
direction where the requested one does not fit, a smaller step where neither
does. Both stream with the step guard disabled, since tracking lag is what
they measure.

### benchmark_arm_step

| Option | Default | Description |
| --- | --- | --- |
| `--side` | right | Which arm to test |
| `--duration` | 5.0 | Test duration per joint (s) |
| `--control-hz` | 200 | Control loop frequency (Hz) |
| `--step-amplitude` | 0.524 (30 deg) | Step amplitude (rad) |
| `--step-time` | 0.3 | Step start time (s) |
| `--transition-time` | 0.3 | Ramp duration (s) |
| `--max-vel` | auto | Ramp velocity (rad/s); derived from the transition time when omitted |
| `--output-dir` | results | Base output directory, relative to the working directory |
| `--settle` | 0.5 | Seconds to settle at zero before verifying (use 0 in simulation) |
| `--no-confirm` | off | Skip the safety confirmation |

PID multipliers are capped to `[0.1, 4]`; a higher P multiplier tracks
tighter but makes the arm stiffer, so choose the lowest value that meets your
tracking requirement.

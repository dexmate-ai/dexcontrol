<!--
Copyright (C) 2026 Dexmate Inc.

This software is dual-licensed:

1. GNU Affero General Public License v3.0 (AGPL-3.0)
   See LICENSE-AGPL for details

2. Commercial License
   For commercial licensing terms, contact: contact@dexmate.ai
-->

# C++ examples

From the repository root, build and run an example with one command:

```bash
./run cpp cycle_arm --simulated --profile vega_1p --delta 0.01
./run cpp --list
./run cpp cycle_arm --help
```

Use `./run --release cpp cycle_arm --simulated` for an optimized build, or
`./run --build-only cpp cycle_arm` to compile without running. See the
[launcher guide](../../README.md#run-examples) for options. Manual build commands
remain available below.

See [what each example does](../../BEHAVIOR.md) for exact defaults, setup motions, output, completion conditions and language differences. [Renamed files](../../RENAMED.md) maps old paths to current paths.

There are **45 shared tasks**: 43 use a Robot control connection and two use read-only diagnostics. A Robot connection may enable compatible arm modes at startup and requests a stop during shutdown, even if the main task only reads data. Shared Robot programs use real hardware unless `--simulated` is supplied.

## Build and run

Activate the matching native SDK, then build with CMake:

```bash
cmake -S examples/vega_1/cpp -B build/cpp -DCMAKE_BUILD_TYPE=Release
cmake --build build/cpp --parallel 4
build/cpp/bin/basic_examples/control/cycle_arm --simulated --profile vega_1 --delta 0.01
```

Sourcing the SDK's `activate` script sets the CMake search path and places the
watchdog executable on `PATH`. Alternatively, pass
`-Ddexcontrol_DIR=/path/to/sdk/lib/cmake/dexcontrol` to CMake.

C++ camera monitors open a Rerun viewer by default; `--no-display` prints frame metadata. Other sensor monitors print summaries. No sensor data is saved by default. Binaries mirror the source paths under `build/bin/`.

## Choosing an example

All `monitor_*` examples run continuously by default until Ctrl+C. Use `--samples N` for a finite number of polls. Native camera monitors also stop when their viewer window is closed.

Names distinguish an all-zero sweep from a small offset, direct torso setpoints from tracked motions, repeated monitoring from a single read, and brake operations from motor modes. Named joint poses are not Cartesian poses or homing procedures.

`replay_trajectory` has significant setup behavior: Python/C++ fold arms, send head home, crouch the torso, and close hands before playback; Rust moves directly to the recorded first row. All three languages read the same CSV recording (`examples/vega_1/data/vega-1_dance.csv`). See the behavior reference before choosing a replay program.

Managed motion does not provide collision avoidance. Wait timeouts bound client waiting and do not cancel the server operation; cleanup requests a stop. Only `display_robot_info` and `measure_robot_clock_offset` use read-only DiagnosticClient connections. `clear_component_errors` is a mutating maintenance command.

## Shared programs

| Program | Main task |
| --- | --- |
| [advanced_examples/steer_chassis_sine.cpp](advanced_examples/steer_chassis_sine.cpp) | Drive forward while sinusoidally changing both steering angles. |
| [advanced_examples/configure_arm_pid.cpp](advanced_examples/configure_arm_pid.cpp) | Read or set arm PID multipliers. |
| [advanced_examples/configure_ee_baud_rate.cpp](advanced_examples/configure_ee_baud_rate.cpp) | Read or set end-effector RS485 baud rates. |
| [advanced_examples/configure_arm_ft_sensor.cpp](advanced_examples/configure_arm_ft_sensor.cpp) | Read, enable, or disable arm force-torque sensor modes. |
| [advanced_examples/send_end_effector_bytes.cpp](advanced_examples/send_end_effector_bytes.cpp) | Send raw bytes to end effectors and poll for replies. |
| [advanced_examples/manage_arm_brakes.cpp](advanced_examples/manage_arm_brakes.cpp) | Read, release, or engage arm brakes. |
| [advanced_examples/monitor_joint_currents.cpp](advanced_examples/monitor_joint_currents.cpp) | Monitor joint currents for components that report per-joint current. |
| [advanced_examples/estop_robot.cpp](advanced_examples/estop_robot.cpp) | Read, activate, or deactivate the software E-stop. |
| [advanced_examples/pose_both_arms.cpp](advanced_examples/pose_both_arms.cpp) | Move both arms to a named pose using independent tracked motions. |
| [advanced_examples/fold_or_unfold_robot.cpp](advanced_examples/fold_or_unfold_robot.cpp) | Fold or unfold the robot in ordered, verified stages. |
| [advanced_examples/pose_arms_sequentially.cpp](advanced_examples/pose_arms_sequentially.cpp) | Move arms sequentially to a named pose without collision checking. |
| [advanced_examples/keyboard_jog_joint.cpp](advanced_examples/keyboard_jog_joint.cpp) | Jog a selected joint using keyboard commands. |
| [advanced_examples/move_to_pose/pose_arm.cpp](advanced_examples/move_to_pose/pose_arm.cpp) | Move one arm to a named pose using its model-declared reference frame. |
| [advanced_examples/move_to_pose/pose_head.cpp](advanced_examples/move_to_pose/pose_head.cpp) | Move the head to a named pose using its model-declared reference frame. |
| [advanced_examples/move_to_pose/pose_torso.cpp](advanced_examples/move_to_pose/pose_torso.cpp) | Move the torso to a named joint pose. |
| [advanced_examples/reboot_control_board.cpp](advanced_examples/reboot_control_board.cpp) | Request a reboot of one robot control board. |
| [advanced_examples/replay_trajectory.cpp](advanced_examples/replay_trajectory.cpp) | Prepare robot joint positions and replay a recorded trajectory. |
| [basic_examples/control/close_hand.cpp](basic_examples/control/close_hand.cpp) | Close one hand using its model-defined pose. |
| [basic_examples/control/set_head_motor_mode.cpp](basic_examples/control/set_head_motor_mode.cpp) | Enable or disable head motors. |
| [basic_examples/control/cycle_arm.cpp](basic_examples/control/cycle_arm.cpp) | Move one arm joint by an offset, then return to its starting position. |
| [basic_examples/control/exercise_chassis.cpp](basic_examples/control/exercise_chassis.cpp) | Drive the chassis in six directions, requesting a stop between them. |
| [basic_examples/control/cycle_hand.cpp](basic_examples/control/cycle_hand.cpp) | Close one hand, then reopen it. |
| [basic_examples/control/sweep_head_joints.cpp](basic_examples/control/sweep_head_joints.cpp) | Sweep head joints through positive, negative, and zero targets. |
| [basic_examples/control/cycle_torso.cpp](basic_examples/control/cycle_torso.cpp) | Move the torso to an offset joint target, then return to its starting position. |
| [basic_examples/control/open_hand.cpp](basic_examples/control/open_hand.cpp) | Open one hand using its model-defined pose. |
| [basic_examples/sensors/read_2d_lidar_scan.cpp](basic_examples/sensors/read_2d_lidar_scan.cpp) | Read and display one 2D LiDAR scan. |
| [basic_examples/sensors/monitor_3d_lidar_points.cpp](basic_examples/sensors/monitor_3d_lidar_points.cpp) | Monitor point clouds from one 3D LiDAR. |
| [basic_examples/sensors/monitor_lidar_imu.cpp](basic_examples/sensors/monitor_lidar_imu.cpp) | Monitor the IMU associated with a 3D LiDAR. |
| [basic_examples/sensors/read_arm_buttons.cpp](basic_examples/sensors/read_arm_buttons.cpp) | Read and print arm wrist-button states. |
| [basic_examples/sensors/read_battery_status.cpp](basic_examples/sensors/read_battery_status.cpp) | Read and print battery status. |
| [basic_examples/sensors/monitor_chassis_cameras.cpp](basic_examples/sensors/monitor_chassis_cameras.cpp) | Monitor frames from configured chassis cameras. |
| [basic_examples/sensors/read_arm_force_torque.cpp](basic_examples/sensors/read_arm_force_torque.cpp) | Read and print one arm force-torque observation. |
| [basic_examples/sensors/read_hand_touch_forces.cpp](basic_examples/sensors/read_hand_touch_forces.cpp) | Read and print fingertip forces from one hand. |
| [basic_examples/sensors/monitor_head_camera.cpp](basic_examples/sensors/monitor_head_camera.cpp) | Monitor frames from the configured head camera. |
| [basic_examples/sensors/read_imu_states.cpp](basic_examples/sensors/read_imu_states.cpp) | Read and print observations from configured IMUs. |
| [basic_examples/sensors/monitor_2d_lidar_scans.cpp](basic_examples/sensors/monitor_2d_lidar_scans.cpp) | Monitor live 2D LiDAR scans. |
| [basic_examples/sensors/read_joint_states.cpp](basic_examples/sensors/read_joint_states.cpp) | Print current joint states and chassis state. |
| [basic_examples/sensors/read_component_temperatures.cpp](basic_examples/sensors/read_component_temperatures.cpp) | Print component temperatures and battery status. |
| [basic_examples/sensors/monitor_wrist_cameras.cpp](basic_examples/sensors/monitor_wrist_cameras.cpp) | Monitor frames from the configured wrist cameras. |
| [benchmark/arm_tracking/benchmark_arm_sine.cpp](benchmark/arm_tracking/benchmark_arm_sine.cpp) | Benchmark each arm joint with a sine trajectory and velocity feed-forward. |
| [benchmark/arm_tracking/benchmark_arm_step.cpp](benchmark/arm_tracking/benchmark_arm_step.cpp) | Benchmark each arm joint with a ramped step trajectory. |
| [benchmark/connection/benchmark_connection_latency.cpp](benchmark/connection/benchmark_connection_latency.cpp) | Measure full Robot connection time and query round-trip latency. |
| [troubleshooting/measure_robot_clock_offset.cpp](troubleshooting/measure_robot_clock_offset.cpp) | Estimate robot clock offset and network round-trip time. |
| [troubleshooting/clear_component_errors.cpp](troubleshooting/clear_component_errors.cpp) | Clear errors on all supported robot components. |
| [troubleshooting/display_robot_info.cpp](troubleshooting/display_robot_info.cpp) | Display robot topology, observed state, and server diagnostics. |

`cycle_torso` supports `--relative` for the outward offset. Both modes use planned moves and return to the stored absolute start only after outward success. The default delta is 0.05 rad; select another value with `--delta`.

## Image visualization

Camera monitors open an owned Rerun window when native visualization support
is built. Packaged control-SDK examples use `--no-display` for terminal output.
Install the viewer with
`pip install rerun-sdk==0.36.2`; native image logging does not call Python.
Use `--no-display` for terminal output or `--viewer-url HOST:PORT` for an
existing viewer. Closing the window stops monitoring; exiting or killing the
monitor closes its owned window. The native viewer build requires Rust 1.95+.

With native visualization enabled, the reusable viewer supports both live
streams and one-time images independently of Robot. See
[visualization behavior](../../BEHAVIOR.md#connections-waiting-and-output).

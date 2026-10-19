<!--
Copyright (C) 2026 Dexmate Inc.

This software is dual-licensed:

1. GNU Affero General Public License v3.0 (AGPL-3.0)
   See LICENSE-AGPL for details

2. Commercial License
   For commercial licensing terms, contact: contact@dexmate.ai
-->

# Python examples

See [what each example does](../../BEHAVIOR.md) for exact defaults, setup motions, output, completion conditions and language differences. [Renamed files](../../RENAMED.md) maps old paths to current paths.

There are **45 shared tasks**: 43 use a Robot control connection and two use read-only diagnostics. A Robot connection may enable compatible arm modes at startup and requests a stop during shutdown, even if the main task only reads data. Shared Robot programs use real hardware unless `--simulated` is supplied.

## Build and run

```bash
pip install -e '.[examples]'
python examples/vega_1/python/basic_examples/control/cycle_arm.py --help
python examples/vega_1/python/basic_examples/control/cycle_arm.py --simulated --profile vega_1 --delta 0.01
```

After Rust changes, `python tools/develop_python.py --profile release` builds/installs the wheel and refreshes the matching source-tree extension. `dexcontrol.build_info()` identifies the loaded build.

Python camera/LiDAR/current monitors open or connect a Rerun viewer by default; `--no-display` prints instead. Each example launches its own viewer on a separate port so an old window is not silently reused. These viewers close when the example exits, including Ctrl+C, SIGTERM, or SIGKILL. An existing viewer selected with `--viewer-url` remains open. Missing `rerun-sdk` falls back to printing. Benchmarks additionally need NumPy/matplotlib; planning and arm gamepad programs need dexmotion; gamepad programs need a DualSense controller.

## Choosing an example

All `monitor_*` examples run continuously by default until Ctrl+C. Use `--samples N` for a finite number of polls. Native camera monitors also stop when their viewer window is closed.

Names distinguish an all-zero sweep from a small offset, direct torso setpoints from tracked motions, repeated monitoring from a single read, and brake operations from motor modes. Named joint poses are not Cartesian poses or homing procedures.

`replay_trajectory` has significant setup behavior: Python/C++ fold arms, send head home, crouch the torso, and close hands before playback; Rust moves directly to the recorded first row. All three languages read the same CSV recording (`examples/vega_1/data/vega-1_dance.csv`). See the behavior reference before choosing a replay program.

Managed motion does not provide collision avoidance. Wait timeouts bound client waiting and do not cancel the server operation; cleanup requests a stop. Only `display_robot_info` and `measure_robot_clock_offset` use read-only DiagnosticClient connections. `clear_component_errors` is a mutating maintenance command.

## Shared programs

| Program | Main task |
| --- | --- |
| [advanced_examples/steer_chassis_sine.py](advanced_examples/steer_chassis_sine.py) | Drive forward while sinusoidally changing both steering angles. |
| [advanced_examples/configure_arm_pid.py](advanced_examples/configure_arm_pid.py) | Read or set arm PID multipliers. |
| [advanced_examples/configure_ee_baud_rate.py](advanced_examples/configure_ee_baud_rate.py) | Read or set end-effector RS485 baud rates. |
| [advanced_examples/configure_arm_ft_sensor.py](advanced_examples/configure_arm_ft_sensor.py) | Read, enable, or disable arm force-torque sensor modes. |
| [advanced_examples/send_end_effector_bytes.py](advanced_examples/send_end_effector_bytes.py) | Send raw bytes to end effectors and poll for replies. |
| [advanced_examples/manage_arm_brakes.py](advanced_examples/manage_arm_brakes.py) | Read, release, or engage arm brakes. |
| [advanced_examples/monitor_joint_currents.py](advanced_examples/monitor_joint_currents.py) | Monitor joint currents for components that report per-joint current. |
| [advanced_examples/estop_robot.py](advanced_examples/estop_robot.py) | Read, activate, or deactivate the software E-stop. |
| [advanced_examples/pose_both_arms.py](advanced_examples/pose_both_arms.py) | Move both arms to a named pose using independent tracked motions. |
| [advanced_examples/fold_or_unfold_robot.py](advanced_examples/fold_or_unfold_robot.py) | Fold or unfold the robot in ordered, verified stages. |
| [advanced_examples/pose_arms_sequentially.py](advanced_examples/pose_arms_sequentially.py) | Move arms sequentially to a named pose without collision checking. |
| [advanced_examples/keyboard_jog_joint.py](advanced_examples/keyboard_jog_joint.py) | Jog a selected joint using keyboard commands. |
| [advanced_examples/move_to_pose/pose_arm.py](advanced_examples/move_to_pose/pose_arm.py) | Move arms to a named pose using their model-declared reference frames. |
| [advanced_examples/move_to_pose/pose_head.py](advanced_examples/move_to_pose/pose_head.py) | Move the head to a named pose using its model-declared reference frame. |
| [advanced_examples/move_to_pose/pose_torso.py](advanced_examples/move_to_pose/pose_torso.py) | Move the torso to a named joint pose. |
| [advanced_examples/reboot_control_board.py](advanced_examples/reboot_control_board.py) | Request a reboot of one robot control board. |
| [advanced_examples/replay_trajectory.py](advanced_examples/replay_trajectory.py) | Prepare robot joint positions and replay a recorded trajectory. |
| [basic_examples/control/close_hand.py](basic_examples/control/close_hand.py) | Close one hand using its model-defined pose. |
| [basic_examples/control/set_head_motor_mode.py](basic_examples/control/set_head_motor_mode.py) | Enable or disable head motors. |
| [basic_examples/control/cycle_arm.py](basic_examples/control/cycle_arm.py) | Move one arm joint by an offset, then return to its starting position. |
| [basic_examples/control/exercise_chassis.py](basic_examples/control/exercise_chassis.py) | Drive the chassis in six directions, requesting a stop between them. |
| [basic_examples/control/cycle_hand.py](basic_examples/control/cycle_hand.py) | Close one hand, then reopen it. |
| [basic_examples/control/sweep_head_joints.py](basic_examples/control/sweep_head_joints.py) | Sweep head joints through positive, negative, and zero targets. |
| [basic_examples/control/cycle_torso.py](basic_examples/control/cycle_torso.py) | Move the torso to an offset joint target, then return to its starting position. |
| [basic_examples/control/open_hand.py](basic_examples/control/open_hand.py) | Open one hand using its model-defined pose. |
| [basic_examples/sensors/read_2d_lidar_scan.py](basic_examples/sensors/read_2d_lidar_scan.py) | Read and display one 2D LiDAR scan. |
| [basic_examples/sensors/monitor_3d_lidar_points.py](basic_examples/sensors/monitor_3d_lidar_points.py) | Monitor point clouds from one 3D LiDAR. |
| [basic_examples/sensors/monitor_lidar_imu.py](basic_examples/sensors/monitor_lidar_imu.py) | Monitor the IMU associated with a 3D LiDAR. |
| [basic_examples/sensors/read_arm_buttons.py](basic_examples/sensors/read_arm_buttons.py) | Read and print arm wrist-button states. |
| [basic_examples/sensors/read_battery_status.py](basic_examples/sensors/read_battery_status.py) | Read and print battery status. |
| [basic_examples/sensors/monitor_chassis_cameras.py](basic_examples/sensors/monitor_chassis_cameras.py) | Monitor frames from configured chassis cameras. |
| [basic_examples/sensors/read_arm_force_torque.py](basic_examples/sensors/read_arm_force_torque.py) | Read and print one arm force-torque observation. |
| [basic_examples/sensors/read_hand_touch_forces.py](basic_examples/sensors/read_hand_touch_forces.py) | Read and print fingertip forces from one hand. |
| [basic_examples/sensors/monitor_head_camera.py](basic_examples/sensors/monitor_head_camera.py) | Monitor frames from the configured head camera. |
| [basic_examples/sensors/read_imu_states.py](basic_examples/sensors/read_imu_states.py) | Read and print observations from configured IMUs. |
| [basic_examples/sensors/monitor_2d_lidar_scans.py](basic_examples/sensors/monitor_2d_lidar_scans.py) | Monitor live 2D LiDAR scans. |
| [basic_examples/sensors/read_joint_states.py](basic_examples/sensors/read_joint_states.py) | Print current joint states and chassis state. |
| [basic_examples/sensors/read_component_temperatures.py](basic_examples/sensors/read_component_temperatures.py) | Print component temperatures and battery status. |
| [basic_examples/sensors/monitor_wrist_cameras.py](basic_examples/sensors/monitor_wrist_cameras.py) | Monitor frames from the configured wrist cameras. |
| [benchmark/arm_tracking/benchmark_arm_sine.py](benchmark/arm_tracking/benchmark_arm_sine.py) | Benchmark each arm joint with a sine trajectory and velocity feed-forward. |
| [benchmark/arm_tracking/benchmark_arm_step.py](benchmark/arm_tracking/benchmark_arm_step.py) | Benchmark each arm joint with a ramped step trajectory. |
| [benchmark/connection/benchmark_connection_latency.py](benchmark/connection/benchmark_connection_latency.py) | Measure full Robot connection time and query round-trip latency. |
| [troubleshooting/measure_robot_clock_offset.py](troubleshooting/measure_robot_clock_offset.py) | Estimate robot clock offset and network round-trip time. |
| [troubleshooting/clear_component_errors.py](troubleshooting/clear_component_errors.py) | Clear errors on all supported robot components. |
| [troubleshooting/display_robot_info.py](troubleshooting/display_robot_info.py) | Display robot topology, observed state, and server diagnostics. |

## Python-only programs and helpers

| File | Behavior |
| --- | --- |
| [advanced_examples/run_cartesian_admittance.py](advanced_examples/run_cartesian_admittance.py) | Run wrench-based Cartesian admittance on both arms until interrupted. |
| [advanced_examples/plan_arm_pose.py](advanced_examples/plan_arm_pose.py) | Plan and execute a both-arm named-pose move with dexmotion. |
| [advanced_examples/offset_end_effectors.py](advanced_examples/offset_end_effectors.py) | Execute four planned relative end-effector moves with Enter prompts. |
| [basic_examples/sensors/_viewer.py](basic_examples/sensors/_viewer.py) | Shared Rerun display helpers; not an executable example. |
| [benchmark/arm_tracking/_tracking_helpers.py](benchmark/arm_tracking/_tracking_helpers.py) | Shared seven-joint tracking reference, CSV, and validation helpers. |
| [benchmark/connection/stress_heartbeat_gil.py](benchmark/connection/stress_heartbeat_gil.py) | Monitor heartbeat status while creating Python GIL contention. |
| [teleop/gamepad_arm_cartesian_control.py](teleop/gamepad_arm_cartesian_control.py) | Control arm end-effector translation and rotation with a DualSense gamepad. |
| [teleop/gamepad_arm_joint_control.py](teleop/gamepad_arm_joint_control.py) | Open available hands, then control arm joints with a DualSense gamepad. |
| [teleop/_arm_teleop_base.py](teleop/_arm_teleop_base.py) | Shared arm teleoperation and dexmotion IK helpers; not an executable example. |
| [teleop/gamepad_chassis_head.py](teleop/gamepad_chassis_head.py) | Move the head home, then control chassis and head with a DualSense gamepad. |
| [teleop/_dualsense_teleop_base.py](teleop/_dualsense_teleop_base.py) | Shared DualSense connection, button mapping, and cleanup helpers. |

Gamepad startup is not entirely L1-gated: the joint controller opens available hands first, and the chassis/head controller sends the head home first. Arm gamepad IK helpers pause the SDK heartbeat monitor and resume it on cleanup. See each program’s help for those side effects.

`cycle_torso` supports `--relative` for the outward offset. Both modes use planned moves and return to the stored absolute start only after outward success. The default delta is 0.05 rad; select another value with `--delta`.

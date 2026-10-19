<!--
Copyright (C) 2026 Dexmate Inc.

This software is dual-licensed:

1. GNU Affero General Public License v3.0 (AGPL-3.0)
   See LICENSE-AGPL for details

2. Commercial License
   For commercial licensing terms, contact: contact@dexmate.ai
-->

# What each example does

This reference describes source behavior with default options on a compatible robot.
It is not evidence of physical execution or convergence. See `--help` for overrides.

## Connections, waiting, and output

- Except for the two DiagnosticClient scripts, a `Robot` connection can enable compatible arm modes during startup. Shutdown requests a stop even for an example whose main body only reads state.
- Shared control examples use real hardware unless `--simulated` is passed. Python-only planning/gamepad/heartbeat programs have no simulation flag. Required action/file arguments are validated before the main task runs.
- Managed joint motion does not imply collision avoidance. A caller wait timeout does not cancel the server operation; cleanup attempts to stop it. Group publication is best-effort, not synchronized execution.
- Many small motion examples use `wait()`, which can return a cancelled or superseded terminal status. Returning from that wait alone does not prove the target was reached; `wait_success()` requires successful completion.
- Sensor loops poll cached observations. A sample/iteration count does not promise that many unique fresh frames; unavailable frames may produce no output. Rust sampling loops generally also pause after the final read, while Python/C++ scalar loops usually pause only between reads.
- Python LiDAR, camera and current monitors open/connect a Rerun viewer unless `--no-display`; missing rerun-sdk falls back to printing. C++/Rust camera monitors open an owned Rerun viewer; --no-display prints frame metadata. Other native sensor monitors print summaries.
- All monitor_* examples run continuously by default until Ctrl+C; --samples N sets a finite poll limit. Native camera monitors also stop when their viewer window is closed.
- Only tracking benchmarks write measurement files by default. Polling a sensor does not save its data. Interactive scripts continue until their documented quit/interrupt condition.

## Shared tasks

### `advanced_examples/steer_chassis_sine`

Drive forward while sinusoidally changing both steering angles.

For 6 s at 50 Hz, command both wheel speeds to 0.5 m/s and both steering angles to 0.6*sin(2*pi*0.5*t) rad. Request a stop on completion/cleanup. Requires a steer-drive base; it does not use generic chassis yaw velocity.

Connection: **control**.

[Python](vega_1/python/advanced_examples/steer_chassis_sine.py) · [C++](vega_1/cpp/advanced_examples/steer_chassis_sine.cpp) · [Rust](vega_1/rust/src/advanced_examples/steer_chassis_sine.rs) (`advanced-steer-chassis-sine`)

### `advanced_examples/configure_arm_pid`

Read or set arm PID multipliers.

Require positional get or set. Address both arms by default; set sends seven multipliers, each defaulting to 1.0. Print service replies. PID writes can take about 40 s; the SDK waits up to 45 s for a reply. A timeout does not cancel a server-side write.

Connection: **control**.

[Python](vega_1/python/advanced_examples/configure_arm_pid.py) · [C++](vega_1/cpp/advanced_examples/configure_arm_pid.cpp) · [Rust](vega_1/rust/src/advanced_examples/configure_arm_pid.rs) (`advanced-configure-arm-pid`)

### `advanced_examples/configure_ee_baud_rate`

Read or set end-effector RS485 baud rates.

Require positional get or set. Address both arm end-effector interfaces by default; set uses 115200 baud unless overridden. Print service replies.

Connection: **control**.

[Python](vega_1/python/advanced_examples/configure_ee_baud_rate.py) · [C++](vega_1/cpp/advanced_examples/configure_ee_baud_rate.cpp) · [Rust](vega_1/rust/src/advanced_examples/configure_ee_baud_rate.rs) (`advanced-configure-ee-baud-rate`)

### `advanced_examples/configure_arm_ft_sensor`

Read, enable, or disable arm force-torque sensor modes.

Require positional get, enable, or disable. Address both arms by default and print service replies; this does not zero the force sensors.

Connection: **control**.

[Python](vega_1/python/advanced_examples/configure_arm_ft_sensor.py) · [C++](vega_1/cpp/advanced_examples/configure_arm_ft_sensor.cpp) · [Rust](vega_1/rust/src/advanced_examples/configure_arm_ft_sensor.rs) (`advanced-configure-arm-ft-sensor`)

### `advanced_examples/send_end_effector_bytes`

Send raw bytes to end effectors and poll for replies.

By default, send hex 09 10 03 E8 00 03 06 01 00 00 00 00 00 72 E1 to both configured arm pass-through interfaces and poll for up to 1 s each. The attached device determines what these bytes do; this is not a generic motion command.

Connection: **control**.

[Python](vega_1/python/advanced_examples/send_end_effector_bytes.py) · [C++](vega_1/cpp/advanced_examples/send_end_effector_bytes.cpp) · [Rust](vega_1/rust/src/advanced_examples/send_end_effector_bytes.rs) (`advanced-send-end-effector-bytes`)

### `advanced_examples/manage_arm_brakes`

Read, release, or engage arm brakes.

Require positional status, release, or engage. status and engage address both arms and all their joints unless --side or --joint narrow them. release has no defaults: it requires --side left, right, or both, and either --joint INDEX (repeatable) or --all. Before releasing, warn that the arm must be physically supported, because a released joint can drop under gravity, and require typing yes; --yes skips the prompt and simulated runs do not prompt. This operates the brake service, not the motor-mode service.

Connection: **control**.

[Python](vega_1/python/advanced_examples/manage_arm_brakes.py) · [C++](vega_1/cpp/advanced_examples/manage_arm_brakes.cpp) · [Rust](vega_1/rust/src/advanced_examples/manage_arm_brakes.rs) (`advanced-manage-arm-brakes`)

### `advanced_examples/monitor_joint_currents`

Monitor joint currents for components that report per-joint current.

Probe all components by default, skip unsupported current readings, then poll continuously at 0.02 s intervals until Ctrl-C; --samples N stops after N polls. Fail if none report current; --component limits the selection.

Connection: **control**.

[Python](vega_1/python/advanced_examples/monitor_joint_currents.py) · [C++](vega_1/cpp/advanced_examples/monitor_joint_currents.cpp) · [Rust](vega_1/rust/src/advanced_examples/monitor_joint_currents.rs) (`advanced-monitor-joint-currents`)

- **python**: Starts/connects a Rerun viewer by default; --no-display prints instead. Missing rerun-sdk falls back to printing.
- **cpp**: Prints numeric summaries/frame metadata; does not open a viewer.
- **rust**: Prints numeric summaries/frame metadata; does not open a viewer.

### `advanced_examples/estop_robot`

Read, activate, or deactivate the software E-stop.

Default positional action is status. activate requests software E-stop; deactivate clears it. Print the resulting observed status.

Connection: **control**.

[Python](vega_1/python/advanced_examples/estop_robot.py) · [C++](vega_1/cpp/advanced_examples/estop_robot.cpp) · [Rust](vega_1/rust/src/advanced_examples/estop_robot.rs) (`advanced-estop-robot`)

### `advanced_examples/pose_both_arms`

Move both arms to a named pose using independent tracked motions.

Default to folded at velocity scale 0.5. Resolve each pose using its model-declared frame: folded poses are joint-relative; orientation reference poses use torso pitch. --passthrough uses raw stored values. Start both targets through best-effort fan-out, wait up to 10 s, and require aggregate success. No synchronized start or collision checking is provided.

Connection: **control**.

[Python](vega_1/python/advanced_examples/pose_both_arms.py) · [C++](vega_1/cpp/advanced_examples/pose_both_arms.cpp) · [Rust](vega_1/rust/src/advanced_examples/pose_both_arms.rs) (`advanced-pose-both-arms`)

### `advanced_examples/fold_or_unfold_robot`

Fold or unfold the robot in ordered, verified stages.

Default: close available hands, move both arms to folded, then torso to folded and head to tucked. With --unfold: move torso to crouch45_high, then resolve/move head home, then resolve/move arms to L_shape. Skip absent components; check motion success and measured positions within 0.1 rad before advancing. No collision checking is provided.

Connection: **control**.

[Python](vega_1/python/advanced_examples/fold_or_unfold_robot.py) · [C++](vega_1/cpp/advanced_examples/fold_or_unfold_robot.cpp) · [Rust](vega_1/rust/src/advanced_examples/fold_or_unfold_robot.rs) (`advanced-fold-or-unfold-robot`)

- **python**: Wait 1 s after closing each available hand before the arm stage.
- **cpp**: Wait 1 s after closing each available hand before the arm stage.
- **rust**: Publish hand-close setpoints without a fixed settling delay before the arm stage.

### `advanced_examples/pose_arms_sequentially`

Move arms sequentially to a named pose without collision checking.

Move the left arm and then the right arm to L_shape at velocity scale 0.5, waiting up to 10 s each. --side can select one arm. Resolve model-declared pose frames, including torso compensation for L_shape. No homing or collision planning is performed.

Connection: **control**.

[Python](vega_1/python/advanced_examples/pose_arms_sequentially.py) · [C++](vega_1/cpp/advanced_examples/pose_arms_sequentially.cpp) · [Rust](vega_1/rust/src/advanced_examples/pose_arms_sequentially.rs) (`advanced-pose-arms-sequentially`)

### `advanced_examples/keyboard_jog_joint`

Jog a selected joint using keyboard commands.

Default component is left_arm and selected joint is 0. w/s move in opposite directions, digits select a joint, and q exits. Interaction and command increments differ by language; see the language notes.

Connection: **control**.

[Python](vega_1/python/advanced_examples/keyboard_jog_joint.py) · [C++](vega_1/cpp/advanced_examples/keyboard_jog_joint.cpp) · [Rust](vega_1/rust/src/advanced_examples/keyboard_jog_joint.rs) (`advanced-keyboard-jog-joint`)

- **python**: Raw-terminal hold/repeat w/s input at 100 Hz and 0.2 rad/s. A tap moves the joint by up to 0.05 rad; a hold keeps moving and stops within 0.05 rad of the release. Runs until q/Ctrl-C; no duration option.
- **cpp**: Same hold/repeat and tap behaviour, with an optional duration limit; raw-terminal input on Linux.
- **rust**: Line input: press Enter after each command. Each w/s applies a 0.02 rad direct step by default; EOF or q exits. Does not implement key-hold velocity control.

### `advanced_examples/move_to_pose/pose_arm`

Move arms to a named pose using their model-declared reference frames.

Move the right arm to L_shape with model-defined torso-pitch compensation; wait up to 10 s. folded and folded_closed_hand stay joint-relative and receive no compensation. zero stores seven zeros relative to an upright torso and compensates only torso tilt from upright; --passthrough returns literal zeros. --passthrough uses raw stored values while retaining target validation. No collision checking is performed.

Connection: **control**.

[Python](vega_1/python/advanced_examples/move_to_pose/pose_arm.py) · [C++](vega_1/cpp/advanced_examples/move_to_pose/pose_arm.cpp) · [Rust](vega_1/rust/src/advanced_examples/move_to_pose/pose_arm.rs) (`advanced-pose-arm`)

- **python**: --side both resolves both arm targets before starting a group motion and waits up to --timeout seconds for the group. Starts are not synchronized.

### `advanced_examples/move_to_pose/pose_head`

Move the head to a named pose using its model-declared reference frame.

Move to home using model-defined torso-pitch compensation; tucked stays joint-relative. --passthrough uses raw stored values. Wait up to 10 s. No collision checking is performed.

Connection: **control**.

[Python](vega_1/python/advanced_examples/move_to_pose/pose_head.py) · [C++](vega_1/cpp/advanced_examples/move_to_pose/pose_head.cpp) · [Rust](vega_1/rust/src/advanced_examples/move_to_pose/pose_head.rs) (`advanced-pose-head`)

### `advanced_examples/move_to_pose/pose_torso`

Move the torso to a named joint pose.

Move to home and wait up to 10 s for tracked completion. No collision checking or automatic return is performed.

Connection: **control**.

[Python](vega_1/python/advanced_examples/move_to_pose/pose_torso.py) · [C++](vega_1/cpp/advanced_examples/move_to_pose/pose_torso.cpp) · [Rust](vega_1/rust/src/advanced_examples/move_to_pose/pose_torso.rs) (`advanced-pose-torso`)

### `advanced_examples/reboot_control_board`

Request a reboot of one robot control board.

Require positional arm, torso, or chassis. The arm board controls both arms. A rebooting board stops controlling its joints, so ask for confirmation first and require typing yes; --yes skips the prompt and simulated runs do not prompt. Send the request and exit; do not wait for the board to come back online.

Connection: **control**.

[Python](vega_1/python/advanced_examples/reboot_control_board.py) · [C++](vega_1/cpp/advanced_examples/reboot_control_board.cpp) · [Rust](vega_1/rust/src/advanced_examples/reboot_control_board.rs) (`advanced-reboot-control-board`)

### `advanced_examples/replay_trajectory`

Prepare robot joint positions and replay a recorded trajectory.

Require a trajectory file and confirmation before motion. Move to preparation/start targets, then stream recorded joint positions with the configured tracking-error guard. Preparation motions and processing options differ by language; see the language notes.

Connection: **control**.

[Python](vega_1/python/advanced_examples/replay_trajectory.py) · [C++](vega_1/cpp/advanced_examples/replay_trajectory.cpp) · [Rust](vega_1/rust/src/advanced_examples/replay_trajectory.rs) (`advanced-replay-trajectory`)

- **python**: CSV input (the format all three languages share; NPZ recordings are still accepted). Prompt, optionally smooth/resample/plot, then prompt before folding both arms, sending head home, crouching torso to crouch20_medium, re-resolving head home for the new torso posture, and closing hands. Press Enter before moving to the recorded first row and streaming. Defaults: 0.1 s position smoothing, speed factor 1, no velocity feed-forward; rate from file or 500 Hz. Checks every sample against the robot model's joint limits before moving and reports violations per joint; --clamp-to-model-limits clips them instead of refusing the file.
- **cpp**: CSV input. Performs the same extra folding/head/torso/hand preparation as Python. Supports smoothing, resampling and velocity feed-forward; --visualize prints a summary instead of a plot. Prompts before preparation and playback.
- **rust**: CSV input. Confirm unless --no-confirm (simulation bypasses the prompt); move directly to the recorded first row, then stream. Does not fold/crouch/close hands or smooth/resample/add velocity feed-forward. Rate from file or 500 Hz.

### `basic_examples/control/close_hand`

Close one hand using its model-defined pose.

Command the right hand closed and wait 2 s. Optional --grasp-torque changes the grip setting for a gripper; omitted values use model defaults. The delay is not a convergence check.

Connection: **control**.

[Python](vega_1/python/basic_examples/control/close_hand.py) · [C++](vega_1/cpp/basic_examples/control/close_hand.cpp) · [Rust](vega_1/rust/src/basic_examples/control/close_hand.rs) (`control-close-hand`)

### `basic_examples/control/set_head_motor_mode`

Enable or disable head motors.

The optional positional mode is enable or disable; without it, send disable. Print the firmware reply; no head target is commanded.

Connection: **control**.

[Python](vega_1/python/basic_examples/control/set_head_motor_mode.py) · [C++](vega_1/cpp/basic_examples/control/set_head_motor_mode.cpp) · [Rust](vega_1/rust/src/basic_examples/control/set_head_motor_mode.rs) (`control-set-head-motor-mode`)

### `basic_examples/control/cycle_arm`

Move one arm joint by an offset, then return to its starting position.

Save the right arm positions, move joint 0 by +0.2 rad, then return to the saved absolute start after outward success. By default compute an absolute outward target; --relative resolves the offset from fresh feedback. Both motions use velocity scale 0.2 and a 10 s wait deadline.

Connection: **control**.

[Python](vega_1/python/basic_examples/control/cycle_arm.py) · [C++](vega_1/cpp/basic_examples/control/cycle_arm.cpp) · [Rust](vega_1/rust/src/basic_examples/control/cycle_arm.rs) (`control-cycle-arm`)

### `basic_examples/control/exercise_chassis`

Drive the chassis in six directions, requesting a stop between them.

Command forward, backward, left, right, counter-clockwise, then clockwise for 3 s each, streaming commands at 50 Hz by default (--control-hz). Linear speed is 0.1 m/s and turn speed is 0.1 rad/s. Request stops between directions and on cleanup; requires a base capable of strafing.

Connection: **control**.

[Python](vega_1/python/basic_examples/control/exercise_chassis.py) · [C++](vega_1/cpp/basic_examples/control/exercise_chassis.cpp) · [Rust](vega_1/rust/src/basic_examples/control/exercise_chassis.rs) (`control-exercise-chassis`)

### `basic_examples/control/cycle_hand`

Close one hand, then reopen it.

Command the right hand to its model-defined closed pose, wait 2 s, command its open pose, then wait 2 s. These delays are not tracked-motion completion checks.

Connection: **control**.

[Python](vega_1/python/basic_examples/control/cycle_hand.py) · [C++](vega_1/cpp/basic_examples/control/cycle_hand.cpp) · [Rust](vega_1/rust/src/basic_examples/control/cycle_hand.rs) (`control-cycle-hand`)

### `basic_examples/control/sweep_head_joints`

Sweep head joints through positive, negative, and zero targets.

First command head joint 0 to -pi/6 rad with other joints zero. Then command each joint to +0.5, -0.5, and zero radians, with all other target joints zero. Finish at all zeros; wait up to 10 s per motion.

Connection: **control**.

[Python](vega_1/python/basic_examples/control/sweep_head_joints.py) · [C++](vega_1/cpp/basic_examples/control/sweep_head_joints.cpp) · [Rust](vega_1/rust/src/basic_examples/control/sweep_head_joints.rs) (`control-sweep-head-joints`)

- **python**: The sequence assumes exactly three head joints.
- **cpp**: The sequence assumes exactly three head joints.
- **rust**: The sequence uses the model joint count.

### `basic_examples/control/cycle_torso`

Move the torso to an offset joint target, then return to its starting position.

Read the torso positions and move joint 0 by +0.1 rad with a planned tracked motion. By default compute an absolute target from the stored start; --relative lets the API resolve the offset against fresh feedback. Always return to the stored absolute start after outward success. Each motion wait has a 10 s deadline.

Connection: **control**.

[Python](vega_1/python/basic_examples/control/cycle_torso.py) · [C++](vega_1/cpp/basic_examples/control/cycle_torso.cpp) · [Rust](vega_1/rust/src/basic_examples/control/cycle_torso.rs) (`control-cycle-torso`)

### `basic_examples/control/open_hand`

Open one hand using its model-defined pose.

Command the right hand open and wait 2 s. Optional --grasp-torque changes the grip setting for a gripper; omitted values use model defaults. The delay is not a convergence check.

Connection: **control**.

[Python](vega_1/python/basic_examples/control/open_hand.py) · [C++](vega_1/cpp/basic_examples/control/open_hand.cpp) · [Rust](vega_1/rust/src/basic_examples/control/open_hand.rs) (`control-open-hand`)

### `basic_examples/sensors/read_2d_lidar_scan`

Read and display one 2D LiDAR scan.

Enable lidar_2d_front, wait up to 10 s for activity, and read one cached scan. Python can visualize XY points; native examples print range statistics. No scan file is saved.

Connection: **control**.

[Python](vega_1/python/basic_examples/sensors/read_2d_lidar_scan.py) · [C++](vega_1/cpp/basic_examples/sensors/read_2d_lidar_scan.cpp) · [Rust](vega_1/rust/src/basic_examples/sensors/read_2d_lidar_scan.rs) (`sensor-read-2d-lidar-scan`)

- **python**: Starts/connects a Rerun viewer by default; --no-display prints instead. Missing rerun-sdk falls back to printing.
- **cpp**: Prints numeric summaries/frame metadata; does not open a viewer.
- **rust**: Prints numeric summaries/frame metadata; does not open a viewer.

### `basic_examples/sensors/monitor_3d_lidar_points`

Monitor point clouds from one 3D LiDAR.

Enable the front 3D LiDAR, wait up to 10 s for activity, and poll continuously at 0.1 s intervals until Ctrl+C. --samples N limits the number of polls. --position back selects the rear LiDAR; no cloud file is saved.

Connection: **control**.

[Python](vega_1/python/basic_examples/sensors/monitor_3d_lidar_points.py) · [C++](vega_1/cpp/basic_examples/sensors/monitor_3d_lidar_points.cpp) · [Rust](vega_1/rust/src/basic_examples/sensors/monitor_3d_lidar_points.rs) (`sensor-monitor-3d-lidar-points`)

- **python**: Starts/connects a Rerun viewer by default; --no-display prints instead. Missing rerun-sdk falls back to printing.
- **cpp**: Prints numeric summaries/frame metadata; does not open a viewer.
- **rust**: Prints numeric summaries/frame metadata; does not open a viewer.

### `basic_examples/sensors/monitor_lidar_imu`

Monitor the IMU associated with a 3D LiDAR.

Enable lidar_3d_front, wait up to 10 s for its IMU stream, and print observations continuously at 0.05 s intervals until Ctrl+C. --samples N limits the number of polls. --sensor selects another sensor with an IMU capability.

Connection: **control**.

[Python](vega_1/python/basic_examples/sensors/monitor_lidar_imu.py) · [C++](vega_1/cpp/basic_examples/sensors/monitor_lidar_imu.cpp) · [Rust](vega_1/rust/src/basic_examples/sensors/monitor_lidar_imu.rs) (`sensor-monitor-lidar-imu`)

### `basic_examples/sensors/read_arm_buttons`

Read and print arm wrist-button states.

Read one right-arm button observation by default. --samples and --period repeat the reads; the default repeat interval is 0.1 s.

Connection: **control**.

[Python](vega_1/python/basic_examples/sensors/read_arm_buttons.py) · [C++](vega_1/cpp/basic_examples/sensors/read_arm_buttons.cpp) · [Rust](vega_1/rust/src/basic_examples/sensors/read_arm_buttons.rs) (`sensor-read-arm-buttons`)

### `basic_examples/sensors/read_battery_status`

Read and print battery status.

Wait up to 5 s for battery activity and print one observation by default, including charge, voltage, current and temperature. Optional repeated reads use a 0.5 s interval.

Connection: **control**.

[Python](vega_1/python/basic_examples/sensors/read_battery_status.py) · [C++](vega_1/cpp/basic_examples/sensors/read_battery_status.cpp) · [Rust](vega_1/rust/src/basic_examples/sensors/read_battery_status.rs) (`sensor-read-battery-status`)

### `basic_examples/sensors/monitor_chassis_cameras`

Monitor frames from configured chassis cameras.

Select base_*_camera sensors from the selected configuration unless --camera is supplied. Subscribe to all advertised streams unless --stream selects a subset; monitor at 30 Hz until Ctrl+C by default. --samples N limits the number of polls. Unsubscribe on exit. No image files are saved.

Connection: **control**.

[Python](vega_1/python/basic_examples/sensors/monitor_chassis_cameras.py) · [C++](vega_1/cpp/basic_examples/sensors/monitor_chassis_cameras.cpp) · [Rust](vega_1/rust/src/basic_examples/sensors/monitor_chassis_cameras.rs) (`sensor-monitor-chassis-cameras`)

- **python**: Starts/connects a Rerun viewer by default; --no-display prints instead. Missing rerun-sdk falls back to printing.
- **cpp**: Opens an owned Rerun image viewer; --no-display prints metadata and --viewer-url HOST:PORT connects to an existing viewer. Closing the window ends monitoring.
- **rust**: Opens an owned Rerun image viewer; --no-display prints metadata and --viewer-url HOST:PORT connects to an existing viewer. Closing the window ends monitoring.

### `basic_examples/sensors/read_arm_force_torque`

Read and print one arm force-torque observation.

Wait up to 5 s for the right-arm wrench stream and print one observation by default. Optional repeated reads use a 0.1 s interval; no calibration or zeroing is requested.

Connection: **control**.

[Python](vega_1/python/basic_examples/sensors/read_arm_force_torque.py) · [C++](vega_1/cpp/basic_examples/sensors/read_arm_force_torque.cpp) · [Rust](vega_1/rust/src/basic_examples/sensors/read_arm_force_torque.rs) (`sensor-read-arm-force-torque`)

### `basic_examples/sensors/read_hand_touch_forces`

Read and print fingertip forces from one hand.

Read the right hand once by default and report missing touch data. Optional repeated reads use a 0.1 s interval.

Connection: **control**.

[Python](vega_1/python/basic_examples/sensors/read_hand_touch_forces.py) · [C++](vega_1/cpp/basic_examples/sensors/read_hand_touch_forces.cpp) · [Rust](vega_1/rust/src/basic_examples/sensors/read_hand_touch_forces.rs) (`sensor-read-hand-touch-forces`)

### `basic_examples/sensors/monitor_head_camera`

Monitor frames from the configured head camera.

Enable head_camera, subscribe to all advertised streams unless selected with --stream, and monitor at 30 Hz until Ctrl+C by default, then unsubscribe. --samples N limits the number of polls. The sensor name is configurable; no particular camera brand is required and no image files are saved.

Connection: **control**.

[Python](vega_1/python/basic_examples/sensors/monitor_head_camera.py) · [C++](vega_1/cpp/basic_examples/sensors/monitor_head_camera.cpp) · [Rust](vega_1/rust/src/basic_examples/sensors/monitor_head_camera.rs) (`sensor-monitor-head-camera`)

- **python**: Starts/connects a Rerun viewer by default; --no-display prints instead. Missing rerun-sdk falls back to printing.
- **cpp**: Opens an owned Rerun image viewer; --no-display prints metadata and --viewer-url HOST:PORT connects to an existing viewer. Closing the window ends monitoring.
- **rust**: Opens an owned Rerun image viewer; --no-display prints metadata and --viewer-url HOST:PORT connects to an existing viewer. Closing the window ends monitoring.

### `basic_examples/sensors/read_imu_states`

Read and print observations from configured IMUs.

Read IMU streams from head_camera and chassis_imu (when declared) once by default, waiting up to 5 s for activity. --sensor changes the requested list; optional repeated reads use a 0.1 s interval.

Connection: **control**.

[Python](vega_1/python/basic_examples/sensors/read_imu_states.py) · [C++](vega_1/cpp/basic_examples/sensors/read_imu_states.cpp) · [Rust](vega_1/rust/src/basic_examples/sensors/read_imu_states.rs) (`sensor-read-imu-states`)

- **python**: Filters undeclared IMU names before connecting.
- **cpp**: Filters undeclared IMU names before connecting.
- **rust**: Filters undeclared IMU names before connecting.

### `basic_examples/sensors/monitor_2d_lidar_scans`

Monitor live 2D LiDAR scans.

Enable lidar_2d_front, wait up to 10 s for activity, and poll continuously at 0.1 s intervals until Ctrl+C. --samples N limits the number of polls. Display XY points or range summaries; no scan file is saved.

Connection: **control**.

[Python](vega_1/python/basic_examples/sensors/monitor_2d_lidar_scans.py) · [C++](vega_1/cpp/basic_examples/sensors/monitor_2d_lidar_scans.cpp) · [Rust](vega_1/rust/src/basic_examples/sensors/monitor_2d_lidar_scans.rs) (`sensor-monitor-2d-lidar-scans`)

- **python**: Starts/connects a Rerun viewer by default; --no-display prints instead. Missing rerun-sdk falls back to printing.
- **cpp**: Prints numeric summaries/frame metadata; does not open a viewer.
- **rust**: Prints numeric summaries/frame metadata; does not open a viewer.

### `basic_examples/sensors/read_joint_states`

Print current joint states and chassis state.

Read all available joint components once, skipping unavailable joint state; --component narrows that list. Also print chassis state when available, independently of the component filter. This is a cached state listing, not a complete diagnostic report.

Connection: **control**.

[Python](vega_1/python/basic_examples/sensors/read_joint_states.py) · [C++](vega_1/cpp/basic_examples/sensors/read_joint_states.cpp) · [Rust](vega_1/rust/src/basic_examples/sensors/read_joint_states.rs) (`sensor-read-joint-states`)

- **python**: Prints named joint rows with model-derived units; unavailable fields appear as —. Chassis steering, wheel encoder positions and wheel speeds are listed separately. Reports nonempty joint error indices by name.
- **cpp**: Prints named joint rows with model-derived units; unavailable fields appear as —. Chassis steering, wheel encoder positions and wheel speeds are listed separately.
- **rust**: Prints named joint rows with model-derived units; unavailable fields appear as —. Chassis steering, wheel encoder positions and wheel speeds are listed separately. Reports nonempty joint error indices.

### `basic_examples/sensors/read_component_temperatures`

Print component temperatures and battery status.

Read each available component temperature stream once, then battery temperature/status. --component narrows component selection; battery reporting is separate. Report unavailable temperature sources.

Connection: **control**.

[Python](vega_1/python/basic_examples/sensors/read_component_temperatures.py) · [C++](vega_1/cpp/basic_examples/sensors/read_component_temperatures.cpp) · [Rust](vega_1/rust/src/basic_examples/sensors/read_component_temperatures.rs) (`sensor-read-component-temperatures`)

### `basic_examples/sensors/monitor_wrist_cameras`

Monitor frames from the configured wrist cameras.

Enable left_wrist_camera and right_wrist_camera unless --sensor supplies names. Subscribe to selected/all advertised streams, monitor at 30 Hz until Ctrl+C by default, then unsubscribe. --samples N limits the number of polls. No particular camera brand is required and no image files are saved.

Connection: **control**.

[Python](vega_1/python/basic_examples/sensors/monitor_wrist_cameras.py) · [C++](vega_1/cpp/basic_examples/sensors/monitor_wrist_cameras.cpp) · [Rust](vega_1/rust/src/basic_examples/sensors/monitor_wrist_cameras.rs) (`sensor-monitor-wrist-cameras`)

- **python**: Starts/connects a Rerun viewer by default; --no-display prints instead. Missing rerun-sdk falls back to printing.
- **cpp**: Opens an owned Rerun image viewer; --no-display prints metadata and --viewer-url HOST:PORT connects to an existing viewer. Closing the window ends monitoring.
- **rust**: Opens an owned Rerun image viewer; --no-display prints metadata and --viewer-url HOST:PORT connects to an existing viewer. Closing the window ends monitoring.

### `benchmark/arm_tracking/benchmark_arm_sine`

Benchmark each arm joint with a sine trajectory and velocity feed-forward.

Prompt for confirmation unless --no-confirm; move the right seven-joint arm to [0,0,0,-0.5,0,0,0] rad. Test each joint for 10 s at 200 Hz with a 0.4 rad, 1 Hz sine; return to the reference between joints and at the end. Write measurement files. This reference is not an all-zero pose. The amplitude is fitted to each joint's limits: the position limits from the reference pose and the velocity limit against the feed-forward peak (amplitude times angular frequency), reduced with a warning where the requested amplitude does not fit. Stream with the step guard disabled (tracking lag is what is measured).

Connection: **control**.

[Python](vega_1/python/benchmark/arm_tracking/benchmark_arm_sine.py) · [C++](vega_1/cpp/benchmark/arm_tracking/benchmark_arm_sine.cpp) · [Rust](vega_1/rust/src/benchmark/arm_tracking/benchmark_arm_sine.rs) (`benchmark-arm-sine`)

- **python**: Writes per-joint CSV, data.npz and summary.png under results beside this script; requires matplotlib.
- **cpp**: Writes per-joint CSV and parameters.txt under the requested output directory.
- **rust**: Writes per-joint CSV and prints tracking errors; no plots/NPZ. --timeout controls managed-motion waits.

### `benchmark/arm_tracking/benchmark_arm_step`

Benchmark each arm joint with a ramped step trajectory.

Prompt unless --no-confirm; move the right seven-joint arm to [0,0,0,-0.5,0,0,0] rad. Test each joint for 5 s at 200 Hz: hold 0.3 s, ramp by pi/6 rad over 0.3 s, then hold. The step is fitted to each joint's limits from the reference pose: the other direction where the requested one does not fit, a smaller step where neither does, each logged. Stream positions without velocity feed-forward and with the step guard disabled (tracking lag is what is measured); return to the reference between joints and at the end. Write measurement files.

Connection: **control**.

[Python](vega_1/python/benchmark/arm_tracking/benchmark_arm_step.py) · [C++](vega_1/cpp/benchmark/arm_tracking/benchmark_arm_step.cpp) · [Rust](vega_1/rust/src/benchmark/arm_tracking/benchmark_arm_step.rs) (`benchmark-arm-step`)

- **python**: Writes per-joint CSV, data.npz and summary.png under results beside this script; requires matplotlib.
- **cpp**: Writes per-joint CSV and parameters.txt under the requested output directory.
- **rust**: Writes per-joint CSV and prints tracking errors; returns to the reference after each joint, with no plots/NPZ.

### `benchmark/connection/benchmark_connection_latency`

Measure full Robot connection time and query round-trip latency.

Create one control connection, then issue 10 version_info queries and print timing statistics. This includes normal Robot startup/cleanup side effects; it is not a read-only DiagnosticClient benchmark.

Connection: **control**.

[Python](vega_1/python/benchmark/connection/benchmark_connection_latency.py) · [C++](vega_1/cpp/benchmark/connection/benchmark_connection_latency.cpp) · [Rust](vega_1/rust/src/benchmark/connection/benchmark_connection_latency.rs) (`benchmark-connection-latency`)

- **python**: Also waits for active state (10 s default timeout) before finishing the connection timer.
- **cpp**: Also waits for active state (10 s default timeout) before finishing the connection timer.
- **rust**: Times builder completion; no separate --timeout/active-state wait.

### `troubleshooting/measure_robot_clock_offset`

Estimate robot clock offset and network round-trip time.

Use a read-only DiagnosticClient to collect 30 time-query samples and report server-minus-client offset, round-trip time, and replies. Does not synchronize or change either clock; no simulation connection is supported.

Connection: **diagnostic**.

[Python](vega_1/python/troubleshooting/measure_robot_clock_offset.py) · [C++](vega_1/cpp/troubleshooting/measure_robot_clock_offset.cpp) · [Rust](vega_1/rust/src/troubleshooting/measure_robot_clock_offset.rs) (`troubleshooting-measure-robot-clock-offset`)

- **python**: Human-readable signed offset and RTT in milliseconds.
- **cpp**: Prints diagnostic JSON.
- **rust**: Pretty-printed diagnostic JSON.

### `troubleshooting/clear_component_errors`

Clear errors on all supported robot components.

Send clear-error requests, print each component outcome, and exit nonzero if any outcome failed. This is a mutating maintenance operation.

Connection: **control**.

[Python](vega_1/python/troubleshooting/clear_component_errors.py) · [C++](vega_1/cpp/troubleshooting/clear_component_errors.cpp) · [Rust](vega_1/rust/src/troubleshooting/clear_component_errors.rs) (`troubleshooting-clear-component-errors`)

### `troubleshooting/display_robot_info`

Display robot topology, observed state, and server diagnostics.

Use a read-only DiagnosticClient and a 0.25 s state observation window, then print the snapshot. Optional --enable-sensor includes declared optional sensors. Does not enable motors or issue stop/motion commands; no simulation connection is supported.

Connection: **diagnostic**.

[Python](vega_1/python/troubleshooting/display_robot_info.py) · [C++](vega_1/cpp/troubleshooting/display_robot_info.cpp) · [Rust](vega_1/rust/src/troubleshooting/display_robot_info.rs) (`troubleshooting-display-robot-info`)

- **python**: Formatted tables, optionally using Rich.
- **cpp**: Formatted native diagnostic tables.
- **rust**: Formatted native diagnostic tables.

## Python-only programs and helpers

### [run_cartesian_admittance.py](vega_1/python/advanced_examples/run_cartesian_admittance.py)

Run wrench-based Cartesian admittance on both arms until interrupted.

Uses dexmotion inverse kinematics and both arm wrench streams. Default gain is 1.0, zero-force mode is enabled, and no blue-button gate is required unless --need-button is set. Streams direct joint targets until Ctrl-C, then requests shutdown. Uses a real Robot control connection with normal startup side effects; no simulation flag.

Connection: **control**.

### [plan_arm_pose.py](vega_1/python/advanced_examples/plan_arm_pose.py)

Plan and execute a both-arm named-pose move with dexmotion.

Connect to the real robot and build a geometry-aware OMPL plan to L_shape with torso-pitch compensation, sampled at 250 Hz. If initially self-colliding, propose a separate escape interpolation and require confirmation. Require confirmation before executing the final path; the dexmotion viewer is on by default (--no-visualize turns it off). Collision checking covers the configured model, not a guarantee about the physical workspace. No simulation flag.

Connection: **control**.

### [offset_end_effectors.py](vega_1/python/advanced_examples/offset_end_effectors.py)

Execute four planned relative end-effector moves with Enter prompts.

Using dexmotion at 250 Hz: move the right end effector +0.1 m in z in frame R_arm_j5; move the left +0.1 m in y; move the left +0.1 m in x in frame L_arm_j3; rotate the right +pi/4 around x. Prompt before each segment and stream both arm joint arrays. Uses a real Robot control connection; no simulation flag.

Connection: **control**.

### [_viewer.py](vega_1/python/basic_examples/sensors/_viewer.py)

Shared Rerun display helpers; not an executable example.

Imported by sensor/current scripts. start() opens/connects a viewer unless --no-display is set; missing rerun-sdk falls back to printing. Running this module directly starts no viewer or robot connection.

Connection: **helper**.

### [_tracking_helpers.py](vega_1/python/benchmark/arm_tracking/_tracking_helpers.py)

Shared seven-joint tracking reference, CSV, and validation helpers.

Imported by arm benchmarks; the reference vector is [0,0,0,-0.5,0,0,0] rad. Contains no standalone benchmark or robot connection.

Connection: **helper**.

### [stress_heartbeat_gil.py](vega_1/python/benchmark/connection/stress_heartbeat_gil.py)

Monitor heartbeat status while creating Python GIL contention.

For 30 s, start four CPU-bound worker threads by default and increase the Python thread switch interval to 0.1 s. Display a Rich heartbeat table; stop workers and restore the interval on normal cleanup. --no-gil-stress disables the load; --disable-heartbeat changes heartbeat policy. Uses a real Robot control connection, not read-only diagnostics; no motion benchmark and no simulation flag.

Connection: **control**.

### [gamepad_arm_cartesian_control.py](vega_1/python/teleop/gamepad_arm_cartesian_control.py)

Control arm end-effector translation and rotation with a DualSense gamepad.

Select the left arm initially; L1 gates interactive movement, R1 switches arms, and R2 toggles translation/rotation mode. Run dexmotion IK and stream arm targets at 400 Hz until interrupted. The arm IK helper pauses the SDK heartbeat monitor and resumes it on cleanup. Uses a real Robot control connection; no simulation flag.

Connection: **control**.

### [gamepad_arm_joint_control.py](vega_1/python/teleop/gamepad_arm_joint_control.py)

Open available hands, then control arm joints with a DualSense gamepad.

Open both available hands before the interactive loop, without requiring L1 for this startup action. Select the left arm initially; L1 gates subsequent joint jogging, R1 switches arms, and face buttons/L2 select joints. Runs until interrupted at 200 Hz by default. Uses dexmotion, pauses the SDK heartbeat monitor in the arm IK helper, resumes it on cleanup, and shuts down the real Robot. No simulation flag.

Connection: **control**.

### [_arm_teleop_base.py](vega_1/python/teleop/_arm_teleop_base.py)

Shared arm teleoperation and dexmotion IK helpers; not an executable example.

Import support for the two arm gamepad scripts. Object construction can connect a Robot and pauses its heartbeat monitor after IK setup; arm cleanup resumes it. Running this module directly defines classes but starts no control loop.

Connection: **helper**.

### [gamepad_chassis_head.py](vega_1/python/teleop/gamepad_chassis_head.py)

Move the head home, then control chassis and head with a DualSense gamepad.

At startup, send a torso-pitch-compensated head home target before L1 is pressed. Then use L1-gated chassis/head control at 400 Hz; default maximum linear/angular speeds are 0.5 m/s and 1 rad/s. Touchpad toggles E-stop; the head-mode control can enable/disable head motors. Runs until interrupted, then requests stop/shutdown. Real Robot connection; no simulation flag.

Connection: **control**.

### [_dualsense_teleop_base.py](vega_1/python/teleop/_dualsense_teleop_base.py)

Shared DualSense connection, button mapping, and cleanup helpers.

Imported by gamepad examples. Instantiation activates the controller and connects a Robot; L1 manages interactive movement state and touchpad toggles software E-stop. Running this module directly defines classes but starts no control loop.

Connection: **helper**.

## Offline conversion tools

## Native support files

`vega_1/rust/src/lib.rs` supplies CLI parsing, connection/cleanup, camera printing and CSV helpers. It is a standalone binary.
C++ headers under `vega_1/cpp/common/` and the arm-tracking directory supply parsing, JSON, CSV, terminal, camera and control helpers. Fixtures under `fixtures/` are test inputs, not programs.

Regenerate with `python tools/generate_example_reference.py`; verify with `--check`.

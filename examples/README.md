<!--
Copyright (C) 2026 Dexmate Inc.

This software is dual-licensed:

1. GNU Affero General Public License v3.0 (AGPL-3.0)
   See LICENSE-AGPL for details

2. Commercial License
   For commercial licensing terms, contact: contact@dexmate.ai
-->

# Cross-language examples

All current robot examples live under `vega_1/`, grouped by language:

```text
examples/vega_1/
  python/
  cpp/
  rust/
```

This suite targets the Vega-1 family (`vega_1`, `vega_1p`, and compatible
upper-body tasks on `vega_1u`). Individual tasks still require their configured
components. The directory name does not override the selected robot or profile.
Future model suites belong in sibling folders; the shared launcher, inventory,
behavior reference and test fixtures remain here.

## Run examples

From the repository root, use the same short example name in any language:

```bash
./run --list
./run rust cycle_arm --simulated --profile vega_1p --delta 0.01
./run cpp cycle_arm --simulated --profile vega_1p --delta 0.01
./run python cycle_arm --simulated --profile vega_1p --delta 0.01
```

The launcher builds the selected Rust/C++ example and reuses incremental builds.
Install and activate the matching native SDK first; it includes the watchdog
executable. Python examples use the installed package in your current Python
environment. See the repository README for package and SDK installation.
The launcher requires Python 3.8+ and has no third-party Python dependencies.

```bash
./run cpp --list
./run rust cycle_arm --help
./run --release cpp cycle_arm --simulated
./run --build-only rust cycle_arm
./run --dry-run cpp cycle_arm --simulated
```

Prerequisite checks follow the selected language: Rust checks Cargo/rustc;
C++ checks its compiler, CMake and the selected Make/Ninja build tool; Python
checks no native compilers. Missing-tool errors print copy-paste installation
commands. Nothing is installed automatically. Help, listing and previews skip
these checks.

For missing C++ tools on Ubuntu/Debian or Ubuntu under WSL2:

```bash
sudo apt-get update
sudo apt-get install build-essential cmake
```

For missing Rust tools:

```bash
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
. "$HOME/.cargo/env"
```

C++ examples link to the installed native SDK and do not need Cargo.
Ninja generators also need `sudo apt-get install ninja-build`.

On macOS, missing compiler tools produce `xcode-select --install`; missing
CMake produces `brew install cmake` for Homebrew users (plus `ninja` when needed).

Launcher options go before the example name; everything after it is passed
through unchanged. `c++` is an alias for `cpp`. `--dry-run` prints commands
without checking dependencies, building, or connecting to a robot.

`examples/vega_1/vega_dance.sh [python|cpp|rust]` replays the bundled dance
recording (`vega_1/data/vega-1_dance.csv`) through the launcher with the
`replay_trajectory` example of that language; `-s SPEED`, `-v` and any
`--option` go to the example (`-h` for details). Python and C++ prepare the
robot and honour speed/visualization; Rust streams the recording as recorded.

Examples retain their existing robot selection and defaults: **control examples
use real hardware unless `--simulated` is supplied**. For a selected physical
robot, `./run rust display_robot_info` or `./run cpp display_robot_info` runs the
read-only diagnostic example; those diagnostics do not support simulation.


See [exact behavior and defaults](BEHAVIOR.md) and the [old-to-new filename map](RENAMED.md).

Filenames use short action-and-subject names, generally two to four words. Exact motion sequences, starting targets, and defaults belong in `--help` and the behavior guide.

Robot connections may enable compatible arm modes during startup and request a stop during shutdown, including programs whose main task only reads state. Only the two DiagnosticClient programs are command-free.

The checked inventory contains **45 tasks**: 43 Robot tasks
and two production-only diagnostics. Every task has Python, C++ and Rust versions;
the Python examples are the reference. Each table row links implementations
of the same task; it does not assert identical flags, outputs or algorithms.

All 43 Robot tasks accept `--simulated --profile vega_1` and `--config FILE`.
Simulation enforces E-stop for joint commands and motion cancellation, checks state arrival, and rejects unsupported publish endpoints. It does not model hardware dynamics, camera streams or every firmware
service. Some examples therefore report unavailable data/services in simulation.
`display_robot_info` and `measure_robot_clock_offset` reject simulation (Python's
argument parser rejects the unsupported flag; native examples reject it explicitly).

Run the same hand-opening task from the repository root. This requires a
hand-equipped profile: `vega_1` alone has no hands, whereas `vega_1_f5d6` does:

```sh
python examples/vega_1/python/basic_examples/control/open_hand.py --simulated --profile vega_1_f5d6 --wait-time 0
/tmp/dexcontrol-examples/bin/basic_examples/control/open_hand --simulated --profile vega_1_f5d6 --wait-time 0
cargo run -p dexcontrol-examples --bin control-open-hand -- --simulated --profile vega_1_f5d6 --wait-time 0
```

Build C++ with `cmake -S examples/vega_1/cpp -B /tmp/dexcontrol-examples
-DDEXCONTROL_IN_TREE=ON -DDEXCONTROL_CARGO_PROFILE=debug` and
`cmake --build /tmp/dexcontrol-examples` after `cargo build -p dexcontrol-capi`.
Install Python as described in its README. Build Rust with
`cargo build -p dexcontrol-examples`.

`python3 tools/check_example_parity.py` checks inventory coverage, Rust binary
registration, Python simulation wiring and Rust cleanup usage. Add
`--smoke --cpp-build /tmp/dexcontrol-examples` to execute nine matching tasks
in all languages and verify simulation typo/diagnostic rejection. This is a
smoke test, not a numerical equivalence or physical safety certification.
The normal verification script runs the inventory check and shared smoke tests.
`tools/check_benchmark_parity.py --cpp-build <build>` additionally checks the
seven-joint step waveform and CSV measurements through all three languages.

Known differences:

- Python camera monitors open Rerun by default (`--no-display` prints metadata). C++ and Rust do the same when native visualization support is built; packaged control-SDK examples use `--no-display`. The reusable native viewer supports live and one-time images without a robot. See [visualization behavior](BEHAVIOR.md#connections-waiting-and-output). Other native sensor examples print readings/statistics.
- Trajectory replay reads the same CSV recording in all three languages
  (`vega_1/data/vega-1_dance.csv`: an optional `control_hz,<rate>` line, a
  `component:joint_index` header, one row per tick). Python still accepts
  NPZ captures.
- Tracking benchmarks share CSV columns, starting pose and step ramp semantics.
  Python additionally saves NPZ files and plots.
- Keyboard interaction differs (Rust is line-oriented). Hardware response and
  timing still require physical validation.
- Python-only workflows: Cartesian planning/IK/admittance through `dexmotion`,
  three DualSense teleoperation programs plus two shared bases, and a GIL
  stress benchmark. `_viewer.py` and benchmark `_tracking_helpers.py` are helpers.
- Native CLI validation uses per-task schemas. Flags from another task are
  rejected before opening a transport.

| Task | Python | C++ | Rust binary / source |
| --- | --- | --- | --- |
| `advanced_examples/steer_chassis_sine` | [source](vega_1/python/advanced_examples/steer_chassis_sine.py) | [source](vega_1/cpp/advanced_examples/steer_chassis_sine.cpp) | [advanced-steer-chassis-sine](vega_1/rust/src/advanced_examples/steer_chassis_sine.rs) |
| `advanced_examples/configure_arm_pid` | [source](vega_1/python/advanced_examples/configure_arm_pid.py) | [source](vega_1/cpp/advanced_examples/configure_arm_pid.cpp) | [advanced-configure-arm-pid](vega_1/rust/src/advanced_examples/configure_arm_pid.rs) |
| `advanced_examples/configure_ee_baud_rate` | [source](vega_1/python/advanced_examples/configure_ee_baud_rate.py) | [source](vega_1/cpp/advanced_examples/configure_ee_baud_rate.cpp) | [advanced-configure-ee-baud-rate](vega_1/rust/src/advanced_examples/configure_ee_baud_rate.rs) |
| `advanced_examples/configure_arm_ft_sensor` | [source](vega_1/python/advanced_examples/configure_arm_ft_sensor.py) | [source](vega_1/cpp/advanced_examples/configure_arm_ft_sensor.cpp) | [advanced-configure-arm-ft-sensor](vega_1/rust/src/advanced_examples/configure_arm_ft_sensor.rs) |
| `advanced_examples/send_end_effector_bytes` | [source](vega_1/python/advanced_examples/send_end_effector_bytes.py) | [source](vega_1/cpp/advanced_examples/send_end_effector_bytes.cpp) | [advanced-send-end-effector-bytes](vega_1/rust/src/advanced_examples/send_end_effector_bytes.rs) |
| `advanced_examples/manage_arm_brakes` | [source](vega_1/python/advanced_examples/manage_arm_brakes.py) | [source](vega_1/cpp/advanced_examples/manage_arm_brakes.cpp) | [advanced-manage-arm-brakes](vega_1/rust/src/advanced_examples/manage_arm_brakes.rs) |
| `advanced_examples/monitor_joint_currents` | [source](vega_1/python/advanced_examples/monitor_joint_currents.py) | [source](vega_1/cpp/advanced_examples/monitor_joint_currents.cpp) | [advanced-monitor-joint-currents](vega_1/rust/src/advanced_examples/monitor_joint_currents.rs) |
| `advanced_examples/estop_robot` | [source](vega_1/python/advanced_examples/estop_robot.py) | [source](vega_1/cpp/advanced_examples/estop_robot.cpp) | [advanced-estop-robot](vega_1/rust/src/advanced_examples/estop_robot.rs) |
| `advanced_examples/pose_both_arms` | [source](vega_1/python/advanced_examples/pose_both_arms.py) | [source](vega_1/cpp/advanced_examples/pose_both_arms.cpp) | [advanced-pose-both-arms](vega_1/rust/src/advanced_examples/pose_both_arms.rs) |
| `advanced_examples/fold_or_unfold_robot` | [source](vega_1/python/advanced_examples/fold_or_unfold_robot.py) | [source](vega_1/cpp/advanced_examples/fold_or_unfold_robot.cpp) | [advanced-fold-or-unfold-robot](vega_1/rust/src/advanced_examples/fold_or_unfold_robot.rs) |
| `advanced_examples/pose_arms_sequentially` | [source](vega_1/python/advanced_examples/pose_arms_sequentially.py) | [source](vega_1/cpp/advanced_examples/pose_arms_sequentially.cpp) | [advanced-pose-arms-sequentially](vega_1/rust/src/advanced_examples/pose_arms_sequentially.rs) |
| `advanced_examples/keyboard_jog_joint` | [source](vega_1/python/advanced_examples/keyboard_jog_joint.py) | [source](vega_1/cpp/advanced_examples/keyboard_jog_joint.cpp) | [advanced-keyboard-jog-joint](vega_1/rust/src/advanced_examples/keyboard_jog_joint.rs) |
| `advanced_examples/move_to_pose/pose_arm` | [source](vega_1/python/advanced_examples/move_to_pose/pose_arm.py) | [source](vega_1/cpp/advanced_examples/move_to_pose/pose_arm.cpp) | [advanced-pose-arm](vega_1/rust/src/advanced_examples/move_to_pose/pose_arm.rs) |
| `advanced_examples/move_to_pose/pose_head` | [source](vega_1/python/advanced_examples/move_to_pose/pose_head.py) | [source](vega_1/cpp/advanced_examples/move_to_pose/pose_head.cpp) | [advanced-pose-head](vega_1/rust/src/advanced_examples/move_to_pose/pose_head.rs) |
| `advanced_examples/move_to_pose/pose_torso` | [source](vega_1/python/advanced_examples/move_to_pose/pose_torso.py) | [source](vega_1/cpp/advanced_examples/move_to_pose/pose_torso.cpp) | [advanced-pose-torso](vega_1/rust/src/advanced_examples/move_to_pose/pose_torso.rs) |
| `advanced_examples/reboot_control_board` | [source](vega_1/python/advanced_examples/reboot_control_board.py) | [source](vega_1/cpp/advanced_examples/reboot_control_board.cpp) | [advanced-reboot-control-board](vega_1/rust/src/advanced_examples/reboot_control_board.rs) |
| `advanced_examples/replay_trajectory` | [source](vega_1/python/advanced_examples/replay_trajectory.py) | [source](vega_1/cpp/advanced_examples/replay_trajectory.cpp) | [advanced-replay-trajectory](vega_1/rust/src/advanced_examples/replay_trajectory.rs) |
| `basic_examples/control/close_hand` | [source](vega_1/python/basic_examples/control/close_hand.py) | [source](vega_1/cpp/basic_examples/control/close_hand.cpp) | [control-close-hand](vega_1/rust/src/basic_examples/control/close_hand.rs) |
| `basic_examples/control/set_head_motor_mode` | [source](vega_1/python/basic_examples/control/set_head_motor_mode.py) | [source](vega_1/cpp/basic_examples/control/set_head_motor_mode.cpp) | [control-set-head-motor-mode](vega_1/rust/src/basic_examples/control/set_head_motor_mode.rs) |
| `basic_examples/control/cycle_arm` | [source](vega_1/python/basic_examples/control/cycle_arm.py) | [source](vega_1/cpp/basic_examples/control/cycle_arm.cpp) | [control-cycle-arm](vega_1/rust/src/basic_examples/control/cycle_arm.rs) |
| `basic_examples/control/exercise_chassis` | [source](vega_1/python/basic_examples/control/exercise_chassis.py) | [source](vega_1/cpp/basic_examples/control/exercise_chassis.cpp) | [control-exercise-chassis](vega_1/rust/src/basic_examples/control/exercise_chassis.rs) |
| `basic_examples/control/cycle_hand` | [source](vega_1/python/basic_examples/control/cycle_hand.py) | [source](vega_1/cpp/basic_examples/control/cycle_hand.cpp) | [control-cycle-hand](vega_1/rust/src/basic_examples/control/cycle_hand.rs) |
| `basic_examples/control/sweep_head_joints` | [source](vega_1/python/basic_examples/control/sweep_head_joints.py) | [source](vega_1/cpp/basic_examples/control/sweep_head_joints.cpp) | [control-sweep-head-joints](vega_1/rust/src/basic_examples/control/sweep_head_joints.rs) |
| `basic_examples/control/cycle_torso` | [source](vega_1/python/basic_examples/control/cycle_torso.py) | [source](vega_1/cpp/basic_examples/control/cycle_torso.cpp) | [control-cycle-torso](vega_1/rust/src/basic_examples/control/cycle_torso.rs) |
| `basic_examples/control/open_hand` | [source](vega_1/python/basic_examples/control/open_hand.py) | [source](vega_1/cpp/basic_examples/control/open_hand.cpp) | [control-open-hand](vega_1/rust/src/basic_examples/control/open_hand.rs) |
| `basic_examples/sensors/read_2d_lidar_scan` | [source](vega_1/python/basic_examples/sensors/read_2d_lidar_scan.py) | [source](vega_1/cpp/basic_examples/sensors/read_2d_lidar_scan.cpp) | [sensor-read-2d-lidar-scan](vega_1/rust/src/basic_examples/sensors/read_2d_lidar_scan.rs) |
| `basic_examples/sensors/monitor_3d_lidar_points` | [source](vega_1/python/basic_examples/sensors/monitor_3d_lidar_points.py) | [source](vega_1/cpp/basic_examples/sensors/monitor_3d_lidar_points.cpp) | [sensor-monitor-3d-lidar-points](vega_1/rust/src/basic_examples/sensors/monitor_3d_lidar_points.rs) |
| `basic_examples/sensors/monitor_lidar_imu` | [source](vega_1/python/basic_examples/sensors/monitor_lidar_imu.py) | [source](vega_1/cpp/basic_examples/sensors/monitor_lidar_imu.cpp) | [sensor-monitor-lidar-imu](vega_1/rust/src/basic_examples/sensors/monitor_lidar_imu.rs) |
| `basic_examples/sensors/read_arm_buttons` | [source](vega_1/python/basic_examples/sensors/read_arm_buttons.py) | [source](vega_1/cpp/basic_examples/sensors/read_arm_buttons.cpp) | [sensor-read-arm-buttons](vega_1/rust/src/basic_examples/sensors/read_arm_buttons.rs) |
| `basic_examples/sensors/read_battery_status` | [source](vega_1/python/basic_examples/sensors/read_battery_status.py) | [source](vega_1/cpp/basic_examples/sensors/read_battery_status.cpp) | [sensor-read-battery-status](vega_1/rust/src/basic_examples/sensors/read_battery_status.rs) |
| `basic_examples/sensors/monitor_chassis_cameras` | [source](vega_1/python/basic_examples/sensors/monitor_chassis_cameras.py) | [source](vega_1/cpp/basic_examples/sensors/monitor_chassis_cameras.cpp) | [sensor-monitor-chassis-cameras](vega_1/rust/src/basic_examples/sensors/monitor_chassis_cameras.rs) |
| `basic_examples/sensors/read_arm_force_torque` | [source](vega_1/python/basic_examples/sensors/read_arm_force_torque.py) | [source](vega_1/cpp/basic_examples/sensors/read_arm_force_torque.cpp) | [sensor-read-arm-force-torque](vega_1/rust/src/basic_examples/sensors/read_arm_force_torque.rs) |
| `basic_examples/sensors/read_hand_touch_forces` | [source](vega_1/python/basic_examples/sensors/read_hand_touch_forces.py) | [source](vega_1/cpp/basic_examples/sensors/read_hand_touch_forces.cpp) | [sensor-read-hand-touch-forces](vega_1/rust/src/basic_examples/sensors/read_hand_touch_forces.rs) |
| `basic_examples/sensors/monitor_head_camera` | [source](vega_1/python/basic_examples/sensors/monitor_head_camera.py) | [source](vega_1/cpp/basic_examples/sensors/monitor_head_camera.cpp) | [sensor-monitor-head-camera](vega_1/rust/src/basic_examples/sensors/monitor_head_camera.rs) |
| `basic_examples/sensors/read_imu_states` | [source](vega_1/python/basic_examples/sensors/read_imu_states.py) | [source](vega_1/cpp/basic_examples/sensors/read_imu_states.cpp) | [sensor-read-imu-states](vega_1/rust/src/basic_examples/sensors/read_imu_states.rs) |
| `basic_examples/sensors/monitor_2d_lidar_scans` | [source](vega_1/python/basic_examples/sensors/monitor_2d_lidar_scans.py) | [source](vega_1/cpp/basic_examples/sensors/monitor_2d_lidar_scans.cpp) | [sensor-monitor-2d-lidar-scans](vega_1/rust/src/basic_examples/sensors/monitor_2d_lidar_scans.rs) |
| `basic_examples/sensors/read_joint_states` | [source](vega_1/python/basic_examples/sensors/read_joint_states.py) | [source](vega_1/cpp/basic_examples/sensors/read_joint_states.cpp) | [sensor-read-joint-states](vega_1/rust/src/basic_examples/sensors/read_joint_states.rs) |
| `basic_examples/sensors/read_component_temperatures` | [source](vega_1/python/basic_examples/sensors/read_component_temperatures.py) | [source](vega_1/cpp/basic_examples/sensors/read_component_temperatures.cpp) | [sensor-read-component-temperatures](vega_1/rust/src/basic_examples/sensors/read_component_temperatures.rs) |
| `basic_examples/sensors/monitor_wrist_cameras` | [source](vega_1/python/basic_examples/sensors/monitor_wrist_cameras.py) | [source](vega_1/cpp/basic_examples/sensors/monitor_wrist_cameras.cpp) | [sensor-monitor-wrist-cameras](vega_1/rust/src/basic_examples/sensors/monitor_wrist_cameras.rs) |
| `benchmark/arm_tracking/benchmark_arm_sine` | [source](vega_1/python/benchmark/arm_tracking/benchmark_arm_sine.py) | [source](vega_1/cpp/benchmark/arm_tracking/benchmark_arm_sine.cpp) | [benchmark-arm-sine](vega_1/rust/src/benchmark/arm_tracking/benchmark_arm_sine.rs) |
| `benchmark/arm_tracking/benchmark_arm_step` | [source](vega_1/python/benchmark/arm_tracking/benchmark_arm_step.py) | [source](vega_1/cpp/benchmark/arm_tracking/benchmark_arm_step.cpp) | [benchmark-arm-step](vega_1/rust/src/benchmark/arm_tracking/benchmark_arm_step.rs) |
| `benchmark/connection/benchmark_connection_latency` | [source](vega_1/python/benchmark/connection/benchmark_connection_latency.py) | [source](vega_1/cpp/benchmark/connection/benchmark_connection_latency.cpp) | [benchmark-connection-latency](vega_1/rust/src/benchmark/connection/benchmark_connection_latency.rs) |
| `troubleshooting/measure_robot_clock_offset` | [source](vega_1/python/troubleshooting/measure_robot_clock_offset.py) | [source](vega_1/cpp/troubleshooting/measure_robot_clock_offset.cpp) | [troubleshooting-measure-robot-clock-offset](vega_1/rust/src/troubleshooting/measure_robot_clock_offset.rs) |
| `troubleshooting/clear_component_errors` | [source](vega_1/python/troubleshooting/clear_component_errors.py) | [source](vega_1/cpp/troubleshooting/clear_component_errors.cpp) | [troubleshooting-clear-component-errors](vega_1/rust/src/troubleshooting/clear_component_errors.rs) |
| `troubleshooting/display_robot_info` | [source](vega_1/python/troubleshooting/display_robot_info.py) | [source](vega_1/cpp/troubleshooting/display_robot_info.cpp) | [troubleshooting-display-robot-info](vega_1/rust/src/troubleshooting/display_robot_info.rs) |

`cycle_torso` supports `--relative` for the outward offset. Both modes use planned moves and return to the stored absolute start only after outward success. The default delta is 0.05 rad; select another value with `--delta`.

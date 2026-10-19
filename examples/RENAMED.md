<!--
Copyright (C) 2026 Dexmate Inc.

This software is dual-licensed:

1. GNU Affero General Public License v3.0 (AGPL-3.0)
   See LICENSE-AGPL for details

2. Commercial License
   For commercial licensing terms, contact: contact@dexmate.ai
-->

# Example filename migration

The language trees moved from `examples/{python,cpp,rust}/` to
`examples/vega_1/{python,cpp,rust}/`. Short `./run LANGUAGE NAME` commands still work.

Names were changed to describe executed actions. Old entrypoints are removed; update commands/imports to the paths below. Runtime behavior is detailed in [BEHAVIOR.md](BEHAVIOR.md).

| Old path | Current path |
| --- | --- |
| `examples/cpp/advanced_examples/chassis_s_curve.cpp` | [examples/vega_1/cpp/advanced_examples/steer_chassis_sine.cpp](vega_1/cpp/advanced_examples/steer_chassis_sine.cpp) |
| `examples/cpp/advanced_examples/config_arm_pid.cpp` | [examples/vega_1/cpp/advanced_examples/configure_arm_pid.cpp](vega_1/cpp/advanced_examples/configure_arm_pid.cpp) |
| `examples/cpp/advanced_examples/config_ee_baud_rate.cpp` | [examples/vega_1/cpp/advanced_examples/configure_ee_baud_rate.cpp](vega_1/cpp/advanced_examples/configure_ee_baud_rate.cpp) |
| `examples/cpp/advanced_examples/config_force_torque_sensor.cpp` | [examples/vega_1/cpp/advanced_examples/configure_arm_ft_sensor.cpp](vega_1/cpp/advanced_examples/configure_arm_ft_sensor.cpp) |
| `examples/cpp/advanced_examples/control_customized_ee.cpp` | [examples/vega_1/cpp/advanced_examples/send_end_effector_bytes.cpp](vega_1/cpp/advanced_examples/send_end_effector_bytes.cpp) |
| `examples/cpp/advanced_examples/data/export_trajectory.py` | removed; every language replays the one CSV recording in [examples/vega_1/data/](vega_1/data/) |
| `examples/vega_1/python/advanced_examples/vega_dance.sh` | [examples/vega_1/vega_dance.sh](vega_1/vega_dance.sh), now `vega_dance.sh [python|cpp|rust]` |
| `examples/cpp/advanced_examples/disable_arm_motors.cpp` | [examples/vega_1/cpp/advanced_examples/manage_arm_brakes.cpp](vega_1/cpp/advanced_examples/manage_arm_brakes.cpp) |
| `examples/cpp/advanced_examples/display_component_current.cpp` | [examples/vega_1/cpp/advanced_examples/monitor_joint_currents.cpp](vega_1/cpp/advanced_examples/monitor_joint_currents.cpp) |
| `examples/cpp/advanced_examples/fold_arms.cpp` | [examples/vega_1/cpp/advanced_examples/pose_both_arms.cpp](vega_1/cpp/advanced_examples/pose_both_arms.cpp) |
| `examples/cpp/advanced_examples/fold_robot.cpp` | [examples/vega_1/cpp/advanced_examples/fold_or_unfold_robot.cpp](vega_1/cpp/advanced_examples/fold_or_unfold_robot.cpp) |
| `examples/cpp/advanced_examples/init_arm_unsafe.cpp` | [examples/vega_1/cpp/advanced_examples/pose_arms_sequentially.cpp](vega_1/cpp/advanced_examples/pose_arms_sequentially.cpp) |
| `examples/cpp/advanced_examples/joint_admittance_control.cpp` | removed; joint admittance is not part of the example set (Python has [run_cartesian_admittance.py](vega_1/python/advanced_examples/run_cartesian_admittance.py)) |
| `examples/cpp/advanced_examples/keyboard_joint_control.cpp` | [examples/vega_1/cpp/advanced_examples/keyboard_jog_joint.cpp](vega_1/cpp/advanced_examples/keyboard_jog_joint.cpp) |
| `examples/cpp/advanced_examples/move_to_pose/move_arm_to_pose.cpp` | [examples/vega_1/cpp/advanced_examples/move_to_pose/pose_arm.cpp](vega_1/cpp/advanced_examples/move_to_pose/pose_arm.cpp) |
| `examples/cpp/advanced_examples/move_to_pose/move_head_to_pose.cpp` | [examples/vega_1/cpp/advanced_examples/move_to_pose/pose_head.cpp](vega_1/cpp/advanced_examples/move_to_pose/pose_head.cpp) |
| `examples/cpp/advanced_examples/move_to_pose/move_torso_to_pose.cpp` | [examples/vega_1/cpp/advanced_examples/move_to_pose/pose_torso.cpp](vega_1/cpp/advanced_examples/move_to_pose/pose_torso.cpp) |
| `examples/cpp/advanced_examples/reboot_robot.cpp` | [examples/vega_1/cpp/advanced_examples/reboot_control_board.cpp](vega_1/cpp/advanced_examples/reboot_control_board.cpp) |
| `examples/cpp/advanced_examples/replay_trajectory.cpp` | [examples/vega_1/cpp/advanced_examples/replay_trajectory.cpp](vega_1/cpp/advanced_examples/replay_trajectory.cpp) |
| `examples/cpp/basic_examples/control/disable_head.cpp` | [examples/vega_1/cpp/basic_examples/control/set_head_motor_mode.cpp](vega_1/cpp/basic_examples/control/set_head_motor_mode.cpp) |
| `examples/cpp/basic_examples/control/move_arm.cpp` | [examples/vega_1/cpp/basic_examples/control/cycle_arm.cpp](vega_1/cpp/basic_examples/control/cycle_arm.cpp) |
| `examples/cpp/basic_examples/control/move_arm_relative.cpp` | [examples/vega_1/cpp/basic_examples/control/cycle_arm.cpp](vega_1/cpp/basic_examples/control/cycle_arm.cpp) |
| `examples/cpp/basic_examples/control/move_arm_target.cpp` | [examples/vega_1/cpp/basic_examples/control/cycle_arm.cpp](vega_1/cpp/basic_examples/control/cycle_arm.cpp) |
| `examples/cpp/basic_examples/control/move_chassis.cpp` | [examples/vega_1/cpp/basic_examples/control/exercise_chassis.cpp](vega_1/cpp/basic_examples/control/exercise_chassis.cpp) |
| `examples/cpp/basic_examples/control/move_hand.cpp` | [examples/vega_1/cpp/basic_examples/control/cycle_hand.cpp](vega_1/cpp/basic_examples/control/cycle_hand.cpp) |
| `examples/cpp/basic_examples/control/move_head.cpp` | [examples/vega_1/cpp/basic_examples/control/sweep_head_joints.cpp](vega_1/cpp/basic_examples/control/sweep_head_joints.cpp) |
| `examples/cpp/basic_examples/control/move_torso.cpp` | [examples/vega_1/cpp/basic_examples/control/cycle_torso.cpp](vega_1/cpp/basic_examples/control/cycle_torso.cpp) |
| `examples/cpp/basic_examples/control/move_torso_relative.cpp` | [examples/vega_1/cpp/basic_examples/control/cycle_torso_relative.cpp](vega_1/cpp/basic_examples/control/cycle_torso.cpp) |
| `examples/cpp/basic_examples/sensors/get_2d_lidar_data.cpp` | [examples/vega_1/cpp/basic_examples/sensors/read_2d_lidar_scan.cpp](vega_1/cpp/basic_examples/sensors/read_2d_lidar_scan.cpp) |
| `examples/cpp/basic_examples/sensors/get_3d_lidar_data.cpp` | [examples/vega_1/cpp/basic_examples/sensors/monitor_3d_lidar_points.cpp](vega_1/cpp/basic_examples/sensors/monitor_3d_lidar_points.cpp) |
| `examples/cpp/basic_examples/sensors/get_3d_lidar_imu_data.cpp` | [examples/vega_1/cpp/basic_examples/sensors/monitor_lidar_imu.cpp](vega_1/cpp/basic_examples/sensors/monitor_lidar_imu.cpp) |
| `examples/cpp/basic_examples/sensors/get_arm_button_state.cpp` | [examples/vega_1/cpp/basic_examples/sensors/read_arm_buttons.cpp](vega_1/cpp/basic_examples/sensors/read_arm_buttons.cpp) |
| `examples/cpp/basic_examples/sensors/get_battery_power.cpp` | [examples/vega_1/cpp/basic_examples/sensors/read_battery_status.cpp](vega_1/cpp/basic_examples/sensors/read_battery_status.cpp) |
| `examples/cpp/basic_examples/sensors/get_chassis_cam_data.cpp` | [examples/vega_1/cpp/basic_examples/sensors/monitor_chassis_cameras.cpp](vega_1/cpp/basic_examples/sensors/monitor_chassis_cameras.cpp) |
| `examples/cpp/basic_examples/sensors/get_force_torque_sensor.cpp` | [examples/vega_1/cpp/basic_examples/sensors/read_arm_force_torque.cpp](vega_1/cpp/basic_examples/sensors/read_arm_force_torque.cpp) |
| `examples/cpp/basic_examples/sensors/get_hand_touch_data.cpp` | [examples/vega_1/cpp/basic_examples/sensors/read_hand_touch_forces.cpp](vega_1/cpp/basic_examples/sensors/read_hand_touch_forces.cpp) |
| `examples/cpp/basic_examples/sensors/get_head_zed_x_mini_data.cpp` | [examples/vega_1/cpp/basic_examples/sensors/monitor_head_camera.cpp](vega_1/cpp/basic_examples/sensors/monitor_head_camera.cpp) |
| `examples/cpp/basic_examples/sensors/get_imu_data.cpp` | [examples/vega_1/cpp/basic_examples/sensors/read_imu_states.cpp](vega_1/cpp/basic_examples/sensors/read_imu_states.cpp) |
| `examples/cpp/basic_examples/sensors/get_live_2d_lidar_data.cpp` | [examples/vega_1/cpp/basic_examples/sensors/monitor_2d_lidar_scans.cpp](vega_1/cpp/basic_examples/sensors/monitor_2d_lidar_scans.cpp) |
| `examples/cpp/basic_examples/sensors/get_robot_state.cpp` | [examples/vega_1/cpp/basic_examples/sensors/read_joint_states.cpp](vega_1/cpp/basic_examples/sensors/read_joint_states.cpp) |
| `examples/cpp/basic_examples/sensors/get_temperature_data.cpp` | [examples/vega_1/cpp/basic_examples/sensors/read_component_temperatures.cpp](vega_1/cpp/basic_examples/sensors/read_component_temperatures.cpp) |
| `examples/cpp/basic_examples/sensors/get_wrist_zed_x_one_data.cpp` | [examples/vega_1/cpp/basic_examples/sensors/monitor_wrist_cameras.cpp](vega_1/cpp/basic_examples/sensors/monitor_wrist_cameras.cpp) |
| `examples/cpp/benchmark/arm_tracking/sin_benchmark.cpp` | [examples/vega_1/cpp/benchmark/arm_tracking/benchmark_arm_sine.cpp](vega_1/cpp/benchmark/arm_tracking/benchmark_arm_sine.cpp) |
| `examples/cpp/benchmark/arm_tracking/step_benchmark.cpp` | [examples/vega_1/cpp/benchmark/arm_tracking/benchmark_arm_step.cpp](vega_1/cpp/benchmark/arm_tracking/benchmark_arm_step.cpp) |
| `examples/cpp/benchmark/connection/connect_robot_latency.cpp` | [examples/vega_1/cpp/benchmark/connection/benchmark_connection_latency.cpp](vega_1/cpp/benchmark/connection/benchmark_connection_latency.cpp) |
| `examples/cpp/troubleshooting/check_time_difference.cpp` | [examples/vega_1/cpp/troubleshooting/measure_robot_clock_offset.cpp](vega_1/cpp/troubleshooting/measure_robot_clock_offset.cpp) |
| `examples/cpp/troubleshooting/clear_error.cpp` | [examples/vega_1/cpp/troubleshooting/clear_component_errors.cpp](vega_1/cpp/troubleshooting/clear_component_errors.cpp) |
| `examples/python/advanced_examples/admittance_control.py` | [examples/vega_1/python/advanced_examples/run_cartesian_admittance.py](vega_1/python/advanced_examples/run_cartesian_admittance.py) |
| `examples/python/advanced_examples/chassis_s_curve.py` | [examples/vega_1/python/advanced_examples/steer_chassis_sine.py](vega_1/python/advanced_examples/steer_chassis_sine.py) |
| `examples/python/advanced_examples/config_arm_pid.py` | [examples/vega_1/python/advanced_examples/configure_arm_pid.py](vega_1/python/advanced_examples/configure_arm_pid.py) |
| `examples/python/advanced_examples/config_ee_baud_rate.py` | [examples/vega_1/python/advanced_examples/configure_ee_baud_rate.py](vega_1/python/advanced_examples/configure_ee_baud_rate.py) |
| `examples/python/advanced_examples/config_force_torque_sensor.py` | [examples/vega_1/python/advanced_examples/configure_arm_ft_sensor.py](vega_1/python/advanced_examples/configure_arm_ft_sensor.py) |
| `examples/python/advanced_examples/control_customized_ee.py` | [examples/vega_1/python/advanced_examples/send_end_effector_bytes.py](vega_1/python/advanced_examples/send_end_effector_bytes.py) |
| `examples/python/advanced_examples/disable_arm_motors.py` | [examples/vega_1/python/advanced_examples/manage_arm_brakes.py](vega_1/python/advanced_examples/manage_arm_brakes.py) |
| `examples/python/advanced_examples/display_component_current.py` | [examples/vega_1/python/advanced_examples/monitor_joint_currents.py](vega_1/python/advanced_examples/monitor_joint_currents.py) |
| `examples/python/advanced_examples/fold_arms.py` | [examples/vega_1/python/advanced_examples/pose_both_arms.py](vega_1/python/advanced_examples/pose_both_arms.py) |
| `examples/python/advanced_examples/fold_robot.py` | [examples/vega_1/python/advanced_examples/fold_or_unfold_robot.py](vega_1/python/advanced_examples/fold_or_unfold_robot.py) |
| `examples/python/advanced_examples/init_arm_safe.py` | [examples/vega_1/python/advanced_examples/plan_arm_pose.py](vega_1/python/advanced_examples/plan_arm_pose.py) |
| `examples/python/advanced_examples/init_arm_unsafe.py` | [examples/vega_1/python/advanced_examples/pose_arms_sequentially.py](vega_1/python/advanced_examples/pose_arms_sequentially.py) |
| `examples/python/advanced_examples/joint_admittance_control.py` | removed; use [run_cartesian_admittance.py](vega_1/python/advanced_examples/run_cartesian_admittance.py) |
| `examples/python/advanced_examples/keyboard_joint_control.py` | [examples/vega_1/python/advanced_examples/keyboard_jog_joint.py](vega_1/python/advanced_examples/keyboard_jog_joint.py) |
| `examples/python/advanced_examples/move_arm_ee_simple.py` | [examples/vega_1/python/advanced_examples/offset_end_effectors.py](vega_1/python/advanced_examples/offset_end_effectors.py) |
| `examples/python/advanced_examples/move_to_pose/move_arm_to_pose.py` | [examples/vega_1/python/advanced_examples/move_to_pose/pose_arm.py](vega_1/python/advanced_examples/move_to_pose/pose_arm.py) |
| `examples/python/advanced_examples/move_to_pose/move_head_to_pose.py` | [examples/vega_1/python/advanced_examples/move_to_pose/pose_head.py](vega_1/python/advanced_examples/move_to_pose/pose_head.py) |
| `examples/python/advanced_examples/move_to_pose/move_torso_to_pose.py` | [examples/vega_1/python/advanced_examples/move_to_pose/pose_torso.py](vega_1/python/advanced_examples/move_to_pose/pose_torso.py) |
| `examples/python/advanced_examples/reboot_robot.py` | [examples/vega_1/python/advanced_examples/reboot_control_board.py](vega_1/python/advanced_examples/reboot_control_board.py) |
| `examples/python/advanced_examples/replay_trajectory.py` | [examples/vega_1/python/advanced_examples/replay_trajectory.py](vega_1/python/advanced_examples/replay_trajectory.py) |
| `examples/python/basic_examples/control/disable_head.py` | [examples/vega_1/python/basic_examples/control/set_head_motor_mode.py](vega_1/python/basic_examples/control/set_head_motor_mode.py) |
| `examples/python/basic_examples/control/move_arm.py` | [examples/vega_1/python/basic_examples/control/cycle_arm.py](vega_1/python/basic_examples/control/cycle_arm.py) |
| `examples/python/basic_examples/control/move_arm_relative.py` | [examples/vega_1/python/basic_examples/control/cycle_arm.py](vega_1/python/basic_examples/control/cycle_arm.py) |
| `examples/python/basic_examples/control/move_arm_target.py` | [examples/vega_1/python/basic_examples/control/cycle_arm.py](vega_1/python/basic_examples/control/cycle_arm.py) |
| `examples/python/basic_examples/control/move_chassis.py` | [examples/vega_1/python/basic_examples/control/exercise_chassis.py](vega_1/python/basic_examples/control/exercise_chassis.py) |
| `examples/python/basic_examples/control/move_hand.py` | [examples/vega_1/python/basic_examples/control/cycle_hand.py](vega_1/python/basic_examples/control/cycle_hand.py) |
| `examples/python/basic_examples/control/move_head.py` | [examples/vega_1/python/basic_examples/control/sweep_head_joints.py](vega_1/python/basic_examples/control/sweep_head_joints.py) |
| `examples/python/basic_examples/control/move_torso.py` | [examples/vega_1/python/basic_examples/control/cycle_torso.py](vega_1/python/basic_examples/control/cycle_torso.py) |
| `examples/python/basic_examples/control/move_torso_relative.py` | [examples/vega_1/python/basic_examples/control/cycle_torso_relative.py](vega_1/python/basic_examples/control/cycle_torso.py) |
| `examples/python/basic_examples/sensors/get_2d_lidar_data.py` | [examples/vega_1/python/basic_examples/sensors/read_2d_lidar_scan.py](vega_1/python/basic_examples/sensors/read_2d_lidar_scan.py) |
| `examples/python/basic_examples/sensors/get_3d_lidar_data.py` | [examples/vega_1/python/basic_examples/sensors/monitor_3d_lidar_points.py](vega_1/python/basic_examples/sensors/monitor_3d_lidar_points.py) |
| `examples/python/basic_examples/sensors/get_3d_lidar_imu_data.py` | [examples/vega_1/python/basic_examples/sensors/monitor_lidar_imu.py](vega_1/python/basic_examples/sensors/monitor_lidar_imu.py) |
| `examples/python/basic_examples/sensors/get_arm_button_state.py` | [examples/vega_1/python/basic_examples/sensors/read_arm_buttons.py](vega_1/python/basic_examples/sensors/read_arm_buttons.py) |
| `examples/python/basic_examples/sensors/get_battery_power.py` | [examples/vega_1/python/basic_examples/sensors/read_battery_status.py](vega_1/python/basic_examples/sensors/read_battery_status.py) |
| `examples/python/basic_examples/sensors/get_chassis_cam_data.py` | [examples/vega_1/python/basic_examples/sensors/monitor_chassis_cameras.py](vega_1/python/basic_examples/sensors/monitor_chassis_cameras.py) |
| `examples/python/basic_examples/sensors/get_force_torque_sensor.py` | [examples/vega_1/python/basic_examples/sensors/read_arm_force_torque.py](vega_1/python/basic_examples/sensors/read_arm_force_torque.py) |
| `examples/python/basic_examples/sensors/get_hand_touch_data.py` | [examples/vega_1/python/basic_examples/sensors/read_hand_touch_forces.py](vega_1/python/basic_examples/sensors/read_hand_touch_forces.py) |
| `examples/python/basic_examples/sensors/get_head_zed_x_mini_data.py` | [examples/vega_1/python/basic_examples/sensors/monitor_head_camera.py](vega_1/python/basic_examples/sensors/monitor_head_camera.py) |
| `examples/python/basic_examples/sensors/get_imu_data.py` | [examples/vega_1/python/basic_examples/sensors/read_imu_states.py](vega_1/python/basic_examples/sensors/read_imu_states.py) |
| `examples/python/basic_examples/sensors/get_live_2d_lidar_data.py` | [examples/vega_1/python/basic_examples/sensors/monitor_2d_lidar_scans.py](vega_1/python/basic_examples/sensors/monitor_2d_lidar_scans.py) |
| `examples/python/basic_examples/sensors/get_robot_state.py` | [examples/vega_1/python/basic_examples/sensors/read_joint_states.py](vega_1/python/basic_examples/sensors/read_joint_states.py) |
| `examples/python/basic_examples/sensors/get_temperature_data.py` | [examples/vega_1/python/basic_examples/sensors/read_component_temperatures.py](vega_1/python/basic_examples/sensors/read_component_temperatures.py) |
| `examples/python/basic_examples/sensors/get_wrist_zed_x_one_data.py` | [examples/vega_1/python/basic_examples/sensors/monitor_wrist_cameras.py](vega_1/python/basic_examples/sensors/monitor_wrist_cameras.py) |
| `examples/python/benchmark/arm_tracking/common.py` | [examples/vega_1/python/benchmark/arm_tracking/_tracking_helpers.py](vega_1/python/benchmark/arm_tracking/_tracking_helpers.py) |
| `examples/python/benchmark/arm_tracking/sin_benchmark.py` | [examples/vega_1/python/benchmark/arm_tracking/benchmark_arm_sine.py](vega_1/python/benchmark/arm_tracking/benchmark_arm_sine.py) |
| `examples/python/benchmark/arm_tracking/step_benchmark.py` | [examples/vega_1/python/benchmark/arm_tracking/benchmark_arm_step.py](vega_1/python/benchmark/arm_tracking/benchmark_arm_step.py) |
| `examples/python/benchmark/connection/connect_robot_latency.py` | [examples/vega_1/python/benchmark/connection/benchmark_connection_latency.py](vega_1/python/benchmark/connection/benchmark_connection_latency.py) |
| `examples/python/benchmark/connection/heartbeat_gil_stress.py` | [examples/vega_1/python/benchmark/connection/stress_heartbeat_gil.py](vega_1/python/benchmark/connection/stress_heartbeat_gil.py) |
| `examples/python/teleop/arm_cartesian_teleop.py` | [examples/vega_1/python/teleop/gamepad_arm_cartesian_control.py](vega_1/python/teleop/gamepad_arm_cartesian_control.py) |
| `examples/python/teleop/arm_joints_teleop.py` | [examples/vega_1/python/teleop/gamepad_arm_joint_control.py](vega_1/python/teleop/gamepad_arm_joint_control.py) |
| `examples/python/teleop/base_arm_teleop.py` | [examples/vega_1/python/teleop/_arm_teleop_base.py](vega_1/python/teleop/_arm_teleop_base.py) |
| `examples/python/teleop/chassis_head_teleop.py` | [examples/vega_1/python/teleop/gamepad_chassis_head.py](vega_1/python/teleop/gamepad_chassis_head.py) |
| `examples/python/teleop/dualsense_teleop_base.py` | [examples/vega_1/python/teleop/_dualsense_teleop_base.py](vega_1/python/teleop/_dualsense_teleop_base.py) |
| `examples/python/troubleshooting/check_time_difference.py` | [examples/vega_1/python/troubleshooting/measure_robot_clock_offset.py](vega_1/python/troubleshooting/measure_robot_clock_offset.py) |
| `examples/python/troubleshooting/clear_error.py` | [examples/vega_1/python/troubleshooting/clear_component_errors.py](vega_1/python/troubleshooting/clear_component_errors.py) |
| `examples/rust/src/advanced_examples/chassis_s_curve.rs` | [examples/vega_1/rust/src/advanced_examples/steer_chassis_sine.rs](vega_1/rust/src/advanced_examples/steer_chassis_sine.rs) |
| `examples/rust/src/advanced_examples/config_arm_pid.rs` | [examples/vega_1/rust/src/advanced_examples/configure_arm_pid.rs](vega_1/rust/src/advanced_examples/configure_arm_pid.rs) |
| `examples/rust/src/advanced_examples/config_ee_baud_rate.rs` | [examples/vega_1/rust/src/advanced_examples/configure_ee_baud_rate.rs](vega_1/rust/src/advanced_examples/configure_ee_baud_rate.rs) |
| `examples/rust/src/advanced_examples/config_force_torque_sensor.rs` | [examples/vega_1/rust/src/advanced_examples/configure_arm_ft_sensor.rs](vega_1/rust/src/advanced_examples/configure_arm_ft_sensor.rs) |
| `examples/rust/src/advanced_examples/control_customized_ee.rs` | [examples/vega_1/rust/src/advanced_examples/send_end_effector_bytes.rs](vega_1/rust/src/advanced_examples/send_end_effector_bytes.rs) |
| `examples/rust/src/advanced_examples/disable_arm_motors.rs` | [examples/vega_1/rust/src/advanced_examples/manage_arm_brakes.rs](vega_1/rust/src/advanced_examples/manage_arm_brakes.rs) |
| `examples/rust/src/advanced_examples/display_component_current.rs` | [examples/vega_1/rust/src/advanced_examples/monitor_joint_currents.rs](vega_1/rust/src/advanced_examples/monitor_joint_currents.rs) |
| `examples/rust/src/advanced_examples/fold_arms.rs` | [examples/vega_1/rust/src/advanced_examples/pose_both_arms.rs](vega_1/rust/src/advanced_examples/pose_both_arms.rs) |
| `examples/rust/src/advanced_examples/fold_robot.rs` | [examples/vega_1/rust/src/advanced_examples/fold_or_unfold_robot.rs](vega_1/rust/src/advanced_examples/fold_or_unfold_robot.rs) |
| `examples/rust/src/advanced_examples/init_arm_unsafe.rs` | [examples/vega_1/rust/src/advanced_examples/pose_arms_sequentially.rs](vega_1/rust/src/advanced_examples/pose_arms_sequentially.rs) |
| `examples/rust/src/advanced_examples/joint_admittance_control.rs` | removed; joint admittance is not part of the example set (Python has [run_cartesian_admittance.py](vega_1/python/advanced_examples/run_cartesian_admittance.py)) |
| `examples/rust/src/advanced_examples/keyboard_joint_control.rs` | [examples/vega_1/rust/src/advanced_examples/keyboard_jog_joint.rs](vega_1/rust/src/advanced_examples/keyboard_jog_joint.rs) |
| `examples/rust/src/advanced_examples/move_to_pose/move_arm_to_pose.rs` | [examples/vega_1/rust/src/advanced_examples/move_to_pose/pose_arm.rs](vega_1/rust/src/advanced_examples/move_to_pose/pose_arm.rs) |
| `examples/rust/src/advanced_examples/move_to_pose/move_head_to_pose.rs` | [examples/vega_1/rust/src/advanced_examples/move_to_pose/pose_head.rs](vega_1/rust/src/advanced_examples/move_to_pose/pose_head.rs) |
| `examples/rust/src/advanced_examples/move_to_pose/move_torso_to_pose.rs` | [examples/vega_1/rust/src/advanced_examples/move_to_pose/pose_torso.rs](vega_1/rust/src/advanced_examples/move_to_pose/pose_torso.rs) |
| `examples/rust/src/advanced_examples/reboot_robot.rs` | [examples/vega_1/rust/src/advanced_examples/reboot_control_board.rs](vega_1/rust/src/advanced_examples/reboot_control_board.rs) |
| `examples/rust/src/advanced_examples/replay_trajectory.rs` | [examples/vega_1/rust/src/advanced_examples/replay_trajectory.rs](vega_1/rust/src/advanced_examples/replay_trajectory.rs) |
| `examples/rust/src/basic_examples/control/disable_head.rs` | [examples/vega_1/rust/src/basic_examples/control/set_head_motor_mode.rs](vega_1/rust/src/basic_examples/control/set_head_motor_mode.rs) |
| `examples/rust/src/basic_examples/control/move_arm.rs` | [examples/vega_1/rust/src/basic_examples/control/cycle_arm.rs](vega_1/rust/src/basic_examples/control/cycle_arm.rs) |
| `examples/rust/src/basic_examples/control/move_arm_relative.rs` | [examples/vega_1/rust/src/basic_examples/control/cycle_arm.rs](vega_1/rust/src/basic_examples/control/cycle_arm.rs) |
| `examples/rust/src/basic_examples/control/move_arm_target.rs` | [examples/vega_1/rust/src/basic_examples/control/cycle_arm.rs](vega_1/rust/src/basic_examples/control/cycle_arm.rs) |
| `examples/rust/src/basic_examples/control/move_chassis.rs` | [examples/vega_1/rust/src/basic_examples/control/exercise_chassis.rs](vega_1/rust/src/basic_examples/control/exercise_chassis.rs) |
| `examples/rust/src/basic_examples/control/move_hand.rs` | [examples/vega_1/rust/src/basic_examples/control/cycle_hand.rs](vega_1/rust/src/basic_examples/control/cycle_hand.rs) |
| `examples/rust/src/basic_examples/control/move_head.rs` | [examples/vega_1/rust/src/basic_examples/control/sweep_head_joints.rs](vega_1/rust/src/basic_examples/control/sweep_head_joints.rs) |
| `examples/rust/src/basic_examples/control/move_torso.rs` | [examples/vega_1/rust/src/basic_examples/control/cycle_torso.rs](vega_1/rust/src/basic_examples/control/cycle_torso.rs) |
| `examples/rust/src/basic_examples/control/move_torso_relative.rs` | [examples/vega_1/rust/src/basic_examples/control/cycle_torso_relative.rs](vega_1/rust/src/basic_examples/control/cycle_torso.rs) |
| `examples/rust/src/basic_examples/sensors/get_2d_lidar_data.rs` | [examples/vega_1/rust/src/basic_examples/sensors/read_2d_lidar_scan.rs](vega_1/rust/src/basic_examples/sensors/read_2d_lidar_scan.rs) |
| `examples/rust/src/basic_examples/sensors/get_3d_lidar_data.rs` | [examples/vega_1/rust/src/basic_examples/sensors/monitor_3d_lidar_points.rs](vega_1/rust/src/basic_examples/sensors/monitor_3d_lidar_points.rs) |
| `examples/rust/src/basic_examples/sensors/get_3d_lidar_imu_data.rs` | [examples/vega_1/rust/src/basic_examples/sensors/monitor_lidar_imu.rs](vega_1/rust/src/basic_examples/sensors/monitor_lidar_imu.rs) |
| `examples/rust/src/basic_examples/sensors/get_arm_button_state.rs` | [examples/vega_1/rust/src/basic_examples/sensors/read_arm_buttons.rs](vega_1/rust/src/basic_examples/sensors/read_arm_buttons.rs) |
| `examples/rust/src/basic_examples/sensors/get_battery_power.rs` | [examples/vega_1/rust/src/basic_examples/sensors/read_battery_status.rs](vega_1/rust/src/basic_examples/sensors/read_battery_status.rs) |
| `examples/rust/src/basic_examples/sensors/get_chassis_cam_data.rs` | [examples/vega_1/rust/src/basic_examples/sensors/monitor_chassis_cameras.rs](vega_1/rust/src/basic_examples/sensors/monitor_chassis_cameras.rs) |
| `examples/rust/src/basic_examples/sensors/get_force_torque_sensor.rs` | [examples/vega_1/rust/src/basic_examples/sensors/read_arm_force_torque.rs](vega_1/rust/src/basic_examples/sensors/read_arm_force_torque.rs) |
| `examples/rust/src/basic_examples/sensors/get_hand_touch_data.rs` | [examples/vega_1/rust/src/basic_examples/sensors/read_hand_touch_forces.rs](vega_1/rust/src/basic_examples/sensors/read_hand_touch_forces.rs) |
| `examples/rust/src/basic_examples/sensors/get_head_zed_x_mini_data.rs` | [examples/vega_1/rust/src/basic_examples/sensors/monitor_head_camera.rs](vega_1/rust/src/basic_examples/sensors/monitor_head_camera.rs) |
| `examples/rust/src/basic_examples/sensors/get_imu_data.rs` | [examples/vega_1/rust/src/basic_examples/sensors/read_imu_states.rs](vega_1/rust/src/basic_examples/sensors/read_imu_states.rs) |
| `examples/rust/src/basic_examples/sensors/get_live_2d_lidar_data.rs` | [examples/vega_1/rust/src/basic_examples/sensors/monitor_2d_lidar_scans.rs](vega_1/rust/src/basic_examples/sensors/monitor_2d_lidar_scans.rs) |
| `examples/rust/src/basic_examples/sensors/get_robot_state.rs` | [examples/vega_1/rust/src/basic_examples/sensors/read_joint_states.rs](vega_1/rust/src/basic_examples/sensors/read_joint_states.rs) |
| `examples/rust/src/basic_examples/sensors/get_temperature_data.rs` | [examples/vega_1/rust/src/basic_examples/sensors/read_component_temperatures.rs](vega_1/rust/src/basic_examples/sensors/read_component_temperatures.rs) |
| `examples/rust/src/basic_examples/sensors/get_wrist_zed_x_one_data.rs` | [examples/vega_1/rust/src/basic_examples/sensors/monitor_wrist_cameras.rs](vega_1/rust/src/basic_examples/sensors/monitor_wrist_cameras.rs) |
| `examples/rust/src/benchmark/arm_tracking/sin_benchmark.rs` | [examples/vega_1/rust/src/benchmark/arm_tracking/benchmark_arm_sine.rs](vega_1/rust/src/benchmark/arm_tracking/benchmark_arm_sine.rs) |
| `examples/rust/src/benchmark/arm_tracking/step_benchmark.rs` | [examples/vega_1/rust/src/benchmark/arm_tracking/benchmark_arm_step.rs](vega_1/rust/src/benchmark/arm_tracking/benchmark_arm_step.rs) |
| `examples/rust/src/benchmark/connection/connect_robot_latency.rs` | [examples/vega_1/rust/src/benchmark/connection/benchmark_connection_latency.rs](vega_1/rust/src/benchmark/connection/benchmark_connection_latency.rs) |
| `examples/rust/src/troubleshooting/check_time_difference.rs` | [examples/vega_1/rust/src/troubleshooting/measure_robot_clock_offset.rs](vega_1/rust/src/troubleshooting/measure_robot_clock_offset.rs) |
| `examples/rust/src/troubleshooting/clear_error.rs` | [examples/vega_1/rust/src/troubleshooting/clear_component_errors.rs](vega_1/rust/src/troubleshooting/clear_component_errors.rs) |

## Shortened descriptive names

| Previous path | Current path |
| --- | --- |
| `examples/cpp/advanced_examples/configure_arm_force_torque_sensor.cpp` | [examples/vega_1/cpp/advanced_examples/configure_arm_ft_sensor.cpp](vega_1/cpp/advanced_examples/configure_arm_ft_sensor.cpp) |
| `examples/cpp/advanced_examples/configure_end_effector_baud_rate.cpp` | [examples/vega_1/cpp/advanced_examples/configure_ee_baud_rate.cpp](vega_1/cpp/advanced_examples/configure_ee_baud_rate.cpp) |
| `examples/cpp/advanced_examples/drive_chassis_with_sinusoidal_steering.cpp` | [examples/vega_1/cpp/advanced_examples/steer_chassis_sine.cpp](vega_1/cpp/advanced_examples/steer_chassis_sine.cpp) |
| `examples/cpp/advanced_examples/move_arms_to_named_pose_sequentially.cpp` | [examples/vega_1/cpp/advanced_examples/pose_arms_sequentially.cpp](vega_1/cpp/advanced_examples/pose_arms_sequentially.cpp) |
| `examples/cpp/advanced_examples/move_both_arms_to_named_pose.cpp` | [examples/vega_1/cpp/advanced_examples/pose_both_arms.cpp](vega_1/cpp/advanced_examples/pose_both_arms.cpp) |
| `examples/cpp/advanced_examples/move_to_pose/move_arm_to_named_pose.cpp` | [examples/vega_1/cpp/advanced_examples/move_to_pose/pose_arm.cpp](vega_1/cpp/advanced_examples/move_to_pose/pose_arm.cpp) |
| `examples/cpp/advanced_examples/move_to_pose/move_head_to_named_pose.cpp` | [examples/vega_1/cpp/advanced_examples/move_to_pose/pose_head.cpp](vega_1/cpp/advanced_examples/move_to_pose/pose_head.cpp) |
| `examples/cpp/advanced_examples/move_to_pose/move_torso_to_named_pose.cpp` | [examples/vega_1/cpp/advanced_examples/move_to_pose/pose_torso.cpp](vega_1/cpp/advanced_examples/move_to_pose/pose_torso.cpp) |
| `examples/cpp/advanced_examples/prepare_robot_and_replay_trajectory.cpp` | [examples/vega_1/cpp/advanced_examples/replay_trajectory.cpp](vega_1/cpp/advanced_examples/replay_trajectory.cpp) |
| `examples/cpp/basic_examples/control/close_then_open_hand.cpp` | [examples/vega_1/cpp/basic_examples/control/cycle_hand.cpp](vega_1/cpp/basic_examples/control/cycle_hand.cpp) |
| `examples/cpp/basic_examples/control/drive_chassis_in_six_directions.cpp` | [examples/vega_1/cpp/basic_examples/control/exercise_chassis.cpp](vega_1/cpp/basic_examples/control/exercise_chassis.cpp) |
| `examples/cpp/basic_examples/control/jog_each_arm_joint_back_and_forth.cpp` | [examples/vega_1/cpp/basic_examples/control/cycle_arm.cpp](vega_1/cpp/basic_examples/control/cycle_arm.cpp) |
| `examples/cpp/basic_examples/control/move_one_arm_joint_by_offset.cpp` | [examples/vega_1/cpp/basic_examples/control/cycle_arm.cpp](vega_1/cpp/basic_examples/control/cycle_arm.cpp) |
| `examples/cpp/basic_examples/control/move_torso_joint_offset_and_return.cpp` | [examples/vega_1/cpp/basic_examples/control/cycle_torso.cpp](vega_1/cpp/basic_examples/control/cycle_torso.cpp) |
| `examples/cpp/basic_examples/control/move_torso_joint_relative_and_return.cpp` | [examples/vega_1/cpp/basic_examples/control/cycle_torso_relative.cpp](vega_1/cpp/basic_examples/control/cycle_torso.cpp) |
| `examples/cpp/basic_examples/control/cycle_arm_joints_from_zero.cpp` | [examples/vega_1/cpp/basic_examples/control/cycle_arm.cpp](vega_1/cpp/basic_examples/control/cycle_arm.cpp) |
| `examples/cpp/benchmark/arm_tracking/benchmark_arm_ramped_step_tracking.cpp` | [examples/vega_1/cpp/benchmark/arm_tracking/benchmark_arm_step.cpp](vega_1/cpp/benchmark/arm_tracking/benchmark_arm_step.cpp) |
| `examples/cpp/benchmark/arm_tracking/benchmark_arm_sine_tracking.cpp` | [examples/vega_1/cpp/benchmark/arm_tracking/benchmark_arm_sine.cpp](vega_1/cpp/benchmark/arm_tracking/benchmark_arm_sine.cpp) |
| `examples/cpp/benchmark/connection/benchmark_connect_and_query_latency.cpp` | [examples/vega_1/cpp/benchmark/connection/benchmark_connection_latency.cpp](vega_1/cpp/benchmark/connection/benchmark_connection_latency.cpp) |
| `examples/python/advanced_examples/configure_arm_force_torque_sensor.py` | [examples/vega_1/python/advanced_examples/configure_arm_ft_sensor.py](vega_1/python/advanced_examples/configure_arm_ft_sensor.py) |
| `examples/python/advanced_examples/configure_end_effector_baud_rate.py` | [examples/vega_1/python/advanced_examples/configure_ee_baud_rate.py](vega_1/python/advanced_examples/configure_ee_baud_rate.py) |
| `examples/python/advanced_examples/drive_chassis_with_sinusoidal_steering.py` | [examples/vega_1/python/advanced_examples/steer_chassis_sine.py](vega_1/python/advanced_examples/steer_chassis_sine.py) |
| `examples/python/advanced_examples/move_arms_to_named_pose_sequentially.py` | [examples/vega_1/python/advanced_examples/pose_arms_sequentially.py](vega_1/python/advanced_examples/pose_arms_sequentially.py) |
| `examples/python/advanced_examples/move_arms_to_named_pose_with_planner.py` | [examples/vega_1/python/advanced_examples/plan_arm_pose.py](vega_1/python/advanced_examples/plan_arm_pose.py) |
| `examples/python/advanced_examples/move_both_arms_to_named_pose.py` | [examples/vega_1/python/advanced_examples/pose_both_arms.py](vega_1/python/advanced_examples/pose_both_arms.py) |
| `examples/python/advanced_examples/move_end_effectors_by_relative_offsets.py` | [examples/vega_1/python/advanced_examples/offset_end_effectors.py](vega_1/python/advanced_examples/offset_end_effectors.py) |
| `examples/python/advanced_examples/move_to_pose/move_arm_to_named_pose.py` | [examples/vega_1/python/advanced_examples/move_to_pose/pose_arm.py](vega_1/python/advanced_examples/move_to_pose/pose_arm.py) |
| `examples/python/advanced_examples/move_to_pose/move_head_to_named_pose.py` | [examples/vega_1/python/advanced_examples/move_to_pose/pose_head.py](vega_1/python/advanced_examples/move_to_pose/pose_head.py) |
| `examples/python/advanced_examples/move_to_pose/move_torso_to_named_pose.py` | [examples/vega_1/python/advanced_examples/move_to_pose/pose_torso.py](vega_1/python/advanced_examples/move_to_pose/pose_torso.py) |
| `examples/python/advanced_examples/prepare_robot_and_replay_trajectory.py` | [examples/vega_1/python/advanced_examples/replay_trajectory.py](vega_1/python/advanced_examples/replay_trajectory.py) |
| `examples/python/basic_examples/control/close_then_open_hand.py` | [examples/vega_1/python/basic_examples/control/cycle_hand.py](vega_1/python/basic_examples/control/cycle_hand.py) |
| `examples/python/basic_examples/control/drive_chassis_in_six_directions.py` | [examples/vega_1/python/basic_examples/control/exercise_chassis.py](vega_1/python/basic_examples/control/exercise_chassis.py) |
| `examples/python/basic_examples/control/jog_each_arm_joint_back_and_forth.py` | [examples/vega_1/python/basic_examples/control/cycle_arm.py](vega_1/python/basic_examples/control/cycle_arm.py) |
| `examples/python/basic_examples/control/move_one_arm_joint_by_offset.py` | [examples/vega_1/python/basic_examples/control/cycle_arm.py](vega_1/python/basic_examples/control/cycle_arm.py) |
| `examples/python/basic_examples/control/move_torso_joint_offset_and_return.py` | [examples/vega_1/python/basic_examples/control/cycle_torso.py](vega_1/python/basic_examples/control/cycle_torso.py) |
| `examples/python/basic_examples/control/move_torso_joint_relative_and_return.py` | [examples/vega_1/python/basic_examples/control/cycle_torso_relative.py](vega_1/python/basic_examples/control/cycle_torso.py) |
| `examples/python/basic_examples/control/cycle_arm_joints_from_zero.py` | [examples/vega_1/python/basic_examples/control/cycle_arm.py](vega_1/python/basic_examples/control/cycle_arm.py) |
| `examples/python/benchmark/arm_tracking/benchmark_arm_ramped_step_tracking.py` | [examples/vega_1/python/benchmark/arm_tracking/benchmark_arm_step.py](vega_1/python/benchmark/arm_tracking/benchmark_arm_step.py) |
| `examples/python/benchmark/arm_tracking/benchmark_arm_sine_tracking.py` | [examples/vega_1/python/benchmark/arm_tracking/benchmark_arm_sine.py](vega_1/python/benchmark/arm_tracking/benchmark_arm_sine.py) |
| `examples/python/benchmark/connection/benchmark_connect_and_query_latency.py` | [examples/vega_1/python/benchmark/connection/benchmark_connection_latency.py](vega_1/python/benchmark/connection/benchmark_connection_latency.py) |
| `examples/python/benchmark/connection/monitor_heartbeat_under_gil_load.py` | [examples/vega_1/python/benchmark/connection/stress_heartbeat_gil.py](vega_1/python/benchmark/connection/stress_heartbeat_gil.py) |
| `examples/python/teleop/gamepad_chassis_and_head_control.py` | [examples/vega_1/python/teleop/gamepad_chassis_head.py](vega_1/python/teleop/gamepad_chassis_head.py) |
| `examples/rust/src/advanced_examples/configure_arm_force_torque_sensor.rs` | [examples/vega_1/rust/src/advanced_examples/configure_arm_ft_sensor.rs](vega_1/rust/src/advanced_examples/configure_arm_ft_sensor.rs) |
| `examples/rust/src/advanced_examples/configure_end_effector_baud_rate.rs` | [examples/vega_1/rust/src/advanced_examples/configure_ee_baud_rate.rs](vega_1/rust/src/advanced_examples/configure_ee_baud_rate.rs) |
| `examples/rust/src/advanced_examples/drive_chassis_with_sinusoidal_steering.rs` | [examples/vega_1/rust/src/advanced_examples/steer_chassis_sine.rs](vega_1/rust/src/advanced_examples/steer_chassis_sine.rs) |
| `examples/rust/src/advanced_examples/move_arms_to_named_pose_sequentially.rs` | [examples/vega_1/rust/src/advanced_examples/pose_arms_sequentially.rs](vega_1/rust/src/advanced_examples/pose_arms_sequentially.rs) |
| `examples/rust/src/advanced_examples/move_both_arms_to_named_pose.rs` | [examples/vega_1/rust/src/advanced_examples/pose_both_arms.rs](vega_1/rust/src/advanced_examples/pose_both_arms.rs) |
| `examples/rust/src/advanced_examples/move_to_pose/move_arm_to_named_pose.rs` | [examples/vega_1/rust/src/advanced_examples/move_to_pose/pose_arm.rs](vega_1/rust/src/advanced_examples/move_to_pose/pose_arm.rs) |
| `examples/rust/src/advanced_examples/move_to_pose/move_head_to_named_pose.rs` | [examples/vega_1/rust/src/advanced_examples/move_to_pose/pose_head.rs](vega_1/rust/src/advanced_examples/move_to_pose/pose_head.rs) |
| `examples/rust/src/advanced_examples/move_to_pose/move_torso_to_named_pose.rs` | [examples/vega_1/rust/src/advanced_examples/move_to_pose/pose_torso.rs](vega_1/rust/src/advanced_examples/move_to_pose/pose_torso.rs) |
| `examples/rust/src/advanced_examples/prepare_robot_and_replay_trajectory.rs` | [examples/vega_1/rust/src/advanced_examples/replay_trajectory.rs](vega_1/rust/src/advanced_examples/replay_trajectory.rs) |
| `examples/rust/src/basic_examples/control/close_then_open_hand.rs` | [examples/vega_1/rust/src/basic_examples/control/cycle_hand.rs](vega_1/rust/src/basic_examples/control/cycle_hand.rs) |
| `examples/rust/src/basic_examples/control/drive_chassis_in_six_directions.rs` | [examples/vega_1/rust/src/basic_examples/control/exercise_chassis.rs](vega_1/rust/src/basic_examples/control/exercise_chassis.rs) |
| `examples/rust/src/basic_examples/control/jog_each_arm_joint_back_and_forth.rs` | [examples/vega_1/rust/src/basic_examples/control/cycle_arm.rs](vega_1/rust/src/basic_examples/control/cycle_arm.rs) |
| `examples/rust/src/basic_examples/control/move_one_arm_joint_by_offset.rs` | [examples/vega_1/rust/src/basic_examples/control/cycle_arm.rs](vega_1/rust/src/basic_examples/control/cycle_arm.rs) |
| `examples/rust/src/basic_examples/control/move_torso_joint_offset_and_return.rs` | [examples/vega_1/rust/src/basic_examples/control/cycle_torso.rs](vega_1/rust/src/basic_examples/control/cycle_torso.rs) |
| `examples/rust/src/basic_examples/control/move_torso_joint_relative_and_return.rs` | [examples/vega_1/rust/src/basic_examples/control/cycle_torso_relative.rs](vega_1/rust/src/basic_examples/control/cycle_torso.rs) |
| `examples/rust/src/basic_examples/control/cycle_arm_joints_from_zero.rs` | [examples/vega_1/rust/src/basic_examples/control/cycle_arm.rs](vega_1/rust/src/basic_examples/control/cycle_arm.rs) |
| `examples/rust/src/benchmark/arm_tracking/benchmark_arm_ramped_step_tracking.rs` | [examples/vega_1/rust/src/benchmark/arm_tracking/benchmark_arm_step.rs](vega_1/rust/src/benchmark/arm_tracking/benchmark_arm_step.rs) |
| `examples/rust/src/benchmark/arm_tracking/benchmark_arm_sine_tracking.rs` | [examples/vega_1/rust/src/benchmark/arm_tracking/benchmark_arm_sine.rs](vega_1/rust/src/benchmark/arm_tracking/benchmark_arm_sine.rs) |
| `examples/rust/src/benchmark/connection/benchmark_connect_and_query_latency.rs` | [examples/vega_1/rust/src/benchmark/connection/benchmark_connection_latency.rs](vega_1/rust/src/benchmark/connection/benchmark_connection_latency.rs) |

Torso cycles now use planned tracked motions: `cycle_torso_setpoint` was renamed to `cycle_torso` in Python, C++, and Rust (Rust binary: `control-cycle-torso`). `cycle_torso --relative` uses a relative planned move followed by an absolute return.

`cycle_torso_relative` is now merged into `cycle_torso --relative` in all three languages. The merged default delta is 0.05 rad; pass `--delta 0.1` to reproduce the former relative example default.

`jog_arm_joints` and `sweep_arm_joints` were merged into `cycle_arm_joints`, which is removed in turn: `cycle_arm` (measured-start out-and-back, `--relative` for relative targets) is the one arm-cycling example in all three languages.

// Copyright (C) 2026 Dexmate Inc.
//
// This software is dual-licensed:
//
// 1. GNU Affero General Public License v3.0 (AGPL-3.0)
//    See LICENSE-AGPL for details
//
// 2. Commercial License
//    For commercial licensing terms, contact: contact@dexmate.ai
use dexcontrol::{MotionOptions, MotionState, Robot, Wait};
use std::time::Duration;

#[test]
fn component_owns_connection_and_motion_completes() {
    let robot = Robot::simulated("vega_1").unwrap();
    let head = robot.joints("head").unwrap();
    drop(robot);
    assert_eq!(head.name().unwrap(), "head");
    let target = head.get_joint_pos().unwrap();
    assert_eq!(target.len(), 3);
    let motion = head
        .move_to_joint_pos(&target, MotionOptions::default())
        .unwrap();
    drop(head);
    motion.wait(Duration::from_secs(5)).unwrap();
    assert_eq!(motion.state().unwrap(), MotionState::Succeeded);
}

#[test]
fn errors_are_preserved_without_sending_invalid_commands() {
    let robot = Robot::simulated("vega_1").unwrap();
    assert!(robot.joints("head\0oops").is_err());
    assert!(robot.joints("missing").is_err());
    let head = robot.joints("head").unwrap();
    let target = head.get_joint_pos().unwrap();
    assert!(head
        .move_to_joint_pos(
            &target,
            MotionOptions {
                velocity_scale: Some(f64::NAN),
                ..Default::default()
            }
        )
        .is_err());
    assert!(head
        .set_joint_pos(&target, Wait::Timeout(Duration::ZERO))
        .is_err());
    robot.set_software_estop(true).unwrap();
    let error = head.set_joint_pos(&target, Wait::NoWait).unwrap_err();
    assert_eq!(error.code, 8);
    assert!(error.message.contains("E-stop"));
    assert!(head.get_joint_pos().is_ok());
    robot.close().unwrap();
    assert!(head.get_joint_pos().is_err());
}

#[test]
fn shared_handles_support_concurrent_reads() {
    let robot = Robot::simulated("vega_1").unwrap();
    let head = std::sync::Arc::new(robot.joints("head").unwrap());
    let readers: Vec<_> = (0..4)
        .map(|_| {
            let head = head.clone();
            std::thread::spawn(move || head.get_joint_pos().unwrap())
        })
        .collect();
    for reader in readers {
        assert_eq!(reader.join().unwrap().len(), 3);
    }
    robot.close().unwrap();
}

#[test]
fn joint_feedback_limits_modes_and_direct_options() {
    use dexcontrol::{CommandOptions, DurationLimit, JointMode, WaitOptions};
    let robot = Robot::simulated("vega_1").unwrap();
    let head = robot.joints("head").unwrap();
    let state = head.joint_state(DurationLimit::Configured).unwrap();
    assert_eq!(state.position.len(), head.joint_names().unwrap().len());
    assert!(state.timestamp_ns > 0);
    let limits = head.joint_limits().unwrap().unwrap();
    assert_eq!(limits.lower.len(), state.position.len());
    assert_eq!(limits.upper.len(), limits.lower.len());
    assert!(head.joint_errors().unwrap().iter().all(|code| *code == 0));
    head.set_modes(&vec![JointMode::Position; state.position.len()])
        .unwrap();
    assert_eq!(
        head.modes().unwrap(),
        vec![Some(JointMode::Position); state.position.len()]
    );
    head.set_joint_pos_with(
        &state.position,
        Some(&vec![0.; state.position.len()]),
        CommandOptions::default(),
        WaitOptions::default(),
    )
    .unwrap();
    assert!(head.is_joint_pos_reached(&state.position, 0.1).unwrap());
    assert!(head
        .set_joint_pos_with(
            &state.position,
            Some(&[]),
            CommandOptions::default(),
            WaitOptions::default()
        )
        .is_err());
    assert!(head
        .command_position(&state.position, Some(&[]), DurationLimit::Configured)
        .is_err());
    assert!(head.release_brake(true, &[]).is_err());
    head.stop().unwrap();
    robot.close().unwrap();
}

#[test]
fn motion_groups_keep_members_alive_and_reject_bad_indices() {
    use std::collections::BTreeMap;
    let robot = Robot::simulated("vega_1").unwrap();
    let mut targets = BTreeMap::new();
    for name in ["left_arm", "right_arm"] {
        targets.insert(
            name.to_owned(),
            robot.joints(name).unwrap().get_joint_pos().unwrap(),
        );
    }
    let group = robot
        .move_to_joint_positions(&targets, MotionOptions::default())
        .unwrap();
    assert_eq!(group.len().unwrap(), 2);
    assert!(!group.is_empty().unwrap());
    assert!(group.member(2).is_err());
    group.wait(Duration::from_secs(3)).unwrap();
    assert_eq!(group.state().unwrap(), MotionState::Succeeded);
    let member = group.member(0).unwrap();
    drop(group);
    assert!(member.id().unwrap() > 0);
    assert_eq!(member.refresh().unwrap(), MotionState::Succeeded);
    assert!(member.message().is_ok());
    let submitted = robot
        .start_joint_motions(&targets, MotionOptions::default())
        .unwrap();
    assert!(submitted.report_json.contains("all_delivered"));
    submitted.group.wait(Duration::from_secs(3)).unwrap();
    robot.close().unwrap();
}

#[test]
fn trajectories_validate_all_shapes_before_starting() {
    use dexcontrol::{TrajectoryOptions, TrajectoryTrack, WaitOptions};
    use std::collections::BTreeMap;
    let robot = Robot::simulated("vega_1").unwrap();
    let head = robot.joints("head").unwrap();
    let start = head.get_joint_pos().unwrap();
    assert!(head
        .move_joint_trajectory(
            &[start.clone(), vec![0.]],
            &[0., 0.1],
            MotionOptions::default(),
            WaitOptions::default()
        )
        .is_err());
    assert!(head
        .move_joint_trajectory(
            std::slice::from_ref(&start),
            &[],
            MotionOptions::default(),
            WaitOptions::default()
        )
        .is_err());
    let track = TrajectoryTrack {
        positions: vec![start.clone(), start.clone()],
        velocities: None,
    };
    let mut tracks = BTreeMap::from([("head".to_owned(), track)]);
    robot
        .execute_trajectory(&tracks, 100., TrajectoryOptions::default())
        .unwrap();
    tracks.get_mut("head").unwrap().velocities = Some(vec![vec![0.]]);
    assert!(robot
        .execute_trajectory(&tracks, 100., TrajectoryOptions::default())
        .is_err());
    assert_eq!(head.get_joint_pos().unwrap(), start);
    robot.close().unwrap();
}

#[test]
fn invalid_options_and_strings_fail_before_a_command_is_sent() {
    use dexcontrol::{CommandOptions, DurationLimit, StepLimit, WaitOptions};
    let robot = Robot::simulated("vega_1").unwrap();
    let head = robot.joints("head").unwrap();
    let start = head.get_joint_pos().unwrap();
    for bad in [f64::NAN, f64::INFINITY, -1., 0.] {
        assert!(head
            .set_joint_pos_with(
                &start,
                None,
                CommandOptions {
                    max_step: StepLimit::Maximum(bad),
                    ..Default::default()
                },
                WaitOptions::default()
            )
            .is_err());
        assert!(head
            .move_to_joint_pos_with(
                &start,
                MotionOptions::default(),
                WaitOptions {
                    convergence_tolerance: Some(bad),
                    ..Default::default()
                }
            )
            .is_err());
    }
    assert!(head
        .joint_state(DurationLimit::Maximum(Duration::from_nanos(1)))
        .is_err());
    assert!(head.request_json("bad\0role", "{}").is_err());
    assert!(robot.query("bad\0service", &[]).is_err());
    assert_eq!(head.get_joint_pos().unwrap(), start);
    robot.close().unwrap();
}

#[test]
fn emergency_stop_blocks_groups_trajectories_and_services_but_not_feedback() {
    use dexcontrol::{TrajectoryOptions, TrajectoryTrack};
    use std::collections::BTreeMap;
    let robot = Robot::simulated("vega_1").unwrap();
    let head = robot.joints("head").unwrap();
    let start = head.get_joint_pos().unwrap();
    robot.set_software_estop(true).unwrap();
    let status = robot.estop_status().unwrap();
    assert!(status.engaged && status.software_estop_enabled);
    let targets = BTreeMap::from([("head".to_owned(), start.clone())]);
    assert!(robot
        .move_to_joint_positions(&targets, MotionOptions::default())
        .is_err());
    let tracks = BTreeMap::from([(
        "head".to_owned(),
        TrajectoryTrack {
            positions: vec![start.clone()],
            velocities: None,
        },
    )]);
    assert!(robot
        .execute_trajectory(&tracks, 100., TrajectoryOptions::default())
        .is_err());
    assert!(robot
        .joints("right_arm")
        .unwrap()
        .set_pid(&[1.; 7])
        .is_err());
    assert_eq!(head.get_joint_pos().unwrap(), start);
    assert!(robot.battery().is_ok());
    robot.close().unwrap();
}

#[test]
fn sensor_snapshots_outlive_connection_and_cameras_report_missing_frames() {
    use dexcontrol::ConnectOptions;
    let robot = Robot::connect(ConnectOptions {
        profile: Some("vega_1"),
        simulated: true,
        enable_sensors: vec!["lidar_2d_front", "head_camera"],
        ..Default::default()
    })
    .unwrap();
    assert!(
        robot
            .battery_with_age(Some(Duration::from_secs(5)))
            .unwrap()
            .value
            .voltage
            > 0.
    );
    assert!(robot.imu("head_camera", None).unwrap().value.timestamp_ns > 0);
    let scan = robot.lidar_2d("lidar_2d_front", None).unwrap();
    let lidar_robot = Robot::connect(ConnectOptions {
        profile: Some("vega_1p"),
        simulated: true,
        enable_sensors: vec!["lidar_3d_front"],
        ..Default::default()
    })
    .unwrap();
    let cloud = lidar_robot.lidar_3d("lidar_3d_front", None).unwrap();
    assert!(
        lidar_robot
            .imu("lidar_3d_front", None)
            .unwrap()
            .value
            .timestamp_ns
            > 0
    );
    lidar_robot.close().unwrap();
    drop(lidar_robot);
    let camera = robot.camera("head_camera").unwrap();
    let streams = camera.streams().unwrap();
    assert!(!streams.is_empty());
    camera.subscribe(&streams[0], 0).unwrap();
    // The public simulator has no image source: an absent frame must stay None.
    // Frame pixel lifetime is covered by the compile-fail ownership test.
    assert!(camera.latest_frame(&streams[0]).unwrap().is_none());
    assert_eq!(camera.stats(&streams[0]).unwrap().frames_received, 0);
    assert!(!camera
        .is_active(Some(&streams[0]), Duration::from_secs(1))
        .unwrap());
    camera.unsubscribe(&streams[0]).unwrap();
    robot.close().unwrap();
    drop(robot);
    assert!(!scan.ranges().unwrap().is_empty());
    assert_eq!(scan.ranges().unwrap().len(), scan.angles().unwrap().len());
    assert!(scan.info().is_ok() && scan.age().is_ok() && scan.intensities().is_ok());
    assert!(!cloud.x().unwrap().is_empty());
    assert_eq!(cloud.y().unwrap().len(), cloud.z().unwrap().len());
    assert!(
        cloud.intensity().is_ok() && cloud.ring().is_ok() && cloud.point_timestamps_ns().is_ok()
    );
    assert!(cloud.info().is_ok() && cloud.age().is_ok());
}

#[test]
fn context_config_and_connection_ownership() {
    use dexcontrol::{ConnectOptions, Context};
    let options = ConnectOptions {
        profile: Some("vega_1"),
        simulated: true,
        ..Default::default()
    };
    let config = options.resolved_config_json().unwrap();
    assert!(config.contains("head"));
    let context = Context::new(2).unwrap();
    let robot = context.connect(options).unwrap();
    drop(context);
    assert!(robot.has_component("head").unwrap());
    assert!(!robot.has_component("absent").unwrap());
    assert!(robot.is_active().unwrap());
    assert!(robot
        .wait_for_state("head", Duration::from_secs(1))
        .unwrap());
    assert!(robot.config_json().unwrap().contains("head"));
    assert!(robot.health_json().is_ok());
    assert!(robot.connection_report_json().is_ok());
    let other = robot.try_clone().unwrap();
    other.close().unwrap();
    assert!(robot.joints("head").unwrap().get_joint_pos().is_err());
    let robot = Robot::from_resolved_json(&config, true).unwrap();
    assert!(robot.joints("head").is_ok());
    robot.close().unwrap();
    assert!(Robot::from_resolved_json("{}", true).is_err());
}

#[test]
fn chassis_and_rate_limiter_validate_input() {
    use dexcontrol::{ChassisOptions, RateLimiter};
    let robot = Robot::simulated("vega_1").unwrap();
    let chassis = robot.chassis(None).unwrap();
    assert!(!chassis.steering_angle().unwrap().is_empty());
    assert!(chassis.wheel_position().is_ok() && chassis.wheel_velocity().is_ok());
    chassis
        .set_velocity(0., 0., 0., ChassisOptions::default())
        .unwrap();
    assert!(chassis
        .set_velocity(f64::NAN, 0., 0., ChassisOptions::default())
        .is_err());
    assert!(chassis.set_motion_state(&[0.], &[]).is_err());
    chassis.stop().unwrap();
    robot.close().unwrap();
    assert!(RateLimiter::new(0., 10, false).is_err());
    let rate = RateLimiter::new(1000., 10, false).unwrap();
    rate.sleep().unwrap();
    assert!(rate.stats().is_ok());
    assert!(rate.actual_rate().unwrap().is_finite());
    assert!(rate.average_rate().unwrap().is_finite());
    rate.reset().unwrap();
    rate.done().unwrap();
}

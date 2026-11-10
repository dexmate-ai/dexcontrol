// Copyright (C) 2026 Dexmate Inc.
// SPDX-License-Identifier: Apache-2.0
//! Move the base forward 1 m, then turn 90 degrees, by measured amounts.
//!
//! Drive straight by --distance metres (default 1.0) with chassis.move_by, then turn in place by --angle degrees (default 90) with chassis.turn_by. Each is a closed loop on the base odometry: the base drives while the odometry is watched, slows down near the target, and corrects only if the settled error is outside the tolerance; --speed overrides the cruise speed. Print the amount measured after each motion, then the odometry pose and its sample counters. The heading comes from an enabled base IMU where the model has one (vega_1: --enable-sensor chassis_imu, optionally --yaw-source chassis_imu), otherwise from the wheels (vega_1p). Stop the base on exit.
//!
//! Uses a Robot control connection with startup arm initialization turned off; shutdown requests a stop. Real hardware is selected unless --simulated is supplied.

use dexcontrol::{ClosedLoopOptions, OdometryOptions, YawSource};
use dexcontrol_examples::{Args, Result};

#[tokio::main]
async fn main() -> Result {
    let args = Args::parse(
        "Move the base forward 1 m, then turn 90 degrees, by measured amounts. Drive straight by --distance metres (default 1.0) with chassis.move_by, then turn in place by --angle degrees (default 90) with chassis.turn_by. Each is a closed loop on the base odometry: the base drives while the odometry is watched, slows down near the target, and corrects only if the settled error is outside the tolerance; --speed overrides the cruise speed. Print the amount measured after each motion, then the odometry pose and its sample counters. The heading comes from an enabled base IMU where the model has one (vega_1: --enable-sensor chassis_imu, optionally --yaw-source chassis_imu), otherwise from the wheels (vega_1p). Stop the base on exit. Uses a Robot control connection with startup arm initialization turned off; shutdown requests a stop. Real hardware is selected unless --simulated is supplied.",
        &["angle", "distance", "speed", "yaw-source"],
    );
    let distance = args.number("distance", 1.0)?;
    let angle = args.number("angle", 90.0)?;
    let speed = args.number("speed", 0.0)?;
    let yaw_source = match args.text("yaw-source", "auto").as_str() {
        "auto" => YawSource::Auto,
        "wheels" => YawSource::Wheels,
        name => YawSource::Imu(name.to_owned()),
    };
    let robot = args
        .connect_with(&[], |options| options.skip_arm_initialization = true)
        .await?;
    dexcontrol_examples::run_robot(&robot, async {
        if !robot.supports_odometry()? {
            return Err("this robot's chassis has no wheel model; odometry is unavailable".into());
        }
        let odometry = robot.start_odometry(OdometryOptions {
            yaw_source,
            ..OdometryOptions::default()
        })?;
        println!("heading from: {}", odometry.status()?.yaw_source);
        let options = ClosedLoopOptions {
            // Zero keeps the default; anything else, including a negative
            // value, reaches the options' own validation.
            speed: (speed != 0.0).then_some(speed),
            ..ClosedLoopOptions::default()
        };
        let chassis = robot.chassis(None)?;
        let result = (|| -> Result {
            let moved = chassis.move_by(distance, options.clone())?;
            println!("moved {moved:+.3} m (asked {distance:+.3} m)");
            let turned = chassis.turn_by(angle.to_radians(), options.clone())?;
            println!(
                "turned {:+.1} deg (asked {angle:+.1} deg)",
                turned.to_degrees()
            );
            let pose = odometry.pose()?;
            let status = odometry.status()?;
            println!(
                "pose: x={:+.3} m  y={:+.3} m  yaw={:+.1} deg | samples={} imu={} slip={} gaps={}",
                pose.x,
                pose.y,
                pose.yaw.to_degrees(),
                status.samples,
                status.imu_samples,
                status.slip_samples,
                status.gaps
            );
            Ok(())
        })();
        chassis.stop()?;
        result
    })
    .await
}

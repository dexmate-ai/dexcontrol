// Copyright (C) 2026 Dexmate Inc.
//
// This software is dual-licensed:
//
// 1. GNU Affero General Public License v3.0 (AGPL-3.0)
//    See LICENSE-AGPL for details
//
// 2. Commercial License
//    For commercial licensing terms, contact: contact@dexmate.ai

//! Read and print observations from configured IMUs.
//!
//! Read IMU streams from head_camera and chassis_imu (when declared) once by default, waiting up to 5 s for activity. --sensor changes the requested list; optional repeated reads use a 0.1 s interval.
//!
//! Filters undeclared IMU names before connecting.
//!
//! Uses a Robot control connection: compatible arm modes may be enabled during startup, and shutdown requests a stop. Real hardware is selected unless --simulated is supplied.

use dexcontrol_examples::{format, pause, Args, Result};
use std::time::Duration;

#[tokio::main]
async fn main() -> Result {
    let args = Args::parse(
        "Read and print observations from configured IMUs. Read IMU streams from head_camera and chassis_imu (when declared) once by default, waiting up to 5 s for activity. --sensor changes the requested list; optional repeated reads use a 0.1 s interval. Filters undeclared IMU names before connecting. Uses a Robot control connection: compatible arm modes may be enabled during startup, and shutdown requests a stop. Real hardware is selected unless --simulated is supplied.",
        &["period", "samples", "sensor"],
    );
    let mut requested = args.list("sensor");
    if requested.is_empty() {
        requested = vec!["head_camera".into(), "chassis_imu".into()];
    }
    let config = args.resolved_config()?;
    let sensors = requested
        .into_iter()
        .filter(|name| {
            let present = config["sensors"].get(name).is_some();
            if !present {
                println!("{name}: not declared by this profile, skipping");
            }
            present
        })
        .collect::<Vec<_>>();
    if sensors.is_empty() {
        return Err(dexcontrol_examples::invalid(
            "none of the requested IMUs exist in this profile",
        ));
    }
    let samples = args.integer("samples", 1)?;
    let period = args.duration("period", 0.1)?;
    let robot = args.connect(&sensors).await?;
    dexcontrol_examples::run_robot(&robot, async {
    for sensor in &sensors {
        dexcontrol_examples::wait_for_imu(&robot, sensor, Duration::from_secs(5)).await?;
    }
    for _ in 0..samples {
        for sensor in &sensors {
            let state = robot.imu(sensor, None)?.value;
            println!("\n{sensor}\n  acceleration:     {} m/s^2\n  angular_velocity: {} rad/s\n  orientation:      {}\n  magnetic_field:   {}\n  timestamp:        {} ns", format(&state.acceleration), format(&state.angular_velocity), format(&state.orientation), format(&state.magnetic_field), state.timestamp_ns);
        }
        pause(period).await;
    }
        Ok(())
    }).await
}

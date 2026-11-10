// Copyright (C) 2026 Dexmate Inc.
// SPDX-License-Identifier: Apache-2.0

//! Print current joint states and chassis state.
//!
//! Read all available joint components once, skipping unavailable joint state; --component narrows that list. Also print chassis state when available, independently of the component filter. This is a cached state listing, not a complete diagnostic report.
//!
//! Prints named joint rows with model-derived units; unavailable fields appear as —. Chassis steering, wheel encoder positions and wheel speeds are listed separately. Reports nonempty joint error indices.
//!
//! Uses a Robot control connection: compatible arm modes may be enabled during startup, and shutdown requests a stop. Real hardware is selected unless --simulated is supplied.

use dexcontrol_examples::{Args, Result};

fn units(kind: &str) -> [&'static str; 3] {
    match kind {
        "prismatic" => ["m", "m/s", "N"],
        "revolute" | "continuous" => ["rad", "rad/s", "N·m"],
        _ => ["units", "units/s", "units"],
    }
}
fn value(values: &[f64], index: usize, unit: &str) -> String {
    match values.get(index) {
        Some(number) => {
            let number = if number.abs() < 0.00005 { 0.0 } else { *number };
            format!("{number:.4} {unit}")
        }
        None => "—".into(),
    }
}

#[tokio::main]
async fn main() -> Result {
    let args = Args::parse(
        "Print current joint states and chassis state. Read all available joint components once, skipping unavailable joint state; --component narrows that list. Also print chassis state when available, independently of the component filter. This is a cached state listing, not a complete diagnostic report. Prints named joint rows with model-derived units; unavailable fields appear as —. Chassis steering, wheel encoder positions and wheel speeds are listed separately. Reports nonempty joint error indices. Uses a Robot control connection: compatible arm modes may be enabled during startup, and shutdown requests a stop. Real hardware is selected unless --simulated is supplied.",
        &["component"],
    );
    let robot = args.connect(&[]).await?;
    dexcontrol_examples::run_robot(&robot, async {
        let config: serde_json::Value = serde_json::from_str(&robot.config_json()?)?;
        println!("Cached joint states (— = not reported)");
        let mut names = args.list("component");
        if names.is_empty() {
            names = robot.component_names()?;
        }
        for name in names {
            let Ok(joints) = robot.joints(&name) else {
                continue;
            };
            let Ok(state) = joints.get_joint_state(dexcontrol::DurationLimit::Configured) else {
                continue;
            };
            println!(
                "\n{name}\n  {:<20} {:>16} {:>16} {:>16} {:>14}",
                "Joint", "Position", "Velocity", "Effort", "Current"
            );
            let metadata = config["joint_metadata"].get(&name);
            for (i, joint_name) in joints.joint_names()?.iter().enumerate() {
                let unit = units(
                    metadata
                        .and_then(|items| items.get(i))
                        .and_then(|joint| joint["joint_type"].as_str())
                        .unwrap_or(""),
                );
                println!(
                    "  {joint_name:<20} {:>16} {:>16} {:>16} {:>14}",
                    value(&state.position, i, unit[0]),
                    value(&state.velocity, i, unit[1]),
                    value(&state.torque, i, unit[2]),
                    value(&state.current, i, "A")
                );
            }
            let raw: serde_json::Value = serde_json::from_str(&robot.joint_state_json(&name)?)?;
            if raw["error_joint_indices"]
                .as_array()
                .is_some_and(|v| !v.is_empty())
            {
                println!(
                    "  Reported joint error indices: {:?}",
                    raw["error_joint_indices"]
                );
            }
        }
        if let Ok(chassis) = robot.chassis(None) {
            match chassis.steering_angle().and_then(|steer| {
                let count = steer.len();
                let mut positions = steer;
                positions.extend(chassis.wheel_encoder_pos()?);
                Ok((count, positions, chassis.wheel_velocity()?))
            }) {
                Ok((steer_count, positions, speeds)) => {
                    println!(
                        "\nchassis\n  {:<20} {:<22} {:>16} {:>16}",
                        "Joint", "Measurement", "Position", "Wheel speed"
                    );
                    let component = config["components"].get("chassis");
                    let metadata = config["joint_metadata"].get("chassis");
                    for i in 0..positions.len() {
                        let name = component
                            .and_then(|c| c["joints"]["names"].get(i))
                            .and_then(|n| n.as_str())
                            .map(str::to_owned)
                            .unwrap_or_else(|| format!("joint_{i}"));
                        let unit = units(
                            metadata
                                .and_then(|items| items.get(i))
                                .and_then(|joint| joint["joint_type"].as_str())
                                .unwrap_or(""),
                        );
                        println!(
                            "  {name:<20} {:<22} {:>16} {:>16}",
                            if i < steer_count {
                                "Steering angle"
                            } else {
                                "Wheel encoder"
                            },
                            value(&positions, i, unit[0]),
                            if i < steer_count {
                                "—".into()
                            } else {
                                value(&speeds, i - steer_count, "m/s")
                            }
                        );
                    }
                }
                Err(_) => println!("\nchassis: state unavailable"),
            }
        }
        Ok(())
    })
    .await
}

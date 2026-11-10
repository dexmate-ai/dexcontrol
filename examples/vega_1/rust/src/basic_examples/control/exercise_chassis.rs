// Copyright (C) 2026 Dexmate Inc.
// SPDX-License-Identifier: Apache-2.0

//! Drive the chassis in six directions, requesting a stop between them.
//!
//! Command forward, backward, left, right, counter-clockwise, then clockwise for 3 s each, streaming commands at 50 Hz by default (--control-hz). Linear speed is 0.1 m/s and turn speed is 0.1 rad/s. Request stops between directions and on cleanup; requires a base capable of strafing.
//!
//! Uses a Robot control connection: compatible arm modes may be enabled during startup, and shutdown requests a stop. Real hardware is selected unless --simulated is supplied.

use dexcontrol_examples::{Args, Result};

#[tokio::main]
async fn main() -> Result {
    let args = Args::parse(
        "Drive the chassis in six directions, requesting a stop between them. Command forward, backward, left, right, counter-clockwise, then clockwise for 3 s each, streaming commands at 50 Hz by default (--control-hz). Linear speed is 0.1 m/s and turn speed is 0.1 rad/s. Request stops between directions and on cleanup; requires a base capable of strafing. Uses a Robot control connection: compatible arm modes may be enabled during startup, and shutdown requests a stop. Real hardware is selected unless --simulated is supplied.",
        &["control-hz", "duration", "speed"],
    );
    let speed = args.number("speed", 0.1)?;
    let duration = args.duration("duration", 3.0)?;
    let robot = args.connect(&[]).await?;
    dexcontrol_examples::run_robot(&robot, async {
        let chassis = robot.chassis(None)?;
        let commands = [
            ("forward", [speed, 0.0, 0.0]),
            ("backward", [-speed, 0.0, 0.0]),
            ("left", [0.0, speed, 0.0]),
            ("right", [0.0, -speed, 0.0]),
            ("counter-clockwise", [0.0, 0.0, speed]),
            ("clockwise", [0.0, 0.0, -speed]),
        ];
        let control_hz = args.number("control-hz", 50.0)?;
        println!(
            "Chassis exercise | {} phases | {control_hz} Hz",
            commands.len()
        );
        println!("Commands in robot frame: +vx forward, +vy left, +wz counter-clockwise.");
        println!("Each phase includes steering alignment if needed, then timed driving.");
        for (index, (label, [vx, vy, wz])) in commands.into_iter().enumerate() {
            println!(
                "[{}/{}] {label} | {:.2} s | vx={vx:+.3} m/s, vy={vy:+.3} m/s, wz={wz:+.3} rad/s",
                index + 1,
                commands.len(),
                duration.as_secs_f64()
            );
            chassis.drive_for(vx, vy, wz, duration, control_hz)?;
            println!("      Command stream complete; stop sent.");
        }
        chassis.stop()?;
        Ok(())
    })
    .await
}

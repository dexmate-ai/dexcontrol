// Copyright (C) 2026 Dexmate Inc.
// SPDX-License-Identifier: Apache-2.0

//! Drive forward while sinusoidally changing both steering angles.
//!
//! For 6 s at 50 Hz, command both wheel speeds to 0.5 m/s and both steering angles to 0.6*sin(2*pi*0.5*t) rad. Request a stop on completion/cleanup. Requires a steer-drive base; it does not use generic chassis yaw velocity.
//!
//! Uses a Robot control connection: compatible arm modes may be enabled during startup, and shutdown requests a stop. Real hardware is selected unless --simulated is supplied.

use dexcontrol::RateLimiter;
use dexcontrol_examples::{Args, Result};
use std::time::Instant;

#[tokio::main]
async fn main() -> Result {
    let args = Args::parse(
        "Drive forward while sinusoidally changing both steering angles. For 6 s at 50 Hz, command both wheel speeds to 0.5 m/s and both steering angles to 0.6*sin(2*pi*0.5*t) rad. Request a stop on completion/cleanup. Requires a steer-drive base; it does not use generic chassis yaw velocity. Uses a Robot control connection: compatible arm modes may be enabled during startup, and shutdown requests a stop. Real hardware is selected unless --simulated is supplied.",
        &["amplitude", "control-hz", "duration", "frequency", "speed"],
    );
    let amplitude = args.number("amplitude", 0.6)?;
    let frequency = args.number("frequency", 0.5)?;
    let speed = args.number("speed", 0.5)?;
    let duration = args.duration("duration", 6.0)?;
    let rate = RateLimiter::new(args.number("control-hz", 50.0)?, 10, false)?;
    let robot = args.connect(&[]).await?;
    dexcontrol_examples::run_robot(&robot, async {
        let chassis = robot.chassis(None)?;
        let started = Instant::now();
        let outcome: dexcontrol::Result<()> = async {
            while started.elapsed() < duration {
                let elapsed = started.elapsed().as_secs_f64();
                let steering = amplitude * (2.0 * std::f64::consts::PI * frequency * elapsed).sin();
                chassis.set_motion_state(&[steering; 2], &[speed; 2])?;
                rate.sleep()?;
                tokio::task::yield_now().await;
            }
            Ok(())
        }
        .await;
        let stopped = chassis.stop();
        outcome?;
        stopped?;
        Ok(())
    })
    .await
}

// Copyright (C) 2026 Dexmate Inc.
// SPDX-License-Identifier: Apache-2.0

//! Read and print one arm force-torque observation.
//!
//! Wait up to 5 s for the right-arm wrench stream and print one observation by default. Optional repeated reads use a 0.1 s interval; no calibration or zeroing is requested.
//!
//! Uses a Robot control connection: compatible arm modes may be enabled during startup, and shutdown requests a stop. Real hardware is selected unless --simulated is supplied.

use dexcontrol_examples::{component_name, format, pause, Args, Result};
use std::time::Duration;

#[tokio::main]
async fn main() -> Result {
    let args = Args::parse(
        "Read and print one arm force-torque observation. Wait up to 5 s for the right-arm wrench stream and print one observation by default. Optional repeated reads use a 0.1 s interval; no calibration or zeroing is requested. Uses a Robot control connection: compatible arm modes may be enabled during startup, and shutdown requests a stop. Real hardware is selected unless --simulated is supplied.",
        &["period", "samples", "side"],
    );
    let arm = component_name(args.side()?, "arm");
    let samples = args.integer("samples", 1)?;
    let period = args.duration("period", 0.1)?;
    let robot = args.connect(&[]).await?;
    dexcontrol_examples::run_robot(&robot, async {
        if !robot.wait_for_state(&arm, Duration::from_secs(5))? {
            return Err(dexcontrol_examples::invalid(
                "wrench state did not become active",
            ));
        }
        for _ in 0..samples {
            let state = robot.sensors().wrench(&arm, None)?.value;
            let split = state.values.len().min(3);
            println!(
                "force={} N torque={} N*m (timestamp {} ns)",
                format(&state.values[..split]),
                format(&state.values[split..]),
                state.timestamp_ns
            );
            pause(period).await;
        }
        Ok(())
    })
    .await
}

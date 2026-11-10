// Copyright (C) 2026 Dexmate Inc.
// SPDX-License-Identifier: Apache-2.0

//! Read and print arm wrist-button states.
//!
//! Read one right-arm button observation by default. --samples and --period repeat the reads; the default repeat interval is 0.1 s.
//!
//! Uses a Robot control connection: compatible arm modes may be enabled during startup, and shutdown requests a stop. Real hardware is selected unless --simulated is supplied.

use dexcontrol_examples::{component_name, pause, Args, Result};

#[tokio::main]
async fn main() -> Result {
    let args = Args::parse(
        "Read and print arm wrist-button states. Read one right-arm button observation by default. --samples and --period repeat the reads; the default repeat interval is 0.1 s. Uses a Robot control connection: compatible arm modes may be enabled during startup, and shutdown requests a stop. Real hardware is selected unless --simulated is supplied.",
        &["period", "samples", "side"],
    );
    let samples = args.integer("samples", 1)?;
    let period = args.duration("period", 0.1)?;
    let robot = args.connect(&[]).await?;
    dexcontrol_examples::run_robot(&robot, async {
        let arm = robot.joints(&component_name(args.side()?, "arm"))?;
        for _ in 0..samples {
            match arm.get_button_state() {
                Ok(Some((blue, green))) => println!(
                    "blue={} green={}",
                    if blue { "pressed" } else { "released" },
                    if green { "pressed" } else { "released" }
                ),
                Ok(None) => println!("no button sample"),
                Err(error) => return Err(error.into()),
            }
            pause(period).await;
        }
        Ok(())
    })
    .await
}

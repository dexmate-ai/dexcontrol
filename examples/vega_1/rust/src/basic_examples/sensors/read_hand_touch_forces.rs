// Copyright (C) 2026 Dexmate Inc.
// SPDX-License-Identifier: Apache-2.0

//! Read and print fingertip forces from one hand.
//!
//! Read the right hand once by default and report missing touch data. Optional repeated reads use a 0.1 s interval.
//!
//! Uses a Robot control connection: compatible arm modes may be enabled during startup, and shutdown requests a stop. Real hardware is selected unless --simulated is supplied.

use dexcontrol_examples::{component_name, format, pause, Args, Result};

#[tokio::main]
async fn main() -> Result {
    let args = Args::parse(
        "Read and print fingertip forces from one hand. Read the right hand once by default and report missing touch data. Optional repeated reads use a 0.1 s interval. Uses a Robot control connection: compatible arm modes may be enabled during startup, and shutdown requests a stop. Real hardware is selected unless --simulated is supplied.",
        &["period", "samples", "side"],
    );
    let samples = args.integer("samples", 1)?;
    let period = args.duration("period", 0.1)?;
    let robot = args.connect(&[]).await?;
    dexcontrol_examples::run_robot(&robot, async {
        let hand = robot.joints(&component_name(args.side()?, "hand"))?;
        for _ in 0..samples {
            match hand.get_touch_data() {
                Ok(Some(forces)) => println!("{}", format(&forces)),
                Ok(None) => println!("no touch sample"),
                Err(error) => return Err(error.into()),
            }
            pause(period).await;
        }
        Ok(())
    })
    .await
}

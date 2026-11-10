// Copyright (C) 2026 Dexmate Inc.
// SPDX-License-Identifier: Apache-2.0

//! Read and print battery status.
//!
//! Wait up to 5 s for battery activity and print one observation by default, including charge, voltage, current and temperature. Optional repeated reads use a 0.5 s interval.
//!
//! Uses a Robot control connection: compatible arm modes may be enabled during startup, and shutdown requests a stop. Real hardware is selected unless --simulated is supplied.

use dexcontrol_examples::{pause, Args, Result};
use std::time::Duration;

#[tokio::main]
async fn main() -> Result {
    let args = Args::parse(
        "Read and print battery status. Wait up to 5 s for battery activity and print one observation by default, including charge, voltage, current and temperature. Optional repeated reads use a 0.5 s interval. Uses a Robot control connection: compatible arm modes may be enabled during startup, and shutdown requests a stop. Real hardware is selected unless --simulated is supplied.",
        &["period", "samples"],
    );
    let samples = args.integer("samples", 1)?;
    let period = args.duration("period", 0.5)?;
    let robot = args.connect(&[]).await?;
    dexcontrol_examples::run_robot(&robot, async {
    if !robot
        .wait_for_state("battery", Duration::from_secs(5))?
    {
        return Err(dexcontrol_examples::invalid(
            "battery state did not become active",
        ));
    }
    for index in 0..samples {
        let state = robot.battery()?;
        println!(
            "Battery status ({}/{samples})\n  Charge       {:.1} %\n  Voltage      {:.2} V\n  Current      {:.2} A\n  Power        {:.2} W\n  Temperature  {:.1} °C\n",
            index + 1, f64::from(state.percentage), state.voltage, state.current,
            state.voltage * state.current, state.temperature
        );
        pause(period).await;
    }
        Ok(())
    }).await
}

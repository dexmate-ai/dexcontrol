// Copyright (C) 2026 Dexmate Inc.
// SPDX-License-Identifier: Apache-2.0

//! Print component temperatures and battery status.
//!
//! Read each available component temperature stream once, then battery temperature/status. --component narrows component selection; battery reporting is separate. Report unavailable temperature sources.
//!
//! Uses a Robot control connection: compatible arm modes may be enabled during startup, and shutdown requests a stop. Real hardware is selected unless --simulated is supplied.

use dexcontrol_examples::{Args, Result};

#[tokio::main]
async fn main() -> Result {
    let args = Args::parse(
        "Print component temperatures and battery status. Read each available component temperature stream once, then battery temperature/status. --component narrows component selection; battery reporting is separate. Report unavailable temperature sources. Uses a Robot control connection: compatible arm modes may be enabled during startup, and shutdown requests a stop. Real hardware is selected unless --simulated is supplied.",
        &["component"],
    );
    let robot = args.connect(&[]).await?;
    dexcontrol_examples::run_robot(&robot, async {
        let mut names = args.list("component");
        let explicit_selection = !names.is_empty();
        if names.is_empty() {
            names = robot.component_names()?;
        }
        let mut unavailable = Vec::new();
        for name in names {
            if name == "battery" { continue; }
            let config: serde_json::Value = serde_json::from_str(&robot.config_json()?)?;
            if config["components"][&name]["endpoints"]["temperature_sub_topic"].is_null() {
                if explicit_selection { println!("No temperature stream configured for: {name}"); }
                continue;
            }
            // Temperature streams are slow (about 1 Hz) and connecting does
            // not wait for them, so give the first sample up to 5 s to arrive.
            let deadline = std::time::Instant::now() + std::time::Duration::from_secs(5);
            let mut latest = robot.sensors().temperature_json(&name);
            while latest.is_err() && std::time::Instant::now() < deadline {
                tokio::time::sleep(std::time::Duration::from_millis(100)).await;
                latest = robot.sensors().temperature_json(&name);
            }
            match latest {
                Ok(json) => {
                    let state: serde_json::Value = serde_json::from_str(&json)?;
                    let mut reported = false;
                    for (group, readings) in state["temperatures"].as_object().into_iter().flatten() {
                        for (label, value) in readings.as_object().into_iter().flatten() {
                            if let Some(value) = value.as_f64() {
                                if !reported { println!("\n{name} temperatures"); reported = true; }
                                println!("  {group}/{label}: {value:.1} °C");
                            }
                        }
                    }
                    if !reported { unavailable.push(name); }
                }
                Err(_) => unavailable.push(name),
            }
        }
        if robot.has_component("battery")? {
            match robot.battery() {
                Ok(state) => println!(
                    "\nBattery status\n  Charge       {:.1} %\n  Voltage      {:.2} V\n  Current      {:.2} A\n  Power        {:.2} W\n  Temperature  {:.1} °C",
                    f64::from(state.percentage), state.voltage, state.current,
                    state.voltage * state.current, state.temperature
                ),
                Err(_) => println!("\nBattery status unavailable."),
            }
        } else { println!("\nNo battery configured on this robot."); }
        if !unavailable.is_empty() {
            println!("\nTemperature readings unavailable: {}", unavailable.join(", "));
        }
        Ok(())
    })
    .await
}

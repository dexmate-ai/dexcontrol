// Copyright (C) 2026 Dexmate Inc.
// SPDX-License-Identifier: Apache-2.0

//! Read, enable, or disable arm force-torque sensor modes.
//!
//! Require positional get, enable, or disable. Address both arms by default and print service replies; this does not zero the force sensors.
//!
//! Uses a Robot control connection: compatible arm modes may be enabled during startup, and shutdown requests a stop. Real hardware is selected unless --simulated is supplied.

use dexcontrol_examples::{component_name, invalid, Args, Result};

#[tokio::main]
async fn main() -> Result {
    let args = Args::parse_with_positionals(
        "Read, enable, or disable arm force-torque sensor modes. Require positional get, enable, or disable. Address both arms by default and print service replies; this does not zero the force sensors. Uses a Robot control connection: compatible arm modes may be enabled during startup, and shutdown requests a stop. Real hardware is selected unless --simulated is supplied.",
        &["side"], "action: get | enable | disable");
    let action = args.positional(0, "");
    if !["get", "enable", "disable"].contains(&action.as_str()) {
        return Err(invalid("action must be get, enable, or disable"));
    }
    let robot = args.connect(&[]).await?;
    dexcontrol_examples::run_robot(&robot, async {
        for side in args.sides()? {
            let arm = robot.joints(&component_name(side, "arm"))?;
            let result = if action == "get" {
                arm.get_force_torque_sensor_mode()?
            } else {
                arm.set_force_torque_sensor(action == "enable")?
            };
            println!(
                "{side}_arm\n{}",
                serde_json::to_string_pretty(&serde_json::from_str::<serde_json::Value>(&result)?)?
            );
        }
        Ok(())
    })
    .await
}

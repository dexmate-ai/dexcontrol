// Copyright (C) 2026 Dexmate Inc.
// SPDX-License-Identifier: Apache-2.0

//! Enable or disable head motors.
//!
//! The optional positional mode is enable or disable; without it, send disable. Print the firmware reply; no head target is commanded.
//!
//! Uses a Robot control connection: compatible arm modes may be enabled during startup, and shutdown requests a stop. Real hardware is selected unless --simulated is supplied.

use dexcontrol_examples::{invalid, Args, Result};

#[tokio::main]
async fn main() -> Result {
    let args = Args::parse_with_positionals("Enable or disable head motors. The optional positional mode is enable or disable; without it, send disable. Print the firmware reply; no head target is commanded. Uses a Robot control connection: compatible arm modes may be enabled during startup, and shutdown requests a stop. Real hardware is selected unless --simulated is supplied.", &[], "mode: enable | disable (default: disable)");
    let mode = args.positional(0, "disable");
    if mode != "enable" && mode != "disable" {
        return Err(invalid("mode must be enable or disable"));
    }
    let robot = args.connect(&[]).await?;
    dexcontrol_examples::run_robot(&robot, async {
        let head = robot.joints("head")?;
        let mode = if mode == "enable" {
            dexcontrol::JointMode::Enable
        } else {
            dexcontrol::JointMode::Disable
        };
        println!(
            "{}",
            head.set_modes(&vec![mode; head.joint_names()?.len()])?
        );
        Ok(())
    })
    .await
}

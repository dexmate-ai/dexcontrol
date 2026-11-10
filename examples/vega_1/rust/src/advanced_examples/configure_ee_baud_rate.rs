// Copyright (C) 2026 Dexmate Inc.
// SPDX-License-Identifier: Apache-2.0

//! Read or set end-effector RS485 baud rates.
//!
//! Require positional get or set. Address both arm end-effector interfaces by default; set uses 115200 baud unless overridden. Print service replies.
//!
//! Uses a Robot control connection: compatible arm modes may be enabled during startup, and shutdown requests a stop. Real hardware is selected unless --simulated is supplied.

use dexcontrol_examples::{component_name, invalid, Args, Result};

#[tokio::main]
async fn main() -> Result {
    let args = Args::parse_with_positionals(
        "Read or set end-effector RS485 baud rates. Require positional get or set. Address both arm end-effector interfaces by default; set uses 115200 baud unless overridden. Print service replies. Uses a Robot control connection: compatible arm modes may be enabled during startup, and shutdown requests a stop. Real hardware is selected unless --simulated is supplied.",
        &["baud-rate", "side"], "action: get | set");
    let action = args.positional(0, "");
    if action != "get" && action != "set" {
        return Err(invalid("action must be get or set"));
    }
    let baud_rate = u32::try_from(args.integer("baud-rate", 115_200)?)?;
    let robot = args.connect(&[]).await?;
    dexcontrol_examples::run_robot(&robot, async {
        for side in args.sides()? {
            let arm = robot.joints(&component_name(side, "arm"))?;
            let result = if action == "get" {
                arm.get_ee_baud_rate()?
            } else {
                arm.set_ee_baud_rate(baud_rate)?
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

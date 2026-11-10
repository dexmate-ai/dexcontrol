// Copyright (C) 2026 Dexmate Inc.
// SPDX-License-Identifier: Apache-2.0

//! Request a reboot of one robot control board.
//!
//! Require positional arm, torso, or chassis. The arm board controls both arms. A rebooting board stops controlling its joints, so ask for confirmation first and require typing yes; --yes skips the prompt and simulated runs do not prompt. Send the request and exit; do not wait for the board to come back online.
//!
//! Uses a Robot control connection with startup arm initialization turned off, so the tool also reaches an arm whose motor cannot enable; shutdown requests a stop. Real hardware is selected unless --simulated is supplied.

use dexcontrol_examples::{invalid, Args, Result};

#[tokio::main]
async fn main() -> Result {
    let args = Args::parse_with_positionals("Request a reboot of one robot control board. Require positional arm, torso, or chassis. The arm board controls both arms. A rebooting board stops controlling its joints, so ask for confirmation first and require typing yes; --yes skips the prompt and simulated runs do not prompt. Send the request and exit; do not wait for the board to come back online. Uses a Robot control connection with startup arm initialization turned off, so the tool also reaches an arm whose motor cannot enable; shutdown requests a stop. Real hardware is selected unless --simulated is supplied.", &["yes"], "board: arm | torso | chassis");
    let board = args.positional(0, "");
    if !["arm", "torso", "chassis"].contains(&board.as_str()) {
        return Err(invalid("board must be arm, torso, or chassis"));
    }
    args.confirm_hazard(&format!(
        "rebooting the {board} control board. Its joints are not controlled while it restarts{}. \
         Make sure the robot is at rest, supported where needed, and that people are clear.",
        if board == "arm" {
            " (the arm board controls BOTH arms)"
        } else {
            ""
        }
    ))?;
    let robot = args
        .connect_with(&[], |options| options.skip_arm_initialization = true)
        .await?;
    dexcontrol_examples::run_robot(&robot, async {
        robot.reboot(&board)?;
        println!("{board}: reboot request sent");
        Ok(())
    })
    .await
}

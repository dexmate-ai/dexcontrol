// Copyright (C) 2026 Dexmate Inc.
// SPDX-License-Identifier: Apache-2.0

//! Close one hand using its model-defined pose.
//!
//! Command the right hand closed and wait 2 s. Optional --grasp-torque changes the grip setting for a gripper; omitted values use model defaults. The delay is not a convergence check.
//!
//! Uses a Robot control connection: compatible arm modes may be enabled during startup, and shutdown requests a stop. Real hardware is selected unless --simulated is supplied.

use dexcontrol::Wait;
use dexcontrol_examples::{component_name, pause, Args, Result};

#[tokio::main]
async fn main() -> Result {
    let args = Args::parse(
        "Close one hand using its model-defined pose. Command the right hand closed and wait 2 s. Optional --grasp-torque changes the grip setting for a gripper; omitted values use model defaults. The delay is not a convergence check. Uses a Robot control connection: compatible arm modes may be enabled during startup, and shutdown requests a stop. Real hardware is selected unless --simulated is supplied.",
        &["grasp-torque", "side", "wait-time"],
    );
    let robot = args.connect(&[]).await?;
    dexcontrol_examples::run_robot(&robot, async {
        let hand = robot.joints(&component_name(args.side()?, "hand"))?;
        if !args.text("grasp-torque", "").is_empty() && hand.grasp_torque()?.is_some() {
            println!(
                "grasp torque {}",
                hand.set_grasp_torque(args.number("grasp-torque", 0.0)?, false)?
            );
        }
        hand.set_joint_pos(&hand.get_pose("close", true)?, Wait::NoWait)?;
        pause(args.duration("wait-time", 2.0)?).await;
        Ok(())
    })
    .await
}

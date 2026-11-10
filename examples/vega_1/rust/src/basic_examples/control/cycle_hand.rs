// Copyright (C) 2026 Dexmate Inc.
// SPDX-License-Identifier: Apache-2.0

//! Close one hand, then reopen it.
//!
//! Command the right hand to its model-defined closed pose, wait 2 s, command its open pose, then wait 2 s. These delays are not tracked-motion completion checks.
//!
//! Uses a Robot control connection: compatible arm modes may be enabled during startup, and shutdown requests a stop. Real hardware is selected unless --simulated is supplied.

use dexcontrol::Wait;
use dexcontrol_examples::{component_name, pause, Args, Result};

#[tokio::main]
async fn main() -> Result {
    let args = Args::parse("Close one hand, then reopen it. Command the right hand to its model-defined closed pose, wait 2 s, command its open pose, then wait 2 s. These delays are not tracked-motion completion checks. Uses a Robot control connection: compatible arm modes may be enabled during startup, and shutdown requests a stop. Real hardware is selected unless --simulated is supplied.", &["side", "wait-time"]);
    let wait = args.duration("wait-time", 2.0)?;
    let robot = args.connect(&[]).await?;
    dexcontrol_examples::run_robot(&robot, async {
        let hand = robot.joints(&component_name(args.side()?, "hand"))?;
        hand.set_joint_pos(&hand.get_pose("close", true)?, Wait::NoWait)?;
        pause(wait).await;
        hand.set_joint_pos(&hand.get_pose("open", true)?, Wait::NoWait)?;
        pause(wait).await;
        Ok(())
    })
    .await
}

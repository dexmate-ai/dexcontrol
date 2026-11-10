// Copyright (C) 2026 Dexmate Inc.
// SPDX-License-Identifier: Apache-2.0

//! Move the head to a named pose using its model-declared reference frame.
//!
//! Move to home using model-defined torso-pitch compensation; tucked stays joint-relative. --passthrough uses raw stored values. Wait up to 10 s. No collision checking is performed.
//!
//! Uses a Robot control connection: compatible arm modes may be enabled during startup, and shutdown requests a stop. Real hardware is selected unless --simulated is supplied.

use dexcontrol::Wait;
use dexcontrol_examples::{Args, Result};

#[tokio::main]
async fn main() -> Result {
    let args = Args::parse(
        "Move the head to a named pose using its model-declared reference frame. Move to home using model-defined torso-pitch compensation; tucked stays joint-relative. --passthrough uses raw stored values. Wait up to 10 s. No collision checking is performed. Uses a Robot control connection: compatible arm modes may be enabled during startup, and shutdown requests a stop. Real hardware is selected unless --simulated is supplied.",
        &["passthrough", "pose", "timeout"],
    );
    let robot = args.connect(&[]).await?;
    dexcontrol_examples::run_robot(&robot, async {
        let head = robot.joints("head")?;
        let pose = args.text("pose", "home");
        let target = robot
            .joints("head")?
            .get_pose(&pose, args.flag("passthrough"))?;
        let motion = head.move_to_joint_pos_with(
            &target,
            dexcontrol::MotionOptions::default(),
            Wait::Timeout(args.duration("timeout", 10.0)?).into(),
        )?;
        println!("{:?}", motion.state()?);
        Ok(())
    })
    .await
}

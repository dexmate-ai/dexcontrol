// Copyright (C) 2026 Dexmate Inc.
// SPDX-License-Identifier: Apache-2.0

//! Move one arm to a named pose using its model-declared reference frame.
//!
//! Move the right arm to L_shape with model-defined torso-pitch compensation; wait up to 10 s. folded and folded_closed_hand stay joint-relative and receive no compensation. zero stores seven zeros relative to an upright torso and compensates only torso tilt from upright; --passthrough returns literal zeros. --passthrough uses raw stored values while retaining target validation. No collision checking is performed.
//!
//! Uses a Robot control connection: compatible arm modes may be enabled during startup, and shutdown requests a stop. Real hardware is selected unless --simulated is supplied.

use dexcontrol::Wait;
use dexcontrol_examples::{component_name, Args, Result};

#[tokio::main]
async fn main() -> Result {
    let args = Args::parse(
        "Move one arm to a named pose using its model-declared reference frame. Move the right arm to L_shape with model-defined torso-pitch compensation; wait up to 10 s. folded and folded_closed_hand stay joint-relative and receive no compensation. zero stores seven zeros relative to an upright torso and compensates only torso tilt from upright; --passthrough returns literal zeros. --passthrough uses raw stored values while retaining target validation. No collision checking is performed. Uses a Robot control connection: compatible arm modes may be enabled during startup, and shutdown requests a stop. Real hardware is selected unless --simulated is supplied.",
        &["passthrough", "pose", "side", "timeout"],
    );
    let name = component_name(args.side()?, "arm");
    let robot = args.connect(&[]).await?;
    dexcontrol_examples::run_robot(&robot, async {
        let arm = robot.joints(&name)?;
        let target = arm.get_pose(&args.text("pose", "L_shape"), args.flag("passthrough"))?;
        let motion = arm.move_to_joint_pos_with(
            &target,
            dexcontrol::MotionOptions::default(),
            Wait::Timeout(args.duration("timeout", 10.0)?).into(),
        )?;
        println!("{:?}", motion.state()?);
        Ok(())
    })
    .await
}

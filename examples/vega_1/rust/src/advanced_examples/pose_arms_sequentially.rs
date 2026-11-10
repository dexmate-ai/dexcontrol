// Copyright (C) 2026 Dexmate Inc.
// SPDX-License-Identifier: Apache-2.0

//! Move arms sequentially to a named pose without collision checking.
//!
//! Move the left arm and then the right arm to L_shape at velocity scale 0.5, waiting up to 10 s each. --side can select one arm. Resolve model-declared pose frames, including torso compensation for L_shape. No homing or collision planning is performed.
//!
//! Uses a Robot control connection: compatible arm modes may be enabled during startup, and shutdown requests a stop. Real hardware is selected unless --simulated is supplied.

use dexcontrol::{MotionOptions, Wait};
use dexcontrol_examples::{component_name, Args, Result};

#[tokio::main]
async fn main() -> Result {
    let args = Args::parse(
        "Move arms sequentially to a named pose without collision checking. Move the left arm and then the right arm to L_shape at velocity scale 0.5, waiting up to 10 s each. --side can select one arm. Resolve model-declared pose frames, including torso compensation for L_shape. No homing or collision planning is performed. Uses a Robot control connection: compatible arm modes may be enabled during startup, and shutdown requests a stop. Real hardware is selected unless --simulated is supplied.",
        &["pose", "side", "timeout", "velocity-scale"],
    );
    let pose = args.text("pose", "L_shape");
    let timeout = Wait::Timeout(args.duration("timeout", 10.0)?);
    let robot = args.connect(&[]).await?;
    dexcontrol_examples::run_robot(&robot, async {
        for side in args.sides()? {
            let name = component_name(side, "arm");
            let arm = robot.joints(&name)?;
            let target = robot.joints(&name)?.get_pose(&pose, false)?;
            let motion = arm.move_to_joint_pos_with(
                &target,
                MotionOptions {
                    velocity_scale: Some(args.number("velocity-scale", 0.5)?),
                    ..MotionOptions::default()
                },
                timeout.into(),
            )?;
            println!("{name}: {:?}", motion.state()?);
        }
        Ok(())
    })
    .await
}

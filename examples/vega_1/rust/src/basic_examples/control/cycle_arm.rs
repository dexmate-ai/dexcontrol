// Copyright (C) 2026 Dexmate Inc.
// SPDX-License-Identifier: Apache-2.0

//! Move one arm joint by an offset, then return to its starting position.
//!
//! Save the right arm positions, move joint 0 by +0.2 rad, then return to the saved absolute start after outward success. By default compute an absolute outward target; --relative resolves the offset from fresh feedback. Both motions use velocity scale 0.2 and a 10 s wait deadline.
//!
//! Uses a Robot control connection: compatible arm modes may be enabled during startup, and shutdown requests a stop. Real hardware is selected unless --simulated is supplied.

use dexcontrol::{MotionOptions, Wait};
use dexcontrol_examples::{component_name, invalid, Args, Result};

#[tokio::main]
async fn main() -> Result {
    let args = Args::parse(
        "Move one arm joint by an offset, then return to its starting position. Save the right arm positions, move joint 0 by +0.2 rad, then return to the saved absolute start after outward success. By default compute an absolute outward target; --relative resolves the offset from fresh feedback. Both motions use velocity scale 0.2 and a 10 s wait deadline. Uses a Robot control connection: compatible arm modes may be enabled during startup, and shutdown requests a stop. Real hardware is selected unless --simulated is supplied.",
        &["delta", "joint", "relative", "side", "timeout", "velocity-scale"],
    );
    let timeout = Wait::Timeout(args.duration("timeout", 10.0)?);
    let joint = args.integer("joint", 0)?;
    let robot = args.connect(&[]).await?;
    dexcontrol_examples::run_robot(&robot, async {
        let arm = robot.joints(&component_name(args.side()?, "arm"))?;
        let start = arm.get_joint_pos()?;
        let relative = args.flag("relative");
        let mut target = if relative {
            vec![0.0; start.len()]
        } else {
            start.clone()
        };
        let value = target
            .get_mut(joint)
            .ok_or_else(|| invalid(format!("--joint must be less than {}", start.len())))?;
        *value += args.number("delta", 0.2)?;
        let mut options = MotionOptions {
            relative,
            velocity_scale: Some(args.number("velocity-scale", 0.2)?),
            ..MotionOptions::default()
        };
        arm.move_to_joint_pos_with(&target, options, Wait::NoWait.into())?
            .wait_with(timeout.into())?;
        options.relative = false;
        arm.move_to_joint_pos_with(&start, options, Wait::NoWait.into())?
            .wait_with(timeout.into())?;
        Ok(())
    })
    .await
}

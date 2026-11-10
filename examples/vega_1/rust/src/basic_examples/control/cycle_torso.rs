// Copyright (C) 2026 Dexmate Inc.
// SPDX-License-Identifier: Apache-2.0

//! Move the torso to an offset joint target, then return to its starting position.
//!
//! Read the torso positions and move joint 0 by +0.1 rad with a planned tracked motion. By default compute an absolute target from the stored start; --relative lets the API resolve the offset against fresh feedback. Always return to the stored absolute start after outward success. Each motion wait has a 10 s deadline.
//!
//! Uses a Robot control connection: compatible arm modes may be enabled during startup, and shutdown requests a stop. Real hardware is selected unless --simulated is supplied.

use dexcontrol::{MotionOptions, Wait};
use dexcontrol_examples::{invalid, Args, Result};

#[tokio::main]
async fn main() -> Result {
    let args = Args::parse(
        "Move the torso to an offset joint target, then return to its starting position. Read the torso positions and move joint 0 by +0.1 rad with a planned tracked motion. By default compute an absolute target from the stored start; --relative lets the API resolve the offset against fresh feedback. Always return to the stored absolute start after outward success. Each motion wait has a 10 s deadline. Uses a Robot control connection: compatible arm modes may be enabled during startup, and shutdown requests a stop. Real hardware is selected unless --simulated is supplied.",
        &["delta", "joint", "relative", "timeout"],
    );
    let timeout = Wait::Timeout(args.duration("timeout", 10.0)?);
    let joint = args.integer("joint", 0)?;
    let robot = args.connect(&[]).await?;
    dexcontrol_examples::run_robot(&robot, async {
        let torso = robot.joints("torso")?;
        let start = torso.get_joint_pos()?;
        let relative = args.flag("relative");
        let mut target = if relative {
            vec![0.0; start.len()]
        } else {
            start.clone()
        };
        let joint_count = target.len();
        let value = target
            .get_mut(joint)
            .ok_or_else(|| invalid(format!("--joint must be less than {joint_count}")))?;
        *value += args.number("delta", 0.1)?;
        torso.move_to_joint_pos_with(
            &target,
            MotionOptions {
                relative,
                ..MotionOptions::default()
            },
            timeout.into(),
        )?;
        torso.move_to_joint_pos_with(&start, MotionOptions::default(), timeout.into())?;
        Ok(())
    })
    .await
}

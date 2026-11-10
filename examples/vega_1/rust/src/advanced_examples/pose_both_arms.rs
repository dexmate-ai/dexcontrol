// Copyright (C) 2026 Dexmate Inc.
// SPDX-License-Identifier: Apache-2.0

//! Move both arms to a named pose using independent tracked motions.
//!
//! Default to folded at velocity scale 0.5. Resolve each pose using its model-declared frame: folded poses are joint-relative; orientation reference poses use torso pitch. --passthrough uses raw stored values. Start both targets through best-effort fan-out, wait up to 10 s, and require aggregate success. No synchronized start or collision checking is provided.
//!
//! Uses a Robot control connection: compatible arm modes may be enabled during startup, and shutdown requests a stop. Real hardware is selected unless --simulated is supplied.

use dexcontrol::MotionOptions;
use dexcontrol_examples::{Args, Result};
use std::collections::BTreeMap;

#[tokio::main]
async fn main() -> Result {
    let args = Args::parse(
        "Move both arms to a named pose using independent tracked motions. Default to folded at velocity scale 0.5. Resolve each pose using its model-declared frame: folded poses are joint-relative; orientation reference poses use torso pitch. --passthrough uses raw stored values. Start both targets through best-effort fan-out, wait up to 10 s, and require aggregate success. No synchronized start or collision checking is provided. Uses a Robot control connection: compatible arm modes may be enabled during startup, and shutdown requests a stop. Real hardware is selected unless --simulated is supplied.",
        &["passthrough", "pose", "timeout", "velocity-scale"],
    );
    let pose = args.text("pose", "folded");
    let robot = args.connect(&[]).await?;
    dexcontrol_examples::run_robot(&robot, async {
        let mut targets = BTreeMap::new();
        for name in ["left_arm", "right_arm"] {
            let target = robot
                .joints(name)?
                .get_pose(&pose, args.flag("passthrough"))?;
            targets.insert(name.to_owned(), target);
        }
        let group = robot.move_to_joint_pos(
            &targets,
            MotionOptions {
                velocity_scale: Some(args.number("velocity-scale", 0.5)?),
                ..MotionOptions::default()
            },
        )?;
        group.wait(args.duration("timeout", 10.0)?)?;
        println!("Both arms: succeeded");
        Ok(())
    })
    .await
}

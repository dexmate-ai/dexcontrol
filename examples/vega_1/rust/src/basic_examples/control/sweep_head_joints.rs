// Copyright (C) 2026 Dexmate Inc.
// SPDX-License-Identifier: Apache-2.0

//! Sweep head joints through positive, negative, and zero targets.
//!
//! First command head joint 0 to -pi/6 rad with other joints zero. Then command each joint to +0.5, -0.5, and zero radians, with all other target joints zero. Finish at all zeros; wait up to 10 s per motion.
//!
//! The sequence uses the model joint count.
//!
//! Uses a Robot control connection: compatible arm modes may be enabled during startup, and shutdown requests a stop. Real hardware is selected unless --simulated is supplied.

use dexcontrol::Wait;
use dexcontrol_examples::{Args, Result};

#[tokio::main]
async fn main() -> Result {
    let args = Args::parse(
        "Sweep head joints through positive, negative, and zero targets. First command head joint 0 to -pi/6 rad with other joints zero. Then command each joint to +0.5, -0.5, and zero radians, with all other target joints zero. Finish at all zeros; wait up to 10 s per motion. The sequence uses the model joint count. Uses a Robot control connection: compatible arm modes may be enabled during startup, and shutdown requests a stop. Real hardware is selected unless --simulated is supplied.",
        &["delta", "timeout"],
    );
    let timeout = Wait::Timeout(args.duration("timeout", 10.0)?);
    let delta = args.number("delta", 0.5)?;
    let robot = args.connect(&[]).await?;
    dexcontrol_examples::run_robot(&robot, async {
        let head = robot.joints("head")?;
        let mut home = vec![0.0; head.joint_names()?.len()];
        if let Some(pitch) = home.first_mut() {
            *pitch = -std::f64::consts::PI / 6.0;
        }
        head.move_to_joint_pos_with(&home, dexcontrol::MotionOptions::default(), timeout.into())?;
        for joint in 0..head.joint_names()?.len() {
            for value in [delta, -delta, 0.0] {
                let mut target = vec![0.0; head.joint_names()?.len()];
                target[joint] = value;
                head.move_to_joint_pos_with(
                    &target,
                    dexcontrol::MotionOptions::default(),
                    timeout.into(),
                )?;
            }
        }
        Ok(())
    })
    .await
}

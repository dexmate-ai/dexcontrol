// Copyright (C) 2026 Dexmate Inc.
// SPDX-License-Identifier: Apache-2.0

//! Fold or unfold the robot in ordered, verified stages.
//!
//! Default: close available hands, move both arms to folded, then torso to folded and head to tucked. With --unfold: move torso to crouch45_high, then resolve/move head home, then resolve/move arms to L_shape. Skip absent components; check motion success and measured positions within 0.1 rad before advancing. No collision checking is provided.
//!
//! Publish hand-close setpoints without a fixed settling delay before the arm stage.
//!
//! Uses a Robot control connection: compatible arm modes may be enabled during startup, and shutdown requests a stop. Real hardware is selected unless --simulated is supplied.

use dexcontrol::{MotionOptions, Robot, Wait};
use dexcontrol_examples::{component_name, invalid, Args, Result};
use std::collections::BTreeMap;

async fn stage(
    robot: &Robot,
    poses: &[(&str, &str)],
    timeout: std::time::Duration,
    tolerance: f64,
    number: usize,
    total: usize,
) -> Result {
    let mut targets = BTreeMap::new();
    for &(name, pose) in poses {
        if robot.has_component(name)? {
            targets.insert(name.to_owned(), robot.joints(name)?.get_pose(pose, false)?);
        }
    }
    if targets.is_empty() {
        println!("Stage {number}/{total}: skipped (components unavailable)");
        return Ok(());
    }
    println!(
        "\nStage {number}/{total}: {} (wait limit: {} s)",
        targets
            .keys()
            .map(|name| {
                let pose = poses
                    .iter()
                    .find(|(component, _)| component == name)
                    .unwrap()
                    .1;
                format!("{name} -> {pose}")
            })
            .collect::<Vec<_>>()
            .join(", "),
        timeout.as_secs_f64()
    );
    let group = robot.move_to_joint_pos(&targets, MotionOptions::default())?;
    let wait_result = group.wait(timeout);
    let mut complete = wait_result.is_ok();
    if let Err(error) = &wait_result {
        println!("  Wait error: {error}");
    }
    for (index, (name, target)) in targets.iter().enumerate() {
        let motion = group.member(index)?;
        let pose = poses
            .iter()
            .find(|(component, _)| component == name)
            .unwrap()
            .1;
        println!("  {name} -> {pose}");
        let status = motion.state()?;
        complete &= status == dexcontrol::MotionState::Succeeded;
        println!(
            "    Motion: {} (id: {})",
            format!("{status:?}").to_lowercase(),
            motion.id()?
        );
        let message = motion.message()?;
        if !message.is_empty() {
            println!("    Reason: {message}");
        }
        let arrival = (|| -> Result {
            let joint = robot.joints(name)?;
            let measured = joint.get_joint_pos()?;
            if measured.len() != target.len()
                || measured.is_empty()
                || measured
                    .iter()
                    .chain(target.iter())
                    .any(|value| !value.is_finite())
            {
                return Err(invalid("invalid joint feedback shape or values"));
            }
            let (worst, error) = measured
                .iter()
                .zip(target.iter())
                .enumerate()
                .map(|(index, (actual, goal))| (index, (actual - goal).abs()))
                .max_by(|a, b| a.1.total_cmp(&b.1))
                .unwrap();
            let reached = error <= tolerance;
            complete &= reached;
            println!(
                "    Target: {} (max error: {error:.4} rad; tolerance: {tolerance} rad)",
                if reached { "reached" } else { "NOT reached" }
            );
            if !reached {
                println!(
                    "    Joint: {}; measured: {:.4} rad; target: {:.4} rad",
                    joint.joint_names()?[worst],
                    measured[worst],
                    target[worst]
                );
            }
            Ok(())
        })();
        if let Err(error) = arrival {
            complete = false;
            println!("    Target: unverified ({error})");
        }
    }
    if !complete {
        return Err(invalid("Stage stopped: motion success and target arrival were not both verified. No later stage will run."));
    }
    println!("  Stage complete.");
    Ok(())
}

#[tokio::main]
async fn main() -> Result {
    let args = Args::parse(
        "Fold or unfold the robot in ordered, verified stages. Default: close available hands, move both arms to folded, then torso to folded and head to tucked. With --unfold: move torso to crouch45_high, then resolve/move head home, then resolve/move arms to L_shape. Skip absent components; check motion success and measured positions within 0.1 rad before advancing. No collision checking is provided. Publish hand-close setpoints without a fixed settling delay before the arm stage. Uses a Robot control connection: compatible arm modes may be enabled during startup, and shutdown requests a stop. Real hardware is selected unless --simulated is supplied.",
        &["timeout", "tolerance", "unfold"],
    );
    let timeout = args.duration("timeout", 10.0)?;
    if timeout.is_zero() {
        return Err(invalid("timeout must be positive"));
    }
    let tolerance = args.number("tolerance", 0.1)?;
    if !tolerance.is_finite() || tolerance < 0.0 {
        return Err(invalid("tolerance must be finite and non-negative"));
    }
    println!(
        "{} robot",
        if args.flag("unfold") {
            "Unfold"
        } else {
            "Fold"
        }
    );
    let robot = args.connect(&[]).await?;
    dexcontrol_examples::run_robot(&robot, async {
        if args.flag("unfold") {
            stage(
                &robot,
                &[("torso", "crouch45_high")],
                timeout,
                tolerance,
                1,
                3,
            )
            .await?;
            stage(&robot, &[("head", "home")], timeout, tolerance, 2, 3).await?;
            stage(
                &robot,
                &[("left_arm", "L_shape"), ("right_arm", "L_shape")],
                timeout,
                tolerance,
                3,
                3,
            )
            .await?;
            println!("\nUnfold complete.");
        } else {
            for side in ["left", "right"] {
                let name = component_name(side, "hand");
                if robot.has_component(&name)? {
                    let hand = robot.joints(&name)?;
                    hand.set_joint_pos(&hand.get_pose("close", true)?, Wait::NoWait)?;
                }
            }
            stage(
                &robot,
                &[("left_arm", "folded"), ("right_arm", "folded")],
                timeout,
                tolerance,
                1,
                2,
            )
            .await?;
            stage(
                &robot,
                &[("torso", "folded"), ("head", "tucked")],
                timeout,
                tolerance,
                2,
                2,
            )
            .await?;
            println!("\nFold complete.");
        }
        Ok(())
    })
    .await
}

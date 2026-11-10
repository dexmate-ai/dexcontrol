// Copyright (C) 2026 Dexmate Inc.
// SPDX-License-Identifier: Apache-2.0

//! Jog a selected joint using keyboard commands.
//!
//! Default component is left_arm and selected joint is 0. w/s move in opposite directions, digits select a joint, and q exits. Interaction and command increments differ by language; see the language notes.
//!
//! Line input: press Enter after each command. Each w/s applies a 0.02 rad direct step by default; EOF or q exits. Does not implement key-hold velocity control.
//!
//! Uses a Robot control connection: compatible arm modes may be enabled during startup, and shutdown requests a stop. Real hardware is selected unless --simulated is supplied.

use dexcontrol::Wait;
use dexcontrol_examples::{invalid, Args, Result};
use std::io::{self, BufRead};

#[tokio::main]
async fn main() -> Result {
    let args = Args::parse(
        "Jog a selected joint using keyboard commands. Default component is left_arm and selected joint is 0. w/s move in opposite directions, digits select a joint, and q exits. Interaction and command increments differ by language; see the language notes. Line input: press Enter after each command. Each w/s applies a 0.02 rad direct step by default; EOF or q exits. Does not implement key-hold velocity control. Uses a Robot control connection: compatible arm modes may be enabled during startup, and shutdown requests a stop. Real hardware is selected unless --simulated is supplied.",
        &["component", "joint", "step"],
    );
    let component = args.text("component", "left_arm");
    let mut selected = args.integer("joint", 0)?;
    let step = args.number("step", 0.02)?;
    let robot = args.connect(&[]).await?;
    dexcontrol_examples::run_robot(&robot, async {
        let joints = robot.joints(&component)?;
        if selected >= joints.joint_names()?.len() {
            return Err(invalid("--joint is out of range"));
        }
        println!("w/s: move +/- {step} rad; 0-9: select joint; q: quit");
        // A detached input thread lets Ctrl-C finish even while stdin is idle.
        // Tokio blocking tasks would keep runtime shutdown waiting for Enter.
        let (sender, mut input) = tokio::sync::mpsc::channel(1);
        std::thread::spawn(move || {
            for line in io::stdin().lock().lines() {
                if sender.blocking_send(line).is_err() {
                    break;
                }
            }
        });
        while let Some(line) = input.recv().await {
            let line = line?;
            let command = line.trim();
            if command == "q" {
                break;
            }
            if let Ok(index) = command.parse::<usize>() {
                if index < joints.joint_names()?.len() {
                    selected = index;
                }
                continue;
            }
            let direction = match command {
                "w" => 1.0,
                "s" => -1.0,
                _ => {
                    println!("expected w, s, a joint index, or q");
                    continue;
                }
            };
            let mut target = joints.get_joint_pos()?;
            target[selected] += direction * step;
            joints.set_joint_pos(&target, Wait::NoWait)?;
            println!(
                "{} = {:.3} rad",
                joints.joint_names()?[selected],
                target[selected]
            );
        }
        joints.stop()?;
        Ok(())
    })
    .await
}

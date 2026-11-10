// Copyright (C) 2026 Dexmate Inc.
// SPDX-License-Identifier: Apache-2.0

//! Read, release, or engage arm brakes.
//!
//! Require positional status, release, or engage. status and engage address both arms and all their joints unless --side or --joint narrow them. release has no defaults: it requires --side left, right, or both, and either --joint INDEX (repeatable) or --all. Before releasing, warn that the arm must be physically supported, because a released joint can drop under gravity, and require typing yes; --yes skips the prompt and simulated runs do not prompt. This operates the brake service, not the motor-mode service.
//!
//! Uses a Robot control connection with startup arm initialization turned off, so the tool also reaches an arm whose motor cannot enable; shutdown requests a stop. Real hardware is selected unless --simulated is supplied.

use dexcontrol_examples::{component_name, invalid, Args, Result};

#[tokio::main]
async fn main() -> Result {
    let args = Args::parse_with_positionals("Read, release, or engage arm brakes. Require positional status, release, or engage. status and engage address both arms and all their joints unless --side or --joint narrow them. release has no defaults: it requires --side left, right, or both, and either --joint INDEX (repeatable) or --all. Before releasing, warn that the arm must be physically supported, because a released joint can drop under gravity, and require typing yes; --yes skips the prompt and simulated runs do not prompt. This operates the brake service, not the motor-mode service. Uses a Robot control connection with startup arm initialization turned off, so the tool also reaches an arm whose motor cannot enable; shutdown requests a stop. Real hardware is selected unless --simulated is supplied.", &["all", "joint", "side", "yes"], "action: status | release | engage");
    let action = args.positional(0, "");
    if !["status", "release", "engage"].contains(&action.as_str()) {
        return Err(invalid("action must be status, release, or engage"));
    }
    let joints = args
        .list("joint")
        .into_iter()
        .map(|value| {
            value
                .parse()
                .map_err(|_| invalid(format!("--joint expects a joint index, got {value:?}")))
        })
        .collect::<Result<Vec<usize>>>()?;
    let all = args.flag("all");
    if all && !joints.is_empty() {
        return Err(invalid("--all and --joint are mutually exclusive"));
    }
    // Releasing a brake lets the joint move freely: nothing about it may be
    // implied. Both the arm and the joints have to be named.
    let sides = if action == "release" {
        if !all && joints.is_empty() {
            return Err(invalid(
                "release requires --joint INDEX (repeatable) or --all; there is no default",
            ));
        }
        args.explicit_sides()?
    } else {
        args.sides()?
    };
    // `None` is the brake service's "every joint". It is only ever passed on
    // purpose (--all, or the status/engage default); an empty index list is
    // never sent, so it cannot be read as "every joint" by accident.
    let selected = (!joints.is_empty()).then_some(joints.as_slice());
    if action == "release" {
        let arms = sides
            .iter()
            .map(|side| component_name(side, "arm"))
            .collect::<Vec<_>>()
            .join(" and ");
        let target = match selected {
            Some(joints) => format!("joints {joints:?}"),
            None => "ALL joints".to_owned(),
        };
        args.confirm_hazard(&format!(
            "releasing the brakes of {target} on {arms}. A released joint is no longer held \
             and the arm can drop under gravity. Physically support the arm and keep people \
             clear before continuing."
        ))?;
    }
    let robot = args
        .connect_with(&[], |options| options.skip_arm_initialization = true)
        .await?;
    dexcontrol_examples::run_robot(&robot, async {
        for side in sides {
            let arm = robot.joints(&component_name(side, "arm"))?;
            let result = if action == "status" {
                arm.get_brake_status()?
            } else {
                match selected {
                    Some(joints) => arm.release_brake(action == "release", joints)?,
                    None => arm.release_all_brakes(action == "release")?,
                }
            };
            println!(
                "{side}_arm\n{}",
                serde_json::to_string_pretty(&serde_json::from_str::<serde_json::Value>(&result)?)?
            );
        }
        Ok(())
    })
    .await
}

// Copyright (C) 2026 Dexmate Inc.
// SPDX-License-Identifier: Apache-2.0

//! Monitor joint currents for components that report per-joint current.
//!
//! Probe all components by default, skip unsupported current readings, then poll continuously at 0.02 s intervals until Ctrl-C; --samples N stops after N polls. Fail if none report current; --component limits the selection.
//!
//! Prints numeric summaries/frame metadata; does not open a viewer.
//!
//! Uses a Robot control connection: compatible arm modes may be enabled during startup, and shutdown requests a stop. Real hardware is selected unless --simulated is supplied.

use dexcontrol::RateLimiter;
use dexcontrol_examples::{invalid, Args, Result};

#[tokio::main]
async fn main() -> Result {
    let args = Args::parse(
        "Monitor joint currents for components that report per-joint current. Probe all components by default, skip unsupported current readings, then poll continuously at 0.02 s intervals until Ctrl-C; --samples N stops after N polls. Fail if none report current; --component limits the selection. Prints numeric summaries/frame metadata; does not open a viewer. Uses a Robot control connection: compatible arm modes may be enabled during startup, and shutdown requests a stop. Real hardware is selected unless --simulated is supplied.",
        &["component", "period", "samples"],
    );
    let samples = args.integer("samples", 0)?; // 0: until Ctrl-C
    let period = args.number("period", 0.02)?;
    let robot = args.connect(&[]).await?;
    dexcontrol_examples::run_robot(&robot, async {
        let mut names = args.list("component");
        if names.is_empty() {
            names = robot.component_names()?;
        }
        let mut handles = Vec::new();
        let mut skipped = Vec::new();
        for name in names {
            match robot.joints(&name).and_then(|handle| {
                handle
                    .get_joint_state(dexcontrol::DurationLimit::Configured)
                    .map(|state| (handle, state))
            }) {
                Ok((handle, state)) if state.current.len() == handle.joint_names()?.len() => {
                    handles.push((name, handle))
                }
                _ => skipped.push(name),
            }
        }
        if !skipped.is_empty() {
            println!("no current data, skipping: {}", skipped.join(", "));
        }
        if handles.is_empty() {
            return Err(invalid("no component on this robot reports joint current"));
        }
        let limiter = RateLimiter::new(1.0 / period, 10, false)?;
        let mut index = 0;
        while samples == 0 || index < samples {
            for (name, handle) in &handles {
                let state = handle.get_joint_state(dexcontrol::DurationLimit::Configured)?;
                let values = handle
                    .joint_names()?
                    .iter()
                    .zip(&state.current)
                    .map(|(joint, value)| format!("{joint}={value:.3}"))
                    .collect::<Vec<_>>()
                    .join(" ");
                println!("{name}: {values}");
            }
            index += 1;
            if samples == 0 || index < samples {
                limiter.sleep()?;
                tokio::task::yield_now().await;
            }
        }
        Ok(())
    })
    .await
}

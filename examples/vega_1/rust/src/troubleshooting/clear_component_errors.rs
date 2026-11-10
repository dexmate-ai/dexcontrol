// Copyright (C) 2026 Dexmate Inc.
// SPDX-License-Identifier: Apache-2.0

//! Clear errors on all supported robot components.
//!
//! Send clear-error requests, print each component outcome, and exit nonzero if any outcome failed. This is a mutating maintenance operation.
//!
//! Uses a Robot control connection: compatible arm modes may be enabled during startup, and shutdown requests a stop. Real hardware is selected unless --simulated is supplied.

use dexcontrol_examples::{Args, Result};

#[tokio::main]
async fn main() -> Result {
    let args = Args::parse("Clear errors on all supported robot components. Send clear-error requests, print each component outcome, and exit nonzero if any outcome failed. This is a mutating maintenance operation. Uses a Robot control connection with startup arm initialization turned off: an arm whose motor cannot enable is the usual reason to clear errors, and the default connection would fail on it before any request is sent. Shutdown requests a stop. Real hardware is selected unless --simulated is supplied.", &[]);
    let robot = args
        .connect_with(&[], |options| options.skip_arm_initialization = true)
        .await?;
    dexcontrol_examples::run_robot(&robot, async {
        let report = serde_json::from_str::<serde_json::Value>(&robot.clear_errors_json()?)?;
        if report["outcomes"].as_array().is_none_or(Vec::is_empty) {
            println!("No clearable components are enabled for this robot.");
        }
        let mut failed = false;
        for outcome in report["outcomes"].as_array().into_iter().flatten() {
            if outcome["success"].as_bool().unwrap_or(false) {
                println!("ok  {}", outcome["component"].as_str().unwrap_or("unknown"));
            } else {
                failed = true;
                println!(
                    "err {}: {}",
                    outcome["component"].as_str().unwrap_or("unknown"),
                    outcome["error"].as_str().unwrap_or("unknown error")
                );
            }
        }
        if failed {
            return Err(dexcontrol_examples::invalid(
                "some components could not be cleared",
            ));
        }
        Ok(())
    })
    .await
}

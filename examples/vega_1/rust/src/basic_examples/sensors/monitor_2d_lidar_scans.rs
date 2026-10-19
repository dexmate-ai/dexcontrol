// Copyright (C) 2026 Dexmate Inc.
//
// This software is dual-licensed:
//
// 1. GNU Affero General Public License v3.0 (AGPL-3.0)
//    See LICENSE-AGPL for details
//
// 2. Commercial License
//    For commercial licensing terms, contact: contact@dexmate.ai

//! Monitor live 2D LiDAR scans.
//!
//! Enable lidar_2d_front, wait up to 10 s for activity, and poll continuously at 0.1 s intervals until Ctrl+C. --samples N limits the number of polls. Display XY points or range summaries; no scan file is saved.
//!
//! Prints numeric summaries/frame metadata; does not open a viewer.
//!
//! Uses a Robot control connection: compatible arm modes may be enabled during startup, and shutdown requests a stop. Real hardware is selected unless --simulated is supplied.

use dexcontrol_examples::{pause, Args, Result};
use std::time::Duration;

#[tokio::main]
async fn main() -> Result {
    let args = Args::parse(
        "Monitor live 2D LiDAR scans. Enable lidar_2d_front, wait up to 10 s for activity, and poll continuously at 0.1 s intervals until Ctrl+C. --samples N limits the number of polls. Display XY points or range summaries; no scan file is saved. Prints numeric summaries/frame metadata; does not open a viewer. Uses a Robot control connection: compatible arm modes may be enabled during startup, and shutdown requests a stop. Real hardware is selected unless --simulated is supplied.",
        &["period", "samples", "sensor"],
    );
    let sensor = args.text("sensor", "lidar_2d_front");
    let samples = if args.has("samples") {
        Some(args.integer("samples", 0)?)
    } else {
        None
    };
    let period = args.duration("period", 0.1)?;
    let robot = args.connect(std::slice::from_ref(&sensor)).await?;
    dexcontrol_examples::run_robot(&robot, async {
        if !robot.wait_for_state(&sensor, Duration::from_secs(10))? {
            return Err(dexcontrol_examples::invalid(format!(
                "{sensor} did not become active"
            )));
        }
        let mut index = 0;
        while samples.is_none_or(|limit| index < limit) {
            let scan = robot.lidar_2d(&sensor, None)?;
            let finite = scan
                .ranges()?
                .iter()
                .copied()
                .filter(|value| value.is_finite())
                .collect::<Vec<_>>();
            let low = finite.iter().copied().reduce(f64::min).unwrap_or(f64::NAN);
            println!("sample={} points={} min={low:.3}", index + 1, finite.len());
            index += 1;
            if samples.is_none_or(|limit| index < limit) {
                pause(period).await;
            }
        }
        Ok(())
    })
    .await
}

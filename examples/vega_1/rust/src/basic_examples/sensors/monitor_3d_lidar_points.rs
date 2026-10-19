// Copyright (C) 2026 Dexmate Inc.
//
// This software is dual-licensed:
//
// 1. GNU Affero General Public License v3.0 (AGPL-3.0)
//    See LICENSE-AGPL for details
//
// 2. Commercial License
//    For commercial licensing terms, contact: contact@dexmate.ai

//! Monitor point clouds from one 3D LiDAR.
//!
//! Enable the front 3D LiDAR, wait up to 10 s for activity, and poll continuously at 0.1 s intervals until Ctrl+C. --samples N limits the number of polls. --position back selects the rear LiDAR; no cloud file is saved.
//!
//! Prints numeric summaries/frame metadata; does not open a viewer.
//!
//! Uses a Robot control connection: compatible arm modes may be enabled during startup, and shutdown requests a stop. Real hardware is selected unless --simulated is supplied.

use dexcontrol_examples::{pause, Args, Result};
use std::time::Duration;

#[tokio::main]
async fn main() -> Result {
    let args = Args::parse(
        "Monitor point clouds from one 3D LiDAR. Enable the front 3D LiDAR, wait up to 10 s for activity, and poll continuously at 0.1 s intervals until Ctrl+C. --samples N limits the number of polls. --position back selects the rear LiDAR; no cloud file is saved. Prints numeric summaries/frame metadata; does not open a viewer. Uses a Robot control connection: compatible arm modes may be enabled during startup, and shutdown requests a stop. Real hardware is selected unless --simulated is supplied.",
        &["period", "position", "samples"],
    );
    let position = args.text("position", "front");
    if position != "front" && position != "back" {
        return Err(dexcontrol_examples::invalid(
            "--position must be front or back",
        ));
    }
    let sensor = format!("lidar_3d_{position}");
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
            let cloud = robot.lidar_3d(&sensor, None)?;
            let range = cloud
                .z()?
                .iter()
                .copied()
                .reduce(f64::min)
                .zip(cloud.z()?.iter().copied().reduce(f64::max));
            if let Some((low, high)) = range {
                println!(
                    "sample={} points={} z={low:.3}..{high:.3} m",
                    index + 1,
                    cloud.z()?.len()
                );
            } else {
                println!("sample={} points=0", index + 1);
            }
            index += 1;
            if samples.is_none_or(|limit| index < limit) {
                pause(period).await;
            }
        }
        Ok(())
    })
    .await
}

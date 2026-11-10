// Copyright (C) 2026 Dexmate Inc.
// SPDX-License-Identifier: Apache-2.0

//! Read and display one 2D LiDAR scan.
//!
//! Enable lidar_2d_front, wait up to 10 s for activity, and read one cached scan. Python can visualize XY points; native examples print range statistics. No scan file is saved.
//!
//! Prints numeric summaries/frame metadata; does not open a viewer.
//!
//! Uses a Robot control connection: compatible arm modes may be enabled during startup, and shutdown requests a stop. Real hardware is selected unless --simulated is supplied.

use dexcontrol_examples::{Args, Result};
use std::time::Duration;

#[tokio::main]
async fn main() -> Result {
    let args = Args::parse(
        "Read and display one 2D LiDAR scan. Enable lidar_2d_front, wait up to 10 s for activity, and read one cached scan. Python can visualize XY points; native examples print range statistics. No scan file is saved. Prints numeric summaries/frame metadata; does not open a viewer. Uses a Robot control connection: compatible arm modes may be enabled during startup, and shutdown requests a stop. Real hardware is selected unless --simulated is supplied.",
        &["sensor"],
    );
    let sensor = args.text("sensor", "lidar_2d_front");
    let robot = args.connect(std::slice::from_ref(&sensor)).await?;
    dexcontrol_examples::run_robot(&robot, async {
        if !robot.wait_for_state(&sensor, Duration::from_secs(10))? {
            return Err(dexcontrol_examples::invalid(format!(
                "{sensor} did not become active"
            )));
        }
        let scan = robot.sensors().lidar_2d(&sensor, None)?;
        let finite = scan
            .ranges()?
            .iter()
            .copied()
            .filter(|value| value.is_finite())
            .collect::<Vec<_>>();
        let low = finite
            .iter()
            .copied()
            .reduce(f64::min)
            .ok_or_else(|| dexcontrol_examples::invalid("no finite returns in the scan"))?;
        let high = finite.iter().copied().reduce(f64::max).unwrap_or(low);
        println!(
            "points={} min={low:.3} max={high:.3} (angle {:.3}..{:.3} rad, range limit {:.3} m)",
            finite.len(),
            scan.info()?.angle_min,
            scan.info()?.angle_max,
            scan.info()?.range_max
        );
        Ok(())
    })
    .await
}

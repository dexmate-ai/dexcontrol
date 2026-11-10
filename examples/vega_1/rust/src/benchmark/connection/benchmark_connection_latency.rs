// Copyright (C) 2026 Dexmate Inc.
// SPDX-License-Identifier: Apache-2.0

//! Measure full Robot connection time and query round-trip latency.
//!
//! Create one control connection, then issue 10 version_info queries and print timing statistics. This includes normal Robot startup/cleanup side effects; it is not a read-only DiagnosticClient benchmark.
//!
//! Times builder completion; no separate --timeout/active-state wait.
//!
//! Uses a Robot control connection: compatible arm modes may be enabled during startup, and shutdown requests a stop. Real hardware is selected unless --simulated is supplied.

use dexcontrol_examples::{Args, Result};
use std::time::Instant;

#[tokio::main]
async fn main() -> Result {
    let args = Args::parse(
        "Measure full Robot connection time and query round-trip latency. Create one control connection, then issue 10 version_info queries and print timing statistics. This includes normal Robot startup/cleanup side effects; it is not a read-only DiagnosticClient benchmark. Times builder completion; no separate --timeout/active-state wait. Uses a Robot control connection: compatible arm modes may be enabled during startup, and shutdown requests a stop. Real hardware is selected unless --simulated is supplied.",
        &["samples"],
    );
    let samples = args.integer("samples", 10)?;
    let started = Instant::now();
    let robot = args.connect(&[]).await?;
    dexcontrol_examples::run_robot(&robot, async {
    let connection = started.elapsed().as_secs_f64();
    let mut values = Vec::new();
    for sample in 0..samples {
        let started = Instant::now();
        match robot.query("version_info", &[]) {
            Ok(_) => values.push(started.elapsed().as_secs_f64()),
            Err(error) => println!("query {} failed: {error}", sample + 1),
        }
    }
    println!(
        "Connect Robot Latency Benchmark\n  Init + discovery: {:.1} ms",
        connection * 1000.0
    );
    if !values.is_empty() {
        let mean = values.iter().sum::<f64>() / values.len() as f64;
        let min = values.iter().copied().reduce(f64::min).unwrap_or_default();
        let max = values.iter().copied().reduce(f64::max).unwrap_or_default();
        let variance = if values.len() > 1 {
            values
                .iter()
                .map(|value| (value - mean).powi(2))
                .sum::<f64>()
                / (values.len() - 1) as f64
        } else {
            0.0
        };
        println!("  Query RTT samples: {}\n  Mean: {:.1} ms\n  Min:  {:.1} ms\n  Max:  {:.1} ms\n  Std:  {:.1} ms", values.len(), mean * 1000.0, min * 1000.0, max * 1000.0, variance.sqrt() * 1000.0);
    }
        Ok(())
    }).await
}

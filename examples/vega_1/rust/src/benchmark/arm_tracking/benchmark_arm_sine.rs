// Copyright (C) 2026 Dexmate Inc.
// SPDX-License-Identifier: Apache-2.0

//! Benchmark each arm joint with a sine trajectory and velocity feed-forward.
//!
//! Prompt for confirmation unless --no-confirm; move the right seven-joint arm to [0,0,0,-0.5,0,0,0] rad. Test each joint for 10 s at 200 Hz with a 0.4 rad, 1 Hz sine; return to the reference between joints and at the end. Write measurement files. This reference is not an all-zero pose. The amplitude is fitted to each joint's limits: the position limits from the reference pose and the velocity limit against the feed-forward peak (amplitude times angular frequency), reduced with a warning where the requested amplitude does not fit. Stream with the step guard disabled (tracking lag is what is measured).
//!
//! Writes per-joint CSV and prints tracking errors; no plots/NPZ. --timeout controls managed-motion waits.
//!
//! Uses a Robot control connection: compatible arm modes may be enabled during startup, and shutdown requests a stop. Real hardware is selected unless --simulated is supplied.

use dexcontrol::{CommandOptions, RateLimiter, Wait};
use dexcontrol_examples::{component_name, Args, Result};
use std::time::Instant;

#[tokio::main]
async fn main() -> Result {
    let args = Args::parse(
        "Benchmark each arm joint with a sine trajectory and velocity feed-forward. Prompt for confirmation unless --no-confirm; move the right seven-joint arm to [0,0,0,-0.5,0,0,0] rad. Test each joint for 10 s at 200 Hz with a 0.4 rad, 1 Hz sine; return to the reference between joints and at the end. Write measurement files. This reference is not an all-zero pose. The amplitude is fitted to each joint's limits: the position limits from the reference pose and the velocity limit against the feed-forward peak (amplitude times angular frequency), reduced with a warning where the requested amplitude does not fit. Stream with the step guard disabled (tracking lag is what is measured). Writes per-joint CSV and prints tracking errors; no plots/NPZ. --timeout controls managed-motion waits. Uses a Robot control connection: compatible arm modes may be enabled during startup, and shutdown requests a stop. Real hardware is selected unless --simulated is supplied.",
        &[
            "amplitude",
            "control-hz",
            "duration",
            "no-confirm",
            "output-dir",
            "settle",
            "side",
            "sin-frequency",
            "timeout",
        ],
    );
    let duration = args.duration("duration", 10.0)?;
    let hz = args.number("control-hz", 200.0)?;
    let requested_amplitude = args.number("amplitude", 0.4)?;
    let omega = 2.0 * std::f64::consts::PI * args.number("sin-frequency", 1.0)?;
    if dexcontrol_examples::sample_count(duration.as_secs_f64(), hz)? == 0 {
        return Err(dexcontrol_examples::invalid(
            "benchmark requires at least one sample",
        ));
    }
    args.confirm_motion()?;
    let robot = args.connect(&[]).await?;
    dexcontrol_examples::run_robot(&robot, async {
        let arm = robot.joints(&component_name(args.side()?, "arm"))?;
        if arm.joint_names()?.len() != 7 {
            return Err(dexcontrol_examples::invalid(
                "benchmark requires seven joints",
            ));
        }
        let zero = vec![0.0, 0.0, 0.0, -0.5, 0.0, 0.0, 0.0];
        arm.move_to_joint_pos_with(
            &zero,
            dexcontrol::MotionOptions::default(),
            Wait::Timeout(args.duration("timeout", 5.0)?).into(),
        )?;
        dexcontrol_examples::pause(args.duration("settle", 0.5)?).await;
        for joint in 0..zero.len() {
            let mut csv = dexcontrol_examples::benchmark_csv(&args, "sin", joint)?;
            if joint > 0 {
                arm.move_to_joint_pos_with(
                    &zero,
                    dexcontrol::MotionOptions::default(),
                    Wait::Timeout(args.duration("timeout", 3.0)?).into(),
                )?;
            }
            // Fitted to the joint's position and velocity limits from the
            // reference pose (reduced with a warning where it does not fit).
            let amplitude = dexcontrol_examples::fit_amplitude(
                &arm,
                joint,
                zero[joint],
                requested_amplitude,
                true,
                omega,
            )?;
            let limiter = RateLimiter::new(hz, 10, false)?;
            let started = Instant::now();
            let mut total = 0.0;
            let mut peak = 0.0_f64;
            let mut samples = 0usize;
            while started.elapsed() < duration {
                let elapsed = started.elapsed().as_secs_f64();
                let mut target = zero.clone();
                target[joint] += amplitude * (omega * elapsed).sin();
                let mut velocity = vec![0.0; zero.len()];
                velocity[joint] = amplitude * omega * (omega * elapsed).cos();
                // Disable the step guard explicitly: tracking lag is what this benchmark measures.
                let options = CommandOptions {
                    max_step: dexcontrol::StepLimit::Unlimited,
                    ..CommandOptions::default()
                };
                arm.set_joint_pos_with(&target, Some(&velocity), options, Wait::NoWait.into())?;
                let actual = arm.get_joint_pos()?;
                dexcontrol_examples::record_sample(&mut csv, elapsed, &target, &actual)?;
                let error = (target[joint] - actual[joint]).abs();
                total += error;
                peak = peak.max(error);
                samples += 1;
                limiter.sleep()?;
                tokio::task::yield_now().await;
            }
            std::io::Write::flush(&mut csv)?;
            println!(
                "joint {joint}: samples={samples} mean|err|={:.4} rad max|err|={peak:.4} rad",
                total / samples.max(1) as f64
            );
        }
        arm.move_to_joint_pos_with(
            &zero,
            dexcontrol::MotionOptions::default(),
            Wait::Timeout(args.duration("timeout", 5.0)?).into(),
        )?;
        Ok(())
    })
    .await
}

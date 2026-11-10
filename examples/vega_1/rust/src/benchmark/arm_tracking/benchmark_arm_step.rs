// Copyright (C) 2026 Dexmate Inc.
// SPDX-License-Identifier: Apache-2.0

//! Benchmark each arm joint with a ramped step trajectory.
//!
//! Prompt unless --no-confirm; move the right seven-joint arm to [0,0,0,-0.5,0,0,0] rad. Test each joint for 5 s at 200 Hz: hold 0.3 s, ramp by pi/6 rad over 0.3 s, then hold. The step is fitted to each joint's limits from the reference pose: the other direction where the requested one does not fit, a smaller step where neither does, each logged. Stream positions without velocity feed-forward and with the step guard disabled (tracking lag is what is measured); return to the reference between joints and at the end. Write measurement files.
//!
//! Writes per-joint CSV and prints tracking errors; returns to the reference after each joint, with no plots/NPZ.
//!
//! Uses a Robot control connection: compatible arm modes may be enabled during startup, and shutdown requests a stop. Real hardware is selected unless --simulated is supplied.

use dexcontrol::{CommandOptions, RateLimiter, Wait};
use dexcontrol_examples::{component_name, Args, Result};

#[tokio::main]
async fn main() -> Result {
    let args = Args::parse(
        "Benchmark each arm joint with a ramped step trajectory. Prompt unless --no-confirm; move the right seven-joint arm to [0,0,0,-0.5,0,0,0] rad. Test each joint for 5 s at 200 Hz: hold 0.3 s, ramp by pi/6 rad over 0.3 s, then hold. The step is fitted to each joint's limits from the reference pose: the other direction where the requested one does not fit, a smaller step where neither does, each logged. Stream positions without velocity feed-forward and with the step guard disabled (tracking lag is what is measured); return to the reference between joints and at the end. Write measurement files. Writes per-joint CSV and prints tracking errors; returns to the reference after each joint, with no plots/NPZ. Uses a Robot control connection: compatible arm modes may be enabled during startup, and shutdown requests a stop. Real hardware is selected unless --simulated is supplied.",
        &[
            "control-hz",
            "duration",
            "max-vel",
            "no-confirm",
            "output-dir",
            "settle",
            "side",
            "step-amplitude",
            "step-time",
            "timeout",
            "transition-time",
        ],
    );
    let hz = args.number("control-hz", 200.0)?;
    let samples = dexcontrol_examples::sample_count(args.number("duration", 5.0)?, hz)?;
    if samples == 0 {
        return Err(dexcontrol_examples::invalid(
            "benchmark requires at least one sample",
        ));
    }
    let before = dexcontrol_examples::sample_count(args.number("step-time", 0.3)?, hz)?;
    let requested_amplitude = args.number("step-amplitude", std::f64::consts::PI / 6.0)?;
    let transition_time = args.number("transition-time", 0.3)?;
    if transition_time <= 0.0 {
        return Err(dexcontrol_examples::invalid(
            "transition time must be positive",
        ));
    }
    let max_vel = args.number("max-vel", requested_amplitude.abs() / transition_time)?;
    if max_vel <= 0.0 {
        return Err(dexcontrol_examples::invalid(
            "max velocity must be positive",
        ));
    }
    // Validates the requested ramp; each joint's fitted amplitude is at most
    // this large, so its ramp is at most this long.
    dexcontrol_examples::sample_count(requested_amplitude.abs() / max_vel, hz)?;
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
            let mut csv = dexcontrol_examples::benchmark_csv(&args, "step", joint)?;
            // The step is fitted to the joint's limits from the reference
            // pose (flipped or reduced with a warning where it does not fit).
            let amplitude = dexcontrol_examples::fit_amplitude(
                &arm,
                joint,
                zero[joint],
                requested_amplitude,
                false,
                0.0,
            )?;
            let transition =
                dexcontrol_examples::sample_count(amplitude.abs() / max_vel, hz)?.max(1);
            let limiter = RateLimiter::new(hz, 10, false)?;
            let mut total = 0.0;
            let mut peak = 0.0_f64;
            let started = std::time::Instant::now();
            for sample in 0..samples {
                let fraction = if sample < before {
                    0.0
                } else {
                    (sample - before + 1).min(transition) as f64 / transition as f64
                };
                let mut target = zero.clone();
                target[joint] += amplitude * fraction;
                // Disable the step guard explicitly: tracking lag is what this benchmark measures.
                arm.set_joint_pos_with(
                    &target,
                    None,
                    CommandOptions {
                        max_step: dexcontrol::StepLimit::Unlimited,
                        ..Default::default()
                    },
                    Wait::NoWait.into(),
                )?;
                let actual = arm.get_joint_pos()?;
                dexcontrol_examples::record_sample(
                    &mut csv,
                    started.elapsed().as_secs_f64(),
                    &target,
                    &actual,
                )?;
                let error = (target[joint] - actual[joint]).abs();
                total += error;
                peak = peak.max(error);
                limiter.sleep()?;
                tokio::task::yield_now().await;
            }
            std::io::Write::flush(&mut csv)?;
            println!(
                "joint {joint}: samples={samples} mean|err|={:.4} rad max|err|={peak:.4} rad",
                total / samples.max(1) as f64
            );
            arm.move_to_joint_pos_with(
                &zero,
                dexcontrol::MotionOptions::default(),
                Wait::Timeout(args.duration("timeout", 5.0)?).into(),
            )?;
        }
        Ok(())
    })
    .await
}

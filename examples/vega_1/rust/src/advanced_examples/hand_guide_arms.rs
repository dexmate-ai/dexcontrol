// Copyright (C) 2026 Dexmate Inc.
// SPDX-License-Identifier: Apache-2.0

//! Guide both arms by hand: yield to pushes sensed from joint current.
//!
//! Capture each arm's joint positions and currents at rest, then at 200 Hz treat a per-joint current offset beyond --threshold A (default 0.25) as a push and send a joint setpoint that moves with it: each joint's velocity is the push divided by a damping of 2*gain (--gain, default 0.3; larger is stiffer, below 0.3 is refused), and no tick commands a setpoint more than 0.2 rad from the measured position. By default the arms stay wherever they are pushed; --hold-pose adds a spring back to the captured pose. Run until Ctrl-C or --duration seconds (default 0, unlimited), then stop both arms. No force sensor, no model and no identification: the current threshold is the only filter, so run it on a machine wired to the robot.
//!
//! Uses a Robot control connection: compatible arm modes may be enabled during startup, and shutdown requests a stop. Real hardware is selected unless --simulated is supplied.

use dexcontrol::{DurationLimit, RateLimiter, Wait};
use dexcontrol_examples::{invalid, Args, Result};

/// Per-joint spring pattern of the 7-joint Vega arm, scaled by --gain when
/// --hold-pose is set: the shoulder and elbow pitch joints carry the load.
const SPRING: [f64; 7] = [2.0, 0.25, 0.25, 2.0, 0.25, 0.25, 0.25];
/// Largest move one tick may command, rad. A direct setpoint is checked
/// against a fresh measurement with a 0.3 rad step guard; a slow loop or a
/// hard shove must slow the arm down, not end the example.
const MAX_STEP: f64 = 0.2;

#[tokio::main]
async fn main() -> Result {
    let args = Args::parse(
        "Guide both arms by hand: yield to pushes sensed from joint current. Capture each arm's joint positions and currents at rest, then at 200 Hz treat a per-joint current offset beyond --threshold A (default 0.25) as a push and send a joint setpoint that moves with it: each joint's velocity is the push divided by a damping of 2*gain (--gain, default 0.3; larger is stiffer, below 0.3 is refused), and no tick commands a setpoint more than 0.2 rad from the measured position. By default the arms stay wherever they are pushed; --hold-pose adds a spring back to the captured pose. Run until Ctrl-C or --duration seconds (default 0, unlimited), then stop both arms. No force sensor, no model and no identification: the current threshold is the only filter, so run it on a machine wired to the robot. Uses a Robot control connection: compatible arm modes may be enabled during startup, and shutdown requests a stop. Real hardware is selected unless --simulated is supplied.",
        &["control-hz", "duration", "gain", "hold-pose", "threshold"],
    );
    let gain = args.number("gain", 0.3)?;
    let threshold = args.number("threshold", 0.25)?;
    let control_hz = args.number("control-hz", 200.0)?;
    let duration = args.number("duration", 0.0)?;
    let hold_pose = args.flag("hold-pose");
    if gain < 0.3 {
        return Err(invalid(
            "--gain below 0.3 lets a current blip fling the arm; refusing",
        ));
    }
    if threshold < 0.0 {
        return Err(invalid("--threshold must not be negative"));
    }
    if control_hz <= 0.0 {
        return Err(invalid("--control-hz must be positive"));
    }
    let robot = args.connect(&[]).await?;
    dexcontrol_examples::run_robot(&robot, async {
        let mut arms = Vec::new();
        for name in ["left_arm", "right_arm"] {
            if !robot.has_component(name)? {
                return Err(invalid(format!("this robot has no {name}")));
            }
            let component = robot.joints(name)?;
            let joints = component.joint_names()?.len();
            if joints != SPRING.len() {
                return Err(invalid(format!(
                    "hand guiding is tuned for the {}-joint Vega arm; {name} has {joints}",
                    SPRING.len()
                )));
            }
            // A rest baseline: everything the joints draw to hold still here
            // is not a push.
            let state = component.get_joint_state(DurationLimit::Configured)?;
            if state.current.len() != SPRING.len() {
                return Err(invalid(format!("{name} reports no per-joint current")));
            }
            arms.push((name, component, state.position, state.current));
        }
        let damping = 2.0 * gain;
        let dt = 1.0 / control_hz;
        println!(
            "hand guiding both arms at {control_hz} Hz; threshold {threshold} A; {}",
            if hold_pose {
                "spring back to the start pose"
            } else {
                "float where pushed"
            }
        );
        let limiter = RateLimiter::new(control_hz, 10, false)?;
        let outcome: Result = async {
            let mut ticks = 0u64;
            while duration <= 0.0 || ticks as f64 * dt < duration {
                for (_, component, rest_pos, rest_current) in &arms {
                    let state = component.get_joint_state(DurationLimit::Configured)?;
                    let target: Vec<f64> = (0..SPRING.len())
                        .map(|joint| {
                            let offset = state.current[joint] - rest_current[joint];
                            // Below the threshold is noise and the load of
                            // holding still; above it, the sign is inverted so
                            // the joint gives way to the push.
                            let push = if offset.abs() < threshold {
                                0.0
                            } else {
                                -offset
                            };
                            let spring = if hold_pose { SPRING[joint] * gain } else { 0.0 };
                            let velocity = (spring * (rest_pos[joint] - state.position[joint])
                                + push)
                                / damping;
                            state.position[joint] + (velocity * dt).clamp(-MAX_STEP, MAX_STEP)
                        })
                        .collect();
                    // The direct setpoint path, as the 0.5 script used: the
                    // motion plugin's own smoothing would lag the target and
                    // its catch-up current reads as a new push.
                    component.set_joint_pos(&target, Wait::NoWait)?;
                }
                ticks += 1;
                limiter.sleep()?;
                tokio::task::yield_now().await;
            }
            Ok(())
        }
        .await;
        let mut stopped = Ok(());
        for (_, component, _, _) in &arms {
            if let Err(error) = component.stop() {
                stopped = Err(error);
            }
        }
        outcome?;
        stopped?;
        Ok(())
    })
    .await
}

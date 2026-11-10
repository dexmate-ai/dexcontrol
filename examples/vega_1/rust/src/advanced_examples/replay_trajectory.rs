// Copyright (C) 2026 Dexmate Inc.
// SPDX-License-Identifier: Apache-2.0

//! Prepare robot joint positions and replay a recorded trajectory.
//!
//! Require a trajectory file and confirmation before motion. Move to preparation/start targets, then stream recorded joint positions with the configured tracking-error guard. Preparation motions and processing options differ by language; see the language notes.
//!
//! CSV input. Confirm unless --no-confirm (simulation bypasses the prompt); move directly to the recorded first row, then stream. Does not fold/crouch/close hands or smooth/resample/add velocity feed-forward. Rate from file or 500 Hz.
//!
//! Uses a Robot control connection: compatible arm modes may be enabled during startup, and shutdown requests a stop. Real hardware is selected unless --simulated is supplied.

use dexcontrol::{MotionOptions, TrajectoryOptions, TrajectoryTrack};
use dexcontrol_examples::{invalid, Args, Result};
use std::collections::BTreeMap;

type Trajectory = BTreeMap<String, Vec<Vec<f64>>>;

fn load(path: &str) -> Result<(f64, Trajectory)> {
    load_csv(&std::fs::read_to_string(path)?)
}

fn load_csv(text: &str) -> Result<(f64, Trajectory)> {
    let mut lines = text.lines();
    let first = lines
        .next()
        .ok_or_else(|| invalid("empty trajectory file"))?;
    let (hz, header) = if let Some(value) = first.strip_prefix("control_hz,") {
        (
            value.parse()?,
            lines.next().ok_or_else(|| invalid("missing CSV header"))?,
        )
    } else {
        (500.0, first)
    };
    let columns = header
        .split(',')
        .map(|column| {
            let (part, joint) = column
                .split_once(':')
                .ok_or_else(|| invalid(format!("bad header column {column:?}")))?;
            Ok((part.to_owned(), joint.parse::<usize>()?))
        })
        .collect::<Result<Vec<_>>>()?;
    if !f64::is_finite(hz) || hz <= 0.0 {
        return Err(invalid("control rate must be finite and positive"));
    }
    let mut seen = std::collections::BTreeSet::new();
    for (part, joint) in &columns {
        if part.is_empty() || *joint >= 1024 || !seen.insert((part.clone(), *joint)) {
            return Err(invalid("invalid or duplicate trajectory column"));
        }
    }
    let mut widths = BTreeMap::new();
    for (part, joint) in &columns {
        widths
            .entry(part.clone())
            .and_modify(|width: &mut usize| *width = (*width).max(joint + 1))
            .or_insert(joint + 1);
    }
    for (part, width) in &widths {
        if (0..*width).any(|j| !seen.contains(&(part.clone(), j))) {
            return Err(invalid(
                "trajectory joint indices must be contiguous starting at zero",
            ));
        }
    }
    let mut tracks = widths
        .keys()
        .map(|part| (part.clone(), Vec::new()))
        .collect::<BTreeMap<_, _>>();
    for line in lines.filter(|line| !line.trim().is_empty()) {
        let cells = line
            .split(',')
            .map(str::parse::<f64>)
            .collect::<std::result::Result<Vec<_>, _>>()?;
        if cells.len() != columns.len() || cells.iter().any(|v| !v.is_finite()) {
            return Err(invalid("trajectory row has the wrong number of columns"));
        }
        let mut tick = widths
            .iter()
            .map(|(part, width)| (part.clone(), vec![0.0; *width]))
            .collect::<BTreeMap<_, _>>();
        for ((part, joint), value) in columns.iter().zip(cells) {
            tick.get_mut(part).expect("known part")[*joint] = value;
        }
        for (part, values) in tick {
            tracks.get_mut(&part).expect("known part").push(values);
        }
    }
    if tracks.values().any(Vec::is_empty) {
        return Err(invalid("trajectory has no samples"));
    }
    Ok((hz, tracks))
}

#[tokio::main]
async fn main() -> Result {
    let args = Args::parse_with_positionals(
        "Prepare robot joint positions and replay a recorded trajectory. Require a trajectory file and confirmation before motion. Move to preparation/start targets, then stream recorded joint positions with the configured tracking-error guard. Preparation motions and processing options differ by language; see the language notes. CSV input. Confirm unless --no-confirm (simulation bypasses the prompt); move directly to the recorded first row, then stream. Does not fold/crouch/close hands or smooth/resample/add velocity feed-forward. Rate from file or 500 Hz. Uses a Robot control connection: compatible arm modes may be enabled during startup, and shutdown requests a stop. Real hardware is selected unless --simulated is supplied.",
        &["control-hz", "max-goal-diff", "no-confirm", "timeout"], "path: trajectory file to replay");
    let path = args.positional(0, "");
    if path.is_empty() {
        return Err(invalid("a trajectory CSV file is required"));
    }
    let (file_hz, tracks) = load(&path)?;
    let hz = args.number("control-hz", file_hz)?;
    if hz <= 0.0 {
        return Err(invalid("control rate must be positive"));
    }
    let max_tracking_error = args.number("max-goal-diff", 1.0)?;
    if max_tracking_error < 0.0 {
        return Err(invalid("max-goal-diff must be non-negative"));
    }
    args.confirm_motion()?;
    let robot = args.connect(&[]).await?;
    dexcontrol_examples::run_robot(&robot, async {
        let names = robot.component_names()?;
        let available = tracks
            .into_iter()
            .filter(|(part, _)| names.contains(part))
            .collect::<BTreeMap<_, _>>();
        if available.is_empty() {
            return Err(invalid("trajectory contains no available components"));
        }
        let starts = available
            .iter()
            .map(|(part, rows)| (part.clone(), rows[0].clone()))
            .collect::<BTreeMap<_, _>>();
        let group = robot.move_to_joint_pos(&starts, MotionOptions::default())?;
        group.wait(args.duration("timeout", 3.0)?)?;
        let tracks = available
            .into_iter()
            .map(|(component, positions)| {
                (
                    component,
                    TrajectoryTrack {
                        positions,
                        velocities: None,
                    },
                )
            })
            .collect();
        robot.execute_trajectory(
            &tracks,
            hz,
            TrajectoryOptions {
                max_tracking_error: Some(max_tracking_error),
                ..Default::default()
            },
        )?;
        Ok(())
    })
    .await
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn rejects_empty_nonfinite_duplicate_and_missing_joints() {
        for csv in [
            "",
            "head:0\n",
            "head:0\nNaN\n",
            "head:0,head:0\n0,1\n",
            "head:1\n0\n",
            "head:18446744073709551615\n0\n",
            "control_hz,0\nhead:0\n0\n",
        ] {
            assert!(
                load_csv(csv).is_err(),
                "accepted invalid recording: {csv:?}"
            );
        }
    }
    #[test]
    fn preserves_track_and_joint_order() {
        let (hz, tracks) =
            load_csv("control_hz,100\nhead:1,arm:0,head:0\n0.2,0.3,0.1\n0.4,0.5,0.6\n").unwrap();
        assert_eq!(hz, 100.);
        assert_eq!(tracks["head"], vec![vec![0.1, 0.2], vec![0.6, 0.4]]);
        assert_eq!(tracks["arm"], vec![vec![0.3], vec![0.5]]);
    }
}

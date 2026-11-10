// Copyright (C) 2026 Dexmate Inc.
// SPDX-License-Identifier: Apache-2.0

//! Human-readable read-only diagnostics, independent of transport and hardware.
use serde_json::Value;
use std::collections::BTreeSet;
use std::fmt::Write;

fn text(v: &Value) -> String {
    let raw = match v {
        Value::Null => "—".into(),
        Value::String(s) => s.clone(),
        _ => v.to_string(),
    };
    raw.chars()
        .map(|c| if c.is_control() { ' ' } else { c })
        .collect()
}
fn items(v: &Value) -> &[Value] {
    v.as_array().map(Vec::as_slice).unwrap_or(&[])
}
fn members(v: &Value) -> impl Iterator<Item = (&String, &Value)> {
    v.as_object().into_iter().flatten()
}
fn yes(v: &Value) -> bool {
    v.as_bool() == Some(true)
}
fn human(v: &Value) -> String {
    text(v).replace('_', " ")
}
fn num(v: &Value, precision: usize, unit: &str) -> String {
    v.as_f64()
        .filter(|v| v.is_finite())
        .map(|v| format!("{v:.precision$}{unit}"))
        .unwrap_or_else(|| "—".into())
}
fn join(v: &Value, empty: &str) -> String {
    let values = items(v).iter().map(text).collect::<Vec<_>>();
    if values.is_empty() {
        empty.into()
    } else {
        values.join(", ")
    }
}
fn table(out: &mut String, title: &str, headers: &[&str], rows: Vec<Vec<String>>) {
    let mut widths = headers
        .iter()
        .map(|s| s.chars().count())
        .collect::<Vec<_>>();
    for row in &rows {
        for (i, cell) in row.iter().enumerate() {
            widths[i] = widths[i].max(cell.chars().count());
        }
    }
    writeln!(out, "\n{title}").unwrap();
    let rule = widths.iter().map(|w| "─".repeat(*w)).collect::<Vec<_>>();
    let headings = headers.iter().map(|s| s.to_string()).collect::<Vec<_>>();
    for row in std::iter::once(headings)
        .chain(std::iter::once(rule))
        .chain(rows)
    {
        for (i, cell) in row.iter().enumerate() {
            out.push_str(cell);
            if i + 1 < row.len() {
                out.push_str(&" ".repeat(widths[i].saturating_sub(cell.chars().count()) + 2));
            }
        }
        out.push('\n');
    }
}

pub fn render(s: &Value) -> String {
    let mut out = String::new();
    table(
        &mut out,
        "Robot overview",
        &["Field", "Value"],
        vec![
            vec!["Profile".into(), text(&s["profile_name"])],
            vec!["Transport".into(), "DexComm".into()],
            vec!["Components".into(), join(&s["component_names"], "None")],
            vec!["Sensors".into(), join(&s["sensor_names"], "None enabled")],
            vec!["End effectors".into(), end_effectors(s)],
        ],
    );
    let components = members(&s["joint_names"])
        .chain(members(&s["joint_states"]))
        .map(|(k, _)| k)
        .collect::<BTreeSet<_>>();
    let mut rows = Vec::new();
    for component in components {
        let state = &s["joint_states"][component];
        let names = items(&s["joint_names"][component]);
        let count = names
            .len()
            .max(items(&state["position"]).len())
            .max(usize::from(state.is_null()));
        for i in 0..count {
            let joint = names
                .get(i)
                .map(text)
                .unwrap_or_else(|| format!("joint_{}", i + 1));
            let status = if state.is_null() {
                let reason = &s["errors"][format!("joint_state:{component}")];
                format!(
                    "No data ({})",
                    reason.as_str().unwrap_or("no state received")
                )
            } else {
                let error = state["errors"]
                    .get(&joint)
                    .or_else(|| state["errors"].get(i.to_string()));
                if let Some(error) = error.filter(|v| !v.is_null() && v.as_str() != Some("")) {
                    text(error)
                } else if items(&state["error_joint_indices"])
                    .iter()
                    .any(|v| v.as_u64() == Some(i as u64))
                {
                    "Error".into()
                } else {
                    "OK".into()
                }
            };
            let mut row = vec![component.clone(), joint];
            for field in ["position", "velocity", "current", "torque"] {
                row.push(if state.is_null() {
                    "N/A".into()
                } else {
                    num(&state[field][i], 4, "")
                });
            }
            row.push(status);
            rows.push(row);
        }
    }
    table(
        &mut out,
        "Joint state",
        &[
            "Component",
            "Joint",
            "Position (rad)",
            "Velocity (rad/s)",
            "Current (A)",
            "Torque (N·m)",
            "Status",
        ],
        rows,
    );
    monitoring(&mut out, &s["monitoring"]);
    let info = &s["version_info"];
    let rows = members(&info["firmware_version"])
        .map(|(board, v)| {
            vec![
                board.clone(),
                text(&v["hardware_version"]),
                text(&v["software_version"]),
                text(&v["release_version"]),
                text(&v["main_hash"]),
                text(&v["compile_time"]),
            ]
        })
        .collect();
    table(
        &mut out,
        "Firmware versions",
        &[
            "Board", "Hardware", "Software", "Release", "Commit", "Compiled",
        ],
        rows,
    );
    writeln!(
        out,
        "Robot server {}  •  minimum client {}",
        text(&info["version"]),
        text(&info["min_client_version"])
    )
    .unwrap();
    let configured = items(&s["component_names"])
        .iter()
        .chain(items(&s["sensor_names"]))
        .map(text)
        .collect::<BTreeSet<_>>();
    let mut rows = members(&s["component_status"]["states"])
        .map(|(name, v)| {
            vec![
                name.clone(),
                human(&v["connection"]),
                human(&v["operation"]),
                text(&v["error"]["error_message"]),
                if configured.contains(name) {
                    "Configured".into()
                } else {
                    "Server only".into()
                },
            ]
        })
        .collect::<Vec<_>>();
    if rows.is_empty() {
        rows.push(vec![
            "—".into(),
            "Unavailable".into(),
            "—".into(),
            "—".into(),
            "—".into(),
        ]);
    }
    table(
        &mut out,
        "Component status",
        &["Component", "Connection", "Operation", "Error", "Scope"],
        rows,
    );
    let ntp = &s["ntp"];
    let mut rows = vec![vec![
        "Status".into(),
        if yes(&ntp["success"]) {
            "Measured".into()
        } else {
            "Unavailable".into()
        },
    ]];
    if yes(&ntp["success"]) {
        for (key, label) in [("offset", "Clock offset"), ("rtt", "Round-trip time")] {
            rows.push(vec![
                label.into(),
                ntp[key]
                    .as_f64()
                    .map(|v| format!("{:.3} ms", v * 1000.0))
                    .unwrap_or_else(|| "—".into()),
            ]);
        }
        rows.push(vec![
            "Samples".into(),
            format!(
                "{} / {} replies",
                text(&ntp["replies"]),
                text(&ntp["samples"])
            ),
        ]);
    }
    if let Some(message) = ntp["message"].as_str() {
        rows.push(vec!["Message".into(), text(&Value::from(message))]);
    }
    table(
        &mut out,
        "Clock synchronization",
        &["Metric", "Value"],
        rows,
    );
    let mut rows = items(&s["connection"]["phases"])
        .iter()
        .map(|v| vec![human(&v["phase"]), num(&v["duration_ms"], 2, " ms")])
        .collect::<Vec<_>>();
    rows.push(vec![
        "Total".into(),
        num(&s["connection"]["total_ms"], 2, " ms"),
    ]);
    table(&mut out, "Connection timing", &["Phase", "Duration"], rows);
    let rows = members(&s["errors"])
        .map(|(name, v)| vec![name.clone(), text(v)])
        .collect::<Vec<_>>();
    if !rows.is_empty() {
        table(
            &mut out,
            "Unavailable diagnostics",
            &["Section", "Reason"],
            rows,
        );
    }
    out
}

fn monitoring(out: &mut String, m: &Value) {
    let mut rows = Vec::new();
    let e = &m["estop"];
    if e.is_object() {
        let sources = [
            ("software_estop_enabled", "software"),
            ("left_base_estop_enabled", "left base hardware"),
            ("right_base_estop_enabled", "right base hardware"),
            ("torso_estop_enabled", "torso hardware"),
            ("remote_estop_enabled", "remote"),
        ]
        .into_iter()
        .filter(|(key, _)| yes(&e[*key]))
        .map(|(_, label)| label)
        .collect::<Vec<_>>();
        let (status, details) = if e["source"] == "no_event_published" {
            if yes(&e["state_known"]) {
                (
                    "Ready",
                    "not engaged (no event published; board connected)".into(),
                )
            } else {
                (
                    "Unknown",
                    if e["board_connected"] == false {
                        "no event published; board disconnected".into()
                    } else {
                        "no event published; board status unavailable".into()
                    },
                )
            }
        } else if !sources.is_empty() {
            ("STOPPED", format!("Active sources: {}", sources.join(", ")))
        } else {
            ("Clear", "No active sources reported".into())
        };
        rows.push(vec!["E-stop".into(), status.into(), details]);
    }
    let h = &m["heartbeat"];
    if h.is_object() {
        rows.push(vec![
            "Heartbeat".into(),
            if yes(&h["monitoring_disabled"]) {
                "Disabled"
            } else if yes(&h["paused"]) {
                "Paused"
            } else if yes(&h["is_active"]) {
                "Active"
            } else {
                "No sample"
            }
            .into(),
            "Native supervision".into(),
        ]);
    }
    let b = &m["battery"];
    if b.is_object() {
        let details = [
            ("voltage", " V"),
            ("current", " A"),
            ("power", " W"),
            ("temperature", " °C"),
        ]
        .into_iter()
        .filter(|(key, _)| b[*key].is_number())
        .map(|(key, unit)| num(&b[key], 2, unit))
        .collect::<Vec<_>>()
        .join(", ");
        rows.push(vec![
            "Battery".into(),
            num(&b["percentage"], 1, "%"),
            details,
        ]);
    }
    table(out, "Monitoring", &["Monitor", "Status", "Details"], rows);
}

/// Vega's bring-up report: the hand each side configures and what the robot
/// says is there.
fn end_effectors(snapshot: &Value) -> String {
    // The core renders the line (same words as the connect log) so a
    // mismatch names the profile and the variant to select. It quotes the
    // robot's hand-type reply, so it is scrubbed like every other cell.
    let summary = &snapshot["model_bringup"]["end_effectors"]["summary"];
    match summary.as_str() {
        Some(line) if !line.is_empty() => text(summary),
        _ => "--".into(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;
    #[test]
    fn formats_units_missing_feedback_and_active_estops() {
        let s = json!({"profile_name":"vega_1p", "joint_names":{"head":["head_j1"], "torso":["torso_j1"]},
          "joint_states":{"head":{"position":[0.123456], "velocity":[], "error_joint_indices":[0]}},
          "monitoring":{"estop":{"software_estop_enabled":true,"torso_estop_enabled":true},"battery":{"percentage":23.0,"voltage":48.4,"temperature":30.0}},
          "ntp":{"success":true,"offset":-0.0001,"rtt":0.0055,"samples":30,"replies":29},
          "errors":{"joint_state:torso":"no feedback"}});
        let out = render(&s);
        for expected in [
            "0.1235",
            "N/A",
            "Position (rad)",
            "48.40 V",
            "30.00 °C",
            "STOPPED",
            "software, torso hardware",
            "-0.100 ms",
            "5.500 ms",
            "29 / 30 replies",
            "no feedback",
        ] {
            assert!(out.contains(expected), "missing {expected}: {out}");
        }
    }
    #[test]
    fn end_effectors_row_prints_the_line_the_core_rendered() {
        let line = "both: none, but the robot reports DexGripper (select vega_1_gripper)";
        let out =
            render(&json!({"model_bringup":{"end_effectors":{"summary": line, "sides": {}}}}));
        assert!(out.contains(line), "{out}");
        // The robot's reply is quoted in the line: control characters are scrubbed.
        let hostile = json!({"model_bringup":{"end_effectors":{"summary":"left: robot reports \u{1b}[31mX\nY"}}});
        assert_eq!(end_effectors(&hostile), "left: robot reports  [31mX Y");
        assert_eq!(end_effectors(&json!({})), "--");
        assert_eq!(
            end_effectors(&json!({"model_bringup":{"end_effectors":{"summary":""}}})),
            "--"
        );
        assert_eq!(
            end_effectors(&json!({"model_bringup":{"end_effectors":{"sides":{}}}})),
            "--"
        );
    }
    #[test]
    fn missing_estop_evidence_is_not_reported_clear() {
        let out = render(
            &json!({"monitoring":{"estop":{"source":"no_event_published","state_known":false}}}),
        );
        assert!(out.contains("Unknown"));
        assert!(!out.contains("Clear"));
        assert!(out.contains("Unavailable"));
    }
}

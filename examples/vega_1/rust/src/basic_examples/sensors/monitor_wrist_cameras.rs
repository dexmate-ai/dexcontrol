// Copyright (C) 2026 Dexmate Inc.
//
// This software is dual-licensed:
//
// 1. GNU Affero General Public License v3.0 (AGPL-3.0)
//    See LICENSE-AGPL for details
//
// 2. Commercial License
//    For commercial licensing terms, contact: contact@dexmate.ai

//! Monitor frames from the configured wrist cameras.
//!
//! Enable left_wrist_camera and right_wrist_camera unless --sensor supplies names. Subscribe to selected/all advertised streams, monitor at 30 Hz until Ctrl+C by default, then unsubscribe. --samples N limits the number of polls. No particular camera brand is required and no image files are saved.
//!
//! Opens an owned Rerun image viewer; --no-display prints metadata and --viewer-url HOST:PORT connects to an existing viewer. Closing the window ends monitoring.
//!
//! Uses a Robot control connection: compatible arm modes may be enabled during startup, and shutdown requests a stop. Real hardware is selected unless --simulated is supplied.

use dexcontrol_examples::{print_camera_frames, Args, Result};

#[tokio::main]
async fn main() -> Result {
    let args = Args::parse(
        "Monitor frames from the configured wrist cameras. Enable left_wrist_camera and right_wrist_camera unless --sensor supplies names. Subscribe to selected/all advertised streams, monitor at 30 Hz until Ctrl+C by default, then unsubscribe. --samples N limits the number of polls. No particular camera brand is required and no image files are saved. Opens an owned Rerun image viewer; --no-display prints metadata and --viewer-url HOST:PORT connects to an existing viewer. Closing the window ends monitoring. Uses a Robot control connection: compatible arm modes may be enabled during startup, and shutdown requests a stop. Real hardware is selected unless --simulated is supplied.",
        &["period", "samples", "sensor", "stream", "no-display", "viewer-url"],
    );
    let mut sensors = args.list("sensor");
    if sensors.is_empty() {
        sensors = vec!["left_wrist_camera".into(), "right_wrist_camera".into()];
    }
    let viewer = args.image_viewer()?;
    let robot = args.connect(&sensors).await?;
    dexcontrol_examples::run_robot(&robot, async {
        print_camera_frames(
            &robot,
            &sensors,
            &args.list("stream"),
            if args.has("samples") {
                Some(args.integer("samples", 0)?)
            } else {
                None
            },
            args.duration("period", 1.0 / 30.0)?,
            viewer.as_ref(),
        )
        .await?;
        Ok(())
    })
    .await
}

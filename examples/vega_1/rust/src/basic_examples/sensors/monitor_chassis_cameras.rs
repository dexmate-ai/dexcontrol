// Copyright (C) 2026 Dexmate Inc.
//
// This software is dual-licensed:
//
// 1. GNU Affero General Public License v3.0 (AGPL-3.0)
//    See LICENSE-AGPL for details
//
// 2. Commercial License
//    For commercial licensing terms, contact: contact@dexmate.ai

//! Monitor frames from configured chassis cameras.
//!
//! Select base_*_camera sensors from the selected configuration unless --camera is supplied. Subscribe to all advertised streams unless --stream selects a subset; monitor at 30 Hz until Ctrl+C by default. --samples N limits the number of polls. Unsubscribe on exit. No image files are saved.
//!
//! Opens an owned Rerun image viewer; --no-display prints metadata and --viewer-url HOST:PORT connects to an existing viewer. Closing the window ends monitoring.
//!
//! Uses a Robot control connection: compatible arm modes may be enabled during startup, and shutdown requests a stop. Real hardware is selected unless --simulated is supplied.

use dexcontrol_examples::{print_camera_frames, Args, Result};

#[tokio::main]
async fn main() -> Result {
    let args = Args::parse(
        "Monitor frames from configured chassis cameras. Select base_*_camera sensors from the selected configuration unless --camera is supplied. Subscribe to all advertised streams unless --stream selects a subset; monitor at 30 Hz until Ctrl+C by default. --samples N limits the number of polls. Unsubscribe on exit. No image files are saved. Opens an owned Rerun image viewer; --no-display prints metadata and --viewer-url HOST:PORT connects to an existing viewer. Closing the window ends monitoring. Uses a Robot control connection: compatible arm modes may be enabled during startup, and shutdown requests a stop. Real hardware is selected unless --simulated is supplied.",
        &["camera", "period", "samples", "stream", "no-display", "viewer-url"],
    );
    let mut cameras = args.list("camera");
    if cameras.is_empty() {
        cameras = args.resolved_config()?["sensors"]
            .as_object()
            .into_iter()
            .flat_map(|s| s.keys())
            .filter(|name| name.starts_with("base_") && name.ends_with("_camera"))
            .cloned()
            .collect();
    }
    if cameras.is_empty() {
        return Err(dexcontrol_examples::invalid(
            "no chassis cameras are declared; pass --camera NAME",
        ));
    }
    let viewer = args.image_viewer()?;
    let robot = args.connect(&cameras).await?;
    dexcontrol_examples::run_robot(&robot, async {
        print_camera_frames(
            &robot,
            &cameras,
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

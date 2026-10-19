// Copyright (C) 2026 Dexmate Inc.
//
// This software is dual-licensed:
//
// 1. GNU Affero General Public License v3.0 (AGPL-3.0)
//    See LICENSE-AGPL for details
//
// 2. Commercial License
//    For commercial licensing terms, contact: contact@dexmate.ai

//! Monitor frames from the configured head camera.
//!
//! Enable head_camera, subscribe to all advertised streams unless selected with --stream, and monitor at 30 Hz until Ctrl+C by default, then unsubscribe. --samples N limits the number of polls. The sensor name is configurable; no particular camera brand is required and no image files are saved.
//!
//! Opens an owned Rerun image viewer; --no-display prints metadata and --viewer-url HOST:PORT connects to an existing viewer. Closing the window ends monitoring.
//!
//! Uses a Robot control connection: compatible arm modes may be enabled during startup, and shutdown requests a stop. Real hardware is selected unless --simulated is supplied.

use dexcontrol_examples::{print_camera_frames, Args, Result};

#[tokio::main]
async fn main() -> Result {
    let args = Args::parse(
        "Monitor frames from the configured head camera. Enable head_camera, subscribe to all advertised streams unless selected with --stream, and monitor at 30 Hz until Ctrl+C by default, then unsubscribe. --samples N limits the number of polls. The sensor name is configurable; no particular camera brand is required and no image files are saved. Opens an owned Rerun image viewer; --no-display prints metadata and --viewer-url HOST:PORT connects to an existing viewer. Closing the window ends monitoring. Uses a Robot control connection: compatible arm modes may be enabled during startup, and shutdown requests a stop. Real hardware is selected unless --simulated is supplied.",
        &["period", "samples", "sensor", "stream", "no-display", "viewer-url"],
    );
    let sensor = args.text("sensor", "head_camera");
    let viewer = args.image_viewer()?;
    let robot = args.connect(std::slice::from_ref(&sensor)).await?;
    dexcontrol_examples::run_robot(&robot, async {
        print_camera_frames(
            &robot,
            &[sensor],
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

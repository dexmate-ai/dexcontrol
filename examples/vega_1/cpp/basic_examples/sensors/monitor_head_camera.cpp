// Copyright (C) 2026 Dexmate Inc.
//
// This software is dual-licensed:
//
// 1. GNU Affero General Public License v3.0 (AGPL-3.0)
//    See LICENSE-AGPL for details
//
// 2. Commercial License
//    For commercial licensing terms, contact: contact@dexmate.ai

// Monitor frames from the configured head camera.
//
// Enable head_camera, subscribe to all advertised streams unless selected with --stream, and monitor at 30 Hz until Ctrl+C by default, then unsubscribe. --samples N limits the number of polls. The sensor name is configurable; no particular camera brand is required and no image files are saved.
//
// Opens an owned Rerun image viewer; --no-display prints metadata and --viewer-url HOST:PORT connects to an existing viewer. Closing the window ends monitoring.
//
// Uses a Robot control connection: compatible arm modes may be enabled during startup, and shutdown requests a stop. Real hardware is selected unless --simulated is supplied.

#include "camera_common.hpp"

int main(int argc, char **argv) {
    return examples::run(argc, argv, "Monitor frames from the configured head camera. Enable head_camera, subscribe to all advertised streams unless selected with --stream, and monitor at 30 Hz until Ctrl+C by default, then unsubscribe. --samples N limits the number of polls. The sensor name is configurable; no particular camera brand is required and no image files are saved. Opens an owned Rerun image viewer; --no-display prints metadata and --viewer-url HOST:PORT connects to an existing viewer. Closing the window ends monitoring. Uses a Robot control connection: compatible arm modes may be enabled during startup, and shutdown requests a stop. Real hardware is selected unless --simulated is supplied.",
                         [](const examples::Args &args) {
        const auto sensor = args.text("sensor", "head_camera");
        const std::optional<int> samples = args.text("samples").empty()
            ? std::nullopt : std::optional<int>{args.integer("samples", 0)};
        if (samples && *samples < 0) throw std::invalid_argument("--samples must be non-negative");
        auto viewer = examples::image_viewer(args);
        auto robot = args.connect({sensor});
        examples::print_frames(robot, {sensor}, args.list("stream"), samples,
                               args.number("period", 1.0 / 30.0), viewer.get());
    }, {"period", "samples", "sensor", "stream", "no-display", "viewer-url"});
}

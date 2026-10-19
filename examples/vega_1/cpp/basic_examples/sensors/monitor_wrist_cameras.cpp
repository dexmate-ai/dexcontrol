// Copyright (C) 2026 Dexmate Inc.
//
// This software is dual-licensed:
//
// 1. GNU Affero General Public License v3.0 (AGPL-3.0)
//    See LICENSE-AGPL for details
//
// 2. Commercial License
//    For commercial licensing terms, contact: contact@dexmate.ai

// Monitor frames from the configured wrist cameras.
//
// Enable left_wrist_camera and right_wrist_camera unless --sensor supplies names. Subscribe to selected/all advertised streams, monitor at 30 Hz until Ctrl+C by default, then unsubscribe. --samples N limits the number of polls. No particular camera brand is required and no image files are saved.
//
// Opens an owned Rerun image viewer; --no-display prints metadata and --viewer-url HOST:PORT connects to an existing viewer. Closing the window ends monitoring.
//
// Uses a Robot control connection: compatible arm modes may be enabled during startup, and shutdown requests a stop. Real hardware is selected unless --simulated is supplied.

#include "camera_common.hpp"

int main(int argc, char **argv) {
    return examples::run(argc, argv, "Monitor frames from the configured wrist cameras. Enable left_wrist_camera and right_wrist_camera unless --sensor supplies names. Subscribe to selected/all advertised streams, monitor at 30 Hz until Ctrl+C by default, then unsubscribe. --samples N limits the number of polls. No particular camera brand is required and no image files are saved. Opens an owned Rerun image viewer; --no-display prints metadata and --viewer-url HOST:PORT connects to an existing viewer. Closing the window ends monitoring. Uses a Robot control connection: compatible arm modes may be enabled during startup, and shutdown requests a stop. Real hardware is selected unless --simulated is supplied.",
                         [](const examples::Args &args) {
        auto sensors = args.list("sensor");
        if (sensors.empty()) sensors = {"left_wrist_camera", "right_wrist_camera"};
        const std::optional<int> samples = args.text("samples").empty()
            ? std::nullopt : std::optional<int>{args.integer("samples", 0)};
        if (samples && *samples < 0) throw std::invalid_argument("--samples must be non-negative");
        auto viewer = examples::image_viewer(args);
        auto robot = args.connect(sensors);
        examples::print_frames(robot, sensors, args.list("stream"), samples,
                               args.number("period", 1.0 / 30.0), viewer.get());
    }, {"period", "samples", "sensor", "stream", "no-display", "viewer-url"});
}

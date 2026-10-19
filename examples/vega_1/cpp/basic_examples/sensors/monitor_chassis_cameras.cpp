// Copyright (C) 2026 Dexmate Inc.
//
// This software is dual-licensed:
//
// 1. GNU Affero General Public License v3.0 (AGPL-3.0)
//    See LICENSE-AGPL for details
//
// 2. Commercial License
//    For commercial licensing terms, contact: contact@dexmate.ai

// Monitor frames from configured chassis cameras.
//
// Select base_*_camera sensors from the selected configuration unless --camera is supplied. Subscribe to all advertised streams unless --stream selects a subset; monitor at 30 Hz until Ctrl+C by default. --samples N limits the number of polls. Unsubscribe on exit. No image files are saved.
//
// Opens an owned Rerun image viewer; --no-display prints metadata and --viewer-url HOST:PORT connects to an existing viewer. Closing the window ends monitoring.
//
// Uses a Robot control connection: compatible arm modes may be enabled during startup, and shutdown requests a stop. Real hardware is selected unless --simulated is supplied.

#include "camera_common.hpp"
#include "json.hpp"

int main(int argc, char **argv) {
    return examples::run(argc, argv, "Monitor frames from configured chassis cameras. Select base_*_camera sensors from the selected configuration unless --camera is supplied. Subscribe to all advertised streams unless --stream selects a subset; monitor at 30 Hz until Ctrl+C by default. --samples N limits the number of polls. Unsubscribe on exit. No image files are saved. Opens an owned Rerun image viewer; --no-display prints metadata and --viewer-url HOST:PORT connects to an existing viewer. Closing the window ends monitoring. Uses a Robot control connection: compatible arm modes may be enabled during startup, and shutdown requests a stop. Real hardware is selected unless --simulated is supplied.",
                         [](const examples::Args &args) {
        auto cameras = args.list("camera");
        if (cameras.empty()) {
            const auto config =
                examples::Json::parse(dexcontrol::resolved_config_json(args.connect_options()));
            for (const auto &[name, sensor] : config["sensors"].members()) {
                (void)sensor;
                if (name.rfind("base_", 0) == 0 && name.size() > 12 &&
                    name.compare(name.size() - 7, 7, "_camera") == 0) {
                    cameras.push_back(name);
                }
            }
        }
        if (cameras.empty()) {
            throw std::runtime_error(
                "no chassis cameras are declared; pass --camera NAME or use a custom profile");
        }
        const std::optional<int> samples = args.text("samples").empty()
            ? std::nullopt : std::optional<int>{args.integer("samples", 0)};
        if (samples && *samples < 0) throw std::invalid_argument("--samples must be non-negative");
        auto viewer = examples::image_viewer(args);
        auto robot = args.connect(cameras);
        examples::print_frames(robot, cameras, args.list("stream"), samples,
                               args.number("period", 1.0 / 30.0), viewer.get());
    }, {"camera", "period", "samples", "stream", "no-display", "viewer-url"});
}

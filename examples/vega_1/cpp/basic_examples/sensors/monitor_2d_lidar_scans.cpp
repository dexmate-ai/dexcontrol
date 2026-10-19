// Copyright (C) 2026 Dexmate Inc.
//
// This software is dual-licensed:
//
// 1. GNU Affero General Public License v3.0 (AGPL-3.0)
//    See LICENSE-AGPL for details
//
// 2. Commercial License
//    For commercial licensing terms, contact: contact@dexmate.ai

// Monitor live 2D LiDAR scans.
//
// Enable lidar_2d_front, wait up to 10 s for activity, and poll continuously at 0.1 s intervals until Ctrl+C. --samples N limits the number of polls. Display XY points or range summaries; no scan file is saved.
//
// Prints numeric summaries/frame metadata; does not open a viewer.
//
// Uses a Robot control connection: compatible arm modes may be enabled during startup, and shutdown requests a stop. Real hardware is selected unless --simulated is supplied.

#include "example.hpp"

int main(int argc, char **argv) {
    return examples::run(argc, argv, "Monitor live 2D LiDAR scans. Enable lidar_2d_front, wait up to 10 s for activity, and poll continuously at 0.1 s intervals until Ctrl+C. --samples N limits the number of polls. Display XY points or range summaries; no scan file is saved. Prints numeric summaries/frame metadata; does not open a viewer. Uses a Robot control connection: compatible arm modes may be enabled during startup, and shutdown requests a stop. Real hardware is selected unless --simulated is supplied.",
                         [](const examples::Args &args) {
        const auto sensor = args.text("sensor", "lidar_2d_front");
        const std::optional<int> samples = args.text("samples").empty()
            ? std::nullopt : std::optional<int>{args.integer("samples", 0)};
        if (samples && *samples < 0) throw std::invalid_argument("--samples must be non-negative");
        const double period = args.number("period", 0.1);

        auto robot = args.connect({sensor});
        if (!robot.wait_for_state(sensor, std::chrono::seconds{10})) {
            throw std::runtime_error(sensor + " did not become active");
        }
        for (size_t index = 0; !examples::interrupted() && (!samples || index < static_cast<size_t>(*samples)); ++index) {
            const auto scan = robot.lidar_2d(sensor);
            size_t finite = 0;
            double low = INFINITY;
            for (const double range : scan.ranges) {
                if (!std::isfinite(range)) continue;
                ++finite;
                low = std::min(low, range);
            }
            std::cout << std::fixed << std::setprecision(3) << "sample=" << index + 1
                      << " points=" << finite << " min=" << low << "\n";
            if (!samples || index + 1 < static_cast<size_t>(*samples)) examples::sleep_for(period);
        }
    }, {"period", "samples", "sensor"});
}

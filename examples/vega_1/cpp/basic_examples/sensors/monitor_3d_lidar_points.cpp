// Copyright (C) 2026 Dexmate Inc.
// SPDX-License-Identifier: Apache-2.0

// Monitor point clouds from one 3D LiDAR.
//
// Enable the front 3D LiDAR, wait up to 10 s for activity, and poll continuously at 0.1 s intervals until Ctrl+C. --samples N limits the number of polls. --position back selects the rear LiDAR; no cloud file is saved.
//
// Prints numeric summaries/frame metadata; does not open a viewer.
//
// Uses a Robot control connection: compatible arm modes may be enabled during startup, and shutdown requests a stop. Real hardware is selected unless --simulated is supplied.

#include "example.hpp"

int main(int argc, char **argv) {
    return examples::run(argc, argv, "Monitor point clouds from one 3D LiDAR. Enable the front 3D LiDAR, wait up to 10 s for activity, and poll continuously at 0.1 s intervals until Ctrl+C. --samples N limits the number of polls. --position back selects the rear LiDAR; no cloud file is saved. Prints numeric summaries/frame metadata; does not open a viewer. Uses a Robot control connection: compatible arm modes may be enabled during startup, and shutdown requests a stop. Real hardware is selected unless --simulated is supplied.",
                         [](const examples::Args &args) {
        const auto name = "lidar_3d_" + args.choice("position", {"front", "back"}, "front");
        const std::optional<int> samples = args.text("samples").empty()
            ? std::nullopt : std::optional<int>{args.integer("samples", 0)};
        if (samples && *samples < 0) throw std::invalid_argument("--samples must be non-negative");
        const double period = args.number("period", 0.1);

        auto robot = args.connect({name});
        if (!robot.wait_for_state(name, std::chrono::seconds{10})) {
            throw std::runtime_error(name + " did not become active");
        }
        for (size_t index = 0; !examples::interrupted() && (!samples || index < static_cast<size_t>(*samples)); ++index) {
            const auto cloud = robot.sensors().lidar_3d(name);
            std::cout << "sample=" << index + 1 << " points=" << cloud.z.size();
            if (!cloud.z.empty()) {
                const auto [low, high] = std::minmax_element(cloud.z.begin(), cloud.z.end());
                std::cout << std::fixed << std::setprecision(3) << " z=" << *low << ".." << *high << " m";
            }
            std::cout << "\n";
            if (!samples || index + 1 < static_cast<size_t>(*samples)) examples::sleep_for(period);
        }
    }, {"period", "position", "samples"});
}

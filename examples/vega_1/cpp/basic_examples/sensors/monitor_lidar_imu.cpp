// Copyright (C) 2026 Dexmate Inc.
//
// This software is dual-licensed:
//
// 1. GNU Affero General Public License v3.0 (AGPL-3.0)
//    See LICENSE-AGPL for details
//
// 2. Commercial License
//    For commercial licensing terms, contact: contact@dexmate.ai

// Monitor the IMU associated with a 3D LiDAR.
//
// Enable lidar_3d_front, wait up to 10 s for its IMU stream, and print observations continuously at 0.05 s intervals until Ctrl+C. --samples N limits the number of polls. --sensor selects another sensor with an IMU capability.
//
// Uses a Robot control connection: compatible arm modes may be enabled during startup, and shutdown requests a stop. Real hardware is selected unless --simulated is supplied.

#include "example.hpp"

int main(int argc, char **argv) {
    return examples::run(argc, argv, "Monitor the IMU associated with a 3D LiDAR. Enable lidar_3d_front, wait up to 10 s for its IMU stream, and print observations continuously at 0.05 s intervals until Ctrl+C. --samples N limits the number of polls. --sensor selects another sensor with an IMU capability. Uses a Robot control connection: compatible arm modes may be enabled during startup, and shutdown requests a stop. Real hardware is selected unless --simulated is supplied.",
                         [](const examples::Args &args) {
        const auto sensor = args.text("sensor", "lidar_3d_front");
        const std::optional<int> samples = args.text("samples").empty()
            ? std::nullopt : std::optional<int>{args.integer("samples", 0)};
        if (samples && *samples < 0) throw std::invalid_argument("--samples must be non-negative");
        const double period = args.number("period", 0.05);

        auto robot = args.connect({sensor});
        const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds{10};
        while (true) {
            try {
                (void)robot.imu(sensor);
                break;
            } catch (const dexcontrol::Error &error) {
                if (error.status() != DEX_STATE_UNAVAILABLE) throw;
            }
            if (std::chrono::steady_clock::now() >= deadline) {
                throw std::runtime_error(sensor + " IMU did not become active");
            }
            examples::sleep_for(0.02);
        }
        for (size_t index = 0; !examples::interrupted() && (!samples || index < static_cast<size_t>(*samples)); ++index) {
            const auto imu = robot.imu(sensor);
            std::cout << "acceleration=" << examples::format(imu.acceleration)
                      << " angular_velocity=" << examples::format(imu.angular_velocity)
                      << " orientation=" << examples::format(imu.orientation, 4) << "\n";
            if (!samples || index + 1 < static_cast<size_t>(*samples)) examples::sleep_for(period);
        }
    }, {"period", "samples", "sensor"});
}

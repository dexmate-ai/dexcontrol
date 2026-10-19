// Copyright (C) 2026 Dexmate Inc.
//
// This software is dual-licensed:
//
// 1. GNU Affero General Public License v3.0 (AGPL-3.0)
//    See LICENSE-AGPL for details
//
// 2. Commercial License
//    For commercial licensing terms, contact: contact@dexmate.ai

// Read and print observations from configured IMUs.
//
// Read IMU streams from head_camera and chassis_imu (when declared) once by default, waiting up to 5 s for activity. --sensor changes the requested list; optional repeated reads use a 0.1 s interval.
//
// Filters undeclared IMU names before connecting.
//
// Uses a Robot control connection: compatible arm modes may be enabled during startup, and shutdown requests a stop. Real hardware is selected unless --simulated is supplied.

#include "example.hpp"
#include "json.hpp"

namespace {
void print_imu(const std::string &name, const dex_imu_t &imu) {
    std::cout << "\n" << name << "\n"
              << "  acceleration:     " << examples::format(imu.acceleration) << " m/s^2\n"
              << "  angular_velocity: " << examples::format(imu.angular_velocity) << " rad/s\n"
              << "  orientation:      " << examples::format(imu.orientation, 4) << "\n";
    if (imu.has_magnetic_field) {
        std::cout << "  magnetic_field:   " << examples::format(imu.magnetic_field) << "\n";
    }
    std::cout << "  timestamp:        " << imu.timestamp_ns << " ns\n";
}
}  // namespace

int main(int argc, char **argv) {
    return examples::run(argc, argv, "Read and print observations from configured IMUs. Read IMU streams from head_camera and chassis_imu (when declared) once by default, waiting up to 5 s for activity. --sensor changes the requested list; optional repeated reads use a 0.1 s interval. Filters undeclared IMU names before connecting. Uses a Robot control connection: compatible arm modes may be enabled during startup, and shutdown requests a stop. Real hardware is selected unless --simulated is supplied.",
                         [](const examples::Args &args) {
        const int samples = args.integer("samples", 1);
        const double period = args.number("period", 0.1);
        auto sensors = args.list("sensor");
        if (sensors.empty()) sensors = {"head_camera", "chassis_imu"};

        // Enabling a sensor the profile lacks fails at connect, so filter the
        // requested names through the profile's declared sensors first.
        const auto declared =
            examples::Json::parse(dexcontrol::resolved_config_json(args.connect_options()))["sensors"];
        std::vector<std::string> available;
        for (const auto &name : sensors) {
            if (declared.contains(name)) {
                available.push_back(name);
            } else {
                std::cout << name << ": not declared by this profile, skipping\n";
            }
        }
        if (available.empty()) throw std::runtime_error("none of the requested IMUs exist in this profile");
        auto robot = args.connect(available);
        for (const auto &name : available) {
            const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds{5};
            while (true) {
                try {
                    (void)robot.imu(name);
                    break;
                } catch (const dexcontrol::Error &error) {
                    if (error.status() != DEX_STATE_UNAVAILABLE) throw;
                }
                if (std::chrono::steady_clock::now() >= deadline) {
                    throw std::runtime_error(name + " IMU did not become active");
                }
                examples::sleep_for(0.02);
            }
        }
        for (int index = 0; index < samples; ++index) {
            for (const auto &name : available) print_imu(name, robot.imu(name));
            if (index + 1 < samples) examples::sleep_for(period);
        }
    }, {"period", "samples", "sensor"});
}

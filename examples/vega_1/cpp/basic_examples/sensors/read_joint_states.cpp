// Copyright (C) 2026 Dexmate Inc.
// SPDX-License-Identifier: Apache-2.0

// Print current joint states and chassis state.
//
// Read all available joint components once, skipping unavailable joint state; --component narrows that list. Also print chassis state when available, independently of the component filter. This is a cached state listing, not a complete diagnostic report.
//
// Prints named joint rows with model-derived units; unavailable fields appear as —. Chassis steering, wheel encoder positions and wheel speeds are listed separately.
//
// Uses a Robot control connection: compatible arm modes may be enabled during startup, and shutdown requests a stop. Real hardware is selected unless --simulated is supplied.

#include "example.hpp"
#include "json.hpp"

static std::array<std::string, 3> units(const std::string &type) {
    if (type == "prismatic") return {"m", "m/s", "N"};
    if (type == "revolute" || type == "continuous") return {"rad", "rad/s", "N·m"};
    return {"units", "units/s", "units"};
}
static std::string value(const std::vector<double> &values, size_t index, const std::string &unit) {
    if (index >= values.size()) return "—";
    auto number = values[index];
    if (std::abs(number) < 0.00005) number = 0.0;
    std::ostringstream out;
    out << std::fixed << std::setprecision(4) << number << " " << unit;
    return out.str();
}

int main(int argc, char **argv) {
    return examples::run(argc, argv, "Print current joint states and chassis state. Read all available joint components once, skipping unavailable joint state; --component narrows that list. Also print chassis state when available, independently of the component filter. This is a cached state listing, not a complete diagnostic report. Prints named joint rows with model-derived units; unavailable fields appear as —. Chassis steering, wheel encoder positions and wheel speeds are listed separately. Uses a Robot control connection: compatible arm modes may be enabled during startup, and shutdown requests a stop. Real hardware is selected unless --simulated is supplied.",
                         [](const examples::Args &args) {
        auto robot = args.connect();
        const auto config = examples::Json::parse(robot.config_json());
        std::cout << "Cached joint states (— = not reported)\n";
        auto names = args.list("component");
        if (names.empty()) names = robot.component_names();
        for (const auto &name : names) {
            dexcontrol::JointState state;
            std::vector<std::string> joint_names;
            try {
                auto joint = robot.joints(name);
                state = joint.get_joint_state();
                joint_names = joint.joint_names();
            } catch (const dexcontrol::Error &) {
                continue;  // not a joint component, or no state yet
            }
            std::cout << "\n" << name << "\n  " << std::left << std::setw(20) << "Joint"
                      << std::right << std::setw(16) << "Position" << std::setw(16) << "Velocity"
                      << std::setw(16) << "Effort" << std::setw(14) << "Current" << "\n";
            for (size_t i = 0; i < joint_names.size(); ++i) {
                const auto unit = units(config["joint_metadata"][name][i]["joint_type"].text());
                std::cout << "  " << std::left << std::setw(20) << joint_names[i] << std::right
                          << std::setw(16) << value(state.position, i, unit[0])
                          << std::setw(16) << value(state.velocity, i, unit[1])
                          << std::setw(16) << value(state.torque, i, unit[2])
                          << std::setw(14) << value(state.current, i, "A") << "\n";
            }
        }
        if (robot.has_component("chassis")) {
            try {
                auto positions = robot.chassis().steering_angle();
                const auto steer_count = positions.size();
                const auto wheels = robot.chassis().wheel_encoder_pos();
                const auto speeds = robot.chassis().wheel_velocity();
                positions.insert(positions.end(), wheels.begin(), wheels.end());
                const auto &names = config["components"]["chassis"]["joints"]["names"];
                std::cout << "\nchassis\n  " << std::left << std::setw(20) << "Joint"
                          << std::setw(22) << "Measurement" << std::right << std::setw(16) << "Position" << std::setw(16) << "Wheel speed" << "\n";
                for (size_t i = 0; i < positions.size(); ++i) {
                    const auto unit = units(config["joint_metadata"]["chassis"][i]["joint_type"].text());
                    std::cout << "  " << std::left << std::setw(20) << names[i].text("joint_" + std::to_string(i))
                              << std::setw(22) << (i < steer_count ? "Steering angle" : "Wheel encoder")
                              << std::right << std::setw(16) << value(positions, i, unit[0])
                              << std::setw(16) << (i < steer_count ? "—" : value(speeds, i - steer_count, "m/s")) << "\n";
                }
            } catch (const dexcontrol::Error &) {
                std::cout << "\nchassis: state unavailable\n";
            }
        }
    }, {"component"});
}

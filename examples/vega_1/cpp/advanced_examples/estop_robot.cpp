// Copyright (C) 2026 Dexmate Inc.
// SPDX-License-Identifier: Apache-2.0

// Read, activate, or deactivate the software E-stop.
//
// Default positional action is status. activate requests software E-stop; deactivate clears it. Print the resulting observed status.
//
// Uses a Robot control connection with startup arm initialization turned off, so the tool also reaches an arm whose motor cannot enable; shutdown requests a stop. Real hardware is selected unless --simulated is supplied.

#include "example.hpp"

int main(int argc, char **argv) {
    return examples::run(argc, argv, "Read, activate, or deactivate the software E-stop. Default positional action is status. activate requests software E-stop; deactivate clears it. Print the resulting observed status. Uses a Robot control connection with startup arm initialization turned off, so the tool also reaches an arm whose motor cannot enable; shutdown requests a stop. Real hardware is selected unless --simulated is supplied.",
                         [](const examples::Args &args) {
        const auto action = args.positional_choice(0, {"status", "activate", "deactivate"}, "status");

        auto options = args.connect_options();
        options.skip_arm_initialization = true;
        auto robot = dexcontrol::Robot::connect(options);
        if (action == "activate") {
            robot.estop_activate();
        } else if (action == "deactivate") {
            robot.estop_deactivate();
        }
        const auto status = robot.estop_status();
        std::string sources;
        const auto add_source = [&](bool active, const char *label) {
            if (active) { if (!sources.empty()) sources += ", "; sources += label; }
        };
        add_source(status.software_estop_enabled, "software");
        add_source(status.left_base_estop_enabled, "left base hardware");
        add_source(status.right_base_estop_enabled, "right base hardware");
        add_source(status.torso_estop_enabled, "torso hardware");
        add_source(status.remote_estop_enabled, "remote hardware");
        std::cout << "E-stop:         " << (status.engaged ? "ACTIVE" : "Not active") << "\n"
                  << "Active sources: " << (sources.empty() ? "None" : sources) << "\n"
                  << "Feedback:       " << (status.state_observed ? "Received" : "No event received (event-driven E-stop)") << "\n";
        if (status.engaged) {
            std::cout << "Robot writes are blocked. Read operations remain available.\n";
        }
    }, {},
       "action: status | activate | deactivate (default: status)");
}

// Copyright (C) 2026 Dexmate Inc.
// SPDX-License-Identifier: Apache-2.0

// Clear errors on all supported robot components.
//
// Send clear-error requests, print each component outcome, and exit nonzero if any outcome failed. This is a mutating maintenance operation.
//
// Uses a Robot control connection: compatible arm modes may be enabled during startup, and shutdown requests a stop. Real hardware is selected unless --simulated is supplied.

#include "example.hpp"
#include "json.hpp"

int main(int argc, char **argv) {
    return examples::run(argc, argv, "Clear errors on all supported robot components. Send clear-error requests, print each component outcome, and exit nonzero if any outcome failed. This is a mutating maintenance operation. Uses a Robot control connection with startup arm initialization turned off: an arm whose motor cannot enable is the usual reason to clear errors, and the default connection would fail on it before any request is sent. Shutdown requests a stop. Real hardware is selected unless --simulated is supplied.",
                         [](const examples::Args &args) {
        auto options = args.connect_options();
        options.skip_arm_initialization = true;
        auto robot = dexcontrol::Robot::connect(options);
        const auto report = examples::Json::parse(robot.clear_errors_json());
        const auto &outcomes = report["outcomes"].items();
        if (outcomes.empty()) {
            std::cout << "No clearable components are enabled for this robot.\n";
        }
        bool failed = false;
        for (const auto &outcome : outcomes) {
            const auto component = outcome["component"].as_string("?");
            if (outcome["success"].as_bool()) {
                std::cout << "✓ " << component << "\n";
            } else {
                failed = true;
                std::cout << "✗ " << component << ": " << outcome["error"].text("unknown error")
                          << "\n";
            }
        }
        if (failed) throw std::runtime_error("some components could not be cleared");
    }, {});
}

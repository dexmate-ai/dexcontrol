// Copyright (C) 2026 Dexmate Inc.
// SPDX-License-Identifier: Apache-2.0

// Read, release, or engage arm brakes.
//
// Require positional status, release, or engage. status and engage address both arms and all their joints unless --side or --joint narrow them. release has no defaults: it requires --side left, right, or both, and either --joint INDEX (repeatable) or --all. Before releasing, warn that the arm must be physically supported, because a released joint can drop under gravity, and require typing yes; --yes skips the prompt and simulated runs do not prompt. This operates the brake service, not the motor-mode service.
//
// Uses a Robot control connection with startup arm initialization turned off, so the tool also reaches an arm whose motor cannot enable; shutdown requests a stop. Real hardware is selected unless --simulated is supplied.

#include "example.hpp"

int main(int argc, char **argv) {
    return examples::run(argc, argv, "Read, release, or engage arm brakes. Require positional status, release, or engage. status and engage address both arms and all their joints unless --side or --joint narrow them. release has no defaults: it requires --side left, right, or both, and either --joint INDEX (repeatable) or --all. Before releasing, warn that the arm must be physically supported, because a released joint can drop under gravity, and require typing yes; --yes skips the prompt and simulated runs do not prompt. This operates the brake service, not the motor-mode service. Uses a Robot control connection with startup arm initialization turned off, so the tool also reaches an arm whose motor cannot enable; shutdown requests a stop. Real hardware is selected unless --simulated is supplied.",
                         [](const examples::Args &args) {
        const auto action = args.positional_choice(0, {"status", "release", "engage"}, "");
        std::vector<size_t> joints;
        for (const auto &value : args.list("joint")) {
            if (value.empty() || value.find_first_not_of("0123456789") != std::string::npos) {
                examples::Args::fail("--joint expects a non-negative joint index, got '" + value + "'");
            }
            joints.push_back(std::stoul(value));
        }
        const bool all = args.flag("all");
        if (all && !joints.empty()) examples::Args::fail("pass either --joint or --all, not both");
        if (action == "release") {
            // Releasing has no defaults: a released joint can drop under
            // gravity, so the arm and the joints are always named.
            if (args.text("side").empty()) {
                examples::Args::fail("release requires --side left, right, or both");
            }
            if (!all && joints.empty()) {
                examples::Args::fail("release requires --joint INDEX (repeatable) or --all");
            }
        }
        const auto side = args.choice("side", {"left", "right", "both"}, "both");
        const std::vector<std::string> sides =
            side == "both" ? std::vector<std::string>{"left", "right"} : std::vector<std::string>{side};
        if (action == "release") {
            examples::confirm_hazard(
                args, "releasing " + std::string(all ? "EVERY brake" : "the selected brakes") + " of the " +
                          (side == "both" ? std::string("left and right arms") : side + " arm") +
                          ". A released joint can drop under gravity: the arm must be physically "
                          "supported before you continue.");
        }

        auto options = args.connect_options();
        options.skip_arm_initialization = true;
        auto robot = dexcontrol::Robot::connect(options);
        for (const auto &name : sides) {
            auto arm = robot.joints(name + "_arm");
            // An empty joint list is an error in the library, never "all":
            // addressing every joint is its own, explicit call.
            const auto result = action == "status" ? arm.get_brake_status()
                                : joints.empty()   ? arm.release_all_brakes(action == "release")
                                                   : arm.release_brake(action == "release", joints);
            std::cout << name << "_arm\n" << result << "\n";
        }
        robot.close();
    }, {"all", "joint", "side", "yes"},
       "action: status | release | engage");
}

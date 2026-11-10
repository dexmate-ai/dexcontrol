// Copyright (C) 2026 Dexmate Inc.
// SPDX-License-Identifier: Apache-2.0

// Enable or disable head motors.
//
// The optional positional mode is enable or disable; without it, send disable. Print the firmware reply; no head target is commanded.
//
// Uses a Robot control connection: compatible arm modes may be enabled during startup, and shutdown requests a stop. Real hardware is selected unless --simulated is supplied.

#include "example.hpp"

int main(int argc, char **argv) {
    return examples::run(argc, argv, "Enable or disable head motors. The optional positional mode is enable or disable; without it, send disable. Print the firmware reply; no head target is commanded. Uses a Robot control connection: compatible arm modes may be enabled during startup, and shutdown requests a stop. Real hardware is selected unless --simulated is supplied.",
                         [](const examples::Args &args) {
        const auto mode = args.positional_choice(0, {"enable", "disable"}, "disable");

        auto robot = args.connect();
        std::cout << robot.joints("head").set_mode(mode == "enable" ? dexcontrol::JointMode::Enable
                                                                     : dexcontrol::JointMode::Disable)
                  << "\n";
    }, {},
       "mode: enable | disable (default: disable)");
}

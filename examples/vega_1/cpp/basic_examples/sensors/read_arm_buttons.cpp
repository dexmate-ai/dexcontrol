// Copyright (C) 2026 Dexmate Inc.
// SPDX-License-Identifier: Apache-2.0

// Read and print arm wrist-button states.
//
// Read one right-arm button observation by default. --samples and --period repeat the reads; the default repeat interval is 0.1 s.
//
// Uses a Robot control connection: compatible arm modes may be enabled during startup, and shutdown requests a stop. Real hardware is selected unless --simulated is supplied.

#include "example.hpp"

int main(int argc, char **argv) {
    return examples::run(argc, argv, "Read and print arm wrist-button states. Read one right-arm button observation by default. --samples and --period repeat the reads; the default repeat interval is 0.1 s. Uses a Robot control connection: compatible arm modes may be enabled during startup, and shutdown requests a stop. Real hardware is selected unless --simulated is supplied.",
                         [](const examples::Args &args) {
        const int samples = args.integer("samples", 1);
        const double period = args.number("period", 0.1);

        auto robot = args.connect();
        auto arm = robot.joints(args.side() + "_arm");
        for (int index = 0; index < samples; ++index) {
            const auto buttons = arm.get_button_state();
            if (buttons) {
                std::cout << "blue=" << (buttons->first ? "pressed" : "released")
                          << " green=" << (buttons->second ? "pressed" : "released") << "\n";
            } else {
                std::cout << "no button sample\n";
            }
            if (index + 1 < samples) examples::sleep_for(period);
        }
    }, {"period", "samples", "side"});
}

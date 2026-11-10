// Copyright (C) 2026 Dexmate Inc.
// SPDX-License-Identifier: Apache-2.0

// Read and print fingertip forces from one hand.
//
// Read the right hand once by default and report missing touch data. Optional repeated reads use a 0.1 s interval.
//
// Uses a Robot control connection: compatible arm modes may be enabled during startup, and shutdown requests a stop. Real hardware is selected unless --simulated is supplied.

#include "example.hpp"

int main(int argc, char **argv) {
    return examples::run(argc, argv, "Read and print fingertip forces from one hand. Read the right hand once by default and report missing touch data. Optional repeated reads use a 0.1 s interval. Uses a Robot control connection: compatible arm modes may be enabled during startup, and shutdown requests a stop. Real hardware is selected unless --simulated is supplied.",
                         [](const examples::Args &args) {
        const int samples = args.integer("samples", 1);
        const double period = args.number("period", 0.1);

        auto robot = args.connect();
        auto hand = robot.joints(args.side() + "_hand");
        for (int index = 0; index < samples; ++index) {
            const auto forces = hand.get_touch_data();
            std::cout << (forces ? examples::format(*forces) : std::string("no touch sample")) << "\n";
            if (index + 1 < samples) examples::sleep_for(period);
        }
    }, {"period", "samples", "side"});
}

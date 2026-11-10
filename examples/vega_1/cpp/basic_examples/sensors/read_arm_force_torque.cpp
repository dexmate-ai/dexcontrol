// Copyright (C) 2026 Dexmate Inc.
// SPDX-License-Identifier: Apache-2.0

// Read and print one arm force-torque observation.
//
// Wait up to 5 s for the right-arm wrench stream and print one observation by default. Optional repeated reads use a 0.1 s interval; no calibration or zeroing is requested.
//
// Uses a Robot control connection: compatible arm modes may be enabled during startup, and shutdown requests a stop. Real hardware is selected unless --simulated is supplied.

#include "example.hpp"

int main(int argc, char **argv) {
    return examples::run(argc, argv, "Read and print one arm force-torque observation. Wait up to 5 s for the right-arm wrench stream and print one observation by default. Optional repeated reads use a 0.1 s interval; no calibration or zeroing is requested. Uses a Robot control connection: compatible arm modes may be enabled during startup, and shutdown requests a stop. Real hardware is selected unless --simulated is supplied.",
                         [](const examples::Args &args) {
        const auto arm = args.side() + "_arm";
        const int samples = args.integer("samples", 1);
        const double period = args.number("period", 0.1);

        auto robot = args.connect();
        if (!robot.wait_for_state(arm, std::chrono::seconds{5})) {
            throw std::runtime_error("wrench state did not become active");
        }
        for (int index = 0; index < samples; ++index) {
            const auto wrench = robot.sensors().wrench(arm);
            std::cout << "force=" << examples::format(std::vector<double>(wrench.values.begin(), wrench.values.begin() + 3))
                      << " N torque=" << examples::format(std::vector<double>(wrench.values.begin() + 3, wrench.values.end()))
                      << " N*m (timestamp " << wrench.timestamp_ns << " ns)\n";
            if (index + 1 < samples) examples::sleep_for(period);
        }
    }, {"period", "samples", "side"});
}

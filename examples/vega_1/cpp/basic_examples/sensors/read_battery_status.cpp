// Copyright (C) 2026 Dexmate Inc.
// SPDX-License-Identifier: Apache-2.0

// Read and print battery status.
//
// Wait up to 5 s for battery activity and print one observation by default, including charge, voltage, current and temperature. Optional repeated reads use a 0.5 s interval.
//
// Uses a Robot control connection: compatible arm modes may be enabled during startup, and shutdown requests a stop. Real hardware is selected unless --simulated is supplied.

#include "example.hpp"

int main(int argc, char **argv) {
    return examples::run(argc, argv, "Read and print battery status. Wait up to 5 s for battery activity and print one observation by default, including charge, voltage, current and temperature. Optional repeated reads use a 0.5 s interval. Uses a Robot control connection: compatible arm modes may be enabled during startup, and shutdown requests a stop. Real hardware is selected unless --simulated is supplied.", [](const examples::Args &args) {
        const int samples = args.integer("samples", 1);
        const double period = args.number("period", 0.5);
        if (samples < 1 || period < 0.0) examples::Args::fail("samples must be positive and period non-negative");

        auto robot = args.connect();
        if (!robot.has_component("battery")) throw std::runtime_error("this robot has no battery");
        if (!robot.wait_for_state("battery", std::chrono::seconds{5})) {
            throw std::runtime_error("battery state did not become active");
        }
        for (int index = 0; index < samples; ++index) {
            const auto battery = robot.battery();
            std::cout << "Battery status (" << index + 1 << "/" << samples << ")\n"
                      << std::fixed << std::setprecision(1)
                      << "  Charge       " << battery.percentage << " %\n"
                      << std::setprecision(2)
                      << "  Voltage      " << battery.voltage << " V\n"
                      << "  Current      " << battery.current << " A\n"
                      << "  Power        " << battery.voltage * battery.current << " W\n"
                      << std::setprecision(1)
                      << "  Temperature  " << battery.temperature << " °C\n" << std::endl;
            if (index + 1 < samples) examples::sleep_for(period);
        }
    }, {"period", "samples"});
}

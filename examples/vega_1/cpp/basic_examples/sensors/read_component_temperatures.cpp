// Copyright (C) 2026 Dexmate Inc.
// SPDX-License-Identifier: Apache-2.0

// Print component temperatures and battery status.
//
// Read each available component temperature stream once, then battery temperature/status. --component narrows component selection; battery reporting is separate. Report unavailable temperature sources.
//
// Uses a Robot control connection: compatible arm modes may be enabled during startup, and shutdown requests a stop. Real hardware is selected unless --simulated is supplied.

#include "example.hpp"
#include "json.hpp"

#include <chrono>
#include <thread>

int main(int argc, char **argv) {
    return examples::run(argc, argv, "Print component temperatures and battery status. Read each available component temperature stream once, then battery temperature/status. --component narrows component selection; battery reporting is separate. Report unavailable temperature sources. Uses a Robot control connection: compatible arm modes may be enabled during startup, and shutdown requests a stop. Real hardware is selected unless --simulated is supplied.",
                         [](const examples::Args &args) {
        auto robot = args.connect();
        const auto config = examples::Json::parse(robot.config_json());
        auto names = args.list("component");
        const bool explicit_selection = !names.empty();
        if (names.empty()) names = robot.component_names();
        std::vector<std::string> unavailable;
        for (const auto &name : names) {
            if (name == "battery") continue;
            if (!config["components"][name]["endpoints"].contains("temperature_sub_topic")) {
                if (explicit_selection) std::cout << "No temperature stream configured for: " << name << "\n";
                continue;
            }
            // Read before printing the heading: a failure would otherwise
            // leave a bare component name with nothing under it.
            // Temperature streams are slow (about 1 Hz) and connecting does
            // not wait for them, so give the first sample up to 5 s to arrive.
            std::string temperatures;
            const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(5);
            while (true) {
                try {
                    temperatures = robot.sensors().temperature_json(name);
                    break;
                } catch (const dexcontrol::Error &) {
                    if (std::chrono::steady_clock::now() >= deadline) break;
                    std::this_thread::sleep_for(std::chrono::milliseconds(100));
                }
            }
            if (temperatures.empty()) {
                unavailable.push_back(name);
                continue;
            }
            const auto state = examples::Json::parse(temperatures);
            bool printed = false;
            for (const auto &[group, readings] : state["temperatures"].members()) {
                for (const auto &[label, value] : readings.members()) {
                    if (!printed) std::cout << "\n" << name << " temperatures\n";
                    printed = true;
                    std::cout << "  " << group << "/" << label << ": " << std::fixed << std::setprecision(1) << value.as_number() << " °C\n";
                }
            }
            if (!printed) unavailable.push_back(name);
        }
        if (robot.has_component("battery")) {
            try {
                const auto battery = robot.battery();
                std::cout << "\nBattery status\n" << std::fixed << std::setprecision(1)
                          << "  Charge       " << battery.percentage << " %\n" << std::setprecision(2)
                          << "  Voltage      " << battery.voltage << " V\n"
                          << "  Current      " << battery.current << " A\n"
                          << "  Power        " << battery.voltage * battery.current << " W\n" << std::setprecision(1)
                          << "  Temperature  " << battery.temperature << " °C\n";
            } catch (const dexcontrol::Error &) {
                std::cout << "\nBattery status unavailable.\n";
            }
        } else {
            std::cout << "\nNo battery configured on this robot.\n";
        }
        // Say so rather than exiting silently: on a robot whose temperature
        // streams are not publishing, every component lands here and the
        // program would otherwise print nothing at all.
        if (!unavailable.empty()) {
            std::cout << "\nTemperature readings unavailable:";
            for (size_t index = 0; index < unavailable.size(); ++index) {
                std::cout << (index == 0 ? " " : ", ") << unavailable[index];
            }
            std::cout << "\n";
        }
    }, {"component"});
}

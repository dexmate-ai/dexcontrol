// Copyright (C) 2026 Dexmate Inc.
// SPDX-License-Identifier: Apache-2.0

// Drive the chassis in six directions, requesting a stop between them.
//
// Command forward, backward, left, right, counter-clockwise, then clockwise for 3 s each, streaming commands at 50 Hz by default (--control-hz). Linear speed is 0.1 m/s and turn speed is 0.1 rad/s. Request stops between directions and on cleanup; requires a base capable of strafing.
//
// Uses a Robot control connection: compatible arm modes may be enabled during startup, and shutdown requests a stop. Real hardware is selected unless --simulated is supplied.

#include "example.hpp"
#include <iomanip>

int main(int argc, char **argv) {
    return examples::run(argc, argv, "Drive the chassis in six directions, requesting a stop between them. Command forward, backward, left, right, counter-clockwise, then clockwise for 3 s each, streaming commands at 50 Hz by default (--control-hz). Linear speed is 0.1 m/s and turn speed is 0.1 rad/s. Request stops between directions and on cleanup; requires a base capable of strafing. Uses a Robot control connection: compatible arm modes may be enabled during startup, and shutdown requests a stop. Real hardware is selected unless --simulated is supplied.",
                         [](const examples::Args &args) {
        const double speed = args.number("speed", 0.1);
        const double duration = args.number("duration", 3.0);
        if (duration < 0.0) examples::Args::fail("--duration must be non-negative");

        auto robot = args.connect();
        if (!robot.has_component("chassis")) throw std::runtime_error("this robot has no chassis");
        const std::vector<std::pair<const char *, std::array<double, 3>>> commands = {
            {"forward", {speed, 0.0, 0.0}},         {"backward", {-speed, 0.0, 0.0}},
            {"left", {0.0, speed, 0.0}},            {"right", {0.0, -speed, 0.0}},
            {"counter-clockwise", {0.0, 0.0, speed}}, {"clockwise", {0.0, 0.0, -speed}},
        };
        const double control_hz = args.number("control-hz", 50.0);
        std::cout << "Chassis exercise | " << commands.size() << " phases | " << control_hz << " Hz\n"
                  << "Commands in robot frame: +vx forward, +vy left, +wz counter-clockwise.\n"
                  << "Each phase includes steering alignment if needed, then timed driving." << std::endl;
        try {
            std::size_t index = 0;
            for (const auto &[label, velocity] : commands) {
                std::cout << "[" << ++index << "/" << commands.size() << "] " << label
                          << " | " << std::fixed << std::setprecision(2) << duration << " s | "
                          << std::showpos << std::setprecision(3)
                          << "vx=" << velocity[0] << " m/s, vy=" << velocity[1]
                          << " m/s, wz=" << velocity[2] << " rad/s" << std::noshowpos << std::endl;
                robot.chassis().drive_for(velocity[0], velocity[1], velocity[2], examples::millis(duration), control_hz);
                std::cout << "      Command stream complete; stop sent." << std::endl;
            }
        } catch (...) {
            robot.chassis().stop();
            throw;
        }
        robot.chassis().stop();
        // Stop and disconnect deliberately: a stop that cannot be delivered
        // is reported here rather than left to the handle's destructor.
        robot.close();
    }, {"control-hz", "duration", "speed"});
}

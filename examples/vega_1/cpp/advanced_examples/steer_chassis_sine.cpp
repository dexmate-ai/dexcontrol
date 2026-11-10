// Copyright (C) 2026 Dexmate Inc.
// SPDX-License-Identifier: Apache-2.0

// Drive forward while sinusoidally changing both steering angles.
//
// For 6 s at 50 Hz, command both wheel speeds to 0.5 m/s and both steering angles to 0.6*sin(2*pi*0.5*t) rad. Request a stop on completion/cleanup. Requires a steer-drive base; it does not use generic chassis yaw velocity.
//
// Uses a Robot control connection: compatible arm modes may be enabled during startup, and shutdown requests a stop. Real hardware is selected unless --simulated is supplied.

#include "example.hpp"

int main(int argc, char **argv) {
    return examples::run(argc, argv, "Drive forward while sinusoidally changing both steering angles. For 6 s at 50 Hz, command both wheel speeds to 0.5 m/s and both steering angles to 0.6*sin(2*pi*0.5*t) rad. Request a stop on completion/cleanup. Requires a steer-drive base; it does not use generic chassis yaw velocity. Uses a Robot control connection: compatible arm modes may be enabled during startup, and shutdown requests a stop. Real hardware is selected unless --simulated is supplied.",
                         [](const examples::Args &args) {
        const double amplitude = args.number("amplitude", 0.6);
        const double frequency = args.number("frequency", 0.5);
        const double speed = args.number("speed", 0.5);
        const double duration = args.number("duration", 6.0);
        const double control_hz = args.number("control-hz", 50.0);

        auto robot = args.connect();
        if (!robot.has_component("chassis")) examples::Args::fail("this robot has no chassis");
        const auto started = std::chrono::steady_clock::now();
        const double period = 1.0 / control_hz;
        try {
            for (double elapsed = 0.0; elapsed < duration; elapsed = examples::elapsed(started)) {
                const double steering = amplitude * std::sin(2.0 * M_PI * frequency * elapsed);
                robot.chassis().set_motion_state({steering, steering}, {speed, speed});
                examples::sleep_for(period);
            }
        } catch (...) {
            robot.chassis().stop();
            throw;
        }
        robot.chassis().stop();
        // Stop and disconnect deliberately: a stop that cannot be delivered
        // is reported here rather than left to the handle's destructor.
        robot.close();
    }, {"amplitude", "control-hz", "duration", "frequency", "speed"});
}

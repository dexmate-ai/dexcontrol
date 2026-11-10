// Copyright (C) 2026 Dexmate Inc.
// SPDX-License-Identifier: Apache-2.0

// Guide both arms by hand: yield to pushes sensed from joint current.
//
// Capture each arm's joint positions and currents at rest, then at 200 Hz treat a per-joint current offset beyond --threshold A (default 0.25) as a push and send a joint setpoint that moves with it: each joint's velocity is the push divided by a damping of 2*gain (--gain, default 0.3; larger is stiffer, below 0.3 is refused), and no tick commands a setpoint more than 0.2 rad from the measured position. By default the arms stay wherever they are pushed; --hold-pose adds a spring back to the captured pose. Run until Ctrl-C or --duration seconds (default 0, unlimited), then stop both arms. No force sensor, no model and no identification: the current threshold is the only filter, so run it on a machine wired to the robot.
//
// Uses a Robot control connection: compatible arm modes may be enabled during startup, and shutdown requests a stop. Real hardware is selected unless --simulated is supplied.

#include "example.hpp"

#include <array>
#include <algorithm>
#include <cmath>

int main(int argc, char **argv) {
    return examples::run(argc, argv, "Guide both arms by hand: yield to pushes sensed from joint current. Capture each arm's joint positions and currents at rest, then at 200 Hz treat a per-joint current offset beyond --threshold A (default 0.25) as a push and send a joint setpoint that moves with it: each joint's velocity is the push divided by a damping of 2*gain (--gain, default 0.3; larger is stiffer, below 0.3 is refused), and no tick commands a setpoint more than 0.2 rad from the measured position. By default the arms stay wherever they are pushed; --hold-pose adds a spring back to the captured pose. Run until Ctrl-C or --duration seconds (default 0, unlimited), then stop both arms. No force sensor, no model and no identification: the current threshold is the only filter, so run it on a machine wired to the robot. Uses a Robot control connection: compatible arm modes may be enabled during startup, and shutdown requests a stop. Real hardware is selected unless --simulated is supplied.",
                         [](const examples::Args &args) {
        // Per-joint spring pattern of the 7-joint Vega arm, scaled by --gain
        // when --hold-pose is set: the shoulder and elbow pitch joints carry
        // the load.
        const std::array<double, 7> spring_pattern{2.0, 0.25, 0.25, 2.0, 0.25, 0.25, 0.25};
        // Largest move one tick may command, rad. A direct setpoint is checked
        // against a fresh measurement with a 0.3 rad step guard; a slow loop or
        // a hard shove must slow the arm down, not end the example.
        constexpr double max_step = 0.2;
        const double gain = args.number("gain", 0.3);
        const double threshold = args.number("threshold", 0.25);
        const double control_hz = args.number("control-hz", 200.0);
        const double duration = args.number("duration", 0.0);
        const bool hold_pose = args.flag("hold-pose");
        if (gain < 0.3) examples::Args::fail("--gain below 0.3 lets a current blip fling the arm; refusing");
        if (threshold < 0.0) examples::Args::fail("--threshold must not be negative");
        if (control_hz <= 0.0) examples::Args::fail("--control-hz must be positive");

        auto robot = args.connect();
        struct Arm {
            std::string name;
            dexcontrol::JointComponent component;
            std::vector<double> rest_pos;
            std::vector<double> rest_current;
        };
        std::vector<Arm> arms;
        for (const char *name : {"left_arm", "right_arm"}) {
            if (!robot.has_component(name)) examples::Args::fail(std::string("this robot has no ") + name);
            auto component = robot.joints(name);
            if (component.joint_count() != spring_pattern.size()) {
                examples::Args::fail("hand guiding is tuned for the 7-joint Vega arm; " + std::string(name) + " has " +
                                     std::to_string(component.joint_count()));
            }
            // A rest baseline: everything the joints draw to hold still here
            // is not a push.
            const auto state = component.get_joint_state();
            if (state.current.size() != spring_pattern.size()) {
                examples::Args::fail(std::string(name) + " reports no per-joint current");
            }
            arms.push_back({name, std::move(component), state.position, state.current});
        }
        const double damping = 2.0 * gain;
        const double dt = 1.0 / control_hz;
        std::cout << "hand guiding both arms at " << control_hz << " Hz; threshold " << threshold << " A; "
                  << (hold_pose ? "spring back to the start pose" : "float where pushed") << std::endl;
        // Stop every arm even if one stop throws; report the first failure.
        auto stop_all = [&arms]() {
            std::exception_ptr first_error;
            for (auto &arm : arms) {
                try {
                    arm.component.stop();
                } catch (...) {
                    if (!first_error) first_error = std::current_exception();
                }
            }
            return first_error;
        };
        dexcontrol::RateLimiter limiter(control_hz);
        std::vector<double> target(spring_pattern.size());
        try {
            for (size_t ticks = 0; duration <= 0.0 || static_cast<double>(ticks) * dt < duration; ++ticks) {
                for (auto &arm : arms) {
                    const auto state = arm.component.get_joint_state();
                    for (size_t joint = 0; joint < target.size(); ++joint) {
                        const double offset = state.current[joint] - arm.rest_current[joint];
                        // Below the threshold is noise and the load of holding
                        // still; above it, the sign is inverted so the joint
                        // gives way to the push.
                        const double push = std::fabs(offset) < threshold ? 0.0 : -offset;
                        const double spring = hold_pose ? spring_pattern[joint] * gain : 0.0;
                        const double velocity = (spring * (arm.rest_pos[joint] - state.position[joint]) + push) / damping;
                        target[joint] = state.position[joint] + std::clamp(velocity * dt, -max_step, max_step);
                    }
                    // The direct setpoint path, as the 0.5 script used: the
                    // motion plugin's own smoothing would lag the target and
                    // its catch-up current reads as a new push.
                    arm.component.set_joint_pos(target);
                }
                if (examples::interrupted()) {
                    std::cout << "stopped" << std::endl;
                    break;
                }
                limiter.sleep();
            }
        } catch (...) {
            (void)stop_all();
            throw;
        }
        if (auto error = stop_all()) std::rethrow_exception(error);
    }, {"control-hz", "duration", "gain", "hold-pose", "threshold"});
}

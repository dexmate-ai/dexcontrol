// Copyright (C) 2026 Dexmate Inc.
// SPDX-License-Identifier: Apache-2.0

// Sweep head joints through positive, negative, and zero targets.
//
// First command head joint 0 to -pi/6 rad with other joints zero. Then command each joint to +0.5, -0.5, and zero radians, with all other target joints zero. Finish at all zeros; wait up to 10 s per motion.
//
// The sequence assumes exactly three head joints.
//
// Uses a Robot control connection: compatible arm modes may be enabled during startup, and shutdown requests a stop. Real hardware is selected unless --simulated is supplied.

#include "example.hpp"

int main(int argc, char **argv) {
    return examples::run(argc, argv, "Sweep head joints through positive, negative, and zero targets. First command head joint 0 to -pi/6 rad with other joints zero. Then command each joint to +0.5, -0.5, and zero radians, with all other target joints zero. Finish at all zeros; wait up to 10 s per motion. The sequence assumes exactly three head joints. Uses a Robot control connection: compatible arm modes may be enabled during startup, and shutdown requests a stop. Real hardware is selected unless --simulated is supplied.",
                         [](const examples::Args &args) {
        const auto timeout = dexcontrol::WaitFor{examples::millis(args.number("timeout", 10.0))};
        const double delta = args.number("delta", 0.5);

        auto robot = args.connect();
        auto head = robot.joints("head");
        head.move_to_joint_pos({-M_PI / 6.0, 0.0, 0.0}).wait(timeout);
        for (size_t joint = 0; joint < 3; ++joint) {
            // Positive, negative, and back to zero.
            for (const double value : {delta, -delta, 0.0}) {
                std::vector<double> target(3, 0.0);
                target[joint] = value;
                head.move_to_joint_pos(target).wait(timeout);
            }
        }
        // Stop and disconnect deliberately: a stop that cannot be delivered
        // is reported here rather than left to the handle's destructor.
        robot.close();
    }, {"delta", "timeout"});
}

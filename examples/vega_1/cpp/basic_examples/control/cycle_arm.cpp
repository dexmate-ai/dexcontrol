// Copyright (C) 2026 Dexmate Inc.
// SPDX-License-Identifier: Apache-2.0

// Move one arm joint by an offset, then return to its starting position.
//
// Save the right arm positions, move joint 0 by +0.2 rad, then return to the saved absolute start after outward success. By default compute an absolute outward target; --relative resolves the offset from fresh feedback. Both motions use velocity scale 0.2 and a 10 s wait deadline.
//
// Uses a Robot control connection: compatible arm modes may be enabled during startup, and shutdown requests a stop. Real hardware is selected unless --simulated is supplied.

#include "example.hpp"

int main(int argc, char **argv) {
    return examples::run(argc, argv, "Move one arm joint by an offset, then return to its starting position. Save the right arm positions, move joint 0 by +0.2 rad, then return to the saved absolute start after outward success. By default compute an absolute outward target; --relative resolves the offset from fresh feedback. Both motions use velocity scale 0.2 and a 10 s wait deadline. Uses a Robot control connection: compatible arm modes may be enabled during startup, and shutdown requests a stop. Real hardware is selected unless --simulated is supplied.",
                         [](const examples::Args &args) {
        const auto timeout = dexcontrol::WaitFor{examples::millis(args.number("timeout", 10.0))};
        const int joint = args.integer("joint", 0);

        auto robot = args.connect();
        auto arm = robot.joints(args.side() + "_arm");
        const auto start = arm.get_joint_pos();
        const bool relative = args.flag("relative");
        auto target = relative ? std::vector<double>(start.size(), 0.0) : start;
        if (joint < 0 || static_cast<size_t>(joint) >= target.size()) {
            examples::Args::fail("--joint must be in [0, " + std::to_string(target.size() - 1) + "]");
        }
        target[static_cast<size_t>(joint)] += args.number("delta", 0.2);
        auto options = dexcontrol::with_velocity_scale(dexcontrol::motion_options(),
                                                             args.number("velocity-scale", 0.2));
        options.relative = relative;
        arm.move_to_joint_pos(target, options).wait_success(timeout);
        options.relative = false;
        arm.move_to_joint_pos(start, options).wait_success(timeout);
        // Stop and disconnect deliberately: a stop that cannot be delivered
        // is reported here rather than left to the handle's destructor.
        robot.close();
    }, {"delta", "joint", "relative", "side", "timeout", "velocity-scale"});
}

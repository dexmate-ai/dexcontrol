// Copyright (C) 2026 Dexmate Inc.
// SPDX-License-Identifier: Apache-2.0

// Move the torso to an offset joint target, then return to its starting position.
//
// Read the torso positions and move joint 0 by +0.1 rad with a planned tracked motion. By default compute an absolute target from the stored start; --relative lets the API resolve the offset against fresh feedback. Always return to the stored absolute start after outward success. Each motion wait has a 10 s deadline.
//
// Uses a Robot control connection: compatible arm modes may be enabled during startup, and shutdown requests a stop. Real hardware is selected unless --simulated is supplied.

#include "example.hpp"

int main(int argc, char **argv) {
    return examples::run(argc, argv, "Move the torso to an offset joint target, then return to its starting position. Read the torso positions and move joint 0 by +0.1 rad with a planned tracked motion. By default compute an absolute target from the stored start; --relative lets the API resolve the offset against fresh feedback. Always return to the stored absolute start after outward success. Each motion wait has a 10 s deadline. Uses a Robot control connection: compatible arm modes may be enabled during startup, and shutdown requests a stop. Real hardware is selected unless --simulated is supplied.",
                         [](const examples::Args &args) {
        const auto timeout = dexcontrol::WaitFor{examples::millis(args.number("timeout", 10.0))};
        const int joint = args.integer("joint", 0);

        auto robot = args.connect();
        auto torso = robot.joints("torso");
        const auto start = torso.get_joint_pos();
        if (joint < 0 || static_cast<size_t>(joint) >= start.size()) {
            examples::Args::fail("--joint must be in [0, " + std::to_string(start.size() - 1) + "]");
        }
        const bool relative = args.flag("relative");
        auto target = relative ? std::vector<double>(start.size(), 0.0) : start;
        target[static_cast<size_t>(joint)] += args.number("delta", 0.1);
        auto options = dexcontrol::motion_options();
        options.relative = relative;
        torso.move_to_joint_pos(target, options, timeout);
        torso.move_to_joint_pos(start, timeout);
        // Stop and disconnect deliberately: a stop that cannot be delivered
        // is reported here rather than left to the handle's destructor.
        robot.close();
    }, {"delta", "joint", "relative", "timeout"});
}

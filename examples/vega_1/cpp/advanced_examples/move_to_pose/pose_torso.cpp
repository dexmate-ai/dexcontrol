// Copyright (C) 2026 Dexmate Inc.
// SPDX-License-Identifier: Apache-2.0

// Move the torso to a named joint pose.
//
// Move to home and wait up to 10 s for tracked completion. No collision checking or automatic return is performed.
//
// Uses a Robot control connection: compatible arm modes may be enabled during startup, and shutdown requests a stop. Real hardware is selected unless --simulated is supplied.

#include "example.hpp"

int main(int argc, char **argv) {
    return examples::run(argc, argv, "Move the torso to a named joint pose. Move to home and wait up to 10 s for tracked completion. No collision checking or automatic return is performed. Uses a Robot control connection: compatible arm modes may be enabled during startup, and shutdown requests a stop. Real hardware is selected unless --simulated is supplied.",
                         [](const examples::Args &args) {
        const auto pose = args.text("pose", "home");
        const auto timeout = dexcontrol::WaitFor{examples::millis(args.number("timeout", 10.0))};

        auto robot = args.connect();
        auto motion = robot.joints("torso").go_to_pose(pose, timeout);
        std::cout << dexcontrol::to_string(motion.motion_state()) << "\n";
        // Stop and disconnect deliberately: a stop that cannot be delivered
        // is reported here rather than left to the handle's destructor.
        robot.close();
    }, {"pose", "timeout"});
}

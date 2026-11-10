// Copyright (C) 2026 Dexmate Inc.
// SPDX-License-Identifier: Apache-2.0

// Move the head to a named pose using its model-declared reference frame.
//
// Move to home using model-defined torso-pitch compensation; tucked stays joint-relative. --passthrough uses raw stored values. Wait up to 10 s. No collision checking is performed.
//
// Uses a Robot control connection: compatible arm modes may be enabled during startup, and shutdown requests a stop. Real hardware is selected unless --simulated is supplied.

#include "example.hpp"

int main(int argc, char **argv) {
    return examples::run(argc, argv, "Move the head to a named pose using its model-declared reference frame. Move to home using model-defined torso-pitch compensation; tucked stays joint-relative. --passthrough uses raw stored values. Wait up to 10 s. No collision checking is performed. Uses a Robot control connection: compatible arm modes may be enabled during startup, and shutdown requests a stop. Real hardware is selected unless --simulated is supplied.",
                         [](const examples::Args &args) {
        const auto pose = args.text("pose", "home");
        const auto timeout = dexcontrol::WaitFor{examples::millis(args.number("timeout", 10.0))};
        const bool passthrough = args.flag("passthrough");

        auto robot = args.connect();
        auto head = robot.joints("head");
        auto target = head.get_pose(pose, passthrough);
        std::cout << dexcontrol::to_string(head.move_to_joint_pos(target).wait(timeout)) << "\n";
        // Stop and disconnect deliberately: a stop that cannot be delivered
        // is reported here rather than left to the handle's destructor.
        robot.close();
    }, {"passthrough", "pose", "timeout"});
}

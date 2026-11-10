// Copyright (C) 2026 Dexmate Inc.
// SPDX-License-Identifier: Apache-2.0

// Move one arm to a named pose using its model-declared reference frame.
//
// Move the right arm to L_shape with model-defined torso-pitch compensation; wait up to 10 s. folded and folded_closed_hand stay joint-relative and receive no compensation. zero stores seven zeros relative to an upright torso and compensates only torso tilt from upright; --passthrough returns literal zeros. --passthrough uses raw stored values while retaining target validation. No collision checking is performed.
//
// Uses a Robot control connection: compatible arm modes may be enabled during startup, and shutdown requests a stop. Real hardware is selected unless --simulated is supplied.

#include "example.hpp"

int main(int argc, char **argv) {
    return examples::run(argc, argv, "Move one arm to a named pose using its model-declared reference frame. Move the right arm to L_shape with model-defined torso-pitch compensation; wait up to 10 s. folded and folded_closed_hand stay joint-relative and receive no compensation. zero stores seven zeros relative to an upright torso and compensates only torso tilt from upright; --passthrough returns literal zeros. --passthrough uses raw stored values while retaining target validation. No collision checking is performed. Uses a Robot control connection: compatible arm modes may be enabled during startup, and shutdown requests a stop. Real hardware is selected unless --simulated is supplied.",
                         [](const examples::Args &args) {
        const auto name = args.side() + "_arm";
        const auto pose = args.text("pose", "L_shape");
        const auto timeout = dexcontrol::WaitFor{examples::millis(args.number("timeout", 10.0))};

        auto robot = args.connect();
        auto target = robot.joints(name).get_pose(pose, args.flag("passthrough"));
        std::cout << dexcontrol::to_string(robot.joints(name).move_to_joint_pos(target).wait(timeout)) << "\n";
        // Stop and disconnect deliberately: a stop that cannot be delivered
        // is reported here rather than left to the handle's destructor.
        robot.close();
    }, {"passthrough", "pose", "side", "timeout"});
}

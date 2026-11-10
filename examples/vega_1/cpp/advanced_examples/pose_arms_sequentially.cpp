// Copyright (C) 2026 Dexmate Inc.
// SPDX-License-Identifier: Apache-2.0

// Move arms sequentially to a named pose without collision checking.
//
// Move the left arm and then the right arm to L_shape at velocity scale 0.5, waiting up to 10 s each. --side can select one arm. Resolve model-declared pose frames, including torso compensation for L_shape. No homing or collision planning is performed.
//
// Uses a Robot control connection: compatible arm modes may be enabled during startup, and shutdown requests a stop. Real hardware is selected unless --simulated is supplied.

#include "example.hpp"

int main(int argc, char **argv) {
    return examples::run(argc, argv, "Move arms sequentially to a named pose without collision checking. Move the left arm and then the right arm to L_shape at velocity scale 0.5, waiting up to 10 s each. --side can select one arm. Resolve model-declared pose frames, including torso compensation for L_shape. No homing or collision planning is performed. Uses a Robot control connection: compatible arm modes may be enabled during startup, and shutdown requests a stop. Real hardware is selected unless --simulated is supplied.",
                         [](const examples::Args &args) {
        const auto side = args.choice("side", {"left", "right", "both"}, "both");
        const auto pose = args.text("pose", "L_shape");
        const double velocity_scale = args.number("velocity-scale", 0.5);
        const auto timeout = dexcontrol::WaitFor{examples::millis(args.number("timeout", 10.0))};
        const std::vector<std::string> sides =
            side == "both" ? std::vector<std::string>{"left", "right"} : std::vector<std::string>{side};

        auto robot = args.connect();
        for (const auto &name : sides) {
            auto arm = robot.joints(name + "_arm");
            auto motion = arm.move_to_joint_pos(
                arm.resolve_pose(pose),
                dexcontrol::with_velocity_scale(dexcontrol::motion_options(), velocity_scale));
            std::cout << name << "_arm: " << dexcontrol::to_string(motion.wait(timeout)) << "\n";
        }
        // Stop and disconnect deliberately: a stop that cannot be delivered
        // is reported here rather than left to the handle's destructor.
        robot.close();
    }, {"pose", "side", "timeout", "velocity-scale"});
}

// Copyright (C) 2026 Dexmate Inc.
// SPDX-License-Identifier: Apache-2.0

// Move both arms to a named pose using independent tracked motions.
//
// Default to folded at velocity scale 0.5. Resolve each pose using its model-declared frame: folded poses are joint-relative; orientation reference poses use torso pitch. --passthrough uses raw stored values. Start both targets through best-effort fan-out, wait up to 10 s, and require aggregate success. No synchronized start or collision checking is provided.
//
// Uses a Robot control connection: compatible arm modes may be enabled during startup, and shutdown requests a stop. Real hardware is selected unless --simulated is supplied.

#include "example.hpp"

int main(int argc, char **argv) {
    return examples::run(argc, argv, "Move both arms to a named pose using independent tracked motions. Default to folded at velocity scale 0.5. Resolve each pose using its model-declared frame: folded poses are joint-relative; orientation reference poses use torso pitch. --passthrough uses raw stored values. Start both targets through best-effort fan-out, wait up to 10 s, and require aggregate success. No synchronized start or collision checking is provided. Uses a Robot control connection: compatible arm modes may be enabled during startup, and shutdown requests a stop. Real hardware is selected unless --simulated is supplied.",
                         [](const examples::Args &args) {
        const auto pose = args.text("pose", "folded");
        const double velocity_scale = args.number("velocity-scale", 0.5);
        const double timeout = args.number("timeout", 10.0);
        const bool passthrough = args.flag("passthrough");

        auto robot = args.connect();
        dexcontrol::Robot::Targets targets;
        for (const std::string name : {"left_arm", "right_arm"}) {
            auto arm = robot.joints(name);
            auto target = arm.get_pose(pose, passthrough);
            targets[name] = target;
        }
        auto group = robot.move_to_joint_pos(
            targets, dexcontrol::with_velocity_scale(dexcontrol::motion_options(), velocity_scale));
        const auto state = group.wait(dexcontrol::WaitFor{examples::millis(timeout)});
        // Reporting the state without checking it exits 0 on a failed fold,
        // which reads as success to anything scripting this.
        if (state != dexcontrol::MotionState::Succeeded) {
            throw std::runtime_error("fold reported '" + std::string(dexcontrol::to_string(state)) +
                                     "' within " + std::to_string(timeout) + "s");
        }
        std::cout << dexcontrol::to_string(state) << "\n";
        // Stop and disconnect deliberately: a stop that cannot be delivered
        // is reported here rather than left to the handle's destructor.
        robot.close();
    }, {"passthrough", "pose", "timeout", "velocity-scale"});
}

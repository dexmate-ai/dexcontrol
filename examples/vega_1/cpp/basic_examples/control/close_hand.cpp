// Copyright (C) 2026 Dexmate Inc.
// SPDX-License-Identifier: Apache-2.0

// Close one hand using its model-defined pose.
//
// Command the right hand closed and wait 2 s. Optional --grasp-torque changes the grip setting for a gripper; omitted values use model defaults. The delay is not a convergence check.
//
// Uses a Robot control connection: compatible arm modes may be enabled during startup, and shutdown requests a stop. Real hardware is selected unless --simulated is supplied.

#include "example.hpp"

int main(int argc, char **argv) {
    return examples::run(argc, argv, "Close one hand using its model-defined pose. Command the right hand closed and wait 2 s. Optional --grasp-torque changes the grip setting for a gripper; omitted values use model defaults. The delay is not a convergence check. Uses a Robot control connection: compatible arm modes may be enabled during startup, and shutdown requests a stop. Real hardware is selected unless --simulated is supplied.",
                         [](const examples::Args &args) {
        auto robot = args.connect();
        auto hand = robot.joints(args.side() + "_hand");
        // Grippers are torque-commanded; five-finger hands are not, and report
        // no grasp torque rather than accepting a grip force.
        if (!args.text("grasp-torque").empty() && hand.grasp_torque().has_value()) {
            std::cout << "grasp torque " << hand.set_grasp_torque(args.number("grasp-torque", 0.0)) << "\n";
        }
        hand.close_hand();
        examples::sleep_for(args.number("wait-time", 2.0));
        // Stop and disconnect deliberately: a stop that cannot be delivered
        // is reported here rather than left to the handle's destructor.
        robot.close();
    }, {"grasp-torque", "side", "wait-time"});
}

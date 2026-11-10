// Copyright (C) 2026 Dexmate Inc.
// SPDX-License-Identifier: Apache-2.0

// Close one hand, then reopen it.
//
// Command the right hand to its model-defined closed pose, wait 2 s, command its open pose, then wait 2 s. These delays are not tracked-motion completion checks.
//
// Uses a Robot control connection: compatible arm modes may be enabled during startup, and shutdown requests a stop. Real hardware is selected unless --simulated is supplied.

#include "example.hpp"

int main(int argc, char **argv) {
    return examples::run(argc, argv, "Close one hand, then reopen it. Command the right hand to its model-defined closed pose, wait 2 s, command its open pose, then wait 2 s. These delays are not tracked-motion completion checks. Uses a Robot control connection: compatible arm modes may be enabled during startup, and shutdown requests a stop. Real hardware is selected unless --simulated is supplied.", [](const examples::Args &args) {
        const double wait_time = args.number("wait-time", 2.0);

        auto robot = args.connect();
        auto hand = robot.joints(args.side() + "_hand");
        hand.close_hand();
        examples::sleep_for(wait_time);
        hand.open_hand();
        examples::sleep_for(wait_time);
        // Stop and disconnect deliberately: a stop that cannot be delivered
        // is reported here rather than left to the handle's destructor.
        robot.close();
    }, {"side", "wait-time"});
}

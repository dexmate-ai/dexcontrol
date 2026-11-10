// Copyright (C) 2026 Dexmate Inc.
// SPDX-License-Identifier: Apache-2.0

// Request a reboot of one robot control board.
//
// Require positional arm, torso, or chassis. The arm board controls both arms. A rebooting board stops controlling its joints, so ask for confirmation first and require typing yes; --yes skips the prompt and simulated runs do not prompt. Send the request and exit; do not wait for the board to come back online.
//
// Uses a Robot control connection with startup arm initialization turned off, so the tool also reaches an arm whose motor cannot enable; shutdown requests a stop. Real hardware is selected unless --simulated is supplied.

#include "example.hpp"

int main(int argc, char **argv) {
    return examples::run(argc, argv, "Request a reboot of one robot control board. Require positional arm, torso, or chassis. The arm board controls both arms. A rebooting board stops controlling its joints, so ask for confirmation first and require typing yes; --yes skips the prompt and simulated runs do not prompt. Send the request and exit; do not wait for the board to come back online. Uses a Robot control connection with startup arm initialization turned off, so the tool also reaches an arm whose motor cannot enable; shutdown requests a stop. Real hardware is selected unless --simulated is supplied.",
                         [](const examples::Args &args) {
        const auto board = args.positional_choice(0, {"arm", "torso", "chassis"}, "");
        examples::confirm_hazard(
            args, "rebooting the " + board + " control board" +
                      (board == "arm" ? " (it drives BOTH arms)" : "") +
                      ". The board stops controlling its joints while it restarts: make sure "
                      "nothing depends on them holding position.");
        auto options = args.connect_options();
        options.skip_arm_initialization = true;
        auto robot = dexcontrol::Robot::connect(options);
        robot.reboot(board);
        // Fire-and-forget: the board stops answering while it restarts, so a
        // clean return means the request was accepted, not that it is back.
        std::cout << board << ": reboot request sent\n";
        robot.close();
    }, {"yes"},
       "board: arm | torso | chassis");
}

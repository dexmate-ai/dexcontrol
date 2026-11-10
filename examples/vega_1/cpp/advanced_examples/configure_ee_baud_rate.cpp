// Copyright (C) 2026 Dexmate Inc.
// SPDX-License-Identifier: Apache-2.0

// Read or set end-effector RS485 baud rates.
//
// Require positional get or set. Address both arm end-effector interfaces by default; set uses 115200 baud unless overridden. Print service replies.
//
// Uses a Robot control connection: compatible arm modes may be enabled during startup, and shutdown requests a stop. Real hardware is selected unless --simulated is supplied.

#include "example.hpp"

int main(int argc, char **argv) {
    return examples::run(argc, argv, "Read or set end-effector RS485 baud rates. Require positional get or set. Address both arm end-effector interfaces by default; set uses 115200 baud unless overridden. Print service replies. Uses a Robot control connection: compatible arm modes may be enabled during startup, and shutdown requests a stop. Real hardware is selected unless --simulated is supplied.",
                         [](const examples::Args &args) {
        const auto action = args.positional_choice(0, {"get", "set"}, "");
        const auto side = args.choice("side", {"left", "right", "both"}, "both");
        const auto baud_rate = static_cast<uint32_t>(args.integer("baud-rate", 115200));
        const std::vector<std::string> sides =
            side == "both" ? std::vector<std::string>{"left", "right"} : std::vector<std::string>{side};

        auto robot = args.connect();
        for (const auto &name : sides) {
            auto arm = robot.joints(name + "_arm");
            const auto result = action == "get" ? arm.get_ee_baud_rate() : arm.set_ee_baud_rate(baud_rate);
            std::cout << name << "_arm\n" << result << "\n";
        }
    }, {"baud-rate", "side"},
       "action: get | set");
}

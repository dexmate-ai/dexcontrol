// Copyright (C) 2026 Dexmate Inc.
// SPDX-License-Identifier: Apache-2.0

// Read or set arm PID multipliers.
//
// Require positional get or set. Address both arms by default; set sends seven multipliers, each defaulting to 1.0. Print service replies. PID writes can take about 40 s; the SDK waits up to 45 s for a reply. A timeout does not cancel a server-side write.
//
// Uses a Robot control connection: compatible arm modes may be enabled during startup, and shutdown requests a stop. Real hardware is selected unless --simulated is supplied.

#include "example.hpp"

int main(int argc, char **argv) {
    return examples::run(argc, argv, "Read or set arm PID multipliers. Require positional get or set. Address both arms by default; set sends seven multipliers, each defaulting to 1.0. Print service replies. PID writes can take about 40 s; the SDK waits up to 45 s for a reply. A timeout does not cancel a server-side write. Uses a Robot control connection: compatible arm modes may be enabled during startup, and shutdown requests a stop. Real hardware is selected unless --simulated is supplied.",
                         [](const examples::Args &args) {
        const auto action = args.positional_choice(0, {"get", "set"}, "");
        const auto side = args.choice("side", {"left", "right", "both"}, "both");
        const auto p = args.numbers("p", std::vector<double>(7, 1.0));
        for (const double value : p) {
            if (value < 0.1 || value > 4.0) {
                examples::Args::fail("every PID multiplier must be in [0.1, 4.0]");
            }
        }
        const std::vector<std::string> sides =
            side == "both" ? std::vector<std::string>{"left", "right"} : std::vector<std::string>{side};

        auto robot = args.connect();
        for (const auto &name : sides) {
            auto arm = robot.joints(name + "_arm");
            if (action == "set") std::cout << "Setting " << name << "_arm PID gains; this can take about 40 seconds..." << std::endl;
            const auto result = action == "get" ? arm.get_pid() : arm.set_pid(p);
            std::cout << name << "_arm\n" << result << "\n";
        }
    }, {"p", "side"},
       "action: get | set");
}

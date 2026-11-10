// Copyright (C) 2026 Dexmate Inc.
// SPDX-License-Identifier: Apache-2.0

// Send raw bytes to end effectors and poll for replies.
//
// By default, send hex 09 10 03 E8 00 03 06 01 00 00 00 00 00 72 E1 to both configured arm pass-through interfaces and poll for up to 1 s each. The attached device determines what these bytes do; this is not a generic motion command.
//
// Uses a Robot control connection: compatible arm modes may be enabled during startup, and shutdown requests a stop. Real hardware is selected unless --simulated is supplied.

#include "example.hpp"

#include <algorithm>

namespace {

std::vector<uint8_t> parse_hex(std::string text) {
    text.erase(std::remove(text.begin(), text.end(), ' '), text.end());
    if (text.size() % 2 != 0) examples::Args::fail("--message-hex needs an even number of digits");
    std::vector<uint8_t> bytes;
    for (size_t index = 0; index < text.size(); index += 2) {
        bytes.push_back(static_cast<uint8_t>(std::stoul(text.substr(index, 2), nullptr, 16)));
    }
    return bytes;
}

std::string hex(const std::vector<uint8_t> &bytes) {
    std::ostringstream out;
    for (size_t index = 0; index < bytes.size(); ++index) {
        out << (index ? " " : "") << std::hex << std::setw(2) << std::setfill('0')
            << static_cast<int>(bytes[index]);
    }
    return out.str();
}

}  // namespace

int main(int argc, char **argv) {
    return examples::run(argc, argv, "Send raw bytes to end effectors and poll for replies. By default, send hex 09 10 03 E8 00 03 06 01 00 00 00 00 00 72 E1 to both configured arm pass-through interfaces and poll for up to 1 s each. The attached device determines what these bytes do; this is not a generic motion command. Uses a Robot control connection: compatible arm modes may be enabled during startup, and shutdown requests a stop. Real hardware is selected unless --simulated is supplied.",
                         [](const examples::Args &args) {
        const auto side = args.choice("side", {"left", "right", "both"}, "both");
        const auto payload = parse_hex(args.text("message-hex", "09 10 03 E8 00 03 06 01 00 00 00 00 00 72 E1"));
        const double timeout = args.number("timeout", 1.0);
        const std::vector<std::string> sides =
            side == "both" ? std::vector<std::string>{"left", "right"} : std::vector<std::string>{side};

        auto robot = args.connect();
        for (const auto &name : sides) {
            auto arm = robot.joints(name + "_arm");
            if (!arm.ee_pass_through_enabled()) {
                std::cout << name << "_arm: pass-through endpoints are not configured\n";
                continue;
            }
            arm.ee_pass_through_send(payload);
            const auto started = std::chrono::steady_clock::now();
            bool answered = false;
            while (examples::elapsed(started) < timeout) {
                if (const auto response = arm.ee_pass_through_latest()) {
                    std::cout << name << "_arm: " << hex(*response) << "\n";
                    answered = true;
                    break;
                }
                examples::sleep_for(0.01);
            }
            if (!answered) {
                std::cout << name << "_arm: no response within " << std::fixed << std::setprecision(2)
                          << timeout << "s\n";
            }
        }
    }, {"message-hex", "side", "timeout"});
}

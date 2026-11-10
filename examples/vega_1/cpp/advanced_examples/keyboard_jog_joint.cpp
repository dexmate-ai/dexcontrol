// Copyright (C) 2026 Dexmate Inc.
// SPDX-License-Identifier: Apache-2.0

// Jog a selected joint using keyboard commands.
//
// Default component is left_arm and selected joint is 0. w/s move in opposite directions, digits select a joint, and q exits. Interaction and command increments differ by language; see the language notes.
//
// Same hold/repeat and tap behaviour, with an optional duration limit; raw-terminal input on Linux.
//
// Uses a Robot control connection: compatible arm modes may be enabled during startup, and shutdown requests a stop. Real hardware is selected unless --simulated is supplied.

#include "example.hpp"

#ifdef __unix__
#include <sys/select.h>
#include <termios.h>
#include <unistd.h>

#include <algorithm>

namespace {

// Terminal auto-repeat is typically 30-50 ms; this is a safe margin above
// that, and the longest the joint keeps moving after the key is released.
constexpr double HOLD_TIMEOUT = 0.15;
// Remaining travel kept when the key is released. Latching the setpoint to
// the measured position instead cancelled a single tap outright: in the 150 ms
// before the release is inferred the arm has barely started moving. This is
// also how far a held joint can still travel after release.
constexpr double RELEASE_LEAD = 0.05;
// How far the commanded setpoint may run ahead of the measured position.
constexpr double MAX_LEAD = 0.2;  // rad

// Raw terminal mode for the lifetime of the object; restores on exit.
class RawTerminal {
public:
    RawTerminal() : interactive_(isatty(STDIN_FILENO) != 0) {
        if (!interactive_) return;
        tcgetattr(STDIN_FILENO, &saved_);
        termios raw = saved_;
        // ISIG off too (like Python's tty.setraw): Ctrl-C arrives as a key so the
        // terminal is restored on the way out instead of left without echo.
        raw.c_lflag &= ~static_cast<tcflag_t>(ICANON | ECHO | ISIG);
        raw.c_cc[VMIN] = 0;
        raw.c_cc[VTIME] = 0;
        tcsetattr(STDIN_FILENO, TCSANOW, &raw);
    }
    ~RawTerminal() {
        if (interactive_) tcsetattr(STDIN_FILENO, TCSADRAIN, &saved_);
    }
    RawTerminal(const RawTerminal &) = delete;
    RawTerminal &operator=(const RawTerminal &) = delete;

    // Every key pressed since the last call, without blocking.
    std::vector<char> drain() const {
        std::vector<char> keys;
        fd_set set;
        timeval zero{0, 0};
        for (;;) {
            FD_ZERO(&set);
            FD_SET(STDIN_FILENO, &set);
            if (select(STDIN_FILENO + 1, &set, nullptr, nullptr, &zero) <= 0) break;
            char key = 0;
            if (read(STDIN_FILENO, &key, 1) != 1) break;
            keys.push_back(key);
        }
        return keys;
    }

private:
    bool interactive_;
    termios saved_{};
};

}  // namespace

int main(int argc, char **argv) {
    return examples::run(argc, argv, "Jog a selected joint using keyboard commands. Default component is left_arm and selected joint is 0. w/s move in opposite directions, digits select a joint, and q exits. Interaction and command increments differ by language; see the language notes. Same hold/repeat and tap behaviour, with an optional duration limit; raw-terminal input on Linux. Uses a Robot control connection: compatible arm modes may be enabled during startup, and shutdown requests a stop. Real hardware is selected unless --simulated is supplied.",
                         [](const examples::Args &args) {
        const auto component_name = args.text("component", "left_arm");
        int selected = args.integer("joint", 0);
        const double speed = args.number("speed", 0.2);
        const double control_hz = args.number("control-hz", 100.0);
        const double duration = args.number("duration", 0.0);
        if (speed <= 0.0 || control_hz <= 0.0) {
            examples::Args::fail("--speed and --control-hz must be positive");
        }

        auto robot = args.connect();
        auto component = robot.joints(component_name);
        const auto names = component.joint_names();
        const auto count = names.size();
        if (selected < 0 || static_cast<size_t>(selected) >= count) {
            examples::Args::fail("--joint must be in [0, " + std::to_string(count - 1) + "]");
        }
        std::vector<double> lower(count, -INFINITY), upper(count, INFINITY), step(count, speed / control_hz);
        if (const auto limits = component.joint_limits()) {
            lower = limits->lower;
            upper = limits->upper;
            // Never jog faster than the joint itself allows.
            for (size_t index = 0; index < count; ++index) {
                step[index] = std::min(speed, limits->velocity[index]) / control_hz;
            }
        }
        auto target = component.get_joint_pos();
        int direction = 0;
        double last_key_time = 0.0;
        std::cout << component_name << ": " << count << " joints, " << speed << " rad/s\n"
                  << "  w / s      hold to move joint +/-\n"
                  << "  0-" << count - 1 << "        select joint\n"
                  << "  q          quit\n";

        dexcontrol::RateLimiter limiter(control_hz);
        const auto started = std::chrono::steady_clock::now();
        RawTerminal keyboard;
        for (;;) {
            if ((duration > 0.0 && examples::elapsed(started) >= duration) || examples::interrupted()) {
                std::cout << "\r\nstopping\r\n";
                return;
            }
            const double now = examples::elapsed(started);
            for (const char key : keyboard.drain()) {
                if (key == 'q' || key == 3) {  // q or Ctrl-C
                    std::cout << "\r\nstopping\r\n";
                    return;
                }
                if (key == 'w') {
                    direction = 1, last_key_time = now;
                } else if (key == 's') {
                    direction = -1, last_key_time = now;
                } else if (key >= '0' && key <= '9' && static_cast<size_t>(key - '0') < count) {
                    selected = key - '0', direction = 0;
                    // Re-seed: the old setpoint belongs to the old joint.
                    target = component.get_joint_pos();
                }
            }
            const auto measured = component.get_joint_pos();
            if (direction != 0 && now - last_key_time > HOLD_TIMEOUT) {
                direction = 0;
                // Cut the lead once on release, then hold that setpoint.
                // Re-reading every idle tick would follow the joint's own sag
                // under gravity and command it progressively downward.
                for (size_t index = 0; index < count; ++index) {
                    target[index] = std::clamp(target[index], measured[index] - RELEASE_LEAD,
                                               measured[index] + RELEASE_LEAD);
                }
            }
            if (direction != 0) target[selected] += direction * step[selected];
            for (size_t index = 0; index < count; ++index) {
                target[index] = std::clamp(target[index], std::max(lower[index], measured[index] - MAX_LEAD),
                                           std::min(upper[index], measured[index] + MAX_LEAD));
            }
            component.set_joint_pos(target);
            const char arrow = direction > 0 ? '+' : direction < 0 ? '-' : ' ';
            std::cout << "\r[" << arrow << "] " << std::setw(10) << names[selected] << " " << std::showpos
                      << std::fixed << std::setprecision(3) << measured[selected] << " rad (limit "
                      << std::setprecision(2) << lower[selected] << " .. " << upper[selected] << ")   "
                      << std::noshowpos << std::flush;
            limiter.sleep();
            if (examples::interrupted()) throw std::runtime_error("interrupted");
        }
        // Stop and disconnect deliberately: a stop that cannot be delivered
        // is reported here rather than left to the handle's destructor.
        robot.close();
    }, {"component", "control-hz", "duration", "joint", "speed"});
}
#else
int main() {
    std::cerr << "keyboard_jog_joint needs a POSIX terminal\n";
    return 1;
}
#endif

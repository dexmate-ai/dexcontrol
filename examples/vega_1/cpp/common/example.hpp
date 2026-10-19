// Copyright (C) 2026 Dexmate Inc.
//
// This software is dual-licensed:
//
// 1. GNU Affero General Public License v3.0 (AGPL-3.0)
//    See LICENSE-AGPL for details
//
// 2. Commercial License
//    For commercial licensing terms, contact: contact@dexmate.ai

#pragma once

/* Shared scaffolding for the C++ examples: a minimal argument reader, the
 * connection options every example accepts, and a `run` wrapper that turns
 * a thrown dexcontrol::Error into an exit status.
 *
 * Every example accepts:
 *   --profile NAME        built-in robot profile (default: from ROBOT_NAME)
 *   --config FILE         a custom robot YAML/JSON file instead of a profile
 *   --enable-sensor NAME  enable a configured sensor; repeatable
 *   --simulated           drive the in-process simulation (dry run, no robot)
 *   --help                print the description and exit
 */

#include <dexcontrol/dexcontrol.hpp>

#include <chrono>
#include <climits>
#include <cmath>
#include <csignal>
#include <cstdio>
#include <cstdlib>
#include <exception>
#include <functional>
#include <initializer_list>
#include <iomanip>
#include <iostream>
#include <set>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

namespace examples {

namespace detail {
inline volatile std::sig_atomic_t interrupt_flag = 0;
inline void on_interrupt(int) { interrupt_flag = 1; }
}  // namespace detail

/* True once SIGINT (Ctrl-C) arrived; `run` installs the handler. Loops
 * without a fixed bound poll this and exit cleanly, the way the Python
 * scripts catch KeyboardInterrupt. */
inline bool interrupted() { return detail::interrupt_flag != 0; }

/* `--key value` options, `--flag` switches and bare positionals, read on
 * demand. `--key=value` is accepted as well.
 *
 * Boolean flags never take the next token as a value, so `--simulated
 * disable` is the flag plus the positional `disable`, never a live run with
 * a swallowed argument. The flags every example uses are registered below;
 * an example with a new one calls `Args::declare_flag` before constructing
 * `Args`. */
class Args {
public:
    static std::set<std::string> &boolean_flags() {
        static std::set<std::string> flags{
            "help",          "simulated",       "unfold",          "visualize",
            "no-confirm",    "no-zero-force", "no-display",   "passthrough", "relative",
            "velocity-compensation", "all",     "yes",
        };
        return flags;
    }
    static void declare_flag(const std::string &name) { boolean_flags().insert(name); }

    Args(int argc, char **argv, std::string description,
         std::initializer_list<const char *> task_options = {})
        : description_(std::move(description)) {
        for (int index = 1; index < argc; ++index) {
            std::string token = argv[index];
            if (token.rfind("--", 0) != 0 || token.size() == 2) {
                positionals_.push_back(token);
                continue;
            }
            const auto equals = token.find('=');
            if (equals != std::string::npos) {
                options_.emplace_back(token.substr(2, equals - 2), token.substr(equals + 1));
                continue;
            }
            const std::string key = token.substr(2);
            const bool is_flag = boolean_flags().count(key) != 0;
            const bool has_value = index + 1 < argc && !looks_like_option(argv[index + 1]);
            if (!is_flag && has_value) {
                options_.emplace_back(key, argv[++index]);
            } else {
                options_.emplace_back(key, "");
            }
        }
        std::set<std::string> known{"help", "profile", "config", "enable-sensor", "simulated"};
        known.insert(task_options.begin(), task_options.end());
        std::set<std::string> seen_flags;
        for (const auto &[key, value] : options_) {
            if (!known.count(key)) fail("unknown option --" + key);
            if (boolean_flags().count(key)) {
                if (!seen_flags.insert(key).second) fail("duplicate flag --" + key);
                (void)flag(key);
            } else if (value.empty()) {
                fail("--" + key + " requires a value");
            }
        }
        if (flag("help")) {
            std::cout << description_ << "\n\n"
                      << "Common options: --profile NAME, --config FILE, "
                         "--enable-sensor NAME (repeatable), --simulated.\n"
                         "Example-specific options are listed in the source file.\n";
            std::cout << "Task options:";
            for (const auto *name : task_options) std::cout << " --" << name;
            std::cout << "\n";
            std::exit(0);
        }
    }

    bool flag(const std::string &name) const {
        for (const auto &[key, value] : options_) {
            if (key == name) {
                if (value.empty() || value == "true" || value == "1") return true;
                if (value == "false" || value == "0") return false;
                fail("--" + name + " expects true, false, 1, or 0");
            }
        }
        return false;
    }
    std::string text(const std::string &name, const std::string &fallback = {}) const {
        for (const auto &[key, value] : options_) {
            if (key == name) return value;
        }
        return fallback;
    }
    double number(const std::string &name, double fallback) const {
        const auto value = text(name);
        if (value.empty()) return fallback;
        try {
            size_t consumed = 0;
            const auto number = std::stod(value, &consumed);
            if (consumed != value.size() || !std::isfinite(number))
                fail("--" + name + " expects a finite number");
            return number;
        } catch (const std::exception &) {
            fail("--" + name + " expects a number, got '" + value + "'");
        }
    }
    int integer(const std::string &name, int fallback) const {
        const double value = number(name, fallback);
        if (!std::isfinite(value) || value < INT_MIN || value > INT_MAX || std::trunc(value) != value) {
            fail("--" + name + " must be an integer in [" + std::to_string(INT_MIN) + ", " +
                 std::to_string(INT_MAX) + "]");
        }
        return static_cast<int>(std::lround(value));
    }
    /* Every value given for a repeatable option, in order. */
    std::vector<std::string> list(const std::string &name) const {
        std::vector<std::string> values;
        for (const auto &[key, value] : options_) {
            if (key == name && !value.empty()) values.push_back(value);
        }
        return values;
    }
    /* A comma-separated list of numbers, e.g. --p 1,1,1,1,1,1,1. */
    std::vector<double> numbers(const std::string &name, std::vector<double> fallback = {}) const {
        const auto value = text(name);
        if (value.empty()) return fallback;
        std::vector<double> values;
        std::stringstream stream(value);
        std::string item;
        while (std::getline(stream, item, ',')) {
            try {
                size_t consumed = 0;
                const auto number = std::stod(item, &consumed);
                if (item.find_first_not_of(" \t\r\n", consumed) != std::string::npos || !std::isfinite(number))
                    fail("--" + name + " expects finite numbers");
                values.push_back(number);
            } catch (const std::exception &) {
                fail("--" + name + " expects comma-separated numbers, got '" + value + "'");
            }
        }
        return values;
    }
    /* An option restricted to a fixed set of values. */
    std::string choice(const std::string &name, std::initializer_list<const char *> allowed,
                       const std::string &fallback) const {
        return validate(name, text(name, fallback), allowed);
    }
    std::string positional(size_t index, const std::string &fallback = {}) const {
        return index < positionals_.size() ? positionals_[index] : fallback;
    }
    std::string positional_choice(size_t index, std::initializer_list<const char *> allowed,
                                  const std::string &fallback) const {
        return validate("argument " + std::to_string(index + 1), positional(index, fallback),
                        allowed);
    }
    /* "left" or "right", from --side. */
    std::string side(const std::string &fallback = "right") const {
        return choice("side", {"left", "right"}, fallback);
    }

    dexcontrol::ConnectOptions connect_options() const {
        dexcontrol::ConnectOptions options;
        options.profile = text("profile");
        options.config_file = text("config");
        options.enable_sensors = list("enable-sensor");
        options.simulated = flag("simulated");
        return options;
    }
    /* Connects with the common options, enabling `sensors` as well. */
    dexcontrol::Robot connect(const std::vector<std::string> &sensors = {}) const {
        auto options = connect_options();
        options.enable_sensors.insert(options.enable_sensors.end(), sensors.begin(), sensors.end());
        return dexcontrol::Robot::connect(options);
    }
    dexcontrol::DiagnosticClient diagnostics() const {
        return dexcontrol::DiagnosticClient::connect(connect_options());
    }

    [[noreturn]] static void fail(const std::string &message) {
        throw std::invalid_argument(message);
    }

private:
    /* `--name` is an option; a negative number such as `-0.2` is a value. */
    static bool looks_like_option(const char *token) {
        return token[0] == '-' && token[1] == '-';
    }
    static std::string validate(const std::string &what, const std::string &value,
                                std::initializer_list<const char *> allowed) {
        for (const char *option : allowed) {
            if (value == option) return value;
        }
        std::string choices;
        for (const char *option : allowed) choices += (choices.empty() ? "" : ", ") + std::string(option);
        fail(what + " must be one of: " + choices + " (got '" + value + "')");
    }
    std::string description_;
    std::vector<std::pair<std::string, std::string>> options_;
    std::vector<std::string> positionals_;
};

/* Runs an example body, mapping thrown errors to a non-zero exit status. */
inline int run(int argc, char **argv, const std::string &description,
               const std::function<void(const Args &)> &body,
               std::initializer_list<const char *> task_options = {}) {
    std::signal(SIGINT, detail::on_interrupt);
    try {
        const Args args(argc, argv, description, task_options);
        body(args);
        return 0;
    } catch (const dexcontrol::Error &error) {
        std::cerr << "dexcontrol error (status " << error.status() << "): " << error.what() << "\n";
    } catch (const std::exception &error) {
        if (interrupted() && std::string(error.what()) == "interrupted") return 0;
        std::cerr << "error: " << error.what() << "\n";
    }
    return 1;
}

inline void sleep_for(double seconds) {
    if (!std::isfinite(seconds) || seconds < 0) Args::fail("invalid sleep duration");
    while (seconds > 0.0) {
        if (interrupted()) throw std::runtime_error("interrupted");
        const double slice = std::min(seconds, 0.05);
        std::this_thread::sleep_for(std::chrono::duration<double>(slice));
        seconds -= slice;
    }
}

inline std::chrono::milliseconds millis(double seconds) {
    const long double value = static_cast<long double>(seconds) * 1000.0L;
    if (!std::isfinite(seconds) || seconds < 0 || value >= static_cast<long double>(LLONG_MAX))
        Args::fail("duration must be finite, non-negative, and representable in milliseconds");
    return std::chrono::milliseconds(static_cast<long long>(std::round(value)));
}

/* Seconds since `since`, for elapsed-time prints. */
inline double elapsed(std::chrono::steady_clock::time_point since) {
    return std::chrono::duration<double>(std::chrono::steady_clock::now() - since).count();
}

/* "[0.000, 0.100, ...]" with a fixed precision. */
template <typename Container>
std::string format(const Container &values, int precision = 3) {
    std::ostringstream out;
    out << std::fixed << std::setprecision(precision) << "[";
    bool first = true;
    for (const auto &value : values) {
        out << (first ? "" : ", ") << value;
        first = false;
    }
    out << "]";
    return out.str();
}

/* Gate for an action that can hurt someone or drop hardware (releasing a
 * brake, rebooting a board under load): prints `warning` and continues only
 * after the operator types the whole word `yes`. A single keystroke is not
 * enough on purpose. `--yes` skips the prompt for scripted use; `--simulated`
 * runs have nothing to protect and never prompt. Throws when declined (or
 * when stdin is closed), so the example exits non-zero without acting. */
inline void confirm_hazard(const Args &args, const std::string &warning) {
    if (args.flag("simulated") || args.flag("yes")) return;
    std::cout << "WARNING: " << warning << "\nType 'yes' to continue: " << std::flush;
    std::string answer;
    if (!std::getline(std::cin, answer) || answer != "yes") {
        Args::fail("aborted: confirmation was not given (pass --yes to skip the prompt)");
    }
}

/* Asks the operator for confirmation on stdin; false on anything but y/Y. */
inline bool confirm(const std::string &prompt) {
    std::cout << prompt << " [y/N]: " << std::flush;
    std::string answer;
    std::getline(std::cin, answer);
    return answer == "y" || answer == "Y";
}

}  // namespace examples

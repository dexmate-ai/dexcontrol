// Copyright (C) 2026 Dexmate Inc.
// SPDX-License-Identifier: Apache-2.0

#pragma once

/* Shared helpers for the arm tracking benchmarks: the zero pose, its
 * verification, result directories, CSV output and tracking statistics. */

#include "example.hpp"

#include <array>
#include <ctime>
#include <filesystem>
#include <fstream>

namespace arm_tracking {

constexpr size_t NUM_JOINTS = 7;
constexpr double PI = 3.14159265358979323846;
const std::vector<double> ZERO_POS{0.0, 0.0, 0.0, -0.5, 0.0, 0.0, 0.0};
constexpr double ZERO_TOLERANCE = 0.05;  // rad
constexpr double LIMIT_MARGIN = 0.02;     // rad kept clear of a joint limit by test excursions
constexpr double VELOCITY_MARGIN = 0.02;  // fraction of the velocity limit kept clear by feed-forward

/* Amplitude of a test excursion from ZERO_POS that stays within the joint's
 * limits. The client refuses a target outside the model's joint limits, and
 * the reference pose sits close to a limit on some joints (R_arm_j2 has
 * 0.45 rad of room upward for a pi/6 step). For a step (symmetric=false) the
 * signed step is returned: the requested direction if it fits, the other
 * direction if only that fits, otherwise the largest step that fits. For a
 * sine (symmetric=true) the magnitude is returned, limited by the smaller of
 * the two sides and, given the angular frequency omega (rad/s), by the
 * joint's velocity limit: the feed-forward peaks at amplitude * omega and the
 * client refuses one above the limit (0.4 rad at 1 Hz is 2.51 rad/s against
 * R_arm_j1's 2.4 rad/s). Every change is printed. */
inline double fit_amplitude(const dexcontrol::JointComponent &arm, size_t joint, double amplitude,
                            bool symmetric, double omega = 0.0) {
    const auto limits = arm.joint_limits();
    if (!limits) return amplitude;
    const auto name = arm.joint_names()[joint];
    double size = std::fabs(amplitude);
    if (symmetric && omega > 0.0 && std::isfinite(limits->velocity[joint])) {
        const double room = limits->velocity[joint] * (1.0 - VELOCITY_MARGIN) / omega;
        if (size > room) {
            std::cout << "WARNING: " << name << ": sine amplitude reduced from " << size << " to " << room
                      << " rad so the velocity feed-forward stays within " << limits->velocity[joint]
                      << " rad/s\n";
            size = room;
        }
    }
    const double lower = limits->lower[joint], upper = limits->upper[joint];
    const double zero = ZERO_POS[joint];
    const double room_up = std::max(0.0, upper - zero - LIMIT_MARGIN);
    const double room_down = std::max(0.0, zero - lower - LIMIT_MARGIN);
    if (symmetric) {
        const double room = std::min(room_up, room_down);
        if (size <= room) return size;
        std::cout << "WARNING: " << name << ": sine amplitude reduced from " << size << " to " << room
                  << " rad to stay within [" << lower << ", " << upper << "]\n";
        return room;
    }
    const bool wanted_up = amplitude >= 0.0;
    if (size <= (wanted_up ? room_up : room_down)) return amplitude;
    if (size <= (wanted_up ? room_down : room_up)) {
        std::cout << "WARNING: " << name << ": stepping " << (wanted_up ? "negative" : "positive")
                  << " instead; " << amplitude << " rad from the reference would leave [" << lower << ", "
                  << upper << "]\n";
        return wanted_up ? -size : size;
    }
    const double best = room_up >= room_down ? room_up : -room_down;
    std::cout << "WARNING: " << name << ": step reduced to " << best << " rad to stay within [" << lower
              << ", " << upper << "]\n";
    return best;
}

/* One recorded loop sample: what was commanded and what was measured. */
struct Sample {
    double time_s;
    std::vector<double> command;
    std::vector<double> actual;
};

/* Reads the current joint positions and verifies they match ZERO_POS. */
inline size_t sample_count(double seconds, double hz) {
    const double count = seconds * hz;
    if (!std::isfinite(seconds) || seconds < 0 || !std::isfinite(hz) || hz <= 0 || !std::isfinite(count) || count > 1000000)
        throw std::invalid_argument("benchmark requires non-negative duration, positive rate, and at most 1000000 samples");
    return static_cast<size_t>(count);
}

inline void verify_zero_position(const dexcontrol::JointComponent &arm) {
    const auto actual = arm.get_joint_pos();
    if (actual.size() != NUM_JOINTS) {
        throw std::runtime_error("expected a 7-joint arm, got " + std::to_string(actual.size()) +
                                 " joints");
    }
    std::vector<double> errors(NUM_JOINTS);
    std::string failed;
    for (size_t joint = 0; joint < NUM_JOINTS; ++joint) {
        errors[joint] = std::fabs(actual[joint] - ZERO_POS[joint]);
        if (errors[joint] > ZERO_TOLERANCE) {
            std::ostringstream detail;
            detail << std::fixed << std::setprecision(4) << "joint " << joint << ": actual=" << actual[joint]
                   << ", err=" << errors[joint] << " rad";
            failed += (failed.empty() ? "" : ", ") + detail.str();
        }
    }
    std::cout << "Actual joint positions: " << examples::format(actual, 4) << "\n";
    std::cout << "Position errors:        " << examples::format(errors, 4) << "\n";
    if (!failed.empty()) {
        std::ostringstream message;
        message << "Zero-position check failed (tolerance=" << ZERO_TOLERANCE << " rad). Failed joints: "
                << failed;
        throw std::runtime_error(message.str());
    }
    std::cout << "Zero position verified (all joints within tolerance)\n";
}

/* Moves the arm to ZERO_POS along a planned path and waits for it. */
inline void go_to_zero(dexcontrol::JointComponent &arm, double timeout_s) {
    arm.move_to_joint_pos(ZERO_POS).wait(dexcontrol::WaitFor{examples::millis(timeout_s)});
}

/* <output_dir>/<timestamp>_<side>_<kind> relative to the working directory,
 * created. */
inline std::filesystem::path result_directory(const std::string &output_dir, const std::string &side,
                                              const std::string &kind) {
    const std::time_t now = std::time(nullptr);
    char stamp[32];
    std::tm local{};
    localtime_r(&now, &local);
    std::strftime(stamp, sizeof stamp, "%Y%m%d_%H%M%S", &local);
    const auto base = std::filesystem::path(output_dir);
    const auto directory = base / (std::string(stamp) + "_" + side + "_" + kind);
    std::filesystem::create_directories(directory);
    return directory;
}

/* One CSV per joint: time, the seven commands, the seven measurements. */
inline void write_csv(const std::filesystem::path &path, const std::vector<Sample> &samples) {
    std::ofstream out(path);
    out << "time_s";
    for (size_t joint = 0; joint < NUM_JOINTS; ++joint) out << ",cmd_" << joint;
    for (size_t joint = 0; joint < NUM_JOINTS; ++joint) out << ",actual_" << joint;
    out << "\n" << std::setprecision(9);
    for (const auto &sample : samples) {
        out << sample.time_s;
        for (const double value : sample.command) out << "," << value;
        for (const double value : sample.actual) out << "," << value;
        out << "\n";
    }
}

/* Mean and peak absolute tracking error of the tested joint. */
inline void print_tracking_error(size_t joint, const std::vector<Sample> &samples) {
    double total = 0.0, peak = 0.0;
    for (const auto &sample : samples) {
        const double error = std::fabs(sample.command[joint] - sample.actual[joint]);
        total += error;
        peak = std::max(peak, error);
    }
    const double mean = samples.empty() ? 0.0 : total / static_cast<double>(samples.size());
    std::cout << std::fixed << std::setprecision(4) << "joint " << joint << ": samples=" << samples.size()
              << " mean|err|=" << mean << " rad max|err|=" << peak << " rad\n";
}

}  // namespace arm_tracking

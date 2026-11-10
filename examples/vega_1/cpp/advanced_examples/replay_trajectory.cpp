// Copyright (C) 2026 Dexmate Inc.
// SPDX-License-Identifier: Apache-2.0

// Prepare robot joint positions and replay a recorded trajectory.
//
// Require a trajectory file and confirmation before motion. Move to preparation/start targets, then stream recorded joint positions with the configured tracking-error guard. Preparation motions and processing options differ by language; see the language notes.
//
// CSV input. Performs the same extra folding/head/torso/hand preparation as Python. Supports smoothing, resampling and velocity feed-forward; --visualize prints a summary instead of a plot. Prompts before preparation and playback.
//
// Uses a Robot control connection: compatible arm modes may be enabled during startup, and shutdown requests a stop. Real hardware is selected unless --simulated is supplied.

#include "example.hpp"

#include <algorithm>
#include <fstream>
#include <map>

namespace {

using Track = std::vector<std::vector<double>>;  // [tick][joint]
using Trajectory = std::map<std::string, Track>;

std::pair<Trajectory, double> load_trajectory(const std::string &path) {
    std::ifstream file(path);
    if (!file) throw std::runtime_error("File not found: " + path);
    std::string line;
    double control_hz = 500.0;  // default
    if (std::getline(file, line) && line.rfind("control_hz,", 0) == 0) {
        control_hz = std::stod(line.substr(11));
        std::cout << "Found control frequency: " << control_hz << "Hz\n";
        std::getline(file, line);
    }
    // Header: component:joint_index per column, grouped by component.
    std::vector<std::pair<std::string, size_t>> columns;
    std::map<std::string, size_t> widths;
    std::stringstream header(line);
    std::string column;
    while (std::getline(header, column, ',')) {
        const auto colon = column.find(':');
        if (colon == std::string::npos) throw std::runtime_error("bad header column: " + column);
        const auto part = column.substr(0, colon);
        columns.emplace_back(part, std::stoul(column.substr(colon + 1)));
        widths[part] = std::max(widths[part], columns.back().second + 1);
    }
    Trajectory trajectory;
    while (std::getline(file, line)) {
        if (line.empty()) continue;
        std::stringstream row(line);
        std::string cell;
        std::map<std::string, std::vector<double>> tick;
        for (auto &[part, width] : widths) tick[part].assign(width, 0.0);
        for (const auto &[part, joint] : columns) {
            if (!std::getline(row, cell, ',')) throw std::runtime_error("short row in " + path);
            tick[part][joint] = std::stod(cell);
        }
        for (auto &[part, values] : tick) trajectory[part].push_back(std::move(values));
    }
    for (const auto &[part, rows] : trajectory) {
        std::cout << "Loaded " << part << ": shape (" << rows.size() << ", " << widths[part] << ")\n";
    }
    return {trajectory, control_hz};
}

// Linear resampling by speed factor (>1 = faster, <1 = slower).
Trajectory resample(const Trajectory &trajectory, double speed_factor) {
    if (speed_factor == 1.0) return trajectory;
    Trajectory resampled;
    for (const auto &[part, rows] : trajectory) {
        const size_t old_len = rows.size();
        const auto new_len = static_cast<size_t>(std::ceil(old_len / speed_factor));
        Track out(new_len, std::vector<double>(rows.front().size()));
        for (size_t index = 0; index < new_len; ++index) {
            const double position = new_len > 1 ? index * double(old_len - 1) / double(new_len - 1) : 0.0;
            const auto low = static_cast<size_t>(std::floor(position));
            const size_t high = std::min(low + 1, old_len - 1);
            const double weight = position - double(low);
            for (size_t joint = 0; joint < out[index].size(); ++joint) {
                out[index][joint] = rows[low][joint] * (1.0 - weight) + rows[high][joint] * weight;
            }
        }
        resampled[part] = std::move(out);
    }
    return resampled;
}

// Gaussian smoothing along time with nearest-edge padding (like
// scipy.ndimage.gaussian_filter1d with mode="nearest", truncate=4).
Track gaussian_smooth(const Track &rows, double sigma_samples) {
    if (sigma_samples <= 0.0 || rows.size() < 2) return rows;
    const auto radius = static_cast<long>(std::lround(4.0 * sigma_samples));
    std::vector<double> kernel(2 * radius + 1);
    double total = 0.0;
    for (long offset = -radius; offset <= radius; ++offset) {
        kernel[offset + radius] = std::exp(-0.5 * (offset * offset) / (sigma_samples * sigma_samples));
        total += kernel[offset + radius];
    }
    for (auto &weight : kernel) weight /= total;
    const long ticks = static_cast<long>(rows.size());
    Track out(rows.size(), std::vector<double>(rows.front().size(), 0.0));
    for (long tick = 0; tick < ticks; ++tick) {
        for (long offset = -radius; offset <= radius; ++offset) {
            const long source = std::clamp(tick + offset, 0L, ticks - 1);
            const double weight = kernel[offset + radius];
            for (size_t joint = 0; joint < out[tick].size(); ++joint) {
                out[tick][joint] += weight * rows[source][joint];
            }
        }
    }
    return out;
}

Trajectory smooth(const Trajectory &trajectory, double sigma_time, double hz) {
    if (sigma_time <= 0.0) return trajectory;
    const double sigma_samples = sigma_time * hz;
    std::cout << "Smoothing with " << std::fixed << std::setprecision(3) << sigma_time << "s window = "
              << std::setprecision(1) << sigma_samples << " samples at " << hz << "Hz\n";
    Trajectory smoothed;
    for (const auto &[part, rows] : trajectory) smoothed[part] = gaussian_smooth(rows, sigma_samples);
    return smoothed;
}

// Finite-difference velocities after optional smoothing.
Trajectory velocities_of(const Trajectory &trajectory, double hz, double smooth_time) {
    const double dt = 1.0 / hz;
    Trajectory velocities;
    for (const auto &[part, original] : trajectory) {
        const size_t ticks = original.size();
        const size_t joints = original.front().size();
        Track velocity(ticks, std::vector<double>(joints, 0.0));
        if (ticks >= 2) {
            const Track rows = smooth_time > 0.0 ? gaussian_smooth(original, smooth_time * hz) : original;
            for (size_t joint = 0; joint < joints; ++joint) {
                velocity[0][joint] = (rows[1][joint] - rows[0][joint]) / dt;
                for (size_t tick = 1; tick + 1 < ticks; ++tick) {
                    velocity[tick][joint] = (rows[tick + 1][joint] - rows[tick - 1][joint]) / (2.0 * dt);
                }
                velocity[ticks - 1][joint] = (rows[ticks - 1][joint] - rows[ticks - 2][joint]) / dt;
            }
        }
        velocities[part] = std::move(velocity);
    }
    return velocities;
}

// A textual stand-in for the Python plots: per-joint ranges.
void summarize(const Trajectory &trajectory, const Trajectory *velocities, double hz) {
    for (const auto &[part, rows] : trajectory) {
        std::cout << part << " (" << rows.size() / hz << "s)\n";
        for (size_t joint = 0; joint < rows.front().size(); ++joint) {
            double low = INFINITY, high = -INFINITY, peak = 0.0;
            for (size_t tick = 0; tick < rows.size(); ++tick) {
                low = std::min(low, rows[tick][joint]);
                high = std::max(high, rows[tick][joint]);
                if (velocities) peak = std::max(peak, std::fabs(velocities->at(part)[tick][joint]));
            }
            std::cout << "  joint " << joint + 1 << ": position " << std::fixed << std::setprecision(3) << low
                      << " .. " << high << " rad";
            if (velocities) std::cout << ", |velocity| <= " << peak << " rad/s";
            std::cout << "\n";
        }
    }
}

// Validates, moves to the first frame, and streams the trajectory natively.
// The executor preflights every waypoint before publishing and checks the
// cached tracking error on every tick when max_goal_diff is set.
void run_replay(dexcontrol::Robot &robot, const Trajectory &trajectory, const Trajectory *velocities, double hz,
                double max_goal_diff) {
    Trajectory available;
    for (const auto &[part, rows] : trajectory) {
        if (robot.has_component(part)) available[part] = rows;
    }
    if (available.empty()) throw std::runtime_error("trajectory contains no components available on this robot");
    dexcontrol::Robot::Targets start;
    for (const auto &[part, rows] : available) start[part] = rows.front();
    robot.move_to_joint_pos(start).wait(dexcontrol::WaitFor{examples::millis(3.0)});
    Trajectory tracked_velocities;
    if (velocities) {
        for (const auto &[part, rows] : *velocities) {
            if (available.count(part)) tracked_velocities[part] = rows;
        }
    }
    robot.execute_trajectory(available, hz, tracked_velocities, max_goal_diff);
}

}  // namespace

int main(int argc, char **argv) {
    return examples::run(argc, argv, "Prepare robot joint positions and replay a recorded trajectory. Require a trajectory file and confirmation before motion. Move to preparation/start targets, then stream recorded joint positions with the configured tracking-error guard. Preparation motions and processing options differ by language; see the language notes. CSV input. Performs the same extra folding/head/torso/hand preparation as Python. Supports smoothing, resampling and velocity feed-forward; --visualize prints a summary instead of a plot. Prompts before preparation and playback. Uses a Robot control connection: compatible arm modes may be enabled during startup, and shutdown requests a stop. Real hardware is selected unless --simulated is supplied.",
                         [](const examples::Args &args) {
        const auto path = args.positional(0);
        if (path.empty()) examples::Args::fail("a trajectory CSV file is required");
        const double control_hz_override = args.number("control-hz", 0.0);
        const double gaussian_sigma = args.number("smooth", 0.1);
        const bool use_velocity = args.flag("velocity-compensation");
        const double velocity_sigma = args.number("vel-smooth", 0.5);
        double speed_factor = args.number("speed-factor", 1.0);
        const bool visualize = args.flag("visualize");
        const double max_goal_diff = args.number("max-goal-diff", 1.0);

        std::cout << "Warning: Be ready to press e-stop if needed! This example does not check for "
                     "self-collisions.\n"
                     "Please ensure the arms and the torso have sufficient space to move.\n";
        if (!examples::confirm("Continue?")) return;
        if (path.find("vega-1_dance") != std::string::npos) {
            const double clamped = std::clamp(speed_factor, 0.2, 3.0);
            if (clamped != speed_factor) {
                std::cout << "Speed factor clamped to " << clamped << " (valid range: 0.2-3.0)\n";
                speed_factor = clamped;
            }
        }
        if (speed_factor > 3.0) {
            std::cout << "Speed factor is greater than 3.0!!! This can be dangerous!!!\n";
            if (!examples::confirm("Continue?")) return;
        }

        auto [trajectory, file_hz] = load_trajectory(path);
        if (trajectory.empty()) {
            std::cerr << "Empty trajectory\n";
            return;
        }
        const double hz = control_hz_override > 0.0 ? control_hz_override : file_hz;
        const size_t original_frames = trajectory.begin()->second.size();
        if (speed_factor != 1.0) {
            std::cout << "Applying speed factor " << speed_factor << "x\n";
            trajectory = resample(trajectory, speed_factor);
        }
        const size_t num_frames = trajectory.begin()->second.size();
        std::cout << std::fixed << std::setprecision(1) << "Original: " << original_frames << " frames, "
                  << original_frames / hz << "s\n"
                  << "Playback: " << num_frames << " frames @ " << hz << "Hz (" << num_frames / hz << "s)\n"
                  << "Parts:";
        for (const auto &[part, rows] : trajectory) std::cout << " " << part;
        std::cout << "\n";
        if (gaussian_sigma > 0.0) {
            std::cout << "Applying Gaussian smoothing (sigma=" << gaussian_sigma << ")\n";
            trajectory = smooth(trajectory, gaussian_sigma, hz);
        }
        Trajectory velocities;
        if (use_velocity) {
            std::cout << "Computing velocities (sigma=" << velocity_sigma << ")\n";
            velocities = velocities_of(trajectory, hz, velocity_sigma);
        }
        if (visualize) {
            summarize(trajectory, use_velocity ? &velocities : nullptr, hz);
            if (!examples::confirm("Continue with execution?")) {
                std::cout << "Execution cancelled\n";
                return;
            }
        }

        auto robot = args.connect();
        std::cout << "Setting joint positions...\n"
                     "Press e-stop if needed!\n"
                     "Please ensure the arms and the torso have sufficient space to move.\n"
                     "If you have an end effector attached, some pre-existing trajectories may cause "
                     "collisions with the robot.\n"
                     "Will move the left arm to folded, the right arm to folded, and the head to home "
                     "pose, then the torso to crouch20_medium and head home again.\n";
        if (!examples::confirm("Continue?")) {
            std::cout << "Execution cancelled\n";
            return;
        }
        dexcontrol::Robot::Targets init_pose{
            {"left_arm", robot.joints("left_arm").resolve_pose("folded")},
            {"right_arm", robot.joints("right_arm").resolve_pose("folded")},
        };
        if (robot.has_component("head")) init_pose["head"] = robot.joints("head").resolve_pose("home");
        robot.move_to_joint_pos(init_pose).wait(dexcontrol::WaitFor{examples::millis(5.0)});
        if (robot.has_component("torso")) {
            robot.joints("torso").go_to_pose("crouch20_medium", dexcontrol::WaitFor{examples::millis(5.0)});
            if (robot.has_component("head")) robot.joints("head").go_to_pose("home", dexcontrol::WaitFor{examples::millis(5.0)});
        }
        for (const std::string side : {"left", "right"}) {
            if (robot.has_component(side + "_hand")) robot.joints(side + "_hand").close_hand();
        }
        std::cout << "Press Enter to start the replay..." << std::flush;
        std::string line;
        std::getline(std::cin, line);
        run_replay(robot, trajectory, use_velocity ? &velocities : nullptr, hz, max_goal_diff);
        std::cout << "Done\n";
        // Stop and disconnect deliberately: a stop that cannot be delivered
        // is reported here rather than left to the handle's destructor.
        robot.close();
    }, {"control-hz", "max-goal-diff", "smooth", "speed-factor", "vel-smooth", "velocity-compensation", "visualize"},
       "path: trajectory file to replay");
}

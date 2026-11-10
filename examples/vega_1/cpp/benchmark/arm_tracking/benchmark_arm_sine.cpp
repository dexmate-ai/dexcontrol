// Copyright (C) 2026 Dexmate Inc.
// SPDX-License-Identifier: Apache-2.0

// Benchmark each arm joint with a sine trajectory and velocity feed-forward.
//
// Prompt for confirmation unless --no-confirm; move the right seven-joint arm to [0,0,0,-0.5,0,0,0] rad. Test each joint for 10 s at 200 Hz with a 0.4 rad, 1 Hz sine; return to the reference between joints and at the end. Write measurement files. This reference is not an all-zero pose. The amplitude is fitted to each joint's limits: the position limits from the reference pose and the velocity limit against the feed-forward peak (amplitude times angular frequency), reduced with a warning where the requested amplitude does not fit. Stream with the step guard disabled (tracking lag is what is measured).
//
// Writes per-joint CSV and parameters.txt under the requested output directory.
//
// Uses a Robot control connection: compatible arm modes may be enabled during startup, and shutdown requests a stop. Real hardware is selected unless --simulated is supplied.

#include "common.hpp"

int main(int argc, char **argv) {
    return examples::run(argc, argv, "Benchmark each arm joint with a sine trajectory and velocity feed-forward. Prompt for confirmation unless --no-confirm; move the right seven-joint arm to [0,0,0,-0.5,0,0,0] rad. Test each joint for 10 s at 200 Hz with a 0.4 rad, 1 Hz sine; return to the reference between joints and at the end. Write measurement files. This reference is not an all-zero pose. The amplitude is fitted to each joint's limits: the position limits from the reference pose and the velocity limit against the feed-forward peak (amplitude times angular frequency), reduced with a warning where the requested amplitude does not fit. Stream with the step guard disabled (tracking lag is what is measured). Writes per-joint CSV and parameters.txt under the requested output directory. Uses a Robot control connection: compatible arm modes may be enabled during startup, and shutdown requests a stop. Real hardware is selected unless --simulated is supplied.",
                         [](const examples::Args &args) {
        using namespace arm_tracking;
        const auto side = args.side();
        const double duration = args.number("duration", 10.0);
        const double control_hz = args.number("control-hz", 200.0);
        const double requested_amplitude = args.number("amplitude", 0.4);
        const double frequency = args.number("sin-frequency", 1.0);
        const auto output_dir = args.text("output-dir", "results");

        std::cout << "WARNING: this benchmark moves the robot arm through sine trajectories. "
                     "No collision checking is performed. Ensure the workspace is clear. "
                     "Make sure the e-stop is accessible.\n";
        if (!args.flag("no-confirm") && !examples::confirm("Continue?")) {
            std::cout << "Aborted by user.\n";
            return;
        }

        if (sample_count(duration, control_hz) == 0) examples::Args::fail("benchmark requires at least one sample");
        auto robot = args.connect();
        auto arm = robot.joints(side + "_arm");
        const auto result_dir = result_directory(output_dir, side, "sin");
        std::cout << "Results will be saved to " << result_dir << "\n";

        std::cout << "Moving " << side << " arm to zero position\n";
        go_to_zero(arm, 5.0);
        // Settle before verifying. A real robot publishes state continuously;
        // the simulation only re-stamps state on a command, so a dry run
        // passes --settle 0 to stay inside the 500 ms freshness limit.
        examples::sleep_for(args.number("settle", 0.5));
        verify_zero_position(arm);

        const double omega = 2.0 * arm_tracking::PI * frequency;
        // Lag is what this benchmark measures: the step guard is off.
        const auto stream = dexcontrol::without_step_guard(dexcontrol::command_options());
        std::vector<std::vector<Sample>> all_samples;
        for (size_t joint = 0; joint < NUM_JOINTS; ++joint) {
            std::cout << "--- Testing joint " << joint << " ---\n";
            if (joint > 0) go_to_zero(arm, 3.0);
            // Fitted to the joint's position and velocity limits from the
            // reference pose (reduced with a warning where it does not fit).
            const double amplitude = fit_amplitude(arm, joint, requested_amplitude, true, omega);

            std::vector<Sample> samples;
            dexcontrol::RateLimiter limiter(control_hz);
            const auto started = std::chrono::steady_clock::now();
            while (examples::elapsed(started) < duration) {
                const double t = examples::elapsed(started);
                auto position = ZERO_POS;
                position[joint] += amplitude * std::sin(omega * t);
                std::vector<double> velocity(NUM_JOINTS, 0.0);
                velocity[joint] = amplitude * omega * std::cos(omega * t);

                arm.set_joint_pos(position, velocity, stream);
                samples.push_back({t, position, arm.get_joint_pos()});
                limiter.sleep();
            if (examples::interrupted()) throw std::runtime_error("interrupted");
            }
            print_tracking_error(joint, samples);
            write_csv(result_dir / ("joint_" + std::to_string(joint) + ".csv"), samples);
            all_samples.push_back(std::move(samples));
        }

        std::ofstream meta(result_dir / "parameters.txt");
        meta << "side=" << side << "\nduration=" << duration << "\ncontrol_hz=" << control_hz
             << "\namplitude=" << requested_amplitude << "\nsin_frequency=" << frequency
             << "\nzero_pos=" << examples::format(ZERO_POS) << "\n";
        std::cout << "Raw data saved to " << result_dir << "\n";

        std::cout << "Returning to zero position\n";
        go_to_zero(arm, 3.0);
        std::cout << "Benchmark complete\n";
        // Stop and disconnect deliberately: a stop that cannot be delivered
        // is reported here rather than left to the handle's destructor.
        robot.close();
    }, {"amplitude", "control-hz", "duration", "no-confirm", "output-dir", "settle", "side", "sin-frequency"});
}

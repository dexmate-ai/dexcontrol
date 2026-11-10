// Copyright (C) 2026 Dexmate Inc.
// SPDX-License-Identifier: Apache-2.0

// Benchmark each arm joint with a ramped step trajectory.
//
// Prompt unless --no-confirm; move the right seven-joint arm to [0,0,0,-0.5,0,0,0] rad. Test each joint for 5 s at 200 Hz: hold 0.3 s, ramp by pi/6 rad over 0.3 s, then hold. The step is fitted to each joint's limits from the reference pose: the other direction where the requested one does not fit, a smaller step where neither does, each logged. Stream positions without velocity feed-forward and with the step guard disabled (tracking lag is what is measured); return to the reference between joints and at the end. Write measurement files.
//
// Writes per-joint CSV and parameters.txt under the requested output directory.
//
// Uses a Robot control connection: compatible arm modes may be enabled during startup, and shutdown requests a stop. Real hardware is selected unless --simulated is supplied.

#include "common.hpp"

int main(int argc, char **argv) {
    return examples::run(argc, argv, "Benchmark each arm joint with a ramped step trajectory. Prompt unless --no-confirm; move the right seven-joint arm to [0,0,0,-0.5,0,0,0] rad. Test each joint for 5 s at 200 Hz: hold 0.3 s, ramp by pi/6 rad over 0.3 s, then hold. The step is fitted to each joint's limits from the reference pose: the other direction where the requested one does not fit, a smaller step where neither does, each logged. Stream positions without velocity feed-forward and with the step guard disabled (tracking lag is what is measured); return to the reference between joints and at the end. Write measurement files. Writes per-joint CSV and parameters.txt under the requested output directory. Uses a Robot control connection: compatible arm modes may be enabled during startup, and shutdown requests a stop. Real hardware is selected unless --simulated is supplied.",
                         [](const examples::Args &args) {
        using namespace arm_tracking;
        const auto side = args.side();
        const double duration = args.number("duration", 5.0);
        const double control_hz = args.number("control-hz", 200.0);
        const double step_amplitude = args.number("step-amplitude", 30.0 * arm_tracking::PI / 180.0);
        const double step_time = args.number("step-time", 0.3);
        const double transition_time = args.number("transition-time", 0.3);
        const double max_vel = args.number("max-vel", std::fabs(step_amplitude) / transition_time);
        if (transition_time <= 0 || max_vel <= 0) examples::Args::fail("transition time and max velocity must be positive");
        (void)sample_count(step_time, control_hz);
        (void)sample_count(std::fabs(step_amplitude) / max_vel, control_hz);
        const auto output_dir = args.text("output-dir", "results");

        std::cout << "WARNING: this benchmark moves the robot arm through step trajectories. "
                     "No collision checking is performed. Ensure the workspace is clear "
                     "and the e-stop is accessible.\n";
        if (!args.flag("no-confirm") && !examples::confirm("Continue?")) {
            std::cout << "Aborted by user.\n";
            return;
        }

        if (sample_count(duration, control_hz) == 0) examples::Args::fail("benchmark requires at least one sample");
        auto robot = args.connect();
        auto arm = robot.joints(side + "_arm");
        const auto result_dir = result_directory(output_dir, side, "step");
        std::cout << "Results will be saved to " << result_dir << "\n";

        // The step trajectory: hold zero, ramp to the amplitude at max_vel
        // (excluding the start point so the first streamed sample already
        // advances the joint), then hold. The amplitude is fitted to each
        // joint's limits from the reference pose.
        const size_t total = sample_count(duration, control_hz);
        const size_t before = sample_count(step_time, control_hz);
        const auto step_profile = [&](double amplitude) {
            const size_t ramp_steps =
                std::max<size_t>(1, sample_count(std::fabs(amplitude) / max_vel, control_hz));
            std::vector<double> trajectory(before, 0.0);
            for (size_t step = 1; step <= ramp_steps; ++step) {
                trajectory.push_back(amplitude * static_cast<double>(step) / static_cast<double>(ramp_steps));
            }
            while (trajectory.size() < total) trajectory.push_back(amplitude);
            trajectory.resize(total);
            return trajectory;
        };
        // Lag is what this benchmark measures: the step guard is off.
        const auto stream = dexcontrol::without_step_guard(dexcontrol::command_options());

        std::cout << "Moving " << side << " arm to zero position\n";
        go_to_zero(arm, 3.0);
        // Settle before verifying. A real robot publishes state continuously;
        // the simulation only re-stamps state on a command, so a dry run
        // passes --settle 0 to stay inside the 500 ms freshness limit.
        examples::sleep_for(args.number("settle", 0.5));
        verify_zero_position(arm);

        for (size_t joint = 0; joint < NUM_JOINTS; ++joint) {
            std::cout << "--- Testing joint " << joint << " ---\n";
            if (joint > 0) go_to_zero(arm, 2.0);
            const auto trajectory = step_profile(fit_amplitude(arm, joint, step_amplitude, false));

            std::vector<Sample> samples;
            dexcontrol::RateLimiter limiter(control_hz);
            const auto started = std::chrono::steady_clock::now();
            size_t index = 0;
            while (index < trajectory.size()) {
                auto position = ZERO_POS;
                position[joint] += trajectory[index];
                arm.set_joint_pos(position, stream);
                samples.push_back({examples::elapsed(started), position, arm.get_joint_pos()});
                limiter.sleep();
            if (examples::interrupted()) throw std::runtime_error("interrupted");
                ++index;
            }
            print_tracking_error(joint, samples);
            write_csv(result_dir / ("joint_" + std::to_string(joint) + ".csv"), samples);
        }

        std::ofstream meta(result_dir / "parameters.txt");
        meta << "side=" << side << "\nduration=" << duration << "\ncontrol_hz=" << control_hz
             << "\nstep_amplitude=" << step_amplitude << "\nstep_time=" << step_time
             << "\ntransition_time=" << transition_time << "\nmax_vel=" << max_vel
             << "\nzero_pos=" << examples::format(ZERO_POS) << "\n";
        std::cout << "Raw data saved to " << result_dir << "\n";

        std::cout << "Returning to zero position\n";
        go_to_zero(arm, 3.0);
        std::cout << "Benchmark complete\n";
        // Stop and disconnect deliberately: a stop that cannot be delivered
        // is reported here rather than left to the handle's destructor.
        robot.close();
    }, {"control-hz", "duration", "max-vel", "no-confirm", "output-dir", "settle", "side", "step-amplitude", "step-time", "transition-time"});
}

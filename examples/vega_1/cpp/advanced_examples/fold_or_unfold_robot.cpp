// Copyright (C) 2026 Dexmate Inc.
// SPDX-License-Identifier: Apache-2.0

// Fold or unfold the robot in ordered, verified stages.
//
// Default: close available hands, move both arms to folded, then torso to folded and head to tucked. With --unfold: move torso to crouch45_high, then resolve/move head home, then resolve/move arms to L_shape. Skip absent components; check motion success and measured positions within 0.1 rad before advancing. No collision checking is provided.
//
// Wait 1 s after closing each available hand before the arm stage.
//
// Uses a Robot control connection: compatible arm modes may be enabled during startup, and shutdown requests a stop. Real hardware is selected unless --simulated is supplied.

#include "example.hpp"

#include <map>

namespace {

// Moves components to named poses and confirms they arrived. Components
// this robot does not have are skipped, so one call covers profiles with
// and without a torso.
void stage(dexcontrol::Robot &robot, const std::map<std::string, std::string> &poses,
           double timeout, double tolerance, size_t number, size_t total) {
    dexcontrol::Robot::Targets targets;
    std::string described;
    for (const auto &[name, pose] : poses) {
        if (!robot.has_component(name)) continue;
        targets[name] = robot.joints(name).resolve_pose(pose);
        described += (described.empty() ? "" : ", ") + name + " -> " + pose;
    }
    if (targets.empty()) {
        std::cout << "Stage " << number << "/" << total << ": skipped (components unavailable)\n";
        return;
    }
    std::cout << "\nStage " << number << "/" << total << ": " << described
              << " (wait limit: " << timeout << " s)" << std::endl;
    auto group = robot.move_to_joint_pos(targets);
    bool complete = false;
    std::string wait_error;
    try {
        complete = group.wait(dexcontrol::WaitFor{examples::millis(timeout)}) ==
                   dexcontrol::MotionState::Succeeded;
    } catch (const std::exception &error) {
        wait_error = error.what();
    }
    if (group.size() != targets.size()) {
        throw std::runtime_error("Stage stopped: motion member count does not match targets.");
    }
    size_t member = 0;
    for (const auto &[name, target] : targets) {
        auto motion = group.member(member++);
        std::cout << "  " << name << " -> " << poses.at(name) << "\n";
        try {
            const auto state = motion.refresh();
            std::cout << "    Motion: " << dexcontrol::to_string(state)
                      << " (id: " << motion.id() << ")\n";
            if (!motion.message().empty()) std::cout << "    Reason: " << motion.message() << "\n";
            complete = complete && state == dexcontrol::MotionState::Succeeded;
        } catch (const std::exception &error) {
            complete = false;
            std::cout << "    Motion: unavailable (" << error.what() << ")\n";
        }
        try {
            auto joint = robot.joints(name);
            const auto measured = joint.get_joint_pos();
            if (measured.size() != target.size() || measured.empty()) {
                throw std::runtime_error("invalid joint feedback shape");
            }
            size_t worst = 0;
            double error = 0.0;
            for (size_t index = 0; index < target.size(); ++index) {
                if (!std::isfinite(measured[index]) || !std::isfinite(target[index])) {
                    throw std::runtime_error("non-finite joint feedback or target");
                }
                const double delta = std::fabs(measured[index] - target[index]);
                if (delta > error) error = delta, worst = index;
            }
            const bool reached = error <= tolerance;
            complete = complete && reached;
            std::cout << "    Target: " << (reached ? "reached" : "NOT reached")
                      << " (max error: " << std::fixed << std::setprecision(4) << error
                      << " rad; tolerance: " << tolerance << " rad)\n";
            if (!reached) {
                std::cout << "    Joint: " << joint.joint_names().at(worst)
                          << "; measured: " << measured[worst] << " rad; target: "
                          << target[worst] << " rad\n";
            }
            std::cout << std::defaultfloat;
        } catch (const std::exception &error) {
            complete = false;
            std::cout << "    Target: unverified (" << error.what() << ")\n";
        }
    }
    if (!wait_error.empty()) std::cout << "  Wait error: " << wait_error << "\n";
    if (!complete) {
        throw std::runtime_error("Stage stopped: motion success and target arrival were not both verified. No later stage will run.");
    }
    std::cout << "  Stage complete.\n";

}

}  // namespace

int main(int argc, char **argv) {
    return examples::run(argc, argv, "Fold or unfold the robot in ordered, verified stages. Default: close available hands, move both arms to folded, then torso to folded and head to tucked. With --unfold: move torso to crouch45_high, then resolve/move head home, then resolve/move arms to L_shape. Skip absent components; check motion success and measured positions within 0.1 rad before advancing. No collision checking is provided. Wait 1 s after closing each available hand before the arm stage. Uses a Robot control connection: compatible arm modes may be enabled during startup, and shutdown requests a stop. Real hardware is selected unless --simulated is supplied.",
                         [](const examples::Args &args) {
        const bool unfold = args.flag("unfold");
        const double timeout = args.number("timeout", 10.0);
        const double tolerance = args.number("tolerance", 0.1);

        if (!std::isfinite(timeout) || timeout <= 0) throw std::runtime_error("timeout must be finite and positive");
        if (!std::isfinite(tolerance) || tolerance < 0) throw std::runtime_error("tolerance must be finite and non-negative");
        std::cout << (unfold ? "Unfold robot\n" : "Fold robot\n");
        auto robot = args.connect();
        std::vector<std::map<std::string, std::string>> stages;
        if (unfold) {
            // Resolve reference poses after the torso reaches its new posture.
            stages = {{{"torso", "crouch45_high"}}, {{"head", "home"}},
                      {{"left_arm", "L_shape"}, {"right_arm", "L_shape"}}};
        } else {
            for (const std::string side : {"left", "right"}) {
                if (robot.has_component(side + "_hand")) {
                    robot.joints(side + "_hand").close_hand();
                    examples::sleep_for(1.0);
                }
            }
            // Arms come in first; the torso and head then tuck over them.
            stages = {{{"left_arm", "folded"}, {"right_arm", "folded"}},
                      {{"torso", "folded"}, {"head", "tucked"}}};
        }
        for (size_t index = 0; index < stages.size(); ++index) {
            stage(robot, stages[index], timeout, tolerance, index + 1, stages.size());
        }
        std::cout << (unfold ? "\nUnfold complete.\n" : "\nFold complete.\n");
        // Stop and disconnect deliberately: a stop that cannot be delivered
        // is reported here rather than left to the handle's destructor.
        robot.close();
    }, {"timeout", "tolerance", "unfold"});
}

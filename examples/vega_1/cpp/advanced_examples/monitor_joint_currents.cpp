// Copyright (C) 2026 Dexmate Inc.
// SPDX-License-Identifier: Apache-2.0

// Monitor joint currents for components that report per-joint current.
//
// Probe all components by default, skip unsupported current readings, then poll continuously at 0.02 s intervals until Ctrl-C; --samples N stops after N polls. Fail if none report current; --component limits the selection.
//
// Prints numeric summaries/frame metadata; does not open a viewer.
//
// Uses a Robot control connection: compatible arm modes may be enabled during startup, and shutdown requests a stop. Real hardware is selected unless --simulated is supplied.

#include "example.hpp"

int main(int argc, char **argv) {
    return examples::run(argc, argv, "Monitor joint currents for components that report per-joint current. Probe all components by default, skip unsupported current readings, then poll continuously at 0.02 s intervals until Ctrl-C; --samples N stops after N polls. Fail if none report current; --component limits the selection. Prints numeric summaries/frame metadata; does not open a viewer. Uses a Robot control connection: compatible arm modes may be enabled during startup, and shutdown requests a stop. Real hardware is selected unless --simulated is supplied.",
                         [](const examples::Args &args) {
        const int samples = args.integer("samples", 0);  // 0: until Ctrl-C
        const double period = args.number("period", 0.02);

        auto robot = args.connect();
        auto names = args.list("component");
        if (names.empty()) names = robot.component_names();

        // Being a joint component is not enough: a component may publish
        // joint state but no per-joint current. Probe once here so the
        // sampling loop below is a plain read.
        std::vector<std::pair<std::string, dexcontrol::JointComponent>> handles;
        std::vector<std::string> skipped;
        for (const auto &name : names) {
            try {
                auto component = robot.joints(name);
                // Python's get_joint_current_dict() raises when the reported
                // current cannot be named per joint; mirror that test.
                if (component.get_joint_state().current.size() != component.joint_count()) {
                    throw std::runtime_error("no current");
                }
                handles.emplace_back(name, std::move(component));
            } catch (const std::exception &) {
                skipped.push_back(name);
            }
        }
        if (!skipped.empty()) {
            std::cout << "no current data, skipping:";
            for (const auto &name : skipped) std::cout << " " << name;
            std::cout << "\n";
        }
        if (handles.empty()) examples::Args::fail("no component on this robot reports joint current");

        dexcontrol::RateLimiter limiter(1.0 / period);
        for (int index = 0; samples <= 0 || index < samples; ++index) {
            for (const auto &[name, component] : handles) {
                const auto joints = component.joint_names();
                const auto current = component.get_joint_state().current;
                std::cout << name << ":";
                for (size_t joint = 0; joint < current.size(); ++joint) {
                    std::cout << " " << (joint < joints.size() ? joints[joint] : std::to_string(joint))
                              << "=" << std::fixed << std::setprecision(3) << current[joint];
                }
                std::cout << "\n";
            }
            if (examples::interrupted()) break;
            if (samples <= 0 || index + 1 < samples) limiter.sleep();
        }
    }, {"component", "period", "samples"});
}

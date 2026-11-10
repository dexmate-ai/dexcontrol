// Copyright (C) 2026 Dexmate Inc.
// SPDX-License-Identifier: Apache-2.0
// Move the base forward 1 m, then turn 90 degrees, by measured amounts.
//
// Drive straight by --distance metres (default 1.0) with chassis.move_by, then turn in place by --angle degrees (default 90) with chassis.turn_by. Each is a closed loop on the base odometry: the base drives while the odometry is watched, slows down near the target, and corrects only if the settled error is outside the tolerance; --speed overrides the cruise speed. Print the amount measured after each motion, then the odometry pose and its sample counters. The heading comes from an enabled base IMU where the model has one (vega_1: --enable-sensor chassis_imu, optionally --yaw-source chassis_imu), otherwise from the wheels (vega_1p). Stop the base on exit.
//
// Uses a Robot control connection with startup arm initialization turned off; shutdown requests a stop. Real hardware is selected unless --simulated is supplied.
#include "example.hpp"

#include <cmath>
#include <iomanip>

int main(int argc, char **argv) {
    return examples::run(argc, argv, "Move the base forward 1 m, then turn 90 degrees, by measured amounts. Drive straight by --distance metres (default 1.0) with chassis.move_by, then turn in place by --angle degrees (default 90) with chassis.turn_by. Each is a closed loop on the base odometry: the base drives while the odometry is watched, slows down near the target, and corrects only if the settled error is outside the tolerance; --speed overrides the cruise speed. Print the amount measured after each motion, then the odometry pose and its sample counters. The heading comes from an enabled base IMU where the model has one (vega_1: --enable-sensor chassis_imu, optionally --yaw-source chassis_imu), otherwise from the wheels (vega_1p). Stop the base on exit. Uses a Robot control connection with startup arm initialization turned off; shutdown requests a stop. Real hardware is selected unless --simulated is supplied.",
                         [](const examples::Args &args) {
        const double distance = args.number("distance", 1.0);
        const double angle = args.number("angle", 90.0);
        const double speed = args.number("speed", 0.0);
        const auto yaw_source = args.text("yaw-source", "auto");
        auto options = args.connect_options();
        options.skip_arm_initialization = true;
        auto robot = dexcontrol::Robot::connect(options);
        if (!robot.supports_odometry()) throw std::runtime_error("this robot's chassis has no wheel model; odometry is unavailable");
        auto odometry = robot.start_odometry(yaw_source);
        std::cout << "heading from: " << odometry.yaw_source() << std::endl;
        auto loop_options = dexcontrol::Chassis::closed_loop_options();
        loop_options.speed = speed;
        auto chassis = robot.chassis();
        try {
            std::cout << std::fixed << std::setprecision(3) << std::showpos;
            const double moved = chassis.move_by(distance, loop_options);
            std::cout << "moved " << moved << " m (asked " << distance << " m)" << std::endl;
            const double turned = chassis.turn_by(angle * M_PI / 180.0, loop_options);
            std::cout << std::setprecision(1) << "turned " << turned * 180.0 / M_PI << " deg (asked " << angle << " deg)" << std::endl;
            const auto pose = odometry.pose();
            const auto status = odometry.status();
            std::cout << std::setprecision(3) << "pose: x=" << pose.x << " m  y=" << pose.y << " m  yaw="
                      << std::setprecision(1) << pose.yaw * 180.0 / M_PI << " deg | " << std::noshowpos
                      << "samples=" << status.samples << " imu=" << status.imu_samples
                      << " slip=" << status.slip_samples << " gaps=" << status.gaps << std::endl;
        } catch (...) {
            chassis.stop();
            throw;
        }
        chassis.stop();
    }, {"angle", "distance", "speed", "yaw-source"});
}

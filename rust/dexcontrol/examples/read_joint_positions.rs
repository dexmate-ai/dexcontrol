// Copyright (C) 2026 Dexmate Inc.
// SPDX-License-Identifier: Apache-2.0
use dexcontrol::Robot;
fn main() -> dexcontrol::Result<()> {
    let robot = Robot::simulated("vega_1")?;
    for name in ["head", "left_arm", "right_arm", "torso"] {
        println!(
            "{name} joint positions (rad): {:.4?}",
            robot.joints(name)?.get_joint_pos()?
        );
    }
    robot.close()
}

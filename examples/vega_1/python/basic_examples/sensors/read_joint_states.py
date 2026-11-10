# Copyright (C) 2026 Dexmate Inc.
# SPDX-License-Identifier: Apache-2.0

"""Print current joint states and chassis state.

Read all available joint components once, skipping unavailable joint state; --component narrows that list. Also print chassis state when available, independently of the component filter. This is a cached state listing, not a complete diagnostic report.

Prints named joint rows with model-derived units; unavailable fields appear as —. Chassis steering, wheel encoder positions and wheel speeds are listed separately. Reports nonempty joint error indices by name.

Uses a Robot control connection: compatible arm modes may be enabled during startup, and shutdown requests a stop. Real hardware is selected unless --simulated is supplied."""

from dataclasses import dataclass, field
from typing import List, Optional

import tyro
from dexcontrol import DexcontrolError, Robot, robot_config
from typing_extensions import Annotated


@dataclass
class Args:
    """Command-line options."""

    component: Annotated[List[str], tyro.conf.UseAppendAction] = field(
        default_factory=lambda: []
    )
    simulated: Annotated[bool, tyro.conf.arg(help="Use the in-process simulation")] = (
        False
    )
    profile: Annotated[Optional[str], tyro.conf.arg(help="Built-in robot profile")] = (
        None
    )
    config: Annotated[
        Optional[str], tyro.conf.arg(help="Custom robot configuration")
    ] = None


def value(values, index, unit):
    if index >= len(values):
        return "—"
    number = float(values[index])
    if abs(number) < 0.00005:
        number = 0.0
    return f"{number:.4f} {unit}".rstrip()


def units(joint_type):
    if joint_type == "prismatic":
        return "m", "m/s", "N"
    if joint_type in ("revolute", "continuous"):
        return "rad", "rad/s", "N·m"
    return "units", "units/s", "units"


def main() -> None:
    args = tyro.cli(Args, description=__doc__, config=(tyro.conf.FlagCreatePairsOff,))
    with Robot(
        simulation=args.simulated, profile=args.profile, config_file=args.config
    ) as robot:
        config = robot_config(profile=args.profile, config_file=args.config)
        metadata = config.get("joint_metadata", {})
        print("Cached joint states (— = not reported)")
        names = args.component or list(robot.component_names)
        for name in names:
            try:
                joint = robot.joints(name)
                state = joint.get_state()
            except (DexcontrolError, RuntimeError, ValueError):
                continue
            print(f"\n{name}")
            print(f"  {'Joint':<20} {'Position':>16} {'Velocity':>16} {'Effort':>16} {'Current':>14}")
            joint_meta = {item['name']: item for item in metadata.get(name, [])}
            for index, joint_name in enumerate(joint.joint_names):
                pos_unit, vel_unit, effort_unit = units(joint_meta.get(joint_name, {}).get('joint_type'))
                print(f"  {joint_name:<20} {value(state['position'], index, pos_unit):>16} "
                      f"{value(state['velocity'], index, vel_unit):>16} "
                      f"{value(state['torque'], index, effort_unit):>16} "
                      f"{value(state['current'], index, 'A'):>14}")
            errors = state.get('error_joint_indices', [])
            if len(errors):
                print("  Reported joint errors: " + ", ".join(
                    joint.joint_names[i] if i < len(joint.joint_names) else f"index {i}"
                    for i in errors))
        if robot.has_component("chassis"):
            chassis = robot.chassis
            try:
                positions = chassis.get_joint_pos_dict()
                steer_count = len(chassis.steering_angle)
                wheel_velocity = chassis.wheel_velocity
            except (DexcontrolError, RuntimeError, ValueError):
                print("\nchassis: state unavailable")
            else:
                print("\nchassis")
                print(f"  {'Joint':<20} {'Measurement':<22} {'Position':>16} {'Wheel speed':>16}")
                joint_meta = {item['name']: item for item in metadata.get('chassis', [])}
                for index, (name, position) in enumerate(positions.items()):
                    unit = units(joint_meta.get(name, {}).get('joint_type'))[0]
                    measurement = 'Steering angle' if index < steer_count else 'Wheel encoder'
                    speed = "—" if index < steer_count else value(wheel_velocity, index - steer_count, "m/s")
                    print(f"  {name:<20} {measurement:<22} {value([position], 0, unit):>16} {speed:>16}")


if __name__ == "__main__":
    main()

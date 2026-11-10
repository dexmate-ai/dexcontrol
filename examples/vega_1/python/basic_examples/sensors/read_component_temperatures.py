# Copyright (C) 2026 Dexmate Inc.
# SPDX-License-Identifier: Apache-2.0

"""Print component temperatures and battery status.

Read each available component temperature stream once, then battery temperature/status. --component narrows component selection; battery reporting is separate. Report unavailable temperature sources.

Uses a Robot control connection: compatible arm modes may be enabled during startup, and shutdown requests a stop. Real hardware is selected unless --simulated is supplied."""

from dataclasses import dataclass, field
from typing import List, Optional

import tyro
from dexcontrol import DexcontrolError, Robot
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


def main() -> None:
    args = tyro.cli(Args, description=__doc__, config=(tyro.conf.FlagCreatePairsOff,))
    with Robot(
        simulation=args.simulated, profile=args.profile, config_file=args.config
    ) as robot:
        names = args.component or list(robot.component_names)
        unavailable = []
        unsupported = []
        for name in names:
            if name == "battery":
                continue  # Battery temperature comes from battery status below.
            try:
                sensor = robot.sensors.get_temperature(name)
            except (DexcontrolError, RuntimeError, ValueError):
                if args.component:
                    unsupported.append(name)
                continue
            # Temperature streams are slow (about 1 Hz) and connecting does not
            # wait for them, so give the first sample time to arrive.
            if not sensor.wait_for_active(timeout=5.0):
                unavailable.append(name)
                continue
            try:
                temperatures = sensor.get_temperatures()
            except (DexcontrolError, RuntimeError, ValueError):
                unavailable.append(name)
                continue
            if not any(temperatures.values()):
                unavailable.append(name)
                continue
            print(f"\n{name} temperatures")
            for group, readings in temperatures.items():
                for label, value in readings.items():
                    print(f"  {group}/{label}: {value:.1f} °C")

        if robot.has_component("battery"):
            try:
                status = robot.battery.get_status()
            except (DexcontrolError, RuntimeError, ValueError):
                print("\nBattery status unavailable.")
            else:
                print(
                    f"\nBattery status\n"
                    f"  Charge       {status['percentage']:.1f} %\n"
                    f"  Voltage      {status['voltage']:.2f} V\n"
                    f"  Current      {status['current']:.2f} A\n"
                    f"  Power        {status['power']:.2f} W\n"
                    f"  Temperature  {status['temperature']:.1f} °C"
                )
        else:
            print("\nNo battery configured on this robot.")
        if unsupported:
            print(f"\nNo temperature stream configured for: {', '.join(unsupported)}")
        if unavailable:
            print(f"\nTemperature readings unavailable: {', '.join(unavailable)}")


if __name__ == "__main__":
    main()

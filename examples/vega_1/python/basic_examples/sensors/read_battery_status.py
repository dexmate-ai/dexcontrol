# Copyright (C) 2026 Dexmate Inc.
# SPDX-License-Identifier: Apache-2.0

"""Read and print battery status.

Wait up to 5 s for battery activity and print one observation by default, including charge, voltage, current and temperature. Optional repeated reads use a 0.5 s interval.

Uses a Robot control connection: compatible arm modes may be enabled during startup, and shutdown requests a stop. Real hardware is selected unless --simulated is supplied."""

import time
from dataclasses import dataclass
from typing import Optional

import tyro
from dexcontrol import Robot
from typing_extensions import Annotated


@dataclass
class Args:
    """Command-line options."""

    samples: int = 1
    period: float = 0.5
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
    if args.samples < 1 or args.period < 0:
        raise SystemExit("samples must be positive and period non-negative")
    with Robot(
        simulation=args.simulated, profile=args.profile, config_file=args.config
    ) as robot:
        if not robot.has_component("battery"):
            raise SystemExit("this robot has no battery")
        battery = robot.battery
        if not battery.wait_for_active(timeout=5.0):
            raise SystemExit("battery state did not become active")
        for index in range(args.samples):
            status = battery.get_status()
            print(
                f"Battery status ({index + 1}/{args.samples})\n"
                f"  Charge       {status['percentage']:.1f} %\n"
                f"  Voltage      {status['voltage']:.2f} V\n"
                f"  Current      {status['current']:.2f} A\n"
                f"  Power        {status['power']:.2f} W\n"
                f"  Temperature  {status['temperature']:.1f} °C\n",
                flush=True,
            )
            if index + 1 < args.samples:
                time.sleep(args.period)


if __name__ == "__main__":
    main()

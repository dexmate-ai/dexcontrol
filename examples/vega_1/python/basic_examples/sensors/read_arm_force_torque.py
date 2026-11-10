# Copyright (C) 2026 Dexmate Inc.
# SPDX-License-Identifier: Apache-2.0

"""Read and print one arm force-torque observation.

Wait up to 5 s for the right-arm wrench stream and print one observation by default. Optional repeated reads use a 0.1 s interval; no calibration or zeroing is requested.

Uses a read-only DiagnosticClient connection; motors are not initialized. Real hardware is selected unless --simulated is supplied."""

import time
from dataclasses import dataclass
from typing import Literal, Optional

import tyro
from dexcontrol import DiagnosticClient
from typing_extensions import Annotated


@dataclass
class Args:
    """Command-line options."""

    side: Literal["left", "right"] = "right"
    samples: int = 1
    period: float = 0.1
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
    with DiagnosticClient(
        simulation=args.simulated, profile=args.profile, config_file=args.config
    ) as robot:
        wrench = robot.sensors.get_wrench(f"{args.side}_arm")
        if not wrench.wait_for_active(timeout=5.0):
            raise SystemExit("wrench state did not become active")
        for index in range(args.samples):
            print(wrench.get_wrench_state())
            if index + 1 < args.samples:
                time.sleep(args.period)


if __name__ == "__main__":
    main()

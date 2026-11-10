# Copyright (C) 2026 Dexmate Inc.
# SPDX-License-Identifier: Apache-2.0

"""Read and print arm wrist-button states.

Read one right-arm button observation by default. --samples and --period repeat the reads; the default repeat interval is 0.1 s.

Uses a Robot control connection: compatible arm modes may be enabled during startup, and shutdown requests a stop. Real hardware is selected unless --simulated is supplied."""

import pprint
import time
from dataclasses import dataclass
from typing import Literal, Optional

import tyro
from dexcontrol import Robot
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
    with Robot(
        simulation=args.simulated, profile=args.profile, config_file=args.config
    ) as robot:
        arm = robot.joints(f"{args.side}_arm")
        for index in range(args.samples):
            pprint.pp(arm.get_button_state(), sort_dicts=False)
            if index + 1 < args.samples:
                time.sleep(args.period)


if __name__ == "__main__":
    main()

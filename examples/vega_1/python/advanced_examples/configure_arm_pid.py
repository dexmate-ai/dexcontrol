# Copyright (C) 2026 Dexmate Inc.
# SPDX-License-Identifier: Apache-2.0

"""Read or set arm PID multipliers.

Require positional get or set. Address both arms by default; set sends seven multipliers, each defaulting to 1.0. Print service replies. PID writes can take about 40 s; the SDK waits up to 45 s for a reply. A timeout does not cancel a server-side write.

Uses a Robot control connection: compatible arm modes may be enabled during startup, and shutdown requests a stop. Real hardware is selected unless --simulated is supplied."""

import pprint
from dataclasses import dataclass
from typing import Literal, Optional, Tuple

import tyro
from dexcontrol import Robot
from typing_extensions import Annotated


@dataclass
class Args:
    """Read or set seven PID multipliers on the selected arms."""

    action: Annotated[Literal["get", "set"], tyro.conf.Positional]
    side: Literal["left", "right", "both"] = "both"
    p: Tuple[float, float, float, float, float, float, float] = (1.0,) * 7
    simulated: bool = False
    profile: Optional[str] = None
    config: Optional[str] = None


def main() -> None:
    args = tyro.cli(Args, description=__doc__, config=(tyro.conf.FlagCreatePairsOff,))
    if any(not 0.1 <= value <= 4.0 for value in args.p):
        raise SystemExit("every PID multiplier must be in [0.1, 4.0]")

    sides = ("left", "right") if args.side == "both" else (args.side,)
    with Robot(
        simulation=args.simulated, profile=args.profile, config_file=args.config
    ) as robot:
        for side in sides:
            arm = robot.joints(f"{side}_arm")
            if args.action == "set":
                print(f"Setting {side}_arm PID gains; this can take about 40 seconds...", flush=True)
            result = arm.get_pid() if args.action == "get" else arm.set_pid(args.p)
            print(f"{side}_arm")
            pprint.pp(result, sort_dicts=False)


if __name__ == "__main__":
    main()

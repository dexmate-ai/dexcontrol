# Copyright (C) 2026 Dexmate Inc.
# SPDX-License-Identifier: Apache-2.0

"""Request a reboot of one robot control board.

Require positional arm, torso, or chassis. The arm board controls both arms. Send the request and exit; do not wait for the board to come back online.

Rebooting a board drops its joints out of control while it restarts, so the script asks for a typed "yes" first unless --yes is supplied. Uses a Robot control connection with startup arm initialization turned off, so the tool also reaches an arm whose motor cannot enable; shutdown requests a stop. Real hardware is selected unless --simulated is supplied; simulated runs skip the confirmation prompt."""

import sys
from dataclasses import dataclass
from typing import Literal, Optional

import tyro
from dexcontrol import Robot, confirm
from typing_extensions import Annotated


@dataclass
class Args:
    """Command-line options."""

    board: Annotated[Literal["arm", "torso", "chassis"], tyro.conf.Positional]
    yes: Annotated[
        bool, tyro.conf.arg(help="Skip the interactive reboot confirmation")
    ] = False
    simulated: Annotated[bool, tyro.conf.arg(help="Use the in-process simulation")] = (
        False
    )
    profile: Annotated[Optional[str], tyro.conf.arg(help="Built-in robot profile")] = (
        None
    )
    config: Annotated[
        Optional[str], tyro.conf.arg(help="Custom robot configuration")
    ] = None


def confirm_reboot(args: Args) -> None:
    print(
        f"WARNING: rebooting the {args.board} control board. Its joints are not "
        "controlled while it restarts; make sure the robot is in a safe, "
        "supported posture.",
        file=sys.stderr,
    )
    if args.yes or args.simulated:
        return
    if not confirm(f"Reboot the {args.board} board?", expected="yes"):
        print("Aborted: no reboot was requested.", file=sys.stderr)
        raise SystemExit(1)


def main() -> None:
    args = tyro.cli(Args, description=__doc__, config=(tyro.conf.FlagCreatePairsOff,))
    confirm_reboot(args)
    with Robot(
        simulation=args.simulated,
        profile=args.profile,
        config_file=args.config,
        initialize_arm_modes=False,
    ) as robot:
        robot.reboot_component(args.board)
        # Fire-and-forget: the board stops answering while it restarts, so a
        # clean return means the request was accepted, not that it is back.
        print(f"{args.board}: reboot request sent")


if __name__ == "__main__":
    main()

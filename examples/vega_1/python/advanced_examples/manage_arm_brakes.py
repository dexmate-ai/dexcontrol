# Copyright (C) 2026 Dexmate Inc.
# SPDX-License-Identifier: Apache-2.0

"""Read, release, or engage arm brakes.

Require positional status, release, or engage. status and engage address both arms and every joint unless --side / --joint narrow them. release is never implicit: it requires --side and either --joint (repeatable) or --all, warns that the arm must be supported, and asks for a typed "yes" unless --yes is supplied. This operates the brake service, not the motor-mode service.

Uses a Robot control connection with startup arm initialization turned off, so the tool also reaches an arm whose motor cannot enable; shutdown requests a stop. Real hardware is selected unless --simulated is supplied; simulated runs skip the confirmation prompt."""

import pprint
import sys
from dataclasses import dataclass, field
from typing import List, Literal, NoReturn, Optional

import tyro
from dexcontrol import Robot, confirm
from typing_extensions import Annotated


@dataclass
class Args:
    """Command-line options."""

    action: Annotated[Literal["status", "release", "engage"], tyro.conf.Positional]
    # Required for release. status and engage default to both arms.
    side: Optional[Literal["left", "right", "both"]] = None
    # Zero-based joint index; repeat --joint to select multiple joints.
    joint: Annotated[List[int], tyro.conf.UseAppendAction] = field(default_factory=list)
    all: Annotated[
        bool, tyro.conf.arg(help="Act on every joint of the selected arm(s)")
    ] = False
    yes: Annotated[
        bool, tyro.conf.arg(help="Skip the interactive release confirmation")
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


def usage_error(message: str) -> NoReturn:
    print(f"error: {message}", file=sys.stderr)
    raise SystemExit(2)


def validate(args: Args) -> None:
    """Rejects an ambiguous scope before any connection is made."""
    if args.all and args.joint:
        usage_error("--all and --joint are mutually exclusive")
    if any(index < 0 for index in args.joint):
        usage_error("--joint indices are zero-based and non-negative")
    if args.action != "release":
        return
    if args.side is None:
        usage_error("release requires an explicit --side {left,right,both}")
    if not args.all and not args.joint:
        usage_error("release requires --joint <index> (repeatable) or --all")


def confirm_release(args: Args, sides) -> None:
    scope = "ALL joints" if args.all else f"joint(s) {args.joint}"
    arms = " and ".join(f"{side}_arm" for side in sides)
    print(
        f"WARNING: releasing the brakes of {scope} on {arms}.\n"
        "A released joint is no longer held: the arm WILL DROP under gravity "
        "unless it is physically supported. Support the arm before continuing.",
        file=sys.stderr,
    )
    if args.yes or args.simulated:
        return
    if not confirm("Release the brakes?", expected="yes"):
        print("Aborted: no brake was released.", file=sys.stderr)
        raise SystemExit(1)


def main() -> None:
    args = tyro.cli(Args, description=__doc__, config=(tyro.conf.FlagCreatePairsOff,))
    validate(args)
    side = args.side or "both"
    sides = ("left", "right") if side == "both" else (side,)
    if args.action == "release":
        confirm_release(args, sides)
    with Robot(
        simulation=args.simulated,
        profile=args.profile,
        config_file=args.config,
        initialize_arm_modes=False,
    ) as robot:
        for side in sides:
            arm = robot.joints(f"{side}_arm")
            if args.action == "status":
                result = arm.get_brake_status()
            elif args.joint:
                result = arm.release_brake(args.action == "release", args.joint)
            else:
                # Every joint: --all for release (validated above), and the
                # default for engage, which is the safe direction.
                result = arm.release_all_brakes(args.action == "release")
            print(f"{side}_arm")
            pprint.pp(result, sort_dicts=False)


if __name__ == "__main__":
    main()

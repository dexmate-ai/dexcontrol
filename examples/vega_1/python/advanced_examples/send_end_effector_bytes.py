# Copyright (C) 2026 Dexmate Inc.
# SPDX-License-Identifier: Apache-2.0

"""Send raw bytes to end effectors and poll for replies.

By default, send hex 09 10 03 E8 00 03 06 01 00 00 00 00 00 72 E1 to both configured arm pass-through interfaces and poll for up to 1 s each. The attached device determines what these bytes do; this is not a generic motion command.

Uses a Robot control connection: compatible arm modes may be enabled during startup, and shutdown requests a stop. Real hardware is selected unless --simulated is supplied."""

import time
from dataclasses import dataclass
from typing import Literal, Optional

import tyro
from dexcontrol import Robot
from typing_extensions import Annotated


@dataclass
class Args:
    """Command-line options."""

    side: Literal["left", "right", "both"] = "both"
    message_hex: str = "09 10 03 E8 00 03 06 01 00 00 00 00 00 72 E1"
    timeout: float = 1.0
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
    payload = bytes.fromhex(args.message_hex.replace(" ", ""))
    sides = ("left", "right") if args.side == "both" else (args.side,)
    with Robot(
        simulation=args.simulated, profile=args.profile, config_file=args.config
    ) as robot:
        for side in sides:
            arm = robot.joints(f"{side}_arm")
            if not arm.enable_ee_pass_through:
                print(f"{side}_arm: pass-through endpoints are not configured")
                continue
            arm.send_ee_pass_through_message(payload)
            deadline = time.monotonic() + args.timeout
            while time.monotonic() < deadline:
                response = arm.get_ee_pass_through_response()
                if response is not None:
                    print(f"{side}_arm: {response.hex(' ')}")
                    break
                time.sleep(0.01)
            else:
                print(f"{side}_arm: no response within {args.timeout:.2f}s")


if __name__ == "__main__":
    main()

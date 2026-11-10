# Copyright (C) 2026 Dexmate Inc.
# SPDX-License-Identifier: Apache-2.0

"""Clear errors on all supported robot components.

Send clear-error requests, print each component outcome, and exit nonzero if any outcome failed. This is a mutating maintenance operation.

Uses a Robot control connection with startup arm initialization turned off: an arm whose motor cannot enable is the usual reason to clear errors, and the default connection would fail on it before any request is sent. Shutdown requests a stop. Real hardware is selected unless --simulated is supplied."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import tyro
from dexcontrol import Robot
from typing_extensions import Annotated


@dataclass
class Args:
    """Command-line options."""

    profile: Optional[str] = None
    config: Optional[str] = None
    simulated: Annotated[bool, tyro.conf.arg(help="Use the in-process simulation")] = (
        False
    )


def main() -> None:
    args = tyro.cli(Args, description=__doc__, config=(tyro.conf.FlagCreatePairsOff,))
    if args.profile is not None and args.config is not None:
        raise SystemExit("--profile and --config are mutually exclusive")

    options: dict[str, Any] = {}
    if args.config:
        options["config_file"] = args.config
    elif args.profile:
        options["profile"] = args.profile

    with Robot(
        **options, simulation=args.simulated, initialize_arm_modes=False
    ) as robot:
        report = robot.clear_errors()

    failed = False
    if not report["outcomes"]:
        print("No clearable components are enabled for this robot.")
    for outcome in report["outcomes"]:
        component = outcome["component"]
        if outcome["success"]:
            print(f"✓ {component}")
        else:
            failed = True
            print(f"✗ {component}: {outcome['error']}")
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

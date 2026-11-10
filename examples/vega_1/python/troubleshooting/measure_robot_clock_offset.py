# Copyright (C) 2026 Dexmate Inc.
# SPDX-License-Identifier: Apache-2.0

"""Estimate robot clock offset and network round-trip time.

Use a read-only DiagnosticClient to collect 30 time-query samples and report server-minus-client offset, round-trip time, and replies. Does not synchronize or change either clock; no simulation connection is supported.

Human-readable signed offset and RTT in milliseconds."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import tyro
from dexcontrol import DiagnosticClient


def format_result(result: dict[str, Any]) -> str:
    """Describe the averaged NTP estimate; positive offset means server ahead."""
    replies = result["replies"]
    samples = result["samples"]
    lines = ["", "Robot clock check", f"  Replies received:  {replies}/{samples}"]
    if not result["success"] or replies == 0:
        lines.extend(
            [
                "  Result:            No valid replies; clock difference is unavailable.",
                "  Check the robot connection and time diagnostic service, then retry.",
            ]
        )
        return "\n".join(lines)

    offset_ms = result["offset"] * 1000
    rtt_ms = result["rtt"] * 1000
    direction = "ahead of" if offset_ms > 0 else "behind"
    lines.extend(
        [
            f"  Mean clock offset: {offset_ms:+.3f} ms (robot/server minus this computer)",
            f"  Mean round trip:   {rtt_ms:.3f} ms (request to reply)",
        ]
    )
    if offset_ms == 0:
        lines.append("  Estimated clock difference is zero.")
    else:
        lines.append(
            f"  The robot/server clock is approximately {abs(offset_ms):.3f} ms "
            f"{direction} this computer's clock."
        )
    if replies < samples:
        lines.append(
            "  Partial result: some requests returned no valid reply before the deadline."
        )
    lines.append(
        "  Timing is an estimate affected by network delays; no clocks were changed."
    )
    return "\n".join(lines)


@dataclass
class Args:
    """Command-line options."""

    profile: Optional[str] = None
    config: Optional[str] = None
    samples: int = 30


def main() -> None:
    args = tyro.cli(Args, description=__doc__, config=(tyro.conf.FlagCreatePairsOff,))
    if args.profile is not None and args.config is not None:
        raise SystemExit("--profile and --config are mutually exclusive")

    options: dict[str, Any] = {}
    if args.config:
        options["config_file"] = args.config
    elif args.profile:
        options["profile"] = args.profile

    with DiagnosticClient(**options) as client:
        result = client.query_ntp(args.samples)
    print(format_result(result))


if __name__ == "__main__":
    main()

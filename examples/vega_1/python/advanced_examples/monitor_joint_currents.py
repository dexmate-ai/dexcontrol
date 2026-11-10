# Copyright (C) 2026 Dexmate Inc.
# SPDX-License-Identifier: Apache-2.0

"""Monitor joint currents for components that report per-joint current.

Probe all components by default, skip unsupported current readings, then poll continuously at 0.02 s intervals until Ctrl-C; --samples N stops after N polls. Fail if none report current; --component limits the selection.

Starts/connects a Rerun viewer by default; --no-display prints instead. Missing rerun-sdk falls back to printing.

Uses a Robot control connection: compatible arm modes may be enabled during startup, and shutdown requests a stop. Real hardware is selected unless --simulated is supplied."""

import pprint
import time
from dataclasses import dataclass, field
from typing import List, Optional

import tyro
from dexcontrol import DexcontrolError, Robot
from typing_extensions import Annotated

from dexcontrol.viewer import (
    ViewerOptions,
    close,
    set_sample_time,
    show_time_series,
    start,
)


@dataclass
class Args(ViewerOptions):
    """Command-line options."""

    component: Annotated[List[str], tyro.conf.UseAppendAction] = field(
        default_factory=lambda: []
    )
    samples: Annotated[int, tyro.conf.arg(help="Polls before exiting; 0 runs until Ctrl-C")] = 0
    period: float = 0.02
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
        # Being a joint component is not enough: the head, for instance,
        # publishes joint state but no per-joint current, and asking for it
        # raises rather than inventing values. Probe once here so the sampling
        # loop below is a plain read.
        handles, skipped = {}, []
        for name in names:
            try:
                component = robot.joints(name)
                component.get_joint_current_dict()
            except (DexcontrolError, RuntimeError, ValueError):
                skipped.append(name)
                continue
            handles[name] = component
        if skipped:
            print(f"no current data, skipping: {', '.join(skipped)}")
        if not handles:
            raise SystemExit("no component on this robot reports joint current")

        visualize = start("joint-current", args)
        if visualize:
            show_time_series("current", list(handles))
        # samples == 0 runs until Ctrl-C; the `with Robot` block then shuts
        # the robot down in order.
        try:
            monitor(args, handles, visualize)
        except KeyboardInterrupt:
            close()
            print("stopped")


def monitor(args: Args, handles: dict, visualize: bool) -> None:
    started = time.monotonic()
    index = 0
    while args.samples <= 0 or index < args.samples:
        if visualize:
            import rerun as rr

            set_sample_time(index, time.monotonic() - started)
        for name, component in handles.items():
            currents = component.get_joint_current_dict()
            if visualize:
                for joint, value in currents.items():
                    rr.log(f"current/{name}/{joint}", rr.Scalars(value))
            else:
                pprint.pp({name: currents})
        index += 1
        if args.samples <= 0 or index < args.samples:
            time.sleep(args.period)


if __name__ == "__main__":
    main()

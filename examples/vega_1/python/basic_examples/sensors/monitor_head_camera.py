# Copyright (C) 2026 Dexmate Inc.
#
# This software is dual-licensed:
#
# 1. GNU Affero General Public License v3.0 (AGPL-3.0)
#    See LICENSE-AGPL for details
#
# 2. Commercial License
#    For commercial licensing terms, contact: contact@dexmate.ai

"""Monitor frames from the configured head camera.

Enable head_camera, subscribe to all advertised streams unless selected with --stream, and monitor at 30 Hz until Ctrl+C by default, then unsubscribe. --samples N limits the number of polls. The sensor name is configurable; no particular camera brand is required and no image files are saved.

Starts/connects a Rerun viewer by default; --no-display prints instead. Missing rerun-sdk falls back to printing.

Uses a Robot control connection: compatible arm modes may be enabled during startup, and shutdown requests a stop. Real hardware is selected unless --simulated is supplied."""

import time
from dataclasses import dataclass, field
from typing import List, Optional

import tyro
from _viewer import ViewerOptions, log_frame, set_sample_time, start
from dexcontrol import Robot
from typing_extensions import Annotated


@dataclass
class Args(ViewerOptions):
    """Command-line options."""

    sensor: str = "head_camera"
    stream: Annotated[List[str], tyro.conf.UseAppendAction] = field(
        default_factory=lambda: []
    )
    samples: Annotated[
        Optional[int], tyro.conf.arg(help="Poll limit; omitted runs until Ctrl+C")
    ] = None
    period: float = 1 / 30
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
    if args.samples is not None and args.samples < 0:
        raise ValueError("--samples must be non-negative")

    names = [args.sensor]
    with Robot(
        enable_sensors=names,
        simulation=args.simulated,
        profile=args.profile,
        config_file=args.config,
    ) as robot:
        cameras = {name: robot.sensors.get_camera(name) for name in names}
        subscribed = [
            (name, stream)
            for name, camera in cameras.items()
            for stream in (args.stream or list(camera.streams))
        ]
        for name, stream in subscribed:
            cameras[name].subscribe(stream)

        try:
            visualize = start("head-camera", args)
            started = time.monotonic()
            index = 0
            while args.samples is None or index < args.samples:
                if visualize:
                    set_sample_time(index, time.monotonic() - started)
                for name, stream in subscribed:
                    frame = cameras[name].latest_frame(stream)
                    if frame is None:
                        continue
                    if visualize:
                        log_frame(f"{name}/{stream}", frame)
                    else:
                        print(f"{name}/{stream}: shape={frame['data'].shape}")
                index += 1
                if args.samples is None or index < args.samples:
                    time.sleep(args.period)
        except KeyboardInterrupt:
            pass
        finally:
            for name, stream in subscribed:
                cameras[name].unsubscribe(stream)


if __name__ == "__main__":
    main()

# Copyright (C) 2026 Dexmate Inc.
# SPDX-License-Identifier: Apache-2.0

"""Optional Rerun display helpers for images, depth and live sensor data.

:func:`start` opens or connects a viewer unless ``ViewerOptions.no_display``
is set; a viewer this module spawned closes when the calling process exits,
including on forced termination. A missing ``rerun-sdk`` falls back to
printing, so a script keeps working without it.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass
class ViewerOptions:
    """Display options shared by sensor examples."""

    viewer_url: Optional[str] = None
    """stream to a Rerun viewer at HOST:PORT instead of spawning one (run `rerun` on your laptop, then point this at it)"""
    no_display: bool = False
    """print values instead of visualizing them"""


def start(name: str, args: Optional[ViewerOptions] = None) -> bool:
    """Starts a Rerun recording unless the caller asked for plain output.

    Returns whether anything should be logged, so an example can branch once
    and keep its loop readable. A missing `rerun-sdk` is reported and treated
    as `--no-display` rather than raised: the example still does something
    useful, which matters when the point of running it is to find out whether
    the sensor works at all.
    """
    args = args or ViewerOptions()
    if args.no_display:
        return False
    try:
        import rerun as rr
    except ImportError:
        print("rerun-sdk is not installed; printing instead (pip install rerun-sdk)")
        return False

    rr.init(f"dexcontrol-{name}")
    if args.viewer_url:
        rr.connect_grpc(f"rerun+http://{args.viewer_url}/proxy")
    else:
        _spawn_attached(rr)
    return True


_viewer_guard = None
_viewer_lifetime_fd: Optional[int] = None


def _spawn_attached(rr) -> None:
    """Launch a viewer owned by this example, including after a forced exit.

    A separate process owns the viewer and watches a pipe held only by this
    process. Closing the pipe, normal exit, SIGTERM, and even SIGKILL all cause
    EOF and let the guard terminate the viewer's entire process group.
    """
    import atexit
    import os
    import subprocess
    import sys

    global _viewer_guard, _viewer_lifetime_fd
    close()
    lifetime_read, lifetime_write = os.pipe()
    ready_read, ready_write = os.pipe()
    try:
        _viewer_guard = subprocess.Popen(
            [
                sys.executable,
                "-c",
                "import runpy, sys; "
                "runpy.run_path(sys.argv[1])['_guard_viewer']"
                "(int(sys.argv[2]), int(sys.argv[3]))",
                os.path.abspath(__file__),
                str(lifetime_read),
                str(ready_write),
            ],
            pass_fds=(lifetime_read, ready_write),
            start_new_session=True,
        )
    except BaseException:
        os.close(lifetime_write)
        os.close(ready_read)
        raise
    finally:
        os.close(lifetime_read)
        os.close(ready_write)

    _viewer_lifetime_fd = lifetime_write
    atexit.register(close)
    try:
        with os.fdopen(ready_read, "rb") as ready:
            port = ready.read()
            if not port:
                raise RuntimeError("Rerun viewer failed to start; see its error above")
        rr.connect_grpc(f"rerun+http://127.0.0.1:{int(port)}/proxy")
    except BaseException:
        close()
        raise


def _guard_viewer(lifetime_fd: int, ready_fd: int) -> None:
    """Run in a separate interpreter so cleanup survives the example's death."""
    import os
    import socket

    # pass_fds keeps these descriptors inheritable across the first exec.
    # The viewer must not inherit them and keep the readiness pipe open.
    os.set_inheritable(lifetime_fd, False)
    os.set_inheritable(ready_fd, False)
    pid = None
    try:
        # Do not fall back to rr.spawn(): it loses ownership of the process.
        from rerun._spawn import _spawn_viewer

        # The SDK silently reuses a viewer if the requested port is occupied,
        # returning no PID. An attached viewer needs its own port and process.
        with socket.socket() as endpoint:
            endpoint.bind(("127.0.0.1", 0))
            port = endpoint.getsockname()[1]
        pid = _spawn_viewer(port=port)
        if pid is None:
            raise RuntimeError("Rerun did not launch an owned viewer")
        with os.fdopen(ready_fd, "wb") as ready:
            ready_fd = -1
            ready.write(str(port).encode("ascii"))
        with os.fdopen(lifetime_fd, "rb") as lifetime:
            lifetime_fd = -1
            lifetime.read()
    finally:
        if ready_fd != -1:
            os.close(ready_fd)
        if lifetime_fd != -1:
            os.close(lifetime_fd)
        if pid is not None:
            _terminate_viewer(pid)


def close() -> None:
    """Close our viewer; externally supplied viewers remain untouched."""
    import os

    global _viewer_guard, _viewer_lifetime_fd
    guard, _viewer_guard = _viewer_guard, None
    fd, _viewer_lifetime_fd = _viewer_lifetime_fd, None
    if fd is not None:
        os.close(fd)
    if guard is not None:
        # The guard also covers exits that cannot run this function at all.
        guard.wait()


def _terminate_viewer(pid: int) -> None:
    """Terminate the wrapper and viewer binary, escalating if they resist."""
    import os
    import signal
    import time

    try:
        group = os.getpgid(pid)
    except ProcessLookupError:
        return
    # Rerun's detached spawn creates its own process group. Refuse to signal
    # a shared group if a future SDK changes that contract.
    if group != pid:
        os.kill(pid, signal.SIGTERM)
        return
    for sig in (signal.SIGTERM, signal.SIGTERM, signal.SIGKILL):
        try:
            os.killpg(group, sig)
        except ProcessLookupError:
            break
        if sig == signal.SIGKILL:
            break
        deadline = time.monotonic() + 0.5
        while time.monotonic() < deadline:
            try:
                os.killpg(group, 0)
            except ProcessLookupError:
                break
            time.sleep(0.05)
        else:
            continue
        break
    # Reap the wrapper if the SDK has not already done so.
    try:
        os.waitpid(pid, 0)
    except ChildProcessError:
        pass


def show_time_series(prefix: str, groups: list[str]) -> None:
    """Lays out one time-series panel per group with every series visible.

    Without a blueprint the viewer picks its own layout and selection, and a
    plot with many joints starts with some of them hidden. `prefix/group/*`
    is the entity layout the current monitor logs.
    """
    import rerun as rr
    import rerun.blueprint as rrb

    panels = [
        rrb.TimeSeriesView(name=group, origin=f"/{prefix}/{group}") for group in groups
    ]
    rr.send_blueprint(rrb.Blueprint(rrb.Grid(*panels), collapse_panels=True))


def set_sample_time(index: int, seconds: float) -> None:
    """Places the next log call on the timeline.

    Two timelines, because the useful question differs: `sample` for stepping
    through readings one at a time, `time` for anything whose shape over time
    is the point, like a current trace.
    """
    import rerun as rr

    rr.set_time("sample", sequence=index)
    rr.set_time("time", duration=seconds)


def log_frame(path: str, frame: dict) -> None:
    """Logs one camera frame, choosing the entity type from its shape.

    Colour arrives as `uint8 (H, W, 3)` and depth as `float32 (H, W)` metres.
    They are logged as different Rerun types on purpose: depth then gets a
    colour map and a readable metre value under the cursor, instead of being
    rendered as a nearly-black grayscale image.
    """
    import rerun as rr

    data = frame["data"]
    if data.ndim == 3 or data.dtype.kind in "uib":
        rr.log(path, rr.Image(data))
    else:
        rr.log(path, rr.DepthImage(data, meter=1.0))


def log_image(path: str, data) -> None:
    """Log a NumPy image once, or call repeatedly for live display.

    Integer images use image rendering; 2D float arrays are depth in metres.
    Call start() first and close() when finished. Install rerun-sdk separately.
    """
    log_frame(path, {"data": data})

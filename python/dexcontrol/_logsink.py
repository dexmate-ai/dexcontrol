# Copyright (C) 2026 Dexmate Inc.
# SPDX-License-Identifier: Apache-2.0

"""Log sink for the native runtime and for the thin Python layer.

The native runtime emits through ``tracing``; the binding forwards each
record here. The surrounding Python stack is loguru-based, so records go
to loguru and land in
whatever sinks an application already configured. Stdlib ``logging`` is a
fallback for installs without loguru, so the library still works if it is
absent.

Native-side filtering happens before this is called (see the ``DEXCONTROL_LOG``
/ ``RUST_LOG`` env vars), so this module is only reached by records that
already passed the level filter.
"""

from __future__ import annotations

import logging
import os
import sys

try:  # pragma: no cover - exercised by whichever branch the install has
    from loguru import logger as _loguru
except ImportError:  # pragma: no cover
    _loguru = None


_CONSOLE_HANDLER_ID: int | None = None


def configure_logging(*, replace_default: bool = True) -> bool:
    """Opt-in console styling for dexcontrol's (and the application's) records.

    dexcontrol is a library: importing it never touches logging
    configuration. Earlier 0.7 development builds restyled loguru's global
    logger as a side effect of the first native log record, from whichever
    thread emitted it; that is gone. Applications (and the bundled examples)
    that want that look call this once, from the main thread, before
    connecting::

        import dexcontrol
        dexcontrol.configure_logging()

    It adds one stderr sink to loguru with a compact format that colors
    WARNING yellow and ERROR/CRITICAL red (honouring ``NO_COLOR`` and
    non-TTY streams). With ``replace_default=True`` loguru's automatic
    stderr handler (id 0) is removed first so records are not printed twice;
    handlers the application added itself are never touched. Calling it
    again is a no-op. Native verbosity is filtered before records reach
    Python: set ``DEXCONTROL_LOG`` (or ``RUST_LOG``), for example
    ``DEXCONTROL_LOG=dexcontrol=debug``.

    Returns:
        True when the sink was installed by this call, False when loguru is
        not installed (records then go to stdlib ``logging``; configure that
        as usual) or the sink was already installed.
    """
    global _CONSOLE_HANDLER_ID
    if _loguru is None or _CONSOLE_HANDLER_ID is not None:
        return False
    if replace_default:
        try:
            _loguru.remove(0)
        except ValueError:
            pass  # the application already replaced loguru's default handler

    def console_format(record):
        color = {"WARNING": "yellow", "ERROR": "red", "CRITICAL": "red"}.get(
            record["level"].name
        )
        start, end = (f"<{color}>", f"</{color}>") if color else ("", "")
        return (
            "{time:YYYY-MM-DD HH:mm:ss.SSS} | "
            + start
            + "{level: <8}"
            + end
            + " | {name}:{function}:{line} - "
            + start
            + "{message}"
            + end
            + "\n{exception}"
        )

    _CONSOLE_HANDLER_ID = _loguru.add(
        sys.stderr,
        format=console_format,
        colorize=bool(getattr(sys.stderr, "isatty", lambda: False)())
        and "NO_COLOR" not in os.environ,
    )
    return True


# loguru level name -> stdlib numeric level, for the fallback path.
_STDLIB_LEVELS = {
    "TRACE": 5,
    "DEBUG": logging.DEBUG,
    "INFO": logging.INFO,
    "WARNING": logging.WARNING,
    "ERROR": logging.ERROR,
}


def _rewrite(target: str):
    """Replaces loguru's caller metadata with the native record's origin.

    Without this every record would be attributed to this module, so a
    format string containing ``{name}`` would read ``dexcontrol._logsink``
    for all of them instead of the emitting Rust module.
    """

    def patch(record) -> None:
        record["name"] = target
        record["function"] = "<native>"
        record["line"] = 0

    return patch


_held: list[tuple[str, str | None, str]] | None = None


class hold_console:
    """Context manager that parks records while an interactive prompt is open.

    Native records arrive on the bridge's drain thread at any moment, and one
    printed after ``input()`` has shown its question overwrites the question
    on a terminal. While a hold is active, :func:`emit` and :func:`log`
    append to a buffer instead of printing; leaving the hold replays the
    buffer in order, through the same sinks. Holds nest; the records are
    released when the outermost one ends.
    """

    _depth = 0

    def __enter__(self) -> "hold_console":
        global _held
        if hold_console._depth == 0:
            _held = []
        hold_console._depth += 1
        return self

    def __exit__(self, *exc: object) -> None:
        global _held
        hold_console._depth -= 1
        if hold_console._depth > 0:
            return
        held, _held = _held or [], None
        for level, target, message in held:
            if target is None:
                log(level, message)
            else:
                emit(level, target, message)


def emit(level: str, target: str, message: str) -> None:
    """Emits one native record. Never raises into the control path."""
    if _held is not None:
        _held.append((level, target, message))
        return
    try:
        if _loguru is not None:
            _loguru.patch(_rewrite(target)).log(level, message)
            return
        logging.getLogger(target).log(_STDLIB_LEVELS.get(level, logging.INFO), message)
    except Exception:  # noqa: BLE001 - logging must never break robot control
        pass


def log(level: str, message: str) -> None:
    """Emits one record originating in Python, from the caller's own frame.

    The thin Python layer has a little of its own to report. Routing it here
    rather than through stdlib ``logging`` keeps every dexcontrol record in
    one stream; unlike :func:`emit` the caller's real module, function and
    line survive, because these records did not come from Rust.
    """
    if _held is not None:
        _held.append((level, None, message))
        return
    try:
        if _loguru is not None:
            _loguru.opt(depth=1).log(level, message)
            return
        logging.getLogger(__name__).log(
            _STDLIB_LEVELS.get(level, logging.INFO), message
        )
    except Exception:  # noqa: BLE001 - logging must never break robot control
        pass


def uses_loguru() -> bool:
    """True when records are routed to loguru rather than stdlib logging."""
    return _loguru is not None

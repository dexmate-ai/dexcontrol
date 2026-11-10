# Copyright (C) 2026 Dexmate Inc.
# SPDX-License-Identifier: Apache-2.0

"""Hidden entry point for the dedicated-process safety watchdog.

``Robot`` spawns ``[sys.executable, "-m", "dexcontrol._watchdog_process"]``
so the wheel needs no extra binary: the native watchdog loop (heartbeat
staleness and E-stop monitoring on its own DexComm session, isolated from
the host application) runs inside this small dedicated interpreter with the
GIL released for the whole run. The configuration arrives as one JSON line
on stdin; events leave as JSON lines on stdout; the process exits on stdin
EOF, SIGTERM, or parent death.
"""

# The log filter for this process is set by the parent when it spawns us
# (see `child_command` in crates/dexcontrol/src/watchdog.rs): `python -m`
# imports the package — installing the log bridge, which reads the filter —
# before this module body would run, so it cannot be set here.
from dexcontrol._native import watchdog_main

if __name__ == "__main__":
    # The native loop ignores SIGINT itself (a terminal Ctrl-C reaches the
    # whole foreground group, and the watchdog must outlast the application's
    # shutdown). This keeps the interpreter from raising KeyboardInterrupt
    # before that handler is installed.
    import signal

    signal.signal(signal.SIGINT, signal.SIG_IGN)
    raise SystemExit(watchdog_main())

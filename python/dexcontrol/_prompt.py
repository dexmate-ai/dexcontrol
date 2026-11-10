# Copyright (C) 2026 Dexmate Inc.
# SPDX-License-Identifier: Apache-2.0

"""Interactive prompts that stay readable next to the native log stream.

A plain ``input()`` in a script that has a connected ``Robot`` is easy to
miss: the native runtime keeps logging from its own threads (connection
progress, the watchdog, safety notifications), and a record printed right
after the question lands on the same terminal line and buries it. These
helpers flush what the runtime has already logged, keep further records
back until the answer is in, and put the question on a line of its own.
"""

from __future__ import annotations

import sys

from dexcontrol import _logsink, _native

#: How long to wait for native records already emitted to reach the console.
_FLUSH_TIMEOUT = 0.5


def ask(question: str) -> str:
    """Shows ``question`` on its own line and returns the stripped answer.

    Native log records emitted before the call are printed first; records
    emitted while the prompt is open are held and printed after it. EOF on
    stdin (no terminal, or Ctrl-D) returns an empty string, which every
    confirmation treats as "no".
    """
    try:
        _native._log_flush(_FLUSH_TIMEOUT)
    except Exception:  # noqa: BLE001 - a prompt must work without the bridge
        pass
    with _logsink.hold_console():
        sys.stderr.flush()
        sys.stdout.flush()
        try:
            answer = input(f"\n>>> {question} ")
        except EOFError:
            answer = ""
        # A terminal echoes the Enter that ends the line; piped input does
        # not, and the next record would continue on the question's line.
        if not sys.stdin.isatty():
            print(file=sys.stderr)
        return answer.strip()


def confirm(question: str, *, expected: str = "y") -> bool:
    """Asks a yes/no question; only the expected answer (case-insensitive)
    counts as yes, and the default is always no.

    ``expected="y"`` renders as ``question [y/N]``; any other word, such as
    ``"yes"`` for hazardous actions, renders as ``question (type "yes")``.
    """
    suffix = "[y/N]" if expected == "y" else f'(type "{expected}")'
    return ask(f"{question} {suffix}").lower() == expected.lower()

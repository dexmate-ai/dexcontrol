# Copyright (C) 2026 Dexmate Inc.
# SPDX-License-Identifier: Apache-2.0

"""Read-only live sensing and diagnostics, with no arm initialization."""
from __future__ import annotations
import weakref
from dexcontrol import _native
from dexcontrol.sensors.manager import Sensors

class DiagnosticClient(_native.DiagnosticClient):
    """Use as a context manager. Supports the same typed sensors as Robot.

    ``simulation=True`` connects only to the in-process mock. This client
    exposes no motion or firmware-write methods and never initializes motors.
    """
    @property
    def sensors(self) -> Sensors:
        """Every sensor on this connection; see :class:`Sensors`.

        Cached on first use (the native base class owns ``__init__``), and
        the manager refers back to this client weakly for the same reason
        :attr:`dexcontrol.Robot.sensors` does.
        """
        manager = self.__dict__.get("_sensor_manager")
        if manager is None:
            manager = self._sensor_manager = Sensors(weakref.proxy(self))
        return manager

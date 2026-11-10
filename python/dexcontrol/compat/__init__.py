# Copyright (C) 2026 Dexmate Inc.
# SPDX-License-Identifier: Apache-2.0

"""Compatibility adapters for earlier DexControl Python APIs.

Each submodule wraps the current native runtime behind the surface of an
older release. The modern ``dexcontrol.Robot`` keeps a single spelling; all
legacy names live here and only here.

* :mod:`dexcontrol.compat.v05` -- the pre-0.6 pure-Python API (0.5.x).
"""

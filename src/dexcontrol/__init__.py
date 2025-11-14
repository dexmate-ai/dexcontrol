# Copyright (C) 2025 Dexmate Inc.
#
# This software is dual-licensed:
#
# 1. GNU Affero General Public License v3.0 (AGPL-3.0)
#    See LICENSE-AGPL for details
#
# 2. Commercial License
#    For commercial licensing terms, contact: contact@dexmate.ai
#    See README.md for details

"""DexControl: Robot Control Interface Library.

This package provides interfaces for controlling and monitoring robot systems.
It serves as the primary API for interacting with Dexmate robots.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Final

from loguru import logger
from rich.logging import RichHandler

# DO NOT REMOVE this following import, it is needed for hydra to find the config
import dexcontrol.config  # pylint: disable=unused-import
from dexcontrol.robot import Robot
from dexcontrol.utils.constants import COMM_CFG_PATH_ENV_VAR

# Package-level constants
LIB_PATH: Final[Path] = Path(__file__).resolve().parent
CFG_PATH: Final[Path] = LIB_PATH / "config"
MIN_SOC_SOFTWARE_VERSION: int = 298

logger.configure(
    handlers=[
        {"sink": RichHandler(markup=True), "format": "{message}", "level": "INFO"}
    ]
)


def get_comm_cfg_path() -> Path | None:
    default_path = list(
        Path("~/.dexmate/comm/zenoh/").expanduser().glob("**/zenoh_peer_config.json5")
    )
    if len(default_path) == 0:
        logger.debug(
            "No zenoh_peer_config.json5 file found in ~/.dexmate/comm/zenoh/ - will use DexComm defaults"
        )
        return None
    return default_path[0]


# Try to get comm config path, but allow None
_comm_cfg = os.getenv(COMM_CFG_PATH_ENV_VAR)
if _comm_cfg:
    COMM_CFG_PATH: Final[Path] = Path(_comm_cfg).expanduser()
else:
    _default = get_comm_cfg_path()
    COMM_CFG_PATH: Final[Path] = _default if _default else Path("/tmp/no_config")

ROBOT_CFG_PATH: Final[Path] = CFG_PATH

__all__ = ["Robot", "LIB_PATH", "CFG_PATH", "COMM_CFG_PATH", "ROBOT_CFG_PATH"]

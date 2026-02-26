# Copyright (C) 2025 Dexmate Inc.
#
# This software is dual-licensed:
#
# 1. GNU Affero General Public License v3.0 (AGPL-3.0)
#    See LICENSE-AGPL for details
#
# 2. Commercial License
#    For commercial licensing terms, contact: contact@dexmate.ai

"""Robotiq 2F-85 gripper adapter for dexcontrol.

Wraps the robotiq_2f_85_controller package to expose the same interface as
DexGripper/Hand so it can be used as a drop-in replacement in VegaRobot.

Requirements:
    pip install robotiq-2f-85-controller
    (or via submodule: pip install -e robotiq_2f_85_controller/)
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
from loguru import logger

# Ensure the submodule takes precedence over any namespace package with the
# same name that may shadow it (e.g. when /home/dexmate/robotiq_2f_85_controller
# is not on sys.path but a .pth editable install points elsewhere).
_SUBMODULE = Path(__file__).resolve().parents[3] / "robotiq_2f_85_controller"
if _SUBMODULE.exists() and str(_SUBMODULE) not in sys.path:
    sys.path.insert(0, str(_SUBMODULE))

try:
    from robotiq_2f_85_controller import Robotiq2FingerGripper
except ImportError as e:
    raise ImportError(
        "robotiq_2f_85_controller is not installed. "
        "Run: pip install -e robotiq_2f_85_controller/ "
        "(or pip install robotiq-2f-85-controller)"
    ) from e

# Robotiq 2F-85 stroke in metres
_STROKE_M = 0.085

# Predefined positions in metres (open = max stroke, close = 0)
_POSE_POOL = {
    "open": np.array([_STROKE_M], dtype=np.float64),
    "close": np.array([0.0], dtype=np.float64),
}


class RobotiqGripper:
    """Robotiq 2F-85 gripper controller compatible with the VegaRobot hand interface.

    Joint space is a single scalar in metres [0, 0.085] where 0 is fully open
    and 0.085 is fully closed, matching Robotiq2FingerGripper.get_pos().
    """

    def __init__(
        self,
        comport: str = "/dev/ttyUSB0",
        init_timeout: float = 15.0,
    ) -> None:
        """Connect to the gripper and run full initialisation.

        Args:
            comport: Serial port the gripper is connected to.
            init_timeout: Seconds to wait for the gripper to become ready.
        """
        logger.info("Connecting to Robotiq 2F-85 on %s …", comport)
        self._gripper = Robotiq2FingerGripper(comport=comport)
        self._gripper.full_init(timeout=init_timeout)
        logger.info("Robotiq 2F-85 ready.")

    # ------------------------------------------------------------------
    # VegaRobot hand interface
    # ------------------------------------------------------------------

    def get_joint_pos(self) -> np.ndarray:
        """Return current gripper position as a 1-element array in metres."""
        self._gripper.getStatus()
        return np.array([self._gripper.get_pos()], dtype=np.float64)

    def set_joint_pos(
        self,
        joint_pos,
        wait_time: float = 0.0,
        **_kwargs,
    ) -> None:
        """Command gripper to a position in metres.

        Args:
            joint_pos: Target position. Scalar, list, or 1-element array in
                metres, clipped to [0, stroke].
            wait_time: Seconds to sleep after sending the command.
        """
        pos_m = float(np.asarray(joint_pos, dtype=np.float64).flat[0])
        pos_m = float(np.clip(pos_m, 0.0, _STROKE_M))
        self._gripper.goto(pos=pos_m, vel=0.05, force=50)
        self._gripper.sendCommand()
        if wait_time > 0.0:
            time.sleep(wait_time)

    def open_hand(self, wait_time: float = 0.0, **_kwargs) -> None:
        """Open the gripper fully."""
        self._gripper.goto(pos=_STROKE_M, vel=0.05, force=50)
        self._gripper.sendCommand()
        if wait_time > 0.0:
            time.sleep(wait_time)

    def close_hand(self, wait_time: float = 0.0, **_kwargs) -> None:
        """Close the gripper fully."""
        self._gripper.goto(pos=0.0, vel=0.05, force=50)
        self._gripper.sendCommand()
        if wait_time > 0.0:
            time.sleep(wait_time)

    def get_predefined_pose(self, name: str) -> np.ndarray:
        """Return a predefined pose by name ('open' or 'close').

        Args:
            name: Pose name.

        Returns:
            1-element numpy array with position in metres.

        Raises:
            KeyError: If name is not a known predefined pose.
        """
        if name not in _POSE_POOL:
            raise KeyError(f"Unknown predefined pose '{name}'. Available: {list(_POSE_POOL)}")
        return _POSE_POOL[name].copy()

    def shutdown(self) -> None:
        """Disconnect from the gripper."""
        try:
            self._gripper.shutdown()
        except Exception:
            pass

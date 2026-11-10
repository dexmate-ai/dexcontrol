# Copyright (C) 2026 Dexmate Inc.
# SPDX-License-Identifier: Apache-2.0

"""Stable language-neutral motion state names and the typed sensor/joint results."""
from __future__ import annotations
from enum import Enum
from typing import TypedDict

import numpy as np

class MotionState(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    CANCELLED = "cancelled"
    FAILED = "failed"
    SUPERSEDED = "superseded"


class CameraFrame(TypedDict):
    """Owned NumPy data: uint8 images or float32 depth in metres."""
    data: np.ndarray
    timestamp_ns: int
    frame_id: str

class FrameSample(CameraFrame):
    """One frame plus a local sequence and arrival age in seconds."""
    sequence: int
    age: float

class JointState(TypedDict):
    """Named channels; absent channels are empty arrays, never invented zeros.

    Revolute position/velocity: rad, rad/s; current: A; torque: N m.
    Prismatic joints use m and m/s; consult component configuration.
    """
    position: np.ndarray
    velocity: np.ndarray
    current: np.ndarray
    torque: np.ndarray
    timestamp_ns: int
    frame_id: str
    error_joint_indices: np.ndarray

class Lidar2dObservation(TypedDict):
    ranges: np.ndarray
    angles: np.ndarray
    qualities: np.ndarray
    timestamp: int
    timestamp_ns: int

class JointSample(JointState):
    age: float | None

class ImuObservation(TypedDict, total=False):
    """Selected float64 channels; timestamp_ns is always present."""
    timestamp_ns: int
    acc: np.ndarray
    gyro: np.ndarray
    quat: np.ndarray
    mag: np.ndarray

class Lidar3dObservation(TypedDict):
    x: np.ndarray
    y: np.ndarray
    z: np.ndarray
    intensity: np.ndarray
    ring: np.ndarray
    point_timestamps_ns: np.ndarray
    timestamp_ns: int
    height: int
    width: int
    is_dense: bool
    point_count: int

"""Tests for Robot motion-plugin target APIs against managed components.

Note: Robot motion APIs wrap all exceptions raised during validation in
DexcontrolError (with the original as __cause__), matching the existing
behaviour for validate_component_names. Tests expect the wrapped form.
"""

import itertools
import threading

import numpy as np
import pytest

from dexcontrol.core.component import (
    ManagedJointComponent,
    MotionPluginManaged,
    RobotJointComponent,
)
from dexcontrol.core.motion_handle import MotionHandle, MultiMotionHandle
from dexcontrol.exceptions import DexcontrolError


class _FakePolicyManager:
    def __init__(self):
        self.touches = 0

    def touch(self):
        self.touches += 1


class _FakePublisher:
    def __init__(self):
        self.messages = []

    def publish(self, msg):
        self.messages.append(msg)


class _MotionApiComponent(ManagedJointComponent):
    """Managed component shell that exercises inherited motion API methods."""

    def __init__(self):
        self._joint_name = ["j0", "j1"]
        self._joint_pos_limit = np.array([[-1.0, 1.0], [-2.0, 2.0]])
        self._default_velocity_scale = None
        self._target_publisher = _FakePublisher()
        self._motion_status_subscriber = object()
        self._motion_status_lock = threading.Lock()
        self._motion_id_counter = itertools.count(42)
        self._active_handles = {}
        self._policy_manager = _FakePolicyManager()

    def get_joint_pos(self):
        return np.array([0.1, 0.2], dtype=np.float32)

    def get_joint_pos_dict(self):
        return {"j0": 0.1, "j1": 0.2}


class _FakeArm(ManagedJointComponent):
    """Stand-in for Arm that skips real initialisation (no node, no zenoh)."""

    def __init__(self):
        self._joint_name = ["j0"]
        self.calls = []
        self.handle = MotionHandle(motion_id=7, publish_cancel_fn=lambda _: None)

    def move_joint_pos(self, pos, velocity_scale=None, relative=False):
        self.calls.append(
            {
                "method": "move_joint_pos",
                "pos": pos,
                "velocity_scale": velocity_scale,
                "relative": relative,
            }
        )
        return None

    def move_to_joint_pos(self, pos, velocity_scale=None, relative=False):
        self.calls.append(
            {
                "method": "move_to_joint_pos",
                "pos": pos,
                "velocity_scale": velocity_scale,
                "relative": relative,
            }
        )
        return self.handle


class _FakeHand(RobotJointComponent):
    """Stand-in for Hand that skips real initialisation."""

    def __init__(self):
        self._joint_name = ["j0"]


def test_validate_managed_targets_rejects_non_managed(monkeypatch):
    """Robot.move_joint_pos should raise DexcontrolError naming the bad component."""
    from dexcontrol.robot import Robot

    # Build a Robot stub that returns our fakes from get_controllable_component_map.
    robot = Robot.__new__(Robot)
    robot._shutdown_called = True
    arm = _FakeArm()
    hand = _FakeHand()
    monkeypatch.setattr(
        robot,
        "get_controllable_component_map",
        lambda: {"left_arm": arm, "left_hand": hand},
    )

    with pytest.raises(DexcontrolError) as exc_info:
        robot.move_joint_pos({"left_hand": np.zeros(1)})

    # The underlying validation error is a ValueError.
    assert isinstance(exc_info.value.__cause__, ValueError)
    msg = str(exc_info.value)
    assert "move_joint_pos()" in msg
    assert "left_hand" in msg
    assert "left_arm" in msg  # supported components listed


def test_validate_managed_targets_accepts_managed(monkeypatch):
    """When only managed components are targeted, no validation error is raised."""
    from dexcontrol.robot import Robot

    robot = Robot.__new__(Robot)
    robot._shutdown_called = True
    arm = _FakeArm()
    monkeypatch.setattr(
        robot, "get_controllable_component_map", lambda: {"left_arm": arm}
    )

    # Should not raise. Result is None because move_joint_pos is untracked.
    result = robot.move_joint_pos({"left_arm": np.zeros(1)})
    assert result is None
    assert arm.calls[-1]["method"] == "move_joint_pos"


def test_fake_arm_is_managed():
    """Sanity: _FakeArm classifies as ManagedJointComponent + MotionPluginManaged."""
    arm = _FakeArm()
    assert isinstance(arm, ManagedJointComponent)
    assert isinstance(arm, MotionPluginManaged)
    assert isinstance(arm, RobotJointComponent)


def test_set_joint_target_api_is_removed():
    """The old tracked flag API should not remain on robot or components."""
    from dexcontrol.robot import Robot

    assert not hasattr(ManagedJointComponent, "set_joint_target")
    assert not hasattr(Robot, "set_joint_target")


def test_move_joint_pos_publishes_untracked_target():
    """move_joint_pos should publish a fire-and-forget target without motion_id."""
    component = _MotionApiComponent()

    result = component.move_joint_pos([2.0, -3.0], velocity_scale=0.25)

    assert result is None
    assert component._policy_manager.touches == 1
    assert len(component._target_publisher.messages) == 1
    msg = component._target_publisher.messages[0]
    np.testing.assert_allclose(msg["pos"], np.array([1.0, -2.0]))
    assert msg["scale"] == [0.25, 0.25]
    assert "motion_id" not in msg


def test_move_to_joint_pos_publishes_tracked_target_with_handle():
    """move_to_joint_pos should publish a tracked target and return its handle."""
    component = _MotionApiComponent()

    handle = component.move_to_joint_pos({"j1": 1.5})

    assert isinstance(handle, MotionHandle)
    assert handle.motion_id == 42
    assert component._active_handles[42] is handle
    assert component._policy_manager.touches == 1
    assert len(component._target_publisher.messages) == 1
    msg = component._target_publisher.messages[0]
    np.testing.assert_allclose(msg["pos"], np.array([0.1, 1.5]))
    assert msg["motion_id"] == 42
    assert "scale" not in msg


def test_robot_move_joint_pos_delegates_to_managed_components(monkeypatch):
    """Robot.move_joint_pos should fan out without returning a handle."""
    from dexcontrol.robot import Robot

    robot = Robot.__new__(Robot)
    robot._shutdown_called = True
    arm = _FakeArm()
    target = np.zeros(1)
    monkeypatch.setattr(
        robot, "get_controllable_component_map", lambda: {"left_arm": arm}
    )

    result = robot.move_joint_pos(
        {"left_arm": target}, velocity_scale={"left_arm": 0.5}, relative=True
    )

    assert result is None
    assert arm.calls[-1]["method"] == "move_joint_pos"
    assert arm.calls[-1]["pos"] is target
    assert arm.calls[-1]["velocity_scale"] == 0.5
    assert arm.calls[-1]["relative"] is True


def test_robot_move_to_joint_pos_returns_multi_motion_handle(monkeypatch):
    """Robot.move_to_joint_pos should return a MultiMotionHandle."""
    from dexcontrol.robot import Robot

    robot = Robot.__new__(Robot)
    robot._shutdown_called = True
    arm = _FakeArm()
    target = np.zeros(1)
    monkeypatch.setattr(
        robot, "get_controllable_component_map", lambda: {"left_arm": arm}
    )

    handle = robot.move_to_joint_pos({"left_arm": target})

    assert isinstance(handle, MultiMotionHandle)
    assert handle._handles["left_arm"] is arm.handle
    assert arm.calls[-1]["method"] == "move_to_joint_pos"
    assert arm.calls[-1]["pos"] is target

"""Structural invariants for the component class hierarchy.

These tests guard the type contract laid out in
``docs/specs/2026-05-17-managed-joint-component-hierarchy-design.md``:
which concrete classes are motion-plugin-managed and which are not.
"""

import pytest

from dexcontrol.core.arm import Arm
from dexcontrol.core.chassis import Chassis, ChassisDrive, ChassisSteer
from dexcontrol.core.component import (
    ManagedJointComponent,
    MotionPluginManaged,
    RobotJointComponent,
)
from dexcontrol.core.hand import DexGripper, Hand, HandF5D6, HandF5D6V2
from dexcontrol.core.head import Head
from dexcontrol.core.torso import Torso

# Classes that should be motion-plugin-managed joint components.
MANAGED_JOINT_CLASSES = [Arm, Head, Torso]

# Classes that are RobotJointComponent but NOT motion-plugin-managed.
RAW_JOINT_CLASSES = [
    Hand,
    HandF5D6,
    HandF5D6V2,
    DexGripper,
    ChassisSteer,
    ChassisDrive,
]


@pytest.mark.parametrize("cls", MANAGED_JOINT_CLASSES)
def test_managed_joint_classes_subclass_managed(cls):
    """Arm/Head/Torso must subclass ManagedJointComponent."""
    assert issubclass(cls, ManagedJointComponent)


@pytest.mark.parametrize("cls", MANAGED_JOINT_CLASSES)
def test_managed_joint_classes_are_motion_plugin_managed(cls):
    """They also pick up the marker mixin via ManagedJointComponent."""
    assert issubclass(cls, MotionPluginManaged)


@pytest.mark.parametrize("cls", MANAGED_JOINT_CLASSES)
def test_managed_joint_classes_are_robot_joint_components(cls):
    """The managed subclass still IS a RobotJointComponent."""
    assert issubclass(cls, RobotJointComponent)


@pytest.mark.parametrize("cls", RAW_JOINT_CLASSES)
def test_raw_joint_classes_are_robot_joint_components(cls):
    """Hand/Gripper/ChassisSteer/ChassisDrive are RobotJointComponents."""
    assert issubclass(cls, RobotJointComponent)


@pytest.mark.parametrize("cls", RAW_JOINT_CLASSES)
def test_raw_joint_classes_are_not_managed(cls):
    """They must NOT inherit ManagedJointComponent."""
    assert not issubclass(cls, ManagedJointComponent)


@pytest.mark.parametrize("cls", RAW_JOINT_CLASSES)
def test_raw_joint_classes_are_not_motion_plugin_managed(cls):
    """They must NOT carry the MotionPluginManaged marker."""
    assert not issubclass(cls, MotionPluginManaged)


@pytest.mark.parametrize("cls", RAW_JOINT_CLASSES)
def test_raw_joint_classes_have_no_motion_plugin_methods(cls):
    """The motion-plugin API surface must be absent on these classes."""
    assert not hasattr(cls, "set_joint_target")
    assert not hasattr(cls, "move_joint_pos")
    assert not hasattr(cls, "move_to_joint_pos")
    assert not hasattr(cls, "go_to_pose")
    assert not hasattr(cls, "_publish_cancel")
    assert not hasattr(cls, "_ensure_target_publisher")
    assert not hasattr(cls, "_ensure_status_subscriber")
    # default_velocity_scale moved to ManagedJointComponent too.
    assert not hasattr(cls, "default_velocity_scale")


@pytest.mark.parametrize("cls", MANAGED_JOINT_CLASSES)
def test_managed_joint_classes_expose_motion_plugin_methods(cls):
    """The motion-plugin API surface must be present on managed subclasses."""
    assert not hasattr(cls, "set_joint_target")
    assert hasattr(cls, "move_joint_pos")
    assert hasattr(cls, "move_to_joint_pos")
    assert hasattr(cls, "go_to_pose")
    assert hasattr(cls, "_publish_cancel")
    assert hasattr(cls, "_ensure_target_publisher")
    assert hasattr(cls, "_ensure_status_subscriber")
    assert hasattr(cls, "default_velocity_scale")


def test_chassis_is_not_managed_joint_component():
    """Chassis is NOT a joint component — it composes ChassisSteer/Drive."""
    assert not issubclass(Chassis, ManagedJointComponent)
    assert not issubclass(Chassis, RobotJointComponent)


def test_supports_motion_target_attribute_removed():
    """The legacy boolean flag must be gone from every class in the hierarchy."""
    for cls in (
        RobotJointComponent,
        ManagedJointComponent,
        *MANAGED_JOINT_CLASSES,
        *RAW_JOINT_CLASSES,
        Chassis,
    ):
        assert not hasattr(cls, "_supports_motion_target"), (
            f"{cls.__name__} still defines _supports_motion_target"
        )

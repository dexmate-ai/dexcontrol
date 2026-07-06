"""Tests for MotionPluginManaged marker mixin and its motion_id helper."""

from collections.abc import Iterator

from dexcontrol.core.component import MotionPluginManaged


def test_mixin_is_importable():
    """The mixin class exists and is bare (no required init args)."""
    instance = MotionPluginManaged()  # type: ignore[abstract]
    assert isinstance(instance, MotionPluginManaged)


def test_new_motion_id_counter_returns_iterator_of_ints():
    counter = MotionPluginManaged._new_motion_id_counter()
    assert isinstance(counter, Iterator)
    first = next(counter)
    second = next(counter)
    assert isinstance(first, int)
    assert isinstance(second, int)
    assert second == first + 1


def test_new_motion_id_counter_has_random_start():
    """Two counters should (almost certainly) start at different values."""
    starts = {next(MotionPluginManaged._new_motion_id_counter()) for _ in range(10)}
    # With a 2**32 random range, the probability of even two collisions
    # in 10 samples is astronomically small.
    assert len(starts) > 5, f"Counter starts collided too often: {starts}"


def test_new_motion_id_counter_start_in_valid_range():
    starts = [next(MotionPluginManaged._new_motion_id_counter()) for _ in range(100)]
    assert all(1 <= s <= 2**32 for s in starts)

"""Tests for MotionHandle state machine logic."""

import threading
import time

import pytest

from dexcontrol.core.motion_handle import MotionHandle, MotionState
from dexcontrol.exceptions import PluginNotAvailableError


class FakeStatusDispatcher:
    """Test helper that simulates status events from the motion plugin."""

    def __init__(self):
        self._handles: dict[int, MotionHandle] = {}

    def register(self, handle: MotionHandle) -> None:
        self._handles[handle.motion_id] = handle

    def unregister(self, motion_id: int) -> None:
        self._handles.pop(motion_id, None)

    def emit(self, motion_id: int, state: MotionState, message: str = "") -> None:
        handle = self._handles.get(motion_id)
        if handle is not None:
            handle._update_state(state, message)


def _make_handle(dispatcher: FakeStatusDispatcher, motion_id: int = 1) -> MotionHandle:
    """Create a MotionHandle and register it with the dispatcher."""
    handle = MotionHandle(
        motion_id=motion_id,
        publish_cancel_fn=lambda mid: None,
    )
    dispatcher.register(handle)
    return handle


class TestMotionHandleProperties:
    def test_initial_state_is_accepted(self):
        dispatcher = FakeStatusDispatcher()
        handle = _make_handle(dispatcher)
        assert handle.state == "accepted"
        assert handle.motion_id == 1
        assert not handle.is_done

    def test_state_transitions_to_finished(self):
        dispatcher = FakeStatusDispatcher()
        handle = _make_handle(dispatcher)
        dispatcher.emit(1, "finished")
        assert handle.state == "finished"
        assert handle.is_done

    def test_state_transitions_to_cancelled(self):
        dispatcher = FakeStatusDispatcher()
        handle = _make_handle(dispatcher)
        dispatcher.emit(1, "cancelled")
        assert handle.state == "cancelled"
        assert handle.is_done

    def test_state_transitions_to_error(self):
        dispatcher = FakeStatusDispatcher()
        handle = _make_handle(dispatcher)
        dispatcher.emit(1, "error", "Ruckig fault")
        assert handle.state == "error"
        assert handle.is_done


class TestMotionHandleWait:
    def test_wait_returns_immediately_if_already_done(self):
        dispatcher = FakeStatusDispatcher()
        handle = _make_handle(dispatcher)
        dispatcher.emit(1, "finished")
        result = handle.wait(timeout=1.0)
        assert result == "finished"

    def test_wait_blocks_until_finished(self):
        dispatcher = FakeStatusDispatcher()
        handle = _make_handle(dispatcher)

        def emit_later():
            time.sleep(0.1)
            dispatcher.emit(1, "finished")

        t = threading.Thread(target=emit_later)
        t.start()
        result = handle.wait(timeout=2.0)
        t.join()
        assert result == "finished"

    def test_wait_raises_plugin_unavailable_when_no_status_received(self):
        # No status messages from the plugin → timeout should surface as
        # PluginNotAvailableError, distinguishing "plugin not running"
        # from "motion still in flight".
        dispatcher = FakeStatusDispatcher()
        handle = _make_handle(dispatcher)
        with pytest.raises(PluginNotAvailableError):
            handle.wait(timeout=0.1)

    def test_wait_raises_timeout_when_status_received_but_not_done(self):
        # Plugin has acknowledged the motion (sent "accepted") but it hasn't
        # finished → timeout should remain TimeoutError, not PluginNotAvailableError.
        dispatcher = FakeStatusDispatcher()
        handle = _make_handle(dispatcher)
        dispatcher.emit(1, "accepted")
        with pytest.raises(
            TimeoutError,
            match="Consider increasing the wait timeout",
        ):
            handle.wait(timeout=0.1)

    def test_wait_raises_runtime_error_on_error_state(self):
        # Motion ended in "error" state → wait() must raise RuntimeError
        # so callers cannot silently ignore a plugin-reported failure.
        dispatcher = FakeStatusDispatcher()
        handle = _make_handle(dispatcher)
        dispatcher.emit(1, "error", "Ruckig fault")
        with pytest.raises(RuntimeError, match="Ruckig fault"):
            handle.wait(timeout=1.0)

    def test_wait_returns_cancelled(self):
        dispatcher = FakeStatusDispatcher()
        handle = _make_handle(dispatcher)
        dispatcher.emit(1, "cancelled")
        result = handle.wait(timeout=1.0)
        assert result == "cancelled"


class TestMotionHandleCancel:
    def test_cancel_publishes_cancel_message(self):
        cancelled_ids = []

        def track_cancel(mid):
            cancelled_ids.append(mid)

        handle = MotionHandle(motion_id=42, publish_cancel_fn=track_cancel)
        handle.cancel()
        assert cancelled_ids == [42]

    def test_cancel_on_already_done_is_noop(self):
        cancelled_ids = []
        dispatcher = FakeStatusDispatcher()

        handle = MotionHandle(
            motion_id=1,
            publish_cancel_fn=lambda mid: cancelled_ids.append(mid),
        )
        dispatcher.register(handle)
        dispatcher.emit(1, "finished")
        handle.cancel()
        assert cancelled_ids == []  # Should not publish cancel for finished motion

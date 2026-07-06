# Copyright (C) 2025 Dexmate Inc.
#
# This software is dual-licensed:
#
# 1. GNU Affero General Public License v3.0 (AGPL-3.0)
#    See LICENSE-AGPL for details
#
# 2. Commercial License
#    For commercial licensing terms, contact: contact@dexmate.ai

"""Tests for subscription policy management."""

import time
from unittest.mock import MagicMock

import pytest

from dexcontrol.core.subscription_policy import (
    IdleMonitor,
    SubscriptionPolicy,
    SubscriptionPolicyManager,
    SubscriptionPolicyMixin,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_subscriber(paused=False, latest="data"):
    """Create a MagicMock subscriber with standard interface."""
    sub = MagicMock()
    sub.is_paused.return_value = paused
    sub.get_latest.return_value = latest
    sub.pause.return_value = None
    sub.resume.return_value = None
    return sub


# ===========================================================================
# TestSubscriptionPolicy
# ===========================================================================


class TestSubscriptionPolicy:
    """Tests for the SubscriptionPolicy enum."""

    def test_enum_values(self):
        assert SubscriptionPolicy.ALWAYS_ON == "always_on"
        assert SubscriptionPolicy.AUTO == "auto"
        assert SubscriptionPolicy.MANUAL == "manual"
        assert SubscriptionPolicy.ALWAYS_OFF == "always_off"

    def test_string_conversion(self):
        assert str(SubscriptionPolicy.ALWAYS_ON) == "always_on"
        assert str(SubscriptionPolicy.AUTO) == "auto"

    def test_from_string(self):
        assert SubscriptionPolicy("always_on") is SubscriptionPolicy.ALWAYS_ON
        assert SubscriptionPolicy("auto") is SubscriptionPolicy.AUTO
        assert SubscriptionPolicy("manual") is SubscriptionPolicy.MANUAL
        assert SubscriptionPolicy("always_off") is SubscriptionPolicy.ALWAYS_OFF

    def test_invalid_string(self):
        with pytest.raises(ValueError):
            SubscriptionPolicy("invalid")


# ===========================================================================
# TestSubscriptionPolicyManager
# ===========================================================================


class TestSubscriptionPolicyManager:
    """Tests for the SubscriptionPolicyManager."""

    def test_default_policy_is_auto(self):
        sub = _make_subscriber()
        mgr = SubscriptionPolicyManager(sub, "test")
        assert mgr.get_policy() is SubscriptionPolicy.AUTO

    def test_set_policy_changes_policy(self):
        sub = _make_subscriber()
        mgr = SubscriptionPolicyManager(sub, "test")
        mgr.set_policy(SubscriptionPolicy.MANUAL)
        assert mgr.get_policy() is SubscriptionPolicy.MANUAL

    # -- ALWAYS_ON ----------------------------------------------------------

    def test_always_on_passthrough(self):
        sub = _make_subscriber(latest="sensor_data")
        mgr = SubscriptionPolicyManager(
            sub, "test", default_policy=SubscriptionPolicy.ALWAYS_ON
        )
        result = mgr.get_latest_managed()
        assert result == "sensor_data"
        sub.get_latest.assert_called_once()

    def test_always_on_ignores_pause(self):
        sub = _make_subscriber()
        mgr = SubscriptionPolicyManager(
            sub, "test", default_policy=SubscriptionPolicy.ALWAYS_ON
        )
        mgr.pause()
        sub.pause.assert_not_called()

    # -- ALWAYS_OFF ---------------------------------------------------------

    def test_always_off_returns_none(self):
        sub = _make_subscriber(paused=True)
        mgr = SubscriptionPolicyManager(
            sub, "test", default_policy=SubscriptionPolicy.ALWAYS_OFF
        )
        result = mgr.get_latest_managed()
        assert result is None
        sub.get_latest.assert_not_called()

    def test_always_off_pauses_on_init(self):
        sub = _make_subscriber()
        SubscriptionPolicyManager(
            sub, "test", default_policy=SubscriptionPolicy.ALWAYS_OFF
        )
        sub.pause.assert_called_once()

    def test_always_off_ignores_resume(self):
        sub = _make_subscriber(paused=True)
        mgr = SubscriptionPolicyManager(
            sub, "test", default_policy=SubscriptionPolicy.ALWAYS_OFF
        )
        sub.pause.reset_mock()
        mgr.resume()
        sub.resume.assert_not_called()

    # -- MANUAL -------------------------------------------------------------

    def test_manual_passthrough_when_active(self):
        sub = _make_subscriber(paused=False, latest="manual_data")
        mgr = SubscriptionPolicyManager(
            sub, "test", default_policy=SubscriptionPolicy.MANUAL
        )
        result = mgr.get_latest_managed()
        assert result == "manual_data"

    def test_manual_returns_none_when_paused(self):
        sub = _make_subscriber(paused=True)
        mgr = SubscriptionPolicyManager(
            sub, "test", default_policy=SubscriptionPolicy.MANUAL
        )
        result = mgr.get_latest_managed()
        assert result is None

    # -- AUTO ---------------------------------------------------------------

    def test_auto_passthrough_when_active(self):
        sub = _make_subscriber(paused=False, latest="auto_data")
        mgr = SubscriptionPolicyManager(sub, "test")
        result = mgr.get_latest_managed()
        assert result == "auto_data"

    def test_auto_updates_last_access_time(self):
        sub = _make_subscriber(paused=False, latest="auto_data")
        mgr = SubscriptionPolicyManager(sub, "test")
        before = time.monotonic()
        mgr.get_latest_managed()
        after = time.monotonic()
        assert before <= mgr._last_access_time <= after

    def test_auto_resume_when_paused(self):
        """AUTO policy: when paused, get_latest_managed resumes and polls for data."""
        sub = _make_subscriber(paused=True, latest="fresh_data")
        mgr = SubscriptionPolicyManager(sub, "test", resume_timeout=0.5)

        # After resume(), is_paused should return False
        def resume_side_effect():
            sub.is_paused.return_value = False

        sub.resume.side_effect = resume_side_effect

        result = mgr.get_latest_managed()
        sub.resume.assert_called_once()
        assert result == "fresh_data"

    def test_auto_resume_timeout_returns_none(self):
        """AUTO policy: returns None when no fresh data within resume_timeout."""
        sub = _make_subscriber(paused=True)
        mgr = SubscriptionPolicyManager(sub, "test", resume_timeout=0.1)

        # Resume but get_latest always returns None
        def resume_side_effect():
            sub.is_paused.return_value = False

        sub.resume.side_effect = resume_side_effect
        sub.get_latest.return_value = None

        result = mgr.get_latest_managed()
        assert result is None
        sub.resume.assert_called_once()

    # -- check_idle ---------------------------------------------------------

    def test_check_idle_pauses_after_timeout(self):
        sub = _make_subscriber(paused=False)
        mgr = SubscriptionPolicyManager(sub, "test", idle_timeout=0.1)
        # Simulate old access time
        mgr._last_access_time = time.monotonic() - 1.0
        mgr.check_idle()
        sub.pause.assert_called_once()

    def test_check_idle_noop_when_recently_accessed(self):
        sub = _make_subscriber(paused=False)
        mgr = SubscriptionPolicyManager(sub, "test", idle_timeout=5.0)
        mgr._last_access_time = time.monotonic()
        mgr.check_idle()
        sub.pause.assert_not_called()

    def test_check_idle_noop_for_always_on(self):
        sub = _make_subscriber(paused=False)
        mgr = SubscriptionPolicyManager(
            sub, "test", default_policy=SubscriptionPolicy.ALWAYS_ON
        )
        mgr._last_access_time = time.monotonic() - 100.0
        mgr.check_idle()
        sub.pause.assert_not_called()

    # -- Other methods ------------------------------------------------------

    def test_set_idle_timeout(self):
        sub = _make_subscriber()
        mgr = SubscriptionPolicyManager(sub, "test")
        mgr.set_idle_timeout(10.0)
        assert mgr._idle_timeout == 10.0

    def test_is_paused_delegates(self):
        sub = _make_subscriber(paused=True)
        mgr = SubscriptionPolicyManager(sub, "test")
        assert mgr.is_paused() is True
        sub.is_paused.assert_called()

    def test_set_policy_to_always_off_pauses(self):
        sub = _make_subscriber(paused=False)
        mgr = SubscriptionPolicyManager(
            sub, "test", default_policy=SubscriptionPolicy.MANUAL
        )
        mgr.set_policy(SubscriptionPolicy.ALWAYS_OFF)
        sub.pause.assert_called_once()

    def test_set_policy_to_always_on_resumes_if_paused(self):
        sub = _make_subscriber(paused=True)
        mgr = SubscriptionPolicyManager(
            sub, "test", default_policy=SubscriptionPolicy.MANUAL
        )
        mgr.set_policy(SubscriptionPolicy.ALWAYS_ON)
        sub.resume.assert_called_once()


# ===========================================================================
# TestSubscriptionPolicyMixin
# ===========================================================================


class TestSubscriptionPolicyMixin:
    """Tests for the SubscriptionPolicyMixin."""

    def _make_component(self, policy=SubscriptionPolicy.AUTO):
        """Create a mock component with the mixin."""

        class Component(SubscriptionPolicyMixin):
            def __init__(self):
                self._policy_manager = MagicMock(spec=SubscriptionPolicyManager)
                self._policy_manager.get_policy.return_value = policy
                self._subcomponents = {}

        return Component()

    def test_set_subscription_policy(self):
        comp = self._make_component()
        comp.set_subscription_policy(SubscriptionPolicy.MANUAL)
        comp._policy_manager.set_policy.assert_called_once_with(
            SubscriptionPolicy.MANUAL
        )

    def test_pause(self):
        comp = self._make_component()
        comp.pause()
        comp._policy_manager.pause.assert_called_once()

    def test_resume(self):
        comp = self._make_component()
        comp.resume()
        comp._policy_manager.resume.assert_called_once()

    def test_is_paused(self):
        comp = self._make_component()
        comp._policy_manager.is_paused.return_value = True
        assert comp.is_paused() is True

    def test_set_idle_timeout(self):
        comp = self._make_component()
        comp.set_idle_timeout(10.0)
        comp._policy_manager.set_idle_timeout.assert_called_once_with(10.0)

    def test_get_subscription_policy(self):
        comp = self._make_component()
        result = comp.get_subscription_policy()
        assert result == SubscriptionPolicy.AUTO

    def test_get_subscription_policy_none_when_no_manager(self):
        comp = self._make_component()
        comp._policy_manager = None
        assert comp.get_subscription_policy() is None

    def test_subcomponents_returns_copy(self):
        comp = self._make_component()
        child = MagicMock()
        comp._subcomponents = {"child": child}
        result = comp.subcomponents
        assert result == {"child": child}
        # Verify it's a copy
        result["new"] = "value"
        assert "new" not in comp._subcomponents

    def test_recursive_set_policy(self):
        comp = self._make_component()
        child = MagicMock()
        child.set_subscription_policy = MagicMock()
        comp._subcomponents = {"child": child}

        comp.set_subscription_policy(SubscriptionPolicy.MANUAL, recursive=True)
        comp._policy_manager.set_policy.assert_called_once_with(
            SubscriptionPolicy.MANUAL
        )
        child.set_subscription_policy.assert_called_once_with(
            SubscriptionPolicy.MANUAL, recursive=True
        )

    def test_recursive_set_policy_bare_manager(self):
        """If subcomponent has set_policy but not set_subscription_policy, use set_policy."""
        comp = self._make_component()
        child = MagicMock(spec=[])  # No methods by default
        child.set_policy = MagicMock()
        comp._subcomponents = {"child": child}

        comp.set_subscription_policy(SubscriptionPolicy.MANUAL, recursive=True)
        child.set_policy.assert_called_once_with(SubscriptionPolicy.MANUAL)

    def test_recursive_pause(self):
        comp = self._make_component()
        child = MagicMock()
        child.pause = MagicMock()
        comp._subcomponents = {"child": child}

        comp.pause(recursive=True)
        comp._policy_manager.pause.assert_called_once()
        child.pause.assert_called_once()

    def test_recursive_resume(self):
        comp = self._make_component()
        child = MagicMock()
        child.resume = MagicMock()
        comp._subcomponents = {"child": child}

        comp.resume(recursive=True)
        comp._policy_manager.resume.assert_called_once()
        child.resume.assert_called_once()


# ===========================================================================
# TestIdleMonitor
# ===========================================================================


class TestIdleMonitor:
    """Tests for the IdleMonitor."""

    def test_start_stop_lifecycle(self):
        monitor = IdleMonitor(check_interval=0.05)
        assert not monitor.is_running()
        monitor.start()
        assert monitor.is_running()
        monitor.stop()
        assert not monitor.is_running()

    def test_register_unregister(self):
        monitor = IdleMonitor()
        mgr = MagicMock(spec=SubscriptionPolicyManager)
        monitor.register(mgr)
        assert mgr in monitor._managers
        monitor.unregister(mgr)
        assert mgr not in monitor._managers

    def test_monitor_calls_check_idle(self):
        """Registered manager with old access time gets check_idle called."""
        sub = _make_subscriber(paused=False)
        mgr = SubscriptionPolicyManager(sub, "test", idle_timeout=0.05)
        mgr._last_access_time = time.monotonic() - 1.0

        monitor = IdleMonitor(check_interval=0.05)
        monitor.register(mgr)
        monitor.start()
        time.sleep(0.2)
        monitor.stop()

        sub.pause.assert_called()

    def test_set_global_idle_timeout(self):
        monitor = IdleMonitor()
        mgr1 = MagicMock(spec=SubscriptionPolicyManager)
        mgr2 = MagicMock(spec=SubscriptionPolicyManager)
        monitor.register(mgr1)
        monitor.register(mgr2)
        monitor.set_global_idle_timeout(15.0)
        mgr1.set_idle_timeout.assert_called_once_with(15.0)
        mgr2.set_idle_timeout.assert_called_once_with(15.0)

    def test_daemon_thread(self):
        """Monitor thread should be a daemon thread."""
        monitor = IdleMonitor(check_interval=0.05)
        monitor.start()
        assert monitor._thread.daemon is True
        monitor.stop()

# Copyright (C) 2025 Dexmate Inc.
#
# This software is dual-licensed:
#
# 1. GNU Affero General Public License v3.0 (AGPL-3.0)
#    See LICENSE-AGPL for details
#
# 2. Commercial License
#    For commercial licensing terms, contact: contact@dexmate.ai

"""Subscriber lifecycle management with policy-based idle control.

Provides four components for managing dexcomm subscriber lifecycles:

- ``SubscriptionPolicy`` — enum defining lifecycle strategies
- ``SubscriptionPolicyManager`` — wraps a subscriber with policy logic
- ``SubscriptionPolicyMixin`` — mixin for robot components
- ``IdleMonitor`` — background daemon that auto-idles unused subscribers
"""

from __future__ import annotations

import threading
import time
from enum import Enum
from typing import TYPE_CHECKING, Any

from loguru import logger

if TYPE_CHECKING:
    from dexcomm import Subscriber as DexcommSubscriber

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_IDLE_TIMEOUT: float = 5.0
"""Seconds of inactivity before AUTO subscribers are paused."""

DEFAULT_RESUME_TIMEOUT: float = 3.0
"""Max seconds to wait for fresh data after resuming an AUTO subscriber."""

_POLL_INTERVAL: float = 0.01
"""Sleep interval (seconds) when polling for fresh data after resume."""


# ===========================================================================
# SubscriptionPolicy
# ===========================================================================


class SubscriptionPolicy(str, Enum):
    """Lifecycle policy for a dexcomm subscriber.

    Values:
        ALWAYS_ON:  Never auto-idle the subscriber.
        AUTO:       Auto-idle after timeout, auto-resume on data access.
        MANUAL:     User controls via pause()/resume().
        ALWAYS_OFF: Always paused; data access returns None.
    """

    ALWAYS_ON = "always_on"
    AUTO = "auto"
    MANUAL = "manual"
    ALWAYS_OFF = "always_off"

    def __str__(self) -> str:  # noqa: D105
        return self.value


# ===========================================================================
# SubscriptionPolicyManager
# ===========================================================================


class SubscriptionPolicyManager:
    """Wraps a dexcomm Subscriber with policy-aware lifecycle management.

    Parameters:
        subscriber: A dexcomm ``Subscriber`` instance (or compatible duck-type)
            that exposes ``get_latest()``, ``pause()``, ``resume()``, and
            ``is_paused()`` methods.
        name: Human-readable name for logging.
        default_policy: Initial lifecycle policy.
        idle_timeout: Seconds of inactivity before AUTO subscribers are paused.
        resume_timeout: Max seconds to block waiting for fresh data on
            AUTO-resume.
    """

    def __init__(
        self,
        subscriber: DexcommSubscriber,
        name: str,
        default_policy: SubscriptionPolicy = SubscriptionPolicy.AUTO,
        idle_timeout: float = DEFAULT_IDLE_TIMEOUT,
        resume_timeout: float = DEFAULT_RESUME_TIMEOUT,
    ) -> None:
        self._subscriber: DexcommSubscriber = subscriber
        self._name = name
        self._policy = default_policy
        self._idle_timeout = idle_timeout
        self._resume_timeout = resume_timeout
        self._last_access_time = time.monotonic()
        self._lock = threading.Lock()
        # Python-level flag mirrors subscriber.is_paused() to avoid FFI on hot path
        self._idle_paused: bool = subscriber.is_paused()

        # Apply initial policy side-effects
        if default_policy is SubscriptionPolicy.ALWAYS_OFF:
            self._subscriber.pause()
            self._idle_paused = True

    # -- Policy control -----------------------------------------------------

    def touch(self) -> None:
        """Reset the idle timer without reading data.

        Call this when the component is being actively used (e.g. publishing
        control commands) to prevent the subscriber from being auto-paused.
        """
        self._last_access_time = time.monotonic()

    def get_policy(self) -> SubscriptionPolicy:
        """Return the current subscription policy."""
        return self._policy

    def set_policy(self, policy: SubscriptionPolicy) -> None:
        """Change the subscription policy.

        Side-effects:
            - ALWAYS_OFF → immediately pauses the subscriber.
            - ALWAYS_ON  → immediately resumes a paused subscriber.
        """
        with self._lock:
            self._policy = policy
            if policy is SubscriptionPolicy.ALWAYS_OFF:
                self._subscriber.pause()
                self._idle_paused = True
            elif policy is SubscriptionPolicy.ALWAYS_ON:
                if self._subscriber.is_paused():
                    self._subscriber.resume()
                self._idle_paused = False
            else:
                # AUTO / MANUAL — sync flag with actual subscriber state
                self._idle_paused = self._subscriber.is_paused()

    # -- Pause / resume -----------------------------------------------------

    def pause(self) -> None:
        """Pause the subscriber.  No-op for ALWAYS_ON policy."""
        with self._lock:
            if self._policy is SubscriptionPolicy.ALWAYS_ON:
                return
            self._subscriber.pause()
            self._idle_paused = True

    def resume(self) -> None:
        """Resume the subscriber.  No-op for ALWAYS_OFF policy."""
        with self._lock:
            if self._policy is SubscriptionPolicy.ALWAYS_OFF:
                return
            self._subscriber.resume()
            self._idle_paused = False
            self._last_access_time = time.monotonic()

    def is_paused(self) -> bool:
        """Check if the subscriber is paused (Python-level flag, no FFI)."""
        return self._idle_paused

    # -- Idle timeout -------------------------------------------------------

    def set_idle_timeout(self, seconds: float) -> None:
        """Update the idle timeout (seconds)."""
        self._idle_timeout = seconds

    def check_idle(self) -> None:
        """Check whether the subscriber should be idled.

        Called periodically by :class:`IdleMonitor`.  Only acts on AUTO
        policy subscribers that are currently active.

        Uses a Python-level ``_idle_paused`` flag for a lock-free fast path
        that avoids FFI calls (``is_paused()``) and lock acquisition for
        already-idle subscribers.  This prevents GIL contention from the
        IdleMonitor's periodic sweep disrupting the main thread (e.g.
        matplotlib's TkAgg event loop).
        """
        # Fast path: pure Python attribute reads — no FFI, no lock
        if self._policy is not SubscriptionPolicy.AUTO:
            return
        if self._idle_paused:
            return
        # Timestamp check without lock — _last_access_time is written
        # atomically under CPython's GIL, so a stale read at worst causes
        # one extra sweep before pausing (harmless).  This skips actively-
        # used AND recently-created subscribers with zero GIL contention.
        if time.monotonic() - self._last_access_time < self._idle_timeout:
            return

        # Slow path: only reached when subscriber is likely idle
        with self._lock:
            if self._subscriber.is_paused():
                self._idle_paused = True
                return
            elapsed = time.monotonic() - self._last_access_time
            if elapsed > self._idle_timeout:
                logger.debug(
                    "Auto-idling subscriber '{}' after {:.1f}s inactivity",
                    self._name,
                    elapsed,
                )
                self._subscriber.pause()
                self._idle_paused = True

    # -- Data access --------------------------------------------------------

    def get_latest_managed(self) -> Any | None:
        """Policy-aware data access.

        The hot path (AUTO, not paused) is lock-free to avoid contention
        with the IdleMonitor background thread, which would otherwise cause
        periodic stutter in real-time consumers like matplotlib animations.

        Single-attribute reads/writes are atomic under CPython's GIL, and the
        existing is_paused() check was already outside the lock, so removing
        locks from the hot path doesn't weaken any thread-safety guarantees.

        Returns:
            The latest message from the subscriber, or ``None`` when the
            policy dictates no data should be returned.
        """
        policy = self._policy

        # ALWAYS_OFF — never return data
        if policy is SubscriptionPolicy.ALWAYS_OFF:
            return None

        # ALWAYS_ON — unconditional passthrough
        if policy is SubscriptionPolicy.ALWAYS_ON:
            return self._subscriber.get_latest()

        # MANUAL — passthrough when active, None when paused
        if policy is SubscriptionPolicy.MANUAL:
            if self._idle_paused:
                return None
            return self._subscriber.get_latest()

        # AUTO — passthrough when active, auto-resume when paused
        # Use Python-level _idle_paused flag instead of FFI is_paused() to
        # avoid contending with the receiver thread's rust_subscriber mutex
        if self._idle_paused:
            return self._auto_resume()

        self._last_access_time = time.monotonic()
        return self._subscriber.get_latest()

    def is_active(self, window: float = 1.0) -> bool:
        """Policy-aware activity check.

        Mirrors :meth:`get_latest_managed`'s policy handling so an AUTO
        subscriber that was idle-paused gets resumed rather than misreported
        as inactive. Without this, ``is_active()`` / ``wait_for_active()``
        would query the raw paused subscriber and report a perfectly healthy
        topic as dead.

        Args:
            window: Time window (seconds) the underlying subscriber uses to
                decide activity.

        Returns:
            True if the component is (or should be) receiving updates under
            the current policy, False otherwise.
        """
        policy = self._policy

        # ALWAYS_OFF — intentionally inactive
        if policy is SubscriptionPolicy.ALWAYS_OFF:
            return False

        # MANUAL — paused means intentionally inactive
        if policy is SubscriptionPolicy.MANUAL and self._idle_paused:
            return False

        # AUTO — resume an idle-paused subscriber so health is reported truthfully
        if policy is SubscriptionPolicy.AUTO and self._idle_paused:
            self._auto_resume()
        else:
            self._last_access_time = time.monotonic()

        return self._subscriber.is_active(window)

    # -- Internal helpers ---------------------------------------------------

    def _auto_resume(self) -> Any | None:
        """Resume an AUTO-paused subscriber and poll for fresh data."""
        with self._lock:
            self._subscriber.resume()
            self._idle_paused = False
            self._last_access_time = time.monotonic()
        logger.warning(
            "Subscriber '{}' was idle-paused; resuming — expect a brief data gap.",
            self._name,
        )

        # Polling loop intentionally outside the lock — it's long-running
        deadline = time.monotonic() + self._resume_timeout
        while time.monotonic() < deadline:
            msg = self._subscriber.get_latest()
            if msg is not None:
                return msg
            time.sleep(_POLL_INTERVAL)

        logger.warning(
            "Subscriber '{}' did not deliver data within {:.1f}s after resume.",
            self._name,
            self._resume_timeout,
        )
        return None


# ===========================================================================
# SubscriptionPolicyMixin
# ===========================================================================


class SubscriptionPolicyMixin:
    """Mixin that adds subscription policy controls to a component.

    Assumes the host class defines:
        ``_policy_manager``:  A :class:`SubscriptionPolicyManager` (or ``None``).
        ``_subcomponents``:   A ``dict[str, Any]`` of child components.
    """

    _policy_manager: SubscriptionPolicyManager | None
    _subcomponents: dict[str, Any]

    # -- Own policy ---------------------------------------------------------

    def set_subscription_policy(
        self, policy: SubscriptionPolicy | str, recursive: bool = False
    ) -> None:
        """Set the subscription policy, optionally propagating to children."""
        if isinstance(policy, str):
            policy = SubscriptionPolicy(policy)

        if self._policy_manager is not None:
            self._policy_manager.set_policy(policy)

        if recursive:
            for child in self._subcomponents.values():
                if hasattr(child, "set_subscription_policy"):
                    child.set_subscription_policy(policy, recursive=True)
                elif hasattr(child, "set_policy"):
                    child.set_policy(policy)

    def pause(self, recursive: bool = False) -> None:
        """Pause this component's subscriber."""
        if self._policy_manager is not None:
            self._policy_manager.pause()

        if recursive:
            for child in self._subcomponents.values():
                if hasattr(child, "pause"):
                    child.pause()

    def resume(self, recursive: bool = False) -> None:
        """Resume this component's subscriber."""
        if self._policy_manager is not None:
            self._policy_manager.resume()

        if recursive:
            for child in self._subcomponents.values():
                if hasattr(child, "resume"):
                    child.resume()

    def is_paused(self) -> bool:
        """Return whether this component's subscriber is paused."""
        if self._policy_manager is None:
            return False
        return self._policy_manager.is_paused()

    def set_idle_timeout(self, seconds: float) -> None:
        """Update the idle timeout on the policy manager."""
        if self._policy_manager is not None:
            self._policy_manager.set_idle_timeout(seconds)

    def get_subscription_policy(self) -> SubscriptionPolicy | None:
        """Return the current policy, or None if no manager is set."""
        if self._policy_manager is None:
            return None
        return self._policy_manager.get_policy()

    @property
    def subcomponents(self) -> dict[str, Any]:
        """Return a shallow copy of the subcomponents dict."""
        return dict(self._subcomponents)


# ===========================================================================
# IdleMonitor
# ===========================================================================


class IdleMonitor:
    """Background daemon that periodically checks for idle subscribers.

    Parameters:
        check_interval: Seconds between idle-check sweeps.
    """

    def __init__(self, check_interval: float = 1.0) -> None:
        self._check_interval = check_interval
        self._managers: list[SubscriptionPolicyManager] = []
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None

    def register(self, manager: SubscriptionPolicyManager) -> None:
        """Register a manager for periodic idle checks."""
        with self._lock:
            if manager not in self._managers:
                self._managers.append(manager)

    def unregister(self, manager: SubscriptionPolicyManager) -> None:
        """Remove a manager from periodic idle checks."""
        with self._lock:
            try:
                self._managers.remove(manager)
            except ValueError:
                pass

    def set_global_idle_timeout(self, seconds: float) -> None:
        """Update the idle timeout on all registered managers."""
        with self._lock:
            for mgr in self._managers:
                mgr.set_idle_timeout(seconds)

    def start(self) -> None:
        """Start the idle-monitor background thread."""
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        """Stop the idle-monitor background thread."""
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=5.0)
            self._thread = None

    def is_running(self) -> bool:
        """Return whether the monitor thread is alive."""
        return self._thread is not None and self._thread.is_alive()

    def _run(self) -> None:
        """Main loop — check idle on all registered managers."""
        while not self._stop_event.is_set():
            with self._lock:
                managers = list(self._managers)
            for mgr in managers:
                try:
                    mgr.check_idle()
                except Exception:
                    logger.opt(exception=True).warning("Error during idle check")
            self._stop_event.wait(self._check_interval)

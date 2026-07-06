# Copyright (C) 2025 Dexmate Inc.
#
# This software is dual-licensed:
#
# 1. GNU Affero General Public License v3.0 (AGPL-3.0)
#    See LICENSE-AGPL for details
#
# 2. Commercial License
#    For commercial licensing terms, contact: contact@dexmate.ai

"""Shared DexComm Node singleton for dexcontrol.

All dexcontrol components use a single shared Node instead of creating
their own. This mirrors dexcomm's global singleton session pattern.
"""

import threading

from dexcomm import Node

_shared_node: Node | None = None
# Number of live top-level owners (each Robot / RobotQueryInterface). The Node
# is only torn down once the last owner releases it, so a standalone interface
# closing cannot destroy the Node still in use by another live instance.
_owner_count: int = 0
# Reentrant so a holder can call get_shared_node() while already holding the lock.
_lock = threading.RLock()


def get_shared_node() -> Node:
    """Get or create the shared dexcontrol Node.

    The Node namespace is auto-resolved from the ROBOT_NAME env var
    (standard dexcomm behavior when namespace is empty).

    This does NOT register ownership — components/sensors call this to obtain
    the shared Node, but only top-level owners (via :func:`acquire_shared_node`)
    govern its lifetime.

    Returns:
        The shared Node instance.
    """
    global _shared_node
    with _lock:
        if _shared_node is None:
            _shared_node = Node(name="dexcontrol")
        return _shared_node


def acquire_shared_node() -> Node:
    """Register a top-level owner and return the shared Node.

    Called once per ``Robot`` / ``RobotQueryInterface`` instance. Each acquire
    must be balanced by exactly one :func:`shutdown_shared_node` call.

    Returns:
        The shared Node instance.
    """
    global _owner_count
    with _lock:
        node = get_shared_node()
        _owner_count += 1
        return node


def shutdown_shared_node() -> None:
    """Release one top-level owner's hold on the shared Node.

    The underlying Node is shut down and the singleton reset only when the last
    owner releases it. This prevents one instance's teardown from destroying a
    Node that other live components in the same process still depend on.
    """
    global _shared_node, _owner_count
    with _lock:
        if _shared_node is None:
            return
        _owner_count -= 1
        if _owner_count <= 0:
            _owner_count = 0
            _shared_node.shutdown()
            _shared_node = None

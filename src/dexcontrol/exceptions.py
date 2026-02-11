"""Custom exceptions for dexcontrol.

This module defines a hierarchy of exceptions for better error handling
and user experience when connection or configuration issues occur.
"""

from __future__ import annotations


class DexcontrolError(Exception):
    """Base exception for all dexcontrol errors.

    All custom dexcontrol exceptions inherit from this class,
    making it easy to catch all dexcontrol-specific errors.
    """

    pass


class ConfigurationError(DexcontrolError):
    """Raised when there is a configuration problem.

    This includes missing or invalid environment variables,
    missing config files, or invalid configuration content.
    """

    pass


class RobotConnectionError(DexcontrolError):
    """Raised when the robot cannot be reached.

    This typically indicates network issues, robot not powered on,
    or Zenoh routing problems.
    """

    pass


class ServiceUnavailableError(DexcontrolError):
    """Raised when a specific service is not responding.

    This indicates the robot is likely connected but a specific
    service (e.g., hand type query) is not available. This could
    happen during robot initialization or if a component is disabled.
    """

    pass

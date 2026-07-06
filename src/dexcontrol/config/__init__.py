import warnings

from dexcontrol.core.config import get_robot_config


def get_robot_cfg(*args, **kwargs):
    """Deprecated: Use `get_robot_config` instead. This function will be removed in a future release."""
    warnings.warn(
        "get_robot_cfg is deprecated and will be removed in dexcontrol>=0.6.0. "
        "Please use get_robot_config instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return get_robot_config(*args, **kwargs)


def get_vega_config(*args, **kwargs):
    """Deprecated: Use `get_robot_config` instead. This function will be removed in a future release."""
    warnings.warn(
        "get_vega_config is deprecated and will be removed in dexcontrol>=0.6.0. "
        "Please use get_robot_config instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return get_robot_config(*args, **kwargs)

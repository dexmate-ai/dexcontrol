from dexcontrol.core.config import get_robot_config


def get_robot_cfg(*args, **kwargs):
    """Deprecated: Use `get_robot_config` instead. This function will be removed in a future release."""
    import loguru

    loguru.logger.warning(
        "get_robot_cfg is deprecated and will be removed in a future release. "
        "Please use get_robot_config instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return get_robot_config(*args, **kwargs)


get_vega_config = get_robot_config

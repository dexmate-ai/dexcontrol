"""Robot Environment Factory.

Factory for creating RobotEnv (local) or RobotEnvClient (gRPC service).
Enables gradual migration and testing of the service architecture.
"""

from typing import Dict, Any, Optional
import os


def create_robot_env(
    use_service: bool = None,
    robot_ip: str = "localhost",
    robot_port: int = 50051,
    **kwargs
):
    """Create RobotEnv or RobotEnvClient based on configuration.

    Args:
        use_service: If True, use RobotEnvClient (gRPC service).
                    If False, use original RobotEnv.
                    If None, read from env var ROBOTENV_USE_SERVICE (default: True)
        robot_ip: RobotEnv service host (for service mode)
        robot_port: In local mode: polymetis port. In service mode: gRPC service port.
        **kwargs: Additional arguments passed to RobotEnv/RobotEnvClient

    Returns:
        RobotEnv or RobotEnvClient instance

    Environment Variables:
        ROBOTENV_USE_SERVICE: Default "1" (service mode). Set to "0", "false", or "no" to use local mode
    """
    # Determine whether to use service (default: True / service mode)
    if use_service is None:
        use_service_env = os.environ.get('ROBOTENV_USE_SERVICE', '1').lower()
        use_service = use_service_env not in ('0', 'false', 'no')

    if use_service:
        # Use RobotEnvClient (gRPC service)
        from dexcontrol.core.robot_env_client import RobotEnvClient

        print(f"[RobotEnvFactory] Using RobotEnvClient (service mode) at {robot_ip}:{robot_port}")

        # Filter kwargs - RobotEnvClient needs reset_joints but not camera/recorder args
        client_kwargs = {
            'robot_ip': robot_ip,
            'robot_port': robot_port,
            'action_space': kwargs.get('action_space', 'cartesian_velocity'),
            'gripper_action_space': kwargs.get('gripper_action_space'),
            'reset_joints': kwargs.get('reset_joints'),
            'randomize': kwargs.get('randomize', False),
        }

        return RobotEnvClient(**client_kwargs)

    else:
        # Use original VegaRobot (local mode for Vega)
        # Note: For custom_dexcontrol, local mode would use VegaRobot directly
        # This path is rarely used in service architecture
        try:
            from dexcontrol.core.vega.robot import VegaRobot
            print(f"[RobotEnvFactory] Using VegaRobot (local mode)")
            # VegaRobot has different initialization than RobotEnv
            # Adapt kwargs as needed
            robot_kwargs = {
                'robot_model': kwargs.get('robot_model', 'vega_1'),
                'arm_side': kwargs.get('arm_side', 'left'),
                'control_hz': kwargs.get('control_hz', 20),
                'gripper_type': kwargs.get('gripper_type', 'default'),
            }
            robot = VegaRobot(**robot_kwargs)
            robot.launch_robot()
            return robot
        except ImportError:
            raise RuntimeError(
                "Local mode (RobotEnv) not available for custom_dexcontrol. "
                "Use service mode (ROBOTENV_USE_SERVICE=1) instead."
            )


def should_use_service_mode() -> bool:
    """Check if service mode should be used based on environment.

    Returns:
        True unless ROBOTENV_USE_SERVICE is explicitly set to "0", "false", or "no"
    """
    use_service_env = os.environ.get('ROBOTENV_USE_SERVICE', '1').lower()
    return use_service_env not in ('0', 'false', 'no')


def get_service_endpoints(config: Dict[str, Any]) -> Dict[str, tuple]:
    """Extract RobotEnv service endpoints from config.

    Args:
        config: Configuration dict with robot_port_1, robot_port_2, etc.

    Returns:
        Dict mapping robot names to (host, port) tuples

    Example:
        >>> config = {'robot_port_1': 50051, 'robot_port_2': 50053}
        >>> get_service_endpoints(config)
        {'robot0': ('localhost', 50051), 'robot1': ('localhost', 50053)}
    """
    endpoints = {}

    # Get robot IP (default: localhost for service mode)
    robot_ip = config.get('robot_ip', 'localhost')

    # Robot 1
    if 'robot_port_1' in config:
        endpoints['robot0'] = (robot_ip, int(config['robot_port_1']))

    # Robot 2
    if 'robot_port_2' in config:
        endpoints['robot1'] = (robot_ip, int(config['robot_port_2']))

    return endpoints

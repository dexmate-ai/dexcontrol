# Changelog

All notable changes to this project will be documented in this file.

## [0.4.1] - 2026-02-10

### Performance Improvements

- **Significantly reduced GIL contention for real-time control loops.** Robot component state updates now happen entirely in Rust without acquiring the Python GIL. Previously, every incoming state message triggered a Python callback that acquired the GIL — this caused latency spikes when running CPU-intensive workloads like neural network inference alongside control loops. With this change, background threads store raw bytes in Rust, and state is only decoded when you read it (with smart caching: <1μs for repeated reads, ~10μs for new data).
- **Heartbeat monitoring is now GIL-free.** The heartbeat safety monitor has been moved from a Python background thread to a Rust-backed `HeartbeatMonitor`. Heartbeat subscription, decoding, and timeout detection all run without GIL involvement, so heartbeat monitoring no longer interferes with Python workloads. The GIL is only briefly acquired if a timeout actually fires (to log the critical error before exiting).

### Added

- **Clearer error messages when things go wrong.** Introduced a hierarchy of specific exceptions to replace generic `RuntimeError`:
  - `ConfigurationError` — raised when `ZENOH_CONFIG` is not set, the config file is missing, or permissions are wrong. The error message tells you exactly what to fix.
  - `RobotConnectionError` — raised when the robot cannot be reached (not powered on, network issues, Zenoh routing problems).
  - `ServiceUnavailableError` — raised when a specific robot service (e.g., hand type query, version info) is not responding. This can happen during robot initialization or if a component is disabled.
  - All exceptions inherit from `DexcontrolError`, so you can catch all dexcontrol errors with a single `except DexcontrolError`.

### Fixed

- **End effector baud rate configuration now works.** The `set_ee_baud_rate` feature was broken because the service client used a hardcoded endpoint name that didn't match the robot's actual service. It now uses the correct service name from the arm configuration.

### Dependencies

- Requires `dexcomm >= 0.4.1` (for Rust-side subscriber storage and `HeartbeatMonitor`).
- Requires `dexbot-utils >= 0.4.1`.

## [0.4.0] - 2026-02-02

### Added
- Unified support for Vega-1U, Vega-1, and Vega-1P robot variants.
- Dex-gripper end effector support.
- Robot arm PID tuning (currently P gain only).
- Arm brake release when joint limits are exceeded.
- Extracted configuration management to `dexbot-utils` package.
- Custom end effector feedback topic for receiving data frames from unrecognized end effectors.
- USB camera support with dedicated `USBCamera` sensor class and configuration.
- Enhanced camera base class with improved RTC and dexcomm integration.
- New robot info helper utilities for better system information management.

### Changed
- **Major dexcomm Migration**: Migrated to dexcomm (>= 0.4.0) with Rust backend for improved performance and reliability.
- Refactored camera architecture with unified base camera implementation supporting both local and RTC modes.
- Simplified subscriber infrastructure by removing redundant subscriber wrapper code.
- Updated all sensor configurations for seamless compatibility with latest dexcomm API.
- Improved ZED camera implementation with better depth handling and timestamp support.
- Improve the error code message parsing for all the components. Now dexcontrol will no longer get any error code, it will directly get the error message for each joint.

### Fixed
- Sensor initialization issues after dexcomm migration.
- Camera streaming bugs with latest dexcomm integration.
- ZED head camera configuration and initialization.
- RTC camera configuration handling.
- IMU, LiDAR, and ultrasonic sensor compatibility with new dexcomm version.
- Error handling and reliability across all sensors.

### Dependencies
- **Breaking**: Requires `dexcomm >= 0.4.0` (migrated from `>= 0.1.18`).
- **Breaking**: Requires `dexbot-utils >= 0.4.0`.
- **Breaking**: Requires firmware >= 0.4.0 on the SOC.
- Removed optional dependencies: `aiortc`, `websockets` (now handled by dexcomm).

### Version Requirements
- **SOC Minimal Version**: 360

## [0.3.3] - 2025-10-09

### Added
- WebRTC camera streaming via `RTCSubscriber` with DexComm-compatible API; added `create_rtc_camera_subscriber`.
- ZED and RGB camera sensors now support RTC mode and DexComm factories; depth stream uses DexComm depth deserializer.
- ZED `get_obs` supports `include_timestamp` to passthrough timestamps when provided by DexComm.
- Core DexComm subscriber utilities: `create_subscriber`, `create_buffered_subscriber`, `create_camera_subscriber`, `create_depth_subscriber`, `create_imu_subscriber`, `create_lidar_subscriber`, `create_generic_subscriber`.
- Ultrasonic sensor now uses typed protobuf deserialization and returns standardized `(4,)` distance array in order `[front_left, front_right, back_left, back_right]`.

### Changed
- Consolidated sensor subscriptions onto DexComm-based factories with consistent topic resolution and config handling.
- Chassis IMU mappings standardized; `timestamp_ns` returned when available for clearer units.

### Dependencies
- Requires `dexcomm >= 0.1.15`.
- Optional WebRTC: `aiortc`, `websockets`; performance: `uvloop` (Unix).

### Version Requirements
- **SOC Minimal Version**: 298

## [0.3.2] - 2025-09-27

### Fixed
- **Arm Mode Switch Query**: Resolved timeout failures in arm mode switch queries, improving reliability during mode transitions.

### Version Requirements
- **SOC Minimal Version**: 298

## [0.3.1] - 2025-09-26

### Added
- **End Effector Control Example**: Added simple example demonstrating end effector control patterns for custom manipulator integration.

### Changed
- **DexComm Integration**: Migrated all Zenoh-based communication to dexcomm library for improved consistency and maintainability.
- **Environment Variable Standardization**: Changed Zenoh config file path environment variable from `DEXMATE_COMM_CFG_PATH` to `ZENOH_CONFIG` for consistency with dexcomm.

### Fixed
- **Hand Type Detection**: Fixed hand type query bug that incorrectly assumed unknown hand types when hands were connected.

### Version Requirements
- **SOC Minimal Version**: 298

## [0.3.0] - 2025-08-29

### Added
- Automatic hand type detection for component initialization. The robot will only have left_hand or right_hand attributes when hands are physically connected.
- Minimum version checking on the server side. Robot firmware now returns a minimum required client version to ensure dexcontrol compatibility.
- RS-485 pass-through support for end-effectors, enabling dexcontrol to send raw bytes directly to unknown end effectors.
- Free-drive motion support with brake release capability for arm motors.
- Support for v2-hand touch sensor readings.
- Unified interface for all motors, providing position, velocity, current/torque, error codes, and timestamps.

### Changed
- Enhanced internal driver communication efficiency. The driver now handles cases where client control frequency (e.g., dexcontrol) exceeds motor limits, preventing communication overload.
- Deployed new internal communication protocol for improved consistency and efficiency.
- Unified arm and head motor enables logic. Both the e-stop button and the software e-stop now disable head motors to prevent head-torso collisions.
- Separated chassis_steer and chassis_wheel into independent modules for better modularity.
- Enhanced logging and debugging capabilities.

### Fixed
- Control Frequency Issue: Fixed bug where control frequency affected reading frequency.
- Display Bug: Resolved false positive issue in display_robot_info function.

### Breaking Changes
- **Protobuf Definition Changes**: Updated protobuf definitions require dexcontrol >= 0.3.0 for all firmware >= 0.3.0.

### Version Requirements
- **SOC Minimal Version**: 286

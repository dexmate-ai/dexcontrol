# Changelog

All notable changes to this project will be documented in this file.

## [0.3.3] - 2025-10-09

### Added
- WebRTC camera streaming via `RTCSubscriber` with DexComm-compatible API; added `create_rtc_camera_subscriber`.
- ZED and RGB camera sensors now support RTC mode and DexComm factories; depth stream uses DexComm depth deserializer.
- ZED `get_obs` supports `include_timestamp` to passthrough timestamps when provided by DexComm.
- Core DexComm subscriber utilities: `create_subscriber`, `create_buffered_subscriber`, `create_camera_subscriber`, `create_depth_subscriber`, `create_imu_subscriber`, `create_lidar_subscriber`, `create_generic_subscriber`, `quick_subscribe`, `wait_for_any_message`.
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

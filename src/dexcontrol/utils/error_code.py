# Copyright (C) 2025 Dexmate Inc.
#
# This software is dual-licensed:
#
# 1. GNU Affero General Public License v3.0 (AGPL-3.0)
#    See LICENSE-AGPL for details
#
# 2. Commercial License
#    For commercial licensing terms, contact: contact@dexmate.ai

"""Error code interpretation utilities for robot components."""


class ErrorCodeInterpreter:
    """Interprets error codes for different robot components."""

    # Arm error codes (left_arm and right_arm)
    ARM_ERROR_CODES = {
        0x00000020: "Overtemperature",
        0x00000080: "Encoder error",
        0x00000200: "Software error",
        0x00000400: "Temperature sensor error",
        0x00000800: "Position limit exceeded",
        0x00002000: "Position tracking error exceeded",
        0x00004000: "Current detection error",
        0x00008000: "Brake failure",
        0x00010000: "Position command limit exceeded",
    }

    # Head error codes
    HEAD_ERROR_CODES = {
        0x00: "Disabled",
        0x04: "Enabled",
        0x20: "Overvoltage",
        0x24: "Undervoltage",
    }

    # Chassis wheel error codes (common)
    CHASSIS_COMMON_CODES = {
        0x00000000: "No error",
        0x00000004: "Overvoltage",
        0x00000008: "Undervoltage",
    }

    # Left drive wheel motor specific
    CHASSIS_LEFT_DRIVE_CODES = {
        0x00000010: "Overcurrent",
        0x00000020: "Overload",
        0x00000080: "Encoder deviation",
        0x00000200: "Reference voltage error",
        0x00000800: "Hall sensor error",
        0x00001000: "Motor overtemperature",
        0x00002000: "Encoder error",
    }

    # Right drive wheel motor specific
    CHASSIS_RIGHT_DRIVE_CODES = {
        0x00100000: "Overcurrent",
        0x00200000: "Overload",
        0x00800000: "Encoder deviation",
        0x02000000: "Reference voltage error",
        0x08000000: "Hall sensor error",
        0x10000000: "Motor overtemperature",
        0x20000000: "Encoder error",
    }

    # Steering wheel error codes (left and right)
    CHASSIS_STEERING_CODES = {
        0xFC00000C: "Undervoltage",
        0xFC000010: "Overvoltage",
    }

    # Torso motor error codes
    TORSO_ERROR_CODES = {
        0x00000008: "Overvoltage",
        0x00000010: "Undervoltage",
        0x00000040: "Startup error",
        0x00000080: "Speed feedback error",
        0x00000100: "Overcurrent",
    }

    # BMS (Battery Management System) alarm codes
    BMS_ERROR_CODES = {
        0x0004: "Single cell overvoltage",
        0x0008: "Single cell undervoltage",
        0x0010: "Total overvoltage",
        0x0020: "Total undervoltage",
        0x0040: "High temperature",
        0x0080: "Low temperature",
        0x0100: "Discharge overcurrent",
        0x0200: "Charge overcurrent",
        # Bits 8-15 are reserved
    }

    @classmethod
    def interpret_error(cls, component: str, error_code: int) -> str:
        """
        Interpret error code for a specific component.

        Args:
            component: Component name (e.g., 'left_arm', 'right_arm', 'head', etc.)
            error_code: Error code to interpret

        Returns:
            Human-readable error description or hex code if unknown
        """
        component_lower = component.lower()

        # Arms (left and right)
        if "left_arm" in component_lower or "right_arm" in component_lower:
            return cls._interpret_bitmask_errors(error_code, cls.ARM_ERROR_CODES)

        # Hands (left and right) - same as arms
        elif "left_hand" in component_lower or "right_hand" in component_lower:
            return cls._interpret_bitmask_errors(error_code, cls.ARM_ERROR_CODES)

        # Head
        elif "head" in component_lower:
            return cls.HEAD_ERROR_CODES.get(
                error_code, f"Unknown error: 0x{error_code:02X}"
            )

        # Chassis (includes all wheel errors)
        elif "chassis" in component_lower:
            return cls._interpret_chassis_errors(error_code)

        # Torso
        elif "torso" in component_lower:
            return cls._interpret_bitmask_errors(error_code, cls.TORSO_ERROR_CODES)

        # BMS (Battery Management System)
        elif "bms" in component_lower:
            return cls._interpret_bitmask_errors(error_code, cls.BMS_ERROR_CODES)

        else:
            return f"Unknown component error: 0x{error_code:08X}"

    @classmethod
    def _interpret_bitmask_errors(
        cls, error_code: int, error_dict: dict[int, str]
    ) -> str:
        """
        Interpret bitmask-style error codes.

        Args:
            error_code: Error code with multiple possible bits set
            error_dict: Dictionary mapping bit masks to error descriptions

        Returns:
            Comma-separated list of active errors or hex code if none found
        """
        if error_code == 0:
            return "No error"

        errors = []
        for mask, description in error_dict.items():
            if error_code & mask:
                errors.append(description)

        if errors:
            return ", ".join(errors)
        else:
            return f"Unknown error: 0x{error_code:08X}"

    @classmethod
    def _interpret_chassis_errors(cls, error_code: int) -> str:
        """
        Interpret chassis-specific error codes (includes all wheel errors).

        Args:
            error_code: Error code to interpret

        Returns:
            Human-readable error description
        """
        # Check steering wheel codes first (they have specific values)
        if error_code in cls.CHASSIS_STEERING_CODES:
            return cls.CHASSIS_STEERING_CODES[error_code]

        # Check common codes
        if error_code == 0:
            return "No error"

        errors = []

        # Check common chassis errors
        for mask, description in cls.CHASSIS_COMMON_CODES.items():
            if mask != 0 and (error_code & mask):
                errors.append(description)

        # Check left drive wheel specific
        for mask, description in cls.CHASSIS_LEFT_DRIVE_CODES.items():
            if error_code & mask:
                errors.append(f"Left drive: {description}")

        # Check right drive wheel specific
        for mask, description in cls.CHASSIS_RIGHT_DRIVE_CODES.items():
            if error_code & mask:
                errors.append(f"Right drive: {description}")

        if errors:
            return ", ".join(errors)
        else:
            return f"Unknown chassis error: 0x{error_code:08X}"


def get_error_description(component: str, error_code: int) -> str:
    """
    Convenience function to get error description.

    Args:
        component: Component name
        error_code: Error code to interpret

    Returns:
        Human-readable error description
    """
    return ErrorCodeInterpreter.interpret_error(component, error_code)


def get_multiple_errors(components_errors: dict[str, int]) -> dict[str, str]:
    """
    Get error descriptions for multiple components.

    Args:
        components_errors: Dictionary mapping component names to error codes

    Returns:
        Dictionary mapping component names to error descriptions
    """
    return {
        component: get_error_description(component, error_code)
        for component, error_code in components_errors.items()
    }

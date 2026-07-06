"""Read temperature data from all components.

This script reads temperature sensor data from arms, head, torso, chassis, and battery.
Temperature values are in °C.
"""

from loguru import logger

from dexcontrol import Robot


def main() -> None:
    with Robot() as robot:
        # Arm temperatures
        for arm_name in ["left_arm", "right_arm"]:
            arm = getattr(robot, arm_name, None)
            if arm is None:
                continue
            if arm.temperature_sensor and arm.temperature_sensor.is_active():
                temps = arm.temperature_sensor.get_temperatures()
                logger.info(f"{arm_name} temperatures: {temps}")
            else:
                logger.warning(f"{arm_name}: no temperature data")

        # Head temperatures (optional component)
        head = getattr(robot, "head", None)
        if head is not None:
            if head.temperature_sensor and head.temperature_sensor.is_active():
                temps = head.temperature_sensor.get_temperatures()
                logger.info(f"head temperatures: {temps}")
            else:
                logger.warning("head: no temperature data")

        # Torso temperatures (optional component)
        if hasattr(robot, "torso"):
            if (
                robot.torso.temperature_sensor
                and robot.torso.temperature_sensor.is_active()
            ):
                temps = robot.torso.temperature_sensor.get_temperatures()
                logger.info(f"torso temperatures: {temps}")
            else:
                logger.warning("torso: no temperature data")

        # Chassis temperatures (optional component, steer + drive)
        if hasattr(robot, "chassis"):
            for sensor_name, sensor in [
                ("chassis steer", robot.chassis.steer_temperature_sensor),
                ("chassis drive", robot.chassis.drive_temperature_sensor),
            ]:
                if sensor and sensor.is_active():
                    temps = sensor.get_temperatures()
                    logger.info(f"{sensor_name} temperatures: {temps}")
                else:
                    logger.warning(f"{sensor_name}: no temperature data")

        # Battery temperature
        if hasattr(robot, "battery") and robot.battery.is_active():
            battery_status = robot.battery.get_status()
            logger.info(f"battery temperature: {battery_status['temperature']}°C")
        else:
            logger.warning("battery: no temperature data")


if __name__ == "__main__":
    main()

"""Runtime settings object built from config files."""

from dataclasses import dataclass

from continuum_robot.config.schemas import RobotConfig, SafetyConfig, SerialConfig


@dataclass
class Settings:
    """Aggregate settings for app runtime."""

    robot: RobotConfig
    serial: SerialConfig
    safety: SafetyConfig

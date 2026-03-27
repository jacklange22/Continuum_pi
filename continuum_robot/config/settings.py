"""Runtime settings object built from config files."""

from dataclasses import dataclass

from continuum_robot.config.schemas import (
    CalibrationConfig,
    ExperimentConfig,
    RegistrationWorkflowConfig,
    RobotConfig,
    RuntimeConfig,
    SafetyConfig,
    SerialConfig,
)


@dataclass
class Settings:
    """Aggregate settings for app runtime."""

    runtime: RuntimeConfig
    robot: RobotConfig
    serial: SerialConfig
    safety: SafetyConfig
    registration: RegistrationWorkflowConfig
    experiment: ExperimentConfig
    calibration: CalibrationConfig

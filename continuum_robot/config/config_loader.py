"""Configuration loading helpers for YAML templates."""

from __future__ import annotations

from pathlib import Path
import yaml

from continuum_robot.config.schemas import RobotConfig, SafetyConfig, SerialConfig
from continuum_robot.config.settings import Settings


class ConfigLoader:
    """Loads YAML config files into typed settings."""

    def __init__(self, base_dir: Path | None = None) -> None:
        self.base_dir = base_dir or Path("/Users/jacklange/Continuum/pi_code/config")

    def load_settings(self) -> Settings:
        """Load core configs from template files.

        The default implementation is intentionally shallow for scaffold use.
        """
        robot_data = self._read_yaml(self.base_dir / "robot_4servo.yaml")
        serial_data = self._read_yaml(self.base_dir / "system.yaml")
        safety_data = self._read_yaml(self.base_dir / "safety.yaml")

        robot = RobotConfig(
            mode=robot_data.get("mode", "4-servo"),
            spool_diameter_cm=robot_data.get("spool_diameter_cm", 1.2),
            servo_ids=robot_data.get("servo_ids", [1, 2, 3, 4]),
            tendon_to_servo=robot_data.get("tendon_to_servo", [1, 2, 3, 4]),
        )
        serial = SerialConfig(
            aurora_port=serial_data.get("aurora_port", ""),
            openrb_port=serial_data.get("openrb_port", ""),
            baudrate=serial_data.get("baudrate", 115200),
        )
        safety = SafetyConfig(
            position_min_offset_ticks=safety_data.get("position_min_offset_ticks", -600),
            position_max_offset_ticks=safety_data.get("position_max_offset_ticks", 600),
            max_current_ma=safety_data.get("max_current_ma", 850),
        )
        return Settings(robot=robot, serial=serial, safety=safety)

    @staticmethod
    def _read_yaml(path: Path) -> dict:
        if not path.exists():
            return {}
        with path.open("r", encoding="utf-8") as handle:
            data = yaml.safe_load(handle) or {}
        if not isinstance(data, dict):
            raise ValueError(f"Expected mapping in {path}")
        return data

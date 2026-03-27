"""Configuration loading helpers for YAML templates."""

from __future__ import annotations

from pathlib import Path
import yaml

from continuum_robot.config.schemas import (
    CalibrationConfig,
    ExperimentConfig,
    RegistrationWorkflowConfig,
    RobotConfig,
    RuntimeConfig,
    SafetyConfig,
    SerialConfig,
)
from continuum_robot.config.settings import Settings


class ConfigLoader:
    """Loads YAML config files into typed settings."""

    def __init__(self, base_dir: Path | None = None) -> None:
        project_root = Path(__file__).resolve().parents[2]
        self.base_dir = base_dir or (project_root / "config")

    def load_settings(self) -> Settings:
        """Load core configs from template files."""
        system_data = self._merge_dicts(
            self._read_yaml(self.base_dir / "system.yaml"),
            self._read_yaml(self.base_dir / "system.local.yaml"),
        )
        robot_path = self.base_dir / str(system_data.get("robot_config", "robot_4servo.yaml"))
        robot_data = self._read_yaml(robot_path)
        safety_data = self._read_yaml(self.base_dir / "safety.yaml")
        registration_data = self._read_yaml(self.base_dir / "registration.yaml")
        experiment_data = self._read_yaml(self.base_dir / "experiment.yaml")

        runtime = RuntimeConfig(
            mock_mode=bool(system_data.get("mock_mode", True)),
            poll_rate_hz=int(system_data.get("poll_rate_hz", 5)),
            robot_config=robot_path.name,
        )
        robot = RobotConfig(
            mode=str(robot_data.get("mode", "4-servo")),
            spool_diameter_cm=float(robot_data.get("spool_diameter_cm", 1.2)),
            ticks_per_revolution=int(robot_data.get("ticks_per_revolution", 4096)),
            servo_ids=[int(v) for v in robot_data.get("servo_ids", [1, 2, 3, 4])],
            tendon_to_servo=[int(v) for v in robot_data.get("tendon_to_servo", [1, 2, 3, 4])],
        )
        serial = SerialConfig(
            aurora_port=str(system_data.get("aurora_port", "")),
            openrb_port=str(system_data.get("openrb_port", "")),
            tracker_backend=str(system_data.get("tracker_backend", "ndi")),
            tracker_type=str(system_data.get("tracker_type", "aurora")),
            baudrate=int(system_data.get("baudrate", 115200)),
            read_timeout_s=float(system_data.get("read_timeout_s", 0.05)),
            frame_timeout_s=float(system_data.get("frame_timeout_s", 0.5)),
            reconnect_delay_s=float(system_data.get("reconnect_delay_s", 1.0)),
            tracker_freshness_timeout_s=float(system_data.get("tracker_freshness_timeout_s", 0.5)),
            tracker_ports_to_probe=[str(v) for v in system_data.get("tracker_ports_to_probe", [])],
            tracker_settings_overrides=dict(system_data.get("tracker_settings_overrides", {}) or {}),
            tracker_socket_path=str(system_data.get("tracker_socket_path", "/tmp/tracker_bridge.sock")),
            tracker_bridge_executable=str(system_data.get("tracker_bridge_executable", "bin/tracker_bridge")),
            tracker_poll_ms=int(system_data.get("tracker_poll_ms", 20)),
            tracker_min_effective_fps=float(system_data.get("tracker_min_effective_fps", 20.0)),
            tracker_max_stale_interval_s=float(system_data.get("tracker_max_stale_interval_s", 0.25)),
            tracker_max_consecutive_missing_frames=int(
                system_data.get("tracker_max_consecutive_missing_frames", 20)
            ),
            tracker_require_valid_transforms=bool(system_data.get("tracker_require_valid_transforms", True)),
            packet_capture_dir=str(system_data.get("packet_capture_dir", "data/tracker_captures")),
        )
        safety = SafetyConfig(
            position_min_offset_ticks=int(safety_data.get("position_min_offset_ticks", -600)),
            position_max_offset_ticks=int(safety_data.get("position_max_offset_ticks", 600)),
            max_current_ma=int(safety_data.get("max_current_ma", 850)),
            pretension_current_balance_tolerance_ma=int(
                safety_data.get("pretension_current_balance_tolerance_ma", 120)
            ),
        )
        registration = RegistrationWorkflowConfig(
            landmark_labels=[str(v) for v in registration_data.get("landmark_labels", ["L1", "L2", "L3", "L4"])],
            captures_per_landmark=int(registration_data.get("captures_per_landmark", 5)),
            nominal_landmarks_robot_xyz_mm={
                str(k): [float(v) for v in values]
                for k, values in registration_data.get("nominal_landmarks_robot_xyz_mm", {}).items()
            },
            capture_tool_id=str(registration_data.get("capture_tool_id", "0B")),
            coil_tool_id=str(registration_data.get("coil_tool_id", "0A")),
            capture_tool_tip_transform=self._maybe_matrix(
                registration_data.get("capture_tool_tip_transform")
            ),
            model_points_file=self._maybe_path(registration_data.get("model_points_file", "tools/12_model_registration_points_in_sw")),
            tip_points_file=self._maybe_path(registration_data.get("tip_points_file", "tools/all_tip_registration_points_in_sw")),
            T_sw_2_model_file=self._maybe_path(registration_data.get("T_sw_2_model_file", "tools/T_sw_2_model")),
            T_sw_2_tip_file=self._maybe_path(registration_data.get("T_sw_2_tip_file", "tools/T_sw_2_tip")),
            penprobe_file=self._maybe_path(registration_data.get("penprobe_file", "tools/penprobe_08_09_24c")),
            quaternion_average_method=str(registration_data.get("quaternion_average_method", "sign_aligned_mean")),
            model_tre_reference_radius_mm=float(registration_data.get("model_tre_reference_radius_mm", 5.0)),
            tip_tre_reference_radius_mm=float(registration_data.get("tip_tre_reference_radius_mm", 3.0)),
            max_fre_mm=self._maybe_float(
                (registration_data.get("validation", {}) or {}).get("max_fre_mm", 2.0)
            ),
        )
        experiment = ExperimentConfig(
            default_settle_time_s=float(experiment_data.get("default_settle_time_s", 2.0)),
            sample_count_per_point=int(experiment_data.get("sample_count_per_point", 1)),
            output_dir=str(experiment_data.get("output_dir", "data/runs")),
        )
        calibration = CalibrationConfig(
            neutral_setpoints_path=str(
                system_data.get("neutral_setpoints_path", "data/calibrations/neutral_setpoints.json")
            ),
            latest_registration_path=str(
                system_data.get("latest_registration_path", "data/registrations/latest_registration.json")
            ),
        )
        return Settings(
            runtime=runtime,
            robot=robot,
            serial=serial,
            safety=safety,
            registration=registration,
            experiment=experiment,
            calibration=calibration,
        )

    @staticmethod
    def _maybe_float(value) -> float | None:
        if value in (None, ""):
            return None
        return float(value)

    @staticmethod
    def _maybe_matrix(value) -> list[list[float]] | None:
        if value in (None, ""):
            return None
        rows = [[float(item) for item in row] for row in value]
        if len(rows) != 4 or any(len(row) != 4 for row in rows):
            raise ValueError("capture_tool_tip_transform must be a 4x4 matrix when provided")
        return rows

    @staticmethod
    def _maybe_path(value) -> str | None:
        if value in (None, ""):
            return None
        return str(value)

    @staticmethod
    def _read_yaml(path: Path) -> dict:
        if not path.exists():
            return {}
        with path.open("r", encoding="utf-8") as handle:
            data = yaml.safe_load(handle) or {}
        if not isinstance(data, dict):
            raise ValueError(f"Expected mapping in {path}")
        return data

    @staticmethod
    def _merge_dicts(base: dict, override: dict) -> dict:
        merged = dict(base)
        merged.update(override)
        return merged

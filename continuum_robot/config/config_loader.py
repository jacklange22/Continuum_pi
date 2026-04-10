"""Configuration loading helpers for YAML templates."""

from __future__ import annotations

from pathlib import Path
import yaml

from continuum_robot.config.schemas import (
    CalibrationConfig,
    ExperimentConfig,
    RegistrationLandmarkConfig,
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
        robot_data = self._merge_dicts(
            self._read_yaml(robot_path),
            dict(system_data.get("robot_overrides", {}) or {}),
        )
        safety_data = self._merge_dicts(
            self._read_yaml(self.base_dir / "safety.yaml"),
            dict(system_data.get("safety_overrides", {}) or {}),
        )
        registration_data = self._read_yaml(self.base_dir / "registration.yaml")
        experiment_data = self._read_yaml(self.base_dir / "experiment.yaml")

        runtime = RuntimeConfig(
            mock_mode=bool(system_data.get("mock_mode", True)),
            poll_rate_hz=int(system_data.get("poll_rate_hz", 5)),
            robot_config=robot_path.name,
            visualization_mode=str(system_data.get("visualization_mode", "auto")),
            visualization_safe_effects=bool(system_data.get("visualization_safe_effects", True)),
        )
        robot = RobotConfig(
            mode=str(robot_data.get("mode", "4-servo")),
            spool_diameter_cm=float(robot_data.get("spool_diameter_cm", 1.2)),
            ticks_per_revolution=int(robot_data.get("ticks_per_revolution", 4096)),
            servo_ids=[int(v) for v in robot_data.get("servo_ids", [1, 2, 3, 4])],
            tendon_to_servo=[int(v) for v in robot_data.get("tendon_to_servo", [1, 2, 3, 4])],
            tightening_rotation_by_servo={
                int(key): str(value).strip().lower()
                for key, value in dict(robot_data.get("tightening_rotation_by_servo", {}) or {}).items()
                if str(value).strip()
            },
        )
        serial = SerialConfig(
            aurora_port=str(system_data.get("aurora_port", "")),
            openrb_port=str(system_data.get("openrb_port", "")),
            tracker_backend=str(system_data.get("tracker_backend", "ndi")),
            tracker_fallback_backend=self._maybe_path(system_data.get("tracker_fallback_backend", "bridge")),
            tracker_fallback_enabled=bool(system_data.get("tracker_fallback_enabled", True)),
            tracker_type=str(system_data.get("tracker_type", "aurora")),
            baudrate=int(system_data.get("baudrate", 115200)),
            read_timeout_s=float(system_data.get("read_timeout_s", 0.05)),
            frame_timeout_s=float(system_data.get("frame_timeout_s", 0.5)),
            reconnect_delay_s=float(system_data.get("reconnect_delay_s", 1.0)),
            tracker_freshness_timeout_s=float(system_data.get("tracker_freshness_timeout_s", 0.5)),
            tracker_ports_to_probe=[str(v) for v in system_data.get("tracker_ports_to_probe", [])],
            tracker_settings_overrides=dict(system_data.get("tracker_settings_overrides", {}) or {}),
            tracker_tool_id_aliases={
                str(key): str(value)
                for key, value in (system_data.get("tracker_tool_id_aliases", {}) or {}).items()
            },
            tracker_socket_path=str(system_data.get("tracker_socket_path", "/tmp/tracker_bridge.sock")),
            tracker_bridge_executable=str(system_data.get("tracker_bridge_executable", "bin/tracker_bridge")),
            tracker_poll_ms=int(system_data.get("tracker_poll_ms", 20)),
            tracker_min_effective_fps=float(system_data.get("tracker_min_effective_fps", 15.0)),
            tracker_max_stale_interval_s=float(system_data.get("tracker_max_stale_interval_s", 0.25)),
            tracker_max_consecutive_missing_frames=int(
                system_data.get("tracker_max_consecutive_missing_frames", 20)
            ),
            tracker_require_valid_transforms=bool(system_data.get("tracker_require_valid_transforms", True)),
            packet_capture_dir=str(system_data.get("packet_capture_dir", "data/tracker_captures")),
            openrb_settings=dict(system_data.get("openrb_settings", {}) or {}),
            dynamixel_settings=dict(system_data.get("dynamixel_settings", {}) or {}),
        )
        safety = SafetyConfig(
            position_min_offset_ticks=int(safety_data.get("position_min_offset_ticks", -600)),
            position_max_offset_ticks=int(safety_data.get("position_max_offset_ticks", 600)),
            max_current_ma=int(safety_data.get("max_current_ma", 850)),
            default_pretension_current_threshold_ma=int(
                safety_data.get("default_pretension_current_threshold_ma", 220)
            ),
            pretension_current_balance_tolerance_ma=int(
                safety_data.get("pretension_current_balance_tolerance_ma", 120)
            ),
            fine_jog_step_ticks=int(safety_data.get("fine_jog_step_ticks", 5)),
            coarse_jog_step_ticks=int(safety_data.get("coarse_jog_step_ticks", 25)),
            software_position_margin_ticks=int(safety_data.get("software_position_margin_ticks", 64)),
            telemetry_stale_after_s=float(safety_data.get("telemetry_stale_after_s", 0.25)),
            pretension_untensioned_reference_tick=int(
                safety_data.get("pretension_untensioned_reference_tick", 4095)
            ),
            pretension_step_ticks=int(safety_data.get("pretension_step_ticks", 2)),
            pretension_timeout_s=float(safety_data.get("pretension_timeout_s", 10.0)),
            pretension_settle_time_s=float(safety_data.get("pretension_settle_time_s", 0.05)),
            pretension_baseline_sample_count=int(
                safety_data.get("pretension_baseline_sample_count", 5)
            ),
            pretension_current_filter_window=int(
                safety_data.get("pretension_current_filter_window", 3)
            ),
            pretension_current_delta_threshold_ma=int(
                safety_data.get("pretension_current_delta_threshold_ma", 60)
            ),
            pretension_absolute_trigger_current_ma=self._maybe_int(
                safety_data.get("pretension_absolute_trigger_current_ma", 220)
            ),
            pretension_hard_current_stop_ma=int(
                safety_data.get(
                    "pretension_hard_current_stop_ma",
                    safety_data.get("max_current_ma", 850),
                )
            ),
            pretension_max_travel_ticks=int(safety_data.get("pretension_max_travel_ticks", 320)),
            max_temperature_c=int(safety_data.get("max_temperature_c", 70)),
            min_input_voltage_mv=int(safety_data.get("min_input_voltage_mv", 4000)),
        )
        candidate_landmarks = self._load_registration_landmarks(registration_data)
        nominal_landmarks = {
            landmark.id: list(landmark.xyz_mm)
            for landmark in candidate_landmarks
        }
        nominal_landmarks.update(
            {
                str(k): [float(v) for v in values]
                for k, values in registration_data.get("nominal_landmarks_robot_xyz_mm", {}).items()
            }
        )
        configured_labels = registration_data.get("landmark_labels")
        if configured_labels is None:
            configured_labels = [landmark.id for landmark in candidate_landmarks if landmark.enabled][:4]
        configured_labels = list(configured_labels or ["L1", "L2", "L3", "L4"])
        registration = RegistrationWorkflowConfig(
            landmark_labels=[str(v) for v in configured_labels],
            captures_per_landmark=int(registration_data.get("captures_per_landmark", 5)),
            nominal_landmarks_robot_xyz_mm=nominal_landmarks,
            candidate_landmarks=candidate_landmarks,
            capture_tool_id=str(registration_data.get("capture_tool_id", "0B")),
            coil_tool_id=str(registration_data.get("coil_tool_id", "0A")),
            capture_tool_tip_transform=self._maybe_matrix(
                registration_data.get("capture_tool_tip_transform")
            ),
            model_points_file=self._maybe_path(registration_data.get("model_points_file")),
            tip_points_file=self._maybe_path(registration_data.get("tip_points_file")),
            T_sw_2_model_file=self._maybe_path(registration_data.get("T_sw_2_model_file")),
            T_sw_2_tip_file=self._maybe_path(registration_data.get("T_sw_2_tip_file")),
            penprobe_file=self._maybe_path(registration_data.get("penprobe_file", "tools/penprobe_08_09_24c")),
            quaternion_average_method=str(registration_data.get("quaternion_average_method", "sign_aligned_mean")),
            model_tre_reference_radius_mm=float(registration_data.get("model_tre_reference_radius_mm", 5.0)),
            tip_tre_reference_radius_mm=float(registration_data.get("tip_tre_reference_radius_mm", 3.0)),
            max_fre_mm=self._maybe_float(
                (registration_data.get("validation", {}) or {}).get("max_fre_mm", 2.0)
            ),
            runtime_tip_truth_points_file=self._maybe_path(
                registration_data.get("runtime_tip_truth_points_file", "tools/all_tip_registration_points_in_sw")
            ),
            runtime_tip_T_sw_2_tip_file=self._maybe_path(
                registration_data.get("runtime_tip_T_sw_2_tip_file", "tools/T_sw_2_tip")
            ),
            runtime_tip_captures_per_landmark=int(
                registration_data.get("runtime_tip_captures_per_landmark", 3)
            ),
            runtime_tip_coil_sample_count=int(
                registration_data.get("runtime_tip_coil_sample_count", 50)
            ),
            runtime_tip_coil_sample_interval_s=float(
                registration_data.get("runtime_tip_coil_sample_interval_s", 0.02)
            ),
            runtime_tip_max_hat_rmse_mm=self._maybe_float(
                registration_data.get("runtime_tip_max_hat_rmse_mm", 2.0)
            ),
            runtime_tip_setup_id=(
                str(registration_data.get("runtime_tip_setup_id"))
                if registration_data.get("runtime_tip_setup_id") not in (None, "")
                else None
            ),
        )
        experiment = ExperimentConfig(
            default_settle_time_s=float(experiment_data.get("default_settle_time_s", 2.0)),
            sample_count_per_point=int(experiment_data.get("sample_count_per_point", 1)),
            output_dir=str(experiment_data.get("output_dir", "data/experiments")),
        )
        calibration = CalibrationConfig(
            neutral_setpoints_path=str(
                system_data.get("neutral_setpoints_path", "data/calibrations/neutral_setpoints.json")
            ),
            latest_registration_path=str(
                system_data.get("latest_registration_path", "data/registrations/latest_registration.json")
            ),
            latest_runtime_tip_calibration_path=str(
                system_data.get(
                    "latest_runtime_tip_calibration_path",
                    "data/calibrations/runtime_tip/latest_runtime_tip_calibration.json",
                )
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

    def list_robot_configs(self) -> list[str]:
        """Return available robot config file names."""
        return sorted(path.name for path in self.base_dir.glob("robot_*.yaml") if path.is_file())

    def load_robot_config(self, robot_config_name: str) -> RobotConfig:
        """Load one robot config file without the rest of the app settings."""
        robot_path = self.base_dir / str(robot_config_name)
        robot_data = self._read_yaml(robot_path)
        if not robot_data:
            raise FileNotFoundError(f"Robot config {robot_config_name} was not found in {self.base_dir}.")
        return RobotConfig(
            mode=str(robot_data.get("mode", "4-servo")),
            spool_diameter_cm=float(robot_data.get("spool_diameter_cm", 1.2)),
            ticks_per_revolution=int(robot_data.get("ticks_per_revolution", 4096)),
            servo_ids=[int(v) for v in robot_data.get("servo_ids", [1, 2, 3, 4])],
            tendon_to_servo=[int(v) for v in robot_data.get("tendon_to_servo", [1, 2, 3, 4])],
            tightening_rotation_by_servo={
                int(key): str(value).strip().lower()
                for key, value in dict(robot_data.get("tightening_rotation_by_servo", {}) or {}).items()
                if str(value).strip()
            },
        )

    def load_system_local_overrides(self) -> dict:
        """Return the current machine-local system override mapping."""
        return self._read_yaml(self.base_dir / "system.local.yaml")

    def save_system_local_overrides(self, overrides: dict) -> Path:
        """Merge and save machine-local runtime overrides."""
        path = self.base_dir / "system.local.yaml"
        existing = self._read_yaml(path)
        merged = self._deep_merge(existing, overrides)
        with path.open("w", encoding="utf-8") as handle:
            yaml.safe_dump(merged, handle, sort_keys=False)
        return path

    @staticmethod
    def _maybe_float(value) -> float | None:
        if value in (None, ""):
            return None
        return float(value)

    @staticmethod
    def _maybe_int(value) -> int | None:
        if value in (None, ""):
            return None
        return int(value)

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

    @classmethod
    def _deep_merge(cls, base: dict, override: dict) -> dict:
        merged = dict(base)
        for key, value in dict(override or {}).items():
            if isinstance(value, dict) and isinstance(merged.get(key), dict):
                merged[key] = cls._deep_merge(dict(merged[key]), value)
            else:
                merged[key] = value
        return merged

    @staticmethod
    def _load_registration_landmarks(payload: dict) -> list[RegistrationLandmarkConfig]:
        landmarks: list[RegistrationLandmarkConfig] = []
        raw_landmarks = payload.get("candidate_landmarks", []) or []
        for index, entry in enumerate(raw_landmarks):
            if not isinstance(entry, dict):
                raise ValueError(f"candidate_landmarks[{index}] must be a mapping")
            landmark_id = str(entry.get("id") or entry.get("label") or entry.get("name") or "").strip()
            xyz = entry.get("xyz_mm") or entry.get("coordinates_mm") or entry.get("xyz")
            if not landmark_id:
                raise ValueError(f"candidate_landmarks[{index}] is missing id")
            if not isinstance(xyz, (list, tuple)) or len(xyz) != 3:
                raise ValueError(f"candidate_landmarks[{index}] must define xyz_mm with exactly 3 values")
            landmarks.append(
                RegistrationLandmarkConfig(
                    id=landmark_id,
                    xyz_mm=[float(value) for value in xyz],
                    display_label=(
                        str(entry.get("display_label")).strip()
                        if entry.get("display_label") not in (None, "")
                        else None
                    ),
                    enabled=bool(entry.get("enabled", True)),
                )
            )
        return landmarks

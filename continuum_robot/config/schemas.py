"""Typed config schemas used across the codebase."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class RuntimeConfig:
    """Application runtime configuration."""

    mock_mode: bool = True
    poll_rate_hz: int = 5
    robot_config: str = "robot_8servo.yaml"
    visualization_mode: str = "auto"
    visualization_safe_effects: bool = True
    figure_output_quality: str = "production"


@dataclass
class RobotSegmentConfig:
    """One physical four-tendon segment group."""

    key: str
    label: str
    segment_label: str = ""
    servo_ids: list[int] = field(default_factory=list)
    pairs: dict[str, list[int]] = field(default_factory=dict)
    segment_role: str = ""
    segment_order_index: int | None = None


@dataclass
class RobotOperatingContext:
    """Resolved servo scope for the selected robot operating mode."""

    operating_mode: str
    expected_servo_ids: list[int]
    commanded_servo_ids: list[int]
    all_configured_servo_ids: list[int]
    segments: dict[str, RobotSegmentConfig]
    selected_servo_id: int | None = None
    active_segment_key: str | None = None
    active_segment_label: str | None = None
    active_segment_servo_ids: list[int] = field(default_factory=list)
    active_pairs: dict[str, list[int]] = field(default_factory=dict)
    segment_order: list[str] = field(default_factory=list)
    mirror_pairs: dict[int, int] = field(default_factory=dict)
    readiness_scope: str = "expected"
    mode_profile: str = ""
    mode_capabilities: dict[str, bool] = field(default_factory=dict)
    mode_notes: list[str] = field(default_factory=list)

    def metadata(self) -> dict:
        return {
            "operating_mode": self.operating_mode,
            "expected_servo_ids": list(self.expected_servo_ids),
            "commanded_servo_ids": list(self.commanded_servo_ids),
            "selected_servo_id": self.selected_servo_id,
            "segment_order": list(self.segment_order),
            "active_segment": (
                {
                    "key": self.active_segment_key,
                    "label": self.active_segment_label,
                    "servo_ids": list(self.active_segment_servo_ids),
                    "pairs": {
                        str(key): [int(value) for value in values]
                        for key, values in dict(self.active_pairs or {}).items()
                    },
                    "segment_role": (
                        self.segments[self.active_segment_key].segment_role
                        if self.active_segment_key in self.segments
                        else ""
                    ),
                    "segment_order_index": (
                        self.segments[self.active_segment_key].segment_order_index
                        if self.active_segment_key in self.segments
                        else None
                    ),
                }
                if self.active_segment_key
                else None
            ),
            "segments": {
                str(key): {
                    "key": str(segment.key),
                    "label": str(segment.label),
                    "segment_label": str(segment.segment_label or segment.label),
                    "servo_ids": [int(value) for value in segment.servo_ids],
                    "segment_role": str(segment.segment_role or ""),
                    "segment_order_index": segment.segment_order_index,
                    "pairs": {
                        str(pair_key): [int(value) for value in pair_values]
                        for pair_key, pair_values in dict(segment.pairs or {}).items()
                    },
                }
                for key, segment in sorted(dict(self.segments or {}).items())
            },
            "mirror_pairs": {str(key): int(value) for key, value in sorted(dict(self.mirror_pairs or {}).items())},
            "readiness_scope": self.readiness_scope,
            "output_mode_family": (
                "mirrored_parallel"
                if self.operating_mode == "parallel_single"
                else ("dual_segment" if self.operating_mode == "dual_segment" else self.operating_mode)
            ),
            "mode_profile": self.mode_profile or self.operating_mode,
            "mode_capabilities": dict(self.mode_capabilities),
            "mode_notes": list(self.mode_notes),
        }


@dataclass
class RobotConfig:
    """Robot hardware configuration."""

    mode: str = "single_segment"
    spool_diameter_cm: float = 2.0
    ticks_per_revolution: int = 4096
    servo_ids: list[int] = field(default_factory=lambda: [1, 2, 3, 4])
    tendon_to_servo: list[int] = field(default_factory=lambda: [1, 2, 3, 4])
    tightening_rotation_by_servo: dict[int, str] = field(default_factory=dict)
    active_segment: str = "segment_a"
    selected_servo_id: int = 1
    segments: dict[str, RobotSegmentConfig] = field(default_factory=dict)

    @staticmethod
    def default_segment_role(key: str) -> str:
        normalized = str(key or "").strip().lower()
        if normalized == "segment_a":
            return "proximal"
        if normalized == "segment_b":
            return "distal"
        return ""

    @staticmethod
    def default_segment_order_index(key: str) -> int | None:
        normalized = str(key or "").strip().lower()
        if normalized == "segment_a":
            return 0
        if normalized == "segment_b":
            return 1
        return None

    def operating_mode(self) -> str:
        raw = str(self.mode or "single_segment").strip().lower().replace("-", "_")
        aliases = {
            "1_servo": "one_servo",
            "one-servo": "one_servo",
            "1-servo": "one_servo",
            "4_servo": "single_segment",
            "4-servo": "single_segment",
            "8_servo": "dual_segment",
            "8-servo": "dual_segment",
            "single": "single_segment",
            "parallel": "parallel_single",
        }
        return aliases.get(raw, raw if raw in {"one_servo", "single_segment", "dual_segment", "parallel_single"} else "single_segment")

    def segment_map(self) -> dict[str, RobotSegmentConfig]:
        if self.segments:
            normalized: dict[str, RobotSegmentConfig] = {}
            for key, segment in dict(self.segments).items():
                segment_key = str(segment.key or key)
                servo_ids = [int(value) for value in segment.servo_ids]
                pairs = {
                    str(pair_key): [int(value) for value in pair_values]
                    for pair_key, pair_values in dict(segment.pairs or {}).items()
                }
                if not pairs and len(servo_ids) >= 4:
                    pairs = {"axis_a": [servo_ids[0], servo_ids[2]], "axis_b": [servo_ids[1], servo_ids[3]]}
                normalized[str(key)] = RobotSegmentConfig(
                    key=segment_key,
                    label=str(segment.label or key),
                    segment_label=str(segment.segment_label or segment.label or key),
                    servo_ids=servo_ids,
                    pairs=pairs,
                    segment_role=str(segment.segment_role or self.default_segment_role(key)),
                    segment_order_index=(
                        segment.segment_order_index
                        if segment.segment_order_index is not None
                        else self.default_segment_order_index(key)
                    ),
                )
            return normalized
        ids = [int(value) for value in (self.tendon_to_servo or self.servo_ids or [])]
        pairs = {}
        if len(ids) >= 4:
            pairs = {"axis_a": [ids[0], ids[2]], "axis_b": [ids[1], ids[3]]}
        return {
            "segment_a": RobotSegmentConfig(
                key="segment_a",
                label="Spine 1",
                segment_label="Segment A",
                servo_ids=list(ids[:4]),
                pairs=pairs,
                segment_role="proximal",
                segment_order_index=0,
            )
        }

    def segment_order(self) -> list[str]:
        segments = self.segment_map()

        def _sort_key(item: tuple[str, RobotSegmentConfig]) -> tuple[int, str]:
            key, segment = item
            index = segment.segment_order_index
            if index is None:
                default_index = self.default_segment_order_index(key)
                index = default_index if default_index is not None else 1000
            return (int(index), str(key))

        return [str(key) for key, _segment in sorted(segments.items(), key=_sort_key)]

    def segment_metadata(self) -> dict[str, dict]:
        metadata: dict[str, dict] = {}
        for key in self.segment_order():
            segment = self.segment_map()[key]
            pairs = {str(pair_key): [int(value) for value in values] for pair_key, values in dict(segment.pairs or {}).items()}
            metadata[str(key)] = {
                "key": str(segment.key),
                "label": str(segment.label),
                "segment_label": str(segment.segment_label or segment.label),
                "segment_role": str(segment.segment_role or self.default_segment_role(key)),
                "segment_order_index": (
                    segment.segment_order_index
                    if segment.segment_order_index is not None
                    else self.default_segment_order_index(key)
                ),
                "servo_ids": [int(value) for value in segment.servo_ids],
                "pairs": pairs,
            }
        return metadata

    def active_segment_config(self) -> RobotSegmentConfig:
        segments = self.segment_map()
        key = str(self.active_segment or "segment_a")
        if key in segments:
            return segments[key]
        return next(iter(segments.values()))

    def active_segment_key(self) -> str:
        return str(self.active_segment_config().key)

    def active_segment_label(self) -> str:
        return str(self.active_segment_config().label)

    def active_segment_servo_ids(self) -> list[int]:
        return [int(value) for value in self.active_segment_config().servo_ids]

    def active_segment_pairs(self) -> dict[str, list[int]]:
        segment = self.active_segment_config()
        pairs = {str(key): [int(value) for value in values] for key, values in dict(segment.pairs or {}).items()}
        ids = [int(value) for value in segment.servo_ids]
        if not pairs and len(ids) >= 4:
            pairs = {"axis_a": [ids[0], ids[2]], "axis_b": [ids[1], ids[3]]}
        return pairs

    def all_segment_servo_ids(self) -> list[int]:
        ids: list[int] = []
        segments = self.segment_map()
        for key in self.segment_order():
            segment = segments[key]
            for servo_id in segment.servo_ids:
                sid = int(servo_id)
                if sid not in ids:
                    ids.append(sid)
        return ids or [int(value) for value in self.servo_ids]

    def parallel_mirror_pairs(self) -> dict[int, int]:
        segments = self.segment_map()
        segment_a = segments.get("segment_a")
        segment_b = segments.get("segment_b")
        if segment_a is None or segment_b is None:
            return {}
        ids_a = [int(value) for value in segment_a.servo_ids]
        ids_b = [int(value) for value in segment_b.servo_ids]
        return {int(a): int(b) for a, b in zip(ids_a, ids_b)}

    def operating_context(self) -> RobotOperatingContext:
        mode = self.operating_mode()
        segments = self.segment_map()
        segment_order = self.segment_order()
        all_ids = self.all_segment_servo_ids()
        active = self.active_segment_config()
        active_ids = [int(value) for value in active.servo_ids]
        active_pairs = self.active_segment_pairs()
        dual_segment_note = (
            "dual_segment currently supports all-8 readiness and manual startup capture; "
            "full two-segment kinematics/control and automatic two-segment pretension are not implemented."
        )
        if mode == "one_servo":
            selected = int(self.selected_servo_id or (all_ids[0] if all_ids else 1))
            return RobotOperatingContext(
                operating_mode=mode,
                expected_servo_ids=[selected],
                commanded_servo_ids=[selected],
                all_configured_servo_ids=list(all_ids),
                segments=segments,
                selected_servo_id=selected,
                segment_order=list(segment_order),
                readiness_scope="selected_servo",
                mode_profile="one_servo_bringup",
                mode_capabilities={
                    "manual_jog": True,
                    "manual_startup_capture": False,
                    "automatic_single_segment_pretension": False,
                    "two_segment_kinematics_control": False,
                },
            )
        if mode == "dual_segment":
            return RobotOperatingContext(
                operating_mode=mode,
                expected_servo_ids=list(all_ids),
                commanded_servo_ids=list(all_ids),
                all_configured_servo_ids=list(all_ids),
                segments=segments,
                segment_order=list(segment_order),
                readiness_scope="all_segments",
                mode_profile="all_8_readiness_manual_startup_foundation",
                mode_capabilities={
                    "all_8_readiness": True,
                    "all_8_telemetry": True,
                    "manual_jog": True,
                    "manual_startup_capture": True,
                    "automatic_single_segment_pretension": False,
                    "automatic_two_segment_pretension": False,
                    "two_segment_kinematics_control": False,
                    "two_segment_modeling": False,
                    "penprobe_chasing": False,
                },
                mode_notes=[dual_segment_note],
            )
        if mode == "parallel_single":
            return RobotOperatingContext(
                operating_mode=mode,
                expected_servo_ids=list(all_ids),
                commanded_servo_ids=list(all_ids),
                all_configured_servo_ids=list(all_ids),
                segments=segments,
                active_segment_key=str(active.key),
                active_segment_label=str(active.label),
                active_segment_servo_ids=list(active_ids),
                active_pairs=active_pairs,
                segment_order=list(segment_order),
                mirror_pairs=self.parallel_mirror_pairs(),
                readiness_scope="mirrored_segments",
                mode_profile="mirrored_single_segment_testing",
                mode_capabilities={
                    "all_8_readiness": True,
                    "manual_jog": True,
                    "manual_startup_capture": True,
                    "mirrored_single_segment_commands": True,
                    "two_segment_kinematics_control": False,
                },
                mode_notes=["parallel_single is mirrored single-segment command testing, not two-segment control."],
            )
        return RobotOperatingContext(
            operating_mode="single_segment",
            expected_servo_ids=list(active_ids),
            commanded_servo_ids=list(active_ids),
            all_configured_servo_ids=list(all_ids),
            segments=segments,
            active_segment_key=str(active.key),
            active_segment_label=str(active.label),
            active_segment_servo_ids=list(active_ids),
            active_pairs=active_pairs,
            segment_order=list(segment_order),
            readiness_scope="active_segment",
            mode_profile="single_segment",
            mode_capabilities={
                "manual_jog": True,
                "manual_startup_capture": True,
                "automatic_single_segment_pretension": True,
                "two_segment_kinematics_control": False,
            },
        )

    def expected_servo_ids(self) -> list[int]:
        return self.operating_context().expected_servo_ids

    def commanded_servo_ids(self) -> list[int]:
        return self.operating_context().commanded_servo_ids


@dataclass
class SerialConfig:
    """Serial defaults for Aurora and OpenRB."""

    aurora_port: str = ""
    openrb_port: str = ""
    tracker_backend: str = "ndi"
    tracker_fallback_backend: str | None = None
    tracker_fallback_enabled: bool = False
    tracker_type: str = "aurora"
    baudrate: int = 115200
    read_timeout_s: float = 0.05
    frame_timeout_s: float = 0.5
    reconnect_delay_s: float = 1.0
    tracker_freshness_timeout_s: float = 0.5
    tracker_ports_to_probe: list[str] = field(default_factory=list)
    tracker_settings_overrides: dict = field(default_factory=dict)
    tracker_tool_id_aliases: dict = field(default_factory=dict)
    # Retained only for explicit legacy bridge fallback. The Python NDI path is
    # the canonical active backend.
    tracker_socket_path: str = "/tmp/tracker_bridge.sock"
    tracker_bridge_executable: str = "bin/tracker_bridge"
    tracker_poll_ms: int = 20
    tracker_min_effective_fps: float = 15.0
    tracker_max_stale_interval_s: float = 0.25
    tracker_max_consecutive_missing_frames: int = 20
    tracker_require_valid_transforms: bool = True
    packet_capture_dir: str = ""
    openrb_settings: dict = field(default_factory=dict)
    dynamixel_settings: dict = field(default_factory=dict)


@dataclass
class SafetyConfig:
    """Safety limits for servo command validation."""

    position_min_offset_ticks: int = -600
    position_max_offset_ticks: int = 600
    max_current_ma: int = 850
    servo_model: str = "XC330-M288-T"
    servo_reported_current_hard_limit_ma: int = 850
    servo_reported_current_warning_ma: int | None = None
    current_safety_basis: str = (
        "XC330-M288-T servo-reported input-current estimate; hard safety uses absolute current magnitude."
    )
    default_pretension_current_threshold_ma: int = 220
    pretension_current_balance_tolerance_ma: int = 120
    fine_jog_step_ticks: int = 5
    coarse_jog_step_ticks: int = 25
    software_position_margin_ticks: int = 64
    telemetry_stale_after_s: float = 0.25
    torque_off_on_disconnect: bool = False
    wrap_risk_margin_ticks: int = 128
    max_raw_jump_without_wrap_risk_ticks: int = 900
    pretension_untensioned_reference_tick: int = 4095
    pretension_start_mode: str = "current_position"
    pretension_step_ticks: int = 2
    pretension_timeout_s: float = 10.0
    pretension_settle_time_s: float = 0.05
    pretension_baseline_sample_count: int = 5
    pretension_current_filter_window: int = 3
    pretension_current_delta_threshold_ma: int = 60
    pretension_absolute_trigger_current_ma: int | None = 220
    pretension_hard_current_stop_ma: int = 850
    pretension_max_travel_ticks: int = 320
    max_temperature_c: int = 70
    min_input_voltage_mv: int = 4000


@dataclass
class RegistrationLandmarkConfig:
    """One selectable registration landmark from the robot/SolidWorks model."""

    id: str
    xyz_mm: list[float]
    display_label: str | None = None
    enabled: bool = True


@dataclass
class TwoSegmentTrackingRoleConfig:
    """Configured Aurora tool role for future two-segment pose labels."""

    role_name: str
    tool_id: str = ""
    physical_meaning: str = ""
    pose_kind: str = "coil_origin"
    required_for_two_segment_model_training: bool = False
    required_for_control: bool = False
    trust_level: str = "configured"
    source: str = "config"
    max_age_s: float | None = None
    enabled: bool = True

    def to_dict(self) -> dict:
        return {
            "role_name": str(self.role_name),
            "tool_id": str(self.tool_id),
            "physical_meaning": str(self.physical_meaning),
            "pose_kind": str(self.pose_kind),
            "required_for_two_segment_model_training": bool(self.required_for_two_segment_model_training),
            "required_for_control": bool(self.required_for_control),
            "trust_level": str(self.trust_level),
            "source": str(self.source),
            "max_age_s": self.max_age_s,
            "enabled": bool(self.enabled),
        }


@dataclass
class RegistrationWorkflowConfig:
    """Registration workflow defaults."""

    landmark_labels: list[str] = field(default_factory=lambda: ["L1", "L2", "L3", "L4"])
    captures_per_landmark: int = 5
    nominal_landmarks_robot_xyz_mm: dict[str, list[float]] = field(default_factory=dict)
    candidate_landmarks: list[RegistrationLandmarkConfig] = field(default_factory=list)
    capture_tool_id: str = "0B"
    coil_tool_id: str = "0A"
    capture_tool_tip_transform: list[list[float]] | None = None
    model_points_file: str | None = None
    tip_points_file: str | None = None
    T_sw_2_model_file: str | None = None
    T_sw_2_tip_file: str | None = None
    penprobe_file: str | None = "tools/penprobe_08_09_24c"
    quaternion_average_method: str = "sign_aligned_mean"
    model_tre_reference_radius_mm: float = 5.0
    tip_tre_reference_radius_mm: float = 3.0
    max_fre_mm: float | None = 2.0
    runtime_tip_truth_points_file: str | None = "tools/all_tip_registration_points_in_sw"
    runtime_tip_T_sw_2_tip_file: str | None = "tools/T_sw_2_tip"
    runtime_tip_captures_per_landmark: int = 3
    runtime_tip_coil_sample_count: int = 50
    runtime_tip_coil_sample_interval_s: float = 0.02
    runtime_tip_max_hat_rmse_mm: float | None = 2.0
    runtime_tip_setup_id: str | None = None
    two_segment_tracking_roles: dict[str, TwoSegmentTrackingRoleConfig] = field(default_factory=dict)


@dataclass
class ExperimentConfig:
    """Experiment execution defaults."""

    default_settle_time_s: float = 2.0
    sample_count_per_point: int = 1
    output_dir: str = "data/experiments"


@dataclass
class CalibrationConfig:
    """Persistent calibration file locations."""

    neutral_setpoints_path: str = "config/neutral_setpoints.json"
    latest_registration_path: str = "data/registrations/latest_registration.json"
    latest_runtime_tip_calibration_path: str = (
        "data/runtime_tip_calibration/latest_runtime_tip_calibration.json"
    )

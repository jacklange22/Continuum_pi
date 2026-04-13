from pathlib import Path
import json
import threading
import time

import numpy as np
import pytest

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
from continuum_robot.experiments.dataset_io import ExperimentDatasetLoader, ExperimentDatasetWriter
from continuum_robot.experiments.experiment_runner import ExperimentRunner
from continuum_robot.experiments.framework import BaseExperiment, ExperimentHardwareRequirements, ExperimentSession
from continuum_robot.experiments.registry import ExperimentRegistry
from continuum_robot.experiments.schedules import CommandScheduleConfig, command_schedule_checksum, generate_command_schedule
from continuum_robot.experiments.schemas import ExperimentMetadata, ExperimentSummary, ExperimentTimeseriesSample
from continuum_robot.hardware.mock_dxl_bus import MockDxlBus
from continuum_robot.services.models import HEALTH_HEALTHY, ServiceHealthSnapshot, ToolTrackingSnapshot, TrackingSnapshot
from continuum_robot.services.tracking_service import TrackingService
from continuum_robot.servos.displacement_mapper import TendonDisplacementMapper
from continuum_robot.servos.neutral_calibration_service import NeutralCalibrationService
from continuum_robot.servos.pretension_validation_service import PretensionValidationService
from continuum_robot.servos.safety_guard import SafetyGuard
from continuum_robot.servos.servo_service import ServoService
from continuum_robot.tracking.mock_tracker_manager import MockTrackerManager


def _settings() -> Settings:
    return Settings(
        runtime=RuntimeConfig(mock_mode=True, poll_rate_hz=20, robot_config="robot_4servo.yaml"),
        robot=RobotConfig(mode="4-servo", spool_diameter_cm=1.2, ticks_per_revolution=4096, servo_ids=[1, 2, 3, 4]),
        serial=SerialConfig(aurora_port="/dev/mock-aurora", openrb_port="/dev/mock-openrb", baudrate=115200),
        safety=SafetyConfig(
            position_min_offset_ticks=-600,
            position_max_offset_ticks=600,
            max_current_ma=850,
            pretension_current_balance_tolerance_ma=120,
        ),
        registration=RegistrationWorkflowConfig(capture_tool_id="0B", coil_tool_id="0A", max_fre_mm=None),
        experiment=ExperimentConfig(default_settle_time_s=0.0, sample_count_per_point=1, output_dir="data/experiments"),
        calibration=CalibrationConfig(
            neutral_setpoints_path="config/neutral_setpoints.json",
            latest_registration_path="data/registrations/latest_registration.json",
        ),
    )


def _servo_service(tmp_path: Path) -> ServoService:
    return ServoService(
        dxl_bus=MockDxlBus([1, 2, 3, 4]),
        mapper=TendonDisplacementMapper(spool_diameter_cm=1.2),
        safety_guard=SafetyGuard(min_offset_ticks=-600, max_offset_ticks=600, max_current_ma=850),
        neutral_calibration=NeutralCalibrationService(path=tmp_path / "neutral.json"),
        pretension_validation=PretensionValidationService(),
    )


class _PretensionExperimentBus(MockDxlBus):
    def __init__(self) -> None:
        super().__init__([1, 2, 3, 4])
        self._current_sequence = [180, 230, 180, 235, 180, 240]
        for telemetry in self._state.values():
            telemetry.torque_enabled = True
            telemetry.present_position = 4031
            telemetry.present_current_ma = 150

    def write_goal_positions(self, positions_by_id: dict[int, int]) -> None:
        super().write_goal_positions(positions_by_id)
        if 1 in positions_by_id and self._current_sequence:
            self._state[1].present_current_ma = self._current_sequence.pop(0)


class _SequencedTrackingService:
    def __init__(self, snapshots: list[TrackingSnapshot]) -> None:
        self._snapshots = list(snapshots)
        self._index = 0
        self._thread = object()

    def peek_snapshot(self) -> TrackingSnapshot:
        if not self._snapshots:
            raise RuntimeError("No tracking snapshots configured for test.")
        if self._index >= len(self._snapshots):
            return self._snapshots[-1]
        return self._snapshots[self._index]

    def get_snapshot(self) -> TrackingSnapshot:
        if not self._snapshots:
            raise RuntimeError("No tracking snapshots configured for test.")
        if self._index >= len(self._snapshots):
            return self._snapshots[-1]
        snapshot = self._snapshots[self._index]
        self._index += 1
        return snapshot

    def start(self) -> None:
        self._thread = object()

    def stop(self) -> None:
        self._thread = None


class _TimingDiagnosticTrackingService:
    def __init__(self, records: list[dict], *, backend_identity: str = "ndi_tracker_python") -> None:
        self._records = [dict(record) for record in records]
        self._listeners: list = []
        self._thread = None
        self._backend_identity = backend_identity
        self._emitter_thread: threading.Thread | None = None

    def start(self) -> None:
        self._thread = object()

    def stop(self) -> None:
        self._thread = None
        emitter = self._emitter_thread
        if emitter is not None and emitter.is_alive():
            emitter.join(timeout=1.0)
        self._emitter_thread = None

    def register_timing_listener(self, listener) -> None:
        self._listeners.append(listener)
        if self._emitter_thread is None:
            self._emitter_thread = threading.Thread(target=self._emit_records, daemon=True)
            self._emitter_thread.start()

    def unregister_timing_listener(self, listener) -> None:
        self._listeners = [item for item in self._listeners if item is not listener]

    def _emit_records(self) -> None:
        for record in self._records:
            payload = dict(record)
            payload.setdefault("backend_identity", self._backend_identity)
            for listener in list(self._listeners):
                listener(payload)
            time.sleep(0.001)

    def peek_snapshot(self) -> TrackingSnapshot:
        return self.get_snapshot()

    def get_snapshot(self) -> TrackingSnapshot:
        return TrackingSnapshot(
            health=ServiceHealthSnapshot(
                name="tracking_service",
                health=HEALTH_HEALTHY,
                state="tracking",
                status="ok",
            ),
            connection_state="tracking",
            canonical_state="streaming_healthy",
            backend_identity=self._backend_identity,
            configured_backend_name="ndi",
            selected_backend_name="ndi",
            runtime_coil_tool_id="0A",
            registration_tool_id="0B",
            tracker_data_age_s=0.01,
            tracker_data_stale=False,
            last_frame_number=(
                int(self._records[-1]["frame_number"])
                if self._records and self._records[-1].get("frame_number") is not None
                else None
            ),
            tools={
                "0A": ToolTrackingSnapshot(tool_id="0A"),
                "0B": ToolTrackingSnapshot(tool_id="0B"),
            },
        )


def _tracking_snapshot(*, translation_mm: list[float]) -> TrackingSnapshot:
    return TrackingSnapshot(
        health=ServiceHealthSnapshot(
            name="tracking_service",
            health=HEALTH_HEALTHY,
            state="tracking",
            status="ok",
        ),
        connection_state="tracking",
        canonical_state="streaming_healthy",
        backend_identity="mock",
        selected_backend_name="mock",
        runtime_coil_tool_id="0A",
        registration_tool_id="0B",
        tracker_data_age_s=0.01,
        tracker_data_stale=False,
        last_frame_number=1,
        normalized_live_tool_ids=["0A"],
        tools={
            "0A": ToolTrackingSnapshot(
                tool_id="0A",
                present=True,
                valid=True,
                validity_known=True,
                tracking_state="valid",
                status="ok",
                frame_number=1,
                translation_mm=tuple(float(value) for value in translation_mm),
                quaternion_wxyz=(1.0, 0.0, 0.0, 0.0),
            ),
            "0B": ToolTrackingSnapshot(tool_id="0B"),
        },
    )


def _tracking_service(settings: Settings, tmp_path: Path, registration_path: Path | None = None) -> TrackingService:
    return TrackingService(
        live_backend=MockTrackerManager(poll_hz=30),
        port=settings.serial.aurora_port,
        registration_path=registration_path or (tmp_path / "latest_registration.json"),
        config_source="test",
        runtime_coil_tool_id=settings.registration.coil_tool_id,
        registration_tool_id=settings.registration.capture_tool_id,
    )


def _runner(
    tmp_path: Path,
    *,
    settings: Settings | None = None,
    servo_service: ServoService | None = None,
    tracking_service: TrackingService | None = None,
    registration_path: Path | None = None,
    registry: ExperimentRegistry | None = None,
) -> ExperimentRunner:
    settings = settings or _settings()
    registration_path = registration_path or (tmp_path / "latest_registration.json")
    return ExperimentRunner(
        project_root=Path(__file__).resolve().parents[1],
        settings=settings,
        tracking_service=tracking_service or _tracking_service(settings, tmp_path, registration_path),
        servo_service=servo_service or _servo_service(tmp_path),
        output_dir=tmp_path / "data" / "experiments",
        default_settle_time_s=0.0,
        registration_path=registration_path,
        sleep_fn=lambda _seconds: None,
        experiment_registry=registry,
    )


class LifecycleProbeExperiment(BaseExperiment):
    name = "lifecycle_probe"
    description = "Exercise lifecycle hooks for tests."
    hardware_requirements = ExperimentHardwareRequirements(mock_compatible=True)

    @classmethod
    def from_dict(cls, payload=None):
        return cls(config=dict(payload or {}))

    def setup(self, session: ExperimentSession) -> None:
        session.set_metric("setup_called", True)

    def precheck(self, session: ExperimentSession) -> None:
        session.set_metric("precheck_called", True)

    def execute(self, session: ExperimentSession) -> None:
        session.add_sample(
            ExperimentTimeseriesSample(
                monotonic_time_s=0.0,
                wall_time_utc="2026-01-01T00:00:00+00:00",
                phase="execute",
                step_index=0,
                sample_index=0,
                status_flags=["ok"],
            )
        )

    def finalize(self, session: ExperimentSession) -> None:
        session.set_metric("finalize_called", True)

    def summarize(self, session: ExperimentSession) -> dict:
        return {"summary_called": True}


def test_experiment_lifecycle_records_stage_results(tmp_path: Path) -> None:
    registry = ExperimentRegistry()
    registry.register(
        name=LifecycleProbeExperiment.name,
        description=LifecycleProbeExperiment.description,
        factory=LifecycleProbeExperiment.from_dict,
    )
    runner = _runner(tmp_path, registry=registry)
    result = runner.run_experiment("lifecycle_probe")

    assert result.success is True
    assert result.summary.stage_pass_fail == {
        "setup": "passed",
        "precheck": "passed",
        "execute": "passed",
        "finalize": "passed",
    }
    assert result.summary.experiment_metrics["setup_called"] is True
    assert result.summary.experiment_metrics["precheck_called"] is True
    assert result.summary.experiment_metrics["finalize_called"] is True
    assert result.summary.experiment_metrics["summary_called"] is True


def test_dataset_writer_roundtrip_loads_canonical_bundle(tmp_path: Path) -> None:
    writer = ExperimentDatasetWriter(tmp_path / "datasets")
    metadata = ExperimentMetadata(
        schema_version="1.0",
        experiment_name="roundtrip_test",
        run_id="abc123",
        timestamp_utc="2026-01-01T00:00:00+00:00",
        git_commit=None,
        backend_info={"mock_mode": True},
        registration_info={"exists": False},
        config_used={"alpha": 1},
        operator_notes="note",
    )
    samples = [
        ExperimentTimeseriesSample(
            monotonic_time_s=0.0,
            wall_time_utc="2026-01-01T00:00:00+00:00",
            phase="sample",
            step_index=0,
            sample_index=0,
            status_flags=["ok"],
        )
    ]
    summary = ExperimentSummary(
        schema_version="1.0",
        experiment_name="roundtrip_test",
        run_id="abc123",
        success=True,
        sample_counts={"total": 1},
        dropped_frames=0,
        invalid_transforms=0,
        stage_pass_fail={"execute": "passed"},
    )

    paths = writer.write_dataset(metadata, samples, summary)
    bundle = ExperimentDatasetLoader().load_dataset(paths.output_dir)

    assert bundle.metadata.experiment_name == "roundtrip_test"
    assert paths.output_dir.parent == tmp_path / "datasets" / "roundtrip_test"
    assert len(bundle.samples) == 1
    assert bundle.summary.success is True


def test_tracker_pipeline_mock_runs_and_logs_samples(tmp_path: Path) -> None:
    runner = _runner(tmp_path)
    result = runner.run_experiment(
        "tracker_pipeline_mock",
        config={"sample_count": 4, "sample_period_s": 0.0},
    )

    assert result.success is True
    assert result.sample_count == 4
    bundle = runner.load_dataset(result.paths.output_dir)
    assert all(sample.phase == "sample" for sample in bundle.samples)
    assert bundle.summary.sample_counts["total"] == 4


def test_transform_chain_validation_reports_zero_error(tmp_path: Path) -> None:
    runner = _runner(tmp_path)
    result = runner.run_experiment("transform_chain_validation")

    assert result.success is True
    assert result.summary.experiment_metrics["translation_error_mm"] < 1e-9


def test_command_schedule_generation_is_deterministic_and_bounded() -> None:
    config = CommandScheduleConfig(kind="babble", dimensions=4, amplitude_cm=0.2, babble_count=8, seed=7)
    first = generate_command_schedule(config)
    second = generate_command_schedule(config)

    assert command_schedule_checksum(first) == command_schedule_checksum(second)
    for point in first:
        assert all(-0.2 <= value <= 0.2 for value in point.tendon_displacement_cm)


def test_replay_runner_loads_existing_dataset(tmp_path: Path) -> None:
    runner = _runner(tmp_path)
    source = runner.run_experiment("dataset_schema_roundtrip", config={"sample_count": 2})
    replay = runner.run_experiment("replay_runner", config={"dataset_path": str(source.paths.output_dir)})

    assert replay.success is True
    assert replay.summary.experiment_metrics["source_sample_count"] == 2
    assert replay.sample_count == 2


def test_tracker_timing_validation_experiment_writes_canonical_outputs_and_summary(tmp_path: Path) -> None:
    timing_records = [
        {
            "sample_start_monotonic_ns": 1_000_000_000,
            "backend_call_start_ns": 1_000_000_000,
            "backend_call_end_ns": 1_012_000_000,
            "parse_complete_ns": 1_013_000_000,
            "state_commit_complete_ns": 1_014_000_000,
            "sample_commit_monotonic_ns": 1_014_000_000,
            "observed_at_utc": "2026-01-01T00:00:00.000Z",
            "frame_number": 11,
            "frame_number_source": "device",
            "is_new_frame": True,
            "is_duplicate_frame": False,
            "raw_payload_available": True,
            "parsed_payload_available": True,
            "output_committed": True,
            "error_flag": False,
            "tools_visible": ["0A", "0B"],
            "raw_tool_ids": ["10", "11"],
            "normalized_tool_ids": ["0A", "0B"],
            "runtime_role_mappings": {"0A": "10", "0B": "11"},
            "tool_validity": {"0A": "tracked", "0B": "tracked"},
            "valid_transform_count": 2,
            "total_cycle_ms": 14.0,
            "backend_call_ms": 12.0,
            "parse_ms": 1.0,
            "state_commit_ms": 1.0,
        },
        {
            "sample_start_monotonic_ns": 1_020_000_000,
            "backend_call_start_ns": 1_020_000_000,
            "backend_call_end_ns": 1_032_000_000,
            "parse_complete_ns": 1_033_000_000,
            "state_commit_complete_ns": 1_034_000_000,
            "sample_commit_monotonic_ns": 1_034_000_000,
            "observed_at_utc": "2026-01-01T00:00:00.020Z",
            "frame_number": 11,
            "frame_number_source": "device",
            "is_new_frame": False,
            "is_duplicate_frame": True,
            "raw_payload_available": True,
            "parsed_payload_available": True,
            "output_committed": True,
            "error_flag": False,
            "tools_visible": ["0A"],
            "raw_tool_ids": ["10"],
            "normalized_tool_ids": ["0A"],
            "runtime_role_mappings": {"0A": "10"},
            "tool_validity": {"0A": "tracked", "0B": "missing"},
            "valid_transform_count": 1,
            "total_cycle_ms": 14.0,
            "backend_call_ms": 12.0,
            "parse_ms": 1.0,
            "state_commit_ms": 1.0,
        },
        {
            "sample_start_monotonic_ns": 1_040_000_000,
            "backend_call_start_ns": 1_040_000_000,
            "backend_call_end_ns": 1_052_000_000,
            "parse_complete_ns": 1_053_000_000,
            "state_commit_complete_ns": 1_054_000_000,
            "sample_commit_monotonic_ns": 1_054_000_000,
            "observed_at_utc": "2026-01-01T00:00:00.040Z",
            "frame_number": 12,
            "frame_number_source": "device",
            "is_new_frame": True,
            "is_duplicate_frame": False,
            "raw_payload_available": True,
            "parsed_payload_available": True,
            "output_committed": True,
            "error_flag": False,
            "tools_visible": ["0A", "0B"],
            "raw_tool_ids": ["10", "11"],
            "normalized_tool_ids": ["0A", "0B"],
            "runtime_role_mappings": {"0A": "10", "0B": "11"},
            "tool_validity": {"0A": "tracked", "0B": "tracked"},
            "valid_transform_count": 2,
            "total_cycle_ms": 14.0,
            "backend_call_ms": 12.0,
            "parse_ms": 1.0,
            "state_commit_ms": 1.0,
        },
    ]
    runner = _runner(
        tmp_path,
        tracking_service=_TimingDiagnosticTrackingService(timing_records),
    )

    result = runner.run_experiment(
        "tracker_timing_validation",
        config={
            "requested_tool_ids": ["0A", "0B"],
            "sample_count_target": 2,
            "warmup_samples": 1,
            "timeout_s": 1.0,
            "run_label": "bench_a",
            "enable_servo_logging": False,
        },
    )

    assert result.success is True
    assert result.summary.experiment_metrics["backend_identity"] == "ndi_tracker_python"
    assert result.summary.experiment_metrics["sample_count_analyzed"] == 2
    assert result.summary.experiment_metrics["warmup_discarded_count"] == 1
    assert result.summary.experiment_metrics["duplicate_frame_count"] == 1
    assert result.summary.experiment_metrics["unique_frame_count"] == 1
    assert result.summary.experiment_metrics["requested_tool_ids"] == ["0A", "0B"]
    assert result.summary.experiment_metrics["servo_sync"]["enabled"] is False
    assert result.paths.output_dir.joinpath("aurora_timing_histogram.png").exists()
    assert result.paths.output_dir.joinpath("aurora_timing_breakdown.png").exists()
    assert result.paths.output_dir.joinpath("aurora_timing_timeseries.png").exists()
    assert result.paths.output_dir.joinpath("aurora_timing_summary.txt").exists()
    assert not result.paths.output_dir.joinpath("aurora_timing_sync_offsets.png").exists()
    summary_text = result.paths.output_dir.joinpath("aurora_timing_summary.txt").read_text(encoding="utf-8")
    assert "GUI refresh rate is not used" in summary_text
    assert "Backend get_frame" in summary_text
    bundle = runner.load_dataset(result.paths.output_dir)
    assert any(sample.extra.get("record_kind") == "tracker_timing" for sample in bundle.samples)


def test_tracker_timing_validation_experiment_outputs_do_not_crash_when_optional_fields_are_missing(tmp_path: Path) -> None:
    timing_records = [
        {
            "sample_start_monotonic_ns": 2_000_000_000,
            "backend_call_start_ns": 2_000_000_000,
            "backend_call_end_ns": 2_180_000_000,
            "parse_complete_ns": None,
            "state_commit_complete_ns": 2_180_000_000,
            "sample_commit_monotonic_ns": 2_180_000_000,
            "observed_at_utc": "2026-01-01T00:00:01.000Z",
            "frame_number": None,
            "frame_number_source": "missing",
            "is_new_frame": None,
            "is_duplicate_frame": None,
            "raw_payload_available": False,
            "parsed_payload_available": False,
            "output_committed": False,
            "error_flag": True,
            "error_stage": "get_frame",
            "error_message": "timeout",
            "tools_visible": [],
            "raw_tool_ids": [],
            "normalized_tool_ids": [],
            "runtime_role_mappings": {},
            "tool_validity": {},
            "valid_transform_count": 0,
            "total_cycle_ms": 180.0,
            "backend_call_ms": 180.0,
            "parse_ms": None,
            "state_commit_ms": 0.0,
        }
    ]
    runner = _runner(
        tmp_path,
        tracking_service=_TimingDiagnosticTrackingService(timing_records),
    )

    result = runner.run_experiment(
        "tracker_timing_validation",
        config={
            "requested_tool_ids": ["0A"],
            "sample_count_target": 1,
            "warmup_samples": 0,
            "timeout_s": 1.0,
            "enable_servo_logging": False,
        },
    )

    assert result.success is True
    assert result.paths.output_dir.joinpath("aurora_timing_histogram.png").exists()
    assert result.paths.output_dir.joinpath("aurora_timing_breakdown.png").exists()
    assert result.paths.output_dir.joinpath("aurora_timing_timeseries.png").exists()
    assert result.summary.experiment_metrics["error_sample_count"] == 1


def test_pretension_validation_experiment_records_current_displacement_trace_and_outputs(tmp_path: Path) -> None:
    settings = _settings()
    servo_service = ServoService(
        dxl_bus=_PretensionExperimentBus(),
        mapper=TendonDisplacementMapper(spool_diameter_cm=1.2),
        safety_guard=SafetyGuard(
            min_offset_ticks=-600,
            max_offset_ticks=600,
            max_current_ma=850,
            default_pretension_current_threshold_ma=220,
            fine_jog_step_ticks=5,
            coarse_jog_step_ticks=25,
            software_position_margin_ticks=64,
            telemetry_stale_after_s=0.25,
            pretension_step_ticks=2,
            pretension_timeout_s=2.0,
            pretension_settle_time_s=0.0,
            pretension_baseline_sample_count=3,
            pretension_current_filter_window=1,
            pretension_current_delta_threshold_ma=60,
            pretension_absolute_trigger_current_ma=500,
            pretension_max_travel_ticks=320,
        ),
        neutral_calibration=NeutralCalibrationService(path=tmp_path / "pretension_neutral.json"),
        pretension_validation=PretensionValidationService(),
        sleep_fn=lambda _seconds: None,
    )
    servo_service.connect("/dev/mock-openrb", 115200)
    tracking_service = _SequencedTrackingService(
        [
            _tracking_snapshot(translation_mm=[0.0, 0.0, 0.0]),
            _tracking_snapshot(translation_mm=[1.0, 0.0, 0.0]),
            _tracking_snapshot(translation_mm=[1.0, 0.0, 0.0]),
            _tracking_snapshot(translation_mm=[0.0, 0.0, 0.0]),
            _tracking_snapshot(translation_mm=[1.2, 0.0, 0.0]),
            _tracking_snapshot(translation_mm=[1.2, 0.0, 0.0]),
        ]
    )
    runner = _runner(
        tmp_path,
        settings=settings,
        servo_service=servo_service,
        tracking_service=tracking_service,
    )

    result = runner.run_experiment(
        "pretension_validation",
        config={
            "servo_id": 1,
            "move_to_reference": False,
            "include_tracker_displacement": True,
        },
    )

    assert result.success is True
    assert result.summary.experiment_metrics["servo_id"] == 1
    assert result.summary.experiment_metrics["accepted"] is True
    assert result.summary.experiment_metrics["stop_reason"] == "baseline_delta_trigger"
    assert result.summary.experiment_metrics["baseline_current_ma"] == pytest.approx(150.0)
    assert result.summary.experiment_metrics["effective_trigger_current_ma"] == 210
    assert result.summary.experiment_metrics["max_observed_displacement_mm"] == pytest.approx(1.2)
    assert result.paths.output_dir.joinpath("pretension_response.png").exists()
    summary_note = result.paths.output_dir / "pretension_summary.txt"
    assert summary_note.exists()
    summary_text = summary_note.read_text(encoding="utf-8").lower()
    assert "engagement proxy" in summary_text
    assert "tendon tension" not in summary_text
    bundle = runner.load_dataset(result.paths.output_dir)
    assert any(sample.phase == "pretension_baseline" for sample in bundle.samples)
    assert any(sample.phase == "pretension_step" for sample in bundle.samples)
    assert any(sample.phase == "pretension_result" for sample in bundle.samples)
    assert any(sample.extra.get("travel_from_untensioned_ticks") is not None for sample in bundle.samples)
    assert any(sample.extra.get("tracker_displacement_mm") is not None for sample in bundle.samples)


def test_pretension_validation_experiment_writes_outputs_without_tracker_data(tmp_path: Path) -> None:
    settings = _settings()
    servo_service = ServoService(
        dxl_bus=_PretensionExperimentBus(),
        mapper=TendonDisplacementMapper(spool_diameter_cm=1.2),
        safety_guard=SafetyGuard(
            min_offset_ticks=-600,
            max_offset_ticks=600,
            max_current_ma=850,
            default_pretension_current_threshold_ma=220,
            fine_jog_step_ticks=5,
            coarse_jog_step_ticks=25,
            software_position_margin_ticks=64,
            telemetry_stale_after_s=0.25,
            pretension_step_ticks=2,
            pretension_timeout_s=2.0,
            pretension_settle_time_s=0.0,
            pretension_baseline_sample_count=3,
            pretension_current_filter_window=1,
            pretension_current_delta_threshold_ma=60,
            pretension_absolute_trigger_current_ma=500,
            pretension_max_travel_ticks=320,
        ),
        neutral_calibration=NeutralCalibrationService(path=tmp_path / "pretension_neutral.json"),
        pretension_validation=PretensionValidationService(),
        sleep_fn=lambda _seconds: None,
    )
    servo_service.connect("/dev/mock-openrb", 115200)
    runner = ExperimentRunner(
        project_root=Path(__file__).resolve().parents[1],
        settings=settings,
        tracking_service=None,
        servo_service=servo_service,
        output_dir=tmp_path / "data" / "experiments",
        default_settle_time_s=0.0,
        registration_path=tmp_path / "latest_registration.json",
        sleep_fn=lambda _seconds: None,
    )

    result = runner.run_experiment(
        "pretension_validation",
        config={
            "servo_id": 1,
            "move_to_reference": False,
            "include_tracker_displacement": True,
        },
    )

    assert result.success is True
    assert result.summary.experiment_metrics["tracker_metric_sample_count"] == 0
    assert result.paths.output_dir.joinpath("pretension_response.png").exists()
    assert result.paths.output_dir.joinpath("pretension_summary.txt").exists()


def test_collect_pose_command_dataset_marks_registration_missing(tmp_path: Path) -> None:
    runner = _runner(tmp_path)
    result = runner.run_experiment(
        "collect_pose_command_dataset",
        config={
            "dry_run": True,
            "sample_count_per_point": 2,
            "command_schedule": {
                "kind": "trajectory",
                "dimensions": 4,
                "trajectory_points_cm": [[0.0, 0.0, 0.0, 0.0], [0.1, 0.0, 0.0, 0.0]],
            },
        },
    )

    assert result.success is True
    bundle = runner.load_dataset(result.paths.output_dir)
    assert any("registration_missing" in sample.status_flags for sample in bundle.samples)
    assert not any("full_pose_available" in sample.status_flags for sample in bundle.samples)


def test_collect_pose_command_dataset_records_full_pose_when_registration_exists(tmp_path: Path) -> None:
    registration_path = tmp_path / "latest_registration.json"
    registration_path.write_text(
        json.dumps({"T_robot_aurora": np.eye(4).tolist(), "T_coil_tip": np.eye(4).tolist()}),
        encoding="utf-8",
    )
    settings = _settings()
    tracking_service = _tracking_service(settings, tmp_path, registration_path=registration_path)
    runner = _runner(tmp_path, settings=settings, tracking_service=tracking_service, registration_path=registration_path)

    result = runner.run_experiment(
        "collect_pose_command_dataset",
        config={
            "dry_run": True,
            "sample_count_per_point": 1,
            "command_schedule": {
                "kind": "trajectory",
                "dimensions": 4,
                "trajectory_points_cm": [[0.0, 0.0, 0.0, 0.0]],
            },
        },
    )

    assert result.success is True
    bundle = runner.load_dataset(result.paths.output_dir)
    assert any("full_pose_available" in sample.status_flags for sample in bundle.samples)

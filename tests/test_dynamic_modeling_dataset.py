"""Tests for the continuous dynamic modeling dataset experiment.

These tests cover trajectory bounds, reproducibility, sync alignment, the
CSV.GZ writer columns, summary metric computation, and the synthetic
not_thesis_evidence path. The experiment itself is exercised end-to-end
through ``ExperimentRunner`` using mock tracker/servo services so the suite
does not depend on live hardware.
"""

from __future__ import annotations

import csv
import gzip
import json
from pathlib import Path
from types import SimpleNamespace
import time
from typing import Any

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
from continuum_robot.experiments.dynamic_modeling_dataset import (
    BoundedRandomWalkTrajectory,
    COMMAND_COLUMNS,
    CommandRecord,
    DynamicModelingDatasetConfig,
    DynamicModelingDatasetExperiment,
    SERVO_COLUMNS,
    SYNC_COLUMNS,
    ServoTelemetryRecord,
    SynchronizedSample,
    TRACKER_COLUMNS,
    TrackerRecord,
    build_synchronized_sample,
    register_dynamic_modeling_dataset,
    summarize_dynamic_run,
    write_commands_csv_gz,
    write_servo_csv_gz,
    write_sync_csv_gz,
    write_tracker_csv_gz,
)
from continuum_robot.experiments.experiment_runner import ExperimentRunner
from continuum_robot.experiments.registry import ExperimentRegistry
from continuum_robot.experiments.builtins import register_builtin_experiments
from continuum_robot.hardware.mock_dxl_bus import MockDxlBus
from continuum_robot.services.models import (
    HEALTH_HEALTHY,
    ServiceHealthSnapshot,
    ToolTrackingSnapshot,
    TrackingSnapshot,
)
from continuum_robot.servos.displacement_mapper import TendonDisplacementMapper
from continuum_robot.servos.neutral_calibration_service import NeutralCalibrationService
from continuum_robot.servos.pretension_validation_service import PretensionValidationService
from continuum_robot.servos.safety_guard import SafetyGuard
from continuum_robot.servos.servo_service import ServoService


# ----------------------------------------------------------------------------
# Test helpers
# ----------------------------------------------------------------------------


def _settings(*, mock_mode: bool = True) -> Settings:
    return Settings(
        runtime=RuntimeConfig(mock_mode=mock_mode, poll_rate_hz=20, robot_config="robot_4servo.yaml"),
        robot=RobotConfig(
            mode="4-servo",
            spool_diameter_cm=1.2,
            ticks_per_revolution=4096,
            servo_ids=[1, 2, 3, 4],
        ),
        serial=SerialConfig(aurora_port="/dev/mock-aurora", openrb_port="/dev/mock-openrb", baudrate=115200),
        safety=SafetyConfig(
            position_min_offset_ticks=-600,
            position_max_offset_ticks=600,
            max_current_ma=850,
            pretension_current_balance_tolerance_ma=120,
        ),
        registration=RegistrationWorkflowConfig(capture_tool_id="0B", coil_tool_id="0A", max_fre_mm=None),
        experiment=ExperimentConfig(
            default_settle_time_s=0.0,
            sample_count_per_point=1,
            output_dir="data/experiments",
        ),
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


def _coil_as_tip_snapshot() -> TrackingSnapshot:
    snapshot = TrackingSnapshot(
        health=ServiceHealthSnapshot(
            name="tracking_service",
            health=HEALTH_HEALTHY,
            state="tracking",
            status="ok",
        ),
        connection_state="tracking",
        canonical_state="streaming_healthy",
        backend_identity="ndi_tracker_python",
        configured_backend_name="ndi",
        selected_backend_name="ndi",
        runtime_coil_tool_id="0A",
        registration_tool_id="0B",
        tracker_data_age_s=0.01,
        tracker_data_stale=False,
        last_frame_number=1,
        registration_state="loaded",
        runtime_tip_calibration_state="coil_as_tip",
        runtime_tip_mode="coil_as_tip",
        runtime_tip_trust_level="thesis_trusted",
        runtime_tip_mode_message="coil-as-tip mode",
        runtime_tip_identity_fallback=True,
        tip_pose_status="coil_as_tip",
        T_robot_tip=[
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
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
                translation_mm=(0.0, 0.0, 0.0),
                quaternion_wxyz=(1.0, 0.0, 0.0, 0.0),
            )
        },
    )
    return snapshot


class _StubTrackingService:
    """Lightweight tracking service stub for the experiment runner."""

    def __init__(self, *, start_frame_number: int = 1, snapshot: TrackingSnapshot | None = None) -> None:
        self._frame_counter = int(start_frame_number)
        self._template = snapshot or _coil_as_tip_snapshot()
        self._thread = object()  # tells the experiment "do not call start"

    def start(self) -> None:  # pragma: no cover - matches contract
        pass

    def stop(self) -> None:  # pragma: no cover - matches contract
        self._thread = None

    def get_snapshot(self) -> TrackingSnapshot:
        return self.peek_snapshot()

    def peek_snapshot(self) -> TrackingSnapshot:
        snapshot = TrackingSnapshot(
            health=self._template.health,
            connection_state=self._template.connection_state,
            canonical_state=self._template.canonical_state,
            backend_identity=self._template.backend_identity,
            configured_backend_name=self._template.configured_backend_name,
            selected_backend_name=self._template.selected_backend_name,
            runtime_coil_tool_id=self._template.runtime_coil_tool_id,
            registration_tool_id=self._template.registration_tool_id,
            tracker_data_age_s=self._template.tracker_data_age_s,
            tracker_data_stale=self._template.tracker_data_stale,
            last_frame_number=int(self._frame_counter),
            registration_state=self._template.registration_state,
            runtime_tip_calibration_state=self._template.runtime_tip_calibration_state,
            runtime_tip_mode=self._template.runtime_tip_mode,
            runtime_tip_trust_level=self._template.runtime_tip_trust_level,
            runtime_tip_mode_message=self._template.runtime_tip_mode_message,
            runtime_tip_identity_fallback=self._template.runtime_tip_identity_fallback,
            tip_pose_status=self._template.tip_pose_status,
            T_robot_tip=self._template.T_robot_tip,
            normalized_live_tool_ids=list(self._template.normalized_live_tool_ids),
            tools={
                key: ToolTrackingSnapshot(
                    tool_id=value.tool_id,
                    present=True,
                    valid=True,
                    validity_known=True,
                    tracking_state="valid",
                    status="ok",
                    frame_number=int(self._frame_counter),
                    translation_mm=value.translation_mm or (0.0, 0.0, 0.0),
                    quaternion_wxyz=value.quaternion_wxyz or (1.0, 0.0, 0.0, 0.0),
                )
                for key, value in (self._template.tools or {}).items()
            },
        )
        self._frame_counter += 1
        return snapshot


def _runner(
    tmp_path: Path,
    *,
    settings: Settings | None = None,
    tracking_service: Any | None = None,
    servo_service: ServoService | None = None,
) -> ExperimentRunner:
    settings = settings or _settings(mock_mode=True)
    return ExperimentRunner(
        project_root=Path(__file__).resolve().parents[1],
        settings=settings,
        tracking_service=tracking_service or _StubTrackingService(),
        servo_service=servo_service or _servo_service(tmp_path),
        output_dir=tmp_path / "data" / "experiments",
        default_settle_time_s=0.0,
        registration_path=tmp_path / "latest_registration.json",
        sleep_fn=lambda _seconds: None,
    )


# ----------------------------------------------------------------------------
# Registration tests
# ----------------------------------------------------------------------------


def test_register_dynamic_modeling_dataset_adds_descriptor() -> None:
    registry = ExperimentRegistry()
    register_dynamic_modeling_dataset(registry)
    descriptor = registry.get("dynamic_modeling_dataset")
    assert descriptor.name == "dynamic_modeling_dataset"
    assert descriptor.category == "dataset"
    assert "Dynamic" in descriptor.tags


def test_builtin_registration_includes_dynamic_modeling_dataset() -> None:
    registry = ExperimentRegistry()
    register_builtin_experiments(registry)
    names = {descriptor.name for descriptor in registry.list_descriptors()}
    assert "dynamic_modeling_dataset" in names


# ----------------------------------------------------------------------------
# Trajectory generator tests
# ----------------------------------------------------------------------------


def test_trajectory_generator_stays_within_bounds_for_many_steps() -> None:
    trajectory = BoundedRandomWalkTrajectory(
        servo_ids=(1, 2, 3, 4),
        max_tick_delta_from_start=50,
        max_tick_delta_hard_cap=600,
        max_step_ticks_per_update=5,
        trajectory_smoothing=0.4,
        random_seed=42,
    )
    for index in range(1000):
        step = trajectory.next_step(elapsed_s=float(index) * 0.1)
        for tick in step.tick_deltas:
            assert -50 <= int(tick) <= 50, f"step {index} produced delta {tick} outside soft cap"


def test_trajectory_generator_is_reproducible_with_same_seed() -> None:
    a = BoundedRandomWalkTrajectory(
        servo_ids=(1, 2, 3, 4),
        max_tick_delta_from_start=100,
        max_tick_delta_hard_cap=600,
        max_step_ticks_per_update=10,
        trajectory_smoothing=0.3,
        random_seed=7,
    )
    b = BoundedRandomWalkTrajectory(
        servo_ids=(1, 2, 3, 4),
        max_tick_delta_from_start=100,
        max_tick_delta_hard_cap=600,
        max_step_ticks_per_update=10,
        trajectory_smoothing=0.3,
        random_seed=7,
    )
    for index in range(64):
        step_a = a.next_step(elapsed_s=float(index) * 0.1)
        step_b = b.next_step(elapsed_s=float(index) * 0.1)
        assert step_a.tick_deltas == step_b.tick_deltas


def test_trajectory_generator_coast_to_zero_drives_back_to_neutral() -> None:
    trajectory = BoundedRandomWalkTrajectory(
        servo_ids=(1, 2),
        max_tick_delta_from_start=40,
        max_tick_delta_hard_cap=200,
        max_step_ticks_per_update=8,
        trajectory_smoothing=0.0,
        random_seed=3,
    )
    # Walk away from zero.
    for _ in range(20):
        trajectory.next_step(elapsed_s=0.0)
    # Coast back. With max_step=8 and bound=40, 8 coast iterations is enough.
    for _ in range(20):
        coast = trajectory.coast_to_zero()
    assert all(int(value) == 0 for value in coast.tick_deltas)


def test_trajectory_generator_step_size_respects_max_step_ticks_per_update() -> None:
    trajectory = BoundedRandomWalkTrajectory(
        servo_ids=(1, 2),
        max_tick_delta_from_start=200,
        max_tick_delta_hard_cap=600,
        max_step_ticks_per_update=3,
        trajectory_smoothing=0.0,
        random_seed=11,
    )
    previous = (0, 0)
    for index in range(500):
        step = trajectory.next_step(elapsed_s=float(index) * 0.1)
        for axis, current in enumerate(step.tick_deltas):
            assert abs(int(current) - int(previous[axis])) <= 3
        previous = step.tick_deltas


# ----------------------------------------------------------------------------
# Sync builder tests
# ----------------------------------------------------------------------------


def _make_command(command_id: int, monotonic_ns: int) -> CommandRecord:
    return CommandRecord(
        command_id=int(command_id),
        monotonic_ns=int(monotonic_ns),
        host_time_s=float(monotonic_ns) / 1e9,
        cable_deltas_cm=(0.0, 0.0, 0.0, 0.0),
        goal_ticks_by_servo={1: 100, 2: 200, 3: 300, 4: 400},
        delta_ticks_by_servo={1: 0, 2: 0, 3: 0, 4: 0},
    )


def _make_tracker(frame_index: int, monotonic_ns: int, *, valid: bool = True) -> TrackerRecord:
    return TrackerRecord(
        frame_index=int(frame_index),
        monotonic_ns=int(monotonic_ns),
        host_time_s=float(monotonic_ns) / 1e9,
        frame_number=int(frame_index),
        frame_time_s=None,
        tip_xyz_mm=(1.0, 2.0, 3.0),
        tip_quat_wxyz=(1.0, 0.0, 0.0, 0.0),
        valid=bool(valid),
        age_ms_at_observation=5.0 if valid else None,
    )


def _make_servo(sample_index: int, monotonic_ns: int, *, valid: bool = True) -> ServoTelemetryRecord:
    return ServoTelemetryRecord(
        sample_index=int(sample_index),
        monotonic_ns=int(monotonic_ns),
        host_time_s=float(monotonic_ns) / 1e9,
        positions_by_servo={1: 1000, 2: 1000, 3: 1000, 4: 1000} if valid else {1: None, 2: None, 3: None, 4: None},
        currents_by_servo={1: 80, 2: 80, 3: 80, 4: 80},
        voltages_by_servo={1: 12000, 2: 12000, 3: 12000, 4: 12000},
        temperatures_by_servo={1: 30, 2: 30, 3: 30, 4: 30},
        valid=bool(valid),
        age_ms_by_servo={1: 5.0, 2: 5.0, 3: 5.0, 4: 5.0},
    )


def test_sync_row_picks_latest_command_at_or_before_sample_time() -> None:
    commands = [_make_command(1, 100_000_000), _make_command(2, 200_000_000), _make_command(3, 300_000_000)]
    trackers = [_make_tracker(1, 250_000_000)]
    servos = [_make_servo(1, 240_000_000)]
    sample = build_synchronized_sample(
        sample_index=0,
        sample_monotonic_ns=260_000_000,
        sample_host_time_s=260_000_000 / 1e9,
        commands=commands,
        trackers=trackers,
        servos=servos,
        max_tracker_age_ms=100.0,
        max_servo_age_ms=200.0,
        require_tracker=True,
    )
    assert sample.command is commands[1]
    assert sample.tracker is trackers[0]
    assert sample.servo is servos[0]
    assert sample.sample_valid is True


def test_sync_row_flags_invalid_when_tracker_is_stale() -> None:
    commands = [_make_command(1, 100_000_000)]
    trackers = [_make_tracker(1, 100_000_000)]
    servos = [_make_servo(1, 250_000_000)]
    sample = build_synchronized_sample(
        sample_index=0,
        sample_monotonic_ns=260_000_000,
        sample_host_time_s=260_000_000 / 1e9,
        commands=commands,
        trackers=trackers,
        servos=servos,
        max_tracker_age_ms=50.0,
        max_servo_age_ms=200.0,
        require_tracker=True,
    )
    assert sample.sample_valid is False
    assert "tracker_stale" in (sample.failure_code or "")


def test_sync_row_flags_invalid_when_servo_is_stale() -> None:
    commands = [_make_command(1, 100_000_000)]
    trackers = [_make_tracker(1, 250_000_000)]
    servos = [_make_servo(1, 100_000_000)]
    sample = build_synchronized_sample(
        sample_index=0,
        sample_monotonic_ns=260_000_000,
        sample_host_time_s=260_000_000 / 1e9,
        commands=commands,
        trackers=trackers,
        servos=servos,
        max_tracker_age_ms=100.0,
        max_servo_age_ms=50.0,
        require_tracker=True,
    )
    assert sample.sample_valid is False
    assert "servo_stale" in (sample.failure_code or "")


def test_sync_row_passes_when_require_tracker_is_false_and_tracker_missing() -> None:
    commands = [_make_command(1, 100_000_000)]
    servos = [_make_servo(1, 250_000_000)]
    sample = build_synchronized_sample(
        sample_index=0,
        sample_monotonic_ns=260_000_000,
        sample_host_time_s=260_000_000 / 1e9,
        commands=commands,
        trackers=[],
        servos=servos,
        max_tracker_age_ms=100.0,
        max_servo_age_ms=200.0,
        require_tracker=False,
    )
    assert sample.sample_valid is True
    assert sample.tracker is None


# ----------------------------------------------------------------------------
# CSV.GZ writer tests
# ----------------------------------------------------------------------------


def _read_csv_gz(path: Path) -> tuple[list[str], list[list[str]]]:
    with gzip.open(path, "rt", encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle)
        rows = list(reader)
    return rows[0], rows[1:]


def test_write_sync_csv_gz_emits_expected_columns_and_row(tmp_path: Path) -> None:
    sample = SynchronizedSample(
        sample_index=0,
        host_time_s=1.0,
        monotonic_ns=1_000_000_000,
        command=_make_command(7, 1_000_000_000),
        tracker=_make_tracker(7, 1_000_000_000),
        servo=_make_servo(7, 1_000_000_000),
        command_to_tracker_dt_ms=0.0,
        servo_to_tracker_dt_ms=0.0,
        sample_valid=True,
        failure_code=None,
    )
    target = tmp_path / "dynamic_samples.csv.gz"
    write_sync_csv_gz(target, [sample], servo_id_slots=[1, 2, 3, 4])
    header, rows = _read_csv_gz(target)
    assert header == list(SYNC_COLUMNS)
    assert len(rows) == 1
    assert "true" in rows[0]  # sample_valid serialized as true
    assert "7" in rows[0]  # command_id


def test_write_commands_csv_gz_emits_expected_columns(tmp_path: Path) -> None:
    command = CommandRecord(
        command_id=3,
        monotonic_ns=2_000_000_000,
        host_time_s=2.0,
        cable_deltas_cm=(0.1, -0.1, -0.1, 0.1),
        goal_ticks_by_servo={1: 1000, 2: 1010, 3: 1020, 4: 1030},
        delta_ticks_by_servo={1: 1, 2: 2, 3: 3, 4: 4},
        safety_status="ok",
    )
    target = tmp_path / "commands.csv.gz"
    write_commands_csv_gz(target, [command], servo_id_slots=[1, 2, 3, 4])
    header, rows = _read_csv_gz(target)
    assert header == list(COMMAND_COLUMNS)
    assert rows[0][0] == "3"
    assert rows[0][-1] == "ok"


def test_write_tracker_csv_gz_emits_expected_columns(tmp_path: Path) -> None:
    tracker = _make_tracker(2, 1_500_000_000)
    target = tmp_path / "tracker_frames.csv.gz"
    write_tracker_csv_gz(target, [tracker])
    header, rows = _read_csv_gz(target)
    assert header == list(TRACKER_COLUMNS)
    assert rows[0][3] == "2"
    assert rows[0][-2] == "true"


def test_write_servo_csv_gz_emits_expected_columns(tmp_path: Path) -> None:
    servo = _make_servo(4, 1_500_000_000)
    target = tmp_path / "servo_telemetry.csv.gz"
    write_servo_csv_gz(target, [servo], servo_id_slots=[1, 2, 3, 4])
    header, rows = _read_csv_gz(target)
    assert header == list(SERVO_COLUMNS)
    assert rows[0][3] == "1000"
    assert rows[0][-2] == "true"


# ----------------------------------------------------------------------------
# Summary metric tests
# ----------------------------------------------------------------------------


def test_summarize_dynamic_run_reports_valid_ratio_and_tracker_rate() -> None:
    config = DynamicModelingDatasetConfig.from_dict({})
    sync_samples = [
        SynchronizedSample(
            sample_index=i,
            host_time_s=float(i) * 0.05,
            monotonic_ns=int(i * 50_000_000),
            command=_make_command(i, int(i * 50_000_000)),
            tracker=_make_tracker(i, int(i * 50_000_000)),
            servo=_make_servo(i, int(i * 50_000_000)),
            command_to_tracker_dt_ms=0.0,
            servo_to_tracker_dt_ms=1.0,
            sample_valid=(i % 5 != 0),
            failure_code=None if i % 5 != 0 else "servo_stale",
        )
        for i in range(20)
    ]
    commands = [_make_command(i, int(i * 50_000_000)) for i in range(20)]
    trackers = [_make_tracker(i, int(i * 50_000_000)) for i in range(20)]
    servos = [_make_servo(i, int(i * 50_000_000)) for i in range(20)]
    metrics = summarize_dynamic_run(
        config=config,
        elapsed_s=1.0,
        sync_samples=sync_samples,
        commands=commands,
        trackers=trackers,
        servo_telemetry=servos,
        failure_code=None,
        thesis_eligible=True,
        eligibility_reasons=[],
        transform_chain_summary={"selected_backend_name": "ndi"},
    )
    assert metrics["dynamic_sample_row_count"] == 20
    assert metrics["valid_sample_count"] == 16
    assert metrics["invalid_sample_count"] == 4
    assert metrics["valid_sample_ratio"] == pytest.approx(0.8)
    assert metrics["tracker_unique_frame_count"] == 20
    assert metrics["servo_telemetry_rate_hz"] == pytest.approx(20.0)
    assert metrics["thesis_eligible"] is True
    assert metrics["transform_chain_summary"]["selected_backend_name"] == "ndi"


# ----------------------------------------------------------------------------
# End-to-end experiment runner tests
# ----------------------------------------------------------------------------


def test_dynamic_modeling_dataset_dry_run_marks_not_thesis_evidence(tmp_path: Path) -> None:
    """Dry-run path must produce the bundle but be marked not_thesis_evidence."""
    runner = _runner(tmp_path)
    result = runner.run_experiment(
        "dynamic_modeling_dataset",
        config={
            "duration_s": 0.2,
            "target_sample_rate_hz": 20.0,
            "command_update_rate_hz": 10.0,
            "max_tick_delta_from_start": 20,
            "max_tick_delta_hard_cap": 200,
            "max_step_ticks_per_update": 5,
            "random_seed": 9,
            "servo_ids": [1, 2, 3, 4],
            "require_tracker": False,
            "return_to_start_at_end": False,
            "tracker_poll_interval_s": 0.01,
            "servo_telemetry_interval_s": 0.02,
            "max_tracker_age_ms": 200.0,
            "max_servo_age_ms": 200.0,
            "dry_run": True,
            "allow_lower_trust_runtime_tip": True,
        },
    )

    # The dataset bundle must be written even if not thesis-evidence.
    output_dir = result.paths.output_dir
    assert (output_dir / "dynamic_samples.csv.gz").exists()
    assert (output_dir / "commands.csv.gz").exists()
    assert (output_dir / "tracker_frames.csv.gz").exists()
    assert (output_dir / "servo_telemetry.csv.gz").exists()
    assert (output_dir / "summary.json").exists()
    assert (output_dir / "dynamic_modeling_summary.json").exists()
    assert (output_dir / "dynamic_modeling_summary.txt").exists()
    assert (output_dir / "config_snapshot.yaml").exists()
    assert (output_dir / "manifest.json").exists()
    summary_metrics = result.summary.experiment_metrics
    assert bool(summary_metrics.get("not_thesis_evidence")) is True


def test_dynamic_modeling_dataset_writes_manifest_with_columns(tmp_path: Path) -> None:
    runner = _runner(tmp_path)
    result = runner.run_experiment(
        "dynamic_modeling_dataset",
        config={
            "duration_s": 0.1,
            "target_sample_rate_hz": 20.0,
            "command_update_rate_hz": 10.0,
            "max_tick_delta_from_start": 10,
            "max_tick_delta_hard_cap": 100,
            "max_step_ticks_per_update": 2,
            "random_seed": 0,
            "servo_ids": [1, 2, 3, 4],
            "require_tracker": False,
            "return_to_start_at_end": False,
            "max_tracker_age_ms": 200.0,
            "max_servo_age_ms": 200.0,
            "tracker_poll_interval_s": 0.02,
            "servo_telemetry_interval_s": 0.02,
            "dry_run": True,
            "allow_lower_trust_runtime_tip": True,
        },
    )
    manifest_path = result.paths.output_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["experiment_name"] == "dynamic_modeling_dataset"
    assert manifest["outputs"]["sync_samples"] == "dynamic_samples.csv.gz"
    assert manifest["columns"]["sync_samples"] == list(SYNC_COLUMNS)
    assert manifest["columns"]["commands"] == list(COMMAND_COLUMNS)
    assert manifest["columns"]["tracker_frames"] == list(TRACKER_COLUMNS)
    assert manifest["columns"]["servo_telemetry"] == list(SERVO_COLUMNS)


def test_dynamic_modeling_dataset_dry_run_emits_synthetic_sample_markers(tmp_path: Path) -> None:
    runner = _runner(tmp_path)
    result = runner.run_experiment(
        "dynamic_modeling_dataset",
        config={
            "duration_s": 0.1,
            "target_sample_rate_hz": 20.0,
            "command_update_rate_hz": 10.0,
            "max_tick_delta_from_start": 10,
            "max_tick_delta_hard_cap": 100,
            "max_step_ticks_per_update": 2,
            "random_seed": 0,
            "servo_ids": [1, 2, 3, 4],
            "require_tracker": False,
            "return_to_start_at_end": False,
            "max_tracker_age_ms": 200.0,
            "max_servo_age_ms": 200.0,
            "tracker_poll_interval_s": 0.02,
            "servo_telemetry_interval_s": 0.02,
            "dry_run": True,
            "allow_lower_trust_runtime_tip": True,
        },
    )
    bundle = runner.load_dataset(result.paths.output_dir)
    assert any(
        sample.extra.get("record_kind") == "dynamic_modeling_sample"
        for sample in bundle.samples
    ), "dynamic_modeling_sample records should be present in the canonical samples bundle"
    assert any(
        "dry_run" in (sample.status_flags or [])
        for sample in bundle.samples
    )


# ---------------------------------------------------------------------------
# GUI integration: preflight branch + controller labels + visibility map
# ---------------------------------------------------------------------------


class TestGuiIntegration:
    """Verify the dynamic_modeling_dataset experiment is wired into every GUI
    surface, not just the page widget: preflight evaluator, controller labels
    (mode/config-summary/history-metric), MODE_EXPERIMENT_VISIBILITY, and the
    runtime tip policy alias. These regressions would silently break the GUI
    flow even though the underlying experiment module is fine."""

    def test_visible_in_single_segment_and_parallel_single_modes(self) -> None:
        from continuum_robot.gui.controllers.experiment_controller import (
            MODE_EXPERIMENT_VISIBILITY,
        )

        assert "dynamic_modeling_dataset" in MODE_EXPERIMENT_VISIBILITY["single_segment"]
        assert "dynamic_modeling_dataset" in MODE_EXPERIMENT_VISIBILITY["parallel_single"]

    def test_runtime_tip_policy_alias_resolves_to_modeling_dataset_workflow(self) -> None:
        from continuum_robot.tracking.runtime_tip_policy import (
            WORKFLOW_MODELING_DATASET,
            resolve_runtime_tip_workflow,
        )

        resolution = resolve_runtime_tip_workflow("dynamic_modeling_dataset")
        assert resolution.canonical_workflow == WORKFLOW_MODELING_DATASET

    def test_mode_label_branches_by_dry_run_and_trust_override(self) -> None:
        from continuum_robot.gui.controllers.experiment_controller import (
            ExperimentController,
        )

        assert ExperimentController._mode_label(
            "dynamic_modeling_dataset",
            {"dry_run": False, "allow_lower_trust_runtime_tip": False},
        ) == "live dynamic modeling dataset"
        assert ExperimentController._mode_label(
            "dynamic_modeling_dataset", {"dry_run": True}
        ) == "dry-run dynamic modeling"
        assert ExperimentController._mode_label(
            "dynamic_modeling_dataset",
            {"dry_run": False, "allow_lower_trust_runtime_tip": True},
        ) == "live dynamic modeling (lower-trust tip)"

    def test_config_summary_label_reports_duration_rate_and_caps(self) -> None:
        from continuum_robot.gui.controllers.experiment_controller import (
            ExperimentController,
        )

        label = ExperimentController._config_summary_label(
            "dynamic_modeling_dataset",
            {
                "duration_s": 300.0,
                "target_sample_rate_hz": 20.0,
                "command_update_rate_hz": 5.0,
                "max_tick_delta_from_start": 100,
                "max_step_ticks_per_update": 10,
            },
        )
        assert "dur 5.0min" in label
        assert "20Hz" in label
        assert "6000 rows" in label  # 300 s * 20 Hz
        assert "cmd 5Hz" in label
        assert "soft cap 100 ticks" in label
        assert "step ≤10 ticks" in label

    def test_history_metric_label_surfaces_rows_rate_validity(self) -> None:
        from continuum_robot.gui.controllers.experiment_controller import (
            ExperimentController,
        )

        label = ExperimentController._history_metric_label(
            experiment_name="dynamic_modeling_dataset",
            metrics={
                "dynamic_sample_count": 6000,
                "achieved_sample_rate_hz": 19.7,
                "valid_sample_ratio": 0.985,
            },
        )
        assert "rows=6000" in label
        assert "rate=19.7Hz" in label
        assert "valid=98%" in label

    def _preflight_kwargs(self, controller, tmp_path: Path) -> dict:
        return {
            "settings": controller.settings,
            "project_root": Path(__file__).resolve().parents[1],
            "tracking_snapshot": controller.tracking_service.get_snapshot(),
            "servo_calibration_summary": controller.servo_service.get_calibration_summary(),
            "servo_connected": False,
            "neutral_setpoints": {},
            "output_root": tmp_path,
            "planned_output_dir": tmp_path / "planned",
            "active_run_output_dir": None,
            "registration_path": tmp_path / "missing.json",
            "config_error": None,
        }

    def test_preflight_branch_fires_and_classifies_checks(self, tmp_path: Path) -> None:
        """With mock_mode=True (default in test settings) and dry_run=False the
        preflight branch must surface mock_mode as blocked plus the standard
        tick_cap + estimates checks. This regression-tests that the experiment
        no longer falls through to the catch-all 'Unsupported experiment'."""
        from tests.test_gui_controllers import _experiment_controller
        from continuum_robot.gui.experiment_preflight import (
            evaluate_preflight,
            PREFLIGHT_BLOCKED,
        )

        controller = _experiment_controller(tmp_path)
        payload = {
            "duration_s": 30.0,
            "target_sample_rate_hz": 20.0,
            "command_update_rate_hz": 5.0,
            "max_tick_delta_from_start": 50,
            "max_step_ticks_per_update": 5,
            "max_tick_delta_hard_cap": 500,
            "dry_run": False,
        }
        report = evaluate_preflight(
            experiment_name="dynamic_modeling_dataset",
            config_payload=payload,
            **self._preflight_kwargs(controller, tmp_path),
        )
        keys = {check.key for check in report.checks}
        # Falling through to the catch-all 'else' would surface only a single
        # "experiment" check; the branch must produce the real check ids.
        assert "tick_cap" in keys
        assert "estimates" in keys
        assert "servo_ids" in keys
        # mock_mode default for test settings is True → must block live runs.
        mock_check = next((c for c in report.checks if c.key == "mock_mode"), None)
        assert mock_check is not None
        assert mock_check.status == PREFLIGHT_BLOCKED

    def test_preflight_dry_run_warns_not_thesis_evidence(self, tmp_path: Path) -> None:
        from tests.test_gui_controllers import _experiment_controller
        from continuum_robot.gui.experiment_preflight import (
            evaluate_preflight,
            PREFLIGHT_WARNING,
        )

        controller = _experiment_controller(tmp_path)
        payload = {
            "duration_s": 30.0,
            "target_sample_rate_hz": 20.0,
            "command_update_rate_hz": 5.0,
            "max_tick_delta_from_start": 50,
            "max_step_ticks_per_update": 5,
            "dry_run": True,
        }
        report = evaluate_preflight(
            experiment_name="dynamic_modeling_dataset",
            config_payload=payload,
            **self._preflight_kwargs(controller, tmp_path),
        )
        mode_check = next((c for c in report.checks if c.key == "mode"), None)
        assert mode_check is not None
        assert mode_check.status == PREFLIGHT_WARNING
        assert "not_thesis_evidence" in mode_check.message

    def test_preflight_blocks_when_soft_cap_exceeds_hard_cap(self, tmp_path: Path) -> None:
        from tests.test_gui_controllers import _experiment_controller
        from continuum_robot.gui.experiment_preflight import (
            evaluate_preflight,
            PREFLIGHT_BLOCKED,
        )

        controller = _experiment_controller(tmp_path)
        payload = {
            "duration_s": 30.0,
            "target_sample_rate_hz": 20.0,
            "max_tick_delta_from_start": 800,  # > hard cap below
            "max_tick_delta_hard_cap": 500,
        }
        report = evaluate_preflight(
            experiment_name="dynamic_modeling_dataset",
            config_payload=payload,
            **self._preflight_kwargs(controller, tmp_path),
        )
        tick_check = next((c for c in report.checks if c.key == "tick_cap"), None)
        assert tick_check is not None
        assert tick_check.status == PREFLIGHT_BLOCKED
        assert "exceeds hard cap" in tick_check.message

    def test_preflight_long_run_emits_warning(self, tmp_path: Path) -> None:
        """A 30-min+ duration should surface the long-run validation warning."""
        from tests.test_gui_controllers import _experiment_controller
        from continuum_robot.gui.experiment_preflight import (
            evaluate_preflight,
            PREFLIGHT_WARNING,
        )

        controller = _experiment_controller(tmp_path)
        payload = {
            "duration_s": 60 * 60.0,  # 1 h
            "target_sample_rate_hz": 20.0,
            "max_tick_delta_from_start": 100,
            "max_tick_delta_hard_cap": 500,
            "max_step_ticks_per_update": 5,
        }
        report = evaluate_preflight(
            experiment_name="dynamic_modeling_dataset",
            config_payload=payload,
            **self._preflight_kwargs(controller, tmp_path),
        )
        long_check = next((c for c in report.checks if c.key == "long_run"), None)
        assert long_check is not None
        assert long_check.status == PREFLIGHT_WARNING

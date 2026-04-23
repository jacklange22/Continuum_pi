from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path

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
from continuum_robot.experiments.calibration_validation import (
    analyze_pivot_runs,
    analyze_registration_runs,
    list_pivot_validation_candidates,
    list_registration_validation_candidates,
)
from continuum_robot.experiments.dataset_io import ExperimentDatasetWriter
from continuum_robot.experiments.experiment_runner import ExperimentRunner
from continuum_robot.experiments.schemas import ExperimentMetadata, ExperimentSummary


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


def _runner(project_root: Path) -> ExperimentRunner:
    return ExperimentRunner(
        project_root=project_root,
        settings=_settings(),
        tracking_service=None,
        servo_service=None,
        output_dir=project_root / "data" / "experiments",
        registration_path=project_root / "data" / "registrations" / "latest_registration.json",
        sleep_fn=lambda _seconds: None,
    )


def _write_registration_record(
    project_root: Path,
    *,
    stem: str,
    timestamp_utc: str,
    fre_mm: float,
    translation_mm: list[float],
    rotation_deg: float = 0.0,
) -> Path:
    root = project_root / "data" / "registrations"
    root.mkdir(parents=True, exist_ok=True)
    angle = np.deg2rad(float(rotation_deg))
    transform = np.eye(4, dtype=float)
    transform[0:3, 0:3] = np.array(
        [
            [np.cos(angle), -np.sin(angle), 0.0],
            [np.sin(angle), np.cos(angle), 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=float,
    )
    transform[0:3, 3] = np.asarray(translation_mm, dtype=float)
    payload = {
        "timestamp_utc": timestamp_utc,
        "landmark_labels": ["L1", "L2", "L3", "L4"],
        "measurement_tool_id": "0B",
        "coil_tool_id": "0A",
        "fre_mm": float(fre_mm),
        "T_robot_aurora": transform.tolist(),
        "validation_metrics": {"overall_fre_mm": float(fre_mm)},
    }
    path = root / stem
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def _write_pivot_run(
    project_root: Path,
    *,
    name: str,
    tip_vector_local_mm: list[float],
    rmse_mm: float,
    sample_count_used: int = 24,
    sample_count_rejected: int = 2,
    experiment_name: str = "pivot_calibration",
) -> Path:
    writer = ExperimentDatasetWriter(project_root / "data" / "experiments")
    metadata = ExperimentMetadata(
        schema_version="1.0",
        experiment_name=experiment_name,
        run_id=name[-12:],
        timestamp_utc=datetime(2026, 1, 1, 0, 0, 0, tzinfo=timezone.utc).isoformat(),
        git_commit=None,
        backend_info={"mock_mode": True},
        registration_info={"exists": False},
        config_used={"tool_id": "0B"},
    )
    summary = ExperimentSummary(
        schema_version="1.0",
        experiment_name=experiment_name,
        run_id=name[-12:],
        success=True,
        status="success",
        sample_counts={"total": 0},
        dropped_frames=0,
        invalid_transforms=0,
        stage_pass_fail={"execute": "passed"},
        experiment_metrics={
            "tip_vector_local_mm": [float(value) for value in tip_vector_local_mm],
            "rmse_mm": float(rmse_mm),
            "sample_count_total": int(sample_count_used + sample_count_rejected),
            "sample_count_used": int(sample_count_used),
            "sample_count_rejected": int(sample_count_rejected),
            "pivot_input_tool_id": "0B",
        },
    )
    return writer.write_dataset(metadata, [], summary, output_dir_name=name).output_dir


def test_registration_validation_analysis_and_runner_outputs(tmp_path: Path) -> None:
    record_a = _write_registration_record(
        tmp_path,
        stem="registration_20260101T000000_000001Z.json",
        timestamp_utc="2026-01-01T00:00:00+00:00",
        fre_mm=0.41,
        translation_mm=[10.0, 2.0, -1.0],
        rotation_deg=0.0,
    )
    record_b = _write_registration_record(
        tmp_path,
        stem="registration_20260101T000100_000002Z.json",
        timestamp_utc="2026-01-01T00:01:00+00:00",
        fre_mm=0.47,
        translation_mm=[10.4, 1.8, -0.8],
        rotation_deg=1.0,
    )
    record_c = _write_registration_record(
        tmp_path,
        stem="registration_20260101T000200_000003Z.json",
        timestamp_utc="2026-01-01T00:02:00+00:00",
        fre_mm=0.52,
        translation_mm=[9.8, 2.3, -1.1],
        rotation_deg=-0.8,
    )

    metrics, warnings = analyze_registration_runs(
        [
            "data/registrations/registration_20260101T000000_000001Z.json",
            "data/registrations/registration_20260101T000100_000002Z.json",
            "data/registrations/registration_20260101T000200_000003Z.json",
            "data/registrations/missing.json",
        ],
        project_root=tmp_path,
    )

    assert metrics["valid_run_count"] == 3
    assert metrics["invalid_run_count"] == 1
    assert metrics["fre_summary_mm"]["mean"] == pytest.approx((0.41 + 0.47 + 0.52) / 3.0)
    assert metrics["translation_delta_to_consensus_summary_mm"]["max"] is not None
    assert metrics["robot_origin_spread_mm"]["rms_distance_mm"] is not None
    assert warnings and "missing.json" in warnings[0]

    runner = _runner(tmp_path)
    result = runner.run_experiment(
        "registration_validation",
        config={
            "run_paths": [
                str(record_a.relative_to(tmp_path)),
                str(record_b.relative_to(tmp_path)),
                str(record_c.relative_to(tmp_path)),
            ]
        },
    )

    assert result.success is True
    assert result.paths.output_dir.name.endswith("_registration_validation")
    assert (result.paths.output_dir / "metrics.csv").exists()
    assert (result.paths.output_dir / "registration_validation_summary.txt").exists()
    assert (result.paths.output_dir / "registration_fre_histogram.png").exists()
    assert (result.paths.output_dir / "registration_frame_origins.png").exists()
    assert (result.paths.output_dir / "registration_transform_spread.png").exists()

    candidates = list_registration_validation_candidates(tmp_path)
    assert [candidate.path for candidate in candidates][:2] == [
        str(record_c.relative_to(tmp_path)),
        str(record_b.relative_to(tmp_path)),
    ]


def test_pivot_validation_analysis_and_runner_outputs(tmp_path: Path) -> None:
    run_a = _write_pivot_run(
        tmp_path,
        name="20260101_000000_pivot_calibration",
        tip_vector_local_mm=[0.0, 0.0, 124.8],
        rmse_mm=0.31,
    )
    run_b = _write_pivot_run(
        tmp_path,
        name="20260101_000100_pivot_calibration",
        tip_vector_local_mm=[0.3, -0.2, 125.4],
        rmse_mm=0.28,
    )
    run_c = _write_pivot_run(
        tmp_path,
        name="20260101_000200_pivot_calibration",
        tip_vector_local_mm=[-0.4, 0.1, 125.1],
        rmse_mm=0.36,
    )
    _write_pivot_run(
        tmp_path,
        name="20260101_000300_not_pivot",
        tip_vector_local_mm=[0.0, 0.0, 125.0],
        rmse_mm=0.4,
        experiment_name="tracker_timing_validation",
    )

    metrics, warnings = analyze_pivot_runs(
        [
            str(run_a.relative_to(tmp_path)),
            str(run_b.relative_to(tmp_path)),
            str(run_c.relative_to(tmp_path)),
            "data/experiments/pivot_calibration/20260101_000300_not_pivot",
        ],
        project_root=tmp_path,
    )

    assert metrics["valid_run_count"] == 3
    assert metrics["invalid_run_count"] == 1
    assert metrics["tip_norm_summary_mm"]["mean"] == pytest.approx(
        np.mean([np.linalg.norm([0.0, 0.0, 124.8]), np.linalg.norm([0.3, -0.2, 125.4]), np.linalg.norm([-0.4, 0.1, 125.1])])
    )
    assert metrics["per_axis_summary_mm"]["z"]["mean"] == pytest.approx(np.mean([124.8, 125.4, 125.1]))
    assert warnings and "not_pivot" in warnings[0]

    runner = _runner(tmp_path)
    result = runner.run_experiment(
        "pivot_validation",
        config={"run_paths": [str(run_a.relative_to(tmp_path)), str(run_b.relative_to(tmp_path)), str(run_c.relative_to(tmp_path))]},
    )

    assert result.success is True
    assert result.paths.output_dir.name.endswith("_pivot_validation")
    assert (result.paths.output_dir / "metrics.csv").exists()
    assert (result.paths.output_dir / "pivot_validation_summary.txt").exists()
    assert (result.paths.output_dir / "pivot_tip_scatter.png").exists()
    assert (result.paths.output_dir / "pivot_axis_histograms.png").exists()
    assert (result.paths.output_dir / "pivot_quality_summary.png").exists()

    candidates = list_pivot_validation_candidates(tmp_path)
    assert candidates[0].path == str(run_c.relative_to(tmp_path))


def test_runner_registers_validation_analysis_experiments(tmp_path: Path) -> None:
    runner = _runner(tmp_path)
    names = [descriptor.name for descriptor in runner.available_experiments()]

    assert "registration_validation" in names
    assert "pivot_validation" in names

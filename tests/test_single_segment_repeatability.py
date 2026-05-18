from __future__ import annotations
import os
from pathlib import Path
import importlib.util
import json
import sys
import threading

import pytest

if sys.version_info < (3, 10):
    pytest.skip("Project tests require Python 3.10+ syntax.", allow_module_level=True)

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

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
from continuum_robot.experiments.framework import ExperimentContext, ExperimentSession
from continuum_robot.experiments.schemas import ExperimentMetadata
from continuum_robot.experiments.single_segment_repeatability import (
    LEGACY_CAPTURE_COUNT,
    LEGACY_TARGET_COUNT,
    LEGACY_VISIT_COUNT,
    SingleSegmentRepeatabilityConfig,
    SingleSegmentRepeatabilityExperiment,
    build_legacy_17_point_targets,
    compute_repeatability_baseline_comparison,
    compute_legacy_style_repeatability_analysis,
    compute_single_segment_repeatability_metrics,
    generate_legacy_revisit_sequence,
    load_repeatability_metrics_from_run,
    repeatability_ring_tick_defaults,
    repeatability_target_tick_profile,
    _read_servo_telemetry_payload,
)
from continuum_robot.experiments import single_segment_repeatability_outputs as repeatability_outputs
from continuum_robot.experiments.plotting import import_matplotlib
from continuum_robot.hardware.mock_dxl_bus import MockDxlBus
from continuum_robot.servos.displacement_mapper import TendonDisplacementMapper
from continuum_robot.servos.neutral_calibration_service import NeutralCalibrationService, ServoCalibrationContext
from continuum_robot.servos.pretension_validation_service import PretensionValidationService
from continuum_robot.servos.safety_guard import SafetyGuard
from continuum_robot.servos.servo_service import ServoService
from continuum_robot.services.models import ServiceHealthSnapshot, ToolTrackingSnapshot, TrackingSnapshot


def test_legacy_17_point_target_generation_is_exact() -> None:
    targets = build_legacy_17_point_targets()

    assert len(targets) == LEGACY_TARGET_COUNT
    assert targets[0].ring == "center"
    assert targets[0].cable_deltas_mm == [0.0, 0.0, 0.0, 0.0]
    assert [target.ring_radius_mm for target in targets[1:9]] == [4.0] * 8
    assert [target.ring_radius_mm for target in targets[9:17]] == [8.0] * 8
    assert targets[1].cable_deltas_mm == [-4.0, 0.0, 4.0, 0.0]
    assert targets[3].cable_deltas_mm == [0.0, -4.0, 0.0, 4.0]
    assert targets[9].cable_deltas_mm == [-8.0, 0.0, 8.0, 0.0]
    assert targets[16].angle_deg == 315.0


def test_repeatability_ring_tick_defaults_match_20mm_spool_math() -> None:
    mapper = TendonDisplacementMapper(spool_diameter_cm=2.0, ticks_per_rev=4096)
    ring_ticks = repeatability_ring_tick_defaults(
        mapper=mapper,
        inner_ring_radius_mm=4.0,
        outer_ring_radius_mm=8.0,
    )
    target_profile = repeatability_target_tick_profile(
        targets=build_legacy_17_point_targets(inner_ring_radius_mm=4.0, outer_ring_radius_mm=8.0),
        mapper=mapper,
    )

    assert ring_ticks["inner_ring_radius_ticks"] == 261
    assert ring_ticks["outer_ring_radius_ticks"] == 522
    assert target_profile["max_abs_tick_delta"] == 522


def test_legacy_revisit_sequence_visits_every_other_target_once_per_target() -> None:
    targets = build_legacy_17_point_targets()
    visits = generate_legacy_revisit_sequence(targets, seed=7)

    assert len(visits) == LEGACY_VISIT_COUNT
    for desired_index in range(LEGACY_TARGET_COUNT):
        rows = [visit for visit in visits if visit.target_index == desired_index]
        assert len(rows) == 16
        assert sorted(visit.approach_index for visit in rows) == [
            index for index in range(LEGACY_TARGET_COUNT) if index != desired_index
        ]
        assert all(visit.repeat_target.target_index == desired_index for visit in rows)
        assert all(visit.approach_target.target_index != desired_index for visit in rows)


def test_repeatability_thesis_02_per_target_rms_bar_writes_valid_png(tmp_path: Path) -> None:
    if importlib.util.find_spec("matplotlib") is None:
        pytest.skip("matplotlib is required for thesis figure rendering")
    metrics = {
        "thesis_goal_rms_mm": 1.0,
        "overall_repeatability_rms_mm": 0.45,
        "thesis_goal_pass": True,
        "per_target_metrics": {
            "0": {"target_index": 0, "label": "Center", "ring": "center", "spread_rms_mm": 0.20},
            "1": {"target_index": 1, "label": "Inner 0", "ring": "inner", "spread_rms_mm": 0.45},
            "9": {"target_index": 9, "label": "Outer 0", "ring": "outer", "spread_rms_mm": 0.62},
        },
    }
    output = tmp_path / "thesis_02_per_target_rms_bar.png"

    repeatability_outputs._write_repeatability_thesis_02_per_target_rms_bar(path=output, metrics=metrics)

    assert output.exists()
    assert output.read_bytes().startswith(b"\x89PNG")


def test_repeatability_metrics_use_repeat_captures_and_robot_frame() -> None:
    samples = []
    for sample_index, (target_index, position) in enumerate(
        [
            (1, [0.0, 0.0, 0.0]),
            (1, [2.0, 0.0, 0.0]),
            (2, [10.0, 0.0, 0.0]),
            (2, [10.0, 3.0, 4.0]),
        ]
    ):
        samples.append(
            _sample(
                phase="repeat",
                sample_index=sample_index,
                target_index=target_index,
                approach_index=0,
                position_mm=position,
            )
        )
    samples.append(
        _sample(
            phase="approach",
            sample_index=99,
            target_index=0,
            approach_index=1,
            position_mm=[100.0, 100.0, 100.0],
        )
    )

    metrics = compute_single_segment_repeatability_metrics(samples, tool_id="0A")

    assert metrics["valid_repeat_sample_count"] == 4
    assert metrics["valid_approach_sample_count"] == 1
    assert metrics["per_target_metrics"]["1"]["centroid_mm"] == [1.0, 0.0, 0.0]
    assert metrics["per_target_metrics"]["1"]["XY_RMSE_mm"] == pytest.approx(1.0)
    assert metrics["per_target_metrics"]["1"]["XYZ_RMSE_mm"] == pytest.approx(1.0)
    assert metrics["per_target_metrics"]["1"]["spread_rms_mm"] == pytest.approx(1.0)
    assert metrics["per_target_metrics"]["2"]["spread_rms_mm"] == pytest.approx(2.5)
    assert metrics["overall_repeatability_rms_mm"] == pytest.approx(((1.0**2 + 1.0**2 + 2.5**2 + 2.5**2) / 4) ** 0.5)
    assert metrics["legacy_style_comparison"]["per_target"]["2"]["XYZ_RMSE_mm"] == pytest.approx(2.5)


def test_legacy_style_repeatability_analysis_matches_legacy_xyz_rmse_definition() -> None:
    samples = [
        _sample(phase="approach", sample_index=0, target_index=1, approach_index=0, position_mm=[100.0, 100.0, 0.0]),
        _sample(phase="repeat", sample_index=1, target_index=1, approach_index=0, position_mm=[0.0, 0.0, 0.0]),
        _sample(phase="repeat", sample_index=2, target_index=1, approach_index=1, position_mm=[2.0, 0.0, 0.0]),
        _sample(phase="repeat", sample_index=3, target_index=1, approach_index=2, position_mm=[1.0, 0.0, 2.0]),
        _sample(phase="repeat", sample_index=4, target_index=1, approach_index=3, position_mm=[150.0, 0.0, 0.0]),
        _sample(
            phase="repeat",
            sample_index=5,
            target_index=1,
            approach_index=4,
            position_mm=[9.0, 9.0, 9.0],
            accepted=False,
        ),
    ]

    analysis = compute_legacy_style_repeatability_analysis(samples, tool_id="0A")

    row = analysis["per_target"]["1"]
    assert row["n_samples"] == 3
    assert row["centroid_mm"] == pytest.approx([1.0, 0.0, 2.0 / 3.0])
    assert row["XY_RMSE_mm"] == pytest.approx((2.0 / 3.0) ** 0.5)
    assert row["XYZ_RMSE_mm"] == pytest.approx((14.0 / 9.0) ** 0.5)
    assert row["max_deviation_mm"] == pytest.approx(4.0 / 3.0)
    assert analysis["sample_selection"]["selected_repeat_sample_count"] == 3
    assert analysis["sample_selection"]["rejected_repeat_sample_count"] == 2


def test_repeatability_metrics_mark_scientifically_weak_partial_run_invalid() -> None:
    metrics = compute_single_segment_repeatability_metrics(
        [
            _sample(
                phase="repeat",
                sample_index=index,
                target_index=1,
                approach_index=index + 1,
                position_mm=[float(index), 0.0, 0.0],
            )
            for index in range(4)
        ],
        tool_id="0A",
    )

    validity = metrics["run_validity"]

    assert validity["thesis_valid_run"] is False
    assert validity["observed"]["targets_below_min_count"] >= 1
    assert validity["observed"]["valid_repeat_sample_count"] == 4
    assert any("accepted repeat coverage" in reason for reason in validity["failure_reasons"])


def test_repeatability_baseline_comparison_reports_improvement_deltas() -> None:
    current = {
        "overall_repeatability_rms_mm": 0.8,
        "overall_max_deviation_mm": 1.6,
        "path_dependence_rms_mm": 0.7,
        "rejected_capture_count": 1,
        "per_target_metrics": {
            "0": {"target_index": 0, "label": "T00", "spread_rms_mm": 0.5, "max_deviation_mm": 1.0},
            "1": {"target_index": 1, "label": "T01", "spread_rms_mm": 0.9, "max_deviation_mm": 1.8},
        },
        "group_metrics": {
            "ring": {
                "inner": {"mean_target_rms_mm": 0.9},
                "outer": {"mean_target_rms_mm": 1.1},
            }
        },
    }
    baseline = {
        "overall_repeatability_rms_mm": 1.2,
        "overall_max_deviation_mm": 2.0,
        "path_dependence_rms_mm": 1.0,
        "rejected_capture_count": 3,
        "per_target_metrics": {
            "0": {"target_index": 0, "label": "T00", "spread_rms_mm": 0.8, "max_deviation_mm": 1.2},
            "1": {"target_index": 1, "label": "T01", "spread_rms_mm": 1.0, "max_deviation_mm": 2.1},
        },
        "group_metrics": {
            "ring": {
                "inner": {"mean_target_rms_mm": 1.2},
                "outer": {"mean_target_rms_mm": 1.5},
            }
        },
    }

    comparison = compute_repeatability_baseline_comparison(
        current_metrics=current,
        baseline_metrics=baseline,
        baseline_path="data/experiments/single_segment_repeatability/baseline",
    )

    assert comparison["available"] is True
    assert comparison["improved_overall_rms"] is True
    assert comparison["overall_rms"]["delta"] == pytest.approx(-0.4)
    assert comparison["overall_max_deviation"]["delta"] == pytest.approx(-0.4)
    assert comparison["path_dependence_rms"]["delta"] == pytest.approx(-0.3)
    assert comparison["rejected_capture_count"]["delta"] == -2
    assert comparison["per_target_rmse"]["0"]["delta_rmse_mm"] == pytest.approx(-0.3)
    assert comparison["group_metrics"]["ring"]["inner"]["delta"] == pytest.approx(-0.3)


def test_repeatability_baseline_loader_caches_summary_reads(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    run_dir = tmp_path / "data" / "experiments" / "single_segment_repeatability" / "baseline_run"
    run_dir.mkdir(parents=True, exist_ok=True)
    summary_path = run_dir / "summary.json"
    summary_path.write_text(
        json.dumps(
            {
                "experiment_name": "single_segment_repeatability",
                "status": "success",
                "success": True,
                "run_id": "baseline",
                "experiment_metrics": {
                    "protocol": "legacy_17_target_single_segment_all_other_approaches",
                    "overall_repeatability_rms_mm": 0.82,
                },
            }
        ),
        encoding="utf-8",
    )

    read_count = 0
    original_read_text = Path.read_text

    def _counted_read_text(self: Path, *args, **kwargs):
        nonlocal read_count
        if self == summary_path:
            read_count += 1
        return original_read_text(self, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", _counted_read_text)

    first = load_repeatability_metrics_from_run(run_dir)
    second = load_repeatability_metrics_from_run(run_dir)

    assert first["overall_repeatability_rms_mm"] == pytest.approx(0.82)
    assert second["overall_repeatability_rms_mm"] == pytest.approx(0.82)
    assert read_count == 1


def test_preflight_blocks_when_transform_chain_is_not_trusted(tmp_path: Path) -> None:
    pytest.importorskip("PySide6")
    from continuum_robot.gui.experiment_preflight import RUN_BLOCKED, evaluate_preflight

    settings = _settings(mock_mode=False)
    service = _servo_service(tmp_path)
    snapshot = _tracking_snapshot(runtime_tip_state="identity_tip_fallback", tip_pose_status="identity_tip_fallback")
    report = evaluate_preflight(
        experiment_name="single_segment_repeatability",
        config_payload={},
        config_error=None,
        settings=settings,
        tracking_snapshot=snapshot,
        servo_connected=True,
        neutral_setpoints={1: 2048, 2: 2048, 3: 2048, 4: 2048},
        registration_path=tmp_path / "latest_registration.json",
        output_root=tmp_path / "data" / "experiments",
        planned_output_dir=tmp_path / "data" / "experiments" / "single_segment_repeatability" / "run",
        project_root=tmp_path,
        servo_calibration_summary=service.neutral_calibration.get_calibration_summary(),
    )

    assert report.overall_status == RUN_BLOCKED
    assert any(check.key == "runtime_tip" and check.status == "blocked" for check in report.checks)


def test_preflight_blocks_repeatability_when_runtime_tip_mode_is_quick_override(tmp_path: Path) -> None:
    pytest.importorskip("PySide6")
    from continuum_robot.gui.experiment_preflight import RUN_BLOCKED, evaluate_preflight

    settings = _settings(mock_mode=False)
    service = _servo_service(tmp_path)
    snapshot = _tracking_snapshot(runtime_tip_state="quick_4_point_loaded", runtime_tip_mode="quick_4_point")
    report = evaluate_preflight(
        experiment_name="single_segment_repeatability",
        config_payload={},
        config_error=None,
        settings=settings,
        tracking_snapshot=snapshot,
        servo_connected=True,
        neutral_setpoints={1: 2048, 2: 2048, 3: 2048, 4: 2048},
        registration_path=tmp_path / "latest_registration.json",
        output_root=tmp_path / "data" / "experiments",
        planned_output_dir=tmp_path / "data" / "experiments" / "single_segment_repeatability" / "run",
        project_root=tmp_path,
        servo_calibration_summary=service.neutral_calibration.get_calibration_summary(),
    )

    assert report.overall_status == RUN_BLOCKED
    assert any("Mode=quick_4_point" in check.message for check in report.checks if check.key == "runtime_tip")


def test_preflight_allows_repeatability_coil_as_tip_as_trusted_policy(tmp_path: Path) -> None:
    pytest.importorskip("PySide6")
    from continuum_robot.gui.experiment_preflight import RUN_WARNING, evaluate_preflight

    settings = _settings(mock_mode=False)
    service = _servo_service(tmp_path)
    pivot_tip_file = tmp_path / "tools" / "penprobe_08_09_24c"
    pivot_tip_file.parent.mkdir(parents=True, exist_ok=True)
    pivot_tip_file.write_text("0,0,0\n", encoding="utf-8")
    snapshot = _tracking_snapshot(
        runtime_tip_state="coil_as_tip",
        tip_pose_status="coil_as_tip",
        runtime_tip_mode="coil_as_tip",
    )
    report = evaluate_preflight(
        experiment_name="single_segment_repeatability",
        config_payload={},
        config_error=None,
        settings=settings,
        tracking_snapshot=snapshot,
        servo_connected=True,
        neutral_setpoints={1: 2048, 2: 2048, 3: 2048, 4: 2048},
        registration_path=tmp_path / "latest_registration.json",
        output_root=tmp_path / "data" / "experiments",
        planned_output_dir=tmp_path / "data" / "experiments" / "single_segment_repeatability" / "run",
        project_root=tmp_path,
        servo_calibration_summary=service.neutral_calibration.get_calibration_summary(),
    )

    assert report.overall_status == RUN_WARNING
    assert any(check.key == "servo_sign_mapping" and check.status == "warning" for check in report.checks)
    assert any(
        check.key == "runtime_tip" and check.status == "ok"
        for check in report.checks
    )


def test_preflight_ignores_legacy_debug_flag_for_trusted_coil_as_tip(tmp_path: Path) -> None:
    pytest.importorskip("PySide6")
    from continuum_robot.gui.experiment_preflight import RUN_WARNING, evaluate_preflight

    settings = _settings(mock_mode=False)
    service = _servo_service(tmp_path)
    pivot_tip_file = tmp_path / "tools" / "penprobe_08_09_24c"
    pivot_tip_file.parent.mkdir(parents=True, exist_ok=True)
    pivot_tip_file.write_text("0,0,0\n", encoding="utf-8")
    snapshot = _tracking_snapshot(
        runtime_tip_state="coil_as_tip",
        tip_pose_status="coil_as_tip",
        runtime_tip_mode="coil_as_tip",
    )
    report = evaluate_preflight(
        experiment_name="single_segment_repeatability",
        config_payload={"allow_debug_coil_as_tip": True},
        config_error=None,
        settings=settings,
        tracking_snapshot=snapshot,
        servo_connected=True,
        neutral_setpoints={1: 2048, 2: 2048, 3: 2048, 4: 2048},
        registration_path=tmp_path / "latest_registration.json",
        output_root=tmp_path / "data" / "experiments",
        planned_output_dir=tmp_path / "data" / "experiments" / "single_segment_repeatability" / "run",
        project_root=tmp_path,
        servo_calibration_summary=service.neutral_calibration.get_calibration_summary(),
    )

    assert report.overall_status == RUN_WARNING
    assert any(check.key == "servo_sign_mapping" and check.status == "warning" for check in report.checks)
    assert any(
        check.key == "runtime_tip" and check.status == "ok"
        for check in report.checks
    )


def test_repeatability_preflight_and_precheck_agree_on_runtime_tip_workflow_resolution(tmp_path: Path) -> None:
    pytest.importorskip("PySide6")
    from continuum_robot.gui.experiment_preflight import RUN_BLOCKED, evaluate_preflight

    settings = _settings(mock_mode=False)
    service = _servo_service(tmp_path)
    pivot_tip_file = tmp_path / "tools" / "penprobe_08_09_24c"
    pivot_tip_file.parent.mkdir(parents=True, exist_ok=True)
    pivot_tip_file.write_text("0,0,0\n", encoding="utf-8")
    snapshot = _tracking_snapshot(
        runtime_tip_state="loaded",
        tip_pose_status="ok",
        runtime_tip_mode="latest_accepted",
    )

    report = evaluate_preflight(
        experiment_name="single_segment_repeatability",
        config_payload={},
        config_error=None,
        settings=settings,
        tracking_snapshot=snapshot,
        servo_connected=True,
        neutral_setpoints={1: 2048, 2: 2048, 3: 2048, 4: 2048},
        registration_path=tmp_path / "latest_registration.json",
        output_root=tmp_path / "data" / "experiments",
        planned_output_dir=tmp_path / "data" / "experiments" / "single_segment_repeatability" / "run",
        project_root=tmp_path,
        servo_calibration_summary=service.neutral_calibration.get_calibration_summary(),
    )

    assert report.overall_status == RUN_BLOCKED
    runtime_tip_check = next(check for check in report.checks if check.key == "runtime_tip")
    assert runtime_tip_check.status == "blocked"
    assert "Requested workflow/platform=single_segment_repeatability" in runtime_tip_check.message
    assert "resolved canonical workflow=repeatability" in runtime_tip_check.message

    experiment = SingleSegmentRepeatabilityExperiment(SingleSegmentRepeatabilityConfig())
    session = _session(tmp_path, service=service, snapshot=snapshot)
    experiment.setup(session)
    with pytest.raises(RuntimeError) as exc_info:
        experiment.precheck(session)
    message = str(exc_info.value)
    assert "requested workflow/platform=single_segment_repeatability" in message
    assert "resolved canonical workflow=repeatability" in message


def test_preflight_accepts_manual_pretension_source_for_repeatability(tmp_path: Path) -> None:
    pytest.importorskip("PySide6")
    from continuum_robot.gui.experiment_preflight import RUN_WARNING, evaluate_preflight

    settings = _settings(mock_mode=False)
    service = _servo_service(tmp_path, pretension_source="manual")
    pivot_tip_file = tmp_path / "tools" / "penprobe_08_09_24c"
    pivot_tip_file.parent.mkdir(parents=True, exist_ok=True)
    pivot_tip_file.write_text("0,0,0\n", encoding="utf-8")
    snapshot = _tracking_snapshot()
    report = evaluate_preflight(
        experiment_name="single_segment_repeatability",
        config_payload={},
        config_error=None,
        settings=settings,
        tracking_snapshot=snapshot,
        servo_connected=True,
        neutral_setpoints={1: 2048, 2: 2048, 3: 2048, 4: 2048},
        registration_path=tmp_path / "latest_registration.json",
        output_root=tmp_path / "data" / "experiments",
        planned_output_dir=tmp_path / "data" / "experiments" / "single_segment_repeatability" / "run",
        project_root=tmp_path,
        servo_calibration_summary=service.neutral_calibration.get_calibration_summary(),
    )

    assert report.overall_status == RUN_WARNING
    assert any(check.key == "servo_sign_mapping" and check.status == "warning" for check in report.checks)
    assert any(
        check.key == "pretension" and "manual pretension" in check.message.lower()
        for check in report.checks
    )


def test_preflight_blocks_repeatability_when_ring_amplitude_exceeds_experiment_cap(tmp_path: Path) -> None:
    pytest.importorskip("PySide6")
    from continuum_robot.gui.experiment_preflight import RUN_BLOCKED, evaluate_preflight

    settings = _settings(mock_mode=False)
    service = _servo_service(tmp_path)
    snapshot = _tracking_snapshot()
    report = evaluate_preflight(
        experiment_name="single_segment_repeatability",
        config_payload={
            "inner_ring_radius_mm": 4.0,
            "outer_ring_radius_mm": 8.0,
            "max_target_tick_delta_from_startup": 500,
        },
        config_error=None,
        settings=settings,
        tracking_snapshot=snapshot,
        servo_connected=True,
        neutral_setpoints={1: 2048, 2: 2048, 3: 2048, 4: 2048},
        registration_path=tmp_path / "latest_registration.json",
        output_root=tmp_path / "data" / "experiments",
        planned_output_dir=tmp_path / "data" / "experiments" / "single_segment_repeatability" / "run",
        project_root=tmp_path,
        servo_calibration_summary=service.neutral_calibration.get_calibration_summary(),
    )

    assert report.overall_status == RUN_BLOCKED
    assert any(
        check.key == "target_geometry" and "exceeds the configured experiment cap" in check.message
        for check in report.checks
    )


def test_repeatability_command_blocks_targets_beyond_experiment_tick_cap(tmp_path: Path) -> None:
    settings = _settings(mock_mode=False)
    service = _servo_service(tmp_path)
    experiment = SingleSegmentRepeatabilityExperiment(
        SingleSegmentRepeatabilityConfig(max_target_tick_delta_from_startup=100)
    )
    metadata = ExperimentMetadata(
        schema_version="1.0",
        experiment_name=experiment.name,
        run_id="cap-test",
        timestamp_utc="2026-04-24T00:00:00Z",
        git_commit=None,
        backend_info={},
        registration_info={},
        config_used=experiment.config_dict(),
    )
    session = ExperimentSession(
        context=ExperimentContext(
            project_root=tmp_path,
            settings=settings,
            tracking_service=_TrackingService(_tracking_snapshot()),
            servo_service=service,
            registration_path=tmp_path / "data" / "registrations" / "latest_registration.json",
            output_root=tmp_path / "data" / "experiments",
            sleep_fn=lambda _seconds: None,
        ),
        metadata=metadata,
    )
    experiment.setup(session)

    with pytest.raises(RuntimeError, match="experiment-safe target envelope"):
        experiment._command_target(session, experiment._targets[9])


def test_repeatability_precheck_allows_coil_as_tip_without_debug_override(tmp_path: Path) -> None:
    service = _servo_service(tmp_path)
    pivot_tip_file = tmp_path / "tools" / "penprobe_08_09_24c"
    pivot_tip_file.parent.mkdir(parents=True, exist_ok=True)
    pivot_tip_file.write_text("0,0,0\n", encoding="utf-8")
    experiment = SingleSegmentRepeatabilityExperiment(SingleSegmentRepeatabilityConfig())
    session = _session(
        tmp_path,
        service=service,
        snapshot=_tracking_snapshot(
            runtime_tip_state="coil_as_tip",
            tip_pose_status="coil_as_tip",
            runtime_tip_mode="coil_as_tip",
        ),
    )
    experiment.setup(session)

    experiment.precheck(session)


def test_repeatability_precheck_records_coil_as_tip_as_thesis_trusted(tmp_path: Path) -> None:
    service = _servo_service(tmp_path)
    pivot_tip_file = tmp_path / "tools" / "penprobe_08_09_24c"
    pivot_tip_file.parent.mkdir(parents=True, exist_ok=True)
    pivot_tip_file.write_text("0,0,0\n", encoding="utf-8")
    experiment = SingleSegmentRepeatabilityExperiment(SingleSegmentRepeatabilityConfig())
    session = _session(
        tmp_path,
        service=service,
        snapshot=_tracking_snapshot(
            runtime_tip_state="coil_as_tip",
            tip_pose_status="coil_as_tip",
            runtime_tip_mode="coil_as_tip",
        ),
    )
    experiment.setup(session)

    experiment.precheck(session)

    provenance = session.metrics["run_provenance"]
    runtime_tip = provenance["runtime_tip_calibration"]
    assert runtime_tip["mode"] == "coil_as_tip"
    assert runtime_tip["trust_level"] == "thesis_trusted"
    assert runtime_tip["artifact_used"] is False
    assert runtime_tip["identity_marker"] == "T_coil_tip_identity__coil_pose_used_as_tip"
    assert runtime_tip["policy"]["uses_coil_as_tip"] is True
    assert provenance["thesis_trusted_runtime_tip"] is True


def test_repeatability_precheck_motion_assessment_uses_experiment_owned_telemetry(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    service = _servo_service(tmp_path)
    pivot_tip_file = tmp_path / "tools" / "penprobe_08_09_24c"
    pivot_tip_file.parent.mkdir(parents=True, exist_ok=True)
    pivot_tip_file.write_text("0,0,0\n", encoding="utf-8")
    experiment = SingleSegmentRepeatabilityExperiment(SingleSegmentRepeatabilityConfig())
    session = _session(
        tmp_path,
        service=service,
        snapshot=_tracking_snapshot(
            runtime_tip_state="coil_as_tip",
            tip_pose_status="coil_as_tip",
            runtime_tip_mode="coil_as_tip",
        ),
    )
    experiment.setup(session)

    sources: list[str | None] = []
    orig = ServoService.assess_experiment_motion

    def _wrap(self, servo_id: int, *, telemetry=None):
        sources.append(getattr(telemetry, "read_source", None) if telemetry is not None else None)
        return orig(self, int(servo_id), telemetry=telemetry)

    monkeypatch.setattr(ServoService, "assess_experiment_motion", _wrap)
    experiment.precheck(session)

    motion_sources = [value for value in sources if value is not None]
    assert motion_sources
    assert all(value == "experiment_owned" for value in motion_sources)


def test_experiment_registration_and_custom_page_routing(tmp_path: Path) -> None:
    pytest.importorskip("PySide6")
    from continuum_robot.experiments.experiment_runner import ExperimentRunner
    from PySide6.QtWidgets import QApplication

    from continuum_robot.gui.widgets.experiment_pages import SingleSegmentRepeatabilityPage, build_experiment_page

    app = QApplication.instance() or QApplication([])
    _ = app
    runner = ExperimentRunner(
        project_root=tmp_path,
        settings=_settings(mock_mode=False),
        tracking_service=_TrackingService(_tracking_snapshot()),
        servo_service=_servo_service(tmp_path),
        output_dir=tmp_path / "data" / "experiments",
        registration_path=tmp_path / "data" / "registrations" / "latest_registration.json",
    )
    names = [descriptor.name for descriptor in runner.available_experiments()]
    assert "single_segment_repeatability" in names
    visibility = {descriptor.name: descriptor.workspace_visible for descriptor in runner.available_experiments()}
    assert visibility["single_segment_repeatability"] is True
    assert visibility["repeatability_dataset"] is False

    page = build_experiment_page(_DummyController(tmp_path), "single_segment_repeatability")
    assert isinstance(page, SingleSegmentRepeatabilityPage)
    page.deleteLater()


def test_live_repeatability_run_writes_canonical_outputs(tmp_path: Path) -> None:
    if importlib.util.find_spec("PySide6") is None:
        pytest.skip("PySide6 is required for saved repeatability figure generation.")
    from continuum_robot.experiments.experiment_runner import ExperimentRunner

    settings = _settings(mock_mode=False)
    service = _servo_service(tmp_path)
    registration_path = tmp_path / "data" / "registrations" / "latest_registration.json"
    registration_path.parent.mkdir(parents=True, exist_ok=True)
    registration_path.write_text("{}", encoding="utf-8")
    runtime_tip_path = tmp_path / "data" / "runtime_tip_calibration" / "latest_runtime_tip_calibration.json"
    runtime_tip_path.parent.mkdir(parents=True, exist_ok=True)
    runtime_tip_path.write_text("{}", encoding="utf-8")
    snapshot = _tracking_snapshot(position_mm=[1.0, 2.0, 3.0])
    snapshot.registration_path = str(registration_path)
    snapshot.stored_registration_timestamp_utc = "2026-04-15T00:00:00Z"
    snapshot.stored_registration_fre_mm = 0.4
    snapshot.runtime_tip_calibration_path = None
    snapshot.stored_runtime_tip_timestamp_utc = "2026-04-15T00:05:00Z"
    tracking = _TrackingService(snapshot)
    pivot_tip_file = tmp_path / "tools" / "penprobe_08_09_24c"
    pivot_tip_file.parent.mkdir(parents=True, exist_ok=True)
    pivot_tip_file.write_text("0,0,0\n", encoding="utf-8")
    runner = ExperimentRunner(
        project_root=tmp_path,
        settings=settings,
        tracking_service=tracking,
        servo_service=service,
        output_dir=tmp_path / "data" / "experiments",
        registration_path=registration_path,
        sleep_fn=lambda _seconds: None,
    )

    result = runner.run_experiment(
        "single_segment_repeatability",
        config={
            "settle_time_s": 0.0,
            "capture_timeout_s": 0.0,
            "random_seed": 3,
        },
        output_dir=tmp_path / "data" / "experiments",
    )

    assert result.success
    assert result.sample_count == LEGACY_CAPTURE_COUNT
    assert result.paths.output_dir.parent.name == "single_segment_repeatability"
    assert (result.paths.output_dir / "debug.json").exists()
    assert (result.paths.output_dir / "thesis_01_target_returns_3d.png").exists()
    assert (result.paths.output_dir / "thesis_02_per_target_rms_bar.png").exists()
    assert (result.paths.output_dir / "thesis_03_path_dependence_vs_total.png").exists()
    for removed in [
        "repeatability_summary.txt",
        "repeatability_clusters_report.png",
        "repeatability_error_by_target_report.png",
        "repeatability_group_summary_report.png",
        "repeatability_clusters.png",
        "tip_pos_clusters.png",
        "repeatability_legacy_style_comparison.csv",
        "repeatability_rmse_summary.png",
        "repeatability_path_dependence.png",
        "repeatability_debug_samples.csv",
        "repeatability_commanded_vs_measured_tip.png",
        "repeatability_acceptance_timeline.png",
        "repeatability_servo_goal_ticks_vs_tip.png",
        "repeatability_provenance_summary.png",
    ]:
        assert not (result.paths.output_dir / removed).exists(), f"deprecated repeatability artifact should be gone: {removed}"
    assert result.summary.experiment_metrics["valid_repeat_sample_count"] == LEGACY_VISIT_COUNT
    assert result.summary.status == "success"
    assert result.summary.experiment_metrics["run_validity"]["thesis_valid_run"] is True
    accepted_samples = [sample for sample in result.paths.samples_path.read_text(encoding="utf-8").splitlines() if sample.strip()]
    assert accepted_samples
    metadata_payload = json.loads(result.paths.metadata_path.read_text(encoding="utf-8"))
    summary_payload = json.loads(result.paths.summary_path.read_text(encoding="utf-8"))
    assert metadata_payload["registration_info"]["pivot_tip"]["path"].endswith("penprobe_08_09_24c")
    assert metadata_payload["registration_info"]["base_registration"]["path"] == str(registration_path)
    assert metadata_payload["registration_info"]["runtime_tip_calibration"]["path"] is None
    assert metadata_payload["registration_info"]["runtime_tip_mode"] == "coil_as_tip"
    assert metadata_payload["registration_info"]["runtime_tip_policy"]["trust_label"] == "thesis_trusted"
    assert metadata_payload["backend_info"]["pretension_source"]["source_type"] == "algorithmic"
    assert summary_payload["experiment_metrics"]["run_provenance"]["pretension_artifact"]["path"].endswith("neutral.json")
    assert summary_payload["experiment_metrics"]["run_provenance"]["pretension_artifact"]["active_source_type"] == "algorithmic"
    assert summary_payload["experiment_metrics"]["run_provenance"]["runtime_tip_calibration"]["mode"] == "coil_as_tip"
    assert summary_payload["experiment_metrics"]["runtime_tip_trust_level"] == "thesis_trusted"
    assert summary_payload["experiment_metrics"]["thesis_trusted_run"] is True
    assert summary_payload["experiment_metrics"]["legacy_style_comparison"]["analysis_mode"] == "legacy_style"
    assert summary_payload["experiment_metrics"]["transform_chain_audit"]["pose_frame_used_for_analysis"] == "robot"
    assert summary_payload["experiment_metrics"]["run_provenance"]["startup_reference_source"] == "algorithmic"
    assert summary_payload["experiment_metrics"]["run_provenance"]["precheck_trust_summary"]["overall_status"] == "ready"
    assert "tracker_bridge" not in (result.paths.output_dir / "config_snapshot.yaml").read_text(encoding="utf-8")
    samples_payload = [
        json.loads(line)
        for line in result.paths.samples_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    accepted_capture = next(sample for sample in samples_payload if sample.get("extra", {}).get("capture_accepted"))
    assert accepted_capture["extra"]["servo_motion_profile"]["operating_mode_label"] == "Position Control"
    assert accepted_capture["extra"]["servo_motion_profile"]["goal_current_ma"] is None
    assert accepted_capture["extra"]["command_target_id"]
    assert accepted_capture["extra"]["resolved_servo_goal_ticks"]
    assert accepted_capture["extra"]["current_servo_positions_before_command"]
    assert accepted_capture["extra"]["current_servo_positions_after_command"]
    assert accepted_capture["extra"]["raw_tracker_tool_pose"]["translation_mm"] == [1.0, 2.0, 3.0]
    assert accepted_capture["extra"]["robot_frame_tip_pose_used"]["translation_mm"] == [1.0, 2.0, 3.0]
    assert accepted_capture["extra"]["accept_reject_reason"] == "ok"
    # legacy_style_comparison content moved from CSV into debug.json.
    debug_payload = json.loads((result.paths.output_dir / "debug.json").read_text(encoding="utf-8"))
    assert debug_payload["experiment_name"] == "single_segment_repeatability"
    assert debug_payload["legacy_style_comparison"]["analysis_mode"] == "legacy_style"
    assert "per_target" in debug_payload["legacy_style_comparison"]
    assert "thesis_goal_pass" in debug_payload
    assert "per_target_metrics" in debug_payload
    assert "path_dependence_by_approach" in debug_payload


def test_coil_as_tip_repeatability_run_is_labeled_thesis_trusted(tmp_path: Path) -> None:
    if importlib.util.find_spec("PySide6") is None:
        pytest.skip("PySide6 is required for saved repeatability figure generation.")
    from continuum_robot.experiments.experiment_runner import ExperimentRunner

    settings = _settings(mock_mode=False)
    service = _servo_service(tmp_path)
    registration_path = tmp_path / "data" / "registrations" / "latest_registration.json"
    registration_path.parent.mkdir(parents=True, exist_ok=True)
    registration_path.write_text("{}", encoding="utf-8")
    snapshot = _tracking_snapshot(
        position_mm=[4.0, 5.0, 6.0],
        runtime_tip_state="coil_as_tip",
        tip_pose_status="coil_as_tip",
        runtime_tip_mode="coil_as_tip",
    )
    snapshot.registration_path = str(registration_path)
    tracking = _TrackingService(snapshot)
    pivot_tip_file = tmp_path / "tools" / "penprobe_08_09_24c"
    pivot_tip_file.parent.mkdir(parents=True, exist_ok=True)
    pivot_tip_file.write_text("0,0,0\n", encoding="utf-8")
    runner = ExperimentRunner(
        project_root=tmp_path,
        settings=settings,
        tracking_service=tracking,
        servo_service=service,
        output_dir=tmp_path / "data" / "experiments",
        registration_path=registration_path,
        sleep_fn=lambda _seconds: None,
    )

    result = runner.run_experiment(
        "single_segment_repeatability",
        config={
            "settle_time_s": 0.0,
            "capture_timeout_s": 0.0,
            "allow_debug_coil_as_tip": True,
        },
        output_dir=tmp_path / "data" / "experiments",
    )

    metrics = result.summary.experiment_metrics
    runtime_tip = metrics["run_provenance"]["runtime_tip_calibration"]
    assert runtime_tip["mode"] == "coil_as_tip"
    assert runtime_tip["trust_level"] == "thesis_trusted"
    assert runtime_tip["artifact_used"] is False
    assert runtime_tip["identity_marker"] == "T_coil_tip_identity__coil_pose_used_as_tip"
    assert runtime_tip["policy"]["thesis_trusted"] is True
    assert metrics["thesis_trusted_run"] is True
    assert metrics["run_validity"]["thesis_valid_run"] is True
    assert result.summary.status == "success"
    assert result.success is True
    sample_payload = json.loads(result.paths.samples_path.read_text(encoding="utf-8").splitlines()[0])
    assert sample_payload["extra"]["runtime_tip_mode"] == "coil_as_tip"
    assert sample_payload["extra"]["runtime_tip_trust_level"] == "thesis_trusted"


def test_capture_gate_rejects_stale_tracker_data(tmp_path: Path) -> None:
    settings = _settings(mock_mode=False)
    service = _servo_service(tmp_path)
    stale_snapshot = _tracking_snapshot(stale=True, age_s=2.0)
    experiment = SingleSegmentRepeatabilityExperiment(
        SingleSegmentRepeatabilityConfig(
            settle_time_s=0.0,
            capture_timeout_s=0.0,
            max_tracker_age_s=0.25,
        )
    )
    metadata = ExperimentMetadata(
        schema_version="1.0",
        experiment_name=experiment.name,
        run_id="test",
        timestamp_utc="2026-04-15T00:00:00Z",
        git_commit=None,
        backend_info={},
        registration_info={},
        config_used=experiment.config_dict(),
    )
    session = ExperimentSession(
        context=ExperimentContext(
            project_root=tmp_path,
            settings=settings,
            tracking_service=_TrackingService(stale_snapshot),
            servo_service=service,
            registration_path=tmp_path / "data" / "registrations" / "latest_registration.json",
            output_root=tmp_path / "data" / "experiments",
            sleep_fn=lambda _seconds: None,
        ),
        metadata=metadata,
    )
    experiment.setup(session)
    sample = experiment._capture_after_move(
        session,
        visit=generate_legacy_revisit_sequence(seed=0)[0],
        target=build_legacy_17_point_targets()[0],
        phase="repeat",
        sample_index=0,
        command_payload={},
    )

    assert sample.extra["capture_accepted"] is False
    assert "tracker_data_stale" in sample.extra["capture_reject_reason"]
    assert "capture_rejected" in sample.status_flags


def test_repeatability_telemetry_payload_classifies_missing_current_and_voltage(tmp_path: Path) -> None:
    service = _servo_service(tmp_path)
    service.dxl_bus._state[2].present_current_ma = None
    service.dxl_bus._state[2].present_current_raw_unit = None
    service.dxl_bus._state[3].present_voltage_mv = None
    service.dxl_bus._state[3].present_voltage_raw_unit = None
    session = _session(tmp_path, service=service)

    payload = _read_servo_telemetry_payload(session, [1, 2, 3, 4], stage="runtime_capture")

    assert payload["ok"] is False
    assert payload["failure_stage"] == "runtime_capture"
    issue_codes = {(int(row.get("servo_id", -1)), str(row.get("code", ""))) for row in payload["issues"]}
    assert (2, "missing_current") in issue_codes
    assert (3, "missing_voltage") in issue_codes


def test_repeatability_telemetry_payload_classifies_owner_conflict(tmp_path: Path) -> None:
    service = _servo_service(tmp_path)
    session = _session(tmp_path, service=service)
    ready = threading.Event()
    release = threading.Event()

    def _hold_bus() -> None:
        with service.exclusive_bus_operation(owner="external_workflow", reason="test_hold"):
            ready.set()
            release.wait(timeout=1.0)

    holder = threading.Thread(target=_hold_bus, daemon=True)
    holder.start()
    assert ready.wait(timeout=1.0)
    try:
        payload = _read_servo_telemetry_payload(session, [1, 2, 3, 4], stage="preflight")
    finally:
        release.set()
        holder.join(timeout=1.0)

    assert payload["ok"] is False
    assert payload["failure_stage"] == "preflight"
    assert payload["failure_code"] in {"owner_conflict", "bus_contention"}


def test_repeatability_precheck_reports_preflight_telemetry_invalid_with_explicit_code(tmp_path: Path) -> None:
    settings = _settings(mock_mode=False)
    service = _servo_service(tmp_path)
    service.dxl_bus._state[1].present_current_ma = None
    service.dxl_bus._state[1].present_current_raw_unit = None
    pivot_tip_file = tmp_path / "tools" / "penprobe_08_09_24c"
    pivot_tip_file.parent.mkdir(parents=True, exist_ok=True)
    pivot_tip_file.write_text("0,0,0\n", encoding="utf-8")
    experiment = SingleSegmentRepeatabilityExperiment(SingleSegmentRepeatabilityConfig())
    metadata = ExperimentMetadata(
        schema_version="1.0",
        experiment_name=experiment.name,
        run_id="precheck-telemetry",
        timestamp_utc="2026-04-25T00:00:00Z",
        git_commit=None,
        backend_info={},
        registration_info={},
        config_used=experiment.config_dict(),
    )
    session = ExperimentSession(
        context=ExperimentContext(
            project_root=tmp_path,
            settings=settings,
            tracking_service=_TrackingService(_tracking_snapshot()),
            servo_service=service,
            registration_path=tmp_path / "data" / "registrations" / "latest_registration.json",
            output_root=tmp_path / "data" / "experiments",
            sleep_fn=lambda _seconds: None,
        ),
        metadata=metadata,
    )
    experiment.setup(session)

    with pytest.raises(RuntimeError, match="preflight_telemetry_invalid: code=missing_current"):
        experiment.precheck(session)


def test_authoritative_repeatability_docs_and_examples_reference_single_segment() -> None:
    project_root = Path(__file__).resolve().parents[1]

    for relative_path in [
        "docs/operator_workflows.md",
        "docs/system_spec.md",
        "docs/validation_plan.md",
        "docs/testing_protocol.md",
        "config/experiment_replay_runner.example.yaml",
    ]:
        text = (project_root / relative_path).read_text(encoding="utf-8")
        assert "single_segment_repeatability" in text
        assert "repeatability_dataset" not in text


def _settings(*, mock_mode: bool) -> Settings:
    return Settings(
        runtime=RuntimeConfig(mock_mode=mock_mode),
        robot=RobotConfig(
            mode="4-servo",
            spool_diameter_cm=2.0,
            ticks_per_revolution=4096,
            servo_ids=[1, 2, 3, 4],
            tendon_to_servo=[1, 2, 3, 4],
        ),
        serial=SerialConfig(tracker_backend="ndi"),
        safety=SafetyConfig(position_min_offset_ticks=-2048, position_max_offset_ticks=2047),
        registration=RegistrationWorkflowConfig(),
        experiment=ExperimentConfig(output_dir="data/experiments"),
        calibration=CalibrationConfig(
            neutral_setpoints_path="config/neutral_setpoints.json",
            latest_registration_path="data/registrations/latest_registration.json",
            latest_runtime_tip_calibration_path="data/runtime_tip_calibration/latest_runtime_tip_calibration.json",
        ),
    )


def _servo_service(tmp_path: Path, *, pretension_source: str = "algorithmic") -> ServoService:
    context = ServoCalibrationContext(
        robot_mode="4-servo",
        servo_ids=[1, 2, 3, 4],
        tendon_to_servo=[1, 2, 3, 4],
        ticks_per_revolution=4096,
        spool_diameter_cm=2.0,
        position_min_offset_ticks=-2048,
        position_max_offset_ticks=2047,
        default_pretension_current_threshold_ma=220,
    )
    service = ServoService(
        dxl_bus=MockDxlBus([1, 2, 3, 4]),
        mapper=TendonDisplacementMapper(spool_diameter_cm=2.0),
        safety_guard=SafetyGuard(min_offset_ticks=-2048, max_offset_ticks=2047, max_current_ma=850),
        neutral_calibration=NeutralCalibrationService(path=tmp_path / "neutral.json", context=context),
        pretension_validation=PretensionValidationService(),
    )
    service.connect("/dev/mock-openrb", 57600)
    for servo_id in [1, 2, 3, 4]:
        service.set_servo_torque_enabled(servo_id, True)
        service.neutral_calibration.save_servo_calibration(
            servo_id=servo_id,
            neutral_setpoint=2048,
            safe_min_tick=0,
            safe_max_tick=4095,
            pretension_current_threshold_ma=220,
            status="startup_calibrated",
            valid=True,
        )
        service.neutral_calibration.save_pretension_result(
            servo_id=servo_id,
            final_position_tick=2020,
            final_current_ma=230,
            threshold_ma=220,
            result_status="completed",
            pretension_source=pretension_source,
        )
        service.neutral_calibration.mark_pretension_accepted(servo_id)
    return service


def _session(
    tmp_path: Path,
    *,
    service: ServoService,
    snapshot: TrackingSnapshot | None = None,
    experiment_name: str = "single_segment_repeatability",
) -> ExperimentSession:
    metadata = ExperimentMetadata(
        schema_version="1.0",
        experiment_name=experiment_name,
        run_id="telemetry-test",
        timestamp_utc="2026-04-25T00:00:00Z",
        git_commit=None,
        backend_info={},
        registration_info={},
        config_used={},
    )
    return ExperimentSession(
        context=ExperimentContext(
            project_root=tmp_path,
            settings=_settings(mock_mode=False),
            tracking_service=_TrackingService(snapshot or _tracking_snapshot()),
            servo_service=service,
            registration_path=tmp_path / "data" / "registrations" / "latest_registration.json",
            output_root=tmp_path / "data" / "experiments",
            sleep_fn=lambda _seconds: None,
        ),
        metadata=metadata,
    )


def _tracking_snapshot(
    *,
    position_mm: list[float] | None = None,
    stale: bool = False,
    age_s: float = 0.01,
    runtime_tip_state: str = "coil_as_tip",
    tip_pose_status: str = "coil_as_tip",
    runtime_tip_mode: str = "coil_as_tip",
) -> TrackingSnapshot:
    position = position_mm or [0.0, 0.0, 50.0]
    matrix = [
        [1.0, 0.0, 0.0, float(position[0])],
        [0.0, 1.0, 0.0, float(position[1])],
        [0.0, 0.0, 1.0, float(position[2])],
        [0.0, 0.0, 0.0, 1.0],
    ]
    tool = ToolTrackingSnapshot(
        tool_id="0A",
        present=True,
        valid=True,
        validity_known=True,
        tracking_state="tracked",
        status="tracked",
        frame_number=10,
        quaternion_wxyz=(1.0, 0.0, 0.0, 0.0),
        translation_mm=tuple(position),
        T_aurora_tool=matrix,
    )
    return TrackingSnapshot(
        health=ServiceHealthSnapshot(name="tracking", health="healthy", state="connected", status="ok"),
        connection_state="connected",
        canonical_state="streaming_healthy",
        backend_identity="scikit-surgerynditracker",
        configured_backend_name="ndi",
        selected_backend_name="ndi",
        tracker_data_age_s=float(age_s),
        tracker_data_stale=bool(stale),
        normalized_live_tool_ids=["0A"],
        raw_live_tool_ids=["0A"],
        last_frame_number=10,
        tools={"0A": tool},
        registration_state="loaded",
        T_robot_aurora=[
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
        runtime_tip_calibration_state=runtime_tip_state,
        runtime_tip_mode=runtime_tip_mode,
        runtime_tip_trust_level=(
            "thesis_trusted"
            if runtime_tip_mode == "coil_as_tip" and runtime_tip_state == "coil_as_tip"
            else ("debug_only" if runtime_tip_mode == "quick_4_point" else "lower_trust")
        ),
        runtime_tip_mode_message=f"mode={runtime_tip_mode}",
        runtime_tip_identity_fallback=(
            runtime_tip_state in {"identity_tip_fallback", "coil_as_tip"} or runtime_tip_mode == "coil_as_tip"
        ),
        tip_pose_status=tip_pose_status,
        T_robot_tip=None if tip_pose_status not in {"ok", "coil_as_tip"} else matrix,
    )


class _TrackingService:
    def __init__(self, snapshot: TrackingSnapshot) -> None:
        self.snapshot = snapshot

    def get_snapshot(self) -> TrackingSnapshot:
        return self.snapshot

    def peek_snapshot(self) -> TrackingSnapshot:
        return self.snapshot


class _DummyController:
    def __init__(self, tmp_path: Path) -> None:
        self.settings = _settings(mock_mode=False)
        self.project_root = tmp_path
        self.experiment_runner = type("Runner", (), {"monotonic_fn": lambda _self: 0.0, "sleep_fn": lambda _self, _s: None})()
        self.tracking_service = _TrackingService(_tracking_snapshot())

    def config_payload(self) -> dict:
        return {}

    def set_config_value(self, _key, _value) -> None:
        return None

    def refresh(self):
        raise RuntimeError("not needed")

    def stop(self) -> None:
        return None

    def set_output_root(self, _path: str) -> None:
        return None

    def set_operator_notes(self, _notes: str) -> None:
        return None


def _sample(
    *,
    phase: str,
    sample_index: int,
    target_index: int,
    approach_index: int,
    position_mm: list[float],
    accepted: bool = True,
):
    from continuum_robot.experiments.schemas import ExperimentTimeseriesSample

    return ExperimentTimeseriesSample(
        monotonic_time_s=float(sample_index),
        wall_time_utc="2026-04-15T00:00:00Z",
        phase=phase,
        step_index=sample_index,
        sample_index=sample_index,
        target_index=target_index,
        revisit_index=sample_index,
        approach_index=approach_index,
        pose_in_robot_frame={"tip": {"translation_mm": [float(value) for value in position_mm]}},
        status_flags=["full_pose_available"] if accepted else ["capture_rejected"],
        extra={"capture_accepted": bool(accepted)},
    )

"""Tests for the two-segment repeatability scaffold experiment."""

from __future__ import annotations

import json
from pathlib import Path

from continuum_robot.experiments.builtins import register_builtin_experiments
from continuum_robot.experiments.registry import ExperimentRegistry
from continuum_robot.experiments.two_segment_repeatability import (
    EXPERIMENT_NAME,
    TwoSegmentRepeatabilityConfig,
    build_two_segment_repeatability_targets,
)

# Reuse the dataset test fixtures (servo service + runner + tracking snapshot).
from tests.test_two_segment_collect_pose_dataset import (
    _DropOneServoBus,
    _FakeTrackingService,
    _runner,
    _save_all8_startup,
    _servo_service,
    _settings,
    _tracking_snapshot,
)


def test_two_segment_repeatability_experiment_is_registered() -> None:
    registry = ExperimentRegistry()
    register_builtin_experiments(registry)

    descriptor = registry.get(EXPERIMENT_NAME)

    assert descriptor.title == "Two-Segment Repeatability (Scaffold)"
    assert "Two Segment" in descriptor.tags
    assert "Repeatability" in descriptor.tags


def test_two_segment_repeatability_target_set_includes_rings_and_combined() -> None:
    config = TwoSegmentRepeatabilityConfig.from_dict({
        "bottom_inner_radius_cm": 0.1,
        "bottom_outer_radius_cm": 0.2,
        "top_inner_radius_cm": 0.1,
        "top_outer_radius_cm": 0.2,
        "points_per_ring": 4,
    })

    targets = build_two_segment_repeatability_targets(config)
    group_tags = {target.group_tag for target in targets}

    assert "center" in group_tags
    assert "bottom_only_inner" in group_tags
    assert "bottom_only_outer" in group_tags
    assert "top_only_inner" in group_tags
    assert "top_only_outer" in group_tags
    assert "combined" in group_tags
    # 1 center + 4*4 rings + 4 combined = 21
    assert len(targets) == 1 + 4 * 4 + 4


def test_two_segment_repeatability_executes_and_reports_per_target_scatter(tmp_path: Path) -> None:
    service = _servo_service(tmp_path)
    _save_all8_startup(service)
    runner = _runner(
        tmp_path,
        service=service,
        tracking_service=_FakeTrackingService(_tracking_snapshot(include_distal=True, include_intermediate=True)),
    )

    result = runner.run_experiment(
        EXPERIMENT_NAME,
        config={
            "bottom_inner_radius_cm": 0.05,
            "bottom_outer_radius_cm": 0.10,
            "top_inner_radius_cm": 0.05,
            "top_outer_radius_cm": 0.10,
            "points_per_ring": 2,
            "repeat_visits": 2,
            "samples_per_visit": 1,
            "settle_time_s": 0.0,
            "allow_servo_only_test_run": True,
            "run_trust_mode": "servo_only",
        },
    )

    assert result.success is True
    metrics = result.summary.experiment_metrics
    assert metrics["dataset_type"] == "two_segment_repeatability"
    assert metrics["target_count"] == 1 + 2 * 2 + 2 * 2 + 4  # 13
    assert metrics["repeat_visits"] == 2
    assert metrics["accepted_sample_count"] >= 1
    scatter = metrics["scatter_metrics"]
    assert isinstance(scatter, dict)
    assert "per_target" in scatter
    assert metrics["bottom_segment_key"] == "segment_a"
    assert metrics["top_segment_key"] == "segment_b"
    # Output files
    assert (result.paths.output_dir / "two_segment_repeatability_summary.txt").exists()
    assert (result.paths.output_dir / "two_segment_repeatability_scatter_metrics.json").exists()
    assert (result.paths.output_dir / "two_segment_repeatability_per_target.csv").exists()
    assert (result.paths.output_dir / "two_segment_repeatability_distal_scatter.png").exists()
    summary = (result.paths.output_dir / "two_segment_repeatability_summary.txt").read_text(encoding="utf-8")
    assert "valid_for_thesis_repeatability: false" in summary
    assert "automatic_two_segment_pretension_validated: false" in summary
    # Policy 10: distal is the primary repeatability metric; intermediate is
    # secondary. The summary and metrics must surface this explicitly.
    assert "primary_metric: distal_tip_rms_mm" in summary
    assert "secondary_metric: intermediate_segment_rms_mm" in summary
    assert "aggregate_distal_rms_mm (primary)" in summary
    assert metrics["primary_repeatability_role"] == "distal_tip"
    assert metrics["secondary_repeatability_role"] == "intermediate_segment"
    # Repeatability summary and per-target CSV must be exportable.
    from continuum_robot.data.export_run_bundle import export_run_bundle

    export = export_run_bundle(
        run_dir=result.paths.output_dir,
        output_root=tmp_path / "exports",
        project_root=tmp_path,
    )
    exported = {entry.bundle_path for entry in export.entries}
    assert "two_segment_repeatability_summary.txt" in exported
    assert "two_segment_repeatability_scatter_metrics.json" in exported
    assert "two_segment_repeatability_per_target.csv" in exported


def test_two_segment_repeatability_records_bottom_top_when_swapped(tmp_path: Path) -> None:
    settings = _settings()
    settings.robot.bottom_segment_key = "segment_b"
    settings.robot.top_segment_key = "segment_a"
    service = _servo_service(tmp_path, settings=settings)
    _save_all8_startup(service)
    runner = _runner(
        tmp_path,
        settings=settings,
        service=service,
        tracking_service=_FakeTrackingService(_tracking_snapshot(include_distal=True, include_intermediate=False)),
    )

    result = runner.run_experiment(
        EXPERIMENT_NAME,
        config={
            "bottom_inner_radius_cm": 0.0,
            "bottom_outer_radius_cm": 0.0,
            "top_inner_radius_cm": 0.0,
            "top_outer_radius_cm": 0.0,
            "points_per_ring": 0,
            "repeat_visits": 1,
            "samples_per_visit": 1,
            "allow_servo_only_test_run": True,
            "run_trust_mode": "servo_only",
        },
    )

    assert result.success is True
    metrics = result.summary.experiment_metrics
    assert metrics["bottom_segment_key"] == "segment_b"
    assert metrics["top_segment_key"] == "segment_a"
    assert metrics["bottom_servo_ids"] == [5, 6, 7, 8]


def test_two_segment_repeatability_blocks_outside_dual_segment(tmp_path: Path) -> None:
    runner = _runner(tmp_path, settings=_settings(mode="single_segment"))

    result = runner.run_experiment(EXPERIMENT_NAME, config={})

    assert result.success is False
    assert "requires operating_mode=dual_segment" in result.message


def test_two_segment_repeatability_emits_current_load_summary_and_assembly_metadata(tmp_path: Path) -> None:
    service = _servo_service(tmp_path)
    _save_all8_startup(service)
    runner = _runner(
        tmp_path,
        service=service,
        tracking_service=_FakeTrackingService(_tracking_snapshot(include_distal=True, include_intermediate=False)),
    )

    result = runner.run_experiment(
        EXPERIMENT_NAME,
        config={
            "bottom_inner_radius_cm": 0.0,
            "bottom_outer_radius_cm": 0.0,
            "top_inner_radius_cm": 0.0,
            "top_outer_radius_cm": 0.0,
            "points_per_ring": 0,
            "repeat_visits": 1,
            "samples_per_visit": 1,
            "allow_servo_only_test_run": True,
            "run_trust_mode": "servo_only",
        },
    )

    assert result.success is True
    metrics = result.summary.experiment_metrics
    summary = metrics.get("current_load_summary") or {}
    assert summary.get("schema_version") == "two_segment_current_load_summary_v1"
    assert "per_servo" in summary
    # Default mock bus should not trip thresholds.
    assert summary.get("sustained_jam_servo_ids") == []


def test_two_segment_repeatability_rejects_samples_with_missing_measured_positions_in_trusted_run(tmp_path: Path) -> None:
    # Repeatability's precheck does not call read_live_telemetry, so the first
    # read is the first capture. warmup_reads=0 means every capture sees the
    # dropped servo 5 position.
    service = _servo_service(tmp_path, dxl_bus=_DropOneServoBus([1, 2, 3, 4, 5, 6, 7, 8], warmup_reads=0))
    _save_all8_startup(service)
    runner = _runner(
        tmp_path,
        service=service,
        tracking_service=_FakeTrackingService(_tracking_snapshot(include_distal=True, include_intermediate=False)),
    )

    result = runner.run_experiment(
        EXPERIMENT_NAME,
        config={
            "bottom_inner_radius_cm": 0.0,
            "bottom_outer_radius_cm": 0.0,
            "top_inner_radius_cm": 0.0,
            "top_outer_radius_cm": 0.0,
            "points_per_ring": 0,
            "repeat_visits": 1,
            "samples_per_visit": 1,
            "allow_servo_only_test_run": False,
            "run_trust_mode": "thesis_trusted",
        },
    )

    assert result.success is True
    metrics = result.summary.experiment_metrics
    samples_text = result.paths.samples_path.read_text(encoding="utf-8").strip().splitlines()
    assert samples_text, "expected at least one captured sample"
    sample = json.loads(samples_text[0])
    extra = sample["extra"]
    # Trusted run flips capture_accepted to False when servo 5 position is missing.
    assert extra["missing_measured_servo_ids"] == [5]
    assert extra["capture_accepted"] is False
    assert extra["capture_rejection_reason"].startswith("measured_servo_position_missing")
    # Counter parity with collect-pose.
    assert metrics["rejected_sample_count"] >= 1


def test_two_segment_repeatability_records_registration_summary_block(tmp_path: Path) -> None:
    """Every repeatability run captures the loaded registration's state + FRE + path."""
    service = _servo_service(tmp_path)
    _save_all8_startup(service)
    # The fake tracking snapshot has registration_state='loaded'. The
    # snapshot fixture doesn't populate FRE/path; we just confirm the block
    # is present and structured.
    runner = _runner(
        tmp_path,
        service=service,
        tracking_service=_FakeTrackingService(_tracking_snapshot(include_distal=True, include_intermediate=False)),
    )

    result = runner.run_experiment(
        EXPERIMENT_NAME,
        config={
            "bottom_inner_radius_cm": 0.0,
            "bottom_outer_radius_cm": 0.0,
            "top_inner_radius_cm": 0.0,
            "top_outer_radius_cm": 0.0,
            "points_per_ring": 0,
            "repeat_visits": 1,
            "samples_per_visit": 1,
            "allow_servo_only_test_run": True,
            "run_trust_mode": "servo_only",
        },
    )

    assert result.success is True
    metrics = result.summary.experiment_metrics
    registration = metrics.get("registration_summary") or {}
    assert registration.get("schema_version") == "two_segment_registration_summary_v1"
    # The fake snapshot reports state='loaded'.
    assert registration.get("registration_state") == "loaded"
    # Summary text should include the registration provenance block.
    summary_text = (result.paths.output_dir / "two_segment_repeatability_summary.txt").read_text(encoding="utf-8")
    assert "Registration Provenance:" in summary_text
    assert "registration_state:" in summary_text


def test_two_segment_repeatability_warns_when_registration_not_loaded(tmp_path: Path) -> None:
    """When no tracker is connected, the warning fires and the registration_state reflects that."""
    service = _servo_service(tmp_path)
    _save_all8_startup(service)
    runner = _runner(tmp_path, service=service, tracking_service=None)  # explicitly no tracker

    result = runner.run_experiment(
        EXPERIMENT_NAME,
        config={
            "bottom_inner_radius_cm": 0.0,
            "bottom_outer_radius_cm": 0.0,
            "top_inner_radius_cm": 0.0,
            "top_outer_radius_cm": 0.0,
            "points_per_ring": 0,
            "repeat_visits": 1,
            "samples_per_visit": 1,
            "allow_servo_only_test_run": True,
            "run_trust_mode": "servo_only",
        },
    )

    assert result.success is True
    metrics = result.summary.experiment_metrics
    registration = metrics.get("registration_summary") or {}
    assert registration.get("registration_state") == "tracker_unavailable"
    # The session warning should mention the registration is not loaded.
    assert any("Registration not fully loaded" in warning for warning in result.summary.warning_messages)


def test_two_segment_repeatability_blocks_invalid_physical_assembly(tmp_path: Path) -> None:
    bad = _settings()
    bad.robot.bottom_segment_key = "segment_a"
    bad.robot.top_segment_key = "segment_a"
    runner = _runner(tmp_path, settings=bad)

    result = runner.run_experiment(EXPERIMENT_NAME, config={})

    assert result.success is False
    assert "Invalid physical assembly" in result.message

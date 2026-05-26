"""Tests for the two_segment_workspace_repeatability experiment.

Covers target generation, visit-order round-robin, dry-run execution flow,
output writers, per-target metrics, and validator + export integration.
"""

from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path

import pytest

from continuum_robot.data.export_run_bundle import CORE_FILENAMES
from continuum_robot.data.validate_run_bundle import validate_run_folder
from continuum_robot.experiments.builtins import register_builtin_experiments
from continuum_robot.experiments.registry import ExperimentRegistry
from continuum_robot.experiments.two_segment_workspace_repeatability import (
    EXPERIMENT_NAME,
    TwoSegmentWorkspaceRepeatabilityConfig,
    TwoSegmentWorkspaceTarget,
    build_round_robin_visit_order,
    build_two_segment_workspace_targets,
)
from continuum_robot.experiments.two_segment_workspace_repeatability_outputs import (
    compute_workspace_repeatability_metrics,
    summarize_workspace_repeatability,
    write_two_segment_workspace_repeatability_outputs,
)


# Reuse the well-tested two-segment runner harness (mock dxl bus + tracking).
from tests.test_two_segment_collect_pose_dataset import (
    _FakeTrackingService,
    _runner,
    _save_all8_startup,
    _servo_service,
    _settings,
    _tracking_snapshot,
)


# ---------------------------------------------------------------------------
# Phase 2: target generation
# ---------------------------------------------------------------------------


def test_default_config_produces_200_targets_with_20_repeats() -> None:
    config = TwoSegmentWorkspaceRepeatabilityConfig.from_dict({})
    assert config.target_count == 200
    assert config.repeats_per_target == 20
    assert config.return_to_neutral_between_visits is True
    assert config.expected_distal_tool_id == "0A"
    assert config.target_generator_mode == "workspace_latin_hypercube"


def test_build_two_segment_workspace_targets_default_count_and_unique_ids() -> None:
    config = TwoSegmentWorkspaceRepeatabilityConfig.from_dict(
        {"max_segment_displacement_cm": 0.5, "random_seed": 7}
    )
    targets = build_two_segment_workspace_targets(config)
    assert len(targets) == 200
    ids = [t.target_id for t in targets]
    assert len(set(ids)) == 200
    indices = [t.target_index for t in targets]
    assert indices == list(range(200))


def test_targets_have_bottom_top_command_fields_and_ordered_8_vector() -> None:
    config = TwoSegmentWorkspaceRepeatabilityConfig.from_dict(
        {"target_count": 12, "max_segment_displacement_cm": 0.25, "random_seed": 1}
    )
    targets = build_two_segment_workspace_targets(config)
    for target in targets:
        assert isinstance(target, TwoSegmentWorkspaceTarget)
        # bottom/top command fields present
        assert isinstance(target.bottom_x_cm, float)
        assert isinstance(target.bottom_y_cm, float)
        assert isinstance(target.top_x_cm, float)
        assert isinstance(target.top_y_cm, float)
        # tendon vectors are 4-vec antagonistic
        assert target.bottom_tendon_cm == pytest.approx(
            [target.bottom_x_cm, target.bottom_y_cm, -target.bottom_x_cm, -target.bottom_y_cm]
        )
        assert target.top_tendon_cm == pytest.approx(
            [target.top_x_cm, target.top_y_cm, -target.top_x_cm, -target.top_y_cm]
        )
        # ordered 8-vec can be derived; values bounded by amplitude
        assert all(abs(v) <= 0.25 + 1e-9 for v in target.bottom_tendon_cm)
        assert all(abs(v) <= 0.25 + 1e-9 for v in target.top_tendon_cm)


def test_target_set_is_reproducible_under_seed() -> None:
    config_a = TwoSegmentWorkspaceRepeatabilityConfig.from_dict(
        {"target_count": 50, "max_segment_displacement_cm": 0.3, "random_seed": 42}
    )
    config_b = TwoSegmentWorkspaceRepeatabilityConfig.from_dict(
        {"target_count": 50, "max_segment_displacement_cm": 0.3, "random_seed": 42}
    )
    config_c = TwoSegmentWorkspaceRepeatabilityConfig.from_dict(
        {"target_count": 50, "max_segment_displacement_cm": 0.3, "random_seed": 9999}
    )
    a = [(t.bottom_x_cm, t.bottom_y_cm, t.top_x_cm, t.top_y_cm) for t in build_two_segment_workspace_targets(config_a)]
    b = [(t.bottom_x_cm, t.bottom_y_cm, t.top_x_cm, t.top_y_cm) for t in build_two_segment_workspace_targets(config_b)]
    c = [(t.bottom_x_cm, t.bottom_y_cm, t.top_x_cm, t.top_y_cm) for t in build_two_segment_workspace_targets(config_c)]
    assert a == b
    assert a != c


def test_requested_amplitude_is_honored_no_silent_cap() -> None:
    """Larger requested amplitudes must produce larger commanded values."""
    small = TwoSegmentWorkspaceRepeatabilityConfig.from_dict(
        {"target_count": 100, "max_segment_displacement_cm": 0.1, "random_seed": 0}
    )
    large = TwoSegmentWorkspaceRepeatabilityConfig.from_dict(
        {"target_count": 100, "max_segment_displacement_cm": 1.0, "random_seed": 0}
    )
    targets_small = build_two_segment_workspace_targets(small)
    targets_large = build_two_segment_workspace_targets(large)
    max_small = max(
        max(abs(t.bottom_x_cm), abs(t.bottom_y_cm), abs(t.top_x_cm), abs(t.top_y_cm))
        for t in targets_small
    )
    max_large = max(
        max(abs(t.bottom_x_cm), abs(t.bottom_y_cm), abs(t.top_x_cm), abs(t.top_y_cm))
        for t in targets_large
    )
    assert 0.05 < max_small <= 0.1
    assert 0.5 < max_large <= 1.0


def test_round_robin_visit_order_covers_each_target_exactly_repeats_times() -> None:
    sequence = build_round_robin_visit_order(target_count=200, repeats_per_target=20, random_seed=11)
    assert len(sequence) == 200 * 20
    target_counts = Counter(target_index for _, _, target_index in sequence)
    assert all(count == 20 for count in target_counts.values())
    assert len(target_counts) == 200
    # Round-robin ordering: each cycle is a permutation of [0..199].
    cycles: dict[int, list[int]] = {}
    for cycle, _visit, target in sequence:
        cycles.setdefault(cycle, []).append(target)
    assert len(cycles) == 20
    for indices in cycles.values():
        assert sorted(indices) == list(range(200))


def test_round_robin_under_same_seed_is_reproducible() -> None:
    a = build_round_robin_visit_order(target_count=200, repeats_per_target=20, random_seed=11)
    b = build_round_robin_visit_order(target_count=200, repeats_per_target=20, random_seed=11)
    c = build_round_robin_visit_order(target_count=200, repeats_per_target=20, random_seed=12)
    assert a == b
    assert a != c


def test_rings_and_axes_generator_is_supported() -> None:
    config = TwoSegmentWorkspaceRepeatabilityConfig.from_dict(
        {"target_count": 50, "target_generator_mode": "rings_and_axes", "max_segment_displacement_cm": 0.5}
    )
    targets = build_two_segment_workspace_targets(config)
    assert len(targets) == 50
    # Origin / neutral target present.
    assert any(t.group_tag == "neutral_or_near_neutral" for t in targets)
    # Bottom-only and top-only groups present.
    tags = {t.group_tag for t in targets}
    assert "bottom_only" in tags
    assert "top_only" in tags
    assert "combined" in tags


# ---------------------------------------------------------------------------
# Phase 3: execution protocol (dry run)
# ---------------------------------------------------------------------------


def test_experiment_is_registered_with_clear_title() -> None:
    registry = ExperimentRegistry()
    register_builtin_experiments(registry)
    descriptor = registry.get(EXPERIMENT_NAME)
    assert descriptor.title == "Two-Segment Workspace Repeatability"
    assert "Two Segment" in descriptor.tags
    assert "Repeatability" in descriptor.tags


def test_dry_run_executes_shortened_protocol_and_writes_outputs(tmp_path: Path) -> None:
    """Phase 3 + 4: dry-run path writes the full run folder with shortened counts."""
    settings = _settings()
    settings.robot.bottom_segment_key = "segment_b"
    settings.robot.top_segment_key = "segment_a"
    settings.robot.physical_assembly_confirmed_by_operator = True
    service = _servo_service(tmp_path, settings=settings)
    _save_all8_startup(service)
    runner = _runner(
        tmp_path,
        settings=settings,
        service=service,
        tracking_service=_FakeTrackingService(_tracking_snapshot(include_distal=True)),
    )

    result = runner.run_experiment(
        EXPERIMENT_NAME,
        config={
            "target_count": 5,
            "repeats_per_target": 2,
            "max_segment_displacement_cm": 0.1,
            "random_seed": 1,
            "neutral_settle_s": 0.0,
            "target_settle_s": 0.0,
            "dry_run": True,
            "allow_servo_only_test_run": True,
        },
    )
    assert result.success is True
    metrics = result.summary.experiment_metrics
    assert metrics["target_count"] == 5
    assert metrics["repeats_per_target"] == 2
    assert metrics["planned_visits"] == 10
    assert metrics["accepted_captures"] == 10
    assert metrics["rejected_captures"] == 0
    assert metrics["demo_only"] is False
    assert metrics["valid_for_repeatability_analysis"] is True
    assert metrics["primary_metric"] == "distal_xyz_repeatability"
    assert metrics["expected_distal_tool_id"] == "0A"
    # Files written. The new experiment writes BOTH the canonical
    # single-segment workspace_map_* filenames AND the two-segment-specific
    # extras so the existing data-plumbing recognises the run shape.
    out = result.paths.output_dir
    for required in (
        "two_segment_workspace_repeatability_summary.txt",
        # Canonical single-segment-shape artifacts.
        "workspace_map_summary.json",
        "workspace_map_visits.jsonl",
        "workspace_map_per_target.csv",
        # Two-segment-specific extras.
        "repeatability_targets.json",
        "repeatability_visit_plan.csv",
        "target_captures.csv",
        "per_target_repeatability.csv",
        "repeatability_metrics.csv",
        "failure_events.jsonl",
    ):
        assert (out / required).exists(), f"missing {required}"
    # Canonical summary JSON uses the same shape as single-segment.
    workspace_summary = json.loads((out / "workspace_map_summary.json").read_text(encoding="utf-8"))
    assert "summary" in workspace_summary
    assert "per_target_rows" in workspace_summary
    for key in ("workspace_rms_mean_mm", "workspace_rms_max_mm", "workspace_rms_p95_mm", "target_count", "targets_with_data"):
        assert key in workspace_summary["summary"], f"summary missing {key}"
    summary_text = (out / "two_segment_workspace_repeatability_summary.txt").read_text(encoding="utf-8")
    assert "Two-Segment Workspace Repeatability" in summary_text
    assert "target_count" in summary_text
    assert "primary_metric: distal_xyz_repeatability" in summary_text


def test_dry_run_returns_to_neutral_between_visits_in_protocol_metadata(tmp_path: Path) -> None:
    settings = _settings()
    settings.robot.bottom_segment_key = "segment_b"
    settings.robot.top_segment_key = "segment_a"
    settings.robot.physical_assembly_confirmed_by_operator = True
    service = _servo_service(tmp_path, settings=settings)
    _save_all8_startup(service)
    runner = _runner(
        tmp_path,
        settings=settings,
        service=service,
        tracking_service=_FakeTrackingService(_tracking_snapshot(include_distal=True)),
    )
    result = runner.run_experiment(
        EXPERIMENT_NAME,
        config={
            "target_count": 3,
            "repeats_per_target": 2,
            "max_segment_displacement_cm": 0.1,
            "random_seed": 0,
            "neutral_settle_s": 0.0,
            "target_settle_s": 0.0,
            "return_to_neutral_between_visits": True,
            "dry_run": True,
            "allow_servo_only_test_run": True,
        },
    )
    assert result.success is True
    metrics = result.summary.experiment_metrics
    assert metrics["return_to_neutral_between_visits"] is True


def test_blocks_outside_dual_segment(tmp_path: Path) -> None:
    settings = _settings(mode="single_segment")
    runner = _runner(tmp_path, settings=settings)
    result = runner.run_experiment(EXPERIMENT_NAME, config={})
    assert result.success is False
    assert "dual_segment" in result.message


def test_blocks_when_amplitude_exceeds_safe_tick_budget(tmp_path: Path) -> None:
    """1 cm amplitude with a tight max_tick_delta_from_startup must reject at precheck."""
    settings = _settings()
    settings.robot.bottom_segment_key = "segment_b"
    settings.robot.top_segment_key = "segment_a"
    settings.robot.physical_assembly_confirmed_by_operator = True
    service = _servo_service(tmp_path, settings=settings)
    _save_all8_startup(service)
    runner = _runner(
        tmp_path,
        settings=settings,
        service=service,
        tracking_service=_FakeTrackingService(_tracking_snapshot(include_distal=True)),
    )
    result = runner.run_experiment(
        EXPERIMENT_NAME,
        config={
            "target_count": 5,
            "repeats_per_target": 1,
            "max_segment_displacement_cm": 1.0,
            "max_tick_delta_from_startup": 5,  # absurdly tight cap
            "dry_run": True,
            "allow_servo_only_test_run": True,
        },
    )
    assert result.success is False
    assert "exceeds configured tick limits" in result.message


# ---------------------------------------------------------------------------
# Phase 5: metrics
# ---------------------------------------------------------------------------


def test_per_target_metrics_correct_on_synthetic_data() -> None:
    target = TwoSegmentWorkspaceTarget(
        target_index=0,
        target_id="WS_0000",
        bottom_x_cm=0.0,
        bottom_y_cm=0.0,
        top_x_cm=0.0,
        top_y_cm=0.0,
        bottom_tendon_cm=[0.0, 0.0, 0.0, 0.0],
        top_tendon_cm=[0.0, 0.0, 0.0, 0.0],
        group_tag="neutral_or_near_neutral",
        amplitude_cm=0.0,
    )
    # Four visits at known offsets from the centroid:
    # mean is (0, 0, 0); rms radial = sqrt((1+1+1+1)/4) = 1.0
    visits = [
        {"accepted": True, "target_index": 0, "distal_xyz_robot_mm": [1.0, 0.0, 0.0]},
        {"accepted": True, "target_index": 0, "distal_xyz_robot_mm": [-1.0, 0.0, 0.0]},
        {"accepted": True, "target_index": 0, "distal_xyz_robot_mm": [0.0, 1.0, 0.0]},
        {"accepted": True, "target_index": 0, "distal_xyz_robot_mm": [0.0, -1.0, 0.0]},
    ]
    per_target = compute_workspace_repeatability_metrics(visit_results=visits, targets=[target])
    row = per_target[0]
    assert row["accepted_repeats"] == 4
    assert row["centroid_xyz_mm"] == pytest.approx([0.0, 0.0, 0.0])
    assert row["rms_spread_mm"] == pytest.approx(1.0)
    assert row["max_radial_mm"] == pytest.approx(1.0)


def test_rejected_visits_excluded_from_metrics() -> None:
    target = TwoSegmentWorkspaceTarget(
        target_index=0,
        target_id="WS_0000",
        bottom_x_cm=0.0,
        bottom_y_cm=0.0,
        top_x_cm=0.0,
        top_y_cm=0.0,
        bottom_tendon_cm=[0.0, 0.0, 0.0, 0.0],
        top_tendon_cm=[0.0, 0.0, 0.0, 0.0],
        group_tag="neutral_or_near_neutral",
        amplitude_cm=0.0,
    )
    visits = [
        {"accepted": True, "target_index": 0, "distal_xyz_robot_mm": [0.0, 0.0, 0.0]},
        {"accepted": False, "target_index": 0, "distal_xyz_robot_mm": [10.0, 0.0, 0.0], "reject_reason": "stale"},
    ]
    per_target = compute_workspace_repeatability_metrics(visit_results=visits, targets=[target])
    assert per_target[0]["accepted_repeats"] == 1


def test_targets_below_minimum_repeats_flagged() -> None:
    target_a = TwoSegmentWorkspaceTarget(
        target_index=0, target_id="WS_0", bottom_x_cm=0.0, bottom_y_cm=0.0, top_x_cm=0.0, top_y_cm=0.0,
        bottom_tendon_cm=[0.0]*4, top_tendon_cm=[0.0]*4, group_tag="neutral", amplitude_cm=0.0,
    )
    target_b = TwoSegmentWorkspaceTarget(
        target_index=1, target_id="WS_1", bottom_x_cm=0.1, bottom_y_cm=0.0, top_x_cm=0.0, top_y_cm=0.0,
        bottom_tendon_cm=[0.1, 0.0, -0.1, 0.0], top_tendon_cm=[0.0]*4, group_tag="bottom_only", amplitude_cm=0.1,
    )
    visits = (
        [{"accepted": True, "target_index": 0, "distal_xyz_robot_mm": [0.0, 0.0, 0.0]} for _ in range(20)]
        + [{"accepted": True, "target_index": 1, "distal_xyz_robot_mm": [1.0, 0.0, 0.0]} for _ in range(5)]
    )
    per_target = compute_workspace_repeatability_metrics(visit_results=visits, targets=[target_a, target_b])
    summary = summarize_workspace_repeatability(per_target, minimum_repeats_per_target=15)
    assert summary["targets_below_minimum_repeats"] == 1
    # Canonical single-segment-shape key for "targets with at least one accepted visit".
    assert summary["targets_with_data"] == 2


# ---------------------------------------------------------------------------
# Phase 6: figures - smoke-test
# ---------------------------------------------------------------------------


def test_outputs_writer_produces_figures_and_csv(tmp_path: Path) -> None:
    targets = [
        TwoSegmentWorkspaceTarget(
            target_index=i, target_id=f"WS_{i:03d}",
            bottom_x_cm=0.01 * i, bottom_y_cm=0.0, top_x_cm=0.0, top_y_cm=0.0,
            bottom_tendon_cm=[0.01 * i, 0.0, -0.01 * i, 0.0],
            top_tendon_cm=[0.0, 0.0, 0.0, 0.0],
            group_tag="bottom_only" if i > 0 else "neutral_or_near_neutral",
            amplitude_cm=abs(0.01 * i),
        )
        for i in range(8)
    ]
    visits = []
    for target in targets:
        for repeat in range(3):
            visits.append(
                {
                    "accepted": True,
                    "target_index": target.target_index,
                    "distal_xyz_robot_mm": [target.bottom_x_cm * 10.0 + 0.01 * repeat, 0.0, 100.0],
                    "timestamp_utc": "2026-01-01T00:00:00Z",
                    "all_8_goal_ticks": {str(s): 2048 for s in range(1, 9)},
                    "all_8_present_position_ticks": {str(s): 2048 for s in range(1, 9)},
                    "all_8_current_load_proxy_ma": {str(s): 100.0 for s in range(1, 9)},
                }
            )
    out_dir = tmp_path / "ws_repeat_out"
    paths = write_two_segment_workspace_repeatability_outputs(
        output_dir=out_dir,
        targets=targets,
        visit_results=visits,
        failure_events=[],
        metrics={
            "experiment_name": "two_segment_workspace_repeatability",
            "target_count": 8,
            "repeats_per_target": 3,
            "planned_visits": 24,
            "accepted_captures": 24,
            "rejected_captures": 0,
            "demo_only": False,
            "valid_for_repeatability_analysis": True,
            "valid_for_model_training": False,
            "valid_for_thesis_repeatability": False,
            "primary_metric": "distal_xyz_repeatability",
            "controlled_point": "distal_tip coil origin in robot base frame",
            "expected_distal_tool_id": "0A",
            "max_segment_displacement_cm": 0.1,
            "target_generator_mode": "workspace_latin_hypercube",
            "random_seed": 0,
            "return_to_neutral_between_visits": True,
            "stop_reason": "test_complete",
        },
    )
    assert (out_dir / "repeatability_targets.json").exists()
    assert (out_dir / "per_target_repeatability.csv").exists()
    assert (out_dir / "workspace_map_summary.json").exists()
    assert (out_dir / "workspace_map_per_target.csv").exists()
    # Figures may or may not exist depending on matplotlib availability, but
    # if they do the names are the canonical thesis-figure filenames.
    for figure_key in ("thesis_01", "thesis_02", "thesis_03", "thesis_04"):
        if figure_key in paths:
            assert paths[figure_key].exists()


def test_outputs_writer_handles_single_target_without_crash(tmp_path: Path) -> None:
    target = TwoSegmentWorkspaceTarget(
        target_index=0, target_id="WS_0000",
        bottom_x_cm=0.0, bottom_y_cm=0.0, top_x_cm=0.0, top_y_cm=0.0,
        bottom_tendon_cm=[0.0]*4, top_tendon_cm=[0.0]*4,
        group_tag="neutral_or_near_neutral", amplitude_cm=0.0,
    )
    visits = [{"accepted": True, "target_index": 0, "distal_xyz_robot_mm": [0.0, 0.0, 0.0]}]
    paths = write_two_segment_workspace_repeatability_outputs(
        output_dir=tmp_path / "one_target",
        targets=[target],
        visit_results=visits,
        failure_events=[],
        metrics={
            "experiment_name": "two_segment_workspace_repeatability",
            "target_count": 1,
            "repeats_per_target": 1,
            "planned_visits": 1,
            "accepted_captures": 1,
            "rejected_captures": 0,
            "demo_only": False,
            "valid_for_repeatability_analysis": True,
            "valid_for_model_training": False,
            "primary_metric": "distal_xyz_repeatability",
            "controlled_point": "distal_tip coil origin in robot base frame",
            "expected_distal_tool_id": "0A",
        },
    )
    assert (tmp_path / "one_target" / "per_target_repeatability.csv").exists()


# ---------------------------------------------------------------------------
# Phase 8: data / export / validator
# ---------------------------------------------------------------------------


def test_core_filenames_includes_new_workspace_repeatability_artifacts() -> None:
    expected = {
        "two_segment_workspace_repeatability_summary.txt",
        # Canonical single-segment-compatible filenames.
        "workspace_map_summary.json",
        "workspace_map_visits.jsonl",
        "workspace_map_per_target.csv",
        "thesis_01_workspace_rms_3d.png",
        "thesis_02_workspace_rms_map.png",
        "thesis_03_rms_vs_amplitude.png",
        "thesis_04_2d_repeatability_map.png",
        # Two-segment-specific extras.
        "repeatability_targets.json",
        "repeatability_visit_plan.csv",
        "repeatability_metrics.csv",
        "per_target_repeatability.csv",
        "target_captures.csv",
        "failure_events.jsonl",
    }
    assert expected.issubset(CORE_FILENAMES)


def _write_good_workspace_repeatability_run(tmp_path: Path) -> Path:
    run_dir = (
        tmp_path / "data" / "experiments" / "two_segment_workspace_repeatability"
        / "20260601_120000_two_segment_workspace_repeatability"
    )
    run_dir.mkdir(parents=True)
    metadata = {
        "experiment_name": "two_segment_workspace_repeatability",
        "run_id": run_dir.name,
        "trust_info": {"run_trust_mode": "repeatability_run"},
        "provenance_info": {"operating_mode": "dual_segment"},
    }
    summary = {
        "experiment_name": "two_segment_workspace_repeatability",
        "run_id": run_dir.name,
        "success": True,
        "status": "success",
        "experiment_metrics": {
            "demo_only": False,
            "valid_for_repeatability_analysis": True,
            "valid_for_model_training": False,
            "valid_for_thesis_repeatability": False,
            "primary_metric": "distal_xyz_repeatability",
            "target_count": 200,
            "repeats_per_target": 20,
            "planned_visits": 4000,
            "accepted_captures": 4000,
            "rejected_captures": 0,
            "repeatability_summary": {"targets_below_minimum_repeats": 0},
        },
    }
    (run_dir / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
    (run_dir / "summary.json").write_text(json.dumps(summary), encoding="utf-8")
    for filename in (
        "two_segment_workspace_repeatability_summary.txt",
        "workspace_map_summary.json",
        "workspace_map_visits.jsonl",
        "workspace_map_per_target.csv",
        "per_target_repeatability.csv",
        "target_captures.csv",
        "repeatability_targets.json",
        "repeatability_visit_plan.csv",
    ):
        (run_dir / filename).write_text("placeholder\n", encoding="utf-8")
    return run_dir


def test_validator_passes_well_formed_workspace_repeatability_run(tmp_path: Path) -> None:
    run_dir = _write_good_workspace_repeatability_run(tmp_path)
    report = validate_run_folder(run_dir)
    fails = [i.message for i in report.issues if i.level == "FAIL"]
    assert not fails, f"unexpected FAIL: {fails}"


def test_validator_fails_when_demo_only_or_model_training_true(tmp_path: Path) -> None:
    run_dir = _write_good_workspace_repeatability_run(tmp_path)
    summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    summary["experiment_metrics"]["valid_for_model_training"] = True
    (run_dir / "summary.json").write_text(json.dumps(summary), encoding="utf-8")
    report = validate_run_folder(run_dir)
    fails = [i.message for i in report.issues if i.level == "FAIL"]
    assert any("valid_for_model_training" in m for m in fails)


def test_validator_warns_when_accepted_less_than_planned(tmp_path: Path) -> None:
    run_dir = _write_good_workspace_repeatability_run(tmp_path)
    summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    summary["experiment_metrics"]["accepted_captures"] = 3500
    (run_dir / "summary.json").write_text(json.dumps(summary), encoding="utf-8")
    report = validate_run_folder(run_dir)
    warns = [i.message for i in report.issues if i.level == "WARN"]
    assert any("3500 < planned 4000" in m for m in warns)

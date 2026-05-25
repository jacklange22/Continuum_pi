"""Tests for the two-segment penprobe lookup demo experiment."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from continuum_robot.demo.two_segment_workspace_lookup import (
    LookupMapBuildConfig,
    build_workspace_lookup_map,
)
from continuum_robot.experiments.builtins import register_builtin_experiments
from continuum_robot.experiments.registry import ExperimentRegistry
from continuum_robot.experiments.two_segment_penprobe_lookup_demo import EXPERIMENT_NAME

# Reuse the same synthetic dataset fixtures + runner from the collect-pose
# tests so the demo runs against format-faithful jsonl rows and the same
# mock servo/tracker harness.
from tests.test_two_segment_collect_pose_dataset import (
    _FakeTrackingService,
    _runner,
    _save_all8_startup,
    _servo_service,
    _settings,
    _tracking_snapshot,
)
from tests.test_two_segment_workspace_lookup import _write_dataset_run


def _build_map_for_demo(tmp_path: Path) -> Path:
    """Build a workspace lookup map and return the JSON path."""
    run_dir = _write_dataset_run(
        tmp_path,
        sample_xyzs=[(x, y, 100.0) for x in (-10.0, 0.0, 10.0) for y in (-10.0, 0.0, 10.0)],
    )
    result = build_workspace_lookup_map(
        [run_dir],
        config=LookupMapBuildConfig(voxel_size_mm=None, output_dir=tmp_path / "demo_map"),
    )
    return result.artifact_paths["map_json"]


def _runner_with_penprobe(tmp_path: Path, *, include_intermediate: bool = False):
    """Build a runner whose mock tracker has 0B (the penprobe) visible."""
    settings = _settings()
    settings.robot.bottom_segment_key = "segment_b"
    settings.robot.top_segment_key = "segment_a"
    settings.robot.physical_assembly_confirmed_by_operator = True
    service = _servo_service(tmp_path, settings=settings)
    _save_all8_startup(service)
    # Add a 0B tool snapshot to the mock tracker so the demo can resolve it
    # via the shared `_extract_robot_frame_tool_pose` helper.
    from continuum_robot.services.models import ToolTrackingSnapshot

    snapshot = _tracking_snapshot(include_distal=True, include_intermediate=include_intermediate)
    snapshot.tools["0B"] = ToolTrackingSnapshot(
        tool_id="0B",
        present=True,
        valid=True,
        tracking_state="tracked",
        translation_mm=(2.0, 2.0, 100.0),  # inside our map's workspace
        quaternion_wxyz=(1.0, 0.0, 0.0, 0.0),
        quality=0.9,
    )
    # The robot-frame pose is computed via T_robot_aurora @ T_aurora_tool.
    # _tracking_snapshot's TrackingSnapshot already includes T_robot_tip but we
    # need T_robot_aurora; set it to identity so target == translation.
    snapshot.T_robot_aurora = [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ]
    tracking_service = _FakeTrackingService(snapshot)
    return _runner(tmp_path, settings=settings, service=service, tracking_service=tracking_service)


def test_two_segment_penprobe_lookup_demo_is_registered() -> None:
    registry = ExperimentRegistry()
    register_builtin_experiments(registry)
    descriptor = registry.get(EXPERIMENT_NAME)
    assert descriptor.title == "Two-Segment Penprobe Lookup Demo"
    assert "Demo" in descriptor.tags
    assert "Not Closed Loop" in descriptor.tags


def test_demo_blocks_outside_dual_segment(tmp_path: Path) -> None:
    map_path = _build_map_for_demo(tmp_path)
    settings = _settings(mode="single_segment")
    runner = _runner(tmp_path, settings=settings)
    result = runner.run_experiment(EXPERIMENT_NAME, config={"map_path": str(map_path)})
    assert result.success is False
    assert "dual_segment" in result.message


def test_demo_blocks_when_map_servo_ids_do_not_match(tmp_path: Path) -> None:
    map_path = _build_map_for_demo(tmp_path)
    # Tamper the map to claim servo IDs we don't have at runtime.
    payload = json.loads(map_path.read_text(encoding="utf-8"))
    payload["metadata"]["commanded_servo_ids"] = [1, 2, 3, 4]  # only 4
    map_path.write_text(json.dumps(payload), encoding="utf-8")
    runner = _runner_with_penprobe(tmp_path)
    result = runner.run_experiment(
        EXPERIMENT_NAME,
        config={"map_path": str(map_path), "max_iterations": 1, "max_duration_s": 1.0, "control_rate_hz": 5.0},
    )
    assert result.success is False
    assert "Map/runtime mismatch" in result.message


def test_demo_runs_and_writes_summary_when_target_valid(tmp_path: Path) -> None:
    map_path = _build_map_for_demo(tmp_path)
    runner = _runner_with_penprobe(tmp_path)
    result = runner.run_experiment(
        EXPERIMENT_NAME,
        config={
            "map_path": str(map_path),
            "max_iterations": 3,
            "max_duration_s": 5.0,
            "control_rate_hz": 5.0,
            "command_update_deadband_mm": 0.0,
            "min_target_motion_mm": 0.0,
            "allow_servo_only_test_run": True,
        },
    )
    assert result.success is True
    metrics = result.summary.experiment_metrics
    # Demo-only flags are stamped on every run.
    assert metrics["demo_only"] is True
    assert metrics["not_closed_loop_validated"] is True
    assert metrics["valid_for_model_training"] is False
    assert metrics["valid_for_thesis_repeatability"] is False
    assert metrics["target_tool_id"] == "0B"
    assert metrics["controlled_point"].startswith("map distal pose")
    assert metrics["target_point"].startswith("0B tool origin")
    assert metrics["iterations"] >= 1
    # Files written: summary, trace csv, jsonl, map metadata copy.
    out = result.paths.output_dir
    assert (out / "two_segment_penprobe_lookup_demo_summary.txt").exists()
    assert (out / "demo_trace.csv").exists()
    assert (out / "demo_trace.jsonl").exists()
    assert (out / "map_used.json").exists()
    summary_text = (out / "two_segment_penprobe_lookup_demo_summary.txt").read_text(encoding="utf-8")
    assert "DEMO ONLY" in summary_text
    assert "not_closed_loop_validated: True" in summary_text


def test_demo_skips_command_when_tracker_stale(tmp_path: Path) -> None:
    map_path = _build_map_for_demo(tmp_path)
    settings = _settings()
    settings.robot.bottom_segment_key = "segment_b"
    settings.robot.top_segment_key = "segment_a"
    settings.robot.physical_assembly_confirmed_by_operator = True
    service = _servo_service(tmp_path, settings=settings)
    _save_all8_startup(service)
    # Build a stale snapshot (no 0B) so the demo cannot resolve a target.
    from continuum_robot.services.models import ToolTrackingSnapshot

    snapshot = _tracking_snapshot(include_distal=True, stale=True)
    snapshot.T_robot_aurora = [
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1],
    ]
    tracking_service = _FakeTrackingService(snapshot)
    runner = _runner(tmp_path, settings=settings, service=service, tracking_service=tracking_service)
    result = runner.run_experiment(
        EXPERIMENT_NAME,
        config={
            "map_path": str(map_path),
            "max_iterations": 2,
            "max_duration_s": 5.0,
            "control_rate_hz": 5.0,
            "allow_servo_only_test_run": True,
        },
    )
    assert result.success is True
    metrics = result.summary.experiment_metrics
    # No commands should have been sent because the target was unreadable/stale.
    assert metrics["commands_sent"] == 0
    # Trace should be populated with skip reasons.
    trace_path = result.paths.output_dir / "demo_trace.jsonl"
    rows = [json.loads(line) for line in trace_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert rows, "trace should not be empty"
    assert any(row.get("block_reason") for row in rows)


def test_demo_marks_itself_demo_only_in_summary_json_and_run_validity(tmp_path: Path) -> None:
    """Every demo run is loudly labelled demo-only across summary + summary.json."""
    map_path = _build_map_for_demo(tmp_path)
    runner = _runner_with_penprobe(tmp_path)
    result = runner.run_experiment(
        EXPERIMENT_NAME,
        config={
            "map_path": str(map_path),
            "max_iterations": 2,
            "max_duration_s": 3.0,
            "control_rate_hz": 5.0,
            "allow_servo_only_test_run": True,
        },
    )
    assert result.success is True
    summary_json = json.loads((result.paths.output_dir / "summary.json").read_text(encoding="utf-8"))
    metrics = summary_json["experiment_metrics"]
    assert metrics["demo_only"] is True
    assert metrics["not_closed_loop_validated"] is True
    assert metrics["valid_for_model_training"] is False
    assert metrics["valid_for_thesis_repeatability"] is False
    assert metrics["physical_tip_chasing"] is False


def test_demo_refuses_to_run_without_map_path(tmp_path: Path) -> None:
    runner = _runner_with_penprobe(tmp_path)
    result = runner.run_experiment(EXPERIMENT_NAME, config={})
    assert result.success is False
    assert "map_path" in result.message

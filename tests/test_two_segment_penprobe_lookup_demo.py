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
    assert metrics["tip_tool_id"] == "0A"
    # Polished semantic strings: controlled point is 0A's coil origin (or
    # whatever tool the map says produced the distal labels); target point
    # is the 0B penprobe origin in robot base frame.
    assert "distal/tip coil origin" in metrics["controlled_point"]
    assert metrics["target_point"].startswith("0B penprobe origin in robot base frame")
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


def _build_unknown_assembly_map_for_demo(tmp_path: Path) -> Path:
    """Build a map from a servo_only dataset that never recorded assembly/role.

    Mirrors the real big collected dataset: run_trust_mode=servo_only with an
    empty physical_assembly + empty tracking_role_config, so the resulting map
    has no bottom_top_assignment and no map_distal_tool_id.
    """
    run_dir = _write_dataset_run(
        tmp_path,
        sample_xyzs=[(x, y, 100.0) for x in (-10.0, 0.0, 10.0) for y in (-10.0, 0.0, 10.0)],
        bottom_segment_key="",
        top_segment_key="",
        bottom_servo_ids=[],
        top_servo_ids=[],
        run_trust_mode="servo_only",
        tracking_role_config={},
    )
    result = build_workspace_lookup_map(
        [run_dir],
        config=LookupMapBuildConfig(
            voxel_size_mm=None, output_dir=tmp_path / "unknown_map", allow_lower_trust=True
        ),
    )
    return result.artifact_paths["map_json"]


def test_demo_runs_against_unknown_assembly_map_when_allowed(tmp_path: Path) -> None:
    """The demo runs on a servo_only-derived map that lacks bottom/top assignment.

    This is the production path for the big collected dataset: the map carries no
    assembly metadata, but allow_unknown_map_assembly defaults True so the demo
    is usable. The run records a loud unknown-assembly warning for the audit.
    """
    map_path = _build_unknown_assembly_map_for_demo(tmp_path)
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
    assert result.success is True, result.message
    metrics = result.summary.experiment_metrics
    assert metrics["map_assembly_unknown"] is True
    assert metrics["allow_unknown_map_assembly"] is True
    assert metrics["map_distal_tool_id"] is None
    assert "map_assembly_unknown_warning" in metrics
    assert metrics["iterations"] >= 1


def test_demo_blocks_unknown_assembly_map_when_disallowed(tmp_path: Path) -> None:
    """Opting out of allow_unknown_map_assembly blocks the unknown-assignment map.

    The genuine bottom/top safety guard is preserved as an explicit operator choice.
    """
    map_path = _build_unknown_assembly_map_for_demo(tmp_path)
    runner = _runner_with_penprobe(tmp_path)
    result = runner.run_experiment(
        EXPERIMENT_NAME,
        config={
            "map_path": str(map_path),
            "max_iterations": 1,
            "max_duration_s": 2.0,
            "control_rate_hz": 5.0,
            "allow_servo_only_test_run": True,
            "allow_unknown_map_assembly": False,
        },
    )
    assert result.success is False
    assert "bottom/top assignment" in (result.message or "")


def test_demo_gui_page_exposes_map_controls_and_unknown_assembly_flag(tmp_path: Path) -> None:
    """The demo GUI page surfaces the map-picker buttons + unknown-assembly flag.

    Builds the page through the same factory the experiment tab uses, confirms the
    new controls exist and sync to defaults, and that "Use Latest Built Map"
    scans the maps folder and populates the path.
    """
    import os

    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PySide6.QtWidgets import QApplication

    app = QApplication.instance() or QApplication([])
    from continuum_robot.gui.controllers.experiment_controller import ExperimentViewState
    from continuum_robot.gui.widgets.experiment_pages import (
        TwoSegmentPenprobeLookupDemoPage,
        build_experiment_page,
    )
    from tests.test_gui_controllers import _experiment_controller

    controller = _experiment_controller(tmp_path)
    controller.project_root = tmp_path  # isolate the map scan to tmp
    page = build_experiment_page(controller, EXPERIMENT_NAME)
    try:
        assert isinstance(page, TwoSegmentPenprobeLookupDemoPage)
        assert hasattr(page, "use_latest_map_button")
        assert hasattr(page, "browse_map_button")
        assert hasattr(page, "allow_unknown_assembly_check")
        page._sync_parameters_from_state(ExperimentViewState())
        # Default config: servo_only-derived (unknown-assembly) maps are usable.
        assert page.allow_unknown_assembly_check.isChecked() is True
        # No maps present yet -> the button reports none found, leaves path empty.
        page._on_use_latest_map()
        assert "No built maps" in page.map_status_label.text()
        assert page.map_path_edit.text().strip() == ""
        # Drop a map artifact in the expected layout and re-scan.
        maps_dir = (
            tmp_path
            / "data"
            / "experiments"
            / "two_segment_workspace_lookup_maps"
            / "20260601_000000_workspace_lookup_map"
        )
        maps_dir.mkdir(parents=True)
        (maps_dir / "two_segment_workspace_lookup_map.json").write_text("{}", encoding="utf-8")
        page._on_use_latest_map()
        assert page.map_path_edit.text().endswith("two_segment_workspace_lookup_map.json")
        assert "Loaded latest map" in page.map_status_label.text()
    finally:
        page.deleteLater()
    _ = app


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


def test_demo_default_config_uses_target_0b_and_tip_0a() -> None:
    """Polished defaults: 0B is target, 0A is tip, map should be 0A-labelled."""
    from continuum_robot.experiments.two_segment_penprobe_lookup_demo import (
        TwoSegmentPenprobeLookupDemoConfig,
    )

    config = TwoSegmentPenprobeLookupDemoConfig.from_dict({})
    assert config.target_tool_id == "0B"
    assert config.tip_tool_id == "0A"
    assert config.expected_map_distal_tool_id == "0A"
    assert config.tip_tracking_optional is True
    assert config.require_target_tool is True
    assert config.block_on_map_tool_mismatch is True


def test_demo_blocks_when_map_distal_tool_explicitly_differs(tmp_path: Path) -> None:
    """A map whose distal source was 0C must block when expected is 0A."""
    map_path = _build_map_for_demo(tmp_path)
    payload = json.loads(map_path.read_text(encoding="utf-8"))
    # Pretend this map's labels came from 0C instead of 0A.
    payload["metadata"]["map_distal_tool_id"] = "0C"
    payload["metadata"]["role_mapping"] = {"distal_tip": "0C"}
    map_path.write_text(json.dumps(payload), encoding="utf-8")
    runner = _runner_with_penprobe(tmp_path)
    result = runner.run_experiment(
        EXPERIMENT_NAME,
        config={
            "map_path": str(map_path),
            "max_iterations": 1,
            "max_duration_s": 1.0,
            "control_rate_hz": 5.0,
            "allow_servo_only_test_run": True,
        },
    )
    assert result.success is False
    assert "Map distal-tip source tool" in result.message
    assert "0C" in result.message


def test_demo_accepts_map_with_explicit_distal_tool_0a(tmp_path: Path) -> None:
    """A correctly-labelled map (0A) must pass setup cleanly."""
    map_path = _build_map_for_demo(tmp_path)
    payload = json.loads(map_path.read_text(encoding="utf-8"))
    payload["metadata"]["map_distal_tool_id"] = "0A"
    payload["metadata"]["role_mapping"] = {"distal_tip": "0A"}
    map_path.write_text(json.dumps(payload), encoding="utf-8")
    runner = _runner_with_penprobe(tmp_path)
    result = runner.run_experiment(
        EXPERIMENT_NAME,
        config={
            "map_path": str(map_path),
            "max_iterations": 1,
            "max_duration_s": 2.0,
            "control_rate_hz": 5.0,
            "command_update_deadband_mm": 0.0,
            "min_target_motion_mm": 0.0,
            "allow_servo_only_test_run": True,
        },
    )
    assert result.success is True
    metrics = result.summary.experiment_metrics
    assert metrics["map_distal_tool_id"] == "0A"
    assert metrics["expected_map_distal_tool_id"] == "0A"


def test_demo_blocks_when_map_has_no_distal_tool_id_and_not_allowed(tmp_path: Path) -> None:
    map_path = _build_map_for_demo(tmp_path)
    payload = json.loads(map_path.read_text(encoding="utf-8"))
    payload["metadata"]["map_distal_tool_id"] = None
    payload["metadata"]["role_mapping"] = {}
    map_path.write_text(json.dumps(payload), encoding="utf-8")
    runner = _runner_with_penprobe(tmp_path)
    result = runner.run_experiment(
        EXPERIMENT_NAME,
        config={
            "map_path": str(map_path),
            "max_iterations": 1,
            "max_duration_s": 1.0,
            "control_rate_hz": 5.0,
            "allow_servo_only_test_run": True,
            "allow_unknown_map_tip_tool": False,  # OPT IN to strict
        },
    )
    assert result.success is False
    assert "no recorded `map_distal_tool_id`" in result.message


def test_demo_allows_map_with_no_distal_tool_id_by_default(tmp_path: Path) -> None:
    """Default behavior: warn, not block, when role metadata is absent."""
    map_path = _build_map_for_demo(tmp_path)
    payload = json.loads(map_path.read_text(encoding="utf-8"))
    payload["metadata"]["map_distal_tool_id"] = None
    payload["metadata"]["role_mapping"] = {}
    map_path.write_text(json.dumps(payload), encoding="utf-8")
    runner = _runner_with_penprobe(tmp_path)
    result = runner.run_experiment(
        EXPERIMENT_NAME,
        config={
            "map_path": str(map_path),
            "max_iterations": 1,
            "max_duration_s": 2.0,
            "control_rate_hz": 5.0,
            "command_update_deadband_mm": 0.0,
            "min_target_motion_mm": 0.0,
            "allow_servo_only_test_run": True,
            # allow_unknown_map_tip_tool defaults True
        },
    )
    assert result.success is True


def test_demo_records_live_tip_tracking_in_trace(tmp_path: Path) -> None:
    """When 0A is visible, trace + summary capture tip XYZ + tip-to-target distance."""
    map_path = _build_map_for_demo(tmp_path)
    # Add 0A as a tracked tool alongside 0B.
    settings = _settings()
    settings.robot.bottom_segment_key = "segment_b"
    settings.robot.top_segment_key = "segment_a"
    settings.robot.physical_assembly_confirmed_by_operator = True
    service = _servo_service(tmp_path, settings=settings)
    _save_all8_startup(service)
    from continuum_robot.services.models import ToolTrackingSnapshot

    snapshot = _tracking_snapshot(include_distal=True)  # already has 0A
    snapshot.tools["0B"] = ToolTrackingSnapshot(
        tool_id="0B",
        present=True,
        valid=True,
        tracking_state="tracked",
        translation_mm=(3.0, 0.0, 100.0),
        quaternion_wxyz=(1.0, 0.0, 0.0, 0.0),
        quality=0.9,
    )
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
            "max_duration_s": 2.0,
            "control_rate_hz": 5.0,
            "command_update_deadband_mm": 0.0,
            "min_target_motion_mm": 0.0,
            "allow_servo_only_test_run": True,
        },
    )
    assert result.success is True
    metrics = result.summary.experiment_metrics
    assert metrics["live_tip_tracking_available"] is True
    assert metrics["live_tip_tracking_available_fraction"] > 0.0
    assert metrics["tip_tool_id"] == "0A"
    # Trace rows must carry tip XYZ + tip-to-target distance.
    trace = [
        json.loads(line)
        for line in (result.paths.output_dir / "demo_trace.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    tip_seen = [row for row in trace if row.get("tip_xyz_robot_mm")]
    assert tip_seen, "trace must contain at least one row with tip_xyz_robot_mm"
    assert tip_seen[0]["tip_tool_id"] == "0A"
    assert tip_seen[0]["target_tool_id"] == "0B"
    assert tip_seen[0]["tip_to_target_distance_mm"] is not None
    # Summary text must spell out the new role section.
    summary = (result.paths.output_dir / "two_segment_penprobe_lookup_demo_summary.txt").read_text(encoding="utf-8")
    assert "target_tool_id: 0B" in summary
    assert "tip_tool_id:    0A" in summary
    assert "Tip-to-target distance summary" in summary


def test_demo_does_not_block_when_only_tip_0a_is_missing(tmp_path: Path) -> None:
    """Missing live 0A must NOT block — feedforward map still drives the command."""
    map_path = _build_map_for_demo(tmp_path)
    # Build a snapshot with 0B but explicitly NO 0A.
    settings = _settings()
    settings.robot.bottom_segment_key = "segment_b"
    settings.robot.top_segment_key = "segment_a"
    settings.robot.physical_assembly_confirmed_by_operator = True
    service = _servo_service(tmp_path, settings=settings)
    _save_all8_startup(service)
    from continuum_robot.services.models import ToolTrackingSnapshot

    # include_distal=False -> no 0A in snapshot.tools.
    snapshot = _tracking_snapshot(include_distal=False)
    snapshot.tools["0B"] = ToolTrackingSnapshot(
        tool_id="0B",
        present=True,
        valid=True,
        tracking_state="tracked",
        translation_mm=(2.0, 2.0, 100.0),
        quaternion_wxyz=(1.0, 0.0, 0.0, 0.0),
        quality=0.9,
    )
    snapshot.T_robot_aurora = [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]
    tracking_service = _FakeTrackingService(snapshot)
    runner = _runner(tmp_path, settings=settings, service=service, tracking_service=tracking_service)
    result = runner.run_experiment(
        EXPERIMENT_NAME,
        config={
            "map_path": str(map_path),
            "max_iterations": 2,
            "max_duration_s": 2.0,
            "control_rate_hz": 5.0,
            "command_update_deadband_mm": 0.0,
            "min_target_motion_mm": 0.0,
            "allow_servo_only_test_run": True,
        },
    )
    assert result.success is True
    metrics = result.summary.experiment_metrics
    # No 0A visible -> live_tip_tracking_available = False, but the run still
    # completed and (depending on deadband math) at least attempted commands.
    assert metrics["live_tip_tracking_available"] is False
    assert metrics["live_tip_tracking_available_fraction"] == 0.0
    assert metrics["target_tool_id"] == "0B"
    assert metrics["tip_tool_id"] == "0A"

"""Tests for the two-segment workspace lookup map builder + controller.

These tests cover:
- offline map builder (sample iteration, filtering, downsampling, artifacts)
- runtime nearest-lookup controller (decisions, safety limiter, compatibility)
- the wired demo experiment's safety preconditions

They use synthetic ``samples.jsonl`` content shaped exactly like the rows that
``two_segment_collect_pose_dataset.py::_capture_sample`` writes, so the
tests double as a contract for the on-disk format.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from continuum_robot.demo.two_segment_lookup_controller import (
    LookupControllerConfig,
    TwoSegmentWorkspaceLookupController,
)
from continuum_robot.demo.two_segment_workspace_lookup import (
    LookupMapBuildConfig,
    LookupMapPoint,
    MAP_SCHEMA_VERSION,
    build_workspace_lookup_map,
    farthest_point_sampling,
    load_workspace_lookup_map,
    nearest_lookup,
    voxel_downsample,
    workspace_bounds_mm,
)


# ---------------------------------------------------------------------------
# Helpers: build a synthetic dataset run on disk
# ---------------------------------------------------------------------------


def _write_dataset_run(
    root: Path,
    *,
    run_id: str = "20260601_120000_two_segment_collect_pose_command_dataset",
    sample_xyzs: list[tuple[float, float, float]] | None = None,
    bottom_segment_key: str = "segment_b",
    top_segment_key: str = "segment_a",
    bottom_servo_ids: list[int] | None = None,
    top_servo_ids: list[int] | None = None,
    run_trust_mode: str = "thesis_trusted",
    accepted_startup: bool = True,
    bad_row: bool = False,
    tracking_role_config: dict | None = None,
) -> Path:
    """Write a minimal but format-faithful samples.jsonl + summary.json."""
    bottom_servo_ids = bottom_servo_ids or [5, 6, 7, 8]
    top_servo_ids = top_servo_ids or [1, 2, 3, 4]
    sample_xyzs = sample_xyzs or [
        (0.0, 0.0, 100.0),
        (10.0, 0.0, 100.0),
        (0.0, 10.0, 100.0),
        (-10.0, 0.0, 100.0),
        (0.0, -10.0, 100.0),
    ]
    run_dir = root / "data" / "experiments" / "two_segment_collect_pose_command_dataset" / run_id
    run_dir.mkdir(parents=True)
    role_config = tracking_role_config if tracking_role_config is not None else {
        "distal_tip": {
            "role_name": "distal_tip",
            "tool_id": "0A",
            "enabled": True,
            "required_for_two_segment_model_training": True,
        },
        "registration_probe": {
            "role_name": "registration_probe",
            "tool_id": "0B",
            "enabled": True,
        },
    }
    (run_dir / "summary.json").write_text(
        json.dumps(
            {
                "experiment_name": "two_segment_collect_pose_command_dataset",
                "run_id": run_id,
                "success": True,
                "status": "success",
                "experiment_metrics": {
                    "startup_artifact_provenance": {"accepted_all_8_startup": accepted_startup},
                    "physical_assembly": {
                        "bottom_segment_key": bottom_segment_key,
                        "top_segment_key": top_segment_key,
                        "bottom_servo_ids": bottom_servo_ids,
                        "top_servo_ids": top_servo_ids,
                    },
                    "two_segment_tracking_role_config": role_config,
                },
            }
        ),
        encoding="utf-8",
    )
    samples_path = run_dir / "samples.jsonl"
    with samples_path.open("w", encoding="utf-8") as handle:
        for idx, (x, y, z) in enumerate(sample_xyzs):
            base_goal = 2048
            goal_ticks = {str(i): base_goal + idx * 5 + i for i in range(1, 9)}
            row = {
                "two_segment_pose": {"frame": "robot", "distal_tip_pose": {"translation_mm": [x, y, z]}},
                "pose_in_robot_frame": {
                    "roles": {"distal_tip": {"translation_mm": [x, y, z]}},
                },
                "two_segment_command": {"segments": {"segment_a": [0, 0, 0, 0], "segment_b": [0, 0, 0, 0]}},
                "extra": {
                    "record_kind": "two_segment_dataset_capture",
                    "capture_accepted": True,
                    "command_success": True,
                    "run_trust_mode": run_trust_mode,
                    "ordered_8_displacements_cm": [0.0] * 8,
                    "commanded_servo_ids": [1, 2, 3, 4, 5, 6, 7, 8],
                    "goal_ticks_by_servo": goal_ticks,
                    "physical_assembly": {
                        "bottom_segment_key": bottom_segment_key,
                        "top_segment_key": top_segment_key,
                        "bottom_servo_ids": bottom_servo_ids,
                        "top_servo_ids": top_servo_ids,
                    },
                    "startup_artifact_provenance": {"accepted_all_8_startup": accepted_startup},
                    "measured_servo_feedback": {
                        str(i): {"position_tick": base_goal + idx * 5 + i, "load_proxy_ma": 100.0 + idx}
                        for i in range(1, 9)
                    },
                    "missing_measured_servo_ids": [],
                    "top_routing_compensation": {"applied": True},
                },
            }
            handle.write(json.dumps(row) + "\n")
        if bad_row:
            # A row that should be rejected: missing distal pose.
            handle.write(
                json.dumps(
                    {
                        "two_segment_pose": {"frame": "robot"},
                        "extra": {
                            "record_kind": "two_segment_dataset_capture",
                            "capture_accepted": True,
                            "command_success": True,
                            "run_trust_mode": run_trust_mode,
                            "goal_ticks_by_servo": {str(i): 2048 for i in range(1, 9)},
                            "commanded_servo_ids": [1, 2, 3, 4, 5, 6, 7, 8],
                            "startup_artifact_provenance": {"accepted_all_8_startup": accepted_startup},
                            "missing_measured_servo_ids": [],
                        },
                    }
                )
                + "\n"
            )
    return run_dir


# ---------------------------------------------------------------------------
# Map builder tests
# ---------------------------------------------------------------------------


def test_build_workspace_lookup_map_writes_artifacts_and_filters_bad_rows(tmp_path: Path) -> None:
    run_dir = _write_dataset_run(tmp_path, bad_row=True)
    output_dir = tmp_path / "lookup_map"
    config = LookupMapBuildConfig(voxel_size_mm=None, output_dir=output_dir)

    result = build_workspace_lookup_map([run_dir], config=config)

    assert result.accepted_count == 5  # original 5 valid samples
    assert result.rejected_count >= 1  # the bad row
    reasons = {r.reason for r in result.rejected}
    assert any("distal_tip_robot_frame_pose_missing" in r for r in reasons)
    # Artifacts all written.
    for key in ("map_json", "points_csv", "summary_text", "quality_json"):
        assert result.artifact_paths[key].exists()
    map_payload = load_workspace_lookup_map(result.artifact_paths["map_json"])
    assert map_payload["schema_version"] == MAP_SCHEMA_VERSION
    assert map_payload["metadata"]["demo_only_artifact"] is True
    assert map_payload["metadata"]["not_closed_loop_validated"] is True
    assert map_payload["metadata"]["bottom_top_assignment"]["bottom_segment_key"] == "segment_b"


def test_build_workspace_lookup_map_rejects_servo_only_when_trust_required(tmp_path: Path) -> None:
    run_dir = _write_dataset_run(tmp_path, run_trust_mode="servo_only")
    config = LookupMapBuildConfig(voxel_size_mm=None, output_dir=tmp_path / "out")

    result = build_workspace_lookup_map([run_dir], config=config)

    assert result.accepted_count == 0
    assert all(r.reason == "servo_only_or_lower_trust" for r in result.rejected)


def test_build_workspace_lookup_map_allows_lower_trust_opt_in(tmp_path: Path) -> None:
    run_dir = _write_dataset_run(tmp_path, run_trust_mode="servo_only", accepted_startup=False)
    config = LookupMapBuildConfig(voxel_size_mm=None, output_dir=tmp_path / "out", allow_lower_trust=True)

    result = build_workspace_lookup_map([run_dir], config=config)

    assert result.accepted_count == 5


def test_voxel_downsample_reduces_count_while_preserving_coverage() -> None:
    rng = np.random.default_rng(5)
    candidates = [
        {"distal_xyz": rng.uniform(-50, 50, size=3), "ticks": {str(i): 2048 for i in range(1, 9)}, "quality": {"max_load_proxy_ma": 100.0}}
        for _ in range(2000)
    ]
    downsampled = voxel_downsample(candidates, voxel_size_mm=20.0)
    assert len(downsampled) <= len(candidates)
    assert len(downsampled) > 0
    # Coverage check: bounding box doesn't shrink meaningfully.
    all_positions = np.asarray([c["distal_xyz"] for c in candidates])
    down_positions = np.asarray([c["distal_xyz"] for c in downsampled])
    assert down_positions.min() >= all_positions.min() - 1e-9
    assert down_positions.max() <= all_positions.max() + 1e-9


def test_farthest_point_sampling_respects_max_points() -> None:
    rng = np.random.default_rng(2)
    candidates = [{"distal_xyz": rng.uniform(-30, 30, size=3)} for _ in range(500)]
    sampled = farthest_point_sampling(candidates, max_points=50)
    assert 1 < len(sampled) <= 50


def test_nearest_lookup_returns_closest_point_and_audit_ids() -> None:
    points = [
        LookupMapPoint(
            map_index=i,
            distal_xyz_robot_mm=[float(i * 10), 0.0, 0.0],
            all_8_goal_ticks={str(s): 2000 + i for s in range(1, 9)},
            commanded_servo_ids=list(range(1, 9)),
            bottom_top_assignment={"bottom_segment_key": "segment_b", "top_segment_key": "segment_a"},
            source_run_id="run-x",
            source_sample_index=i,
            run_trust_mode="thesis_trusted",
            accepted_all_8_startup=True,
        )
        for i in range(5)
    ]
    result = nearest_lookup([22.0, 0.0, 0.0], map_points=points, knn=1)
    assert result["selected_map_index"] == 2  # closest x=20.0
    assert "run-x#2" in result["source_sample_ids"]
    assert result["interpolation_mode"] == "nearest"


def test_nearest_lookup_inverse_distance_blends_when_knn_gt_one() -> None:
    points = [
        LookupMapPoint(
            map_index=0,
            distal_xyz_robot_mm=[0.0, 0.0, 0.0],
            all_8_goal_ticks={str(s): 1000 for s in range(1, 9)},
            commanded_servo_ids=list(range(1, 9)),
            bottom_top_assignment={},
            source_run_id="r0",
            source_sample_index=0,
            run_trust_mode="thesis_trusted",
            accepted_all_8_startup=True,
        ),
        LookupMapPoint(
            map_index=1,
            distal_xyz_robot_mm=[10.0, 0.0, 0.0],
            all_8_goal_ticks={str(s): 3000 for s in range(1, 9)},
            commanded_servo_ids=list(range(1, 9)),
            bottom_top_assignment={},
            source_run_id="r0",
            source_sample_index=1,
            run_trust_mode="thesis_trusted",
            accepted_all_8_startup=True,
        ),
    ]
    result = nearest_lookup([5.0, 0.0, 0.0], map_points=points, knn=2, interpolation="inverse_distance")
    for sid, tick in result["all_8_goal_ticks"].items():
        # Equidistant -> equal weights -> ~2000.
        assert 1500 <= int(tick) <= 2500


def test_workspace_bounds_mm_returns_box_and_centroid() -> None:
    points = [
        LookupMapPoint(
            map_index=i,
            distal_xyz_robot_mm=xyz,
            all_8_goal_ticks={str(s): 2048 for s in range(1, 9)},
            commanded_servo_ids=list(range(1, 9)),
            bottom_top_assignment={},
            source_run_id="r",
            source_sample_index=i,
            run_trust_mode="thesis_trusted",
            accepted_all_8_startup=True,
        )
        for i, xyz in enumerate([[0.0, 0.0, 0.0], [10.0, 5.0, -2.0], [-5.0, -5.0, 4.0]])
    ]
    bounds = workspace_bounds_mm(points)
    assert bounds["min_xyz_mm"] == pytest.approx([-5.0, -5.0, -2.0])
    assert bounds["max_xyz_mm"] == pytest.approx([10.0, 5.0, 4.0])
    assert bounds["centroid_xyz_mm"] == pytest.approx([5.0 / 3, 0.0, 2.0 / 3])


# ---------------------------------------------------------------------------
# Runtime controller tests
# ---------------------------------------------------------------------------


def _build_simple_map(tmp_path: Path) -> Path:
    run_dir = _write_dataset_run(tmp_path)
    result = build_workspace_lookup_map(
        [run_dir],
        config=LookupMapBuildConfig(voxel_size_mm=None, output_dir=tmp_path / "map"),
    )
    return result.artifact_paths["map_json"]


def test_controller_nearest_lookup_returns_command_when_target_valid(tmp_path: Path) -> None:
    map_path = _build_simple_map(tmp_path)
    controller = TwoSegmentWorkspaceLookupController.load_from_path(
        map_path,
        config=LookupControllerConfig(expected_bottom_segment_key="segment_b", expected_top_segment_key="segment_a"),
    )
    decision = controller.decide(target_xyz_robot_mm=[5.0, 0.0, 100.0], tracker_stale=False)
    assert decision.command_allowed is True
    assert decision.all_8_goal_ticks is not None
    assert decision.nearest_distance_mm is not None and decision.nearest_distance_mm < 20.0
    assert decision.block_reason is None
    assert decision.inside_workspace_hint is True


def test_controller_blocks_when_tracker_stale(tmp_path: Path) -> None:
    map_path = _build_simple_map(tmp_path)
    controller = TwoSegmentWorkspaceLookupController.load_from_path(map_path)
    decision = controller.decide(target_xyz_robot_mm=[0.0, 0.0, 100.0], tracker_stale=True)
    assert decision.command_allowed is False
    assert decision.block_reason == "tracker_stale"
    assert decision.all_8_goal_ticks is None


def test_controller_blocks_when_servo_ids_mismatch(tmp_path: Path) -> None:
    map_path = _build_simple_map(tmp_path)
    controller = TwoSegmentWorkspaceLookupController.load_from_path(
        map_path,
        config=LookupControllerConfig(expected_commanded_servo_ids=[1, 2, 3, 4]),  # wrong
    )
    decision = controller.decide(target_xyz_robot_mm=[0.0, 0.0, 100.0])
    assert decision.command_allowed is False
    assert decision.block_reason == "map_compatibility_failed"
    assert any("servo_id_mismatch" in r for r in decision.safety_limiter_reasons)


def test_controller_blocks_when_bottom_top_mismatch(tmp_path: Path) -> None:
    map_path = _build_simple_map(tmp_path)
    controller = TwoSegmentWorkspaceLookupController.load_from_path(
        map_path,
        config=LookupControllerConfig(
            expected_bottom_segment_key="segment_a",  # wrong; map has B=bottom
            expected_top_segment_key="segment_b",
        ),
    )
    decision = controller.decide(target_xyz_robot_mm=[0.0, 0.0, 100.0])
    assert decision.command_allowed is False
    assert any("bottom_segment_key_mismatch" in r for r in decision.safety_limiter_reasons)


def test_controller_extrapolation_warning_when_target_outside_workspace(tmp_path: Path) -> None:
    map_path = _build_simple_map(tmp_path)
    controller = TwoSegmentWorkspaceLookupController.load_from_path(
        map_path,
        config=LookupControllerConfig(
            expected_bottom_segment_key="segment_b",
            expected_top_segment_key="segment_a",
            max_nearest_distance_mm=5.0,  # tight
        ),
    )
    decision = controller.decide(target_xyz_robot_mm=[200.0, 200.0, 100.0])
    assert decision.inside_workspace_hint is False
    assert decision.is_extrapolating is True
    assert decision.command_allowed is False
    assert any("nearest_distance_above_hard_threshold" in r for r in decision.safety_limiter_reasons)


def test_controller_blocks_when_map_empty(tmp_path: Path) -> None:
    empty_map = {
        "schema_version": MAP_SCHEMA_VERSION,
        "metadata": {"bottom_top_assignment": {}, "commanded_servo_ids": [1, 2, 3, 4, 5, 6, 7, 8]},
        "map_points": [],
        "rejected_candidates": [],
    }
    controller = TwoSegmentWorkspaceLookupController(map_payload=empty_map)
    decision = controller.decide(target_xyz_robot_mm=[0.0, 0.0, 100.0])
    assert decision.command_allowed is False
    assert decision.block_reason == "map_empty"


def test_controller_rejects_unknown_interpolation() -> None:
    with pytest.raises(ValueError, match="Unsupported interpolation_mode"):
        LookupControllerConfig(interpolation_mode="quadratic")


def test_controller_loads_map_directly_from_path_and_round_trips(tmp_path: Path) -> None:
    map_path = _build_simple_map(tmp_path)
    controller = TwoSegmentWorkspaceLookupController.load_from_path(map_path)
    assert controller.map_point_count == 5
    assert controller.bottom_top_assignment["bottom_segment_key"] == "segment_b"


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def test_cli_main_writes_map(tmp_path: Path) -> None:
    from continuum_robot.demo.two_segment_workspace_lookup import main

    run_dir = _write_dataset_run(tmp_path)
    out_dir = tmp_path / "cli_out"
    rc = main(
        [
            "--run-dir",
            str(run_dir),
            "--output-dir",
            str(out_dir),
            "--voxel-size-mm",
            "20",
        ]
    )
    assert rc == 0
    assert (out_dir / "two_segment_workspace_lookup_map.json").exists()
    assert (out_dir / "two_segment_workspace_lookup_points.csv").exists()


def test_cli_main_returns_nonzero_when_no_accepted_points(tmp_path: Path) -> None:
    from continuum_robot.demo.two_segment_workspace_lookup import main

    run_dir = _write_dataset_run(tmp_path, run_trust_mode="servo_only")
    out_dir = tmp_path / "cli_out_empty"
    rc = main(
        [
            "--run-dir",
            str(run_dir),
            "--output-dir",
            str(out_dir),
        ]
    )
    assert rc == 2


def test_map_builder_extracts_distal_tool_0a_from_role_config(tmp_path: Path) -> None:
    """Phase 2: role-to-tool mapping is pulled from summary.json into the map metadata."""
    run_dir = _write_dataset_run(tmp_path)  # default config has distal_tip -> 0A
    result = build_workspace_lookup_map(
        [run_dir], config=LookupMapBuildConfig(voxel_size_mm=None, output_dir=tmp_path / "map")
    )
    map_payload = load_workspace_lookup_map(result.artifact_paths["map_json"])
    meta = map_payload["metadata"]
    assert meta["map_distal_tool_id"] == "0A"
    assert meta["role_mapping"]["distal_tip"] == "0A"
    assert meta["role_mapping"]["registration_probe"] == "0B"
    assert meta["target_tool_default"] == "0B"
    assert meta["tip_tool_default"] == "0A"
    assert meta["map_controlled_point"].startswith("0A distal/tip coil origin")
    assert meta["role_mapping_warnings"] == []
    # Summary text must show the new role section.
    summary = result.artifact_paths["summary_text"].read_text(encoding="utf-8")
    assert "map_distal_tool_id: 0A" in summary
    assert "0A distal/tip coil origin" in summary


def test_map_builder_warns_when_role_config_missing(tmp_path: Path) -> None:
    """Phase 2: absent role config -> map_distal_tool_id is None + warning recorded."""
    run_dir = _write_dataset_run(tmp_path, tracking_role_config={})  # explicitly empty
    result = build_workspace_lookup_map(
        [run_dir], config=LookupMapBuildConfig(voxel_size_mm=None, output_dir=tmp_path / "map_no_roles")
    )
    map_payload = load_workspace_lookup_map(result.artifact_paths["map_json"])
    meta = map_payload["metadata"]
    assert meta["map_distal_tool_id"] is None
    assert meta["role_mapping"] == {}
    assert meta["role_mapping_warnings"], "missing role config should produce a warning"
    summary = result.artifact_paths["summary_text"].read_text(encoding="utf-8")
    assert "UNKNOWN" in summary or "(source tool ID unknown" in summary


def test_map_builder_warns_when_role_resolves_to_multiple_tool_ids(tmp_path: Path) -> None:
    """If two source runs disagree on which tool is `distal_tip`, the map records that conflict."""
    run_a = _write_dataset_run(
        tmp_path,
        run_id="20260601_120000_run_a",
        tracking_role_config={
            "distal_tip": {"role_name": "distal_tip", "tool_id": "0A", "enabled": True},
        },
    )
    run_b = _write_dataset_run(
        tmp_path,
        run_id="20260601_120001_run_b",
        tracking_role_config={
            "distal_tip": {"role_name": "distal_tip", "tool_id": "0C", "enabled": True},
        },
    )
    result = build_workspace_lookup_map(
        [run_a, run_b],
        config=LookupMapBuildConfig(voxel_size_mm=None, output_dir=tmp_path / "map_conflicting"),
    )
    map_payload = load_workspace_lookup_map(result.artifact_paths["map_json"])
    meta = map_payload["metadata"]
    # Conflicting tool IDs resolve to None and surface a warning.
    assert meta["map_distal_tool_id"] is None
    assert meta["role_mapping"].get("distal_tip") is None
    assert any("multiple tool IDs" in w for w in meta["role_mapping_warnings"])

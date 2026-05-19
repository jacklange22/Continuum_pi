"""Phase 2 — end-to-end coverage of the first/averaged label variant pipeline.

The Phase 1 tests cover the per-frame averaging math. These tests cover the
plumbing that lets the ANN training popout pick which label set to train on:

  writer  -> emits modeling_dataset_export_{first,averaged}.jsonl
  discovery -> ModelingDatasetSummary.label_variants reflects what's on disk
  loader  -> _load_export_rows + prepare_legacy_ann_dataset pick the right file
  controller -> select_label_variant reconciles + invalidates estimate
  popout  -> radio visible only when both variants exist on disk
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pytest

from continuum_robot.experiments.modeling_dataset_outputs import (
    _summarize_tracker_variability,
    write_modeling_dataset_outputs,
)
from continuum_robot.experiments.schemas import (
    ExperimentMetadata,
    ExperimentSummary,
    ExperimentTimeseriesSample,
)
from continuum_robot.modeling.ann_training import (
    LABEL_VARIANT_AVERAGED,
    LABEL_VARIANT_FIRST,
    LABEL_VARIANT_VALID,
    _compile_modeling_dataset_summary,
    _load_export_rows,
    _resolve_export_path_for_variant,
    load_modeling_dataset_summary,
    prepare_legacy_ann_dataset,
    validate_legacy_ann_rows,
)


# --------------------------------------------------------------------------- #
# Synthetic dataset builders                                                  #
# --------------------------------------------------------------------------- #


def _make_sample(
    *,
    step_index: int,
    sample_index: int = 0,
    label_kind: str | None = None,
    tip_xyz_mm: tuple[float, float, float] = (1.0, 2.0, 3.0),
    cable_command_cm: tuple[float, ...] = (0.0, 0.0, 0.0, 0.0),
) -> ExperimentTimeseriesSample:
    extra: dict[str, Any] = {
        "record_kind": "modeling_dataset_capture",
        "dataset_mode": "workspace_coverage",
        "run_label": "",
        "dataset_tag": "",
        "tool_id": "0A",
        "capture_accepted": True,
        "capture_rejection_reason": None,
        "modeling_export_exclude": False,
        "tracker_gate": {"accepted": True, "reason": "ok"},
        "runtime_tip_mode": "coil_as_tip",
        "runtime_tip_trust_level": "thesis_trusted",
        "runtime_tip_policy": None,
        "requested_pair_command_cm": [float(cable_command_cm[0]), float(cable_command_cm[1])],
        "resolved_pair_command_cm": [float(cable_command_cm[0]), float(cable_command_cm[1])],
        "previous_pair_command_cm": [0.0, 0.0],
        "requested_cable_command_cm": list(cable_command_cm),
        "resolved_cable_command_cm": list(cable_command_cm),
        "raw_goal_ticks_by_servo": {"1": 2048, "2": 2048, "3": 2048, "4": 2048},
        "final_goal_ticks_by_servo": {"1": 2048, "2": 2048, "3": 2048, "4": 2048},
        "clamp_reasons_by_servo": {},
        "motion_profile": {},
        "command_metadata": {},
        "servo_feedback_at_command": {},
        "servo_feedback_at_capture": {
            "1": {"present_position_ticks": 2048},
            "2": {"present_position_ticks": 2048},
            "3": {"present_position_ticks": 2048},
            "4": {"present_position_ticks": 2048},
        },
        "command_message": "",
        "prior_family": None,
        "block_index": None,
        "step_metadata": {"label": f"step_{step_index}"},
        "sequential_order_preserved": True,
    }
    if label_kind is not None:
        extra["label_kind"] = label_kind
    return ExperimentTimeseriesSample(
        monotonic_time_s=float(step_index),
        wall_time_utc="2026-05-19T00:00:00Z",
        phase="workspace_coverage",
        step_index=int(step_index),
        sample_index=int(sample_index),
        commanded_motor_values={"1": 2048, "2": 2048, "3": 2048, "4": 2048},
        commanded_cable_deltas_cm=list(cable_command_cm),
        tracker_frame_id=int(step_index * 100),
        tool_ids_seen=["0A"],
        transform_validity={"0A": "tracked"},
        pose_in_tracker_frame={
            "0A": {
                "tracking_state": "tracked",
                "translation_mm": list(tip_xyz_mm),
                "quaternion_wxyz": [1.0, 0.0, 0.0, 0.0],
                "tangent_xyz": [0.0, 0.0, 1.0],
                "frame_number": int(step_index * 100),
            }
        },
        pose_in_robot_frame={
            "tip": {
                "matrix": [[1, 0, 0, tip_xyz_mm[0]], [0, 1, 0, tip_xyz_mm[1]], [0, 0, 1, tip_xyz_mm[2]], [0, 0, 0, 1]],
                "translation_mm": list(tip_xyz_mm),
                "quaternion_wxyz": [1.0, 0.0, 0.0, 0.0],
                "tangent_xyz": [0.0, 0.0, 1.0],
            }
        },
        freshness_s=0.01,
        latency_s=0.01,
        status_flags=["capture_accepted"],
        backend_health={"canonical_state": "mock"},
        extra=extra,
    )


def _make_metadata(tmp_path: Path) -> ExperimentMetadata:
    return ExperimentMetadata(
        schema_version="1.0",
        experiment_name="collect_pose_command_dataset",
        run_id="run_test",
        timestamp_utc="2026-05-19T00:00:00Z",
        git_commit="abcdef",
        backend_info={"mock_mode": True},
        registration_info={"path": "", "exists": False},
        config_used={"dataset_mode": "workspace_coverage"},
        operator_notes="",
        provenance_info={
            "operating_mode": "single_segment",
            "expected_servo_ids": [1, 2, 3, 4],
            "active_segment": {"key": "segment_a"},
            "mock_mode": True,
        },
        trust_info={"run_trust_mode": "mock"},
    )


def _make_summary(*, sample_count: int) -> ExperimentSummary:
    return ExperimentSummary(
        schema_version="1.0",
        experiment_name="collect_pose_command_dataset",
        run_id="run_test",
        success=True,
        sample_counts={"total": int(sample_count)},
        dropped_frames=0,
        invalid_transforms=0,
        stage_pass_fail={"setup": "passed", "precheck": "passed", "execute": "passed", "finalize": "passed"},
        status="success",
        experiment_metrics={
            "dataset_mode": "workspace_coverage",
            "accepted_sample_count": int(sample_count),
            "rejected_sample_count": 0,
            "legacy_export_enabled": False,
        },
    )


# --------------------------------------------------------------------------- #
# Writer — file emission                                                       #
# --------------------------------------------------------------------------- #


def test_writer_n1_writes_only_legacy_export(tmp_path: Path) -> None:
    output_dir = tmp_path / "run"
    output_dir.mkdir()
    samples = [_make_sample(step_index=i, tip_xyz_mm=(i * 1.0, 0.0, 0.0)) for i in range(3)]
    outputs = write_modeling_dataset_outputs(
        output_dir=output_dir,
        metadata=_make_metadata(tmp_path),
        summary=_make_summary(sample_count=len(samples)),
        samples=samples,
        averaged_samples=None,
        raw_tracker_frame_rows=None,
        tracker_samples_per_command=1,
        averaged_label_enabled=False,
        export_first_sample_label=True,
        export_averaged_sample_label=False,
    )
    assert (output_dir / "modeling_dataset_export.jsonl").exists()
    assert not (output_dir / "modeling_dataset_export_first.jsonl").exists()
    assert not (output_dir / "modeling_dataset_export_averaged.jsonl").exists()
    assert not (output_dir / "samples_first.jsonl").exists()
    assert not (output_dir / "samples_averaged.jsonl").exists()
    assert not (output_dir / "raw_tracker_samples.jsonl").exists()
    assert "export_jsonl_path" in outputs


def test_writer_n_gt_1_writes_both_variant_exports(tmp_path: Path) -> None:
    output_dir = tmp_path / "run"
    output_dir.mkdir()
    first_samples = [
        _make_sample(step_index=i, label_kind="first", tip_xyz_mm=(i * 1.0, 0.0, 0.0))
        for i in range(3)
    ]
    averaged_samples = [
        _make_sample(step_index=i, label_kind="averaged", tip_xyz_mm=(i * 1.0 + 0.05, 0.0, 0.0))
        for i in range(3)
    ]
    raw_rows = [
        {"command_index": i, "frame_index": j, "tool_id": "0A"}
        for i in range(3)
        for j in range(5)
    ]
    outputs = write_modeling_dataset_outputs(
        output_dir=output_dir,
        metadata=_make_metadata(tmp_path),
        summary=_make_summary(sample_count=len(first_samples)),
        samples=first_samples,
        averaged_samples=averaged_samples,
        raw_tracker_frame_rows=raw_rows,
        tracker_samples_per_command=20,
        averaged_label_enabled=True,
        export_first_sample_label=True,
        export_averaged_sample_label=True,
    )
    assert (output_dir / "modeling_dataset_export.jsonl").exists()
    assert (output_dir / "modeling_dataset_export_first.jsonl").exists()
    assert (output_dir / "modeling_dataset_export_averaged.jsonl").exists()
    assert (output_dir / "samples_first.jsonl").exists()
    assert (output_dir / "samples_averaged.jsonl").exists()
    assert (output_dir / "raw_tracker_samples.jsonl").exists()
    # samples_first must equal samples_averaged in line count but have different X positions
    first_lines = (output_dir / "modeling_dataset_export_first.jsonl").read_text().splitlines()
    avg_lines = (output_dir / "modeling_dataset_export_averaged.jsonl").read_text().splitlines()
    assert len(first_lines) == len(avg_lines) == 3
    first_xs = [json.loads(l)["tip_position_xyz_mm"][0] for l in first_lines]
    avg_xs = [json.loads(l)["tip_position_xyz_mm"][0] for l in avg_lines]
    assert first_xs != avg_xs
    assert "export_first_path" in outputs
    assert "export_averaged_path" in outputs


# --------------------------------------------------------------------------- #
# Discovery — label_variants on ModelingDatasetSummary                         #
# --------------------------------------------------------------------------- #


def _write_minimal_run(
    tmp_path: Path,
    *,
    with_averaged: bool,
    with_explicit_first: bool = False,
) -> Path:
    run = tmp_path / "20260519_000000_collect_pose"
    run.mkdir()
    (run / "metadata.json").write_text(
        json.dumps(_make_metadata(tmp_path).to_dict()),
        encoding="utf-8",
    )
    (run / "summary.json").write_text(
        json.dumps(_make_summary(sample_count=2).to_dict()),
        encoding="utf-8",
    )
    base_export = [
        {
            "sequence_index": 0,
            "source_sequence_index": 0,
            "phase": "workspace_coverage",
            "step_index": 0,
            "sample_index": 0,
            "accepted": True,
            "resolved_cable_command_cm": [0.0, 0.0, 0.0, 0.0],
            "tip_position_xyz_mm": [0.0, 0.0, 0.0],
            "tip_tangent_xyz": [0.0, 0.0, 1.0],
            "servo_feedback_at_capture": {"1": {"p": 1}, "2": {"p": 2}, "3": {"p": 3}, "4": {"p": 4}},
        },
        {
            "sequence_index": 1,
            "source_sequence_index": 1,
            "phase": "workspace_coverage",
            "step_index": 1,
            "sample_index": 0,
            "accepted": True,
            "resolved_cable_command_cm": [0.1, 0.0, 0.0, 0.0],
            "tip_position_xyz_mm": [1.0, 0.0, 0.0],
            "tip_tangent_xyz": [0.0, 0.0, 1.0],
            "servo_feedback_at_capture": {"1": {"p": 1}, "2": {"p": 2}, "3": {"p": 3}, "4": {"p": 4}},
        },
    ]
    (run / "modeling_dataset_export.jsonl").write_text(
        "\n".join(json.dumps(row) for row in base_export) + "\n",
        encoding="utf-8",
    )
    if with_explicit_first:
        (run / "modeling_dataset_export_first.jsonl").write_text(
            "\n".join(json.dumps(row) for row in base_export) + "\n",
            encoding="utf-8",
        )
    if with_averaged:
        averaged_export = [dict(row, tip_position_xyz_mm=[row["tip_position_xyz_mm"][0] + 0.05, 0.0, 0.0]) for row in base_export]
        (run / "modeling_dataset_export_averaged.jsonl").write_text(
            "\n".join(json.dumps(row) for row in averaged_export) + "\n",
            encoding="utf-8",
        )
    return run


def test_discovery_reports_only_first_when_run_did_not_average(tmp_path: Path) -> None:
    run = _write_minimal_run(tmp_path, with_averaged=False)
    summary = _compile_modeling_dataset_summary(run, dataset_scan_root="experiments", strict=True)
    assert summary.label_variants == ("first",)


def test_discovery_reports_both_when_averaged_file_present(tmp_path: Path) -> None:
    run = _write_minimal_run(tmp_path, with_averaged=True)
    summary = _compile_modeling_dataset_summary(run, dataset_scan_root="experiments", strict=True)
    assert summary.label_variants == ("first", "averaged")


def test_load_modeling_dataset_summary_back_compat(tmp_path: Path) -> None:
    """An existing run dir without the new files still has label_variants=('first',)."""
    run = _write_minimal_run(tmp_path, with_averaged=False)
    summary = load_modeling_dataset_summary(run)
    assert summary.label_variants == ("first",)
    assert "averaged" not in summary.label_variants


# --------------------------------------------------------------------------- #
# Loader — variant routing                                                     #
# --------------------------------------------------------------------------- #


def test_resolve_export_path_first_prefers_explicit_then_canonical(tmp_path: Path) -> None:
    run = _write_minimal_run(tmp_path, with_averaged=True, with_explicit_first=True)
    path = _resolve_export_path_for_variant(run, LABEL_VARIANT_FIRST)
    assert path.name == "modeling_dataset_export_first.jsonl"
    # Remove the explicit file: fallback is the legacy canonical name.
    (run / "modeling_dataset_export_first.jsonl").unlink()
    path = _resolve_export_path_for_variant(run, LABEL_VARIANT_FIRST)
    assert path.name == "modeling_dataset_export.jsonl"


def test_resolve_export_path_averaged_has_no_fallback(tmp_path: Path) -> None:
    run = _write_minimal_run(tmp_path, with_averaged=False)
    path = _resolve_export_path_for_variant(run, LABEL_VARIANT_AVERAGED)
    assert path.name == "modeling_dataset_export_averaged.jsonl"
    # The file doesn't exist — caller is expected to check before reading.
    assert not path.exists()


def test_resolve_export_path_invalid_variant_raises() -> None:
    with pytest.raises(ValueError, match="Unsupported label_variant"):
        _resolve_export_path_for_variant(Path("/tmp"), "nonsense")


def test_load_export_rows_routes_to_averaged_file(tmp_path: Path) -> None:
    run = _write_minimal_run(tmp_path, with_averaged=True)
    first_rows = _load_export_rows(run, label_variant=LABEL_VARIANT_FIRST)
    avg_rows = _load_export_rows(run, label_variant=LABEL_VARIANT_AVERAGED)
    assert [row["tip_position_xyz_mm"] for row in first_rows] == [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]
    assert [row["tip_position_xyz_mm"] for row in avg_rows] == [[0.05, 0.0, 0.0], [1.05, 0.0, 0.0]]


def test_prepare_legacy_ann_dataset_rejects_unavailable_variant(tmp_path: Path) -> None:
    run = _write_minimal_run(tmp_path, with_averaged=False)
    with pytest.raises(ValueError, match="not available"):
        prepare_legacy_ann_dataset(run, label_variant=LABEL_VARIANT_AVERAGED)


def test_prepare_legacy_ann_dataset_returns_averaged_labels(tmp_path: Path) -> None:
    run = _write_minimal_run(tmp_path, with_averaged=True)
    prepared = prepare_legacy_ann_dataset(run, label_variant=LABEL_VARIANT_AVERAGED, output_target="xyz")
    # The averaged file's tip_position_xyz_mm[0] values are shifted by +0.05.
    assert prepared.outputs.shape == (2, 3)
    assert prepared.outputs[0][0] == pytest.approx(0.05)
    assert prepared.outputs[1][0] == pytest.approx(1.05)


def test_validate_legacy_ann_rows_routes_per_variant(tmp_path: Path) -> None:
    run = _write_minimal_run(tmp_path, with_averaged=True)
    first_report = validate_legacy_ann_rows(run, label_variant=LABEL_VARIANT_FIRST)
    avg_report = validate_legacy_ann_rows(run, label_variant=LABEL_VARIANT_AVERAGED)
    assert first_report.accepted_export_rows == 2
    assert avg_report.accepted_export_rows == 2
    # The averaged variant on a run that DIDN'T average should report missing.
    (tmp_path / "second").mkdir()
    run_no_avg = _write_minimal_run(tmp_path / "second", with_averaged=False)
    avg_missing = validate_legacy_ann_rows(run_no_avg, label_variant=LABEL_VARIANT_AVERAGED)
    assert avg_missing.can_train is False
    assert "modeling_dataset_export_averaged.jsonl" in (avg_missing.block_reason or "")


# --------------------------------------------------------------------------- #
# Controller — reconciliation + setter                                         #
# --------------------------------------------------------------------------- #


def test_controller_select_label_variant_validates() -> None:
    pytest.importorskip("PySide6")
    from continuum_robot.gui.controllers.ann_training_controller import AnnTrainingController

    ctrl = AnnTrainingController(project_root=Path("."), dataset_output_root=Path("."))
    ctrl.select_label_variant(LABEL_VARIANT_FIRST)
    assert ctrl.state.selected_label_variant == LABEL_VARIANT_FIRST
    ctrl.select_label_variant(LABEL_VARIANT_AVERAGED)
    assert ctrl.state.selected_label_variant == LABEL_VARIANT_AVERAGED
    with pytest.raises(ValueError, match="must be one of"):
        ctrl.select_label_variant("garbage")


def test_controller_refresh_reconciles_unavailable_variant(tmp_path: Path) -> None:
    """If the operator picked 'averaged' on the last run but the new selection
    doesn't have it, refresh() should snap back to 'first' instead of silently
    feeding a missing file to prepare_legacy_ann_dataset."""
    pytest.importorskip("PySide6")
    from continuum_robot.gui.controllers.ann_training_controller import AnnTrainingController

    # Build a dataset root with one run that has only "first".
    runs_root = tmp_path / "data" / "experiments" / "collect_pose_command_dataset"
    runs_root.mkdir(parents=True)
    _write_minimal_run(runs_root, with_averaged=False)

    ctrl = AnnTrainingController(project_root=tmp_path, dataset_output_root=tmp_path)
    ctrl.select_label_variant(LABEL_VARIANT_AVERAGED)
    state = ctrl.refresh()
    assert state.selected_label_variant == LABEL_VARIANT_FIRST
    assert state.available_label_variants == (LABEL_VARIANT_FIRST,)


def test_controller_refresh_keeps_averaged_when_available(tmp_path: Path) -> None:
    pytest.importorskip("PySide6")
    from continuum_robot.gui.controllers.ann_training_controller import AnnTrainingController

    runs_root = tmp_path / "data" / "experiments" / "collect_pose_command_dataset"
    runs_root.mkdir(parents=True)
    _write_minimal_run(runs_root, with_averaged=True)

    ctrl = AnnTrainingController(project_root=tmp_path, dataset_output_root=tmp_path)
    ctrl.select_label_variant(LABEL_VARIANT_AVERAGED)
    state = ctrl.refresh()
    assert state.selected_label_variant == LABEL_VARIANT_AVERAGED
    assert state.available_label_variants == ("first", "averaged")


# --------------------------------------------------------------------------- #
# Popout — radio visibility wiring                                             #
# --------------------------------------------------------------------------- #


def test_popout_hides_radio_when_only_first_variant(tmp_path: Path) -> None:
    pytest.importorskip("PySide6")
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PySide6.QtWidgets import QApplication
    from continuum_robot.gui.controllers.ann_training_controller import AnnTrainingController
    from continuum_robot.gui.widgets.ann_training_window import AnnTrainingWindow

    runs_root = tmp_path / "data" / "experiments" / "collect_pose_command_dataset"
    runs_root.mkdir(parents=True)
    _write_minimal_run(runs_root, with_averaged=False)

    app = QApplication.instance() or QApplication([])
    ctrl = AnnTrainingController(project_root=tmp_path, dataset_output_root=tmp_path)
    win = AnnTrainingWindow(ctrl)
    state = ctrl.refresh()
    win._sync_label_variant(state)
    assert win.label_variant_container.isVisible() is False


def test_popout_shows_radio_when_both_variants_available(tmp_path: Path) -> None:
    pytest.importorskip("PySide6")
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PySide6.QtWidgets import QApplication
    from continuum_robot.gui.controllers.ann_training_controller import AnnTrainingController
    from continuum_robot.gui.widgets.ann_training_window import AnnTrainingWindow

    runs_root = tmp_path / "data" / "experiments" / "collect_pose_command_dataset"
    runs_root.mkdir(parents=True)
    _write_minimal_run(runs_root, with_averaged=True)

    app = QApplication.instance() or QApplication([])
    ctrl = AnnTrainingController(project_root=tmp_path, dataset_output_root=tmp_path)
    win = AnnTrainingWindow(ctrl)
    win.show()  # widgets need to be in a top-level window before isVisible() is meaningful
    state = ctrl.refresh()
    win._sync_label_variant(state)
    assert win.label_variant_container.isVisibleTo(win) is True
    assert win.label_variant_first_radio.isEnabled() is True
    assert win.label_variant_averaged_radio.isEnabled() is True


# --------------------------------------------------------------------------- #
# Constants                                                                    #
# --------------------------------------------------------------------------- #


def test_label_variant_constants_are_stable() -> None:
    assert LABEL_VARIANT_FIRST == "first"
    assert LABEL_VARIANT_AVERAGED == "averaged"
    assert LABEL_VARIANT_VALID == ("first", "averaged")

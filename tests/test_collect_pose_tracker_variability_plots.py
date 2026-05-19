"""Phase 3 — render-side tests for the 5 tracker-variability figures.

These don't pixel-diff the plots (matplotlib's pixel output is sensitive to
font / backend); they verify the writers produce non-trivial PNG bytes for
representative inputs, handle empty inputs without crashing, and the file
set is gated correctly on averaging being active. Pixel-level eyeballing
is done manually by the operator after a real run.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from continuum_robot.experiments.modeling_dataset_outputs import (
    _tracker_variability_records,
    _write_tracker_variability_first_vs_mean,
    _write_tracker_variability_sample_spread,
    _write_tracker_variability_std_histogram,
    _write_tracker_variability_std_vs_command_index,
    _write_tracker_variability_workspace_xy,
    write_modeling_dataset_outputs,
)
from continuum_robot.experiments.schemas import (
    ExperimentMetadata,
    ExperimentSummary,
    ExperimentTimeseriesSample,
)


_PLACEHOLDER_PNG_BYTES = 90  # the placeholder PNG written on render failure is < 100 bytes


# --------------------------------------------------------------------------- #
# Fixtures                                                                    #
# --------------------------------------------------------------------------- #


def _make_records(n: int, *, rng: np.random.Generator | None = None) -> list[dict[str, Any]]:
    rng = rng or np.random.default_rng(0)
    records: list[dict[str, Any]] = []
    for i in range(n):
        theta = rng.uniform(0, 2 * np.pi)
        r = rng.uniform(0.5, 10.0)
        x = float(r * np.cos(theta))
        y = float(r * np.sin(theta))
        std = float(abs(rng.normal(0.08 + 0.02 * (i / max(1, n - 1)), 0.02)))
        records.append(
            {
                "command_index": i,
                "averaged_x_mm": x,
                "averaged_y_mm": y,
                "averaged_z_mm": -150.0,
                "position_std_rms_mm": std,
                "position_max_deviation_mm": std * 2.5,
                "first_vs_mean_position_diff_mm": float(abs(rng.normal(0.05, 0.04))),
                "orientation_max_spread_deg": float(abs(rng.normal(0.1, 0.05))),
                "valid_sample_count": 20,
                "sample_window_s": 0.4,
            }
        )
    return records


def _make_raw_rows(records: list[dict[str, Any]], *, frames_per_command: int = 20) -> list[dict[str, Any]]:
    rng = np.random.default_rng(13)
    raw: list[dict[str, Any]] = []
    for record in records:
        for j in range(frames_per_command):
            rx = record["averaged_x_mm"] + rng.normal(0, record["position_std_rms_mm"])
            ry = record["averaged_y_mm"] + rng.normal(0, record["position_std_rms_mm"])
            raw.append(
                {
                    "command_index": int(record["command_index"]),
                    "frame_index": int(j),
                    "pose_in_robot_frame": {"translation_mm": [float(rx), float(ry), -150.0]},
                }
            )
    return raw


def _assert_real_png(path: Path) -> None:
    assert path.exists(), f"{path} not written"
    size = path.stat().st_size
    assert size > _PLACEHOLDER_PNG_BYTES, (
        f"{path} looks like a placeholder ({size} bytes) — render likely raised"
    )


# --------------------------------------------------------------------------- #
# Per-plot tests                                                              #
# --------------------------------------------------------------------------- #


def test_workspace_xy_renders_with_real_records(tmp_path: Path) -> None:
    records = _make_records(40)
    path = tmp_path / "ws_xy.png"
    _write_tracker_variability_workspace_xy(path=path, records=records)
    _assert_real_png(path)


def test_workspace_xy_renders_empty_records_without_crashing(tmp_path: Path) -> None:
    path = tmp_path / "ws_xy_empty.png"
    _write_tracker_variability_workspace_xy(path=path, records=[])
    _assert_real_png(path)


def test_std_histogram_renders_with_real_records(tmp_path: Path) -> None:
    records = _make_records(40)
    path = tmp_path / "hist.png"
    _write_tracker_variability_std_histogram(path=path, records=records)
    _assert_real_png(path)


def test_std_histogram_empty_input(tmp_path: Path) -> None:
    path = tmp_path / "hist_empty.png"
    _write_tracker_variability_std_histogram(path=path, records=[])
    _assert_real_png(path)


def test_first_vs_mean_renders(tmp_path: Path) -> None:
    records = _make_records(40)
    path = tmp_path / "fvm.png"
    _write_tracker_variability_first_vs_mean(path=path, records=records)
    _assert_real_png(path)


def test_sample_spread_renders_with_three_picks(tmp_path: Path) -> None:
    records = _make_records(40)
    raw_rows = _make_raw_rows(records)
    path = tmp_path / "spread.png"
    _write_tracker_variability_sample_spread(path=path, records=records, raw_rows=raw_rows)
    _assert_real_png(path)


def test_sample_spread_empty_raw_rows(tmp_path: Path) -> None:
    records = _make_records(10)
    path = tmp_path / "spread_empty.png"
    _write_tracker_variability_sample_spread(path=path, records=records, raw_rows=[])
    _assert_real_png(path)


def test_std_vs_command_index_renders(tmp_path: Path) -> None:
    records = _make_records(40)
    path = tmp_path / "trend.png"
    _write_tracker_variability_std_vs_command_index(path=path, records=records)
    _assert_real_png(path)


def test_std_vs_command_index_small_n_skips_smoothing(tmp_path: Path) -> None:
    """When N < smoothing window, plot still renders without the rolling line."""
    records = _make_records(3)
    path = tmp_path / "trend_small.png"
    _write_tracker_variability_std_vs_command_index(path=path, records=records)
    _assert_real_png(path)


# --------------------------------------------------------------------------- #
# Records extractor                                                           #
# --------------------------------------------------------------------------- #


def test_records_extractor_skips_samples_missing_translation() -> None:
    samples = [
        ExperimentTimeseriesSample(
            monotonic_time_s=0.0,
            wall_time_utc="x",
            phase="workspace_coverage",
            step_index=0,
            sample_index=0,
            pose_in_robot_frame={"tip": {"translation_mm": [1.0, 2.0, 3.0]}},
            extra={"tracker_averaging": {"position_std_rms_mm": 0.1}},
        ),
        ExperimentTimeseriesSample(
            monotonic_time_s=0.0,
            wall_time_utc="x",
            phase="workspace_coverage",
            step_index=1,
            sample_index=0,
            pose_in_robot_frame={"tip": {}},  # no translation -> skip
            extra={"tracker_averaging": {"position_std_rms_mm": 0.2}},
        ),
    ]
    records = _tracker_variability_records(samples)
    assert len(records) == 1
    assert records[0]["command_index"] == 0
    assert records[0]["averaged_x_mm"] == 1.0


# --------------------------------------------------------------------------- #
# End-to-end gating: writer emits the figure set only when averaging is on    #
# --------------------------------------------------------------------------- #


def _make_metadata() -> ExperimentMetadata:
    return ExperimentMetadata(
        schema_version="1.0",
        experiment_name="collect_pose_command_dataset",
        run_id="run_p3_test",
        timestamp_utc="2026-05-19T00:00:00Z",
        git_commit="abc",
        backend_info={"mock_mode": True},
        registration_info={},
        config_used={},
    )


def _make_summary(n: int) -> ExperimentSummary:
    return ExperimentSummary(
        schema_version="1.0",
        experiment_name="collect_pose_command_dataset",
        run_id="run_p3_test",
        success=True,
        sample_counts={"total": int(n)},
        dropped_frames=0,
        invalid_transforms=0,
        stage_pass_fail={"setup": "passed", "precheck": "passed", "execute": "passed", "finalize": "passed"},
        status="success",
        experiment_metrics={"legacy_export_enabled": False},
    )


def _make_averaged_sample(*, step_index: int, x_mm: float, std_rms: float) -> ExperimentTimeseriesSample:
    return ExperimentTimeseriesSample(
        monotonic_time_s=float(step_index),
        wall_time_utc="2026-05-19T00:00:00Z",
        phase="workspace_coverage",
        step_index=int(step_index),
        sample_index=0,
        commanded_motor_values={"1": 2048, "2": 2048, "3": 2048, "4": 2048},
        commanded_cable_deltas_cm=[0.0, 0.0, 0.0, 0.0],
        pose_in_robot_frame={"tip": {"translation_mm": [x_mm, 0.0, -150.0]}},
        extra={
            "record_kind": "modeling_dataset_capture",
            "capture_accepted": True,
            "modeling_export_exclude": False,
            "label_kind": "averaged",
            "tracker_averaging": {
                "position_std_rms_mm": float(std_rms),
                "position_max_deviation_mm": float(std_rms * 2.5),
                "first_vs_mean_position_diff_mm": float(std_rms * 0.5),
                "orientation_max_spread_deg": 0.1,
                "valid_sample_count": 20,
                "sample_window_s": 0.4,
            },
        },
    )


def test_writer_emits_all_5_figures_when_averaging_active(tmp_path: Path) -> None:
    averaged_samples = [
        _make_averaged_sample(step_index=i, x_mm=float(i) * 0.5, std_rms=0.05 + i * 0.01)
        for i in range(8)
    ]
    raw_rows = [
        {
            "command_index": i,
            "frame_index": j,
            "pose_in_robot_frame": {"translation_mm": [float(i) * 0.5 + j * 0.01, 0.0, -150.0]},
        }
        for i in range(8)
        for j in range(20)
    ]
    outputs = write_modeling_dataset_outputs(
        output_dir=tmp_path,
        metadata=_make_metadata(),
        summary=_make_summary(len(averaged_samples)),
        samples=[],  # no first-frame rows needed; we're only checking figure emission
        averaged_samples=averaged_samples,
        raw_tracker_frame_rows=raw_rows,
        tracker_samples_per_command=20,
        averaged_label_enabled=True,
        export_first_sample_label=False,
        export_averaged_sample_label=True,
    )
    expected = {
        "tracker_variability_workspace_xy_path",
        "tracker_variability_std_histogram_path",
        "tracker_variability_first_vs_mean_path",
        "tracker_variability_sample_spread_path",
        "tracker_variability_std_vs_command_index_path",
    }
    assert expected.issubset(outputs.keys())
    for key in expected:
        _assert_real_png(outputs[key])


def test_writer_skips_variability_figures_when_averaging_off(tmp_path: Path) -> None:
    outputs = write_modeling_dataset_outputs(
        output_dir=tmp_path,
        metadata=_make_metadata(),
        summary=_make_summary(0),
        samples=[],
        averaged_samples=None,
        raw_tracker_frame_rows=None,
        tracker_samples_per_command=1,
        averaged_label_enabled=False,
        export_first_sample_label=True,
        export_averaged_sample_label=False,
    )
    for key in (
        "tracker_variability_workspace_xy_path",
        "tracker_variability_std_histogram_path",
        "tracker_variability_first_vs_mean_path",
        "tracker_variability_sample_spread_path",
        "tracker_variability_std_vs_command_index_path",
    ):
        assert key not in outputs
        assert not (tmp_path / f"{key.removesuffix('_path')}.png").exists()


def test_writer_skips_variability_figures_when_averaged_samples_empty(tmp_path: Path) -> None:
    """Averaging configured ON but produced zero averaged rows (degenerate run)."""
    outputs = write_modeling_dataset_outputs(
        output_dir=tmp_path,
        metadata=_make_metadata(),
        summary=_make_summary(0),
        samples=[],
        averaged_samples=[],
        raw_tracker_frame_rows=[],
        tracker_samples_per_command=20,
        averaged_label_enabled=True,
        export_first_sample_label=True,
        export_averaged_sample_label=True,
    )
    for key in (
        "tracker_variability_workspace_xy_path",
        "tracker_variability_std_histogram_path",
    ):
        assert key not in outputs

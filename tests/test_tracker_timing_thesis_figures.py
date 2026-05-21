"""Render-side tests for the tracker-timing thesis figures.

We don't pixel-diff matplotlib output (fragile); each writer is exercised
against a synthetic record stream and an empty stream, and the resulting
PNG is checked for being a real (non-placeholder) PNG.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from continuum_robot.experiments import tracker_timing_outputs as outputs


_PLACEHOLDER_PNG_BYTES = 90


def _assert_real_png(path: Path) -> None:
    assert path.exists(), f"{path} not written"
    size = path.stat().st_size
    assert size > _PLACEHOLDER_PNG_BYTES, (
        f"{path} looks like a placeholder ({size} bytes) — render likely raised"
    )
    assert path.read_bytes().startswith(b"\x89PNG")


def _make_records(n: int = 60, *, dropout_at: int | None = None) -> list[dict[str, Any]]:
    """Synthetic tracker-timing records at ~38 Hz with optional dropout."""
    records: list[dict[str, Any]] = []
    t = 0
    for i in range(n):
        delta_ns = 26_000_000  # ~38 Hz
        if dropout_at is not None and dropout_at <= i < dropout_at + 4:
            delta_ns = 80_000_000
        t += delta_ns
        is_duplicate = (i % 11 == 0)
        records.append(
            {
                "sample_commit_monotonic_ns": int(t),
                "sample_index": i,
                "frame_number": i,
                "frame_number_source": "device",
                "warmup_discarded": False,
                "is_new_frame": not is_duplicate,
                "is_duplicate_frame": is_duplicate,
                "tool_validity": {"0A": "tracked" if (i % 23 != 0) else "missing"},
                "backend_call_ms": 20.0 + (i % 5) * 0.2,
                "parse_ms": 0.8 + (i % 3) * 0.05,
                "state_commit_ms": 0.03,
                "total_cycle_ms": 21.0 + (i % 5) * 0.2,
            }
        )
    return records


def _metrics() -> dict[str, Any]:
    return {
        "polling_rate_hz": 38.0,
        "unique_frame_rate_hz": 35.5,
        "valid_pose_rate_hz": 34.0,
        "requested_tool_ids": ["0A"],
    }


# --------------------------------------------------------------------------- #
# Per-figure rendering                                                         #
# --------------------------------------------------------------------------- #


def test_thesis_01_rate_vs_ceiling_renders(tmp_path: Path) -> None:
    path = tmp_path / "thesis_01.png"
    outputs._write_tracker_thesis_01_rate_vs_ceiling(
        path=path, tracker_records=_make_records(), metrics=_metrics(),
    )
    _assert_real_png(path)


def test_thesis_01_rate_vs_ceiling_handles_no_rate(tmp_path: Path) -> None:
    """No achieved-rate metric — the writer must still emit a real PNG."""
    path = tmp_path / "thesis_01_empty.png"
    outputs._write_tracker_thesis_01_rate_vs_ceiling(
        path=path, tracker_records=[], metrics={},
    )
    _assert_real_png(path)


def test_thesis_01_rate_vs_ceiling_with_dropout_renders(tmp_path: Path) -> None:
    """A mid-run dropout should not crash the stability strip."""
    path = tmp_path / "thesis_01_dropout.png"
    outputs._write_tracker_thesis_01_rate_vs_ceiling(
        path=path, tracker_records=_make_records(dropout_at=25), metrics=_metrics(),
    )
    _assert_real_png(path)


def test_thesis_02_inter_frame_interval_renders(tmp_path: Path) -> None:
    path = tmp_path / "thesis_02.png"
    outputs._write_tracker_thesis_02_inter_frame_interval(
        path=path, tracker_records=_make_records(), metrics=_metrics(),
    )
    _assert_real_png(path)


def test_thesis_02_inter_frame_interval_handles_empty(tmp_path: Path) -> None:
    path = tmp_path / "thesis_02_empty.png"
    outputs._write_tracker_thesis_02_inter_frame_interval(
        path=path, tracker_records=[], metrics=_metrics(),
    )
    _assert_real_png(path)


def test_thesis_03_cycle_time_budget_renders(tmp_path: Path) -> None:
    path = tmp_path / "thesis_03.png"
    outputs._write_tracker_thesis_03_cycle_time_budget(
        path=path, tracker_records=_make_records(), metrics=_metrics(),
    )
    _assert_real_png(path)


def test_thesis_03_cycle_time_budget_handles_empty(tmp_path: Path) -> None:
    path = tmp_path / "thesis_03_empty.png"
    outputs._write_tracker_thesis_03_cycle_time_budget(
        path=path, tracker_records=[], metrics=_metrics(),
    )
    _assert_real_png(path)


# --------------------------------------------------------------------------- #
# Writer integration                                                           #
# --------------------------------------------------------------------------- #


def test_writer_returns_all_3_thesis_paths(tmp_path: Path) -> None:
    """write_tracker_timing_outputs must surface the 3 thesis figure paths
    in its return dict so a downstream caller (CLI, GUI summary) can find
    them without scanning the directory."""
    from continuum_robot.experiments.schemas import (
        ExperimentMetadata,
        ExperimentSummary,
    )

    metadata = ExperimentMetadata(
        schema_version="1.0",
        experiment_name="tracker_timing_validation",
        run_id="run_thesis_figures_test",
        timestamp_utc="2026-05-21T00:00:00Z",
        git_commit="abc",
        backend_info={"mock_mode": True},
        registration_info={},
        config_used={},
    )
    summary = ExperimentSummary(
        schema_version="1.0",
        experiment_name="tracker_timing_validation",
        run_id="run_thesis_figures_test",
        success=True,
        sample_counts={"total": 0},
        dropped_frames=0,
        invalid_transforms=0,
        stage_pass_fail={
            "setup": "passed", "precheck": "passed",
            "execute": "passed", "finalize": "passed",
        },
        status="success",
        experiment_metrics=_metrics(),
    )
    paths = outputs.write_tracker_timing_outputs(
        output_dir=tmp_path,
        metadata=metadata,
        summary=summary,
        samples=[],
    )
    for key in ("thesis_01_path", "thesis_02_path", "thesis_03_path", "debug_json_path"):
        assert key in paths, f"Missing key in writer return dict: {key}"
        assert paths[key].exists(), f"Writer did not produce {key} → {paths[key]}"

    # The old supplementary figure keys must NOT be returned anymore.
    for stale_key in (
        "inter_frame_interval_histogram_path",
        "unique_frame_rate_over_time_path",
        "duplicate_invalid_timeline_path",
        "polling_vs_unique_rate_path",
        "valid_pose_rate_over_time_path",
    ):
        assert stale_key not in paths, (
            f"Removed supplementary figure key still in return dict: {stale_key}"
        )

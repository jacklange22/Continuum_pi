"""Phase 3 supplementary figures for tracker timing experiments.

Render-side tests for the 5 figures the spec lists. We don't pixel-diff
matplotlib output (fragile); instead we verify each writer produces a
non-trivial PNG for representative inputs and handles empty inputs
without crashing.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import pytest

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
    """Build a ~30 Hz synthetic tracker timing record stream."""
    records: list[dict[str, Any]] = []
    t = 0
    for i in range(n):
        # Most intervals are ~26 ms; a few stretched intervals around dropout_at.
        delta_ns = 26_000_000
        if dropout_at is not None and dropout_at <= i < dropout_at + 4:
            delta_ns = 80_000_000
        t += delta_ns
        is_duplicate = (i % 11 == 0)  # roughly 9% duplicates
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


def test_thesis_01_frame_interval_histogram_renders(tmp_path: Path) -> None:
    """Headline figure 1 is now a SIMPLE inter-frame interval histogram."""
    path = tmp_path / "frame_interval.png"
    outputs._write_tracker_thesis_01_frame_interval_histogram(
        path=path, tracker_records=_make_records(), metrics=_metrics(),
    )
    _assert_real_png(path)


def test_thesis_01_frame_interval_histogram_handles_empty(tmp_path: Path) -> None:
    path = tmp_path / "frame_interval_empty.png"
    outputs._write_tracker_thesis_01_frame_interval_histogram(
        path=path, tracker_records=[], metrics=_metrics(),
    )
    _assert_real_png(path)


def test_thesis_01_frame_interval_minimal_reference_set(tmp_path: Path) -> None:
    """Operator wanted the figure SIMPLE: just histogram + median +
    Aurora 25ms ceiling. No CDF / twin axis, no p95/p99 lines crowding
    the chart. Checks code patterns, not docstring text."""
    import inspect

    src = inspect.getsource(outputs._write_tracker_thesis_01_frame_interval_histogram)
    # Strip docstring (so the word "CDF" in the explanatory comment doesn't
    # trip the check). Walk the function body only.
    body = src.split('"""', 2)[-1]
    assert ".twinx(" not in body, "thesis_01 must stay a single-axis histogram"
    assert "percentile(" not in body, "thesis_01 must not plot any percentile lines"


def test_unique_frame_rate_over_time_renders(tmp_path: Path) -> None:
    path = tmp_path / "unique.png"
    outputs._write_tracker_unique_frame_rate_over_time(
        path=path, tracker_records=_make_records(dropout_at=20), metrics=_metrics(),
    )
    _assert_real_png(path)


def test_unique_frame_rate_over_time_handles_empty(tmp_path: Path) -> None:
    path = tmp_path / "unique_empty.png"
    outputs._write_tracker_unique_frame_rate_over_time(
        path=path, tracker_records=[], metrics=_metrics(),
    )
    _assert_real_png(path)


def test_writer_no_longer_emits_duplicate_invalid_timeline(tmp_path: Path) -> None:
    """Operator dropped the duplicate/invalid timeline figure from the
    output contract (it nearly always rendered empty on a clean run, and
    the counts are still in debug.json + summary metrics)."""
    from continuum_robot.experiments.schemas import (
        ExperimentMetadata, ExperimentSummary,
    )
    metadata = ExperimentMetadata(
        schema_version="1.0", experiment_name="tracker_timing_validation",
        run_id="dropped_fig_test", timestamp_utc="2026-05-20T00:00:00Z",
        git_commit="abc", backend_info={"mock_mode": True},
        registration_info={}, config_used={},
    )
    summary = ExperimentSummary(
        schema_version="1.0", experiment_name="tracker_timing_validation",
        run_id="dropped_fig_test", success=True,
        sample_counts={"total": 0}, dropped_frames=0, invalid_transforms=0,
        stage_pass_fail={"setup": "passed", "precheck": "passed",
                         "execute": "passed", "finalize": "passed"},
        status="success", experiment_metrics=_metrics(),
    )
    paths = outputs.write_tracker_timing_outputs(
        output_dir=tmp_path, metadata=metadata, summary=summary, samples=[],
    )
    assert "duplicate_invalid_timeline_path" not in paths, (
        "Figure dropped by operator; do not re-add to the output contract."
    )
    # Also confirm no stray PNG with that name was written.
    assert not (tmp_path / "tracker_duplicate_invalid_timeline.png").exists()


def test_polling_vs_unique_rate_summary_renders(tmp_path: Path) -> None:
    path = tmp_path / "rates.png"
    outputs._write_tracker_polling_vs_unique_rate_summary(
        path=path, metrics=_metrics(),
    )
    _assert_real_png(path)


def test_polling_vs_unique_rate_summary_handles_missing_metrics(tmp_path: Path) -> None:
    """When the metrics dict has no rate fields, the writer must still emit
    a real PNG (with an 'no rate metrics available' message) — never raise."""
    path = tmp_path / "rates_empty.png"
    outputs._write_tracker_polling_vs_unique_rate_summary(
        path=path, metrics={},
    )
    _assert_real_png(path)


def test_valid_pose_rate_over_time_renders(tmp_path: Path) -> None:
    path = tmp_path / "valid_pose.png"
    outputs._write_tracker_valid_pose_rate_over_time(
        path=path, tracker_records=_make_records(), metrics=_metrics(),
    )
    _assert_real_png(path)


def test_valid_pose_rate_over_time_handles_no_requested_tools(tmp_path: Path) -> None:
    path = tmp_path / "valid_pose_no_tools.png"
    outputs._write_tracker_valid_pose_rate_over_time(
        path=path, tracker_records=_make_records(),
        metrics={"polling_rate_hz": 30.0},  # no requested_tool_ids
    )
    _assert_real_png(path)


# --------------------------------------------------------------------------- #
# Writer integration                                                           #
# --------------------------------------------------------------------------- #


def test_writer_returns_all_remaining_figure_paths(tmp_path: Path) -> None:
    """write_tracker_timing_outputs must surface every emitted figure path
    in its return dict so a downstream caller (CLI, GUI summary) can find
    them without scanning the directory.

    After the operator-requested simplification: thesis_01 (simple frame
    interval histogram) + thesis_02 (stage breakdown) + 3 supplementary
    figures. The old inter_frame_interval_histogram and
    duplicate_invalid_timeline paths are intentionally absent."""
    from continuum_robot.experiments.schemas import (
        ExperimentMetadata,
        ExperimentSummary,
    )

    metadata = ExperimentMetadata(
        schema_version="1.0",
        experiment_name="tracker_timing_validation",
        run_id="run_phase3_test",
        timestamp_utc="2026-05-20T00:00:00Z",
        git_commit="abc",
        backend_info={"mock_mode": True},
        registration_info={},
        config_used={},
    )
    summary = ExperimentSummary(
        schema_version="1.0",
        experiment_name="tracker_timing_validation",
        run_id="run_phase3_test",
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
    for key in (
        "thesis_01_path",
        "thesis_02_path",
        "unique_frame_rate_over_time_path",
        "polling_vs_unique_rate_path",
        "valid_pose_rate_over_time_path",
    ):
        assert key in paths, f"Figure path missing from writer return: {key}"
        assert paths[key].exists(), f"Figure not written: {key}"
    # Negative guard: dropped figures must NOT come back in the return dict.
    for dropped_key in (
        "inter_frame_interval_histogram_path",
        "duplicate_invalid_timeline_path",
    ):
        assert dropped_key not in paths, (
            f"Operator-dropped figure {dropped_key} reappeared in writer output."
        )

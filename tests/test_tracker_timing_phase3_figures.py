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


def test_inter_frame_interval_histogram_renders(tmp_path: Path) -> None:
    path = tmp_path / "iface.png"
    outputs._write_tracker_inter_frame_interval_histogram(
        path=path, tracker_records=_make_records(), metrics=_metrics(),
    )
    _assert_real_png(path)


def test_inter_frame_interval_histogram_handles_empty(tmp_path: Path) -> None:
    path = tmp_path / "iface_empty.png"
    outputs._write_tracker_inter_frame_interval_histogram(
        path=path, tracker_records=[], metrics=_metrics(),
    )
    _assert_real_png(path)


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


def test_duplicate_invalid_timeline_renders(tmp_path: Path) -> None:
    path = tmp_path / "dup.png"
    outputs._write_tracker_duplicate_invalid_timeline(
        path=path, tracker_records=_make_records(), metrics=_metrics(),
    )
    _assert_real_png(path)


def test_duplicate_invalid_timeline_handles_empty(tmp_path: Path) -> None:
    path = tmp_path / "dup_empty.png"
    outputs._write_tracker_duplicate_invalid_timeline(
        path=path, tracker_records=[], metrics=_metrics(),
    )
    _assert_real_png(path)


def test_duplicate_invalid_timeline_clean_run_uses_success_card(tmp_path: Path) -> None:
    """When the run had records but ZERO duplicates and ZERO invalid frames,
    the writer must render a clean success card, not an empty scatter on a
    meaningless ±0.05 s axis.

    The success-card path is a code branch reached only when records exist
    AND there are no duplicate/invalid events; previously this fell through
    to the scatter writer and produced an unreadable empty figure.
    """
    records = _make_records(n=60)
    for rec in records:
        rec["is_duplicate_frame"] = False
        rec["tool_validity"] = {"0A": "tracked"}
    path = tmp_path / "dup_clean.png"
    outputs._write_tracker_duplicate_invalid_timeline(
        path=path, tracker_records=records, metrics=_metrics(),
    )
    _assert_real_png(path)
    # Clean-run path must produce a real PNG (not the tiny placeholder).
    assert path.stat().st_size > 5_000, (
        "Clean-run PNG should be a real success card, not a placeholder"
    )


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


def test_writer_returns_all_5_supplementary_paths(tmp_path: Path) -> None:
    """write_tracker_timing_outputs must surface the 5 new figure paths
    in its return dict so a downstream caller (CLI, GUI summary) can find
    them without scanning the directory."""
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
        "inter_frame_interval_histogram_path",
        "unique_frame_rate_over_time_path",
        "duplicate_invalid_timeline_path",
        "polling_vs_unique_rate_path",
        "valid_pose_rate_over_time_path",
    ):
        assert key in paths, f"Phase 3 figure path missing: {key}"
        assert paths[key].exists(), f"Phase 3 figure not written: {key}"

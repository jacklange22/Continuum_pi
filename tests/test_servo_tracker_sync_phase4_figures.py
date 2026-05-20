"""Phase 4 supplementary figures for servo-tracker sync experiment.

Render-side tests for the 6 supplementary figures (alongside the 2 thesis
figures). We don't pixel-diff matplotlib output (fragile); instead we verify
each writer produces a non-trivial PNG for representative inputs and handles
empty inputs without crashing.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from continuum_robot.experiments import servo_tracker_sync_outputs as outputs


_PLACEHOLDER_PNG_BYTES = 90


def _assert_real_png(path: Path) -> None:
    assert path.exists(), f"{path} not written"
    size = path.stat().st_size
    assert size > _PLACEHOLDER_PNG_BYTES, (
        f"{path} looks like a placeholder ({size} bytes) — render likely raised"
    )
    assert path.read_bytes().startswith(b"\x89PNG")


# --------------------------------------------------------------------------- #
# Synthetic record builders                                                    #
# --------------------------------------------------------------------------- #


def _make_tracker_records(
    n: int = 60,
    *,
    start_ns: int = 1_000_000_000,
    interval_ns: int = 26_000_000,
) -> list[dict[str, Any]]:
    """Build a ~38 Hz synthetic tracker record stream."""
    records: list[dict[str, Any]] = []
    for i in range(n):
        records.append(
            {
                "sample_commit_monotonic_ns": int(start_ns + i * interval_ns),
                "sample_index": i,
                "frame_number": i,
                "frame_number_source": "device",
                "warmup_discarded": False,
                "is_new_frame": True,
                "is_duplicate_frame": False,
                "tool_validity": {"0A": "tracked"},
            }
        )
    return records


def _make_servo_command_records(
    n: int = 12,
    *,
    start_ns: int = 1_005_000_000,
    interval_ns: int = 100_000_000,
) -> list[dict[str, Any]]:
    """Build synthetic servo command dispatch events at ~10 Hz."""
    return [
        {
            "command_monotonic_ns": int(start_ns + i * interval_ns),
            "sample_index": i,
            "servo_id": 1,
            "commanded_position_ticks": 1500 + (i % 2) * 50,
            "warmup_discarded": False,
            "error_flag": False,
        }
        for i in range(n)
    ]


def _make_servo_telemetry_records(
    n: int = 100,
    *,
    start_ns: int = 1_002_000_000,
    interval_ns: int = 15_000_000,
) -> list[dict[str, Any]]:
    """Build synthetic servo telemetry samples at ~66 Hz."""
    return [
        {
            "sample_monotonic_ns": int(start_ns + i * interval_ns),
            "sample_index": i,
            "servo_id": 1,
            "present_position_ticks": 1500 + (i % 50),
            "warmup_discarded": False,
            "error_flag": False,
        }
        for i in range(n)
    ]


def _make_metrics() -> dict[str, Any]:
    """Synthesize the `metrics` dict shape that real summaries hand to the writers."""
    return {
        "requested_tool_ids": ["0A"],
        "servo_tracker_sync": {
            "tracker_to_servo_command_offsets_ms": [
                1.2, 2.4, 3.7, 5.1, 6.8, 4.2, 2.9, 1.5, 3.1, 4.7,
            ],
            "tracker_to_servo_command_median_offset_ms": 3.4,
            "tracker_to_servo_command_mean_offset_ms": 3.6,
            "tracker_to_servo_command_p95_offset_ms": 6.4,
            "tracker_to_servo_command_max_offset_ms": 6.8,
            "tracker_to_servo_command_sample_count": 10,
            "tracker_to_servo_telemetry_offsets_ms": [
                0.5, 1.2, 0.8, 1.5, 2.1, 0.6, 1.8, 2.4, 1.1, 0.9,
                1.7, 0.4, 2.0, 1.3, 0.7,
            ],
            "tracker_to_servo_telemetry_median_offset_ms": 1.2,
            "tracker_to_servo_telemetry_mean_offset_ms": 1.27,
            "tracker_to_servo_telemetry_p95_offset_ms": 2.3,
            "tracker_to_servo_telemetry_max_offset_ms": 2.4,
            "tracker_to_servo_telemetry_sample_count": 15,
        },
    }


# --------------------------------------------------------------------------- #
# Per-figure rendering                                                         #
# --------------------------------------------------------------------------- #


def test_command_events_on_tracker_timeline_renders(tmp_path: Path) -> None:
    path = tmp_path / "events.png"
    outputs._write_sync_command_events_on_tracker_timeline(
        path=path,
        tracker_records=_make_tracker_records(),
        servo_command_records=_make_servo_command_records(),
    )
    _assert_real_png(path)


def test_command_events_on_tracker_timeline_handles_empty_tracker(tmp_path: Path) -> None:
    path = tmp_path / "events_empty.png"
    outputs._write_sync_command_events_on_tracker_timeline(
        path=path,
        tracker_records=[],
        servo_command_records=_make_servo_command_records(),
    )
    _assert_real_png(path)


def test_command_events_on_tracker_timeline_handles_no_commands(tmp_path: Path) -> None:
    """Tracker frames present, but no command dispatch events — should still render."""
    path = tmp_path / "events_no_cmds.png"
    outputs._write_sync_command_events_on_tracker_timeline(
        path=path,
        tracker_records=_make_tracker_records(),
        servo_command_records=[],
    )
    _assert_real_png(path)


def test_command_offset_histogram_renders(tmp_path: Path) -> None:
    path = tmp_path / "cmd_offset.png"
    outputs._write_sync_command_offset_histogram(
        path=path, metrics=_make_metrics(),
    )
    _assert_real_png(path)


def test_command_offset_histogram_handles_missing_metrics(tmp_path: Path) -> None:
    """Metrics dict has no sync block — writer must still emit a real PNG."""
    path = tmp_path / "cmd_offset_empty.png"
    outputs._write_sync_command_offset_histogram(
        path=path, metrics={},
    )
    _assert_real_png(path)


def test_telemetry_offset_histogram_renders(tmp_path: Path) -> None:
    path = tmp_path / "tel_offset.png"
    outputs._write_sync_telemetry_offset_histogram(
        path=path, metrics=_make_metrics(),
    )
    _assert_real_png(path)


def test_telemetry_offset_histogram_handles_missing_metrics(tmp_path: Path) -> None:
    path = tmp_path / "tel_offset_empty.png"
    outputs._write_sync_telemetry_offset_histogram(
        path=path, metrics={},
    )
    _assert_real_png(path)


def test_tracker_frame_age_at_servo_events_renders(tmp_path: Path) -> None:
    path = tmp_path / "frame_age.png"
    outputs._write_sync_tracker_frame_age_at_servo_events(
        path=path,
        tracker_records=_make_tracker_records(),
        servo_command_records=_make_servo_command_records(),
        servo_telemetry_records=_make_servo_telemetry_records(),
    )
    _assert_real_png(path)


def test_tracker_frame_age_at_servo_events_handles_no_servo_events(tmp_path: Path) -> None:
    """Tracker frames but no commands or telemetry — writer must emit a real PNG."""
    path = tmp_path / "frame_age_empty.png"
    outputs._write_sync_tracker_frame_age_at_servo_events(
        path=path,
        tracker_records=_make_tracker_records(),
        servo_command_records=[],
        servo_telemetry_records=[],
    )
    _assert_real_png(path)


def test_tracker_frame_age_at_servo_events_handles_no_tracker(tmp_path: Path) -> None:
    """Servo events but no tracker frames — writer must emit a real PNG."""
    path = tmp_path / "frame_age_no_tracker.png"
    outputs._write_sync_tracker_frame_age_at_servo_events(
        path=path,
        tracker_records=[],
        servo_command_records=_make_servo_command_records(),
        servo_telemetry_records=_make_servo_telemetry_records(),
    )
    _assert_real_png(path)


def test_servo_telemetry_interval_histogram_renders(tmp_path: Path) -> None:
    path = tmp_path / "telemetry_intervals.png"
    outputs._write_sync_servo_telemetry_interval_histogram(
        path=path,
        servo_telemetry_records=_make_servo_telemetry_records(),
    )
    _assert_real_png(path)


def test_servo_telemetry_interval_histogram_handles_empty(tmp_path: Path) -> None:
    path = tmp_path / "telemetry_intervals_empty.png"
    outputs._write_sync_servo_telemetry_interval_histogram(
        path=path,
        servo_telemetry_records=[],
    )
    _assert_real_png(path)


def test_combined_timing_summary_renders(tmp_path: Path) -> None:
    path = tmp_path / "combined.png"
    outputs._write_sync_combined_timing_summary(
        path=path,
        metrics=_make_metrics(),
        tracker_records=_make_tracker_records(),
        servo_command_records=_make_servo_command_records(),
        servo_telemetry_records=_make_servo_telemetry_records(),
    )
    _assert_real_png(path)


def test_combined_timing_summary_handles_no_data(tmp_path: Path) -> None:
    """No metrics, no records — combined panel must still render with placeholders."""
    path = tmp_path / "combined_empty.png"
    outputs._write_sync_combined_timing_summary(
        path=path,
        metrics={},
        tracker_records=[],
        servo_command_records=[],
        servo_telemetry_records=[],
    )
    _assert_real_png(path)


# --------------------------------------------------------------------------- #
# Helper smoke tests                                                           #
# --------------------------------------------------------------------------- #


def test_nearest_preceding_offset_ms_returns_positive_ms() -> None:
    """nearest_preceding returns ns->ms conversion and only positive (no future) values."""
    targets = [1_000_000_000, 2_000_000_000, 3_000_000_000]
    # 2.5 s comes 0.5 s after the most recent target (2.0 s)
    assert outputs._nearest_preceding_offset_ms(2_500_000_000, targets) == pytest.approx(500.0)
    # Event before all targets → None
    assert outputs._nearest_preceding_offset_ms(500_000_000, targets) is None
    # Event exactly at a target → ms gap from that exact target = 0
    assert outputs._nearest_preceding_offset_ms(1_000_000_000, targets) == pytest.approx(0.0)


def test_analyzed_tracker_commit_times_ns_drops_warmup() -> None:
    records = [
        {"sample_commit_monotonic_ns": 100, "warmup_discarded": True},
        {"sample_commit_monotonic_ns": 200, "warmup_discarded": False},
        {"sample_commit_monotonic_ns": 300, "warmup_discarded": False},
    ]
    times = outputs._analyzed_tracker_commit_times_ns(records)
    assert times == [200, 300]


def test_analyzed_servo_command_times_ns_drops_warmup() -> None:
    records = [
        {"command_monotonic_ns": 100, "warmup_discarded": True},
        {"command_monotonic_ns": 200, "warmup_discarded": False},
    ]
    times = outputs._analyzed_servo_command_times_ns(records)
    assert times == [200]


def test_analyzed_servo_telemetry_times_ns_drops_warmup() -> None:
    records = [
        {"sample_monotonic_ns": 100, "warmup_discarded": True},
        {"sample_monotonic_ns": 200, "warmup_discarded": False},
    ]
    times = outputs._analyzed_servo_telemetry_times_ns(records)
    assert times == [200]


# --------------------------------------------------------------------------- #
# Writer integration                                                           #
# --------------------------------------------------------------------------- #


def test_writer_returns_all_6_phase4_paths(tmp_path: Path) -> None:
    """write_servo_tracker_sync_outputs must surface all 6 Phase 4 figure paths
    in its return dict so a downstream caller (CLI, GUI summary) can find
    them without scanning the directory."""
    from continuum_robot.experiments.schemas import (
        ExperimentMetadata,
        ExperimentSummary,
    )

    metadata = ExperimentMetadata(
        schema_version="1.0",
        experiment_name="servo_tracker_sync_validation",
        run_id="run_phase4_test",
        timestamp_utc="2026-05-20T00:00:00Z",
        git_commit="abc",
        backend_info={"mock_mode": True},
        registration_info={},
        config_used={},
    )
    summary = ExperimentSummary(
        schema_version="1.0",
        experiment_name="servo_tracker_sync_validation",
        run_id="run_phase4_test",
        success=True,
        sample_counts={"total": 0},
        dropped_frames=0,
        invalid_transforms=0,
        stage_pass_fail={
            "setup": "passed", "precheck": "passed",
            "execute": "passed", "finalize": "passed",
        },
        status="success",
        experiment_metrics=_make_metrics(),
    )
    paths = outputs.write_servo_tracker_sync_outputs(
        output_dir=tmp_path,
        metadata=metadata,
        summary=summary,
        samples=[],
    )
    for key in (
        "command_events_on_tracker_timeline_path",
        "command_offset_histogram_path",
        "telemetry_offset_histogram_path",
        "tracker_frame_age_at_servo_events_path",
        "servo_telemetry_interval_histogram_path",
        "combined_timing_summary_path",
    ):
        assert key in paths, f"Phase 4 figure path missing: {key}"
        assert paths[key].exists(), f"Phase 4 figure not written: {key}"

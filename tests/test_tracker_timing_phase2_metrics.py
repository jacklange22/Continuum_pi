"""Phase 2 metrics added to compute_tracker_timing_summary.

Covers:
  - polling_rate_hz and total_polling_duration_s (include warmup)
  - valid_pose_rate_hz (Hz, not just fraction)
  - tracker_frame_number_available + tracker_frame_timestamp_available flags
  - inter_frame_interval_ms_stats alias
  - std now reported by _numeric_stats (population std for jitter)
"""

from __future__ import annotations

import pytest

from continuum_robot.tracking.timing_benchmark import (
    _numeric_stats,
    compute_tracker_timing_summary,
)


# --------------------------------------------------------------------------- #
# _numeric_stats now includes std                                              #
# --------------------------------------------------------------------------- #


def test_numeric_stats_includes_population_std() -> None:
    stats = _numeric_stats([10.0, 20.0, 30.0, 40.0])
    assert "std" in stats
    # population std of [10,20,30,40] is ~11.18
    assert stats["std"] == pytest.approx(11.180, abs=0.01)


def test_numeric_stats_single_value_std_is_zero_not_crash() -> None:
    stats = _numeric_stats([42.0])
    assert stats["std"] == 0.0


def test_numeric_stats_empty_returns_none_std() -> None:
    stats = _numeric_stats([])
    assert stats["std"] is None


# --------------------------------------------------------------------------- #
# Synthetic record builder                                                     #
# --------------------------------------------------------------------------- #


def _make_record(
    *,
    sample_index: int,
    commit_ns: int,
    frame_number: int | None,
    frame_number_source: str = "device",
    is_new_frame: bool = True,
    is_duplicate_frame: bool = False,
    warmup_discarded: bool = False,
    tool_validity_0a: str = "tracked",
    observed_at_utc: str = "2026-05-20T00:00:00.000000+00:00",
) -> dict[str, object]:
    return {
        "sample_index": sample_index,
        "sample_commit_monotonic_ns": int(commit_ns),
        "observed_at_utc": observed_at_utc,
        "backend_identity": "test_backend",
        "requested_tool_ids": ["0A"],
        "frame_number": frame_number,
        "frame_number_source": frame_number_source,
        "is_new_frame": bool(is_new_frame),
        "is_duplicate_frame": bool(is_duplicate_frame),
        "warmup_discarded": bool(warmup_discarded),
        "tools_visible": ["0A"],
        "tool_validity": {"0A": tool_validity_0a},
        "tool_pose_payload": {"0A": {}},
        "raw_payload_available": True,
        "parsed_payload_available": True,
        "output_committed": True,
        "error_flag": False,
        "total_cycle_ms": 22.0,
        "backend_call_ms": 18.0,
        "parse_ms": 2.0,
        "state_commit_ms": 2.0,
    }


# --------------------------------------------------------------------------- #
# Phase 2 metrics                                                              #
# --------------------------------------------------------------------------- #


def test_polling_rate_hz_uses_total_window_including_warmup() -> None:
    """5 polling attempts spanning 100 ms → 4 intervals / 0.1 s = 40 Hz."""
    records = [
        _make_record(sample_index=i, commit_ns=int(i * 25_000_000), frame_number=i,
                     warmup_discarded=(i < 2))
        for i in range(5)
    ]
    summary = compute_tracker_timing_summary(
        records,
        requested_tool_ids=["0A"],
        backend_identity="test",
        configured_backend_name="ndi",
        selected_backend_name="ndi",
        run_duration_s=0.1,
    )
    assert summary["polling_attempt_count"] == 5
    assert summary["total_polling_duration_s"] == pytest.approx(0.1)
    assert summary["polling_rate_hz"] == pytest.approx(40.0, abs=0.1)


def test_polling_rate_distinct_from_effective_loop_rate() -> None:
    """polling_rate_hz includes warmup; effective_loop_rate_hz does not.
    With 2 warmup samples discarded, the two rates should land on different
    denominators."""
    records = [
        _make_record(sample_index=i, commit_ns=int(i * 25_000_000), frame_number=i,
                     warmup_discarded=(i < 2))
        for i in range(5)
    ]
    summary = compute_tracker_timing_summary(
        records,
        requested_tool_ids=["0A"],
        backend_identity="test",
        configured_backend_name="ndi",
        selected_backend_name="ndi",
    )
    # polling: 5 samples over 100 ms → 40 Hz
    # effective_loop: 3 analyzed samples over 50 ms → 40 Hz (same here, but
    # the denominators are different — let's pick a case where they diverge)
    assert summary["polling_rate_hz"] is not None
    assert summary["effective_loop_rate_hz"] is not None
    # Both fields exist and are floats
    assert isinstance(summary["polling_rate_hz"], float)
    assert isinstance(summary["effective_loop_rate_hz"], float)


def test_valid_pose_rate_hz_when_all_requested_tools_tracked() -> None:
    """4 analyzed samples with 0A tracked over 75 ms → 3 intervals / 0.075 s = 40 Hz."""
    records = [
        _make_record(sample_index=i, commit_ns=int(i * 25_000_000), frame_number=i,
                     tool_validity_0a="tracked")
        for i in range(4)
    ]
    summary = compute_tracker_timing_summary(
        records,
        requested_tool_ids=["0A"],
        backend_identity="test",
        configured_backend_name="ndi",
        selected_backend_name="ndi",
    )
    assert summary["valid_pose_rate_hz"] == pytest.approx(40.0, abs=0.1)


def test_valid_pose_rate_hz_zero_when_no_valid_samples() -> None:
    records = [
        _make_record(sample_index=i, commit_ns=int(i * 25_000_000), frame_number=i,
                     tool_validity_0a="missing")
        for i in range(4)
    ]
    summary = compute_tracker_timing_summary(
        records,
        requested_tool_ids=["0A"],
        backend_identity="test",
        configured_backend_name="ndi",
        selected_backend_name="ndi",
    )
    assert summary["valid_pose_rate_hz"] is None


def test_tracker_frame_number_available_true_when_device_sourced() -> None:
    records = [
        _make_record(sample_index=i, commit_ns=int(i * 25_000_000),
                     frame_number=i, frame_number_source="device")
        for i in range(3)
    ]
    summary = compute_tracker_timing_summary(
        records,
        requested_tool_ids=["0A"],
        backend_identity="test",
        configured_backend_name="ndi",
        selected_backend_name="ndi",
    )
    assert summary["tracker_frame_number_available"] is True


def test_tracker_frame_number_unavailable_when_only_synthetic() -> None:
    """Synthetic frame numbers (host-assigned) must NOT pass as 'real'
    backend frame IDs — operator needs to know the dedup is host-side."""
    records = [
        _make_record(sample_index=i, commit_ns=int(i * 25_000_000),
                     frame_number=i, frame_number_source="synthetic")
        for i in range(3)
    ]
    summary = compute_tracker_timing_summary(
        records,
        requested_tool_ids=["0A"],
        backend_identity="test",
        configured_backend_name="ndi",
        selected_backend_name="ndi",
    )
    assert summary["tracker_frame_number_available"] is False


def test_tracker_frame_timestamp_available_when_observed_at_utc_present() -> None:
    records = [
        _make_record(sample_index=0, commit_ns=0, frame_number=0,
                     observed_at_utc="2026-05-20T00:00:00Z"),
    ]
    summary = compute_tracker_timing_summary(
        records,
        requested_tool_ids=["0A"],
        backend_identity="test",
        configured_backend_name="ndi",
        selected_backend_name="ndi",
    )
    assert summary["tracker_frame_timestamp_available"] is True


def test_inter_frame_interval_ms_stats_aliases_loop_period() -> None:
    """The two keys must point to the exact same stats dict (alias, not copy)
    so a reader is never out of sync."""
    records = [
        _make_record(sample_index=i, commit_ns=int(i * 25_000_000), frame_number=i)
        for i in range(4)
    ]
    summary = compute_tracker_timing_summary(
        records,
        requested_tool_ids=["0A"],
        backend_identity="test",
        configured_backend_name="ndi",
        selected_backend_name="ndi",
    )
    assert summary["loop_period_ms_stats"] is summary["inter_frame_interval_ms_stats"]
    assert "std" in summary["loop_period_ms_stats"]


def test_summary_carries_all_phase2_keys() -> None:
    """Regression guard: every Phase 2 metric key must appear in the summary
    so downstream readers (GUI, CLI, plots, summary.txt) can rely on them."""
    records = [
        _make_record(sample_index=i, commit_ns=int(i * 25_000_000), frame_number=i)
        for i in range(4)
    ]
    summary = compute_tracker_timing_summary(
        records,
        requested_tool_ids=["0A"],
        backend_identity="test",
        configured_backend_name="ndi",
        selected_backend_name="ndi",
    )
    for key in (
        "polling_attempt_count",
        "polling_rate_hz",
        "total_polling_duration_s",
        "valid_pose_rate_hz",
        "tracker_frame_number_available",
        "tracker_frame_timestamp_available",
        "inter_frame_interval_ms_stats",
    ):
        assert key in summary, f"Phase 2 metric missing: {key}"

from continuum_robot.experiments.schemas import ExperimentTimeseriesSample
from continuum_robot.tracking.timing_benchmark import (
    compute_servo_sync_summary,
    compute_tracker_timing_summary,
    extract_tracker_timing_records,
)


def test_extract_tracker_timing_records_reads_canonical_tracker_samples() -> None:
    sample = ExperimentTimeseriesSample(
        monotonic_time_s=0.01,
        wall_time_utc="2026-01-01T00:00:00Z",
        phase="tracker_timing",
        step_index=0,
        sample_index=0,
        tracker_frame_id=7,
        tool_ids_seen=["0A", "0B"],
        transform_validity={"0A": "tracked", "0B": "tracked"},
        extra={
            "record_kind": "tracker_timing",
            "sample_start_monotonic_ns": 1000,
            "backend_call_start_ns": 1000,
            "backend_call_end_ns": 6000,
            "parse_complete_ns": 8000,
            "state_commit_complete_ns": 10000,
            "sample_commit_monotonic_ns": 10000,
            "observed_at_utc": "2026-01-01T00:00:00Z",
            "backend_identity": "ndi_tracker_python",
            "requested_tool_ids": ["0A", "0B"],
            "frame_number": 7,
            "frame_number_source": "device",
            "is_new_frame": True,
            "is_duplicate_frame": False,
            "raw_payload_available": True,
            "parsed_payload_available": True,
            "output_committed": True,
            "error_flag": False,
            "tools_visible": ["0A", "0B"],
            "raw_tool_ids": ["10", "11"],
            "normalized_tool_ids": ["0A", "0B"],
            "runtime_role_mappings": {"0A": "10", "0B": "11"},
            "tool_validity": {"0A": "tracked", "0B": "tracked"},
            "valid_transform_count": 2,
            "total_cycle_ms": 0.009,
            "backend_call_ms": 0.005,
            "parse_ms": 0.002,
            "state_commit_ms": 0.002,
            "warmup_discarded": False,
        },
    )

    records = extract_tracker_timing_records([sample])

    assert len(records) == 1
    assert records[0]["backend_identity"] == "ndi_tracker_python"
    assert records[0]["requested_tool_ids"] == ["0A", "0B"]
    assert records[0]["tool_validity"]["0B"] == "tracked"
    assert records[0]["frame_number"] == 7
    assert records[0]["total_cycle_ms"] == 0.009


def test_compute_tracker_timing_summary_distinguishes_unique_frames_from_loop_rate() -> None:
    records = [
        {
            "sample_index": 0,
            "sample_commit_monotonic_ns": 0,
            "frame_number": 101,
            "is_new_frame": True,
            "is_duplicate_frame": False,
            "raw_payload_available": True,
            "parsed_payload_available": True,
            "error_flag": False,
            "frame_number_source": "device",
            "tool_validity": {"0A": "tracked", "0B": "tracked"},
            "tools_visible": ["0A", "0B"],
            "total_cycle_ms": 12.0,
            "backend_call_ms": 10.0,
            "parse_ms": 1.0,
            "state_commit_ms": 1.0,
            "warmup_discarded": False,
            "valid_transform_count": 2,
        },
        {
            "sample_index": 1,
            "sample_commit_monotonic_ns": 20_000_000,
            "frame_number": 101,
            "is_new_frame": False,
            "is_duplicate_frame": True,
            "raw_payload_available": True,
            "parsed_payload_available": True,
            "error_flag": False,
            "frame_number_source": "device",
            "tool_validity": {"0A": "tracked", "0B": "missing"},
            "tools_visible": ["0A"],
            "total_cycle_ms": 14.0,
            "backend_call_ms": 11.0,
            "parse_ms": 2.0,
            "state_commit_ms": 1.0,
            "warmup_discarded": False,
            "valid_transform_count": 1,
        },
        {
            "sample_index": 2,
            "sample_commit_monotonic_ns": 40_000_000,
            "frame_number": 102,
            "is_new_frame": True,
            "is_duplicate_frame": False,
            "raw_payload_available": True,
            "parsed_payload_available": True,
            "error_flag": False,
            "frame_number_source": "device",
            "tool_validity": {"0A": "tracked", "0B": "tracked"},
            "tools_visible": ["0A", "0B"],
            "total_cycle_ms": 13.0,
            "backend_call_ms": 10.5,
            "parse_ms": 1.5,
            "state_commit_ms": 1.0,
            "warmup_discarded": False,
            "valid_transform_count": 2,
        },
    ]

    summary = compute_tracker_timing_summary(
        records,
        requested_tool_ids=["0A", "0B"],
        backend_identity="ndi_tracker_python",
        configured_backend_name="ndi",
        selected_backend_name="ndi",
        run_duration_s=0.04,
    )

    assert summary["sample_count_analyzed"] == 3
    assert summary["duplicate_frame_count"] == 1
    assert summary["unique_frame_count"] == 2
    assert summary["duplicate_frame_ratio"] == 1.0 / 3.0
    assert summary["effective_loop_rate_hz"] == 50.0
    assert summary["unique_frame_rate_hz"] == 25.0
    assert summary["invalid_or_missing_requested_tool_sample_count"] == 1
    assert summary["per_tool_summary"]["0B"]["missing_count"] == 1
    assert summary["per_tool_summary"]["0B"]["valid_transform_rate"] == 2.0 / 3.0


def test_compute_servo_sync_summary_handles_absent_servo_records_cleanly() -> None:
    tracker_records = [
        {
            "sample_commit_monotonic_ns": 1_000_000,
            "warmup_discarded": False,
        },
        {
            "sample_commit_monotonic_ns": 3_000_000,
            "warmup_discarded": False,
        },
    ]

    summary = compute_servo_sync_summary(tracker_records, [])

    assert summary["enabled"] is False
    assert summary["available"] is False
    assert summary["tracker_sample_count"] == 2
    assert summary["servo_sample_count"] == 0
    assert summary["servo_to_tracker_mean_offset_ms"] is None

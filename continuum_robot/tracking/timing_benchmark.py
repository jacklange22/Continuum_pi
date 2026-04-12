"""Backend timing helpers for tracker-throughput validation."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
import math
import statistics
from typing import Any


def _as_float(value: Any) -> float | None:
    if value in (None, ""):
        return None
    try:
        numeric = float(value)
    except Exception:
        return None
    return numeric if math.isfinite(numeric) else None


def _as_int(value: Any) -> int | None:
    if value in (None, ""):
        return None
    try:
        return int(value)
    except Exception:
        return None


def _quantile(sorted_values: list[float], fraction: float) -> float | None:
    if not sorted_values:
        return None
    if len(sorted_values) == 1:
        return float(sorted_values[0])
    rank = max(0.0, min(1.0, float(fraction))) * float(len(sorted_values) - 1)
    lower = int(math.floor(rank))
    upper = int(math.ceil(rank))
    if lower == upper:
        return float(sorted_values[lower])
    weight = rank - float(lower)
    return float(sorted_values[lower] * (1.0 - weight) + sorted_values[upper] * weight)


def _numeric_stats(values: Iterable[Any]) -> dict[str, Any]:
    finite_values = [numeric for numeric in (_as_float(value) for value in values) if numeric is not None]
    if not finite_values:
        return {
            "count": 0,
            "min": None,
            "max": None,
            "mean": None,
            "median": None,
            "p95": None,
            "p99": None,
        }
    ordered = sorted(finite_values)
    return {
        "count": len(ordered),
        "min": float(ordered[0]),
        "max": float(ordered[-1]),
        "mean": float(statistics.fmean(ordered)),
        "median": _quantile(ordered, 0.5),
        "p95": _quantile(ordered, 0.95),
        "p99": _quantile(ordered, 0.99),
    }


def extract_tracker_timing_records(samples) -> list[dict[str, Any]]:
    """Return normalized tracker timing records from canonical experiment samples."""
    rows: list[dict[str, Any]] = []
    for sample in samples:
        extra = dict(getattr(sample, "extra", {}) or {})
        if str(extra.get("record_kind", "")).strip().lower() != "tracker_timing":
            continue
        row = {
            "sample_index": _as_int(extra.get("sample_index")),
            "sample_start_monotonic_ns": _as_int(extra.get("sample_start_monotonic_ns")),
            "backend_call_start_ns": _as_int(extra.get("backend_call_start_ns")),
            "backend_call_end_ns": _as_int(extra.get("backend_call_end_ns")),
            "parse_complete_ns": _as_int(extra.get("parse_complete_ns")),
            "state_commit_complete_ns": _as_int(extra.get("state_commit_complete_ns")),
            "sample_commit_monotonic_ns": _as_int(
                extra.get("sample_commit_monotonic_ns") or extra.get("state_commit_complete_ns")
            ),
            "observed_at_utc": str(extra.get("observed_at_utc", "") or getattr(sample, "wall_time_utc", "")),
            "backend_identity": str(extra.get("backend_identity", "")),
            "requested_tool_ids": [str(value) for value in (extra.get("requested_tool_ids") or [])],
            "frame_number": _as_int(extra.get("frame_number")),
            "frame_number_source": str(extra.get("frame_number_source", "unknown")),
            "is_new_frame": bool(extra.get("is_new_frame", False)),
            "is_duplicate_frame": bool(extra.get("is_duplicate_frame", False)),
            "raw_payload_available": bool(extra.get("raw_payload_available", False)),
            "parsed_payload_available": bool(extra.get("parsed_payload_available", False)),
            "output_committed": bool(extra.get("output_committed", False)),
            "error_flag": bool(extra.get("error_flag", False)),
            "error_stage": str(extra.get("error_stage", "")) or None,
            "error_message": str(extra.get("error_message", "")) or None,
            "tools_visible": [str(value) for value in (extra.get("tools_visible") or [])],
            "raw_tool_ids": [str(value) for value in (extra.get("raw_tool_ids") or [])],
            "normalized_tool_ids": [str(value) for value in (extra.get("normalized_tool_ids") or [])],
            "runtime_role_mappings": {
                str(key): str(value)
                for key, value in dict(extra.get("runtime_role_mappings", {}) or {}).items()
            },
            "tool_validity": {
                str(key): str(value)
                for key, value in dict(extra.get("tool_validity", {}) or {}).items()
            },
            "valid_transform_count": _as_int(extra.get("valid_transform_count")) or 0,
            "total_cycle_ms": _as_float(extra.get("total_cycle_ms")),
            "backend_call_ms": _as_float(extra.get("backend_call_ms")),
            "parse_ms": _as_float(extra.get("parse_ms")),
            "state_commit_ms": _as_float(extra.get("state_commit_ms")),
            "warmup_discarded": bool(extra.get("warmup_discarded", False)),
        }
        rows.append(row)
    rows.sort(key=lambda row: (row.get("sample_commit_monotonic_ns") or 0, row.get("sample_index") or 0))
    return rows


def extract_servo_timing_records(samples) -> list[dict[str, Any]]:
    """Return normalized servo timing rows from canonical experiment samples."""
    rows: list[dict[str, Any]] = []
    for sample in samples:
        extra = dict(getattr(sample, "extra", {}) or {})
        if str(extra.get("record_kind", "")).strip().lower() != "servo_timing":
            continue
        rows.append(
            {
                "sample_index": _as_int(extra.get("sample_index")),
                "sample_monotonic_ns": _as_int(extra.get("sample_monotonic_ns")),
                "servo_id": _as_int(extra.get("servo_id")),
                "commanded_position_ticks": _as_int(extra.get("commanded_position_ticks")),
                "present_position_ticks": _as_int(extra.get("present_position_ticks")),
                "present_current_ma": _as_int(extra.get("present_current_ma")),
                "telemetry_age_s": _as_float(extra.get("telemetry_age_s")),
                "error_flag": bool(extra.get("error_flag", False)),
                "error_message": str(extra.get("error_message", "")) or None,
            }
        )
    rows.sort(key=lambda row: (row.get("sample_monotonic_ns") or 0, row.get("servo_id") or 0))
    return rows


def compute_servo_sync_summary(
    tracker_records: list[dict[str, Any]],
    servo_records: list[dict[str, Any]],
    *,
    pairing_threshold_ms: float = 25.0,
) -> dict[str, Any]:
    """Summarize nearest-neighbor tracker/servo time offsets."""
    tracker_times_ns = [
        int(value)
        for value in (
            record.get("sample_commit_monotonic_ns")
            for record in tracker_records
            if not bool(record.get("warmup_discarded", False))
        )
        if value is not None
    ]
    servo_times_ns = [
        int(value)
        for value in (record.get("sample_monotonic_ns") for record in servo_records)
        if value is not None
    ]
    if not tracker_times_ns or not servo_times_ns:
        return {
            "enabled": bool(servo_records),
            "available": False,
            "pairing_threshold_ms": float(pairing_threshold_ms),
            "tracker_sample_count": len(tracker_times_ns),
            "servo_sample_count": len(servo_times_ns),
            "servo_to_tracker_offsets_ms": [],
            "tracker_to_servo_offsets_ms": [],
            "servo_to_tracker_mean_offset_ms": None,
            "servo_to_tracker_median_offset_ms": None,
            "servo_to_tracker_p95_offset_ms": None,
            "servo_to_tracker_max_offset_ms": None,
            "tracker_to_servo_mean_offset_ms": None,
            "tracker_to_servo_median_offset_ms": None,
            "tracker_to_servo_p95_offset_ms": None,
            "tracker_to_servo_max_offset_ms": None,
            "servo_samples_over_threshold_count": None,
            "tracker_samples_over_threshold_count": None,
        }

    def _nearest_offsets_ms(source_ns: list[int], target_ns: list[int]) -> list[float]:
        offsets_ms: list[float] = []
        for value in source_ns:
            best = min(abs(int(value) - int(target)) for target in target_ns)
            offsets_ms.append(float(best) / 1_000_000.0)
        return offsets_ms

    servo_to_tracker = _nearest_offsets_ms(servo_times_ns, tracker_times_ns)
    tracker_to_servo = _nearest_offsets_ms(tracker_times_ns, servo_times_ns)
    servo_stats = _numeric_stats(servo_to_tracker)
    tracker_stats = _numeric_stats(tracker_to_servo)
    return {
        "enabled": True,
        "available": True,
        "pairing_threshold_ms": float(pairing_threshold_ms),
        "tracker_sample_count": len(tracker_times_ns),
        "servo_sample_count": len(servo_times_ns),
        "servo_to_tracker_offsets_ms": servo_to_tracker,
        "tracker_to_servo_offsets_ms": tracker_to_servo,
        "servo_to_tracker_mean_offset_ms": servo_stats["mean"],
        "servo_to_tracker_median_offset_ms": servo_stats["median"],
        "servo_to_tracker_p95_offset_ms": servo_stats["p95"],
        "servo_to_tracker_max_offset_ms": servo_stats["max"],
        "tracker_to_servo_mean_offset_ms": tracker_stats["mean"],
        "tracker_to_servo_median_offset_ms": tracker_stats["median"],
        "tracker_to_servo_p95_offset_ms": tracker_stats["p95"],
        "tracker_to_servo_max_offset_ms": tracker_stats["max"],
        "servo_samples_over_threshold_count": sum(offset > float(pairing_threshold_ms) for offset in servo_to_tracker),
        "tracker_samples_over_threshold_count": sum(offset > float(pairing_threshold_ms) for offset in tracker_to_servo),
    }


def compute_tracker_timing_summary(
    tracker_records: list[dict[str, Any]],
    *,
    requested_tool_ids: Iterable[str],
    backend_identity: str,
    configured_backend_name: str,
    selected_backend_name: str,
    run_duration_s: float | None = None,
    run_label: str = "",
    servo_sync_summary: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Summarize backend timing/throughput metrics from normalized tracker records."""
    requested = [str(tool_id).upper() for tool_id in requested_tool_ids if str(tool_id).strip()]
    all_records = sorted(
        [dict(record) for record in tracker_records],
        key=lambda record: (record.get("sample_commit_monotonic_ns") or 0, record.get("sample_index") or 0),
    )
    analyzed_records = [record for record in all_records if not bool(record.get("warmup_discarded", False))]
    commit_times_ns = [
        int(value)
        for value in (record.get("sample_commit_monotonic_ns") for record in analyzed_records)
        if value is not None
    ]
    analyzed_duration_s = None
    if len(commit_times_ns) >= 2:
        analyzed_duration_s = max(0.0, float(commit_times_ns[-1] - commit_times_ns[0]) / 1_000_000_000.0)

    total_cycle_values = [record.get("total_cycle_ms") for record in analyzed_records]
    backend_call_values = [record.get("backend_call_ms") for record in analyzed_records]
    parse_values = [record.get("parse_ms") for record in analyzed_records]
    commit_values = [record.get("state_commit_ms") for record in analyzed_records]
    total_stats = _numeric_stats(total_cycle_values)
    backend_stats = _numeric_stats(backend_call_values)
    parse_stats = _numeric_stats(parse_values)
    commit_stats = _numeric_stats(commit_values)

    loop_period_values_ms: list[float] = []
    for earlier, later in zip(commit_times_ns[:-1], commit_times_ns[1:]):
        loop_period_values_ms.append(float(later - earlier) / 1_000_000.0)
    loop_period_stats = _numeric_stats(loop_period_values_ms)

    effective_loop_rate_hz = None
    if analyzed_duration_s and analyzed_duration_s > 0.0 and len(analyzed_records) >= 2:
        effective_loop_rate_hz = float(len(analyzed_records) - 1) / float(analyzed_duration_s)

    comparable_records = [
        record
        for record in analyzed_records
        if record.get("is_new_frame") is not None or record.get("is_duplicate_frame") is not None
    ]
    new_frame_count = sum(bool(record.get("is_new_frame", False)) for record in comparable_records)
    duplicate_frame_count = sum(bool(record.get("is_duplicate_frame", False)) for record in comparable_records)
    duplicate_frame_ratio = None
    if comparable_records:
        duplicate_frame_ratio = float(duplicate_frame_count) / float(len(comparable_records))

    unique_frame_rate_hz = None
    if analyzed_duration_s and analyzed_duration_s > 0.0 and new_frame_count >= 2:
        unique_frame_rate_hz = float(new_frame_count - 1) / float(analyzed_duration_s)

    error_sample_count = sum(bool(record.get("error_flag", False)) for record in analyzed_records)
    invalid_or_missing_requested_tool_count = 0
    valid_transform_sample_count = 0
    raw_payload_missing_count = 0
    parsed_payload_missing_count = 0
    under_25ms_count = 0
    synthetic_frame_count = 0
    per_tool_summary: dict[str, dict[str, Any]] = {}
    for tool_id in requested:
        tracked_count = 0
        invalid_count = 0
        missing_count = 0
        present_count = 0
        for record in analyzed_records:
            validity = str((record.get("tool_validity", {}) or {}).get(tool_id, "missing")).strip().lower() or "missing"
            if tool_id in (record.get("tools_visible") or []):
                present_count += 1
            if validity == "tracked":
                tracked_count += 1
            elif validity == "invalid":
                invalid_count += 1
            else:
                missing_count += 1
        denominator = max(1, len(analyzed_records))
        per_tool_summary[tool_id] = {
            "tracked_count": tracked_count,
            "invalid_count": invalid_count,
            "missing_count": missing_count,
            "present_count": present_count,
            "valid_transform_rate": float(tracked_count) / float(denominator),
            "present_rate": float(present_count) / float(denominator),
        }

    for record in analyzed_records:
        if not bool(record.get("raw_payload_available", False)):
            raw_payload_missing_count += 1
        if not bool(record.get("parsed_payload_available", False)):
            parsed_payload_missing_count += 1
        if str(record.get("frame_number_source", "")).strip().lower() == "synthetic":
            synthetic_frame_count += 1
        total_cycle_ms = _as_float(record.get("total_cycle_ms"))
        if total_cycle_ms is not None and total_cycle_ms <= 25.0:
            under_25ms_count += 1
        requested_states = [
            str((record.get("tool_validity", {}) or {}).get(tool_id, "missing")).strip().lower() or "missing"
            for tool_id in requested
        ]
        if requested and any(state != "tracked" for state in requested_states):
            invalid_or_missing_requested_tool_count += 1
        if requested and all(state == "tracked" for state in requested_states):
            valid_transform_sample_count += 1

    analyzed_count = len(analyzed_records)
    summary = {
        "run_label": str(run_label or ""),
        "backend_identity": str(backend_identity or ""),
        "configured_backend_name": str(configured_backend_name or ""),
        "selected_backend_name": str(selected_backend_name or ""),
        "requested_tool_ids": list(requested),
        "sample_count_total": len(all_records),
        "sample_count_analyzed": analyzed_count,
        "warmup_discarded_count": sum(bool(record.get("warmup_discarded", False)) for record in all_records),
        "configured_run_duration_s": _as_float(run_duration_s),
        "analyzed_duration_s": analyzed_duration_s,
        "effective_loop_rate_hz": effective_loop_rate_hz,
        "unique_frame_rate_hz": unique_frame_rate_hz,
        "comparable_frame_count": len(comparable_records),
        "unique_frame_count": new_frame_count,
        "duplicate_frame_count": duplicate_frame_count,
        "duplicate_frame_ratio": duplicate_frame_ratio,
        "error_sample_count": error_sample_count,
        "invalid_or_missing_requested_tool_sample_count": invalid_or_missing_requested_tool_count,
        "invalid_or_missing_requested_tool_ratio": (
            float(invalid_or_missing_requested_tool_count) / float(analyzed_count)
            if analyzed_count > 0
            else None
        ),
        "valid_requested_tool_sample_count": valid_transform_sample_count,
        "valid_requested_tool_rate": (
            float(valid_transform_sample_count) / float(analyzed_count)
            if analyzed_count > 0
            else None
        ),
        "raw_payload_missing_count": raw_payload_missing_count,
        "parsed_payload_missing_count": parsed_payload_missing_count,
        "percent_under_25ms": (
            100.0 * float(under_25ms_count) / float(analyzed_count)
            if analyzed_count > 0
            else None
        ),
        "samples_under_25ms_count": under_25ms_count,
        "synthetic_frame_count": synthetic_frame_count,
        "synthetic_frame_ratio": (
            float(synthetic_frame_count) / float(analyzed_count)
            if analyzed_count > 0
            else None
        ),
        "total_cycle_ms_stats": total_stats,
        "backend_call_ms_stats": backend_stats,
        "parse_ms_stats": parse_stats,
        "state_commit_ms_stats": commit_stats,
        "loop_period_ms_stats": loop_period_stats,
        "mean_total_cycle_ms": total_stats["mean"],
        "median_total_cycle_ms": total_stats["median"],
        "p95_total_cycle_ms": total_stats["p95"],
        "p99_total_cycle_ms": total_stats["p99"],
        "mean_backend_call_ms": backend_stats["mean"],
        "p95_backend_call_ms": backend_stats["p95"],
        "mean_parse_ms": parse_stats["mean"],
        "p95_parse_ms": parse_stats["p95"],
        "mean_state_commit_ms": commit_stats["mean"],
        "p95_state_commit_ms": commit_stats["p95"],
        "per_tool_summary": per_tool_summary,
    }
    if servo_sync_summary:
        summary["servo_sync"] = dict(servo_sync_summary)
    return summary

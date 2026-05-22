"""Tests for the neutral setpoint drift validation experiment."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from continuum_robot.experiments.neutral_setpoint_drift_validation import (
    KNOWN_EVENT_TYPES,
    ACCEPTED_EVENT_TYPES,
    NEUTRAL_ZERO_LOG_SCHEMA_VERSION,
    NeutralSetpointDriftValidationConfig,
    NeutralSetpointDriftValidationExperiment,
    analyze_neutral_setpoint_drift_events,
    analyze_neutral_setpoint_drift_logs,
    list_neutral_setpoint_drift_candidates,
)


def _make_event(
    *,
    timestamp: str,
    event_type: str,
    by_servo: dict[int, dict],
    capture_source: str = "bench",
    accepted: bool | None = None,
    operator_note: str | None = None,
    active_segment_key: str = "segment_a",
) -> dict:
    return {
        "schema_version": NEUTRAL_ZERO_LOG_SCHEMA_VERSION,
        "timestamp_utc": timestamp,
        "event_type": event_type,
        "capture_source": capture_source,
        "operator_note": operator_note,
        "accepted": accepted,
        "active_segment_key": active_segment_key,
        "active_segment_label": "Spine 1" if active_segment_key == "segment_a" else "Spine 2",
        "active_segment_servo_ids": list(by_servo.keys()),
        "operating_mode": "single_segment",
        "robot_config_name": "robot_8servo.yaml",
        "mode_profile": "single_segment",
        "servo_count": len(by_servo),
        "by_servo": {str(int(sid)): dict(state) for sid, state in by_servo.items()},
    }


def _make_state(
    *,
    neutral: int,
    safe_offset: int = 600,
    final_position: int | None = None,
    final_current_ma: int | None = None,
    pretension_status: str | None = None,
    pretension_source: str | None = None,
) -> dict:
    return {
        "neutral_setpoint": neutral,
        "safe_min_tick": neutral - safe_offset,
        "safe_max_tick": neutral + safe_offset,
        "pretension_current_threshold_ma": 850,
        "pretension_final_position_tick": final_position,
        "pretension_result_status": pretension_status,
        "pretension_source": pretension_source,
        "pretension_note": None,
        "pretension_completed_at_utc": None,
        "last_measured_current_ma": final_current_ma,
        "tightening_rotation": "cw",
        "status": "neutral_captured",
        "valid": True,
        "calibrated_at_utc": None,
    }


# --------------------------------------------------------------------------- #
# Config + registration
# --------------------------------------------------------------------------- #


def test_config_default_values_are_permissive() -> None:
    """Defaults: no path filter, no event-type filter, no servo filter.
    The runtime resolves log_paths to the canonical project log if empty."""
    cfg = NeutralSetpointDriftValidationConfig.from_dict({})
    assert cfg.log_paths == []
    assert cfg.include_event_types == []
    assert cfg.include_servo_ids == []
    assert cfg.accepted_events_only is False


def test_config_from_dict_normalises_filters() -> None:
    cfg = NeutralSetpointDriftValidationConfig.from_dict(
        {
            "log_paths": ["data/calibration/neutral_zero_log.jsonl", "  /tmp/extra.jsonl  "],
            "include_event_types": ["Neutral_Captured", " manual_pretension_accepted "],
            "include_servo_ids": [1, 2, "3"],
            "accepted_events_only": True,
        }
    )
    assert cfg.log_paths == ["data/calibration/neutral_zero_log.jsonl", "  /tmp/extra.jsonl  "]
    assert cfg.include_event_types == ["neutral_captured", "manual_pretension_accepted"]
    assert cfg.include_servo_ids == [1, 2, 3]
    assert cfg.accepted_events_only is True


def test_known_event_types_includes_all_calibration_service_emissions() -> None:
    """Every event type the NeutralCalibrationService emits must be known to
    the analysis layer so we don't silently drop new event types."""
    expected = {
        "neutral_captured",
        "servo_calibration_saved",
        "manual_pretension_captured",
        "manual_pretension_accepted",
        "pretension_result_saved",
        "pretension_accepted",
    }
    assert expected.issubset(set(KNOWN_EVENT_TYPES))


# --------------------------------------------------------------------------- #
# Analysis — event filtering, sorting, per-servo statistics
# --------------------------------------------------------------------------- #


def test_analyze_events_orders_by_timestamp() -> None:
    """Events are sorted ascending by timestamp before per-servo analysis so
    the drift timeline is chronologically correct even when the log records
    arrived out of order (e.g. a back-fill or concurrent writer)."""
    events = [
        _make_event(
            timestamp="2026-05-22T12:00:00Z",
            event_type="neutral_captured",
            by_servo={1: _make_state(neutral=1810)},
        ),
        _make_event(
            timestamp="2026-05-22T08:00:00Z",
            event_type="neutral_captured",
            by_servo={1: _make_state(neutral=1800)},
        ),
        _make_event(
            timestamp="2026-05-22T10:00:00Z",
            event_type="neutral_captured",
            by_servo={1: _make_state(neutral=1805)},
        ),
    ]
    metrics, _ = analyze_neutral_setpoint_drift_events(events)
    timeline = metrics["timelines_by_servo"]["1"]
    assert [row["neutral_setpoint"] for row in timeline] == [1800, 1805, 1810]


def test_analyze_events_filters_by_event_type() -> None:
    """When ``include_event_types`` is set, only matching events make it into
    the analysis (used for "accepted-only" drift plots)."""
    events = [
        _make_event(
            timestamp="2026-05-22T08:00:00Z",
            event_type="neutral_captured",
            by_servo={1: _make_state(neutral=1800)},
        ),
        _make_event(
            timestamp="2026-05-22T09:00:00Z",
            event_type="manual_pretension_captured",
            by_servo={1: _make_state(neutral=1801)},
        ),
        _make_event(
            timestamp="2026-05-22T10:00:00Z",
            event_type="manual_pretension_accepted",
            by_servo={1: _make_state(neutral=1802)},
        ),
    ]
    metrics, _ = analyze_neutral_setpoint_drift_events(events, include_event_types=ACCEPTED_EVENT_TYPES)
    assert metrics["event_count"] == 2  # neutral_captured + manual_pretension_accepted
    types_in_metrics = set(metrics["event_type_counts"].keys())
    assert "manual_pretension_captured" not in types_in_metrics


def test_analyze_events_filters_by_servo_id() -> None:
    """``include_servo_ids`` narrows the per-servo timeline to a subset."""
    events = [
        _make_event(
            timestamp="2026-05-22T08:00:00Z",
            event_type="neutral_captured",
            by_servo={
                1: _make_state(neutral=1800),
                2: _make_state(neutral=2285),
                5: _make_state(neutral=2000),
            },
        ),
    ]
    metrics, _ = analyze_neutral_setpoint_drift_events(events, include_servo_ids=[1, 5])
    timelines = metrics["timelines_by_servo"]
    assert set(timelines.keys()) == {"1", "5"}
    assert metrics["servo_count"] == 2


def test_per_servo_drift_metrics_match_first_and_last_neutral() -> None:
    """The per-servo summary must report:
      - neutral_drift_ticks = last - first (signed)
      - max_excursion_from_first_ticks = max |value - first| (always >= 0)
    """
    events = [
        _make_event(
            timestamp=f"2026-05-22T{hour:02d}:00:00Z",
            event_type="neutral_captured",
            by_servo={1: _make_state(neutral=neutral)},
        )
        for hour, neutral in enumerate([1800, 1812, 1795, 1808])
    ]
    metrics, _ = analyze_neutral_setpoint_drift_events(events)
    row = next(r for r in metrics["per_servo_rows"] if r["servo_id"] == 1)
    assert row["event_count"] == 4
    assert row["first_neutral_setpoint"] == 1800
    assert row["last_neutral_setpoint"] == 1808
    # Signed end-to-end drift.
    assert row["neutral_drift_ticks"] == 8
    # max |1800-1800|=0, |1812-1800|=12, |1795-1800|=5, |1808-1800|=8 -> 12
    assert row["max_excursion_from_first_ticks"] == 12
    # And the summary statistics are populated.
    assert row["neutral_setpoint"]["count"] == 4
    assert row["neutral_setpoint"]["min"] == 1795
    assert row["neutral_setpoint"]["max"] == 1812


def test_analyze_events_handles_missing_neutral_gracefully() -> None:
    """Events whose ``by_servo`` entry has ``neutral_setpoint: None`` are
    still kept in the timeline (to preserve the chronological row count) but
    don't appear in the numeric drift calculations."""
    events = [
        _make_event(
            timestamp="2026-05-22T08:00:00Z",
            event_type="neutral_captured",
            by_servo={1: _make_state(neutral=1800)},
        ),
        _make_event(
            timestamp="2026-05-22T09:00:00Z",
            event_type="servo_calibration_saved",
            by_servo={1: {**_make_state(neutral=0), "neutral_setpoint": None}},
        ),
    ]
    metrics, _ = analyze_neutral_setpoint_drift_events(events)
    row = next(r for r in metrics["per_servo_rows"] if r["servo_id"] == 1)
    assert row["neutral_setpoint"]["count"] == 1  # only the one with a value


# --------------------------------------------------------------------------- #
# Log file loading (round-trip JSONL)
# --------------------------------------------------------------------------- #


def test_analyze_neutral_setpoint_drift_logs_reads_jsonl_file(tmp_path: Path) -> None:
    """Round-trip: write a JSONL log of events, read it back, verify the
    same per-servo metrics fall out. This is the file-IO entry point used
    by the experiment's ``execute`` method."""
    log_path = tmp_path / "neutral_zero_log.jsonl"
    events = [
        _make_event(
            timestamp="2026-05-22T08:00:00Z",
            event_type="neutral_captured",
            by_servo={
                sid: _make_state(neutral=1800 + sid * 100)
                for sid in (1, 2, 3, 4)
            },
        ),
        _make_event(
            timestamp="2026-05-22T10:00:00Z",
            event_type="manual_pretension_accepted",
            by_servo={
                sid: _make_state(
                    neutral=1800 + sid * 100,
                    final_position=1799 + sid * 100,
                    final_current_ma=-30,
                    pretension_status="accepted",
                    pretension_source="manual",
                )
                for sid in (1, 2, 3, 4)
            },
            capture_source="manual",
            accepted=True,
            operator_note="bench tensioning",
        ),
    ]
    with log_path.open("w", encoding="utf-8") as fh:
        for event in events:
            fh.write(json.dumps(event))
            fh.write("\n")

    metrics, warnings = analyze_neutral_setpoint_drift_logs([log_path])
    assert metrics["event_count"] == 2
    assert metrics["servo_count"] == 4
    assert metrics["event_type_counts"]["neutral_captured"] == 1
    assert metrics["event_type_counts"]["manual_pretension_accepted"] == 1
    assert warnings == []
    # Each per-servo row preserves the path the event came from.
    row1 = next(r for r in metrics["per_servo_rows"] if r["servo_id"] == 1)
    assert row1["event_count"] == 2
    timeline1 = metrics["timelines_by_servo"]["1"]
    assert timeline1[1]["_source_log_path"] == str(log_path)
    assert timeline1[1]["pretension_source"] == "manual"
    assert timeline1[1]["accepted"] is True


def test_analyze_logs_skips_records_with_unknown_schema_version(tmp_path: Path) -> None:
    """Future log-format changes must not silently mis-render in older code.
    A record whose ``schema_version`` doesn't match the constant is dropped."""
    log_path = tmp_path / "neutral_zero_log.jsonl"
    good_event = _make_event(
        timestamp="2026-05-22T08:00:00Z",
        event_type="neutral_captured",
        by_servo={1: _make_state(neutral=1800)},
    )
    future_event = dict(good_event)
    future_event["schema_version"] = "neutral_zero_log_v999"
    future_event["timestamp_utc"] = "2026-05-22T09:00:00Z"
    with log_path.open("w", encoding="utf-8") as fh:
        fh.write(json.dumps(good_event))
        fh.write("\n")
        fh.write(json.dumps(future_event))
        fh.write("\n")
    metrics, _ = analyze_neutral_setpoint_drift_logs([log_path])
    assert metrics["event_count"] == 1
    assert metrics["first_event_timestamp_utc"] == "2026-05-22T08:00:00Z"


def test_analyze_logs_handles_missing_file(tmp_path: Path) -> None:
    """A missing log file produces a warning, not an exception."""
    metrics, warnings = analyze_neutral_setpoint_drift_logs([tmp_path / "does_not_exist.jsonl"])
    assert metrics["event_count"] == 0
    assert any("not found" in msg.lower() for msg in warnings)


def test_analyze_logs_handles_malformed_jsonl_line(tmp_path: Path) -> None:
    """A non-JSON line is silently skipped — the rest of the log is still
    analyzed."""
    log_path = tmp_path / "neutral_zero_log.jsonl"
    good_event = _make_event(
        timestamp="2026-05-22T08:00:00Z",
        event_type="neutral_captured",
        by_servo={1: _make_state(neutral=1800)},
    )
    with log_path.open("w", encoding="utf-8") as fh:
        fh.write(json.dumps(good_event))
        fh.write("\n")
        fh.write("this is not valid JSON\n")
        fh.write("\n")  # blank line
    metrics, _ = analyze_neutral_setpoint_drift_logs([log_path])
    assert metrics["event_count"] == 1


# --------------------------------------------------------------------------- #
# Candidate discovery for the GUI page
# --------------------------------------------------------------------------- #


def test_list_candidates_finds_canonical_log(tmp_path: Path) -> None:
    """``list_neutral_setpoint_drift_candidates`` looks at
    ``data/calibration/neutral_zero_log.jsonl`` under the project root and
    returns a single candidate summarizing it."""
    log_path = tmp_path / "data" / "calibration" / "neutral_zero_log.jsonl"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    events = [
        _make_event(
            timestamp="2026-05-22T08:00:00Z",
            event_type="neutral_captured",
            by_servo={1: _make_state(neutral=1800), 2: _make_state(neutral=2285)},
        ),
        _make_event(
            timestamp="2026-05-22T10:00:00Z",
            event_type="manual_pretension_accepted",
            by_servo={1: _make_state(neutral=1801), 2: _make_state(neutral=2284)},
            accepted=True,
        ),
    ]
    with log_path.open("w", encoding="utf-8") as fh:
        for event in events:
            fh.write(json.dumps(event))
            fh.write("\n")

    candidates = list_neutral_setpoint_drift_candidates(tmp_path)
    assert len(candidates) == 1
    candidate = candidates[0]
    assert candidate.event_count == 2
    assert candidate.servo_count == 2
    assert candidate.first_timestamp_utc == "2026-05-22T08:00:00Z"
    assert candidate.last_timestamp_utc == "2026-05-22T10:00:00Z"
    assert "neutral_captured" in candidate.event_types
    assert "manual_pretension_accepted" in candidate.event_types
    # The path is rendered relative to project root.
    assert candidate.path.endswith("neutral_zero_log.jsonl")


def test_list_candidates_returns_empty_when_no_log_exists(tmp_path: Path) -> None:
    """An empty project_root (no log file) returns an empty list, not a
    crash. The GUI uses this to show "no logs found yet"."""
    candidates = list_neutral_setpoint_drift_candidates(tmp_path)
    assert candidates == []


# --------------------------------------------------------------------------- #
# End-to-end: experiment execution + output file generation
# --------------------------------------------------------------------------- #


def test_experiment_writes_4_thesis_figures_plus_debug_json(tmp_path: Path) -> None:
    """End-to-end: run the experiment, write outputs, confirm all four
    thesis-quality PNGs and debug.json are written to the canonical
    locations."""
    from continuum_robot.experiments.neutral_setpoint_drift_validation_outputs import (
        write_neutral_setpoint_drift_validation_outputs,
    )

    # Build a synthetic log with 6 events across 4 servos.
    events = [
        _make_event(
            timestamp=f"2026-05-{20 + (i % 3):02d}T{10 + i:02d}:00:00Z",
            event_type=("manual_pretension_accepted" if i % 2 else "neutral_captured"),
            by_servo={
                sid: _make_state(
                    neutral=1800 + sid * 100 + i * 2,
                    final_position=1799 + sid * 100,
                    final_current_ma=-25 - sid,
                )
                for sid in (1, 2, 3, 4)
            },
            accepted=(i % 2 == 1),
        )
        for i in range(6)
    ]
    metrics, _ = analyze_neutral_setpoint_drift_events(events)
    metadata = SimpleNamespace(run_id="test_run", timestamp_utc="2026-05-22T12:00:00Z")
    summary = SimpleNamespace(status="success", experiment_metrics=metrics)

    output_dir = tmp_path / "data" / "neutral_setpoint_drift_validation" / "20260522_test"
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = write_neutral_setpoint_drift_validation_outputs(
        output_dir=output_dir, metadata=metadata, summary=summary,
    )

    assert paths["thesis_01_path"].exists()
    assert paths["thesis_02_path"].exists()
    assert paths["thesis_03_path"].exists()
    assert paths["thesis_04_path"].exists()
    assert paths["debug_json_path"].exists()
    # Each PNG must be non-trivial (real figure render).
    for key in ("thesis_01_path", "thesis_02_path", "thesis_03_path", "thesis_04_path"):
        assert paths[key].stat().st_size > 10000, f"{key} too small (placeholder?)"

    debug = json.loads(paths["debug_json_path"].read_text(encoding="utf-8"))
    assert debug["schema_version"] == "neutral_setpoint_drift_validation_debug_v1"
    assert debug["event_count"] == 6
    assert debug["servo_count"] == 4
    assert len(debug["per_servo_rows"]) == 4
    assert len(debug["events"]) == 6


def test_experiment_class_metadata() -> None:
    """The experiment class declares the right name and mock-compatible
    hardware requirements (no live servos needed for offline analysis)."""
    assert NeutralSetpointDriftValidationExperiment.name == "neutral_setpoint_drift_validation"
    assert NeutralSetpointDriftValidationExperiment.hardware_requirements.mock_compatible is True
    assert (
        NeutralSetpointDriftValidationExperiment.hardware_requirements.servo_required is False
    )


def test_experiment_registers_with_registry() -> None:
    """The experiment must be discoverable via the GUI experiment registry
    so it shows up in the experiment-selection dropdown."""
    from continuum_robot.experiments.registry import ExperimentRegistry
    from continuum_robot.experiments.neutral_setpoint_drift_validation import (
        register_neutral_setpoint_drift_validation_experiment,
    )

    registry = ExperimentRegistry()
    register_neutral_setpoint_drift_validation_experiment(registry)
    entry = registry.get("neutral_setpoint_drift_validation")
    assert entry.title == "Neutral Setpoint Drift Validation"
    assert entry.category == "analysis"
    assert "Calibration" in entry.tags
    assert "Drift" in entry.tags


def test_register_builtin_experiments_includes_neutral_drift_validation() -> None:
    """The new experiment must be wired into register_builtin_experiments so
    the GUI's default registry contains it after bootstrap."""
    from continuum_robot.experiments.registry import ExperimentRegistry
    from continuum_robot.experiments.builtins import register_builtin_experiments

    registry = ExperimentRegistry()
    register_builtin_experiments(registry)
    entry = registry.get("neutral_setpoint_drift_validation")
    assert entry is not None
    assert entry.factory is not None

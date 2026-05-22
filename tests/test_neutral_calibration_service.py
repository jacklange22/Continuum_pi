import json
from pathlib import Path

from continuum_robot.servos.neutral_calibration_service import (
    NeutralCalibrationService,
    ServoCalibrationContext,
)


def test_neutral_calibration_service_archives_previous_latest(tmp_path: Path) -> None:
    service = NeutralCalibrationService(
        path=tmp_path / "neutral_setpoints.json",
        context=ServoCalibrationContext(
            robot_mode="4-servo",
            robot_config_name="robot_4servo.yaml",
            servo_ids=[1],
            tendon_to_servo=[1],
            position_min_offset_ticks=-600,
            position_max_offset_ticks=600,
            default_pretension_current_threshold_ma=850,
            tightening_rotation_by_servo={1: "cw"},
        ),
    )

    service.save_neutral_setpoints({1: 100}, capture_source="bench_neutral_capture")
    service.save_neutral_setpoints({1: 200}, capture_source="bench_neutral_capture")

    latest = json.loads((tmp_path / "neutral_setpoints.json").read_text(encoding="utf-8"))
    archives = sorted((tmp_path / "history").glob("*.json"))

    assert latest["schema_version"] == 4
    assert latest["servos"]["1"]["neutral_setpoint"] == 200
    assert latest["servos"]["1"]["safe_min_tick"] == -400
    assert latest["servos"]["1"]["safe_max_tick"] == 800
    assert latest["servos"]["1"]["pretension_current_threshold_ma"] == 850
    assert latest["servos"]["1"]["tightening_rotation"] == "cw"
    assert latest["servos"]["1"]["capture_source"] == "bench_neutral_capture"
    assert latest["servos"]["1"]["latest_pretension_run"] is None
    assert latest["robot"]["robot_config_name"] == "robot_4servo.yaml"
    assert len(archives) == 1
    assert archives[0].name.endswith("_neutral_setpoints.json")
    archived_payload = json.loads(archives[0].read_text(encoding="utf-8"))
    assert archived_payload["servos"]["1"]["neutral_setpoint"] == 100


def test_neutral_calibration_service_migrates_legacy_neutral_map(tmp_path: Path) -> None:
    path = tmp_path / "neutral_setpoints.json"
    path.write_text(json.dumps({"1": 1234, "2": 2345}), encoding="utf-8")
    service = NeutralCalibrationService(
        path=path,
        context=ServoCalibrationContext(
            robot_mode="4-servo",
            servo_ids=[1, 2],
            tendon_to_servo=[1, 2],
            position_min_offset_ticks=-100,
            position_max_offset_ticks=200,
            default_pretension_current_threshold_ma=700,
            tightening_rotation_by_servo={1: "cw", 2: "cw"},
        ),
    )

    artifact = service.load_calibration_artifact()
    summary = service.get_calibration_summary()

    assert artifact.source_format == "legacy_neutral_map"
    assert artifact.servos[1].neutral_setpoint == 1234
    assert artifact.servos[1].safe_min_tick == 1134
    assert artifact.servos[1].safe_max_tick == 1434
    assert summary.migrated_legacy_format is True
    assert summary.compatible is True


def test_neutral_calibration_service_reports_incompatible_robot_context(tmp_path: Path) -> None:
    service = NeutralCalibrationService(
        path=tmp_path / "neutral_setpoints.json",
        context=ServoCalibrationContext(
            robot_mode="4-servo",
            servo_ids=[1, 2, 3, 4],
            tendon_to_servo=[1, 2, 3, 4],
            tightening_rotation_by_servo={1: "cw", 2: "cw", 3: "cw", 4: "cw"},
        ),
    )
    service.save_neutral_setpoints({1: 100, 2: 200})

    incompatible = NeutralCalibrationService(
        path=service.path,
        context=ServoCalibrationContext(
            robot_mode="8-servo",
            servo_ids=[1, 2, 3, 4],
            tendon_to_servo=[1, 2, 3, 4],
            tightening_rotation_by_servo={1: "cw", 2: "cw", 3: "cw", 4: "cw"},
        ),
    )
    summary = incompatible.get_calibration_summary()

    assert summary.exists is True
    assert summary.compatible is False
    assert "does not match current mode" in summary.message


def test_neutral_calibration_service_marks_pretension_result_as_accepted(tmp_path: Path) -> None:
    service = NeutralCalibrationService(
        path=tmp_path / "neutral_setpoints.json",
        context=ServoCalibrationContext(
            robot_mode="1-servo",
            servo_ids=[1],
            tendon_to_servo=[1],
            position_min_offset_ticks=-200,
            position_max_offset_ticks=200,
            default_pretension_current_threshold_ma=220,
            tightening_rotation_by_servo={1: "cw"},
        ),
    )
    service.save_servo_calibration(
        servo_id=1,
        neutral_setpoint=2048,
        safe_min_tick=1948,
        safe_max_tick=2148,
        pretension_current_threshold_ma=220,
        tightening_rotation="cw",
    )
    service.save_pretension_result(
        servo_id=1,
        final_position_tick=2020,
        final_current_ma=225,
        threshold_ma=220,
        result_status="threshold_reached",
    )

    accepted = service.mark_pretension_accepted(1)

    assert accepted.pretension_result_status == "accepted"
    assert accepted.pretension_final_position_tick == 2020


def test_neutral_calibration_service_tracks_manual_pretension_source_summary(tmp_path: Path) -> None:
    service = NeutralCalibrationService(
        path=tmp_path / "neutral_setpoints.json",
        context=ServoCalibrationContext(
            robot_mode="4-servo",
            servo_ids=[1, 2, 3, 4],
            tendon_to_servo=[1, 2, 3, 4],
            position_min_offset_ticks=-200,
            position_max_offset_ticks=200,
            default_pretension_current_threshold_ma=220,
            tightening_rotation_by_servo={1: "cw", 2: "cw", 3: "cw", 4: "cw"},
        ),
    )
    service.save_neutral_setpoints({1: 2048, 2: 2048, 3: 2048, 4: 2048})

    service.save_manual_pretension_state(
        states_by_servo={
            servo_id: {
                "measured_position_tick": 2020 - servo_id,
                "measured_current_ma": 220 + servo_id,
            }
            for servo_id in [1, 2, 3, 4]
        },
        note="bench startup state",
        accepted=False,
    )
    pending = service.get_calibration_summary().pretension_source_summary([1, 2, 3, 4])
    assert pending.accepted is False
    assert pending.usable is False
    assert pending.source_type == "none"

    for servo_id in [1, 2, 3, 4]:
        service.mark_pretension_accepted(servo_id)

    accepted = service.get_calibration_summary().pretension_source_summary([1, 2, 3, 4])
    assert accepted.accepted is True
    assert accepted.usable is True
    assert accepted.source_type == "manual"
    assert accepted.note == "bench startup state"


def test_neutral_calibration_service_clears_manual_pretension_without_touching_neutral(tmp_path: Path) -> None:
    service = NeutralCalibrationService(
        path=tmp_path / "neutral_setpoints.json",
        context=ServoCalibrationContext(
            robot_mode="4-servo",
            servo_ids=[1, 2, 3, 4],
            tendon_to_servo=[1, 2, 3, 4],
            position_min_offset_ticks=-200,
            position_max_offset_ticks=200,
            default_pretension_current_threshold_ma=220,
            tightening_rotation_by_servo={1: "cw", 2: "cw", 3: "cw", 4: "cw"},
        ),
    )
    service.save_neutral_setpoints({1: 2048, 2: 2048, 3: 2048, 4: 2048})
    service.save_manual_pretension_state(
        states_by_servo={
            servo_id: {
                "measured_position_tick": 2020,
                "measured_current_ma": 225,
            }
            for servo_id in [1, 2, 3, 4]
        },
        accepted=True,
    )

    cleared = service.clear_manual_pretension_state([1, 2, 3, 4])
    summary = service.get_calibration_summary()

    assert cleared == [1, 2, 3, 4]
    assert summary.servo_entries[1].neutral_setpoint == 2048
    assert summary.servo_entries[1].pretension_result_status is None
    assert summary.servo_entries[1].pretension_source is None


# --------------------------------------------------------------------------- #
# Neutral/zero append-only history log — drift analysis over time.
# Operator wants to plot per-servo neutral_setpoint drift like the pivot
# validation does. Every write path (save_neutral_setpoints,
# save_servo_calibration, save_manual_pretension_state, mark_pretension_accepted,
# save_pretension_result) appends one record to the log.
# --------------------------------------------------------------------------- #


def _read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    out: list[dict] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        out.append(json.loads(line))
    return out


def test_neutral_zero_log_default_path_under_data_calibration(tmp_path: Path) -> None:
    """The default history-log path must sit one level above the per-snapshot
    archive root, so one file captures every neutral/zero update across all
    capture paths. Operators can grep / load that single file for drift
    analysis without touching the archive snapshots."""
    service = NeutralCalibrationService(
        path=tmp_path / "neutral_setpoints.json",
        context=ServoCalibrationContext(
            robot_mode="4-servo",
            robot_config_name="robot_4servo.yaml",
            servo_ids=[1],
            tendon_to_servo=[1],
            position_min_offset_ticks=-600,
            position_max_offset_ticks=600,
            default_pretension_current_threshold_ma=850,
            tightening_rotation_by_servo={1: "cw"},
        ),
    )
    # archive_root defaults to tmp_path/"history"; the log should land at
    # tmp_path/"neutral_zero_log.jsonl" (one level above).
    assert service.history_log_path == tmp_path / "neutral_zero_log.jsonl"


def test_save_neutral_setpoints_appends_one_log_record(tmp_path: Path) -> None:
    """``save_neutral_setpoints`` (the bulk capture path) must append exactly
    one line to the history log per call, with event_type=neutral_captured
    and a per-servo dict containing the new neutral_setpoint."""
    log_path = tmp_path / "neutral_zero_log.jsonl"
    service = NeutralCalibrationService(
        path=tmp_path / "neutral_setpoints.json",
        history_log_path=log_path,
        context=ServoCalibrationContext(
            robot_mode="4-servo",
            robot_config_name="robot_4servo.yaml",
            servo_ids=[1, 2, 3, 4],
            tendon_to_servo=[1, 2, 3, 4],
            active_segment_key="segment_a",
            active_segment_label="Spine 1",
            active_segment_servo_ids=[1, 2, 3, 4],
            position_min_offset_ticks=-600,
            position_max_offset_ticks=600,
            default_pretension_current_threshold_ma=850,
            tightening_rotation_by_servo={1: "cw", 2: "cw", 3: "cw", 4: "cw"},
        ),
    )

    service.save_neutral_setpoints({1: 1800, 2: 2285, 3: 2058, 4: 2723}, capture_source="bench_neutral_capture")

    records = _read_jsonl(log_path)
    assert len(records) == 1
    record = records[0]
    assert record["schema_version"] == "neutral_zero_log_v1"
    assert record["event_type"] == "neutral_captured"
    assert record["capture_source"] == "bench_neutral_capture"
    assert record["active_segment_key"] == "segment_a"
    assert record["active_segment_label"] == "Spine 1"
    assert record["operating_mode"] == "4-servo"
    assert record["servo_count"] == 4
    # All four servos' neutrals captured in this record.
    assert record["by_servo"]["1"]["neutral_setpoint"] == 1800
    assert record["by_servo"]["2"]["neutral_setpoint"] == 2285
    assert record["by_servo"]["3"]["neutral_setpoint"] == 2058
    assert record["by_servo"]["4"]["neutral_setpoint"] == 2723
    # Safe bounds derived from neutral +/- offset (the context's
    # position_min/max_offset_ticks defaults to -600/+600).
    assert record["by_servo"]["1"]["safe_min_tick"] == 1200
    assert record["by_servo"]["1"]["safe_max_tick"] == 2400
    # Timestamp present and ISO-ish.
    assert "timestamp_utc" in record
    assert "T" in record["timestamp_utc"] and "Z" in record["timestamp_utc"]


def test_neutral_zero_log_accumulates_across_multiple_writes(tmp_path: Path) -> None:
    """Append-only semantics: every write path appends a new record without
    overwriting the prior ones. Drift over time is plotted by reading every
    record in order. Verifies multiple updates produce multiple records."""
    log_path = tmp_path / "neutral_zero_log.jsonl"
    service = NeutralCalibrationService(
        path=tmp_path / "neutral_setpoints.json",
        history_log_path=log_path,
        context=ServoCalibrationContext(
            robot_mode="4-servo",
            robot_config_name="robot_4servo.yaml",
            servo_ids=[1],
            tendon_to_servo=[1],
            position_min_offset_ticks=-600,
            position_max_offset_ticks=600,
            default_pretension_current_threshold_ma=850,
            tightening_rotation_by_servo={1: "cw"},
        ),
    )

    # Three successive bulk captures (e.g. operator re-zeros each day).
    service.save_neutral_setpoints({1: 1800})
    service.save_neutral_setpoints({1: 1810})
    service.save_neutral_setpoints({1: 1795})

    records = _read_jsonl(log_path)
    assert len(records) == 3
    # The drift trace.
    neutrals_over_time = [int(r["by_servo"]["1"]["neutral_setpoint"]) for r in records]
    assert neutrals_over_time == [1800, 1810, 1795]


def test_save_servo_calibration_appends_log_record(tmp_path: Path) -> None:
    """The single-servo write path also appends to the log so the operator
    sees every neutral/bounds update no matter which API the caller used."""
    log_path = tmp_path / "neutral_zero_log.jsonl"
    service = NeutralCalibrationService(
        path=tmp_path / "neutral_setpoints.json",
        history_log_path=log_path,
        context=ServoCalibrationContext(
            robot_mode="4-servo",
            robot_config_name="robot_4servo.yaml",
            servo_ids=[1],
            tendon_to_servo=[1],
            position_min_offset_ticks=-600,
            position_max_offset_ticks=600,
            default_pretension_current_threshold_ma=850,
            tightening_rotation_by_servo={1: "cw"},
        ),
    )

    service.save_servo_calibration(
        servo_id=1,
        neutral_setpoint=2000,
        safe_min_tick=1400,
        safe_max_tick=2600,
        pretension_current_threshold_ma=850,
        tightening_rotation="cw",
    )

    records = _read_jsonl(log_path)
    assert len(records) == 1
    assert records[0]["event_type"] == "servo_calibration_saved"
    assert records[0]["by_servo"]["1"]["neutral_setpoint"] == 2000
    assert records[0]["by_servo"]["1"]["safe_min_tick"] == 1400
    assert records[0]["by_servo"]["1"]["safe_max_tick"] == 2600


def test_save_manual_pretension_state_appends_log_with_accepted_flag(tmp_path: Path) -> None:
    """``save_manual_pretension_state(accepted=True)`` must log a
    ``manual_pretension_accepted`` event so the drift plot can distinguish
    "captured but not accepted" from "accepted by operator". The note must
    also be preserved for context."""
    log_path = tmp_path / "neutral_zero_log.jsonl"
    service = NeutralCalibrationService(
        path=tmp_path / "neutral_setpoints.json",
        history_log_path=log_path,
        context=ServoCalibrationContext(
            robot_mode="4-servo",
            robot_config_name="robot_4servo.yaml",
            servo_ids=[1, 2, 3, 4],
            tendon_to_servo=[1, 2, 3, 4],
            active_segment_key="segment_a",
            active_segment_label="Spine 1",
            active_segment_servo_ids=[1, 2, 3, 4],
            position_min_offset_ticks=-600,
            position_max_offset_ticks=600,
            default_pretension_current_threshold_ma=850,
            tightening_rotation_by_servo={1: "cw", 2: "cw", 3: "cw", 4: "cw"},
        ),
    )

    # First seed neutrals so the manual pretension call has servos to update.
    service.save_neutral_setpoints({1: 1800, 2: 2285, 3: 2058, 4: 2723})
    service.save_manual_pretension_state(
        states_by_servo={
            sid: {"measured_position_tick": tick, "measured_current_ma": -30}
            for sid, tick in [(1, 1799), (2, 2284), (3, 2057), (4, 2722)]
        },
        note="bench tensioned manually",
        accepted=True,
    )

    records = _read_jsonl(log_path)
    # First event was the bulk neutral_captured; second is the manual pretension.
    assert len(records) == 2
    manual_event = records[1]
    assert manual_event["event_type"] == "manual_pretension_accepted"
    assert manual_event["accepted"] is True
    assert manual_event["capture_source"] == "manual"
    assert manual_event["operator_note"] == "bench tensioned manually"
    # Every servo's final position is captured in the per-servo record.
    assert manual_event["by_servo"]["1"]["pretension_final_position_tick"] == 1799
    assert manual_event["by_servo"]["1"]["pretension_source"] == "manual"
    assert manual_event["by_servo"]["1"]["last_measured_current_ma"] == -30


def test_save_manual_pretension_state_captured_but_not_accepted_emits_captured_event(tmp_path: Path) -> None:
    """When accepted=False the event_type must be ``manual_pretension_captured``
    so downstream readers can filter to only-accepted records for plots."""
    log_path = tmp_path / "neutral_zero_log.jsonl"
    service = NeutralCalibrationService(
        path=tmp_path / "neutral_setpoints.json",
        history_log_path=log_path,
        context=ServoCalibrationContext(
            robot_mode="4-servo",
            robot_config_name="robot_4servo.yaml",
            servo_ids=[1, 2, 3, 4],
            tendon_to_servo=[1, 2, 3, 4],
            position_min_offset_ticks=-600,
            position_max_offset_ticks=600,
            default_pretension_current_threshold_ma=850,
            tightening_rotation_by_servo={1: "cw", 2: "cw", 3: "cw", 4: "cw"},
        ),
    )
    service.save_neutral_setpoints({1: 1800, 2: 2285, 3: 2058, 4: 2723})
    service.save_manual_pretension_state(
        states_by_servo={
            sid: {"measured_position_tick": tick, "measured_current_ma": -25}
            for sid, tick in [(1, 1799), (2, 2284), (3, 2057), (4, 2722)]
        },
        note="snapshot for comparison",
        accepted=False,
    )
    records = _read_jsonl(log_path)
    assert any(r["event_type"] == "manual_pretension_captured" for r in records)
    assert all(r["accepted"] in (None, False) for r in records)


def test_mark_pretension_accepted_appends_log_record(tmp_path: Path) -> None:
    """``mark_pretension_accepted`` is the final-finalization step. It must
    emit a ``pretension_accepted`` record so the drift plot shows the
    operator-final state, not just the captured intermediate."""
    log_path = tmp_path / "neutral_zero_log.jsonl"
    service = NeutralCalibrationService(
        path=tmp_path / "neutral_setpoints.json",
        history_log_path=log_path,
        context=ServoCalibrationContext(
            robot_mode="4-servo",
            robot_config_name="robot_4servo.yaml",
            servo_ids=[1, 2, 3, 4],
            tendon_to_servo=[1, 2, 3, 4],
            position_min_offset_ticks=-600,
            position_max_offset_ticks=600,
            default_pretension_current_threshold_ma=850,
            tightening_rotation_by_servo={1: "cw", 2: "cw", 3: "cw", 4: "cw"},
        ),
    )
    service.save_neutral_setpoints({1: 1800, 2: 2285, 3: 2058, 4: 2723})
    service.save_manual_pretension_state(
        states_by_servo={
            sid: {"measured_position_tick": tick, "measured_current_ma": -28}
            for sid, tick in [(1, 1799), (2, 2284), (3, 2057), (4, 2722)]
        },
        accepted=False,
    )
    service.mark_pretension_accepted(1)
    records = _read_jsonl(log_path)
    # neutral_captured + manual_pretension_captured + pretension_accepted
    assert len(records) == 3
    final = records[-1]
    assert final["event_type"] == "pretension_accepted"
    assert final["accepted"] is True
    assert "1" in final["by_servo"]
    assert final["by_servo"]["1"]["pretension_result_status"] == "accepted"


def test_neutral_zero_log_survives_when_path_invalid(tmp_path: Path) -> None:
    """Logging failure must never raise — a permission error or invalid path
    should be silently swallowed so the primary calibration write succeeds.
    Operators on a read-only data dir still get a working pretension flow,
    just no drift log."""
    # Point the history log at a path that cannot be written (a file under a
    # path that's already a file, making mkdir fail).
    obstruction = tmp_path / "block.txt"
    obstruction.write_text("blocking", encoding="utf-8")
    invalid_log_path = obstruction / "subdir" / "neutral_zero_log.jsonl"
    service = NeutralCalibrationService(
        path=tmp_path / "neutral_setpoints.json",
        history_log_path=invalid_log_path,
        context=ServoCalibrationContext(
            robot_mode="4-servo",
            robot_config_name="robot_4servo.yaml",
            servo_ids=[1],
            tendon_to_servo=[1],
            position_min_offset_ticks=-600,
            position_max_offset_ticks=600,
            default_pretension_current_threshold_ma=850,
            tightening_rotation_by_servo={1: "cw"},
        ),
    )
    # Must not raise — the primary write still succeeds.
    service.save_neutral_setpoints({1: 1800})
    # Canonical artifact is written.
    assert (tmp_path / "neutral_setpoints.json").exists()

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

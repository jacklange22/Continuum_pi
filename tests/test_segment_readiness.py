from __future__ import annotations

from pathlib import Path

from continuum_robot.servos.neutral_calibration_service import NeutralCalibrationService, ServoCalibrationContext
from continuum_robot.servos.segment_readiness import STATUS_FAIL, STATUS_PASS, STATUS_WARN, evaluate_selected_segment_readiness
from continuum_robot.servos.sign_mapping_check import (
    ServoMappingCheckEntry,
    ServoMappingCheckRecord,
    ServoMappingCheckRepository,
    configured_axis_mapping,
)


def _context(*, active_segment: str = "segment_b") -> ServoCalibrationContext:
    segments = {
        "segment_a": {
            "key": "segment_a",
            "label": "Spine 1",
            "segment_label": "Segment A",
            "segment_role": "proximal",
            "segment_order_index": 0,
            "servo_ids": [1, 2, 3, 4],
            "pairs": {"axis_a": [1, 3], "axis_b": [2, 4]},
        },
        "segment_b": {
            "key": "segment_b",
            "label": "Spine 2",
            "segment_label": "Segment B",
            "segment_role": "distal",
            "segment_order_index": 1,
            "servo_ids": [5, 6, 7, 8],
            "pairs": {"axis_a": [5, 7], "axis_b": [6, 8]},
        },
    }
    active = segments[active_segment]
    return ServoCalibrationContext(
        robot_mode="single_segment",
        robot_config_name="robot_8servo.yaml",
        servo_ids=[1, 2, 3, 4, 5, 6, 7, 8],
        tendon_to_servo=[1, 2, 3, 4, 5, 6, 7, 8],
        active_segment_key=active_segment,
        active_segment_label=str(active["label"]),
        active_segment_servo_ids=[int(value) for value in active["servo_ids"]],
        active_segment_pairs=dict(active["pairs"]),
        segments=segments,
        segment_order=["segment_a", "segment_b"],
        expected_servo_ids=[int(value) for value in active["servo_ids"]],
        commanded_servo_ids=[int(value) for value in active["servo_ids"]],
        position_min_offset_ticks=-600,
        position_max_offset_ticks=600,
        default_pretension_current_threshold_ma=220,
        tightening_rotation_by_servo={servo_id: "cw" for servo_id in range(1, 9)},
    )


def _readiness(service: NeutralCalibrationService, *, active_segment: str = "segment_b"):
    context = _context(active_segment=active_segment)
    return evaluate_selected_segment_readiness(
        operating_mode="single_segment",
        active_segment_key=context.active_segment_key,
        active_segment_label=context.active_segment_label,
        expected_servo_ids=context.expected_servo_ids,
        calibration_summary=service.get_calibration_summary(),
        mock_mode=False,
        servo_connected=True,
        telemetry_rows={
            servo_id: {"position": 2048, "telemetry_fresh": True, "hardware_error": 0, "error": None}
            for servo_id in context.expected_servo_ids
        },
    )


def test_startup_artifact_without_neutral_safe_bounds_is_not_experiment_ready(tmp_path: Path) -> None:
    service = NeutralCalibrationService(path=tmp_path / "neutral.json", context=_context(active_segment="segment_b"))
    service.save_manual_pretension_state(
        states_by_servo={
            servo_id: {"measured_position_tick": 2100 + servo_id, "measured_current_ma": 180}
            for servo_id in [5, 6, 7, 8]
        },
        accepted=True,
    )

    readiness = _readiness(service, active_segment="segment_b")

    assert readiness.startup_pretension.ready is True
    assert readiness.startup_pretension.source_type == "manual_startup"
    assert readiness.neutral_safe_calibration.ready is False
    assert readiness.neutral_safe_calibration.status == STATUS_FAIL
    assert readiness.experiment.ready_for_repeatability is False
    assert "Startup reference exists" in readiness.next_action


def test_stale_display_cache_warns_without_marking_transport_missing(tmp_path: Path) -> None:
    service = NeutralCalibrationService(path=tmp_path / "neutral.json", context=_context(active_segment="segment_b"))
    context = _context(active_segment="segment_b")

    readiness = evaluate_selected_segment_readiness(
        operating_mode="single_segment",
        active_segment_key=context.active_segment_key,
        active_segment_label=context.active_segment_label,
        expected_servo_ids=context.expected_servo_ids,
        calibration_summary=service.get_calibration_summary(),
        mock_mode=False,
        servo_connected=True,
        telemetry_rows={
            servo_id: {
                "position": 2048,
                "telemetry_fresh": False,
                "packet_read_ok": True,
                "hardware_error": 0,
                "error": None,
            }
            for servo_id in context.expected_servo_ids
        },
    )

    assert readiness.transport.status == STATUS_WARN
    assert readiness.transport.missing_servo_ids == []
    assert "GUI display cache is stale" in readiness.transport.message
    assert "fresh pre-motion reads" in readiness.transport.message
    assert readiness.experiment.ready_for_raw_tiny_jog is True
    assert "fresh pre-motion read" in readiness.next_action


def test_mock_calibration_cannot_be_hardware_valid(tmp_path: Path) -> None:
    service = NeutralCalibrationService(path=tmp_path / "neutral.json", context=_context(active_segment="segment_b"))
    service.save_neutral_setpoints(
        {5: 2050, 6: 2051, 7: 2052, 8: 2053},
        calibration_metadata={"mock_mode": True, "calibration_trust": "mock", "valid_for_hardware_startup": False},
    )

    readiness = _readiness(service, active_segment="segment_b")

    assert readiness.neutral_safe_calibration.ready is False
    assert readiness.neutral_safe_calibration.is_mock is True
    assert "mock/debug" in readiness.neutral_safe_calibration.message


def test_mismatched_active_segment_artifact_fails_compatibility(tmp_path: Path) -> None:
    service = NeutralCalibrationService(path=tmp_path / "neutral.json", context=_context(active_segment="segment_b"))
    service.save_neutral_setpoints({1: 2048, 2: 2048, 3: 2048, 4: 2048})
    service_a = NeutralCalibrationService(path=service.path, context=_context(active_segment="segment_a"))

    readiness = _readiness(service_a, active_segment="segment_a")

    assert readiness.neutral_safe_calibration.ready is False
    assert readiness.neutral_safe_calibration.mismatch_reasons
    assert "active_segment_key" in readiness.neutral_safe_calibration.message


def test_segment_b_valid_neutral_safe_and_startup_passes(tmp_path: Path) -> None:
    service = NeutralCalibrationService(path=tmp_path / "neutral.json", context=_context(active_segment="segment_b"))
    service.save_neutral_setpoints({5: 2050, 6: 2051, 7: 2052, 8: 2053})
    service.save_manual_pretension_state(
        states_by_servo={
            servo_id: {"measured_position_tick": 2100 + servo_id, "measured_current_ma": 180}
            for servo_id in [5, 6, 7, 8]
        },
        accepted=True,
    )

    readiness = _readiness(service, active_segment="segment_b")

    assert readiness.expected_servo_ids == [5, 6, 7, 8]
    assert readiness.neutral_safe_calibration.status == STATUS_PASS
    assert readiness.startup_pretension.ready is True
    assert readiness.experiment.ready_for_repeatability is True


class _CachedRuntimeSnapshot:
    detected_servo_ids: list[int] = []
    missing_servo_ids: list[int] = [5, 6, 7, 8]
    entries: dict[int, object] = {servo_id: object() for servo_id in [5, 6, 7, 8]}
    message: str = "Showing cached servo state; cached=0/4."


def test_cached_empty_runtime_snapshot_is_unknown_not_missing(tmp_path: Path) -> None:
    service = NeutralCalibrationService(path=tmp_path / "neutral.json", context=_context(active_segment="segment_b"))
    context = _context(active_segment="segment_b")

    readiness = evaluate_selected_segment_readiness(
        operating_mode="single_segment",
        active_segment_key=context.active_segment_key,
        active_segment_label=context.active_segment_label,
        expected_servo_ids=context.expected_servo_ids,
        calibration_summary=service.get_calibration_summary(),
        mock_mode=False,
        servo_connected=True,
        runtime_snapshot=_CachedRuntimeSnapshot(),
    )

    assert readiness.transport.status == STATUS_WARN
    assert readiness.transport.missing_servo_ids == []
    assert "unknown, not confirmed missing" in readiness.transport.message


def test_sign_mapping_check_write_read_and_missing_warning(tmp_path: Path) -> None:
    repo = ServoMappingCheckRepository(tmp_path / "checks")
    missing = repo.latest_for_segment(active_segment_key="segment_b", expected_servo_ids=[5, 6, 7, 8])

    assert missing.exists is False
    assert missing.confirmed is False
    assert "not been recorded" in missing.message


def test_configured_segment_b_mapping_and_one_click_confirmation(tmp_path: Path) -> None:
    repo = ServoMappingCheckRepository(tmp_path / "checks")
    mapping = configured_axis_mapping(
        expected_servo_ids=[5, 6, 7, 8],
        active_segment_pairs={"axis_a": [5, 7], "axis_b": [6, 8]},
    )

    record = repo.confirm_configured_mapping(
        operating_mode="single_segment",
        active_segment_key="segment_b",
        active_segment_label="Segment B",
        expected_servo_ids=[5, 6, 7, 8],
        active_segment_pairs={"axis_a": [5, 7], "axis_b": [6, 8]},
        notes="hardware-day configured mapping",
    )
    summary = repo.latest_for_segment(active_segment_key="segment_b", expected_servo_ids=[5, 6, 7, 8])

    assert {servo_id: data["tendon"] for servo_id, data in mapping.items()} == {
        5: "+X",
        6: "+Y",
        7: "-X",
        8: "-Y",
    }
    assert record.lower_tick_means_tension is True
    assert record.confirmed_by_operator is True
    assert record.all_expected_confirmed() is True
    assert summary.confirmed is True
    assert summary.record is not None
    assert summary.record.entries[0].positive_tendon_direction == "+X"

    record = repo.write(
        ServoMappingCheckRecord(
            timestamp_utc="2026-05-12T12:00:00Z",
            operator="operator",
            operating_mode="single_segment",
            active_segment_key="segment_b",
            active_segment_label="Spine 2",
            expected_servo_ids=[5, 6, 7, 8],
            entries=[
                ServoMappingCheckEntry(
                    servo_id=servo_id,
                    tiny_jog_direction="tighten_fine",
                    observed_tendon_behavior="tendon tightened",
                    expected_sign_confirmed=True,
                )
                for servo_id in [5, 6, 7, 8]
            ],
        )
    )
    loaded = repo.read(Path(record.artifact_path))
    latest = repo.latest_for_segment(active_segment_key="segment_b", expected_servo_ids=[5, 6, 7, 8])

    assert loaded.expected_servo_ids == [5, 6, 7, 8]
    assert latest.exists is True
    assert latest.confirmed is True
    assert latest.record is not None
    assert latest.record.confirmed_servo_ids == [5, 6, 7, 8]

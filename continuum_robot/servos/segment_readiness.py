"""Canonical selected-segment servo readiness model."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

from continuum_robot.servos.neutral_calibration_service import ServoCalibrationSummary
from continuum_robot.servos.sign_mapping_check import ServoMappingCheckSummary


STATUS_PASS = "PASS"
STATUS_WARN = "WARN"
STATUS_FAIL = "FAIL"


@dataclass(frozen=True)
class ServoTransportReadiness:
    """Transport/readback readiness for the selected servo set."""

    status: str
    expected_servo_ids: list[int]
    responding_servo_ids: list[int]
    missing_servo_ids: list[int]
    stale_servo_ids: list[int]
    hardware_error_servo_ids: list[int]
    telemetry_by_servo: dict[int, dict[str, Any]] = field(default_factory=dict)
    message: str = ""


@dataclass(frozen=True)
class NeutralSafeCalibrationReadiness:
    """Neutral and safe-bound calibration truth for the selected segment."""

    status: str
    ready: bool
    hardware_valid: bool
    artifact_path: str
    updated_at_utc: str | None
    source: str
    is_mock: bool
    missing_neutral_servo_ids: list[int]
    missing_bounds_servo_ids: list[int]
    invalid_bounds_servo_ids: list[int]
    mismatch_reasons: list[str]
    message: str


@dataclass(frozen=True)
class StartupPretensionReadiness:
    """Accepted startup/manual pretension reference truth."""

    status: str
    ready: bool
    accepted: bool
    usable: bool
    artifact_path: str
    updated_at_utc: str | None
    source_type: str
    raw_source_type: str
    is_mock: bool
    mismatch_reasons: list[str]
    positions_by_servo: dict[int, int | None] = field(default_factory=dict)
    message: str = ""


@dataclass(frozen=True)
class ExperimentReadiness:
    """Workflow gate outcomes derived from transport/calibration/startup truth."""

    ready_for_manual_jog: bool
    ready_for_raw_tiny_jog: bool
    ready_for_calibrated_motion: bool
    ready_for_pretension_validation: bool
    ready_for_repeatability: bool
    ready_for_collect_pose: bool
    ready_for_penprobe_chasing: bool
    raw_tiny_jog_label: str
    message: str


@dataclass(frozen=True)
class SelectedSegmentReadiness:
    """Canonical readiness state for the active operating mode and segment."""

    operating_mode: str
    active_segment_key: str
    active_segment_label: str
    expected_servo_ids: list[int]
    transport: ServoTransportReadiness
    neutral_safe_calibration: NeutralSafeCalibrationReadiness
    startup_pretension: StartupPretensionReadiness
    experiment: ExperimentReadiness
    sign_mapping: ServoMappingCheckSummary | None = None
    next_action: str = ""

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        if self.sign_mapping is None:
            payload["sign_mapping"] = None
        return payload

    def compact_lines(self) -> list[str]:
        mapping_message = self.sign_mapping.message if self.sign_mapping is not None else ""
        lines = [
            f"Active segment: {self.active_segment_label or self.active_segment_key} ({self.active_segment_key or 'unknown'})",
            f"Expected servo IDs: {self.expected_servo_ids}",
            f"Servo transport: {self.transport.status} - {self.transport.message}",
            f"Neutral/safe calibration: {self.neutral_safe_calibration.status} - {self.neutral_safe_calibration.message}",
            f"Startup/pretension reference: {self.startup_pretension.status} - {self.startup_pretension.message}",
        ]
        if mapping_message:
            mapping_status = STATUS_PASS if self.sign_mapping and self.sign_mapping.confirmed else STATUS_WARN
            lines.append(f"Servo sign/mapping: {mapping_status} - {mapping_message}")
        lines.append(f"Raw tiny jog: {self.experiment.raw_tiny_jog_label}")
        lines.append(f"Next: {self.next_action}")
        return lines

    def compact_text(self) -> str:
        return "\n".join(self.compact_lines())


def evaluate_selected_segment_readiness(
    *,
    operating_mode: str,
    active_segment_key: str | None,
    active_segment_label: str | None,
    expected_servo_ids: list[int],
    calibration_summary: ServoCalibrationSummary | None,
    mock_mode: bool,
    servo_connected: bool = False,
    runtime_snapshot: Any | None = None,
    telemetry_rows: dict[int, dict[str, Any]] | None = None,
    sign_mapping: ServoMappingCheckSummary | None = None,
) -> SelectedSegmentReadiness:
    """Build the canonical selected-segment readiness model."""

    expected = [int(value) for value in expected_servo_ids]
    mode = str(operating_mode or "").strip().lower().replace("-", "_")
    segment_key = str(active_segment_key or "")
    segment_label = str(active_segment_label or segment_key or "active segment")
    transport = _evaluate_transport(
        expected_servo_ids=expected,
        servo_connected=bool(servo_connected),
        runtime_snapshot=runtime_snapshot,
        telemetry_rows=telemetry_rows,
    )
    neutral = _evaluate_neutral_safe(
        expected_servo_ids=expected,
        operating_mode=mode,
        active_segment_key=segment_key,
        active_segment_label=segment_label,
        calibration_summary=calibration_summary,
        runtime_mock_mode=bool(mock_mode),
    )
    startup = _evaluate_startup(
        expected_servo_ids=expected,
        calibration_summary=calibration_summary,
        neutral_ready=neutral.ready,
    )
    experiment = _evaluate_experiments(
        transport=transport,
        neutral=neutral,
        startup=startup,
        mock_mode=bool(mock_mode),
    )
    next_action = _next_action(
        active_segment_label=segment_label,
        expected_servo_ids=expected,
        transport=transport,
        neutral=neutral,
        startup=startup,
        sign_mapping=sign_mapping,
    )
    return SelectedSegmentReadiness(
        operating_mode=mode,
        active_segment_key=segment_key,
        active_segment_label=segment_label,
        expected_servo_ids=expected,
        transport=transport,
        neutral_safe_calibration=neutral,
        startup_pretension=startup,
        experiment=experiment,
        sign_mapping=sign_mapping,
        next_action=next_action,
    )


def _evaluate_transport(
    *,
    expected_servo_ids: list[int],
    servo_connected: bool,
    runtime_snapshot: Any | None,
    telemetry_rows: dict[int, dict[str, Any]] | None,
) -> ServoTransportReadiness:
    rows = {int(key): dict(value or {}) for key, value in dict(telemetry_rows or {}).items()}
    if runtime_snapshot is not None:
        responding = [int(value) for value in getattr(runtime_snapshot, "detected_servo_ids", []) or []]
        missing = [int(value) for value in getattr(runtime_snapshot, "missing_servo_ids", []) or []]
    else:
        responding = sorted(
            int(servo_id)
            for servo_id, row in rows.items()
            if row.get("position") is not None or row.get("telemetry_status") not in {"Unreadable", "Missing"}
        )
        missing = [int(value) for value in expected_servo_ids if int(value) not in responding]
    if not rows and runtime_snapshot is not None:
        for servo_id, entry in dict(getattr(runtime_snapshot, "entries", {}) or {}).items():
            telemetry = getattr(entry, "telemetry", None)
            rows[int(servo_id)] = _telemetry_row_from_object(telemetry)
    stale = sorted(
        int(servo_id)
        for servo_id, row in rows.items()
        if int(servo_id) in expected_servo_ids and row.get("telemetry_fresh") is False
    )
    errors = sorted(
        int(servo_id)
        for servo_id, row in rows.items()
        if int(servo_id) in expected_servo_ids
        and (row.get("hardware_error") not in (None, 0, "", False) or bool(row.get("error")))
    )
    missing = sorted({int(value) for value in missing if int(value) in expected_servo_ids})
    if not servo_connected:
        status = STATUS_WARN
        message = "OpenRB is disconnected; transport has not been validated."
    elif runtime_snapshot is not None and not responding and not any(bool(row) for row in rows.values()):
        status = STATUS_WARN
        missing = []
        message = (
            "Live readiness scan unavailable; no cached servo telemetry was available. "
            "Transport is unknown, not confirmed missing."
        )
    elif missing:
        status = STATUS_FAIL
        message = "missing servo ID(s): " + ", ".join(str(value) for value in missing) + "."
    elif stale:
        status = STATUS_WARN
        message = (
            "GUI display cache is stale for servo ID(s): "
            + ", ".join(str(value) for value in stale)
            + ". Experiments perform fresh pre-motion reads before commanding."
        )
    elif errors:
        status = STATUS_FAIL
        message = "Hardware/telemetry errors on servo ID(s): " + ", ".join(str(value) for value in errors) + "."
    else:
        status = STATUS_PASS if expected_servo_ids else STATUS_WARN
        message = (
            "Readable packet telemetry is available for expected servo IDs."
            if expected_servo_ids
            else "No expected servo IDs are configured."
        )
    return ServoTransportReadiness(
        status=status,
        expected_servo_ids=list(expected_servo_ids),
        responding_servo_ids=sorted({int(value) for value in responding}),
        missing_servo_ids=missing,
        stale_servo_ids=stale,
        hardware_error_servo_ids=errors,
        telemetry_by_servo=rows,
        message=message,
    )


def _evaluate_neutral_safe(
    *,
    expected_servo_ids: list[int],
    operating_mode: str,
    active_segment_key: str,
    active_segment_label: str,
    calibration_summary: ServoCalibrationSummary | None,
    runtime_mock_mode: bool,
) -> NeutralSafeCalibrationReadiness:
    if calibration_summary is None:
        return _neutral_result(
            status=STATUS_FAIL,
            ready=False,
            message="Servo calibration artifact state is unavailable.",
        )
    artifact_path = str(getattr(calibration_summary, "path", "") or "")
    updated = getattr(calibration_summary, "updated_at_utc", None)
    if not calibration_summary.exists:
        return _neutral_result(
            status=STATUS_FAIL,
            ready=False,
            artifact_path=artifact_path,
            updated_at_utc=updated,
            message="Neutral/safe calibration artifact is missing.",
        )
    if not calibration_summary.compatible:
        return _neutral_result(
            status=STATUS_FAIL,
            ready=False,
            artifact_path=artifact_path,
            updated_at_utc=updated,
            message=f"Calibration artifact is incompatible: {calibration_summary.message}",
            mismatch_reasons=[calibration_summary.message],
        )
    robot = dict(getattr(calibration_summary, "robot_metadata", {}) or {})
    mismatches = _artifact_segment_mismatches(
        robot=robot,
        operating_mode=operating_mode,
        active_segment_key=active_segment_key,
        expected_servo_ids=expected_servo_ids,
    )
    missing_neutral: list[int] = []
    missing_bounds: list[int] = []
    invalid_bounds: list[int] = []
    for servo_id in expected_servo_ids:
        entry = dict(getattr(calibration_summary, "servo_entries", {}) or {}).get(int(servo_id))
        if entry is None or entry.neutral_setpoint is None or not bool(entry.valid):
            missing_neutral.append(int(servo_id))
            continue
        if entry.safe_min_tick is None or entry.safe_max_tick is None:
            missing_bounds.append(int(servo_id))
            continue
        if int(entry.safe_min_tick) > int(entry.neutral_setpoint) or int(entry.safe_max_tick) < int(entry.neutral_setpoint):
            invalid_bounds.append(int(servo_id))
        if int(entry.safe_min_tick) >= int(entry.safe_max_tick):
            invalid_bounds.append(int(servo_id))
    invalid_bounds = sorted(set(invalid_bounds))
    is_mock = bool(getattr(calibration_summary, "is_mock_calibration", False))
    source = str(getattr(calibration_summary, "calibration_trust", "unknown") or "unknown")
    if is_mock:
        return NeutralSafeCalibrationReadiness(
            status=STATUS_FAIL if not runtime_mock_mode else STATUS_WARN,
            ready=False,
            hardware_valid=False,
            artifact_path=artifact_path,
            updated_at_utc=updated,
            source=source,
            is_mock=True,
            missing_neutral_servo_ids=missing_neutral,
            missing_bounds_servo_ids=missing_bounds,
            invalid_bounds_servo_ids=invalid_bounds,
            mismatch_reasons=mismatches,
            message="Neutral calibration is mock/debug and is not valid for hardware startup.",
        )
    if mismatches:
        return NeutralSafeCalibrationReadiness(
            status=STATUS_FAIL,
            ready=False,
            hardware_valid=False,
            artifact_path=artifact_path,
            updated_at_utc=updated,
            source=source,
            is_mock=False,
            missing_neutral_servo_ids=missing_neutral,
            missing_bounds_servo_ids=missing_bounds,
            invalid_bounds_servo_ids=invalid_bounds,
            mismatch_reasons=mismatches,
            message="Calibration artifact does not match the active segment: " + "; ".join(mismatches),
        )
    if missing_neutral or missing_bounds or invalid_bounds:
        parts: list[str] = []
        if missing_neutral:
            parts.append("missing neutral ticks for " + ", ".join(str(value) for value in missing_neutral))
        if missing_bounds:
            parts.append("missing safe bounds for " + ", ".join(str(value) for value in missing_bounds))
        if invalid_bounds:
            parts.append("invalid safe bounds for " + ", ".join(str(value) for value in invalid_bounds))
        return NeutralSafeCalibrationReadiness(
            status=STATUS_FAIL,
            ready=False,
            hardware_valid=False,
            artifact_path=artifact_path,
            updated_at_utc=updated,
            source=source,
            is_mock=False,
            missing_neutral_servo_ids=missing_neutral,
            missing_bounds_servo_ids=missing_bounds,
            invalid_bounds_servo_ids=invalid_bounds,
            mismatch_reasons=[],
            message=(
                f"{active_segment_label} neutral/safe calibration is incomplete: "
                + "; ".join(parts)
                + "."
            ),
        )
    return NeutralSafeCalibrationReadiness(
        status=STATUS_PASS,
        ready=True,
        hardware_valid=True,
        artifact_path=artifact_path,
        updated_at_utc=updated,
        source=source,
        is_mock=False,
        missing_neutral_servo_ids=[],
        missing_bounds_servo_ids=[],
        invalid_bounds_servo_ids=[],
        mismatch_reasons=[],
        message=(
            f"{active_segment_label} has neutral ticks and calibrated safe bounds for servo IDs "
            + ", ".join(str(value) for value in expected_servo_ids)
            + "."
        ),
    )


def _evaluate_startup(
    *,
    expected_servo_ids: list[int],
    calibration_summary: ServoCalibrationSummary | None,
    neutral_ready: bool,
) -> StartupPretensionReadiness:
    if calibration_summary is None:
        return _startup_result(
            status=STATUS_FAIL,
            ready=False,
            message="Startup/pretension artifact state is unavailable.",
        )
    artifact_path = str(getattr(calibration_summary, "path", "") or "")
    if not calibration_summary.exists or not calibration_summary.compatible:
        return _startup_result(
            status=STATUS_FAIL,
            ready=False,
            artifact_path=artifact_path,
            updated_at_utc=getattr(calibration_summary, "updated_at_utc", None),
            message=f"Servo calibration artifact is not ready: {calibration_summary.message}",
            mismatch_reasons=[calibration_summary.message],
        )
    source_summary = calibration_summary.pretension_source_summary(expected_servo_ids)
    normalized_source = _normalized_startup_source(getattr(source_summary, "source_type", "unknown"), calibration_summary, expected_servo_ids)
    positions = {
        int(servo_id): (
            int(position)
            if position is not None
            else None
        )
        for servo_id, position in dict(source_summary.positions_by_servo or {}).items()
        if int(servo_id) in expected_servo_ids
    }
    missing_positions = [int(value) for value in expected_servo_ids if positions.get(int(value)) is None]
    status = STATUS_PASS
    ready = bool(getattr(source_summary, "accepted", False) and getattr(source_summary, "usable", False) and not missing_positions)
    message = str(getattr(source_summary, "message", "") or "")
    if not ready:
        status = STATUS_FAIL if neutral_ready else STATUS_WARN
        if missing_positions and getattr(source_summary, "accepted", False):
            message = (
                "Accepted startup/pretension reference is missing positions for servo(s): "
                + ", ".join(str(value) for value in missing_positions)
            )
    is_mock = bool(getattr(calibration_summary, "is_mock_calibration", False))
    if is_mock:
        status = STATUS_WARN
        message = message + " Reference is from mock/debug calibration and is not hardware-valid."
    return StartupPretensionReadiness(
        status=status,
        ready=ready and not is_mock,
        accepted=bool(getattr(source_summary, "accepted", False)),
        usable=bool(getattr(source_summary, "usable", False)),
        artifact_path=artifact_path,
        updated_at_utc=getattr(source_summary, "updated_at_utc", None) or getattr(calibration_summary, "updated_at_utc", None),
        source_type=normalized_source,
        raw_source_type=str(getattr(source_summary, "source_type", "none") or "none"),
        is_mock=is_mock,
        mismatch_reasons=[],
        positions_by_servo=positions,
        message=message,
    )


def _evaluate_experiments(
    *,
    transport: ServoTransportReadiness,
    neutral: NeutralSafeCalibrationReadiness,
    startup: StartupPretensionReadiness,
    mock_mode: bool,
) -> ExperimentReadiness:
    transport_ok = transport.status == STATUS_PASS
    expected_ids = {int(value) for value in transport.expected_servo_ids}
    responding_ids = {int(value) for value in transport.responding_servo_ids}
    stale_display_only = bool(
        transport.status == STATUS_WARN
        and transport.stale_servo_ids
        and not transport.missing_servo_ids
        and not transport.hardware_error_servo_ids
        and expected_ids.issubset(responding_ids)
    )
    raw_jog = bool(transport_ok or stale_display_only)
    calibrated = bool(transport_ok and neutral.ready)
    startup_ready = bool(calibrated and startup.ready)
    if raw_jog and not calibrated:
        raw_label = "Allowed as raw/hardware-bound bring-up only; calibrated motion remains blocked."
    elif raw_jog:
        raw_label = "Allowed; calibration is available, but tiny jog remains a raw hardware-bound action."
    else:
        raw_label = "Blocked until selected servos are connected, readable, and error-free."
    if mock_mode:
        message = "Mock/debug runtime: runs may execute only as non-thesis/non-training output."
    elif startup_ready:
        message = "Trusted single-segment experiment preflight may proceed after tracker/registration checks."
    elif calibrated:
        message = "Neutral/safe calibration is ready; capture or validate startup/pretension reference before trusted runs."
    elif raw_jog:
        message = "Raw tiny jog is available for bring-up; capture valid neutral/safe bounds before calibrated experiments."
    else:
        message = "Transport is not ready for selected segment bring-up."
    return ExperimentReadiness(
        ready_for_manual_jog=raw_jog,
        ready_for_raw_tiny_jog=raw_jog,
        ready_for_calibrated_motion=calibrated and not mock_mode,
        ready_for_pretension_validation=raw_jog,
        ready_for_repeatability=startup_ready and not mock_mode,
        ready_for_collect_pose=startup_ready and not mock_mode,
        ready_for_penprobe_chasing=startup_ready and not mock_mode,
        raw_tiny_jog_label=raw_label,
        message=message,
    )


def _next_action(
    *,
    active_segment_label: str,
    expected_servo_ids: list[int],
    transport: ServoTransportReadiness,
    neutral: NeutralSafeCalibrationReadiness,
    startup: StartupPretensionReadiness,
    sign_mapping: ServoMappingCheckSummary | None,
) -> str:
    if transport.missing_servo_ids:
        return (
            f"{active_segment_label} expects servos {expected_servo_ids}. Missing: "
            f"{transport.missing_servo_ids}. Connect/scan before calibration."
        )
    if transport.status == STATUS_WARN:
        if transport.stale_servo_ids and not transport.missing_servo_ids and not transport.hardware_error_servo_ids:
            return (
                "GUI display cache is stale; refresh the display or proceed to an experiment "
                "that performs a fresh pre-motion read."
            )
        return "Refresh servo telemetry and resolve readback warnings before calibration."
    if transport.status == STATUS_FAIL:
        return transport.message
    if neutral.is_mock:
        return "Capture hardware neutral/safe calibration; mock/debug calibration is not valid for hardware startup."
    if not neutral.ready:
        if startup.accepted:
            return (
                "Startup reference exists, but calibrated neutral/safe bounds are missing or incompatible. "
                "Raw tiny jog is allowed for bring-up; calibrated experiments remain blocked."
            )
        return "Capture valid neutral/safe calibration before repeatability."
    if sign_mapping is not None and not sign_mapping.confirmed:
        return "Run tiny jog sign checklist before pretension."
    if not startup.ready:
        return "Capture manual startup or run conservative pretension validation before repeatability."
    return (
        f"{active_segment_label} has fresh telemetry, valid neutral/safe calibration, and accepted startup reference. "
        "Pretension/repeatability preflight may proceed."
    )


def _artifact_segment_mismatches(
    *,
    robot: dict[str, Any],
    operating_mode: str,
    active_segment_key: str,
    expected_servo_ids: list[int],
) -> list[str]:
    if operating_mode != "single_segment":
        return []
    mismatches: list[str] = []
    saved_active_key = str(robot.get("active_segment_key", "") or "")
    if saved_active_key and active_segment_key and saved_active_key != active_segment_key:
        mismatches.append(f"artifact active_segment_key={saved_active_key}, current={active_segment_key}")
    saved_active_ids = _int_list(robot.get("active_segment_servo_ids"))
    if saved_active_ids and sorted(saved_active_ids) != sorted(expected_servo_ids):
        mismatches.append(f"artifact active_segment_servo_ids={saved_active_ids}, current={expected_servo_ids}")
    saved_expected_ids = _int_list(robot.get("expected_servo_ids"))
    if saved_expected_ids and sorted(saved_expected_ids) != sorted(expected_servo_ids):
        mismatches.append(f"artifact expected_servo_ids={saved_expected_ids}, current={expected_servo_ids}")
    return mismatches


def _normalized_startup_source(
    raw_source_type: str,
    calibration_summary: ServoCalibrationSummary,
    servo_ids: list[int],
) -> str:
    raw = str(raw_source_type or "unknown").strip().lower()
    if raw == "algorithmic":
        return "automatic_pretension_validation"
    if raw == "manual":
        records = []
        entries = dict(getattr(calibration_summary, "servo_entries", {}) or {})
        for servo_id in servo_ids:
            entry = entries.get(int(servo_id))
            if entry is not None and entry.latest_pretension_run:
                records.append(dict(entry.latest_pretension_run))
        if any(record.get("startup_artifact_scope") for record in records):
            return "manual_startup"
        return "manual_pretension"
    if raw in {"none", ""}:
        return "unknown"
    return raw


def _neutral_result(
    *,
    status: str,
    ready: bool,
    message: str,
    artifact_path: str = "",
    updated_at_utc: str | None = None,
    mismatch_reasons: list[str] | None = None,
) -> NeutralSafeCalibrationReadiness:
    return NeutralSafeCalibrationReadiness(
        status=status,
        ready=ready,
        hardware_valid=False,
        artifact_path=artifact_path,
        updated_at_utc=updated_at_utc,
        source="unknown",
        is_mock=False,
        missing_neutral_servo_ids=[],
        missing_bounds_servo_ids=[],
        invalid_bounds_servo_ids=[],
        mismatch_reasons=list(mismatch_reasons or []),
        message=message,
    )


def _startup_result(
    *,
    status: str,
    ready: bool,
    message: str,
    artifact_path: str = "",
    updated_at_utc: str | None = None,
    mismatch_reasons: list[str] | None = None,
) -> StartupPretensionReadiness:
    return StartupPretensionReadiness(
        status=status,
        ready=ready,
        accepted=False,
        usable=False,
        artifact_path=artifact_path,
        updated_at_utc=updated_at_utc,
        source_type="unknown",
        raw_source_type="none",
        is_mock=False,
        mismatch_reasons=list(mismatch_reasons or []),
        positions_by_servo={},
        message=message,
    )


def _telemetry_row_from_object(telemetry: Any) -> dict[str, Any]:
    if telemetry is None:
        return {}
    return {
        "position": getattr(telemetry, "present_position", None),
        "current_ma": getattr(telemetry, "present_current_ma", None),
        "voltage_mv": getattr(telemetry, "present_voltage_mv", None),
        "temperature_c": getattr(telemetry, "present_temperature_c", None),
        "hardware_error": getattr(telemetry, "hardware_error_code", None),
        "error": getattr(telemetry, "hardware_error", None)
        or getattr(telemetry, "identity_error", None)
        or getattr(telemetry, "telemetry_error", None),
    }


def _int_list(value: Any) -> list[int]:
    result: list[int] = []
    for item in list(value or []):
        if item is None:
            continue
        try:
            result.append(int(item))
        except Exception:
            continue
    return result

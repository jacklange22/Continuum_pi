"""Servo calibration artifact storage.

This service preserves the old neutral-setpoint API while storing a richer
single-file calibration artifact for future neutral/pretension workflows.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
import json
from typing import Any

from continuum_robot.experiments.dataset_io import canonical_timestamped_path


SCHEMA_VERSION = 4


@dataclass
class ServoCalibrationContext:
    """Robot/config context used to validate one saved calibration artifact."""

    robot_mode: str | None = None
    robot_config_name: str | None = None
    servo_ids: list[int] = field(default_factory=list)
    tendon_to_servo: list[int] = field(default_factory=list)
    active_segment_key: str | None = None
    active_segment_label: str | None = None
    active_segment_servo_ids: list[int] = field(default_factory=list)
    active_segment_pairs: dict[str, list[int]] = field(default_factory=dict)
    segments: dict[str, dict[str, Any]] = field(default_factory=dict)
    segment_order: list[str] = field(default_factory=list)
    selected_servo_id: int | None = None
    expected_servo_ids: list[int] = field(default_factory=list)
    commanded_servo_ids: list[int] = field(default_factory=list)
    mirror_pairs: dict[int, int] = field(default_factory=dict)
    mode_profile: str | None = None
    mode_capabilities: dict[str, bool] = field(default_factory=dict)
    mode_notes: list[str] = field(default_factory=list)
    ticks_per_revolution: int | None = None
    spool_diameter_cm: float | None = None
    position_min_offset_ticks: int | None = None
    position_max_offset_ticks: int | None = None
    default_pretension_current_threshold_ma: int | None = None
    tightening_rotation_by_servo: dict[int, str] = field(default_factory=dict)


@dataclass
class ServoCalibrationEntry:
    """Calibration state for one servo/tendon path."""

    servo_id: int
    tendon_index: int | None = None
    capture_source: str | None = None
    neutral_setpoint: int | None = None
    safe_min_tick: int | None = None
    safe_max_tick: int | None = None
    pretension_current_threshold_ma: int | None = None
    tightening_rotation: str | None = None
    hardware_min_tick: int | None = None
    hardware_max_tick: int | None = None
    hardware_current_limit_ma: int | None = None
    last_measured_current_ma: int | None = None
    pretension_final_position_tick: int | None = None
    pretension_result_status: str | None = None
    pretension_source: str | None = None
    pretension_note: str | None = None
    pretension_completed_at_utc: str | None = None
    latest_pretension_run: dict[str, Any] | None = None
    calibrated_at_utc: str | None = None
    status: str = "missing"
    valid: bool = False


@dataclass
class PretensionSourceSummary:
    """Accepted startup-state source summary for one configured servo set."""

    source_type: str
    accepted: bool
    usable: bool
    message: str
    updated_at_utc: str | None = None
    note: str | None = None
    source_by_servo: dict[int, str] = field(default_factory=dict)
    positions_by_servo: dict[int, int | None] = field(default_factory=dict)
    currents_by_servo: dict[int, int | None] = field(default_factory=dict)


@dataclass
class ServoCalibrationArtifact:
    """Single persisted calibration artifact for the servo subsystem."""

    schema_version: int = SCHEMA_VERSION
    updated_at_utc: str | None = None
    robot: dict[str, Any] = field(default_factory=dict)
    servos: dict[int, ServoCalibrationEntry] = field(default_factory=dict)
    source_format: str = "servo_calibration_v2"


@dataclass
class ServoCalibrationSummary:
    """UI-facing summary of the current calibration artifact."""

    exists: bool
    compatible: bool
    path: str
    updated_at_utc: str | None
    status: str
    message: str
    schema_version: int
    migrated_legacy_format: bool = False
    servo_entries: dict[int, ServoCalibrationEntry] = field(default_factory=dict)
    robot_metadata: dict[str, Any] = field(default_factory=dict)
    source_format: str = "servo_calibration_v2"

    @property
    def calibrated_servo_ids(self) -> list[int]:
        return sorted(
            servo_id
            for servo_id, entry in self.servo_entries.items()
            if entry.valid and entry.neutral_setpoint is not None
        )

    @property
    def missing_bounds_servo_ids(self) -> list[int]:
        return sorted(
            servo_id
            for servo_id, entry in self.servo_entries.items()
            if entry.safe_min_tick is None or entry.safe_max_tick is None
        )

    @property
    def missing_threshold_servo_ids(self) -> list[int]:
        return sorted(
            servo_id
            for servo_id, entry in self.servo_entries.items()
            if entry.pretension_current_threshold_ma is None
        )

    @property
    def calibration_trust(self) -> str:
        raw = str(self.robot_metadata.get("calibration_trust", "") or "").strip().lower()
        if raw:
            return raw
        if bool(self.robot_metadata.get("mock_mode", False)):
            return "mock"
        return "unknown"

    @property
    def is_mock_calibration(self) -> bool:
        trust = self.calibration_trust
        if trust in {"mock", "debug", "synthetic"}:
            return True
        if bool(self.robot_metadata.get("mock_mode", False)):
            return True
        hardware_valid = self.robot_metadata.get("valid_for_hardware_startup")
        return hardware_valid is False

    def pretension_source_summary(self, servo_ids: list[int]) -> PretensionSourceSummary:
        requested_ids = [int(servo_id) for servo_id in servo_ids]
        if not self.exists or not self.compatible:
            return PretensionSourceSummary(
                source_type="none",
                accepted=False,
                usable=False,
                message=f"Servo calibration artifact is not ready: {self.message}",
                updated_at_utc=self.updated_at_utc,
            )
        missing = [
            int(servo_id)
            for servo_id in requested_ids
            if (
                (entry := self.servo_entries.get(int(servo_id))) is None
                or entry.pretension_result_status != "accepted"
            )
        ]
        if missing:
            return PretensionSourceSummary(
                source_type="none",
                accepted=False,
                usable=False,
                message=(
                    "Accepted pretension/startup state is missing for servo(s): "
                    + ", ".join(str(value) for value in missing)
                ),
                updated_at_utc=self.updated_at_utc,
            )

        source_by_servo: dict[int, str] = {}
        positions_by_servo: dict[int, int | None] = {}
        currents_by_servo: dict[int, int | None] = {}
        notes: list[str] = []
        for servo_id in requested_ids:
            entry = self.servo_entries[int(servo_id)]
            source = _pretension_source_for_entry(entry)
            source_by_servo[int(servo_id)] = source
            positions_by_servo[int(servo_id)] = entry.pretension_final_position_tick
            currents_by_servo[int(servo_id)] = entry.last_measured_current_ma
            if entry.pretension_note not in (None, ""):
                notes.append(str(entry.pretension_note))
        distinct_sources = sorted({value for value in source_by_servo.values() if value})
        if len(distinct_sources) != 1 or distinct_sources[0] not in {"manual", "algorithmic"}:
            return PretensionSourceSummary(
                source_type="mixed",
                accepted=True,
                usable=False,
                message=(
                    "Accepted pretension exists, but the source is mixed or ambiguous across the configured servos: "
                    + ", ".join(f"{servo_id}={source_by_servo[servo_id]}" for servo_id in requested_ids)
                ),
                updated_at_utc=self.updated_at_utc,
                note=notes[0] if len(set(notes)) == 1 else None,
                source_by_servo=source_by_servo,
                positions_by_servo=positions_by_servo,
                currents_by_servo=currents_by_servo,
            )
        source = distinct_sources[0]
        return PretensionSourceSummary(
            source_type=source,
            accepted=True,
            usable=True,
            message=(
                f"Accepted {source} pretension/startup state is recorded for servo(s): "
                + ", ".join(str(value) for value in requested_ids)
            ),
            updated_at_utc=self.updated_at_utc,
            note=notes[0] if len(set(notes)) == 1 else None,
            source_by_servo=source_by_servo,
            positions_by_servo=positions_by_servo,
            currents_by_servo=currents_by_servo,
        )


class NeutralCalibrationService:
    """Stores and retrieves the canonical servo calibration artifact."""

    def __init__(
        self,
        path: Path | None = None,
        *,
        archive_root: Path | None = None,
        context: ServoCalibrationContext | None = None,
    ) -> None:
        project_root = Path(__file__).resolve().parents[2]
        self.path = path or (project_root / "config" / "neutral_setpoints.json")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.archive_root = archive_root or self._default_archive_root(self.path)
        self.archive_root.mkdir(parents=True, exist_ok=True)
        self.context = context or ServoCalibrationContext()

    def save_neutral_setpoints(
        self,
        setpoints_by_id: dict[int, int],
        *,
        safe_bounds_by_id: dict[int, tuple[int, int]] | None = None,
        capture_source: str = "live_present_position",
        calibration_metadata: dict[str, Any] | None = None,
    ) -> None:
        artifact = self.load_calibration_artifact()
        tendon_index_by_servo = {
            int(servo_id): index
            for index, servo_id in enumerate(self.context.tendon_to_servo)
        }
        timestamp = _utc_now()
        explicit_bounds = {
            int(servo_id): (int(bounds[0]), int(bounds[1]))
            for servo_id, bounds in dict(safe_bounds_by_id or {}).items()
        }
        for servo_id, neutral_tick in sorted(setpoints_by_id.items()):
            existing = artifact.servos.get(int(servo_id), ServoCalibrationEntry(servo_id=int(servo_id)))
            safe_bounds = explicit_bounds.get(int(servo_id))
            if safe_bounds is not None:
                safe_min, safe_max = safe_bounds
            else:
                safe_min = (
                    int(neutral_tick) + int(self.context.position_min_offset_ticks)
                    if self.context.position_min_offset_ticks is not None
                    else existing.safe_min_tick
                )
                safe_max = (
                    int(neutral_tick) + int(self.context.position_max_offset_ticks)
                    if self.context.position_max_offset_ticks is not None
                    else existing.safe_max_tick
                )
            threshold = existing.pretension_current_threshold_ma
            if threshold is None:
                threshold = self.context.default_pretension_current_threshold_ma
            artifact.servos[int(servo_id)] = ServoCalibrationEntry(
                servo_id=int(servo_id),
                tendon_index=tendon_index_by_servo.get(int(servo_id), existing.tendon_index),
                capture_source=str(capture_source).strip() if capture_source else existing.capture_source,
                neutral_setpoint=int(neutral_tick),
                safe_min_tick=int(safe_min) if safe_min is not None else None,
                safe_max_tick=int(safe_max) if safe_max is not None else None,
                pretension_current_threshold_ma=int(threshold) if threshold is not None else None,
                tightening_rotation=(
                    self.context.tightening_rotation_by_servo.get(int(servo_id))
                    or existing.tightening_rotation
                ),
                hardware_min_tick=existing.hardware_min_tick,
                hardware_max_tick=existing.hardware_max_tick,
                hardware_current_limit_ma=existing.hardware_current_limit_ma,
                last_measured_current_ma=existing.last_measured_current_ma,
                pretension_final_position_tick=existing.pretension_final_position_tick,
                pretension_result_status=existing.pretension_result_status,
                pretension_source=existing.pretension_source,
                pretension_note=existing.pretension_note,
                pretension_completed_at_utc=existing.pretension_completed_at_utc,
                latest_pretension_run=dict(existing.latest_pretension_run or {}) or None,
                calibrated_at_utc=timestamp,
                status="neutral_captured",
                valid=True,
            )
        artifact.updated_at_utc = timestamp
        artifact.robot = self._robot_metadata()
        artifact.robot.update(dict(calibration_metadata or {}))
        artifact.source_format = "servo_calibration_v2"
        self.save_calibration_artifact(artifact)

    def save_servo_calibration(
        self,
        *,
        servo_id: int,
        neutral_setpoint: int,
        safe_min_tick: int,
        safe_max_tick: int,
        pretension_current_threshold_ma: int | None,
        tightening_rotation: str | None = None,
        hardware_min_tick: int | None = None,
        hardware_max_tick: int | None = None,
        hardware_current_limit_ma: int | None = None,
        last_measured_current_ma: int | None = None,
        status: str = "startup_calibrated",
        valid: bool = True,
    ) -> ServoCalibrationEntry:
        artifact = self.load_calibration_artifact()
        tendon_index_by_servo = {
            int(mapped_servo_id): index
            for index, mapped_servo_id in enumerate(self.context.tendon_to_servo)
        }
        if int(safe_min_tick) > int(neutral_setpoint) or int(safe_max_tick) < int(neutral_setpoint):
            raise ValueError("Safe bounds must contain the neutral setpoint.")
        timestamp = _utc_now()
        existing = artifact.servos.get(int(servo_id), ServoCalibrationEntry(servo_id=int(servo_id)))
        entry = ServoCalibrationEntry(
            servo_id=int(servo_id),
            tendon_index=tendon_index_by_servo.get(int(servo_id), existing.tendon_index),
            capture_source=existing.capture_source,
            neutral_setpoint=int(neutral_setpoint),
            safe_min_tick=int(safe_min_tick),
            safe_max_tick=int(safe_max_tick),
            pretension_current_threshold_ma=(
                int(pretension_current_threshold_ma)
                if pretension_current_threshold_ma is not None
                else existing.pretension_current_threshold_ma
            ),
            tightening_rotation=(
                (str(tightening_rotation).strip().lower() if tightening_rotation else None)
                or existing.tightening_rotation
                or self.context.tightening_rotation_by_servo.get(int(servo_id))
            ),
            hardware_min_tick=(
                int(hardware_min_tick) if hardware_min_tick is not None else existing.hardware_min_tick
            ),
            hardware_max_tick=(
                int(hardware_max_tick) if hardware_max_tick is not None else existing.hardware_max_tick
            ),
            hardware_current_limit_ma=(
                int(hardware_current_limit_ma)
                if hardware_current_limit_ma is not None
                else existing.hardware_current_limit_ma
            ),
            last_measured_current_ma=(
                int(last_measured_current_ma)
                if last_measured_current_ma is not None
                else existing.last_measured_current_ma
            ),
            pretension_final_position_tick=existing.pretension_final_position_tick,
            pretension_result_status=existing.pretension_result_status,
            pretension_source=existing.pretension_source,
            pretension_note=existing.pretension_note,
            pretension_completed_at_utc=existing.pretension_completed_at_utc,
            latest_pretension_run=dict(existing.latest_pretension_run or {}) or None,
            calibrated_at_utc=timestamp,
            status=str(status),
            valid=bool(valid),
        )
        artifact.servos[int(servo_id)] = entry
        artifact.updated_at_utc = timestamp
        artifact.robot = self._robot_metadata()
        self.save_calibration_artifact(artifact)
        return entry

    def save_pretension_result(
        self,
        *,
        servo_id: int,
        final_position_tick: int | None,
        final_current_ma: int | None,
        threshold_ma: int | None,
        result_status: str,
        run_record: dict[str, Any] | None = None,
        pretension_source: str | None = None,
        pretension_note: str | None = None,
    ) -> ServoCalibrationEntry:
        artifact = self.load_calibration_artifact()
        existing = artifact.servos.get(int(servo_id), ServoCalibrationEntry(servo_id=int(servo_id)))
        timestamp = _utc_now()
        latest_pretension_run = self._sanitize_run_record(run_record) or dict(existing.latest_pretension_run or {}) or None
        entry = ServoCalibrationEntry(
            servo_id=int(servo_id),
            tendon_index=existing.tendon_index,
            capture_source=existing.capture_source,
            neutral_setpoint=existing.neutral_setpoint,
            safe_min_tick=existing.safe_min_tick,
            safe_max_tick=existing.safe_max_tick,
            pretension_current_threshold_ma=(
                int(threshold_ma) if threshold_ma is not None else existing.pretension_current_threshold_ma
            ),
            tightening_rotation=existing.tightening_rotation,
            hardware_min_tick=existing.hardware_min_tick,
            hardware_max_tick=existing.hardware_max_tick,
            hardware_current_limit_ma=existing.hardware_current_limit_ma,
            last_measured_current_ma=(
                int(final_current_ma) if final_current_ma is not None else existing.last_measured_current_ma
            ),
            pretension_final_position_tick=(
                int(final_position_tick) if final_position_tick is not None else existing.pretension_final_position_tick
            ),
            pretension_result_status=str(result_status),
            pretension_source=(
                str(pretension_source).strip().lower()
                if pretension_source not in (None, "")
                else existing.pretension_source
            ),
            pretension_note=(
                str(pretension_note).strip()
                if pretension_note not in (None, "")
                else existing.pretension_note
            ),
            pretension_completed_at_utc=timestamp,
            latest_pretension_run=latest_pretension_run,
            calibrated_at_utc=existing.calibrated_at_utc,
            status=existing.status if existing.status != "missing" else str(result_status),
            valid=existing.valid,
        )
        artifact.servos[int(servo_id)] = entry
        artifact.updated_at_utc = timestamp
        artifact.robot = self._robot_metadata()
        self.save_calibration_artifact(artifact)
        return entry

    def save_manual_pretension_state(
        self,
        *,
        states_by_servo: dict[int, dict[str, Any]],
        note: str | None = None,
        accepted: bool = False,
    ) -> dict[int, ServoCalibrationEntry]:
        if not states_by_servo:
            raise ValueError("At least one servo state is required for manual pretension capture.")
        artifact = self.load_calibration_artifact()
        timestamp = _utc_now()
        saved_entries: dict[int, ServoCalibrationEntry] = {}
        clean_note = str(note).strip() if note not in (None, "") else None
        for servo_id, raw_state in sorted(states_by_servo.items()):
            state = dict(raw_state or {})
            existing = artifact.servos.get(int(servo_id), ServoCalibrationEntry(servo_id=int(servo_id)))
            final_position_tick = (
                int(state["measured_position_tick"])
                if state.get("measured_position_tick") is not None
                else existing.pretension_final_position_tick
            )
            final_current_ma = (
                int(state["measured_current_ma"])
                if state.get("measured_current_ma") is not None
                else existing.last_measured_current_ma
            )
            run_record = self._sanitize_run_record(
                {
                    **state,
                    "source": "manual",
                    "status": "accepted" if accepted else "manual_captured",
                    "captured_at_utc": timestamp,
                    "operator_note": clean_note,
                    "active_segment_key": self.context.active_segment_key,
                    "active_segment_label": self.context.active_segment_label,
                    "active_segment_servo_ids": list(self.context.active_segment_servo_ids),
                    "active_segment_pairs": {
                        str(key): [int(value) for value in values]
                        for key, values in dict(self.context.active_segment_pairs or {}).items()
                    },
                    "segments": self._segment_metadata(),
                    "segment_order": list(self.context.segment_order),
                    "operating_mode": self.context.robot_mode,
                    "mode_profile": self.context.mode_profile,
                    "mode_capabilities": dict(self.context.mode_capabilities),
                    "mode_notes": list(self.context.mode_notes),
                    "startup_artifact_scope": self._startup_artifact_scope(),
                    "selected_servo_id": self.context.selected_servo_id,
                    "expected_servo_ids": list(self.context.expected_servo_ids),
                    "commanded_servo_ids": list(self.context.commanded_servo_ids),
                    "mirror_pairs": {str(key): int(value) for key, value in dict(self.context.mirror_pairs or {}).items()},
                }
            )
            entry = ServoCalibrationEntry(
                servo_id=int(servo_id),
                tendon_index=existing.tendon_index,
                capture_source=existing.capture_source,
                neutral_setpoint=existing.neutral_setpoint,
                safe_min_tick=existing.safe_min_tick,
                safe_max_tick=existing.safe_max_tick,
                pretension_current_threshold_ma=existing.pretension_current_threshold_ma,
                tightening_rotation=existing.tightening_rotation,
                hardware_min_tick=existing.hardware_min_tick,
                hardware_max_tick=existing.hardware_max_tick,
                hardware_current_limit_ma=existing.hardware_current_limit_ma,
                last_measured_current_ma=final_current_ma,
                pretension_final_position_tick=final_position_tick,
                pretension_result_status="accepted" if accepted else "manual_captured",
                pretension_source="manual",
                pretension_note=clean_note,
                pretension_completed_at_utc=timestamp,
                latest_pretension_run=run_record,
                calibrated_at_utc=existing.calibrated_at_utc,
                status="pretension_accepted" if accepted else "manual_pretension_captured",
                valid=existing.valid,
            )
            artifact.servos[int(servo_id)] = entry
            saved_entries[int(servo_id)] = entry
        artifact.updated_at_utc = timestamp
        artifact.robot = self._robot_metadata()
        self.save_calibration_artifact(artifact)
        return saved_entries

    def clear_manual_pretension_state(self, servo_ids: list[int]) -> list[int]:
        artifact = self.load_calibration_artifact()
        cleared_ids: list[int] = []
        timestamp = _utc_now()
        for servo_id in [int(value) for value in servo_ids]:
            existing = artifact.servos.get(int(servo_id))
            if existing is None:
                continue
            if _pretension_source_for_entry(existing) != "manual":
                continue
            artifact.servos[int(servo_id)] = ServoCalibrationEntry(
                servo_id=existing.servo_id,
                tendon_index=existing.tendon_index,
                capture_source=existing.capture_source,
                neutral_setpoint=existing.neutral_setpoint,
                safe_min_tick=existing.safe_min_tick,
                safe_max_tick=existing.safe_max_tick,
                pretension_current_threshold_ma=existing.pretension_current_threshold_ma,
                tightening_rotation=existing.tightening_rotation,
                hardware_min_tick=existing.hardware_min_tick,
                hardware_max_tick=existing.hardware_max_tick,
                hardware_current_limit_ma=existing.hardware_current_limit_ma,
                last_measured_current_ma=None,
                pretension_final_position_tick=None,
                pretension_result_status=None,
                pretension_source=None,
                pretension_note=None,
                pretension_completed_at_utc=None,
                latest_pretension_run=None,
                calibrated_at_utc=existing.calibrated_at_utc,
                status=(
                    "neutral_captured"
                    if existing.neutral_setpoint is not None
                    else "missing"
                ),
                valid=existing.valid,
            )
            cleared_ids.append(int(servo_id))
        if cleared_ids:
            artifact.updated_at_utc = timestamp
            artifact.robot = self._robot_metadata()
            self.save_calibration_artifact(artifact)
        return cleared_ids

    def load_neutral_setpoints(self) -> dict[int, int]:
        artifact = self.load_calibration_artifact()
        return {
            int(servo_id): int(entry.neutral_setpoint)
            for servo_id, entry in artifact.servos.items()
            if entry.neutral_setpoint is not None
        }

    def load_calibration_artifact(self) -> ServoCalibrationArtifact:
        payload = self._read_payload()
        if payload is None:
            return self._empty_artifact()
        if self._is_legacy_neutral_payload(payload):
            return self._migrate_legacy_payload(payload)
        return self._coerce_artifact(payload)

    def save_calibration_artifact(self, artifact: ServoCalibrationArtifact) -> None:
        self._archive_latest_if_present()
        payload = {
            "schema_version": int(artifact.schema_version),
            "updated_at_utc": artifact.updated_at_utc,
            "robot": dict(artifact.robot),
            "source_format": artifact.source_format,
            "servos": {
                str(servo_id): asdict(entry)
                for servo_id, entry in sorted(artifact.servos.items())
            },
        }
        self.path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")

    def get_calibration_summary(self) -> ServoCalibrationSummary:
        payload = self._read_payload()
        if payload is None:
            return ServoCalibrationSummary(
                exists=False,
                compatible=False,
                path=str(self.path),
                updated_at_utc=None,
                status="missing",
                message="No servo calibration file has been saved yet.",
                schema_version=SCHEMA_VERSION,
            )
        migrated_legacy = self._is_legacy_neutral_payload(payload)
        artifact = self._migrate_legacy_payload(payload) if migrated_legacy else self._coerce_artifact(payload)
        compatible, compatibility_message = self._compatibility_check(artifact)
        calibrated_count = len(
            [entry for entry in artifact.servos.values() if entry.valid and entry.neutral_setpoint is not None]
        )
        if compatible:
            status = "ready" if calibrated_count else "empty"
        else:
            status = "incompatible"
        message = (
            compatibility_message
            if compatibility_message
            else f"Calibration file contains {calibrated_count} servo entries."
        )
        return ServoCalibrationSummary(
            exists=True,
            compatible=compatible,
            path=str(self.path),
            updated_at_utc=artifact.updated_at_utc,
            status=status,
            message=message,
            schema_version=int(artifact.schema_version),
            migrated_legacy_format=migrated_legacy,
            servo_entries=dict(sorted(artifact.servos.items())),
            robot_metadata=dict(artifact.robot or {}),
            source_format=str(artifact.source_format),
        )

    def bounds_by_servo_id(self, servo_ids: list[int]) -> dict[int, tuple[int, int]]:
        summary = self.get_calibration_summary()
        if not summary.exists or not summary.compatible:
            return {}
        bounds: dict[int, tuple[int, int]] = {}
        for servo_id in servo_ids:
            entry = summary.servo_entries.get(int(servo_id))
            if entry is None or entry.safe_min_tick is None or entry.safe_max_tick is None:
                return {}
            bounds[int(servo_id)] = (int(entry.safe_min_tick), int(entry.safe_max_tick))
        return bounds

    def thresholds_by_servo_id(self, servo_ids: list[int]) -> dict[int, int]:
        summary = self.get_calibration_summary()
        if not summary.exists or not summary.compatible:
            return {}
        thresholds: dict[int, int] = {}
        for servo_id in servo_ids:
            entry = summary.servo_entries.get(int(servo_id))
            if entry is None or entry.pretension_current_threshold_ma is None:
                return {}
            thresholds[int(servo_id)] = int(entry.pretension_current_threshold_ma)
        return thresholds

    def entry_by_servo_id(self, servo_id: int) -> ServoCalibrationEntry | None:
        summary = self.get_calibration_summary()
        if not summary.exists or not summary.compatible:
            return None
        return summary.servo_entries.get(int(servo_id))

    def mark_pretension_accepted(self, servo_id: int) -> ServoCalibrationEntry:
        artifact = self.load_calibration_artifact()
        existing = artifact.servos.get(int(servo_id))
        if existing is None:
            raise ValueError(f"Servo {servo_id} has no saved calibration entry.")
        if existing.pretension_final_position_tick is None:
            raise ValueError(f"Servo {servo_id} has no pretension result to accept.")
        timestamp = _utc_now()
        entry = ServoCalibrationEntry(
            servo_id=existing.servo_id,
            tendon_index=existing.tendon_index,
            capture_source=existing.capture_source,
            neutral_setpoint=existing.neutral_setpoint,
            safe_min_tick=existing.safe_min_tick,
            safe_max_tick=existing.safe_max_tick,
            pretension_current_threshold_ma=existing.pretension_current_threshold_ma,
            tightening_rotation=existing.tightening_rotation,
            hardware_min_tick=existing.hardware_min_tick,
            hardware_max_tick=existing.hardware_max_tick,
            hardware_current_limit_ma=existing.hardware_current_limit_ma,
            last_measured_current_ma=existing.last_measured_current_ma,
            pretension_final_position_tick=existing.pretension_final_position_tick,
            pretension_result_status="accepted",
            pretension_source=existing.pretension_source,
            pretension_note=existing.pretension_note,
            pretension_completed_at_utc=timestamp,
            latest_pretension_run=self._mark_run_record_accepted(existing.latest_pretension_run),
            calibrated_at_utc=existing.calibrated_at_utc,
            status="pretension_accepted",
            valid=existing.valid,
        )
        artifact.servos[int(servo_id)] = entry
        artifact.updated_at_utc = timestamp
        artifact.robot = self._robot_metadata()
        self.save_calibration_artifact(artifact)
        return entry

    def _read_payload(self) -> dict[str, Any] | None:
        if not self.path.exists():
            return None
        payload = json.loads(self.path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError(f"Expected calibration payload mapping in {self.path}")
        return payload

    def _coerce_artifact(self, payload: dict[str, Any]) -> ServoCalibrationArtifact:
        servos: dict[int, ServoCalibrationEntry] = {}
        for raw_key, raw_value in (payload.get("servos", {}) or {}).items():
            servo_id = int(raw_key)
            data = dict(raw_value or {})
            servos[servo_id] = ServoCalibrationEntry(
                servo_id=int(data.get("servo_id", servo_id)),
                tendon_index=int(data["tendon_index"]) if data.get("tendon_index") is not None else None,
                capture_source=(
                    str(data.get("capture_source"))
                    if data.get("capture_source") not in (None, "")
                    else None
                ),
                neutral_setpoint=(
                    int(data["neutral_setpoint"]) if data.get("neutral_setpoint") is not None else None
                ),
                safe_min_tick=int(data["safe_min_tick"]) if data.get("safe_min_tick") is not None else None,
                safe_max_tick=int(data["safe_max_tick"]) if data.get("safe_max_tick") is not None else None,
                pretension_current_threshold_ma=(
                    int(data["pretension_current_threshold_ma"])
                    if data.get("pretension_current_threshold_ma") is not None
                    else None
                ),
                tightening_rotation=(
                    str(data.get("tightening_rotation")).strip().lower()
                    if data.get("tightening_rotation") not in (None, "")
                    else None
                ),
                hardware_min_tick=int(data["hardware_min_tick"]) if data.get("hardware_min_tick") is not None else None,
                hardware_max_tick=int(data["hardware_max_tick"]) if data.get("hardware_max_tick") is not None else None,
                hardware_current_limit_ma=(
                    int(data["hardware_current_limit_ma"])
                    if data.get("hardware_current_limit_ma") is not None
                    else None
                ),
                last_measured_current_ma=(
                    int(data["last_measured_current_ma"])
                    if data.get("last_measured_current_ma") is not None
                    else None
                ),
                pretension_final_position_tick=(
                    int(data["pretension_final_position_tick"])
                    if data.get("pretension_final_position_tick") is not None
                    else None
                ),
                pretension_result_status=(
                    str(data.get("pretension_result_status"))
                    if data.get("pretension_result_status") not in (None, "")
                    else None
                ),
                pretension_source=(
                    str(data.get("pretension_source")).strip().lower()
                    if data.get("pretension_source") not in (None, "")
                    else None
                ),
                pretension_note=(
                    str(data.get("pretension_note")).strip()
                    if data.get("pretension_note") not in (None, "")
                    else None
                ),
                pretension_completed_at_utc=(
                    str(data.get("pretension_completed_at_utc"))
                    if data.get("pretension_completed_at_utc")
                    else None
                ),
                latest_pretension_run=self._sanitize_run_record(data.get("latest_pretension_run")),
                calibrated_at_utc=str(data.get("calibrated_at_utc")) if data.get("calibrated_at_utc") else None,
                status=str(data.get("status", "missing")),
                valid=bool(data.get("valid", False)),
            )
        return ServoCalibrationArtifact(
            schema_version=int(payload.get("schema_version", SCHEMA_VERSION)),
            updated_at_utc=str(payload.get("updated_at_utc")) if payload.get("updated_at_utc") else None,
            robot=dict(payload.get("robot", {}) or {}),
            servos=servos,
            source_format=str(payload.get("source_format", "servo_calibration_v2")),
        )

    def _migrate_legacy_payload(self, payload: dict[str, Any]) -> ServoCalibrationArtifact:
        artifact = self._empty_artifact()
        artifact.source_format = "legacy_neutral_map"
        artifact.updated_at_utc = None
        artifact.robot = self._robot_metadata()
        timestamp = _utc_now()
        tendon_index_by_servo = {
            int(servo_id): index
            for index, servo_id in enumerate(self.context.tendon_to_servo)
        }
        for raw_key, raw_value in sorted(payload.items(), key=lambda item: int(item[0])):
            servo_id = int(raw_key)
            neutral = int(raw_value)
            artifact.servos[servo_id] = ServoCalibrationEntry(
                servo_id=servo_id,
                tendon_index=tendon_index_by_servo.get(servo_id),
                capture_source="legacy_neutral_map",
                neutral_setpoint=neutral,
                safe_min_tick=(
                    neutral + int(self.context.position_min_offset_ticks)
                    if self.context.position_min_offset_ticks is not None
                    else None
                ),
                safe_max_tick=(
                    neutral + int(self.context.position_max_offset_ticks)
                    if self.context.position_max_offset_ticks is not None
                    else None
                ),
                pretension_current_threshold_ma=(
                    int(self.context.default_pretension_current_threshold_ma)
                    if self.context.default_pretension_current_threshold_ma is not None
                    else None
                ),
                tightening_rotation=self.context.tightening_rotation_by_servo.get(servo_id),
                latest_pretension_run=None,
                calibrated_at_utc=timestamp,
                status="migrated_legacy_neutral",
                valid=True,
            )
        return artifact

    def _empty_artifact(self) -> ServoCalibrationArtifact:
        return ServoCalibrationArtifact(
            schema_version=SCHEMA_VERSION,
            updated_at_utc=None,
            robot=self._robot_metadata(),
            servos={},
        )

    @staticmethod
    def _sanitize_run_record(run_record: Any) -> dict[str, Any] | None:
        if not isinstance(run_record, dict):
            return None
        clean = NeutralCalibrationService._sanitize_value(run_record)
        return clean if isinstance(clean, dict) and clean else None

    @staticmethod
    def _sanitize_value(value: Any) -> Any:
        if value is None or isinstance(value, (str, int, float, bool)):
            return value
        if isinstance(value, dict):
            clean: dict[str, Any] = {}
            for child_key, child_value in value.items():
                sanitized = NeutralCalibrationService._sanitize_value(child_value)
                if sanitized is not None or child_value is None:
                    clean[str(child_key)] = sanitized
            return clean
        if isinstance(value, (list, tuple)):
            return [
                sanitized
                for item in value
                if (sanitized := NeutralCalibrationService._sanitize_value(item)) is not None or item is None
            ]
        return None

    def _mark_run_record_accepted(self, run_record: dict[str, Any] | None) -> dict[str, Any] | None:
        clean = self._sanitize_run_record(run_record)
        if clean is None:
            return None
        clean["accepted"] = True
        clean["accepted_at_utc"] = _utc_now()
        return clean

    def _robot_metadata(self) -> dict[str, Any]:
        return {
            "robot_mode": self.context.robot_mode,
            "robot_config_name": self.context.robot_config_name,
            "servo_ids": list(self.context.servo_ids),
            "tendon_to_servo": list(self.context.tendon_to_servo),
            "active_segment_key": self.context.active_segment_key,
            "active_segment_label": self.context.active_segment_label,
            "active_segment_servo_ids": list(self.context.active_segment_servo_ids),
            "active_segment_pairs": {
                str(key): [int(value) for value in values]
                for key, values in dict(self.context.active_segment_pairs or {}).items()
            },
            "segments": self._segment_metadata(),
            "segment_order": list(self.context.segment_order),
            "selected_servo_id": self.context.selected_servo_id,
            "expected_servo_ids": list(self.context.expected_servo_ids),
            "commanded_servo_ids": list(self.context.commanded_servo_ids),
            "mirror_pairs": {str(key): int(value) for key, value in dict(self.context.mirror_pairs or {}).items()},
            "mode_profile": self.context.mode_profile,
            "mode_capabilities": dict(self.context.mode_capabilities),
            "mode_notes": list(self.context.mode_notes),
            "startup_artifact_scope": self._startup_artifact_scope(),
            "dual_segment_foundation_only": self.context.robot_mode == "dual_segment",
            "automatic_two_segment_pretension_validated": False,
            "two_segment_kinematics_control_validated": False,
            "ticks_per_revolution": self.context.ticks_per_revolution,
            "spool_diameter_cm": self.context.spool_diameter_cm,
            "position_min_offset_ticks": self.context.position_min_offset_ticks,
            "position_max_offset_ticks": self.context.position_max_offset_ticks,
            "default_pretension_current_threshold_ma": self.context.default_pretension_current_threshold_ma,
            "tightening_rotation_by_servo": {
                str(servo_id): str(rotation)
                for servo_id, rotation in sorted(self.context.tightening_rotation_by_servo.items())
            },
        }

    def _segment_metadata(self) -> dict[str, dict[str, Any]]:
        clean: dict[str, dict[str, Any]] = {}
        for key, value in dict(self.context.segments or {}).items():
            data = dict(value or {})
            clean[str(key)] = {
                "key": str(data.get("key", key)),
                "label": str(data.get("label", key)),
                "segment_label": str(data.get("segment_label", data.get("label", key))),
                "segment_role": str(data.get("segment_role", "")),
                "segment_order_index": (
                    int(data["segment_order_index"])
                    if data.get("segment_order_index") is not None
                    else None
                ),
                "servo_ids": [int(item) for item in list(data.get("servo_ids", []) or [])],
                "pairs": {
                    str(pair_key): [int(item) for item in list(pair_values or [])]
                    for pair_key, pair_values in dict(data.get("pairs", {}) or {}).items()
                },
            }
        return clean

    def _startup_artifact_scope(self) -> str:
        if self.context.robot_mode == "dual_segment":
            return "dual_segment_all_8_manual_startup_foundation"
        if self.context.robot_mode == "parallel_single":
            return "parallel_single_mirrored_manual_startup"
        if self.context.robot_mode == "single_segment":
            return "single_segment_manual_startup"
        return str(self.context.robot_mode or "unknown")

    def _compatibility_check(self, artifact: ServoCalibrationArtifact) -> tuple[bool, str]:
        if not artifact.servos:
            return False, "Calibration file exists but does not contain any servo entries."
        robot = artifact.robot or {}
        if self.context.robot_mode and robot.get("robot_mode") not in (None, "", self.context.robot_mode):
            return False, (
                f"Calibration robot mode {robot.get('robot_mode')} does not match current mode {self.context.robot_mode}."
            )
        required_servo_ids = list(
            self.context.expected_servo_ids
            or self.context.active_segment_servo_ids
            or self.context.servo_ids
        )
        if required_servo_ids:
            saved_ids = [int(value) for value in robot.get("servo_ids", []) if value is not None]
            active_saved_ids = [int(value) for value in robot.get("active_segment_servo_ids", []) if value is not None]
            allowed_saved_sets = [list(self.context.servo_ids), list(required_servo_ids)]
            if active_saved_ids:
                allowed_saved_sets.append(active_saved_ids)
            if saved_ids and saved_ids not in allowed_saved_sets:
                return False, (
                    f"Calibration servo IDs {saved_ids} do not match current active segment IDs {required_servo_ids} "
                    f"or configured IDs {self.context.servo_ids}."
                )
            if self.context.robot_mode == "dual_segment":
                saved_expected_ids = [
                    int(value) for value in robot.get("expected_servo_ids", []) if value is not None
                ]
                saved_commanded_ids = [
                    int(value) for value in robot.get("commanded_servo_ids", []) if value is not None
                ]
                if saved_expected_ids and saved_expected_ids != required_servo_ids:
                    return False, (
                        f"Dual-segment startup artifact expected IDs {saved_expected_ids} do not match "
                        f"current all-8 expected IDs {required_servo_ids}."
                    )
                if saved_commanded_ids and saved_commanded_ids != required_servo_ids:
                    return False, (
                        f"Dual-segment startup artifact commanded IDs {saved_commanded_ids} do not match "
                        f"current all-8 commanded IDs {required_servo_ids}."
                    )
                saved_segment_order = [str(value) for value in robot.get("segment_order", []) if value is not None]
                current_segment_order = [str(value) for value in self.context.segment_order]
                if saved_segment_order and current_segment_order and saved_segment_order != current_segment_order:
                    return False, (
                        f"Dual-segment startup artifact segment order {saved_segment_order} does not match "
                        f"current segment order {current_segment_order}."
                    )
            missing = [servo_id for servo_id in required_servo_ids if servo_id not in artifact.servos]
            if missing:
                return False, f"Calibration is missing servo entries for {missing}."
        if self.context.tendon_to_servo:
            saved_mapping = [int(value) for value in robot.get("tendon_to_servo", []) if value is not None]
            allowed_mappings = [list(self.context.tendon_to_servo)]
            if self.context.active_segment_servo_ids:
                allowed_mappings.append(list(self.context.active_segment_servo_ids))
            if saved_mapping and saved_mapping not in allowed_mappings:
                return False, (
                    f"Calibration tendon mapping {saved_mapping} does not match current mapping {self.context.tendon_to_servo} "
                    f"or active segment mapping {self.context.active_segment_servo_ids}."
                )
        return True, ""

    def _archive_latest_if_present(self) -> None:
        if self.path.exists():
            archive_path = canonical_timestamped_path(
                self.archive_root,
                self.path.stem,
                extension=self.path.suffix,
            )
            archive_path.write_text(self.path.read_text(encoding="utf-8"), encoding="utf-8")

    @staticmethod
    def _default_archive_root(path: Path) -> Path:
        resolved = Path(path)
        if resolved.name == "neutral_setpoints.json" and resolved.parent.name == "config":
            return resolved.parent.parent / "data" / "calibration" / "servo_calibration"
        return resolved.parent / "history"

    @staticmethod
    def _is_legacy_neutral_payload(payload: dict[str, Any]) -> bool:
        if not payload:
            return False
        if "servos" in payload or "schema_version" in payload:
            return False
        return all(isinstance(value, (int, float)) for value in payload.values())


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _pretension_source_for_entry(entry: ServoCalibrationEntry) -> str:
    if entry.pretension_source not in (None, ""):
        return str(entry.pretension_source).strip().lower()
    run_record = dict(entry.latest_pretension_run or {})
    if run_record.get("source") not in (None, ""):
        return str(run_record.get("source")).strip().lower()
    if entry.pretension_result_status == "accepted" and entry.latest_pretension_run:
        return "algorithmic"
    return "unknown"

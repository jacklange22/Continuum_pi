"""Servo sign and tendon mapping verification artifacts."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
import json
from typing import Any

from continuum_robot.experiments.dataset_io import canonical_timestamp_token


SCHEMA_VERSION = 1


@dataclass
class ServoMappingCheckEntry:
    """One tiny-jog sign check observation."""

    servo_id: int
    expected_tendon_position: str = ""
    axis_label: str = ""
    positive_tendon_direction: str = ""
    tiny_jog_direction: str = ""
    observed_tendon_behavior: str = ""
    expected_sign_confirmed: bool | None = None
    notes: str = ""


@dataclass
class ServoMappingCheckRecord:
    """Saved sign/mapping verification record for one active segment."""

    schema_version: int = SCHEMA_VERSION
    timestamp_utc: str = ""
    operator: str = ""
    operating_mode: str = ""
    active_segment_key: str = ""
    active_segment_label: str = ""
    expected_servo_ids: list[int] = field(default_factory=list)
    source: str = "tiny_jog_sign_mapping_check"
    artifact_path: str = ""
    mock_mode: bool = False
    lower_tick_means_tension: bool = True
    confirmed_by_operator: bool = False
    entries: list[ServoMappingCheckEntry] = field(default_factory=list)
    notes: str = ""

    @property
    def confirmed_servo_ids(self) -> list[int]:
        return sorted(
            int(entry.servo_id)
            for entry in self.entries
            if entry.expected_sign_confirmed is True
        )

    def all_expected_confirmed(self) -> bool:
        expected = [int(value) for value in self.expected_servo_ids]
        confirmed = set(self.confirmed_servo_ids)
        return bool(expected) and all(int(servo_id) in confirmed for servo_id in expected)


@dataclass(frozen=True)
class ServoMappingCheckSummary:
    """Compact readiness view of the latest sign/mapping artifact."""

    exists: bool
    compatible: bool
    confirmed: bool
    path: str
    timestamp_utc: str | None
    message: str
    missing_servo_ids: list[int] = field(default_factory=list)
    unconfirmed_servo_ids: list[int] = field(default_factory=list)
    record: ServoMappingCheckRecord | None = None


class ServoMappingCheckRepository:
    """Read and write JSON sign/mapping checklist sidecars."""

    def __init__(self, root: Path | None = None) -> None:
        self.root = Path(root or Path.cwd() / "data" / "calibration" / "servo_mapping_checks")
        self.root.mkdir(parents=True, exist_ok=True)

    def create_checklist(
        self,
        *,
        operating_mode: str,
        active_segment_key: str,
        active_segment_label: str,
        expected_servo_ids: list[int],
        operator: str = "",
        mock_mode: bool = False,
        notes: str = "",
    ) -> ServoMappingCheckRecord:
        record = ServoMappingCheckRecord(
            timestamp_utc=_utc_now(),
            operator=str(operator or ""),
            operating_mode=str(operating_mode or ""),
            active_segment_key=str(active_segment_key or ""),
            active_segment_label=str(active_segment_label or ""),
            expected_servo_ids=[int(value) for value in expected_servo_ids],
            mock_mode=bool(mock_mode),
            entries=[
                ServoMappingCheckEntry(
                    servo_id=int(servo_id),
                    tiny_jog_direction="tighten_fine_then_loosen_fine",
                )
                for servo_id in expected_servo_ids
            ],
            notes=str(notes or ""),
        )
        return self.write(record)

    def confirm_configured_mapping(
        self,
        *,
        operating_mode: str,
        active_segment_key: str,
        active_segment_label: str,
        expected_servo_ids: list[int],
        active_segment_pairs: dict[str, list[int]],
        operator: str = "",
        mock_mode: bool = False,
        notes: str = "",
    ) -> ServoMappingCheckRecord:
        """Write a one-click configured mapping confirmation artifact."""
        mapping = configured_axis_mapping(
            expected_servo_ids=expected_servo_ids,
            active_segment_pairs=active_segment_pairs,
        )
        record = ServoMappingCheckRecord(
            timestamp_utc=_utc_now(),
            operator=str(operator or ""),
            operating_mode=str(operating_mode or ""),
            active_segment_key=str(active_segment_key or ""),
            active_segment_label=str(active_segment_label or ""),
            expected_servo_ids=[int(value) for value in expected_servo_ids],
            source="configured_mapping_confirmation",
            mock_mode=bool(mock_mode),
            lower_tick_means_tension=True,
            confirmed_by_operator=True,
            entries=[
                ServoMappingCheckEntry(
                    servo_id=int(servo_id),
                    expected_tendon_position=str(mapping[int(servo_id)]["tendon"]),
                    axis_label=str(mapping[int(servo_id)]["axis"]),
                    positive_tendon_direction=str(mapping[int(servo_id)]["tendon"]),
                    tiny_jog_direction="configured_lower_tick_tensions",
                    observed_tendon_behavior="configured_mapping_confirmed_by_operator",
                    expected_sign_confirmed=True,
                )
                for servo_id in [int(value) for value in expected_servo_ids]
            ],
            notes=str(notes or ""),
        )
        return self.write(record)

    def write(self, record: ServoMappingCheckRecord) -> ServoMappingCheckRecord:
        timestamp = record.timestamp_utc or _utc_now()
        safe_segment = _safe_token(record.active_segment_key or "segment")
        filename = f"{canonical_timestamp_token()}_{safe_segment}_servo_mapping_check.json"
        path = self.root / filename
        clean = ServoMappingCheckRecord(
            schema_version=SCHEMA_VERSION,
            timestamp_utc=timestamp,
            operator=str(record.operator or ""),
            operating_mode=str(record.operating_mode or ""),
            active_segment_key=str(record.active_segment_key or ""),
            active_segment_label=str(record.active_segment_label or ""),
            expected_servo_ids=[int(value) for value in record.expected_servo_ids],
            source=str(record.source or "tiny_jog_sign_mapping_check"),
            artifact_path=str(path),
            mock_mode=bool(record.mock_mode),
            lower_tick_means_tension=bool(record.lower_tick_means_tension),
            confirmed_by_operator=bool(record.confirmed_by_operator),
            entries=[
                ServoMappingCheckEntry(
                    servo_id=int(entry.servo_id),
                    expected_tendon_position=str(entry.expected_tendon_position or ""),
                    axis_label=str(entry.axis_label or ""),
                    positive_tendon_direction=str(entry.positive_tendon_direction or ""),
                    tiny_jog_direction=str(entry.tiny_jog_direction or ""),
                    observed_tendon_behavior=str(entry.observed_tendon_behavior or ""),
                    expected_sign_confirmed=(
                        None
                        if entry.expected_sign_confirmed is None
                        else bool(entry.expected_sign_confirmed)
                    ),
                    notes=str(entry.notes or ""),
                )
                for entry in record.entries
            ],
            notes=str(record.notes or ""),
        )
        path.write_text(json.dumps(asdict(clean), indent=2, sort_keys=True), encoding="utf-8")
        return clean

    def read(self, path: Path) -> ServoMappingCheckRecord:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError(f"Expected mapping check payload mapping in {path}")
        return _coerce_record(payload, path=Path(path))

    def latest_for_segment(
        self,
        *,
        active_segment_key: str,
        expected_servo_ids: list[int],
    ) -> ServoMappingCheckSummary:
        expected = [int(value) for value in expected_servo_ids]
        candidates = sorted(self.root.glob("*_servo_mapping_check.json"), reverse=True)
        latest: ServoMappingCheckRecord | None = None
        for path in candidates:
            try:
                record = self.read(path)
            except Exception:
                continue
            if str(record.active_segment_key) != str(active_segment_key):
                continue
            latest = record
            break
        if latest is None:
            return ServoMappingCheckSummary(
                exists=False,
                compatible=False,
                confirmed=False,
                path="",
                timestamp_utc=None,
                message=(
                    "Servo sign/mapping has not been recorded for this segment in this session. "
                    "Verify tiny jogs before live experiment."
                ),
                missing_servo_ids=list(expected),
                unconfirmed_servo_ids=list(expected),
            )
        recorded = [int(value) for value in latest.expected_servo_ids]
        missing = [int(value) for value in expected if int(value) not in recorded]
        unconfirmed = [
            int(value)
            for value in expected
            if int(value) not in latest.confirmed_servo_ids
        ]
        compatible = not missing and sorted(recorded) == sorted(expected)
        confirmed = compatible and not unconfirmed
        if not compatible:
            message = (
                f"Latest sign/mapping check expected servo IDs {recorded}, "
                f"but active segment expects {expected}."
            )
        elif not confirmed:
            message = (
                "Configured servo sign/mapping exists but is not operator-confirmed for servo(s): "
                + ", ".join(str(value) for value in unconfirmed)
            )
        else:
            message = (
                "Servo sign/mapping checklist is confirmed for active segment "
                f"{latest.active_segment_label or latest.active_segment_key}: {expected}."
            )
        return ServoMappingCheckSummary(
            exists=True,
            compatible=compatible,
            confirmed=confirmed,
            path=str(latest.artifact_path),
            timestamp_utc=latest.timestamp_utc,
            message=message,
            missing_servo_ids=missing,
            unconfirmed_servo_ids=unconfirmed,
            record=latest,
        )


def _coerce_record(payload: dict[str, Any], *, path: Path) -> ServoMappingCheckRecord:
    entries = []
    for item in list(payload.get("entries", []) or []):
        data = dict(item or {})
        confirmed = data.get("expected_sign_confirmed")
        entries.append(
            ServoMappingCheckEntry(
                servo_id=int(data.get("servo_id")),
                expected_tendon_position=str(data.get("expected_tendon_position", "") or ""),
                axis_label=str(data.get("axis_label", "") or ""),
                positive_tendon_direction=str(data.get("positive_tendon_direction", "") or ""),
                tiny_jog_direction=str(data.get("tiny_jog_direction", "") or ""),
                observed_tendon_behavior=str(data.get("observed_tendon_behavior", "") or ""),
                expected_sign_confirmed=None if confirmed is None else bool(confirmed),
                notes=str(data.get("notes", "") or ""),
            )
        )
    return ServoMappingCheckRecord(
        schema_version=int(payload.get("schema_version", SCHEMA_VERSION)),
        timestamp_utc=str(payload.get("timestamp_utc", "") or ""),
        operator=str(payload.get("operator", "") or ""),
        operating_mode=str(payload.get("operating_mode", "") or ""),
        active_segment_key=str(payload.get("active_segment_key", "") or ""),
        active_segment_label=str(payload.get("active_segment_label", "") or ""),
        expected_servo_ids=[int(value) for value in list(payload.get("expected_servo_ids", []) or [])],
        source=str(payload.get("source", "tiny_jog_sign_mapping_check") or "tiny_jog_sign_mapping_check"),
        artifact_path=str(payload.get("artifact_path", "") or str(path)),
        mock_mode=bool(payload.get("mock_mode", False)),
        lower_tick_means_tension=bool(payload.get("lower_tick_means_tension", True)),
        confirmed_by_operator=bool(payload.get("confirmed_by_operator", False)),
        entries=entries,
        notes=str(payload.get("notes", "") or ""),
    )


def configured_axis_mapping(
    *,
    expected_servo_ids: list[int],
    active_segment_pairs: dict[str, list[int]],
) -> dict[int, dict[str, str]]:
    """Return the configured single-segment axis/tendon label for each servo."""
    ids = [int(value) for value in expected_servo_ids]
    pairs = {str(key): [int(value) for value in values] for key, values in dict(active_segment_pairs or {}).items()}
    axis_a = pairs.get("axis_a") or ([ids[0], ids[2]] if len(ids) >= 4 else [])
    axis_b = pairs.get("axis_b") or ([ids[1], ids[3]] if len(ids) >= 4 else [])
    mapping: dict[int, dict[str, str]] = {}
    if len(axis_a) >= 2:
        mapping[int(axis_a[0])] = {"axis": "X", "tendon": "+X"}
        mapping[int(axis_a[1])] = {"axis": "X", "tendon": "-X"}
    if len(axis_b) >= 2:
        mapping[int(axis_b[0])] = {"axis": "Y", "tendon": "+Y"}
        mapping[int(axis_b[1])] = {"axis": "Y", "tendon": "-Y"}
    for servo_id in ids:
        mapping.setdefault(int(servo_id), {"axis": "", "tendon": ""})
    return mapping


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _safe_token(value: str) -> str:
    token = "".join(char.lower() if char.isalnum() else "_" for char in str(value or "segment"))
    return "_".join(part for part in token.split("_") if part) or "segment"

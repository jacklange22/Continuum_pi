"""Offline neutral / zero setpoint drift validation experiment.

Reads the append-only history log written by
``NeutralCalibrationService._append_neutral_zero_log_event`` (default location:
``data/calibration/neutral_zero_log.jsonl``) and analyzes how the per-servo
neutral setpoint has drifted over time across every operator capture, manual
pretension accept, automatic pretension save, and bulk re-zero.

Output contract (mirrors ``pivot_validation`` / ``registration_validation``):

  - 4 thesis-quality PNGs
  - debug.json with every analyzed event + per-servo statistics
  - per-event rows live in ``summary.json`` under
    ``experiment_metrics.per_servo_rows`` and ``experiment_metrics.events``
  - samples.jsonl is intentionally empty (typed timeseries contract)
  - metrics.csv and a summary text are intentionally NOT written here;
    debug.json is the structured fallback.

This is a pure-analysis experiment (no hardware required, ``mock_compatible=True``).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
import json
import math
from pathlib import Path
from typing import Any, Iterable

from continuum_robot.experiments.framework import BaseExperiment, ExperimentHardwareRequirements, ExperimentSession


# Schema version of the input log we know how to read.
NEUTRAL_ZERO_LOG_SCHEMA_VERSION = "neutral_zero_log_v1"

# The two event-type buckets the analysis layer cares about. Operators can
# include all events, or filter to the "accepted only" set for plots that
# show the operator-final timeline.
ACCEPTED_EVENT_TYPES = (
    "neutral_captured",
    "servo_calibration_saved",
    "manual_pretension_accepted",
    "pretension_accepted",
    "pretension_result_saved",
)
INTERMEDIATE_EVENT_TYPES = ("manual_pretension_captured",)
KNOWN_EVENT_TYPES = ACCEPTED_EVENT_TYPES + INTERMEDIATE_EVENT_TYPES


@dataclass
class NeutralSetpointDriftValidationConfig:
    """Config for the neutral / zero setpoint drift analysis.

    Inputs:
      - ``log_paths``: one or more JSONL history logs to read. When empty the
        default ``data/calibration/neutral_zero_log.jsonl`` is used at runtime.
      - ``include_event_types``: subset of event types to include in the
        analysis. Empty list = all known events. Operators typically narrow
        this to ``ACCEPTED_EVENT_TYPES`` to look at "what we actually
        committed to" rather than every intermediate capture.
      - ``include_servo_ids``: subset of servo IDs to include. Empty = all
        servos seen in the log. Useful when the operator wants to plot only
        the active segment.
      - ``accepted_events_only``: convenience toggle equivalent to setting
        ``include_event_types = ACCEPTED_EVENT_TYPES``. Defaults to False so
        the full timeline (including intermediate captures) is the default
        plot — operators can flip the gate from the GUI.
    """

    log_paths: list[str] = field(default_factory=list)
    include_event_types: list[str] = field(default_factory=list)
    include_servo_ids: list[int] = field(default_factory=list)
    accepted_events_only: bool = False

    @classmethod
    def from_dict(cls, payload: dict[str, Any] | None) -> "NeutralSetpointDriftValidationConfig":
        payload = dict(payload or {})
        return cls(
            log_paths=[str(value) for value in (payload.get("log_paths") or []) if str(value).strip()],
            include_event_types=[
                str(value).strip().lower()
                for value in (payload.get("include_event_types") or [])
                if str(value).strip()
            ],
            include_servo_ids=[
                int(value)
                for value in (payload.get("include_servo_ids") or [])
                if str(value).strip()
            ],
            accepted_events_only=bool(payload.get("accepted_events_only", False)),
        )


@dataclass
class NeutralSetpointLogCandidate:
    """One selectable JSONL log for the GUI page."""

    path: str
    label: str
    event_count: int
    first_timestamp_utc: str
    last_timestamp_utc: str
    servo_count: int
    event_types: list[str] = field(default_factory=list)


class NeutralSetpointDriftValidationExperiment(BaseExperiment):
    """Analyze neutral / zero setpoint drift over time across capture events.

    This is the GUI counterpart to ``registration_validation`` and
    ``pivot_validation``: a thesis-grade offline analysis that bundles the
    history log into citeable figures + summary metrics without touching the
    live calibration artifact.
    """

    name = "neutral_setpoint_drift_validation"
    description = (
        "Analyze the append-only neutral / zero setpoint history log "
        "(data/calibration/neutral_zero_log.jsonl) for per-servo drift over "
        "time, distribution per servo, event-type breakdown, and safe-bound "
        "stability. Produces thesis-quality figures without touching the "
        "canonical calibration artifact."
    )
    hardware_requirements = ExperimentHardwareRequirements(mock_compatible=True)

    def __init__(self, config: NeutralSetpointDriftValidationConfig) -> None:
        super().__init__(config)

    @classmethod
    def from_dict(cls, payload: dict[str, Any] | None = None) -> "NeutralSetpointDriftValidationExperiment":
        return cls(config=NeutralSetpointDriftValidationConfig.from_dict(payload))

    def precheck(self, session: ExperimentSession) -> None:
        log_paths = self._resolved_log_paths(session)
        if not log_paths:
            raise RuntimeError(
                "Neutral setpoint drift validation requires at least one history log "
                "(data/calibration/neutral_zero_log.jsonl). No log was found."
            )
        missing = [str(path) for path in log_paths if not path.exists()]
        if missing:
            raise RuntimeError(
                "Selected neutral-zero history logs are missing: "
                + ", ".join(missing[:3])
            )

    def execute(self, session: ExperimentSession) -> None:
        log_paths = self._resolved_log_paths(session)
        metrics, warnings = analyze_neutral_setpoint_drift_logs(
            log_paths,
            include_event_types=self._effective_event_types(),
            include_servo_ids=list(self.config.include_servo_ids),
        )
        if int(metrics.get("event_count", 0) or 0) <= 0:
            raise RuntimeError(
                "No matching events found in the selected neutral-zero history log(s). "
                "Loosen the event-type / servo-id filters and retry."
            )
        metrics["summary_requirements"] = {"force_status": "success"}
        session.metrics.update(metrics)
        for message in warnings:
            session.add_warning(message)

    def write_outputs(self, session: ExperimentSession, paths, summary) -> None:
        from continuum_robot.experiments.neutral_setpoint_drift_validation_outputs import (
            write_neutral_setpoint_drift_validation_outputs,
        )

        write_neutral_setpoint_drift_validation_outputs(
            output_dir=paths.output_dir,
            metadata=session.metadata,
            summary=summary,
        )

    # ------------------------------------------------------------------ #
    # helpers
    # ------------------------------------------------------------------ #

    def _resolved_log_paths(self, session: ExperimentSession) -> list[Path]:
        project_root = Path(session.context.project_root)
        if self.config.log_paths:
            return [_resolve_path(project_root, raw) for raw in self.config.log_paths]
        # Default: the canonical project-relative log.
        default = project_root / "data" / "calibration" / "neutral_zero_log.jsonl"
        return [default] if default.exists() else []

    def _effective_event_types(self) -> list[str]:
        if self.config.include_event_types:
            return list(self.config.include_event_types)
        if self.config.accepted_events_only:
            return list(ACCEPTED_EVENT_TYPES)
        return []  # empty = include all known types


def register_neutral_setpoint_drift_validation_experiment(registry) -> None:
    """Register the offline neutral/zero drift validation experiment."""
    registry.register(
        name=NeutralSetpointDriftValidationExperiment.name,
        title="Neutral Setpoint Drift Validation",
        description=NeutralSetpointDriftValidationExperiment.description,
        category="analysis",
        tags=["Calibration", "NeutralSetpoints", "Drift", "Validation"],
        default_config_path="config/experiment_neutral_setpoint_drift_validation.example.yaml",
        factory=NeutralSetpointDriftValidationExperiment.from_dict,
    )


# --------------------------------------------------------------------------- #
# Candidate discovery (for the GUI selection page).
# --------------------------------------------------------------------------- #


def list_neutral_setpoint_drift_candidates(
    project_root: Path,
    *,
    limit: int | None = None,
) -> list[NeutralSetpointLogCandidate]:
    """List JSONL history logs that look like the neutral-zero schema.

    The canonical log sits at ``data/calibration/neutral_zero_log.jsonl`` but
    operators can also keep archived copies (e.g. one per bench day) under
    ``data/calibration/neutral_zero_log_history/``. This function scans both
    locations and returns one candidate per usable file with a quick header
    summary (event count, time range, servo count) so the operator can pick
    which log to analyze.
    """
    project_root = Path(project_root)
    search_paths: list[Path] = []
    canonical = project_root / "data" / "calibration" / "neutral_zero_log.jsonl"
    if canonical.exists():
        search_paths.append(canonical)
    archive_dir = project_root / "data" / "calibration" / "neutral_zero_log_history"
    if archive_dir.exists():
        for path in sorted(archive_dir.glob("*.jsonl"), reverse=True):
            search_paths.append(path)
    if limit is not None:
        search_paths = search_paths[: max(0, int(limit))]
    candidates: list[NeutralSetpointLogCandidate] = []
    for path in search_paths:
        try:
            events = _load_log_events(path)
        except Exception:
            continue
        if not events:
            continue
        servo_ids: set[int] = set()
        event_types: set[str] = set()
        first_ts = events[0].get("timestamp_utc", "")
        last_ts = events[-1].get("timestamp_utc", "")
        for event in events:
            for sid_str in (event.get("by_servo") or {}):
                try:
                    servo_ids.add(int(sid_str))
                except (TypeError, ValueError):
                    continue
            event_types.add(str(event.get("event_type") or ""))
        candidates.append(
            NeutralSetpointLogCandidate(
                path=_display_path(path, project_root=project_root),
                label=path.name,
                event_count=len(events),
                first_timestamp_utc=str(first_ts),
                last_timestamp_utc=str(last_ts),
                servo_count=len(servo_ids),
                event_types=sorted(event_types),
            )
        )
    return candidates


# --------------------------------------------------------------------------- #
# Analysis — pure-data pipeline so the unit tests can hit it without I/O.
# --------------------------------------------------------------------------- #


def analyze_neutral_setpoint_drift_logs(
    log_paths: Iterable[Path | str],
    *,
    include_event_types: Iterable[str] | None = None,
    include_servo_ids: Iterable[int] | None = None,
) -> tuple[dict[str, Any], list[str]]:
    """Read every JSONL log and aggregate events into per-servo timelines.

    Returns ``(metrics, warnings)``. ``metrics`` is the experiment summary
    dict written to ``summary.json`` and consumed by the output writers;
    ``warnings`` are operator-facing messages about skipped events / files.
    """
    warnings: list[str] = []
    raw_events: list[dict[str, Any]] = []
    for raw in log_paths:
        path = Path(raw)
        try:
            file_events = _load_log_events(path)
        except FileNotFoundError:
            warnings.append(f"Log not found: {path}")
            continue
        except Exception as exc:  # pragma: no cover - defensive
            warnings.append(f"Skipped {path.name}: {exc}")
            continue
        for event in file_events:
            event = dict(event)
            event.setdefault("_source_log_path", str(path))
            raw_events.append(event)

    return analyze_neutral_setpoint_drift_events(
        raw_events,
        include_event_types=include_event_types,
        include_servo_ids=include_servo_ids,
        warnings=warnings,
    )


def analyze_neutral_setpoint_drift_events(
    events: Iterable[dict[str, Any]],
    *,
    include_event_types: Iterable[str] | None = None,
    include_servo_ids: Iterable[int] | None = None,
    warnings: list[str] | None = None,
) -> tuple[dict[str, Any], list[str]]:
    """Pure analysis pass: events in, metrics out. Unit tests use this entry
    point directly without touching the filesystem."""
    warnings = list(warnings or [])
    event_type_filter = {str(value).strip().lower() for value in (include_event_types or []) if str(value).strip()}
    servo_filter = {int(value) for value in (include_servo_ids or [])}

    # Filter + sort by timestamp.
    filtered_events: list[dict[str, Any]] = []
    for event in events:
        event_type = str(event.get("event_type") or "").strip().lower()
        if event_type_filter and event_type not in event_type_filter:
            continue
        filtered_events.append(event)
    filtered_events.sort(key=lambda e: str(e.get("timestamp_utc") or ""))

    # Build per-servo timelines.
    timelines: dict[int, list[dict[str, Any]]] = {}
    event_type_counts: dict[str, int] = {}
    capture_source_counts: dict[str, int] = {}
    for event_index, event in enumerate(filtered_events):
        event_type = str(event.get("event_type") or "").strip().lower()
        event_type_counts[event_type] = int(event_type_counts.get(event_type, 0)) + 1
        capture_source = str(event.get("capture_source") or "")
        capture_source_counts[capture_source] = int(capture_source_counts.get(capture_source, 0)) + 1
        timestamp = str(event.get("timestamp_utc") or "")
        by_servo = dict(event.get("by_servo") or {})
        for sid_str, state in by_servo.items():
            try:
                sid = int(sid_str)
            except (TypeError, ValueError):
                continue
            if servo_filter and sid not in servo_filter:
                continue
            timelines.setdefault(sid, []).append(
                {
                    "event_index": int(event_index),
                    "timestamp_utc": timestamp,
                    "event_type": event_type,
                    "capture_source": capture_source,
                    "operator_note": event.get("operator_note"),
                    "accepted": event.get("accepted"),
                    "neutral_setpoint": _as_optional_int(state.get("neutral_setpoint")),
                    "safe_min_tick": _as_optional_int(state.get("safe_min_tick")),
                    "safe_max_tick": _as_optional_int(state.get("safe_max_tick")),
                    "pretension_final_position_tick": _as_optional_int(state.get("pretension_final_position_tick")),
                    "pretension_result_status": state.get("pretension_result_status"),
                    "pretension_source": state.get("pretension_source"),
                    "last_measured_current_ma": _as_optional_int(state.get("last_measured_current_ma")),
                    "tightening_rotation": state.get("tightening_rotation"),
                    "calibrated_at_utc": state.get("calibrated_at_utc"),
                    "status": state.get("status"),
                    "active_segment_key": event.get("active_segment_key"),
                    "active_segment_label": event.get("active_segment_label"),
                    "operating_mode": event.get("operating_mode"),
                    "_source_log_path": event.get("_source_log_path"),
                }
            )

    # Per-servo summary statistics.
    per_servo_rows: list[dict[str, Any]] = []
    for sid in sorted(timelines):
        entries = timelines[sid]
        neutrals = [int(row["neutral_setpoint"]) for row in entries if row["neutral_setpoint"] is not None]
        finals = [int(row["pretension_final_position_tick"]) for row in entries if row["pretension_final_position_tick"] is not None]
        currents = [
            int(row["last_measured_current_ma"])
            for row in entries
            if row["last_measured_current_ma"] is not None
        ]
        safe_mins = [int(row["safe_min_tick"]) for row in entries if row["safe_min_tick"] is not None]
        safe_maxs = [int(row["safe_max_tick"]) for row in entries if row["safe_max_tick"] is not None]
        per_servo_rows.append(
            {
                "servo_id": int(sid),
                "event_count": int(len(entries)),
                "first_event_index": int(entries[0]["event_index"]) if entries else None,
                "last_event_index": int(entries[-1]["event_index"]) if entries else None,
                "first_timestamp_utc": entries[0]["timestamp_utc"] if entries else "",
                "last_timestamp_utc": entries[-1]["timestamp_utc"] if entries else "",
                "neutral_setpoint": _summarize_scalars(neutrals),
                "pretension_final_position_tick": _summarize_scalars(finals),
                "last_measured_current_ma": _summarize_scalars(currents),
                "safe_min_tick": _summarize_scalars(safe_mins),
                "safe_max_tick": _summarize_scalars(safe_maxs),
                "first_neutral_setpoint": neutrals[0] if neutrals else None,
                "last_neutral_setpoint": neutrals[-1] if neutrals else None,
                "neutral_drift_ticks": (
                    int(neutrals[-1] - neutrals[0])
                    if len(neutrals) >= 2
                    else 0
                ),
                "max_excursion_from_first_ticks": (
                    int(max(abs(value - neutrals[0]) for value in neutrals))
                    if neutrals
                    else 0
                ),
            }
        )

    # Overall metrics.
    timestamps_sorted = [
        str(event.get("timestamp_utc") or "") for event in filtered_events
    ]
    metrics: dict[str, Any] = {
        "experiment_name": NeutralSetpointDriftValidationExperiment.name,
        "schema_version_input": NEUTRAL_ZERO_LOG_SCHEMA_VERSION,
        "event_count": int(len(filtered_events)),
        "servo_count": int(len(timelines)),
        "event_type_counts": dict(event_type_counts),
        "capture_source_counts": dict(capture_source_counts),
        "first_event_timestamp_utc": timestamps_sorted[0] if timestamps_sorted else "",
        "last_event_timestamp_utc": timestamps_sorted[-1] if timestamps_sorted else "",
        "filter_event_types": sorted(event_type_filter),
        "filter_servo_ids": sorted(servo_filter),
        "per_servo_rows": per_servo_rows,
        "events": filtered_events,
        "timelines_by_servo": {str(sid): timelines[sid] for sid in sorted(timelines)},
        "definitions": {
            "neutral_setpoint": "Per-servo neutral / zero position in raw ticks recorded at the timestamp.",
            "neutral_drift_ticks": "Signed difference between the last and first observed neutral_setpoint in the filtered timeline.",
            "max_excursion_from_first_ticks": "Maximum absolute deviation of any captured neutral from the first observed value.",
            "safe_min_tick / safe_max_tick": "Per-servo safe operating bounds at capture time; usually neutral ± offset.",
            "last_measured_current_ma": "Signed current at the time of the capture (negative = motor holding against tendon load).",
        },
        "units": {
            "neutral_setpoint": "ticks (raw XC330 position, 0..4095)",
            "safe_min_tick": "ticks",
            "safe_max_tick": "ticks",
            "last_measured_current_ma": "mA (signed; negative = tensioned on this rig)",
            "neutral_drift_ticks": "ticks",
            "max_excursion_from_first_ticks": "ticks",
        },
    }
    return metrics, warnings


# --------------------------------------------------------------------------- #
# Internal helpers
# --------------------------------------------------------------------------- #


def _load_log_events(path: Path) -> list[dict[str, Any]]:
    """Load and validate a JSONL neutral-zero log file."""
    if not path.exists():
        raise FileNotFoundError(str(path))
    events: list[dict[str, Any]] = []
    for line_index, line in enumerate(path.read_text(encoding="utf-8").splitlines()):
        line = line.strip()
        if not line:
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(record, dict):
            continue
        # Hard-skip records with an unknown schema version so future formats
        # don't quietly mis-render. Records lacking the field are treated as
        # v1 (legacy logs written before the constant was added).
        schema_version = str(record.get("schema_version") or NEUTRAL_ZERO_LOG_SCHEMA_VERSION)
        if schema_version != NEUTRAL_ZERO_LOG_SCHEMA_VERSION:
            continue
        events.append(record)
    return events


def _summarize_scalars(values: list[int]) -> dict[str, Any]:
    if not values:
        return {
            "count": 0,
            "mean": None,
            "median": None,
            "std": None,
            "min": None,
            "max": None,
            "range": None,
        }
    n = len(values)
    sorted_values = sorted(values)
    mean = sum(values) / float(n)
    var = sum((v - mean) ** 2 for v in values) / float(n)
    std = math.sqrt(var)
    if n % 2 == 1:
        median = float(sorted_values[n // 2])
    else:
        median = (float(sorted_values[n // 2 - 1]) + float(sorted_values[n // 2])) / 2.0
    return {
        "count": int(n),
        "mean": float(mean),
        "median": float(median),
        "std": float(std),
        "min": int(sorted_values[0]),
        "max": int(sorted_values[-1]),
        "range": int(sorted_values[-1] - sorted_values[0]),
    }


def _as_optional_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _resolve_path(project_root: Path, raw_path: str | Path) -> Path:
    candidate = Path(raw_path)
    if candidate.is_absolute():
        return candidate
    return Path(project_root) / candidate


def _display_path(path: Path, *, project_root: Path) -> str:
    try:
        return str(path.relative_to(project_root))
    except ValueError:
        return str(path)

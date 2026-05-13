"""Servo transport soak diagnostics and bundle writer."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
import csv
import json
from pathlib import Path
import time
from typing import Any, Callable

import yaml

from continuum_robot.app.bootstrap import build_app_context
from continuum_robot.experiments.dataset_io import canonical_timestamped_path
from continuum_robot.hardware.dxl_bus import ServoTelemetry
from continuum_robot.servos.servo_service import ServoBusBusyError
from continuum_robot.utils.logging_setup import current_session_log_path


SOAK_STATIC = "static_telemetry"
SOAK_ONE_SERVO = "one_servo_motion"
SOAK_NEUTRAL_HOLD = "neutral_hold_after_move"
SOAK_COORDINATED = "coordinated_micro_motion"
SOAK_MODES = (SOAK_STATIC, SOAK_ONE_SERVO, SOAK_NEUTRAL_HOLD, SOAK_COORDINATED)

PACKET_FAILURE_TYPES = {
    "no_status_packet",
    "incorrect_status_packet",
    "tx_rx_error",
    "missing_position",
    "missing_current",
    "missing_voltage",
    "stale_telemetry",
    "servo_missing",
    "owner_conflict",
    "bus_contention",
    "safety_limit_rejected",
    "arming_failed",
    "torque_enable_failed",
    "post_move_validation_failed",
}


@dataclass
class ServoTransportSoakConfig:
    """Configuration for one bounded servo transport soak."""

    mode: str = SOAK_STATIC
    servo_ids: list[int] = field(default_factory=list)
    selected_servo_id: int | None = None
    duration_s: float = 30.0
    sample_period_s: float = 0.1
    cycles: int = 20
    step_ticks: int = 8
    coordinated_delta_ticks: int = 6
    stop_on_first_serious_failure: bool = False
    output_dir: str | None = None
    session_log_path: str | None = None

    @classmethod
    def from_dict(cls, payload: dict[str, Any] | None) -> "ServoTransportSoakConfig":
        payload = dict(payload or {})
        mode = str(payload.get("mode", SOAK_STATIC) or SOAK_STATIC).strip().lower()
        if mode not in SOAK_MODES:
            raise ValueError(f"Unsupported servo transport soak mode: {mode}")
        servo_ids = [int(value) for value in list(payload.get("servo_ids", []) or [])]
        selected = payload.get("selected_servo_id")
        return cls(
            mode=mode,
            servo_ids=servo_ids,
            selected_servo_id=None if selected in (None, "") else int(selected),
            duration_s=max(0.0, float(payload.get("duration_s", 30.0))),
            sample_period_s=max(0.0, float(payload.get("sample_period_s", 0.1))),
            cycles=max(1, int(payload.get("cycles", 20))),
            step_ticks=max(1, int(payload.get("step_ticks", 8))),
            coordinated_delta_ticks=max(1, int(payload.get("coordinated_delta_ticks", 6))),
            stop_on_first_serious_failure=bool(payload.get("stop_on_first_serious_failure", mode == SOAK_COORDINATED)),
            output_dir=(str(payload.get("output_dir")).strip() if payload.get("output_dir") not in (None, "") else None),
            session_log_path=(
                str(payload.get("session_log_path")).strip()
                if payload.get("session_log_path") not in (None, "")
                else None
            ),
        )


@dataclass
class ServoTransportFailure:
    """One classified servo transport or telemetry failure."""

    timestamp_utc: str
    monotonic_time_s: float
    servo_ids: list[int]
    failure_type: str
    field: str | None
    mode: str
    stage: str
    message: str
    last_commanded_goals: dict[str, int]
    last_known_telemetry: dict[str, Any]
    bus_ownership_state: dict[str, Any]
    stale_age_s: float | None
    just_moved: bool
    idle: bool


@dataclass
class ServoTransportSample:
    """One soak sample row written to JSONL and CSV."""

    timestamp_utc: str
    monotonic_time_s: float
    elapsed_s: float
    mode: str
    stage: str
    cycle_index: int
    sample_index: int
    just_moved: bool
    commanded_goals: dict[str, int]
    telemetry_by_servo: dict[str, Any]
    failures: list[dict[str, Any]]


@dataclass
class ServoTransportSoakResult:
    """Completed soak run with generated bundle paths."""

    success: bool
    status: str
    mode: str
    output_dir: Path
    summary_path: Path
    sample_count: int
    failure_count: int
    serious_failure_count: int
    message: str


def classify_packet_failure(
    *,
    message: str | None = None,
    telemetry: ServoTelemetry | None = None,
    field: str | None = None,
    stale: bool = False,
    owner_conflict: bool = False,
    stage: str = "",
) -> str | None:
    """Map SDK/status/telemetry symptoms into stable diagnostic failure labels."""
    if owner_conflict:
        return "owner_conflict"
    lowered = str(message or "").lower()
    if "incorrect status packet" in lowered:
        return "incorrect_status_packet"
    if "no status packet" in lowered:
        return "no_status_packet"
    if "txrx" in lowered or "tx rx" in lowered or "tx_rx" in lowered or "comm_" in lowered:
        return "tx_rx_error"
    if "bus is owned" in lowered or "background refresh is paused" in lowered:
        return "bus_contention"
    if "torque" in lowered and ("failed" in lowered or "unavailable" in lowered):
        return "torque_enable_failed"
    if "outside" in lowered or "rejected" in lowered or "limit" in lowered:
        return "safety_limit_rejected"
    if "post" in str(stage).lower() or "post-command" in lowered or "post motion" in lowered:
        return "post_move_validation_failed"
    if telemetry is None:
        return "servo_missing"
    merged = " | ".join(
        value
        for value in (
            telemetry.identity_error,
            telemetry.telemetry_error,
            telemetry.hardware_error,
        )
        if value
    ).lower()
    if "incorrect status packet" in merged:
        return "incorrect_status_packet"
    if "no status packet" in merged:
        return "no_status_packet"
    if "[txrxresult]" in merged or "txrx" in merged:
        return "tx_rx_error"
    if "missing servo" in merged or "missing" == merged.strip():
        return "servo_missing"
    if field == "position" and telemetry.present_position is None:
        return "missing_position"
    if field == "current" and telemetry.present_current_ma is None:
        return "missing_current"
    if field == "voltage" and telemetry.present_voltage_mv is None:
        return "missing_voltage"
    if stale:
        return "stale_telemetry"
    return None


def run_servo_transport_soak(
    servo_service,
    config: ServoTransportSoakConfig,
    *,
    output_root: Path,
    sleep_fn: Callable[[float], None] = time.sleep,
    time_fn: Callable[[], float] = time.monotonic,
) -> ServoTransportSoakResult:
    """Run one servo transport soak and write a retrieval-friendly bundle."""
    servo_ids = [int(value) for value in (config.servo_ids or [])]
    if not servo_ids:
        servo_ids = [int(value) for value in getattr(servo_service.neutral_calibration.context, "servo_ids", [])]
    if not servo_ids:
        raise ValueError("No servo IDs were supplied or configured.")
    selected_servo_id = int(config.selected_servo_id or servo_ids[0])
    if selected_servo_id not in servo_ids:
        servo_ids.append(selected_servo_id)
        servo_ids = sorted(set(servo_ids))
    output_dir = Path(config.output_dir) if config.output_dir else canonical_timestamped_path(
        Path(output_root),
        f"servo_transport_{config.mode}",
    )
    output_dir.mkdir(parents=True, exist_ok=False)
    samples: list[ServoTransportSample] = []
    failures: list[ServoTransportFailure] = []
    run_started = float(time_fn())
    last_commanded: dict[int, int] = {}
    last_moved = False
    status = "success"
    message = f"Completed servo transport soak mode={config.mode}."

    def _record_exception(stage: str, exc: Exception, *, just_moved: bool) -> None:
        nonlocal status, message
        failure_type = classify_packet_failure(
            message=str(exc),
            owner_conflict=isinstance(exc, ServoBusBusyError),
            stage=stage,
        ) or "tx_rx_error"
        status = "failed"
        message = f"Servo transport soak failed during {stage}: {exc}"
        failures.append(
            _failure_event(
                servo_service=servo_service,
                servo_ids=servo_ids,
                failure_type=failure_type,
                field=None,
                mode=config.mode,
                stage=stage,
                message=str(exc),
                last_commanded=last_commanded,
                telemetry=None,
                stale_age_s=None,
                just_moved=just_moved,
                now=float(time_fn()),
            )
        )

    def _read_sample(stage: str, cycle_index: int, sample_index: int, *, just_moved: bool) -> bool:
        nonlocal last_moved
        now = float(time_fn())
        try:
            telemetry = servo_service.read_live_telemetry(servo_ids)
        except Exception as exc:
            _record_exception(stage, exc, just_moved=just_moved)
            sample = ServoTransportSample(
                timestamp_utc=_utc_now_iso(),
                monotonic_time_s=now,
                elapsed_s=max(0.0, now - run_started),
                mode=config.mode,
                stage=stage,
                cycle_index=int(cycle_index),
                sample_index=int(sample_index),
                just_moved=bool(just_moved),
                commanded_goals={str(k): int(v) for k, v in last_commanded.items()},
                telemetry_by_servo={},
                failures=[asdict(failures[-1])],
            )
            samples.append(sample)
            last_moved = bool(just_moved)
            return False
        sample_failures: list[ServoTransportFailure] = []
        for servo_id in servo_ids:
            item = telemetry.get(int(servo_id))
            sample_failures.extend(
                _classify_telemetry_failures(
                    servo_service=servo_service,
                    servo_id=int(servo_id),
                    telemetry=item,
                    mode=config.mode,
                    stage=stage,
                    last_commanded=last_commanded,
                    just_moved=just_moved,
                    now=now,
                )
            )
        failures.extend(sample_failures)
        sample = ServoTransportSample(
            timestamp_utc=_utc_now_iso(),
            monotonic_time_s=now,
            elapsed_s=max(0.0, now - run_started),
            mode=config.mode,
            stage=stage,
            cycle_index=int(cycle_index),
            sample_index=int(sample_index),
            just_moved=bool(just_moved),
            commanded_goals={str(k): int(v) for k, v in last_commanded.items()},
            telemetry_by_servo={str(k): _telemetry_payload(v, servo_service=servo_service) for k, v in telemetry.items()},
            failures=[asdict(item) for item in sample_failures],
        )
        samples.append(sample)
        last_moved = bool(just_moved)
        return not sample_failures

    try:
        with servo_service.exclusive_bus_operation(
            owner="servo_transport_soak",
            servo_id=selected_servo_id if config.mode == SOAK_ONE_SERVO else None,
            reason=config.mode,
        ):
            if config.mode == SOAK_STATIC:
                _run_static_soak(config, _read_sample, sleep_fn, time_fn, run_started)
            elif config.mode == SOAK_ONE_SERVO:
                last_commanded.update(
                    _run_one_servo_motion(config, servo_service, selected_servo_id, _read_sample, sleep_fn)
                )
            elif config.mode == SOAK_NEUTRAL_HOLD:
                last_commanded.update(_command_neutral_hold(servo_service, servo_ids))
                _run_static_soak(config, _read_sample, sleep_fn, time_fn, run_started, stage="hold_read")
            elif config.mode == SOAK_COORDINATED:
                last_commanded.update(
                    _run_coordinated_micro_motion(config, servo_service, servo_ids, _read_sample, sleep_fn)
                )
    except Exception as exc:
        _record_exception("run", exc, just_moved=last_moved)

    if failures and (status == "success" or config.stop_on_first_serious_failure):
        status = "failed" if config.stop_on_first_serious_failure else "completed_with_failures"
        message = f"Servo transport soak completed with {len(failures)} classified failure(s)."
    success = status == "success"
    paths = _write_bundle(
        output_dir=output_dir,
        config=config,
        servo_ids=servo_ids,
        samples=samples,
        failures=failures,
        success=success,
        status=status,
        message=message,
        servo_service=servo_service,
        session_log_path=Path(config.session_log_path) if config.session_log_path else current_session_log_path(),
    )
    return ServoTransportSoakResult(
        success=success,
        status=status,
        mode=config.mode,
        output_dir=output_dir,
        summary_path=paths["summary_json"],
        sample_count=len(samples),
        failure_count=len(failures),
        serious_failure_count=len(failures),
        message=message,
    )


def _run_static_soak(config, read_sample, sleep_fn, time_fn, run_started, *, stage: str = "idle_read") -> None:
    sample_index = 0
    while True:
        elapsed = max(0.0, float(time_fn()) - float(run_started))
        if sample_index > 0 and float(config.duration_s) <= 0:
            break
        if config.duration_s > 0 and elapsed >= float(config.duration_s) and sample_index > 0:
            break
        ok = read_sample(stage, 0, sample_index, just_moved=False)
        sample_index += 1
        if config.stop_on_first_serious_failure and not ok:
            break
        sleep_fn(float(config.sample_period_s))


def _run_one_servo_motion(config, servo_service, selected_servo_id, read_sample, sleep_fn) -> dict[int, int]:
    telemetry = servo_service.read_live_telemetry([int(selected_servo_id)])[int(selected_servo_id)]
    if telemetry.present_position is None:
        raise RuntimeError(f"Servo {selected_servo_id} position is unavailable before one-servo soak.")
    base = int(telemetry.present_position)
    last_commanded: dict[int, int] = {}
    for cycle in range(int(config.cycles)):
        target = base + (int(config.step_ticks) if cycle % 2 == 0 else -int(config.step_ticks))
        result = servo_service.move_servo_to_raw_target(
            servo_id=int(selected_servo_id),
            target_tick=int(target),
            reason="servo_transport_one_servo_soak",
        )
        if result.goal_tick is not None:
            last_commanded[int(selected_servo_id)] = int(result.goal_tick)
        ok = read_sample("one_servo_post_move", cycle, cycle, just_moved=True)
        if config.stop_on_first_serious_failure and not ok:
            break
        sleep_fn(float(config.sample_period_s))
    return last_commanded


def _command_neutral_hold(servo_service, servo_ids: list[int]) -> dict[int, int]:
    resolved = servo_service.resolve_startup_reference_ticks(servo_ids)
    commanded: dict[int, int] = {}
    for servo_id, target in dict(resolved.ticks_by_servo).items():
        result = servo_service.move_servo_to_raw_target(
            servo_id=int(servo_id),
            target_tick=int(target),
            reason="servo_transport_neutral_hold",
        )
        commanded[int(servo_id)] = int(result.goal_tick if result.goal_tick is not None else target)
    return commanded


def _run_coordinated_micro_motion(config, servo_service, servo_ids: list[int], read_sample, sleep_fn) -> dict[int, int]:
    if len(servo_ids) != 4:
        raise RuntimeError("Coordinated micro-motion soak requires exactly four configured servos.")
    reference = servo_service.resolve_startup_reference_ticks(servo_ids)
    neutral_ticks = [int(reference.ticks_by_servo[int(servo_id)]) for servo_id in servo_ids]
    delta_cm = _ticks_to_cm(servo_service, int(config.coordinated_delta_ticks))
    last_commanded: dict[int, int] = {}
    patterns = ([delta_cm, 0.0, -delta_cm, 0.0], [-delta_cm, 0.0, delta_cm, 0.0], [0.0, delta_cm, 0.0, -delta_cm], [0.0, -delta_cm, 0.0, delta_cm])
    for cycle in range(int(config.cycles)):
        pattern = patterns[cycle % len(patterns)]
        result = servo_service.command_displacement(
            tendon_displacements_cm=list(pattern),
            neutral_ticks=list(neutral_ticks),
            servo_ids=list(servo_ids),
            motion_workflow="experiment_motion",
        )
        last_commanded = {int(k): int(v) for k, v in result.positions_by_id.items()}
        ok = read_sample("coordinated_post_move", cycle, cycle, just_moved=True)
        if config.stop_on_first_serious_failure and not ok:
            break
        sleep_fn(float(config.sample_period_s))
    return last_commanded


def _ticks_to_cm(servo_service, ticks: int) -> float:
    mapper = servo_service.mapper
    ticks_per_rev = float(getattr(mapper, "ticks_per_rev", 4096) or 4096)
    spool_diameter_cm = float(getattr(mapper, "spool_diameter_cm", 1.2) or 1.2)
    return float(ticks) / ticks_per_rev * 3.141592653589793 * spool_diameter_cm


def _classify_telemetry_failures(
    *,
    servo_service,
    servo_id: int,
    telemetry: ServoTelemetry | None,
    mode: str,
    stage: str,
    last_commanded: dict[int, int],
    just_moved: bool,
    now: float,
) -> list[ServoTransportFailure]:
    events: list[ServoTransportFailure] = []
    fields = {"position": "missing_position", "current": "missing_current", "voltage": "missing_voltage"}
    stale_age = servo_service.telemetry_age_s(telemetry)
    stale = servo_service.telemetry_is_fresh(telemetry) is False
    for field_name in fields:
        failure_type = classify_packet_failure(telemetry=telemetry, field=field_name, stale=False)
        if failure_type:
            events.append(
                _failure_event(
                    servo_service=servo_service,
                    servo_ids=[int(servo_id)],
                    failure_type=failure_type,
                    field=field_name,
                    mode=mode,
                    stage=stage,
                    message=_telemetry_error_message(telemetry, fallback=failure_type),
                    last_commanded=last_commanded,
                    telemetry=telemetry,
                    stale_age_s=stale_age,
                    just_moved=just_moved,
                    now=now,
                )
            )
    packet_failure = classify_packet_failure(telemetry=telemetry)
    if packet_failure and packet_failure not in {event.failure_type for event in events}:
        events.append(
            _failure_event(
                servo_service=servo_service,
                servo_ids=[int(servo_id)],
                failure_type=packet_failure,
                field=None,
                mode=mode,
                stage=stage,
                message=_telemetry_error_message(telemetry, fallback=packet_failure),
                last_commanded=last_commanded,
                telemetry=telemetry,
                stale_age_s=stale_age,
                just_moved=just_moved,
                now=now,
            )
        )
    if stale:
        events.append(
            _failure_event(
                servo_service=servo_service,
                servo_ids=[int(servo_id)],
                failure_type="stale_telemetry",
                field=None,
                mode=mode,
                stage=stage,
                message=f"Telemetry age {stale_age} exceeded freshness threshold.",
                last_commanded=last_commanded,
                telemetry=telemetry,
                stale_age_s=stale_age,
                just_moved=just_moved,
                now=now,
            )
        )
    return events


def _failure_event(
    *,
    servo_service,
    servo_ids: list[int],
    failure_type: str,
    field: str | None,
    mode: str,
    stage: str,
    message: str,
    last_commanded: dict[int, int],
    telemetry: ServoTelemetry | None,
    stale_age_s: float | None,
    just_moved: bool,
    now: float,
) -> ServoTransportFailure:
    owner = servo_service.bus_ownership_status()
    return ServoTransportFailure(
        timestamp_utc=_utc_now_iso(),
        monotonic_time_s=float(now),
        servo_ids=[int(value) for value in servo_ids],
        failure_type=str(failure_type),
        field=field,
        mode=str(mode),
        stage=str(stage),
        message=str(message),
        last_commanded_goals={str(k): int(v) for k, v in dict(last_commanded).items()},
        last_known_telemetry=_telemetry_payload(telemetry, servo_service=servo_service),
        bus_ownership_state=asdict(owner),
        stale_age_s=stale_age_s,
        just_moved=bool(just_moved),
        idle=not bool(just_moved),
    )


def _telemetry_payload(telemetry: ServoTelemetry | None, *, servo_service) -> dict[str, Any]:
    if telemetry is None:
        return {}
    age_s = servo_service.telemetry_age_s(telemetry)
    suspect = bool(
        telemetry.telemetry_error
        or telemetry.identity_error
        or telemetry.present_current_ma is None
    )
    return {
        "servo_id": int(telemetry.servo_id),
        "reported_servo_id": telemetry.reported_servo_id,
        "position_tick": telemetry.present_position,
        "current_raw_unit": telemetry.present_current_raw_unit,
        "current_ma": telemetry.present_current_ma,
        "current_source": "servo_reported_register_estimate",
        "current_trust": "suspect_packet_or_missing" if suspect else "servo_register_readback_not_bench_calibrated",
        "current_note": "Not bench supply current; this is the servo Present Current register converted by configured scale.",
        "voltage_mv": telemetry.present_voltage_mv,
        "voltage_source": "servo_reported_input_voltage_register",
        "temperature_c": telemetry.present_temperature_c,
        "torque_enabled": telemetry.torque_enabled,
        "operating_mode": telemetry.operating_mode,
        "hardware_error_code": telemetry.hardware_error_code,
        "hardware_error": telemetry.hardware_error,
        "identity_error": telemetry.identity_error,
        "telemetry_error": telemetry.telemetry_error,
        "telemetry_age_s": age_s,
        "last_read_monotonic_s": telemetry.last_read_monotonic_s,
        "last_valid_packet_monotonic_s": getattr(telemetry, "last_valid_packet_monotonic_s", None),
        "last_valid_packet_wall_time": getattr(telemetry, "last_valid_packet_wall_time", None),
        "last_read_attempt_monotonic_s": getattr(telemetry, "last_read_attempt_monotonic_s", None),
        "read_duration_ms": getattr(telemetry, "read_duration_ms", None),
        "packet_age_s": getattr(telemetry, "packet_age_s", None),
        "read_source": getattr(telemetry, "read_source", None),
        "telemetry_error_code": getattr(telemetry, "telemetry_error_code", None),
        "telemetry_error_detail": getattr(telemetry, "telemetry_error_detail", None),
        "bus_owner": getattr(telemetry, "bus_owner", None),
        "read_sequence_index": getattr(telemetry, "read_sequence_index", None),
    }


def _telemetry_error_message(telemetry: ServoTelemetry | None, *, fallback: str) -> str:
    if telemetry is None:
        return fallback
    return telemetry.telemetry_error or telemetry.identity_error or telemetry.hardware_error or fallback


def _write_bundle(
    *,
    output_dir: Path,
    config: ServoTransportSoakConfig,
    servo_ids: list[int],
    samples: list[ServoTransportSample],
    failures: list[ServoTransportFailure],
    success: bool,
    status: str,
    message: str,
    servo_service,
    session_log_path: Path | None,
) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    failure_counter = Counter(event.failure_type for event in failures)
    servo_failure_counter: Counter[int] = Counter()
    field_failure_counter: Counter[str] = Counter()
    for event in failures:
        for servo_id in event.servo_ids:
            servo_failure_counter[int(servo_id)] += 1
        if event.field:
            field_failure_counter[str(event.field)] += 1
    summary = {
        "schema_version": "1.0",
        "success": bool(success),
        "status": str(status),
        "mode": config.mode,
        "message": message,
        "timestamp_utc": _utc_now_iso(),
        "servo_ids": [int(value) for value in servo_ids],
        "sample_count": len(samples),
        "failure_count": len(failures),
        "failure_counts_by_type": dict(sorted(failure_counter.items())),
        "failure_counts_by_servo": {str(k): int(v) for k, v in sorted(servo_failure_counter.items())},
        "failure_counts_by_field": dict(sorted(field_failure_counter.items())),
        "bus_ownership_final": asdict(servo_service.bus_ownership_status()),
        "last_commanded_goals": {str(k): int(v) for k, v in servo_service.last_goal_positions().items()},
        "current_telemetry_semantics": {
            "servo_current": "present_current_ma is a DYNAMIXEL Present Current register estimate.",
            "bench_supply_current": "not measured by this software unless an external sensor is explicitly added.",
            "suspect_policy": "current is marked suspect when packet/status/field validity is degraded.",
        },
        "config": asdict(config),
    }
    paths: dict[str, Path] = {}
    paths["summary_json"] = output_dir / "summary.json"
    paths["summary_json"].write_text(json.dumps(summary, indent=2), encoding="utf-8")
    paths["samples_jsonl"] = output_dir / "samples.jsonl"
    with paths["samples_jsonl"].open("w", encoding="utf-8") as handle:
        for sample in samples:
            handle.write(json.dumps(asdict(sample), separators=(",", ":")) + "\n")
    paths["metrics_csv"] = output_dir / "metrics.csv"
    _write_metrics_csv(paths["metrics_csv"], samples)
    paths["config_snapshot"] = output_dir / "config_snapshot.yaml"
    paths["config_snapshot"].write_text(yaml.safe_dump(asdict(config), sort_keys=False), encoding="utf-8")
    paths["summary_text"] = output_dir / "servo_transport_summary.txt"
    paths["summary_text"].write_text(_render_summary_text(summary), encoding="utf-8")
    paths["session_log_excerpt"] = output_dir / "session_log_excerpt.txt"
    paths["session_log_excerpt"].write_text(_read_log_excerpt(session_log_path), encoding="utf-8")
    plot_paths = _write_plots(output_dir / "plots", samples, failures, servo_ids)
    paths.update(plot_paths)
    paths["manifest"] = output_dir / "manifest.json"
    manifest = {
        "output_dir": str(output_dir),
        "files": {key: str(path) for key, path in sorted(paths.items()) if key != "manifest"},
    }
    paths["manifest"].write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return paths


def _write_metrics_csv(path: Path, samples: list[ServoTransportSample]) -> None:
    fields = [
        "sample_index",
        "cycle_index",
        "elapsed_s",
        "mode",
        "stage",
        "just_moved",
        "servo_id",
        "position_tick",
        "current_ma",
        "current_trust",
        "voltage_mv",
        "temperature_c",
        "telemetry_age_s",
        "failure_types",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for sample in samples:
            failures_by_servo: dict[int, list[str]] = defaultdict(list)
            for failure in sample.failures:
                for servo_id in failure.get("servo_ids", []):
                    failures_by_servo[int(servo_id)].append(str(failure.get("failure_type")))
            for servo_id_text, telemetry in sample.telemetry_by_servo.items():
                servo_id = int(servo_id_text)
                writer.writerow(
                    {
                        "sample_index": sample.sample_index,
                        "cycle_index": sample.cycle_index,
                        "elapsed_s": f"{sample.elapsed_s:.6f}",
                        "mode": sample.mode,
                        "stage": sample.stage,
                        "just_moved": sample.just_moved,
                        "servo_id": servo_id,
                        "position_tick": telemetry.get("position_tick"),
                        "current_ma": telemetry.get("current_ma"),
                        "current_trust": telemetry.get("current_trust"),
                        "voltage_mv": telemetry.get("voltage_mv"),
                        "temperature_c": telemetry.get("temperature_c"),
                        "telemetry_age_s": telemetry.get("telemetry_age_s"),
                        "failure_types": "|".join(failures_by_servo.get(servo_id, [])),
                    }
                )


def _write_plots(root: Path, samples: list[ServoTransportSample], failures: list[ServoTransportFailure], servo_ids: list[int]) -> dict[str, Path]:
    root.mkdir(parents=True, exist_ok=True)
    paths = {
        "plot_packet_failures_by_servo": root / "packet_failure_counts_by_servo.svg",
        "plot_failure_type_histogram": root / "failure_type_histogram.svg",
        "plot_telemetry_freshness": root / "telemetry_freshness_over_time.svg",
        "plot_current_voltage_position": root / "current_voltage_position_timeseries.svg",
        "plot_failure_timeline": root / "per_servo_failure_timeline.svg",
        "plot_commanded_vs_measured": root / "commanded_vs_measured_position.svg",
    }
    servo_counts = Counter()
    type_counts = Counter()
    for failure in failures:
        type_counts[failure.failure_type] += 1
        for servo_id in failure.servo_ids:
            servo_counts[int(servo_id)] += 1
    _write_bar_svg(paths["plot_packet_failures_by_servo"], {str(sid): servo_counts[int(sid)] for sid in servo_ids}, "Packet failures by servo")
    _write_bar_svg(paths["plot_failure_type_histogram"], dict(type_counts), "Failure type histogram")
    freshness = {
        str(sid): [
            (sample.elapsed_s, sample.telemetry_by_servo.get(str(sid), {}).get("telemetry_age_s"))
            for sample in samples
        ]
        for sid in servo_ids
    }
    _write_line_svg(paths["plot_telemetry_freshness"], freshness, "Telemetry freshness over time", "age_s")
    current_series = {
        f"{sid}_current": [(sample.elapsed_s, sample.telemetry_by_servo.get(str(sid), {}).get("current_ma")) for sample in samples]
        for sid in servo_ids
    }
    _write_line_svg(paths["plot_current_voltage_position"], current_series, "Servo-reported current time series", "mA")
    timeline = {
        str(sid): [(failure.monotonic_time_s, idx + 1) for idx, failure in enumerate(failures) if int(sid) in failure.servo_ids]
        for sid in servo_ids
    }
    _write_line_svg(paths["plot_failure_timeline"], timeline, "Per-servo failure timeline", "failure_index")
    measured = {
        f"{sid}_measured": [(sample.elapsed_s, sample.telemetry_by_servo.get(str(sid), {}).get("position_tick")) for sample in samples]
        for sid in servo_ids
    }
    for sid in servo_ids:
        measured[f"{sid}_commanded"] = [(sample.elapsed_s, sample.commanded_goals.get(str(sid))) for sample in samples]
    _write_line_svg(paths["plot_commanded_vs_measured"], measured, "Commanded vs measured position", "ticks")
    return paths


def _write_bar_svg(path: Path, values: dict[str, int], title: str) -> None:
    items = list(values.items()) or [("none", 0)]
    width, height = 900, 420
    plot_h = 280
    max_value = max([int(v) for _, v in items] + [1])
    bar_w = max(12, int((width - 140) / max(1, len(items))))
    parts = [_svg_header(width, height), f'<text x="30" y="36" font-size="20">{_xml(title)}</text>']
    for idx, (label, value) in enumerate(items):
        x = 70 + idx * bar_w
        h = int((int(value) / max_value) * plot_h)
        y = 350 - h
        parts.append(f'<rect x="{x}" y="{y}" width="{max(8, bar_w - 8)}" height="{h}" fill="#2f6f9f"/>')
        parts.append(f'<text x="{x}" y="376" font-size="11">{_xml(label)}</text>')
        parts.append(f'<text x="{x}" y="{max(52, y - 6)}" font-size="11">{int(value)}</text>')
    parts.append("</svg>\n")
    path.write_text("\n".join(parts), encoding="utf-8")


def _write_line_svg(path: Path, series: dict[str, list[tuple[float, Any]]], title: str, y_label: str) -> None:
    width, height = 1000, 460
    left, top, right, bottom = 70, 55, 950, 380
    points_by_name: dict[str, list[tuple[float, float]]] = {}
    xs: list[float] = []
    ys: list[float] = []
    for name, points in series.items():
        clean = [(float(x), float(y)) for x, y in points if y not in (None, "")]
        points_by_name[name] = clean
        xs.extend(x for x, _ in clean)
        ys.extend(y for _, y in clean)
    min_x, max_x = (min(xs), max(xs)) if xs else (0.0, 1.0)
    min_y, max_y = (min(ys), max(ys)) if ys else (0.0, 1.0)
    if max_x <= min_x:
        max_x = min_x + 1.0
    if max_y <= min_y:
        max_y = min_y + 1.0
    colors = ["#2f6f9f", "#b23a48", "#4b8f5a", "#7a5c99", "#a66f2f", "#3d6b63", "#7f7f7f", "#c15f2e"]
    parts = [
        _svg_header(width, height),
        f'<text x="30" y="36" font-size="20">{_xml(title)}</text>',
        f'<text x="20" y="226" font-size="12" transform="rotate(-90 20,226)">{_xml(y_label)}</text>',
        f'<line x1="{left}" y1="{bottom}" x2="{right}" y2="{bottom}" stroke="#555"/>',
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{bottom}" stroke="#555"/>',
    ]
    for idx, (name, points) in enumerate(points_by_name.items()):
        if not points:
            continue
        color = colors[idx % len(colors)]
        coords = []
        for x, y in points:
            sx = left + ((x - min_x) / (max_x - min_x)) * (right - left)
            sy = bottom - ((y - min_y) / (max_y - min_y)) * (bottom - top)
            coords.append(f"{sx:.1f},{sy:.1f}")
        parts.append(f'<polyline points="{" ".join(coords)}" fill="none" stroke="{color}" stroke-width="2"/>')
        parts.append(f'<text x="{right - 180}" y="{70 + idx * 16}" font-size="11" fill="{color}">{_xml(name)}</text>')
    parts.append("</svg>\n")
    path.write_text("\n".join(parts), encoding="utf-8")


def _svg_header(width: int, height: int) -> str:
    return f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}"><rect width="100%" height="100%" fill="#fff"/>'


def _xml(value: object) -> str:
    return str(value).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def _render_summary_text(summary: dict[str, Any]) -> str:
    lines = [
        "Servo transport soak diagnostics",
        f"status={summary['status']}",
        f"success={summary['success']}",
        f"mode={summary['mode']}",
        f"sample_count={summary['sample_count']}",
        f"failure_count={summary['failure_count']}",
        f"message={summary['message']}",
        "",
        "Failure counts by type:",
    ]
    lines.extend(f"- {key}: {value}" for key, value in dict(summary["failure_counts_by_type"]).items())
    lines.extend(
        [
            "",
            "Current telemetry semantics:",
            "- current_ma is the servo Present Current register converted by configured scale.",
            "- bench supply current is not measured by this bundle.",
            "- packet-corrupted or missing current is marked suspect.",
        ]
    )
    return "\n".join(lines) + "\n"


def _read_log_excerpt(path: Path | None, *, max_lines: int = 300) -> str:
    if path is None or not Path(path).exists():
        return "Session log unavailable.\n"
    lines = Path(path).read_text(encoding="utf-8", errors="replace").splitlines()
    return "\n".join(lines[-max_lines:]) + "\n"


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run servo transport soak diagnostics.")
    parser.add_argument("--mode", choices=SOAK_MODES, default=SOAK_STATIC)
    parser.add_argument("--servo-ids", type=int, nargs="*", default=None)
    parser.add_argument("--selected-servo-id", type=int, default=None)
    parser.add_argument("--duration-s", type=float, default=30.0)
    parser.add_argument("--sample-period-s", type=float, default=0.1)
    parser.add_argument("--cycles", type=int, default=20)
    parser.add_argument("--step-ticks", type=int, default=8)
    parser.add_argument("--coordinated-delta-ticks", type=int, default=6)
    parser.add_argument("--stop-on-first-serious-failure", action="store_true")
    parser.add_argument("--assume-connected", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--session-log-path", type=Path, default=None)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    ctx = build_app_context()
    settings = ctx.settings
    servo_service = ctx.services.get("servo_service")
    openrb_client = ctx.services.get("openrb_client")
    connected_here = False
    if not args.assume_connected and not bool(getattr(servo_service, "is_connected", False)):
        if not settings.serial.openrb_port:
            print("ERROR: openrb_port is empty in the active config.")
            return 2
        openrb_client.connect(settings.serial.openrb_port, settings.serial.baudrate)
        openrb_client.prepare_for_dynamixel_use()
        servo_service.connect(settings.serial.openrb_port, settings.serial.baudrate)
        connected_here = True
    try:
        config = ServoTransportSoakConfig.from_dict(
            {
                "mode": args.mode,
                "servo_ids": args.servo_ids or settings.robot.servo_ids,
                "selected_servo_id": args.selected_servo_id,
                "duration_s": args.duration_s,
                "sample_period_s": args.sample_period_s,
                "cycles": args.cycles,
                "step_ticks": args.step_ticks,
                "coordinated_delta_ticks": args.coordinated_delta_ticks,
                "stop_on_first_serious_failure": args.stop_on_first_serious_failure or args.mode == SOAK_COORDINATED,
                "output_dir": str(args.output_dir) if args.output_dir else None,
                "session_log_path": str(args.session_log_path) if args.session_log_path else str(ctx.session_log_path or ""),
            }
        )
        result = run_servo_transport_soak(
            servo_service,
            config,
            output_root=ctx.project_root / "data" / "diagnostics" / "servo_transport",
        )
        print(f"status={result.status}")
        print(f"success={result.success}")
        print(f"mode={result.mode}")
        print(f"samples={result.sample_count}")
        print(f"failures={result.failure_count}")
        print(f"output_dir={result.output_dir}")
        print(f"summary_json={result.summary_path}")
        return 0 if result.success else 1
    finally:
        if connected_here:
            try:
                servo_service.disconnect()
            finally:
                openrb_client.disconnect()


if __name__ == "__main__":
    raise SystemExit(main())

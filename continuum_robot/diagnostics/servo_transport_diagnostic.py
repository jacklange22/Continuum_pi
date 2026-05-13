"""Focused live-read timing diagnostic for the DYNAMIXEL servo transport."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
import json
import statistics
import time
from typing import Any

from continuum_robot.app.bootstrap import build_app_context


@dataclass
class ServoTransportDiagnosticResult:
    servo_ids: list[int]
    duration_s: float
    read_rate_hz: float
    achieved_read_rate_hz: float | None
    fields: str
    sample_count: int
    success_count_by_servo: dict[str, int]
    failure_count_by_type: dict[str, int]
    mean_read_duration_ms_by_servo: dict[str, float | None]
    max_read_duration_ms_by_servo: dict[str, float | None]
    mean_packet_age_s_by_servo: dict[str, float | None]
    max_packet_age_s_by_servo: dict[str, float | None]
    position_summary_by_servo: dict[str, dict[str, int | None]]
    current_summary_by_servo: dict[str, dict[str, int | None]]
    voltage_summary_by_servo: dict[str, dict[str, int | None]]
    temperature_summary_by_servo: dict[str, dict[str, int | None]]
    recommended_next_action: str


def run_diagnostic(
    servo_service,
    *,
    servo_ids: list[int],
    duration_s: float,
    read_rate_hz: float,
    fields: str,
) -> ServoTransportDiagnosticResult:
    resolved_ids = [int(value) for value in servo_ids]
    if not resolved_ids:
        raise ValueError("servo_ids must not be empty")
    fields_name = "full" if str(fields).strip().lower() == "full" else "minimal"
    read_fn = servo_service.read_telemetry if fields_name == "full" else servo_service.read_minimal_telemetry
    period_s = 1.0 / max(1e-6, float(read_rate_hz))
    run_started = time.monotonic()
    deadline = time.monotonic() + max(0.0, float(duration_s))
    sample_count = 0
    successes: Counter[int] = Counter()
    failures: Counter[str] = Counter()
    durations: dict[int, list[float]] = defaultdict(list)
    ages: dict[int, list[float]] = defaultdict(list)
    positions: dict[int, list[int]] = defaultdict(list)
    currents: dict[int, list[int]] = defaultdict(list)
    voltages: dict[int, list[int]] = defaultdict(list)
    temperatures: dict[int, list[int]] = defaultdict(list)

    with servo_service.exclusive_bus_operation(
        owner="servo_transport_diagnostic",
        reason=f"{fields_name} telemetry timing diagnostic",
    ):
        while sample_count == 0 or time.monotonic() < deadline:
            started = time.monotonic()
            try:
                telemetry = read_fn(resolved_ids)
            except Exception as exc:
                failures[_classify_error(str(exc))] += 1
                telemetry = {}
            for servo_id in resolved_ids:
                item = telemetry.get(int(servo_id))
                if item is None:
                    failures["servo_missing"] += 1
                    continue
                failure_type = _classify_telemetry(item)
                if failure_type:
                    failures[failure_type] += 1
                else:
                    successes[int(servo_id)] += 1
                if item.read_duration_ms is not None:
                    durations[int(servo_id)].append(float(item.read_duration_ms))
                age = servo_service.telemetry_age_s(item)
                if age is not None:
                    ages[int(servo_id)].append(float(age))
                if item.present_position is not None:
                    positions[int(servo_id)].append(int(item.present_position))
                if item.present_current_ma is not None:
                    currents[int(servo_id)].append(int(item.present_current_ma))
                if item.present_voltage_mv is not None:
                    voltages[int(servo_id)].append(int(item.present_voltage_mv))
                if item.present_temperature_c is not None:
                    temperatures[int(servo_id)].append(int(item.present_temperature_c))
            sample_count += 1
            elapsed = time.monotonic() - started
            if duration_s <= 0:
                break
            sleep_s = period_s - elapsed
            if sleep_s > 0:
                time.sleep(sleep_s)

    elapsed_s = max(0.0, time.monotonic() - run_started)
    achieved_read_rate_hz = (float(sample_count) / elapsed_s) if elapsed_s > 0.0 else None
    recommendation = _recommendation(failures=failures, durations=durations, read_rate_hz=float(read_rate_hz))
    return ServoTransportDiagnosticResult(
        servo_ids=resolved_ids,
        duration_s=float(duration_s),
        read_rate_hz=float(read_rate_hz),
        achieved_read_rate_hz=achieved_read_rate_hz,
        fields=fields_name,
        sample_count=int(sample_count),
        success_count_by_servo={str(sid): int(successes[int(sid)]) for sid in resolved_ids},
        failure_count_by_type={str(k): int(v) for k, v in sorted(failures.items())},
        mean_read_duration_ms_by_servo={str(sid): _mean(durations[int(sid)]) for sid in resolved_ids},
        max_read_duration_ms_by_servo={str(sid): _max(durations[int(sid)]) for sid in resolved_ids},
        mean_packet_age_s_by_servo={str(sid): _mean(ages[int(sid)]) for sid in resolved_ids},
        max_packet_age_s_by_servo={str(sid): _max(ages[int(sid)]) for sid in resolved_ids},
        position_summary_by_servo={str(sid): _int_summary(positions[int(sid)]) for sid in resolved_ids},
        current_summary_by_servo={str(sid): _int_summary(currents[int(sid)]) for sid in resolved_ids},
        voltage_summary_by_servo={str(sid): _int_summary(voltages[int(sid)]) for sid in resolved_ids},
        temperature_summary_by_servo={str(sid): _int_summary(temperatures[int(sid)]) for sid in resolved_ids},
        recommended_next_action=recommendation,
    )


def _classify_telemetry(item) -> str | None:
    detail = " | ".join(
        str(value)
        for value in (item.telemetry_error, item.identity_error, item.hardware_error)
        if value
    )
    if detail:
        return _classify_error(detail)
    if item.present_position is None:
        return "missing_position"
    return None


def _classify_error(message: str) -> str:
    lowered = str(message).lower()
    if "incorrect status packet" in lowered:
        return "incorrect_status_packet"
    if "no status packet" in lowered:
        return "no_status_packet"
    if "timeout" in lowered:
        return "timeout"
    if "missing" in lowered:
        return "servo_missing"
    if "hardware" in lowered or "status=0x" in lowered:
        return "hardware_error"
    return "tx_rx_error"


def _mean(values: list[float]) -> float | None:
    return float(statistics.fmean(values)) if values else None


def _max(values: list[float]) -> float | None:
    return float(max(values)) if values else None


def _int_summary(values: list[int]) -> dict[str, int | None]:
    return {
        "min": min(values) if values else None,
        "max": max(values) if values else None,
        "last": values[-1] if values else None,
    }


def _recommendation(*, failures: Counter[str], durations: dict[int, list[float]], read_rate_hz: float) -> str:
    if failures:
        return (
            "Resolve reported packet/status failures first. If failures cluster at higher read rates, rerun with "
            "--fields minimal and lower --read-rate-hz to find a stable health-check cadence."
        )
    max_duration = max((max(values) for values in durations.values() if values), default=0.0)
    if max_duration > (1000.0 / max(1e-6, read_rate_hz)):
        return "Read duration exceeds the requested period; lower read-rate-hz or use --fields minimal at 57600 baud."
    return "Transport read timing is stable for this diagnostic profile."


def _parse_servo_ids(raw: str) -> list[int]:
    return [int(part.strip()) for part in str(raw).split(",") if part.strip()]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run a focused DYNAMIXEL servo transport timing diagnostic.")
    parser.add_argument("--port", default=None, help="OpenRB/DYNAMIXEL serial port. Defaults to config/system*.yaml.")
    parser.add_argument("--baud", type=int, default=None, help="DYNAMIXEL baudrate. Defaults to config/system*.yaml.")
    parser.add_argument("--servo-ids", default="5,6,7,8", help="Comma-separated servo IDs, e.g. 5,6,7,8.")
    parser.add_argument("--duration", type=float, default=10.0, help="Diagnostic duration in seconds.")
    parser.add_argument("--read-rate-hz", type=float, default=5.0, help="Requested live read rate.")
    parser.add_argument("--fields", choices=("minimal", "full"), default="minimal", help="Telemetry field profile.")
    args = parser.parse_args(argv)

    context = build_app_context()
    port = args.port if args.port is not None else context.settings.serial.openrb_port
    baud = int(args.baud if args.baud is not None else context.settings.serial.baudrate)
    servo_ids = _parse_servo_ids(args.servo_ids)
    servo_service = context.services.get("servo_service")
    servo_service.connect(str(port), baud)
    try:
        result = run_diagnostic(
            servo_service,
            servo_ids=servo_ids,
            duration_s=float(args.duration),
            read_rate_hz=float(args.read_rate_hz),
            fields=str(args.fields),
        )
    finally:
        servo_service.disconnect(torque_off=False, requested_by_operator=False, reason="servo_transport_diagnostic_complete")
    print(json.dumps(asdict(result), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Quick single-segment current/voltage sweep for overcurrent diagnosis.

This sweep does not require tracker connectivity.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from datetime import datetime
import json
from pathlib import Path
import time
from typing import Any

from continuum_robot.app.bootstrap import build_app_context


@dataclass
class SweepPoint:
    amplitude_cm: float
    command_cm: list[float]
    peak_abs_current_ma_by_servo: dict[str, int | None]
    voltage_mv_by_servo: dict[str, int | None]


@dataclass
class OvercurrentSweepSummary:
    schema_version: str
    generated_at_local: str
    port: str
    baud: int
    servo_ids: list[int]
    amplitudes_cm: list[float]
    points: list[dict[str, Any]]
    limiting_servo_id: int | None
    limiting_servo_peak_abs_current_ma: int | None
    recommendation: str


def _parse_amplitudes(raw: str) -> list[float]:
    values = [float(part.strip()) for part in str(raw).split(",") if part.strip()]
    return [float(value) for value in values if value >= 0.0]


def _default_command_for_amplitude(amplitude_cm: float) -> list[float]:
    # Single-segment axis-A antagonistic pair command.
    a = float(amplitude_cm)
    return [a, 0.0, -a, 0.0]


def run_sweep(
    servo_service,
    *,
    servo_ids: list[int],
    amplitudes_cm: list[float],
    settle_s: float,
) -> OvercurrentSweepSummary:
    ids = [int(value) for value in servo_ids]
    if len(ids) != 4:
        raise ValueError(f"overcurrent_sweep requires exactly 4 servo IDs; got {ids}.")
    startup = servo_service.resolve_startup_reference_ticks(ids)
    neutral_ticks = [int(startup.ticks_by_servo[sid]) for sid in ids]
    points: list[SweepPoint] = []

    with servo_service.exclusive_bus_operation(owner="overcurrent_sweep", reason="single-segment current/voltage diagnosis"):
        # Start from neutral.
        servo_service.command_displacement(
            tendon_displacements_cm=[0.0, 0.0, 0.0, 0.0],
            neutral_ticks=neutral_ticks,
            servo_ids=ids,
            motion_workflow="experiment_motion",
            telemetry_retry_count=0,
            allow_recovered_packet_errors=False,
        )
        for amplitude in amplitudes_cm:
            command = _default_command_for_amplitude(float(amplitude))
            result = servo_service.command_displacement(
                tendon_displacements_cm=list(command),
                neutral_ticks=neutral_ticks,
                servo_ids=ids,
                motion_workflow="experiment_motion",
                telemetry_retry_count=0,
                allow_recovered_packet_errors=False,
            )
            if float(settle_s) > 0.0:
                time.sleep(float(settle_s))
            telemetry = dict(result.telemetry_by_id or {})
            peak_abs_current = {
                str(sid): (
                    abs(int(telemetry[sid].present_current_ma))
                    if sid in telemetry and telemetry[sid].present_current_ma is not None
                    else None
                )
                for sid in ids
            }
            voltage_mv = {
                str(sid): (
                    int(telemetry[sid].present_voltage_mv)
                    if sid in telemetry and telemetry[sid].present_voltage_mv is not None
                    else None
                )
                for sid in ids
            }
            points.append(
                SweepPoint(
                    amplitude_cm=float(amplitude),
                    command_cm=list(command),
                    peak_abs_current_ma_by_servo=peak_abs_current,
                    voltage_mv_by_servo=voltage_mv,
                )
            )

        # Return to neutral at end.
        servo_service.command_displacement(
            tendon_displacements_cm=[0.0, 0.0, 0.0, 0.0],
            neutral_ticks=neutral_ticks,
            servo_ids=ids,
            motion_workflow="experiment_motion",
            telemetry_retry_count=0,
            allow_recovered_packet_errors=False,
        )

    limiting_servo_id: int | None = None
    limiting_servo_peak: int | None = None
    for point in points:
        for sid, peak in point.peak_abs_current_ma_by_servo.items():
            if peak is None:
                continue
            if limiting_servo_peak is None or int(peak) > int(limiting_servo_peak):
                limiting_servo_peak = int(peak)
                limiting_servo_id = int(sid)

    if limiting_servo_id is None:
        recommendation = "No current samples were available; verify telemetry/current fields and rerun."
    else:
        recommendation = (
            f"Servo {limiting_servo_id} is current-limiting in this sweep (peak |I|={limiting_servo_peak} mA). "
            "Inspect neutral/startup alignment, tendon routing/friction, and transition aggressiveness before larger amplitudes."
        )

    return OvercurrentSweepSummary(
        schema_version="overcurrent_sweep_v1",
        generated_at_local=datetime.now().isoformat(timespec="seconds"),
        port="",
        baud=0,
        servo_ids=ids,
        amplitudes_cm=[float(value) for value in amplitudes_cm],
        points=[asdict(point) for point in points],
        limiting_servo_id=limiting_servo_id,
        limiting_servo_peak_abs_current_ma=limiting_servo_peak,
        recommendation=recommendation,
    )


def _render_text(summary: OvercurrentSweepSummary) -> str:
    lines = [
        "Overcurrent Sweep Summary",
        f"Generated: {summary.generated_at_local}",
        f"Port: {summary.port}",
        f"Baud: {summary.baud}",
        f"Servo IDs: {summary.servo_ids}",
        f"Amplitudes (cm): {summary.amplitudes_cm}",
        "",
    ]
    for point in summary.points:
        lines.append(
            f"amp={point['amplitude_cm']:.3f} cm | peak_abs_current_ma={point['peak_abs_current_ma_by_servo']} | voltage_mv={point['voltage_mv_by_servo']}"
        )
    lines.extend([
        "",
        f"Limiting servo: {summary.limiting_servo_id}",
        f"Limiting peak |I| (mA): {summary.limiting_servo_peak_abs_current_ma}",
        f"Recommendation: {summary.recommendation}",
    ])
    return "\n".join(lines).strip() + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run a quick single-segment overcurrent sweep without tracker.")
    parser.add_argument("--port", default=None, help="OpenRB/DYNAMIXEL serial port. Defaults to config/system*.yaml.")
    parser.add_argument("--baud", type=int, default=None, help="DYNAMIXEL baudrate. Defaults to config/system*.yaml.")
    parser.add_argument("--servo-ids", default="1,2,3,4", help="Comma-separated 4-servo single-segment IDs.")
    parser.add_argument("--amplitudes-cm", default="0.10,0.25,0.50,0.75", help="Comma-separated command amplitudes.")
    parser.add_argument("--settle-s", type=float, default=0.1, help="Post-command settle time before capture.")
    parser.add_argument("--output-dir", default="data/diagnostics/overcurrent_sweep", help="Output root directory.")
    args = parser.parse_args(argv)

    context = build_app_context()
    servo_service = context.services.get("servo_service")
    port = args.port if args.port is not None else context.settings.serial.openrb_port
    baud = int(args.baud if args.baud is not None else context.settings.serial.baudrate)
    servo_ids = [int(part.strip()) for part in str(args.servo_ids).split(",") if part.strip()]
    amplitudes = _parse_amplitudes(args.amplitudes_cm)
    if not amplitudes:
        raise SystemExit("No valid amplitudes were provided.")

    output_root = Path(args.output_dir)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = output_root / stamp
    run_dir.mkdir(parents=True, exist_ok=True)

    servo_service.connect(str(port), baud)
    try:
        summary = run_sweep(
            servo_service,
            servo_ids=servo_ids,
            amplitudes_cm=amplitudes,
            settle_s=float(args.settle_s),
        )
    finally:
        servo_service.disconnect(torque_off=False, requested_by_operator=False, reason="overcurrent_sweep_complete")

    summary.port = str(port)
    summary.baud = int(baud)
    json_path = run_dir / "overcurrent_sweep_summary.json"
    txt_path = run_dir / "overcurrent_sweep_summary.txt"
    json_path.write_text(json.dumps(asdict(summary), indent=2, sort_keys=True), encoding="utf-8")
    txt_path.write_text(_render_text(summary), encoding="utf-8")
    print(str(json_path))
    print(str(txt_path))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Servo telemetry policy summaries and practical benchmark helpers."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
import statistics
import time
from typing import Any

from continuum_robot.app.bootstrap import build_app_context
from continuum_robot.experiments.dataset_io import canonical_timestamped_path
from continuum_robot.hardware.dxl_bus import DxlBusConfig


DEFAULT_SERVO_FULL_REFRESH_DIVISOR = 4
DEFAULT_SYSTEM_SUMMARY_REFRESH_DIVISOR = 4

LIVE_REGISTER_KEYS = (
    "operating_mode",
    "torque_enable",
    "present_position",
    "present_current",
    "present_input_voltage",
    "present_temperature",
    "hardware_error_status",
)

FULL_ONLY_REGISTER_KEYS = (
    "servo_id",
    "model_number",
    "firmware_version",
    "current_limit",
    "max_position_limit",
    "min_position_limit",
    "bus_watchdog",
)


@dataclass(frozen=True)
class TelemetryProfileSpec:
    """One register-selection profile used by the live bus path."""

    name: str
    register_keys: tuple[str, ...]
    include_identity: bool
    include_limits: bool
    include_reported_id: bool

    @property
    def register_transaction_count_per_servo(self) -> int:
        return len(self.register_keys)


@dataclass(frozen=True)
class TelemetryGuiPolicy:
    """Compact operator-facing summary of the current GUI telemetry policy."""

    gui_refresh_target_hz: float
    servos_selected_refresh_target_hz: float
    servos_full_refresh_target_hz: float
    system_summary_refresh_target_hz: float
    telemetry_stale_after_s: float
    live_profile: TelemetryProfileSpec
    full_profile: TelemetryProfileSpec
    primary_limiter: str
    throughput_limited: bool
    cadence_summary: str
    field_summary: str
    bottleneck_summary: str


@dataclass(frozen=True)
class TelemetryBenchmarkResult:
    """Measured throughput for one servo telemetry profile."""

    profile_name: str
    servo_ids: list[int]
    iterations: int
    baudrate: int | None
    register_keys: list[str]
    register_transaction_count_per_servo: int
    total_duration_s: float
    mean_loop_s: float
    median_loop_s: float
    p95_loop_s: float
    effective_loop_hz: float
    aggregate_servo_samples_hz: float
    telemetry_fields: list[str]


def live_telemetry_profile() -> TelemetryProfileSpec:
    """Return the lightweight runtime profile used by live telemetry polling."""
    return TelemetryProfileSpec(
        name="live",
        register_keys=tuple(LIVE_REGISTER_KEYS),
        include_identity=False,
        include_limits=False,
        include_reported_id=False,
    )


def full_telemetry_profile() -> TelemetryProfileSpec:
    """Return the heavier profile used by readiness and safety checks."""
    return TelemetryProfileSpec(
        name="full",
        register_keys=tuple((*FULL_ONLY_REGISTER_KEYS, *LIVE_REGISTER_KEYS)),
        include_identity=True,
        include_limits=True,
        include_reported_id=True,
    )


def register_descriptions(config: DxlBusConfig | None = None, *, profile: str = "live") -> list[str]:
    """Return human-readable register descriptions for the requested profile."""
    control_table = dict((config.control_table if config is not None else DxlBusConfig().control_table) or {})
    spec = full_telemetry_profile() if str(profile).strip().lower() == "full" else live_telemetry_profile()
    descriptions: list[str] = []
    for key in spec.register_keys:
        address = control_table.get(str(key))
        if address is None:
            descriptions.append(str(key))
        else:
            descriptions.append(f"{key}@0x{int(address):02X}")
    return descriptions


def build_telemetry_gui_policy(
    *,
    baudrate: int | None,
    poll_rate_hz: int,
    telemetry_stale_after_s: float,
    servo_full_refresh_divisor: int = DEFAULT_SERVO_FULL_REFRESH_DIVISOR,
    system_summary_refresh_divisor: int = DEFAULT_SYSTEM_SUMMARY_REFRESH_DIVISOR,
) -> TelemetryGuiPolicy:
    """Return a grounded summary of the current GUI servo polling policy."""
    gui_refresh_target_hz = max(1.0, float(poll_rate_hz))
    full_divisor = max(1, int(servo_full_refresh_divisor))
    system_divisor = max(1, int(system_summary_refresh_divisor))
    live_profile = live_telemetry_profile()
    full_profile = full_telemetry_profile()
    servos_selected_refresh_target_hz = gui_refresh_target_hz
    servos_full_refresh_target_hz = gui_refresh_target_hz / float(full_divisor)
    system_summary_refresh_target_hz = gui_refresh_target_hz / float(system_divisor)

    resolved_baudrate = None if baudrate in (None, 0) else int(baudrate)
    if resolved_baudrate is not None and resolved_baudrate <= 57600:
        primary_limiter = "baudrate"
        throughput_limited = True
        bottleneck_summary = (
            "Current servo telemetry is most likely limited by the configured 57600-style DYNAMIXEL baudrate. "
            "The OpenRB-150 and XC330 path supports 1 Mbps; raising baud will matter more than increasing the GUI timer."
        )
    else:
        primary_limiter = "register_fanout"
        throughput_limited = True
        bottleneck_summary = (
            "Current telemetry is dominated by many one-register Tx/Rx calls per servo. "
            "The next clear optimization after baudrate is reducing per-servo register fanout and avoiding repeated scan pings."
        )

    cadence_summary = (
        f"GUI {gui_refresh_target_hz:.1f} Hz | System auto servo summary {system_summary_refresh_target_hz:.1f} Hz "
        f"(no background scan) | Servos selected-servo target {servos_selected_refresh_target_hz:.1f} Hz | "
        f"Servos all-servo target {servos_full_refresh_target_hz:.1f} Hz"
    )
    field_summary = (
        "Live reads: operating mode, torque enable, present position, present current, input voltage, "
        "temperature, hardware error. Full reads add servo ID, model number, firmware version, current limit, "
        "min/max position limits, and bus watchdog."
    )
    return TelemetryGuiPolicy(
        gui_refresh_target_hz=gui_refresh_target_hz,
        servos_selected_refresh_target_hz=servos_selected_refresh_target_hz,
        servos_full_refresh_target_hz=servos_full_refresh_target_hz,
        system_summary_refresh_target_hz=system_summary_refresh_target_hz,
        telemetry_stale_after_s=float(telemetry_stale_after_s),
        live_profile=live_profile,
        full_profile=full_profile,
        primary_limiter=primary_limiter,
        throughput_limited=throughput_limited,
        cadence_summary=cadence_summary,
        field_summary=field_summary,
        bottleneck_summary=(
            f"{bottleneck_summary} "
            f"Current code issues {live_profile.register_transaction_count_per_servo} register reads/servo for live polling "
            f"and {full_profile.register_transaction_count_per_servo} reads/servo for full readiness."
        ),
    )


def benchmark_telemetry_profile(
    servo_service,
    servo_ids: list[int],
    *,
    profile: str = "live",
    iterations: int = 100,
) -> TelemetryBenchmarkResult:
    """Measure practical telemetry read throughput for the requested profile."""
    resolved_ids = [int(servo_id) for servo_id in servo_ids]
    if not resolved_ids:
        raise ValueError("servo_ids must not be empty")
    if int(iterations) <= 0:
        raise ValueError("iterations must be positive")

    profile_name = str(profile).strip().lower()
    if profile_name == "full":
        spec = full_telemetry_profile()
        read_fn = servo_service.read_telemetry
    else:
        spec = live_telemetry_profile()
        read_fn = servo_service.read_live_telemetry
        profile_name = "live"

    durations_s: list[float] = []
    for _ in range(int(iterations)):
        started = time.perf_counter()
        read_fn(resolved_ids)
        durations_s.append(max(0.0, time.perf_counter() - started))

    total_duration_s = sum(durations_s)
    if total_duration_s <= 0.0:
        raise RuntimeError("Telemetry benchmark duration was zero.")
    ordered = sorted(durations_s)
    p95_index = min(len(ordered) - 1, max(0, int(round(0.95 * (len(ordered) - 1)))))
    effective_loop_hz = float(len(durations_s)) / float(total_duration_s)
    aggregate_servo_samples_hz = float(len(durations_s) * len(resolved_ids)) / float(total_duration_s)
    return TelemetryBenchmarkResult(
        profile_name=profile_name,
        servo_ids=list(resolved_ids),
        iterations=int(iterations),
        baudrate=getattr(getattr(servo_service, "dxl_bus", None), "baudrate", None),
        register_keys=list(spec.register_keys),
        register_transaction_count_per_servo=spec.register_transaction_count_per_servo,
        total_duration_s=float(total_duration_s),
        mean_loop_s=float(statistics.fmean(durations_s)),
        median_loop_s=float(statistics.median(durations_s)),
        p95_loop_s=float(ordered[p95_index]),
        effective_loop_hz=float(effective_loop_hz),
        aggregate_servo_samples_hz=float(aggregate_servo_samples_hz),
        telemetry_fields=list(spec.register_keys),
    )


def write_benchmark_outputs(
    *,
    output_dir: Path,
    results: list[TelemetryBenchmarkResult],
    metadata: dict[str, Any] | None = None,
) -> dict[str, Path]:
    """Persist benchmark summaries to a canonical diagnostic folder."""
    payload = {
        "metadata": dict(metadata or {}),
        "results": [asdict(result) for result in results],
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    text_lines = [
        "Servo telemetry benchmark",
        f"written_at_utc={datetime.now(timezone.utc).isoformat()}",
    ]
    for key, value in dict(metadata or {}).items():
        text_lines.append(f"{key}={value}")
    for result in results:
        text_lines.extend(
            [
                "",
                f"profile={result.profile_name}",
                f"servo_ids={result.servo_ids}",
                f"iterations={result.iterations}",
                f"baudrate={result.baudrate}",
                f"register_transaction_count_per_servo={result.register_transaction_count_per_servo}",
                f"effective_loop_hz={result.effective_loop_hz:.3f}",
                f"aggregate_servo_samples_hz={result.aggregate_servo_samples_hz:.3f}",
                f"mean_loop_ms={result.mean_loop_s * 1000.0:.3f}",
                f"median_loop_ms={result.median_loop_s * 1000.0:.3f}",
                f"p95_loop_ms={result.p95_loop_s * 1000.0:.3f}",
                "telemetry_fields=" + ", ".join(result.telemetry_fields),
            ]
        )
    summary_text_path = output_dir / "summary.txt"
    summary_text_path.write_text("\n".join(text_lines) + "\n", encoding="utf-8")
    return {
        "summary_json": summary_path,
        "summary_text": summary_text_path,
    }


def build_arg_parser() -> argparse.ArgumentParser:
    """Build the CLI parser for real-bus telemetry benchmarks."""
    parser = argparse.ArgumentParser(description="Benchmark servo telemetry throughput on the active OpenRB/XC330 path")
    parser.add_argument("--profile", choices=["live", "full", "both"], default="both")
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--servo-ids", type=int, nargs="*", default=None)
    parser.add_argument("--assume-connected", action="store_true", help="Do not connect OpenRB/DYNAMIXEL inside the CLI")
    parser.add_argument("--output-dir", type=Path, default=None, help="Optional output directory override")
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run the telemetry benchmark CLI against the configured runtime path."""
    args = build_arg_parser().parse_args(argv)
    ctx = build_app_context()
    settings = ctx.settings
    servo_service = ctx.services.get("servo_service")
    openrb_client = ctx.services.get("openrb_client")
    servo_ids = [int(value) for value in (args.servo_ids or settings.robot.servo_ids or [])]
    if not servo_ids:
        print("ERROR: no servo IDs are configured or supplied.")
        return 2

    connected_here = False
    if not bool(args.assume_connected) and not bool(getattr(servo_service, "is_connected", False)):
        if not settings.serial.openrb_port:
            print("ERROR: openrb_port is empty in the active config.")
            return 2
        openrb_client.connect(settings.serial.openrb_port, settings.serial.baudrate)
        openrb_client.prepare_for_dynamixel_use()
        servo_service.connect(settings.serial.openrb_port, settings.serial.baudrate)
        connected_here = True

    try:
        profiles = ["live", "full"] if args.profile == "both" else [str(args.profile)]
        results = [
            benchmark_telemetry_profile(
                servo_service,
                servo_ids,
                profile=profile_name,
                iterations=int(args.iterations),
            )
            for profile_name in profiles
        ]
        run_dir = args.output_dir
        if run_dir is None:
            run_dir = canonical_timestamped_path(
                ctx.project_root / "data" / "diagnostics" / "servo_telemetry",
                f"servo_telemetry_{'_'.join(profiles)}_{len(servo_ids)}servos",
            )
        outputs = write_benchmark_outputs(
            output_dir=run_dir,
            results=results,
            metadata={
                "openrb_port": settings.serial.openrb_port,
                "baudrate": settings.serial.baudrate,
                "servo_ids": servo_ids,
                "profile": args.profile,
                "iterations": int(args.iterations),
            },
        )
        for result in results:
            print(f"profile={result.profile_name}")
            print(f"servo_ids={result.servo_ids}")
            print(f"iterations={result.iterations}")
            print(f"baudrate={result.baudrate}")
            print(f"register_transaction_count_per_servo={result.register_transaction_count_per_servo}")
            print(f"effective_loop_hz={result.effective_loop_hz:.3f}")
            print(f"aggregate_servo_samples_hz={result.aggregate_servo_samples_hz:.3f}")
            print(f"mean_loop_ms={result.mean_loop_s * 1000.0:.3f}")
            print(f"median_loop_ms={result.median_loop_s * 1000.0:.3f}")
            print(f"p95_loop_ms={result.p95_loop_s * 1000.0:.3f}")
            print("telemetry_fields=" + ",".join(result.telemetry_fields))
        print(f"summary_json={outputs['summary_json']}")
        print(f"summary_text={outputs['summary_text']}")
        return 0
    finally:
        if connected_here:
            try:
                servo_service.disconnect()
            finally:
                openrb_client.disconnect()


if __name__ == "__main__":
    raise SystemExit(main())

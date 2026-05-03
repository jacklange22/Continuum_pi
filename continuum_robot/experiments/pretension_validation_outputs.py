"""Extra artifacts for the pretension-validation experiment."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
import struct
from typing import Any
import zlib


@dataclass(frozen=True)
class PretensionTracePoint:
    """Normalized one-sample trace row for pretension validation."""

    monotonic_time_s: float
    phase: str
    run_state: str
    servo_id: int
    commanded_position_ticks: int | None
    current_position_ticks: int | None
    travel_from_untensioned_ticks: int | None
    travel_from_untensioned_mm: float | None
    raw_current_ma: float | None
    filtered_current_ma: float | None
    baseline_current_ma: float | None
    effective_trigger_current_ma: float | None
    hard_current_stop_ma: float | None
    trigger_met: bool
    stop_reason: str | None
    tracker_displacement_mm: float | None
    tracker_metric_frame: str | None


def write_pretension_validation_outputs(
    *,
    output_dir: Path,
    metadata,
    summary,
    samples,
) -> dict[str, Path]:
    """Write stable plot/text artifacts for one pretension-validation run."""
    output_dir = Path(output_dir)
    metrics = summary.experiment_metrics if isinstance(summary.experiment_metrics, dict) else {}
    mode = str(metrics.get("mode", "single_servo_trace") or "single_servo_trace").strip().lower()
    if mode in {"single_segment_staged", "staged", "four_servo_staged"}:
        return _write_staged_pretension_outputs(
            output_dir=output_dir,
            metadata=metadata,
            summary=summary,
            metrics=metrics,
        )
    plot_path = output_dir / "pretension_response.png"
    summary_text_path = output_dir / "pretension_summary.txt"
    metrics_csv_path = output_dir / "metrics.csv"
    trace_points = extract_pretension_trace_points(samples)
    _write_single_servo_metrics_csv(metrics_csv_path=metrics_csv_path, metrics=metrics)
    _write_summary_text(
        summary_text_path=summary_text_path,
        metadata=metadata,
        summary=summary,
        trace_points=trace_points,
    )
    _write_response_plot(
        plot_path=plot_path,
        trace_points=trace_points,
        metrics=(summary.experiment_metrics if isinstance(summary.experiment_metrics, dict) else {}),
    )
    return {
        "plot_path": plot_path,
        "summary_text_path": summary_text_path,
        "metrics_csv_path": metrics_csv_path,
    }


def _write_staged_pretension_outputs(
    *,
    output_dir: Path,
    metadata,
    summary,
    metrics: dict[str, Any],
) -> dict[str, Path]:
    summary_text_path = output_dir / "pretension_summary.txt"
    metrics_csv_path = output_dir / "metrics.csv"
    current_vs_position_path = output_dir / "pretension_current_vs_position.png"
    tendon_vs_tip_path = output_dir / "pretension_tendon_displacement_vs_tip_xy.png"
    tendon_vs_current_path = output_dir / "pretension_tendon_displacement_vs_current.png"
    current_vs_tip_error_path = output_dir / "pretension_current_vs_tip_error.png"
    balance_over_stages_path = output_dir / "pretension_balance_over_stages.png"
    tip_xy_path = output_dir / "pretension_tip_xy_path.png"
    final_tip_scatter_path = output_dir / "pretension_final_tip_xy_scatter.png"
    final_current_dist_path = output_dir / "pretension_final_current_distribution.png"
    final_position_dist_path = output_dir / "pretension_final_position_distribution.png"
    quality_dist_path = output_dir / "pretension_quality_score_distribution.png"
    repeatability_path = output_dir / "pretension_repeatability_summary.png"
    response_alias_path = output_dir / "pretension_response.png"

    run_rows = list(metrics.get("run_rows") or [])
    trace_rows = list(metrics.get("trace_rows") or [])
    _write_staged_metrics_csv(metrics_csv_path=metrics_csv_path, run_rows=run_rows)
    _write_staged_summary_text(
        summary_text_path=summary_text_path,
        metadata=metadata,
        summary=summary,
        metrics=metrics,
    )
    _write_staged_current_vs_position_plot(current_vs_position_path=current_vs_position_path, trace_rows=trace_rows)
    _write_staged_tendon_vs_tip_plot(plot_path=tendon_vs_tip_path, trace_rows=trace_rows)
    _write_staged_tendon_vs_current_plot(plot_path=tendon_vs_current_path, trace_rows=trace_rows)
    _write_staged_current_vs_tip_error_plot(plot_path=current_vs_tip_error_path, trace_rows=trace_rows)
    _write_staged_balance_over_stages_plot(plot_path=balance_over_stages_path, trace_rows=trace_rows, run_rows=run_rows)
    _write_staged_tip_xy_plot(tip_xy_path=tip_xy_path, run_rows=run_rows)
    _write_staged_final_tip_scatter_plot(plot_path=final_tip_scatter_path, run_rows=run_rows)
    _write_staged_final_current_distribution_plot(final_current_dist_path=final_current_dist_path, run_rows=run_rows)
    _write_staged_final_position_distribution_plot(plot_path=final_position_dist_path, run_rows=run_rows)
    _write_staged_quality_distribution_plot(plot_path=quality_dist_path, run_rows=run_rows)
    _write_staged_repeatability_plot(repeatability_path=repeatability_path, metrics=metrics)
    if current_vs_position_path.exists():
        response_alias_path.write_bytes(current_vs_position_path.read_bytes())
    return {
        "summary_text_path": summary_text_path,
        "metrics_csv_path": metrics_csv_path,
        "current_vs_position_plot_path": current_vs_position_path,
        "tendon_displacement_vs_tip_xy_plot_path": tendon_vs_tip_path,
        "tendon_displacement_vs_current_plot_path": tendon_vs_current_path,
        "current_vs_tip_error_plot_path": current_vs_tip_error_path,
        "balance_over_stages_plot_path": balance_over_stages_path,
        "tip_xy_path_plot_path": tip_xy_path,
        "final_tip_xy_scatter_plot_path": final_tip_scatter_path,
        "final_current_distribution_plot_path": final_current_dist_path,
        "final_position_distribution_plot_path": final_position_dist_path,
        "quality_score_distribution_plot_path": quality_dist_path,
        "repeatability_plot_path": repeatability_path,
        "plot_path": response_alias_path,
    }


def extract_pretension_trace_points(samples) -> list[PretensionTracePoint]:
    """Return normalized trace rows from canonical experiment samples."""
    rows: list[PretensionTracePoint] = []
    for sample in samples:
        extra = dict(getattr(sample, "extra", {}) or {})
        phase = str(getattr(sample, "phase", "") or "")
        servo_id = extra.get("servo_id")
        if servo_id is None:
            servo_id = getattr(sample, "commanded_motor_values", {}).get("servo_id")
        if servo_id in (None, ""):
            continue
        rows.append(
            PretensionTracePoint(
                monotonic_time_s=float(getattr(sample, "monotonic_time_s", 0.0) or 0.0),
                phase=phase,
                run_state=str(extra.get("run_state", phase or "unknown")),
                servo_id=int(servo_id),
                commanded_position_ticks=_as_int(extra.get("commanded_position_ticks")),
                current_position_ticks=_as_int(extra.get("current_position_ticks")),
                travel_from_untensioned_ticks=_as_int(extra.get("travel_from_untensioned_ticks")),
                travel_from_untensioned_mm=_as_float(extra.get("travel_from_untensioned_mm")),
                raw_current_ma=_as_float(extra.get("raw_current_ma")),
                filtered_current_ma=_as_float(extra.get("filtered_current_ma")),
                baseline_current_ma=_as_float(extra.get("baseline_current_ma")),
                effective_trigger_current_ma=_as_float(extra.get("effective_trigger_current_ma")),
                hard_current_stop_ma=_as_float(extra.get("hard_current_stop_ma")),
                trigger_met=bool(extra.get("trigger_met", False)),
                stop_reason=(
                    str(extra.get("stop_reason"))
                    if extra.get("stop_reason") not in (None, "")
                    else None
                ),
                tracker_displacement_mm=_as_float(extra.get("tracker_displacement_mm")),
                tracker_metric_frame=(
                    str(extra.get("tracker_metric_frame"))
                    if extra.get("tracker_metric_frame") not in (None, "")
                    else None
                ),
            )
        )
    rows.sort(key=lambda row: (row.servo_id, row.monotonic_time_s, row.phase))
    return rows


def _write_single_servo_metrics_csv(*, metrics_csv_path: Path, metrics: dict[str, Any]) -> None:
    fieldnames = [
        "servo_id",
        "accepted",
        "status",
        "stop_reason",
        "final_position_tick",
        "travel_used_ticks",
        "travel_used_mm",
        "baseline_current_ma",
        "effective_trigger_current_ma",
        "trigger_current_ma",
        "hard_current_stop_ma",
        "max_observed_current_ma",
        "max_observed_filtered_current_ma",
        "max_observed_displacement_mm",
        "trigger_displacement_mm",
        "tracker_metric_frame",
        "tracker_metric_sample_count",
        "trace_sample_count",
    ]
    with metrics_csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerow({key: metrics.get(key) for key in fieldnames})


def _write_staged_metrics_csv(*, metrics_csv_path: Path, run_rows: list[dict[str, Any]]) -> None:
    fieldnames = [
        "run_index",
        "accepted",
        "reject_reasons",
        "servo_id",
        "baseline_current_ma",
        "final_current_ma",
        "signed_raw_current_ma",
        "current_above_baseline_ma",
        "load_proxy_current_ma",
        "start_position_ticks",
        "startup_reference_ticks",
        "final_position_ticks",
        "tendon_displacement_mm",
        "travel_used_ticks",
        "travel_used_mm",
        "stop_reason",
        "startup_source",
        "trust_status",
        "packet_retry_count",
        "telemetry_event_counts",
        "quality_flags",
        "load_balance_error_ma",
        "pair_balance_error_ma",
        "final_tip_x_mm",
        "final_tip_y_mm",
        "final_tip_z_mm",
        "final_tip_xy_offset_mm",
        "settle_tip_drift_mm",
        "quality_score_0_100",
        "quality_components",
        "tip_centering_status",
        "equalization_status",
    ]
    with metrics_csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for run in run_rows:
            servo_map = run.get("final_position_ticks_by_servo") or {}
            servo_ids = sorted(int(key) for key in servo_map.keys())
            tip_xyz = run.get("final_tip_xyz_mm") if isinstance(run.get("final_tip_xyz_mm"), list) else None
            for servo_id in servo_ids:
                baseline_map = dict(run.get("baseline_current_ma_by_servo") or {})
                final_current_map = dict(run.get("final_current_ma_by_servo") or {})
                current_above_map = dict(run.get("current_above_baseline_ma_by_servo") or {})
                load_proxy_map = dict(run.get("load_proxy_current_ma_by_servo") or current_above_map)
                start_map = dict(run.get("start_position_ticks_by_servo") or {})
                startup_reference_map = dict(run.get("startup_reference_ticks_by_servo") or {})
                final_map = dict(run.get("final_position_ticks_by_servo") or {})
                tendon_displacement_map = dict(run.get("final_tendon_displacement_mm_by_servo") or {})
                stop_map = dict(run.get("stop_reason_by_servo") or {})
                start_tick = start_map.get(str(int(servo_id)))
                final_tick = final_map.get(str(int(servo_id)))
                writer.writerow(
                    {
                        "run_index": run.get("run_index"),
                        "accepted": bool(run.get("accepted")),
                        "reject_reasons": ",".join(str(value) for value in (run.get("reject_reasons") or [])),
                        "servo_id": int(servo_id),
                        "baseline_current_ma": baseline_map.get(str(int(servo_id))),
                        "final_current_ma": final_current_map.get(str(int(servo_id))),
                        "signed_raw_current_ma": final_current_map.get(str(int(servo_id))),
                        "current_above_baseline_ma": current_above_map.get(str(int(servo_id))),
                        "load_proxy_current_ma": load_proxy_map.get(str(int(servo_id))),
                        "start_position_ticks": start_tick,
                        "startup_reference_ticks": startup_reference_map.get(str(int(servo_id))),
                        "final_position_ticks": final_tick,
                        "tendon_displacement_mm": tendon_displacement_map.get(str(int(servo_id))),
                        "travel_used_ticks": (
                            None
                            if start_tick is None or final_tick is None
                            else int(start_tick) - int(final_tick)
                        ),
                        "travel_used_mm": None,
                        "stop_reason": stop_map.get(str(int(servo_id))),
                        "startup_source": run.get("startup_source"),
                        "trust_status": run.get("trust_status"),
                        "packet_retry_count": run.get("packet_retry_count"),
                        "telemetry_event_counts": run.get("telemetry_event_counts"),
                        "quality_flags": ",".join(str(value) for value in (run.get("quality_flags") or [])),
                        "load_balance_error_ma": run.get("load_balance_error_ma"),
                        "pair_balance_error_ma": run.get("pair_balance_error_ma"),
                        "final_tip_x_mm": (tip_xyz[0] if tip_xyz is not None and len(tip_xyz) > 0 else None),
                        "final_tip_y_mm": (tip_xyz[1] if tip_xyz is not None and len(tip_xyz) > 1 else None),
                        "final_tip_z_mm": (tip_xyz[2] if tip_xyz is not None and len(tip_xyz) > 2 else None),
                        "final_tip_xy_offset_mm": run.get("final_tip_xy_offset_mm"),
                        "settle_tip_drift_mm": run.get("settle_tip_drift_mm"),
                        "quality_score_0_100": run.get("quality_score_0_100"),
                        "quality_components": run.get("quality_components"),
                        "tip_centering_status": run.get("tip_centering_status"),
                        "equalization_status": run.get("equalization_status"),
                    }
                )


def _write_staged_summary_text(
    *,
    summary_text_path: Path,
    metadata,
    summary,
    metrics: dict[str, Any],
) -> None:
    lines = [
        "Pretension Validation Summary (Single-Segment Staged)",
        "Current is treated as a load/engagement proxy only; values are not calibrated tendon force.",
        "",
        f"Run ID: {metadata.run_id}",
        f"Timestamp: {metadata.timestamp_utc}",
        f"Status: {summary.status}",
        f"Algorithm: {metrics.get('algorithm')}",
        f"Algorithm mode: {metrics.get('algorithm_mode')}",
        f"Staged strategy: {metrics.get('staged_strategy')}",
        f"Servo IDs: {metrics.get('servo_ids')}",
        f"Repeat runs: {metrics.get('repeat_runs')}",
        f"Accepted runs: {metrics.get('accepted_run_count')} / {metrics.get('run_count')}",
        f"Accepted fraction: {_fmt_float(metrics.get('accepted_run_fraction'))}",
        "",
        "Units:",
        "- Current: mA servo-reported current estimate; signed current is saved separately from absolute load proxy.",
        "- Load proxy: absolute current delta from baseline in mA; not calibrated tendon force.",
        "- Position: ticks",
        "- Travel / tip position: mm",
        "",
        "Telemetry / trust:",
        f"- Runtime tip mode used: {metrics.get('runtime_tip_mode_used')}",
        f"- Runtime tip trust level: {metrics.get('runtime_tip_trust_level')}",
        f"- Thesis-trusted runtime tip: {metrics.get('thesis_trusted_runtime_tip')}",
        f"- Aggregate telemetry event counts: {metrics.get('telemetry_event_counts')}",
        f"- Aggregate packet retry count: {metrics.get('packet_retry_count')}",
        "",
        "Repeatability:",
        f"- Final position std by servo (ticks): {metrics.get('final_position_std_ticks_by_servo')}",
        f"- Final current std by servo (mA): {metrics.get('final_current_std_ma_by_servo')}",
        f"- Final tip XY std (mm): {_fmt_float(metrics.get('final_tip_xy_std_mm'))}",
        f"- Quality score mean (0-100): {_fmt_float(metrics.get('quality_score_mean_0_100'))}",
        f"- Quality score std (0-100): {_fmt_float(metrics.get('quality_score_std_0_100'))}",
        f"- Failure reasons: {metrics.get('failure_reason_counts')}",
        f"- Current characterization: {[row.get('current_characterization') for row in metrics.get('run_rows', [])]}",
        "",
        "Manual vs advanced artifact:",
        f"- Manual startup artifact: {metrics.get('manual_startup_artifact')}",
        f"- Advanced startup artifacts: {metrics.get('advanced_startup_artifacts')}",
        "",
        "Saved plots:",
        "- pretension_tendon_displacement_vs_tip_xy.png",
        "- pretension_tendon_displacement_vs_current.png",
        "- pretension_current_vs_tip_error.png",
        "- pretension_balance_over_stages.png",
        "- pretension_tip_xy_path.png",
        "- pretension_final_tip_xy_scatter.png",
        "- pretension_final_current_distribution.png",
        "- pretension_final_position_distribution.png",
        "- pretension_quality_score_distribution.png",
        "- pretension_repeatability_summary.png",
    ]
    summary_text_path.write_text("\n".join(lines).strip() + "\n", encoding="utf-8")


def _write_staged_current_vs_position_plot(*, current_vs_position_path: Path, trace_rows: list[dict[str, Any]]) -> None:
    grouped: dict[int, list[tuple[float, float]]] = {}
    for row in trace_rows:
        servo_id = row.get("servo_id")
        position = row.get("final_position_tick")
        if position is None:
            position = row.get("position_tick")
        current = row.get("final_current_ma")
        if current is None:
            current = row.get("raw_current_ma")
        if isinstance(current, dict):
            current = None
        if servo_id in (None, "") or position in (None, "") or current in (None, ""):
            continue
        grouped.setdefault(int(servo_id), []).append((float(position), float(current)))
    for point in _flatten_staged_trace_servo_points(trace_rows):
        if point["position_tick"] is None or point["current_ma"] is None:
            continue
        grouped.setdefault(int(point["servo_id"]), []).append((float(point["position_tick"]), float(point["current_ma"])))
    width, height = 1080, 720
    canvas = _Canvas(width, height, background=(255, 255, 255))
    canvas.text(28, 16, "PRETENSION CURRENT VS POSITION", color=(15, 23, 42), scale=2)
    canvas.text(28, 40, "POSITION (TICKS) / CURRENT (MA)", color=(71, 85, 105), scale=1)
    rect = (40, 72, width - 80, height - 120)
    _draw_plot_frame(canvas, rect, "CURRENT VS POSITION")
    points: list[tuple[float, float, tuple[int, int, int]]] = []
    palette = [(37, 99, 235), (217, 119, 6), (22, 163, 74), (139, 92, 246)]
    for index, servo_id in enumerate(sorted(grouped)):
        color = palette[index % len(palette)]
        for x_value, y_value in grouped[servo_id]:
            points.append((x_value, y_value, color))
    if not points:
        canvas.text(84, 220, "NO POSITION/CURRENT TRACE DATA", color=(100, 116, 139), scale=2)
        canvas.save_png(current_vs_position_path)
        return
    x_values = [point[0] for point in points]
    y_values = [point[1] for point in points]
    x_min, x_max = _expand_range(_min_max(x_values), pad_fraction=0.05, minimum_span=50.0)
    y_min, y_max = _expand_range(_min_max(y_values), pad_fraction=0.08, minimum_span=20.0)
    for x_value, y_value, color in points:
        canvas.circle(
            int(_plot_x(rect, x_min, x_max, x_value)),
            int(_plot_y(rect, y_min, y_max, y_value)),
            3,
            color,
        )
    canvas.save_png(current_vs_position_path)


def _flatten_staged_trace_servo_points(trace_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    points: list[dict[str, Any]] = []
    first_position_by_run_servo: dict[tuple[int, int], float] = {}
    for row in trace_rows:
        run_index = _as_int(row.get("run_index")) or 0
        measured_positions = dict(row.get("measured_positions_ticks") or {})
        raw_currents = dict(row.get("raw_current_ma") or {})
        filtered_currents = dict(row.get("filtered_current_ma") or {})
        tip_error = _as_float(
            row.get("tip_xy_error_mm")
            if row.get("tip_xy_error_mm") is not None
            else row.get("tip_xy_offset_mm") or row.get("final_tip_xy_offset_mm")
        )
        servo_keys = set(measured_positions.keys()) | set(raw_currents.keys()) | set(filtered_currents.keys())
        for raw_servo_id in servo_keys:
            try:
                servo_id = int(raw_servo_id)
            except Exception:
                continue
            position = _as_float(measured_positions.get(str(servo_id), measured_positions.get(servo_id)))
            raw_current = _as_float(raw_currents.get(str(servo_id), raw_currents.get(servo_id)))
            filtered_current = _as_float(filtered_currents.get(str(servo_id), filtered_currents.get(servo_id)))
            key = (int(run_index), int(servo_id))
            if position is not None and key not in first_position_by_run_servo:
                first_position_by_run_servo[key] = float(position)
            start_position = first_position_by_run_servo.get(key)
            travel_ticks = (
                None
                if position is None or start_position is None
                else abs(float(position) - float(start_position))
            )
            points.append(
                {
                    "run_index": int(run_index),
                    "servo_id": int(servo_id),
                    "position_tick": position,
                    "current_ma": filtered_current if filtered_current is not None else raw_current,
                    "raw_current_ma": raw_current,
                    "filtered_current_ma": filtered_current,
                    "travel_ticks": travel_ticks,
                    "tip_error_mm": tip_error,
                }
            )
    return points


def _write_staged_tendon_vs_tip_plot(*, plot_path: Path, trace_rows: list[dict[str, Any]]) -> None:
    points: list[tuple[float, float]] = []
    latest_tip_error_by_run: dict[int, float] = {}
    for row in trace_rows:
        run_index = _as_int(row.get("run_index")) or 0
        tip_error = _as_float(row.get("tip_xy_offset_mm") or row.get("final_tip_xy_offset_mm"))
        if tip_error is not None:
            latest_tip_error_by_run[int(run_index)] = float(tip_error)
        travel = _as_float(row.get("travel_used_mm"))
        if travel is None:
            travel_ticks = _as_float(row.get("travel_used_ticks"))
            travel = travel_ticks
        if travel is not None and int(run_index) in latest_tip_error_by_run:
            points.append((float(travel), float(latest_tip_error_by_run[int(run_index)])))
    for point in _flatten_staged_trace_servo_points(trace_rows):
        if point["travel_ticks"] is not None and point["tip_error_mm"] is not None:
            points.append((float(point["travel_ticks"]), float(point["tip_error_mm"])))
    _write_basic_scatter_plot(
        plot_path=plot_path,
        title="TENDON DISPLACEMENT VS TIP XY ERROR",
        subtitle="TENDON DISPLACEMENT (MM WHEN AVAILABLE; TICKS FALLBACK) / TIP XY ERROR (MM)",
        points=points,
        color=(124, 58, 237),
        empty_message="NO TENDON/TIP DATA",
    )


def _write_staged_tendon_vs_current_plot(*, plot_path: Path, trace_rows: list[dict[str, Any]]) -> None:
    points: list[tuple[float, float]] = []
    for row in trace_rows:
        raw_current_value = row.get("final_current_ma") or row.get("raw_current_ma") or row.get("filtered_current_ma")
        current = None if isinstance(raw_current_value, dict) else _as_float(raw_current_value)
        travel = _as_float(row.get("travel_used_mm"))
        if travel is None:
            travel = _as_float(row.get("travel_used_ticks"))
        if travel is not None and current is not None:
            points.append((float(travel), float(current)))
    for point in _flatten_staged_trace_servo_points(trace_rows):
        if point["travel_ticks"] is not None and point["current_ma"] is not None:
            points.append((float(point["travel_ticks"]), float(point["current_ma"])))
    _write_basic_scatter_plot(
        plot_path=plot_path,
        title="TENDON DISPLACEMENT VS CURRENT",
        subtitle="TENDON DISPLACEMENT (MM WHEN AVAILABLE; TICKS FALLBACK) / CURRENT (MA)",
        points=points,
        color=(37, 99, 235),
        empty_message="NO TENDON/CURRENT DATA",
    )


def _write_staged_current_vs_tip_error_plot(*, plot_path: Path, trace_rows: list[dict[str, Any]]) -> None:
    points: list[tuple[float, float]] = []
    latest_tip_error_by_run: dict[int, float] = {}
    for row in trace_rows:
        run_index = _as_int(row.get("run_index")) or 0
        tip_error = _as_float(row.get("tip_xy_offset_mm") or row.get("final_tip_xy_offset_mm"))
        if tip_error is not None:
            latest_tip_error_by_run[int(run_index)] = float(tip_error)
        raw_current_value = row.get("final_current_ma") or row.get("raw_current_ma") or row.get("filtered_current_ma")
        current = None if isinstance(raw_current_value, dict) else _as_float(raw_current_value)
        if current is not None and int(run_index) in latest_tip_error_by_run:
            points.append((float(current), float(latest_tip_error_by_run[int(run_index)])))
    for point in _flatten_staged_trace_servo_points(trace_rows):
        if point["current_ma"] is not None and point["tip_error_mm"] is not None:
            points.append((float(point["current_ma"]), float(point["tip_error_mm"])))
    _write_basic_scatter_plot(
        plot_path=plot_path,
        title="CURRENT VS TIP XY ERROR",
        subtitle="CURRENT (MA) / TIP XY ERROR (MM)",
        points=points,
        color=(217, 119, 6),
        empty_message="NO CURRENT/TIP DATA",
    )


def _write_staged_balance_over_stages_plot(
    *,
    plot_path: Path,
    trace_rows: list[dict[str, Any]],
    run_rows: list[dict[str, Any]],
) -> None:
    points: list[tuple[float, float, tuple[int, int, int]]] = []
    index = 0
    for row in trace_rows:
        for key, color in (
            ("load_balance_error_ma", (37, 99, 235)),
            ("pair_balance_error_ma", (217, 119, 6)),
            ("pair_current_mismatch_ma", (139, 92, 246)),
        ):
            value = _as_float(row.get(key))
            if value is None:
                continue
            points.append((float(index), float(value), color))
        index += 1
    for row in run_rows:
        for key, color in (
            ("load_balance_error_ma", (37, 99, 235)),
            ("pair_balance_error_ma", (217, 119, 6)),
        ):
            value = _as_float(row.get(key))
            if value is not None:
                points.append((float(index), float(value), color))
        index += 1
    _write_basic_scatter_plot(
        plot_path=plot_path,
        title="PAIR / LOAD BALANCE OVER STAGES",
        subtitle="STAGE SAMPLE INDEX / BALANCE ERROR (MA)",
        points=[(x, y) for x, y, _color in points],
        colors=[color for _x, _y, color in points],
        color=(37, 99, 235),
        empty_message="NO BALANCE DATA",
    )


def _write_staged_tip_xy_plot(*, tip_xy_path: Path, run_rows: list[dict[str, Any]]) -> None:
    points = []
    for row in run_rows:
        xyz = row.get("final_tip_xyz_mm")
        if not isinstance(xyz, list) or len(xyz) < 2:
            continue
        points.append((float(xyz[0]), float(xyz[1]), bool(row.get("accepted", False))))
    width, height = 1080, 720
    canvas = _Canvas(width, height, background=(255, 255, 255))
    canvas.text(28, 16, "PRETENSION FINAL TIP XY", color=(15, 23, 42), scale=2)
    canvas.text(28, 40, "TIP XY IN ROBOT FRAME (MM)", color=(71, 85, 105), scale=1)
    rect = (40, 72, width - 80, height - 120)
    _draw_plot_frame(canvas, rect, "TIP XY")
    if not points:
        canvas.text(84, 220, "NO TIP XY DATA AVAILABLE", color=(100, 116, 139), scale=2)
        canvas.save_png(tip_xy_path)
        return
    x_values = [point[0] for point in points]
    y_values = [point[1] for point in points]
    x_min, x_max = _expand_range(_min_max(x_values), pad_fraction=0.2, minimum_span=2.0)
    y_min, y_max = _expand_range(_min_max(y_values), pad_fraction=0.2, minimum_span=2.0)
    for x_value, y_value, accepted in points:
        color = (22, 163, 74) if accepted else (220, 38, 38)
        canvas.circle(
            int(_plot_x(rect, x_min, x_max, x_value)),
            int(_plot_y(rect, y_min, y_max, y_value)),
            4,
            color,
        )
    canvas.save_png(tip_xy_path)


def _write_staged_final_tip_scatter_plot(*, plot_path: Path, run_rows: list[dict[str, Any]]) -> None:
    points = []
    for row in run_rows:
        xyz = row.get("final_tip_xyz_mm")
        if not isinstance(xyz, list) or len(xyz) < 2:
            continue
        points.append((float(xyz[0]), float(xyz[1]), bool(row.get("accepted", False))))
    width, height = 1080, 720
    canvas = _Canvas(width, height, background=(255, 255, 255))
    canvas.text(28, 16, "FINAL TIP XY SCATTER", color=(15, 23, 42), scale=2)
    canvas.text(28, 40, "FINAL TIP XY IN ROBOT FRAME (MM)", color=(71, 85, 105), scale=1)
    rect = (40, 72, width - 80, height - 120)
    _draw_plot_frame(canvas, rect, "FINAL TIP XY")
    if not points:
        canvas.text(84, 220, "NO FINAL TIP XY DATA", color=(100, 116, 139), scale=2)
        canvas.save_png(plot_path)
        return
    x_values = [point[0] for point in points]
    y_values = [point[1] for point in points]
    x_min, x_max = _expand_range(_min_max(x_values + [0.0]), pad_fraction=0.2, minimum_span=3.0)
    y_min, y_max = _expand_range(_min_max(y_values + [0.0]), pad_fraction=0.2, minimum_span=3.0)
    canvas.circle(int(_plot_x(rect, x_min, x_max, 0.0)), int(_plot_y(rect, y_min, y_max, 0.0)), 5, (15, 23, 42))
    for x_value, y_value, accepted in points:
        color = (22, 163, 74) if accepted else (220, 38, 38)
        canvas.circle(
            int(_plot_x(rect, x_min, x_max, x_value)),
            int(_plot_y(rect, y_min, y_max, y_value)),
            4,
            color,
        )
    canvas.save_png(plot_path)


def _write_staged_final_current_distribution_plot(*, final_current_dist_path: Path, run_rows: list[dict[str, Any]]) -> None:
    by_servo: dict[int, list[float]] = {}
    for row in run_rows:
        for key, value in dict(row.get("final_current_ma_by_servo") or {}).items():
            if value is None:
                continue
            by_servo.setdefault(int(key), []).append(float(value))
    width, height = 1080, 680
    canvas = _Canvas(width, height, background=(255, 255, 255))
    canvas.text(28, 16, "FINAL CURRENT DISTRIBUTION", color=(15, 23, 42), scale=2)
    canvas.text(28, 40, "MEAN FINAL CURRENT BY SERVO (MA)", color=(71, 85, 105), scale=1)
    rect = (52, 88, width - 104, height - 150)
    _draw_plot_frame(canvas, rect, "FINAL CURRENT (MA)")
    if not by_servo:
        canvas.text(84, 220, "NO FINAL CURRENT DATA", color=(100, 116, 139), scale=2)
        canvas.save_png(final_current_dist_path)
        return
    servo_ids = sorted(by_servo)
    means = [sum(values) / len(values) for values in [by_servo[sid] for sid in servo_ids]]
    y_min, y_max = _expand_range(_min_max(means), pad_fraction=0.2, minimum_span=20.0)
    bar_width = max(12, int((rect[2] - 90) / max(1, len(servo_ids))))
    left_start = rect[0] + 46
    for index, servo_id in enumerate(servo_ids):
        mean_value = means[index]
        x_left = left_start + (index * (bar_width + 12))
        y_top = int(_plot_y(rect, y_min, y_max, mean_value))
        y_base = int(_plot_y(rect, y_min, y_max, y_min))
        for x in range(x_left, x_left + bar_width):
            canvas.line(x, y_top, x, y_base, color=(37, 99, 235), thickness=1)
        canvas.text(x_left, y_base + 8, f"S{servo_id}", color=(15, 23, 42), scale=1)
    canvas.save_png(final_current_dist_path)


def _write_staged_final_position_distribution_plot(*, plot_path: Path, run_rows: list[dict[str, Any]]) -> None:
    by_servo: dict[int, list[float]] = {}
    for row in run_rows:
        for key, value in dict(row.get("final_position_ticks_by_servo") or {}).items():
            if value is None:
                continue
            by_servo.setdefault(int(key), []).append(float(value))
    _write_distribution_by_servo_plot(
        plot_path=plot_path,
        title="FINAL SERVO POSITION DISTRIBUTION",
        subtitle="MEAN FINAL POSITION BY SERVO (TICKS)",
        values_by_servo=by_servo,
        empty_message="NO FINAL POSITION DATA",
        color=(22, 163, 74),
    )


def _write_staged_quality_distribution_plot(*, plot_path: Path, run_rows: list[dict[str, Any]]) -> None:
    values = [
        float(row["quality_score_0_100"])
        for row in run_rows
        if row.get("quality_score_0_100") is not None
    ]
    width, height = 1080, 680
    canvas = _Canvas(width, height, background=(255, 255, 255))
    canvas.text(28, 16, "QUALITY SCORE DISTRIBUTION", color=(15, 23, 42), scale=2)
    canvas.text(28, 40, "RUN QUALITY SCORE (0-100)", color=(71, 85, 105), scale=1)
    rect = (52, 88, width - 104, height - 150)
    _draw_plot_frame(canvas, rect, "QUALITY SCORE")
    if not values:
        canvas.text(84, 220, "NO QUALITY SCORE DATA", color=(100, 116, 139), scale=2)
        canvas.save_png(plot_path)
        return
    y_min, y_max = 0.0, 100.0
    bar_width = max(12, int((rect[2] - 90) / max(1, len(values))))
    left_start = rect[0] + 46
    for index, value in enumerate(values):
        x_left = left_start + (index * (bar_width + 8))
        y_top = int(_plot_y(rect, y_min, y_max, value))
        y_base = int(_plot_y(rect, y_min, y_max, y_min))
        for x in range(x_left, x_left + bar_width):
            canvas.line(x, y_top, x, y_base, color=(139, 92, 246), thickness=1)
        canvas.text(x_left, y_base + 8, f"R{index + 1}", color=(15, 23, 42), scale=1)
    canvas.save_png(plot_path)


def _write_staged_repeatability_plot(*, repeatability_path: Path, metrics: dict[str, Any]) -> None:
    position_std = dict(metrics.get("final_position_std_ticks_by_servo") or {})
    current_std = dict(metrics.get("final_current_std_ma_by_servo") or {})
    width, height = 1080, 680
    canvas = _Canvas(width, height, background=(255, 255, 255))
    canvas.text(28, 16, "PRETENSION REPEATABILITY SUMMARY", color=(15, 23, 42), scale=2)
    canvas.text(28, 40, "STD OF FINAL POSITION (TICKS) / CURRENT (MA)", color=(71, 85, 105), scale=1)
    canvas.rect(36, 84, width - 72, height - 132, color=(203, 213, 225), thickness=1)
    lines = [
        f"Accepted runs: {metrics.get('accepted_run_count')} / {metrics.get('run_count')}",
        f"Accepted fraction: {_fmt_float(metrics.get('accepted_run_fraction'))}",
        f"Final tip XY std (mm): {_fmt_float(metrics.get('final_tip_xy_std_mm'))}",
        f"Failure reasons: {metrics.get('failure_reason_counts')}",
        f"Position std by servo (ticks): {position_std}",
        f"Current std by servo (mA): {current_std}",
    ]
    y = 108
    for line in lines:
        canvas.text(52, y, line, color=(15, 23, 42), scale=1)
        y += 28
    canvas.save_png(repeatability_path)


def _write_basic_scatter_plot(
    *,
    plot_path: Path,
    title: str,
    subtitle: str,
    points: list[tuple[float, float]],
    color: tuple[int, int, int],
    empty_message: str,
    colors: list[tuple[int, int, int]] | None = None,
) -> None:
    width, height = 1080, 720
    canvas = _Canvas(width, height, background=(255, 255, 255))
    canvas.text(28, 16, title, color=(15, 23, 42), scale=2)
    canvas.text(28, 40, subtitle, color=(71, 85, 105), scale=1)
    rect = (40, 72, width - 80, height - 120)
    _draw_plot_frame(canvas, rect, "SCATTER")
    if not points:
        canvas.text(84, 220, empty_message, color=(100, 116, 139), scale=2)
        canvas.save_png(plot_path)
        return
    x_values = [float(point[0]) for point in points]
    y_values = [float(point[1]) for point in points]
    x_min, x_max = _expand_range(_min_max(x_values), pad_fraction=0.08, minimum_span=1.0)
    y_min, y_max = _expand_range(_min_max(y_values), pad_fraction=0.12, minimum_span=1.0)
    for index, (x_value, y_value) in enumerate(points):
        point_color = colors[index] if colors is not None and index < len(colors) else color
        canvas.circle(
            int(_plot_x(rect, x_min, x_max, float(x_value))),
            int(_plot_y(rect, y_min, y_max, float(y_value))),
            4,
            point_color,
        )
    canvas.save_png(plot_path)


def _write_distribution_by_servo_plot(
    *,
    plot_path: Path,
    title: str,
    subtitle: str,
    values_by_servo: dict[int, list[float]],
    empty_message: str,
    color: tuple[int, int, int],
) -> None:
    width, height = 1080, 680
    canvas = _Canvas(width, height, background=(255, 255, 255))
    canvas.text(28, 16, title, color=(15, 23, 42), scale=2)
    canvas.text(28, 40, subtitle, color=(71, 85, 105), scale=1)
    rect = (52, 88, width - 104, height - 150)
    _draw_plot_frame(canvas, rect, "DISTRIBUTION")
    if not values_by_servo:
        canvas.text(84, 220, empty_message, color=(100, 116, 139), scale=2)
        canvas.save_png(plot_path)
        return
    servo_ids = sorted(values_by_servo)
    means = [sum(values_by_servo[sid]) / len(values_by_servo[sid]) for sid in servo_ids]
    y_min, y_max = _expand_range(_min_max(means), pad_fraction=0.2, minimum_span=20.0)
    bar_width = max(12, int((rect[2] - 90) / max(1, len(servo_ids))))
    left_start = rect[0] + 46
    for index, servo_id in enumerate(servo_ids):
        mean_value = means[index]
        x_left = left_start + (index * (bar_width + 12))
        y_top = int(_plot_y(rect, y_min, y_max, mean_value))
        y_base = int(_plot_y(rect, y_min, y_max, y_min))
        for x in range(x_left, x_left + bar_width):
            canvas.line(x, y_top, x, y_base, color=color, thickness=1)
        canvas.text(x_left, y_base + 8, f"S{servo_id}", color=(15, 23, 42), scale=1)
    canvas.save_png(plot_path)


def _write_summary_text(*, summary_text_path: Path, metadata, summary, trace_points: list[PretensionTracePoint]) -> None:
    metrics = summary.experiment_metrics if isinstance(summary.experiment_metrics, dict) else {}
    lines = [
        "Pretension Validation Summary",
        "Current is used here as an engagement proxy only. This run does not claim tendon-force sensing.",
        "",
        f"Run ID: {metadata.run_id}",
        f"Timestamp: {metadata.timestamp_utc}",
        f"Servo ID: {metrics.get('servo_id', 'n/a')}",
        f"Accepted: {'yes' if metrics.get('accepted') else 'no'}",
        f"Status: {summary.status}",
        f"Stop Reason: {metrics.get('stop_reason', 'n/a')}",
        f"Final Position (ticks): {_fmt_int(metrics.get('final_position_tick'))}",
        f"Travel Used (ticks): {_fmt_int(metrics.get('travel_used_ticks'))}",
        f"Travel Used (mm): {_fmt_float(metrics.get('travel_used_mm'))}",
        f"Baseline Current (mA): {_fmt_float(metrics.get('baseline_current_ma'))}",
        f"Effective Trigger Current (mA): {_fmt_float(metrics.get('effective_trigger_current_ma'))}",
        f"Observed Trigger Current (mA): {_fmt_float(metrics.get('trigger_current_ma'))}",
        f"Hard Current Stop (mA): {_fmt_float(metrics.get('hard_current_stop_ma'))}",
        f"Max Observed Current (mA): {_fmt_float(metrics.get('max_observed_current_ma'))}",
        f"Max Observed Filtered Current (mA): {_fmt_float(metrics.get('max_observed_filtered_current_ma'))}",
        f"Max Observed Displacement (mm): {_fmt_float(metrics.get('max_observed_displacement_mm'))}",
        f"Trigger Displacement (mm): {_fmt_float(metrics.get('trigger_displacement_mm'))}",
        f"Tracker Metric Frame: {metrics.get('tracker_metric_frame', 'n/a')}",
        f"Trace Sample Count: {len(trace_points)}",
    ]
    summary_text_path.write_text("\n".join(lines).strip() + "\n", encoding="utf-8")


def _write_response_plot(
    *,
    plot_path: Path,
    trace_points: list[PretensionTracePoint],
    metrics: dict[str, Any],
) -> None:
    grouped: dict[int, list[PretensionTracePoint]] = {}
    for row in trace_points:
        grouped.setdefault(int(row.servo_id), []).append(row)
    if not grouped:
        grouped = {int(metrics.get("servo_id", 1) or 1): []}

    servo_ids = sorted(grouped)
    block_height = 300
    width = 1080
    height = 70 + (block_height * len(servo_ids))
    canvas = _Canvas(width, height, background=(255, 255, 255))
    canvas.text(28, 18, "PRETENSION RESPONSE VS TRAVEL", color=(15, 23, 42), scale=2)
    canvas.text(28, 44, "CURRENT IS AN ENGAGEMENT PROXY ONLY", color=(71, 85, 105), scale=1)

    for index, servo_id in enumerate(servo_ids):
        top = 70 + (index * block_height)
        _draw_servo_block(
            canvas=canvas,
            left=28,
            top=top,
            width=width - 56,
            height=block_height - 18,
            servo_id=int(servo_id),
            rows=list(grouped.get(int(servo_id), [])),
        )

    canvas.save_png(plot_path)


def _draw_servo_block(*, canvas: "_Canvas", left: int, top: int, width: int, height: int, servo_id: int, rows: list[PretensionTracePoint]) -> None:
    canvas.text(left, top, f"SERVO {servo_id}", color=(15, 23, 42), scale=2)
    current_rect = (left, top + 26, width, 160)
    disp_rect = (left, top + 206, width, 70)
    _draw_plot_frame(canvas, current_rect, "CURRENT")
    _draw_plot_frame(canvas, disp_rect, "DISP")

    use_mm = any(row.travel_from_untensioned_mm is not None for row in rows)
    raw_points = [
        (_trace_x(row, use_mm), float(row.raw_current_ma))
        for row in rows
        if _trace_x(row, use_mm) is not None and row.raw_current_ma is not None
    ]
    filtered_points = [
        (_trace_x(row, use_mm), float(row.filtered_current_ma))
        for row in rows
        if _trace_x(row, use_mm) is not None and row.filtered_current_ma is not None
    ]
    disp_points = [
        (_trace_x(row, use_mm), float(row.tracker_displacement_mm))
        for row in rows
        if _trace_x(row, use_mm) is not None and row.tracker_displacement_mm is not None
    ]

    if raw_points or filtered_points:
        x_values = [point[0] for point in raw_points + filtered_points if point[0] is not None]
        y_values = [point[1] for point in raw_points + filtered_points]
        for extra_value in (
            next((row.baseline_current_ma for row in rows if row.baseline_current_ma is not None), None),
            next((row.effective_trigger_current_ma for row in rows if row.effective_trigger_current_ma is not None), None),
            next((row.hard_current_stop_ma for row in rows if row.hard_current_stop_ma is not None), None),
        ):
            if extra_value is not None:
                y_values.append(float(extra_value))
        x_min, x_max = _min_max(x_values)
        y_min, y_max = _expand_range(_min_max(y_values), pad_fraction=0.08, minimum_span=20.0)
        _draw_threshold(canvas, current_rect, x_min, x_max, y_min, y_max, next((row.baseline_current_ma for row in rows if row.baseline_current_ma is not None), None), (22, 163, 74))
        _draw_threshold(canvas, current_rect, x_min, x_max, y_min, y_max, next((row.effective_trigger_current_ma for row in rows if row.effective_trigger_current_ma is not None), None), (217, 119, 6))
        _draw_threshold(canvas, current_rect, x_min, x_max, y_min, y_max, next((row.hard_current_stop_ma for row in rows if row.hard_current_stop_ma is not None), None), (220, 38, 38))
        _draw_series(canvas, current_rect, raw_points, x_min, x_max, y_min, y_max, (148, 163, 184), width_px=1)
        _draw_series(canvas, current_rect, filtered_points, x_min, x_max, y_min, y_max, (37, 99, 235), width_px=2)
        trigger_row = next((row for row in rows if row.trigger_met and _trace_x(row, use_mm) is not None and row.filtered_current_ma is not None), None)
        if trigger_row is not None:
            _draw_marker(
                canvas,
                current_rect,
                float(_trace_x(trigger_row, use_mm)),
                float(trigger_row.filtered_current_ma),
                x_min,
                x_max,
                y_min,
                y_max,
                (245, 158, 11),
            )
        canvas.text(left + 8, top + 194, "RAW/FILT/BASE/TRIG/STOP", color=(71, 85, 105), scale=1)
    else:
        canvas.text(left + 16, top + 98, "NO CURRENT TRACE", color=(100, 116, 139), scale=1)

    if disp_points:
        x_values = [point[0] for point in disp_points if point[0] is not None]
        y_values = [point[1] for point in disp_points]
        x_min, x_max = _min_max(x_values)
        y_min, y_max = _expand_range(_min_max(y_values), pad_fraction=0.12, minimum_span=0.5)
        _draw_series(canvas, disp_rect, disp_points, x_min, x_max, y_min, y_max, (124, 58, 237), width_px=2)
        last = disp_points[-1]
        _draw_marker(canvas, disp_rect, float(last[0]), float(last[1]), x_min, x_max, y_min, y_max, (124, 58, 237))
    else:
        canvas.text(left + 16, top + 236, "NO TRACKER DISPLACEMENT", color=(100, 116, 139), scale=1)


def _draw_plot_frame(canvas: "_Canvas", rect: tuple[int, int, int, int], title: str) -> None:
    left, top, width, height = rect
    canvas.rect(left, top, width, height, color=(203, 213, 225), thickness=1)
    canvas.text(left + 8, top + 6, title, color=(15, 23, 42), scale=1)
    canvas.line(left + 34, top + 18, left + 34, top + height - 16, color=(226, 232, 240), thickness=1)
    canvas.line(left + 34, top + height - 16, left + width - 10, top + height - 16, color=(226, 232, 240), thickness=1)


def _draw_threshold(
    canvas: "_Canvas",
    rect: tuple[int, int, int, int],
    x_min: float,
    x_max: float,
    y_min: float,
    y_max: float,
    value: float | None,
    color: tuple[int, int, int],
) -> None:
    if value is None:
        return
    x0, x1, y = _plot_x(rect, x_min, x_max, x_min), _plot_x(rect, x_min, x_max, x_max), _plot_y(rect, y_min, y_max, float(value))
    for dash_start in range(int(x0), int(x1), 8):
        canvas.line(dash_start, int(y), min(dash_start + 4, int(x1)), int(y), color=color, thickness=1)


def _draw_series(
    canvas: "_Canvas",
    rect: tuple[int, int, int, int],
    points: list[tuple[float | None, float]],
    x_min: float,
    x_max: float,
    y_min: float,
    y_max: float,
    color: tuple[int, int, int],
    *,
    width_px: int,
) -> None:
    mapped = [
        (
            _plot_x(rect, x_min, x_max, float(x_value)),
            _plot_y(rect, y_min, y_max, float(y_value)),
        )
        for x_value, y_value in points
        if x_value is not None
    ]
    for point in mapped:
        canvas.circle(int(point[0]), int(point[1]), 2, color)
    for start, end in zip(mapped[:-1], mapped[1:]):
        canvas.line(int(start[0]), int(start[1]), int(end[0]), int(end[1]), color=color, thickness=width_px)


def _draw_marker(
    canvas: "_Canvas",
    rect: tuple[int, int, int, int],
    x_value: float,
    y_value: float,
    x_min: float,
    x_max: float,
    y_min: float,
    y_max: float,
    color: tuple[int, int, int],
) -> None:
    canvas.circle(
        int(_plot_x(rect, x_min, x_max, x_value)),
        int(_plot_y(rect, y_min, y_max, y_value)),
        4,
        color,
    )


def _plot_x(rect: tuple[int, int, int, int], x_min: float, x_max: float, x_value: float) -> float:
    left, _top, width, _height = rect
    plot_left = left + 36
    plot_right = left + width - 10
    return _map_value(x_value, x_min, x_max, float(plot_left), float(plot_right))


def _plot_y(rect: tuple[int, int, int, int], y_min: float, y_max: float, y_value: float) -> float:
    _left, top, _width, height = rect
    plot_top = top + 22
    plot_bottom = top + height - 18
    return _map_value(y_value, y_min, y_max, float(plot_bottom), float(plot_top))


def _trace_x(row: PretensionTracePoint, use_mm: bool) -> float | None:
    if use_mm:
        return row.travel_from_untensioned_mm
    if row.travel_from_untensioned_ticks is None:
        return None
    return float(row.travel_from_untensioned_ticks)


def _map_value(value: float, source_min: float, source_max: float, target_min: float, target_max: float) -> float:
    if source_max <= source_min:
        return float((target_min + target_max) * 0.5)
    ratio = (float(value) - float(source_min)) / float(source_max - source_min)
    return float(target_min + (ratio * (target_max - target_min)))


def _min_max(values: list[float]) -> tuple[float, float]:
    minimum = min(values)
    maximum = max(values)
    if maximum <= minimum:
        return float(minimum), float(minimum + 1.0)
    return float(minimum), float(maximum)


def _expand_range(bounds: tuple[float, float], *, pad_fraction: float, minimum_span: float) -> tuple[float, float]:
    lower, upper = float(bounds[0]), float(bounds[1])
    span = max(float(upper - lower), float(minimum_span))
    padding = span * float(pad_fraction)
    return float(lower - padding), float(upper + padding)


def _fmt_float(value: Any) -> str:
    if value is None:
        return "n/a"
    return f"{float(value):.3f}"


def _fmt_int(value: Any) -> str:
    if value is None:
        return "n/a"
    return str(int(value))


def _as_int(value: Any) -> int | None:
    if value in (None, ""):
        return None
    return int(value)


def _as_float(value: Any) -> float | None:
    if value in (None, ""):
        return None
    return float(value)


class _Canvas:
    """Small RGB canvas with line/text primitives and PNG output."""

    def __init__(self, width: int, height: int, *, background: tuple[int, int, int]) -> None:
        self.width = int(width)
        self.height = int(height)
        self._pixels = bytearray(background * (self.width * self.height))

    def save_png(self, path: Path) -> None:
        path = Path(path)
        raw = bytearray()
        stride = self.width * 3
        for row in range(self.height):
            raw.append(0)
            start = row * stride
            raw.extend(self._pixels[start : start + stride])
        compressed = zlib.compress(bytes(raw), level=9)
        with path.open("wb") as handle:
            handle.write(b"\x89PNG\r\n\x1a\n")
            handle.write(self._png_chunk(b"IHDR", struct.pack(">IIBBBBB", self.width, self.height, 8, 2, 0, 0, 0)))
            handle.write(self._png_chunk(b"IDAT", compressed))
            handle.write(self._png_chunk(b"IEND", b""))

    def rect(self, x: int, y: int, width: int, height: int, *, color: tuple[int, int, int], thickness: int = 1) -> None:
        self.line(x, y, x + width, y, color=color, thickness=thickness)
        self.line(x, y, x, y + height, color=color, thickness=thickness)
        self.line(x + width, y, x + width, y + height, color=color, thickness=thickness)
        self.line(x, y + height, x + width, y + height, color=color, thickness=thickness)

    def line(self, x0: int, y0: int, x1: int, y1: int, *, color: tuple[int, int, int], thickness: int = 1) -> None:
        dx = float(x1 - x0)
        dy = float(y1 - y0)
        steps = int(max(abs(dx), abs(dy), 1))
        for step in range(steps + 1):
            ratio = float(step) / float(steps)
            x = int(round(float(x0) + (dx * ratio)))
            y = int(round(float(y0) + (dy * ratio)))
            self._stamp(x, y, color=color, radius=max(0, int(thickness) - 1))

    def circle(self, cx: int, cy: int, radius: int, color: tuple[int, int, int]) -> None:
        radius_sq = int(radius * radius)
        for y in range(cy - radius, cy + radius + 1):
            for x in range(cx - radius, cx + radius + 1):
                if ((x - cx) * (x - cx)) + ((y - cy) * (y - cy)) <= radius_sq:
                    self._set_pixel(x, y, color)

    def text(self, x: int, y: int, text: str, *, color: tuple[int, int, int], scale: int = 1) -> None:
        cursor = int(x)
        for char in str(text).upper():
            glyph = _FONT_5X7.get(char, _FONT_5X7["?"])
            for row_index, row_bits in enumerate(glyph):
                for col_index, bit in enumerate(row_bits):
                    if bit != "1":
                        continue
                    for dy in range(scale):
                        for dx in range(scale):
                            self._set_pixel(
                                cursor + (col_index * scale) + dx,
                                int(y) + (row_index * scale) + dy,
                                color,
                            )
            cursor += (6 * scale)

    def _stamp(self, x: int, y: int, *, color: tuple[int, int, int], radius: int) -> None:
        for yy in range(y - radius, y + radius + 1):
            for xx in range(x - radius, x + radius + 1):
                self._set_pixel(xx, yy, color)

    def _set_pixel(self, x: int, y: int, color: tuple[int, int, int]) -> None:
        if x < 0 or y < 0 or x >= self.width or y >= self.height:
            return
        index = (int(y) * self.width + int(x)) * 3
        self._pixels[index : index + 3] = bytes((int(color[0]), int(color[1]), int(color[2])))

    @staticmethod
    def _png_chunk(kind: bytes, data: bytes) -> bytes:
        crc = zlib.crc32(kind)
        crc = zlib.crc32(data, crc)
        return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", crc & 0xFFFFFFFF)


_FONT_5X7 = {
    " ": ["00000", "00000", "00000", "00000", "00000", "00000", "00000"],
    "-": ["00000", "00000", "00000", "11111", "00000", "00000", "00000"],
    ".": ["00000", "00000", "00000", "00000", "00000", "00110", "00110"],
    "/": ["00001", "00010", "00100", "01000", "10000", "00000", "00000"],
    ":": ["00000", "00110", "00110", "00000", "00110", "00110", "00000"],
    "?": ["01110", "10001", "00010", "00100", "00100", "00000", "00100"],
    "0": ["01110", "10001", "10011", "10101", "11001", "10001", "01110"],
    "1": ["00100", "01100", "00100", "00100", "00100", "00100", "01110"],
    "2": ["01110", "10001", "00001", "00010", "00100", "01000", "11111"],
    "3": ["11110", "00001", "00001", "01110", "00001", "00001", "11110"],
    "4": ["00010", "00110", "01010", "10010", "11111", "00010", "00010"],
    "5": ["11111", "10000", "10000", "11110", "00001", "00001", "11110"],
    "6": ["01110", "10000", "10000", "11110", "10001", "10001", "01110"],
    "7": ["11111", "00001", "00010", "00100", "01000", "01000", "01000"],
    "8": ["01110", "10001", "10001", "01110", "10001", "10001", "01110"],
    "9": ["01110", "10001", "10001", "01111", "00001", "00001", "01110"],
    "A": ["01110", "10001", "10001", "11111", "10001", "10001", "10001"],
    "B": ["11110", "10001", "10001", "11110", "10001", "10001", "11110"],
    "C": ["01110", "10001", "10000", "10000", "10000", "10001", "01110"],
    "D": ["11100", "10010", "10001", "10001", "10001", "10010", "11100"],
    "E": ["11111", "10000", "10000", "11110", "10000", "10000", "11111"],
    "F": ["11111", "10000", "10000", "11110", "10000", "10000", "10000"],
    "G": ["01110", "10001", "10000", "10111", "10001", "10001", "01110"],
    "H": ["10001", "10001", "10001", "11111", "10001", "10001", "10001"],
    "I": ["01110", "00100", "00100", "00100", "00100", "00100", "01110"],
    "J": ["00001", "00001", "00001", "00001", "10001", "10001", "01110"],
    "K": ["10001", "10010", "10100", "11000", "10100", "10010", "10001"],
    "L": ["10000", "10000", "10000", "10000", "10000", "10000", "11111"],
    "M": ["10001", "11011", "10101", "10101", "10001", "10001", "10001"],
    "N": ["10001", "11001", "10101", "10011", "10001", "10001", "10001"],
    "O": ["01110", "10001", "10001", "10001", "10001", "10001", "01110"],
    "P": ["11110", "10001", "10001", "11110", "10000", "10000", "10000"],
    "Q": ["01110", "10001", "10001", "10001", "10101", "10010", "01101"],
    "R": ["11110", "10001", "10001", "11110", "10100", "10010", "10001"],
    "S": ["01111", "10000", "10000", "01110", "00001", "00001", "11110"],
    "T": ["11111", "00100", "00100", "00100", "00100", "00100", "00100"],
    "U": ["10001", "10001", "10001", "10001", "10001", "10001", "01110"],
    "V": ["10001", "10001", "10001", "10001", "10001", "01010", "00100"],
    "W": ["10001", "10001", "10001", "10101", "10101", "10101", "01010"],
    "X": ["10001", "10001", "01010", "00100", "01010", "10001", "10001"],
    "Y": ["10001", "10001", "01010", "00100", "00100", "00100", "00100"],
    "Z": ["11111", "00001", "00010", "00100", "01000", "10000", "11111"],
}

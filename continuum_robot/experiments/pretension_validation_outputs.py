"""Extra artifacts for the pretension-validation experiment."""

from __future__ import annotations

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
    plot_path = output_dir / "pretension_response.png"
    summary_text_path = output_dir / "pretension_summary.txt"
    trace_points = extract_pretension_trace_points(samples)
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

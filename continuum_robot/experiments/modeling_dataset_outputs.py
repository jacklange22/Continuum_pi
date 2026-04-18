"""Saved artifacts for the Motor Babble modeling dataset collector."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from continuum_robot.experiments.dat_writer import DatRunWriter
from continuum_robot.experiments.tracker_timing_outputs import (
    _draw_bar_chart,
    _draw_panel,
    _draw_summary_pairs,
    _draw_title,
    _ensure_plot_qt_app,
    _fmt,
    _new_image,
)
from continuum_robot.gui.theme import COLORS
from PySide6.QtCore import QPointF, QRectF, Qt
from PySide6.QtGui import QColor, QPainter, QPen


def build_modeling_dataset_summary_pairs(*, metrics: dict[str, Any]) -> list[tuple[str, str]]:
    provenance = dict(metrics.get("run_provenance", {}) or {})
    runtime_tip = dict(provenance.get("runtime_tip_calibration", {}) or {})
    pretension = dict(provenance.get("pretension_artifact", {}) or {})
    command_range = dict(metrics.get("command_pair_range_cm", {}) or {})
    workspace_span = dict(metrics.get("workspace_span_mm", {}) or {})
    return [
        ("Dataset Mode", str(metrics.get("dataset_mode", "unknown") or "unknown").replace("_", " ")),
        ("Run Label", str(metrics.get("run_label", "") or "n/a")),
        ("Dataset Tag", str(metrics.get("dataset_tag", "") or "n/a")),
        ("Accepted Samples", str(int(metrics.get("accepted_sample_count", 0) or 0))),
        ("Rejected Samples", str(int(metrics.get("rejected_sample_count", 0) or 0))),
        (
            "Acceptance Rate",
            (
                f"{float(metrics.get('accepted_capture_rate', 0.0) or 0.0) * 100.0:.1f}%"
                if metrics.get("accepted_capture_rate") is not None
                else "n/a"
            ),
        ),
        ("Command Steps", str(int(metrics.get("command_step_count", 0) or 0))),
        ("Samples / Command", str(int(metrics.get("samples_per_command", 0) or 0))),
        ("Runtime Tip", f"{runtime_tip.get('mode', 'unknown')} ({runtime_tip.get('trust_level', 'unknown')})"),
        ("Pretension", f"{pretension.get('active_source_type', 'unknown')} ({pretension.get('status', 'unknown')})"),
        (
            "Pair 1/3 Range",
            (
                f"{_fmt(command_range.get('pair_13_min_cm'))} .. {_fmt(command_range.get('pair_13_max_cm'))} cm"
                if command_range
                else "n/a"
            ),
        ),
        (
            "Pair 2/4 Range",
            (
                f"{_fmt(command_range.get('pair_24_min_cm'))} .. {_fmt(command_range.get('pair_24_max_cm'))} cm"
                if command_range
                else "n/a"
            ),
        ),
        (
            "Workspace Span",
            (
                f"x={_fmt(workspace_span.get('x_span_mm'))} mm, "
                f"y={_fmt(workspace_span.get('y_span_mm'))} mm, "
                f"z={_fmt(workspace_span.get('z_span_mm'))} mm"
                if workspace_span
                else "n/a"
            ),
        ),
        ("Lower-Trust Override", "yes" if metrics.get("lower_trust_active") else "no"),
    ]


def build_modeling_dataset_summary_lines(*, metadata, summary, metrics: dict[str, Any]) -> list[str]:
    provenance = dict(metrics.get("run_provenance", {}) or {})
    runtime_tip = dict(provenance.get("runtime_tip_calibration", {}) or {})
    registration = dict(provenance.get("base_registration", {}) or {})
    pretension = dict(provenance.get("pretension_artifact", {}) or {})
    lines = [
        "Motor Babble Modeling Dataset Summary",
        "This run collected ordered single-segment command/pose samples for offline forward-model training and later state-aware model comparison.",
        "",
        f"Run ID: {metadata.run_id}",
        f"Timestamp: {metadata.timestamp_utc}",
        f"Status: {summary.status}",
    ]
    for label, value in build_modeling_dataset_summary_pairs(metrics=metrics):
        lines.append(f"{label}: {value}")
    lines.extend(
        [
            "",
            "Provenance:",
            f"- Registration: {registration.get('path', 'n/a')} @ {registration.get('stored_timestamp_utc', registration.get('modified_at_utc', 'n/a'))}",
            (
                f"- Runtime tip: {runtime_tip.get('path', 'n/a')} "
                f"(mode={runtime_tip.get('mode', 'unknown')}, trust={runtime_tip.get('trust_level', 'unknown')}, "
                f"timestamp={runtime_tip.get('stored_timestamp_utc', runtime_tip.get('modified_at_utc', 'n/a'))})"
            ),
            (
                f"- Pretension: {pretension.get('path', 'n/a')} "
                f"(source={pretension.get('active_source_type', 'unknown')}, "
                f"updated={pretension.get('active_source_updated_at_utc', pretension.get('modified_at_utc', 'n/a'))})"
            ),
        ]
    )
    if runtime_tip.get("mode_message"):
        lines.append(f"- Runtime tip detail: {runtime_tip.get('mode_message')}")
    if pretension.get("active_source_message"):
        lines.append(f"- Pretension detail: {pretension.get('active_source_message')}")
    rejection_reasons = dict(metrics.get("rejection_reasons", {}) or {})
    if rejection_reasons:
        lines.extend(["", "Rejected captures:"])
        for reason, count in sorted(rejection_reasons.items()):
            lines.append(f"- {reason}: {int(count)}")
    lines.extend(
        [
            "",
            "Offline handoff:",
            "- `samples.jsonl` preserves canonical full-fidelity ordered samples.",
            "- `modeling_dataset_export.jsonl` preserves ordered export rows with an explicit accepted flag for offline filtering or sequential multi-input modeling.",
            "- `modeling_dataset_legacy_compat.dat` contains accepted rows only for quick legacy ANN/comparison-stack conversion.",
        ]
    )
    return lines


def write_modeling_dataset_outputs(*, output_dir: Path, metadata, summary, samples) -> dict[str, Path]:
    _ensure_plot_qt_app()
    output_dir = Path(output_dir)
    metrics = summary.experiment_metrics if isinstance(summary.experiment_metrics, dict) else {}
    summary_text_path = output_dir / "modeling_dataset_summary.txt"
    export_jsonl_path = output_dir / "modeling_dataset_export.jsonl"
    legacy_dat_path = output_dir / "modeling_dataset_legacy_compat.dat"
    workspace_plot_path = output_dir / "modeling_workspace_coverage.png"
    command_plot_path = output_dir / "modeling_command_distribution.png"

    summary_text_path.write_text(
        "\n".join(
            build_modeling_dataset_summary_lines(
                metadata=metadata,
                summary=summary,
                metrics=metrics,
            )
        ).strip()
        + "\n",
        encoding="utf-8",
    )
    export_rows = _build_export_rows(samples=samples)
    with export_jsonl_path.open("w", encoding="utf-8") as handle:
        for row in export_rows:
            handle.write(json.dumps(row, separators=(",", ":")) + "\n")
    legacy_written_path: Path | None = None
    if bool(metrics.get("legacy_export_enabled", True)):
        rows = _build_legacy_dat_rows(export_rows=export_rows)
        if rows:
            writer = DatRunWriter(output_dir=output_dir)
            legacy_written_path = writer.write_run(
                num_cables=4,
                rows=rows,
                filename_stem="modeling_dataset_legacy_compat",
            )
    _write_workspace_plot(workspace_plot_path=workspace_plot_path, export_rows=export_rows, metrics=metrics)
    _write_command_plot(command_plot_path=command_plot_path, export_rows=export_rows, metrics=metrics)
    outputs: dict[str, Path] = {
        "summary_text_path": summary_text_path,
        "export_jsonl_path": export_jsonl_path,
        "workspace_plot_path": workspace_plot_path,
        "command_plot_path": command_plot_path,
    }
    if legacy_written_path is not None:
        outputs["legacy_dat_path"] = legacy_written_path
    elif legacy_dat_path.exists():
        outputs["legacy_dat_path"] = legacy_dat_path
    return outputs


def _build_export_rows(*, samples) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for sequence_index, sample in enumerate(samples):
        tip_payload = dict(sample.pose_in_robot_frame.get("tip", {}) or {})
        tool_payload = dict(sample.pose_in_tracker_frame.get("0A", {}) or {})
        extra = dict(sample.extra or {})
        rows.append(
            {
                "sequence_index": int(sequence_index),
                "phase": str(sample.phase or ""),
                "step_index": int(sample.step_index),
                "sample_index": int(sample.sample_index),
                "accepted": bool(extra.get("capture_accepted")),
                "capture_rejection_reason": extra.get("capture_rejection_reason"),
                "dataset_mode": extra.get("dataset_mode"),
                "requested_pair_command_cm": list(extra.get("requested_pair_command_cm", []) or []),
                "resolved_pair_command_cm": list(extra.get("resolved_pair_command_cm", []) or []),
                "previous_pair_command_cm": list(extra.get("previous_pair_command_cm", []) or []),
                "requested_cable_command_cm": list(extra.get("requested_cable_command_cm", []) or []),
                "resolved_cable_command_cm": list(extra.get("resolved_cable_command_cm", []) or []),
                "raw_goal_ticks_by_servo": dict(extra.get("raw_goal_ticks_by_servo", {}) or {}),
                "final_goal_ticks_by_servo": dict(extra.get("final_goal_ticks_by_servo", {}) or {}),
                "servo_feedback_at_capture": dict(extra.get("servo_feedback_at_capture", {}) or {}),
                "tracker_frame_id": sample.tracker_frame_id,
                "tracker_freshness_s": sample.freshness_s,
                "tracker_status_flags": list(sample.status_flags or []),
                "tip_position_xyz_mm": list(tip_payload.get("translation_mm", []) or []),
                "tip_quaternion_wxyz": list(tip_payload.get("quaternion_wxyz", []) or []),
                "tip_tangent_xyz": list(tip_payload.get("tangent_xyz", []) or []),
                "tool_0A_translation_mm": list(tool_payload.get("translation_mm", []) or []),
                "tool_0A_quaternion_wxyz": list(tool_payload.get("quaternion_wxyz", []) or []),
                "tool_0A_tangent_xyz": list(tool_payload.get("tangent_xyz", []) or []),
            }
        )
    return rows


def _build_legacy_dat_rows(*, export_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row in export_rows:
        if not bool(row.get("accepted")):
            continue
        servo_feedback = dict(row.get("servo_feedback_at_capture", {}) or {})
        ordered_servo_ids = sorted(servo_feedback, key=lambda value: int(value))
        rows.append(
            {
                "timestamp_utc": "",
                "index": int(row.get("sequence_index", 0) or 0),
                "repeat_index": int(row.get("sample_index", 0) or 0),
                "commanded_displacement_cm": list(row.get("resolved_cable_command_cm", []) or []),
                "commanded_goal_ticks": [
                    (dict(row.get("final_goal_ticks_by_servo", {}) or {}).get(servo_id))
                    for servo_id in ordered_servo_ids
                ],
                "servo_position_ticks": [
                    (dict(servo_feedback.get(servo_id, {}) or {}).get("present_position_ticks"))
                    for servo_id in ordered_servo_ids
                ],
                "servo_current_ma": [
                    (dict(servo_feedback.get(servo_id, {}) or {}).get("present_current_ma"))
                    for servo_id in ordered_servo_ids
                ],
                "servo_voltage_mv": [
                    (dict(servo_feedback.get(servo_id, {}) or {}).get("present_voltage_mv"))
                    for servo_id in ordered_servo_ids
                ],
                "tool_0A_translation_mm": list(row.get("tool_0A_translation_mm", []) or []),
                "tool_0B_translation_mm": [],
                "tip_position_xyz": list(row.get("tip_position_xyz_mm", []) or []),
                "tip_tangent_xyz": list(row.get("tip_tangent_xyz", []) or []),
            }
        )
    return rows


def _write_workspace_plot(*, workspace_plot_path: Path, export_rows: list[dict[str, Any]], metrics: dict[str, Any]) -> None:
    image = _new_image(1280, 860)
    painter = QPainter(image)
    painter.setRenderHint(QPainter.Antialiasing, True)
    _draw_title(
        painter,
        QRectF(34.0, 24.0, 1212.0, 58.0),
        "MODELING DATASET WORKSPACE COVERAGE",
        "Accepted robot-frame tip positions for the single-segment Motor Babble dataset.",
    )
    scatter_rect = QRectF(36.0, 108.0, 820.0, 700.0)
    summary_rect = QRectF(888.0, 108.0, 356.0, 700.0)
    _draw_panel(painter, scatter_rect, "Accepted Tip Positions (XY)")
    _draw_panel(painter, summary_rect, "Run Summary")
    accepted_rows = [row for row in export_rows if row.get("accepted") and len(row.get("tip_position_xyz_mm", [])) == 3]
    rejected_rows = [row for row in export_rows if not row.get("accepted") and len(row.get("tip_position_xyz_mm", [])) == 3]
    _draw_xy_scatter(
        painter,
        scatter_rect.adjusted(22.0, 42.0, -22.0, -28.0),
        accepted_points=[(float(row["tip_position_xyz_mm"][0]), float(row["tip_position_xyz_mm"][1])) for row in accepted_rows],
        rejected_points=[(float(row["tip_position_xyz_mm"][0]), float(row["tip_position_xyz_mm"][1])) for row in rejected_rows],
        x_label="X (mm)",
        y_label="Y (mm)",
        empty_text="No accepted robot-frame tip samples were available.",
    )
    _draw_summary_pairs(
        painter,
        summary_rect.adjusted(16.0, 38.0, -16.0, -16.0),
        build_modeling_dataset_summary_pairs(metrics=metrics),
    )
    painter.end()
    image.save(str(workspace_plot_path))


def _write_command_plot(*, command_plot_path: Path, export_rows: list[dict[str, Any]], metrics: dict[str, Any]) -> None:
    image = _new_image(1280, 820)
    painter = QPainter(image)
    painter.setRenderHint(QPainter.Antialiasing, True)
    _draw_title(
        painter,
        QRectF(34.0, 24.0, 1212.0, 58.0),
        "MODELING DATASET COMMAND DISTRIBUTION",
        "Resolved antagonistic pair commands and capture acceptance summary.",
    )
    scatter_rect = QRectF(36.0, 108.0, 820.0, 660.0)
    summary_rect = QRectF(888.0, 108.0, 356.0, 660.0)
    _draw_panel(painter, scatter_rect, "Pair Command Space")
    _draw_panel(painter, summary_rect, "Acceptance Summary")
    accepted_rows = [row for row in export_rows if row.get("accepted") and len(row.get("resolved_pair_command_cm", [])) == 2]
    rejected_rows = [row for row in export_rows if not row.get("accepted") and len(row.get("resolved_pair_command_cm", [])) == 2]
    _draw_xy_scatter(
        painter,
        scatter_rect.adjusted(22.0, 42.0, -22.0, -28.0),
        accepted_points=[(float(row["resolved_pair_command_cm"][0]), float(row["resolved_pair_command_cm"][1])) for row in accepted_rows],
        rejected_points=[(float(row["resolved_pair_command_cm"][0]), float(row["resolved_pair_command_cm"][1])) for row in rejected_rows],
        x_label="Pair 1/3 command (cm)",
        y_label="Pair 2/4 command (cm)",
        empty_text="No pair-command rows were available.",
    )
    _draw_bar_chart(
        painter,
        summary_rect.adjusted(16.0, 38.0, -16.0, -320.0),
        labels=["Accepted", "Rejected"],
        values=[
            float(metrics.get("accepted_sample_count", 0) or 0),
            float(metrics.get("rejected_sample_count", 0) or 0),
        ],
        color=QColor(COLORS.scene_measurement),
        y_label="Samples",
        empty_text="No capture rows were available.",
    )
    phase_counts = dict(metrics.get("phase_counts", {}) or {})
    _draw_bar_chart(
        painter,
        summary_rect.adjusted(16.0, 370.0, -16.0, -16.0),
        labels=[str(key) for key in phase_counts],
        values=[float(value) for value in phase_counts.values()],
        color=QColor(COLORS.scene_residual),
        y_label="Samples",
        empty_text="No phase counts were recorded.",
    )
    painter.end()
    image.save(str(command_plot_path))


def _draw_xy_scatter(
    painter: QPainter,
    rect: QRectF,
    *,
    accepted_points: list[tuple[float, float]],
    rejected_points: list[tuple[float, float]],
    x_label: str,
    y_label: str,
    empty_text: str,
) -> None:
    painter.save()
    if not accepted_points and not rejected_points:
        painter.setPen(QColor(COLORS.text_muted))
        painter.drawText(rect, Qt.AlignCenter | Qt.TextWordWrap, empty_text)
        painter.restore()
        return
    all_points = accepted_points + rejected_points
    xs = [point[0] for point in all_points]
    ys = [point[1] for point in all_points]
    min_x = min(xs)
    max_x = max(xs)
    min_y = min(ys)
    max_y = max(ys)
    if abs(max_x - min_x) < 1e-6:
        min_x -= 1.0
        max_x += 1.0
    if abs(max_y - min_y) < 1e-6:
        min_y -= 1.0
        max_y += 1.0
    plot_rect = rect.adjusted(56.0, 18.0, -22.0, -48.0)
    painter.setPen(QPen(QColor(COLORS.surface_border), 1.0))
    painter.drawRect(plot_rect)
    painter.setPen(QColor(COLORS.text_muted))
    painter.drawText(QRectF(plot_rect.left(), rect.bottom() - 26.0, plot_rect.width(), 20.0), Qt.AlignCenter, x_label)
    painter.save()
    painter.translate(rect.left() + 18.0, plot_rect.center().y())
    painter.rotate(-90.0)
    painter.drawText(QRectF(-plot_rect.height() / 2.0, -10.0, plot_rect.height(), 20.0), Qt.AlignCenter, y_label)
    painter.restore()
    for value in np.linspace(0.0, 1.0, 5):
        x = plot_rect.left() + (plot_rect.width() * float(value))
        y = plot_rect.bottom() - (plot_rect.height() * float(value))
        grid_pen = QPen(QColor(COLORS.chart_grid), 1.0)
        painter.setPen(grid_pen)
        painter.drawLine(QPointF(x, plot_rect.top()), QPointF(x, plot_rect.bottom()))
        painter.drawLine(QPointF(plot_rect.left(), y), QPointF(plot_rect.right(), y))
    accepted_pen = QColor(COLORS.scene_measurement)
    rejected_pen = QColor(COLORS.scene_residual)
    for point in accepted_points:
        painter.setPen(Qt.NoPen)
        painter.setBrush(accepted_pen)
        painter.drawEllipse(_map_point_xy(point, plot_rect, min_x=min_x, max_x=max_x, min_y=min_y, max_y=max_y), 4.0, 4.0)
    for point in rejected_points:
        painter.setPen(Qt.NoPen)
        painter.setBrush(rejected_pen)
        painter.drawEllipse(_map_point_xy(point, plot_rect, min_x=min_x, max_x=max_x, min_y=min_y, max_y=max_y), 3.5, 3.5)
    painter.restore()


def _map_point_xy(
    point: tuple[float, float],
    rect: QRectF,
    *,
    min_x: float,
    max_x: float,
    min_y: float,
    max_y: float,
) -> QPointF:
    x_ratio = (float(point[0]) - float(min_x)) / float(max_x - min_x)
    y_ratio = (float(point[1]) - float(min_y)) / float(max_y - min_y)
    x = rect.left() + (rect.width() * x_ratio)
    y = rect.bottom() - (rect.height() * y_ratio)
    return QPointF(float(x), float(y))

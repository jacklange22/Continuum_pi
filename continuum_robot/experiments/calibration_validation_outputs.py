"""Saved artifacts for offline registration/pivot validation experiments."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

import numpy as np

from continuum_robot.experiments.tracker_timing_outputs import (
    _body_font,
    _body_bold_font,
    _draw_bar_chart,
    _draw_histogram,
    _draw_panel,
    _draw_summary_pairs,
    _draw_title,
    _ensure_plot_qt_app,
    _new_image,
    _panel_font,
    _write_plot_placeholder,
)

try:
    from PySide6.QtCore import QRectF, Qt
    from PySide6.QtGui import QColor, QPainter, QPen

    _QT_AVAILABLE = True
except ModuleNotFoundError:
    _QT_AVAILABLE = False


def write_registration_validation_outputs(*, output_dir: Path, metadata, summary) -> dict[str, Path]:
    """Write canonical output artifacts for registration validation."""
    metrics = summary.experiment_metrics if isinstance(summary.experiment_metrics, dict) else {}
    output_dir = Path(output_dir)
    csv_path = output_dir / "metrics.csv"
    text_path = output_dir / "registration_validation_summary.txt"
    fre_plot_path = output_dir / "registration_fre_histogram.png"
    origin_plot_path = output_dir / "registration_frame_origins.png"
    spread_plot_path = output_dir / "registration_transform_spread.png"
    _write_rows_csv(
        csv_path,
        rows=metrics.get("per_run_rows") or [],
        header_overrides={
            "path": "path",
            "label": "label",
            "timestamp_utc": "timestamp_utc",
            "measurement_tool_id": "measurement_tool_id",
            "coil_tool_id": "coil_tool_id",
            "fre_mm": "fre_mm",
            "landmark_count": "landmark_count",
            "robot_origin_in_aurora_mm": "robot_origin_in_aurora_mm",
            "translation_delta_to_consensus_mm": "translation_delta_to_consensus_mm",
            "rotation_delta_to_consensus_deg": "rotation_delta_to_consensus_deg",
            "origin_distance_to_consensus_mm": "origin_distance_to_consensus_mm",
            "run_index": "run_index",
        },
    )
    text_path.write_text(
        "\n".join(build_registration_validation_summary_lines(metadata=metadata, summary=summary, metrics=metrics)).strip() + "\n",
        encoding="utf-8",
    )
    if _QT_AVAILABLE:
        _ensure_plot_qt_app()
        _write_registration_fre_histogram(fre_plot_path=fre_plot_path, metrics=metrics)
        _write_registration_origin_plot(origin_plot_path=origin_plot_path, metrics=metrics)
        _write_registration_spread_plot(spread_plot_path=spread_plot_path, metrics=metrics)
    else:
        _write_plot_placeholder(fre_plot_path)
        _write_plot_placeholder(origin_plot_path)
        _write_plot_placeholder(spread_plot_path)
    return {
        "metrics_csv_path": csv_path,
        "summary_text_path": text_path,
        "fre_plot_path": fre_plot_path,
        "origin_plot_path": origin_plot_path,
        "spread_plot_path": spread_plot_path,
    }


def write_pivot_validation_outputs(*, output_dir: Path, metadata, summary) -> dict[str, Path]:
    """Write canonical output artifacts for pivot validation."""
    metrics = summary.experiment_metrics if isinstance(summary.experiment_metrics, dict) else {}
    output_dir = Path(output_dir)
    csv_path = output_dir / "metrics.csv"
    text_path = output_dir / "pivot_validation_summary.txt"
    scatter_plot_path = output_dir / "pivot_tip_scatter.png"
    axis_plot_path = output_dir / "pivot_axis_histograms.png"
    quality_plot_path = output_dir / "pivot_quality_summary.png"
    _write_rows_csv(
        csv_path,
        rows=metrics.get("per_run_rows") or [],
        header_overrides={
            "path": "path",
            "label": "label",
            "timestamp_utc": "timestamp_utc",
            "status": "status",
            "tool_id": "tool_id",
            "tip_vector_local_mm": "tip_vector_local_mm",
            "tip_norm_mm": "tip_offset_norm_mm",
            "distance_to_consensus_mm": "tip_offset_deviation_to_consensus_mm",
            "rmse_mm": "pivot_rmse_mm",
            "sample_count_total": "sample_count_total",
            "sample_count_used": "sample_count_used",
            "sample_count_rejected": "sample_count_rejected",
            "input_format": "input_format",
            "run_index": "run_index",
        },
    )
    text_path.write_text(
        "\n".join(build_pivot_validation_summary_lines(metadata=metadata, summary=summary, metrics=metrics)).strip() + "\n",
        encoding="utf-8",
    )
    if _QT_AVAILABLE:
        _ensure_plot_qt_app()
        _write_pivot_scatter_plot(scatter_plot_path=scatter_plot_path, metrics=metrics)
        _write_pivot_axis_histograms(axis_plot_path=axis_plot_path, metrics=metrics)
        _write_pivot_quality_plot(quality_plot_path=quality_plot_path, metrics=metrics)
    else:
        _write_plot_placeholder(scatter_plot_path)
        _write_plot_placeholder(axis_plot_path)
        _write_plot_placeholder(quality_plot_path)
    return {
        "metrics_csv_path": csv_path,
        "summary_text_path": text_path,
        "scatter_plot_path": scatter_plot_path,
        "axis_plot_path": axis_plot_path,
        "quality_plot_path": quality_plot_path,
    }


def build_registration_validation_summary_lines(*, metadata, summary, metrics: dict[str, Any]) -> list[str]:
    """Return human-readable registration-validation summary text."""
    spread = dict(metrics.get("robot_origin_spread_mm", {}) or {})
    lines = [
        "Registration Validation Summary",
        "Offline analysis of repeated saved registration solves. Metrics quantify FRE spread, transform spread, and where the downstream robot/model frame origin lands in Aurora space (all distances in mm, rotation in degrees).",
        "",
        f"Run ID: {metadata.run_id}",
        f"Timestamp: {metadata.timestamp_utc}",
        f"Status: {summary.status}",
        f"Selected runs: {int(metrics.get('selected_run_count', 0) or 0)}",
        f"Valid runs: {int(metrics.get('valid_run_count', 0) or 0)}",
        f"Skipped runs: {int(metrics.get('invalid_run_count', 0) or 0)}",
        "",
        "Definitions:",
        "- FRE is residual error on the fiducials used by each registration solve (not an external ground-truth held-out error).",
        "- Translation/rotation spread values are transform deltas relative to a consensus transform across selected runs.",
        "",
        f"FRE mean / std / max (mm): {_metric_triplet(metrics.get('fre_summary_mm'))}",
        f"Translation spread mean / std / max (mm): {_metric_triplet(metrics.get('translation_delta_to_consensus_summary_mm'))}",
        f"Rotation spread mean / std / max (deg): {_metric_triplet(metrics.get('rotation_delta_to_consensus_summary_deg'))}",
        f"Robot-origin RMS / max distance to consensus (mm): {_fmt(spread.get('rms_distance_mm'))} / {_fmt(spread.get('max_distance_mm'))}",
        f"Robot-origin XYZ span (mm): x={_fmt(spread.get('span_x_mm'))}, y={_fmt(spread.get('span_y_mm'))}, z={_fmt(spread.get('span_z_mm'))}",
        "",
        "Saved plots:",
        "- registration_fre_histogram.png",
        "- registration_frame_origins.png",
        "- registration_transform_spread.png",
    ]
    invalid_runs = metrics.get("invalid_runs") or []
    if invalid_runs:
        lines.extend(["", "Skipped source artifacts:"])
        for row in invalid_runs:
            lines.append(f"- {row.get('path')}: {row.get('reason')}")
    return lines


def build_pivot_validation_summary_lines(*, metadata, summary, metrics: dict[str, Any]) -> list[str]:
    """Return human-readable pivot-validation summary text."""
    lines = [
        "Pivot Validation Summary",
        "Offline analysis of repeated saved pivot-calibration runs. Metrics quantify tip-offset spread only; no arbitrary orientation estimate is implied (all lengths in mm).",
        "",
        f"Run ID: {metadata.run_id}",
        f"Timestamp: {metadata.timestamp_utc}",
        f"Status: {summary.status}",
        f"Selected runs: {int(metrics.get('selected_run_count', 0) or 0)}",
        f"Valid runs: {int(metrics.get('valid_run_count', 0) or 0)}",
        f"Skipped runs: {int(metrics.get('invalid_run_count', 0) or 0)}",
        "",
        "Definitions:",
        "- Pivot RMSE is per-run solve residual quality.",
        "- Distance to consensus is cross-run repeatability spread of solved tip offsets.",
        "",
        f"Tip-offset norm mean / std / max (mm): {_metric_triplet(metrics.get('tip_norm_summary_mm'))}",
        f"Pivot solve RMSE mean / std / max (mm): {_metric_triplet(metrics.get('rmse_summary_mm'))}",
        f"Tip-offset deviation to consensus mean / std / max (mm): {_metric_triplet(metrics.get('distance_to_consensus_summary_mm'))}",
        f"Consensus tip-offset vector [x, y, z] (mm): {metrics.get('consensus_tip_vector_local_mm')}",
        "",
        "Saved plots:",
        "- pivot_tip_scatter.png",
        "- pivot_axis_histograms.png",
        "- pivot_quality_summary.png",
    ]
    invalid_runs = metrics.get("invalid_runs") or []
    if invalid_runs:
        lines.extend(["", "Skipped source artifacts:"])
        for row in invalid_runs:
            lines.append(f"- {row.get('path')}: {row.get('reason')}")
    return lines


def _write_rows_csv(
    path: Path,
    *,
    rows: list[dict[str, Any]],
    header_overrides: dict[str, str] | None = None,
) -> None:
    fieldnames: list[str] = []
    header_overrides = dict(header_overrides or {})
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(str(key))
    output_headers = [header_overrides.get(key, key) for key in fieldnames]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=output_headers or ["empty"])
        writer.writeheader()
        if not fieldnames:
            writer.writerow({"empty": ""})
            return
        for row in rows:
            writer.writerow(
                {
                    header_overrides.get(key, key): _csv_value(row.get(key))
                    for key in fieldnames
                }
            )


def _csv_value(value: Any) -> Any:
    if isinstance(value, (list, dict)):
        return str(value)
    return value


def _write_registration_fre_histogram(*, fre_plot_path: Path, metrics: dict[str, Any]) -> None:
    image = _new_publication_image(1180, 720)
    painter = QPainter(image)
    painter.setRenderHint(QPainter.Antialiasing, True)
    _draw_title(
        painter,
        QRectF(34.0, 24.0, 1112.0, 56.0),
        "REGISTRATION FRE DISTRIBUTION (MM)",
        "Histogram of per-run fiducial registration error (FRE) across the selected saved solves.",
    )
    chart_rect = QRectF(36.0, 104.0, 740.0, 560.0)
    summary_rect = QRectF(808.0, 104.0, 338.0, 560.0)
    _draw_panel(painter, chart_rect, "FRE Histogram")
    _draw_panel(painter, summary_rect, "Summary")
    values = [float(row.get("fre_mm")) for row in (metrics.get("per_run_rows") or []) if row.get("fre_mm") is not None]
    _draw_histogram(
        painter,
        chart_rect.adjusted(18.0, 38.0, -18.0, -28.0),
        values=values,
        color=QColor("#2563eb"),
        x_label="FRE (mm)",
        y_label="Count",
        empty_text="No valid FRE values were available.",
    )
    _draw_summary_pairs(
        painter,
        summary_rect.adjusted(16.0, 36.0, -16.0, -16.0),
        [
            ("Valid Runs", str(int(metrics.get("valid_run_count", 0) or 0))),
            ("Mean FRE (mm)", _metric_stat(metrics.get("fre_summary_mm"), "mean")),
            ("Std FRE (mm)", _metric_stat(metrics.get("fre_summary_mm"), "std")),
            ("Max FRE (mm)", _metric_stat(metrics.get("fre_summary_mm"), "max")),
            ("Mean dT (mm)", _metric_stat(metrics.get("translation_delta_to_consensus_summary_mm"), "mean")),
            ("Mean dR (deg)", _metric_stat(metrics.get("rotation_delta_to_consensus_summary_deg"), "mean")),
        ],
    )
    painter.end()
    image.save(str(fre_plot_path))


def _write_registration_origin_plot(*, origin_plot_path: Path, metrics: dict[str, Any]) -> None:
    image = _new_publication_image(1180, 760)
    painter = QPainter(image)
    painter.setRenderHint(QPainter.Antialiasing, True)
    _draw_title(
        painter,
        QRectF(34.0, 24.0, 1112.0, 56.0),
        "REGISTRATION FRAME ORIGINS (MM)",
        "Projected 3D scatter of solved robot/model-frame origins expressed in Aurora space with equal projected scaling.",
    )
    scatter_rect = QRectF(36.0, 104.0, 820.0, 600.0)
    summary_rect = QRectF(888.0, 104.0, 258.0, 600.0)
    _draw_panel(painter, scatter_rect, "Solved Frame Origins (mm)")
    _draw_panel(painter, summary_rect, "Spread")
    points = [
        np.asarray(row.get("robot_origin_in_aurora_mm") or [], dtype=float)
        for row in (metrics.get("per_run_rows") or [])
        if isinstance(row.get("robot_origin_in_aurora_mm"), list) and len(row.get("robot_origin_in_aurora_mm")) == 3
    ]
    _draw_projected_scatter(
        painter,
        scatter_rect.adjusted(18.0, 38.0, -18.0, -28.0),
        points=points,
        point_color=QColor("#0f766e"),
        point_radius=5.0,
        empty_text="No valid registration origins were available.",
        axis_note="Projected axes in mm (equal projected scale)",
    )
    spread = dict(metrics.get("robot_origin_spread_mm", {}) or {})
    _draw_summary_pairs(
        painter,
        summary_rect.adjusted(16.0, 36.0, -16.0, -16.0),
        [
            ("Std X (mm)", _fmt(spread.get("std_x_mm"))),
            ("Std Y (mm)", _fmt(spread.get("std_y_mm"))),
            ("Std Z (mm)", _fmt(spread.get("std_z_mm"))),
            ("RMS Dist (mm)", _fmt(spread.get("rms_distance_mm"))),
            ("Max Dist (mm)", _fmt(spread.get("max_distance_mm"))),
            ("Span X (mm)", _fmt(spread.get("span_x_mm"))),
            ("Span Y (mm)", _fmt(spread.get("span_y_mm"))),
            ("Span Z (mm)", _fmt(spread.get("span_z_mm"))),
        ],
    )
    painter.end()
    image.save(str(origin_plot_path))


def _write_registration_spread_plot(*, spread_plot_path: Path, metrics: dict[str, Any]) -> None:
    rows = list(metrics.get("per_run_rows") or [])
    labels = [f"R{index + 1}" for index in range(len(rows))]
    translation_values = [float(row.get("translation_delta_to_consensus_mm", 0.0) or 0.0) for row in rows]
    rotation_values = [float(row.get("rotation_delta_to_consensus_deg", 0.0) or 0.0) for row in rows]
    image = _new_publication_image(1180, 760)
    painter = QPainter(image)
    painter.setRenderHint(QPainter.Antialiasing, True)
    _draw_title(
        painter,
        QRectF(34.0, 24.0, 1112.0, 56.0),
        "REGISTRATION TRANSFORM SPREAD (MM / DEG)",
        "Per-run translation and rotation deltas relative to the consensus transform.",
    )
    top_rect = QRectF(36.0, 104.0, 1108.0, 280.0)
    bottom_rect = QRectF(36.0, 414.0, 1108.0, 280.0)
    _draw_panel(painter, top_rect, "Translation Delta To Consensus (mm)")
    _draw_panel(painter, bottom_rect, "Rotation Delta To Consensus (deg)")
    _draw_bar_chart(
        painter,
        top_rect.adjusted(18.0, 38.0, -18.0, -28.0),
        labels=labels,
        values=translation_values,
        color=QColor("#2563eb"),
        y_label="mm",
        empty_text="No valid runs were available.",
    )
    _draw_bar_chart(
        painter,
        bottom_rect.adjusted(18.0, 38.0, -18.0, -28.0),
        labels=labels,
        values=rotation_values,
        color=QColor("#dc2626"),
        y_label="deg",
        empty_text="No valid runs were available.",
    )
    painter.end()
    image.save(str(spread_plot_path))


def _write_pivot_scatter_plot(*, scatter_plot_path: Path, metrics: dict[str, Any]) -> None:
    image = _new_publication_image(1180, 760)
    painter = QPainter(image)
    painter.setRenderHint(QPainter.Antialiasing, True)
    _draw_title(
        painter,
        QRectF(34.0, 24.0, 1112.0, 56.0),
        "PIVOT TIP-OFFSET SCATTER (MM)",
        "Projected 3D scatter of per-run local tip-offset solutions with equal projected scaling.",
    )
    scatter_rect = QRectF(36.0, 104.0, 820.0, 600.0)
    summary_rect = QRectF(888.0, 104.0, 258.0, 600.0)
    _draw_panel(painter, scatter_rect, "Tip Offset Vectors (mm)")
    _draw_panel(painter, summary_rect, "Summary")
    points = [
        np.asarray(row.get("tip_vector_local_mm") or [], dtype=float)
        for row in (metrics.get("per_run_rows") or [])
        if isinstance(row.get("tip_vector_local_mm"), list) and len(row.get("tip_vector_local_mm")) == 3
    ]
    _draw_projected_scatter(
        painter,
        scatter_rect.adjusted(18.0, 38.0, -18.0, -28.0),
        points=points,
        point_color=QColor("#7c3aed"),
        point_radius=5.0,
        empty_text="No valid tip vectors were available.",
        axis_note="Projected axes in mm (equal projected scale)",
    )
    _draw_summary_pairs(
        painter,
        summary_rect.adjusted(16.0, 36.0, -16.0, -16.0),
        [
            ("Valid Runs", str(int(metrics.get("valid_run_count", 0) or 0))),
            ("Mean Norm (mm)", _metric_stat(metrics.get("tip_norm_summary_mm"), "mean")),
            ("Std Norm (mm)", _metric_stat(metrics.get("tip_norm_summary_mm"), "std")),
            ("Mean RMSE (mm)", _metric_stat(metrics.get("rmse_summary_mm"), "mean")),
            ("Mean dC (mm)", _metric_stat(metrics.get("distance_to_consensus_summary_mm"), "mean")),
            ("Consensus [x,y,z] mm", str(metrics.get("consensus_tip_vector_local_mm"))),
        ],
    )
    painter.end()
    image.save(str(scatter_plot_path))


def _write_pivot_axis_histograms(*, axis_plot_path: Path, metrics: dict[str, Any]) -> None:
    rows = list(metrics.get("per_run_rows") or [])
    values_by_axis = {
        "X": [float(row["tip_vector_local_mm"][0]) for row in rows if isinstance(row.get("tip_vector_local_mm"), list)],
        "Y": [float(row["tip_vector_local_mm"][1]) for row in rows if isinstance(row.get("tip_vector_local_mm"), list)],
        "Z": [float(row["tip_vector_local_mm"][2]) for row in rows if isinstance(row.get("tip_vector_local_mm"), list)],
    }
    image = _new_publication_image(1280, 780)
    painter = QPainter(image)
    painter.setRenderHint(QPainter.Antialiasing, True)
    _draw_title(
        painter,
        QRectF(34.0, 24.0, 1212.0, 56.0),
        "PIVOT AXIS DISTRIBUTIONS (MM)",
        "Per-axis histograms for saved tip-offset vectors.",
    )
    rects = [
        QRectF(36.0, 104.0, 388.0, 600.0),
        QRectF(446.0, 104.0, 388.0, 600.0),
        QRectF(856.0, 104.0, 388.0, 600.0),
    ]
    colors = [QColor("#2563eb"), QColor("#0f766e"), QColor("#dc2626")]
    for rect, (axis, values), color in zip(rects, values_by_axis.items(), colors, strict=False):
        _draw_panel(painter, rect, f"{axis} Offset (mm)")
        _draw_histogram(
            painter,
            rect.adjusted(18.0, 38.0, -18.0, -28.0),
            values=values,
            color=color,
            x_label=f"{axis} (mm)",
            y_label="Count",
            empty_text="No valid values were available.",
        )
    painter.end()
    image.save(str(axis_plot_path))


def _write_pivot_quality_plot(*, quality_plot_path: Path, metrics: dict[str, Any]) -> None:
    rows = list(metrics.get("per_run_rows") or [])
    labels = [f"R{index + 1}" for index in range(len(rows))]
    rmse_values = [float(row.get("rmse_mm", 0.0) or 0.0) for row in rows]
    distance_values = [float(row.get("distance_to_consensus_mm", 0.0) or 0.0) for row in rows]
    image = _new_publication_image(1180, 760)
    painter = QPainter(image)
    painter.setRenderHint(QPainter.Antialiasing, True)
    _draw_title(
        painter,
        QRectF(34.0, 24.0, 1112.0, 56.0),
        "PIVOT QUALITY SUMMARY (MM)",
        "Per-run pivot solve RMSE and tip-offset deviation to the consensus vector.",
    )
    top_rect = QRectF(36.0, 104.0, 1108.0, 280.0)
    bottom_rect = QRectF(36.0, 414.0, 1108.0, 280.0)
    _draw_panel(painter, top_rect, "RMSE (mm)")
    _draw_panel(painter, bottom_rect, "Distance To Consensus (mm)")
    _draw_bar_chart(
        painter,
        top_rect.adjusted(18.0, 38.0, -18.0, -28.0),
        labels=labels,
        values=rmse_values,
        color=QColor("#7c3aed"),
        y_label="mm",
        empty_text="No valid runs were available.",
    )
    _draw_bar_chart(
        painter,
        bottom_rect.adjusted(18.0, 38.0, -18.0, -28.0),
        labels=labels,
        values=distance_values,
        color=QColor("#2563eb"),
        y_label="mm",
        empty_text="No valid runs were available.",
    )
    painter.end()
    image.save(str(quality_plot_path))


def _draw_projected_scatter(
    painter: QPainter,
    rect: QRectF,
    *,
    points: list[np.ndarray],
    point_color: QColor,
    point_radius: float,
    empty_text: str,
    axis_note: str,
) -> None:
    if not points:
        painter.setFont(_body_font())
        painter.setPen(QColor("#64748b"))
        painter.drawText(rect, Qt.AlignCenter, empty_text)
        return
    projected = np.asarray([_project_point(point) for point in points], dtype=float)
    minimum = projected.min(axis=0)
    maximum = projected.max(axis=0)
    span = np.maximum(maximum - minimum, 1e-6)
    shared_span = float(max(span[0], span[1], 1e-6))
    painter.setPen(QPen(QColor("#cbd5e1"), 1.0))
    painter.drawRect(rect)
    for point in projected:
        x_norm = float((point[0] - minimum[0]) / shared_span)
        y_norm = float((point[1] - minimum[1]) / shared_span)
        x = rect.left() + (x_norm * rect.width())
        y = rect.bottom() - (y_norm * rect.height())
        painter.setPen(Qt.NoPen)
        painter.setBrush(point_color)
        painter.drawEllipse(QRectF(x - point_radius, y - point_radius, point_radius * 2.0, point_radius * 2.0))
    painter.setPen(QColor("#475569"))
    painter.setFont(_body_font())
    painter.drawText(rect.adjusted(8.0, 8.0, -8.0, -8.0), Qt.AlignLeft | Qt.AlignTop, axis_note)


def _new_publication_image(width: int, height: int):
    image = _new_image(width, height)
    # 300 dpi export target for thesis-ready raster figures.
    dots_per_meter = int(round(300.0 / 0.0254))
    image.setDotsPerMeterX(dots_per_meter)
    image.setDotsPerMeterY(dots_per_meter)
    return image


def _project_point(point: np.ndarray) -> tuple[float, float]:
    x, y, z = [float(value) for value in np.asarray(point, dtype=float).reshape(3)]
    return (x - (0.55 * y), z + (0.35 * y))


def _metric_triplet(summary: Any) -> str:
    summary = dict(summary or {})
    return f"{_fmt(summary.get('mean'))} / {_fmt(summary.get('std'))} / {_fmt(summary.get('max'))}"


def _metric_stat(summary: Any, key: str) -> str:
    summary = dict(summary or {})
    return _fmt(summary.get(key))


def _fmt(value: Any) -> str:
    if value is None:
        return "n/a"
    return f"{float(value):.3f}"

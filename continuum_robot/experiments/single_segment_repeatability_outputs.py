"""Saved artifacts for the single-segment repeatability experiment."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from continuum_robot.experiments.dataset_tools import extract_tip_or_tool_position_mm
from continuum_robot.experiments.tracker_timing_outputs import (
    _body_font,
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
from PySide6.QtGui import QColor, QPen, QPainter


def build_single_segment_repeatability_summary_pairs(*, metrics: dict[str, Any]) -> list[tuple[str, str]]:
    """Return compact rows for GUI/report summaries."""
    return [
        ("Protocol", "Legacy 17 target / all-other approaches"),
        ("Targets", str(int(metrics.get("target_count", 0) or 0))),
        ("Planned Captures", str(int(metrics.get("planned_capture_count", 0) or 0))),
        ("Valid Repeat Captures", str(int(metrics.get("valid_repeat_sample_count", 0) or 0))),
        ("Valid Approach Captures", str(int(metrics.get("valid_approach_sample_count", 0) or 0))),
        ("Rejected Captures", str(int(metrics.get("rejected_capture_count", 0) or 0))),
        ("Pose Frame", str(metrics.get("position_frame", "unknown") or "unknown")),
        ("Overall RMS", _fmt(metrics.get("overall_repeatability_rms_mm"), suffix=" mm")),
        ("Overall Max", _fmt(metrics.get("overall_max_deviation_mm"), suffix=" mm")),
        ("Path Dependence RMS", _fmt(metrics.get("path_dependence_rms_mm"), suffix=" mm")),
        ("Thesis Goal", f"<= {_fmt(metrics.get('thesis_goal_rms_mm'), suffix=' mm')}"),
        ("Goal Result", "PASS" if metrics.get("thesis_goal_pass") else "not met"),
    ]


def build_single_segment_repeatability_summary_lines(*, metadata, summary, metrics: dict[str, Any]) -> list[str]:
    """Return human-readable summary text for a repeatability run."""
    lines = [
        "Single-Segment Repeatability Summary",
        (
            "This run recreates the legacy 17-point single-segment protocol using the current "
            "tracker, servo, registration, and runtime-tip chain. Repeatability metrics are computed "
            "from repeat captures after returning to the desired target from every other target."
        ),
        "",
        f"Run ID: {metadata.run_id}",
        f"Timestamp: {metadata.timestamp_utc}",
        f"Status: {summary.status}",
    ]
    for label, value in build_single_segment_repeatability_summary_pairs(metrics=metrics):
        lines.append(f"{label}: {value}")
    per_target = sorted(
        (metrics.get("per_target_metrics", {}) or {}).values(),
        key=lambda item: int(item.get("target_index", 10**9)),
    )
    lines.extend(["", "Per-target repeat captures:"])
    for row in per_target:
        lines.append(
            f"- {row.get('label', 'target')} ({row.get('ring', 'unknown')}): "
            f"n={int(row.get('repeat_sample_count', 0) or 0)}, "
            f"RMSE={_fmt(row.get('spread_rms_mm'), suffix=' mm')}, "
            f"max={_fmt(row.get('max_deviation_mm'), suffix=' mm')}"
        )
    lines.extend(
        [
            "",
            "Interpretation:",
            "- This validates repeatability after startup calibration and pretension, not spatial accuracy by itself.",
            "- Use robot-frame metrics only when registration and 0A runtime tip calibration are accepted and active.",
            "- Path-dependence values summarize approach-conditioned deviations from each target centroid.",
        ]
    )
    return lines


def write_single_segment_repeatability_outputs(*, output_dir: Path, metadata, summary, samples) -> dict[str, Path]:
    """Write thesis-oriented figures and summary text for one run."""
    _ensure_plot_qt_app()
    output_dir = Path(output_dir)
    metrics = summary.experiment_metrics if isinstance(summary.experiment_metrics, dict) else {}
    summary_text_path = output_dir / "repeatability_summary.txt"
    clusters_path = output_dir / "repeatability_clusters.png"
    rmse_path = output_dir / "repeatability_rmse_summary.png"
    path_dependence_path = output_dir / "repeatability_path_dependence.png"
    summary_text_path.write_text(
        "\n".join(
            build_single_segment_repeatability_summary_lines(
                metadata=metadata,
                summary=summary,
                metrics=metrics,
            )
        ).strip()
        + "\n",
        encoding="utf-8",
    )
    _write_cluster_figure(clusters_path=clusters_path, samples=samples, metrics=metrics)
    _write_rmse_figure(rmse_path=rmse_path, metrics=metrics)
    _write_path_dependence_figure(path_dependence_path=path_dependence_path, metrics=metrics)
    return {
        "summary_text_path": summary_text_path,
        "clusters_path": clusters_path,
        "rmse_path": rmse_path,
        "path_dependence_path": path_dependence_path,
    }


def _write_cluster_figure(*, clusters_path: Path, samples, metrics: dict[str, Any]) -> None:
    image = _new_image(1280, 920)
    painter = QPainter(image)
    painter.setRenderHint(QPainter.Antialiasing, True)
    _draw_title(
        painter,
        QRectF(34.0, 24.0, 1212.0, 58.0),
        "SINGLE-SEGMENT REPEATABILITY CLUSTERS",
        "Accepted repeat captures by target. Metrics use repeat captures after approach from every other target.",
    )
    chart_rect = QRectF(36.0, 108.0, 820.0, 760.0)
    summary_rect = QRectF(888.0, 108.0, 356.0, 760.0)
    _draw_panel(painter, chart_rect, "Robot-Frame Tip Clusters (XY)")
    _draw_panel(painter, summary_rect, "Run Summary")
    _draw_cluster_scatter(painter, chart_rect.adjusted(22.0, 42.0, -22.0, -28.0), samples=samples, metrics=metrics)
    _draw_summary_pairs(
        painter,
        summary_rect.adjusted(16.0, 38.0, -16.0, -16.0),
        build_single_segment_repeatability_summary_pairs(metrics=metrics),
    )
    painter.end()
    image.save(str(clusters_path))


def _write_rmse_figure(*, rmse_path: Path, metrics: dict[str, Any]) -> None:
    per_target = _ordered_per_target(metrics)
    labels = [str(row.get("label", f"T{index:02d}")) for index, row in enumerate(per_target)]
    rmse_values = [row.get("spread_rms_mm") for row in per_target]
    max_values = [row.get("max_deviation_mm") for row in per_target]
    image = _new_image(1280, 720)
    painter = QPainter(image)
    painter.setRenderHint(QPainter.Antialiasing, True)
    _draw_title(
        painter,
        QRectF(34.0, 24.0, 1212.0, 58.0),
        "PER-TARGET REPEATABILITY SUMMARY",
        "RMSE and maximum deviation from each target centroid for accepted repeat captures.",
    )
    rmse_rect = QRectF(36.0, 112.0, 586.0, 540.0)
    max_rect = QRectF(658.0, 112.0, 586.0, 540.0)
    _draw_panel(painter, rmse_rect, "Repeatability RMSE")
    _draw_panel(painter, max_rect, "Maximum Deviation")
    _draw_bar_chart(
        painter,
        rmse_rect.adjusted(14.0, 38.0, -14.0, -20.0),
        labels=labels,
        values=rmse_values,
        color=QColor(COLORS.scene_truth),
        y_label="RMSE (mm)",
        empty_text="No accepted repeat captures were available.",
    )
    _draw_bar_chart(
        painter,
        max_rect.adjusted(14.0, 38.0, -14.0, -20.0),
        labels=labels,
        values=max_values,
        color=QColor(COLORS.scene_residual),
        y_label="Max deviation (mm)",
        empty_text="No accepted repeat captures were available.",
    )
    painter.end()
    image.save(str(rmse_path))


def _write_path_dependence_figure(*, path_dependence_path: Path, metrics: dict[str, Any]) -> None:
    per_target = _ordered_per_target(metrics)
    labels = [str(row.get("label", f"T{index:02d}")) for index, row in enumerate(per_target)]
    path_entries = list(metrics.get("path_dependence_by_approach", []) or [])
    max_by_target: dict[int, float] = {}
    mean_by_target: dict[int, float] = {}
    for row in per_target:
        target_index = int(row.get("target_index", -1))
        values = [
            float(entry.get("deviation_mm", 0.0) or 0.0)
            for entry in path_entries
            if int(entry.get("target_index", -2)) == target_index
        ]
        if values:
            max_by_target[target_index] = float(np.max(values))
            mean_by_target[target_index] = float(np.mean(values))
    max_values = [max_by_target.get(int(row.get("target_index", -1))) for row in per_target]
    mean_values = [mean_by_target.get(int(row.get("target_index", -1))) for row in per_target]
    image = _new_image(1280, 720)
    painter = QPainter(image)
    painter.setRenderHint(QPainter.Antialiasing, True)
    _draw_title(
        painter,
        QRectF(34.0, 24.0, 1212.0, 58.0),
        "PATH-DEPENDENCE SUMMARY",
        "Approach-conditioned deviation from each target centroid. One repeat capture is recorded per approach target.",
    )
    mean_rect = QRectF(36.0, 112.0, 586.0, 540.0)
    max_rect = QRectF(658.0, 112.0, 586.0, 540.0)
    _draw_panel(painter, mean_rect, "Mean Approach-Conditioned Deviation")
    _draw_panel(painter, max_rect, "Max Approach-Conditioned Deviation")
    _draw_bar_chart(
        painter,
        mean_rect.adjusted(14.0, 38.0, -14.0, -20.0),
        labels=labels,
        values=mean_values,
        color=QColor(COLORS.scene_measurement),
        y_label="Mean deviation (mm)",
        empty_text="No path-dependence deviations were available.",
    )
    _draw_bar_chart(
        painter,
        max_rect.adjusted(14.0, 38.0, -14.0, -20.0),
        labels=labels,
        values=max_values,
        color=QColor(COLORS.scene_residual),
        y_label="Max deviation (mm)",
        empty_text="No path-dependence deviations were available.",
    )
    painter.end()
    image.save(str(path_dependence_path))


def _draw_cluster_scatter(painter: QPainter, rect: QRectF, *, samples, metrics: dict[str, Any]) -> None:
    points = _repeat_points_by_target(samples=samples)
    centroids = {
        int(row.get("target_index", -1)): row.get("centroid_mm")
        for row in _ordered_per_target(metrics)
        if isinstance(row.get("centroid_mm"), list) and len(row.get("centroid_mm")) >= 2
    }
    combined_xy: list[tuple[float, float]] = []
    for target_points in points.values():
        combined_xy.extend((float(point[0]), float(point[1])) for point in target_points)
    combined_xy.extend((float(point[0]), float(point[1])) for point in centroids.values())
    if not combined_xy:
        painter.setFont(_body_font())
        painter.setPen(QColor("#64748b"))
        painter.drawText(rect, Qt.AlignCenter, "No accepted robot-frame repeat captures were available.")
        return
    xs = [point[0] for point in combined_xy]
    ys = [point[1] for point in combined_xy]
    min_x, max_x = _expand_range(min(xs), max(xs), minimum_span=5.0, pad_fraction=0.12)
    min_y, max_y = _expand_range(min(ys), max(ys), minimum_span=5.0, pad_fraction=0.12)
    painter.setPen(QPen(QColor("#e2e8f0"), 1.0))
    painter.drawLine(QPointF(rect.left(), rect.bottom()), QPointF(rect.right(), rect.bottom()))
    painter.drawLine(QPointF(rect.left(), rect.top()), QPointF(rect.left(), rect.bottom()))
    palette = [
        QColor(COLORS.scene_truth),
        QColor(COLORS.scene_measurement),
        QColor(COLORS.scene_tip),
        QColor(COLORS.scene_residual),
        QColor(COLORS.selection_bg),
    ]
    per_target = {int(row.get("target_index", -1)): row for row in _ordered_per_target(metrics)}
    for index, target_index in enumerate(sorted(points)):
        color = palette[index % len(palette)]
        painter.setBrush(color)
        painter.setPen(Qt.NoPen)
        for point in points[target_index]:
            painter.drawEllipse(
                QPointF(
                    _scale_x(rect, float(point[0]), min_x, max_x),
                    _scale_y(rect, float(point[1]), min_y, max_y),
                ),
                3.0,
                3.0,
            )
        centroid = centroids.get(target_index)
        if centroid:
            center = QPointF(
                _scale_x(rect, float(centroid[0]), min_x, max_x),
                _scale_y(rect, float(centroid[1]), min_y, max_y),
            )
            painter.setPen(QPen(QColor("#0f172a"), 1.4))
            painter.setBrush(QColor("#ffffff"))
            painter.drawRect(QRectF(center.x() - 4.5, center.y() - 4.5, 9.0, 9.0))
            painter.setPen(QColor("#334155"))
            painter.setFont(_body_font())
            label = str(per_target.get(target_index, {}).get("label", f"T{target_index:02d}"))
            rmse = per_target.get(target_index, {}).get("spread_rms_mm")
            painter.drawText(QPointF(center.x() + 6.0, center.y() - 4.0), f"{label} {_fmt(rmse, suffix=' mm')}")
    painter.setPen(QColor("#475569"))
    painter.setFont(_body_font())
    painter.drawText(QRectF(rect.left(), rect.bottom() + 4.0, rect.width(), 18.0), Qt.AlignCenter, "X (mm)")
    painter.drawText(QRectF(rect.left(), rect.top() - 2.0, rect.width(), 18.0), Qt.AlignRight, "Y (mm)")


def _repeat_points_by_target(*, samples) -> dict[int, list[list[float]]]:
    points: dict[int, list[list[float]]] = {}
    for sample in samples:
        if sample.phase != "repeat":
            continue
        if not bool(sample.extra.get("capture_accepted", True)):
            continue
        position, frame = extract_tip_or_tool_position_mm(sample, tool_id="0A", prefer_robot_frame=True)
        if position is None or frame != "robot":
            continue
        target_index = int(sample.target_index if sample.target_index is not None else -1)
        points.setdefault(target_index, []).append([float(value) for value in position])
    return points


def _ordered_per_target(metrics: dict[str, Any]) -> list[dict[str, Any]]:
    return sorted(
        [dict(value) for value in (metrics.get("per_target_metrics", {}) or {}).values()],
        key=lambda item: int(item.get("target_index", 10**9)),
    )


def _expand_range(low: float, high: float, *, minimum_span: float, pad_fraction: float) -> tuple[float, float]:
    span = max(float(high - low), float(minimum_span))
    pad = span * float(pad_fraction)
    center = (float(low) + float(high)) / 2.0
    half = (span / 2.0) + pad
    return center - half, center + half


def _scale_x(rect: QRectF, value: float, low: float, high: float) -> float:
    if high <= low:
        return rect.center().x()
    return rect.left() + ((float(value) - low) / (high - low)) * rect.width()


def _scale_y(rect: QRectF, value: float, low: float, high: float) -> float:
    if high <= low:
        return rect.center().y()
    return rect.bottom() - ((float(value) - low) / (high - low)) * rect.height()

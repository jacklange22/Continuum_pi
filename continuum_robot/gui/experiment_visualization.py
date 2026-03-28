"""Pure-Python visualization models for the experiment workspace."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from continuum_robot.experiments.dataset_tools import extract_tip_or_tool_position_mm


@dataclass
class ScatterSeries3D:
    """One 3D scatter series."""

    name: str
    color_hex: str
    points_xyz: list[tuple[float, float, float]] = field(default_factory=list)
    point_size: float = 0.12
    mesh: str = "sphere"


@dataclass
class ChartModel:
    """Simple chart payload for QtCharts rendering."""

    kind: str
    title: str
    x_title: str
    y_title: str
    categories: list[str] = field(default_factory=list)
    values: list[float] = field(default_factory=list)
    points_xy: list[tuple[float, float]] = field(default_factory=list)
    color_hex: str = "#2563eb"


@dataclass
class VisualizationModel:
    """Full GUI-facing visualization payload."""

    series_3d: list[ScatterSeries3D] = field(default_factory=list)
    charts: list[ChartModel] = field(default_factory=list)
    summary_lines: list[str] = field(default_factory=list)


_PALETTE = [
    "#0f766e",
    "#2563eb",
    "#dc2626",
    "#7c3aed",
    "#d97706",
    "#059669",
    "#db2777",
    "#1d4ed8",
]


def build_visualization_model(
    *,
    experiment_name: str,
    samples,
    metrics: dict[str, Any] | None,
    config_payload: dict[str, Any] | None,
    color_mode: str,
    show_centroids: bool,
    show_truth: bool,
) -> VisualizationModel:
    """Build a visualization payload from canonical samples and summary metrics."""
    metrics = dict(metrics or {})
    config_payload = dict(config_payload or {})
    if experiment_name == "repeatability_dataset":
        return _build_repeatability_model(
            samples=samples,
            metrics=metrics,
            tool_id=str(config_payload.get("tool_id", "0A")),
            color_mode=color_mode,
            show_centroids=show_centroids,
        )
    if experiment_name == "aurora_grid_accuracy":
        return _build_grid_model(
            samples=samples,
            metrics=metrics,
            tool_id=str(config_payload.get("tool_id", "0B")),
            color_mode=color_mode,
            show_truth=show_truth,
        )
    if experiment_name == "pivot_calibration":
        return _build_pivot_model(
            samples=samples,
            metrics=metrics,
            tool_id=str(config_payload.get("tool_id", "0B")),
            color_mode=color_mode,
        )
    return VisualizationModel(summary_lines=["No visualization is available for the selected experiment."])


def _build_repeatability_model(*, samples, metrics: dict[str, Any], tool_id: str, color_mode: str, show_centroids: bool) -> VisualizationModel:
    measurement_samples = [sample for sample in samples if sample.phase == "sample"]
    grouped: dict[str, list[tuple[float, float, float]]] = {}
    distances_mm: list[float] = []
    centroid_map = {
        key: np.asarray(value.get("centroid_mm", []), dtype=float)
        for key, value in (metrics.get("per_target_metrics", {}) or {}).items()
        if value.get("centroid_mm")
    }
    for sample in measurement_samples:
        position, _ = extract_tip_or_tool_position_mm(sample, tool_id=tool_id, prefer_robot_frame=True)
        if position is None:
            continue
        key = _repeatability_group_key(sample=sample, color_mode=color_mode)
        grouped.setdefault(key, []).append(tuple(float(value) for value in position))
        center = centroid_map.get(str(sample.target_index))
        if center is not None and center.shape == (3,):
            distances_mm.append(float(np.linalg.norm(np.asarray(position, dtype=float) - center)))

    series = _grouped_series(grouped, base_label="Samples")
    if show_centroids:
        centroid_points = [
            tuple(float(value) for value in point_metrics["centroid_mm"])
            for _, point_metrics in sorted((metrics.get("per_target_metrics", {}) or {}).items(), key=lambda item: int(item[0]))
            if point_metrics.get("centroid_mm")
        ]
        if centroid_points:
            series.append(
                ScatterSeries3D(
                    name="Centroids",
                    color_hex="#111827",
                    points_xyz=centroid_points,
                    point_size=0.18,
                    mesh="cube",
                )
            )

    per_target = metrics.get("per_target_metrics", {}) or {}
    spread_chart = ChartModel(
        kind="bar",
        title="Per-Target Spread",
        x_title="Target",
        y_title="Spread RMS (mm)",
        categories=[str(target) for target in sorted(per_target, key=int)],
        values=[float(per_target[target]["spread_rms_mm"]) for target in sorted(per_target, key=int)],
        color_hex="#2563eb",
    )
    histogram = _histogram_chart(
        title="Repeatability Error Distribution",
        values=distances_mm,
        x_title="Distance To Target Centroid (mm)",
        y_title="Count",
        color_hex="#0f766e",
    )
    summary_lines = [
        f"status={metrics.get('status', 'unknown')}",
        f"position_frame={metrics.get('position_frame', 'unknown')}",
        f"valid_samples={metrics.get('valid_sample_count', 0)}",
        f"invalid_samples={metrics.get('invalid_sample_count', 0)}",
        f"overall_repeatability_rms_mm={_fmt(metrics.get('overall_repeatability_rms_mm'))}",
    ]
    return VisualizationModel(
        series_3d=series,
        charts=[spread_chart, histogram],
        summary_lines=summary_lines,
    )


def _build_grid_model(*, samples, metrics: dict[str, Any], tool_id: str, color_mode: str, show_truth: bool) -> VisualizationModel:
    measurement_samples = [sample for sample in samples if sample.phase == "sample"]
    grouped: dict[str, list[tuple[float, float, float]]] = {}
    truth_points: list[tuple[float, float, float]] = []
    for sample in measurement_samples:
        position, _ = extract_tip_or_tool_position_mm(sample, tool_id=tool_id, prefer_robot_frame=True)
        if position is None:
            continue
        key = _grid_group_key(sample=sample, color_mode=color_mode)
        grouped.setdefault(key, []).append(tuple(float(value) for value in position))
        truth_point = sample.extra.get("truth_point_mm")
        if isinstance(truth_point, list) and len(truth_point) == 3:
            truth_points.append(tuple(float(value) for value in truth_point))
    series = _grouped_series(grouped, base_label="Measured")
    if show_truth and truth_points:
        unique_truth = sorted(set(truth_points))
        series.append(
            ScatterSeries3D(
                name="Truth Grid",
                color_hex="#111827",
                points_xyz=list(unique_truth),
                point_size=0.18,
                mesh="cube",
            )
        )
    per_point = metrics.get("per_point_metrics", {}) or {}
    point_rms = metrics.get("pointwise_rms_error_mm", {}) or {}
    point_chart = ChartModel(
        kind="bar",
        title="Per-Point RMS Error",
        x_title="Grid Point",
        y_title="RMS Error (mm)",
        categories=[str(point) for point in sorted(point_rms, key=int)],
        values=[float(point_rms[point]) for point in sorted(point_rms, key=int)],
        color_hex="#dc2626",
    )
    bias = [float(value) for value in (metrics.get("per_axis_bias_mm") or [0.0, 0.0, 0.0])]
    bias_chart = ChartModel(
        kind="bar",
        title="Per-Axis Bias",
        x_title="Axis",
        y_title="Bias (mm)",
        categories=["X", "Y", "Z"],
        values=bias,
        color_hex="#2563eb",
    )
    summary_lines = [
        f"status={metrics.get('status', 'unknown')}",
        f"overall_rms_error_mm={_fmt(metrics.get('overall_rms_error_mm'))}",
        f"outlier_count={metrics.get('outlier_count', 0)}",
        f"registration_available={metrics.get('registration_available', False)}",
        f"tip_calibration_available={metrics.get('tip_calibration_available', False)}",
        f"points_summarized={len(per_point)}",
    ]
    return VisualizationModel(
        series_3d=series,
        charts=[point_chart, bias_chart],
        summary_lines=summary_lines,
    )


def _build_pivot_model(*, samples, metrics: dict[str, Any], tool_id: str, color_mode: str) -> VisualizationModel:
    sample_points: list[tuple[float, float, float]] = []
    inlier_mask = [bool(value) for value in (metrics.get("pivot_inlier_mask") or [])]
    grouped: dict[str, list[tuple[float, float, float]]] = {}
    for index, sample in enumerate(samples):
        position, _ = extract_tip_or_tool_position_mm(sample, tool_id=tool_id, prefer_robot_frame=False)
        if position is None:
            continue
        sample_points.append(tuple(float(value) for value in position))
        if color_mode == "inlier_outlier" and index < len(inlier_mask):
            key = "Inlier" if inlier_mask[index] else "Outlier"
        else:
            key = sample.phase or "sample"
        grouped.setdefault(key, []).append(tuple(float(value) for value in position))
    series = _grouped_series(grouped, base_label="Pivot")
    residual_rows = metrics.get("pivot_residuals_mm") or []
    residual_norms = [
        float(np.linalg.norm(np.asarray(row, dtype=float)))
        for row in residual_rows
        if isinstance(row, list) and len(row) == 3
    ]
    residual_hist = _histogram_chart(
        title="Pivot Residual Distribution",
        values=residual_norms,
        x_title="Residual Norm (mm)",
        y_title="Count",
        color_hex="#7c3aed",
    )
    inlier_count = int(metrics.get("sample_count_used", 0))
    rejected_count = int(metrics.get("sample_count_rejected", 0))
    inlier_chart = ChartModel(
        kind="bar",
        title="Inlier / Outlier Summary",
        x_title="Class",
        y_title="Count",
        categories=["Inliers", "Rejected"],
        values=[float(inlier_count), float(rejected_count)],
        color_hex="#0f766e",
    )
    summary_lines = [
        f"status={metrics.get('status', 'unknown')}",
        f"rmse_mm={_fmt(metrics.get('rmse_mm'))}",
        f"sample_count_total={metrics.get('sample_count_total', len(sample_points))}",
        f"sample_count_used={inlier_count}",
        f"sample_count_rejected={rejected_count}",
        f"tip_vector_local_mm={metrics.get('tip_vector_local_mm')}",
    ]
    return VisualizationModel(
        series_3d=series,
        charts=[residual_hist, inlier_chart],
        summary_lines=summary_lines,
    )


def _repeatability_group_key(*, sample, color_mode: str) -> str:
    if color_mode == "phase":
        return sample.phase or "sample"
    if color_mode == "revisit_index":
        return f"Revisit {sample.revisit_index}"
    if color_mode == "validity":
        return "Valid" if "tracker_data_stale" not in sample.status_flags else "Stale"
    return f"Target {sample.target_index}"


def _grid_group_key(*, sample, color_mode: str) -> str:
    if color_mode == "phase":
        return sample.phase or "sample"
    if color_mode == "revisit_index":
        return f"Rep {sample.revisit_index}"
    if color_mode == "validity":
        return "Valid" if "tracker_data_stale" not in sample.status_flags else "Stale"
    return f"Point {sample.target_index}"


def _grouped_series(grouped: dict[str, list[tuple[float, float, float]]], *, base_label: str) -> list[ScatterSeries3D]:
    series: list[ScatterSeries3D] = []
    for index, key in enumerate(sorted(grouped)):
        points = grouped[key]
        if not points:
            continue
        mesh = "cube" if "Centroid" in key or "Truth" in key else "sphere"
        series.append(
            ScatterSeries3D(
                name=f"{base_label}: {key}",
                color_hex=_PALETTE[index % len(_PALETTE)],
                points_xyz=points,
                point_size=0.12 if mesh == "sphere" else 0.18,
                mesh=mesh,
            )
        )
    return series


def _histogram_chart(*, title: str, values: list[float], x_title: str, y_title: str, color_hex: str) -> ChartModel:
    if not values:
        return ChartModel(kind="bar", title=title, x_title=x_title, y_title=y_title, categories=[], values=[], color_hex=color_hex)
    hist, edges = np.histogram(np.asarray(values, dtype=float), bins=min(8, max(3, len(values))))
    categories = [f"{edges[index]:.2f}-{edges[index + 1]:.2f}" for index in range(len(hist))]
    return ChartModel(
        kind="bar",
        title=title,
        x_title=x_title,
        y_title=y_title,
        categories=categories,
        values=[float(value) for value in hist.tolist()],
        color_hex=color_hex,
    )


def _fmt(value: Any) -> str:
    if value is None:
        return "n/a"
    return f"{float(value):.3f}"

"""Pure-Python visualization models for the experiment workspace."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from continuum_robot.experiments.dataset_tools import extract_tip_or_tool_position_mm
from continuum_robot.experiments.pretension_validation_outputs import extract_pretension_trace_points
from continuum_robot.gui.theme import COLORS, chart_palette
from continuum_robot.tracking.timing_benchmark import (
    compute_servo_sync_summary,
    compute_tracker_timing_summary,
    extract_servo_timing_records,
    extract_tracker_timing_records,
)


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
    caption: str = ""
    categories: list[str] = field(default_factory=list)
    values: list[float] = field(default_factory=list)
    points_xy: list[tuple[float, float]] = field(default_factory=list)
    color_hex: str = COLORS.selection_bg


@dataclass
class VisualizationModel:
    """Full GUI-facing visualization payload."""

    series_3d: list[ScatterSeries3D] = field(default_factory=list)
    charts: list[ChartModel] = field(default_factory=list)
    summary_lines: list[str] = field(default_factory=list)


_PALETTE = chart_palette()


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
            acceptance=config_payload.get("acceptance", {}),
        )
    if experiment_name == "aurora_grid_accuracy":
        return _build_grid_model(
            samples=samples,
            metrics=metrics,
            tool_id=str(config_payload.get("tool_id", "0B")),
            color_mode=color_mode,
            show_centroids=show_centroids,
            show_truth=show_truth,
            acceptance=config_payload.get("acceptance", {}),
        )
    if experiment_name == "pivot_calibration":
        return _build_pivot_model(
            samples=samples,
            metrics=metrics,
            tool_id=str(config_payload.get("tool_id", "0B")),
            color_mode=color_mode,
            acceptance=config_payload.get("acceptance", {}),
        )
    if experiment_name == "pretension_validation":
        return _build_pretension_validation_model(samples=samples, metrics=metrics)
    if experiment_name == "tracker_timing_validation":
        return _build_tracker_timing_model(samples=samples, metrics=metrics, config_payload=config_payload)
    return _build_generic_model(
        experiment_name=experiment_name,
        samples=samples,
        metrics=metrics,
        tool_id=str(config_payload.get("tool_id") or config_payload.get("tracker_tool_id") or "0A"),
    )


def _build_repeatability_model(
    *,
    samples,
    metrics: dict[str, Any],
    tool_id: str,
    color_mode: str,
    show_centroids: bool,
    acceptance: dict[str, Any],
) -> VisualizationModel:
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
    ordered_target_metrics = sorted(
        (metrics.get("per_target_metrics", {}) or {}).values(),
        key=lambda item: int(item.get("target_index", 10**9)),
    )
    if show_centroids:
        centroid_points = [
            tuple(float(value) for value in point_metrics["centroid_mm"])
            for point_metrics in ordered_target_metrics
            if point_metrics.get("centroid_mm")
        ]
        if centroid_points:
            series.append(
                ScatterSeries3D(
                    name="Centroids",
                    color_hex=COLORS.text_secondary,
                    points_xyz=centroid_points,
                    point_size=0.18,
                    mesh="cube",
                )
            )

    per_target = metrics.get("per_target_metrics", {}) or {}
    spread_chart = ChartModel(
        kind="bar",
        title="Per-Target Repeatability RMS",
        x_title="Target",
        y_title="Repeatability RMS (mm)",
        caption="Each bar shows the RMS spread for one revisited target after repeated returns from different prior states.",
        categories=[str(point_metrics.get("label", f"T{index + 1:02d}")) for index, point_metrics in enumerate(ordered_target_metrics)],
        values=[float(point_metrics.get("spread_rms_mm", 0.0) or 0.0) for point_metrics in ordered_target_metrics],
        color_hex=COLORS.scene_truth,
    )
    max_chart = ChartModel(
        kind="bar",
        title="Per-Target Max Deviation",
        x_title="Target",
        y_title="Max Deviation (mm)",
        caption="Maximum deviation from the target centroid for each revisited target.",
        categories=[str(point_metrics.get("label", f"T{index + 1:02d}")) for index, point_metrics in enumerate(ordered_target_metrics)],
        values=[float(point_metrics.get("max_deviation_mm", 0.0) or 0.0) for point_metrics in ordered_target_metrics],
        color_hex=COLORS.scene_residual,
    )
    charts: list[ChartModel] = [spread_chart, max_chart]
    group_metrics = metrics.get("group_metrics", {}) or {}
    group_categories: list[str] = []
    group_values: list[float] = []
    axis_groups = group_metrics.get("axis_class", {}) or {}
    magnitude_groups = group_metrics.get("magnitude_class", {}) or {}
    for label, value in sorted(axis_groups.items()):
        group_categories.append(f"axis:{label}")
        group_values.append(float(value.get("mean_target_rms_mm", 0.0) or 0.0))
    for label, value in sorted(magnitude_groups.items()):
        group_categories.append(f"mag:{label}")
        group_values.append(float(value.get("mean_target_rms_mm", 0.0) or 0.0))
    if group_categories:
        charts.append(
            ChartModel(
                kind="bar",
                title="Grouped Mean Target RMS",
                x_title="Target Group",
                y_title="Mean Target RMS (mm)",
                caption="Legacy-inspired grouped comparison across on-axis/off-axis and low/high-magnitude target classes when those classes are defined.",
                categories=group_categories,
                values=group_values,
                color_hex=COLORS.scene_measurement,
            )
        )
    else:
        charts.append(
            _histogram_chart(
                title="Repeatability Error Distribution",
                values=distances_mm,
                x_title="Distance To Target Centroid (mm)",
                y_title="Count",
                caption="Histogram of sample distances from each target centroid.",
                color_hex=COLORS.scene_measurement,
            )
        )
    acceptance_lines = _acceptance_lines(
        experiment_name="repeatability_dataset",
        metrics=metrics,
        acceptance=acceptance,
    )
    summary_lines = [
        f"Run status: {metrics.get('status', 'unknown')}",
        f"Pose frame: {metrics.get('position_frame', 'unknown')}",
        f"Targets summarized: {metrics.get('target_count', len(per_target))}",
        f"Target revisits: {metrics.get('visit_count', 0)}",
        f"Valid samples: {metrics.get('valid_sample_count', 0)}",
        f"Invalid samples: {metrics.get('invalid_sample_count', 0)}",
        f"Overall repeatability RMS: {_fmt(metrics.get('overall_repeatability_rms_mm'))} mm",
        f"Overall max deviation: {_fmt(metrics.get('overall_max_deviation_mm'))} mm",
        f"Path-dependence RMS shift: {_fmt(metrics.get('path_dependence_rms_mm'))} mm",
        (
            "Thesis target (< 1.000 mm): PASS"
            if metrics.get("thesis_goal_pass")
            else "Thesis target (< 1.000 mm): not yet met"
        ),
    ]
    summary_lines.extend(acceptance_lines)
    return VisualizationModel(
        series_3d=series,
        charts=charts,
        summary_lines=summary_lines,
    )


def _build_grid_model(
    *,
    samples,
    metrics: dict[str, Any],
    tool_id: str,
    color_mode: str,
    show_centroids: bool,
    show_truth: bool,
    acceptance: dict[str, Any],
) -> VisualizationModel:
    _ = samples, tool_id, color_mode
    per_point = metrics.get("per_point_metrics", {}) or {}
    ordered_labels = sorted(per_point, key=_label_sort_key)
    truth_points = [
        tuple(float(value) for value in per_point[label].get("truth_point_mm", []))
        for label in ordered_labels
        if isinstance(per_point[label].get("truth_point_mm"), list) and len(per_point[label]["truth_point_mm"]) == 3
    ]
    aligned_centroids = [
        tuple(float(value) for value in per_point[label].get("aligned_centroid_truth_mm", []))
        for label in ordered_labels
        if isinstance(per_point[label].get("aligned_centroid_truth_mm"), list)
        and len(per_point[label]["aligned_centroid_truth_mm"]) == 3
    ]
    series: list[ScatterSeries3D] = []
    if show_truth and truth_points:
        series.append(
            ScatterSeries3D(
                name="Truth Grid",
                color_hex=COLORS.text_secondary,
                points_xyz=truth_points,
                point_size=0.18,
                mesh="cube",
            )
        )
    if show_centroids and aligned_centroids:
        series.append(
            ScatterSeries3D(
                name="Aligned Measured Centroids",
                color_hex=COLORS.scene_truth,
                points_xyz=aligned_centroids,
                point_size=0.16,
                mesh="sphere",
            )
        )
    point_rms = metrics.get("per_point_residual_mm") or metrics.get("pointwise_rms_error_mm") or {}
    point_chart = ChartModel(
        kind="bar",
        title="Per-Point Residual",
        x_title="Grid Point",
        y_title="Residual (mm)",
        caption="Residual norm for each labeled point after best-fit rigid alignment of the ideal grid to the measured centroids.",
        categories=[str(point) for point in sorted(point_rms, key=_label_sort_key)],
        values=[float(point_rms[point]) for point in sorted(point_rms, key=_label_sort_key)],
        color_hex=COLORS.scene_residual,
    )
    spread_chart = ChartModel(
        kind="bar",
        title="Within-Point Spread",
        x_title="Grid Point",
        y_title="Spread RMS (mm)",
        caption="RMS spread of accepted samples around each point centroid before the global alignment solve.",
        categories=ordered_labels,
        values=[
            float(per_point[label].get("sample_spread_rms_mm", 0.0) or 0.0)
            for label in ordered_labels
        ],
        color_hex=COLORS.scene_measurement,
    )
    acceptance_lines = _acceptance_lines(
        experiment_name="aurora_grid_accuracy",
        metrics=metrics,
        acceptance=acceptance,
    )
    point_count_captured = int(metrics.get("point_count_captured", 0) or 0)
    point_count_total = int(metrics.get("point_count_total", len(ordered_labels)) or 0)
    point_count_complete = int(metrics.get("point_count_complete", 0) or 0)
    point_count_partial = int(metrics.get("point_count_partial", 0) or 0)
    point_count_not_started = int(metrics.get("point_count_not_started", 0) or 0)
    position_sources = metrics.get("position_source_counts", {}) or {}
    position_source_summary = ", ".join(
        f"{str(source)}={int(count)}"
        for source, count in sorted(position_sources.items())
    ) or "n/a"
    summary_lines = [
        f"Run status: {metrics.get('status', 'unknown')}",
        f"Coverage: {point_count_captured} / {point_count_total} labeled points captured",
        f"Point status: {point_count_complete} complete, {point_count_partial} partial, {point_count_not_started} not started",
        f"Solve readiness: {metrics.get('alignment_ready_reason', 'n/a')}",
        (
            f"Raw / accepted / rejected samples: "
            f"{int(metrics.get('raw_sample_count', 0) or 0)} / "
            f"{int(metrics.get('accepted_sample_count', 0) or 0)} / "
            f"{int(metrics.get('rejected_sample_count', metrics.get('outlier_count', 0)) or 0)}"
        ),
        f"Aligned RMS residual: {_fmt(metrics.get('overall_rms_residual_mm') or metrics.get('overall_rms_error_mm'))} mm",
        f"Max residual: {_fmt(metrics.get('max_residual_mm'))} mm",
        f"Mean within-point spread: {_fmt(metrics.get('mean_within_point_spread_mm'))} mm",
        f"Max within-point spread: {_fmt(metrics.get('max_within_point_spread_mm'))} mm",
        f"Aligned points: {metrics.get('point_count_aligned', 0)}",
        f"Position source counts: {position_source_summary}",
        (
            "Tip calibration used: yes"
            if metrics.get("tip_calibration_used")
            else ("Tip calibration used: fallback only" if metrics.get("coil_origin_fallback_used") else "Tip calibration used: no")
        ),
        f"Points summarized: {len(ordered_labels)}",
    ]
    summary_lines.extend(acceptance_lines)
    return VisualizationModel(
        series_3d=series,
        charts=[point_chart, spread_chart],
        summary_lines=summary_lines,
    )


def _build_pivot_model(
    *,
    samples,
    metrics: dict[str, Any],
    tool_id: str,
    color_mode: str,
    acceptance: dict[str, Any],
) -> VisualizationModel:
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
        caption="Distribution of point-to-pivot residuals after outlier rejection.",
        color_hex=COLORS.scene_residual,
    )
    inlier_count = int(metrics.get("sample_count_used", 0))
    rejected_count = int(metrics.get("sample_count_rejected", 0))
    inlier_chart = ChartModel(
        kind="bar",
        title="Inlier / Outlier Summary",
        x_title="Class",
        y_title="Count",
        caption="Samples used by the final least-squares solve versus rejected outliers.",
        categories=["Inliers", "Rejected"],
        values=[float(inlier_count), float(rejected_count)],
        color_hex=COLORS.scene_measurement,
    )
    acceptance_lines = _acceptance_lines(
        experiment_name="pivot_calibration",
        metrics=metrics,
        acceptance=acceptance,
    )
    summary_lines = [
        f"Run status: {metrics.get('status', 'unknown')}",
        f"Pivot RMSE: {_fmt(metrics.get('rmse_mm'))} mm",
        f"Samples collected: {metrics.get('sample_count_total', len(sample_points))}",
        f"Samples used: {inlier_count}",
        f"Samples rejected: {rejected_count}",
        f"Tip vector (local mm): {metrics.get('tip_vector_local_mm')}",
    ]
    summary_lines.extend(acceptance_lines)
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
    return str(sample.extra.get("target_label") or f"Target {sample.target_index}")


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
        color = _semantic_color(key, fallback_index=index)
        series.append(
            ScatterSeries3D(
                name=f"{base_label}: {key}",
                color_hex=color,
                points_xyz=points,
                point_size=0.12 if mesh == "sphere" else 0.18,
                mesh=mesh,
            )
        )
    return series


def _histogram_chart(
    *,
    title: str,
    values: list[float],
    x_title: str,
    y_title: str,
    color_hex: str,
    caption: str = "",
) -> ChartModel:
    if not values:
        return ChartModel(
            kind="bar",
            title=title,
            x_title=x_title,
            y_title=y_title,
            caption=caption,
            categories=[],
            values=[],
            color_hex=color_hex,
        )
    hist, edges = np.histogram(np.asarray(values, dtype=float), bins=min(8, max(3, len(values))))
    categories = [f"{edges[index]:.2f}-{edges[index + 1]:.2f}" for index in range(len(hist))]
    return ChartModel(
        kind="bar",
        title=title,
        x_title=x_title,
        y_title=y_title,
        caption=caption,
        categories=categories,
        values=[float(value) for value in hist.tolist()],
        color_hex=color_hex,
    )


def _fmt(value: Any) -> str:
    if value is None:
        return "n/a"
    return f"{float(value):.3f}"


def _semantic_color(key: str, *, fallback_index: int) -> str:
    normalized = str(key).strip().lower()
    if "truth" in normalized:
        return COLORS.text_primary
    if "centroid" in normalized:
        return COLORS.text_secondary
    if "inlier" in normalized:
        return COLORS.scene_measurement
    if "outlier" in normalized:
        return COLORS.scene_residual
    if "stale" in normalized:
        return COLORS.warning_fg
    return _PALETTE[fallback_index % len(_PALETTE)]


def _acceptance_lines(*, experiment_name: str, metrics: dict[str, Any], acceptance: dict[str, Any]) -> list[str]:
    acceptance = dict(acceptance or {})
    if not acceptance:
        return ["Acceptance thresholds: not configured for this run."]

    lines: list[str] = []
    status = "pass"
    reasons: list[str] = []

    def _apply_threshold(metric_key: str, warn_key: str, fail_key: str, label: str) -> None:
        nonlocal status
        value = metrics.get(metric_key)
        if value is None:
            return
        warn_value = acceptance.get(warn_key)
        fail_value = acceptance.get(fail_key)
        if fail_value is not None and float(value) > float(fail_value):
            status = "fail"
            reasons.append(f"{label} {float(value):.3f} > fail {float(fail_value):.3f}")
            return
        if warn_value is not None and float(value) > float(warn_value) and status != "fail":
            status = "warn"
            reasons.append(f"{label} {float(value):.3f} > warn {float(warn_value):.3f}")

    if experiment_name == "repeatability_dataset":
        _apply_threshold(
            "overall_repeatability_rms_mm",
            "repeatability_rms_warn_mm",
            "repeatability_rms_fail_mm",
            "repeatability RMS",
        )
        min_valid = acceptance.get("min_valid_sample_count")
        valid_count = metrics.get("valid_sample_count")
        if min_valid is not None and valid_count is not None and int(valid_count) < int(min_valid):
            status = "fail" if status != "fail" else status
            reasons.append(f"valid samples {int(valid_count)} < minimum {int(min_valid)}")
    elif experiment_name == "aurora_grid_accuracy":
        _apply_threshold("overall_rms_error_mm", "grid_rms_warn_mm", "grid_rms_fail_mm", "grid RMS")
        total_points = max(1, int(metrics.get("raw_sample_count", 0) or metrics.get("valid_sample_count", 0) or 0))
        outlier_count = int(metrics.get("outlier_count", 0) or 0)
        outlier_rate = float(outlier_count) / float(total_points)
        warn_rate = acceptance.get("outlier_rate_warn")
        fail_rate = acceptance.get("outlier_rate_fail")
        if fail_rate is not None and outlier_rate > float(fail_rate):
            status = "fail"
            reasons.append(f"outlier rate {outlier_rate:.3f} > fail {float(fail_rate):.3f}")
        elif warn_rate is not None and outlier_rate > float(warn_rate) and status != "fail":
            status = "warn"
            reasons.append(f"outlier rate {outlier_rate:.3f} > warn {float(warn_rate):.3f}")
    elif experiment_name == "pivot_calibration":
        _apply_threshold("rmse_mm", "pivot_rmse_warn_mm", "pivot_rmse_fail_mm", "pivot RMSE")
        total = max(1, int(metrics.get("sample_count_total", 0) or 0))
        rejected = int(metrics.get("sample_count_rejected", 0) or 0)
        outlier_rate = float(rejected) / float(total)
        warn_rate = acceptance.get("outlier_rate_warn")
        fail_rate = acceptance.get("outlier_rate_fail")
        if fail_rate is not None and outlier_rate > float(fail_rate):
            status = "fail"
            reasons.append(f"rejected-sample rate {outlier_rate:.3f} > fail {float(fail_rate):.3f}")
        elif warn_rate is not None and outlier_rate > float(warn_rate) and status != "fail":
            status = "warn"
            reasons.append(f"rejected-sample rate {outlier_rate:.3f} > warn {float(warn_rate):.3f}")

    rendered_thresholds = ", ".join(f"{key}={value}" for key, value in sorted(acceptance.items()))
    lines.append(f"Acceptance check: {status.upper()}")
    lines.append(f"Configured thresholds: {rendered_thresholds}")
    if reasons:
        lines.append("Threshold reasons: " + "; ".join(reasons))
    else:
        lines.append("Threshold reasons: all configured thresholds passed.")
    return lines


def _label_sort_key(value: Any) -> tuple[int, str]:
    text = str(value)
    digits = "".join(ch for ch in text if ch.isdigit())
    return (int(digits) if digits else 10**9, text)


def _build_generic_model(
    *,
    experiment_name: str,
    samples,
    metrics: dict[str, Any],
    tool_id: str,
) -> VisualizationModel:
    grouped: dict[str, list[tuple[float, float, float]]] = {}
    phase_counts: dict[str, int] = {}
    command_norms: list[tuple[float, float]] = []
    for index, sample in enumerate(samples):
        phase = str(sample.phase or "sample")
        phase_counts[phase] = int(phase_counts.get(phase, 0) + 1)
        if sample.commanded_cable_deltas_cm:
            norm = float(np.linalg.norm(np.asarray(sample.commanded_cable_deltas_cm, dtype=float)))
            command_norms.append((float(index), norm))
        position, _frame_name = extract_tip_or_tool_position_mm(sample, tool_id=tool_id, prefer_robot_frame=True)
        if position is not None:
            grouped.setdefault(phase, []).append(tuple(float(value) for value in position))

    summary_lines = [
        f"Experiment: {experiment_name}",
        f"Samples loaded: {len(samples)}",
        f"Metric count: {len(metrics)}",
    ]
    scalar_metrics = [
        (key, value)
        for key, value in metrics.items()
        if isinstance(value, (str, int, float, bool))
    ]
    for key, value in scalar_metrics[:8]:
        rendered = f"{float(value):.3f}" if isinstance(value, float) else str(value)
        summary_lines.append(f"{key}: {rendered}")
    if not scalar_metrics:
        summary_lines.append("No scalar summary metrics are available for this experiment yet.")

    charts: list[ChartModel] = []
    if phase_counts:
        charts.append(
            ChartModel(
                kind="bar",
                title="Samples By Phase",
                x_title="Phase",
                y_title="Count",
                caption="How many canonical samples were recorded in each experiment phase.",
                categories=list(phase_counts.keys()),
                values=[float(value) for value in phase_counts.values()],
                color_hex=COLORS.scene_truth,
            )
        )
    if command_norms:
        charts.append(
            ChartModel(
                kind="line",
                title="Command Magnitude",
                x_title="Sample Index",
                y_title="Command Norm",
                caption="Euclidean norm of commanded cable displacement by sample index.",
                points_xy=command_norms,
                color_hex=COLORS.scene_measurement,
            )
        )
    return VisualizationModel(
        series_3d=_grouped_series(grouped, base_label="Samples"),
        charts=charts,
        summary_lines=summary_lines,
    )


def _build_pretension_validation_model(*, samples, metrics: dict[str, Any]) -> VisualizationModel:
    trace_points = extract_pretension_trace_points(samples)
    use_mm = any(point.travel_from_untensioned_mm is not None for point in trace_points)
    x_title = "Travel From Untensioned (mm)" if use_mm else "Travel From Untensioned (ticks)"
    filtered_points = []
    raw_points = []
    displacement_points = []
    for point in trace_points:
        x_value = point.travel_from_untensioned_mm if use_mm else point.travel_from_untensioned_ticks
        if x_value is None:
            continue
        if point.filtered_current_ma is not None:
            filtered_points.append((float(x_value), float(point.filtered_current_ma)))
        if point.raw_current_ma is not None:
            raw_points.append((float(x_value), float(point.raw_current_ma)))
        if point.tracker_displacement_mm is not None:
            displacement_points.append((float(x_value), float(point.tracker_displacement_mm)))

    summary_lines = [
        f"Status: {metrics.get('status', 'unknown')}",
        f"Servo: {metrics.get('servo_id', 'n/a')}",
        f"Accepted: {'yes' if metrics.get('accepted') else 'no'}",
        f"Stop Reason: {metrics.get('stop_reason', 'n/a')}",
        f"Final Position: {_fmt(metrics.get('final_position_tick'))} ticks",
        f"Travel Used (ticks): {_fmt(metrics.get('travel_used_ticks'))}",
        f"Travel Used (mm): {_fmt(metrics.get('travel_used_mm'))}",
        f"Baseline Current: {_fmt(metrics.get('baseline_current_ma'))} mA",
        f"Effective Trigger: {_fmt(metrics.get('effective_trigger_current_ma'))} mA",
        f"Trigger Current: {_fmt(metrics.get('trigger_current_ma'))} mA",
        f"Hard Current Stop: {_fmt(metrics.get('hard_current_stop_ma'))} mA",
        f"Max Tracker Displacement: {_fmt(metrics.get('max_observed_displacement_mm'))} mm",
        "Current is treated here as an engagement proxy only. This run does not estimate tendon force.",
    ]
    charts: list[ChartModel] = []
    if filtered_points:
        charts.append(
            ChartModel(
                kind="line",
                title="Filtered Current vs Travel",
                x_title=x_title,
                y_title="Filtered Current (mA)",
                caption="Filtered present current across the pretension run, indexed by travel from the untensioned reference.",
                points_xy=filtered_points,
                color_hex=COLORS.scene_truth,
            )
        )
    if raw_points:
        charts.append(
            ChartModel(
                kind="line",
                title="Raw Current vs Travel",
                x_title=x_title,
                y_title="Raw Current (mA)",
                caption="Raw present current recorded at each pretension trace sample.",
                points_xy=raw_points,
                color_hex=COLORS.text_muted,
            )
        )
    if displacement_points:
        charts.append(
            ChartModel(
                kind="line",
                title="Tracker Displacement vs Travel",
                x_title=x_title,
                y_title="Displacement (mm)",
                caption="Tracker-side displacement relative to the run start, when live tracking was available.",
                points_xy=displacement_points,
                color_hex=COLORS.selection_bg,
            )
        )
    return VisualizationModel(charts=charts, summary_lines=summary_lines)


def _build_tracker_timing_model(*, samples, metrics: dict[str, Any], config_payload: dict[str, Any]) -> VisualizationModel:
    tracker_records = extract_tracker_timing_records(samples)
    servo_records = extract_servo_timing_records(samples)
    requested_tool_ids = [
        str(value).strip().upper()
        for value in (config_payload.get("requested_tool_ids") or ["0A", "0B"])
        if str(value).strip()
    ] or ["0A", "0B"]
    if not metrics:
        servo_sync = compute_servo_sync_summary(tracker_records, servo_records)
        servo_sync["enabled"] = bool(config_payload.get("enable_servo_logging", False))
        backend_identity = next(
            (str(record.get("backend_identity")) for record in tracker_records if record.get("backend_identity")),
            "",
        )
        metrics = compute_tracker_timing_summary(
            tracker_records,
            requested_tool_ids=requested_tool_ids,
            backend_identity=backend_identity,
            configured_backend_name="",
            selected_backend_name="",
            run_duration_s=config_payload.get("run_duration_s"),
            run_label=str(config_payload.get("run_label", "") or ""),
            servo_sync_summary=servo_sync,
        )

    analyzed_records = [record for record in tracker_records if not bool(record.get("warmup_discarded", False))]
    total_points = [
        (float(index), float(record["total_cycle_ms"]))
        for index, record in enumerate(analyzed_records)
        if record.get("total_cycle_ms") is not None
    ]
    duplicate_points_present = any(bool(record.get("is_duplicate_frame", False)) for record in analyzed_records)
    stage_chart = ChartModel(
        kind="bar",
        title="Stage Mean Timing",
        x_title="Stage",
        y_title="Mean Time (ms)",
        caption="Mean host monotonic time spent in backend get_frame(), payload parsing, runtime commit, and the full sample cycle.",
        categories=["get_frame", "parse", "commit", "total"],
        values=[
            float(metrics.get("mean_backend_call_ms", 0.0) or 0.0),
            float(metrics.get("mean_parse_ms", 0.0) or 0.0),
            float(metrics.get("mean_state_commit_ms", 0.0) or 0.0),
            float(metrics.get("mean_total_cycle_ms", 0.0) or 0.0),
        ],
        color_hex=COLORS.scene_measurement,
    )
    per_tool_summary = dict(metrics.get("per_tool_summary", {}) or {})
    tool_rate_chart = ChartModel(
        kind="bar",
        title="Per-Tool Valid Transform Rate",
        x_title="Tool",
        y_title="Valid Transform Rate (%)",
        caption="Fraction of analyzed samples in which each requested tool produced a tracked transform.",
        categories=list(per_tool_summary.keys()),
        values=[
            100.0 * float(value.get("valid_transform_rate", 0.0) or 0.0)
            for value in per_tool_summary.values()
        ],
        color_hex=COLORS.scene_truth,
    )
    charts = [
        ChartModel(
            kind="line",
            title="Total Cycle Time",
            x_title="Analyzed Sample Index",
            y_title="Total Cycle Time (ms)",
            caption="Total backend sample time over analyzed tracker samples. Duplicate device frames remain visible in the same series and are also reported separately in the summary.",
            points_xy=total_points,
            color_hex=COLORS.scene_truth,
        ),
        stage_chart,
    ]
    if per_tool_summary:
        charts.append(tool_rate_chart)
    servo_sync = dict(metrics.get("servo_sync", {}) or {})
    if servo_sync.get("enabled") and servo_sync.get("available"):
        charts.append(
            ChartModel(
                kind="line",
                title="Servo To Tracker Offset",
                x_title="Servo Sample Index",
                y_title="Absolute Offset (ms)",
                caption="Nearest analyzed tracker sample offset for each logged servo telemetry sample.",
                points_xy=[
                    (float(index), float(value))
                    for index, value in enumerate(servo_sync.get("servo_to_tracker_offsets_ms", []) or [])
                ],
                color_hex=COLORS.selection_bg,
            )
        )
    summary_lines = [
        f"Backend: {metrics.get('backend_identity', 'n/a')}",
        f"Configured backend: {metrics.get('configured_backend_name', 'n/a') or 'n/a'}",
        f"Selected backend: {metrics.get('selected_backend_name', 'n/a') or 'n/a'}",
        f"Requested tools: {', '.join(metrics.get('requested_tool_ids', []) or requested_tool_ids)}",
        f"Analyzed tracker samples: {int(metrics.get('sample_count_analyzed', 0) or 0)}",
        f"Warmup discarded: {int(metrics.get('warmup_discarded_count', 0) or 0)}",
        f"Effective loop rate: {_fmt(metrics.get('effective_loop_rate_hz'))} Hz",
        f"Unique-frame rate: {_fmt(metrics.get('unique_frame_rate_hz'))} Hz",
        f"Mean total cycle: {_fmt(metrics.get('mean_total_cycle_ms'))} ms",
        f"P95 total cycle: {_fmt(metrics.get('p95_total_cycle_ms'))} ms",
        f"P99 total cycle: {_fmt(metrics.get('p99_total_cycle_ms'))} ms",
        (
            f"Duplicate frames: {int(metrics.get('duplicate_frame_count', 0) or 0)} "
            f"({100.0 * float(metrics.get('duplicate_frame_ratio', 0.0) or 0.0):.1f}%)"
            if metrics.get("duplicate_frame_ratio") is not None
            else "Duplicate frames: n/a"
        ),
        (
            f"Samples under 25 ms: {float(metrics.get('percent_under_25ms', 0.0) or 0.0):.1f}%"
            if metrics.get("percent_under_25ms") is not None
            else "Samples under 25 ms: n/a"
        ),
        (
            f"Invalid/missing requested-tool samples: {int(metrics.get('invalid_or_missing_requested_tool_sample_count', 0) or 0)}"
        ),
    ]
    if duplicate_points_present:
        summary_lines.append("Duplicate-frame samples remain included in the time-series; use unique-frame Hz to judge fresh-frame throughput.")
    if servo_sync.get("enabled"):
        if servo_sync.get("available"):
            summary_lines.append(
                f"Servo->tracker mean/p95 offset: {_fmt(servo_sync.get('servo_to_tracker_mean_offset_ms'))} / {_fmt(servo_sync.get('servo_to_tracker_p95_offset_ms'))} ms"
            )
        else:
            summary_lines.append("Servo sync logging was requested, but no valid tracker-servo pairings were available.")
    return VisualizationModel(charts=charts, summary_lines=summary_lines)

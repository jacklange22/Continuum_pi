"""Output writers + repeatability metrics for two-segment workspace repeatability.

Mirrors the single-segment ``workspace_repeatability_map_outputs`` structure
in shape and figure set, adapted for the 4D two-segment command space.
"""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


# Canonical artifact filenames. The single-segment ``workspace_repeatability_map``
# uses ``workspace_map_*`` + ``thesis_0*_workspace_*`` names, so the
# two-segment artifacts use the IDENTICAL names — same Data tab / export /
# validator surface. Two-segment-specific extras are added alongside.
WORKSPACE_MAP_SUMMARY_JSON = "workspace_map_summary.json"
WORKSPACE_MAP_VISITS_JSONL = "workspace_map_visits.jsonl"
WORKSPACE_MAP_PER_TARGET_CSV = "workspace_map_per_target.csv"
THESIS_01_PNG = "thesis_01_workspace_rms_3d.png"
THESIS_02_PNG = "thesis_02_workspace_rms_map.png"
THESIS_03_PNG = "thesis_03_rms_vs_amplitude.png"
THESIS_04_PNG = "thesis_04_2d_repeatability_map.png"

# Two-segment-specific extras (no single-segment equivalent).
TARGETS_JSON = "repeatability_targets.json"
VISIT_PLAN_CSV = "repeatability_visit_plan.csv"
TARGET_CAPTURES_CSV = "target_captures.csv"
PER_TARGET_REPEATABILITY_CSV = "per_target_repeatability.csv"
REPEATABILITY_METRICS_CSV = "repeatability_metrics.csv"
FAILURE_EVENTS_JSONL = "failure_events.jsonl"
SUMMARY_TXT = "two_segment_workspace_repeatability_summary.txt"

THESIS_FIGURE_FILENAMES = (THESIS_01_PNG, THESIS_02_PNG, THESIS_03_PNG, THESIS_04_PNG)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


def compute_workspace_repeatability_metrics(
    *,
    visit_results: Sequence[Any],
    targets: Sequence[Any],
) -> list[dict[str, Any]]:
    """Per-target measured-space scatter stats for the workspace repeatability run.

    Accepts both the experiment's ``_VisitResult`` dataclass instances and
    plain dicts (so tests can feed synthetic rows). Drops rejected captures
    before computing per-target metrics.
    """
    visit_dicts = [_as_dict(visit) for visit in visit_results]
    accepted_by_target: dict[int, list[dict[str, Any]]] = {}
    for visit in visit_dicts:
        if not bool(visit.get("accepted") or visit.get("capture_accepted")):
            continue
        xyz = _xyz_array(visit.get("distal_xyz_robot_mm") or visit.get("position_mm"))
        if xyz is None:
            continue
        target_idx = int(visit.get("target_index"))
        accepted_by_target.setdefault(target_idx, []).append(visit)

    targets_by_index = {int(_as_dict(t).get("target_index", index)): _as_dict(t) for index, t in enumerate(targets)}

    rows: list[dict[str, Any]] = []
    for target_index, target_dict in sorted(targets_by_index.items()):
        visits = accepted_by_target.get(target_index, [])
        n = len(visits)
        base_row = _target_base_row(target_index=target_index, target_dict=target_dict)
        if n == 0:
            base_row.update(_empty_distal_stats())
            rows.append(base_row)
            continue
        distal_positions = _visit_xyz_matrix(visits, key="distal_xyz_robot_mm", fallback_key="position_mm")
        base_row.update(_position_stats(distal_positions, prefix=""))

        intermediate_positions = _visit_xyz_matrix(visits, key="intermediate_xyz_robot_mm")
        base_row.update(_position_stats(intermediate_positions, prefix="proximal_"))

        paired_relative_positions = _paired_relative_xyz_matrix(
            visits,
            distal_key="distal_xyz_robot_mm",
            proximal_key="intermediate_xyz_robot_mm",
        )
        base_row.update(_position_stats(paired_relative_positions, prefix="distal_relative_"))
        rows.append(base_row)
    _attach_measured_workspace_fields(rows)
    return rows


def _target_base_row(*, target_index: int, target_dict: Mapping[str, Any]) -> dict[str, Any]:
    bottom_x = float(target_dict.get("bottom_x_cm") or 0.0)
    bottom_y = float(target_dict.get("bottom_y_cm") or 0.0)
    top_x = float(target_dict.get("top_x_cm") or 0.0)
    top_y = float(target_dict.get("top_y_cm") or 0.0)
    proximal_command_norm_cm = float(math.hypot(bottom_x, bottom_y))
    distal_command_norm_cm = float(math.hypot(top_x, top_y))
    command_l2_cm = float(math.sqrt(bottom_x ** 2 + bottom_y ** 2 + top_x ** 2 + top_y ** 2))
    return {
        "target_index": int(target_index),
        "target_id": str(target_dict.get("target_id", f"WS_{target_index:04d}")),
        "group_tag": str(target_dict.get("group_tag", "")),
        "amplitude_cm": float(target_dict.get("amplitude_cm") or command_l2_cm),
        "bottom_x_cm": bottom_x,
        "bottom_y_cm": bottom_y,
        "top_x_cm": top_x,
        "top_y_cm": top_y,
        "proximal_command_norm_cm": proximal_command_norm_cm,
        "distal_command_norm_cm": distal_command_norm_cm,
        "command_l2_cm": command_l2_cm,
        "target_amplitude_mm": float(target_dict.get("amplitude_cm") or command_l2_cm) * 10.0,
        # Backward-compatible workspace-map columns. Older consumers read
        # x_mm/y_mm as the map coordinates. They now represent measured
        # distal-tip displacement and are filled by _attach_measured_workspace_fields.
        "x_mm": None,
        "y_mm": None,
    }


def _empty_distal_stats() -> dict[str, Any]:
    return {
        "accepted_repeats": 0,
        "centroid_xyz_mm": None,
        "rms_spread_mm": None,
        "mean_radial_mm": None,
        "median_radial_mm": None,
        "max_radial_mm": None,
        "std_x_mm": None,
        "std_y_mm": None,
        "std_z_mm": None,
        "proximal_accepted_repeats": 0,
        "proximal_centroid_xyz_mm": None,
        "proximal_rms_spread_mm": None,
        "proximal_mean_radial_mm": None,
        "proximal_median_radial_mm": None,
        "proximal_max_radial_mm": None,
        "proximal_std_x_mm": None,
        "proximal_std_y_mm": None,
        "proximal_std_z_mm": None,
        "distal_relative_accepted_repeats": 0,
        "distal_relative_centroid_xyz_mm": None,
        "distal_relative_rms_spread_mm": None,
        "distal_relative_mean_radial_mm": None,
        "distal_relative_median_radial_mm": None,
        "distal_relative_max_radial_mm": None,
        "distal_relative_std_x_mm": None,
        "distal_relative_std_y_mm": None,
        "distal_relative_std_z_mm": None,
    }


def _position_stats(positions: np.ndarray | None, *, prefix: str) -> dict[str, Any]:
    if positions is None or positions.size == 0:
        accepted_key = f"{prefix}accepted_repeats" if prefix else "accepted_repeats"
        centroid_key = f"{prefix}centroid_xyz_mm" if prefix else "centroid_xyz_mm"
        rms_key = f"{prefix}rms_spread_mm" if prefix else "rms_spread_mm"
        mean_key = f"{prefix}mean_radial_mm" if prefix else "mean_radial_mm"
        median_key = f"{prefix}median_radial_mm" if prefix else "median_radial_mm"
        max_key = f"{prefix}max_radial_mm" if prefix else "max_radial_mm"
        return {
            accepted_key: 0,
            centroid_key: None,
            rms_key: None,
            mean_key: None,
            median_key: None,
            max_key: None,
            f"{prefix}std_x_mm" if prefix else "std_x_mm": None,
            f"{prefix}std_y_mm" if prefix else "std_y_mm": None,
            f"{prefix}std_z_mm" if prefix else "std_z_mm": None,
        }
    centroid = positions.mean(axis=0)
    deltas = positions - centroid
    radial = np.linalg.norm(deltas, axis=1)
    rms_spread = float(np.sqrt(np.mean(radial ** 2)))
    std = positions.std(axis=0)
    accepted_key = f"{prefix}accepted_repeats" if prefix else "accepted_repeats"
    centroid_key = f"{prefix}centroid_xyz_mm" if prefix else "centroid_xyz_mm"
    rms_key = f"{prefix}rms_spread_mm" if prefix else "rms_spread_mm"
    mean_key = f"{prefix}mean_radial_mm" if prefix else "mean_radial_mm"
    median_key = f"{prefix}median_radial_mm" if prefix else "median_radial_mm"
    max_key = f"{prefix}max_radial_mm" if prefix else "max_radial_mm"
    return {
        accepted_key: int(positions.shape[0]),
        centroid_key: [float(c) for c in centroid.tolist()],
        rms_key: rms_spread,
        mean_key: float(np.mean(radial)),
        median_key: float(np.median(radial)),
        max_key: float(np.max(radial)),
        f"{prefix}std_x_mm" if prefix else "std_x_mm": float(std[0]),
        f"{prefix}std_y_mm" if prefix else "std_y_mm": float(std[1]),
        f"{prefix}std_z_mm" if prefix else "std_z_mm": float(std[2]),
    }


def _visit_xyz_matrix(
    visits: Sequence[Mapping[str, Any]],
    *,
    key: str,
    fallback_key: str | None = None,
) -> np.ndarray | None:
    values: list[np.ndarray] = []
    for visit in visits:
        xyz = _xyz_array(visit.get(key))
        if xyz is None and fallback_key:
            xyz = _xyz_array(visit.get(fallback_key))
        if xyz is not None:
            values.append(xyz)
    if not values:
        return None
    return np.vstack(values).astype(float)


def _paired_relative_xyz_matrix(
    visits: Sequence[Mapping[str, Any]],
    *,
    distal_key: str,
    proximal_key: str,
) -> np.ndarray | None:
    values: list[np.ndarray] = []
    for visit in visits:
        distal = _xyz_array(visit.get(distal_key) or visit.get("position_mm"))
        proximal = _xyz_array(visit.get(proximal_key))
        if distal is not None and proximal is not None:
            values.append(distal - proximal)
    if not values:
        return None
    return np.vstack(values).astype(float)


def _attach_measured_workspace_fields(rows: list[dict[str, Any]]) -> None:
    """Add real-space displacement fields and segment contribution estimates.

    The plots should show measured robot-space motion, not commanded tendon
    coordinates. When an intermediate/proximal tracker marker exists we record
    direct proximal and distal-relative displacement fields. When it is absent
    (common in older runs), we still separate the measured distal-tip workspace
    into proximal-command and distal-command contributions with a least-squares
    fit from the measured centroids.
    """
    for row in rows:
        _set_default_measured_fields(row)

    data_indices = [idx for idx, row in enumerate(rows) if _xyz_array(row.get("centroid_xyz_mm")) is not None]
    if not data_indices:
        return

    command = np.asarray(
        [
            [
                float(rows[idx].get("bottom_x_cm") or 0.0),
                float(rows[idx].get("bottom_y_cm") or 0.0),
                float(rows[idx].get("top_x_cm") or 0.0),
                float(rows[idx].get("top_y_cm") or 0.0),
            ]
            for idx in data_indices
        ],
        dtype=float,
    )
    centroids = np.asarray([_xyz_array(rows[idx].get("centroid_xyz_mm")) for idx in data_indices], dtype=float)
    reference_xyz, reference_method = _measured_zero_reference(
        rows=rows,
        data_indices=data_indices,
        centroids=centroids,
        command=command,
    )
    coeffs, fit_rank = _least_squares_workspace_coefficients(command=command, centroids=centroids)
    segment_method = (
        "direct_intermediate_marker"
        if any(_xyz_array(row.get("proximal_centroid_xyz_mm")) is not None for row in rows)
        else "least_squares_distal_tip_decomposition"
    )

    for row in rows:
        row["measured_tip_reference_xyz_mm"] = _array_to_xyz_list(reference_xyz)
        row["measured_displacement_reference_method"] = reference_method
        row["segment_separation_method"] = segment_method
        row["workspace_fit_rank"] = int(fit_rank)
        centroid = _xyz_array(row.get("centroid_xyz_mm"))
        command_vec = np.asarray(
            [
                float(row.get("bottom_x_cm") or 0.0),
                float(row.get("bottom_y_cm") or 0.0),
                float(row.get("top_x_cm") or 0.0),
                float(row.get("top_y_cm") or 0.0),
            ],
            dtype=float,
        )
        proximal_est = command_vec[0] * coeffs[1] + command_vec[1] * coeffs[2]
        distal_est = command_vec[2] * coeffs[3] + command_vec[3] * coeffs[4]
        row["proximal_estimated_displacement_xyz_mm"] = _array_to_xyz_list(proximal_est)
        row["proximal_estimated_displacement_norm_mm"] = float(np.linalg.norm(proximal_est))
        row["distal_estimated_displacement_xyz_mm"] = _array_to_xyz_list(distal_est)
        row["distal_estimated_displacement_norm_mm"] = float(np.linalg.norm(distal_est))
        if centroid is None:
            continue
        measured_displacement = centroid - reference_xyz
        row["measured_tip_displacement_xyz_mm"] = _array_to_xyz_list(measured_displacement)
        row["measured_tip_displacement_norm_mm"] = float(np.linalg.norm(measured_displacement))
        row["measured_tip_displacement_xy_mm"] = float(np.linalg.norm(measured_displacement[:2]))
        row["x_mm"] = float(measured_displacement[0])
        row["y_mm"] = float(measured_displacement[1])
        fit_prediction = coeffs[0] + proximal_est + distal_est
        fit_residual = centroid - fit_prediction
        row["workspace_fit_residual_xyz_mm"] = _array_to_xyz_list(fit_residual)
        row["workspace_fit_residual_norm_mm"] = float(np.linalg.norm(fit_residual))

    _attach_direct_segment_displacements(
        rows,
        centroid_key="proximal_centroid_xyz_mm",
        displacement_key="proximal_measured_displacement_xyz_mm",
        norm_key="proximal_measured_displacement_norm_mm",
    )
    _attach_direct_segment_displacements(
        rows,
        centroid_key="distal_relative_centroid_xyz_mm",
        displacement_key="distal_relative_measured_displacement_xyz_mm",
        norm_key="distal_relative_measured_displacement_norm_mm",
    )


def _set_default_measured_fields(row: dict[str, Any]) -> None:
    row.setdefault("measured_tip_reference_xyz_mm", None)
    row.setdefault("measured_displacement_reference_method", None)
    row.setdefault("measured_tip_displacement_xyz_mm", None)
    row.setdefault("measured_tip_displacement_norm_mm", None)
    row.setdefault("measured_tip_displacement_xy_mm", None)
    row.setdefault("proximal_estimated_displacement_xyz_mm", None)
    row.setdefault("proximal_estimated_displacement_norm_mm", None)
    row.setdefault("distal_estimated_displacement_xyz_mm", None)
    row.setdefault("distal_estimated_displacement_norm_mm", None)
    row.setdefault("proximal_measured_displacement_xyz_mm", None)
    row.setdefault("proximal_measured_displacement_norm_mm", None)
    row.setdefault("distal_relative_measured_displacement_xyz_mm", None)
    row.setdefault("distal_relative_measured_displacement_norm_mm", None)
    row.setdefault("workspace_fit_rank", None)
    row.setdefault("workspace_fit_residual_xyz_mm", None)
    row.setdefault("workspace_fit_residual_norm_mm", None)
    row.setdefault("segment_separation_method", None)


def _measured_zero_reference(
    *,
    rows: Sequence[Mapping[str, Any]],
    data_indices: Sequence[int],
    centroids: np.ndarray,
    command: np.ndarray,
) -> tuple[np.ndarray, str]:
    neutral_positions: list[np.ndarray] = []
    for row in rows:
        command_l2 = float(row.get("command_l2_cm") or 0.0)
        centroid = _xyz_array(row.get("centroid_xyz_mm"))
        if command_l2 <= 1e-9 and centroid is not None:
            neutral_positions.append(centroid)
    if neutral_positions:
        return np.vstack(neutral_positions).mean(axis=0), "captured_zero_command_target"

    coeffs, rank = _least_squares_workspace_coefficients(command=command, centroids=centroids)
    if rank >= 5:
        return np.asarray(coeffs[0], dtype=float), "least_squares_zero_command_intercept"

    closest_index = min(data_indices, key=lambda idx: float(rows[idx].get("command_l2_cm") or 0.0))
    closest = _xyz_array(rows[closest_index].get("centroid_xyz_mm"))
    if closest is not None:
        return closest, f"nearest_available_target:{rows[closest_index].get('target_id')}"
    return centroids.mean(axis=0), "workspace_centroid"


def _least_squares_workspace_coefficients(*, command: np.ndarray, centroids: np.ndarray) -> tuple[np.ndarray, int]:
    if command.size == 0 or centroids.size == 0:
        return np.zeros((5, 3), dtype=float), 0
    design = np.column_stack([np.ones(command.shape[0], dtype=float), command])
    coeffs, _residuals, rank, _singular_values = np.linalg.lstsq(design, centroids, rcond=None)
    if coeffs.shape != (5, 3):
        padded = np.zeros((5, 3), dtype=float)
        padded[: coeffs.shape[0], : coeffs.shape[1]] = coeffs
        coeffs = padded
    return np.asarray(coeffs, dtype=float), int(rank)


def _attach_direct_segment_displacements(
    rows: list[dict[str, Any]],
    *,
    centroid_key: str,
    displacement_key: str,
    norm_key: str,
) -> None:
    data = [(idx, _xyz_array(row.get(centroid_key))) for idx, row in enumerate(rows)]
    data = [(idx, xyz) for idx, xyz in data if xyz is not None]
    if not data:
        return
    command = np.asarray(
        [
            [
                float(rows[idx].get("bottom_x_cm") or 0.0),
                float(rows[idx].get("bottom_y_cm") or 0.0),
                float(rows[idx].get("top_x_cm") or 0.0),
                float(rows[idx].get("top_y_cm") or 0.0),
            ]
            for idx, _xyz in data
        ],
        dtype=float,
    )
    centroids = np.asarray([xyz for _idx, xyz in data], dtype=float)
    reference_xyz, _method = _measured_zero_reference(
        rows=rows,
        data_indices=[idx for idx, _xyz in data],
        centroids=centroids,
        command=command,
    )
    for idx, xyz in data:
        displacement = xyz - reference_xyz
        rows[idx][displacement_key] = _array_to_xyz_list(displacement)
        rows[idx][norm_key] = float(np.linalg.norm(displacement))


def _xyz_array(value: Any) -> np.ndarray | None:
    if not isinstance(value, (list, tuple, np.ndarray)) or len(value) < 3:
        return None
    try:
        arr = np.asarray([float(value[0]), float(value[1]), float(value[2])], dtype=float)
    except (TypeError, ValueError):
        return None
    if not np.all(np.isfinite(arr)):
        return None
    return arr


def _array_to_xyz_list(value: np.ndarray | Sequence[float]) -> list[float]:
    arr = np.asarray(value, dtype=float).reshape(-1)
    return [float(v) for v in arr[:3]]


def _numeric_row_values(rows: Sequence[Mapping[str, Any]], key: str) -> np.ndarray:
    values: list[float] = []
    for row in rows:
        value = row.get(key)
        if isinstance(value, (int, float)) and math.isfinite(float(value)):
            values.append(float(value))
    return np.asarray(values, dtype=float)


def summarize_workspace_repeatability(
    per_target_rows: Sequence[Mapping[str, Any]],
    *,
    minimum_repeats_per_target: int,
    target_distal_rms_threshold_mm: float | None = None,
    thesis_goal_rms_mm: float | None = None,
) -> dict[str, Any]:
    """Roll up per-target rows into the overall repeatability summary.

    Key names match the single-segment ``workspace_repeatability_map``
    contract (``workspace_rms_mean_mm`` / ``workspace_rms_max_mm`` /
    ``workspace_rms_p95_mm`` / ``targets_with_data`` / ``worst_targets``)
    so the Data tab + downstream tools recognize the shape; we add a
    handful of two-segment-aware extras alongside.
    """
    rows_with_data = [row for row in per_target_rows if row.get("rms_spread_mm") is not None]
    goal_mm = float(thesis_goal_rms_mm) if thesis_goal_rms_mm is not None else (
        float(target_distal_rms_threshold_mm) if target_distal_rms_threshold_mm is not None else None
    )
    if not rows_with_data:
        return {
            # Single-segment-compatible keys (operator + Data tab look for these).
            "target_count": int(len(per_target_rows)),
            "targets_with_data": 0,
            "workspace_rms_mean_mm": None,
            "workspace_rms_median_mm": None,
            "workspace_rms_p95_mm": None,
            "workspace_rms_max_mm": None,
            "workspace_max_spread_max_mm": None,
            "thesis_goal_rms_mm": goal_mm,
            "fraction_above_thesis_goal": None,
            "worst_targets": [],
            # Two-segment-specific bookkeeping.
            "targets_below_minimum_repeats": int(len(per_target_rows)),
            "minimum_repeats_per_target": int(minimum_repeats_per_target),
            "overall_distal_rms_mm": None,
            "median_per_target_rms_mm": None,
            "p95_per_target_rms_mm": None,
            "worst_target_rms_mm": None,
            "worst_target_id": None,
            "best_target_rms_mm": None,
            "best_target_id": None,
            "mean_repeats_per_target": 0.0,
            "target_distal_rms_threshold_mm": target_distal_rms_threshold_mm,
            "targets_above_threshold": None,
            "measured_displacement_reference_method": None,
            "measured_tip_reference_xyz_mm": None,
            "measured_tip_displacement_median_mm": None,
            "measured_tip_displacement_max_mm": None,
            "intermediate_marker_targets_with_data": 0,
            "direct_proximal_rms_mean_mm": None,
            "direct_distal_relative_rms_mean_mm": None,
            "segment_separation_method": None,
            "proximal_estimated_displacement_max_mm": None,
            "distal_estimated_displacement_max_mm": None,
        }
    rms_values = np.asarray([float(row["rms_spread_mm"]) for row in rows_with_data])
    max_radial = np.asarray([float(row.get("max_radial_mm") or 0.0) for row in rows_with_data])
    repeats = np.asarray([int(row.get("accepted_repeats", 0)) for row in rows_with_data])
    weighted_rms = float(np.sqrt(np.sum((rms_values ** 2) * repeats) / max(1, np.sum(repeats))))
    median_rms = float(np.median(rms_values))
    p95_rms = float(np.percentile(rms_values, 95.0))
    worst_idx = int(np.argmax(rms_values))
    best_idx = int(np.argmin(rms_values))
    below_min = sum(
        1 for row in per_target_rows if int(row.get("accepted_repeats", 0)) < int(minimum_repeats_per_target)
    )
    targets_above_threshold: int | None = None
    if target_distal_rms_threshold_mm is not None:
        targets_above_threshold = int(np.sum(rms_values > float(target_distal_rms_threshold_mm)))
    fraction_above_goal: float | None = None
    if goal_mm is not None:
        fraction_above_goal = float(np.sum(rms_values > goal_mm) / float(len(rms_values)))
    measured_displacements = _numeric_row_values(rows_with_data, "measured_tip_displacement_norm_mm")
    proximal_estimated = _numeric_row_values(rows_with_data, "proximal_estimated_displacement_norm_mm")
    distal_estimated = _numeric_row_values(rows_with_data, "distal_estimated_displacement_norm_mm")
    proximal_rms = _numeric_row_values(rows_with_data, "proximal_rms_spread_mm")
    distal_relative_rms = _numeric_row_values(rows_with_data, "distal_relative_rms_spread_mm")
    first_row = rows_with_data[0]
    # Top-5 worst targets by RMS (id, rms, accepted_repeats).
    worst_order = np.argsort(-rms_values)
    worst_targets: list[dict[str, Any]] = [
        {
            "target_id": str(rows_with_data[int(idx)].get("target_id", "")),
            "rms_spread_mm": float(rms_values[int(idx)]),
            "accepted_repeats": int(rows_with_data[int(idx)].get("accepted_repeats", 0)),
            "amplitude_cm": float(rows_with_data[int(idx)].get("amplitude_cm") or 0.0),
            "group_tag": str(rows_with_data[int(idx)].get("group_tag", "")),
        }
        for idx in worst_order[: min(5, len(worst_order))]
    ]
    return {
        # Single-segment-compatible keys.
        "target_count": int(len(per_target_rows)),
        "targets_with_data": int(len(rows_with_data)),
        "workspace_rms_mean_mm": float(np.mean(rms_values)),
        "workspace_rms_median_mm": float(median_rms),
        "workspace_rms_p95_mm": float(p95_rms),
        "workspace_rms_max_mm": float(rms_values[worst_idx]),
        "workspace_max_spread_max_mm": float(np.max(max_radial)) if max_radial.size else None,
        "thesis_goal_rms_mm": goal_mm,
        "fraction_above_thesis_goal": fraction_above_goal,
        "worst_targets": worst_targets,
        # Two-segment-specific bookkeeping.
        "targets_below_minimum_repeats": int(below_min),
        "minimum_repeats_per_target": int(minimum_repeats_per_target),
        "overall_distal_rms_mm": float(weighted_rms),
        "median_per_target_rms_mm": float(median_rms),
        "p95_per_target_rms_mm": float(p95_rms),
        "worst_target_rms_mm": float(rms_values[worst_idx]),
        "worst_target_id": str(rows_with_data[worst_idx].get("target_id", "")),
        "best_target_rms_mm": float(rms_values[best_idx]),
        "best_target_id": str(rows_with_data[best_idx].get("target_id", "")),
        "mean_repeats_per_target": float(np.mean(repeats)),
        "target_distal_rms_threshold_mm": target_distal_rms_threshold_mm,
        "targets_above_threshold": targets_above_threshold,
        "measured_displacement_reference_method": first_row.get("measured_displacement_reference_method"),
        "measured_tip_reference_xyz_mm": first_row.get("measured_tip_reference_xyz_mm"),
        "measured_tip_displacement_median_mm": (
            float(np.median(measured_displacements)) if measured_displacements.size else None
        ),
        "measured_tip_displacement_max_mm": (
            float(np.max(measured_displacements)) if measured_displacements.size else None
        ),
        "intermediate_marker_targets_with_data": int(
            sum(1 for row in rows_with_data if int(row.get("proximal_accepted_repeats") or 0) > 0)
        ),
        "direct_proximal_rms_mean_mm": float(np.mean(proximal_rms)) if proximal_rms.size else None,
        "direct_distal_relative_rms_mean_mm": (
            float(np.mean(distal_relative_rms)) if distal_relative_rms.size else None
        ),
        "segment_separation_method": first_row.get("segment_separation_method"),
        "proximal_estimated_displacement_max_mm": (
            float(np.max(proximal_estimated)) if proximal_estimated.size else None
        ),
        "distal_estimated_displacement_max_mm": (
            float(np.max(distal_estimated)) if distal_estimated.size else None
        ),
    }


# ---------------------------------------------------------------------------
# Output writers
# ---------------------------------------------------------------------------


def write_two_segment_workspace_repeatability_outputs(
    *,
    output_dir: Path,
    targets: Sequence[Any],
    visit_results: Sequence[Any],
    failure_events: Sequence[Mapping[str, Any]],
    metrics: Mapping[str, Any],
) -> dict[str, Path]:
    """Write the full experiment run folder (targets, captures, metrics, figures)."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    target_dicts = [_as_dict(t) for t in targets]
    visit_dicts = [_as_dict(v) for v in visit_results]
    per_target_rows = compute_workspace_repeatability_metrics(
        visit_results=visit_results, targets=targets
    )
    minimum_repeats = int(metrics.get("repeats_per_target", 20))
    summary = summarize_workspace_repeatability(
        per_target_rows,
        minimum_repeats_per_target=minimum_repeats,
    )

    paths: dict[str, Path] = {}

    # ---- Canonical single-segment-shape outputs --------------------------
    # These three filenames match `workspace_repeatability_map_outputs` so
    # the existing Data tab + export + analysis tools recognise the run.
    paths["workspace_map_summary_json"] = output_dir / WORKSPACE_MAP_SUMMARY_JSON
    paths["workspace_map_summary_json"].write_text(
        json.dumps(
            {
                "schema_version": "two_segment_workspace_repeatability_summary_v1",
                "summary": summary,
                "per_target_rows": per_target_rows,
            },
            indent=2,
            default=_json_default,
        ),
        encoding="utf-8",
    )
    paths["workspace_map_visits_jsonl"] = _write_workspace_map_visits_jsonl(
        output_dir / WORKSPACE_MAP_VISITS_JSONL, visit_dicts
    )
    paths["workspace_map_per_target_csv"] = _write_workspace_map_per_target_csv(
        output_dir / WORKSPACE_MAP_PER_TARGET_CSV, per_target_rows
    )

    # ---- Two-segment-specific extras -------------------------------------
    paths["targets_json"] = output_dir / TARGETS_JSON
    paths["targets_json"].write_text(
        json.dumps(
            {"schema_version": "two_segment_workspace_repeatability_targets_v1", "targets": target_dicts},
            indent=2,
        ),
        encoding="utf-8",
    )
    paths["visit_plan_csv"] = _write_visit_plan_csv(output_dir / VISIT_PLAN_CSV, visit_dicts)
    paths["target_captures_csv"] = _write_target_captures_csv(output_dir / TARGET_CAPTURES_CSV, visit_dicts)
    paths["per_target_csv"] = _write_per_target_csv(
        output_dir / PER_TARGET_REPEATABILITY_CSV, per_target_rows
    )
    paths["metrics_csv"] = _write_metrics_summary_csv(output_dir / REPEATABILITY_METRICS_CSV, summary)

    # failure events jsonl
    paths["failure_events_jsonl"] = output_dir / FAILURE_EVENTS_JSONL
    with paths["failure_events_jsonl"].open("w", encoding="utf-8") as handle:
        for event in failure_events:
            handle.write(json.dumps(_as_dict(event), default=_json_default) + "\n")

    # summary text
    paths["summary_text"] = _write_summary_text(output_dir / SUMMARY_TXT, metrics=dict(metrics), summary=summary)

    # figures (best-effort)
    try:
        from continuum_robot.experiments.plotting import (
            create_figure,
            create_3d_figure,
            save_figure,
            set_equal_xy,
            set_equal_xyz,
            style_axes,
            style_3d_axes,
        )

        max_amplitude_mm = float(metrics.get("max_segment_displacement_cm") or 0.0) * 10.0
        rows_with_data = [row for row in per_target_rows if row.get("rms_spread_mm") is not None]
        if rows_with_data:
            paths["thesis_01"] = _write_thesis_01(
                output_dir / THESIS_01_PNG,
                rows_with_data=rows_with_data,
                visit_dicts=visit_dicts,
                summary=summary,
                max_amplitude_mm=max_amplitude_mm,
                create_3d_figure=create_3d_figure,
                save_figure=save_figure,
                set_equal_xyz=set_equal_xyz,
                style_3d_axes=style_3d_axes,
            )
            paths["thesis_02"] = _write_thesis_02(
                output_dir / THESIS_02_PNG,
                rows_with_data=rows_with_data,
                visit_dicts=visit_dicts,
                summary=summary,
                max_amplitude_mm=max_amplitude_mm,
                create_figure=create_figure,
                save_figure=save_figure,
                set_equal_xy=set_equal_xy,
                style_axes=style_axes,
            )
            paths["thesis_03"] = _write_thesis_03(
                output_dir / THESIS_03_PNG,
                rows_with_data=rows_with_data,
                summary=summary,
                max_amplitude_mm=max_amplitude_mm,
                create_figure=create_figure,
                save_figure=save_figure,
                style_axes=style_axes,
            )
            paths["thesis_04"] = _write_thesis_04(
                output_dir / THESIS_04_PNG,
                rows_with_data=rows_with_data,
                summary=summary,
                max_amplitude_mm=max_amplitude_mm,
                save_figure=save_figure,
                set_equal_xy=set_equal_xy,
                style_axes=style_axes,
            )
    except Exception:
        # Figures are nice-to-have. Never block on missing matplotlib.
        pass

    return paths


# ---------------------------------------------------------------------------
# CSV / JSON writers
# ---------------------------------------------------------------------------


def _write_workspace_map_visits_jsonl(path: Path, visit_dicts: Sequence[Mapping[str, Any]]) -> Path:
    """Visit-level JSONL with the same shape the single-segment reader expects.

    The single-segment `workspace_map_visits.jsonl` carries
    `target_index`, `target_label`, `target_x_mm`, `target_y_mm`,
    `target_amplitude_mm`, `position_mm`, `capture_accepted`, etc. We
    emit the same keys so existing Data tab + analysis hooks work, plus
    two-segment extras (`bottom_x_cm`, `top_x_cm`, ...).
    """
    with Path(path).open("w", encoding="utf-8") as handle:
        for visit in visit_dicts:
            distal = visit.get("distal_xyz_robot_mm") or [None, None, None]
            handle.write(
                json.dumps(
                    {
                        # Single-segment-shape keys (operator tools recognize these).
                        "target_index": visit.get("target_index"),
                        "target_label": visit.get("target_id"),
                        "target_x_mm": float(visit.get("bottom_x_cm") or 0.0) * 10.0,
                        "target_y_mm": float(visit.get("bottom_y_cm") or 0.0) * 10.0,
                        "target_amplitude_mm": float(visit.get("amplitude_cm") or 0.0) * 10.0,
                        "position_mm": [float(v) for v in distal] if distal[0] is not None else None,
                        "capture_accepted": bool(visit.get("accepted") or visit.get("capture_accepted")),
                        "capture_reject_reason": visit.get("reject_reason"),
                        "cycle_index": visit.get("cycle_index"),
                        "rejected": not bool(visit.get("accepted") or visit.get("capture_accepted")),
                        "protocol": "two_segment_workspace_repeatability_lhs_from_neutral",
                        "tracker_stale_age_s": visit.get("tracker_age_s"),
                        "resolved_servo_goal_ticks": visit.get("all_8_goal_ticks"),
                        # Two-segment extras.
                        "bottom_x_cm": visit.get("bottom_x_cm"),
                        "bottom_y_cm": visit.get("bottom_y_cm"),
                        "top_x_cm": visit.get("top_x_cm"),
                        "top_y_cm": visit.get("top_y_cm"),
                        "group_tag": visit.get("group_tag"),
                        "visit_in_cycle": visit.get("visit_in_cycle"),
                        "visit_position": visit.get("visit_position"),
                        "intermediate_xyz_robot_mm": visit.get("intermediate_xyz_robot_mm"),
                        "ordered_8_displacements_cm": visit.get("ordered_8_displacements_cm"),
                    },
                    default=_json_default,
                )
                + "\n"
            )
    return path


def _write_workspace_map_per_target_csv(path: Path, per_target_rows: Sequence[Mapping[str, Any]]) -> Path:
    """Per-target CSV with single-segment-compatible column names plus extras.

    Matches the single-segment `workspace_map_per_target.csv` shape
    (target_index/target_label/target_x_mm/target_y_mm/target_amplitude_mm/
    rms_spread_mm/...). Bottom/top XY are added as extra columns at the end.
    """
    fields = [
        "target_index",
        "target_label",
        "target_x_mm",
        "target_y_mm",
        "target_amplitude_mm",
        "accepted_repeats",
        "rms_spread_mm",
        "mean_radial_mm",
        "median_radial_mm",
        "max_radial_mm",
        "std_x_mm",
        "std_y_mm",
        "std_z_mm",
        "centroid_x_mm",
        "centroid_y_mm",
        "centroid_z_mm",
        "measured_tip_displacement_x_mm",
        "measured_tip_displacement_y_mm",
        "measured_tip_displacement_z_mm",
        "measured_tip_displacement_norm_mm",
        "proximal_estimated_displacement_norm_mm",
        "distal_estimated_displacement_norm_mm",
        "proximal_measured_displacement_norm_mm",
        "distal_relative_measured_displacement_norm_mm",
        "group_tag",
        # Two-segment extras.
        "bottom_x_cm",
        "bottom_y_cm",
        "top_x_cm",
        "top_y_cm",
    ]
    with Path(path).open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in per_target_rows:
            centroid = row.get("centroid_xyz_mm") or [None, None, None]
            measured = row.get("measured_tip_displacement_xyz_mm") or [None, None, None]
            writer.writerow(
                {
                    "target_index": row.get("target_index"),
                    "target_label": row.get("target_id"),
                    "target_x_mm": row.get("x_mm"),
                    "target_y_mm": row.get("y_mm"),
                    "target_amplitude_mm": row.get("target_amplitude_mm"),
                    "accepted_repeats": row.get("accepted_repeats"),
                    "rms_spread_mm": row.get("rms_spread_mm"),
                    "mean_radial_mm": row.get("mean_radial_mm"),
                    "median_radial_mm": row.get("median_radial_mm"),
                    "max_radial_mm": row.get("max_radial_mm"),
                    "std_x_mm": row.get("std_x_mm"),
                    "std_y_mm": row.get("std_y_mm"),
                    "std_z_mm": row.get("std_z_mm"),
                    "centroid_x_mm": centroid[0] if centroid else None,
                    "centroid_y_mm": centroid[1] if centroid else None,
                    "centroid_z_mm": centroid[2] if centroid else None,
                    "measured_tip_displacement_x_mm": measured[0] if measured else None,
                    "measured_tip_displacement_y_mm": measured[1] if measured else None,
                    "measured_tip_displacement_z_mm": measured[2] if measured else None,
                    "measured_tip_displacement_norm_mm": row.get("measured_tip_displacement_norm_mm"),
                    "proximal_estimated_displacement_norm_mm": row.get("proximal_estimated_displacement_norm_mm"),
                    "distal_estimated_displacement_norm_mm": row.get("distal_estimated_displacement_norm_mm"),
                    "proximal_measured_displacement_norm_mm": row.get("proximal_measured_displacement_norm_mm"),
                    "distal_relative_measured_displacement_norm_mm": row.get("distal_relative_measured_displacement_norm_mm"),
                    "group_tag": row.get("group_tag"),
                    "bottom_x_cm": row.get("bottom_x_cm"),
                    "bottom_y_cm": row.get("bottom_y_cm"),
                    "top_x_cm": row.get("top_x_cm"),
                    "top_y_cm": row.get("top_y_cm"),
                }
            )
    return path


def _write_visit_plan_csv(path: Path, visit_dicts: Sequence[Mapping[str, Any]]) -> Path:
    fields = ["visit_position", "cycle_index", "visit_in_cycle", "target_index", "target_id", "group_tag", "amplitude_cm"]
    with Path(path).open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for visit in visit_dicts:
            writer.writerow({k: visit.get(k) for k in fields})
    return path


def _write_target_captures_csv(path: Path, visit_dicts: Sequence[Mapping[str, Any]]) -> Path:
    fields = [
        "timestamp_utc",
        "monotonic_time_s",
        "target_index",
        "target_id",
        "cycle_index",
        "visit_in_cycle",
        "visit_position",
        "bottom_x_cm",
        "bottom_y_cm",
        "top_x_cm",
        "top_y_cm",
        "distal_x_mm",
        "distal_y_mm",
        "distal_z_mm",
        "intermediate_x_mm",
        "intermediate_y_mm",
        "intermediate_z_mm",
        "command_success",
        "capture_accepted",
        "reject_reason",
        "tracker_age_s",
        "servo_telemetry_age_s",
        "group_tag",
        "amplitude_cm",
    ] + [f"goal_tick_servo_{i}" for i in range(1, 9)] + [
        f"present_tick_servo_{i}" for i in range(1, 9)
    ] + [f"load_proxy_servo_{i}_ma" for i in range(1, 9)]
    with Path(path).open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for visit in visit_dicts:
            distal = visit.get("distal_xyz_robot_mm") or [None, None, None]
            intermediate = visit.get("intermediate_xyz_robot_mm") or [None, None, None]
            goal_ticks = visit.get("all_8_goal_ticks") or {}
            present_ticks = visit.get("all_8_present_position_ticks") or {}
            load_proxies = visit.get("all_8_current_load_proxy_ma") or {}
            row = {
                "timestamp_utc": visit.get("timestamp_utc"),
                "monotonic_time_s": visit.get("monotonic_time_s"),
                "target_index": visit.get("target_index"),
                "target_id": visit.get("target_id"),
                "cycle_index": visit.get("cycle_index"),
                "visit_in_cycle": visit.get("visit_in_cycle"),
                "visit_position": visit.get("visit_position"),
                "bottom_x_cm": visit.get("bottom_x_cm"),
                "bottom_y_cm": visit.get("bottom_y_cm"),
                "top_x_cm": visit.get("top_x_cm"),
                "top_y_cm": visit.get("top_y_cm"),
                "distal_x_mm": distal[0] if len(distal) >= 1 else None,
                "distal_y_mm": distal[1] if len(distal) >= 2 else None,
                "distal_z_mm": distal[2] if len(distal) >= 3 else None,
                "intermediate_x_mm": intermediate[0] if intermediate and len(intermediate) >= 1 else None,
                "intermediate_y_mm": intermediate[1] if intermediate and len(intermediate) >= 2 else None,
                "intermediate_z_mm": intermediate[2] if intermediate and len(intermediate) >= 3 else None,
                "command_success": int(bool(visit.get("command_success"))) if visit.get("command_success") is not None else None,
                "capture_accepted": int(bool(visit.get("accepted") or visit.get("capture_accepted"))),
                "reject_reason": visit.get("reject_reason"),
                "tracker_age_s": visit.get("tracker_age_s"),
                "servo_telemetry_age_s": visit.get("servo_telemetry_age_s"),
                "group_tag": visit.get("group_tag"),
                "amplitude_cm": visit.get("amplitude_cm"),
            }
            for sid in range(1, 9):
                row[f"goal_tick_servo_{sid}"] = goal_ticks.get(str(sid))
                row[f"present_tick_servo_{sid}"] = present_ticks.get(str(sid))
                row[f"load_proxy_servo_{sid}_ma"] = load_proxies.get(str(sid))
            writer.writerow(row)
    return path


def _write_per_target_csv(path: Path, per_target_rows: Sequence[Mapping[str, Any]]) -> Path:
    fields = [
        "target_index",
        "target_id",
        "group_tag",
        "amplitude_cm",
        "bottom_x_cm",
        "bottom_y_cm",
        "top_x_cm",
        "top_y_cm",
        "accepted_repeats",
        "rms_spread_mm",
        "mean_radial_mm",
        "median_radial_mm",
        "max_radial_mm",
        "std_x_mm",
        "std_y_mm",
        "std_z_mm",
        "centroid_x_mm",
        "centroid_y_mm",
        "centroid_z_mm",
        "measured_tip_displacement_x_mm",
        "measured_tip_displacement_y_mm",
        "measured_tip_displacement_z_mm",
        "measured_tip_displacement_norm_mm",
        "proximal_command_norm_cm",
        "distal_command_norm_cm",
        "proximal_accepted_repeats",
        "proximal_centroid_x_mm",
        "proximal_centroid_y_mm",
        "proximal_centroid_z_mm",
        "proximal_rms_spread_mm",
        "distal_relative_accepted_repeats",
        "distal_relative_centroid_x_mm",
        "distal_relative_centroid_y_mm",
        "distal_relative_centroid_z_mm",
        "distal_relative_rms_spread_mm",
        "proximal_estimated_displacement_x_mm",
        "proximal_estimated_displacement_y_mm",
        "proximal_estimated_displacement_z_mm",
        "proximal_estimated_displacement_norm_mm",
        "distal_estimated_displacement_x_mm",
        "distal_estimated_displacement_y_mm",
        "distal_estimated_displacement_z_mm",
        "distal_estimated_displacement_norm_mm",
        "proximal_measured_displacement_norm_mm",
        "distal_relative_measured_displacement_norm_mm",
        "measured_displacement_reference_method",
        "segment_separation_method",
    ]
    with Path(path).open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in per_target_rows:
            centroid = row.get("centroid_xyz_mm") or [None, None, None]
            measured = row.get("measured_tip_displacement_xyz_mm") or [None, None, None]
            proximal_centroid = row.get("proximal_centroid_xyz_mm") or [None, None, None]
            distal_relative_centroid = row.get("distal_relative_centroid_xyz_mm") or [None, None, None]
            proximal_est = row.get("proximal_estimated_displacement_xyz_mm") or [None, None, None]
            distal_est = row.get("distal_estimated_displacement_xyz_mm") or [None, None, None]
            writer.writerow(
                {
                    "target_index": row.get("target_index"),
                    "target_id": row.get("target_id"),
                    "group_tag": row.get("group_tag"),
                    "amplitude_cm": row.get("amplitude_cm"),
                    "bottom_x_cm": row.get("bottom_x_cm"),
                    "bottom_y_cm": row.get("bottom_y_cm"),
                    "top_x_cm": row.get("top_x_cm"),
                    "top_y_cm": row.get("top_y_cm"),
                    "accepted_repeats": row.get("accepted_repeats"),
                    "rms_spread_mm": row.get("rms_spread_mm"),
                    "mean_radial_mm": row.get("mean_radial_mm"),
                    "median_radial_mm": row.get("median_radial_mm"),
                    "max_radial_mm": row.get("max_radial_mm"),
                    "std_x_mm": row.get("std_x_mm"),
                    "std_y_mm": row.get("std_y_mm"),
                    "std_z_mm": row.get("std_z_mm"),
                    "centroid_x_mm": centroid[0] if centroid else None,
                    "centroid_y_mm": centroid[1] if centroid else None,
                    "centroid_z_mm": centroid[2] if centroid else None,
                    "measured_tip_displacement_x_mm": measured[0] if measured else None,
                    "measured_tip_displacement_y_mm": measured[1] if measured else None,
                    "measured_tip_displacement_z_mm": measured[2] if measured else None,
                    "measured_tip_displacement_norm_mm": row.get("measured_tip_displacement_norm_mm"),
                    "proximal_command_norm_cm": row.get("proximal_command_norm_cm"),
                    "distal_command_norm_cm": row.get("distal_command_norm_cm"),
                    "proximal_accepted_repeats": row.get("proximal_accepted_repeats"),
                    "proximal_centroid_x_mm": proximal_centroid[0] if proximal_centroid else None,
                    "proximal_centroid_y_mm": proximal_centroid[1] if proximal_centroid else None,
                    "proximal_centroid_z_mm": proximal_centroid[2] if proximal_centroid else None,
                    "proximal_rms_spread_mm": row.get("proximal_rms_spread_mm"),
                    "distal_relative_accepted_repeats": row.get("distal_relative_accepted_repeats"),
                    "distal_relative_centroid_x_mm": (
                        distal_relative_centroid[0] if distal_relative_centroid else None
                    ),
                    "distal_relative_centroid_y_mm": (
                        distal_relative_centroid[1] if distal_relative_centroid else None
                    ),
                    "distal_relative_centroid_z_mm": (
                        distal_relative_centroid[2] if distal_relative_centroid else None
                    ),
                    "distal_relative_rms_spread_mm": row.get("distal_relative_rms_spread_mm"),
                    "proximal_estimated_displacement_x_mm": proximal_est[0] if proximal_est else None,
                    "proximal_estimated_displacement_y_mm": proximal_est[1] if proximal_est else None,
                    "proximal_estimated_displacement_z_mm": proximal_est[2] if proximal_est else None,
                    "proximal_estimated_displacement_norm_mm": row.get("proximal_estimated_displacement_norm_mm"),
                    "distal_estimated_displacement_x_mm": distal_est[0] if distal_est else None,
                    "distal_estimated_displacement_y_mm": distal_est[1] if distal_est else None,
                    "distal_estimated_displacement_z_mm": distal_est[2] if distal_est else None,
                    "distal_estimated_displacement_norm_mm": row.get("distal_estimated_displacement_norm_mm"),
                    "proximal_measured_displacement_norm_mm": row.get("proximal_measured_displacement_norm_mm"),
                    "distal_relative_measured_displacement_norm_mm": row.get("distal_relative_measured_displacement_norm_mm"),
                    "measured_displacement_reference_method": row.get("measured_displacement_reference_method"),
                    "segment_separation_method": row.get("segment_separation_method"),
                }
            )
    return path


def _write_metrics_summary_csv(path: Path, summary: Mapping[str, Any]) -> Path:
    with Path(path).open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["metric", "value"])
        for key, value in summary.items():
            writer.writerow([str(key), _csv_value(value)])
    return path


def _write_summary_text(path: Path, *, metrics: Mapping[str, Any], summary: Mapping[str, Any]) -> Path:
    lines = [
        "Two-Segment Workspace Repeatability",
        f"schema_version: {metrics.get('schema_version', '')}",
        f"experiment_name: {metrics.get('experiment_name', 'two_segment_workspace_repeatability')}",
        f"demo_only: {metrics.get('demo_only', False)}",
        f"valid_for_repeatability_analysis: {metrics.get('valid_for_repeatability_analysis', True)}",
        f"valid_for_thesis_repeatability: {metrics.get('valid_for_thesis_repeatability', False)}",
        f"valid_for_model_training: {metrics.get('valid_for_model_training', False)}",
        f"primary_metric: {metrics.get('primary_metric', 'distal_xyz_repeatability')}",
        f"controlled_point: {metrics.get('controlled_point', 'distal_tip coil origin in robot base frame')}",
        f"expected_distal_tool_id: {metrics.get('expected_distal_tool_id', '0A')}",
        "",
        "Protocol:",
        f"  target_count:                  {metrics.get('target_count', 0)}",
        f"  repeats_per_target:            {metrics.get('repeats_per_target', 0)}",
        f"  planned_visits:                {metrics.get('planned_visits', 0)}",
        f"  accepted_captures:             {metrics.get('accepted_captures', 0)}",
        f"  rejected_captures:             {metrics.get('rejected_captures', 0)}",
        f"  stop_reason:                   {metrics.get('stop_reason', '')}",
        f"  target_generator_mode:         {metrics.get('target_generator_mode', '')}",
        f"  random_seed:                   {metrics.get('random_seed', 0)}",
        f"  max_segment_displacement_cm:   {metrics.get('max_segment_displacement_cm', 0)}",
        f"  return_to_neutral_between:     {metrics.get('return_to_neutral_between_visits', True)}",
        f"  neutral_settle_s:              {metrics.get('neutral_settle_s', 0)}",
        f"  target_settle_s:               {metrics.get('target_settle_s', 0)}",
        f"  bottom_segment_key:            {metrics.get('bottom_segment_key', '')}",
        f"  top_segment_key:               {metrics.get('top_segment_key', '')}",
        "",
        "Repeatability summary (distal XYZ):",
        f"  overall_distal_rms_mm:         {summary.get('overall_distal_rms_mm')}",
        f"  median_per_target_rms_mm:      {summary.get('median_per_target_rms_mm')}",
        f"  p95_per_target_rms_mm:         {summary.get('p95_per_target_rms_mm')}",
        f"  worst_target_rms_mm:           {summary.get('worst_target_rms_mm')} (id={summary.get('worst_target_id')})",
        f"  best_target_rms_mm:            {summary.get('best_target_rms_mm')} (id={summary.get('best_target_id')})",
        f"  targets_with_data:             {summary.get('targets_with_data')}",
        f"  targets_below_minimum_repeats: {summary.get('targets_below_minimum_repeats')}",
        f"  minimum_repeats_per_target:    {summary.get('minimum_repeats_per_target')}",
        f"  mean_repeats_per_target:       {summary.get('mean_repeats_per_target')}",
        "",
        "Measured-space analysis:",
        f"  measured_reference_method:      {summary.get('measured_displacement_reference_method')}",
        f"  measured_tip_reference_xyz_mm:  {summary.get('measured_tip_reference_xyz_mm')}",
        f"  measured_tip_disp_median_mm:    {summary.get('measured_tip_displacement_median_mm')}",
        f"  measured_tip_disp_max_mm:       {summary.get('measured_tip_displacement_max_mm')}",
        f"  intermediate_marker_targets:    {summary.get('intermediate_marker_targets_with_data')}",
        f"  segment_separation_method:      {summary.get('segment_separation_method')}",
        f"  proximal_est_disp_max_mm:       {summary.get('proximal_estimated_displacement_max_mm')}",
        f"  distal_est_disp_max_mm:         {summary.get('distal_estimated_displacement_max_mm')}",
        "",
        "This is a data-collection workflow. Repeatability is reported honestly;",
        "the run can be valid even if RMS exceeds any operator-configured target.",
    ]
    Path(path).write_text("\n".join(lines).strip() + "\n", encoding="utf-8")
    return path


# ---------------------------------------------------------------------------
# Figures — mirror single-segment workspace_repeatability_map figure set
# ---------------------------------------------------------------------------


def _color_vmax(values: Sequence[float]) -> float:
    if not values:
        return 1.0
    arr = np.asarray(values, dtype=float)
    p95 = float(np.percentile(arr, 95.0))
    return max(p95, float(arr.max()) * 0.5, 1e-6)


def _accepted_distal_capture_displacements(
    visit_dicts: Sequence[Mapping[str, Any]],
    *,
    reference_xyz: np.ndarray | None,
) -> np.ndarray | None:
    if reference_xyz is None:
        return None
    values: list[np.ndarray] = []
    for visit in visit_dicts:
        if not bool(visit.get("accepted") or visit.get("capture_accepted")):
            continue
        xyz = _xyz_array(visit.get("distal_xyz_robot_mm") or visit.get("position_mm"))
        if xyz is not None:
            values.append(xyz - reference_xyz)
    if not values:
        return None
    return np.vstack(values).astype(float)


def _row_xyz_values(rows: Sequence[Mapping[str, Any]], key: str) -> np.ndarray | None:
    values = [_xyz_array(row.get(key)) for row in rows]
    values = [value for value in values if value is not None]
    if not values:
        return None
    return np.vstack(values).astype(float)


def _write_thesis_01(
    path: Path,
    *,
    rows_with_data: Sequence[Mapping[str, Any]],
    visit_dicts: Sequence[Mapping[str, Any]],
    summary: Mapping[str, Any],
    max_amplitude_mm: float,
    create_3d_figure,
    save_figure,
    set_equal_xyz,
    style_3d_axes,
) -> Path:
    fig, ax = create_3d_figure(size="thesis_3d")
    reference = _xyz_array(summary.get("measured_tip_reference_xyz_mm"))
    capture_xyz = _accepted_distal_capture_displacements(visit_dicts, reference_xyz=reference)
    if capture_xyz is not None:
        ax.scatter(
            capture_xyz[:, 0],
            capture_xyz[:, 1],
            capture_xyz[:, 2],
            color="#cbd5e1",
            s=3,
            alpha=0.16,
            depthshade=False,
        )
    plotted_rows = [
        r
        for r in rows_with_data
        if _xyz_array(r.get("measured_tip_displacement_xyz_mm")) is not None
        and r.get("rms_spread_mm") is not None
    ]
    points = np.asarray(
        [_xyz_array(r.get("measured_tip_displacement_xyz_mm")) for r in plotted_rows],
        dtype=float,
    )
    rms = [float(r["rms_spread_mm"]) for r in plotted_rows]
    vmax = _color_vmax(rms)
    sc = ax.scatter(
        points[:, 0],
        points[:, 1],
        points[:, 2],
        c=rms,
        cmap="viridis",
        vmin=0.0,
        vmax=vmax,
        s=24,
        edgecolors="#111827",
        linewidths=0.25,
        depthshade=False,
    )
    all_xyz = points if capture_xyz is None else np.vstack([capture_xyz, points])
    set_equal_xyz(ax, x_values=all_xyz[:, 0], y_values=all_xyz[:, 1], z_values=all_xyz[:, 2], pad_fraction=0.08)
    style_3d_axes(
        ax,
        title=(
            "Measured Workspace Repeatability\n"
            f"overall RMS = {_fmt(summary.get('overall_distal_rms_mm'))} mm"
        ),
        xlabel="Tip dX (mm)",
        ylabel="Tip dY (mm)",
        zlabel="Tip dZ (mm)",
        labelpad=7.0,
        view_elev=22.0,
        view_azim=-45.0,
    )
    fig.colorbar(sc, ax=ax, label="RMS (mm)", shrink=0.68, pad=0.12)
    save_figure(fig, path)
    return path


def _write_thesis_02(
    path: Path,
    *,
    rows_with_data: Sequence[Mapping[str, Any]],
    visit_dicts: Sequence[Mapping[str, Any]],
    summary: Mapping[str, Any],
    max_amplitude_mm: float,
    create_figure,
    save_figure,
    set_equal_xy,
    style_axes,
) -> Path:
    fig, ax = create_figure(size="wide")
    reference = _xyz_array(summary.get("measured_tip_reference_xyz_mm"))
    capture_xyz = _accepted_distal_capture_displacements(visit_dicts, reference_xyz=reference)
    if capture_xyz is not None:
        ax.scatter(
            capture_xyz[:, 0],
            capture_xyz[:, 1],
            color="#cbd5e1",
            s=5,
            alpha=0.14,
            linewidths=0,
            label="captures",
        )
    plotted_rows = [r for r in rows_with_data if r.get("x_mm") is not None and r.get("y_mm") is not None]
    xs = [float(r["x_mm"]) for r in plotted_rows]
    ys = [float(r["y_mm"]) for r in plotted_rows]
    zs = [float(r["rms_spread_mm"]) for r in plotted_rows]
    vmax = _color_vmax(zs)
    sc = ax.scatter(xs, ys, c=zs, cmap="viridis", vmin=0.0, vmax=vmax, s=34, edgecolors="white", linewidths=0.4)
    ax.axhline(0.0, color="#94a3b8", linewidth=0.7)
    ax.axvline(0.0, color="#94a3b8", linewidth=0.7)
    fig.colorbar(sc, ax=ax, label="RMS (mm)", pad=0.03)
    style_axes(
        ax,
        title="Measured Tip Repeatability",
        xlabel="Tip dX (mm)",
        ylabel="Tip dY (mm)",
    )
    x_for_limits = xs if capture_xyz is None else list(capture_xyz[:, 0]) + xs
    y_for_limits = ys if capture_xyz is None else list(capture_xyz[:, 1]) + ys
    set_equal_xy(ax, x_values=x_for_limits, y_values=y_for_limits, pad_fraction=0.08)
    save_figure(fig, path)
    return path


def _write_thesis_03(
    path: Path,
    *,
    rows_with_data: Sequence[Mapping[str, Any]],
    summary: Mapping[str, Any],
    max_amplitude_mm: float,
    create_figure,
    save_figure,
    style_axes,
) -> Path:
    fig, ax = create_figure(size="wide")
    amplitudes = [float(r.get("measured_tip_displacement_norm_mm") or 0.0) for r in rows_with_data]
    rms = [float(r["rms_spread_mm"]) for r in rows_with_data]
    color_values = [float(r.get("distal_command_norm_cm") or 0.0) for r in rows_with_data]
    sc = ax.scatter(amplitudes, rms, c=color_values, cmap="plasma", s=22, alpha=0.78, edgecolors="none")
    fig.colorbar(sc, ax=ax, label="distal command (cm)")
    if summary.get("overall_distal_rms_mm") is not None:
        ax.axhline(
            float(summary["overall_distal_rms_mm"]),
            color="#dc2626",
            linewidth=0.8,
            linestyle="--",
            label=f"overall RMS = {float(summary['overall_distal_rms_mm']):.3f} mm",
        )
    if summary.get("target_distal_rms_threshold_mm") is not None:
        ax.axhline(
            float(summary["target_distal_rms_threshold_mm"]),
            color="#0ea5e9",
            linewidth=0.8,
            linestyle=":",
            label=f"operator threshold = {float(summary['target_distal_rms_threshold_mm']):.3f} mm",
        )
    style_axes(
        ax,
        title="Repeatability vs Measured Tip Displacement",
        xlabel="Measured tip displacement (mm)",
        ylabel="RMS spread (mm)",
    )
    ax.legend(loc="upper left", fontsize=8)
    save_figure(fig, path)
    return path


def _write_thesis_04(
    path: Path,
    *,
    rows_with_data: Sequence[Mapping[str, Any]],
    summary: Mapping[str, Any],
    max_amplitude_mm: float,
    save_figure,
    set_equal_xy,
    style_axes,
) -> Path:
    from continuum_robot.experiments.plotting import report_style

    with report_style() as plt:
        fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.9), constrained_layout=True)
    segment_method = str(summary.get("segment_separation_method") or "")
    use_direct = segment_method == "direct_intermediate_marker"
    proximal_key = "proximal_measured_displacement_xyz_mm" if use_direct else "proximal_estimated_displacement_xyz_mm"
    distal_key = "distal_relative_measured_displacement_xyz_mm" if use_direct else "distal_estimated_displacement_xyz_mm"
    proximal = _row_xyz_values(rows_with_data, proximal_key)
    distal = _row_xyz_values(rows_with_data, distal_key)
    _plot_segment_displacement_panel(
        fig=fig,
        ax=axes[0],
        values=proximal,
        title="Proximal Segment",
        set_equal_xy=set_equal_xy,
        style_axes=style_axes,
    )
    _plot_segment_displacement_panel(
        fig=fig,
        ax=axes[1],
        values=distal,
        title="Distal Segment",
        set_equal_xy=set_equal_xy,
        style_axes=style_axes,
    )
    save_figure(fig, path)
    return path


def _plot_segment_displacement_panel(
    *,
    fig,
    ax,
    values: np.ndarray | None,
    title: str,
    set_equal_xy,
    style_axes,
) -> None:
    if values is None or values.size == 0:
        style_axes(ax, title=title, xlabel="dX (mm)", ylabel="dY (mm)")
        ax.text(0.5, 0.5, "not recorded", transform=ax.transAxes, ha="center", va="center")
        return
    norms = np.linalg.norm(values, axis=1)
    vmax = _color_vmax([float(v) for v in norms])
    ax.axhline(0.0, color="#94a3b8", linewidth=0.7)
    ax.axvline(0.0, color="#94a3b8", linewidth=0.7)
    sc = ax.scatter(
        values[:, 0],
        values[:, 1],
        c=norms,
        cmap="magma",
        vmin=0.0,
        vmax=vmax,
        s=20,
        edgecolors="white",
        linewidths=0.25,
    )
    fig.colorbar(sc, ax=ax, label="norm (mm)", pad=0.03)
    style_axes(
        ax,
        title=title,
        xlabel="dX (mm)",
        ylabel="dY (mm)",
    )
    set_equal_xy(ax, x_values=values[:, 0], y_values=values[:, 1], pad_fraction=0.12)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _as_dict(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    if hasattr(value, "to_dict"):
        return dict(value.to_dict())
    if hasattr(value, "__dict__"):
        return {k: getattr(value, k) for k in vars(value)}
    return {}


def _json_default(value: Any) -> Any:
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def _csv_value(value: Any) -> Any:
    if value is None:
        return ""
    if isinstance(value, (list, dict, tuple)):
        return json.dumps(value, default=_json_default)
    return value


def _fmt(value: Any) -> str:
    if value is None:
        return "n/a"
    try:
        return f"{float(value):.3f}"
    except Exception:
        return str(value)

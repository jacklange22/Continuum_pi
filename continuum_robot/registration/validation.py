"""Registration validation and comparison helpers."""

from __future__ import annotations

from itertools import combinations
from typing import Any, Mapping, Sequence

import numpy as np


def compute_fre_mm(residuals_xyz_mm: Sequence[Sequence[float]]) -> float:
    """Compute fiducial registration error from residual vectors."""
    if not residuals_xyz_mm:
        raise ValueError("residuals_xyz_mm must not be empty")
    total = 0.0
    count = 0
    for x, y, z in residuals_xyz_mm:
        total += x * x + y * y + z * z
        count += 1
    return (total / count) ** 0.5


def compute_residual_norms_mm(residuals_xyz_mm_by_label: Mapping[str, Sequence[float]]) -> dict[str, float]:
    """Return Euclidean residual norms keyed by landmark label."""
    return {
        str(label): float(np.linalg.norm(np.asarray(vector, dtype=float)))
        for label, vector in residuals_xyz_mm_by_label.items()
    }


def summarize_residual_norms_mm(residual_norms_mm_by_label: Mapping[str, float]) -> dict[str, Any]:
    """Summarize per-landmark residual norms."""
    if not residual_norms_mm_by_label:
        return {
            "landmark_count": 0,
            "mean_residual_mm": None,
            "median_residual_mm": None,
            "std_residual_mm": None,
            "max_residual_mm": None,
            "min_residual_mm": None,
            "worst_landmark_label": None,
            "worst_landmark_residual_mm": None,
        }

    ordered = {str(label): float(value) for label, value in residual_norms_mm_by_label.items()}
    labels = list(ordered.keys())
    values = np.asarray(list(ordered.values()), dtype=float)
    worst_index = int(np.argmax(values))
    return {
        "landmark_count": len(labels),
        "mean_residual_mm": float(values.mean()),
        "median_residual_mm": float(np.median(values)),
        "std_residual_mm": float(values.std(ddof=0)),
        "max_residual_mm": float(values.max()),
        "min_residual_mm": float(values.min()),
        "worst_landmark_label": labels[worst_index],
        "worst_landmark_residual_mm": float(values[worst_index]),
    }


def compute_geometry_diagnostics(points_xyz_mm: Sequence[Sequence[float]]) -> dict[str, Any]:
    """Describe landmark geometry conditioning for registration trust reporting."""
    points = np.asarray(points_xyz_mm, dtype=float)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("points_xyz_mm must have shape (N, 3)")
    if points.shape[0] == 0:
        raise ValueError("points_xyz_mm must not be empty")

    centroid = points.mean(axis=0)
    centered = points - centroid
    singular_values = np.linalg.svd(centered, compute_uv=False)
    nonzero_singular_values = [float(value) for value in singular_values if float(value) > 1e-9]
    rank = int(np.linalg.matrix_rank(centered))
    condition_number = None
    if len(nonzero_singular_values) >= 2:
        condition_number = float(max(nonzero_singular_values) / min(nonzero_singular_values))

    pairwise_distances = [
        float(np.linalg.norm(points[left] - points[right]))
        for left, right in combinations(range(points.shape[0]), 2)
    ]
    return {
        "point_count": int(points.shape[0]),
        "centroid_mm": centroid.astype(float).tolist(),
        "geometry_rank": rank,
        "singular_values_mm": [float(value) for value in singular_values.tolist()],
        "condition_number": condition_number,
        "min_pairwise_distance_mm": min(pairwise_distances) if pairwise_distances else None,
        "mean_pairwise_distance_mm": float(np.mean(pairwise_distances)) if pairwise_distances else None,
        "max_pairwise_distance_mm": max(pairwise_distances) if pairwise_distances else None,
    }


def compute_transform_delta(T_reference: Sequence[Sequence[float]], T_candidate: Sequence[Sequence[float]]) -> dict[str, float]:
    """Return translation and rotation difference between rigid transforms."""
    reference = np.asarray(T_reference, dtype=float)
    candidate = np.asarray(T_candidate, dtype=float)
    if reference.shape != (4, 4) or candidate.shape != (4, 4):
        raise ValueError("Both transforms must have shape (4, 4)")
    delta = np.linalg.inv(reference) @ candidate
    translation_mm = float(np.linalg.norm(delta[0:3, 3]))
    rotation_deg = _rotation_angle_deg(delta[0:3, 0:3])
    return {
        "translation_delta_mm": translation_mm,
        "rotation_delta_deg": rotation_deg,
    }


def build_registration_history_summary(
    *,
    current_payload: Mapping[str, Any],
    current_path: str | None,
    previous_runs: Sequence[tuple[str, Mapping[str, Any]]],
) -> dict[str, Any]:
    """Summarize repeated registration solves against prior accepted runs."""
    current_T = _extract_transform(current_payload, "T_robot_aurora")
    current_fre = _extract_fre_mm(current_payload)
    current_residual_norms = _extract_residual_norms_mm_by_label(current_payload)

    comparisons: list[dict[str, Any]] = []
    all_fres: list[float] = [current_fre] if current_fre is not None else []
    all_residuals_by_label: dict[str, list[float]] = {
        label: [float(value)]
        for label, value in current_residual_norms.items()
    }

    for path, payload in previous_runs:
        previous_T = _extract_transform(payload, "T_robot_aurora")
        if current_T is None or previous_T is None:
            delta = {"translation_delta_mm": None, "rotation_delta_deg": None}
        else:
            delta = compute_transform_delta(previous_T, current_T)

        previous_fre = _extract_fre_mm(payload)
        if previous_fre is not None:
            all_fres.append(previous_fre)
        previous_residual_norms = _extract_residual_norms_mm_by_label(payload)
        for label, value in previous_residual_norms.items():
            all_residuals_by_label.setdefault(label, []).append(float(value))

        comparisons.append(
            {
                "path": str(path),
                "timestamp_utc": payload.get("timestamp_utc"),
                "fre_mm": previous_fre,
                "fre_delta_mm": (
                    abs(float(current_fre) - float(previous_fre))
                    if current_fre is not None and previous_fre is not None
                    else None
                ),
                "translation_delta_mm": delta["translation_delta_mm"],
                "rotation_delta_deg": delta["rotation_delta_deg"],
            }
        )

    translation_deltas = [
        float(item["translation_delta_mm"])
        for item in comparisons
        if item.get("translation_delta_mm") is not None
    ]
    rotation_deltas = [
        float(item["rotation_delta_deg"])
        for item in comparisons
        if item.get("rotation_delta_deg") is not None
    ]
    residual_summary_by_label: dict[str, Any] = {}
    worst_label = None
    worst_mean = None
    for label, values in sorted(all_residuals_by_label.items()):
        arr = np.asarray(values, dtype=float)
        mean_value = float(arr.mean())
        residual_summary_by_label[label] = {
            "count": int(arr.size),
            "mean_mm": mean_value,
            "std_mm": float(arr.std(ddof=0)),
            "max_mm": float(arr.max()),
            "min_mm": float(arr.min()),
        }
        if worst_mean is None or mean_value > worst_mean:
            worst_mean = mean_value
            worst_label = label

    fre_summary = _summarize_scalar_values(all_fres)
    translation_summary = _summarize_scalar_values(translation_deltas)
    rotation_summary = _summarize_scalar_values(rotation_deltas)

    return {
        "current_registration_path": str(current_path) if current_path else None,
        "current_timestamp_utc": current_payload.get("timestamp_utc"),
        "history_run_count": 1 + len(previous_runs),
        "comparison_count": len(comparisons),
        "comparisons_to_recent_runs": comparisons,
        "fre_summary_mm": fre_summary,
        "translation_delta_summary_mm": translation_summary,
        "rotation_delta_summary_deg": rotation_summary,
        "per_landmark_residual_summary_mm_by_label": residual_summary_by_label,
        "worst_landmark_by_mean_residual": worst_label,
        "worst_landmark_mean_residual_mm": worst_mean,
    }


def _extract_transform(payload: Mapping[str, Any], key: str) -> np.ndarray | None:
    matrix = payload.get(key)
    if matrix is None and key == "T_robot_aurora":
        matrix = payload.get("T_aurora_2_model")
    if matrix is None and key == "T_coil_tip":
        matrix = payload.get("T_tip_2_coil")
        if matrix is not None:
            matrix = np.linalg.inv(np.asarray(matrix, dtype=float)).tolist()
    if matrix is None:
        return None
    array = np.asarray(matrix, dtype=float)
    if array.shape != (4, 4):
        return None
    return array


def _extract_fre_mm(payload: Mapping[str, Any]) -> float | None:
    metrics = payload.get("validation_metrics", {})
    if isinstance(metrics, Mapping):
        value = metrics.get("overall_fre_mm")
        if value is not None:
            return float(value)
    value = payload.get("fre_mm")
    return float(value) if value is not None else None


def _extract_residual_norms_mm_by_label(payload: Mapping[str, Any]) -> dict[str, float]:
    metrics = payload.get("validation_metrics", {})
    if isinstance(metrics, Mapping):
        residuals = metrics.get("residual_norms_mm_by_label")
        if isinstance(residuals, Mapping):
            return {str(label): float(value) for label, value in residuals.items()}
    raw_residuals = payload.get("residuals_robot_xyz_mm")
    if isinstance(raw_residuals, Mapping):
        return compute_residual_norms_mm(
            {
                str(label): [float(component) for component in value]
                for label, value in raw_residuals.items()
                if isinstance(value, Sequence)
            }
        )
    return {}


def _rotation_angle_deg(R: np.ndarray) -> float:
    value = float((np.trace(R) - 1.0) / 2.0)
    value = min(1.0, max(-1.0, value))
    return float(np.degrees(np.arccos(value)))


def _summarize_scalar_values(values: Sequence[float]) -> dict[str, Any]:
    if not values:
        return {
            "count": 0,
            "mean": None,
            "std": None,
            "min": None,
            "max": None,
        }
    arr = np.asarray(values, dtype=float)
    return {
        "count": int(arr.size),
        "mean": float(arr.mean()),
        "std": float(arr.std(ddof=0)),
        "min": float(arr.min()),
        "max": float(arr.max()),
    }

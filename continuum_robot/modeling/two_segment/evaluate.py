"""Evaluation metrics for two-segment modeling."""

from __future__ import annotations

from typing import Any

import numpy as np


def position_metrics(y_true: np.ndarray, y_pred: np.ndarray, *, prefix: str = "") -> dict[str, float]:
    """Return XYZ position metrics in mm."""

    if y_true.size == 0 or y_pred.size == 0:
        return {}
    truth = np.asarray(y_true[:, :3], dtype=float)
    pred = np.asarray(y_pred[:, :3], dtype=float)
    error = pred - truth
    norm = np.linalg.norm(error, axis=1)
    xy = np.linalg.norm(error[:, :2], axis=1)
    axis_rmse = np.sqrt(np.mean(error * error, axis=0))
    base = {
        "xyz_rmse_mm": float(np.sqrt(np.mean(norm * norm))),
        "xy_rmse_mm": float(np.sqrt(np.mean(xy * xy))),
        "z_rmse_mm": float(np.sqrt(np.mean(error[:, 2] * error[:, 2]))),
        "x_rmse_mm": float(axis_rmse[0]),
        "y_rmse_mm": float(axis_rmse[1]),
        "mean_error_mm": float(np.mean(norm)),
        "median_error_mm": float(np.median(norm)),
        "p95_error_mm": float(np.percentile(norm, 95)),
        "max_error_mm": float(np.max(norm)),
    }
    return {f"{prefix}{key}": value for key, value in base.items()}


def orientation_metrics(y_true: np.ndarray, y_pred: np.ndarray, *, prefix: str = "") -> dict[str, float]:
    """Return angular tangent/orientation metrics in degrees when available."""

    if y_true.shape[1] < 3 or y_pred.shape[1] < 3:
        return {}
    if y_true.shape[1] >= 6 and y_pred.shape[1] >= 6:
        truth = _normalize(y_true[:, 3:6])
        pred = _normalize(y_pred[:, 3:6])
    else:
        truth = _normalize(y_true[:, :3])
        pred = _normalize(y_pred[:, :3])
    dots = np.clip(np.sum(truth * pred, axis=1), -1.0, 1.0)
    angles = np.degrees(np.arccos(dots))
    tangent_rmse = np.sqrt(np.mean((pred - truth) * (pred - truth), axis=0))
    base = {
        "orientation_mean_error_deg": float(np.mean(angles)),
        "orientation_median_error_deg": float(np.median(angles)),
        "orientation_p95_error_deg": float(np.percentile(angles, 95)),
        "orientation_max_error_deg": float(np.max(angles)),
        "tangent_rmse_x": float(tangent_rmse[0]),
        "tangent_rmse_y": float(tangent_rmse[1]),
        "tangent_rmse_z": float(tangent_rmse[2]),
    }
    return {f"{prefix}{key}": value for key, value in base.items()}


def all_metrics(y_true: np.ndarray, y_pred: np.ndarray, *, label_metadata: dict[str, Any] | None = None) -> dict[str, Any]:
    if not label_metadata:
        metrics = position_metrics(y_true, y_pred)
        metrics.update(orientation_metrics(y_true, y_pred))
        return metrics
    slices = label_metadata.get("label_slices") if isinstance(label_metadata.get("label_slices"), dict) else {}
    distal_slice = _slice_for(slices, role="distal_tip", kind="position") or [0, 1, 2]
    metrics = position_metrics(y_true[:, distal_slice], y_pred[:, distal_slice], prefix="distal_")
    for key in list(metrics):
        if key.startswith("distal_"):
            metrics[key.removeprefix("distal_")] = metrics[key]
    intermediate_slice = _slice_for(slices, role="intermediate_segment", kind="position")
    if intermediate_slice:
        inter = position_metrics(y_true[:, intermediate_slice], y_pred[:, intermediate_slice], prefix="intermediate_")
        metrics.update(inter)
        combined_truth = np.concatenate([y_true[:, intermediate_slice], y_true[:, distal_slice]], axis=1)
        combined_pred = np.concatenate([y_pred[:, intermediate_slice], y_pred[:, distal_slice]], axis=1)
        combined_error = combined_pred - combined_truth
        metrics["two_coil_combined_xyz_rmse_mm"] = float(np.sqrt(np.mean(combined_error * combined_error)))
    for role, prefix in [("distal_tip", "distal_"), ("intermediate_segment", "intermediate_")]:
        orient_slice = _slice_for(slices, role=role, kind="orientation")
        if orient_slice:
            role_metrics = orientation_metrics(y_true[:, orient_slice], y_pred[:, orient_slice], prefix=prefix)
            metrics.update(role_metrics)
            if role == "distal_tip":
                for key, value in role_metrics.items():
                    if key.startswith(prefix):
                        metrics[key.removeprefix(prefix)] = value
    return metrics


def _slice_for(slices: dict[str, Any], *, role: str, kind: str) -> list[int] | None:
    role_slices = slices.get(role) if isinstance(slices.get(role), dict) else {}
    value = role_slices.get(kind)
    if isinstance(value, list) and len(value) == 3:
        return [int(item) for item in value]
    return None


def _normalize(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1)
    norms = np.where(norms == 0.0, 1.0, norms)
    return matrix / norms[:, None]

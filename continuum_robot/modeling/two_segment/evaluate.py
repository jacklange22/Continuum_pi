"""Evaluation metrics for two-segment modeling."""

from __future__ import annotations

from typing import Any

import numpy as np


def position_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """Return XYZ position metrics in mm."""

    if y_true.size == 0 or y_pred.size == 0:
        return {}
    truth = np.asarray(y_true[:, :3], dtype=float)
    pred = np.asarray(y_pred[:, :3], dtype=float)
    error = pred - truth
    norm = np.linalg.norm(error, axis=1)
    xy = np.linalg.norm(error[:, :2], axis=1)
    axis_rmse = np.sqrt(np.mean(error * error, axis=0))
    return {
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


def orientation_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """Return angular tangent/orientation metrics in degrees when available."""

    if y_true.shape[1] < 6 or y_pred.shape[1] < 6:
        return {}
    truth = _normalize(y_true[:, 3:6])
    pred = _normalize(y_pred[:, 3:6])
    dots = np.clip(np.sum(truth * pred, axis=1), -1.0, 1.0)
    angles = np.degrees(np.arccos(dots))
    tangent_rmse = np.sqrt(np.mean((pred - truth) * (pred - truth), axis=0))
    return {
        "orientation_mean_error_deg": float(np.mean(angles)),
        "orientation_median_error_deg": float(np.median(angles)),
        "orientation_p95_error_deg": float(np.percentile(angles, 95)),
        "orientation_max_error_deg": float(np.max(angles)),
        "tangent_rmse_x": float(tangent_rmse[0]),
        "tangent_rmse_y": float(tangent_rmse[1]),
        "tangent_rmse_z": float(tangent_rmse[2]),
    }


def all_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, Any]:
    metrics = position_metrics(y_true, y_pred)
    metrics.update(orientation_metrics(y_true, y_pred))
    return metrics


def _normalize(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1)
    norms = np.where(norms == 0.0, 1.0, norms)
    return matrix / norms[:, None]

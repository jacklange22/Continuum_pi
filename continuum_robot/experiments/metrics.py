"""Shared point-set metrics for experiment analysis."""

from __future__ import annotations

from collections import defaultdict
from typing import Any

import numpy as np


def as_points_n3(points: list[list[float]] | np.ndarray, *, name: str) -> np.ndarray:
    """Coerce points into an ``(N, 3)`` float array."""
    arr = np.asarray(points, dtype=float)
    if arr.ndim != 2 or arr.shape[1] != 3:
        raise ValueError(f"{name} must have shape (N, 3)")
    if arr.shape[0] == 0:
        raise ValueError(f"{name} must contain at least one point")
    if not np.isfinite(arr).all():
        raise ValueError(f"{name} contains non-finite values")
    return arr


def centroid_mm(points: list[list[float]] | np.ndarray) -> np.ndarray:
    """Return the centroid of an ``(N, 3)`` point set."""
    arr = as_points_n3(points, name="points")
    return arr.mean(axis=0)


def spread_rms_mm(points: list[list[float]] | np.ndarray, *, centroid_xyz: np.ndarray | None = None) -> float:
    """Return the RMS distance of points to their centroid."""
    arr = as_points_n3(points, name="points")
    center = np.asarray(centroid_xyz, dtype=float) if centroid_xyz is not None else arr.mean(axis=0)
    distances_sq = np.sum((arr - center.reshape(1, 3)) ** 2, axis=1)
    return float(np.sqrt(np.mean(distances_sq)))


def rms_error_mm(
    measured_points: list[list[float]] | np.ndarray,
    truth_points: list[list[float]] | np.ndarray,
) -> float:
    """Return RMS pointwise error between measured and truth coordinates."""
    measured = as_points_n3(measured_points, name="measured_points")
    truth = as_points_n3(truth_points, name="truth_points")
    if measured.shape != truth.shape:
        raise ValueError("measured_points and truth_points must have the same shape")
    errors_sq = np.sum((measured - truth) ** 2, axis=1)
    return float(np.sqrt(np.mean(errors_sq)))


def per_axis_bias_mm(
    measured_points: list[list[float]] | np.ndarray,
    truth_points: list[list[float]] | np.ndarray,
) -> list[float]:
    """Return mean signed bias per axis."""
    measured = as_points_n3(measured_points, name="measured_points")
    truth = as_points_n3(truth_points, name="truth_points")
    if measured.shape != truth.shape:
        raise ValueError("measured_points and truth_points must have the same shape")
    return [float(value) for value in (measured - truth).mean(axis=0)]


def pointwise_rms_errors_mm(
    measured_by_key: dict[Any, list[list[float]]],
    truth_by_key: dict[Any, list[float]],
) -> dict[str, float]:
    """Return RMS error per point key."""
    output: dict[str, float] = {}
    for key, measured_points in measured_by_key.items():
        truth = truth_by_key.get(key)
        if truth is None or not measured_points:
            continue
        truth_stack = np.repeat(np.asarray(truth, dtype=float).reshape(1, 3), len(measured_points), axis=0)
        output[str(key)] = rms_error_mm(measured_points, truth_stack.tolist())
    return output


def group_positions_by_key(
    samples: list[Any],
    *,
    key_fn,
    position_fn,
) -> dict[Any, list[list[float]]]:
    """Group extracted positions by a caller-defined sample key."""
    grouped: dict[Any, list[list[float]]] = defaultdict(list)
    for sample in samples:
        key = key_fn(sample)
        position = position_fn(sample)
        if key is None or position is None:
            continue
        grouped[key].append([float(value) for value in position])
    return dict(grouped)

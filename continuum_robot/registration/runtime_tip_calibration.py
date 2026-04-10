"""Runtime 0A hat-based tip calibration helpers."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from continuum_robot.registration.legacy_compat import (
    AuroraPoseSample,
    average_tool_samples_as_transform,
)
from continuum_robot.registration.rigid_solver import RigidRegistrationSolver
from continuum_robot.registration.validation import (
    compute_fre_mm,
    compute_residual_norms_mm,
    summarize_residual_norms_mm,
)
from continuum_robot.tracking.transforms import (
    assert_rigid_transform_matrix,
    make_transform_A_B,
)


@dataclass(frozen=True)
class RuntimeTipTruthGeometry:
    """Known hat geometry used to solve the runtime tip frame."""

    points_file: Path
    transform_file: Path
    labels: list[str]
    truth_points_in_sw_by_label: dict[str, list[float]]
    truth_points_in_tip_by_label: dict[str, list[float]]
    T_sw_tip: np.ndarray


@dataclass(frozen=True)
class RuntimeTipCalibrationResult:
    """Solved runtime 0A-to-tip transform plus validation metrics."""

    labels: list[str]
    averaged_points_by_label: dict[str, list[float]]
    residuals_tip_xyz_mm_by_label: dict[str, list[float]]
    residual_norms_mm_by_label: dict[str, float]
    point_spread_mm_by_label: dict[str, float]
    fit_rmse_mm: float
    max_residual_mm: float
    T_tip_aurora: np.ndarray
    T_aurora_tip: np.ndarray
    T_aurora_coil_avg: np.ndarray
    T_coil_tip: np.ndarray
    averaged_coil_quaternion_wxyz: np.ndarray
    averaged_coil_translation_mm: np.ndarray
    coil_translation_spread_mm: dict[str, Any]
    coil_rotation_spread_deg: dict[str, Any]
    quaternion_average_details: dict[str, Any]
    validation_metrics: dict[str, Any]


def load_runtime_tip_truth_geometry(points_file: Path, transform_file: Path) -> RuntimeTipTruthGeometry:
    """Load the protected hat-point truth set and map it into the Tip frame."""
    truth_in_sw = _load_points_3xN(points_file)
    T_sw_tip = _load_transform_4x4(transform_file)
    truth_in_tip = _apply_transform_to_points(T_sw_tip, truth_in_sw)
    labels = [f"T{index + 1:02d}" for index in range(truth_in_tip.shape[1])]
    return RuntimeTipTruthGeometry(
        points_file=points_file,
        transform_file=transform_file,
        labels=labels,
        truth_points_in_sw_by_label={
            label: truth_in_sw[:, index].astype(float).tolist()
            for index, label in enumerate(labels)
        },
        truth_points_in_tip_by_label={
            label: truth_in_tip[:, index].astype(float).tolist()
            for index, label in enumerate(labels)
        },
        T_sw_tip=T_sw_tip,
    )


def solve_runtime_tip_calibration(
    *,
    solver: RigidRegistrationSolver,
    labels: Sequence[str],
    raw_points_by_label: Mapping[str, Sequence[Sequence[float]]],
    truth_points_in_tip_by_label: Mapping[str, Sequence[float]],
    coil_samples: Sequence[AuroraPoseSample],
    quaternion_average_method: str,
) -> RuntimeTipCalibrationResult:
    """Solve the thesis/reference runtime tip calibration.

    The solved chain is:
        T_robot_tip = T_robot_aurora @ T_aurora_coil @ T_coil_tip

    and this routine computes:
        T_coil_tip = inverse(T_aurora_coil_avg) @ inverse(T_tip_aurora)
    """
    ordered_labels = [str(label) for label in labels]
    if not ordered_labels:
        raise ValueError("At least one hat point label is required")
    if len(coil_samples) == 0:
        raise ValueError("At least one 0A coil sample is required")

    averaged_points_by_label: dict[str, list[float]] = {}
    point_spread_mm_by_label: dict[str, float] = {}
    measured_points: list[list[float]] = []
    truth_points: list[list[float]] = []
    for label in ordered_labels:
        raw_points = np.asarray(raw_points_by_label.get(label, []), dtype=float)
        if raw_points.ndim != 2 or raw_points.shape[1] != 3 or raw_points.shape[0] == 0:
            raise ValueError(f"Label {label} is missing captured Aurora-space hat points")
        centroid = raw_points.mean(axis=0)
        averaged_points_by_label[label] = centroid.astype(float).tolist()
        point_spread_mm_by_label[label] = float(
            np.sqrt(np.mean(np.sum(np.square(raw_points - centroid), axis=1)))
        )
        measured_points.append(centroid.astype(float).tolist())

        truth_point = np.asarray(truth_points_in_tip_by_label.get(label), dtype=float)
        if truth_point.shape != (3,):
            raise ValueError(f"Label {label} is missing tip-frame truth geometry")
        truth_points.append(truth_point.astype(float).tolist())

    measured = np.asarray(measured_points, dtype=float)
    truth = np.asarray(truth_points, dtype=float)
    T_tip_aurora = solver.solve_T_robot_aurora(measured, truth)
    transformed = solver.apply_transform(T_tip_aurora, measured)
    residuals = truth - transformed
    residuals_tip_xyz_mm_by_label = {
        label: residuals[index, :].astype(float).tolist()
        for index, label in enumerate(ordered_labels)
    }
    residual_norms_mm_by_label = compute_residual_norms_mm(residuals_tip_xyz_mm_by_label)
    residual_summary = summarize_residual_norms_mm(residual_norms_mm_by_label)
    fit_rmse_mm = float(compute_fre_mm(list(residuals_tip_xyz_mm_by_label.values())))
    max_residual_mm = float(residual_summary["max_residual_mm"] or 0.0)

    T_aurora_coil_avg, averaged_quaternion, averaged_translation, averaging_details = average_tool_samples_as_transform(
        list(coil_samples),
        method=quaternion_average_method,
    )
    T_aurora_tip = np.linalg.inv(T_tip_aurora)
    T_coil_tip = np.linalg.inv(T_aurora_coil_avg) @ T_aurora_tip
    assert_rigid_transform_matrix(T_coil_tip, "T_coil_tip")

    translation_spread = _summarize_translation_spread(coil_samples, averaged_translation)
    rotation_spread = _summarize_rotation_spread(coil_samples, T_aurora_coil_avg)
    validation_metrics = {
        "calibration_kind": "runtime_tip_calibration_hat",
        "measurement_tool_id": "0B",
        "coil_tool_id": "0A",
        "landmark_count": len(ordered_labels),
        "hat_fit_rmse_mm": fit_rmse_mm,
        "max_hat_residual_mm": max_residual_mm,
        "residual_norms_mm_by_label": dict(residual_norms_mm_by_label),
        "residual_summary": residual_summary,
        "point_spread_mm_by_label": dict(point_spread_mm_by_label),
        "point_spread_summary_mm": _summarize_scalar_values(point_spread_mm_by_label.values()),
        "coil_sample_count": len(coil_samples),
        "coil_translation_spread_mm": translation_spread,
        "coil_rotation_spread_deg": rotation_spread,
        "quaternion_average_method": quaternion_average_method,
        "quaternion_average_details": dict(averaging_details),
        "frame_metadata": {
            "T_tip_aurora": {
                "target_frame": "tip",
                "source_frame": "aurora",
                "description": "Hat-point registration solve from Aurora-space 0B tip points into the known Tip frame.",
            },
            "T_aurora_coil_avg": {
                "target_frame": "aurora",
                "source_frame": "coil",
                "description": "Stationary averaged 0A coil pose collected while the hat defines a fixed tip frame.",
            },
            "T_coil_tip": {
                "target_frame": "coil",
                "source_frame": "tip",
                "description": (
                    "Computed as inverse(T_aurora_coil_avg) @ inverse(T_tip_aurora), matching the thesis/runtime chain."
                ),
            },
        },
    }

    return RuntimeTipCalibrationResult(
        labels=ordered_labels,
        averaged_points_by_label=averaged_points_by_label,
        residuals_tip_xyz_mm_by_label=residuals_tip_xyz_mm_by_label,
        residual_norms_mm_by_label=residual_norms_mm_by_label,
        point_spread_mm_by_label=point_spread_mm_by_label,
        fit_rmse_mm=fit_rmse_mm,
        max_residual_mm=max_residual_mm,
        T_tip_aurora=T_tip_aurora,
        T_aurora_tip=T_aurora_tip,
        T_aurora_coil_avg=T_aurora_coil_avg,
        T_coil_tip=T_coil_tip,
        averaged_coil_quaternion_wxyz=averaged_quaternion,
        averaged_coil_translation_mm=averaged_translation,
        coil_translation_spread_mm=translation_spread,
        coil_rotation_spread_deg=rotation_spread,
        quaternion_average_details=dict(averaging_details),
        validation_metrics=validation_metrics,
    )


def _load_points_3xN(path: Path) -> np.ndarray:
    rows = []
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        stripped = raw_line.strip()
        if not stripped:
            continue
        rows.append([float(value) for value in stripped.split(",") if value.strip() != ""])
    points = np.asarray(rows, dtype=float)
    if points.ndim != 2 or points.shape[0] != 3:
        raise ValueError(f"{path} must contain 3 CSV rows of point coordinates")
    return points


def _load_transform_4x4(path: Path) -> np.ndarray:
    rows = []
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        stripped = raw_line.strip()
        if not stripped:
            continue
        rows.append([float(value) for value in stripped.split(",") if value.strip() != ""])
    matrix = np.asarray(rows, dtype=float)
    assert_rigid_transform_matrix(matrix, str(path))
    return matrix


def _apply_transform_to_points(T_target_source: np.ndarray, points_source_3xN: np.ndarray) -> np.ndarray:
    points = np.asarray(points_source_3xN, dtype=float)
    if points.ndim != 2 or points.shape[0] != 3:
        raise ValueError("points_source_3xN must have shape (3, N)")
    homogeneous = np.vstack([points, np.ones((1, points.shape[1]), dtype=float)])
    transformed = T_target_source @ homogeneous
    return transformed[0:3, :]


def _summarize_translation_spread(
    coil_samples: Sequence[AuroraPoseSample],
    averaged_translation_mm: np.ndarray,
) -> dict[str, Any]:
    deltas = [
        float(np.linalg.norm(np.asarray(sample.translation_mm, dtype=float) - averaged_translation_mm))
        for sample in coil_samples
    ]
    return _summarize_scalar_values(deltas)


def _summarize_rotation_spread(
    coil_samples: Sequence[AuroraPoseSample],
    T_aurora_coil_avg: np.ndarray,
) -> dict[str, Any]:
    R_avg = np.asarray(T_aurora_coil_avg[0:3, 0:3], dtype=float)
    deltas_deg: list[float] = []
    for sample in coil_samples:
        T_aurora_coil = make_transform_A_B(sample.quaternion_wxyz, sample.translation_mm)
        delta = np.linalg.inv(R_avg) @ T_aurora_coil[0:3, 0:3]
        value = float((np.trace(delta) - 1.0) / 2.0)
        value = min(1.0, max(-1.0, value))
        deltas_deg.append(float(np.degrees(np.arccos(value))))
    return _summarize_scalar_values(deltas_deg)


def _summarize_scalar_values(values: Sequence[float]) -> dict[str, Any]:
    numeric = [float(value) for value in values]
    if not numeric:
        return {
            "count": 0,
            "mean": None,
            "std": None,
            "min": None,
            "max": None,
        }
    array = np.asarray(numeric, dtype=float)
    return {
        "count": int(array.size),
        "mean": float(array.mean()),
        "std": float(array.std(ddof=0)),
        "min": float(array.min()),
        "max": float(array.max()),
    }

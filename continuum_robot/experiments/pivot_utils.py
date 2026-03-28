"""Least-squares pivot calibration helpers."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from continuum_robot.experiments.dataset_tools import extract_tool_transforms_from_dataset
from continuum_robot.registration.legacy_compat import parse_aurora_csv
from continuum_robot.tracking.transforms import quat_wxyz_to_rotmat


@dataclass
class PivotCalibrationResult:
    """Output of a pivot calibration solve."""

    tip_vector_local_mm: list[float]
    pivot_point_tracker_mm: list[float]
    rmse_mm: float
    sample_count_total: int
    sample_count_used: int
    sample_count_rejected: int
    residuals_mm: list[list[float]]
    inlier_mask: list[bool]
    rejected_indices: list[int]


def solve_pivot_calibration(
    rotations: list[np.ndarray],
    translations_mm: list[np.ndarray],
    *,
    std_dev_threshold: float = 3.0,
    min_samples: int = 8,
) -> PivotCalibrationResult:
    """Solve the standard least-squares pivot calibration problem."""
    if len(rotations) != len(translations_mm):
        raise ValueError("rotations and translations_mm length mismatch")
    if len(rotations) < int(min_samples):
        raise ValueError("Insufficient samples for pivot calibration")
    A, b = _assemble_system(rotations, translations_mm)
    x_initial, *_ = np.linalg.lstsq(A, b, rcond=None)
    residuals_initial = (A @ x_initial - b).reshape(-1, 3)
    keep_mask = _inlier_mask_from_residuals(residuals_initial, std_dev_threshold=std_dev_threshold)
    if int(np.count_nonzero(keep_mask)) < int(min_samples):
        raise ValueError("Pivot calibration has insufficient inlier samples after outlier rejection")
    inlier_rows = np.repeat(keep_mask, 3)
    A_inlier = A[inlier_rows, :]
    b_inlier = b[inlier_rows]
    x_final, *_ = np.linalg.lstsq(A_inlier, b_inlier, rcond=None)
    residuals_final = (A_inlier @ x_final - b_inlier).reshape(-1, 3)
    rmse_mm = float(np.sqrt(np.mean(np.sum(residuals_final**2, axis=1))))
    return PivotCalibrationResult(
        tip_vector_local_mm=[float(value) for value in x_final[0:3]],
        pivot_point_tracker_mm=[float(value) for value in x_final[3:6]],
        rmse_mm=rmse_mm,
        sample_count_total=len(rotations),
        sample_count_used=int(np.count_nonzero(keep_mask)),
        sample_count_rejected=int(len(rotations) - int(np.count_nonzero(keep_mask))),
        residuals_mm=[[float(value) for value in row] for row in residuals_final],
        inlier_mask=[bool(value) for value in keep_mask.tolist()],
        rejected_indices=[int(index) for index, keep in enumerate(keep_mask.tolist()) if not keep],
    )


def load_pivot_transforms(
    path: Path,
    *,
    tool_id: str = "0B",
) -> list[np.ndarray]:
    """Load pivot pose transforms from a canonical dataset or legacy Aurora CSV."""
    source = Path(path)
    if source.suffix.lower() == ".csv":
        samples = parse_aurora_csv(source).get(tool_id, [])
        transforms: list[np.ndarray] = []
        for sample in samples:
            T = np.eye(4, dtype=float)
            T[0:3, 0:3] = quat_wxyz_to_rotmat(sample.quaternion_wxyz)
            T[0:3, 3] = np.asarray(sample.translation_mm, dtype=float)
            transforms.append(T)
        return transforms
    return extract_tool_transforms_from_dataset(source, tool_id=tool_id)


def write_tip_vector_file(path: Path, tip_vector_local_mm: list[float]) -> Path:
    """Write the pen-probe tip vector in the repo's expected 3-value format."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = ",".join(f"{float(value):+0.4f}" for value in tip_vector_local_mm)
    target.write_text(payload, encoding="utf-8")
    return target


def _assemble_system(rotations: list[np.ndarray], translations_mm: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    rows: list[np.ndarray] = []
    rhs: list[np.ndarray] = []
    for rotation, translation in zip(rotations, translations_mm):
        R = np.asarray(rotation, dtype=float)
        t = np.asarray(translation, dtype=float).reshape(3)
        rows.append(np.hstack([R, -1.0 * np.eye(3)]))
        rhs.append(-t.reshape(3, 1))
    A = np.vstack(rows)
    b = np.vstack(rhs).reshape(-1)
    return A, b


def _inlier_mask_from_residuals(residuals_mm: np.ndarray, *, std_dev_threshold: float) -> np.ndarray:
    residuals = np.asarray(residuals_mm, dtype=float)
    means = residuals.mean(axis=0)
    stdevs = residuals.std(axis=0, ddof=0)
    deviations = np.zeros_like(residuals)
    for axis in range(3):
        if stdevs[axis] <= 1e-12:
            deviations[:, axis] = 0.0
        else:
            deviations[:, axis] = np.abs((residuals[:, axis] - means[axis]) / stdevs[axis])
    outlier_mask = np.any(deviations > float(std_dev_threshold), axis=1)
    return ~outlier_mask

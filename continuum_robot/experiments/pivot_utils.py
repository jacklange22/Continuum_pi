"""Least-squares pivot calibration helpers."""

from __future__ import annotations

import csv
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from continuum_robot.experiments.dataset_tools import extract_tool_transforms_from_dataset
from continuum_robot.registration.legacy_compat import AuroraPoseSample, parse_aurora_csv
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


@dataclass
class RansacPivotCalibrationResult:
    """Output of a RANSAC-augmented pivot calibration solve.

    Mirrors :class:`PivotCalibrationResult` so downstream code can treat the
    two interchangeably, then adds RANSAC-specific diagnostics that make the
    run auditable (best-consensus growth, iteration budget, seed used).
    """

    tip_vector_local_mm: list[float]
    pivot_point_tracker_mm: list[float]
    rmse_mm: float
    sample_count_total: int
    sample_count_used: int
    sample_count_rejected: int
    residuals_mm: list[list[float]]
    inlier_mask: list[bool]
    rejected_indices: list[int]
    iterations_run: int
    best_consensus_size: int
    iteration_history: list[dict[str, Any]]
    converged: bool
    inlier_threshold_mm: float
    minimum_sample_size: int
    min_consensus_size: int
    seed: int | None

    def to_pivot_calibration_result(self) -> PivotCalibrationResult:
        """Project to the classical :class:`PivotCalibrationResult` shape.

        Useful when wiring RANSAC into an experiment path that already
        consumes a :class:`PivotCalibrationResult` and shouldn't have to
        learn the richer schema.
        """
        return PivotCalibrationResult(
            tip_vector_local_mm=list(self.tip_vector_local_mm),
            pivot_point_tracker_mm=list(self.pivot_point_tracker_mm),
            rmse_mm=float(self.rmse_mm),
            sample_count_total=int(self.sample_count_total),
            sample_count_used=int(self.sample_count_used),
            sample_count_rejected=int(self.sample_count_rejected),
            residuals_mm=[list(row) for row in self.residuals_mm],
            inlier_mask=list(self.inlier_mask),
            rejected_indices=list(self.rejected_indices),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "tip_vector_local_mm": list(self.tip_vector_local_mm),
            "pivot_point_tracker_mm": list(self.pivot_point_tracker_mm),
            "rmse_mm": float(self.rmse_mm),
            "sample_count_total": int(self.sample_count_total),
            "sample_count_used": int(self.sample_count_used),
            "sample_count_rejected": int(self.sample_count_rejected),
            "residuals_mm": [list(row) for row in self.residuals_mm],
            "inlier_mask": list(self.inlier_mask),
            "rejected_indices": list(self.rejected_indices),
            "iterations_run": int(self.iterations_run),
            "best_consensus_size": int(self.best_consensus_size),
            "iteration_history": [dict(row) for row in self.iteration_history],
            "converged": bool(self.converged),
            "inlier_threshold_mm": float(self.inlier_threshold_mm),
            "minimum_sample_size": int(self.minimum_sample_size),
            "min_consensus_size": int(self.min_consensus_size),
            "seed": int(self.seed) if self.seed is not None else None,
        }


class PivotRansacFailure(ValueError):
    """Raised when RANSAC cannot produce a consensus that meets the floor.

    Carries the partial iteration record so the operator can see how the
    run progressed (best consensus size achieved, iterations attempted).
    """

    def __init__(self, message: str, *, partial: dict[str, Any] | None = None) -> None:
        super().__init__(message)
        self.partial = dict(partial or {})


@dataclass
class PivotCsvLoadReport:
    """Operator-facing parse report for pivot input files."""

    source_path: str
    detected_format: str
    tool_id: str
    header_columns: list[str] = field(default_factory=list)
    total_data_rows: int = 0
    usable_rows: int = 0
    filtered_other_tool_rows: int = 0
    rejected_rows: list[dict[str, object]] = field(default_factory=list)

    def to_metrics(self) -> dict[str, object]:
        """Convert the parse report into experiment-summary metrics."""
        return {
            "pivot_input_path": self.source_path,
            "pivot_input_format": self.detected_format,
            "pivot_input_tool_id": self.tool_id,
            "pivot_input_header_columns": list(self.header_columns),
            "pivot_input_total_rows": int(self.total_data_rows),
            "pivot_input_usable_rows": int(self.usable_rows),
            "pivot_input_filtered_other_tool_rows": int(self.filtered_other_tool_rows),
            "pivot_input_rejected_row_count": len(self.rejected_rows),
            "pivot_input_rejected_rows": [dict(row) for row in self.rejected_rows],
        }


@dataclass
class PivotTransformLoadResult:
    """Transforms plus a parse/load report."""

    transforms: list[np.ndarray]
    report: PivotCsvLoadReport


class PivotInputParseError(ValueError):
    """Raised when a pivot input file cannot be parsed into usable tool samples."""

    def __init__(self, message: str, report: PivotCsvLoadReport) -> None:
        super().__init__(message)
        self.report = report


_CANONICAL_PIVOT_COLUMNS = ("tool_id", "qw", "qx", "qy", "qz", "x", "y", "z")


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


def solve_pivot_calibration_ransac(
    rotations: list[np.ndarray],
    translations_mm: list[np.ndarray],
    *,
    inlier_threshold_mm: float = 1.0,
    minimum_sample_size: int = 3,
    min_consensus_size: int | None = None,
    max_iterations: int = 1000,
    confidence: float = 0.99,
    seed: int | None = None,
) -> RansacPivotCalibrationResult:
    """Solve pivot calibration with RANSAC outlier rejection.

    The classical pivot system ``A x = b`` (with ``A_i = [R_i | -I]`` and
    ``b_i = -t_i``) yields ``x = [tip_local; pivot_tracker]``. Each pose i
    contributes a 3D residual ``R_i @ tip + t_i - pivot``; an inlier is a
    pose whose residual norm is at most ``inlier_threshold_mm``.

    The algorithm samples ``minimum_sample_size`` poses uniformly at random,
    fits ``x`` via least squares on the sample, scores inliers on every
    pose, and keeps the largest-consensus result. The iteration budget
    shrinks adaptively once a good consensus appears, following
    ``k = log(1 - p) / log(1 - w^n)``. The final tip/pivot are refit on
    the best inlier set via the same least-squares solver used by
    :func:`solve_pivot_calibration`, so the recovered math is identical to
    the standard solve -- just on a clean subset of poses.

    Parameters
    ----------
    rotations : Per-pose 3x3 rotation matrices.
    translations_mm : Per-pose 3-vector translations in millimeters.
    inlier_threshold_mm : Residual-norm cutoff in mm for a pose to count
        as an inlier.
    minimum_sample_size : Number of poses in a random sample. The unknown
        vector has 6 components, so technically 2 poses are sufficient
        (6 equations); 3 is a more conservative default that conditions
        the per-sample fit better when rotations are similar.
    min_consensus_size : Minimum inlier count for the run to be considered
        converged. Defaults to ``max(6, ceil(0.5 N))`` so a half-corrupted
        dataset of at least 12 poses can still pass.
    max_iterations : Hard cap on iterations; adaptive shrinking will usually
        stop sooner once a strong consensus is found.
    confidence : Desired probability of sampling an all-inlier minimum set
        at least once, used to compute the adaptive iteration budget.
    seed : Optional integer seed for reproducibility.

    Returns
    -------
    :class:`RansacPivotCalibrationResult` containing the recovered tip and
    pivot, per-pose residuals on every input (not just inliers), the
    inlier/reject masks, and a per-iteration history.

    Raises
    ------
    ValueError
        If inputs are inconsistent or there are not enough poses to draw a
        single minimum sample.
    PivotRansacFailure
        If no sample produced a consensus that met ``min_consensus_size``.
    """
    if len(rotations) != len(translations_mm):
        raise ValueError("rotations and translations_mm length mismatch")
    n_total = len(rotations)
    sample_size = int(minimum_sample_size)
    if sample_size < 2:
        raise ValueError(
            "minimum_sample_size must be at least 2 (6 equations are required to determine "
            "the 6-vector [tip; pivot])"
        )
    if n_total < sample_size:
        raise ValueError(
            f"At least {sample_size} poses are required for RANSAC pivot calibration; received {n_total}"
        )
    if inlier_threshold_mm <= 0:
        raise ValueError("inlier_threshold_mm must be positive")
    if not (0.0 < confidence < 1.0):
        raise ValueError("confidence must lie in the open interval (0, 1)")
    if max_iterations < 1:
        raise ValueError("max_iterations must be at least 1")
    consensus_floor = (
        int(min_consensus_size)
        if min_consensus_size is not None
        else max(6, math.ceil(0.5 * n_total))
    )
    consensus_floor = max(consensus_floor, sample_size)

    A_full, b_full = _assemble_system(rotations, translations_mm)
    rng = np.random.default_rng(seed)
    best_mask: np.ndarray | None = None
    best_consensus = 0
    best_inlier_rmse = math.inf
    iteration_history: list[dict[str, Any]] = []
    iteration_budget = int(max_iterations)
    iteration = 0

    while iteration < iteration_budget:
        sample_indices = rng.choice(n_total, size=sample_size, replace=False)
        sample_rows = np.repeat(_indices_to_mask(sample_indices, n_total), 3)
        A_sample = A_full[sample_rows, :]
        b_sample = b_full[sample_rows]
        try:
            x_candidate, *_ = np.linalg.lstsq(A_sample, b_sample, rcond=None)
        except np.linalg.LinAlgError:
            iteration += 1
            continue
        residuals_full = (A_full @ x_candidate - b_full).reshape(-1, 3)
        residual_norms = np.linalg.norm(residuals_full, axis=1)
        inlier_mask = residual_norms <= inlier_threshold_mm
        consensus_size = int(inlier_mask.sum())
        inlier_rmse_mm = (
            float(np.sqrt(np.mean(residual_norms[inlier_mask] ** 2)))
            if consensus_size > 0
            else float("inf")
        )

        improved = consensus_size > best_consensus or (
            consensus_size == best_consensus and inlier_rmse_mm < best_inlier_rmse
        )
        if improved:
            best_consensus = consensus_size
            best_mask = inlier_mask
            best_inlier_rmse = inlier_rmse_mm
            inlier_ratio = consensus_size / n_total
            if 0.0 < inlier_ratio < 1.0:
                denom = math.log(1.0 - inlier_ratio ** sample_size)
                if denom < 0.0:
                    required = math.ceil(math.log(1.0 - confidence) / denom)
                    iteration_budget = min(int(max_iterations), max(iteration + 1, required))
            elif inlier_ratio >= 1.0:
                iteration_budget = iteration + 1

        iteration_history.append(
            {
                "iteration": int(iteration),
                "sample_indices": [int(value) for value in sample_indices.tolist()],
                "consensus_size": int(consensus_size),
                "inlier_rmse_mm": inlier_rmse_mm,
                "best_consensus_size_so_far": int(best_consensus),
                "iteration_budget": int(iteration_budget),
            }
        )
        iteration += 1

    iterations_run = iteration
    if best_mask is None or best_consensus < consensus_floor:
        partial = {
            "best_consensus_size": int(best_consensus),
            "min_consensus_size": consensus_floor,
            "iterations_run": int(iterations_run),
            "iteration_history": [dict(row) for row in iteration_history],
            "inlier_threshold_mm": float(inlier_threshold_mm),
            "sample_count_total": int(n_total),
        }
        raise PivotRansacFailure(
            f"RANSAC pivot calibration failed: best consensus was {best_consensus} of "
            f"{n_total} poses (need at least {consensus_floor}) within "
            f"{inlier_threshold_mm:.3f} mm after {iterations_run} iterations.",
            partial=partial,
        )

    inlier_indices = np.flatnonzero(best_mask)
    inlier_rows = np.repeat(best_mask, 3)
    A_inlier = A_full[inlier_rows, :]
    b_inlier = b_full[inlier_rows]
    x_final, *_ = np.linalg.lstsq(A_inlier, b_inlier, rcond=None)
    residuals_final_all = (A_full @ x_final - b_full).reshape(-1, 3)
    residual_norms_final = np.linalg.norm(residuals_final_all, axis=1)
    final_mask = residual_norms_final <= inlier_threshold_mm
    # Honor the consensus indices even if the refit nudges a borderline
    # residual past the threshold by epsilon -- stable inlier accounting.
    final_mask[inlier_indices] = True
    final_inlier_indices = np.flatnonzero(final_mask)
    if final_inlier_indices.size == 0:
        raise PivotRansacFailure(
            "RANSAC pivot calibration final refit produced no inliers; this should not happen "
            "and likely indicates degenerate input geometry.",
            partial={
                "best_consensus_size": int(best_consensus),
                "iterations_run": int(iterations_run),
            },
        )
    inlier_residual_norms = residual_norms_final[final_inlier_indices]
    inlier_rmse = float(np.sqrt(np.mean(inlier_residual_norms ** 2)))
    rejected = np.flatnonzero(~final_mask)
    return RansacPivotCalibrationResult(
        tip_vector_local_mm=[float(value) for value in x_final[0:3]],
        pivot_point_tracker_mm=[float(value) for value in x_final[3:6]],
        rmse_mm=inlier_rmse,
        sample_count_total=int(n_total),
        sample_count_used=int(final_inlier_indices.size),
        sample_count_rejected=int(rejected.size),
        residuals_mm=[[float(value) for value in row] for row in residuals_final_all],
        inlier_mask=[bool(value) for value in final_mask.tolist()],
        rejected_indices=[int(value) for value in rejected.tolist()],
        iterations_run=int(iterations_run),
        best_consensus_size=int(best_consensus),
        iteration_history=iteration_history,
        converged=True,
        inlier_threshold_mm=float(inlier_threshold_mm),
        minimum_sample_size=sample_size,
        min_consensus_size=int(consensus_floor),
        seed=seed,
    )


def _indices_to_mask(indices: np.ndarray, total: int) -> np.ndarray:
    mask = np.zeros(int(total), dtype=bool)
    mask[indices] = True
    return mask


def load_pivot_transforms(
    path: Path,
    *,
    tool_id: str = "0B",
) -> list[np.ndarray]:
    """Load pivot pose transforms from a canonical dataset or CSV."""
    return load_pivot_transforms_with_report(path, tool_id=tool_id).transforms


def load_pivot_transforms_with_report(
    path: Path,
    *,
    tool_id: str = "0B",
) -> PivotTransformLoadResult:
    """Load pivot transforms and return a parse report suitable for operator review."""
    source = Path(path)
    if source.suffix.lower() != ".csv":
        transforms = extract_tool_transforms_from_dataset(source, tool_id=tool_id)
        report = PivotCsvLoadReport(
            source_path=str(source),
            detected_format="canonical_dataset_bundle",
            tool_id=tool_id,
            total_data_rows=len(transforms),
            usable_rows=len(transforms),
        )
        return PivotTransformLoadResult(transforms=transforms, report=report)

    format_hint = _detect_csv_format(source)
    if format_hint == "canonical_headered_csv":
        return _load_canonical_headered_pivot_csv(source, tool_id=tool_id)
    if format_hint == "legacy_aurora_csv":
        return _load_legacy_pivot_csv(source, tool_id=tool_id)
    report = PivotCsvLoadReport(source_path=str(source), detected_format="unknown_csv", tool_id=tool_id)
    raise PivotInputParseError(
        f"Pivot input CSV format is ambiguous: {source}. Expected either the canonical header "
        f"{','.join(_CANONICAL_PIVOT_COLUMNS)} or a supported legacy Aurora CSV layout.",
        report,
    )


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


def _detect_csv_format(path: Path) -> str:
    with Path(path).open("r", encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle)
        for row in reader:
            cells = [str(item).strip() for item in row if str(item).strip() != ""]
            if not cells:
                continue
            lowered = [cell.lower() for cell in cells]
            if "tool_id" in lowered or any(cell in {"qw", "qx", "qy", "qz", "x", "y", "z"} for cell in lowered):
                missing = [column for column in _CANONICAL_PIVOT_COLUMNS if column not in lowered]
                if missing:
                    report = PivotCsvLoadReport(
                        source_path=str(path),
                        detected_format="canonical_headered_csv",
                        tool_id="",
                        header_columns=lowered,
                    )
                    raise PivotInputParseError(
                        "CSV header detected but required columns missing: "
                        + ", ".join(missing)
                        + f". Expected columns: {','.join(_CANONICAL_PIVOT_COLUMNS)}",
                        report,
                    )
                return "canonical_headered_csv"
            return "legacy_aurora_csv"
    raise PivotInputParseError(
        f"Pivot input CSV contains no rows: {path}",
        PivotCsvLoadReport(source_path=str(path), detected_format="empty_csv", tool_id=""),
    )


def _load_canonical_headered_pivot_csv(path: Path, *, tool_id: str) -> PivotTransformLoadResult:
    with Path(path).open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            report = PivotCsvLoadReport(source_path=str(path), detected_format="canonical_headered_csv", tool_id=tool_id)
            raise PivotInputParseError(f"Pivot input CSV has no header row: {path}", report)
        header_lookup = {str(field).strip().lower(): str(field) for field in reader.fieldnames if field is not None}
        missing = [column for column in _CANONICAL_PIVOT_COLUMNS if column not in header_lookup]
        report = PivotCsvLoadReport(
            source_path=str(path),
            detected_format="canonical_headered_csv",
            tool_id=tool_id,
            header_columns=list(header_lookup.keys()),
        )
        if missing:
            raise PivotInputParseError(
                "CSV header detected but required columns missing: "
                + ", ".join(missing)
                + f". Expected columns: {','.join(_CANONICAL_PIVOT_COLUMNS)}",
                report,
            )
        samples: list[AuroraPoseSample] = []
        for row_index, raw in enumerate(reader, start=2):
            if raw is None:
                continue
            if all((value is None or str(value).strip() == "") for value in raw.values()):
                continue
            report.total_data_rows += 1
            raw_tool_value = raw.get(header_lookup["tool_id"], "")
            raw_tool_id = "" if raw_tool_value is None else str(raw_tool_value).strip()
            if not raw_tool_id:
                report.rejected_rows.append({"row": row_index, "reason": "missing tool_id"})
                continue
            if raw_tool_id != tool_id:
                report.filtered_other_tool_rows += 1
                continue
            missing_values = [
                column
                for column in _CANONICAL_PIVOT_COLUMNS
                if _is_blank_csv_value(raw.get(header_lookup[column], ""))
            ]
            if missing_values:
                report.rejected_rows.append(
                    {"row": row_index, "reason": "missing values for " + ", ".join(missing_values)}
                )
                continue
            try:
                quat = tuple(float(raw[header_lookup[column]]) for column in ("qw", "qx", "qy", "qz"))
                trans = tuple(float(raw[header_lookup[column]]) for column in ("x", "y", "z"))
            except (TypeError, ValueError) as exc:
                report.rejected_rows.append({"row": row_index, "reason": f"non-float pose value: {exc}"})
                continue
            samples.append(
                AuroraPoseSample(
                    tool_id=raw_tool_id,
                    quaternion_wxyz=quat,  # type: ignore[arg-type]
                    translation_mm=trans,  # type: ignore[arg-type]
                    source_row=row_index,
                    source_token="canonical_headered_csv",
                )
            )
        report.usable_rows = len(samples)
    if not samples:
        rejected_summary = _render_rejected_rows(report.rejected_rows)
        message = (
            f"No usable {tool_id} samples found in canonical headered CSV {path}. "
            f"Filtered other-tool rows={report.filtered_other_tool_rows}, rejected rows={len(report.rejected_rows)}."
        )
        if rejected_summary:
            message += f" Rejected rows: {rejected_summary}"
        raise PivotInputParseError(message, report)
    return PivotTransformLoadResult(transforms=_samples_to_transforms(samples), report=report)


def _load_legacy_pivot_csv(path: Path, *, tool_id: str) -> PivotTransformLoadResult:
    report = PivotCsvLoadReport(
        source_path=str(path),
        detected_format="legacy_aurora_csv",
        tool_id=tool_id,
    )
    try:
        parsed = parse_aurora_csv(path)
    except Exception as exc:
        raise PivotInputParseError(f"Legacy Aurora CSV parse failed for {path}: {exc}", report) from exc
    all_samples = [sample for samples in parsed.values() for sample in samples]
    report.total_data_rows = len(all_samples)
    target_samples = list(parsed.get(tool_id, []))
    report.usable_rows = len(target_samples)
    report.filtered_other_tool_rows = max(0, len(all_samples) - len(target_samples))
    if not target_samples:
        raise PivotInputParseError(
            f"No usable {tool_id} samples found in legacy Aurora CSV {path}. "
            f"Rows parsed={report.total_data_rows}, filtered other-tool rows={report.filtered_other_tool_rows}.",
            report,
        )
    return PivotTransformLoadResult(transforms=_samples_to_transforms(target_samples), report=report)


def _samples_to_transforms(samples: list[AuroraPoseSample]) -> list[np.ndarray]:
    transforms: list[np.ndarray] = []
    for sample in samples:
        T = np.eye(4, dtype=float)
        T[0:3, 0:3] = quat_wxyz_to_rotmat(sample.quaternion_wxyz)
        T[0:3, 3] = np.asarray(sample.translation_mm, dtype=float)
        transforms.append(T)
    return transforms


def _render_rejected_rows(rejected_rows: list[dict[str, object]], *, limit: int = 4) -> str:
    preview = rejected_rows[:limit]
    rendered = ", ".join(
        f"row {row.get('row')}: {row.get('reason')}"
        for row in preview
    )
    if len(rejected_rows) > limit:
        rendered += f", and {len(rejected_rows) - limit} more"
    return rendered


def _is_blank_csv_value(value: object) -> bool:
    return value is None or str(value).strip() == ""

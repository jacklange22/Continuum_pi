"""Rigid registration solver."""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class RansacRegistrationResult:
    """Output of RANSAC-augmented rigid registration.

    Carries the recovered transform plus enough diagnostics that an operator
    can audit the run after the fact (which points were rejected, how many
    iterations ran, how the best-consensus size grew).
    """

    T_target_source: np.ndarray
    transformed_points: np.ndarray
    residuals_xyz_mm: np.ndarray
    residual_norms_mm: np.ndarray
    inlier_mask: list[bool]
    inlier_indices: list[int]
    rejected_indices: list[int]
    inlier_rmse_mm: float
    all_point_rmse_mm: float
    sample_count_total: int
    sample_count_used: int
    iterations_run: int
    best_consensus_size: int
    iteration_history: list[dict[str, Any]]
    converged: bool
    inlier_threshold_mm: float
    minimum_sample_size: int
    min_consensus_size: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "T_target_source": self.T_target_source.tolist(),
            "residuals_xyz_mm": self.residuals_xyz_mm.tolist(),
            "residual_norms_mm": self.residual_norms_mm.tolist(),
            "inlier_mask": list(self.inlier_mask),
            "inlier_indices": list(self.inlier_indices),
            "rejected_indices": list(self.rejected_indices),
            "inlier_rmse_mm": float(self.inlier_rmse_mm),
            "all_point_rmse_mm": float(self.all_point_rmse_mm),
            "sample_count_total": int(self.sample_count_total),
            "sample_count_used": int(self.sample_count_used),
            "iterations_run": int(self.iterations_run),
            "best_consensus_size": int(self.best_consensus_size),
            "iteration_history": [dict(row) for row in self.iteration_history],
            "converged": bool(self.converged),
            "inlier_threshold_mm": float(self.inlier_threshold_mm),
            "minimum_sample_size": int(self.minimum_sample_size),
            "min_consensus_size": int(self.min_consensus_size),
        }


class RigidRegistrationFailure(ValueError):
    """Raised when no RANSAC consensus meets the acceptance criteria.

    Carries the partial iteration history so the operator can see how close
    the run got (best consensus size, threshold used, iterations attempted).
    """

    def __init__(self, message: str, *, partial: dict[str, Any] | None = None) -> None:
        super().__init__(message)
        self.partial = dict(partial or {})


class RigidRegistrationSolver:
    """Solve best-fit rigid transform with SVD and reflection correction."""

    def solve_T_robot_aurora(self, measured_in_aurora: np.ndarray, truth_in_robot: np.ndarray) -> np.ndarray:
        """Return best-fit ``T_robot_aurora``.

        Supports points arranged as ``(3, N)`` or ``(N, 3)``.
        """
        src = self._as_n_by_3(measured_in_aurora, "measured_in_aurora")
        dst = self._as_n_by_3(truth_in_robot, "truth_in_robot")
        if src.shape != dst.shape:
            raise ValueError("measured_in_aurora and truth_in_robot must have identical point shape")
        if src.shape[0] < 3:
            raise ValueError("At least 3 point correspondences are required")

        src_centroid = src.mean(axis=0)
        dst_centroid = dst.mean(axis=0)
        src_centered = src - src_centroid
        dst_centered = dst - dst_centroid

        H = src_centered.T @ dst_centered
        U, _S, Vt = np.linalg.svd(H)
        R = Vt.T @ U.T

        # Enforce proper rotation when SVD returns a reflection.
        if np.linalg.det(R) < 0:
            Vt[-1, :] *= -1.0
            R = Vt.T @ U.T

        t = dst_centroid - (R @ src_centroid)

        T_robot_aurora = np.eye(4, dtype=float)
        T_robot_aurora[0:3, 0:3] = R
        T_robot_aurora[0:3, 3] = t
        return T_robot_aurora

    def solve_alignment(self, measured_in_source: np.ndarray, truth_in_target: np.ndarray) -> dict[str, np.ndarray | float]:
        """Return transform, transformed points, residuals, and RMSE."""
        src = self._as_n_by_3(measured_in_source, "measured_in_source")
        dst = self._as_n_by_3(truth_in_target, "truth_in_target")
        T_target_source = self.solve_T_robot_aurora(src, dst)
        transformed = self.apply_transform(T_target_source, src)
        residuals = dst - transformed
        rmse_mm = float(np.sqrt(np.mean(np.sum(np.square(residuals), axis=1))))
        return {
            "transform": T_target_source,
            "transformed_points": transformed.T,
            "residuals": residuals.T,
            "rmse_mm": rmse_mm,
        }

    def solve_T_robot_aurora_ransac(
        self,
        measured_in_aurora: np.ndarray,
        truth_in_robot: np.ndarray,
        *,
        inlier_threshold_mm: float = 1.0,
        minimum_sample_size: int = 3,
        min_consensus_size: int | None = None,
        max_iterations: int = 1000,
        confidence: float = 0.99,
        seed: int | None = None,
    ) -> RansacRegistrationResult:
        """Solve ``T_target_source`` with RANSAC outlier rejection.

        Classical RANSAC: repeatedly pick a minimum sample, fit the closed-form
        SVD-Procrustes transform on the sample, count inliers within
        ``inlier_threshold_mm``, and keep the largest-consensus result. The
        iteration budget is reduced adaptively once a good consensus appears,
        following ``k = log(1 - p) / log(1 - w^n)`` where ``w`` is the current
        best inlier ratio and ``n`` is the minimum sample size. The final
        transform is refit on the best consensus using the same SVD solver
        used by :meth:`solve_T_robot_aurora`, so the RANSAC path produces the
        same math the rest of the codebase already trusts -- just on a clean
        subset of correspondences.

        Parameters
        ----------
        measured_in_aurora : (N, 3) or (3, N) source-frame points.
        truth_in_robot : (N, 3) or (3, N) target-frame points, same N.
        inlier_threshold_mm : Residual-norm cutoff in mm for a correspondence
            to be counted as an inlier.
        minimum_sample_size : Random-sample size per iteration. Three is the
            minimum that uniquely determines a rigid 3D transform.
        min_consensus_size : Minimum inlier count for the run to be considered
            converged. Defaults to ``max(minimum_sample_size + 1, ceil(0.5 N))``.
        max_iterations : Hard cap on iterations; the adaptive schedule will
            usually stop sooner once a strong consensus is found.
        confidence : Desired probability of sampling an all-inlier minimum
            set at least once, used to compute the adaptive iteration budget.
        seed : Optional integer seed for reproducibility.

        Returns
        -------
        :class:`RansacRegistrationResult` with the recovered transform plus
        diagnostics. The result reports ``converged=True`` when the best
        consensus size meets ``min_consensus_size``.

        Raises
        ------
        ValueError
            If input shapes are inconsistent or there are not enough
            correspondences to form even one minimum sample.
        RigidRegistrationFailure
            If no candidate sample produces a consensus large enough to
            satisfy ``min_consensus_size``.
        """
        src = self._as_n_by_3(measured_in_aurora, "measured_in_aurora")
        dst = self._as_n_by_3(truth_in_robot, "truth_in_robot")
        if src.shape != dst.shape:
            raise ValueError("measured_in_aurora and truth_in_robot must have identical point shape")
        n_total = int(src.shape[0])
        sample_size = int(minimum_sample_size)
        if sample_size < 3:
            raise ValueError("minimum_sample_size must be at least 3 for a rigid 3D transform")
        if n_total < sample_size:
            raise ValueError(
                f"At least {sample_size} correspondences are required for RANSAC; received {n_total}"
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
            else max(sample_size + 1, math.ceil(0.5 * n_total))
        )
        if consensus_floor < sample_size:
            raise ValueError("min_consensus_size cannot be smaller than minimum_sample_size")

        rng = np.random.default_rng(seed)
        best_mask: np.ndarray | None = None
        best_consensus = 0
        best_inlier_rmse = math.inf
        iteration_history: list[dict[str, Any]] = []
        iteration_budget = int(max_iterations)
        iteration = 0

        while iteration < iteration_budget:
            sample_indices = rng.choice(n_total, size=sample_size, replace=False)
            try:
                candidate_T = self.solve_T_robot_aurora(src[sample_indices], dst[sample_indices])
            except ValueError:
                iteration += 1
                continue
            transformed_all = self.apply_transform(candidate_T, src)
            residuals_all = dst - transformed_all
            residual_norms = np.linalg.norm(residuals_all, axis=1)
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
            raise RigidRegistrationFailure(
                f"RANSAC registration failed: best consensus was {best_consensus} of "
                f"{n_total} (need at least {consensus_floor}) within "
                f"{inlier_threshold_mm:.3f} mm after {iterations_run} iterations.",
                partial=partial,
            )

        # Final refit on the best inlier set with the standard SVD-Procrustes solver,
        # then recompute residuals on every input point with the refit transform.
        inlier_indices = np.flatnonzero(best_mask)
        T_final = self.solve_T_robot_aurora(src[inlier_indices], dst[inlier_indices])
        transformed_final = self.apply_transform(T_final, src)
        residuals_final = dst - transformed_final
        residual_norms_final = np.linalg.norm(residuals_final, axis=1)
        final_inlier_mask = residual_norms_final <= inlier_threshold_mm
        # Guarantee the original best-consensus indices are honored as inliers
        # even if the refit nudges a borderline point past the threshold by
        # epsilon. This keeps inlier accounting stable for the operator.
        final_inlier_mask[inlier_indices] = True
        final_inlier_indices = np.flatnonzero(final_inlier_mask)
        if final_inlier_indices.size == 0:
            raise RigidRegistrationFailure(
                "RANSAC registration final refit produced no inliers; this should not happen "
                "and likely indicates degenerate input geometry.",
                partial={
                    "best_consensus_size": int(best_consensus),
                    "iterations_run": int(iterations_run),
                },
            )
        inlier_rmse_final = float(
            np.sqrt(np.mean(residual_norms_final[final_inlier_indices] ** 2))
        )
        all_rmse_final = float(np.sqrt(np.mean(residual_norms_final ** 2)))
        rejected_indices = np.flatnonzero(~final_inlier_mask)

        return RansacRegistrationResult(
            T_target_source=T_final,
            transformed_points=transformed_final,
            residuals_xyz_mm=residuals_final,
            residual_norms_mm=residual_norms_final,
            inlier_mask=[bool(value) for value in final_inlier_mask.tolist()],
            inlier_indices=[int(value) for value in final_inlier_indices.tolist()],
            rejected_indices=[int(value) for value in rejected_indices.tolist()],
            inlier_rmse_mm=inlier_rmse_final,
            all_point_rmse_mm=all_rmse_final,
            sample_count_total=n_total,
            sample_count_used=int(final_inlier_indices.size),
            iterations_run=iterations_run,
            best_consensus_size=int(best_consensus),
            iteration_history=iteration_history,
            converged=True,
            inlier_threshold_mm=float(inlier_threshold_mm),
            minimum_sample_size=sample_size,
            min_consensus_size=consensus_floor,
        )

    @staticmethod
    def apply_transform(T_target_source: np.ndarray, points: np.ndarray) -> np.ndarray:
        """Apply a homogeneous transform to points with shape ``(N, 3)`` or ``(3, N)``."""
        arr = RigidRegistrationSolver._as_n_by_3(points, "points")
        T = np.asarray(T_target_source, dtype=float)
        if T.shape != (4, 4):
            raise ValueError("T_target_source must have shape (4, 4)")
        return (T[0:3, 0:3] @ arr.T).T + T[0:3, 3]

    @staticmethod
    def _as_n_by_3(points: np.ndarray, name: str) -> np.ndarray:
        arr = np.asarray(points, dtype=float)
        if arr.ndim != 2:
            raise ValueError(f"{name} must be a 2D array")
        if arr.shape[1] == 3:
            return arr
        if arr.shape[0] == 3:
            return arr.T
        raise ValueError(f"{name} must have shape (N, 3) or (3, N)")

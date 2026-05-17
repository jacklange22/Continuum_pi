"""Tests for RANSAC-augmented rigid registration and pivot calibration.

These exercise the new RANSAC paths end-to-end on synthetic data with planted
outliers, plus the integration glue in ``trial_analysis`` and the
``pivot_calibration`` experiment in ``critical_experiments``. The classical
solvers are unchanged and have their own coverage; what matters here is that
the RANSAC variants

* recover the same transforms when given clean data,
* identify and reject planted outliers when the data is contaminated,
* return rich diagnostics (iteration history, inlier masks, threshold used),
* are reproducible under a fixed seed,
* fail loudly with a clear error when no consensus can be found, and
* slot cleanly into the registration trial and pivot experiment without
  breaking the classical default paths.
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest

from continuum_robot.experiments.pivot_utils import (
    PivotCalibrationResult,
    PivotRansacFailure,
    RansacPivotCalibrationResult,
    solve_pivot_calibration,
    solve_pivot_calibration_ransac,
)
from continuum_robot.registration.rigid_solver import (
    RansacRegistrationResult,
    RigidRegistrationFailure,
    RigidRegistrationSolver,
)
from continuum_robot.registration.trial_analysis import (
    RansacOptions,
    evaluate_method,
    solve_with_metrics,
    solve_with_metrics_ransac,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _rotation_matrix(axis: np.ndarray, angle: float) -> np.ndarray:
    """Return a 3x3 rotation via the standard Rodrigues formula."""
    axis = np.asarray(axis, dtype=float)
    axis = axis / np.linalg.norm(axis)
    K = np.asarray(
        [
            [0.0, -axis[2], axis[1]],
            [axis[2], 0.0, -axis[0]],
            [-axis[1], axis[0], 0.0],
        ],
        dtype=float,
    )
    return np.eye(3, dtype=float) + math.sin(angle) * K + (1.0 - math.cos(angle)) * (K @ K)


def _make_ground_truth_transform(seed: int = 0) -> np.ndarray:
    """Return a fixed 4x4 transform reproducible from ``seed``."""
    rng = np.random.default_rng(seed)
    axis = rng.normal(size=3)
    angle = float(rng.uniform(-math.pi / 2.0, math.pi / 2.0))
    R = _rotation_matrix(axis, angle)
    t = rng.uniform(-50.0, 50.0, size=3)
    T = np.eye(4, dtype=float)
    T[0:3, 0:3] = R
    T[0:3, 3] = t
    return T


def _generate_paired_points(
    *,
    n_total: int = 14,
    noise_std_mm: float = 0.05,
    outlier_count: int = 0,
    outlier_magnitude_mm: float = 25.0,
    seed: int = 1,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[int]]:
    """Construct (source, target, ground-truth T, outlier_indices).

    Points are sampled in source-frame, transformed by a ground-truth T into
    target-frame, then perturbed by small Gaussian noise. ``outlier_count``
    target points get a large extra offset added to simulate gross outliers.
    The returned ``outlier_indices`` records which positions were corrupted
    so tests can assert RANSAC identifies them.
    """
    rng = np.random.default_rng(seed)
    T = _make_ground_truth_transform(seed=seed + 99)
    src = rng.uniform(-100.0, 100.0, size=(n_total, 3))
    transformed = (T[0:3, 0:3] @ src.T).T + T[0:3, 3]
    target = transformed + rng.normal(scale=noise_std_mm, size=transformed.shape)
    outlier_indices: list[int] = []
    if outlier_count > 0:
        outlier_indices = sorted(rng.choice(n_total, size=int(outlier_count), replace=False).tolist())
        for index in outlier_indices:
            direction = rng.normal(size=3)
            direction /= np.linalg.norm(direction)
            target[index] += direction * float(outlier_magnitude_mm)
    return src, target, T, outlier_indices


def _generate_pivot_samples(
    *,
    n_total: int = 24,
    tip_local_mm: np.ndarray | None = None,
    pivot_tracker_mm: np.ndarray | None = None,
    noise_std_mm: float = 0.05,
    outlier_count: int = 0,
    outlier_magnitude_mm: float = 25.0,
    seed: int = 7,
) -> tuple[list[np.ndarray], list[np.ndarray], np.ndarray, np.ndarray, list[int]]:
    """Synthesize pivot poses: ``R_i @ tip + t_i = pivot``.

    Builds ``n_total`` rotation matrices spanning a range of axes and angles
    (no two are parallel), solves for ``t_i`` exactly, then perturbs
    translations with small Gaussian noise. ``outlier_count`` poses get a
    large random offset added to ``t_i`` to simulate gross outliers.
    Returns the rotations, translations, ground-truth tip + pivot, and the
    list of outlier indices.
    """
    rng = np.random.default_rng(seed)
    tip = np.asarray(tip_local_mm if tip_local_mm is not None else [10.0, 20.0, 100.0], dtype=float)
    pivot = np.asarray(pivot_tracker_mm if pivot_tracker_mm is not None else [25.0, -10.0, 40.0], dtype=float)
    rotations: list[np.ndarray] = []
    translations: list[np.ndarray] = []
    for _ in range(int(n_total)):
        axis = rng.normal(size=3)
        if float(np.linalg.norm(axis)) < 1e-9:
            axis = np.asarray([0.0, 0.0, 1.0], dtype=float)
        angle = float(rng.uniform(-math.pi, math.pi))
        R = _rotation_matrix(axis, angle)
        t = pivot - (R @ tip) + rng.normal(scale=noise_std_mm, size=3)
        rotations.append(R)
        translations.append(t)
    outlier_indices: list[int] = []
    if outlier_count > 0:
        outlier_indices = sorted(rng.choice(n_total, size=int(outlier_count), replace=False).tolist())
        for index in outlier_indices:
            direction = rng.normal(size=3)
            direction /= np.linalg.norm(direction)
            translations[index] = translations[index] + direction * float(outlier_magnitude_mm)
    return rotations, translations, tip, pivot, outlier_indices


# ---------------------------------------------------------------------------
# RigidRegistrationSolver.solve_T_robot_aurora_ransac
# ---------------------------------------------------------------------------


class TestRigidRegistrationRansac:
    def test_clean_data_recovers_ground_truth(self) -> None:
        src, dst, T_truth, _ = _generate_paired_points(n_total=12, noise_std_mm=0.0, seed=11)
        solver = RigidRegistrationSolver()
        result = solver.solve_T_robot_aurora_ransac(src, dst, seed=0)
        assert isinstance(result, RansacRegistrationResult)
        assert result.converged is True
        assert result.sample_count_used == src.shape[0]
        assert result.best_consensus_size == src.shape[0]
        assert result.inlier_rmse_mm < 1e-6
        assert np.allclose(result.T_target_source, T_truth, atol=1e-6)

    def test_clean_data_matches_classical_solver(self) -> None:
        src, dst, _, _ = _generate_paired_points(n_total=10, noise_std_mm=0.02, seed=12)
        solver = RigidRegistrationSolver()
        classical_T = solver.solve_T_robot_aurora(src, dst)
        ransac_result = solver.solve_T_robot_aurora_ransac(src, dst, seed=0)
        # On clean data with small noise, both solvers should agree to noise level
        assert np.allclose(ransac_result.T_target_source, classical_T, atol=1e-3)

    def test_identifies_planted_outliers(self) -> None:
        src, dst, T_truth, planted = _generate_paired_points(
            n_total=15, noise_std_mm=0.05, outlier_count=4, outlier_magnitude_mm=20.0, seed=13
        )
        solver = RigidRegistrationSolver()
        result = solver.solve_T_robot_aurora_ransac(
            src, dst, inlier_threshold_mm=1.0, seed=0, max_iterations=200
        )
        assert result.converged is True
        # Every planted outlier should have been rejected by the final mask.
        assert set(planted).issubset(set(result.rejected_indices))
        # Recovered transform should be close to truth despite the outliers.
        assert np.allclose(result.T_target_source, T_truth, atol=0.1)

    def test_recovers_under_heavy_outlier_contamination(self) -> None:
        src, dst, T_truth, planted = _generate_paired_points(
            n_total=20, noise_std_mm=0.05, outlier_count=8, outlier_magnitude_mm=30.0, seed=14
        )
        solver = RigidRegistrationSolver()
        result = solver.solve_T_robot_aurora_ransac(
            src,
            dst,
            inlier_threshold_mm=1.0,
            min_consensus_size=10,
            max_iterations=500,
            seed=0,
        )
        assert result.converged is True
        assert set(planted).issubset(set(result.rejected_indices))
        assert np.allclose(result.T_target_source, T_truth, atol=0.1)

    def test_raises_when_no_consensus_meets_floor(self) -> None:
        # Generate data where every point is essentially an outlier relative
        # to the inlier threshold so RANSAC cannot find a 10-point consensus.
        rng = np.random.default_rng(15)
        src = rng.uniform(-100.0, 100.0, size=(12, 3))
        dst = rng.uniform(-100.0, 100.0, size=(12, 3))  # totally unrelated targets
        solver = RigidRegistrationSolver()
        with pytest.raises(RigidRegistrationFailure) as excinfo:
            solver.solve_T_robot_aurora_ransac(
                src,
                dst,
                inlier_threshold_mm=0.5,
                min_consensus_size=10,
                max_iterations=80,
                seed=0,
            )
        partial = excinfo.value.partial
        assert "best_consensus_size" in partial
        assert "iterations_run" in partial
        assert int(partial["best_consensus_size"]) < 10

    def test_seed_makes_iteration_history_reproducible(self) -> None:
        src, dst, _, _ = _generate_paired_points(
            n_total=14, noise_std_mm=0.1, outlier_count=2, seed=16
        )
        solver = RigidRegistrationSolver()
        first = solver.solve_T_robot_aurora_ransac(src, dst, seed=42, max_iterations=50)
        second = solver.solve_T_robot_aurora_ransac(src, dst, seed=42, max_iterations=50)
        assert first.iterations_run == second.iterations_run
        assert np.allclose(first.T_target_source, second.T_target_source, atol=1e-12)
        assert first.inlier_mask == second.inlier_mask
        assert first.iteration_history == second.iteration_history

    def test_adaptive_budget_shrinks_under_high_inlier_ratio(self) -> None:
        src, dst, _, _ = _generate_paired_points(n_total=15, noise_std_mm=0.0, seed=17)
        solver = RigidRegistrationSolver()
        result = solver.solve_T_robot_aurora_ransac(
            src, dst, inlier_threshold_mm=1.0, max_iterations=1000, seed=0
        )
        # With every point an inlier the adaptive schedule should stop within
        # only a handful of iterations, well below the budget cap.
        assert result.iterations_run < 50
        # The recorded iteration budget should reflect the shrunk schedule.
        last_budget = int(result.iteration_history[-1]["iteration_budget"])
        assert last_budget <= 50

    def test_input_shape_3xN_is_accepted(self) -> None:
        src, dst, T_truth, _ = _generate_paired_points(n_total=10, noise_std_mm=0.0, seed=18)
        solver = RigidRegistrationSolver()
        result = solver.solve_T_robot_aurora_ransac(src.T, dst.T, seed=0)
        assert np.allclose(result.T_target_source, T_truth, atol=1e-6)
        assert result.sample_count_total == src.shape[0]

    def test_validates_minimum_sample_size(self) -> None:
        src, dst, _, _ = _generate_paired_points(n_total=10, noise_std_mm=0.0, seed=19)
        solver = RigidRegistrationSolver()
        with pytest.raises(ValueError, match="minimum_sample_size"):
            solver.solve_T_robot_aurora_ransac(src, dst, minimum_sample_size=2, seed=0)

    def test_validates_inlier_threshold_positive(self) -> None:
        src, dst, _, _ = _generate_paired_points(n_total=10, noise_std_mm=0.0, seed=20)
        solver = RigidRegistrationSolver()
        with pytest.raises(ValueError, match="inlier_threshold_mm"):
            solver.solve_T_robot_aurora_ransac(src, dst, inlier_threshold_mm=0.0, seed=0)

    def test_validates_confidence_in_open_unit_interval(self) -> None:
        src, dst, _, _ = _generate_paired_points(n_total=10, noise_std_mm=0.0, seed=21)
        solver = RigidRegistrationSolver()
        with pytest.raises(ValueError, match="confidence"):
            solver.solve_T_robot_aurora_ransac(src, dst, confidence=1.0, seed=0)
        with pytest.raises(ValueError, match="confidence"):
            solver.solve_T_robot_aurora_ransac(src, dst, confidence=0.0, seed=0)

    def test_validates_min_consensus_versus_sample_size(self) -> None:
        src, dst, _, _ = _generate_paired_points(n_total=10, noise_std_mm=0.0, seed=22)
        solver = RigidRegistrationSolver()
        with pytest.raises(ValueError, match="min_consensus_size"):
            solver.solve_T_robot_aurora_ransac(
                src, dst, minimum_sample_size=4, min_consensus_size=3, seed=0
            )

    def test_diagnostics_to_dict_roundtrip(self) -> None:
        src, dst, _, _ = _generate_paired_points(n_total=10, noise_std_mm=0.0, seed=23)
        solver = RigidRegistrationSolver()
        result = solver.solve_T_robot_aurora_ransac(src, dst, seed=0)
        payload = result.to_dict()
        assert payload["converged"] is True
        assert payload["sample_count_total"] == src.shape[0]
        assert payload["minimum_sample_size"] == 3
        assert isinstance(payload["iteration_history"], list)
        assert isinstance(payload["T_target_source"], list)
        assert len(payload["T_target_source"]) == 4
        assert len(payload["T_target_source"][0]) == 4


# ---------------------------------------------------------------------------
# solve_pivot_calibration_ransac
# ---------------------------------------------------------------------------


class TestPivotCalibrationRansac:
    def test_clean_data_recovers_ground_truth(self) -> None:
        rotations, translations, tip, pivot, _ = _generate_pivot_samples(
            n_total=20, noise_std_mm=0.0, seed=31
        )
        result = solve_pivot_calibration_ransac(rotations, translations, seed=0)
        assert isinstance(result, RansacPivotCalibrationResult)
        assert result.converged is True
        assert np.allclose(result.tip_vector_local_mm, tip, atol=1e-6)
        assert np.allclose(result.pivot_point_tracker_mm, pivot, atol=1e-6)
        assert result.sample_count_used == len(rotations)
        assert result.rmse_mm < 1e-6

    def test_clean_data_matches_classical_solve(self) -> None:
        rotations, translations, tip, pivot, _ = _generate_pivot_samples(
            n_total=24, noise_std_mm=0.02, seed=32
        )
        classical = solve_pivot_calibration(rotations, translations, std_dev_threshold=10.0, min_samples=8)
        ransac_result = solve_pivot_calibration_ransac(rotations, translations, seed=0)
        # Both should land near the planted tip with no real outliers around.
        assert np.allclose(classical.tip_vector_local_mm, tip, atol=0.05)
        assert np.allclose(ransac_result.tip_vector_local_mm, tip, atol=0.05)
        assert np.allclose(classical.pivot_point_tracker_mm, pivot, atol=0.05)
        assert np.allclose(ransac_result.pivot_point_tracker_mm, pivot, atol=0.05)

    def test_identifies_planted_outliers(self) -> None:
        rotations, translations, tip, pivot, planted = _generate_pivot_samples(
            n_total=24, noise_std_mm=0.05, outlier_count=5, outlier_magnitude_mm=20.0, seed=33
        )
        result = solve_pivot_calibration_ransac(
            rotations,
            translations,
            inlier_threshold_mm=1.0,
            max_iterations=500,
            seed=0,
        )
        assert result.converged is True
        assert set(planted).issubset(set(result.rejected_indices))
        assert np.allclose(result.tip_vector_local_mm, tip, atol=0.1)
        assert np.allclose(result.pivot_point_tracker_mm, pivot, atol=0.1)

    def test_recovers_under_heavy_outlier_contamination(self) -> None:
        rotations, translations, tip, pivot, planted = _generate_pivot_samples(
            n_total=30, noise_std_mm=0.05, outlier_count=12, outlier_magnitude_mm=30.0, seed=34
        )
        result = solve_pivot_calibration_ransac(
            rotations,
            translations,
            inlier_threshold_mm=1.0,
            min_consensus_size=14,
            max_iterations=1000,
            seed=0,
        )
        assert result.converged is True
        assert set(planted).issubset(set(result.rejected_indices))
        assert np.allclose(result.tip_vector_local_mm, tip, atol=0.2)
        assert np.allclose(result.pivot_point_tracker_mm, pivot, atol=0.2)

    def test_raises_when_no_consensus_meets_floor(self) -> None:
        rng = np.random.default_rng(35)
        rotations = []
        translations = []
        for _ in range(14):
            axis = rng.normal(size=3)
            angle = float(rng.uniform(-math.pi, math.pi))
            rotations.append(_rotation_matrix(axis, angle))
            translations.append(rng.uniform(-100.0, 100.0, size=3))
        with pytest.raises(PivotRansacFailure) as excinfo:
            solve_pivot_calibration_ransac(
                rotations,
                translations,
                inlier_threshold_mm=0.5,
                min_consensus_size=10,
                max_iterations=80,
                seed=0,
            )
        partial = excinfo.value.partial
        assert "best_consensus_size" in partial
        assert int(partial["best_consensus_size"]) < 10

    def test_seed_makes_results_reproducible(self) -> None:
        rotations, translations, _, _, _ = _generate_pivot_samples(
            n_total=24, noise_std_mm=0.1, outlier_count=3, seed=36
        )
        first = solve_pivot_calibration_ransac(rotations, translations, seed=7, max_iterations=50)
        second = solve_pivot_calibration_ransac(rotations, translations, seed=7, max_iterations=50)
        assert first.iterations_run == second.iterations_run
        assert first.inlier_mask == second.inlier_mask
        assert np.allclose(first.tip_vector_local_mm, second.tip_vector_local_mm, atol=1e-12)
        assert np.allclose(first.pivot_point_tracker_mm, second.pivot_point_tracker_mm, atol=1e-12)

    def test_adaptive_budget_shrinks_under_high_inlier_ratio(self) -> None:
        rotations, translations, _, _, _ = _generate_pivot_samples(
            n_total=20, noise_std_mm=0.0, seed=37
        )
        result = solve_pivot_calibration_ransac(
            rotations, translations, inlier_threshold_mm=1.0, max_iterations=1000, seed=0
        )
        assert result.iterations_run < 60
        last_budget = int(result.iteration_history[-1]["iteration_budget"])
        assert last_budget <= 60

    def test_validates_minimum_sample_size_floor(self) -> None:
        rotations, translations, _, _, _ = _generate_pivot_samples(n_total=10, seed=38)
        with pytest.raises(ValueError, match="minimum_sample_size"):
            solve_pivot_calibration_ransac(rotations, translations, minimum_sample_size=1, seed=0)

    def test_validates_input_length_match(self) -> None:
        rotations, translations, _, _, _ = _generate_pivot_samples(n_total=10, seed=39)
        with pytest.raises(ValueError, match="length mismatch"):
            solve_pivot_calibration_ransac(rotations, translations[:-1], seed=0)

    def test_to_pivot_calibration_result_projects_cleanly(self) -> None:
        rotations, translations, _, _, _ = _generate_pivot_samples(n_total=14, noise_std_mm=0.0, seed=40)
        result = solve_pivot_calibration_ransac(rotations, translations, seed=0)
        projected = result.to_pivot_calibration_result()
        assert isinstance(projected, PivotCalibrationResult)
        assert projected.tip_vector_local_mm == result.tip_vector_local_mm
        assert projected.pivot_point_tracker_mm == result.pivot_point_tracker_mm
        assert projected.rmse_mm == result.rmse_mm
        assert projected.sample_count_total == result.sample_count_total
        assert projected.sample_count_used == result.sample_count_used
        assert projected.sample_count_rejected == result.sample_count_rejected
        assert projected.inlier_mask == result.inlier_mask
        assert projected.rejected_indices == result.rejected_indices

    def test_to_dict_roundtrip(self) -> None:
        rotations, translations, _, _, _ = _generate_pivot_samples(n_total=14, noise_std_mm=0.0, seed=41)
        result = solve_pivot_calibration_ransac(rotations, translations, seed=0)
        payload = result.to_dict()
        assert payload["converged"] is True
        assert payload["seed"] == 0
        assert payload["minimum_sample_size"] == 3
        assert isinstance(payload["iteration_history"], list)
        assert isinstance(payload["residuals_mm"], list)


# ---------------------------------------------------------------------------
# trial_analysis wire-in
# ---------------------------------------------------------------------------


class TestTrialAnalysisRansacIntegration:
    def _registration_inputs(
        self,
        *,
        n_landmarks: int = 8,
        outlier_count: int = 0,
        outlier_magnitude_mm: float = 25.0,
        seed: int = 51,
    ) -> tuple[dict[str, list[float]], dict[str, list[float]], list[str], np.ndarray]:
        src, dst, T_truth, planted = _generate_paired_points(
            n_total=n_landmarks,
            noise_std_mm=0.05,
            outlier_count=outlier_count,
            outlier_magnitude_mm=outlier_magnitude_mm,
            seed=seed,
        )
        labels = [f"L{idx:02d}" for idx in range(n_landmarks)]
        measured = {label: src[idx].tolist() for idx, label in enumerate(labels)}
        truth = {label: dst[idx].tolist() for idx, label in enumerate(labels)}
        return measured, truth, [labels[idx] for idx in planted], T_truth

    def test_solve_with_metrics_ransac_returns_same_shape_plus_ransac_key(self) -> None:
        measured, truth, _, _ = self._registration_inputs(n_landmarks=10, seed=52)
        classical = solve_with_metrics(measured, truth)
        ransac_solve = solve_with_metrics_ransac(
            measured, truth, options=RansacOptions(seed=0, inlier_threshold_mm=2.0)
        )
        for key in (
            "labels",
            "T_robot_aurora",
            "fre_mm",
            "residuals_xyz_by_label",
            "residual_norms_mm_by_label",
            "max_residual_mm",
            "worst_landmark_label",
            "geometry",
        ):
            assert key in classical
            assert key in ransac_solve
        assert "ransac" not in classical
        assert "ransac" in ransac_solve
        diagnostics = ransac_solve["ransac"]
        assert isinstance(diagnostics, dict)
        assert diagnostics["converged"] is True
        assert diagnostics["sample_count_total"] == len(measured)
        assert isinstance(diagnostics["inlier_labels"], list)

    def test_ransac_solve_rejects_outlier_landmark_labels(self) -> None:
        measured, truth, planted_labels, T_truth = self._registration_inputs(
            n_landmarks=12, outlier_count=3, outlier_magnitude_mm=20.0, seed=53
        )
        result = solve_with_metrics_ransac(
            measured,
            truth,
            options=RansacOptions(seed=0, inlier_threshold_mm=1.0, max_iterations=200),
        )
        rejected = set(result["ransac"]["rejected_labels"])
        assert set(planted_labels).issubset(rejected)
        recovered = np.asarray(result["T_robot_aurora"], dtype=float)
        assert np.allclose(recovered, T_truth, atol=0.1)

    def test_evaluate_method_populates_ransac_field_when_options_given(self) -> None:
        measured, truth, _, _ = self._registration_inputs(n_landmarks=10, seed=54)
        captures = {label: [point] for label, point in measured.items()}
        result_classical = evaluate_method(captures, truth, method="mean")
        result_ransac = evaluate_method(
            captures,
            truth,
            method="mean",
            ransac_options=RansacOptions(seed=0, inlier_threshold_mm=2.0),
        )
        assert result_classical.ransac is None
        assert isinstance(result_ransac.ransac, dict)
        assert result_ransac.ransac["converged"] is True
        # FRE should be close on essentially clean data.
        assert abs(result_classical.fre_mm - result_ransac.fre_mm) < 0.5

    def test_evaluate_method_with_ransac_rejects_outlier_landmark(self) -> None:
        measured, truth, planted_labels, _ = self._registration_inputs(
            n_landmarks=12, outlier_count=3, outlier_magnitude_mm=20.0, seed=55
        )
        captures = {label: [point] for label, point in measured.items()}
        result = evaluate_method(
            captures,
            truth,
            method="mean",
            ransac_options=RansacOptions(seed=0, inlier_threshold_mm=1.0, max_iterations=200),
        )
        assert result.ransac is not None
        rejected = set(result.ransac["rejected_labels"])
        assert set(planted_labels).issubset(rejected)


# ---------------------------------------------------------------------------
# critical_experiments pivot path
# ---------------------------------------------------------------------------


class TestPivotExperimentRansacIntegration:
    def test_config_round_trips_ransac_fields(self) -> None:
        from continuum_robot.experiments.critical_experiments import PivotCalibrationConfig

        payload = {
            "use_ransac": True,
            "ransac_inlier_threshold_mm": 0.5,
            "ransac_minimum_sample_size": 4,
            "ransac_min_consensus_size": 18,
            "ransac_max_iterations": 250,
            "ransac_confidence": 0.999,
            "ransac_seed": 11,
        }
        config = PivotCalibrationConfig.from_dict(payload)
        assert config.use_ransac is True
        assert config.ransac_inlier_threshold_mm == 0.5
        assert config.ransac_minimum_sample_size == 4
        assert config.ransac_min_consensus_size == 18
        assert config.ransac_max_iterations == 250
        assert config.ransac_confidence == 0.999
        assert config.ransac_seed == 11

    def test_config_default_keeps_classical_behavior(self) -> None:
        from continuum_robot.experiments.critical_experiments import PivotCalibrationConfig

        config = PivotCalibrationConfig.from_dict({})
        assert config.use_ransac is False
        assert config.ransac_seed is None
        assert config.ransac_min_consensus_size is None

    @pytest.mark.parametrize("use_ransac", [False, True])
    def test_pivot_calibration_offline_with_optional_ransac(
        self,
        tmp_path: Path,
        use_ransac: bool,
    ) -> None:
        # Reuse the same helpers and runner the existing pivot tests use so
        # any divergence shows up next to existing coverage.
        from tests.test_critical_experiments import (
            _pivot_rotations_and_translations,
            _runner,
        )
        from continuum_robot.tracking.transforms import rotmat_to_quat_wxyz

        rotations, translations = _pivot_rotations_and_translations()
        # Plant a single gross outlier in the translation set so RANSAC has
        # something visible to reject; the classical std-dev path also
        # tolerates this with threshold 3.0.
        translations_with_outlier = [np.asarray(t, dtype=float).copy() for t in translations]
        translations_with_outlier[0] = translations_with_outlier[0] + np.asarray(
            [12.0, -10.0, 8.0], dtype=float
        )

        csv_path = tmp_path / "pivot_samples_ransac.csv"
        with csv_path.open("w", encoding="utf-8") as handle:
            for rotation, translation in zip(rotations, translations_with_outlier):
                quat = rotmat_to_quat_wxyz(np.asarray(rotation, dtype=float))
                handle.write(
                    ",".join(
                        [
                            "0",
                            "0B",
                            f"{float(quat[0]):.10f}",
                            f"{float(quat[1]):.10f}",
                            f"{float(quat[2]):.10f}",
                            f"{float(quat[3]):.10f}",
                            f"{float(translation[0]):.6f}",
                            f"{float(translation[1]):.6f}",
                            f"{float(translation[2]):.6f}",
                        ]
                    )
                    + "\n"
                )

        runner = _runner(tmp_path)
        config: dict[str, object] = {
            "tool_id": "0B",
            "input_path": str(csv_path),
            "output_tip_file": str(tmp_path / "tip.csv"),
            "std_dev_threshold": 3.0,
            "min_samples": 8,
        }
        if use_ransac:
            config.update(
                {
                    "use_ransac": True,
                    "ransac_inlier_threshold_mm": 1.0,
                    "ransac_minimum_sample_size": 3,
                    "ransac_max_iterations": 400,
                    "ransac_seed": 0,
                }
            )

        result = runner.run_experiment("pivot_calibration", config=config)
        assert result.success is True
        metrics = result.summary.experiment_metrics
        assert metrics["pivot_solver"] == ("ransac" if use_ransac else "classical_std_dev")
        if use_ransac:
            assert "pivot_ransac" in metrics
            assert metrics["pivot_ransac"]["converged"] is True
            # The planted outlier (index 0) should be rejected by RANSAC.
            assert 0 in metrics["pivot_ransac"]["rejected_indices"]
            # Recovered tip should still match the planted ground truth.
            assert np.allclose(
                metrics["tip_vector_local_mm"], [10.0, 20.0, 100.0], atol=0.1
            )
        else:
            assert "pivot_ransac" not in metrics

    def test_pivot_calibration_ransac_failure_marks_status_invalid(
        self,
        tmp_path: Path,
    ) -> None:
        from tests.test_critical_experiments import (
            _pivot_rotations_and_translations,
            _runner,
        )
        from continuum_robot.tracking.transforms import rotmat_to_quat_wxyz

        rotations, translations = _pivot_rotations_and_translations()
        # Corrupt every translation with a large random offset so no consensus
        # can form within the configured inlier threshold.
        rng = np.random.default_rng(99)
        corrupted = [
            np.asarray(t, dtype=float) + rng.uniform(-50.0, 50.0, size=3)
            for t in translations
        ]
        csv_path = tmp_path / "pivot_samples_bad.csv"
        with csv_path.open("w", encoding="utf-8") as handle:
            for rotation, translation in zip(rotations, corrupted):
                quat = rotmat_to_quat_wxyz(np.asarray(rotation, dtype=float))
                handle.write(
                    ",".join(
                        [
                            "0",
                            "0B",
                            f"{float(quat[0]):.10f}",
                            f"{float(quat[1]):.10f}",
                            f"{float(quat[2]):.10f}",
                            f"{float(quat[3]):.10f}",
                            f"{float(translation[0]):.6f}",
                            f"{float(translation[1]):.6f}",
                            f"{float(translation[2]):.6f}",
                        ]
                    )
                    + "\n"
                )

        runner = _runner(tmp_path)
        result = runner.run_experiment(
            "pivot_calibration",
            config={
                "tool_id": "0B",
                "input_path": str(csv_path),
                "output_tip_file": str(tmp_path / "tip.csv"),
                "min_samples": 4,
                "use_ransac": True,
                "ransac_inlier_threshold_mm": 0.1,
                "ransac_min_consensus_size": 9,
                "ransac_max_iterations": 80,
                "ransac_seed": 0,
            },
        )
        assert result.success is False
        metrics = result.summary.experiment_metrics
        assert metrics["pivot_solver"] == "ransac"
        assert metrics["status"].startswith("invalid_")
        partial = metrics.get("ransac_failure_partial")
        assert isinstance(partial, dict)
        assert "best_consensus_size" in partial

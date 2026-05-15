from __future__ import annotations

import numpy as np
import pytest

from continuum_robot.registration.trial_analysis import (
    AVERAGING_METHODS,
    average_captures,
    evaluate_all_subsets,
    evaluate_method,
    leave_one_out_fre,
    mad_outlier_mask,
    solve_with_metrics,
    summarize_subset_search,
    summarize_trial,
    sweep_methods,
)


def _rotation_z(theta_rad: float) -> np.ndarray:
    c, s = float(np.cos(theta_rad)), float(np.sin(theta_rad))
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=float)


def _make_truth_landmarks() -> dict[str, list[float]]:
    return {
        "L1": [0.0, 35.0, -5.0],
        "L2": [-35.0, 0.0, -5.0],
        "L3": [0.0, -35.0, -5.0],
        "L4": [35.0, 0.0, -5.0],
    }


def _make_clean_captures(
    truth_by_label: dict[str, list[float]],
    *,
    T_robot_aurora: np.ndarray,
    captures_per_label: int = 20,
    rng_seed: int = 0,
) -> dict[str, np.ndarray]:
    """Each capture sits at the inverse-transformed truth point plus tiny Gaussian noise."""
    rng = np.random.default_rng(rng_seed)
    T_aurora_robot = np.linalg.inv(T_robot_aurora)
    captures: dict[str, np.ndarray] = {}
    for label, xyz_robot in truth_by_label.items():
        truth_h = np.append(np.asarray(xyz_robot, dtype=float), 1.0)
        center_aurora = (T_aurora_robot @ truth_h)[:3]
        noise = rng.normal(scale=0.05, size=(captures_per_label, 3))
        captures[label] = center_aurora + noise
    return captures


def test_mad_outlier_mask_keeps_consistent_cluster_drops_one_outlier() -> None:
    points = np.zeros((11, 3), dtype=float)
    rng = np.random.default_rng(0)
    points[:10] = rng.normal(scale=0.01, size=(10, 3))
    points[10] = [5.0, 5.0, 5.0]  # obvious outlier 50+ stddev away

    mask = mad_outlier_mask(points, k=3.5)
    assert mask.shape == (11,)
    assert bool(mask[10]) is False
    assert int(mask[:10].sum()) >= 8  # the inliers are kept


def test_mad_outlier_mask_handles_zero_spread() -> None:
    points = np.tile(np.array([1.0, 2.0, 3.0]), (5, 1))
    mask = mad_outlier_mask(points, k=3.5)
    assert mask.tolist() == [True] * 5


def test_average_captures_mean_matches_numpy_mean() -> None:
    rng = np.random.default_rng(1)
    pts = rng.normal(size=(8, 3))
    avg = average_captures(pts, method="mean")
    np.testing.assert_allclose(avg.averaged_xyz_mm, pts.mean(axis=0))
    assert avg.n_input == 8 and avg.n_kept == 8 and avg.n_rejected == 0


def test_average_captures_median_matches_numpy_median() -> None:
    pts = np.array([[0.0, 0.0, 0.0], [10.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    avg = average_captures(pts, method="median")
    np.testing.assert_allclose(avg.averaged_xyz_mm, [1.0, 0.0, 0.0])


def test_average_captures_mad_filtered_drops_outlier_only() -> None:
    pts = np.zeros((11, 3), dtype=float)
    pts[10] = [5.0, 5.0, 5.0]
    avg = average_captures(pts, method="mad_filtered_mean")
    # Outlier rejected; averaged_xyz should be near the inlier cluster (zeros).
    assert avg.n_rejected >= 1
    np.testing.assert_allclose(avg.averaged_xyz_mm, [0.0, 0.0, 0.0], atol=1e-6)


def test_average_captures_trimmed_mean_drops_top_and_bottom() -> None:
    pts = np.array(
        [
            [0.0, 0.0, 0.0],
            [0.1, 0.0, 0.0],
            [-0.1, 0.0, 0.0],
            [10.0, 0.0, 0.0],
            [-10.0, 0.0, 0.0],
        ]
    )
    avg = average_captures(pts, method="trimmed_mean", trimmed_fraction=0.2)
    # Median is [0,0,0]; trimmed_fraction=0.2 of 5 = 1 sample dropped (farthest single).
    # Result should ignore the worst outlier.
    assert avg.n_rejected >= 1
    assert abs(avg.averaged_xyz_mm[0]) < 5.0


def test_solve_with_metrics_recovers_known_transform_with_zero_noise() -> None:
    truth = _make_truth_landmarks()
    T_robot_aurora = np.eye(4, dtype=float)
    T_robot_aurora[0:3, 0:3] = _rotation_z(0.3)
    T_robot_aurora[0:3, 3] = [12.0, -7.0, 4.0]
    T_aurora_robot = np.linalg.inv(T_robot_aurora)
    measured = {
        label: ((T_aurora_robot @ np.append(np.asarray(xyz), 1.0))[:3]).tolist()
        for label, xyz in truth.items()
    }
    result = solve_with_metrics(measured, truth)
    assert result["fre_mm"] < 1e-9
    np.testing.assert_allclose(np.asarray(result["T_robot_aurora"]), T_robot_aurora, atol=1e-9)


def test_solve_with_metrics_rejects_two_correspondences() -> None:
    with pytest.raises(ValueError):
        solve_with_metrics({"L1": [0, 0, 0], "L2": [1, 1, 1]}, {"L1": [0, 0, 0], "L2": [1, 1, 1]})


def test_evaluate_method_clean_data_gives_low_fre() -> None:
    truth = _make_truth_landmarks()
    T_robot_aurora = np.eye(4, dtype=float)
    T_robot_aurora[0:3, 3] = [10.0, 5.0, 2.0]
    captures = _make_clean_captures(
        truth, T_robot_aurora=T_robot_aurora, captures_per_label=30, rng_seed=2
    )
    result = evaluate_method(captures, truth, method="mean")
    # 0.05mm per-axis noise over 30 captures averaged: each landmark error ~0.05/sqrt(30) ~= 0.009 mm,
    # FRE across 4 landmarks should comfortably sit below 0.05 mm.
    assert result.fre_mm < 0.05
    assert all(value.n_input == 30 for value in result.averaging.values())
    assert all(value.n_kept == 30 for value in result.averaging.values())


def test_evaluate_method_mad_beats_mean_when_one_capture_is_garbage() -> None:
    truth = _make_truth_landmarks()
    T_robot_aurora = np.eye(4, dtype=float)
    captures = _make_clean_captures(
        truth, T_robot_aurora=T_robot_aurora, captures_per_label=20, rng_seed=3
    )
    # Inject a single 5 mm bad capture into one landmark.
    captures["L1"][-1] = captures["L1"][-1] + np.array([5.0, 0.0, 0.0])
    mean_result = evaluate_method(captures, truth, method="mean")
    mad_result = evaluate_method(captures, truth, method="mad_filtered_mean")
    assert mad_result.fre_mm < mean_result.fre_mm
    # The bad capture should have been excluded by MAD.
    assert mad_result.averaging["L1"].n_rejected >= 1


def test_leave_one_out_fre_drops_significantly_when_dropping_bad_landmark() -> None:
    truth = _make_truth_landmarks()
    T_robot_aurora = np.eye(4, dtype=float)
    captures = _make_clean_captures(
        truth, T_robot_aurora=T_robot_aurora, captures_per_label=20, rng_seed=4
    )
    # Bias the whole L2 cluster by 1 mm — makes L2 the bad landmark.
    captures["L2"] = captures["L2"] + np.array([1.0, 0.0, 0.0])
    averaged = {label: pts.mean(axis=0).tolist() for label, pts in captures.items()}
    all_in = solve_with_metrics(averaged, truth)
    loo = leave_one_out_fre(averaged, truth)
    assert set(loo.keys()) == set(averaged.keys())
    assert loo["L2"] < float(all_in["fre_mm"])  # dropping the bad one helps


def test_leave_one_out_fre_returns_empty_when_only_3_landmarks() -> None:
    measured = {"L1": [0, 0, 0], "L2": [1, 0, 0], "L3": [0, 1, 0]}
    truth = {"L1": [0, 0, 0], "L2": [1, 0, 0], "L3": [0, 1, 0]}
    assert leave_one_out_fre(measured, truth) == {}


def test_sweep_methods_produces_all_methods() -> None:
    truth = _make_truth_landmarks()
    T_robot_aurora = np.eye(4, dtype=float)
    captures = _make_clean_captures(
        truth, T_robot_aurora=T_robot_aurora, captures_per_label=15, rng_seed=5
    )
    results = sweep_methods(captures, truth)
    assert set(results.keys()) == set(AVERAGING_METHODS)
    summary = summarize_trial(results)
    assert summary["best_method"] in AVERAGING_METHODS
    assert summary["best_fre_mm"] >= 0
    assert len(summary["method_rows"]) == len(AVERAGING_METHODS)


def _make_12_landmark_truth_with_height_variation() -> dict[str, list[float]]:
    """A non-coplanar 12-landmark set so subset search has 3D spread to chew on."""
    return {
        "L1": [0.0, 35.0, -5.0],
        "L2": [-35.0, 0.0, -5.0],
        "L3": [0.0, -35.0, -5.0],
        "L4": [35.0, 0.0, -5.0],
        "L5": [0.0, 25.0, 5.0],
        "L6": [-25.0, 0.0, 5.0],
        "L7": [0.0, -25.0, 5.0],
        "L8": [25.0, 0.0, 5.0],
        "L9": [10.0, 25.0, 15.0],
        "L10": [-25.0, 10.0, 15.0],
        "L11": [-10.0, -25.0, 15.0],
        "L12": [25.0, -10.0, 15.0],
    }


def test_evaluate_all_subsets_returns_one_entry_per_combination() -> None:
    truth = {"L1": [0, 0, 0], "L2": [1, 0, 0], "L3": [0, 1, 0], "L4": [1, 1, 1], "L5": [-1, 0, 1]}
    measured = {label: truth[label] for label in truth}  # identity (perfect data)
    subsets = evaluate_all_subsets(measured, truth, sizes=(4, 5))
    # C(5,4)=5 plus C(5,5)=1 = 6
    assert len(subsets) == 6
    assert all(result.fre_mm < 1e-9 for result in subsets)
    assert {result.size for result in subsets} == {4, 5}


def test_evaluate_all_subsets_skips_invalid_sizes() -> None:
    truth = {"L1": [0, 0, 0], "L2": [1, 0, 0], "L3": [0, 1, 0]}
    measured = dict(truth)
    # 4 is impossible with only 3 labels; 2 is below the solver minimum.
    subsets = evaluate_all_subsets(measured, truth, sizes=(2, 3, 4))
    assert {result.size for result in subsets} == {3}
    assert len(subsets) == 1


def test_subset_search_finds_best_for_each_size() -> None:
    truth = _make_12_landmark_truth_with_height_variation()
    rng = np.random.default_rng(7)
    # Identity transform + small noise per label, but bias L3 by 1.5 mm so subsets that
    # avoid L3 should fit much better.
    measured: dict[str, list[float]] = {}
    for label, xyz in truth.items():
        noise = rng.normal(scale=0.02, size=3)
        bias = np.array([1.5, 0.0, 0.0]) if label == "L3" else np.zeros(3)
        measured[label] = (np.asarray(xyz, dtype=float) + noise + bias).tolist()

    subsets = evaluate_all_subsets(measured, truth, sizes=(4, 5, 6, 7, 8))
    summary = summarize_subset_search(subsets)
    per_size = summary["per_size_best"]
    assert set(per_size.keys()) == {4, 5, 6, 7, 8}
    # The global best should exclude L3 because L3 has the 1.5mm bias.
    global_best = summary["global_best"]
    assert global_best is not None
    assert "L3" not in global_best["labels"]
    # Including L3 in any subset would jack FRE; the best-of-4 should be tighter
    # than any subset that includes L3.
    best_size_4 = per_size[4]
    assert "L3" not in best_size_4["best_labels"]
    assert float(best_size_4["best_fre_mm"]) < 0.5


def test_subset_search_summary_with_no_results_is_empty() -> None:
    summary = summarize_subset_search([])
    assert summary["global_best"] is None
    assert summary["per_size_best"] == {}
    assert summary["total_subsets_evaluated"] == 0


def test_summary_picks_lowest_fre_method() -> None:
    truth = _make_truth_landmarks()
    captures = _make_clean_captures(
        truth, T_robot_aurora=np.eye(4), captures_per_label=20, rng_seed=6
    )
    # Inject a single tail outlier into L3 so mean is hurt but mad/trimmed are not.
    captures["L3"][-1] = captures["L3"][-1] + np.array([0.0, 3.0, 0.0])
    results = sweep_methods(captures, truth)
    summary = summarize_trial(results)
    assert summary["best_method"] in ("mad_filtered_mean", "trimmed_mean", "median")
    assert summary["best_fre_mm"] <= results["mean"].fre_mm

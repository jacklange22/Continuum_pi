"""Registration trial-mode analysis helpers.

These helpers exist so an operator can capture a single dense bench dataset
(N landmarks * K captures per landmark) and compare several honest, well-defined
post-processing choices side by side, without re-capturing for each one.

Nothing here changes the meaning of FRE or invents new math. Every method
exposes its own residuals and the same FRE definition is applied across them.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import combinations
from typing import Iterable, Mapping, Sequence

import numpy as np

from continuum_robot.registration.rigid_solver import RigidRegistrationSolver
from continuum_robot.registration.validation import (
    compute_fre_mm,
    compute_geometry_diagnostics,
    compute_residual_norms_mm,
)


AVERAGING_METHODS = ("mean", "median", "trimmed_mean", "mad_filtered_mean")
"""Method identifiers recognized by :func:`average_captures` and the sweep."""


@dataclass
class CaptureAveraging:
    """Result of collapsing K captures of a landmark into one averaged point."""

    method: str
    n_input: int
    n_kept: int
    n_rejected: int
    averaged_xyz_mm: list[float]
    intra_capture_stddev_mm: float


@dataclass
class RegistrationTrialResult:
    """One configuration's worth of registration outcome on a fixed capture set."""

    method: str
    averaging: dict[str, CaptureAveraging]
    fre_mm: float
    max_residual_mm: float
    per_label_residual_mm: dict[str, float]
    loo_fre_mm_by_excluded_label: dict[str, float]
    loo_max_minus_keep_mm: float
    """Largest reduction in FRE attainable by dropping a single landmark."""
    worst_landmark_label: str | None
    geometry: dict[str, object]
    """Output of :func:`compute_geometry_diagnostics` on the *truth* landmarks."""
    method_params: dict[str, object] = field(default_factory=dict)


def mad_outlier_mask(
    points_xyz_mm: Sequence[Sequence[float]],
    *,
    k: float = 3.5,
) -> np.ndarray:
    """Return a boolean mask of captures kept by an MAD outlier filter.

    The distance metric is the Euclidean distance of each capture from the
    coordinate-wise median. Captures with ``distance > k * MAD_scaled`` are
    flagged. ``MAD_scaled = 1.4826 * median_abs_deviation`` so the threshold is
    in units of an approximate Gaussian standard deviation. The 1.4826 factor
    is the standard consistency factor for a normal distribution, not a fudge.

    If all captures sit at the same point (``MAD == 0``) we keep them all,
    which matches the spirit of "no spread, nothing to reject".
    """
    points = np.asarray(points_xyz_mm, dtype=float)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("points_xyz_mm must have shape (N, 3)")
    if points.shape[0] == 0:
        return np.zeros((0,), dtype=bool)
    median = np.median(points, axis=0)
    distances = np.linalg.norm(points - median, axis=1)
    mad = float(np.median(np.abs(distances - np.median(distances))))
    if mad == 0.0:
        # MAD collapses to zero when most points coincide with the median. In that
        # case the cluster is at distance ~0; reject anything visibly off it.
        if np.all(distances < 1e-9):
            return np.ones(points.shape[0], dtype=bool)
        return distances < 1e-9
    threshold = float(k) * 1.4826 * mad
    return distances <= threshold


def average_captures(
    points_xyz_mm: Sequence[Sequence[float]],
    *,
    method: str = "mean",
    trimmed_fraction: float = 0.2,
    mad_k: float = 3.5,
) -> CaptureAveraging:
    """Collapse multiple captures of one landmark to a single averaged point.

    Available ``method`` values:

    - ``"mean"`` — plain arithmetic mean of all captures.
    - ``"median"`` — coordinate-wise median.
    - ``"trimmed_mean"`` — drop the top and bottom ``trimmed_fraction`` of
      samples ranked by Euclidean distance from the median, then mean the rest.
    - ``"mad_filtered_mean"`` — drop samples flagged by :func:`mad_outlier_mask`,
      then mean the kept ones. If everything is rejected (degenerate input), the
      plain mean is returned and ``n_rejected`` reflects the attempted drop count.
    """
    if method not in AVERAGING_METHODS:
        raise ValueError(f"Unknown averaging method {method!r}; expected one of {AVERAGING_METHODS}")
    points = np.asarray(points_xyz_mm, dtype=float)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("points_xyz_mm must have shape (N, 3)")
    if points.shape[0] == 0:
        raise ValueError("points_xyz_mm must have at least one capture")
    n_input = int(points.shape[0])
    intra_std = float(np.linalg.norm(points.std(axis=0, ddof=0)))

    if method == "mean":
        averaged = points.mean(axis=0)
        kept = n_input
    elif method == "median":
        averaged = np.median(points, axis=0)
        kept = n_input
    elif method == "trimmed_mean":
        if not (0.0 <= trimmed_fraction < 0.5):
            raise ValueError("trimmed_fraction must be in [0, 0.5)")
        median = np.median(points, axis=0)
        distances = np.linalg.norm(points - median, axis=1)
        order = np.argsort(distances)
        n_drop_each = int(np.floor(trimmed_fraction * n_input))
        if n_drop_each * 2 >= n_input:
            n_drop_each = max(0, (n_input - 1) // 2)
        keep_idx = order[: n_input - n_drop_each] if n_drop_each else order
        kept_points = points[keep_idx]
        averaged = kept_points.mean(axis=0)
        kept = int(kept_points.shape[0])
    elif method == "mad_filtered_mean":
        mask = mad_outlier_mask(points, k=mad_k)
        if not bool(mask.any()):
            averaged = points.mean(axis=0)
            kept = 0
        else:
            kept_points = points[mask]
            averaged = kept_points.mean(axis=0)
            kept = int(kept_points.shape[0])
    else:  # pragma: no cover — guarded by AVERAGING_METHODS check above.
        raise AssertionError("unreachable")

    return CaptureAveraging(
        method=method,
        n_input=n_input,
        n_kept=int(kept),
        n_rejected=int(n_input - kept),
        averaged_xyz_mm=[float(v) for v in averaged],
        intra_capture_stddev_mm=intra_std,
    )


def solve_with_metrics(
    measured_aurora_by_label: Mapping[str, Sequence[float]],
    truth_robot_by_label: Mapping[str, Sequence[float]],
) -> dict[str, object]:
    """Solve registration and return solve + residual + geometry diagnostics.

    Inputs are already-averaged single 3D points per landmark. This is the
    "one configuration" inner solve used by :func:`evaluate_method` and the
    leave-one-out path.
    """
    labels = sorted(set(measured_aurora_by_label.keys()) & set(truth_robot_by_label.keys()))
    if len(labels) < 3:
        raise ValueError(
            "At least 3 landmark correspondences are required to solve a rigid registration"
        )
    measured = np.asarray([measured_aurora_by_label[label] for label in labels], dtype=float)
    truth = np.asarray([truth_robot_by_label[label] for label in labels], dtype=float)
    solver = RigidRegistrationSolver()
    T_robot_aurora = solver.solve_T_robot_aurora(measured, truth)
    transformed = solver.apply_transform(T_robot_aurora, measured)
    residuals = truth - transformed
    residuals_by_label = {
        label: residuals[idx, :].tolist() for idx, label in enumerate(labels)
    }
    fre = float(compute_fre_mm(list(residuals_by_label.values())))
    residual_norms = compute_residual_norms_mm(residuals_by_label)
    worst_label = max(residual_norms, key=residual_norms.get) if residual_norms else None
    geometry = compute_geometry_diagnostics(truth)
    return {
        "labels": labels,
        "T_robot_aurora": T_robot_aurora.tolist(),
        "fre_mm": fre,
        "residuals_xyz_by_label": residuals_by_label,
        "residual_norms_mm_by_label": residual_norms,
        "max_residual_mm": max(residual_norms.values()) if residual_norms else 0.0,
        "worst_landmark_label": worst_label,
        "geometry": geometry,
    }


def leave_one_out_fre(
    measured_aurora_by_label: Mapping[str, Sequence[float]],
    truth_robot_by_label: Mapping[str, Sequence[float]],
) -> dict[str, float]:
    """Return FRE that would be obtained if each landmark were excluded.

    Useful for spotting one bad landmark that dominates the fit. If the
    excluded-FRE drops far below the all-in FRE, that landmark is the suspect.

    Requires at least 4 shared landmarks; for fewer, the result is empty
    (a 3-point fit is exactly determined and LOO is meaningless).
    """
    shared = sorted(set(measured_aurora_by_label.keys()) & set(truth_robot_by_label.keys()))
    if len(shared) < 4:
        return {}
    out: dict[str, float] = {}
    for excluded in shared:
        subset_measured = {label: measured_aurora_by_label[label] for label in shared if label != excluded}
        subset_truth = {label: truth_robot_by_label[label] for label in shared if label != excluded}
        result = solve_with_metrics(subset_measured, subset_truth)
        out[excluded] = float(result["fre_mm"])
    return out


def evaluate_method(
    captures_by_label: Mapping[str, Sequence[Sequence[float]]],
    truth_robot_by_label: Mapping[str, Sequence[float]],
    *,
    method: str = "mean",
    trimmed_fraction: float = 0.2,
    mad_k: float = 3.5,
) -> RegistrationTrialResult:
    """Run a single averaging method end-to-end and return a trial result.

    ``captures_by_label`` maps each landmark label to its raw K-by-3 capture
    array (already in the Aurora frame, with any tip offset applied at capture
    time). ``truth_robot_by_label`` maps each label to its nominal location in
    the robot frame. This function does no I/O.
    """
    averaging: dict[str, CaptureAveraging] = {}
    averaged_measured: dict[str, list[float]] = {}
    for label, points in captures_by_label.items():
        result = average_captures(
            points,
            method=method,
            trimmed_fraction=trimmed_fraction,
            mad_k=mad_k,
        )
        averaging[label] = result
        averaged_measured[label] = result.averaged_xyz_mm

    solve = solve_with_metrics(averaged_measured, truth_robot_by_label)
    loo = leave_one_out_fre(averaged_measured, truth_robot_by_label)
    keep_fre = float(solve["fre_mm"])
    loo_max_drop = 0.0
    if loo:
        loo_max_drop = max(0.0, keep_fre - min(loo.values()))
    return RegistrationTrialResult(
        method=method,
        averaging=averaging,
        fre_mm=keep_fre,
        max_residual_mm=float(solve["max_residual_mm"]),
        per_label_residual_mm=dict(solve["residual_norms_mm_by_label"]),
        loo_fre_mm_by_excluded_label=loo,
        loo_max_minus_keep_mm=float(loo_max_drop),
        worst_landmark_label=solve["worst_landmark_label"],
        geometry=dict(solve["geometry"]),
        method_params={
            "trimmed_fraction": float(trimmed_fraction),
            "mad_k": float(mad_k),
        },
    )


def sweep_methods(
    captures_by_label: Mapping[str, Sequence[Sequence[float]]],
    truth_robot_by_label: Mapping[str, Sequence[float]],
    *,
    methods: Sequence[str] = AVERAGING_METHODS,
    trimmed_fraction: float = 0.2,
    mad_k: float = 3.5,
) -> dict[str, RegistrationTrialResult]:
    """Run every averaging method on the same captures and return a comparison.

    No data is altered; each method is a fresh post-processing pass over the
    same raw captures. The caller decides what to do with the resulting FREs.
    """
    return {
        method: evaluate_method(
            captures_by_label,
            truth_robot_by_label,
            method=method,
            trimmed_fraction=trimmed_fraction,
            mad_k=mad_k,
        )
        for method in methods
    }


@dataclass
class SubsetSolveResult:
    """One solved subset of labels."""

    labels: tuple[str, ...]
    size: int
    fre_mm: float
    max_residual_mm: float
    worst_landmark_label: str | None
    per_label_residual_mm: dict[str, float]
    geometry_rank: int | None
    geometry_condition_number: float | None


def evaluate_all_subsets(
    averaged_measured_by_label: Mapping[str, Sequence[float]],
    truth_robot_by_label: Mapping[str, Sequence[float]],
    *,
    sizes: Iterable[int] = (4, 5, 6, 7, 8),
) -> list[SubsetSolveResult]:
    """Solve registration for every label subset of the requested sizes.

    Each subset is fit with the standard Kabsch solve; nothing about the math
    changes between subsets. The point is to surface which subset of the
    captured landmarks happens to fit best, and at what size the residual stops
    improving — a measurement-driven answer to "do I need 4, 5, or 8 points?".
    """
    shared_labels = sorted(set(averaged_measured_by_label.keys()) & set(truth_robot_by_label.keys()))
    out: list[SubsetSolveResult] = []
    for size in sorted({int(s) for s in sizes}):
        if size < 3 or size > len(shared_labels):
            continue
        for combo in combinations(shared_labels, size):
            subset_measured = {label: averaged_measured_by_label[label] for label in combo}
            subset_truth = {label: truth_robot_by_label[label] for label in combo}
            solved = solve_with_metrics(subset_measured, subset_truth)
            geom = solved.get("geometry") if isinstance(solved, dict) else None
            geom_rank = geom.get("geometry_rank") if isinstance(geom, dict) else None
            geom_cond = geom.get("condition_number") if isinstance(geom, dict) else None
            out.append(
                SubsetSolveResult(
                    labels=tuple(combo),
                    size=int(size),
                    fre_mm=float(solved["fre_mm"]),
                    max_residual_mm=float(solved["max_residual_mm"]),
                    worst_landmark_label=solved["worst_landmark_label"],
                    per_label_residual_mm=dict(solved["residual_norms_mm_by_label"]),
                    geometry_rank=int(geom_rank) if geom_rank is not None else None,
                    geometry_condition_number=(
                        float(geom_cond) if geom_cond is not None else None
                    ),
                )
            )
    return out


def summarize_subset_search(
    subset_results: Sequence[SubsetSolveResult],
) -> dict[str, object]:
    """Reduce the per-subset list to a per-size best-of and a global best."""
    by_size: dict[int, list[SubsetSolveResult]] = {}
    for item in subset_results:
        by_size.setdefault(item.size, []).append(item)
    per_size_best: dict[int, dict[str, object]] = {}
    for size, items in sorted(by_size.items()):
        best = min(items, key=lambda result: result.fre_mm)
        per_size_best[size] = {
            "size": int(size),
            "subset_count": len(items),
            "best_labels": list(best.labels),
            "best_fre_mm": float(best.fre_mm),
            "best_max_residual_mm": float(best.max_residual_mm),
            "best_worst_landmark_label": best.worst_landmark_label,
            "best_geometry_rank": best.geometry_rank,
            "best_geometry_condition_number": best.geometry_condition_number,
        }
    global_best: dict[str, object] | None = None
    if subset_results:
        best = min(subset_results, key=lambda result: result.fre_mm)
        global_best = {
            "size": int(best.size),
            "labels": list(best.labels),
            "fre_mm": float(best.fre_mm),
            "max_residual_mm": float(best.max_residual_mm),
            "worst_landmark_label": best.worst_landmark_label,
            "geometry_rank": best.geometry_rank,
            "geometry_condition_number": best.geometry_condition_number,
        }
    return {
        "per_size_best": per_size_best,
        "global_best": global_best,
        "total_subsets_evaluated": len(subset_results),
    }


def summarize_trial(results: Mapping[str, RegistrationTrialResult]) -> dict[str, object]:
    """Reduce a method-sweep to a small human-reviewable comparison record."""
    rows = []
    for method, result in results.items():
        rows.append(
            {
                "method": method,
                "fre_mm": result.fre_mm,
                "max_residual_mm": result.max_residual_mm,
                "worst_landmark_label": result.worst_landmark_label,
                "n_input_per_label": {
                    label: avg.n_input for label, avg in result.averaging.items()
                },
                "n_kept_per_label": {
                    label: avg.n_kept for label, avg in result.averaging.items()
                },
                "intra_capture_stddev_mm_by_label": {
                    label: avg.intra_capture_stddev_mm for label, avg in result.averaging.items()
                },
                "loo_max_minus_keep_mm": result.loo_max_minus_keep_mm,
                "method_params": result.method_params,
            }
        )
    best = min(results.values(), key=lambda r: r.fre_mm) if results else None
    return {
        "method_rows": rows,
        "best_method": best.method if best else None,
        "best_fre_mm": float(best.fre_mm) if best else None,
        "best_max_residual_mm": float(best.max_residual_mm) if best else None,
        "best_loo_max_minus_keep_mm": float(best.loo_max_minus_keep_mm) if best else None,
    }


# ---------------------------------------------------------------------------
# Samples-per-point study
# ---------------------------------------------------------------------------
#
# The method sweep and the subset search both use the *full* captured pool
# at each landmark. That answers "which method and which subset are best?"
# but not "did I capture enough samples per landmark?" — the operator's
# real question on the bench.
#
# The samples-per-point study draws random subsamples of size k from the
# captured pool (with k = 1, 3, 5, ..., up to K) and solves the full-N-point
# registration using mean averaging for each draw. Bootstrap iterations
# smooth out the random choice of which k were drawn. The output curve
# (FRE vs k) lets the operator pick the smallest k that lands within an
# operator-tunable epsilon of the best FRE the dataset can produce — i.e.
# diminishing-returns answered from the actual data, not intuition.
#
# Math: same Kabsch solve as everywhere else. Mean averaging is fixed to
# keep this axis honest; comparing across averaging methods is the job of
# `sweep_methods()`, and stacking the two knobs makes both noisier.

DEFAULT_SAMPLES_PER_POINT_LADDER: tuple[int, ...] = (1, 3, 5, 10, 20, 30, 50)
"""Default k values to evaluate. The actual ladder is capped at the dataset's K."""


def samples_per_point_study(
    captures_by_label: Mapping[str, Sequence[Sequence[float]]],
    truth_robot_by_label: Mapping[str, Sequence[float]],
    *,
    k_values: Iterable[int] = DEFAULT_SAMPLES_PER_POINT_LADDER,
    bootstrap_iterations: int = 40,
    random_seed: int = 7,
) -> list[dict[str, object]]:
    """Return per-(k, iteration) FRE rows for a samples-per-point sweep.

    For each k in ``k_values`` (capped to the minimum pool size across all
    labels), draw ``bootstrap_iterations`` random k-subsets from each label,
    take the mean of each subset, solve the full-N-point registration, and
    record FRE. Returns one row per (k, iteration). Aggregation is the
    caller's job — see :func:`aggregate_samples_per_point`.
    """
    shared = sorted(set(captures_by_label.keys()) & set(truth_robot_by_label.keys()))
    if len(shared) < 3:
        return []
    pools = {label: np.asarray(captures_by_label[label], dtype=float) for label in shared}
    truth_subset = {label: list(truth_robot_by_label[label]) for label in shared}
    min_pool = min(int(pool.shape[0]) for pool in pools.values())
    if min_pool < 1:
        return []
    rng = np.random.default_rng(int(random_seed))
    rows: list[dict[str, object]] = []
    iterations = max(1, int(bootstrap_iterations))
    for k_raw in sorted({int(v) for v in k_values}):
        k = int(min(k_raw, min_pool))
        if k < 1:
            continue
        for iteration in range(iterations):
            averaged: dict[str, list[float]] = {}
            for label in shared:
                pool = pools[label]
                if pool.shape[0] == k:
                    draw = pool
                else:
                    idx = rng.choice(pool.shape[0], size=k, replace=False)
                    draw = pool[idx]
                averaged[label] = [float(np.mean(draw[:, axis])) for axis in range(3)]
            solved = solve_with_metrics(averaged, truth_subset)
            rows.append(
                {
                    "k": int(k),
                    "k_requested": int(k_raw),
                    "iteration": int(iteration),
                    "fre_mm": float(solved["fre_mm"]),
                    "max_residual_mm": float(solved["max_residual_mm"]),
                }
            )
        if k == min_pool:
            # Drawing without replacement at k=min_pool returns the whole pool every
            # iteration, so further k values would be capped to the same draw. Stop
            # to avoid pointless duplicate rows.
            break
    return rows


def aggregate_samples_per_point(rows: Sequence[Mapping[str, object]]) -> list[dict[str, object]]:
    """Reduce per-iteration rows to per-k summary statistics."""
    by_k: dict[int, list[float]] = {}
    for row in rows:
        fre = row.get("fre_mm")
        if fre is None:
            continue
        by_k.setdefault(int(row["k"]), []).append(float(fre))
    out: list[dict[str, object]] = []
    for k, values in sorted(by_k.items()):
        arr = np.asarray(values, dtype=float)
        out.append(
            {
                "k": int(k),
                "iteration_count": int(arr.size),
                "fre_mean_mm": float(np.mean(arr)) if arr.size else None,
                "fre_std_mm": float(np.std(arr, ddof=0)) if arr.size else None,
                "fre_min_mm": float(np.min(arr)) if arr.size else None,
                "fre_max_mm": float(np.max(arr)) if arr.size else None,
                "fre_p95_mm": float(np.percentile(arr, 95)) if arr.size else None,
            }
        )
    return out


def recommend_samples_per_point(
    aggregated: Sequence[Mapping[str, object]],
    *,
    epsilon_mm: float = 0.02,
) -> dict[str, object]:
    """Pick the smallest k whose mean FRE is within ``epsilon_mm`` of the best.

    Returns the recommended k, the FRE at that k, the best FRE in the ladder,
    and a one-line plain-English rationale.
    """
    eligible = [row for row in aggregated if row.get("fre_mean_mm") is not None]
    if not eligible:
        return {
            "recommended_k": None,
            "recommended_fre_mean_mm": None,
            "best_fre_mean_mm": None,
            "rationale": "Insufficient samples-per-point data to make a recommendation.",
        }
    best_fre = float(min(float(row["fre_mean_mm"]) for row in eligible))
    threshold = best_fre + float(max(0.0, epsilon_mm))
    recommended = None
    for row in sorted(eligible, key=lambda r: int(r["k"])):
        if float(row["fre_mean_mm"]) <= threshold:
            recommended = int(row["k"])
            recommended_fre = float(row["fre_mean_mm"])
            break
    if recommended is None:
        last = max(eligible, key=lambda r: int(r["k"]))
        recommended = int(last["k"])
        recommended_fre = float(last["fre_mean_mm"])
    return {
        "recommended_k": int(recommended),
        "recommended_fre_mean_mm": float(recommended_fre),
        "best_fre_mean_mm": best_fre,
        "epsilon_mm": float(epsilon_mm),
        "rationale": (
            f"FRE bottoms out at {best_fre:.4f} mm in the captured ladder; "
            f"k={recommended} reaches within {epsilon_mm:.3f} mm of that "
            f"(actual={recommended_fre:.4f} mm). Capturing more samples per "
            "landmark beyond that point does not measurably improve FRE on this data."
        ),
    }

"""Offline analysis + writers for `registration_sampling_study` runs.

All math is a thin layer on top of `RigidRegistrationSolver`. No registration
math is changed.

Outputs produced inside the run directory:
- `metrics.csv` — per-(subset_size, averaging_method) bootstrap summary.
- `point_centers.csv` — per-label center under each averaging method, plus spread.
- `subset_results.csv` — every bootstrap subset's solve result.
- `leave_one_out_results.csv` — per-label leave-one-out residuals.
- `samples_per_point_results.csv` — FRE as a function of samples-per-point.
- `raw_point_samples.jsonl` — raw aurora captures per label (one row per sample).
- `registration_sampling_study_summary.txt` — operator-readable summary.
- `registration_point_spread_report.png`
- `registration_subset_rms_report.png`
- `registration_samples_per_point_report.png`
- `registration_transform_consistency_report.png`
- `registration_candidate.json` — best full-subset registration; never auto-promoted.
- `summary.json` is written by the runner; this module only fills in
  `experiment_metrics` via the canonical session-metric path.
"""

from __future__ import annotations

import csv
import json
import math
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from continuum_robot.experiments.plotting import (
    add_metric_box,
    color,
    create_figure,
    legend,
    save_figure,
    style_axes,
)
from continuum_robot.experiments.schemas import ExperimentTimeseriesSample
from continuum_robot.registration.rigid_solver import RigidRegistrationSolver


REGISTRATION_CANDIDATE_FILENAME = "registration_candidate.json"
SUMMARY_TEXT_FILENAME = "registration_sampling_study_summary.txt"
RAW_SAMPLES_FILENAME = "raw_point_samples.jsonl"
POINT_CENTERS_FILENAME = "point_centers.csv"
SUBSET_RESULTS_FILENAME = "subset_results.csv"
LEAVE_ONE_OUT_FILENAME = "leave_one_out_results.csv"
METRICS_FILENAME = "metrics.csv"
SAMPLES_PER_POINT_FILENAME = "samples_per_point_results.csv"

# Sample-counts evaluated for the "how many samples per point are enough" study.
SAMPLES_PER_POINT_LADDER: tuple[int, ...] = (1, 3, 5, 10, 20, 50)

REPORT_FIGURES: tuple[str, ...] = (
    "registration_point_spread_report.png",
    "registration_subset_rms_report.png",
    "registration_samples_per_point_report.png",
    "registration_transform_consistency_report.png",
)


# ---------------------------------------------------------------------------
# Pure analysis (no I/O, no plotting). These functions are unit-tested
# directly so the math is honest and reproducible.
# ---------------------------------------------------------------------------


def group_samples_by_label(
    samples: list[ExperimentTimeseriesSample],
    labels: list[str],
) -> dict[str, list[list[float]]]:
    """Return `{label: [[ax,ay,az], ...]}` aurora captures for each label.

    Samples without an aurora capture (rejected or missing translation) are
    silently skipped. Captures retain the order they were recorded in.
    """
    pool: dict[str, list[list[float]]] = {label: [] for label in labels}
    for sample in samples:
        record = (sample.extra or {}).get("registration_sample")
        if not isinstance(record, dict):
            continue
        if not bool((sample.extra or {}).get("capture_accepted", record.get("capture_accepted", True))):
            continue
        label = str(record.get("label", "")).strip()
        if label not in pool:
            continue
        aurora_xyz = record.get("aurora_xyz_mm")
        if aurora_xyz is None or len(list(aurora_xyz)) < 3:
            continue
        pool[label].append([float(value) for value in list(aurora_xyz)[:3]])
    return pool


def compute_point_centers(
    samples_by_label: dict[str, list[list[float]]],
    *,
    methods: list[str],
    trimmed_mean_proportion: float,
) -> dict[str, dict[str, Any]]:
    """Per-label center under each averaging method, plus spread metrics."""
    out: dict[str, dict[str, Any]] = {}
    for label, samples in samples_by_label.items():
        arr = np.asarray(samples, dtype=float) if samples else np.empty((0, 3))
        if arr.size == 0:
            out[label] = {
                "sample_count": 0,
                "centers_by_method": {method: None for method in methods},
                "within_point_rms_mm": None,
                "within_point_max_mm": None,
                "within_point_std_mm": [None, None, None],
                "max_z_residual": None,
            }
            continue
        centers = {}
        for method in methods:
            centers[method] = _averaging_center(arr, method, trimmed_mean_proportion)
        # Use the "mean" center as the reference for spread reporting.
        mean_center = centers.get("mean") or [float(np.mean(arr[:, k])) for k in range(3)]
        deltas = arr - np.asarray(mean_center, dtype=float)
        distances = np.linalg.norm(deltas, axis=1)
        rms = float(np.sqrt(np.mean(np.square(distances)))) if distances.size else None
        std_per_axis = [float(np.std(arr[:, k], ddof=0)) for k in range(3)]
        max_within_point_mm = float(np.max(distances)) if distances.size else None
        # z-score per sample (using the largest axis std as a conservative denominator)
        denom = max(float(np.max(std_per_axis)) or 0.0, 1e-9)
        max_z = float(np.max(distances) / denom) if distances.size else None
        out[label] = {
            "sample_count": int(arr.shape[0]),
            "centers_by_method": centers,
            "within_point_rms_mm": rms,
            "within_point_max_mm": max_within_point_mm,
            "within_point_std_mm": std_per_axis,
            "max_z_residual": max_z,
        }
    return out


def _averaging_center(arr: np.ndarray, method: str, trimmed_proportion: float) -> list[float]:
    if arr.size == 0:
        return [float("nan"), float("nan"), float("nan")]
    if method == "mean":
        return [float(np.mean(arr[:, k])) for k in range(3)]
    if method == "median":
        return [float(np.median(arr[:, k])) for k in range(3)]
    if method == "trimmed_mean":
        proportion = float(max(0.0, min(0.49, trimmed_proportion)))
        n = arr.shape[0]
        k = int(math.floor(proportion * n))
        if n - 2 * k <= 0:
            return [float(np.median(arr[:, idx])) for idx in range(3)]
        trimmed = [
            float(np.mean(np.sort(arr[:, idx])[k : n - k])) if n - 2 * k > 0 else float(np.median(arr[:, idx]))
            for idx in range(3)
        ]
        return trimmed
    # Unknown method falls back to mean.
    return [float(np.mean(arr[:, k])) for k in range(3)]


def solve_registration_for_subset(
    labels_subset: list[str],
    *,
    centers_by_label: dict[str, list[float]],
    truth_by_label: dict[str, list[float]],
) -> dict[str, Any]:
    """Solve `T_robot_aurora` from the given subset of labels and return metrics."""
    aurora = np.asarray(
        [centers_by_label[label] for label in labels_subset if label in centers_by_label],
        dtype=float,
    )
    truth = np.asarray(
        [truth_by_label[label] for label in labels_subset if label in truth_by_label],
        dtype=float,
    )
    if aurora.shape[0] < 3 or aurora.shape != truth.shape:
        return {
            "labels": list(labels_subset),
            "T_robot_aurora": None,
            "fre_mm": None,
            "residuals_mm": [],
            "max_residual_mm": None,
        }
    solver = RigidRegistrationSolver()
    result = solver.solve_alignment(aurora, truth)
    residuals = np.asarray(result["residuals"]).T if "residuals" in result else np.zeros((aurora.shape[0], 3))
    if residuals.ndim == 1:
        residuals = residuals.reshape(-1, 3)
    # `residuals` from solve_alignment is returned shape (3, N) per its docstring; normalize to (N, 3)
    if residuals.shape[0] == 3 and residuals.shape[1] != 3:
        residuals = residuals.T
    elif residuals.shape[1] == 3 and residuals.shape[0] != aurora.shape[0]:
        # already (N,3) — keep
        pass
    per_point = np.linalg.norm(residuals, axis=1) if residuals.size else np.zeros(0)
    return {
        "labels": list(labels_subset),
        "T_robot_aurora": np.asarray(result["transform"], dtype=float),
        "fre_mm": float(result.get("rmse_mm")) if result.get("rmse_mm") is not None else None,
        "residuals_mm": [float(value) for value in per_point.tolist()],
        "max_residual_mm": float(np.max(per_point)) if per_point.size else None,
    }


def bootstrap_subset_solves(
    *,
    labels: list[str],
    centers_by_label: dict[str, list[float]],
    truth_by_label: dict[str, list[float]],
    subset_sizes: list[int],
    bootstrap_iterations: int,
    rng: np.random.Generator,
) -> list[dict[str, Any]]:
    """For each subset size, sample subsets, solve, and record FRE / leave-one-out."""
    rows: list[dict[str, Any]] = []
    eligible_labels = [label for label in labels if label in centers_by_label and label in truth_by_label]
    for size in subset_sizes:
        if size > len(eligible_labels) or size < 3:
            continue
        # If the bootstrap count is large enough to enumerate all C(n,size), enumerate.
        n = len(eligible_labels)
        total_choose = math.comb(n, size)
        if bootstrap_iterations >= total_choose:
            subsets: Iterable[tuple[str, ...]] = list(combinations(eligible_labels, size))
        else:
            seen: set[tuple[str, ...]] = set()
            subsets_list: list[tuple[str, ...]] = []
            attempts = 0
            max_attempts = max(bootstrap_iterations * 5, bootstrap_iterations + 50)
            while len(subsets_list) < bootstrap_iterations and attempts < max_attempts:
                chosen = tuple(sorted(rng.choice(eligible_labels, size=size, replace=False).tolist()))
                if chosen not in seen:
                    seen.add(chosen)
                    subsets_list.append(chosen)
                attempts += 1
            subsets = subsets_list
        for subset in subsets:
            res = solve_registration_for_subset(
                list(subset),
                centers_by_label=centers_by_label,
                truth_by_label=truth_by_label,
            )
            loo = leave_one_out_residuals(
                list(subset),
                centers_by_label=centers_by_label,
                truth_by_label=truth_by_label,
            )
            rows.append(
                {
                    "subset_size": int(size),
                    "labels": list(subset),
                    "fre_mm": res.get("fre_mm"),
                    "max_residual_mm": res.get("max_residual_mm"),
                    "leave_one_out_rms_mm": loo.get("rms_mm"),
                    "leave_one_out_max_mm": loo.get("max_mm"),
                    "leave_one_out_per_label_mm": loo.get("per_label_mm", {}),
                }
            )
    return rows


def leave_one_out_residuals(
    labels_subset: list[str],
    *,
    centers_by_label: dict[str, list[float]],
    truth_by_label: dict[str, list[float]],
) -> dict[str, Any]:
    """For each label in `labels_subset`, leave it out, solve on the rest, and
    compute the predicted-vs-truth error at the left-out label."""
    per_label: dict[str, float] = {}
    if len(labels_subset) < 4:
        return {"rms_mm": None, "max_mm": None, "per_label_mm": per_label}
    solver = RigidRegistrationSolver()
    for label in labels_subset:
        remaining = [other for other in labels_subset if other != label]
        if len(remaining) < 3:
            continue
        try:
            aurora = np.asarray([centers_by_label[other] for other in remaining], dtype=float)
            truth = np.asarray([truth_by_label[other] for other in remaining], dtype=float)
            T = solver.solve_T_robot_aurora(aurora, truth)
        except Exception:
            continue
        predicted = (T[0:3, 0:3] @ np.asarray(centers_by_label[label], dtype=float)) + T[0:3, 3]
        err = float(np.linalg.norm(predicted - np.asarray(truth_by_label[label], dtype=float)))
        per_label[label] = err
    if not per_label:
        return {"rms_mm": None, "max_mm": None, "per_label_mm": per_label}
    values = np.asarray(list(per_label.values()), dtype=float)
    return {
        "rms_mm": float(np.sqrt(np.mean(np.square(values)))),
        "max_mm": float(np.max(values)),
        "per_label_mm": {label: float(value) for label, value in per_label.items()},
    }


def samples_per_point_study(
    *,
    samples_by_label: dict[str, list[list[float]]],
    truth_by_label: dict[str, list[float]],
    sample_counts: Iterable[int],
    bootstrap_iterations: int,
    rng: np.random.Generator,
) -> list[dict[str, Any]]:
    """For each sample-count k in `sample_counts`, draw k samples per label,
    average them, solve the full-N-points registration, and record FRE.

    Returns a list of rows, one per (k, bootstrap_iteration) pair.
    """
    labels = [label for label in samples_by_label.keys() if label in truth_by_label]
    rows: list[dict[str, Any]] = []
    for k in sample_counts:
        k = int(k)
        if k < 1:
            continue
        for iteration in range(max(1, int(bootstrap_iterations))):
            centers: dict[str, list[float]] = {}
            usable_labels: list[str] = []
            for label in labels:
                pool = np.asarray(samples_by_label[label], dtype=float)
                if pool.shape[0] == 0:
                    continue
                k_eff = min(k, pool.shape[0])
                idx = rng.choice(pool.shape[0], size=k_eff, replace=False)
                draw = pool[idx]
                centers[label] = [float(np.mean(draw[:, axis])) for axis in range(3)]
                usable_labels.append(label)
            if len(usable_labels) < 3:
                continue
            res = solve_registration_for_subset(
                usable_labels,
                centers_by_label=centers,
                truth_by_label=truth_by_label,
            )
            rows.append(
                {
                    "samples_per_point": int(k),
                    "iteration": int(iteration),
                    "fre_mm": res.get("fre_mm"),
                    "max_residual_mm": res.get("max_residual_mm"),
                    "label_count": int(len(usable_labels)),
                }
            )
    return rows


def aggregate_subset_metrics(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Aggregate bootstrap subset rows into per-subset-size summary stats."""
    by_size: dict[int, list[dict[str, Any]]] = {}
    for row in rows:
        by_size.setdefault(int(row["subset_size"]), []).append(row)
    out: list[dict[str, Any]] = []
    for size, group in sorted(by_size.items()):
        fre = np.asarray([row["fre_mm"] for row in group if row.get("fre_mm") is not None], dtype=float)
        loo = np.asarray([row["leave_one_out_rms_mm"] for row in group if row.get("leave_one_out_rms_mm") is not None], dtype=float)
        out.append(
            {
                "subset_size": int(size),
                "trial_count": int(len(group)),
                "fre_mean_mm": float(np.mean(fre)) if fre.size else None,
                "fre_std_mm": float(np.std(fre, ddof=0)) if fre.size else None,
                "fre_min_mm": float(np.min(fre)) if fre.size else None,
                "fre_max_mm": float(np.max(fre)) if fre.size else None,
                "fre_p95_mm": float(np.percentile(fre, 95)) if fre.size else None,
                "leave_one_out_mean_mm": float(np.mean(loo)) if loo.size else None,
                "leave_one_out_max_mm": float(np.max(loo)) if loo.size else None,
            }
        )
    return out


def aggregate_samples_per_point(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_k: dict[int, list[float]] = {}
    for row in rows:
        if row.get("fre_mm") is None:
            continue
        by_k.setdefault(int(row["samples_per_point"]), []).append(float(row["fre_mm"]))
    out: list[dict[str, Any]] = []
    for k, fres in sorted(by_k.items()):
        arr = np.asarray(fres, dtype=float)
        out.append(
            {
                "samples_per_point": int(k),
                "trial_count": int(arr.size),
                "fre_mean_mm": float(np.mean(arr)) if arr.size else None,
                "fre_std_mm": float(np.std(arr, ddof=0)) if arr.size else None,
                "fre_p95_mm": float(np.percentile(arr, 95)) if arr.size else None,
            }
        )
    return out


def flag_outlier_labels(
    point_centers: dict[str, dict[str, Any]],
    *,
    z_threshold: float,
) -> list[dict[str, Any]]:
    """Flag labels whose within-point spread is unusual relative to the
    other labels. This never deletes samples — it only labels them.
    """
    rms_values = [
        float(meta["within_point_rms_mm"])
        for meta in point_centers.values()
        if meta.get("within_point_rms_mm") is not None
    ]
    if len(rms_values) < 3:
        return []
    median = float(np.median(rms_values))
    mad = float(np.median(np.abs(np.asarray(rms_values) - median))) or 1e-9
    flagged: list[dict[str, Any]] = []
    for label, meta in point_centers.items():
        rms = meta.get("within_point_rms_mm")
        if rms is None:
            continue
        # Modified z-score using MAD.
        z = float(0.6745 * (rms - median) / mad)
        if z >= float(z_threshold):
            flagged.append(
                {
                    "label": label,
                    "within_point_rms_mm": float(rms),
                    "modified_z_score": z,
                }
            )
    flagged.sort(key=lambda entry: -float(entry["modified_z_score"]))
    return flagged


def recommend_protocol(
    *,
    subset_summary: list[dict[str, Any]],
    samples_per_point_summary: list[dict[str, Any]],
    optimize_for: str,
) -> dict[str, Any]:
    """Decide a recommended protocol and a one-line rationale."""
    if not subset_summary:
        return {
            "recommended_subset_size": None,
            "recommended_samples_per_point": None,
            "recommended_averaging_method": "mean",
            "rationale": "Insufficient bootstrap data to recommend a protocol.",
        }
    key = "fre_mean_mm" if optimize_for == "fre_mm" else (
        "leave_one_out_mean_mm" if optimize_for == "leave_one_out_mm" else "fre_mean_mm"
    )
    eligible = [row for row in subset_summary if row.get(key) is not None]
    if not eligible:
        return {
            "recommended_subset_size": None,
            "recommended_samples_per_point": None,
            "recommended_averaging_method": "mean",
            "rationale": f"Optimization key {key!r} unavailable; no subset has a usable score.",
        }
    best_subset = min(eligible, key=lambda row: float(row[key]))
    # For samples-per-point: pick the smallest k whose mean FRE is within
    # 5% of the lowest mean FRE in the ladder (diminishing-returns rule).
    eligible_k = [row for row in samples_per_point_summary if row.get("fre_mean_mm") is not None]
    recommended_k: int | None = None
    if eligible_k:
        min_fre = float(min(row["fre_mean_mm"] for row in eligible_k))
        threshold = float(min_fre) * 1.05 if min_fre > 0 else 0.0
        for row in sorted(eligible_k, key=lambda row: row["samples_per_point"]):
            if float(row["fre_mean_mm"]) <= threshold or float(row["fre_mean_mm"]) <= min_fre + 0.05:
                recommended_k = int(row["samples_per_point"])
                break
        if recommended_k is None:
            recommended_k = int(eligible_k[-1]["samples_per_point"])
    rationale_parts = [
        f"Optimized for {optimize_for}.",
        f"Best subset size = {best_subset['subset_size']} ({key}={best_subset[key]:.3f} mm).",
    ]
    if recommended_k is not None:
        rationale_parts.append(
            f"Samples/point reaches diminishing returns at k={recommended_k}."
        )
    return {
        "recommended_subset_size": int(best_subset["subset_size"]),
        "recommended_samples_per_point": recommended_k,
        "recommended_averaging_method": "mean",  # The bootstrap above uses mean; method study is in point_centers.
        "rationale": " ".join(rationale_parts),
    }


# ---------------------------------------------------------------------------
# Writers (file I/O + plotting). Keep separate from analysis so the analysis
# can be unit-tested headlessly.
# ---------------------------------------------------------------------------


def write_registration_sampling_study_outputs(
    *,
    output_dir: Path,
    metadata,
    summary,
    samples: list[ExperimentTimeseriesSample],
    labels: list[str],
    truth_points_by_label: dict[str, list[float]],
    subset_sizes: list[int],
    averaging_methods: list[str],
    trimmed_mean_proportion: float,
    bootstrap_iterations: int,
    random_seed: int,
    outlier_flag_z_threshold: float,
    optimize_for: str,
) -> dict[str, Any]:
    """Top-level writer. Returns the recommended-protocol payload."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    samples_by_label = group_samples_by_label(samples, labels)
    point_centers = compute_point_centers(
        samples_by_label,
        methods=list(averaging_methods),
        trimmed_mean_proportion=float(trimmed_mean_proportion),
    )
    rng = np.random.default_rng(int(random_seed))
    # Use "mean" centers for the subset bootstrap; the per-method centers are
    # already reported in point_centers.csv. This keeps the subset study
    # focused on the size/sample-count axis.
    centers_for_subsets = {
        label: meta["centers_by_method"]["mean"]
        for label, meta in point_centers.items()
        if meta.get("centers_by_method", {}).get("mean") is not None
    }
    subset_rows = bootstrap_subset_solves(
        labels=labels,
        centers_by_label=centers_for_subsets,
        truth_by_label=truth_points_by_label,
        subset_sizes=list(subset_sizes),
        bootstrap_iterations=int(bootstrap_iterations),
        rng=rng,
    )
    subset_summary = aggregate_subset_metrics(subset_rows)
    spp_rows = samples_per_point_study(
        samples_by_label=samples_by_label,
        truth_by_label=truth_points_by_label,
        sample_counts=SAMPLES_PER_POINT_LADDER,
        bootstrap_iterations=max(20, int(bootstrap_iterations) // 4),
        rng=rng,
    )
    spp_summary = aggregate_samples_per_point(spp_rows)
    flagged = flag_outlier_labels(point_centers, z_threshold=outlier_flag_z_threshold)
    recommendation = recommend_protocol(
        subset_summary=subset_summary,
        samples_per_point_summary=spp_summary,
        optimize_for=optimize_for,
    )
    # Candidate registration: use all available labels with mean centers.
    full_subset_labels = sorted(centers_for_subsets.keys(), key=lambda label: labels.index(label) if label in labels else 0)
    candidate = solve_registration_for_subset(
        full_subset_labels,
        centers_by_label=centers_for_subsets,
        truth_by_label=truth_points_by_label,
    )
    _write_raw_samples_jsonl(output_dir / RAW_SAMPLES_FILENAME, samples_by_label, truth_points_by_label)
    _write_point_centers_csv(output_dir / POINT_CENTERS_FILENAME, point_centers, averaging_methods)
    _write_subset_results_csv(output_dir / SUBSET_RESULTS_FILENAME, subset_rows)
    _write_leave_one_out_csv(output_dir / LEAVE_ONE_OUT_FILENAME, subset_rows)
    _write_metrics_csv(output_dir / METRICS_FILENAME, subset_summary, spp_summary)
    _write_samples_per_point_csv(output_dir / SAMPLES_PER_POINT_FILENAME, spp_rows)
    _write_summary_text(
        output_dir / SUMMARY_TEXT_FILENAME,
        metadata=metadata,
        labels=labels,
        truth_points_by_label=truth_points_by_label,
        point_centers=point_centers,
        subset_summary=subset_summary,
        spp_summary=spp_summary,
        flagged=flagged,
        recommendation=recommendation,
        candidate=candidate,
        optimize_for=optimize_for,
        averaging_methods=averaging_methods,
    )
    _write_registration_candidate_json(
        output_dir / REGISTRATION_CANDIDATE_FILENAME,
        candidate=candidate,
        labels=full_subset_labels,
        centers=centers_for_subsets,
        truth_points_by_label=truth_points_by_label,
        samples_by_label=samples_by_label,
        metadata=metadata,
    )
    # Plots
    try:
        _plot_point_spread(
            output_dir / "registration_point_spread_report.png",
            point_centers=point_centers,
            labels=labels,
        )
        _plot_subset_rms(
            output_dir / "registration_subset_rms_report.png",
            subset_summary=subset_summary,
        )
        _plot_samples_per_point(
            output_dir / "registration_samples_per_point_report.png",
            samples_per_point_summary=spp_summary,
        )
        _plot_transform_consistency(
            output_dir / "registration_transform_consistency_report.png",
            subset_rows=subset_rows,
        )
    except Exception:
        # Headless or matplotlib-broken environments: still produce CSVs/text.
        pass
    # Surface key metrics into the summary that the runner writes.
    metrics = dict(getattr(summary, "experiment_metrics", {}) or {})
    metrics.update(
        {
            "registration_sampling_study_schema_version": "1.0",
            "captured_label_count": int(len(samples_by_label)),
            "captured_sample_count_total": int(sum(len(v) for v in samples_by_label.values())),
            "subset_summary": subset_summary,
            "samples_per_point_summary": spp_summary,
            "flagged_outlier_labels": flagged,
            "recommended_protocol": recommendation,
            "candidate_registration_fre_mm": candidate.get("fre_mm"),
            "candidate_registration_max_residual_mm": candidate.get("max_residual_mm"),
            "candidate_registration_label_count": int(len(full_subset_labels)),
            "report_figures": list(REPORT_FIGURES),
            "valid_for_model_training": False,
            "valid_for_thesis_repeatability": False,
            "valid_for_registration_protocol_recommendation": bool(
                recommendation.get("recommended_subset_size") is not None
            ),
        }
    )
    if hasattr(summary, "experiment_metrics") and isinstance(summary.experiment_metrics, dict):
        summary.experiment_metrics.update(metrics)
    return recommendation


# ---------------------------------------------------------------------------
# Individual file writers
# ---------------------------------------------------------------------------


def _write_raw_samples_jsonl(
    path: Path,
    samples_by_label: dict[str, list[list[float]]],
    truth_by_label: dict[str, list[float]],
) -> None:
    with path.open("w", encoding="utf-8") as fh:
        for label, samples in samples_by_label.items():
            for index, xyz in enumerate(samples):
                fh.write(
                    json.dumps(
                        {
                            "label": label,
                            "sample_index": int(index),
                            "aurora_xyz_mm": [float(value) for value in xyz],
                            "truth_body_xyz_mm": [float(value) for value in truth_by_label.get(label, [])],
                        },
                        sort_keys=True,
                    )
                )
                fh.write("\n")


def _write_point_centers_csv(
    path: Path,
    point_centers: dict[str, dict[str, Any]],
    averaging_methods: list[str],
) -> None:
    header = ["label", "sample_count", "within_point_rms_mm", "within_point_max_mm"]
    for method in averaging_methods:
        for axis in ("x", "y", "z"):
            header.append(f"center_{method}_{axis}_mm")
    for axis in ("x", "y", "z"):
        header.append(f"within_point_std_{axis}_mm")
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(header)
        for label, meta in point_centers.items():
            row: list[Any] = [label, meta.get("sample_count", 0), meta.get("within_point_rms_mm"), meta.get("within_point_max_mm")]
            centers = meta.get("centers_by_method", {})
            for method in averaging_methods:
                xyz = centers.get(method) or [None, None, None]
                row.extend(xyz)
            stds = meta.get("within_point_std_mm") or [None, None, None]
            row.extend(stds)
            writer.writerow(row)


def _write_subset_results_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    header = ["subset_size", "labels", "fre_mm", "max_residual_mm", "leave_one_out_rms_mm", "leave_one_out_max_mm"]
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(header)
        for row in rows:
            writer.writerow(
                [
                    int(row.get("subset_size", 0)),
                    "|".join(row.get("labels", [])),
                    row.get("fre_mm"),
                    row.get("max_residual_mm"),
                    row.get("leave_one_out_rms_mm"),
                    row.get("leave_one_out_max_mm"),
                ]
            )


def _write_leave_one_out_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["subset_size", "labels", "left_out_label", "residual_mm"])
        for row in rows:
            per_label = dict(row.get("leave_one_out_per_label_mm", {}) or {})
            for left_out_label, residual in per_label.items():
                writer.writerow(
                    [
                        int(row.get("subset_size", 0)),
                        "|".join(row.get("labels", [])),
                        left_out_label,
                        float(residual),
                    ]
                )


def _write_metrics_csv(
    path: Path,
    subset_summary: list[dict[str, Any]],
    spp_summary: list[dict[str, Any]],
) -> None:
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["section", "x", "trial_count", "fre_mean_mm", "fre_std_mm", "fre_min_mm", "fre_max_mm", "fre_p95_mm", "leave_one_out_mean_mm", "leave_one_out_max_mm"])
        for row in subset_summary:
            writer.writerow(
                [
                    "subset_size",
                    int(row.get("subset_size", 0)),
                    int(row.get("trial_count", 0)),
                    row.get("fre_mean_mm"),
                    row.get("fre_std_mm"),
                    row.get("fre_min_mm"),
                    row.get("fre_max_mm"),
                    row.get("fre_p95_mm"),
                    row.get("leave_one_out_mean_mm"),
                    row.get("leave_one_out_max_mm"),
                ]
            )
        for row in spp_summary:
            writer.writerow(
                [
                    "samples_per_point",
                    int(row.get("samples_per_point", 0)),
                    int(row.get("trial_count", 0)),
                    row.get("fre_mean_mm"),
                    row.get("fre_std_mm"),
                    None,
                    None,
                    row.get("fre_p95_mm"),
                    None,
                    None,
                ]
            )


def _write_samples_per_point_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["samples_per_point", "iteration", "label_count", "fre_mm", "max_residual_mm"])
        for row in rows:
            writer.writerow(
                [
                    int(row.get("samples_per_point", 0)),
                    int(row.get("iteration", 0)),
                    int(row.get("label_count", 0)),
                    row.get("fre_mm"),
                    row.get("max_residual_mm"),
                ]
            )


def _write_summary_text(
    path: Path,
    *,
    metadata,
    labels: list[str],
    truth_points_by_label: dict[str, list[float]],
    point_centers: dict[str, dict[str, Any]],
    subset_summary: list[dict[str, Any]],
    spp_summary: list[dict[str, Any]],
    flagged: list[dict[str, Any]],
    recommendation: dict[str, Any],
    candidate: dict[str, Any],
    optimize_for: str,
    averaging_methods: list[str],
) -> None:
    lines: list[str] = []
    run_id = getattr(metadata, "run_id", "") or ""
    timestamp = getattr(metadata, "timestamp_utc", "") or ""
    lines.append("# Registration Sampling Study")
    lines.append("")
    lines.append(f"Run ID: {run_id}")
    lines.append(f"Timestamp UTC: {timestamp}")
    lines.append(f"Optimization criterion: {optimize_for}")
    lines.append(f"Averaging methods studied: {', '.join(averaging_methods)}")
    lines.append(f"Landmarks captured: {len(labels)}  Truth labels with coords: {len(truth_points_by_label)}")
    lines.append("")
    lines.append("## Recommended protocol")
    lines.append(f"- Recommended subset size: {recommendation.get('recommended_subset_size')}")
    lines.append(f"- Recommended samples per point: {recommendation.get('recommended_samples_per_point')}")
    lines.append(f"- Recommended averaging method: {recommendation.get('recommended_averaging_method')}")
    lines.append(f"- Rationale: {recommendation.get('rationale')}")
    lines.append("")
    lines.append("## Candidate full-subset registration (NOT auto-promoted)")
    if candidate.get("fre_mm") is not None:
        lines.append(f"- Labels: {len(candidate.get('labels', []))} ({', '.join(candidate.get('labels', []))})")
        lines.append(f"- FRE: {candidate['fre_mm']:.4f} mm")
        lines.append(f"- Max per-point residual: {candidate.get('max_residual_mm'):.4f} mm")
    else:
        lines.append("- Candidate registration could not be solved (insufficient data).")
    lines.append("")
    lines.append("## Subset-size summary")
    lines.append("size  trials  fre_mean_mm  fre_std_mm  fre_p95_mm  leave_one_out_mean_mm")
    for row in subset_summary:
        lines.append(
            f"  {row['subset_size']:>3d}   {row['trial_count']:>4d}   "
            f"{_fmt_mm(row.get('fre_mean_mm')):>10s}  {_fmt_mm(row.get('fre_std_mm')):>10s}  "
            f"{_fmt_mm(row.get('fre_p95_mm')):>10s}  {_fmt_mm(row.get('leave_one_out_mean_mm')):>10s}"
        )
    lines.append("")
    lines.append("## Samples-per-point summary")
    lines.append("k    trials  fre_mean_mm  fre_std_mm  fre_p95_mm")
    for row in spp_summary:
        lines.append(
            f"  {row['samples_per_point']:>3d}   {row['trial_count']:>4d}   "
            f"{_fmt_mm(row.get('fre_mean_mm')):>10s}  {_fmt_mm(row.get('fre_std_mm')):>10s}  "
            f"{_fmt_mm(row.get('fre_p95_mm')):>10s}"
        )
    lines.append("")
    if flagged:
        lines.append("## Flagged-but-not-deleted outlier labels")
        for entry in flagged:
            lines.append(
                f"- {entry['label']}: within-point RMS={entry['within_point_rms_mm']:.4f} mm, "
                f"modified z-score={entry['modified_z_score']:.2f}"
            )
        lines.append("")
    lines.append("## Per-point spread (mean center reference)")
    lines.append("label  samples  within_point_rms_mm  within_point_max_mm")
    for label in labels:
        meta = point_centers.get(label, {})
        lines.append(
            f"  {label:>4s}  {meta.get('sample_count', 0):>5d}  "
            f"{_fmt_mm(meta.get('within_point_rms_mm')):>10s}  {_fmt_mm(meta.get('within_point_max_mm')):>10s}"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _fmt_mm(value: Any) -> str:
    if value is None:
        return "n/a"
    try:
        return f"{float(value):.4f}"
    except Exception:
        return "n/a"


def _write_registration_candidate_json(
    path: Path,
    *,
    candidate: dict[str, Any],
    labels: list[str],
    centers: dict[str, list[float]],
    truth_points_by_label: dict[str, list[float]],
    samples_by_label: dict[str, list[list[float]]],
    metadata,
) -> None:
    """Write a `registration_candidate.json` that mirrors the canonical
    `latest_registration.json` schema closely enough for the promote tool to
    rename it into the active artifact slot.
    """
    T = candidate.get("T_robot_aurora")
    payload = {
        "schema_version": "registration_sampling_study_candidate_v1",
        "timestamp_utc": getattr(metadata, "timestamp_utc", "") or "",
        "source_run_id": getattr(metadata, "run_id", "") or "",
        "candidate_kind": "registration_sampling_study_full_subset_mean_centers",
        "landmark_labels": list(labels),
        "raw_captured_landmarks_aurora_xyz": {
            label: [[float(c) for c in xyz] for xyz in samples_by_label.get(label, [])]
            for label in labels
        },
        "averaged_landmarks_aurora_xyz": {
            label: [float(value) for value in centers.get(label, [])] for label in labels
        },
        "truth_points_in_robot_xyz": {
            label: [float(value) for value in truth_points_by_label.get(label, [])] for label in labels
        },
        "fre_mm": candidate.get("fre_mm"),
        "max_residual_mm": candidate.get("max_residual_mm"),
        "residuals_robot_xyz_mm_per_point": list(candidate.get("residuals_mm", []) or []),
        "T_robot_aurora": (
            [[float(value) for value in row] for row in np.asarray(T, dtype=float).tolist()]
            if T is not None
            else None
        ),
        "promote_warning": (
            "This candidate registration is NEVER auto-promoted. Use "
            "continuum_robot.data.promote_registration_study to copy it into "
            "data/registrations/latest_registration.json after manual review."
        ),
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------


def _plot_point_spread(path: Path, *, point_centers: dict[str, dict[str, Any]], labels: list[str]) -> None:
    rms = [
        float(point_centers.get(label, {}).get("within_point_rms_mm") or 0.0)
        for label in labels
    ]
    max_mm = [
        float(point_centers.get(label, {}).get("within_point_max_mm") or 0.0)
        for label in labels
    ]
    fig, ax = create_figure(size="wide")
    positions = np.arange(len(labels))
    width = 0.4
    ax.bar(positions - width / 2, rms, width=width, label="RMS within point", color=color("measured"))
    ax.bar(positions + width / 2, max_mm, width=width, label="Max within point", color=color("target"))
    ax.set_xticks(positions.tolist())
    ax.set_xticklabels(labels, rotation=45, ha="right")
    style_axes(ax, title="Per-Point Spread (Aurora frame)", xlabel="Landmark", ylabel="Spread (mm)")
    legend(ax, loc="upper right")
    save_figure(fig, path)


def _plot_subset_rms(path: Path, *, subset_summary: list[dict[str, Any]]) -> None:
    sizes = [row["subset_size"] for row in subset_summary]
    means = [float(row.get("fre_mean_mm") or 0.0) for row in subset_summary]
    p95 = [float(row.get("fre_p95_mm") or 0.0) for row in subset_summary]
    fig, ax = create_figure(size="wide")
    ax.plot(sizes, means, marker="o", color=color("measured"), label="FRE mean (mm)")
    ax.plot(sizes, p95, marker="s", color=color("threshold"), linestyle="--", label="FRE p95 (mm)")
    style_axes(ax, title="Registration FRE vs Subset Size", xlabel="Number of points", ylabel="FRE (mm)")
    legend(ax, loc="upper right")
    save_figure(fig, path)


def _plot_samples_per_point(path: Path, *, samples_per_point_summary: list[dict[str, Any]]) -> None:
    xs = [row["samples_per_point"] for row in samples_per_point_summary]
    ys = [float(row.get("fre_mean_mm") or 0.0) for row in samples_per_point_summary]
    p95 = [float(row.get("fre_p95_mm") or 0.0) for row in samples_per_point_summary]
    fig, ax = create_figure(size="wide")
    ax.plot(xs, ys, marker="o", color=color("measured"), label="FRE mean (mm)")
    ax.plot(xs, p95, marker="s", color=color("threshold"), linestyle="--", label="FRE p95 (mm)")
    style_axes(ax, title="Registration FRE vs Samples per Point", xlabel="Samples per point", ylabel="FRE (mm)")
    legend(ax, loc="upper right")
    save_figure(fig, path)


def _plot_transform_consistency(path: Path, *, subset_rows: list[dict[str, Any]]) -> None:
    """Box-plot-ish view of FRE distribution per subset size using scatter."""
    fig, ax = create_figure(size="wide")
    by_size: dict[int, list[float]] = {}
    for row in subset_rows:
        if row.get("fre_mm") is None:
            continue
        by_size.setdefault(int(row["subset_size"]), []).append(float(row["fre_mm"]))
    positions = sorted(by_size.keys())
    for idx, size in enumerate(positions):
        values = by_size[size]
        xs = np.full(len(values), idx + 1, dtype=float)
        jitter = (np.random.default_rng(int(size)).random(len(values)) - 0.5) * 0.30
        ax.scatter(xs + jitter, values, s=24, c=color("measured"), alpha=0.55, linewidths=0)
        ax.plot([idx + 1 - 0.2, idx + 1 + 0.2], [float(np.mean(values))] * 2, color=color("target"), lw=2)
    ax.set_xticks(list(range(1, len(positions) + 1)))
    ax.set_xticklabels([str(size) for size in positions])
    style_axes(ax, title="FRE consistency by subset size", xlabel="Subset size", ylabel="FRE (mm)")
    save_figure(fig, path)

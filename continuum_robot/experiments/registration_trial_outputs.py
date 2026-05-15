"""Output artifacts for the registration trial experiment."""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any, Mapping, Sequence


REPORT_FIGURES: tuple[str, ...] = (
    "registration_trial_point_spread_report.png",
    "registration_trial_subset_rms_report.png",
    "registration_trial_samples_per_point_report.png",
    "registration_trial_method_comparison_report.png",
)


def write_registration_trial_outputs(
    *, output_dir: Path, metadata, summary
) -> dict[str, Path]:
    """Write the trial report (markdown + JSON + PNG reports) into the run folder.

    The experiment framework already writes ``metadata.json`` and ``summary.json``;
    this function adds:

    - ``trial_report.md`` — human-readable comparison + recommendations.
    - ``trial_report.json`` — structured payload (method sweep, subset search,
      samples-per-point study, raw captures, truth).
    - Four PNG reports for advisor/thesis handoff (best-effort: headless or
      matplotlib-broken environments still get the md + json above).
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics: dict[str, Any] = dict(summary.experiment_metrics or {}) if hasattr(summary, "experiment_metrics") else {}
    if not metrics and isinstance(summary, dict):
        metrics = dict(summary.get("experiment_metrics") or {})
    md_path = output_dir / "trial_report.md"
    json_path = output_dir / "trial_report.json"
    md_path.write_text(_render_markdown(metadata, metrics), encoding="utf-8")
    payload = {
        "schema_version": "1.1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "method_summary": metrics.get("method_summary", {}),
        "method_sweep": metrics.get("method_sweep", {}),
        "subset_search_summary": metrics.get("subset_search_summary", {}),
        "subset_search_count": metrics.get("subset_search_count", 0),
        "samples_per_point_summary": metrics.get("samples_per_point_summary", []),
        "samples_per_point_recommendation": metrics.get("samples_per_point_recommendation", {}),
        "landmark_labels_captured": metrics.get("landmark_labels_captured", []),
        "captures_per_landmark_target": metrics.get("captures_per_landmark_target"),
        "trial_recommendations": metrics.get("trial_recommendations", []),
        "report_figures": list(REPORT_FIGURES),
        "truth_by_label": metrics.get("truth_by_label", {}),
        "raw_captures_by_label": metrics.get("raw_captures_by_label", {}),
    }
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    try:
        _emit_plots(output_dir, metrics)
    except Exception:
        # Plot failures must not lose a successful trial run.
        pass
    return {"trial_report_md": md_path, "trial_report_json": json_path}


def _emit_plots(output_dir: Path, metrics: Mapping[str, Any]) -> None:
    """Write the four PNG reports. Imports matplotlib lazily."""
    from continuum_robot.experiments.plotting import (
        add_metric_box,
        color,
        create_figure,
        legend,
        save_figure,
        style_axes,
    )
    import numpy as np

    raw_captures = dict(metrics.get("raw_captures_by_label") or {})
    labels = list(metrics.get("landmark_labels_captured") or sorted(raw_captures.keys()))

    # 1. Per-point spread (within-point RMS + max around the mean center).
    if raw_captures:
        rms_list: list[float] = []
        max_list: list[float] = []
        sample_count_list: list[int] = []
        for label in labels:
            captures = raw_captures.get(label) or []
            if not captures:
                rms_list.append(0.0)
                max_list.append(0.0)
                sample_count_list.append(0)
                continue
            arr = np.asarray(captures, dtype=float)
            mean = arr.mean(axis=0)
            distances = np.linalg.norm(arr - mean, axis=1)
            rms_list.append(float(np.sqrt(np.mean(np.square(distances)))) if distances.size else 0.0)
            max_list.append(float(np.max(distances)) if distances.size else 0.0)
            sample_count_list.append(int(arr.shape[0]))
        fig, ax = create_figure(size="wide")
        positions = np.arange(len(labels))
        width = 0.4
        ax.bar(positions - width / 2, rms_list, width=width, label="RMS within point", color=color("measured"))
        ax.bar(positions + width / 2, max_list, width=width, label="Max within point", color=color("target"))
        ax.set_xticks(positions.tolist())
        ax.set_xticklabels(labels, rotation=45, ha="right")
        style_axes(ax, title="Per-Point Capture Spread (Aurora frame)", xlabel="Landmark", ylabel="Spread (mm)")
        if sample_count_list:
            add_metric_box(
                ax,
                [
                    f"min K = {min(sample_count_list)}",
                    f"max K = {max(sample_count_list)}",
                ],
                loc="upper right",
            )
        legend(ax, loc="upper left")
        save_figure(fig, output_dir / "registration_trial_point_spread_report.png")

    # 2. Subset FRE vs size (per-size best from the exhaustive search).
    subset_summary = dict(metrics.get("subset_search_summary") or {})
    per_size = dict(subset_summary.get("per_size_best") or {})
    if per_size:
        sizes = sorted(int(k) for k in per_size.keys())
        best_fres = [float(per_size[k].get("best_fre_mm") or 0.0) for k in sizes]
        max_residuals = [float(per_size[k].get("best_max_residual_mm") or 0.0) for k in sizes]
        fig, ax = create_figure(size="wide")
        ax.plot(sizes, best_fres, marker="o", color=color("measured"), label="Best FRE (mm)")
        ax.plot(sizes, max_residuals, marker="s", color=color("threshold"), linestyle="--", label="Best max residual (mm)")
        style_axes(ax, title="Registration FRE vs Subset Size", xlabel="Number of points in fit", ylabel="mm")
        legend(ax, loc="upper right")
        save_figure(fig, output_dir / "registration_trial_subset_rms_report.png")

    # 3. Samples-per-point curve.
    spp_summary = list(metrics.get("samples_per_point_summary") or [])
    if spp_summary:
        xs = [int(row.get("k") or 0) for row in spp_summary]
        means = [float(row.get("fre_mean_mm") or 0.0) for row in spp_summary]
        p95 = [float(row.get("fre_p95_mm") or 0.0) for row in spp_summary]
        fig, ax = create_figure(size="wide")
        ax.plot(xs, means, marker="o", color=color("measured"), label="FRE mean (mm)")
        ax.plot(xs, p95, marker="s", color=color("threshold"), linestyle="--", label="FRE p95 (mm)")
        recommendation = dict(metrics.get("samples_per_point_recommendation") or {})
        recommended_k = recommendation.get("recommended_k")
        if recommended_k is not None:
            ax.axvline(int(recommended_k), color=color("target"), linestyle=":", linewidth=1.4)
        style_axes(ax, title="Registration FRE vs Samples per Point", xlabel="Samples per landmark", ylabel="FRE (mm)")
        if recommended_k is not None and recommendation.get("best_fre_mean_mm") is not None:
            add_metric_box(
                ax,
                [
                    f"recommended k = {int(recommended_k)}",
                    f"best mean = {float(recommendation.get('best_fre_mean_mm') or 0.0):.4f} mm",
                ],
                loc="upper right",
            )
        legend(ax, loc="upper left")
        save_figure(fig, output_dir / "registration_trial_samples_per_point_report.png")

    # 4. Averaging-method comparison (bar of FRE and max LOO drop per method).
    method_summary = dict(metrics.get("method_summary") or {})
    rows = list(method_summary.get("method_rows") or [])
    if rows:
        method_names = [str(row.get("method")) for row in rows]
        fres = [float(row.get("fre_mm") or 0.0) for row in rows]
        loo_drops = [float(row.get("loo_max_minus_keep_mm") or 0.0) for row in rows]
        fig, ax = create_figure(size="wide")
        positions = np.arange(len(method_names))
        width = 0.4
        ax.bar(positions - width / 2, fres, width=width, label="FRE (mm)", color=color("measured"))
        ax.bar(positions + width / 2, loo_drops, width=width, label="Max LOO drop (mm)", color=color("threshold"))
        ax.set_xticks(positions.tolist())
        ax.set_xticklabels(method_names, rotation=20)
        style_axes(ax, title="Averaging Methods Compared", xlabel="Method", ylabel="mm")
        best_method = method_summary.get("best_method")
        if best_method:
            add_metric_box(ax, [f"best: {best_method}"], loc="upper right")
        legend(ax, loc="upper left")
        save_figure(fig, output_dir / "registration_trial_method_comparison_report.png")


def _render_markdown(metadata, metrics: Mapping[str, Any]) -> str:
    lines: list[str] = []
    lines.append("# Registration Trial Report")
    lines.append("")
    timestamp = getattr(metadata, "timestamp_utc", None) or datetime.now(timezone.utc).isoformat()
    run_id = getattr(metadata, "run_id", None) or "unknown"
    lines.append(f"- Run ID: `{run_id}`")
    lines.append(f"- Generated: {timestamp}")
    labels = metrics.get("landmark_labels_captured") or []
    lines.append(f"- Landmarks captured: {labels}")
    captures_target = metrics.get("captures_per_landmark_target")
    method_summary = metrics.get("method_summary") or {}
    captures_min = method_summary.get("captures_min")
    captures_max = method_summary.get("captures_max")
    lines.append(
        f"- Captures per landmark: target={captures_target} actual_min={captures_min} actual_max={captures_max}"
    )
    lines.append("")

    method_rows = sorted(
        list(method_summary.get("method_rows") or []),
        key=lambda row: float(row.get("fre_mm") or 0.0),
    )
    lines.append("## Method comparison")
    lines.append("")
    if method_rows:
        lines.append("| Method | FRE (mm) | Max residual (mm) | Worst label | LOO max drop (mm) |")
        lines.append("|---|---:|---:|---|---:|")
        for row in method_rows:
            lines.append(
                "| {method} | {fre:.4f} | {maxr:.4f} | {worst} | {loo:.4f} |".format(
                    method=row.get("method"),
                    fre=float(row.get("fre_mm") or 0.0),
                    maxr=float(row.get("max_residual_mm") or 0.0),
                    worst=row.get("worst_landmark_label") or "-",
                    loo=float(row.get("loo_max_minus_keep_mm") or 0.0),
                )
            )
        lines.append("")
        best_method = method_summary.get("best_method")
        best_fre = method_summary.get("best_fre_mm")
        if best_method is not None and best_fre is not None:
            lines.append(f"**Best method: `{best_method}` at {float(best_fre):.4f} mm**")
            lines.append("")
    else:
        lines.append("_No method rows recorded._")
        lines.append("")

    subset_summary = metrics.get("subset_search_summary") or {}
    lines.append("## Subset search")
    lines.append("")
    averaging_used = subset_summary.get("averaging_method_used") or "(unknown)"
    lines.append(
        f"_Subset search used the averaged points from method `{averaging_used}`. "
        "Averaging knob is not stacked on top of subset knob to keep results comparable._"
    )
    lines.append("")
    per_size_best = subset_summary.get("per_size_best") or {}
    if per_size_best:
        lines.append("| Size | # subsets | Best FRE (mm) | Best subset | Max residual (mm) | Rank | Cond. # |")
        lines.append("|---:|---:|---:|---|---:|---:|---:|")
        for size in sorted(per_size_best.keys(), key=lambda v: int(v)):
            row = per_size_best[size]
            cond = row.get("best_geometry_condition_number")
            cond_text = f"{float(cond):.1f}" if cond is not None else "-"
            lines.append(
                "| {size} | {n} | {fre:.4f} | {labels} | {maxr:.4f} | {rank} | {cond} |".format(
                    size=row.get("size"),
                    n=row.get("subset_count"),
                    fre=float(row.get("best_fre_mm") or 0.0),
                    labels=row.get("best_labels"),
                    maxr=float(row.get("best_max_residual_mm") or 0.0),
                    rank=row.get("best_geometry_rank"),
                    cond=cond_text,
                )
            )
        lines.append("")
    global_best = subset_summary.get("global_best")
    if isinstance(global_best, Mapping):
        lines.append(
            "**Global best subset: size={size} labels={labels} FRE={fre:.4f} mm**".format(
                size=global_best.get("size"),
                labels=global_best.get("labels"),
                fre=float(global_best.get("fre_mm") or 0.0),
            )
        )
        lines.append("")

    # Samples-per-point study table.
    spp_rows = list(metrics.get("samples_per_point_summary") or [])
    if spp_rows:
        lines.append("## Samples per landmark")
        lines.append("")
        lines.append("_Mean averaging with random k-subsets of the captured pool. Bootstrap iterations "
                     "smooth the choice of which k samples were drawn._")
        lines.append("")
        lines.append("| k | iterations | FRE mean (mm) | FRE std (mm) | FRE p95 (mm) | FRE min (mm) | FRE max (mm) |")
        lines.append("|---:|---:|---:|---:|---:|---:|---:|")
        for row in spp_rows:
            lines.append(
                "| {k} | {n} | {mean:.4f} | {std:.4f} | {p95:.4f} | {mn:.4f} | {mx:.4f} |".format(
                    k=int(row.get("k") or 0),
                    n=int(row.get("iteration_count") or 0),
                    mean=float(row.get("fre_mean_mm") or 0.0),
                    std=float(row.get("fre_std_mm") or 0.0),
                    p95=float(row.get("fre_p95_mm") or 0.0),
                    mn=float(row.get("fre_min_mm") or 0.0),
                    mx=float(row.get("fre_max_mm") or 0.0),
                )
            )
        lines.append("")
        recommendation = dict(metrics.get("samples_per_point_recommendation") or {})
        recommended_k = recommendation.get("recommended_k")
        if recommended_k is not None and recommendation.get("best_fre_mean_mm") is not None:
            lines.append(
                "**Recommended k = {k} (mean FRE {fre:.4f} mm, within "
                "{eps:.3f} mm of the captured pool's best {best:.4f} mm).**".format(
                    k=int(recommended_k),
                    fre=float(recommendation.get("recommended_fre_mean_mm") or 0.0),
                    eps=float(recommendation.get("epsilon_mm") or 0.02),
                    best=float(recommendation.get("best_fre_mean_mm") or 0.0),
                )
            )
            lines.append("")

    recs = metrics.get("trial_recommendations") or []
    if recs:
        lines.append("## Recommendations")
        lines.append("")
        for rec in recs:
            lines.append(f"- {rec}")
        lines.append("")
    figures = metrics.get("report_figures") if isinstance(metrics.get("report_figures"), list) else list(REPORT_FIGURES)
    if figures:
        lines.append("## Report figures")
        lines.append("")
        for figure in figures:
            lines.append(f"- `{figure}`")
        lines.append("")
    return "\n".join(lines) + "\n"

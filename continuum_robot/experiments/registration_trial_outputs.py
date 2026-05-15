"""Output artifacts for the registration trial experiment."""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any, Mapping, Sequence


def write_registration_trial_outputs(
    *, output_dir: Path, metadata, summary
) -> dict[str, Path]:
    """Write the trial report (markdown + JSON) into the run folder.

    The experiment framework already writes ``metadata.json`` and ``summary.json``;
    this function adds:

    - ``trial_report.md`` — human-readable comparison + recommendations.
    - ``trial_report.json`` — structured payload (method sweep, subset search,
      raw captures, truth).
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
        "schema_version": "1.0",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "method_summary": metrics.get("method_summary", {}),
        "method_sweep": metrics.get("method_sweep", {}),
        "subset_search_summary": metrics.get("subset_search_summary", {}),
        "subset_search_count": metrics.get("subset_search_count", 0),
        "landmark_labels_captured": metrics.get("landmark_labels_captured", []),
        "captures_per_landmark_target": metrics.get("captures_per_landmark_target"),
        "trial_recommendations": metrics.get("trial_recommendations", []),
        "truth_by_label": metrics.get("truth_by_label", {}),
        "raw_captures_by_label": metrics.get("raw_captures_by_label", {}),
    }
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return {"trial_report_md": md_path, "trial_report_json": json_path}


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

    recs = metrics.get("trial_recommendations") or []
    if recs:
        lines.append("## Recommendations")
        lines.append("")
        for rec in recs:
            lines.append(f"- {rec}")
        lines.append("")
    return "\n".join(lines) + "\n"

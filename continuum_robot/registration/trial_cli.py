"""CLI for the registration trial sweep.

Replays the trial analysis on already-saved registration records so an operator
can ask "what would MAD/trimmed/median have done on captures I already have?"
without bench time.

Example:

    .venv/bin/python -m continuum_robot.registration.trial_cli \
        data/registrations/latest_registration.json --output-dir data/diagnostics/registration_trial

The script writes ``trial_report.md`` and ``trial_report.json`` to the chosen
output directory and prints a small comparison table to stdout.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np
import yaml

from continuum_robot.registration.trial_analysis import (
    AVERAGING_METHODS,
    RegistrationTrialResult,
    sweep_methods,
    summarize_trial,
)


def load_truth_points(registration_yaml_path: Path) -> dict[str, list[float]]:
    """Read candidate landmark truth points from ``config/registration.yaml``."""
    payload = yaml.safe_load(registration_yaml_path.read_text(encoding="utf-8")) or {}
    truth: dict[str, list[float]] = {}
    nominal = payload.get("nominal_landmarks_robot_xyz_mm") or {}
    if isinstance(nominal, Mapping):
        for label, xyz in nominal.items():
            truth[str(label)] = [float(v) for v in xyz]
    for entry in payload.get("candidate_landmarks", []) or []:
        if not isinstance(entry, Mapping):
            continue
        landmark_id = str(entry.get("id") or entry.get("label") or "").strip()
        xyz = entry.get("xyz_mm") or entry.get("coordinates_mm") or entry.get("xyz")
        if not landmark_id or xyz is None:
            continue
        truth.setdefault(landmark_id, [float(v) for v in xyz])
    if not truth:
        raise RuntimeError(
            f"No truth points found in {registration_yaml_path}. Provide candidate_landmarks or"
            " nominal_landmarks_robot_xyz_mm."
        )
    return truth


def load_captures_from_registration_record(
    record_path: Path,
) -> dict[str, list[list[float]]]:
    """Load the raw per-landmark captures stored alongside a saved registration."""
    payload = json.loads(record_path.read_text(encoding="utf-8"))
    captures = payload.get("raw_captured_landmarks_aurora_xyz")
    if not isinstance(captures, Mapping):
        raise RuntimeError(
            f"{record_path} has no raw_captured_landmarks_aurora_xyz; this script needs raw per-capture points"
        )
    out: dict[str, list[list[float]]] = {}
    for label, points in captures.items():
        if not isinstance(points, Sequence):
            continue
        rows = [list(map(float, p)) for p in points if isinstance(p, Sequence) and len(p) == 3]
        if rows:
            out[str(label)] = rows
    if not out:
        raise RuntimeError(f"{record_path}: raw captures present but empty")
    return out


def run_trial_for_record(
    record_path: Path,
    truth_by_label: Mapping[str, Sequence[float]],
    *,
    methods: Sequence[str] = AVERAGING_METHODS,
    trimmed_fraction: float = 0.2,
    mad_k: float = 3.5,
) -> tuple[dict[str, RegistrationTrialResult], dict[str, object]]:
    """Run the method sweep and return raw + summary records for one registration."""
    captures = load_captures_from_registration_record(record_path)
    # Limit truth to the labels that have captures so we don't fight the solver.
    truth_subset = {label: list(truth_by_label[label]) for label in captures if label in truth_by_label}
    missing = sorted(set(captures.keys()) - set(truth_subset.keys()))
    if missing:
        raise RuntimeError(
            f"{record_path}: captures present for labels {missing} but no truth available"
        )
    results = sweep_methods(
        captures,
        truth_subset,
        methods=methods,
        trimmed_fraction=trimmed_fraction,
        mad_k=mad_k,
    )
    summary = summarize_trial(results)
    capture_counts = {label: len(points) for label, points in captures.items()}
    summary["capture_counts_per_label"] = capture_counts
    summary["captures_min"] = min(capture_counts.values())
    summary["captures_max"] = max(capture_counts.values())
    summary["record_path"] = str(record_path)
    return results, summary


def render_markdown(
    record_path: Path,
    truth_by_label: Mapping[str, Sequence[float]],
    results: Mapping[str, RegistrationTrialResult],
    summary: Mapping[str, object],
) -> str:
    lines: list[str] = []
    lines.append("# Registration Trial Replay Report")
    lines.append("")
    lines.append(f"- Source: `{record_path}`")
    lines.append(f"- Generated: {datetime.now(timezone.utc).isoformat()}")
    lines.append(f"- Captures per landmark: min={summary.get('captures_min')}, max={summary.get('captures_max')}")
    used_labels = sorted({label for result in results.values() for label in result.per_label_residual_mm})
    lines.append(f"- Labels solved against: {used_labels}")
    lines.append("")
    lines.append("## Method comparison")
    lines.append("")
    lines.append("| Method | FRE (mm) | Max residual (mm) | Worst label | LOO max drop (mm) |")
    lines.append("|---|---:|---:|---|---:|")
    for method, result in results.items():
        lines.append(
            f"| {method} | {result.fre_mm:.4f} | {result.max_residual_mm:.4f} | "
            f"{result.worst_landmark_label or '-'} | {result.loo_max_minus_keep_mm:.4f} |"
        )
    lines.append("")
    lines.append(
        f"**Best method by FRE: `{summary.get('best_method')}` at "
        f"{summary.get('best_fre_mm'):.4f} mm**"
        if summary.get("best_fre_mm") is not None
        else "**Best method by FRE: n/a**"
    )
    lines.append("")
    lines.append("## Per-method detail")
    lines.append("")
    for method, result in results.items():
        lines.append(f"### {method}")
        lines.append("")
        lines.append("Per-landmark residual norms (mm):")
        for label, value in sorted(result.per_label_residual_mm.items()):
            avg = result.averaging.get(label)
            kept = avg.n_kept if avg else "?"
            n_in = avg.n_input if avg else "?"
            intra = f"{avg.intra_capture_stddev_mm:.4f}" if avg else "?"
            lines.append(
                f"- `{label}` residual={value:.4f} | kept={kept}/{n_in} | intra-capture stddev={intra}"
            )
        if result.loo_fre_mm_by_excluded_label:
            lines.append("")
            lines.append("Leave-one-out FRE (mm) by excluded label:")
            for label, fre in sorted(result.loo_fre_mm_by_excluded_label.items()):
                delta = result.fre_mm - fre
                arrow = "↓" if delta > 0 else "↑"
                lines.append(f"- excluding `{label}`: FRE = {fre:.4f}  {arrow}{abs(delta):.4f} from keep-all")
        geom = result.geometry if isinstance(result.geometry, dict) else {}
        cond = geom.get("condition_number")
        rank = geom.get("geometry_rank")
        svals = geom.get("singular_values_mm")
        if any(v is not None for v in (cond, rank, svals)):
            lines.append("")
            lines.append("Truth-landmark geometry:")
            if rank is not None:
                lines.append(f"- rank = {rank}")
            if cond is not None:
                lines.append(f"- condition number = {cond:.3f}")
            if svals is not None:
                lines.append(f"- singular values (mm) = {[round(float(v), 3) for v in svals]}")
        lines.append("")

    lines.append("## Recommendations")
    lines.append("")
    geom = next(iter(results.values())).geometry if results else {}
    geom = geom if isinstance(geom, dict) else {}
    rank = geom.get("geometry_rank")
    svals = geom.get("singular_values_mm") or []
    if rank is not None and int(rank) < 3:
        lines.append(
            "- **Truth landmarks are coplanar / rank-deficient (rank={}).** A rigid Kabsch fit "
            "on coplanar points is well-defined in-plane but degenerate out-of-plane; the third "
            "axis is recovered only from measurement noise. Add a landmark at a different height "
            "(non-coplanar z) before expecting sub-0.5 mm FRE consistently.".format(rank)
        )
    if svals and min((float(v) for v in svals if float(v) > 0.0), default=None) is not None:
        ratio = float(max(svals)) / max(min(float(v) for v in svals if float(v) > 0.0), 1e-12)
        if ratio > 50:
            lines.append(
                f"- Truth-geometry singular-value ratio is large ({ratio:.0f}); landmarks are "
                "poorly spread along at least one axis. Choose landmarks with better 3D spread."
            )
    best_method = summary.get("best_method")
    best_fre = summary.get("best_fre_mm")
    rows = sorted(summary.get("method_rows") or [], key=lambda row: float(row["fre_mm"]))
    if rows and len(rows) > 1:
        worst_fre = float(rows[-1]["fre_mm"])
        best_observed = float(rows[0]["fre_mm"])
        if worst_fre - best_observed < 0.01:
            lines.append(
                "- The four averaging methods agree to within 0.01 mm. With your current capture "
                "count, the averaging choice is not the bottleneck. To get more separation between "
                "methods, capture more samples per landmark (50+ recommended) so MAD / trimmed mean "
                "have something to act on."
            )
        else:
            lines.append(
                f"- Best averaging method on this run: **{best_method}** at {float(best_fre):.4f} mm. "
                f"Worst method on the same data: {float(worst_fre):.4f} mm. Difference = "
                f"{worst_fre - best_observed:.4f} mm."
            )
    loo_drops = []
    for result in results.values():
        for excluded, fre in result.loo_fre_mm_by_excluded_label.items():
            loo_drops.append((excluded, result.fre_mm - fre))
    if loo_drops:
        # Aggregate by excluded label, take max drop.
        per_label_max_drop: dict[str, float] = {}
        for label, drop in loo_drops:
            if drop > per_label_max_drop.get(label, float("-inf")):
                per_label_max_drop[label] = drop
        worst = max(per_label_max_drop.items(), key=lambda item: item[1])
        if worst[1] > 0.05:
            lines.append(
                f"- Excluding landmark **{worst[0]}** drops FRE by up to "
                f"{worst[1]:.4f} mm. Either recapture {worst[0]} with the operator paying "
                "extra attention to probe contact and Aurora visibility, or capture more landmarks "
                "so a single bad one is less load-bearing."
            )
    lines.append("")
    return "\n".join(lines) + "\n"


def write_report(output_dir: Path, markdown: str, results_payload: dict) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    md_path = output_dir / "trial_report.md"
    json_path = output_dir / "trial_report.json"
    md_path.write_text(markdown, encoding="utf-8")
    json_path.write_text(json.dumps(results_payload, indent=2, default=_json_default), encoding="utf-8")
    return md_path, json_path


def _json_default(value):  # pragma: no cover - trivial
    if isinstance(value, np.ndarray):
        return value.tolist()
    if hasattr(value, "__dict__"):
        return asdict(value)
    raise TypeError(f"Cannot serialize {type(value)}")


def _results_to_payload(results: Mapping[str, RegistrationTrialResult]) -> dict:
    return {method: asdict(result) for method, result in results.items()}


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Replay the registration trial sweep on a saved registration record."
    )
    parser.add_argument("record", type=Path, help="Path to a saved registration JSON.")
    parser.add_argument(
        "--registration-config",
        type=Path,
        default=Path("config/registration.yaml"),
        help="Registration config used to look up truth landmark coordinates. Default: config/registration.yaml.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/diagnostics/registration_trial"),
        help="Where to write trial_report.md and trial_report.json.",
    )
    parser.add_argument("--mad-k", type=float, default=3.5, help="MAD rejection threshold (default 3.5).")
    parser.add_argument(
        "--trimmed-fraction",
        type=float,
        default=0.2,
        help="Trim fraction for trimmed_mean (default 0.2).",
    )
    args = parser.parse_args(list(argv) if argv is not None else None)

    truth_by_label = load_truth_points(args.registration_config)
    results, summary = run_trial_for_record(
        args.record,
        truth_by_label,
        trimmed_fraction=float(args.trimmed_fraction),
        mad_k=float(args.mad_k),
    )
    markdown = render_markdown(args.record, truth_by_label, results, summary)
    payload = {
        "summary": summary,
        "results_by_method": _results_to_payload(results),
    }
    md_path, json_path = write_report(args.output_dir, markdown, payload)

    print(f"Wrote {md_path}")
    print(f"Wrote {json_path}")
    print()
    print("Method comparison (lower FRE is better):")
    rows = summary.get("method_rows") or []
    rows = sorted(rows, key=lambda row: float(row["fre_mm"]))
    width = max((len(row["method"]) for row in rows), default=10)
    for row in rows:
        print(
            f"  {row['method']:<{width}}  FRE={float(row['fre_mm']):.4f} mm  "
            f"max_res={float(row['max_residual_mm']):.4f} mm  "
            f"worst={row['worst_landmark_label']}"
        )
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entry
    sys.exit(main())

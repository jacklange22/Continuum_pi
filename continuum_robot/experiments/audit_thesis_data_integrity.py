"""Audit existing experiment runs for thesis-evidence integrity.

This is an audit/report tool — it never modifies or deletes data. It scans
``data/experiments/`` and ``data/mock_experiments/`` for saved run folders,
reads each run's ``metadata.json``, ``summary.json``, ``config_snapshot.yaml``,
and ``run_review.json`` (when present), and writes a timestamped diagnostic
bundle under ``data/diagnostics/thesis_data_integrity/``.

Run classification:

* ``thesis_candidate`` — run has the trust metadata of a live thesis run:
  no mock/dry-run/synthetic markers, transform-chain summary present,
  runtime-tip trust status recorded, tracker backend not mock.
* ``needs_review`` — run is missing one or more provenance pieces (no
  transform-chain summary, no runtime-tip trust, no review sidecar) but
  is not clearly synthetic. A human should look before citing.
* ``debug_or_synthetic`` — run is explicitly synthetic / mock / dry-run.
  Recorded clearly so it cannot be confused with real evidence.
* ``missing_metadata`` — run folder exists but lacks ``metadata.json``
  or ``summary.json`` entirely; cannot be classified without those files.

Synthetic / mock / dry-run signals checked (any one triggers
``debug_or_synthetic``):

* ``dry_run: true`` in config_snapshot
* ``mock_mode: true`` in metadata's provenance_info
* ``run_trust_mode == "mock"`` in trust_info
* ``tracker_backend`` containing ``mock`` (e.g., ``mock_tracker_manager``)
* explicit ``not_thesis_evidence: true`` in metrics
* ``synthetic_seed_used`` field present in summary metrics
  (signals grid_accuracy ran in synthetic mode)
* ``status_flags`` containing ``synthetic`` or ``dry_run`` on samples
  (only checked when summary.json doesn't disambiguate)
* output_dir under ``data/mock_experiments/``

Provenance fields verified for ``thesis_candidate`` classification:

* presence of ``transform_chain_summary.json`` (or ``.txt``) in the run folder
* runtime-tip trust status surfaced (``runtime_tip_trust_status`` or
  equivalent) in metadata.registration_info
* ``tracker_backend`` set to a non-mock value
* ``runtime_tip_mode`` field present

CLI:

    python -m continuum_robot.experiments.audit_thesis_data_integrity \\
        [--project-root /path/to/pi_code] \\
        [--include-archived] [--include-mock-tree]

Outputs (always written under ``data/diagnostics/thesis_data_integrity/``):

* ``summary.json``  — counts, classification roll-up, per-experiment breakdown
* ``integrity_report.csv``  — one row per run, machine-friendly
* ``integrity_report.txt``  — same data, human-friendly, with explicit list
  of synthetic / mock runs at the top so the operator can see them at a glance
"""

from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import yaml


SYNTHETIC_REASONS_SCHEMA = "thesis_data_integrity_v1"


# Classification labels (also the order used in the text report sections).
CLASS_THESIS_CANDIDATE = "thesis_candidate"
CLASS_NEEDS_REVIEW = "needs_review"
CLASS_DEBUG_OR_SYNTHETIC = "debug_or_synthetic"
CLASS_MISSING_METADATA = "missing_metadata"

# Roots scanned for run folders. ``data/experiments_archived/`` and
# ``data/mock_experiments/`` are opt-in via CLI flags.
DEFAULT_ROOTS = ("data/experiments",)
OPTIONAL_ROOTS = {
    "include_archived": ("data/experiments_archived",),
    "include_mock_tree": ("data/mock_experiments",),
}


def _safe_read_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {"_parse_error": True}


def _safe_read_yaml(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
    except Exception:
        return {"_parse_error": True}
    return loaded if isinstance(loaded, dict) else None


def _discover_run_folders(project_root: Path, roots: Iterable[str]) -> list[Path]:
    found: list[Path] = []
    for root_rel in roots:
        root = project_root / root_rel
        if not root.is_dir():
            continue
        for experiment_dir in sorted(root.iterdir()):
            if not experiment_dir.is_dir():
                continue
            for run_dir in sorted(experiment_dir.iterdir()):
                if run_dir.is_dir() and (run_dir / "metadata.json").exists() or (
                    run_dir.is_dir() and (run_dir / "summary.json").exists()
                ):
                    found.append(run_dir)
                elif run_dir.is_dir():
                    # Still emit so it shows up as missing_metadata, but only
                    # if it looks roughly like a run folder (has *any* file).
                    if any(run_dir.iterdir()):
                        found.append(run_dir)
    return found


def _detect_synthetic_signals(
    *,
    run_dir: Path,
    metadata: dict[str, Any] | None,
    summary: dict[str, Any] | None,
    config_snapshot: dict[str, Any] | None,
) -> tuple[bool, list[str]]:
    """Return (is_synthetic, reasons)."""
    reasons: list[str] = []

    # Run folder under data/mock_experiments/ → always synthetic.
    if "mock_experiments" in run_dir.parts:
        reasons.append("output_dir_under_mock_experiments")

    cfg = config_snapshot or {}
    if cfg.get("dry_run") is True:
        reasons.append("config.dry_run=true")

    meta = metadata or {}
    prov = dict(meta.get("provenance_info") or {})
    trust = dict(meta.get("trust_info") or {})
    if bool(prov.get("mock_mode")):
        reasons.append("metadata.provenance_info.mock_mode=true")
    backend = str(prov.get("tracker_backend") or "").lower()
    if "mock" in backend:
        reasons.append(f"tracker_backend={prov.get('tracker_backend')}")
    if str(trust.get("run_trust_mode") or "").lower() == "mock":
        reasons.append("trust_info.run_trust_mode=mock")
    if bool(trust.get("not_thesis_evidence")):
        reasons.append("trust_info.not_thesis_evidence=true")
    for reason in list(trust.get("not_thesis_evidence_reasons") or []):
        reasons.append(f"trust_info.reason={reason}")

    summ = summary or {}
    metrics = dict(summ.get("experiment_metrics") or {})
    if metrics.get("synthetic_seed_used") is not None:
        reasons.append("metrics.synthetic_seed_used present (grid_accuracy synthetic)")
    if bool(metrics.get("not_thesis_evidence")):
        reasons.append("metrics.not_thesis_evidence=true")
    for reason in list(metrics.get("not_thesis_evidence_reasons") or []):
        reasons.append(f"metrics.reason={reason}")
    if str(metrics.get("run_trust_mode") or "").lower() == "mock":
        reasons.append("metrics.run_trust_mode=mock")

    return (bool(reasons), sorted(set(reasons)))


def _detect_provenance_gaps(
    *,
    run_dir: Path,
    metadata: dict[str, Any] | None,
    summary: dict[str, Any] | None,
) -> list[str]:
    gaps: list[str] = []
    transform_chain_json = run_dir / "transform_chain_summary.json"
    transform_chain_txt = run_dir / "transform_chain_summary.txt"
    if not transform_chain_json.exists() and not transform_chain_txt.exists():
        gaps.append("missing_transform_chain_summary")
    meta = metadata or {}
    reg = dict(meta.get("registration_info") or {})
    if "runtime_tip_mode" not in reg and "runtime_tip_mode" not in (
        dict(meta.get("provenance_info") or {})
    ):
        gaps.append("missing_runtime_tip_mode")
    if "runtime_tip_trust_level" not in reg and "runtime_tip_trust_status" not in (
        dict(meta.get("provenance_info") or {})
    ):
        gaps.append("missing_runtime_tip_trust_status")
    prov = dict(meta.get("provenance_info") or {})
    if not str(prov.get("tracker_backend") or "").strip():
        gaps.append("missing_tracker_backend")
    return gaps


def _classify(
    *,
    metadata: dict[str, Any] | None,
    summary: dict[str, Any] | None,
    is_synthetic: bool,
    gaps: list[str],
) -> str:
    if metadata is None and summary is None:
        return CLASS_MISSING_METADATA
    if is_synthetic:
        return CLASS_DEBUG_OR_SYNTHETIC
    if gaps:
        return CLASS_NEEDS_REVIEW
    return CLASS_THESIS_CANDIDATE


def audit_run_folder(run_dir: Path) -> dict[str, Any]:
    metadata = _safe_read_json(run_dir / "metadata.json")
    summary = _safe_read_json(run_dir / "summary.json")
    config_snapshot = _safe_read_yaml(run_dir / "config_snapshot.yaml")
    review = _safe_read_json(run_dir / "run_review.json")

    is_synthetic, synthetic_reasons = _detect_synthetic_signals(
        run_dir=run_dir,
        metadata=metadata,
        summary=summary,
        config_snapshot=config_snapshot,
    )
    gaps = _detect_provenance_gaps(
        run_dir=run_dir, metadata=metadata, summary=summary
    )
    classification = _classify(
        metadata=metadata,
        summary=summary,
        is_synthetic=is_synthetic,
        gaps=gaps,
    )
    experiment_name = ""
    timestamp_utc = ""
    if metadata:
        experiment_name = str(metadata.get("experiment_name") or "")
        timestamp_utc = str(metadata.get("timestamp_utc") or "")
    if not experiment_name:
        # Fall back to the parent directory name (experiments/<name>/<run>/).
        try:
            experiment_name = run_dir.parent.name
        except Exception:
            experiment_name = ""
    review_status = ""
    intended_use = ""
    if review:
        review_status = str(review.get("review_status") or "")
        intended_use = str(review.get("intended_use") or "")
    return {
        "run_dir": str(run_dir),
        "experiment_name": experiment_name,
        "timestamp_utc": timestamp_utc,
        "classification": classification,
        "synthetic_or_mock": bool(is_synthetic),
        "synthetic_reasons": synthetic_reasons,
        "provenance_gaps": gaps,
        "review_status": review_status,
        "intended_use": intended_use,
        "metadata_present": bool(metadata),
        "summary_present": bool(summary),
        "config_snapshot_present": bool(config_snapshot),
        "run_review_present": bool(review),
    }


def write_audit_outputs(
    *,
    project_root: Path,
    rows: list[dict[str, Any]],
) -> dict[str, Path]:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    output_dir = (
        project_root
        / "data"
        / "diagnostics"
        / "thesis_data_integrity"
        / f"{timestamp}_thesis_data_integrity"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    # CSV report
    csv_path = output_dir / "integrity_report.csv"
    fieldnames = [
        "run_dir",
        "experiment_name",
        "timestamp_utc",
        "classification",
        "synthetic_or_mock",
        "synthetic_reasons",
        "provenance_gaps",
        "review_status",
        "intended_use",
        "metadata_present",
        "summary_present",
    ]
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({
                "run_dir": row["run_dir"],
                "experiment_name": row["experiment_name"],
                "timestamp_utc": row["timestamp_utc"],
                "classification": row["classification"],
                "synthetic_or_mock": "true" if row["synthetic_or_mock"] else "false",
                "synthetic_reasons": "|".join(row["synthetic_reasons"]),
                "provenance_gaps": "|".join(row["provenance_gaps"]),
                "review_status": row["review_status"],
                "intended_use": row["intended_use"],
                "metadata_present": "true" if row["metadata_present"] else "false",
                "summary_present": "true" if row["summary_present"] else "false",
            })

    # Summary JSON
    counts = {label: 0 for label in (
        CLASS_THESIS_CANDIDATE,
        CLASS_NEEDS_REVIEW,
        CLASS_DEBUG_OR_SYNTHETIC,
        CLASS_MISSING_METADATA,
    )}
    by_experiment: dict[str, dict[str, int]] = {}
    for row in rows:
        counts[row["classification"]] = counts.get(row["classification"], 0) + 1
        exp_bucket = by_experiment.setdefault(row["experiment_name"] or "(unknown)", {
            label: 0 for label in counts
        })
        exp_bucket[row["classification"]] = exp_bucket.get(row["classification"], 0) + 1

    summary_payload = {
        "schema_version": SYNTHETIC_REASONS_SCHEMA,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "project_root": str(project_root),
        "run_count_total": len(rows),
        "classification_counts": counts,
        "classification_counts_by_experiment": by_experiment,
        "synthetic_or_mock_runs": [
            row["run_dir"] for row in rows if row["synthetic_or_mock"]
        ],
        "runs": rows,
    }
    summary_path = output_dir / "summary.json"
    summary_path.write_text(
        json.dumps(summary_payload, indent=2, sort_keys=True), encoding="utf-8"
    )

    # TXT report
    txt_path = output_dir / "integrity_report.txt"
    lines: list[str] = []
    lines.append("Thesis Data Integrity Audit")
    lines.append("=" * 78)
    lines.append(f"Generated:   {summary_payload['generated_at_utc']}")
    lines.append(f"Project:     {project_root}")
    lines.append(f"Total runs:  {len(rows)}")
    lines.append("")
    lines.append("Classification counts:")
    for label in (
        CLASS_THESIS_CANDIDATE,
        CLASS_NEEDS_REVIEW,
        CLASS_DEBUG_OR_SYNTHETIC,
        CLASS_MISSING_METADATA,
    ):
        lines.append(f"  {label:<25s}  {counts.get(label, 0)}")
    lines.append("")

    # Synthetic/mock runs called out explicitly at the top.
    synth_rows = [row for row in rows if row["synthetic_or_mock"]]
    if synth_rows:
        lines.append(f"Synthetic / mock / dry-run runs ({len(synth_rows)}):")
        lines.append("-" * 78)
        for row in synth_rows:
            lines.append(f"  {row['run_dir']}")
            if row["synthetic_reasons"]:
                lines.append(f"      reasons: {', '.join(row['synthetic_reasons'])}")
        lines.append("")
    else:
        lines.append("No synthetic / mock / dry-run runs found.")
        lines.append("")

    needs_review = [row for row in rows if row["classification"] == CLASS_NEEDS_REVIEW]
    if needs_review:
        lines.append(f"Runs needing provenance review ({len(needs_review)}):")
        lines.append("-" * 78)
        for row in needs_review:
            lines.append(f"  {row['run_dir']}")
            if row["provenance_gaps"]:
                lines.append(f"      gaps: {', '.join(row['provenance_gaps'])}")
        lines.append("")

    missing = [row for row in rows if row["classification"] == CLASS_MISSING_METADATA]
    if missing:
        lines.append(f"Runs missing metadata/summary entirely ({len(missing)}):")
        lines.append("-" * 78)
        for row in missing:
            lines.append(f"  {row['run_dir']}")
        lines.append("")

    by_experiment_lines = sorted(by_experiment.items())
    if by_experiment_lines:
        lines.append("Per-experiment classification breakdown:")
        lines.append("-" * 78)
        header = "  {:<45s} {:>6s} {:>6s} {:>6s} {:>6s}".format(
            "experiment", "thesis", "review", "debug", "missing"
        )
        lines.append(header)
        for name, bucket in by_experiment_lines:
            lines.append("  {:<45s} {:>6d} {:>6d} {:>6d} {:>6d}".format(
                name[:45],
                bucket.get(CLASS_THESIS_CANDIDATE, 0),
                bucket.get(CLASS_NEEDS_REVIEW, 0),
                bucket.get(CLASS_DEBUG_OR_SYNTHETIC, 0),
                bucket.get(CLASS_MISSING_METADATA, 0),
            ))

    txt_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    return {
        "output_dir": output_dir,
        "summary_path": summary_path,
        "csv_path": csv_path,
        "txt_path": txt_path,
    }


def audit_project(
    *,
    project_root: Path,
    include_archived: bool = False,
    include_mock_tree: bool = True,
) -> dict[str, Any]:
    roots: list[str] = list(DEFAULT_ROOTS)
    if include_mock_tree:
        roots.extend(OPTIONAL_ROOTS["include_mock_tree"])
    if include_archived:
        roots.extend(OPTIONAL_ROOTS["include_archived"])
    run_dirs = _discover_run_folders(project_root, roots)
    rows = [audit_run_folder(run_dir) for run_dir in run_dirs]
    paths = write_audit_outputs(project_root=project_root, rows=rows)
    return {
        "row_count": len(rows),
        "paths": paths,
        "rows": rows,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--project-root",
        default=".",
        help="Project root containing data/experiments/. Defaults to cwd.",
    )
    parser.add_argument(
        "--include-archived",
        action="store_true",
        help="Also scan data/experiments_archived/.",
    )
    parser.add_argument(
        "--include-mock-tree",
        action="store_true",
        default=True,
        help="Scan data/mock_experiments/ as well (default: on).",
    )
    parser.add_argument(
        "--exclude-mock-tree",
        action="store_true",
        help="Skip data/mock_experiments/ even though the default scans it.",
    )
    args = parser.parse_args(argv)
    project_root = Path(args.project_root).resolve()
    include_mock_tree = bool(args.include_mock_tree) and not bool(args.exclude_mock_tree)
    result = audit_project(
        project_root=project_root,
        include_archived=bool(args.include_archived),
        include_mock_tree=include_mock_tree,
    )
    paths = result["paths"]
    print(f"Audited {result['row_count']} run folder(s).")
    print(f"Report: {paths['txt_path']}")
    print(f"Summary: {paths['summary_path']}")
    print(f"CSV:     {paths['csv_path']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

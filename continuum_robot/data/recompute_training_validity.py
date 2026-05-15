"""Recompute collect-pose training-validity metadata from saved run artifacts."""

from __future__ import annotations

import argparse
import ast
import json
import re
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from continuum_robot.data.model_training_validity import (
    collect_pose_training_row_stats_from_samples,
    evaluate_collect_pose_model_training_validity,
)
from continuum_robot.experiments.builtins import (
    _collect_pose_dataset_quality_summary,
    _render_collect_pose_dataset_quality_summary,
)
from continuum_robot.experiments.modeling_dataset_outputs import build_modeling_dataset_summary_lines
from continuum_robot.experiments.modeling_dataset_outputs import _build_export_rows, _build_legacy_dat_rows
from continuum_robot.experiments.schemas import ExperimentTimeseriesSample
from continuum_robot.experiments.dat_writer import DatRunWriter


TOKEN_FIXES = (
    (re.compile(r"\bTrue\b"), "true"),
    (re.compile(r"\bFalse\b"), "false"),
    (re.compile(r"\bNone\b"), "null"),
)


def _load_json_tolerant(path: Path) -> tuple[dict[str, Any], str]:
    if not path.exists():
        return {}, "missing"
    text = path.read_text(encoding="utf-8")
    try:
        payload = json.loads(text)
        return (payload if isinstance(payload, dict) else {}), "json"
    except Exception:
        pass
    patched = text
    for rx, replacement in TOKEN_FIXES:
        patched = rx.sub(replacement, patched)
    if patched != text:
        try:
            payload = json.loads(patched)
            return (payload if isinstance(payload, dict) else {}), "json_token_patched"
        except Exception:
            pass
    try:
        payload = ast.literal_eval(text)
        return (payload if isinstance(payload, dict) else {}), "python_literal"
    except Exception:
        return {}, "unparsed"


def _load_samples(path: Path) -> list[ExperimentTimeseriesSample]:
    if not path.exists():
        return []
    samples: list[ExperimentTimeseriesSample] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        payload = json.loads(line)
        if isinstance(payload, dict):
            samples.append(ExperimentTimeseriesSample.from_dict(payload))
    return samples


def _legacy_row_count(path: Path) -> int:
    if not path.exists():
        return 0
    try:
        with path.open("r", encoding="utf-8", errors="replace") as handle:
            for _ in range(24):
                line = handle.readline()
                if not line:
                    break
                if line.startswith("NUM_MEASUREMENTS:"):
                    return int(line.split(":", 1)[1].strip())
    except Exception:
        pass
    rows = 0
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for _ in handle:
            rows += 1
    return max(0, rows - 6)


def _resolve_run_dirs(targets: list[str]) -> list[Path]:
    resolved: list[Path] = []
    for value in targets:
        path = Path(value).expanduser()
        if path.is_dir() and (path / "summary.json").exists():
            resolved.append(path)
            continue
        if path.is_dir() and path.name == "collect_pose_command_dataset":
            resolved.extend(sorted(p for p in path.iterdir() if p.is_dir()))
            continue
        if path.exists() and path.is_file() and path.name == "summary.json":
            resolved.append(path.parent)
            continue
        parent = path.parent
        if path.name.endswith("*") and parent.exists():
            pattern = path.name
            resolved.extend(sorted(p for p in parent.glob(pattern) if p.is_dir()))
    unique: dict[str, Path] = {}
    for run_dir in resolved:
        unique[str(run_dir.resolve())] = run_dir
    return list(unique.values())


def _recompute_one_run(run_dir: Path, *, apply: bool) -> dict[str, Any]:
    summary_path = run_dir / "summary.json"
    metadata_path = run_dir / "metadata.json"
    quality_path = run_dir / "dataset_quality_summary.json"
    quality_txt_path = run_dir / "dataset_quality_summary.txt"
    modeling_summary_path = run_dir / "modeling_dataset_summary.txt"
    samples_path = run_dir / "samples.jsonl"
    export_path = run_dir / "modeling_dataset_export.jsonl"
    legacy_path = run_dir / "modeling_dataset_legacy_compat.dat"

    summary_payload, summary_parse_mode = _load_json_tolerant(summary_path)
    metadata_payload, metadata_parse_mode = _load_json_tolerant(metadata_path)
    quality_payload, quality_parse_mode = _load_json_tolerant(quality_path)
    experiment_name = str(summary_payload.get("experiment_name") or metadata_payload.get("experiment_name") or "")
    if experiment_name.strip().lower() != "collect_pose_command_dataset":
        return {
            "run_dir": str(run_dir),
            "status": "skipped",
            "reason": f"experiment_name={experiment_name or 'unknown'}",
            "parse_modes": {
                "summary": summary_parse_mode,
                "metadata": metadata_parse_mode,
                "quality": quality_parse_mode,
            },
        }
    metrics = (
        dict(summary_payload.get("experiment_metrics", {}) or {})
        if isinstance(summary_payload.get("experiment_metrics"), dict)
        else {}
    )
    samples = _load_samples(samples_path)
    expected_export_rows = _build_export_rows(samples=samples)
    export_contaminated = False
    current_export_rows: list[dict[str, Any]] = []
    if export_path.exists():
        for line in export_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except Exception:
                export_contaminated = True
                continue
            if isinstance(row, dict):
                current_export_rows.append(row)
    if len(current_export_rows) != len(expected_export_rows):
        export_contaminated = True
    elif current_export_rows and expected_export_rows:
        for lhs, rhs in zip(current_export_rows, expected_export_rows):
            if lhs != rhs:
                export_contaminated = True
                break
    row_stats = collect_pose_training_row_stats_from_samples(samples)
    success = bool(summary_payload.get("success"))
    validity = evaluate_collect_pose_model_training_validity(
        success=success,
        experiment_name=experiment_name,
        metrics=metrics,
        row_stats=row_stats,
        legacy_row_count=len(expected_export_rows),
    )
    metrics["valid_for_model_training"] = bool(validity.get("valid_for_model_training"))
    metrics["not_model_training_ready"] = bool(validity.get("not_model_training_ready"))
    metrics["model_training_validity_status"] = str(validity.get("model_training_validity_status"))
    metrics["model_training_validity_reason"] = str(validity.get("model_training_validity_reason"))
    metrics["model_training_validity_checks"] = dict(validity.get("model_training_validity_checks", {}) or {})
    metrics["model_training_warnings"] = list(validity.get("model_training_warnings", []) or [])
    metrics["model_training_hard_invalidation_reasons"] = list(
        validity.get("model_training_hard_invalidation_reasons", []) or []
    )
    metrics["dropped_samples_excluded_from_training"] = bool(validity.get("dropped_samples_excluded_from_training"))
    metrics["accepted_rows_complete"] = bool(validity.get("accepted_rows_complete"))
    metrics["modeling_export_row_count"] = int(validity.get("modeling_export_row_count", 0) or 0)
    metrics["modeling_legacy_row_count"] = int(validity.get("modeling_legacy_row_count", 0) or 0)
    metrics["accepted_training_row_count"] = int(validity.get("accepted_training_row_count", 0) or 0)
    metrics["complete_training_row_count"] = int(validity.get("complete_training_row_count", 0) or 0)
    metrics["accepted_workspace_sample_count"] = int(validity.get("accepted_workspace_sample_count", 0) or 0)
    metrics["non_training_accepted_row_count"] = int(validity.get("non_training_accepted_row_count", 0) or 0)
    metrics["incomplete_accepted_workspace_row_count"] = int(validity.get("incomplete_accepted_workspace_row_count", 0) or 0)
    metrics["dropped_quarantined_sample_count"] = int(validity.get("dropped_quarantined_sample_count", 0) or 0)
    run_provenance = dict(metrics.get("run_provenance", {}) or {})
    if run_provenance:
        run_provenance["valid_for_model_training"] = bool(metrics["valid_for_model_training"])
        metrics["run_provenance"] = run_provenance
    summary_payload["experiment_metrics"] = metrics

    quality = _collect_pose_dataset_quality_summary(
        samples=samples,
        metrics=metrics,
        max_current_warning_ma=metrics.get("dataset_stability_parameters", {}).get("max_current_warning_ma"),
    )
    quality_payload = {**quality_payload, **quality}

    trust_info = (
        dict(metadata_payload.get("trust_info", {}) or {})
        if isinstance(metadata_payload.get("trust_info"), dict)
        else {}
    )
    trust_info["valid_for_model_training"] = bool(metrics["valid_for_model_training"])
    metadata_payload["trust_info"] = trust_info

    if apply:
        if export_contaminated or (not export_path.exists()):
            with export_path.open("w", encoding="utf-8") as handle:
                for row in expected_export_rows:
                    handle.write(json.dumps(row, separators=(",", ":")) + "\n")
            legacy_rows = _build_legacy_dat_rows(export_rows=expected_export_rows)
            if legacy_rows:
                writer = DatRunWriter(output_dir=run_dir)
                writer.write_run(
                    num_cables=4,
                    rows=legacy_rows,
                    filename_stem="modeling_dataset_legacy_compat",
                )
        summary_path.write_text(json.dumps(summary_payload, indent=2, sort_keys=True), encoding="utf-8")
        metadata_path.write_text(json.dumps(metadata_payload, indent=2, sort_keys=True), encoding="utf-8")
        quality_path.write_text(json.dumps(quality_payload, indent=2, sort_keys=True), encoding="utf-8")
        quality_txt_path.write_text(_render_collect_pose_dataset_quality_summary(quality_payload), encoding="utf-8")
        summary_lines = build_modeling_dataset_summary_lines(
            metadata=SimpleNamespace(
                run_id=str(summary_payload.get("run_id", run_dir.name)),
                timestamp_utc=str(summary_payload.get("timestamp_utc", metadata_payload.get("timestamp_utc", ""))),
            ),
            summary=SimpleNamespace(status=str(summary_payload.get("status", "unknown"))),
            metrics=metrics,
        )
        modeling_summary_path.write_text("\n".join(summary_lines).strip() + "\n", encoding="utf-8")

    return {
        "run_dir": str(run_dir),
        "status": "updated" if apply else "dry_run",
        "valid_for_model_training": bool(metrics["valid_for_model_training"]),
        "not_model_training_ready": bool(metrics["not_model_training_ready"]),
        "model_training_validity_status": str(metrics.get("model_training_validity_status")),
        "model_training_validity_reason": str(metrics.get("model_training_validity_reason")),
        "model_training_warnings": list(metrics.get("model_training_warnings", []) or []),
        "accepted_training_row_count": int(metrics.get("accepted_training_row_count", 0) or 0),
        "modeling_export_row_count": int(metrics.get("modeling_export_row_count", 0) or 0),
        "export_contaminated": bool(export_contaminated),
        "parse_modes": {
            "summary": summary_parse_mode,
            "metadata": metadata_parse_mode,
            "quality": quality_parse_mode,
        },
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Recompute collect-pose trainability metadata for one or more run folders.")
    parser.add_argument("targets", nargs="+", help="Run folder(s), summary.json path(s), or collect-pose root folder.")
    parser.add_argument("--apply", action="store_true", help="Write updated derived metadata files.")
    parser.add_argument("--dry-run", action="store_true", help="Report only (default behavior).")
    args = parser.parse_args(argv)

    apply = bool(args.apply)
    if bool(args.dry_run):
        apply = False
    run_dirs = _resolve_run_dirs(list(args.targets))
    if not run_dirs:
        print("No run directories resolved from targets.")
        return 2
    results = [_recompute_one_run(run_dir, apply=apply) for run_dir in run_dirs]
    for result in results:
        mode = result.get("status")
        run_dir = result.get("run_dir")
        if mode == "skipped":
            print(f"SKIP {run_dir}: {result.get('reason')}")
            continue
        print(
            f"{'APPLY' if apply else 'DRY'} {run_dir} | "
            f"valid_for_model_training={result.get('valid_for_model_training')} | "
            f"status={result.get('model_training_validity_status')} | "
            f"reason={result.get('model_training_validity_reason')} | "
            f"export_contaminated={result.get('export_contaminated')}"
        )
        warnings = list(result.get("model_training_warnings", []) or [])
        for warning in warnings:
            print(f"  warning: {warning}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Export one experiment run into a small advisor/thesis handoff bundle."""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import sys
from typing import Any

from continuum_robot.experiments.dataset_io import canonical_timestamp_token, sanitize_output_name


DEFAULT_DATA_ROOT = Path("data")
DEFAULT_MAX_SAMPLE_BYTES = 25 * 1024 * 1024
EXPORT_PROFILE_HUMAN = "human_advisor"
EXPORT_PROFILE_AI_DEBUG = "ai_debug"
EXPORT_PROFILES = {EXPORT_PROFILE_HUMAN, EXPORT_PROFILE_AI_DEBUG}

CORE_FILENAMES = {
    "summary.json",
    "metadata.json",
    "run_review.json",
    "config_snapshot.yaml",
    "metrics.csv",
    "comparison_metrics.csv",
    "phase_metrics.csv",
    "modeling_dataset_summary.txt",
    "modeling_dataset_export.jsonl",
    "failure_context.json",
    "dataset_quality_summary.json",
    "dataset_quality_summary.txt",
    "pretension_summary.txt",
    "two_segment_startup_summary.txt",
    "two_segment_startup_artifact_metadata.json",
    "two_segment_dataset_summary.txt",
    "two_segment_tracking_role_provenance.json",
    "two_segment_modeling_summary.txt",
    "long_run_health.json",
    "sample_failure_events.jsonl",
    "transport_recovery_report.json",
    "two_segment_repeatability_summary.txt",
    "two_segment_repeatability_scatter_metrics.json",
    "two_segment_repeatability_per_target.csv",
    "model_config.yaml",
    "feature_metadata.json",
    "label_metadata.json",
    "train_test_split.json",
    "rejected_samples.jsonl",
    "predictions.csv",
    "run_provenance.json",
    "model_status.json",
    "physics_model_parameter_report.json",
    "physics_model_parameter_report.txt",
    "registration_validation_summary.txt",
    "pivot_validation_summary.txt",
    "repeatability_summary.txt",
    "penprobe_chasing_summary.txt",
    "training_summary.txt",
    "training_metadata.json",
    "training_config.json",
    "loss_history.csv",
    "evaluation_metadata.json",
}

OPTIONAL_LARGE_FILENAMES = {
    "samples.jsonl",
    "modeling_dataset_legacy_compat.dat",
}

DEBUG_NAME_MARKERS = ("_debug", "debug_", "diagnostic")


@dataclass(frozen=True)
class BundleEntry:
    source_path: str
    bundle_path: str
    size_bytes: int
    sha256: str
    category: str


@dataclass(frozen=True)
class SkippedEntry:
    source_path: str
    reason: str
    size_bytes: int = 0


@dataclass(frozen=True)
class ExportBundleResult:
    run_dir: Path
    bundle_dir: Path
    zip_path: Path | None
    manifest_path: Path
    entries: list[BundleEntry] = field(default_factory=list)
    skipped: list[SkippedEntry] = field(default_factory=list)

    @property
    def final_path(self) -> Path:
        return self.zip_path if self.zip_path is not None else self.bundle_dir

    @property
    def total_size_bytes(self) -> int:
        return sum(entry.size_bytes for entry in self.entries)


def find_latest_run(*, project_root: Path, experiment_name: str, data_root: Path = DEFAULT_DATA_ROOT) -> Path:
    """Return the newest canonical run directory for an experiment."""
    root = Path(project_root)
    safe_name = sanitize_output_name(experiment_name, default="experiment")
    experiment_root = root / data_root / "experiments" / safe_name
    candidates: list[Path] = []
    if experiment_root.exists():
        candidates.extend(path for path in experiment_root.iterdir() if path.is_dir())
    legacy_root = root / data_root / "experiments"
    if legacy_root.exists():
        candidates.extend(
            path
            for path in legacy_root.iterdir()
            if path.is_dir() and path.name.endswith(f"_{safe_name}")
        )
    valid = [path for path in candidates if (path / "summary.json").exists() or (path / "metadata.json").exists()]
    if not valid:
        raise FileNotFoundError(f"No runs found for experiment '{experiment_name}' under {root / data_root / 'experiments'}")
    return max(valid, key=lambda path: (path.stat().st_mtime, path.name))


def export_run_bundle(
    *,
    run_dir: Path,
    output_root: Path | None = None,
    project_root: Path | None = None,
    include_samples: bool = False,
    include_debug: bool = False,
    max_sample_bytes: int = DEFAULT_MAX_SAMPLE_BYTES,
    make_zip: bool = False,
    profile: str = EXPORT_PROFILE_HUMAN,
) -> ExportBundleResult:
    """Copy the useful contents of one run directory into a bounded export bundle."""
    run_dir = Path(run_dir).resolve()
    if not run_dir.exists() or not run_dir.is_dir():
        raise FileNotFoundError(f"Run directory does not exist: {run_dir}")
    project_root = Path(project_root or Path.cwd()).resolve()
    profile = _normalize_profile(profile)
    effective_include_debug = bool(include_debug or profile == EXPORT_PROFILE_AI_DEBUG)
    experiment_name = _experiment_name_for_run(run_dir)
    stamp = canonical_timestamp_token()
    bundle_name = f"{stamp}_{sanitize_output_name(experiment_name)}_{profile}_bundle"
    output_root = Path(output_root or (project_root / DEFAULT_DATA_ROOT / "exports")).resolve()
    bundle_dir = _collision_safe_dir(output_root / bundle_name)
    bundle_dir.mkdir(parents=True, exist_ok=False)

    entries: list[BundleEntry] = []
    skipped: list[SkippedEntry] = []
    for source in _candidate_files(run_dir):
        category = _classify_file(source)
        if category == "debug" and not effective_include_debug:
            skipped.append(SkippedEntry(source_path=str(source), reason="debug output excluded by default", size_bytes=source.stat().st_size))
            continue
        if category == "figure" and not effective_include_debug:
            skipped.append(SkippedEntry(source_path=str(source), reason="non-report figure excluded from human/advisor profile", size_bytes=source.stat().st_size))
            continue
        sample_size = source.stat().st_size
        if source.name in OPTIONAL_LARGE_FILENAMES and sample_size > int(max_sample_bytes):
            skipped.append(SkippedEntry(source_path=str(source), reason=f"large sample file exceeds {int(max_sample_bytes)} bytes", size_bytes=source.stat().st_size))
            continue
        if source.name in OPTIONAL_LARGE_FILENAMES and not (include_samples or profile == EXPORT_PROFILE_AI_DEBUG):
            skipped.append(SkippedEntry(source_path=str(source), reason="large/raw sample file excluded; pass --include-samples or use ai_debug profile", size_bytes=source.stat().st_size))
            continue
        relative = source.relative_to(run_dir)
        destination = bundle_dir / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
        entries.append(_bundle_entry(source=source, destination=destination, bundle_dir=bundle_dir, category=category))

    final_export_path = bundle_dir.with_suffix(".zip") if make_zip else bundle_dir
    transfer_path = bundle_dir / "transfer_commands.txt"
    transfer_path.write_text(build_transfer_commands(final_export_path), encoding="utf-8")
    entries.append(_bundle_entry(source=transfer_path, destination=transfer_path, bundle_dir=bundle_dir, category="metadata"))

    trust_path = bundle_dir / "trust_provenance.json"
    trust_payload = _extract_trust_provenance(run_dir=run_dir)
    trust_path.write_text(json.dumps(trust_payload, indent=2), encoding="utf-8")
    entries.append(_bundle_entry(source=trust_path, destination=trust_path, bundle_dir=bundle_dir, category="metadata"))

    readme_path = bundle_dir / "README.txt"
    readme_path.write_text(
        _render_bundle_readme(
            run_dir=run_dir,
            experiment_name=experiment_name,
            trust_payload=trust_payload,
            profile=profile,
        ),
        encoding="utf-8",
    )
    entries.append(_bundle_entry(source=readme_path, destination=readme_path, bundle_dir=bundle_dir, category="metadata"))

    manifest_path = bundle_dir / "manifest.json"
    manifest_payload = {
        "schema_version": "1.0",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_run_dir": str(run_dir),
        "experiment_name": experiment_name,
        "profile": profile,
        "include_samples": bool(include_samples),
        "include_debug": bool(effective_include_debug),
        "max_sample_bytes": int(max_sample_bytes),
        "file_count": len(entries),
        "total_size_bytes": sum(entry.size_bytes for entry in entries),
        "files": [entry.__dict__ for entry in entries],
        "skipped": [entry.__dict__ for entry in skipped],
    }
    manifest_path.write_text(json.dumps(manifest_payload, indent=2), encoding="utf-8")
    entries.append(_bundle_entry(source=manifest_path, destination=manifest_path, bundle_dir=bundle_dir, category="manifest"))

    zip_path: Path | None = None
    if make_zip:
        zip_stem = str(bundle_dir)
        archive = shutil.make_archive(zip_stem, "zip", root_dir=bundle_dir)
        zip_path = Path(archive)
    return ExportBundleResult(
        run_dir=run_dir,
        bundle_dir=bundle_dir,
        zip_path=zip_path,
        manifest_path=manifest_path,
        entries=entries,
        skipped=skipped,
    )


def build_transfer_commands(export_path: Path, *, remote_user: str = "<pi-user>", remote_host: str = "<pi-host>") -> str:
    """Return copy/paste transfer command templates for an exported bundle."""
    path = Path(export_path)
    remote = f"{remote_user}@{remote_host}:{path}"
    return "\n".join(
        [
            "Transfer command templates",
            "",
            "Run these from the destination Mac/workstation after replacing placeholders:",
            "",
            f"rsync -av {remote} ~/Downloads/",
            f"scp -r {remote} ~/Downloads/",
            "",
            "Notes:",
            "- Prefer rsync for resumed/repeated transfers.",
            "- The exporter never copies the full data directory by default.",
            "- Inspect trust_provenance.json before using a run for thesis/modeling claims.",
        ]
    ).strip() + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    project_root = Path(args.project_root).resolve()
    if args.run_dir:
        run_dir = Path(args.run_dir)
    elif args.latest:
        run_dir = find_latest_run(project_root=project_root, experiment_name=str(args.latest))
    else:
        parser.error("Provide --latest EXPERIMENT_NAME or --run-dir PATH.")
    result = export_run_bundle(
        run_dir=run_dir,
        output_root=Path(args.output_dir) if args.output_dir else None,
        project_root=project_root,
        include_samples=bool(args.include_samples),
        include_debug=bool(args.include_debug),
        max_sample_bytes=int(args.max_sample_bytes),
        make_zip=bool(args.zip),
        profile=str(args.profile),
    )
    print(f"Exported run bundle: {result.final_path}")
    print(f"Source run: {result.run_dir}")
    print(f"Files included: {len(result.entries)}")
    print(f"Bundle size: {_format_bytes(result.total_size_bytes)}")
    if result.skipped:
        print(f"Skipped files: {len(result.skipped)} (see manifest.json)")
    return 0


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Export one experiment run into a bounded handoff bundle.")
    selector = parser.add_mutually_exclusive_group(required=True)
    selector.add_argument("--latest", metavar="EXPERIMENT_NAME", help="Export the latest run for this experiment.")
    selector.add_argument("--run-dir", metavar="PATH", help="Export a specific run directory.")
    parser.add_argument("--project-root", default=".", help="Project root containing data/experiments.")
    parser.add_argument("--output-dir", help="Destination root for bundles. Defaults to data/exports.")
    parser.add_argument("--include-samples", action="store_true", help="Include samples.jsonl and other large raw sample files.")
    parser.add_argument("--include-debug", action="store_true", help="Include debug subfolders and debug-marked figures.")
    parser.add_argument("--max-sample-bytes", type=int, default=DEFAULT_MAX_SAMPLE_BYTES, help="Maximum size for optional sample files.")
    parser.add_argument("--zip", action="store_true", help="Also create a .zip archive of the bundle folder.")
    parser.add_argument(
        "--profile",
        choices=sorted(EXPORT_PROFILES),
        default=EXPORT_PROFILE_HUMAN,
        help="Export profile: human_advisor is clean/thesis-facing; ai_debug is fuller for assistant debugging.",
    )
    return parser


def _candidate_files(run_dir: Path) -> list[Path]:
    files: list[Path] = []
    for path in sorted(run_dir.rglob("*")):
        if not path.is_file():
            continue
        if _is_export_artifact(path):
            files.append(path)
    return files


def _is_export_artifact(path: Path) -> bool:
    name = path.name
    suffix = path.suffix.lower()
    if name in CORE_FILENAMES or name in OPTIONAL_LARGE_FILENAMES:
        return True
    if suffix == ".png":
        return True
    if suffix in {".json", ".pt"} and "models" in [part.lower() for part in path.parts]:
        return True
    if suffix in {".json", ".csv", ".txt", ".yaml", ".yml"} and "debug" in str(path.parent).lower():
        return True
    return False


def _classify_file(path: Path) -> str:
    lower = str(path).lower()
    name = path.name.lower()
    if any(marker in name for marker in DEBUG_NAME_MARKERS) or "debug" in lower:
        return "debug"
    if name.endswith("_report.png") or "report" in name or name.startswith("thesis_"):
        return "report_figure"
    if name.endswith(".png"):
        return "figure"
    if name in OPTIONAL_LARGE_FILENAMES:
        return "samples"
    if name in {"summary.json", "metadata.json", "config_snapshot.yaml", "evaluation_metadata.json", "training_metadata.json"}:
        return "metadata"
    if name.endswith(".csv"):
        return "metrics"
    return "artifact"


def _bundle_entry(*, source: Path, destination: Path, bundle_dir: Path, category: str) -> BundleEntry:
    return BundleEntry(
        source_path=str(source),
        bundle_path=str(destination.relative_to(bundle_dir)),
        size_bytes=destination.stat().st_size,
        sha256=_sha256(destination),
        category=category,
    )


def _extract_trust_provenance(*, run_dir: Path) -> dict[str, Any]:
    metadata = _read_json_if_present(run_dir / "metadata.json")
    summary = _read_json_if_present(run_dir / "summary.json")
    summary_metrics = dict(summary.get("experiment_metrics", {}) or {}) if isinstance(summary.get("experiment_metrics"), dict) else {}
    metadata_trust = _as_dict(metadata.get("trust_info"))
    metadata_provenance = _as_dict(metadata.get("provenance_info"))
    summary_trust = _as_dict(summary_metrics.get("run_trust"))
    summary_provenance = _as_dict(summary_metrics.get("run_provenance"))
    return {
        "run_id": metadata.get("run_id") or summary.get("run_id"),
        "experiment_name": metadata.get("experiment_name") or summary.get("experiment_name") or _experiment_name_for_run(run_dir),
        "status": summary.get("status"),
        "success": summary.get("success"),
        "stop_reason": summary_metrics.get("stop_reason") or summary_metrics.get("failure_reason") or summary.get("stop_reason"),
        "metadata_trust_info": metadata_trust,
        "metadata_provenance_info": metadata_provenance,
        "summary_run_trust": summary_trust,
        "summary_run_provenance": summary_provenance,
        "selected_segment_readiness": (
            summary_provenance.get("selected_segment_readiness")
            or metadata_provenance.get("selected_segment_readiness")
            or {}
        ),
        "startup_pretension_artifact": (
            summary_provenance.get("startup_pretension_artifact")
            or metadata_provenance.get("startup_pretension_artifact")
            or _as_dict(_as_dict(metadata.get("backend_info")).get("pretension_source"))
        ),
        "servo_sign_mapping_check": (
            summary_provenance.get("servo_sign_mapping_check")
            or metadata_provenance.get("servo_sign_mapping_check")
            or _as_dict(_as_dict(metadata.get("backend_info")).get("servo_sign_mapping_check"))
        ),
        "active_segment": (
            summary_provenance.get("active_segment")
            or metadata_provenance.get("active_segment")
            or _as_dict(_as_dict(metadata.get("backend_info")).get("active_segment"))
        ),
        "run_trust_mode": (
            summary_metrics.get("run_trust_mode")
            or summary_trust.get("run_trust_mode")
            or metadata_trust.get("run_trust_mode")
        ),
        "valid_for_model_training": (
            summary_metrics.get("valid_for_model_training")
            if summary_metrics.get("valid_for_model_training") is not None
            else metadata_trust.get("valid_for_model_training")
        ),
        "valid_for_thesis_repeatability": (
            summary_metrics.get("valid_for_thesis_repeatability")
            if summary_metrics.get("valid_for_thesis_repeatability") is not None
            else metadata_trust.get("valid_for_thesis_repeatability")
        ),
        "data_quality_warnings": (
            summary_metrics.get("data_quality_warnings")
            or metadata_trust.get("data_quality_warnings")
            or []
        ),
    }


def _render_bundle_readme(*, run_dir: Path, experiment_name: str, trust_payload: dict[str, Any], profile: str) -> str:
    profile_note = (
        "Human / Advisor Export: clean review packet; raw samples and debug-heavy files are excluded unless requested."
        if profile == EXPORT_PROFILE_HUMAN
        else "AI Debug Export: complete debugging packet; samples are included when below the configured size cap."
    )
    return "\n".join(
        [
            f"Continuum Robot Run Bundle: {experiment_name}",
            "",
            f"Profile: {profile}",
            profile_note,
            "",
            f"Source run: {run_dir}",
            f"Run ID: {trust_payload.get('run_id', 'n/a')}",
            f"Status: {trust_payload.get('status', 'n/a')}",
            f"Run trust mode: {trust_payload.get('run_trust_mode', 'unknown')}",
            f"Valid for model training: {trust_payload.get('valid_for_model_training')}",
            f"Valid for thesis repeatability: {trust_payload.get('valid_for_thesis_repeatability')}",
            "",
            "Contents:",
            "- summary.json / metadata.json: canonical run status and provenance when present.",
            "- *_report.png: thesis-facing figures.",
            "- Other PNGs: compatibility or operator review figures.",
            "- metrics.csv / comparison_metrics.csv: tabular metrics when present.",
            "- trust_provenance.json: condensed trust/provenance block, including calibration/startup/sign-check context when present.",
            "- manifest.json: file list, sizes, checksums, and skipped large/debug files.",
            "- run_review.json: operator review status when present in the source run.",
            "",
            "Raw samples are included only for AI debug profile or when explicitly requested, and still obey the size cap.",
        ]
    ).strip() + "\n"


def _normalize_profile(profile: str) -> str:
    value = str(profile or EXPORT_PROFILE_HUMAN).strip().lower().replace("-", "_")
    aliases = {
        "human": EXPORT_PROFILE_HUMAN,
        "advisor": EXPORT_PROFILE_HUMAN,
        "human_advisor": EXPORT_PROFILE_HUMAN,
        "ai": EXPORT_PROFILE_AI_DEBUG,
        "debug": EXPORT_PROFILE_AI_DEBUG,
        "ai_debug": EXPORT_PROFILE_AI_DEBUG,
    }
    resolved = aliases.get(value, value)
    if resolved not in EXPORT_PROFILES:
        raise ValueError(f"Unknown export profile '{profile}'. Expected one of: {', '.join(sorted(EXPORT_PROFILES))}")
    return resolved


def _experiment_name_for_run(run_dir: Path) -> str:
    metadata = _read_json_if_present(run_dir / "metadata.json")
    if metadata.get("experiment_name"):
        return sanitize_output_name(str(metadata["experiment_name"]))
    summary = _read_json_if_present(run_dir / "summary.json")
    if summary.get("experiment_name"):
        return sanitize_output_name(str(summary["experiment_name"]))
    parent = run_dir.parent.name
    if parent != "experiments":
        return sanitize_output_name(parent)
    parts = run_dir.name.split("_", 2)
    return sanitize_output_name(parts[2] if len(parts) >= 3 else run_dir.name)


def _read_json_if_present(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def _as_dict(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, dict) else {}


def _collision_safe_dir(path: Path) -> Path:
    candidate = Path(path)
    if not candidate.exists():
        return candidate
    suffix = 1
    while True:
        next_candidate = candidate.with_name(f"{candidate.name}_{suffix:02d}")
        if not next_candidate.exists():
            return next_candidate
        suffix += 1


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _format_bytes(value: int) -> str:
    size = float(value)
    for unit in ("B", "KB", "MB", "GB"):
        if size < 1024.0 or unit == "GB":
            return f"{size:.1f} {unit}"
        size /= 1024.0
    return f"{size:.1f} GB"


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())

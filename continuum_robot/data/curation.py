"""Data inventory, curation classification, and safe cleanup planning."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
import csv
import hashlib
import json
from pathlib import Path
import shutil
from typing import Any

from continuum_robot.data.run_management import (
    REVIEW_FILENAME,
    archive_run,
    load_run_review,
    summarize_run,
    trash_run,
)


CURATION_REPORT_ROOT = Path("data/diagnostics/data_curation")
DEFAULT_DATA_ROOTS = [
    Path("data/experiments"),
    Path("data/mock_experiments"),
    Path("data/experiments_archived"),
    Path("data/trash"),
    Path("data/exports"),
    Path("data/calibration"),
    Path("data/pivot_calibration"),
    Path("data/runtime_tip_calibration"),
    Path("data/diagnostics"),
    Path("data/modeling_results"),
    Path("data/models"),
]
PROTECTED_NAMES = {
    "latest_registration.json",
    "latest_runtime_tip_calibration.json",
    "latest_quick_4_point_runtime_tip.json",
    "generated_penprobe_tip.csv",
}
GENERATED_ROOTS = {
    "data/diagnostics",
    "data/exports",
    "data/modeling_results",
}
IMPORTANT_REVIEW_STATUSES = {"keep", "thesis_candidate", "advisor_share"}
LOWER_TRUST_MODES = {"mock", "debug", "servo_only", "current_only", "lower_trust", "unknown"}


@dataclass(frozen=True)
class CurationItem:
    """One curation candidate discovered under a managed data root."""

    path: Path
    root: str
    item_type: str
    classification: str
    reasons: list[str] = field(default_factory=list)
    size_bytes: int = 0
    mtime_utc: str = ""
    experiment_name: str = ""
    run_id: str = ""
    review_status: str = ""
    has_run_review: bool = False
    mock_mode: bool | None = None
    trust_mode: str = ""
    success: Any = None
    status: str = ""
    duplicate_key: str = ""

    def to_dict(self, project_root: Path) -> dict[str, Any]:
        return {
            "path": _rel(project_root, self.path),
            "root": self.root,
            "item_type": self.item_type,
            "classification": self.classification,
            "reasons": list(self.reasons),
            "size_bytes": self.size_bytes,
            "mtime_utc": self.mtime_utc,
            "experiment_name": self.experiment_name,
            "run_id": self.run_id,
            "review_status": self.review_status,
            "has_run_review": self.has_run_review,
            "mock_mode": self.mock_mode,
            "trust_mode": self.trust_mode,
            "success": self.success,
            "status": self.status,
            "duplicate_key": self.duplicate_key,
        }


@dataclass(frozen=True)
class RootInventory:
    """Summary for one scanned root."""

    root: str
    exists: bool
    item_count: int = 0
    total_size_bytes: int = 0
    oldest_mtime_utc: str = ""
    newest_mtime_utc: str = ""
    missing_run_review_count: int = 0
    mock_count: int = 0
    lower_trust_count: int = 0
    thesis_candidate_count: int = 0
    advisor_share_count: int = 0


@dataclass(frozen=True)
class CurationReport:
    """Full non-mutating curation inventory."""

    generated_at_utc: str
    project_root: Path
    roots: list[RootInventory]
    items: list[CurationItem]
    large_files: list[dict[str, Any]]
    duplicate_groups: list[list[str]]
    report_dir: Path | None = None
    json_path: Path | None = None
    markdown_path: Path | None = None
    csv_path: Path | None = None

    def to_dict(self) -> dict[str, Any]:
        counts: dict[str, int] = {}
        for item in self.items:
            counts[item.classification] = counts.get(item.classification, 0) + 1
        return {
            "generated_at_utc": self.generated_at_utc,
            "project_root": str(self.project_root),
            "classification_counts": counts,
            "roots": [root.__dict__ for root in self.roots],
            "large_files": list(self.large_files),
            "duplicate_groups": list(self.duplicate_groups),
            "items": [item.to_dict(self.project_root) for item in self.items],
        }


@dataclass(frozen=True)
class CleanupAction:
    """One previewed or applied cleanup action."""

    action: str
    source_path: Path
    destination_path: Path | None
    classification: str
    status: str
    reason: str

    def to_dict(self, project_root: Path) -> dict[str, Any]:
        return {
            "action": self.action,
            "source_path": _rel(project_root, self.source_path),
            "destination_path": _rel(project_root, self.destination_path) if self.destination_path else "",
            "classification": self.classification,
            "status": self.status,
            "reason": self.reason,
        }


@dataclass(frozen=True)
class CleanupManifest:
    """Cleanup preview/apply manifest."""

    mode: str
    generated_at_utc: str
    action_count: int
    applied_count: int
    skipped_count: int
    error_count: int
    actions: list[CleanupAction]
    manifest_path: Path | None = None
    summary_path: Path | None = None

    def to_dict(self, project_root: Path) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "generated_at_utc": self.generated_at_utc,
            "action_count": self.action_count,
            "applied_count": self.applied_count,
            "skipped_count": self.skipped_count,
            "error_count": self.error_count,
            "actions": [action.to_dict(project_root) for action in self.actions],
        }


def build_curation_report(
    project_root: Path,
    *,
    roots: list[str | Path] | None = None,
    large_file_bytes: int = 25 * 1024 * 1024,
) -> CurationReport:
    """Scan managed data roots and classify items without mutating data."""
    project_root = Path(project_root).resolve()
    relative_roots = [Path(root) for root in (roots or DEFAULT_DATA_ROOTS)]
    items: list[CurationItem] = []
    root_reports: list[RootInventory] = []
    large_files: list[dict[str, Any]] = []

    for relative_root in relative_roots:
        root = project_root / relative_root
        root_items = _discover_items_for_root(project_root, relative_root)
        items.extend(root_items)
        root_reports.append(_root_inventory(project_root, relative_root, root_items))
        if root.exists():
            for path in root.rglob("*"):
                if path.is_file():
                    try:
                        size = path.stat().st_size
                    except OSError:
                        continue
                    if size >= int(large_file_bytes):
                        large_files.append(
                            {
                                "path": _rel(project_root, path),
                                "size_bytes": size,
                                "root": str(relative_root),
                            }
                        )

    duplicate_groups = _duplicate_groups(project_root, items)
    report = CurationReport(
        generated_at_utc=_utc_now(),
        project_root=project_root,
        roots=root_reports,
        items=sorted(items, key=lambda item: (item.root, item.mtime_utc, str(item.path)), reverse=True),
        large_files=sorted(large_files, key=lambda row: int(row["size_bytes"]), reverse=True),
        duplicate_groups=duplicate_groups,
    )
    return report


def write_curation_report(
    report: CurationReport,
    *,
    output_dir: Path | None = None,
    also_write_root_aliases: bool = False,
) -> CurationReport:
    """Persist JSON, Markdown, and CSV curation reports."""
    project_root = report.project_root
    target_dir = Path(output_dir) if output_dir is not None else project_root / CURATION_REPORT_ROOT / _stamp("data_curation")
    if not target_dir.is_absolute():
        target_dir = project_root / target_dir
    target_dir.mkdir(parents=True, exist_ok=True)
    json_path = target_dir / "data_curation_report.json"
    markdown_path = target_dir / "data_curation_report.md"
    csv_path = target_dir / "data_curation_candidates.csv"

    payload = report.to_dict()
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    markdown_path.write_text(render_curation_markdown(report), encoding="utf-8")
    _write_candidates_csv(report, csv_path)

    if also_write_root_aliases:
        (project_root / "data_curation_report.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
        (project_root / "data_curation_report.md").write_text(render_curation_markdown(report), encoding="utf-8")

    return CurationReport(
        generated_at_utc=report.generated_at_utc,
        project_root=project_root,
        roots=report.roots,
        items=report.items,
        large_files=report.large_files,
        duplicate_groups=report.duplicate_groups,
        report_dir=target_dir,
        json_path=json_path,
        markdown_path=markdown_path,
        csv_path=csv_path,
    )


def render_curation_markdown(report: CurationReport) -> str:
    counts: dict[str, int] = {}
    for item in report.items:
        counts[item.classification] = counts.get(item.classification, 0) + 1
    lines = [
        "# Data Curation Report",
        "",
        f"Generated: {report.generated_at_utc}",
        "",
        "## Classification Summary",
        "",
    ]
    for key in sorted(counts):
        lines.append(f"- {key}: {counts[key]}")
    lines.extend(["", "## Root Summary", ""])
    for root in report.roots:
        status = "present" if root.exists else "missing"
        lines.append(
            f"- {root.root}: {status}, {root.item_count} item(s), {_format_bytes(root.total_size_bytes)}, "
            f"missing_review={root.missing_run_review_count}, mock={root.mock_count}, lower_trust={root.lower_trust_count}, "
            f"thesis_candidate={root.thesis_candidate_count}, advisor_share={root.advisor_share_count}"
        )
    lines.extend(["", "## Large Files", ""])
    if report.large_files:
        for row in report.large_files[:40]:
            lines.append(f"- {row['path']}: {_format_bytes(int(row['size_bytes']))}")
    else:
        lines.append("- none")
    lines.extend(["", "## Duplicate / Near-Duplicate Runs", ""])
    if report.duplicate_groups:
        for group in report.duplicate_groups[:30]:
            lines.append("- " + " | ".join(group))
    else:
        lines.append("- none detected")
    lines.extend(["", "## Candidates", ""])
    for item in report.items:
        if item.classification in {"trash_candidate", "archive_candidate", "keep_candidate", "needs_human_review"}:
            reason = "; ".join(item.reasons) or "n/a"
            lines.append(f"- {item.classification}: {_rel(report.project_root, item.path)} ({reason})")
    return "\n".join(lines) + "\n"


def preview_cleanup(
    project_root: Path,
    *,
    apply_trash_candidates: bool = False,
    apply_archive_candidates: bool = False,
    permanently_delete_trash: bool = False,
    force: bool = False,
    dry_run: bool = True,
    filters: dict[str, Any] | None = None,
) -> CleanupManifest:
    """Preview or apply safe cleanup actions from curation classifications."""
    project_root = Path(project_root).resolve()
    report = build_curation_report(project_root)
    selected = _filter_items(report.items, filters or {})
    actions: list[CleanupAction] = []
    if apply_trash_candidates:
        actions.extend(_plan_move_actions(project_root, selected, classification="trash_candidate", action="trash"))
    if apply_archive_candidates:
        actions.extend(_plan_move_actions(project_root, selected, classification="archive_candidate", action="archive"))
    if permanently_delete_trash:
        actions.extend(_plan_delete_trash_actions(project_root, selected))
    if not (apply_trash_candidates or apply_archive_candidates or permanently_delete_trash):
        actions = _plan_move_actions(project_root, selected, classification="trash_candidate", action="trash")
        actions.extend(_plan_move_actions(project_root, selected, classification="archive_candidate", action="archive"))

    if dry_run:
        manifest = CleanupManifest(
            mode="preview",
            generated_at_utc=_utc_now(),
            action_count=len(actions),
            applied_count=0,
            skipped_count=sum(1 for action in actions if action.status == "skipped"),
            error_count=0,
            actions=actions,
        )
        return write_cleanup_manifest(project_root, manifest)

    applied: list[CleanupAction] = []
    for action in actions:
        if action.status == "skipped":
            applied.append(action)
            continue
        try:
            if action.action == "trash":
                result = trash_run(action.source_path, project_root=project_root, force=force)
                applied.append(
                    CleanupAction(action.action, action.source_path, result.destination_path, action.classification, "applied", action.reason)
                )
            elif action.action == "archive":
                result = archive_run(action.source_path, project_root=project_root, force=force)
                applied.append(
                    CleanupAction(action.action, action.source_path, result.destination_path, action.classification, "applied", action.reason)
                )
            elif action.action == "delete_trash":
                if not force:
                    applied.append(
                        CleanupAction(action.action, action.source_path, None, action.classification, "skipped", "Permanent delete requires --force.")
                    )
                    continue
                _assert_inside_trash(project_root, action.source_path)
                if action.source_path.is_dir():
                    shutil.rmtree(action.source_path)
                elif action.source_path.exists():
                    action.source_path.unlink()
                applied.append(
                    CleanupAction(action.action, action.source_path, None, action.classification, "applied", action.reason)
                )
        except Exception as exc:
            applied.append(CleanupAction(action.action, action.source_path, None, action.classification, "error", str(exc)))
    manifest = CleanupManifest(
        mode="apply",
        generated_at_utc=_utc_now(),
        action_count=len(applied),
        applied_count=sum(1 for action in applied if action.status == "applied"),
        skipped_count=sum(1 for action in applied if action.status == "skipped"),
        error_count=sum(1 for action in applied if action.status == "error"),
        actions=applied,
    )
    return write_cleanup_manifest(project_root, manifest)


def write_cleanup_manifest(project_root: Path, manifest: CleanupManifest) -> CleanupManifest:
    project_root = Path(project_root).resolve()
    out_dir = project_root / CURATION_REPORT_ROOT / _stamp(f"cleanup_{manifest.mode}")
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "cleanup_manifest.json"
    md_path = out_dir / "cleanup_manifest.md"
    json_path.write_text(json.dumps(manifest.to_dict(project_root), indent=2), encoding="utf-8")
    md_path.write_text(_render_cleanup_markdown(project_root, manifest), encoding="utf-8")
    return CleanupManifest(
        mode=manifest.mode,
        generated_at_utc=manifest.generated_at_utc,
        action_count=manifest.action_count,
        applied_count=manifest.applied_count,
        skipped_count=manifest.skipped_count,
        error_count=manifest.error_count,
        actions=manifest.actions,
        manifest_path=json_path,
        summary_path=md_path,
    )


def _discover_items_for_root(project_root: Path, relative_root: Path) -> list[CurationItem]:
    root = project_root / relative_root
    if not root.exists():
        return []
    root_key = str(relative_root)
    if root_key in {"data/experiments", "data/mock_experiments", "data/experiments_archived", "data/trash"}:
        return _discover_run_items(project_root, relative_root)
    if root_key == "data/diagnostics":
        items: list[CurationItem] = []
        for family_root in [path for path in root.iterdir() if path.is_dir()]:
            child_dirs = [path for path in family_root.iterdir() if path.is_dir()]
            if child_dirs:
                items.extend(_classify_generic_item(project_root, relative_root, path) for path in child_dirs)
            else:
                items.append(_classify_generic_item(project_root, relative_root, family_root))
        return items
    if root_key in {"data/exports", "data/modeling_results", "data/models"}:
        return [_classify_generic_item(project_root, relative_root, path) for path in root.iterdir() if not path.name.startswith(".")]
    items: list[CurationItem] = []
    for path in root.iterdir():
        if path.name.startswith("."):
            continue
        if path.is_dir() and ((path / "summary.json").exists() or (path / "metadata.json").exists()):
            items.append(_classify_run_item(project_root, relative_root, path))
        else:
            items.append(_classify_generic_item(project_root, relative_root, path))
    return items


def _discover_run_items(project_root: Path, relative_root: Path) -> list[CurationItem]:
    root = project_root / relative_root
    items: list[CurationItem] = []
    for experiment_root in sorted([path for path in root.iterdir() if path.is_dir()]):
        if _looks_like_run_dir(experiment_root):
            items.append(_classify_run_item(project_root, relative_root, experiment_root))
            continue
        for run_dir in sorted([path for path in experiment_root.iterdir() if path.is_dir()]):
            if _looks_like_run_dir(run_dir):
                items.append(_classify_run_item(project_root, relative_root, run_dir))
    return items


def _classify_run_item(project_root: Path, relative_root: Path, run_dir: Path) -> CurationItem:
    root_key = str(relative_root)
    reasons: list[str] = []
    try:
        summary = summarize_run(run_dir)
        review = summary.review
        experiment_name = summary.experiment_name
        run_id = summary.run_id
        mock_mode = summary.mock_mode
        trust_mode = summary.run_trust_mode
        success = summary.success
        status = summary.status
        has_review = summary.has_run_review
        size_bytes = summary.total_size_bytes
    except Exception as exc:
        review = load_run_review(run_dir)
        experiment_name = run_dir.parent.name
        run_id = run_dir.name
        mock_mode = None
        trust_mode = "unknown"
        success = None
        status = "unknown"
        has_review = (run_dir / REVIEW_FILENAME).exists()
        size_bytes = _path_size(run_dir)
        reasons.append(f"could_not_summarize={exc}")

    classification = "needs_human_review"
    if root_key == "data/trash":
        classification = "trash_candidate"
        reasons.append("already_in_trash")
    elif root_key == "data/mock_experiments":
        classification = "trash_candidate"
        reasons.append("mock_experiment_root")
    elif review.review_status in IMPORTANT_REVIEW_STATUSES:
        classification = "keep_candidate"
        reasons.append(f"review_status={review.review_status}")
    elif review.review_status == "garbage":
        classification = "trash_candidate"
        reasons.append("review_status=garbage")
    elif root_key == "data/experiments_archived":
        classification = "archive_candidate"
        reasons.append("already_archived")
    elif mock_mode is True or trust_mode == "mock":
        classification = "trash_candidate"
        reasons.append("mock_or_mock_trust")
    elif not has_review:
        classification = "needs_human_review"
        reasons.append("missing_run_review")
    elif trust_mode in LOWER_TRUST_MODES or success is False or status in {"failed", "error", "partial_success"}:
        classification = "archive_candidate"
        reasons.append(f"lower_trust_or_failed={trust_mode}/{status}")
    else:
        classification = "needs_human_review"
        reasons.append(f"review_status={review.review_status or 'debug'}")

    duplicate_key = _duplicate_key(experiment_name, run_id, run_dir)
    return CurationItem(
        path=run_dir,
        root=root_key,
        item_type="run",
        classification=classification,
        reasons=reasons,
        size_bytes=size_bytes,
        mtime_utc=_mtime_utc(run_dir),
        experiment_name=experiment_name,
        run_id=run_id,
        review_status=review.review_status,
        has_run_review=has_review,
        mock_mode=mock_mode,
        trust_mode=trust_mode,
        success=success,
        status=status,
        duplicate_key=duplicate_key,
    )


def _classify_generic_item(project_root: Path, relative_root: Path, path: Path) -> CurationItem:
    root_key = str(relative_root)
    reasons: list[str] = []
    classification = "needs_human_review"
    if path.name in PROTECTED_NAMES or path.name.startswith("latest_"):
        classification = "protected_active_alias"
        reasons.append("active_alias")
    elif root_key in GENERATED_ROOTS:
        classification = "generated_ignore"
        reasons.append("generated_root")
    elif root_key == "data/models":
        classification = "keep_candidate"
        reasons.append("trained_model_artifact")
    elif root_key in {"data/calibration", "data/pivot_calibration", "data/runtime_tip_calibration"}:
        classification = "keep_candidate"
        reasons.append("calibration_artifact")
    return CurationItem(
        path=path,
        root=root_key,
        item_type="dir" if path.is_dir() else "file",
        classification=classification,
        reasons=reasons,
        size_bytes=_path_size(path),
        mtime_utc=_mtime_utc(path),
    )


def _root_inventory(project_root: Path, relative_root: Path, items: list[CurationItem]) -> RootInventory:
    mtimes = [item.mtime_utc for item in items if item.mtime_utc]
    return RootInventory(
        root=str(relative_root),
        exists=(project_root / relative_root).exists(),
        item_count=len(items),
        total_size_bytes=sum(item.size_bytes for item in items),
        oldest_mtime_utc=min(mtimes) if mtimes else "",
        newest_mtime_utc=max(mtimes) if mtimes else "",
        missing_run_review_count=sum(1 for item in items if item.item_type == "run" and not item.has_run_review),
        mock_count=sum(1 for item in items if item.mock_mode is True or item.trust_mode == "mock"),
        lower_trust_count=sum(1 for item in items if item.trust_mode in LOWER_TRUST_MODES),
        thesis_candidate_count=sum(1 for item in items if item.review_status == "thesis_candidate"),
        advisor_share_count=sum(1 for item in items if item.review_status == "advisor_share"),
    )


def _filter_items(items: list[CurationItem], filters: dict[str, Any]) -> list[CurationItem]:
    selected = list(items)
    root = str(filters.get("root") or "").strip()
    if root:
        selected = [item for item in selected if item.root == root]
    experiment = str(filters.get("experiment") or "").strip()
    if experiment:
        selected = [item for item in selected if item.experiment_name == experiment or item.path.name == experiment]
    review_status = str(filters.get("review_status") or "").strip()
    if review_status:
        selected = [item for item in selected if item.review_status == review_status]
    older_than = str(filters.get("older_than") or "").strip()
    if older_than:
        selected = [item for item in selected if item.mtime_utc and item.mtime_utc[:10] < older_than]
    if bool(filters.get("mock_only")):
        selected = [item for item in selected if item.mock_mode is True or item.root == "data/mock_experiments" or item.trust_mode == "mock"]
    if bool(filters.get("generated_only")):
        selected = [item for item in selected if item.classification == "generated_ignore"]
    max_size = filters.get("max_size")
    if max_size is not None:
        selected = [item for item in selected if item.size_bytes <= int(max_size)]
    return selected


def _plan_move_actions(project_root: Path, items: list[CurationItem], *, classification: str, action: str) -> list[CleanupAction]:
    actions: list[CleanupAction] = []
    for item in items:
        if item.classification != classification:
            continue
        if item.item_type != "run":
            actions.append(CleanupAction(action, item.path, None, classification, "skipped", "Only run folders are moved by cleanup apply."))
            continue
        if item.root == "data/trash":
            actions.append(CleanupAction(action, item.path, None, classification, "skipped", "Already in data/trash."))
            continue
        if action == "archive" and item.root == "data/experiments_archived":
            actions.append(CleanupAction(action, item.path, None, classification, "skipped", "Already archived."))
            continue
        destination_root = project_root / "data" / ("trash" if action == "trash" else "experiments_archived") / (item.experiment_name or item.path.parent.name)
        actions.append(
            CleanupAction(action, item.path, destination_root / item.path.name, classification, "candidate", "; ".join(item.reasons))
        )
    return actions


def _plan_delete_trash_actions(project_root: Path, items: list[CurationItem]) -> list[CleanupAction]:
    actions: list[CleanupAction] = []
    for item in items:
        if item.root != "data/trash":
            continue
        actions.append(CleanupAction("delete_trash", item.path, None, item.classification, "candidate", "permanent_delete_from_trash"))
    return actions


def _render_cleanup_markdown(project_root: Path, manifest: CleanupManifest) -> str:
    lines = [
        "# Data Cleanup Manifest",
        "",
        f"Mode: {manifest.mode}",
        f"Generated: {manifest.generated_at_utc}",
        f"Actions: {manifest.action_count}",
        f"Applied: {manifest.applied_count}",
        f"Skipped: {manifest.skipped_count}",
        f"Errors: {manifest.error_count}",
        "",
    ]
    for action in manifest.actions:
        target = f" -> {_rel(project_root, action.destination_path)}" if action.destination_path else ""
        lines.append(f"- {action.status}: {action.action} {_rel(project_root, action.source_path)}{target} ({action.reason})")
    return "\n".join(lines) + "\n"


def _write_candidates_csv(report: CurationReport, path: Path) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "classification",
                "root",
                "item_type",
                "path",
                "experiment_name",
                "review_status",
                "trust_mode",
                "mock_mode",
                "size_bytes",
                "reasons",
            ],
        )
        writer.writeheader()
        for item in report.items:
            writer.writerow(
                {
                    "classification": item.classification,
                    "root": item.root,
                    "item_type": item.item_type,
                    "path": _rel(report.project_root, item.path),
                    "experiment_name": item.experiment_name,
                    "review_status": item.review_status,
                    "trust_mode": item.trust_mode,
                    "mock_mode": item.mock_mode,
                    "size_bytes": item.size_bytes,
                    "reasons": "; ".join(item.reasons),
                }
            )


def _duplicate_groups(project_root: Path, items: list[CurationItem]) -> list[list[str]]:
    groups: dict[str, list[CurationItem]] = {}
    for item in items:
        if item.item_type == "run" and item.duplicate_key:
            groups.setdefault(item.duplicate_key, []).append(item)
    return [
        [_rel(project_root, item.path) for item in sorted(group, key=lambda entry: entry.mtime_utc, reverse=True)]
        for group in groups.values()
        if len(group) > 1
    ]


def _duplicate_key(experiment_name: str, run_id: str, run_dir: Path) -> str:
    if run_id and run_id != run_dir.name:
        return f"{experiment_name}:{run_id}"
    summary_path = run_dir / "summary.json"
    if summary_path.exists():
        try:
            digest = hashlib.sha1(summary_path.read_bytes()).hexdigest()[:12]
            return f"{experiment_name}:summary:{digest}"
        except OSError:
            pass
    return f"{experiment_name}:{run_dir.name}"


def _looks_like_run_dir(path: Path) -> bool:
    return path.is_dir() and ((path / "summary.json").exists() or (path / "metadata.json").exists())


def _path_size(path: Path) -> int:
    if path.is_file():
        try:
            return path.stat().st_size
        except OSError:
            return 0
    total = 0
    if path.is_dir():
        for candidate in path.rglob("*"):
            if candidate.is_file():
                try:
                    total += candidate.stat().st_size
                except OSError:
                    continue
    return total


def _mtime_utc(path: Path) -> str:
    try:
        stamp = path.stat().st_mtime
    except OSError:
        return ""
    return datetime.fromtimestamp(stamp, tz=timezone.utc).isoformat()


def _assert_inside_trash(project_root: Path, path: Path) -> None:
    resolved = Path(path).resolve()
    trash = (Path(project_root).resolve() / "data" / "trash").resolve()
    try:
        resolved.relative_to(trash)
    except ValueError as exc:
        raise ValueError(f"Refusing to permanently delete outside data/trash: {path}") from exc
    if resolved == trash:
        raise ValueError("Refusing to permanently delete data/trash root.")


def _rel(project_root: Path, path: Path | None) -> str:
    if path is None:
        return ""
    try:
        return str(Path(path).resolve().relative_to(Path(project_root).resolve()))
    except ValueError:
        return str(path)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _stamp(label: str) -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_") + label


def _format_bytes(size: int) -> str:
    value = float(size)
    for unit in ("B", "KB", "MB", "GB"):
        if value < 1024.0 or unit == "GB":
            return f"{value:.1f} {unit}" if unit != "B" else f"{int(value)} B"
        value /= 1024.0
    return f"{size} B"

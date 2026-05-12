"""Discovery, normalization, migration, and safe deletion helpers for operator data management."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import shutil
from typing import Any

from continuum_robot.experiments.dataset_io import (
    canonical_timestamp_token,
    sanitize_output_name,
)
from continuum_robot.modeling.ann_training import discover_trained_artifacts
from continuum_robot.registration.repository import RegistrationRepository
from continuum_robot.registration.runtime_tip_repository import RuntimeTipCalibrationRepository
from continuum_robot.servos.neutral_calibration_service import NeutralCalibrationService


DIAGNOSTIC_EXPERIMENT_NAMES = {"tracker_timing_validation", "servo_tracker_sync_validation"}
MIGRATION_REPORT_ROOT = "data/diagnostics/data_management_migration"


@dataclass(frozen=True)
class ManagedDataItem:
    """One normalized operator-visible data bundle or artifact."""

    category_key: str
    category_label: str
    item_type: str
    readable_name: str
    timestamp_label: str
    timestamp_sort_key: str
    path: Path
    root_path: Path
    canonical_root_path: Path
    canonical_name: str
    canonical_path: Path | None
    path_kind: str
    status: str = ""
    details: str = ""
    deletable: bool = True
    delete_reason: str = ""
    protected: bool = False
    original_name: str = ""
    original_path: str = ""
    legacy_naming: bool = False
    legacy_root: bool = False
    legacy_reason: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def is_legacy(self) -> bool:
        return bool(self.legacy_naming or self.legacy_root)

    @property
    def display_status(self) -> str:
        flags: list[str] = []
        if self.status not in ("", "saved"):
            flags.append(str(self.status))
        elif self.status:
            flags.append("saved")
        if self.protected:
            flags.append("protected")
        if self.is_legacy:
            flags.append("legacy")
        if not flags:
            return "saved"
        return " | ".join(dict.fromkeys(flags))


@dataclass(frozen=True)
class MigrationEntry:
    """One previewed or applied migration action."""

    category_key: str
    display_name: str
    source_path: Path
    target_path: Path | None
    action: str
    status: str
    reason: str


@dataclass(frozen=True)
class MigrationReport:
    """Dry-run or applied migration ledger."""

    mode: str
    generated_at_utc: str
    scanned_count: int
    candidate_count: int
    applied_count: int
    skipped_count: int
    protected_count: int
    error_count: int
    entries: list[MigrationEntry] = field(default_factory=list)
    report_dir: Path | None = None
    manifest_path: Path | None = None
    summary_path: Path | None = None

    @property
    def actionable_entries(self) -> list[MigrationEntry]:
        return [entry for entry in self.entries if entry.status == "candidate"]


def discover_managed_data(project_root: Path) -> list[ManagedDataItem]:
    """Return all operator-facing data items across the active runtime roots."""
    root = Path(project_root)
    items = [
        *_discover_calibration_items(root),
        *_discover_experiment_items(root),
        *_discover_modeling_items(root),
        *_discover_diagnostic_items(root),
        *_discover_trash_items(root),
    ]
    return sorted(
        items,
        key=lambda item: (item.timestamp_sort_key, item.category_label, item.readable_name, str(item.path)),
        reverse=True,
    )


def filter_managed_data(
    items: list[ManagedDataItem],
    *,
    category_key: str = "all",
    search_text: str = "",
) -> list[ManagedDataItem]:
    """Filter discovered items by category and a simple substring search."""
    selected_category = str(category_key or "all").strip().lower()
    needle = str(search_text or "").strip().lower()
    filtered: list[ManagedDataItem] = []
    for item in items:
        if selected_category not in {"", "all"} and item.category_key != selected_category:
            continue
        if needle:
            haystack = " ".join(
                [
                    item.category_label,
                    item.item_type,
                    item.readable_name,
                    item.display_status,
                    item.details,
                    item.original_name,
                    item.original_path,
                    str(item.path),
                ]
            ).lower()
            if needle not in haystack:
                continue
        filtered.append(item)
    return filtered


def delete_managed_items(project_root: Path, items: list[ManagedDataItem]) -> list[Path]:
    """Delete selected artifact bundles or files, never category roots."""
    project_root = Path(project_root).resolve()
    deleted: list[Path] = []
    for item in items:
        if not item.deletable:
            raise ValueError(f"Deletion is disabled for {item.path.name}: {item.delete_reason or 'protected item'}")
        resolved_path = item.path.resolve()
        resolved_root = item.root_path.resolve()
        try:
            resolved_path.relative_to(project_root)
            resolved_path.relative_to(resolved_root)
        except ValueError as exc:
            raise ValueError(f"Refusing to delete non-project or non-root path: {item.path}") from exc
        if resolved_path == resolved_root:
            raise ValueError(f"Refusing to delete root path: {item.path}")
        if not resolved_path.exists():
            continue
        if resolved_path.is_dir():
            shutil.rmtree(resolved_path)
        else:
            resolved_path.unlink()
        deleted.append(item.path)
    return deleted


def preview_migration(
    project_root: Path,
    items: list[ManagedDataItem],
) -> MigrationReport:
    """Build a dry-run migration ledger for the selected items and persist it."""
    return _write_migration_report(
        project_root=Path(project_root),
        report=_build_migration_report(mode="preview", items=items),
    )


def apply_migration(
    project_root: Path,
    items: list[ManagedDataItem],
) -> MigrationReport:
    """Apply safe migration actions for the selected items and persist the applied ledger."""
    project_root = Path(project_root)
    preview = _build_migration_report(mode="preview", items=items)
    applied_entries: list[MigrationEntry] = []
    applied_count = 0
    error_count = 0
    for entry in preview.entries:
        if entry.status != "candidate":
            applied_entries.append(entry)
            continue
        if entry.target_path is None:
            applied_entries.append(replace(entry, status="skipped", reason="Missing migration target."))
            continue
        try:
            entry.target_path.parent.mkdir(parents=True, exist_ok=True)
            if entry.source_path.exists():
                shutil.move(str(entry.source_path), str(entry.target_path))
            applied_entries.append(replace(entry, status="applied"))
            applied_count += 1
        except Exception as exc:
            applied_entries.append(replace(entry, status="error", reason=str(exc)))
            error_count += 1
    report = MigrationReport(
        mode="apply",
        generated_at_utc=_utc_now_iso(),
        scanned_count=preview.scanned_count,
        candidate_count=preview.candidate_count,
        applied_count=applied_count,
        skipped_count=sum(1 for entry in applied_entries if entry.status == "skipped"),
        protected_count=sum(1 for entry in applied_entries if "protected" in entry.reason.lower()),
        error_count=error_count,
        entries=applied_entries,
    )
    return _write_migration_report(project_root=project_root, report=report)


def build_root_summary(project_root: Path) -> list[tuple[str, str]]:
    """Return the canonical active roots surfaced by the Data Management tab."""
    _ = Path(project_root)
    return [
        (
            "Calibration",
            "data/registrations | data/runtime_tip_calibration | data/pivot_calibration | "
            "config/neutral_setpoints.json | data/calibration/servo_calibration",
        ),
        ("Experiments", "data/experiments/*"),
        ("Modeling / Training", "data/models/ann | data/modeling_results"),
        (
            "Diagnostics",
            "data/diagnostics/* | data/experiments/tracker_timing_validation | "
            "data/experiments/servo_tracker_sync_validation",
        ),
        ("Trash", "data/trash/*"),
        ("Migration Ledgers", "data/diagnostics/data_management_migration"),
    ]


def _discover_calibration_items(project_root: Path) -> list[ManagedDataItem]:
    items: list[ManagedDataItem] = []

    registration_root = project_root / "data" / "registrations"
    repository = RegistrationRepository(root_dir=registration_root)
    latest_registration = registration_root / "latest_registration.json"
    if latest_registration.exists():
        payload = _read_json(latest_registration)
        fre_mm = _payload_registration_fre(payload)
        items.append(
            _build_item(
                category_key="calibration",
                category_label="Calibration",
                item_type="registration_latest",
                readable_name="Latest Accepted Registration",
                path=latest_registration,
                root_path=registration_root,
                canonical_root_path=registration_root,
                canonical_label="registration",
                timestamp_label=_timestamp_from_payload_or_name(payload, latest_registration.name),
                path_kind="file",
                extension=".json",
                status="active",
                details=(f"Latest accepted registration | FRE {fre_mm:.3f} mm" if fre_mm is not None else "Latest accepted registration"),
                deletable=False,
                delete_reason="Active calibration alias",
                protected=True,
                original_name=latest_registration.name,
                treat_current_name_as_canonical=True,
            )
        )
    for path in repository.list_saved_records():
        payload = _read_json(path)
        fre_mm = _payload_registration_fre(payload)
        items.append(
            _build_item(
                category_key="calibration",
                category_label="Calibration",
                item_type="registration",
                readable_name="Registration",
                path=path,
                root_path=registration_root,
                canonical_root_path=registration_root,
                canonical_label="registration",
                timestamp_label=_timestamp_from_payload_or_name(payload, path.name),
                path_kind="file",
                extension=".json",
                status="saved",
                details=(f"Registration | FRE {fre_mm:.3f} mm" if fre_mm is not None else "Registration"),
                original_name=path.name,
                metadata={"fre_mm": fre_mm},
            )
        )

    runtime_tip_root = project_root / "data" / "runtime_tip_calibration"
    for alias_name, display_name, canonical_label, status_label in (
        ("latest_runtime_tip_calibration.json", "Latest Runtime Tip Calibration", "runtime_tip_calibration", "active latest"),
        ("latest_quick_4_point_runtime_tip.json", "Latest Quick 4-Point Runtime Tip", "runtime_tip_quick_4_point", "active quick override"),
    ):
        alias_path = runtime_tip_root / alias_name
        if not alias_path.exists():
            continue
        payload = _read_json(alias_path)
        items.append(
            _build_item(
                category_key="calibration",
                category_label="Calibration",
                item_type="runtime_tip_latest",
                readable_name=display_name,
                path=alias_path,
                root_path=runtime_tip_root,
                canonical_root_path=runtime_tip_root,
                canonical_label=canonical_label,
                timestamp_label=_timestamp_from_payload_or_name(payload, alias_name),
                path_kind="file",
                extension=".json",
                status="active",
                details=f"{display_name} | {status_label}",
                deletable=False,
                delete_reason="Active runtime-tip alias",
                protected=True,
                original_name=alias_name,
                treat_current_name_as_canonical=True,
            )
        )
    for path in sorted(runtime_tip_root.glob("*.json"), reverse=True):
        if path.name in {"latest_runtime_tip_calibration.json", "latest_quick_4_point_runtime_tip.json"}:
            continue
        if not _is_runtime_tip_record_name(path.name):
            continue
        payload = _read_json(path)
        calibration_kind = str(payload.get("calibration_kind", "") or "")
        item_type = "runtime_tip_quick_4_point" if "quick" in calibration_kind.lower() or "quick" in path.name else "runtime_tip_calibration"
        readable_name = "Runtime Tip Quick 4-Point" if item_type == "runtime_tip_quick_4_point" else "Runtime Tip Calibration"
        canonical_label = "runtime_tip_quick_4_point" if item_type == "runtime_tip_quick_4_point" else "runtime_tip_calibration"
        rmse = payload.get("fit_rmse_mm")
        details = readable_name + (f" | RMSE {float(rmse):.3f} mm" if rmse is not None else "")
        items.append(
            _build_item(
                category_key="calibration",
                category_label="Calibration",
                item_type=item_type,
                readable_name=readable_name,
                path=path,
                root_path=runtime_tip_root,
                canonical_root_path=runtime_tip_root,
                canonical_label=canonical_label,
                timestamp_label=_timestamp_from_payload_or_name(payload, path.name),
                path_kind="file",
                extension=".json",
                status="saved",
                details=details,
                original_name=path.name,
                metadata={"fit_rmse_mm": float(rmse) if rmse is not None else None},
            )
        )

    neutral_path = project_root / "config" / "neutral_setpoints.json"
    service = NeutralCalibrationService(path=neutral_path)
    neutral_archive_root = service.archive_root
    if neutral_path.exists():
        summary = service.get_calibration_summary()
        calibrated_count = len(summary.calibrated_servo_ids)
        items.append(
            _build_item(
                category_key="calibration",
                category_label="Calibration",
                item_type="servo_calibration_active",
                readable_name="Active Servo Calibration",
                path=neutral_path,
                root_path=neutral_path.parent,
                canonical_root_path=neutral_path.parent,
                canonical_label="neutral_setpoints",
                timestamp_label=_timestamp_from_payload_or_name(_read_json(neutral_path), neutral_path.name),
                path_kind="file",
                extension=".json",
                status="active",
                details=f"{summary.status} | {calibrated_count} calibrated servo(s)",
                deletable=False,
                delete_reason="Active servo calibration file",
                protected=True,
                original_name=neutral_path.name,
                treat_current_name_as_canonical=True,
            )
        )
    for archive in sorted(neutral_archive_root.glob("*.json"), reverse=True):
        if archive.name == neutral_path.name or not _is_neutral_archive_name(archive.name):
            continue
        payload = _read_json(archive)
        items.append(
            _build_item(
                category_key="calibration",
                category_label="Calibration",
                item_type="servo_calibration_archive",
                readable_name="Servo Calibration Archive",
                path=archive,
                root_path=neutral_archive_root,
                canonical_root_path=neutral_archive_root,
                canonical_label="neutral_setpoints",
                timestamp_label=_timestamp_from_payload_or_name(payload, archive.name),
                path_kind="file",
                extension=".json",
                status="archived",
                details=f"Servo calibration archive | {_servo_count_from_payload(payload)} servo(s)",
                original_name=archive.name,
            )
        )
    if neutral_path.parent.exists():
        for archive in sorted(neutral_path.parent.glob("*.json"), reverse=True):
            if archive.name == neutral_path.name or not _is_neutral_archive_name(archive.name):
                continue
            payload = _read_json(archive)
            items.append(
                _build_item(
                    category_key="calibration",
                    category_label="Calibration",
                    item_type="servo_calibration_archive",
                    readable_name="Servo Calibration Archive",
                    path=archive,
                    root_path=neutral_path.parent,
                    canonical_root_path=neutral_archive_root,
                    canonical_label="neutral_setpoints",
                    timestamp_label=_timestamp_from_payload_or_name(payload, archive.name),
                    path_kind="file",
                    extension=".json",
                    status="archived",
                    details=(
                        f"Servo calibration archive | {_servo_count_from_payload(payload)} servo(s)"
                        " | legacy config root"
                    ),
                    original_name=archive.name,
                )
            )
    items.extend(_discover_pivot_calibration_items(project_root))
    return items


def _discover_experiment_items(project_root: Path) -> list[ManagedDataItem]:
    items: list[ManagedDataItem] = []
    experiments_root = project_root / "data" / "experiments"
    if not experiments_root.exists():
        return items
    for experiment_root in sorted((path for path in experiments_root.iterdir() if path.is_dir()), reverse=True):
        if _looks_like_timestamped_run_dir(experiment_root.name):
            experiment_name = _experiment_name_from_run_dir(experiment_root.name)
            if not experiment_name or experiment_name in DIAGNOSTIC_EXPERIMENT_NAMES or experiment_name == "tracker_validation":
                continue
            items.append(
                _build_experiment_item(
                    run_dir=experiment_root,
                    experiment_name=experiment_name,
                    root_path=experiments_root,
                    canonical_root_path=experiments_root / experiment_name,
                    legacy_note="legacy experiments root",
                )
            )
            continue
        experiment_name = experiment_root.name
        if experiment_name in DIAGNOSTIC_EXPERIMENT_NAMES or experiment_name == "tracker_validation":
            continue
        for run_dir in sorted((path for path in experiment_root.iterdir() if path.is_dir()), reverse=True):
            items.append(
                _build_experiment_item(
                    run_dir=run_dir,
                    experiment_name=experiment_name,
                    root_path=experiment_root,
                    canonical_root_path=experiments_root / experiment_name,
                )
            )
    return items


def _discover_modeling_items(project_root: Path) -> list[ManagedDataItem]:
    items: list[ManagedDataItem] = []
    artifact_root = project_root / "data" / "models" / "ann"
    valid_artifacts = {artifact.path: artifact for artifact in discover_trained_artifacts(artifact_root=artifact_root)}
    if artifact_root.exists():
        for path in sorted((entry for entry in artifact_root.iterdir() if entry.is_dir()), reverse=True):
            display_name, canonical_label, metadata_payload = _ann_artifact_identity(path)
            artifact = valid_artifacts.get(path)
            if artifact is None:
                items.append(
                    _build_item(
                        category_key="modeling",
                        category_label="Modeling / Training",
                        item_type="ann_artifact",
                        readable_name=display_name,
                        path=path,
                        root_path=artifact_root,
                        canonical_root_path=artifact_root,
                        canonical_label=canonical_label,
                        timestamp_label=_timestamp_from_name(path.name),
                        path_kind="dir",
                        extension="",
                        status="invalid",
                        details="Missing or unreadable ANN artifact metadata",
                        original_name=path.name,
                        metadata=metadata_payload,
                    )
                )
                continue
            validation_loss = (
                f"{artifact.best_validation_loss:.6f}"
                if artifact.best_validation_loss is not None
                else "n/a"
            )
            items.append(
                _build_item(
                    category_key="modeling",
                    category_label="Modeling / Training",
                    item_type="ann_artifact",
                    readable_name=display_name,
                    path=artifact.path,
                    root_path=artifact_root,
                    canonical_root_path=artifact_root,
                    canonical_label=canonical_label,
                    timestamp_label=_timestamp_from_payload_or_name({"created_at_utc": artifact.created_at_utc}, artifact.path.name),
                    path_kind="dir",
                    extension="",
                    status=artifact.status,
                    details=f"{artifact.dataset_name} | {artifact.backend_name} | best val {validation_loss}",
                    original_name=artifact.path.name,
                    metadata={"dataset_name": artifact.dataset_name, **metadata_payload},
                )
            )

    results_root = project_root / "data" / "modeling_results"
    if results_root.exists():
        for path in sorted((entry for entry in results_root.iterdir() if entry.is_dir()), reverse=True):
            summary_path = path / "summary.json"
            metadata_path = path / "evaluation_metadata.json"
            readable_name = "Modeling Comparison"
            canonical_label = "modeling_results"
            timestamp_label = _timestamp_from_name(path.name)
            status = "invalid"
            details = "Missing summary.json"
            metadata: dict[str, Any] = {}
            if summary_path.exists():
                summary = _read_json(summary_path)
                metadata = _read_json(metadata_path) if metadata_path.exists() else {}
                dataset_name = (
                    str(metadata.get("dataset", {}).get("run_name", "") or "")
                    if isinstance(metadata.get("dataset"), dict)
                    else ""
                )
                readable_name = dataset_name or "Modeling Comparison"
                canonical_label = sanitize_output_name(dataset_name or "modeling_results", default="modeling_results")
                timestamp_label = _timestamp_from_payload_or_name(metadata, path.name)
                status = str(summary.get("status", metadata.get("status", "")) or "saved")
                details = readable_name
                scope = metadata.get("evaluation_scope_used")
                if scope:
                    details += f" | {scope}"
            items.append(
                _build_item(
                    category_key="modeling",
                    category_label="Modeling / Training",
                    item_type="modeling_results",
                    readable_name=readable_name,
                    path=path,
                    root_path=results_root,
                    canonical_root_path=results_root,
                    canonical_label=canonical_label,
                    timestamp_label=timestamp_label,
                    path_kind="dir",
                    extension="",
                    status=status,
                    details=details,
                    original_name=path.name,
                    metadata=metadata,
                )
            )
    return items


def _discover_trash_items(project_root: Path) -> list[ManagedDataItem]:
    items: list[ManagedDataItem] = []
    trash_root = project_root / "data" / "trash"
    if not trash_root.exists():
        return items
    for experiment_root in sorted((path for path in trash_root.iterdir() if path.is_dir()), reverse=True):
        experiment_name = experiment_root.name
        for run_dir in sorted((path for path in experiment_root.iterdir() if path.is_dir()), reverse=True):
            metadata_path = run_dir / "metadata.json"
            summary_path = run_dir / "summary.json"
            readable_name = f"Trash: {_humanize_name(experiment_name)}"
            timestamp_label = _timestamp_from_name(run_dir.name)
            details = "Trashed run; permanent delete is available only from data/trash."
            status = "trash"
            metadata_payload: dict[str, Any] = {"experiment_name": experiment_name}
            if metadata_path.exists() and summary_path.exists():
                metadata = _read_json(metadata_path)
                summary = _read_json(summary_path)
                run_experiment_name = str(metadata.get("experiment_name", experiment_name) or experiment_name)
                readable_name = f"Trash: {_humanize_name(run_experiment_name)}"
                timestamp_label = _timestamp_from_payload_or_name(metadata, run_dir.name)
                details = _experiment_details(metadata, summary) + " | trashed run"
                metadata_payload = {"experiment_name": run_experiment_name, **metadata}
            items.append(
                _build_item(
                    category_key="trash",
                    category_label="Trash",
                    item_type=experiment_name,
                    readable_name=readable_name,
                    path=run_dir,
                    root_path=trash_root,
                    canonical_root_path=trash_root / experiment_name,
                    canonical_label=experiment_name,
                    timestamp_label=timestamp_label,
                    path_kind="dir",
                    extension="",
                    status=status,
                    details=details,
                    original_name=run_dir.name,
                    treat_current_name_as_canonical=True,
                    metadata=metadata_payload,
                )
            )
    return items


def _discover_diagnostic_items(project_root: Path) -> list[ManagedDataItem]:
    items: list[ManagedDataItem] = []
    diagnostics_root = project_root / "data" / "diagnostics"
    if diagnostics_root.exists():
        for family_root in sorted((path for path in diagnostics_root.iterdir() if path.is_dir()), reverse=True):
            for run_dir in sorted((path for path in family_root.iterdir() if path.is_dir()), reverse=True):
                summary_path = run_dir / "summary.json"
                report_path = run_dir / "tracker_validation_report.json"
                readable_name = _humanize_name(family_root.name)
                canonical_label = family_root.name
                timestamp_label = _timestamp_from_name(run_dir.name)
                details = "Missing summary.json or tracker_validation_report.json"
                status = "invalid"
                metadata_payload: dict[str, Any] = {}
                if summary_path.exists():
                    summary = _read_json(summary_path)
                    metadata_payload = dict(summary.get("metadata", {}) or {})
                    readable_name, canonical_label = _diagnostic_identity(family_root.name, summary, run_dir.name)
                    timestamp_label = _timestamp_from_payload_or_name(metadata_payload, run_dir.name)
                    details = _diagnostic_details(family_root.name, summary)
                    status = "saved"
                elif report_path.exists():
                    report = _read_json(report_path)
                    readable_name = "Tracker Validation"
                    canonical_label = "tracker_validation"
                    timestamp_label = _timestamp_from_name(run_dir.name)
                    details = _tracker_validation_details(report)
                    status = "saved"
                items.append(
                    _build_item(
                        category_key="diagnostics",
                        category_label="Diagnostics",
                        item_type=family_root.name,
                        readable_name=readable_name,
                        path=run_dir,
                        root_path=family_root,
                        canonical_root_path=family_root,
                        canonical_label=canonical_label,
                        timestamp_label=timestamp_label,
                        path_kind="dir",
                        extension="",
                        status=status,
                        details=details,
                        original_name=run_dir.name,
                        metadata=metadata_payload,
                    )
                )

    canonical_tracker_root = project_root / "data" / "diagnostics" / "tracker_validation"
    for legacy_root, note in (
        (project_root / "data" / "experiments" / "tracker_validation", "legacy experiments root"),
        (project_root / "data" / "tracker_validations", "legacy tracker_validations root"),
    ):
        if not legacy_root.exists():
            continue
        for run_dir in sorted((path for path in legacy_root.iterdir() if path.is_dir()), reverse=True):
            report_path = run_dir / "tracker_validation_report.json"
            report = _read_json(report_path) if report_path.exists() else {}
            items.append(
                _build_item(
                    category_key="diagnostics",
                    category_label="Diagnostics",
                    item_type="tracker_validation",
                    readable_name="Tracker Validation",
                    path=run_dir,
                    root_path=legacy_root,
                    canonical_root_path=canonical_tracker_root,
                    canonical_label="tracker_validation",
                    timestamp_label=_timestamp_from_name(run_dir.name),
                    path_kind="dir",
                    extension="",
                    status=("saved" if report else "invalid"),
                    details=(
                        _tracker_validation_details(report) + f" | {note}"
                        if report
                        else f"Missing tracker_validation_report.json | {note}"
                    ),
                    original_name=run_dir.name,
                )
            )
        for report_path in sorted((path for path in legacy_root.glob("*.json") if path.is_file()), reverse=True):
            report = _read_json(report_path)
            stamp = _timestamp_from_payload_or_name(report, report_path.name)
            items.append(
                _build_item(
                    category_key="diagnostics",
                    category_label="Diagnostics",
                    item_type="tracker_validation",
                    readable_name="Tracker Validation",
                    path=report_path,
                    root_path=legacy_root,
                    canonical_root_path=canonical_tracker_root,
                    canonical_label="tracker_validation",
                    timestamp_label=stamp,
                    path_kind="file",
                    extension=".json",
                    status="saved",
                    details=_tracker_validation_details(report) + f" | {note}",
                    original_name=report_path.name,
                    canonical_path_override=(
                        canonical_tracker_root / f"{stamp}_tracker_validation" / "tracker_validation_report.json"
                        if stamp
                        else None
                    ),
                )
            )

    experiments_root = project_root / "data" / "experiments"
    for experiment_name in sorted(DIAGNOSTIC_EXPERIMENT_NAMES):
        experiment_root = experiments_root / experiment_name
        if not experiment_root.exists():
            continue
        for run_dir in sorted((path for path in experiment_root.iterdir() if path.is_dir()), reverse=True):
            metadata_path = run_dir / "metadata.json"
            summary_path = run_dir / "summary.json"
            readable_name = _humanize_name(experiment_name)
            timestamp_label = _timestamp_from_name(run_dir.name)
            status = "invalid"
            details = "Missing metadata.json or summary.json"
            if metadata_path.exists() and summary_path.exists():
                metadata = _read_json(metadata_path)
                summary = _read_json(summary_path)
                readable_name = _humanize_name(str(metadata.get("experiment_name", experiment_name) or experiment_name))
                timestamp_label = _timestamp_from_payload_or_name(metadata, run_dir.name)
                status = str(summary.get("status", "") or "saved")
                details = _experiment_details(metadata, summary)
            items.append(
                _build_item(
                    category_key="diagnostics",
                    category_label="Diagnostics",
                    item_type=experiment_name,
                    readable_name=readable_name,
                    path=run_dir,
                    root_path=experiment_root,
                    canonical_root_path=experiments_root / experiment_name,
                    canonical_label=experiment_name,
                    timestamp_label=timestamp_label,
                    path_kind="dir",
                    extension="",
                    status=status,
                    details=details,
                    original_name=run_dir.name,
                )
            )
    return items


def _discover_pivot_calibration_items(project_root: Path) -> list[ManagedDataItem]:
    items: list[ManagedDataItem] = []
    pivot_root = project_root / "data" / "pivot_calibration"
    if not pivot_root.exists():
        return items

    accepted_tip_path = pivot_root / "generated_penprobe_tip.csv"
    if accepted_tip_path.exists():
        items.append(
            _build_item(
                category_key="calibration",
                category_label="Calibration",
                item_type="pivot_tip_active",
                readable_name="Accepted Pivot Tip File",
                path=accepted_tip_path,
                root_path=pivot_root,
                canonical_root_path=pivot_root,
                canonical_label="generated_penprobe_tip",
                timestamp_label=_timestamp_from_name(accepted_tip_path.name),
                path_kind="file",
                extension=".csv",
                status="active",
                details="Accepted 0B tip file",
                deletable=False,
                delete_reason="Active pivot tip file",
                protected=True,
                original_name=accepted_tip_path.name,
                treat_current_name_as_canonical=True,
            )
        )

    captures_root = pivot_root / "captures"
    if captures_root.exists():
        for path in sorted((entry for entry in captures_root.iterdir() if entry.is_file()), reverse=True):
            if path.suffix.lower() != ".csv":
                continue
            items.append(
                _build_item(
                    category_key="calibration",
                    category_label="Calibration",
                    item_type="pivot_capture",
                    readable_name="Pivot Capture CSV",
                    path=path,
                    root_path=captures_root,
                    canonical_root_path=captures_root,
                    canonical_label="pivot_0B_samples",
                    timestamp_label=_timestamp_from_name(path.name),
                    path_kind="file",
                    extension=".csv",
                    status="saved",
                    details="Raw tracker-driven pivot capture",
                    original_name=path.name,
                )
            )

    staged_root = pivot_root / "staged"
    if staged_root.exists():
        for path in sorted((entry for entry in staged_root.iterdir() if entry.is_file()), reverse=True):
            if path.suffix.lower() != ".csv":
                continue
            items.append(
                _build_item(
                    category_key="calibration",
                    category_label="Calibration",
                    item_type="pivot_tip_staged",
                    readable_name="Staged Pivot Tip File",
                    path=path,
                    root_path=staged_root,
                    canonical_root_path=staged_root,
                    canonical_label="generated_penprobe_tip",
                    timestamp_label=_timestamp_from_name(path.name),
                    path_kind="file",
                    extension=".csv",
                    status="staged",
                    details="Pending pivot tip acceptance",
                    original_name=path.name,
                )
            )

    for run_dir in sorted((entry for entry in pivot_root.iterdir() if entry.is_dir()), reverse=True):
        if run_dir.name in {"captures", "staged"}:
            continue
        metadata_path = run_dir / "metadata.json"
        summary_path = run_dir / "summary.json"
        readable_name = "Pivot Calibration"
        canonical_label = "pivot_calibration_review" if "review" in run_dir.name else "pivot_calibration"
        timestamp_label = _timestamp_from_name(run_dir.name)
        status = "invalid"
        details = "Missing metadata.json or summary.json"
        metadata_payload: dict[str, Any] = {}
        if metadata_path.exists() and summary_path.exists():
            metadata_payload = _read_json(metadata_path)
            summary_payload = _read_json(summary_path)
            metrics = summary_payload.get("experiment_metrics", {})
            if not isinstance(metrics, dict):
                metrics = {}
            timestamp_label = _timestamp_from_payload_or_name(metadata_payload, run_dir.name)
            status = str(summary_payload.get("status", "") or "saved")
            if "review" in run_dir.name:
                readable_name = "Pivot Calibration Review"
                canonical_label = "pivot_calibration_review"
            else:
                readable_name = "Pivot Calibration"
                canonical_label = "pivot_calibration"
            rmse = metrics.get("rmse_mm")
            used = metrics.get("sample_count_used")
            rejected = metrics.get("sample_count_rejected")
            details = readable_name
            if rmse is not None:
                details += f" | RMSE {float(rmse):.3f} mm"
            if used is not None or rejected is not None:
                details += f" | {int(used or 0)} used / {int(rejected or 0)} rejected"
        items.append(
            _build_item(
                category_key="calibration",
                category_label="Calibration",
                item_type="pivot_calibration_run",
                readable_name=readable_name,
                path=run_dir,
                root_path=pivot_root,
                canonical_root_path=pivot_root,
                canonical_label=canonical_label,
                timestamp_label=timestamp_label,
                path_kind="dir",
                extension="",
                status=status,
                details=details,
                original_name=run_dir.name,
                metadata=metadata_payload,
            )
        )
    return items


def _build_experiment_item(
    *,
    run_dir: Path,
    experiment_name: str,
    root_path: Path,
    canonical_root_path: Path,
    legacy_note: str = "",
) -> ManagedDataItem:
    metadata_path = run_dir / "metadata.json"
    summary_path = run_dir / "summary.json"
    readable_name = _humanize_name(experiment_name)
    timestamp_label = _timestamp_from_name(run_dir.name)
    details = "Missing metadata.json or summary.json"
    status = "invalid"
    metadata_payload: dict[str, Any] = {"experiment_name": experiment_name}
    if metadata_path.exists() and summary_path.exists():
        metadata = _read_json(metadata_path)
        summary = _read_json(summary_path)
        readable_name = _humanize_name(str(metadata.get("experiment_name", experiment_name) or experiment_name))
        timestamp_label = _timestamp_from_payload_or_name(metadata, run_dir.name)
        details = _experiment_details(metadata, summary)
        status = str(summary.get("status", "") or "saved")
        metadata_payload = {"experiment_name": experiment_name, **metadata}
    if legacy_note:
        details = f"{details} | {legacy_note}"
    return _build_item(
        category_key="experiments",
        category_label="Experiments",
        item_type=experiment_name,
        readable_name=readable_name,
        path=run_dir,
        root_path=root_path,
        canonical_root_path=canonical_root_path,
        canonical_label=experiment_name,
        timestamp_label=timestamp_label,
        path_kind="dir",
        extension="",
        status=status,
        details=details,
        original_name=run_dir.name,
        metadata=metadata_payload,
    )


def _build_item(
    *,
    category_key: str,
    category_label: str,
    item_type: str,
    readable_name: str,
    path: Path,
    root_path: Path,
    canonical_root_path: Path,
    canonical_label: str,
    timestamp_label: str,
    path_kind: str,
    extension: str,
    status: str = "",
    details: str = "",
    deletable: bool = True,
    delete_reason: str = "",
    protected: bool = False,
    original_name: str = "",
    treat_current_name_as_canonical: bool = False,
    canonical_path_override: Path | None = None,
    metadata: dict[str, Any] | None = None,
) -> ManagedDataItem:
    canonical_path = canonical_path_override or _proposed_canonical_path(
        canonical_root_path,
        timestamp_label=timestamp_label,
        canonical_label=canonical_label,
        extension=extension,
        current_path=path,
    )
    legacy_naming = (
        False
        if treat_current_name_as_canonical
        else not _matches_canonical_pattern(
            name=path.name,
            timestamp_label=timestamp_label,
            canonical_label=canonical_label,
            extension=extension,
        )
    )
    legacy_root = path.parent != canonical_root_path
    legacy_reason_parts: list[str] = []
    if legacy_naming:
        legacy_reason_parts.append("legacy name")
    if legacy_root:
        legacy_reason_parts.append("legacy root")
    return ManagedDataItem(
        category_key=category_key,
        category_label=category_label,
        item_type=item_type,
        readable_name=readable_name,
        timestamp_label=timestamp_label or "unknown",
        timestamp_sort_key=timestamp_label or "00000000_000000",
        path=Path(path),
        root_path=Path(root_path),
        canonical_root_path=Path(canonical_root_path),
        canonical_name=(path.name if treat_current_name_as_canonical else (canonical_path.name if canonical_path is not None else "")),
        canonical_path=(path if treat_current_name_as_canonical else canonical_path),
        path_kind=path_kind,
        status=status,
        details=details,
        deletable=deletable,
        delete_reason=delete_reason,
        protected=protected,
        original_name=(original_name or path.name),
        original_path=(str(path) if (legacy_naming or legacy_root) else ""),
        legacy_naming=legacy_naming,
        legacy_root=legacy_root,
        legacy_reason=", ".join(legacy_reason_parts),
        metadata=dict(metadata or {}),
    )


def _build_migration_report(*, mode: str, items: list[ManagedDataItem]) -> MigrationReport:
    entries: list[MigrationEntry] = []
    for item in items:
        if item.protected:
            entries.append(
                MigrationEntry(
                    category_key=item.category_key,
                    display_name=item.readable_name,
                    source_path=item.path,
                    target_path=None,
                    action="skip",
                    status="skipped",
                    reason=f"Protected: {item.delete_reason or 'alias or active artifact'}",
                )
            )
            continue
        if not item.is_legacy:
            continue
        if item.canonical_path is None or not item.timestamp_label or item.timestamp_label == "unknown":
            entries.append(
                MigrationEntry(
                    category_key=item.category_key,
                    display_name=item.readable_name,
                    source_path=item.path,
                    target_path=None,
                    action="skip",
                    status="skipped",
                    reason="Legacy artifact is ambiguous; no trusted canonical timestamp/target could be derived.",
                )
            )
            continue
        action = "move" if item.path.parent != item.canonical_path.parent else "rename"
        entries.append(
            MigrationEntry(
                category_key=item.category_key,
                display_name=item.readable_name,
                source_path=item.path,
                target_path=item.canonical_path,
                action=action,
                status="candidate",
                reason=item.legacy_reason or "Legacy artifact",
            )
        )
    return MigrationReport(
        mode=mode,
        generated_at_utc=_utc_now_iso(),
        scanned_count=len(items),
        candidate_count=sum(1 for entry in entries if entry.status == "candidate"),
        applied_count=0,
        skipped_count=sum(1 for entry in entries if entry.status == "skipped"),
        protected_count=sum(1 for entry in entries if "protect" in entry.reason.lower()),
        error_count=0,
        entries=entries,
    )


def _write_migration_report(*, project_root: Path, report: MigrationReport) -> MigrationReport:
    root = Path(project_root) / MIGRATION_REPORT_ROOT
    root.mkdir(parents=True, exist_ok=True)
    report_dir = _proposed_canonical_path(
        root,
        timestamp_label=canonical_timestamp_token(report.generated_at_utc),
        canonical_label=f"data_management_migration_{report.mode}",
        extension="",
        current_path=None,
    )
    if report_dir is None:
        raise RuntimeError("Failed to allocate data-management migration report directory.")
    report_dir.mkdir(parents=True, exist_ok=False)
    manifest_path = report_dir / "migration_manifest.json"
    summary_path = report_dir / "migration_summary.txt"
    manifest_payload = {
        "mode": report.mode,
        "generated_at_utc": report.generated_at_utc,
        "scanned_count": report.scanned_count,
        "candidate_count": report.candidate_count,
        "applied_count": report.applied_count,
        "skipped_count": report.skipped_count,
        "protected_count": report.protected_count,
        "error_count": report.error_count,
        "entries": [
            {
                "category_key": entry.category_key,
                "display_name": entry.display_name,
                "source_path": str(entry.source_path),
                "target_path": (str(entry.target_path) if entry.target_path is not None else None),
                "action": entry.action,
                "status": entry.status,
                "reason": entry.reason,
            }
            for entry in report.entries
        ],
    }
    manifest_path.write_text(json.dumps(manifest_payload, indent=2), encoding="utf-8")
    summary_lines = [
        f"mode={report.mode}",
        f"generated_at_utc={report.generated_at_utc}",
        f"scanned_count={report.scanned_count}",
        f"candidate_count={report.candidate_count}",
        f"applied_count={report.applied_count}",
        f"skipped_count={report.skipped_count}",
        f"protected_count={report.protected_count}",
        f"error_count={report.error_count}",
    ]
    for entry in report.entries[:20]:
        target = f" -> {entry.target_path}" if entry.target_path is not None else ""
        summary_lines.append(f"{entry.status}: {entry.source_path}{target} ({entry.reason})")
    summary_path.write_text("\n".join(summary_lines) + "\n", encoding="utf-8")
    return replace(report, report_dir=report_dir, manifest_path=manifest_path, summary_path=summary_path)


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _proposed_canonical_path(
    root: Path,
    *,
    timestamp_label: str,
    canonical_label: str,
    extension: str,
    current_path: Path | None,
) -> Path | None:
    root = Path(root)
    stamp = str(timestamp_label or "").strip()
    label = sanitize_output_name(canonical_label, default="")
    if not stamp or stamp == "unknown" or not label:
        return None
    suffix = str(extension or "")
    if current_path is not None and current_path.parent == root and _matches_canonical_pattern(
        name=current_path.name,
        timestamp_label=stamp,
        canonical_label=label,
        extension=suffix,
    ):
        return current_path
    base_stem = f"{stamp}_{label}"
    candidate = root / f"{base_stem}{suffix}"
    if current_path is not None and current_path == candidate:
        return candidate
    if not candidate.exists():
        return candidate
    suffix_index = 1
    while True:
        named = root / f"{base_stem}_{suffix_index:02d}{suffix}"
        if current_path is not None and current_path == named:
            return named
        if not named.exists():
            return named
        suffix_index += 1


def _matches_canonical_pattern(
    *,
    name: str,
    timestamp_label: str,
    canonical_label: str,
    extension: str,
) -> bool:
    stamp = str(timestamp_label or "").strip()
    label = sanitize_output_name(canonical_label, default="")
    if not stamp or stamp == "unknown" or not label:
        return False
    pattern = rf"^{re.escape(stamp)}_{re.escape(label)}(?:_\d{{2}})?{re.escape(str(extension or ''))}$"
    return bool(re.match(pattern, str(name or "")))


def _timestamp_from_payload_or_name(payload: dict[str, Any], name: str) -> str:
    for key in ("timestamp_utc", "created_at_utc", "updated_at_utc"):
        value = payload.get(key)
        if value:
            return canonical_timestamp_token(str(value))
    metadata = payload.get("metadata")
    if isinstance(metadata, dict):
        for key in ("timestamp_utc", "created_at_utc", "updated_at_utc"):
            value = metadata.get(key)
            if value:
                return canonical_timestamp_token(str(value))
    return _timestamp_from_name(name)


def _timestamp_from_name(name: str) -> str:
    match = re.search(r"(\d{8})[T_]?(\d{6})", str(name or ""))
    if not match:
        return ""
    return f"{match.group(1)}_{match.group(2)}"


def _looks_like_timestamped_run_dir(name: str) -> bool:
    return bool(re.match(r"^\d{8}_\d{6}_.+", str(name or "")))


def _experiment_name_from_run_dir(name: str) -> str:
    raw = str(name or "")
    if not _looks_like_timestamped_run_dir(raw):
        return ""
    stripped = re.sub(r"^\d{8}_\d{6}_", "", raw)
    stripped = re.sub(r"_\d{2}$", "", stripped)
    return sanitize_output_name(stripped, default="")


def _payload_registration_fre(payload: dict[str, Any]) -> float | None:
    validation_metrics = dict(payload.get("validation_metrics", {}) or {})
    fre = validation_metrics.get("overall_fre_mm", payload.get("fre_mm"))
    return float(fre) if fre not in (None, "") else None


def _servo_count_from_payload(payload: dict[str, Any]) -> int:
    servos = payload.get("servos")
    return len(servos) if isinstance(servos, dict) else 0


def _ann_artifact_identity(path: Path) -> tuple[str, str, dict[str, Any]]:
    training_config_path = path / "training_config.json"
    payload: dict[str, Any] = {}
    if training_config_path.exists():
        try:
            payload = _read_json(training_config_path)
        except Exception:
            payload = {}
    artifact_name = str(payload.get("artifact_name", "") or "").strip()
    if artifact_name:
        return artifact_name, sanitize_output_name(artifact_name, default="ann_training"), payload
    stripped = re.sub(r"^\d{8}_\d{6}_", "", path.name)
    stripped = re.sub(r"_\d{2}$", "", stripped)
    humanized = _humanize_name(stripped or "ann_training")
    return humanized, sanitize_output_name(stripped or "ann_training", default="ann_training"), payload


def _diagnostic_identity(family_name: str, summary_payload: dict[str, Any], run_name: str) -> tuple[str, str]:
    if family_name == "servo_telemetry":
        results = list(summary_payload.get("results") or [])
        profile_names = [sanitize_output_name(str(result.get("profile_name", "") or ""), default="live") for result in results if result.get("profile_name")]
        metadata = dict(summary_payload.get("metadata", {}) or {})
        servo_ids = metadata.get("servo_ids") or []
        servo_count = len(servo_ids) if isinstance(servo_ids, list) and servo_ids else _servo_count_from_run_name(run_name)
        readable = "Servo Telemetry"
        if profile_names:
            readable += f" ({', '.join(profile_names)})"
        label = "servo_telemetry"
        if profile_names:
            label += "_" + "_".join(profile_names)
        if servo_count:
            label += f"_{servo_count}servos"
        return readable, label
    if family_name == "tracker_validation":
        return "Tracker Validation", "tracker_validation"
    return _humanize_name(family_name), family_name


def _servo_count_from_run_name(name: str) -> int:
    match = re.search(r"(\d+)servos", str(name or ""))
    return int(match.group(1)) if match else 0


def _experiment_details(metadata: dict[str, Any], summary: dict[str, Any]) -> str:
    experiment_name = str(metadata.get("experiment_name", "") or "")
    status = str(summary.get("status", "") or "").strip()
    metrics = summary.get("experiment_metrics", {})
    details = _humanize_name(experiment_name) if experiment_name else "Experiment"
    if status and status != "saved":
        details += f" | {status}"
    if isinstance(metrics, dict):
        if metrics.get("sample_count_used") is not None:
            details += f" | {int(metrics.get('sample_count_used', 0) or 0)} used"
        elif metrics.get("sample_count") is not None:
            details += f" | {int(metrics.get('sample_count', 0) or 0)} samples"
        elif metrics.get("accepted_sample_count") is not None:
            details += f" | {int(metrics.get('accepted_sample_count', 0) or 0)} accepted"
    return details.strip(" |")


def _diagnostic_details(family_name: str, summary_payload: dict[str, Any]) -> str:
    if family_name == "servo_telemetry":
        results = list(summary_payload.get("results") or [])
        if not results:
            return "Servo telemetry diagnostic"
        profile_names = [str(result.get("profile_name", "") or "") for result in results if result.get("profile_name")]
        baudrate = (summary_payload.get("metadata") or {}).get("baudrate")
        detail = "Servo telemetry"
        if profile_names:
            detail += f" | {', '.join(profile_names)}"
        if baudrate:
            detail += f" | {int(baudrate)} baud"
        return detail
    return f"{_humanize_name(family_name)} diagnostic"


def _tracker_validation_details(report: dict[str, Any]) -> str:
    tracker_ready = report.get("tracker_ready")
    effective_rate = report.get("effective_frame_rate_hz")
    parts = ["Tracker validation"]
    if tracker_ready is not None:
        parts.append("ready" if bool(tracker_ready) else "not ready")
    if effective_rate not in (None, ""):
        parts.append(f"{float(effective_rate):.2f} Hz")
    return " | ".join(parts)


def _is_runtime_tip_record_name(name: str) -> bool:
    if name.startswith("runtime_tip_calibration_") and name.endswith(".json"):
        return True
    if name.startswith("runtime_tip_quick_4_point_") and name.endswith(".json"):
        return True
    return name.endswith("_runtime_tip_calibration.json") or name.endswith("_runtime_tip_quick_4_point.json")


def _is_neutral_archive_name(name: str) -> bool:
    return (
        name.startswith("neutral_setpoints_")
        or name.endswith("_neutral_setpoints.json")
        or name.endswith("_neutral.json")
    )


def _humanize_name(value: str) -> str:
    parts = [segment for segment in re.split(r"[_\-\s]+", str(value or "").strip()) if segment]
    if not parts:
        return "Artifact"
    rendered: list[str] = []
    for part in parts:
        if part.upper() in {"ANN", "MPS", "CPU", "CUDA", "FRE"}:
            rendered.append(part.upper())
        elif part.isdigit():
            rendered.append(part)
        else:
            rendered.append(part.capitalize())
    return " ".join(rendered)


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

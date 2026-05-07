"""Controller for the operator-facing Data Management workspace."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from continuum_robot.data.build_thesis_evidence_index import build_thesis_evidence_index
from continuum_robot.data.export_run_bundle import ExportBundleResult, build_transfer_commands, export_run_bundle
from continuum_robot.data.run_management import (
    MoveRunResult,
    REVIEW_STATUSES,
    archive_run,
    detail_pairs_for_run,
    discover_experiment_run_dirs,
    summarize_run,
    trash_run,
    write_run_review,
)
from continuum_robot.data.validate_run_bundle import render_validation_report, validate_run_folder
from continuum_robot.data_management import (
    ManagedDataItem,
    build_root_summary,
    delete_managed_items,
    discover_managed_data,
    filter_managed_data,
    apply_migration,
    preview_migration,
    MigrationReport,
)


@dataclass
class DataManagementViewState:
    """GUI-facing state for the data browser."""

    items: list[ManagedDataItem] = field(default_factory=list)
    filtered_items: list[ManagedDataItem] = field(default_factory=list)
    selected_paths: list[str] = field(default_factory=list)
    root_summary_pairs: list[tuple[str, str]] = field(default_factory=list)
    detail_pairs: list[tuple[str, str]] = field(default_factory=list)
    category_filter: str = "all"
    search_text: str = ""
    status_message: str = "Browse saved calibration, experiment, modeling, and diagnostic artifacts."
    can_open: bool = False
    can_reveal: bool = False
    can_copy_path: bool = False
    can_delete: bool = False
    selected_delete_summary: str = ""
    migration_summary_pairs: list[tuple[str, str]] = field(default_factory=list)
    last_migration_report_path: str | None = None
    can_preview_migration: bool = False
    can_apply_migration: bool = False
    can_export_selected: bool = False
    can_export_latest: bool = False
    can_copy_export_path: bool = False
    can_copy_transfer_command: bool = False
    last_export_path: str | None = None
    last_transfer_command: str = ""
    can_validate_selected_run: bool = False
    can_validate_latest_run: bool = False
    can_validate_experiment_runs: bool = False
    can_mark_selected_run: bool = False
    can_archive_selected_run: bool = False
    can_trash_selected_run: bool = False
    can_build_evidence_index: bool = True
    review_status_options: list[str] = field(default_factory=lambda: sorted(REVIEW_STATUSES))
    last_evidence_index_path: str | None = None


class DataManagementController:
    """Owns the normalized data catalog and safe deletion flow."""

    def __init__(self, *, project_root: Path) -> None:
        self.project_root = Path(project_root)
        self.state = DataManagementViewState(root_summary_pairs=build_root_summary(self.project_root))
        self._catalog_dirty = True
        self._last_migration_report: MigrationReport | None = None

    def refresh(self) -> DataManagementViewState:
        if self._catalog_dirty:
            self.state.items = discover_managed_data(self.project_root)
            self._catalog_dirty = False
        self.state.filtered_items = filter_managed_data(
            self.state.items,
            category_key=self.state.category_filter,
            search_text=self.state.search_text,
        )
        visible_paths = {str(item.path) for item in self.state.filtered_items}
        self.state.selected_paths = [path for path in self.state.selected_paths if path in visible_paths]
        selected_items = self.selected_items()
        self.state.root_summary_pairs = build_root_summary(self.project_root)
        self.state.detail_pairs = _build_detail_pairs(self.project_root, selected_items)
        self.state.can_open = bool(selected_items)
        self.state.can_reveal = bool(selected_items)
        self.state.can_copy_path = bool(selected_items)
        self.state.can_delete = (
            bool(selected_items)
            and all(item.deletable for item in selected_items)
            and all(_exportable_run_dir(item) is None for item in selected_items)
        )
        self.state.can_export_selected = _selected_run_dir(selected_items) is not None
        self.state.can_export_latest = any(_exportable_run_dir(item) is not None for item in (self.state.filtered_items or self.state.items))
        selected_run_dir = _selected_run_dir(selected_items)
        any_visible_run = self.state.can_export_latest
        self.state.can_validate_selected_run = selected_run_dir is not None
        self.state.can_validate_latest_run = any_visible_run
        self.state.can_validate_experiment_runs = selected_run_dir is not None or any_visible_run
        self.state.can_mark_selected_run = selected_run_dir is not None
        self.state.can_archive_selected_run = selected_run_dir is not None
        self.state.can_trash_selected_run = selected_run_dir is not None
        self.state.can_copy_export_path = bool(self.state.last_export_path)
        self.state.can_copy_transfer_command = bool(self.state.last_transfer_command)
        self.state.selected_delete_summary = _delete_summary(selected_items)
        current_scope = self.state.filtered_items if self.state.filtered_items else self.state.items
        legacy_items = [item for item in current_scope if item.is_legacy or item.protected]
        self.state.can_preview_migration = bool(legacy_items)
        self.state.can_apply_migration = bool(
            self._last_migration_report is not None and self._last_migration_report.candidate_count > 0
        )
        self.state.last_migration_report_path = (
            str(self._last_migration_report.manifest_path)
            if self._last_migration_report is not None and self._last_migration_report.manifest_path is not None
            else None
        )
        self.state.migration_summary_pairs = _migration_summary_pairs(self._last_migration_report)
        if not self.state.filtered_items:
            self.state.status_message = "No data items match the current filter."
        else:
            self.state.status_message = (
                f"{len(self.state.filtered_items)} item(s) shown across {len(self.state.items)} discovered artifact(s)."
            )
        return self.state

    def set_category_filter(self, value: str) -> None:
        self.state.category_filter = str(value or "all")

    def set_search_text(self, value: str) -> None:
        self.state.search_text = str(value or "")

    def set_selected_paths(self, paths: list[str]) -> None:
        selected = {str(path) for path in paths if str(path).strip()}
        self.state.selected_paths = [
            str(item.path)
            for item in self.state.filtered_items
            if str(item.path) in selected
        ]

    def selected_items(self) -> list[ManagedDataItem]:
        selected = set(self.state.selected_paths)
        if not selected:
            return []
        return [item for item in self.state.items if str(item.path) in selected]

    def delete_selected(self) -> list[Path]:
        selected_items = self.selected_items()
        deleted = delete_managed_items(self.project_root, selected_items)
        self.state.selected_paths = []
        self._catalog_dirty = True
        self.refresh()
        if deleted:
            self.state.status_message = f"Deleted {len(deleted)} selected item(s)."
        return deleted

    def preview_migration(self) -> MigrationReport:
        scope = self.state.filtered_items if self.state.filtered_items else self.state.items
        self._last_migration_report = preview_migration(self.project_root, scope)
        self.refresh()
        self.state.status_message = (
            f"Migration preview saved to {self._last_migration_report.report_dir.name} "
            f"with {self._last_migration_report.candidate_count} candidate(s)."
        )
        return self._last_migration_report

    def apply_previewed_migration(self) -> MigrationReport:
        scope = self.state.filtered_items if self.state.filtered_items else self.state.items
        self._last_migration_report = apply_migration(self.project_root, scope)
        self._catalog_dirty = True
        self.state.selected_paths = []
        self.refresh()
        self.state.status_message = (
            f"Applied {self._last_migration_report.applied_count} migration(s). "
            f"Ledger saved to {self._last_migration_report.report_dir.name}."
        )
        return self._last_migration_report

    def invalidate_catalog(self) -> None:
        self._catalog_dirty = True

    def export_selected(
        self,
        *,
        include_samples: bool = False,
        include_debug: bool = False,
        make_zip: bool = True,
    ) -> ExportBundleResult:
        selected = self.selected_items()
        run_dir = _selected_run_dir(selected)
        if run_dir is None:
            raise ValueError("Select exactly one exportable run directory.")
        return self._export_run(run_dir=run_dir, include_samples=include_samples, include_debug=include_debug, make_zip=make_zip)

    def export_latest_visible(
        self,
        *,
        include_samples: bool = False,
        include_debug: bool = False,
        make_zip: bool = True,
    ) -> ExportBundleResult:
        for item in self.state.filtered_items or self.state.items:
            run_dir = _exportable_run_dir(item)
            if run_dir is not None:
                return self._export_run(run_dir=run_dir, include_samples=include_samples, include_debug=include_debug, make_zip=make_zip)
        raise ValueError("No exportable run bundle is visible.")

    def exports_folder(self) -> Path:
        return self.project_root / "data" / "exports"

    def validate_selected_run(self) -> str:
        run_dir = self._selected_run_or_error()
        report = validate_run_folder(run_dir)
        text = render_validation_report(report)
        self.state.status_message = text
        return text

    def validate_latest_visible_run(self) -> str:
        run_dir = self._latest_visible_run_or_error()
        report = validate_run_folder(run_dir)
        text = render_validation_report(report)
        self.state.status_message = text
        return text

    def validate_visible_experiment_runs(self) -> str:
        run_dir = self._selected_run_or_none() or self._latest_visible_run_or_error()
        experiment_name = summarize_run(run_dir).experiment_name
        run_dirs = discover_experiment_run_dirs(self.project_root, experiment_name=experiment_name)
        reports = [validate_run_folder(path) for path in run_dirs]
        counts = {status: sum(1 for report in reports if report.status == status) for status in ("PASS", "WARN", "FAIL")}
        lines = [
            f"Validation for {experiment_name}: {len(reports)} run(s)",
            f"PASS={counts['PASS']} WARN={counts['WARN']} FAIL={counts['FAIL']}",
        ]
        for report in reports[:12]:
            lines.append(f"- {report.status}: {report.run_dir.name}")
        if len(reports) > 12:
            lines.append(f"... and {len(reports) - 12} more")
        text = "\n".join(lines)
        self.state.status_message = text
        return text

    def mark_selected_run(
        self,
        *,
        status: str,
        notes: str = "",
        include_in_evidence_index: bool | None = None,
    ):
        run_dir = self._selected_run_or_error()
        review = write_run_review(
            run_dir,
            status=status,
            notes=notes,
            include_in_evidence_index=include_in_evidence_index,
        )
        self._catalog_dirty = True
        self.refresh()
        self.state.status_message = f"Marked {run_dir.name} as {review.review_status}."
        return review

    def archive_selected_run(self, *, force: bool = False) -> MoveRunResult:
        run_dir = self._selected_run_or_error()
        result = archive_run(run_dir, project_root=self.project_root, force=force)
        self.state.selected_paths = []
        self._catalog_dirty = True
        self.refresh()
        self.state.status_message = f"Archived run to {result.destination_path}."
        return result

    def trash_selected_run(self, *, force: bool = False) -> MoveRunResult:
        run_dir = self._selected_run_or_error()
        result = trash_run(run_dir, project_root=self.project_root, force=force)
        self.state.selected_paths = []
        self._catalog_dirty = True
        self.refresh()
        self.state.status_message = f"Moved run to trash: {result.destination_path}."
        return result

    def build_evidence_index(self) -> Path:
        output_dir = build_thesis_evidence_index(project_root=self.project_root)
        self.state.last_evidence_index_path = str(output_dir)
        self._catalog_dirty = True
        self.refresh()
        self.state.last_evidence_index_path = str(output_dir)
        self.state.status_message = f"Built thesis evidence index at {output_dir}."
        return output_dir

    def _export_run(
        self,
        *,
        run_dir: Path,
        include_samples: bool,
        include_debug: bool,
        make_zip: bool,
    ) -> ExportBundleResult:
        result = export_run_bundle(
            run_dir=run_dir,
            project_root=self.project_root,
            include_samples=include_samples,
            include_debug=include_debug,
            make_zip=make_zip,
        )
        self.state.last_export_path = str(result.final_path)
        self.state.last_transfer_command = build_transfer_commands(result.final_path)
        self.state.status_message = (
            f"Exported {len(result.entries)} file(s) to {result.final_path} "
            f"({result.total_size_bytes} bytes)."
        )
        status_message = self.state.status_message
        self._catalog_dirty = True
        self.refresh()
        self.state.status_message = status_message
        self.state.last_export_path = str(result.final_path)
        self.state.last_transfer_command = build_transfer_commands(result.final_path)
        self.state.can_copy_export_path = True
        self.state.can_copy_transfer_command = True
        return result

    def _selected_run_or_none(self) -> Path | None:
        return _selected_run_dir(self.selected_items())

    def _selected_run_or_error(self) -> Path:
        run_dir = self._selected_run_or_none()
        if run_dir is None:
            raise ValueError("Select one experiment/modeling/diagnostic run folder first.")
        return run_dir

    def _latest_visible_run_or_error(self) -> Path:
        for item in self.state.filtered_items or self.state.items:
            run_dir = _exportable_run_dir(item)
            if run_dir is not None:
                return run_dir
        raise ValueError("No exportable run bundle is visible.")


def _build_detail_pairs(project_root: Path, selected_items: list[ManagedDataItem]) -> list[tuple[str, str]]:
    if not selected_items:
        return [("Selection", "Choose one or more items to inspect paths, type, and delete readiness.")]
    if len(selected_items) > 1:
        deletable_count = sum(1 for item in selected_items if item.deletable)
        return [
            ("Selection", f"{len(selected_items)} items selected"),
            ("Deletable", f"{deletable_count}/{len(selected_items)}"),
            ("Categories", ", ".join(sorted({item.category_label for item in selected_items}))),
        ]
    item = selected_items[0]
    try:
        relative_path = str(item.path.relative_to(project_root))
    except ValueError:
        relative_path = str(item.path)
    pairs = [
        ("Category", item.category_label),
        ("Type", item.item_type),
        ("Name", item.readable_name),
        ("Timestamp", item.timestamp_label or "unknown"),
        ("Flags", item.display_status),
        ("Path", relative_path),
        ("Canonical Path", _display_path(project_root, item.canonical_path)),
        ("Details", item.details or "n/a"),
    ]
    if item.original_path:
        pairs.append(("Legacy Path", _display_path(project_root, Path(item.original_path))))
    if item.original_name and item.original_name != item.readable_name:
        pairs.append(("Original Name", item.original_name))
    if item.legacy_reason:
        pairs.append(("Legacy", item.legacy_reason))
    if not item.deletable:
        pairs.append(("Delete", item.delete_reason or "Protected"))
    run_dir = _exportable_run_dir(item)
    if run_dir is not None:
        try:
            pairs.extend(detail_pairs_for_run(summarize_run(run_dir), project_root=project_root))
        except Exception as exc:
            pairs.append(("Run Summary", f"Could not summarize run: {exc}"))
    return pairs


def _delete_summary(selected_items: list[ManagedDataItem]) -> str:
    if not selected_items:
        return ""
    run_count = sum(1 for item in selected_items if _exportable_run_dir(item) is not None)
    if run_count:
        return "Use Archive Selected Run or Move Run to Trash for experiment runs."
    if not all(item.deletable for item in selected_items):
        blocked = [item.readable_name for item in selected_items if not item.deletable]
        return "Delete disabled for protected items: " + ", ".join(blocked[:3])
    if len(selected_items) == 1:
        return f"Delete {selected_items[0].readable_name}"
    return f"Delete {len(selected_items)} selected items"


def _migration_summary_pairs(report: MigrationReport | None) -> list[tuple[str, str]]:
    if report is None:
        return [
            ("Migration", "Preview legacy artifacts to see proposed canonical names and roots."),
        ]
    pairs = [
        ("Mode", report.mode),
        ("Scanned", str(report.scanned_count)),
        ("Candidates", str(report.candidate_count)),
        ("Applied", str(report.applied_count)),
        ("Skipped", str(report.skipped_count)),
    ]
    if report.report_dir is not None:
        pairs.append(("Ledger", str(report.report_dir)))
    return pairs


def _exportable_run_dir(item: ManagedDataItem) -> Path | None:
    if item.category_key not in {"experiments", "modeling", "diagnostics"}:
        return None
    path = item.path if item.path.is_dir() else item.path.parent
    if (path / "summary.json").exists() or (path / "metadata.json").exists() or (path / "evaluation_metadata.json").exists():
        return path
    return None


def _selected_run_dir(selected_items: list[ManagedDataItem]) -> Path | None:
    if len(selected_items) != 1:
        return None
    return _exportable_run_dir(selected_items[0])


def _display_path(project_root: Path, path: Path | None) -> str:
    if path is None:
        return "n/a"
    try:
        return str(path.relative_to(project_root))
    except ValueError:
        return str(path)

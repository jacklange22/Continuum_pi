"""Controller for the operator-facing Data Management workspace."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

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
        self.state.can_delete = bool(selected_items) and all(item.deletable for item in selected_items)
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
    return pairs


def _delete_summary(selected_items: list[ManagedDataItem]) -> str:
    if not selected_items:
        return ""
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


def _display_path(project_root: Path, path: Path | None) -> str:
    if path is None:
        return "n/a"
    try:
        return str(path.relative_to(project_root))
    except ValueError:
        return str(path)

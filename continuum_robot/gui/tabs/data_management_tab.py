"""Operator-facing data and artifact browser."""

from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import QSignalBlocker, Qt, QUrl
from PySide6.QtGui import QDesktopServices
from PySide6.QtWidgets import (
    QApplication,
    QComboBox,
    QFrame,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from continuum_robot.gui.controllers.data_management_controller import (
    DataManagementController,
    DataManagementViewState,
)
from continuum_robot.gui.theme import COLORS, grouped_workspace_stylesheet


class DataManagementTab(QWidget):
    """Centralized workspace for browsing and safely deleting saved data bundles."""

    CATEGORY_OPTIONS = [
        ("All Data", "all"),
        ("Calibration", "calibration"),
        ("Experiments", "experiments"),
        ("Modeling / Training", "modeling"),
        ("Diagnostics", "diagnostics"),
    ]

    COLUMN_LABELS = ["Timestamp", "Category", "Name", "Type", "Flags", "Path"]

    def __init__(self, controller: DataManagementController, parent=None) -> None:
        super().__init__(parent)
        self.controller = controller
        self.setObjectName("dataManagementWorkspace")
        self.setStyleSheet(
            grouped_workspace_stylesheet(
                object_name="dataManagementWorkspace",
                input_selectors=["QComboBox", "QLineEdit", "QTableWidget"],
            )
        )

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)

        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setFrameShape(QFrame.NoFrame)
        outer.addWidget(self.scroll_area)

        content = QWidget()
        self.scroll_area.setWidget(content)
        root = QVBoxLayout(content)
        root.setContentsMargins(16, 16, 16, 16)
        root.setSpacing(12)

        title = QLabel("Data Management")
        title.setProperty("role", "title")
        hint = QLabel(
            "Browse canonical calibration, experiment, modeling, and diagnostic outputs in one place. "
            "Use this tab for safe multi-select cleanup instead of deleting data from scattered workflow pages."
        )
        hint.setProperty("role", "hint")
        hint.setWordWrap(True)
        root.addWidget(title)
        root.addWidget(hint)

        filters_card = _Card("Browse", "Filter saved artifacts by category or search text, then act on the selected rows.")
        filter_row = QHBoxLayout()
        filter_row.setContentsMargins(0, 0, 0, 0)
        filter_row.setSpacing(10)
        self.category_combo = QComboBox()
        for label, value in self.CATEGORY_OPTIONS:
            self.category_combo.addItem(label, value)
        self.category_combo.currentIndexChanged.connect(self._on_category_changed)
        self.search_input = QLineEdit()
        self.search_input.setPlaceholderText("Search name, type, status, or path")
        self.search_input.textChanged.connect(self._on_search_changed)
        self.refresh_button = QPushButton("Refresh")
        self.refresh_button.setProperty("variant", "ghost")
        self.refresh_button.clicked.connect(self._refresh_catalog)
        filter_row.addWidget(QLabel("Category"))
        filter_row.addWidget(self.category_combo, 0)
        filter_row.addWidget(self.search_input, 1)
        filter_row.addWidget(self.refresh_button, 0)
        filters_card.body_layout.addLayout(filter_row)

        self.table = QTableWidget(0, len(self.COLUMN_LABELS))
        self.table.setHorizontalHeaderLabels(self.COLUMN_LABELS)
        self.table.setSelectionBehavior(QTableWidget.SelectRows)
        self.table.setSelectionMode(QTableWidget.ExtendedSelection)
        self.table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.table.setAlternatingRowColors(True)
        self.table.verticalHeader().setVisible(False)
        self.table.setMinimumHeight(360)
        header = self.table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(1, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(2, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(3, QHeaderView.Stretch)
        header.setSectionResizeMode(4, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(5, QHeaderView.Stretch)
        self.table.itemSelectionChanged.connect(self._sync_selection_from_table)
        filters_card.body_layout.addWidget(self.table)
        root.addWidget(filters_card)

        lower = QHBoxLayout()
        lower.setContentsMargins(0, 0, 0, 0)
        lower.setSpacing(12)
        root.addLayout(lower, 1)

        actions_card = _Card("Actions", "Primary operator actions for the selected bundle(s).")
        action_row = QHBoxLayout()
        action_row.setContentsMargins(0, 0, 0, 0)
        action_row.setSpacing(10)
        self.open_button = QPushButton("Open")
        self.open_button.clicked.connect(self._open_selected)
        self.reveal_button = QPushButton("Reveal Folder")
        self.reveal_button.setProperty("variant", "ghost")
        self.reveal_button.clicked.connect(self._reveal_selected)
        self.copy_path_button = QPushButton("Copy Path")
        self.copy_path_button.setProperty("variant", "ghost")
        self.copy_path_button.clicked.connect(self._copy_selected_paths)
        self.delete_button = QPushButton("Delete Selected")
        self.delete_button.setProperty("variant", "danger")
        self.delete_button.clicked.connect(self._delete_selected)
        action_row.addWidget(self.open_button)
        action_row.addWidget(self.reveal_button)
        action_row.addWidget(self.copy_path_button)
        action_row.addWidget(self.delete_button)
        action_row.addStretch(1)
        actions_card.body_layout.addLayout(action_row)
        self.status_label = QLabel("Browse saved calibration, experiment, modeling, and diagnostic artifacts.")
        self.status_label.setWordWrap(True)
        self.status_label.setStyleSheet(f"color: {COLORS.text_primary};")
        actions_card.body_layout.addWidget(self.status_label)
        lower.addWidget(actions_card, 3)

        details_card = _Card("Details", "Canonical roots, normalized selection details, and legacy migration status.")
        self.selection_pairs = _PairsWidget()
        self.root_pairs = _PairsWidget()
        migration_row = QHBoxLayout()
        migration_row.setContentsMargins(0, 0, 0, 0)
        migration_row.setSpacing(10)
        self.preview_migration_button = QPushButton("Preview Legacy Migration")
        self.preview_migration_button.clicked.connect(self._preview_migration)
        self.apply_migration_button = QPushButton("Apply Previewed Migration")
        self.apply_migration_button.setProperty("variant", "danger")
        self.apply_migration_button.clicked.connect(self._apply_migration)
        self.open_migration_report_button = QPushButton("Open Migration Ledger")
        self.open_migration_report_button.setProperty("variant", "ghost")
        self.open_migration_report_button.clicked.connect(self._open_migration_report)
        migration_row.addWidget(self.preview_migration_button)
        migration_row.addWidget(self.apply_migration_button)
        migration_row.addWidget(self.open_migration_report_button)
        migration_row.addStretch(1)
        details_card.body_layout.addLayout(migration_row)
        self.migration_pairs = _PairsWidget()
        details_card.body_layout.addWidget(self.selection_pairs)
        details_card.body_layout.addWidget(self.migration_pairs)
        details_card.body_layout.addWidget(self.root_pairs)
        lower.addWidget(details_card, 2)

    def update(self, state: DataManagementViewState) -> None:
        self._set_combo(self.category_combo, state.category_filter)
        self._set_line_edit(self.search_input, state.search_text)
        self._sync_table(state)
        self.selection_pairs.set_pairs(state.detail_pairs)
        self.migration_pairs.set_pairs(state.migration_summary_pairs)
        self.root_pairs.set_pairs(state.root_summary_pairs)
        self.status_label.setText(state.status_message)
        self.open_button.setEnabled(state.can_open)
        self.reveal_button.setEnabled(state.can_reveal)
        self.copy_path_button.setEnabled(state.can_copy_path)
        self.delete_button.setEnabled(state.can_delete)
        self.preview_migration_button.setEnabled(state.can_preview_migration)
        self.apply_migration_button.setEnabled(state.can_apply_migration)
        self.open_migration_report_button.setEnabled(bool(state.last_migration_report_path))
        if state.selected_delete_summary:
            self.delete_button.setToolTip(state.selected_delete_summary)
        else:
            self.delete_button.setToolTip("")

    def _refresh_catalog(self) -> None:
        self.controller.invalidate_catalog()
        self.update(self.controller.refresh())

    def _on_category_changed(self, _index: int) -> None:
        self.controller.set_category_filter(str(self.category_combo.currentData() or "all"))
        self.update(self.controller.refresh())

    def _on_search_changed(self, value: str) -> None:
        self.controller.set_search_text(value)
        self.update(self.controller.refresh())

    def _sync_table(self, state: DataManagementViewState) -> None:
        current_paths = [
            self.table.item(row, 0).data(Qt.UserRole)
            for row in range(self.table.rowCount())
            if self.table.item(row, 0) is not None
        ]
        target_paths = [str(item.path) for item in state.filtered_items]
        if current_paths != target_paths:
            with QSignalBlocker(self.table):
                self.table.setRowCount(len(state.filtered_items))
                for row, item in enumerate(state.filtered_items):
                    values = [
                        item.timestamp_label,
                        item.category_label,
                        item.readable_name,
                        item.item_type,
                        item.display_status,
                        _relative_path(item.path, self.controller.project_root),
                    ]
                    for column, value in enumerate(values):
                        cell = QTableWidgetItem(str(value))
                        cell.setData(Qt.UserRole, str(item.path))
                        if column == 5:
                            cell.setToolTip(str(item.path))
                        self.table.setItem(row, column, cell)
        with QSignalBlocker(self.table):
            self.table.clearSelection()
            selected = set(state.selected_paths)
            for row in range(self.table.rowCount()):
                item = self.table.item(row, 0)
                if item is not None and item.data(Qt.UserRole) in selected:
                    self.table.selectRow(row)

    def _sync_selection_from_table(self) -> None:
        paths = []
        for index in self.table.selectionModel().selectedRows():
            item = self.table.item(index.row(), 0)
            if item is not None:
                paths.append(str(item.data(Qt.UserRole)))
        self.controller.set_selected_paths(paths)
        self.update(self.controller.refresh())

    def _open_selected(self) -> None:
        selected = self.controller.selected_items()
        if not selected:
            return
        for item in selected:
            QDesktopServices.openUrl(QUrl.fromLocalFile(str(item.path)))

    def _reveal_selected(self) -> None:
        selected = self.controller.selected_items()
        if not selected:
            return
        for item in selected:
            QDesktopServices.openUrl(QUrl.fromLocalFile(str(item.path.parent if item.path.is_file() else item.path)))

    def _copy_selected_paths(self) -> None:
        selected = self.controller.selected_items()
        if not selected:
            return
        text = "\n".join(str(item.path) for item in selected)
        clipboard = QApplication.clipboard()
        if clipboard is not None:
            clipboard.setText(text)
        self.status_label.setText(f"Copied {len(selected)} path(s) to the clipboard.")

    def _delete_selected(self) -> None:
        selected = self.controller.selected_items()
        if not selected:
            return
        if not all(item.deletable for item in selected):
            self.status_label.setText(self.controller.refresh().selected_delete_summary)
            return
        lines = [f"Delete {len(selected)} selected item(s)?", ""]
        for item in selected[:8]:
            lines.append(_relative_path(item.path, self.controller.project_root))
        if len(selected) > 8:
            lines.append(f"... and {len(selected) - 8} more")
        lines.append("")
        lines.append("This deletes only the selected bundles/files, not their parent category folders.")
        choice = QMessageBox.question(
            self,
            "Delete Selected Data",
            "\n".join(lines),
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No,
        )
        if choice != QMessageBox.Yes:
            return
        self.controller.delete_selected()
        self.update(self.controller.refresh())

    def _preview_migration(self) -> None:
        report = self.controller.preview_migration()
        self.update(self.controller.refresh())
        if report.summary_path is not None:
            self.status_label.setText(f"Preview saved to {report.summary_path}.")

    def _apply_migration(self) -> None:
        state = self.controller.refresh()
        if not state.can_apply_migration:
            return
        choice = QMessageBox.question(
            self,
            "Apply Previewed Migration",
            "Apply the currently previewed legacy migration actions?\n\n"
            "Protected aliases stay untouched. A migration ledger will be written before and after the move.",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No,
        )
        if choice != QMessageBox.Yes:
            return
        report = self.controller.apply_previewed_migration()
        self.update(self.controller.refresh())
        if report.summary_path is not None:
            self.status_label.setText(f"Applied migration. Ledger saved to {report.summary_path}.")

    def _open_migration_report(self) -> None:
        state = self.controller.refresh()
        if state.last_migration_report_path:
            QDesktopServices.openUrl(QUrl.fromLocalFile(str(Path(state.last_migration_report_path).parent)))

    @staticmethod
    def _set_combo(combo: QComboBox, value: str) -> None:
        index = combo.findData(value)
        if index >= 0 and combo.currentIndex() != index:
            with QSignalBlocker(combo):
                combo.setCurrentIndex(index)

    @staticmethod
    def _set_line_edit(widget: QLineEdit, value: str) -> None:
        if widget.text() != value:
            with QSignalBlocker(widget):
                widget.setText(value)


class _Card(QFrame):
    def __init__(self, title: str, subtitle: str = "") -> None:
        super().__init__()
        self.setProperty("card", True)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(10)

        title_label = QLabel(title)
        title_label.setProperty("role", "section_title")
        layout.addWidget(title_label)
        if subtitle:
            subtitle_label = QLabel(subtitle)
            subtitle_label.setProperty("role", "hint")
            subtitle_label.setWordWrap(True)
            layout.addWidget(subtitle_label)

        self.body_layout = QVBoxLayout()
        self.body_layout.setContentsMargins(0, 0, 0, 0)
        self.body_layout.setSpacing(10)
        layout.addLayout(self.body_layout)


class _PairsWidget(QWidget):
    def __init__(self) -> None:
        super().__init__()
        self._layout = QVBoxLayout(self)
        self._layout.setContentsMargins(0, 0, 0, 0)
        self._layout.setSpacing(6)

    def set_pairs(self, pairs: list[tuple[str, str]]) -> None:
        while self._layout.count():
            item = self._layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()
        for label, value in pairs:
            row = QLabel(f"<b>{label}:</b> {value}")
            row.setWordWrap(True)
            row.setTextFormat(Qt.RichText)
            self._layout.addWidget(row)
        self._layout.addStretch(1)


def _relative_path(path: Path, project_root: Path) -> str:
    try:
        return str(Path(path).relative_to(project_root))
    except ValueError:
        return str(path)

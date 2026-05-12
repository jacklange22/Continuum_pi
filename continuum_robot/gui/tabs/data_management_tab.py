"""Operator-facing data and artifact browser."""

from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import QSignalBlocker, Qt, QUrl
from PySide6.QtGui import QDesktopServices
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
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
from continuum_robot.data.run_management import summarize_run


class DataManagementTab(QWidget):
    """Centralized workspace for browsing and safely deleting saved data bundles."""

    CATEGORY_OPTIONS = [
        ("All Data", "all"),
        ("Calibration", "calibration"),
        ("Experiments", "experiments"),
        ("Exports", "exports"),
        ("Modeling / Training", "modeling"),
        ("Diagnostics", "diagnostics"),
        ("Trash", "trash"),
    ]

    COLUMN_LABELS = ["Timestamp", "Experiment", "Run", "Validation", "Trust", "Mode / Segment", "Flags", "Path"]

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

        filter_row_2 = QHBoxLayout()
        filter_row_2.setContentsMargins(0, 0, 0, 0)
        filter_row_2.setSpacing(8)
        self.root_filter_combo = _combo_with_options([("Any root", "all")])
        self.experiment_filter_combo = _combo_with_options([("Any experiment", "all")])
        self.review_filter_combo = _combo_with_options(
            [("Any review", "all"), ("Unreviewed", "unreviewed")]
            + [(status, status) for status in ("debug", "garbage", "keep", "thesis_candidate", "advisor_share", "archived")]
        )
        self.trust_filter_combo = _combo_with_options([("Any trust", "all")])
        self.sort_combo = _combo_with_options(
            [
                ("Newest", "newest"),
                ("Largest", "largest"),
                ("Experiment", "experiment"),
                ("Review", "review"),
                ("Trust", "trust"),
            ]
        )
        for combo, key in (
            (self.root_filter_combo, "root_filter"),
            (self.experiment_filter_combo, "experiment_filter"),
            (self.review_filter_combo, "review_status_filter"),
            (self.trust_filter_combo, "trust_mode_filter"),
            (self.sort_combo, "sort_mode"),
        ):
            combo.currentIndexChanged.connect(lambda _index, k=key, c=combo: self._on_filter_changed(k, str(c.currentData() or "all")))
        filter_row_2.addWidget(QLabel("Root"))
        filter_row_2.addWidget(self.root_filter_combo)
        filter_row_2.addWidget(QLabel("Experiment"))
        filter_row_2.addWidget(self.experiment_filter_combo)
        filter_row_2.addWidget(QLabel("Review"))
        filter_row_2.addWidget(self.review_filter_combo)
        filter_row_2.addWidget(QLabel("Trust"))
        filter_row_2.addWidget(self.trust_filter_combo)
        filter_row_2.addWidget(QLabel("Sort"))
        filter_row_2.addWidget(self.sort_combo)
        filters_card.body_layout.addLayout(filter_row_2)

        filter_row_3 = QHBoxLayout()
        filter_row_3.setContentsMargins(0, 0, 0, 0)
        filter_row_3.setSpacing(8)
        self.mock_filter_combo = _bool_filter_combo("Mock mode")
        self.model_valid_filter_combo = _bool_filter_combo("Model valid")
        self.thesis_valid_filter_combo = _bool_filter_combo("Thesis valid")
        self.samples_filter_combo = _bool_filter_combo("Samples")
        self.figures_filter_combo = _bool_filter_combo("Figures")
        self.review_sidecar_filter_combo = _bool_filter_combo("run_review")
        self.size_filter_combo = _combo_with_options(
            [("Any size", "all"), ("<1 MB", "small"), ("1-25 MB", "medium"), (">=25 MB", "large")]
        )
        for combo, key in (
            (self.mock_filter_combo, "mock_mode_filter"),
            (self.model_valid_filter_combo, "valid_model_filter"),
            (self.thesis_valid_filter_combo, "valid_thesis_filter"),
            (self.samples_filter_combo, "has_samples_filter"),
            (self.figures_filter_combo, "has_figures_filter"),
            (self.review_sidecar_filter_combo, "has_review_filter"),
            (self.size_filter_combo, "size_filter"),
        ):
            combo.currentIndexChanged.connect(lambda _index, k=key, c=combo: self._on_filter_changed(k, str(c.currentData() or "all")))
        for label, combo in (
            ("Mock", self.mock_filter_combo),
            ("Model", self.model_valid_filter_combo),
            ("Thesis", self.thesis_valid_filter_combo),
            ("Samples", self.samples_filter_combo),
            ("Figures", self.figures_filter_combo),
            ("Review file", self.review_sidecar_filter_combo),
            ("Size", self.size_filter_combo),
        ):
            filter_row_3.addWidget(QLabel(label))
            filter_row_3.addWidget(combo)
        filter_row_3.addStretch(1)
        filters_card.body_layout.addLayout(filter_row_3)

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
        header.setSectionResizeMode(3, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(4, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(5, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(6, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(7, QHeaderView.Stretch)
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
        self.delete_button = QPushButton("Delete Selected File/Bundle")
        self.delete_button.setProperty("variant", "danger")
        self.delete_button.clicked.connect(self._delete_selected)
        action_row.addWidget(self.open_button)
        action_row.addWidget(self.reveal_button)
        action_row.addWidget(self.copy_path_button)
        action_row.addWidget(self.delete_button)
        action_row.addStretch(1)
        actions_card.body_layout.addLayout(action_row)
        export_row = QHBoxLayout()
        export_row.setContentsMargins(0, 0, 0, 0)
        export_row.setSpacing(10)
        self.export_selected_button = QPushButton("Export Human Packet")
        self.export_selected_button.clicked.connect(self._export_selected_run)
        self.export_latest_button = QPushButton("Export Latest Human")
        self.export_latest_button.setProperty("variant", "ghost")
        self.export_latest_button.clicked.connect(self._export_latest_run)
        self.export_ai_selected_button = QPushButton("Export AI Debug Packet")
        self.export_ai_selected_button.clicked.connect(self._export_ai_selected_run)
        self.export_ai_latest_button = QPushButton("Export Latest AI Debug")
        self.export_ai_latest_button.setProperty("variant", "ghost")
        self.export_ai_latest_button.clicked.connect(self._export_ai_latest_run)
        self.include_samples_checkbox = QCheckBox("Include samples")
        self.include_debug_checkbox = QCheckBox("Include debug")
        self.zip_export_checkbox = QCheckBox("Zip")
        self.zip_export_checkbox.setChecked(True)
        export_row.addWidget(self.export_selected_button)
        export_row.addWidget(self.export_latest_button)
        export_row.addWidget(self.export_ai_selected_button)
        export_row.addWidget(self.export_ai_latest_button)
        export_row.addWidget(self.include_samples_checkbox)
        export_row.addWidget(self.include_debug_checkbox)
        export_row.addWidget(self.zip_export_checkbox)
        export_row.addStretch(1)
        actions_card.body_layout.addLayout(export_row)
        export_copy_row = QHBoxLayout()
        export_copy_row.setContentsMargins(0, 0, 0, 0)
        export_copy_row.setSpacing(10)
        self.copy_export_path_button = QPushButton("Copy Export Path")
        self.copy_export_path_button.setProperty("variant", "ghost")
        self.copy_export_path_button.clicked.connect(self._copy_export_path)
        self.copy_transfer_command_button = QPushButton("Copy Transfer Command")
        self.copy_transfer_command_button.setProperty("variant", "ghost")
        self.copy_transfer_command_button.clicked.connect(self._copy_transfer_command)
        self.open_exports_folder_button = QPushButton("Open Exports Folder")
        self.open_exports_folder_button.setProperty("variant", "ghost")
        self.open_exports_folder_button.clicked.connect(self._open_exports_folder)
        export_copy_row.addWidget(self.copy_export_path_button)
        export_copy_row.addWidget(self.copy_transfer_command_button)
        export_copy_row.addWidget(self.open_exports_folder_button)
        export_copy_row.addStretch(1)
        actions_card.body_layout.addLayout(export_copy_row)
        validation_row = QHBoxLayout()
        validation_row.setContentsMargins(0, 0, 0, 0)
        validation_row.setSpacing(10)
        self.validate_selected_button = QPushButton("Validate Selected Run")
        self.validate_selected_button.clicked.connect(self._validate_selected_run)
        self.validate_latest_button = QPushButton("Validate Latest Run")
        self.validate_latest_button.setProperty("variant", "ghost")
        self.validate_latest_button.clicked.connect(self._validate_latest_run)
        self.validate_experiment_button = QPushButton("Validate Experiment Runs")
        self.validate_experiment_button.setProperty("variant", "ghost")
        self.validate_experiment_button.clicked.connect(self._validate_experiment_runs)
        validation_row.addWidget(self.validate_selected_button)
        validation_row.addWidget(self.validate_latest_button)
        validation_row.addWidget(self.validate_experiment_button)
        validation_row.addStretch(1)
        actions_card.body_layout.addLayout(validation_row)
        modeling_row = QHBoxLayout()
        modeling_row.setContentsMargins(0, 0, 0, 0)
        modeling_row.setSpacing(10)
        self.run_two_segment_modeling_button = QPushButton("Run Two-Segment Modeling")
        self.run_two_segment_modeling_button.clicked.connect(self._run_two_segment_modeling)
        self.open_modeling_summary_button = QPushButton("Open Modeling Summary")
        self.open_modeling_summary_button.setProperty("variant", "ghost")
        self.open_modeling_summary_button.clicked.connect(self._open_modeling_summary)
        self.export_modeling_bundle_button = QPushButton("Export Modeling Bundle")
        self.export_modeling_bundle_button.setProperty("variant", "ghost")
        self.export_modeling_bundle_button.clicked.connect(self._export_modeling_bundle)
        modeling_row.addWidget(self.run_two_segment_modeling_button)
        modeling_row.addWidget(self.open_modeling_summary_button)
        modeling_row.addWidget(self.export_modeling_bundle_button)
        modeling_row.addStretch(1)
        actions_card.body_layout.addLayout(modeling_row)
        review_row = QHBoxLayout()
        review_row.setContentsMargins(0, 0, 0, 0)
        review_row.setSpacing(10)
        self.review_status_combo = QComboBox()
        for status in controller.state.review_status_options:
            self.review_status_combo.addItem(status, status)
        self.review_notes_input = QLineEdit()
        self.review_notes_input.setPlaceholderText("Review notes")
        self.include_evidence_checkbox = QCheckBox("Evidence index")
        self.intended_use_combo = _combo_with_options(
            [
                ("debug", "debug"),
                ("mock", "mock"),
                ("thesis", "thesis"),
                ("advisor", "advisor"),
                ("ai_debug", "ai_debug"),
            ]
        )
        self.save_review_button = QPushButton("Save Review")
        self.save_review_button.clicked.connect(self._save_review)
        review_row.addWidget(QLabel("Mark"))
        review_row.addWidget(self.review_status_combo)
        review_row.addWidget(self.review_notes_input, 1)
        review_row.addWidget(QLabel("Use"))
        review_row.addWidget(self.intended_use_combo)
        review_row.addWidget(self.include_evidence_checkbox)
        review_row.addWidget(self.save_review_button)
        actions_card.body_layout.addLayout(review_row)
        lifecycle_row = QHBoxLayout()
        lifecycle_row.setContentsMargins(0, 0, 0, 0)
        lifecycle_row.setSpacing(10)
        self.archive_run_button = QPushButton("Archive Selected Run")
        self.archive_run_button.setProperty("variant", "ghost")
        self.archive_run_button.clicked.connect(self._archive_selected_run)
        self.trash_run_button = QPushButton("Move Selected Run to Trash")
        self.trash_run_button.setProperty("variant", "danger")
        self.trash_run_button.clicked.connect(self._trash_selected_run)
        self.build_evidence_index_button = QPushButton("Build Evidence Index")
        self.build_evidence_index_button.setProperty("variant", "ghost")
        self.build_evidence_index_button.clicked.connect(self._build_evidence_index)
        self.include_debug_evidence_checkbox = QCheckBox("Include debug")
        self.include_mock_evidence_checkbox = QCheckBox("Include mock")
        lifecycle_row.addWidget(self.archive_run_button)
        lifecycle_row.addWidget(self.trash_run_button)
        lifecycle_row.addWidget(self.build_evidence_index_button)
        lifecycle_row.addWidget(self.include_debug_evidence_checkbox)
        lifecycle_row.addWidget(self.include_mock_evidence_checkbox)
        lifecycle_row.addStretch(1)
        actions_card.body_layout.addLayout(lifecycle_row)
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
        self._sync_filter_options(state)
        self._set_combo(self.category_combo, state.category_filter)
        self._set_line_edit(self.search_input, state.search_text)
        self._set_combo(self.root_filter_combo, state.root_filter)
        self._set_combo(self.experiment_filter_combo, state.experiment_filter)
        self._set_combo(self.review_filter_combo, state.review_status_filter)
        self._set_combo(self.trust_filter_combo, state.trust_mode_filter)
        self._set_combo(self.mock_filter_combo, state.mock_mode_filter)
        self._set_combo(self.model_valid_filter_combo, state.valid_model_filter)
        self._set_combo(self.thesis_valid_filter_combo, state.valid_thesis_filter)
        self._set_combo(self.samples_filter_combo, state.has_samples_filter)
        self._set_combo(self.figures_filter_combo, state.has_figures_filter)
        self._set_combo(self.review_sidecar_filter_combo, state.has_review_filter)
        self._set_combo(self.size_filter_combo, state.size_filter)
        self._set_combo(self.sort_combo, state.sort_mode)
        self._sync_table(state)
        self.selection_pairs.set_pairs(state.detail_pairs)
        self.migration_pairs.set_pairs(state.migration_summary_pairs)
        self.root_pairs.set_pairs(state.root_summary_pairs)
        self.status_label.setText(state.status_message)
        self.open_button.setEnabled(state.can_open)
        self.reveal_button.setEnabled(state.can_reveal)
        self.copy_path_button.setEnabled(state.can_copy_path)
        self.delete_button.setEnabled(state.can_delete)
        self.delete_button.setText(state.delete_button_label)
        self.preview_migration_button.setEnabled(state.can_preview_migration)
        self.apply_migration_button.setEnabled(state.can_apply_migration)
        self.open_migration_report_button.setEnabled(bool(state.last_migration_report_path))
        self.export_selected_button.setEnabled(state.can_export_selected)
        self.export_latest_button.setEnabled(state.can_export_latest)
        self.export_ai_selected_button.setEnabled(state.can_export_selected)
        self.export_ai_latest_button.setEnabled(state.can_export_latest)
        self.copy_export_path_button.setEnabled(state.can_copy_export_path)
        self.copy_transfer_command_button.setEnabled(state.can_copy_transfer_command)
        self.open_exports_folder_button.setEnabled(True)
        self.validate_selected_button.setEnabled(state.can_validate_selected_run)
        self.validate_latest_button.setEnabled(state.can_validate_latest_run)
        self.validate_experiment_button.setEnabled(state.can_validate_experiment_runs)
        self.run_two_segment_modeling_button.setEnabled(state.can_run_two_segment_modeling)
        self.open_modeling_summary_button.setEnabled(state.can_open_modeling_summary)
        self.export_modeling_bundle_button.setEnabled(state.can_export_modeling_bundle)
        self.save_review_button.setEnabled(state.can_mark_selected_run)
        self.archive_run_button.setEnabled(state.can_archive_selected_run)
        self.trash_run_button.setEnabled(state.can_trash_selected_run)
        self.build_evidence_index_button.setEnabled(state.can_build_evidence_index)
        self._sync_review_controls()
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

    def _on_filter_changed(self, key: str, value: str) -> None:
        self.controller.set_filter_value(key, value)
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
                    values = _table_values(item, self.controller.project_root)
                    for column, value in enumerate(values):
                        cell = QTableWidgetItem(str(value))
                        cell.setData(Qt.UserRole, str(item.path))
                        if column == 7:
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

    def _export_selected_run(self) -> None:
        self._run_export(selected=True, profile="human_advisor")

    def _export_latest_run(self) -> None:
        self._run_export(selected=False, profile="human_advisor")

    def _export_ai_selected_run(self) -> None:
        self._run_export(selected=True, profile="ai_debug")

    def _export_ai_latest_run(self) -> None:
        self._run_export(selected=False, profile="ai_debug")

    def _run_export(self, *, selected: bool, profile: str) -> None:
        try:
            if selected:
                result = self.controller.export_selected(
                    include_samples=self.include_samples_checkbox.isChecked(),
                    include_debug=self.include_debug_checkbox.isChecked(),
                    make_zip=self.zip_export_checkbox.isChecked(),
                    profile=profile,
                )
            else:
                result = self.controller.export_latest_visible(
                    include_samples=self.include_samples_checkbox.isChecked(),
                    include_debug=self.include_debug_checkbox.isChecked(),
                    make_zip=self.zip_export_checkbox.isChecked(),
                    profile=profile,
                )
        except Exception as exc:
            self.status_label.setText(f"Export failed: {exc}")
            return
        self.update(self.controller.refresh())
        self.status_label.setText(
            f"Exported {len(result.entries)} file(s) to {result.final_path} "
            f"({result.total_size_bytes} bytes)."
        )

    def _copy_export_path(self) -> None:
        state = self.controller.refresh()
        if not state.last_export_path:
            return
        clipboard = QApplication.clipboard()
        if clipboard is not None:
            clipboard.setText(state.last_export_path)
        self.status_label.setText("Copied export path to the clipboard.")

    def _copy_transfer_command(self) -> None:
        state = self.controller.refresh()
        if not state.last_transfer_command:
            return
        clipboard = QApplication.clipboard()
        if clipboard is not None:
            clipboard.setText(state.last_transfer_command)
        self.status_label.setText("Copied transfer command template to the clipboard.")

    def _open_exports_folder(self) -> None:
        folder = self.controller.exports_folder()
        folder.mkdir(parents=True, exist_ok=True)
        QDesktopServices.openUrl(QUrl.fromLocalFile(str(folder)))

    def _validate_selected_run(self) -> None:
        self._run_text_action(self.controller.validate_selected_run)

    def _validate_latest_run(self) -> None:
        self._run_text_action(self.controller.validate_latest_visible_run)

    def _validate_experiment_runs(self) -> None:
        self._run_text_action(self.controller.validate_visible_experiment_runs)

    def _run_two_segment_modeling(self) -> None:
        try:
            output_dir = self.controller.run_two_segment_modeling_for_selected()
        except Exception as exc:
            self.status_label.setText(f"Two-segment modeling failed: {exc}")
            return
        self.update(self.controller.refresh())
        self.status_label.setText(f"Two-segment modeling saved to {output_dir}")

    def _open_modeling_summary(self) -> None:
        try:
            path = self.controller.open_modeling_summary_path_for_selected()
        except Exception as exc:
            self.status_label.setText(f"Open modeling summary failed: {exc}")
            return
        QDesktopServices.openUrl(QUrl.fromLocalFile(str(path)))

    def _export_modeling_bundle(self) -> None:
        try:
            result = self.controller.export_selected_modeling_bundle()
        except Exception as exc:
            self.status_label.setText(f"Modeling export failed: {exc}")
            return
        self.update(self.controller.refresh())
        self.status_label.setText(f"Exported modeling bundle to {result.final_path}")

    def _save_review(self) -> None:
        try:
            self.controller.mark_selected_run(
                status=str(self.review_status_combo.currentData() or "debug"),
                notes=self.review_notes_input.text(),
                include_in_evidence_index=self.include_evidence_checkbox.isChecked(),
                intended_use=str(self.intended_use_combo.currentData() or "debug"),
            )
        except Exception as exc:
            self.status_label.setText(f"Review save failed: {exc}")
            return
        self.update(self.controller.refresh())

    def _archive_selected_run(self) -> None:
        self._move_selected_run(action="archive")

    def _trash_selected_run(self) -> None:
        self._move_selected_run(action="trash")

    def _move_selected_run(self, *, action: str) -> None:
        selected = self.controller.selected_items()
        if not selected:
            return
        verb = "Archive" if action == "archive" else "Move to Trash"
        choice = QMessageBox.question(
            self,
            f"{verb} Run",
            f"{verb} the selected run?\n\nThis moves the folder out of active data/experiments; it does not delete exports.",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No,
        )
        if choice != QMessageBox.Yes:
            return
        try:
            result = self.controller.archive_selected_run() if action == "archive" else self.controller.trash_selected_run()
        except ValueError as exc:
            force_choice = QMessageBox.question(
                self,
                f"Protected Run: {verb}",
                f"{exc}\n\nProceed anyway?",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No,
            )
            if force_choice != QMessageBox.Yes:
                self.status_label.setText(str(exc))
                return
            try:
                result = (
                    self.controller.archive_selected_run(force=True)
                    if action == "archive"
                    else self.controller.trash_selected_run(force=True)
                )
            except Exception as final_exc:
                self.status_label.setText(f"{verb} failed: {final_exc}")
                return
        except Exception as exc:
            self.status_label.setText(f"{verb} failed: {exc}")
            return
        self.update(self.controller.refresh())
        self.status_label.setText(f"{verb} complete: {result.destination_path}")

    def _build_evidence_index(self) -> None:
        try:
            output_dir = self.controller.build_evidence_index(
                include_debug=self.include_debug_evidence_checkbox.isChecked(),
                include_mock=self.include_mock_evidence_checkbox.isChecked(),
            )
        except Exception as exc:
            self.status_label.setText(f"Evidence index failed: {exc}")
            return
        self.update(self.controller.refresh())
        self.status_label.setText(f"Evidence index built at {output_dir}")

    def _run_text_action(self, action) -> None:
        try:
            text = action()
        except Exception as exc:
            self.status_label.setText(f"Action failed: {exc}")
            return
        self.update(self.controller.refresh())
        self.status_label.setText(text)

    def _sync_review_controls(self) -> None:
        selected = self.controller.selected_items()
        if len(selected) != 1:
            return
        run_dir = _run_dir_for_item(selected[0])
        if run_dir is None:
            return
        try:
            review = summarize_run(run_dir).review
        except Exception:
            return
        index = self.review_status_combo.findData(review.review_status)
        if index >= 0 and self.review_status_combo.currentIndex() != index:
            with QSignalBlocker(self.review_status_combo):
                self.review_status_combo.setCurrentIndex(index)
        if not self.review_notes_input.hasFocus() and self.review_notes_input.text() != review.notes:
            with QSignalBlocker(self.review_notes_input):
                self.review_notes_input.setText(review.notes)
        if self.include_evidence_checkbox.isChecked() != review.include_in_evidence_index:
            with QSignalBlocker(self.include_evidence_checkbox):
                self.include_evidence_checkbox.setChecked(review.include_in_evidence_index)
        intended_index = self.intended_use_combo.findData(review.intended_use or "debug")
        if intended_index >= 0 and self.intended_use_combo.currentIndex() != intended_index:
            with QSignalBlocker(self.intended_use_combo):
                self.intended_use_combo.setCurrentIndex(intended_index)

    def _delete_selected(self) -> None:
        selected = self.controller.selected_items()
        if not selected:
            return
        if not all(item.deletable for item in selected):
            self.status_label.setText(self.controller.refresh().selected_delete_summary)
            return
        trash_only = all(item.category_key == "trash" for item in selected)
        action_label = "Permanently delete from trash" if trash_only else "Delete"
        lines = [f"{action_label} {len(selected)} selected item(s)?", ""]
        for item in selected[:8]:
            lines.append(_relative_path(item.path, self.controller.project_root))
        if len(selected) > 8:
            lines.append(f"... and {len(selected) - 8} more")
        lines.append("")
        if trash_only:
            lines.append("This permanently deletes selected item(s) from data/trash and cannot be undone.")
        else:
            lines.append("This deletes only the selected bundles/files, not active experiment run folders.")
        choice = QMessageBox.question(
            self,
            "Permanently Delete Trash Item" if trash_only else "Delete Selected File/Bundle",
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

    def _sync_filter_options(self, state: DataManagementViewState) -> None:
        _replace_combo_options(
            self.root_filter_combo,
            [("Any root", "all"), *[(value, value) for value in state.root_filter_options if value != "all"]],
        )
        _replace_combo_options(
            self.experiment_filter_combo,
            [("Any experiment", "all"), *[(value, value) for value in state.experiment_filter_options if value != "all"]],
        )
        _replace_combo_options(
            self.trust_filter_combo,
            [("Any trust", "all"), *[(value, value) for value in state.trust_mode_filter_options if value != "all"]],
        )

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


def _combo_with_options(options: list[tuple[str, str]]) -> QComboBox:
    combo = QComboBox()
    for label, value in options:
        combo.addItem(label, value)
    return combo


def _bool_filter_combo(label: str) -> QComboBox:
    return _combo_with_options([(f"Any {label}", "all"), ("true", "true"), ("false", "false")])


def _replace_combo_options(combo: QComboBox, options: list[tuple[str, str]]) -> None:
    current = combo.currentData()
    existing = [combo.itemData(index) for index in range(combo.count())]
    target = [value for _, value in options]
    if existing == target:
        return
    with QSignalBlocker(combo):
        combo.clear()
        for label, value in options:
            combo.addItem(label, value)
        index = combo.findData(current)
        if index >= 0:
            combo.setCurrentIndex(index)


def _table_values(item, project_root: Path) -> list[str]:
    run_dir = _run_dir_for_item(item)
    if run_dir is not None:
        try:
            summary = summarize_run(run_dir)
            mode_segment = summary.operating_mode
            if summary.active_segment:
                mode_segment = f"{mode_segment} / {summary.active_segment}"
            flags = " | ".join(
                value
                for value in [
                    item.display_status,
                    f"model={summary.valid_for_model_training}",
                    f"thesis={summary.valid_for_thesis_repeatability}",
                    summary.stop_or_failure_reason,
                    f"review={summary.review.review_status}",
                ]
                if value
            )
            return [
                summary.timestamp_label,
                summary.experiment_name,
                summary.run_id,
                summary.validation_status,
                summary.run_trust_mode,
                mode_segment,
                flags,
                _relative_path(item.path, project_root),
            ]
        except Exception:
            pass
    return [
        item.timestamp_label,
        item.category_label,
        item.readable_name,
        item.item_type,
        item.status or item.display_status,
        "",
        item.display_status,
        _relative_path(item.path, project_root),
    ]


def _run_dir_for_item(item) -> Path | None:
    if item.category_key not in {"experiments", "modeling", "diagnostics", "trash"}:
        return None
    path = item.path if item.path.is_dir() else item.path.parent
    if (path / "summary.json").exists() or (path / "metadata.json").exists() or (path / "evaluation_metadata.json").exists():
        return path
    return None

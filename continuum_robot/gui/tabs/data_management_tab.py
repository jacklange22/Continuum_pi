"""Operator-facing data and artifact browser."""

from __future__ import annotations

from pathlib import Path
from time import perf_counter

from PySide6.QtCore import QSignalBlocker, Qt, QUrl
from PySide6.QtGui import QColor, QDesktopServices

from typing import Any
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
    QToolButton,
    QTreeWidget,
    QTreeWidgetItem,
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

    COLUMN_LABELS = ["Name", "Status", "Size", "When", "Path"]
    TREE_PATH_ROLE = Qt.UserRole + 1
    TREE_GROUP_PATHS_ROLE = Qt.UserRole + 2
    TREE_DEBUG_PATHS_ROLE = Qt.UserRole + 3

    def __init__(
        self, controller: DataManagementController, *, open_in_ann_training=None, parent=None
    ) -> None:
        super().__init__(parent)
        self.controller = controller
        self._ann_training_opener = open_in_ann_training
        self._tree_fingerprint: tuple | None = None
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

        title = QLabel("Data")
        title.setProperty("role", "title")
        self.summary_stats_label = QLabel("Scanning…")
        self.summary_stats_label.setStyleSheet(
            f"color: {COLORS.text_primary}; font-weight: 600; padding: 4px 0;"
        )
        self.summary_stats_label.setWordWrap(True)
        root.addWidget(title)
        root.addWidget(self.summary_stats_label)

        filters_card = _Section()
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
        self.preset_combo = _combo_with_options(
            [
                ("Presets", "all"),
                ("Today", "today"),
                ("Real hardware", "real_hardware"),
                ("Single segment", "single_segment"),
                ("Segment B", "segment_b"),
                ("Needs review", "needs_review"),
                ("Thesis/advisor", "thesis_advisor"),
                ("Large files", "large_files"),
                ("Generated diagnostics", "generated_diagnostics"),
                ("Trash candidates", "trash_candidates"),
            ]
        )
        self.preset_combo.currentIndexChanged.connect(self._on_preset_changed)
        self.refresh_button = QPushButton("Refresh")
        self.refresh_button.setProperty("variant", "ghost")
        self.refresh_button.clicked.connect(self._refresh_catalog)
        filter_row.addWidget(QLabel("Category"))
        filter_row.addWidget(self.category_combo, 0)
        filter_row.addWidget(QLabel("Preset"))
        filter_row.addWidget(self.preset_combo, 0)
        filter_row.addWidget(self.search_input, 1)
        filter_row.addWidget(self.refresh_button, 0)
        filters_card.body_layout.addLayout(filter_row)

        more_filters_section = _CollapsibleSection("More filters", expanded=False)

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
        more_filters_section.body_layout.addLayout(filter_row_2)

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
        more_filters_section.body_layout.addLayout(filter_row_3)
        filters_card.body_layout.addWidget(more_filters_section)

        self.tree = QTreeWidget()
        self.tree.setColumnCount(len(self.COLUMN_LABELS))
        self.tree.setHeaderLabels(self.COLUMN_LABELS)
        self.tree.setSelectionMode(QTreeWidget.ExtendedSelection)
        self.tree.setUniformRowHeights(True)
        self.tree.setAllColumnsShowFocus(True)
        self.tree.setAlternatingRowColors(True)
        self.tree.setRootIsDecorated(True)
        self.tree.setExpandsOnDoubleClick(True)
        self.tree.setMinimumHeight(360)
        tree_header = self.tree.header()
        tree_header.setSectionResizeMode(0, QHeaderView.Stretch)
        tree_header.setSectionResizeMode(1, QHeaderView.ResizeToContents)
        tree_header.setSectionResizeMode(2, QHeaderView.ResizeToContents)
        tree_header.setSectionResizeMode(3, QHeaderView.ResizeToContents)
        tree_header.setSectionResizeMode(4, QHeaderView.ResizeToContents)
        self.tree.itemSelectionChanged.connect(self._sync_selection_from_tree)
        # Hidden table retained for back-compat with existing helpers/tests that
        # may still reference it. The tree above is the active operator surface.
        self.table = QTableWidget(0, len(self.COLUMN_LABELS))
        self.table.setVisible(False)

        # "Quick clean" call-to-action: surfaces only when there are debug-marked runs.
        self.quick_clean_button = QPushButton("Trash debug-marked runs")
        self.quick_clean_button.setProperty("role", "primary")
        self.quick_clean_button.clicked.connect(self._quick_clean_debug)
        self.quick_clean_button.setVisible(False)
        self.quick_clean_label = QLabel("")
        self.quick_clean_label.setStyleSheet(f"color: {COLORS.text_muted};")
        self.quick_clean_label.setVisible(False)
        quick_clean_row = QHBoxLayout()
        quick_clean_row.setContentsMargins(0, 0, 0, 0)
        quick_clean_row.setSpacing(8)
        quick_clean_row.addWidget(self.quick_clean_button)
        quick_clean_row.addWidget(self.quick_clean_label, 1)

        root.addWidget(filters_card)
        root.addLayout(quick_clean_row)
        root.addWidget(self.tree, 1)

        actions_card = _Section()
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
        self.open_ann_training_button = QPushButton("Open in ANN Training")
        self.open_ann_training_button.setProperty("variant", "ghost")
        self.open_ann_training_button.clicked.connect(self._open_ann_training_from_data_tab)
        self.delete_button = QPushButton("Delete Selected File/Bundle")
        self.delete_button.setProperty("variant", "danger")
        self.delete_button.clicked.connect(self._delete_selected)
        action_row.addWidget(self.open_button)
        action_row.addWidget(self.reveal_button)
        action_row.addWidget(self.copy_path_button)
        action_row.addWidget(self.open_ann_training_button)
        action_row.addSpacing(16)

        advanced_actions_section = _CollapsibleSection("Advanced actions", expanded=False)

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
        advanced_actions_section.body_layout.addLayout(export_row)
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
        advanced_actions_section.body_layout.addLayout(export_copy_row)
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
        advanced_actions_section.body_layout.addLayout(validation_row)
        # Two-segment modeling lives on the Modeling tab; modeling-bundle export
        # is covered by Export Human Packet on any run. Hidden no-op widgets
        # kept for back-compat with controller state flags and tests.
        self.run_two_segment_modeling_button = QPushButton("Run Two-Segment Modeling")
        self.run_two_segment_modeling_button.clicked.connect(self._run_two_segment_modeling)
        self.run_two_segment_modeling_button.setVisible(False)
        self.open_modeling_summary_button = QPushButton("Open Modeling Summary")
        self.open_modeling_summary_button.clicked.connect(self._open_modeling_summary)
        self.open_modeling_summary_button.setVisible(False)
        self.export_modeling_bundle_button = QPushButton("Export Modeling Bundle")
        self.export_modeling_bundle_button.clicked.connect(self._export_modeling_bundle)
        self.export_modeling_bundle_button.setVisible(False)
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
        advanced_actions_section.body_layout.addLayout(review_row)
        self.archive_run_button = QPushButton("Archive Selected Run")
        self.archive_run_button.setProperty("variant", "ghost")
        self.archive_run_button.clicked.connect(self._archive_selected_run)
        self.trash_run_button = QPushButton("Move Selected Run to Trash")
        self.trash_run_button.setProperty("variant", "danger")
        self.trash_run_button.clicked.connect(self._trash_selected_run)
        # Evidence-index build is a one-off thesis-rollup operation; available
        # via the CLI script when needed. Hidden no-op kept for compatibility.
        self.build_evidence_index_button = QPushButton("Build Evidence Index")
        self.build_evidence_index_button.clicked.connect(self._build_evidence_index)
        self.build_evidence_index_button.setVisible(False)
        self.include_debug_evidence_checkbox = QCheckBox("Include debug")
        self.include_debug_evidence_checkbox.setVisible(False)
        self.include_mock_evidence_checkbox = QCheckBox("Include mock")
        self.include_mock_evidence_checkbox.setVisible(False)
        action_row.addWidget(self.archive_run_button)
        action_row.addWidget(self.trash_run_button)
        action_row.addWidget(self.delete_button)
        action_row.addStretch(1)
        actions_card.body_layout.addLayout(action_row)
        actions_card.body_layout.addWidget(advanced_actions_section)
        self.status_label = QLabel("Browse saved calibration, experiment, modeling, and diagnostic artifacts.")
        self.status_label.setWordWrap(True)
        self.status_label.setStyleSheet(f"color: {COLORS.text_primary};")
        actions_card.body_layout.addWidget(self.status_label)
        root.addWidget(actions_card)

        # Migration buttons + diagnostic key/value widgets removed. The
        # project has no legacy `runs/` directory; migration is a dead
        # feature. selection_pairs / root_pairs / performance_pairs are kept
        # only as hidden widgets so update() and tests that reference them
        # don't break. The new tree already surfaces per-run status, size,
        # and path on each row.
        self.selection_pairs = _PairsWidget()
        self.selection_pairs.setVisible(False)
        self.root_pairs = _PairsWidget()
        self.root_pairs.setVisible(False)
        self.migration_pairs = _PairsWidget()
        self.migration_pairs.setVisible(False)
        self.performance_pairs = _PairsWidget()
        self.performance_pairs.setVisible(False)
        self.preview_migration_button = QPushButton("Preview Legacy Migration")
        self.preview_migration_button.clicked.connect(self._preview_migration)
        self.preview_migration_button.setVisible(False)
        self.apply_migration_button = QPushButton("Apply Previewed Migration")
        self.apply_migration_button.clicked.connect(self._apply_migration)
        self.apply_migration_button.setVisible(False)
        self.open_migration_report_button = QPushButton("Open Migration Ledger")
        self.open_migration_report_button.clicked.connect(self._open_migration_report)
        self.open_migration_report_button.setVisible(False)

    def update(self, state: DataManagementViewState) -> None:
        self._sync_filter_options(state)
        self._update_summary_stats(state)
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
        table_ms = self._sync_table(state)
        self.selection_pairs.set_pairs(state.detail_pairs)
        self.migration_pairs.set_pairs(state.migration_summary_pairs)
        self.performance_pairs.set_pairs([*state.performance_pairs, ("Table population", f"{table_ms:.1f} ms")])
        self.root_pairs.set_pairs(state.root_summary_pairs)
        self.status_label.setText(state.status_message)
        self.open_button.setEnabled(state.can_open)
        self.reveal_button.setEnabled(state.can_reveal)
        self.copy_path_button.setEnabled(state.can_copy_path)
        ann_ready = (
            self._ann_training_opener is not None
            and len(state.selected_paths) == 1
            and any("collect_pose_command_dataset" in str(p).replace("\\", "/") for p in state.selected_paths)
        )
        self.open_ann_training_button.setVisible(self._ann_training_opener is not None)
        self.open_ann_training_button.setEnabled(bool(ann_ready))
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

    def _update_summary_stats(self, state: DataManagementViewState) -> None:
        total_count = len(state.items)
        shown_count = len(state.filtered_items)
        total_bytes = 0
        trash_count = 0
        mock_count = 0
        keep_count = 0
        for summary in state.run_summaries_by_path.values():
            try:
                total_bytes += int(getattr(summary, "total_size_bytes", 0) or 0)
            except (TypeError, ValueError):
                pass
            review = getattr(summary, "review", None)
            review_status = str(getattr(review, "review_status", "") or "").lower()
            if review_status in {"garbage", "debug"}:
                trash_count += 1
            elif review_status in {"keep", "thesis_candidate", "advisor_share", "archived"}:
                keep_count += 1
            if bool(getattr(summary, "mock_mode", False)):
                mock_count += 1
        parts = [
            f"{shown_count} of {total_count} shown",
            f"{_format_bytes(total_bytes)}",
        ]
        if mock_count:
            parts.append(f"{mock_count} mock-mode")
        if trash_count:
            parts.append(f"{trash_count} marked trash")
        if keep_count:
            parts.append(f"{keep_count} kept")
        self.summary_stats_label.setText("  ·  ".join(parts))

    def _on_category_changed(self, _index: int) -> None:
        self.controller.set_category_filter(str(self.category_combo.currentData() or "all"))
        self.update(self.controller.refresh())

    def _on_search_changed(self, value: str) -> None:
        self.controller.set_search_text(value)
        self.update(self.controller.refresh())

    def _on_filter_changed(self, key: str, value: str) -> None:
        self.controller.set_filter_value(key, value)
        self.update(self.controller.refresh())

    def _on_preset_changed(self, _index: int) -> None:
        self.controller.apply_filter_preset(str(self.preset_combo.currentData() or "all"))
        self.update(self.controller.refresh())

    def _sync_table(self, state: DataManagementViewState) -> float:
        return self._sync_tree(state)

    def _sync_tree(self, state: DataManagementViewState) -> float:
        started = perf_counter()
        project_root = self.controller.project_root
        # Build hierarchical buckets: category_label -> experiment_name -> [items]
        category_order: list[str] = []
        category_buckets: dict[str, dict[str, list]] = {}
        item_summary: dict[str, Any] = {}
        for item in state.filtered_items:
            run_dir = _run_dir_for_item(item)
            summary = state.run_summaries_by_path.get(str(run_dir)) if run_dir is not None else None
            item_summary[str(item.path)] = summary
            cat = item.category_label or item.category_key or "Other"
            if cat not in category_buckets:
                category_buckets[cat] = {}
                category_order.append(cat)
            experiment_key = ""
            if summary is not None:
                experiment_key = str(getattr(summary, "experiment_name", "") or "")
            elif item.category_key == "experiments":
                experiment_key = item.readable_name
            category_buckets[cat].setdefault(experiment_key, []).append(item)

        selected_paths = set(state.selected_paths)

        # Skip the rebuild when the underlying data hasn't changed. The
        # AppWindow refresh timer fires this method ~10 Hz; rebuilding the
        # tree every tick collapses expanded groups and disrupts selection.
        fingerprint = self._compute_tree_fingerprint(state, item_summary)
        if fingerprint == self._tree_fingerprint:
            self._update_tree_selection(selected_paths)
            self._update_quick_clean_button_from_state(state, item_summary)
            return (perf_counter() - started) * 1000.0

        expanded_keys = self._capture_expanded_keys()
        scroll_value = self.tree.verticalScrollBar().value() if self.tree.verticalScrollBar() else 0
        self._tree_fingerprint = fingerprint

        debug_run_count = 0
        debug_run_size = 0

        with QSignalBlocker(self.tree):
            self.tree.clear()
            for cat in category_order:
                experiments = category_buckets[cat]
                cat_items = [it for items in experiments.values() for it in items]
                cat_size = sum(self._size_for(it, item_summary) for it in cat_items)
                cat_paths = [str(it.path) for it in cat_items]
                cat_node = QTreeWidgetItem([
                    f"{cat}  ({len(cat_items)})",
                    "",
                    _format_bytes(cat_size),
                    "",
                    "",
                ])
                cat_node.setData(0, self.TREE_GROUP_PATHS_ROLE, cat_paths)
                cat_node.setData(0, self.TREE_DEBUG_PATHS_ROLE, [
                    str(it.path) for it in cat_items
                    if self._is_debug(item_summary.get(str(it.path)))
                ])
                cat_node.setFirstColumnSpanned(False)
                self.tree.addTopLevelItem(cat_node)
                # If only one experiment in this category, fold runs directly under category.
                if len(experiments) == 1 and not next(iter(experiments)).strip():
                    runs = next(iter(experiments.values()))
                    for run_item in runs:
                        run_node = self._build_run_node(run_item, item_summary.get(str(run_item.path)), project_root)
                        cat_node.addChild(run_node)
                        if str(run_item.path) in selected_paths:
                            run_node.setSelected(True)
                        if self._is_debug(item_summary.get(str(run_item.path))):
                            debug_run_count += 1
                            debug_run_size += self._size_for(run_item, item_summary)
                else:
                    for experiment_name, runs in sorted(experiments.items()):
                        runs_sorted = sorted(runs, key=lambda it: it.timestamp_sort_key, reverse=True)
                        exp_size = sum(self._size_for(it, item_summary) for it in runs_sorted)
                        keep_count = sum(1 for it in runs_sorted if self._is_keeper(item_summary.get(str(it.path))))
                        debug_count = sum(1 for it in runs_sorted if self._is_debug(item_summary.get(str(it.path))))
                        title = experiment_name or "(other)"
                        signals = []
                        if keep_count:
                            signals.append(f"★ {keep_count} keep")
                        if debug_count:
                            signals.append(f"🗑 {debug_count} debug")
                        exp_node = QTreeWidgetItem([
                            f"{title}  ({len(runs_sorted)})",
                            "  ·  ".join(signals),
                            _format_bytes(exp_size),
                            "",
                            "",
                        ])
                        exp_paths = [str(it.path) for it in runs_sorted]
                        exp_debug_paths = [
                            str(it.path) for it in runs_sorted
                            if self._is_debug(item_summary.get(str(it.path)))
                        ]
                        exp_node.setData(0, self.TREE_GROUP_PATHS_ROLE, exp_paths)
                        exp_node.setData(0, self.TREE_DEBUG_PATHS_ROLE, exp_debug_paths)
                        cat_node.addChild(exp_node)
                        for run_item in runs_sorted:
                            run_node = self._build_run_node(run_item, item_summary.get(str(run_item.path)), project_root)
                            exp_node.addChild(run_node)
                            if str(run_item.path) in selected_paths:
                                run_node.setSelected(True)
                            if self._is_debug(item_summary.get(str(run_item.path))):
                                debug_run_count += 1
                                debug_run_size += self._size_for(run_item, item_summary)
            self._restore_expanded_keys(expanded_keys)

        if self.tree.verticalScrollBar() is not None:
            self.tree.verticalScrollBar().setValue(scroll_value)
        self._update_quick_clean_button(debug_run_count, debug_run_size)
        return (perf_counter() - started) * 1000.0

    def _compute_tree_fingerprint(self, state: DataManagementViewState, item_summary: dict) -> tuple:
        parts: list[tuple] = []
        for item in state.filtered_items:
            path = str(item.path)
            summary = item_summary.get(path)
            if summary is not None:
                review = getattr(getattr(summary, "review", None), "review_status", "")
                parts.append((
                    path,
                    item.category_key,
                    str(getattr(summary, "experiment_name", "") or ""),
                    int(getattr(summary, "total_size_bytes", 0) or 0),
                    str(review or ""),
                    bool(getattr(summary, "mock_mode", False)),
                    str(getattr(summary, "run_trust_mode", "") or ""),
                    item.timestamp_label,
                ))
            else:
                parts.append((
                    path,
                    item.category_key,
                    "",
                    0,
                    str(getattr(item, "status", "") or ""),
                    False,
                    "",
                    item.timestamp_label,
                ))
        return tuple(parts)

    def _capture_expanded_keys(self) -> set[str]:
        keys: set[str] = set()

        def walk(node: QTreeWidgetItem, path: str) -> None:
            child_key = f"{path}>{node.text(0)}"
            if node.isExpanded():
                keys.add(child_key)
            for index in range(node.childCount()):
                walk(node.child(index), child_key)

        for index in range(self.tree.topLevelItemCount()):
            walk(self.tree.topLevelItem(index), "")
        return keys

    def _restore_expanded_keys(self, expanded_keys: set[str]) -> None:
        # If the user has never touched the tree, default to expanding the
        # Experiments category so the most-interesting groups are visible.
        default_to_experiments = not expanded_keys

        def walk(node: QTreeWidgetItem, path: str) -> None:
            child_key = f"{path}>{node.text(0)}"
            if child_key in expanded_keys:
                node.setExpanded(True)
            elif default_to_experiments and not path and node.text(0).lower().startswith("experiments"):
                node.setExpanded(True)
            for index in range(node.childCount()):
                walk(node.child(index), child_key)

        for index in range(self.tree.topLevelItemCount()):
            walk(self.tree.topLevelItem(index), "")

    def _update_tree_selection(self, selected_paths: set[str]) -> None:
        with QSignalBlocker(self.tree):
            def walk(node: QTreeWidgetItem) -> None:
                path = node.data(0, self.TREE_PATH_ROLE)
                if path is not None:
                    node.setSelected(str(path) in selected_paths)
                for index in range(node.childCount()):
                    walk(node.child(index))

            for index in range(self.tree.topLevelItemCount()):
                walk(self.tree.topLevelItem(index))

    def _update_quick_clean_button_from_state(self, state: DataManagementViewState, item_summary: dict) -> None:
        debug_count = 0
        debug_size = 0
        for item in state.filtered_items:
            summary = item_summary.get(str(item.path))
            if self._is_debug(summary):
                debug_count += 1
                debug_size += self._size_for(item, {str(item.path): summary})
        self._update_quick_clean_button(debug_count, debug_size)

    def _build_run_node(self, item, summary, project_root: Path) -> QTreeWidgetItem:
        status = _row_status_label(item, summary) if summary is not None else (item.status or item.display_status or "—")
        size_bytes = self._size_for(item, {str(item.path): summary})
        timestamp = item.timestamp_label
        if summary is not None:
            ts = getattr(summary, "timestamp_label", None)
            if ts:
                timestamp = ts
        node = QTreeWidgetItem([
            f"  {item.readable_name}",
            status,
            _format_bytes(size_bytes),
            timestamp,
            _relative_path(item.path, project_root),
        ])
        node.setData(0, self.TREE_PATH_ROLE, str(item.path))
        node.setToolTip(0, str(item.path))
        node.setToolTip(4, str(item.path))
        if self._is_keeper(summary):
            node.setForeground(0, QColor(COLORS.success_fg))
        elif self._is_debug(summary):
            node.setForeground(0, QColor(COLORS.text_muted))
        return node

    @staticmethod
    def _size_for(item, item_summary: dict) -> int:
        summary = item_summary.get(str(item.path)) if isinstance(item_summary, dict) else None
        if summary is not None:
            try:
                return int(getattr(summary, "total_size_bytes", 0) or 0)
            except (TypeError, ValueError):
                return 0
        return _item_size_bytes_fallback(item.path)

    @staticmethod
    def _is_debug(summary) -> bool:
        if summary is None:
            return False
        review = getattr(getattr(summary, "review", None), "review_status", "")
        return str(review or "").lower() in {"debug", "garbage"}

    @staticmethod
    def _is_keeper(summary) -> bool:
        if summary is None:
            return False
        review = getattr(getattr(summary, "review", None), "review_status", "")
        return str(review or "").lower() in {"keep", "thesis_candidate", "advisor_share", "archived"}

    def _update_quick_clean_button(self, debug_count: int, debug_size: int) -> None:
        if debug_count > 0:
            self.quick_clean_button.setText(
                f"Trash {debug_count} debug-marked run{'s' if debug_count != 1 else ''} ({_format_bytes(debug_size)})"
            )
            self.quick_clean_button.setVisible(True)
            self.quick_clean_label.setText("Removes runs marked debug or garbage. Asks for confirmation.")
            self.quick_clean_label.setVisible(True)
        else:
            self.quick_clean_button.setVisible(False)
            self.quick_clean_label.setVisible(False)

    def _quick_clean_debug(self) -> None:
        state = self.controller.refresh()
        # Collect all debug-marked paths visible across the tree (top-level GROUP_PATHS_ROLE).
        debug_paths: list[str] = []
        for index in range(self.tree.topLevelItemCount()):
            node = self.tree.topLevelItem(index)
            paths = node.data(0, self.TREE_DEBUG_PATHS_ROLE) or []
            debug_paths.extend(str(p) for p in paths)
            # also descend one level for sub-experiment buckets
            for child_index in range(node.childCount()):
                child = node.child(child_index)
                child_paths = child.data(0, self.TREE_DEBUG_PATHS_ROLE) or []
                debug_paths.extend(str(p) for p in child_paths)
        # dedupe
        debug_paths = sorted(set(debug_paths))
        if not debug_paths:
            return
        choice = QMessageBox.question(
            self,
            "Trash debug-marked runs",
            f"Move {len(debug_paths)} debug-marked run(s) to data/trash?\n\n"
            "They will not be permanently deleted. Empty Trash later from the Trash category.",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No,
        )
        if choice != QMessageBox.Yes:
            return
        self.controller.set_selected_paths(debug_paths)
        self.update(self.controller.refresh())
        try:
            self.controller.trash_selected_run()
        except Exception as exc:
            self.status_label.setText(f"Quick clean failed: {exc}")
        else:
            self.status_label.setText(f"Moved {len(debug_paths)} run(s) to trash.")
        self.update(self.controller.refresh())

    def _sync_selection_from_tree(self) -> None:
        paths: list[str] = []
        for node in self.tree.selectedItems():
            path = node.data(0, self.TREE_PATH_ROLE)
            if path:
                paths.append(str(path))
                continue
            group_paths = node.data(0, self.TREE_GROUP_PATHS_ROLE) or []
            for child_path in group_paths:
                paths.append(str(child_path))
        # dedupe while preserving order
        seen = set()
        unique = []
        for p in paths:
            if p not in seen:
                seen.add(p)
                unique.append(p)
        self.controller.set_selected_paths(unique)
        self.update(self.controller.refresh())

    def _sync_selection_from_table(self) -> None:
        # Back-compat shim; real selection now syncs from the tree.
        self._sync_selection_from_tree()

    def _open_selected(self) -> None:
        selected = self.controller.selected_items()
        if not selected:
            return
        for item in selected:
            QDesktopServices.openUrl(QUrl.fromLocalFile(str(item.path)))

    def _open_ann_training_from_data_tab(self) -> None:
        if self._ann_training_opener is None:
            return
        state = self.controller.refresh()
        paths = [str(p) for p in state.selected_paths]
        if len(paths) != 1:
            return
        self._ann_training_opener(paths[0])

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


def _format_bytes(value: int) -> str:
    size = float(max(0, int(value or 0)))
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if size < 1024.0 or unit == "TB":
            if unit == "B":
                return f"{int(size)} {unit}"
            return f"{size:.1f} {unit}"
        size /= 1024.0
    return f"{size:.1f} TB"


class _CollapsibleSection(QWidget):
    """Inline collapsible section with a chevron toggle. Hidden body by default."""

    def __init__(self, title: str, *, expanded: bool = False, parent=None) -> None:
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)
        header = QHBoxLayout()
        header.setContentsMargins(0, 0, 0, 0)
        header.setSpacing(6)
        self.toggle = QToolButton()
        self.toggle.setCheckable(True)
        self.toggle.setChecked(bool(expanded))
        self.toggle.setArrowType(Qt.DownArrow if expanded else Qt.RightArrow)
        self.toggle.setToolButtonStyle(Qt.ToolButtonIconOnly)
        self.toggle.setAutoRaise(True)
        title_label = QLabel(title)
        title_label.setStyleSheet(f"color: {COLORS.text_secondary}; font-weight: 600;")
        title_label.setCursor(Qt.PointingHandCursor)
        title_label.mousePressEvent = lambda _e: self.toggle.setChecked(not self.toggle.isChecked())
        header.addWidget(self.toggle)
        header.addWidget(title_label, 1)
        layout.addLayout(header)
        self.body = QWidget()
        self.body.setVisible(bool(expanded))
        self.body_layout = QVBoxLayout(self.body)
        self.body_layout.setContentsMargins(20, 0, 0, 0)
        self.body_layout.setSpacing(8)
        layout.addWidget(self.body)
        self.toggle.toggled.connect(self._on_toggled)

    def _on_toggled(self, checked: bool) -> None:
        self.toggle.setArrowType(Qt.DownArrow if checked else Qt.RightArrow)
        self.body.setVisible(bool(checked))


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


class _Section(QWidget):
    """Borderless container, used for thin un-titled sections in the data tab."""

    def __init__(self) -> None:
        super().__init__()
        self.body_layout = QVBoxLayout(self)
        self.body_layout.setContentsMargins(0, 0, 0, 0)
        self.body_layout.setSpacing(8)


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


_STATUS_PRIORITY = {
    "trash": 0,
    "mock": 1,
    "review": 2,
    "thesis": 3,
    "keep": 4,
    "active": 5,
    "archived": 6,
}


def _row_status_label(item, summary) -> str:
    review = str(getattr(getattr(summary, "review", None), "review_status", "") or "").lower()
    if review in {"garbage", "debug"}:
        return "trash"
    if review == "thesis_candidate":
        return "thesis"
    if review == "keep":
        return "keep"
    if review == "advisor_share":
        return "advisor"
    if review == "archived":
        return "archived"
    if bool(getattr(summary, "mock_mode", False)):
        return "mock"
    trust = str(getattr(summary, "run_trust_mode", "") or "").lower()
    if trust:
        return trust
    return str(getattr(item, "status", "") or getattr(item, "display_status", "") or "").lower() or "—"


def _table_values(item, project_root: Path, *, summary=None) -> list[str]:
    run_dir = _run_dir_for_item(item)
    if run_dir is not None:
        try:
            if summary is None:
                summary = summarize_run(run_dir)
            size_bytes = int(getattr(summary, "total_size_bytes", 0) or 0)
            return [
                summary.timestamp_label,
                summary.experiment_name,
                summary.run_id,
                _row_status_label(item, summary),
                _format_bytes(size_bytes),
                _relative_path(item.path, project_root),
            ]
        except Exception:
            pass
    return [
        item.timestamp_label,
        item.category_label,
        item.readable_name,
        item.status or item.display_status or "—",
        _format_bytes(_item_size_bytes_fallback(item.path)),
        _relative_path(item.path, project_root),
    ]


def _item_size_bytes_fallback(path: Path) -> int:
    try:
        if path.is_file():
            return int(path.stat().st_size)
        if path.is_dir():
            total = 0
            for entry in path.rglob("*"):
                if entry.is_file():
                    try:
                        total += int(entry.stat().st_size)
                    except OSError:
                        continue
            return total
    except OSError:
        pass
    return 0


def _run_dir_for_item(item) -> Path | None:
    if item.category_key not in {"experiments", "modeling", "diagnostics", "trash"}:
        return None
    path = item.path if item.path.is_dir() else item.path.parent
    if (path / "summary.json").exists() or (path / "metadata.json").exists() or (path / "evaluation_metadata.json").exists():
        return path
    return None

"""Canonical experiment workspace tab."""

from __future__ import annotations

from PySide6.QtCore import QSignalBlocker, Qt, QUrl
from PySide6.QtGui import QDesktopServices
from PySide6.QtWidgets import (
    QComboBox,
    QFileDialog,
    QFormLayout,
    QFrame,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QPlainTextEdit,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QSplitter,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from continuum_robot.gui.controllers.experiment_controller import ExperimentViewState
from continuum_robot.gui.widgets.experiment_3d_widget import Experiment3DWidget
from continuum_robot.gui.widgets.experiment_parameter_editor import ExperimentParameterEditor
from continuum_robot.gui.widgets.experiment_preflight_widget import ExperimentPreflightWidget
from continuum_robot.gui.widgets.experiment_results_widget import ExperimentResultsWidget


class ExperimentTab(QWidget):
    """Generic validation and data-generation workspace for canonical experiments."""

    _COLOR_MODES = [
        ("Target Point", "target_point"),
        ("Validity", "validity"),
        ("Phase", "phase"),
        ("Revisit Index", "revisit_index"),
        ("Inlier / Outlier", "inlier_outlier"),
    ]

    def __init__(self, controller, parent=None) -> None:
        super().__init__(parent)
        self.controller = controller
        self.setObjectName("experimentWorkspace")
        self.setStyleSheet(
            """
            QWidget#experimentWorkspace {
                background: #eef3f8;
                color: #0f172a;
            }
            QWidget#experimentWorkspace QLabel[role="page-title"] {
                font-size: 24px;
                font-weight: 700;
                color: #0f172a;
            }
            QWidget#experimentWorkspace QLabel[role="section-title"] {
                font-size: 15px;
                font-weight: 700;
                color: #0f172a;
            }
            QWidget#experimentWorkspace QLabel[role="body"] {
                color: #475569;
            }
            QWidget#experimentWorkspace QLabel[role="muted"] {
                color: #556476;
            }
            QWidget#experimentWorkspace QLabel[role="chip"] {
                padding: 5px 10px;
                border-radius: 999px;
                background: #e2e8f0;
                color: #0f172a;
                font-weight: 700;
            }
            QWidget#experimentWorkspace QFrame[role="card"] {
                background: #ffffff;
                border: 1px solid #d9e3ec;
                border-radius: 16px;
            }
            QWidget#experimentWorkspace QLineEdit,
            QWidget#experimentWorkspace QPlainTextEdit,
            QWidget#experimentWorkspace QTextEdit,
            QWidget#experimentWorkspace QListWidget,
            QWidget#experimentWorkspace QComboBox {
                border: 1px solid #dbe4ee;
                border-radius: 12px;
                background: #fbfdff;
                padding: 6px 8px;
            }
            QWidget#experimentWorkspace QListWidget {
                padding: 6px;
            }
            QWidget#experimentWorkspace QPushButton {
                min-height: 36px;
                padding: 0 14px;
                border: 1px solid #dbe4ee;
                border-radius: 10px;
                background: #f8fafc;
                color: #0f172a;
                font-weight: 600;
            }
            QWidget#experimentWorkspace QPushButton[variant="primary"] {
                background: #0f172a;
                color: #ffffff;
                border-color: #0f172a;
            }
            QWidget#experimentWorkspace QPushButton[variant="danger"] {
                background: #ffffff;
                color: #b91c1c;
                border-color: #fecaca;
            }
            QWidget#experimentWorkspace QPushButton[variant="ghost"] {
                background: transparent;
                color: #334155;
            }
            QWidget#experimentWorkspace QProgressBar {
                min-height: 22px;
                border: 1px solid #dbe4ee;
                border-radius: 10px;
                background: #eef2f7;
                text-align: center;
            }
            QWidget#experimentWorkspace QProgressBar::chunk {
                background: #2563eb;
                border-radius: 9px;
            }
            QWidget#experimentWorkspace QScrollArea {
                border: none;
                background: transparent;
            }
            """
        )

        self.page_title = QLabel("Experiment Workspace")
        self.page_title.setProperty("role", "page-title")
        self.page_subtitle = QLabel(
            "Use this workspace for structured validation runs, characterization, and dataset generation. "
            "Routine setup workflows stay in their dedicated tabs."
        )
        self.page_subtitle.setProperty("role", "body")
        self.page_subtitle.setWordWrap(True)

        self.selected_status_chip = QLabel("Ready")
        self.selected_status_chip.setProperty("role", "chip")
        self.selected_experiment_title = QLabel("Experiment")
        self.selected_experiment_title.setProperty("role", "section-title")
        self.selected_experiment_description = QLabel()
        self.selected_experiment_description.setWordWrap(True)
        self.selected_experiment_description.setProperty("role", "body")
        self.selected_badges_label = QLabel()
        self.selected_badges_label.setWordWrap(True)
        self.selected_badges_label.setProperty("role", "muted")

        header_card = _SectionCard()
        header_row = QHBoxLayout()
        header_row.setContentsMargins(0, 0, 0, 0)
        header_row.setSpacing(18)
        header_left = QWidget()
        header_left_layout = QVBoxLayout(header_left)
        header_left_layout.setContentsMargins(0, 0, 0, 0)
        header_left_layout.setSpacing(6)
        header_left_layout.addWidget(self.page_title)
        header_left_layout.addWidget(self.page_subtitle)
        header_right = QWidget()
        header_right_layout = QVBoxLayout(header_right)
        header_right_layout.setContentsMargins(0, 0, 0, 0)
        header_right_layout.setSpacing(8)
        header_right_layout.addWidget(self.selected_status_chip, 0, Qt.AlignRight)
        header_right_layout.addWidget(self.selected_experiment_title)
        header_right_layout.addWidget(self.selected_experiment_description)
        header_right_layout.addWidget(self.selected_badges_label)
        header_row.addWidget(header_left, 3)
        header_row.addWidget(header_right, 4)
        header_card.body_layout.addLayout(header_row)

        self.experiment_combo = QComboBox()
        self.experiment_combo.currentIndexChanged.connect(self._on_experiment_selected)
        self.load_defaults_button = QPushButton("Load Defaults")
        self.load_defaults_button.clicked.connect(self.controller.load_defaults)
        self.load_run_button = QPushButton("Load Run Folder")
        self.load_run_button.clicked.connect(self._browse_run)
        selector_card = _SectionCard(
            "Experiment",
            "Select a structured validation or data-generation run. Setup and calibration stay in their dedicated tabs.",
        )
        selector_row = QHBoxLayout()
        selector_row.setContentsMargins(0, 0, 0, 0)
        selector_row.setSpacing(10)
        selector_row.addWidget(self.experiment_combo, 1)
        selector_row.addWidget(self.load_defaults_button)
        selector_row.addWidget(self.load_run_button)
        selector_card.body_layout.addLayout(selector_row)

        self.output_root_edit = QLineEdit()
        self.output_root_edit.editingFinished.connect(self._on_output_root_changed)
        self.notes_edit = QPlainTextEdit()
        self.notes_edit.setPlaceholderText("Short operator note stored in metadata.json")
        self.notes_edit.setMinimumHeight(72)
        self.notes_edit.setMaximumHeight(96)
        self.notes_edit.textChanged.connect(lambda: self.controller.set_operator_notes(self.notes_edit.toPlainText()))

        self.parameter_editor = ExperimentParameterEditor()
        self.parameter_editor.setMinimumHeight(260)
        self.parameter_editor.fieldChanged.connect(self.controller.set_parameter_value)

        self.config_preview = QPlainTextEdit()
        self.config_preview.setReadOnly(True)
        self.config_preview.setPlaceholderText("Effective config preview")
        self.config_preview.setMinimumHeight(120)
        self.config_preview.setMaximumHeight(180)

        parameters_card = _SectionCard(
            "Parameters",
            "Edit the selected experiment parameters here. The canonical YAML snapshot is saved into each run folder.",
        )
        parameters_form = QFormLayout()
        parameters_form.setContentsMargins(0, 0, 0, 0)
        parameters_form.setSpacing(10)
        parameters_form.addRow("Output Root", self.output_root_edit)
        parameters_form.addRow("Operator Note", self.notes_edit)
        parameters_card.body_layout.addLayout(parameters_form)
        parameters_card.body_layout.addWidget(self.parameter_editor)
        preview_label = QLabel("Effective Config Preview")
        preview_label.setProperty("role", "muted")
        parameters_card.body_layout.addWidget(preview_label)
        parameters_card.body_layout.addWidget(self.config_preview)

        self.preflight_widget = ExperimentPreflightWidget()
        preflight_card = _SectionCard("Preflight", "Review blockers, warnings, and validation context before you run.")
        preflight_card.body_layout.addWidget(self.preflight_widget)

        self.checklist_widget = _KeyValueSummaryWidget()
        checklist_card = _SectionCard("Ready To Run", "A compact summary of what will happen when you start the run.")
        checklist_card.body_layout.addWidget(self.checklist_widget)

        self.run_button = QPushButton("Run Experiment")
        self.run_button.setProperty("variant", "primary")
        self.stop_button = QPushButton("Stop")
        self.stop_button.setProperty("variant", "danger")
        self.refresh_button = QPushButton("Refresh Checks")
        self.refresh_button.setProperty("variant", "ghost")
        self.refresh_button.clicked.connect(self.controller.refresh)
        self.run_button.clicked.connect(self._run)
        self.stop_button.clicked.connect(self.controller.stop)
        self.progress_bar = QProgressBar()
        self.status_text = QTextEdit()
        self.status_text.setReadOnly(True)
        self.status_text.setPlaceholderText("Run status and validation messages appear here.")
        self.status_text.setMinimumHeight(68)
        self.status_text.setMaximumHeight(92)
        self.destination_widget = _KeyValueSummaryWidget()
        self.open_output_button = QPushButton("Open Run Folder")
        self.open_output_button.setProperty("variant", "ghost")
        self.open_output_button.clicked.connect(self._open_current_run_folder)

        run_controls_card = _SectionCard("Run Controls", "Use the canonical runner, save outputs, and monitor status here.")
        run_buttons = QHBoxLayout()
        run_buttons.setContentsMargins(0, 0, 0, 0)
        run_buttons.setSpacing(10)
        run_buttons.addWidget(self.run_button, 3)
        run_buttons.addWidget(self.stop_button, 2)
        run_buttons.addWidget(self.refresh_button, 2)
        run_controls_card.body_layout.addLayout(run_buttons)
        run_controls_card.body_layout.addWidget(self.progress_bar)
        run_controls_card.body_layout.addWidget(self.destination_widget)
        run_controls_card.body_layout.addWidget(self.status_text)
        run_controls_card.body_layout.addWidget(self.open_output_button, 0, Qt.AlignLeft)

        control_stack = QWidget()
        control_stack_layout = QVBoxLayout(control_stack)
        control_stack_layout.setContentsMargins(0, 0, 0, 0)
        control_stack_layout.setSpacing(14)
        control_stack_layout.addWidget(selector_card)
        control_stack_layout.addWidget(parameters_card)
        control_stack_layout.addWidget(preflight_card)
        control_stack_layout.addWidget(checklist_card)
        control_stack_layout.addWidget(run_controls_card)
        control_stack_layout.addStretch(1)

        left_scroll = QScrollArea()
        left_scroll.setWidgetResizable(True)
        left_scroll.setWidget(control_stack)
        left_scroll.setMinimumWidth(340)

        self.viewer_3d = Experiment3DWidget(
            requested_mode=self.controller.settings.runtime.visualization_mode,
            safe_effects=self.controller.settings.runtime.visualization_safe_effects,
        )
        self.viewer_3d.setMinimumHeight(360)
        self.color_mode_combo = QComboBox()
        for label, value in self._COLOR_MODES:
            self.color_mode_combo.addItem(label, value)
        self.color_mode_combo.currentIndexChanged.connect(self._on_color_mode_changed)
        self.show_axes_button = QPushButton("Axes")
        self.show_axes_button.setProperty("variant", "ghost")
        self.show_axes_button.clicked.connect(lambda: self.controller.set_show_axes(not self.controller.state.show_axes))
        self.show_labels_button = QPushButton("Labels")
        self.show_labels_button.setProperty("variant", "ghost")
        self.show_labels_button.clicked.connect(
            lambda: self.controller.set_show_labels(not self.controller.state.show_labels)
        )
        self.show_centroids_button = QPushButton("Centroids")
        self.show_centroids_button.setProperty("variant", "ghost")
        self.show_centroids_button.clicked.connect(
            lambda: self.controller.set_show_centroids(not self.controller.state.show_centroids)
        )
        self.show_truth_button = QPushButton("Truth")
        self.show_truth_button.setProperty("variant", "ghost")
        self.show_truth_button.clicked.connect(lambda: self.controller.set_show_truth(not self.controller.state.show_truth))
        self.save_view_button = QPushButton("Save View")
        self.save_view_button.setProperty("variant", "ghost")
        self.save_view_button.clicked.connect(self._save_view_image)

        visualization_card = _SectionCard(
            "Visualization",
            "Current run samples or loaded history appear here. Specialized plots are used when available; generic summaries are shown otherwise.",
        )
        viz_toolbar = QHBoxLayout()
        viz_toolbar.setContentsMargins(0, 0, 0, 0)
        viz_toolbar.setSpacing(10)
        viz_toolbar.addWidget(QLabel("Coloring"))
        viz_toolbar.addWidget(self.color_mode_combo, 0)
        viz_toolbar.addStretch(1)
        viz_toolbar.addWidget(self.show_axes_button)
        viz_toolbar.addWidget(self.show_labels_button)
        viz_toolbar.addWidget(self.show_centroids_button)
        viz_toolbar.addWidget(self.show_truth_button)
        viz_toolbar.addWidget(self.save_view_button)
        visualization_card.body_layout.addLayout(viz_toolbar)
        visualization_card.body_layout.addWidget(self.viewer_3d)

        self.result_details_widget = _KeyValueSummaryWidget()
        self.results_widget = ExperimentResultsWidget()
        self.export_plot_button = QPushButton("Save Current Plot")
        self.export_plot_button.setProperty("variant", "ghost")
        self.export_plot_button.clicked.connect(self._save_plot_image)
        results_card = _SectionCard(
            "Results",
            "Review paths, summary metrics, warnings, and any built-in plots after the run completes or when a historical run is loaded.",
        )
        results_toolbar = QHBoxLayout()
        results_toolbar.setContentsMargins(0, 0, 0, 0)
        results_toolbar.addStretch(1)
        results_toolbar.addWidget(self.export_plot_button)
        results_card.body_layout.addLayout(results_toolbar)
        results_card.body_layout.addWidget(self.result_details_widget)
        results_card.body_layout.addWidget(self.results_widget)

        center_splitter = QSplitter(Qt.Vertical)
        center_splitter.setChildrenCollapsible(False)
        center_splitter.setHandleWidth(8)
        center_splitter.addWidget(visualization_card)
        center_splitter.addWidget(results_card)
        center_splitter.setStretchFactor(0, 4)
        center_splitter.setStretchFactor(1, 3)
        center_splitter.setSizes([620, 360])

        self.history_list = QListWidget()
        self.history_list.itemDoubleClicked.connect(self._load_selected_history_item)
        self.history_list.setSpacing(10)
        self.history_list.setMinimumWidth(280)
        self.history_refresh_button = QPushButton("Refresh")
        self.history_refresh_button.setProperty("variant", "ghost")
        self.history_refresh_button.clicked.connect(self.controller.refresh)
        history_card = _SectionCard(
            "Run History",
            "Recent runs for the currently selected experiment. Double-click any entry to reload its outputs and summaries.",
        )
        history_header = QHBoxLayout()
        history_header.setContentsMargins(0, 0, 0, 0)
        history_header.addWidget(QLabel("Recent runs"))
        history_header.addStretch(1)
        history_header.addWidget(self.history_refresh_button)
        history_card.body_layout.addLayout(history_header)
        history_card.body_layout.addWidget(self.history_list)

        main_splitter = QSplitter(Qt.Horizontal)
        main_splitter.setChildrenCollapsible(False)
        main_splitter.setHandleWidth(8)
        main_splitter.addWidget(left_scroll)
        main_splitter.addWidget(center_splitter)
        main_splitter.addWidget(history_card)
        main_splitter.setStretchFactor(0, 2)
        main_splitter.setStretchFactor(1, 5)
        main_splitter.setStretchFactor(2, 2)
        main_splitter.setSizes([360, 980, 280])

        layout = QVBoxLayout(self)
        layout.setContentsMargins(18, 18, 18, 18)
        layout.setSpacing(16)
        layout.addWidget(header_card)
        layout.addWidget(main_splitter)

    def update(self, state: ExperimentViewState) -> None:
        with QSignalBlocker(self.notes_edit):
            if self.notes_edit.toPlainText() != state.operator_notes:
                self.notes_edit.setPlainText(state.operator_notes)
        with QSignalBlocker(self.output_root_edit):
            if self.output_root_edit.text() != state.output_root:
                self.output_root_edit.setText(state.output_root)
        with QSignalBlocker(self.config_preview):
            if self.config_preview.toPlainText() != state.config_text:
                self.config_preview.setPlainText(state.config_text)

        self.selected_experiment_title.setText(state.experiment_title)
        self.selected_experiment_description.setText(state.experiment_description)
        self.selected_badges_label.setText("  •  ".join(state.experiment_badges))
        self._update_status_chip(state)

        self.parameter_editor.set_fields(state.parameter_fields)
        self.preflight_widget.set_report(state.preflight_report)
        self.checklist_widget.set_pairs(state.run_checklist)
        self.result_details_widget.set_pairs(state.result_details)
        self.progress_bar.setMaximum(max(1, state.progress_total or 1))
        self.progress_bar.setValue(state.progress_current)
        self.run_button.setEnabled(not state.run_active and state.preflight_report.overall_status != "blocked")
        self.stop_button.setEnabled(state.run_active)
        self.open_output_button.setEnabled(bool(state.last_output_path or state.loaded_run_path))
        self.export_plot_button.setEnabled(bool(state.visualization_model.charts))
        self.destination_widget.set_pairs(
            [
                ("Planned Output", state.planned_output_dir or "n/a"),
                ("Current Run", state.last_output_path or state.loaded_run_path or "none"),
            ]
        )
        lines = [state.status_message]
        if state.config_error:
            lines.append(f"Config: {state.config_error}")
        if state.last_error:
            lines.append(f"Error: {state.last_error}")
        self.status_text.setPlainText("\n".join(lines))

        self.viewer_3d.set_view_options(show_axes=state.show_axes, show_labels=state.show_labels)
        self.viewer_3d.set_series(state.visualization_model.series_3d)
        self.results_widget.set_model(state.visualization_model)

        self._update_experiment_selector(state)
        self._update_history_list(state)
        self._update_color_mode(state)

    def _run(self) -> None:
        try:
            report = self.controller.refresh().preflight_report
            if report.requires_confirmation:
                details = "\n".join(report.overwrite_targets)
                response = QMessageBox.question(
                    self,
                    "Confirm Overwrite",
                    f"The following output(s) already exist and will be overwritten:\n\n{details}\n\nContinue?",
                )
                if response != QMessageBox.Yes:
                    return
                self.controller.run(confirm_overwrite=True)
            else:
                self.controller.run()
        except Exception:
            return

    def _browse_run(self) -> None:
        path = QFileDialog.getExistingDirectory(self, "Open Experiment Run", "")
        if path:
            try:
                self.controller.load_run(path)
            except Exception:
                return

    def _load_selected_history_item(self, item: QListWidgetItem) -> None:
        raw_path = item.data(Qt.UserRole)
        if raw_path:
            try:
                self.controller.load_run(raw_path)
            except Exception:
                return

    def _on_experiment_selected(self, row: int) -> None:
        if row < 0:
            return
        raw_name = self.experiment_combo.itemData(row)
        if raw_name and raw_name != self.controller.state.selected_experiment:
            self.controller.select_experiment(str(raw_name))

    def _on_output_root_changed(self) -> None:
        self.controller.set_output_root(self.output_root_edit.text())

    def _on_color_mode_changed(self, row: int) -> None:
        if row < 0:
            return
        value = self.color_mode_combo.itemData(row)
        if value:
            self.controller.set_color_mode(str(value))

    def _open_current_run_folder(self) -> None:
        path = self.controller.state.last_output_path or self.controller.state.loaded_run_path
        if path:
            QDesktopServices.openUrl(QUrl.fromLocalFile(str(path)))

    def _save_plot_image(self) -> None:
        path, _ = QFileDialog.getSaveFileName(self, "Save Plot Image", "experiment_plot.png", "PNG Images (*.png)")
        if path:
            self.results_widget.save_current_view(path)

    def _save_view_image(self) -> None:
        path, _ = QFileDialog.getSaveFileName(
            self, "Save Visualization Image", "experiment_view.png", "PNG Images (*.png)"
        )
        if path:
            self.viewer_3d.save_screenshot(path)

    def _update_experiment_selector(self, state: ExperimentViewState) -> None:
        current_keys = [self.experiment_combo.itemData(index) for index in range(self.experiment_combo.count())]
        target_keys = [option.name for option in state.experiment_options]
        if current_keys != target_keys:
            with QSignalBlocker(self.experiment_combo):
                self.experiment_combo.clear()
                for option in state.experiment_options:
                    self.experiment_combo.addItem(option.title, option.name)
        target_index = 0
        for index in range(self.experiment_combo.count()):
            if self.experiment_combo.itemData(index) == state.selected_experiment:
                target_index = index
                break
        with QSignalBlocker(self.experiment_combo):
            self.experiment_combo.setCurrentIndex(target_index)

    def _update_history_list(self, state: ExperimentViewState) -> None:
        self.history_list.clear()
        for entry in state.history:
            item = QListWidgetItem()
            item.setData(Qt.UserRole, entry.path)
            item.setToolTip(entry.path)
            self.history_list.addItem(item)
            widget = _HistoryItemWidget(
                experiment_name=entry.experiment_name,
                timestamp_utc=entry.timestamp_utc,
                status=entry.status,
                metric_summary=entry.metric_summary,
                run_path=entry.path,
            )
            item.setSizeHint(widget.sizeHint())
            self.history_list.setItemWidget(item, widget)

    def _update_color_mode(self, state: ExperimentViewState) -> None:
        target_row = 0
        for index in range(self.color_mode_combo.count()):
            if self.color_mode_combo.itemData(index) == state.color_mode:
                target_row = index
                break
        with QSignalBlocker(self.color_mode_combo):
            self.color_mode_combo.setCurrentIndex(target_row)

    def _update_status_chip(self, state: ExperimentViewState) -> None:
        status = state.preflight_report.overall_status
        if status == "blocked":
            bg, fg, text = "#fee2e2", "#991b1b", "Blocked"
        elif status == "ok_with_warning":
            bg, fg, text = "#fef3c7", "#92400e", "Ready With Warning"
        else:
            bg, fg, text = "#dcfce7", "#166534", "Ready"
        self.selected_status_chip.setText(text)
        self.selected_status_chip.setStyleSheet(
            f"padding: 5px 12px; border-radius: 999px; background: {bg}; color: {fg}; font-weight: 700;"
        )


class _SectionCard(QFrame):
    def __init__(self, title: str | None = None, subtitle: str | None = None, parent=None) -> None:
        super().__init__(parent)
        self.setProperty("role", "card")
        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(12)
        if title:
            title_label = QLabel(title)
            title_label.setProperty("role", "section-title")
            layout.addWidget(title_label)
        if subtitle:
            subtitle_label = QLabel(subtitle)
            subtitle_label.setProperty("role", "body")
            subtitle_label.setWordWrap(True)
            layout.addWidget(subtitle_label)
        self.body_layout = layout


class _KeyValueSummaryWidget(QWidget):
    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.layout_ = QVBoxLayout(self)
        self.layout_.setContentsMargins(0, 0, 0, 0)
        self.layout_.setSpacing(8)

    def set_pairs(self, pairs: list[tuple[str, str]]) -> None:
        while self.layout_.count():
            item = self.layout_.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()
        for label, value in pairs:
            row = QFrame()
            row.setStyleSheet("background: #f8fafc; border: 1px solid #e2e8f0; border-radius: 10px;")
            row_layout = QHBoxLayout(row)
            row_layout.setContentsMargins(10, 8, 10, 8)
            row_layout.setSpacing(12)
            key = QLabel(label)
            key.setProperty("role", "muted")
            val = QLabel(value)
            val.setWordWrap(True)
            val.setStyleSheet("color: #0f172a; font-weight: 600;")
            row_layout.addWidget(key, 1)
            row_layout.addWidget(val, 2)
            self.layout_.addWidget(row)
        self.layout_.addStretch(1)


class _HistoryItemWidget(QWidget):
    def __init__(self, *, experiment_name: str, timestamp_utc: str, status: str, metric_summary: str, run_path: str, parent=None) -> None:
        super().__init__(parent)
        title = QLabel(experiment_name.replace("_", " ").title())
        title.setStyleSheet("font-weight: 700; color: #0f172a;")
        chip = QLabel(_status_label(status))
        chip.setStyleSheet(
            f"padding: 4px 8px; border-radius: 999px; background: {_status_bg(status)}; color: {_status_fg(status)}; font-weight: 700;"
        )
        stamp = QLabel(timestamp_utc.replace("T", " ").replace("+00:00", "Z"))
        stamp.setStyleSheet("color: #64748b;")
        metric = QLabel(metric_summary or "No metric summary")
        metric.setWordWrap(True)
        metric.setStyleSheet("color: #334155;")
        run_name = QLabel(run_path.split("/")[-1])
        run_name.setStyleSheet("color: #94a3b8;")

        title_row = QHBoxLayout()
        title_row.setContentsMargins(0, 0, 0, 0)
        title_row.addWidget(title, 1)
        title_row.addWidget(chip)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 10, 12, 10)
        layout.setSpacing(4)
        layout.addLayout(title_row)
        layout.addWidget(stamp)
        layout.addWidget(metric)
        layout.addWidget(run_name)
        self.setStyleSheet("background: #f8fafc; border: 1px solid #e2e8f0; border-radius: 12px;")


def _status_label(status: str) -> str:
    mapping = {
        "success": "Pass",
        "partial_success": "Partial",
        "invalid_due_to_missing_registration": "Needs Registration",
        "invalid_due_to_missing_tip_cal": "Needs Tip Cal",
        "invalid_due_to_insufficient_samples": "Too Few Samples",
        "invalid_due_to_invalid_transforms": "Invalid",
    }
    return mapping.get(status, status.replace("_", " ").title())


def _status_bg(status: str) -> str:
    if status == "success":
        return "#dcfce7"
    if status == "partial_success":
        return "#fef3c7"
    return "#fee2e2"


def _status_fg(status: str) -> str:
    if status == "success":
        return "#166534"
    if status == "partial_success":
        return "#92400e"
    return "#991b1b"

"""Canonical experiment workspace tab."""

from __future__ import annotations

from PySide6.QtCore import QSignalBlocker, Qt, QUrl
from PySide6.QtGui import QDesktopServices
from PySide6.QtWidgets import (
    QFileDialog,
    QFormLayout,
    QFrame,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QPlainTextEdit,
    QProgressBar,
    QPushButton,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from continuum_robot.gui.controllers.experiment_controller import ExperimentViewState
from continuum_robot.gui.widgets.experiment_3d_widget import Experiment3DWidget
from continuum_robot.gui.widgets.experiment_preflight_widget import ExperimentPreflightWidget
from continuum_robot.gui.widgets.experiment_results_widget import ExperimentResultsWidget


class ExperimentTab(QWidget):
    """Workflow-focused operator workspace for the three canonical experiments."""

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
                background: #f4f7fb;
            }
            QWidget#experimentWorkspace QGroupBox {
                border: 1px solid #dbe4ee;
                border-radius: 12px;
                margin-top: 16px;
                padding-top: 10px;
                background: #ffffff;
            }
            QWidget#experimentWorkspace QGroupBox::title {
                subcontrol-origin: margin;
                left: 12px;
                padding: 0 6px;
                color: #0f172a;
                font-weight: 700;
            }
            QWidget#experimentWorkspace QLabel[role="hint"] {
                color: #475569;
            }
            QWidget#experimentWorkspace QLabel[role="chip"] {
                padding: 5px 10px;
                border-radius: 999px;
                background: #e2e8f0;
                color: #0f172a;
                font-weight: 600;
            }
            QWidget#experimentWorkspace QLabel[role="headline"] {
                color: #0f172a;
                font-size: 18px;
                font-weight: 700;
            }
            QWidget#experimentWorkspace QPushButton {
                min-height: 34px;
                padding: 0 12px;
            }
            QWidget#experimentWorkspace QPushButton[variant="primary"] {
                background: #0f172a;
                color: #ffffff;
                border-radius: 8px;
            }
            QWidget#experimentWorkspace QPushButton[variant="danger"] {
                background: #b91c1c;
                color: #ffffff;
                border-radius: 8px;
            }
            QWidget#experimentWorkspace QPlainTextEdit,
            QWidget#experimentWorkspace QTextEdit,
            QWidget#experimentWorkspace QLineEdit,
            QWidget#experimentWorkspace QListWidget,
            QWidget#experimentWorkspace QTableWidget {
                border: 1px solid #dbe4ee;
                border-radius: 10px;
                background: #fbfdff;
            }
            QWidget#experimentWorkspace QProgressBar {
                min-height: 22px;
                border: 1px solid #dbe4ee;
                border-radius: 8px;
                background: #eef2f7;
                text-align: center;
            }
            QWidget#experimentWorkspace QProgressBar::chunk {
                background: #2563eb;
                border-radius: 7px;
            }
            """
        )

        self.workspace_title = QLabel("Experiment Workspace")
        self.workspace_title.setProperty("role", "headline")
        self.workspace_hint = QLabel(
            "Run pivot calibration, Aurora grid accuracy, and repeatability from one guarded workspace. "
            "Preflight must pass before a run can start."
        )
        self.workspace_hint.setWordWrap(True)
        self.workspace_hint.setProperty("role", "hint")

        self.experiment_combo = QLineEdit()
        self.experiment_combo.setReadOnly(True)
        self.experiment_list = QListWidget()
        self.experiment_list.currentTextChanged.connect(self._on_experiment_selected)
        self.badges_label = QLabel()
        self.badges_label.setWordWrap(True)
        self.badges_label.setProperty("role", "chip")
        self.description_label = QLabel()
        self.description_label.setWordWrap(True)
        self.output_root_edit = QLineEdit()
        self.output_root_edit.editingFinished.connect(self._on_output_root_changed)
        self.notes_edit = QPlainTextEdit()
        self.notes_edit.setPlaceholderText("Optional operator notes stored in metadata.json")
        self.notes_edit.textChanged.connect(lambda: self.controller.set_operator_notes(self.notes_edit.toPlainText()))

        header_box = QGroupBox("Experiment Selection")
        header_layout = QFormLayout(header_box)
        header_layout.addRow("Selected", self.experiment_combo)
        header_layout.addRow("Available", self.experiment_list)
        header_layout.addRow("Badges", self.badges_label)
        header_layout.addRow("Description", self.description_label)
        header_layout.addRow("Output Root", self.output_root_edit)
        header_layout.addRow("Operator Notes", self.notes_edit)

        self.config_hint = QLabel(
            "Edit the run configuration here. The exact effective config will be written into each run folder as "
            "`config_snapshot.yaml`."
        )
        self.config_hint.setWordWrap(True)
        self.config_hint.setProperty("role", "hint")
        self.config_edit = QPlainTextEdit()
        self.config_edit.setPlaceholderText("YAML config")
        self.config_edit.textChanged.connect(lambda: self.controller.set_config_text(self.config_edit.toPlainText()))
        self.config_edit.setTabStopDistance(28)
        reset_button = QPushButton("Reset Example")
        reset_button.clicked.connect(lambda: self.controller.select_experiment(self.controller.state.selected_experiment))
        load_run_button = QPushButton("Load Run Folder")
        load_run_button.clicked.connect(self._browse_run)

        config_box = QGroupBox("Config")
        config_layout = QVBoxLayout(config_box)
        config_layout.addWidget(self.config_hint)
        config_layout.addWidget(self.config_edit)
        config_buttons = QHBoxLayout()
        config_buttons.addWidget(reset_button)
        config_buttons.addWidget(load_run_button)
        config_layout.addLayout(config_buttons)

        self.preflight_widget = ExperimentPreflightWidget()
        preflight_box = QGroupBox("Preflight")
        preflight_layout = QVBoxLayout(preflight_box)
        preflight_layout.addWidget(self.preflight_widget)

        self.checklist_table = QTableWidget(0, 2)
        self.checklist_table.setHorizontalHeaderLabels(["Item", "Value"])
        self.checklist_table.horizontalHeader().setStretchLastSection(True)
        self.checklist_table.verticalHeader().setVisible(False)
        checklist_hint = QLabel(
            "This card is the final run summary. Confirm the backend, mode, tool IDs, paths, and key config values before starting."
        )
        checklist_hint.setWordWrap(True)
        checklist_hint.setProperty("role", "hint")
        checklist_box = QGroupBox("Ready-To-Run Checklist")
        checklist_layout = QVBoxLayout(checklist_box)
        checklist_layout.addWidget(checklist_hint)
        checklist_layout.addWidget(self.checklist_table)

        self.viewer_3d = Experiment3DWidget(
            requested_mode=self.controller.settings.runtime.visualization_mode,
            safe_effects=self.controller.settings.runtime.visualization_safe_effects,
        )
        self.results_widget = ExperimentResultsWidget()
        self.color_mode_list = QListWidget()
        for label, _value in self._COLOR_MODES:
            self.color_mode_list.addItem(label)
        self.color_mode_list.currentRowChanged.connect(self._on_color_mode_changed)
        self.color_mode_list.setMaximumHeight(120)

        self.show_axes_button = QPushButton("Axes")
        self.show_axes_button.clicked.connect(lambda: self.controller.set_show_axes(not self.controller.state.show_axes))
        self.show_labels_button = QPushButton("Labels")
        self.show_labels_button.clicked.connect(lambda: self.controller.set_show_labels(not self.controller.state.show_labels))
        self.show_centroids_button = QPushButton("Centroids")
        self.show_centroids_button.clicked.connect(
            lambda: self.controller.set_show_centroids(not self.controller.state.show_centroids)
        )
        self.show_truth_button = QPushButton("Truth")
        self.show_truth_button.clicked.connect(lambda: self.controller.set_show_truth(not self.controller.state.show_truth))
        self.save_view_button = QPushButton("Save View Image")
        self.save_view_button.clicked.connect(self._save_view_image)

        view_controls = QHBoxLayout()
        view_controls.addWidget(self.show_axes_button)
        view_controls.addWidget(self.show_labels_button)
        view_controls.addWidget(self.show_centroids_button)
        view_controls.addWidget(self.show_truth_button)
        view_controls.addStretch(1)
        view_controls.addWidget(self.save_view_button)

        visualization_hint = QLabel(
            "Rotate, pan, and zoom when native 3D is available. On unstable platforms the workspace falls back to a safe projection view."
        )
        visualization_hint.setWordWrap(True)
        visualization_hint.setProperty("role", "hint")

        view_box = QGroupBox("Visualization")
        view_layout = QVBoxLayout(view_box)
        view_layout.addWidget(visualization_hint)
        view_layout.addLayout(view_controls)
        view_layout.addWidget(self.viewer_3d)
        view_layout.addWidget(QLabel("Coloring"))
        view_layout.addWidget(self.color_mode_list)

        results_box = QGroupBox("Results")
        results_layout = QVBoxLayout(results_box)
        results_layout.addWidget(self.results_widget)

        center_splitter = QSplitter(Qt.Vertical)
        center_splitter.addWidget(view_box)
        center_splitter.addWidget(results_box)
        center_splitter.setStretchFactor(0, 3)
        center_splitter.setStretchFactor(1, 2)

        self.run_button = QPushButton("Run Experiment")
        self.run_button.setProperty("variant", "primary")
        self.stop_button = QPushButton("Stop Run")
        self.stop_button.setProperty("variant", "danger")
        self.refresh_button = QPushButton("Refresh Preflight")
        self.refresh_button.clicked.connect(self.controller.refresh)
        self.run_button.clicked.connect(self._run)
        self.stop_button.clicked.connect(self.controller.stop)
        self.open_output_button = QPushButton("Open Current Run Folder")
        self.open_output_button.clicked.connect(self._open_current_run_folder)
        self.export_plot_button = QPushButton("Save Current Plot")
        self.export_plot_button.clicked.connect(self._save_plot_image)
        self.progress_bar = QProgressBar()
        self.planned_output_label = QLabel()
        self.planned_output_label.setWordWrap(True)
        self.last_output_label = QLabel()
        self.last_output_label.setWordWrap(True)
        self.status_text = QTextEdit()
        self.status_text.setReadOnly(True)

        controls_box = QGroupBox("Run Controls")
        controls_layout = QVBoxLayout(controls_box)
        controls_layout.addWidget(self.run_button)
        controls_layout.addWidget(self.stop_button)
        controls_layout.addWidget(self.refresh_button)
        controls_layout.addWidget(self.open_output_button)
        controls_layout.addWidget(self.export_plot_button)
        controls_layout.addWidget(self.progress_bar)
        controls_layout.addWidget(QLabel("Planned Output Folder"))
        controls_layout.addWidget(self.planned_output_label)
        controls_layout.addWidget(QLabel("Last Loaded / Saved Run"))
        controls_layout.addWidget(self.last_output_label)
        controls_layout.addWidget(self.status_text)

        self.history_list = QListWidget()
        self.history_list.itemDoubleClicked.connect(self._load_selected_history_item)
        self.history_list.setAlternatingRowColors(True)
        history_hint = QLabel("Double-click a prior run to reload its summary, plots, and visualization without hardware attached.")
        history_hint.setWordWrap(True)
        history_hint.setProperty("role", "hint")
        history_refresh_button = QPushButton("Refresh History")
        history_refresh_button.clicked.connect(self.controller.refresh)

        history_box = QGroupBox("Run History")
        history_layout = QVBoxLayout(history_box)
        history_layout.addWidget(history_hint)
        history_layout.addWidget(history_refresh_button)
        history_layout.addWidget(self.history_list)

        left_column = QWidget()
        left_layout = QVBoxLayout(left_column)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.setSpacing(12)
        left_layout.addWidget(config_box)
        left_layout.addWidget(preflight_box)
        left_layout.addWidget(checklist_box)

        right_column = QWidget()
        right_layout = QVBoxLayout(right_column)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.setSpacing(12)
        right_layout.addWidget(controls_box)
        right_layout.addWidget(history_box)

        main_splitter = QSplitter(Qt.Horizontal)
        main_splitter.addWidget(left_column)
        main_splitter.addWidget(center_splitter)
        main_splitter.addWidget(right_column)
        main_splitter.setStretchFactor(0, 3)
        main_splitter.setStretchFactor(1, 5)
        main_splitter.setStretchFactor(2, 3)

        header_card = QFrame()
        header_layout = QVBoxLayout(header_card)
        header_layout.setContentsMargins(0, 0, 0, 0)
        header_layout.addWidget(self.workspace_title)
        header_layout.addWidget(self.workspace_hint)
        header_layout.addWidget(header_box)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(14)
        layout.addWidget(header_card)
        layout.addWidget(main_splitter)

    def update(self, state: ExperimentViewState) -> None:
        with QSignalBlocker(self.notes_edit):
            if self.notes_edit.toPlainText() != state.operator_notes:
                self.notes_edit.setPlainText(state.operator_notes)
        with QSignalBlocker(self.config_edit):
            if self.config_edit.toPlainText() != state.config_text:
                self.config_edit.setPlainText(state.config_text)
        with QSignalBlocker(self.output_root_edit):
            if self.output_root_edit.text() != state.output_root:
                self.output_root_edit.setText(state.output_root)

        self.experiment_combo.setText(state.experiment_title)
        self.badges_label.setText("  |  ".join(state.experiment_badges))
        self.description_label.setText(state.experiment_description)
        self.preflight_widget.set_report(state.preflight_report)
        self._update_checklist(state)
        self.progress_bar.setMaximum(max(1, state.progress_total or 1))
        self.progress_bar.setValue(state.progress_current)
        self.run_button.setEnabled(not state.run_active and state.preflight_report.overall_status != "blocked")
        self.stop_button.setEnabled(state.run_active)
        self.open_output_button.setEnabled(bool(state.last_output_path))
        self.export_plot_button.setEnabled(bool(state.visualization_model.charts))
        self.planned_output_label.setText(state.planned_output_dir or "n/a")
        self.last_output_label.setText(state.last_output_path or state.loaded_run_path or "none")
        lines = [state.status_message]
        if state.last_error:
            lines.append(f"Error: {state.last_error}")
        self.status_text.setPlainText("\n".join(lines))

        self.viewer_3d.set_view_options(show_axes=state.show_axes, show_labels=state.show_labels)
        self.viewer_3d.set_series(state.visualization_model.series_3d)
        self.results_widget.set_model(state.visualization_model)

        self._update_experiment_list(state)
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

    def _on_experiment_selected(self, title: str) -> None:
        title = str(title or "").strip()
        if not title:
            return
        for option in self.controller.state.experiment_options:
            if option.title == title:
                self.controller.select_experiment(option.name)
                return

    def _on_output_root_changed(self) -> None:
        self.controller.set_output_root(self.output_root_edit.text())

    def _on_color_mode_changed(self, row: int) -> None:
        if row < 0 or row >= len(self._COLOR_MODES):
            return
        self.controller.set_color_mode(self._COLOR_MODES[row][1])

    def _open_current_run_folder(self) -> None:
        path = self.controller.state.last_output_path or self.controller.state.loaded_run_path
        if path:
            QDesktopServices.openUrl(QUrl.fromLocalFile(str(path)))

    def _save_plot_image(self) -> None:
        path, _ = QFileDialog.getSaveFileName(self, "Save Plot Image", "experiment_plot.png", "PNG Images (*.png)")
        if path:
            self.results_widget.save_current_view(path)

    def _save_view_image(self) -> None:
        path, _ = QFileDialog.getSaveFileName(self, "Save Visualization Image", "experiment_view.png", "PNG Images (*.png)")
        if path:
            self.viewer_3d.save_screenshot(path)

    def _update_experiment_list(self, state: ExperimentViewState) -> None:
        labels = [option.title for option in state.experiment_options]
        current_labels = [self.experiment_list.item(index).text() for index in range(self.experiment_list.count())]
        if labels != current_labels:
            with QSignalBlocker(self.experiment_list):
                self.experiment_list.clear()
                for label in labels:
                    self.experiment_list.addItem(label)
        target_title = state.experiment_title
        for index in range(self.experiment_list.count()):
            if self.experiment_list.item(index).text() == target_title:
                with QSignalBlocker(self.experiment_list):
                    self.experiment_list.setCurrentRow(index)
                break

    def _update_history_list(self, state: ExperimentViewState) -> None:
        self.history_list.clear()
        for entry in state.history:
            item = QListWidgetItem(entry.label)
            item.setData(Qt.UserRole, entry.path)
            item.setToolTip(entry.path)
            self.history_list.addItem(item)

    def _update_color_mode(self, state: ExperimentViewState) -> None:
        target_row = 0
        for index, (_label, value) in enumerate(self._COLOR_MODES):
            if value == state.color_mode:
                target_row = index
                break
        with QSignalBlocker(self.color_mode_list):
            self.color_mode_list.setCurrentRow(target_row)

    def _update_checklist(self, state: ExperimentViewState) -> None:
        self.checklist_table.setRowCount(len(state.run_checklist))
        for row, (label, value) in enumerate(state.run_checklist):
            self.checklist_table.setItem(row, 0, QTableWidgetItem(label))
            self.checklist_table.setItem(row, 1, QTableWidgetItem(value))
        self.checklist_table.resizeColumnsToContents()

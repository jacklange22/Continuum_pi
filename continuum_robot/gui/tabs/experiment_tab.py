"""Canonical experiment workspace tab."""

from __future__ import annotations

from PySide6.QtCore import QSignalBlocker, Qt
from PySide6.QtWidgets import (
    QFileDialog,
    QFormLayout,
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

        self.experiment_combo = QLineEdit()
        self.experiment_combo.setReadOnly(True)
        self.experiment_list = QListWidget()
        self.experiment_list.currentTextChanged.connect(self._on_experiment_selected)
        self.badges_label = QLabel()
        self.badges_label.setWordWrap(True)
        self.description_label = QLabel()
        self.description_label.setWordWrap(True)
        self.output_root_edit = QLineEdit()
        self.output_root_edit.editingFinished.connect(self._on_output_root_changed)
        self.notes_edit = QPlainTextEdit()
        self.notes_edit.setPlaceholderText("Operator notes stored in metadata.json")
        self.notes_edit.textChanged.connect(lambda: self.controller.set_operator_notes(self.notes_edit.toPlainText()))

        header_box = QGroupBox("Experiment Workspace")
        header_layout = QFormLayout(header_box)
        header_layout.addRow("Selected", self.experiment_combo)
        header_layout.addRow("Available", self.experiment_list)
        header_layout.addRow("Badges", self.badges_label)
        header_layout.addRow("Description", self.description_label)
        header_layout.addRow("Output Root", self.output_root_edit)
        header_layout.addRow("Notes", self.notes_edit)

        self.config_edit = QPlainTextEdit()
        self.config_edit.setPlaceholderText("YAML config")
        self.config_edit.textChanged.connect(lambda: self.controller.set_config_text(self.config_edit.toPlainText()))
        reset_button = QPushButton("Reset Example")
        reset_button.clicked.connect(lambda: self.controller.select_experiment(self.controller.state.selected_experiment))
        load_run_button = QPushButton("Open Run Folder")
        load_run_button.clicked.connect(self._browse_run)

        config_box = QGroupBox("Config")
        config_layout = QVBoxLayout(config_box)
        config_layout.addWidget(self.config_edit)
        config_buttons = QHBoxLayout()
        config_buttons.addWidget(reset_button)
        config_buttons.addWidget(load_run_button)
        config_layout.addLayout(config_buttons)

        self.preflight_widget = ExperimentPreflightWidget()
        preflight_box = QGroupBox("Preflight")
        preflight_layout = QVBoxLayout(preflight_box)
        preflight_layout.addWidget(self.preflight_widget)

        self.checklist_text = QTextEdit()
        self.checklist_text.setReadOnly(True)
        checklist_box = QGroupBox("Run Checklist")
        checklist_layout = QVBoxLayout(checklist_box)
        checklist_layout.addWidget(self.checklist_text)

        self.viewer_3d = Experiment3DWidget()
        self.results_widget = ExperimentResultsWidget()
        self.color_mode_list = QListWidget()
        for label, _value in self._COLOR_MODES:
            self.color_mode_list.addItem(label)
        self.color_mode_list.currentRowChanged.connect(self._on_color_mode_changed)
        self.color_mode_list.setMaximumHeight(110)

        self.show_axes_button = QPushButton("Toggle Axes")
        self.show_axes_button.clicked.connect(lambda: self.controller.set_show_axes(not self.controller.state.show_axes))
        self.show_labels_button = QPushButton("Toggle Labels")
        self.show_labels_button.clicked.connect(lambda: self.controller.set_show_labels(not self.controller.state.show_labels))
        self.show_centroids_button = QPushButton("Toggle Centroids")
        self.show_centroids_button.clicked.connect(
            lambda: self.controller.set_show_centroids(not self.controller.state.show_centroids)
        )
        self.show_truth_button = QPushButton("Toggle Truth")
        self.show_truth_button.clicked.connect(lambda: self.controller.set_show_truth(not self.controller.state.show_truth))

        view_controls = QHBoxLayout()
        view_controls.addWidget(self.show_axes_button)
        view_controls.addWidget(self.show_labels_button)
        view_controls.addWidget(self.show_centroids_button)
        view_controls.addWidget(self.show_truth_button)

        view_box = QGroupBox("3D View")
        view_layout = QVBoxLayout(view_box)
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
        center_splitter.setStretchFactor(0, 2)
        center_splitter.setStretchFactor(1, 2)

        self.run_button = QPushButton("Run Experiment")
        self.stop_button = QPushButton("Stop")
        self.refresh_button = QPushButton("Refresh Preflight")
        self.refresh_button.clicked.connect(self.controller.refresh)
        self.run_button.clicked.connect(self._run)
        self.stop_button.clicked.connect(self.controller.stop)
        self.progress_bar = QProgressBar()
        self.planned_output_label = QLabel()
        self.last_output_label = QLabel()
        self.status_text = QTextEdit()
        self.status_text.setReadOnly(True)

        controls_box = QGroupBox("Run Controls")
        controls_layout = QVBoxLayout(controls_box)
        controls_layout.addWidget(self.run_button)
        controls_layout.addWidget(self.stop_button)
        controls_layout.addWidget(self.refresh_button)
        controls_layout.addWidget(self.progress_bar)
        controls_layout.addWidget(QLabel("Planned Output"))
        controls_layout.addWidget(self.planned_output_label)
        controls_layout.addWidget(QLabel("Last Output"))
        controls_layout.addWidget(self.last_output_label)
        controls_layout.addWidget(self.status_text)

        self.history_list = QListWidget()
        self.history_list.itemDoubleClicked.connect(self._load_selected_history_item)
        history_refresh_button = QPushButton("Refresh History")
        history_refresh_button.clicked.connect(self.controller.refresh)

        history_box = QGroupBox("Run History")
        history_layout = QVBoxLayout(history_box)
        history_layout.addWidget(history_refresh_button)
        history_layout.addWidget(self.history_list)

        left_column = QWidget()
        left_layout = QVBoxLayout(left_column)
        left_layout.addWidget(config_box)
        left_layout.addWidget(preflight_box)
        left_layout.addWidget(checklist_box)

        right_column = QWidget()
        right_layout = QVBoxLayout(right_column)
        right_layout.addWidget(controls_box)
        right_layout.addWidget(history_box)

        main_splitter = QSplitter(Qt.Horizontal)
        main_splitter.addWidget(left_column)
        main_splitter.addWidget(center_splitter)
        main_splitter.addWidget(right_column)
        main_splitter.setStretchFactor(0, 2)
        main_splitter.setStretchFactor(1, 4)
        main_splitter.setStretchFactor(2, 2)

        layout = QVBoxLayout(self)
        layout.addWidget(header_box)
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
        self.badges_label.setText(" | ".join(state.experiment_badges))
        self.description_label.setText(state.experiment_description)
        self.preflight_widget.set_report(state.preflight_report)
        self.checklist_text.setPlainText(
            "\n".join(f"{label}: {value}" for label, value in state.run_checklist)
        )
        self.progress_bar.setMaximum(max(1, state.progress_total or 1))
        self.progress_bar.setValue(state.progress_current)
        self.run_button.setEnabled(not state.run_active and state.preflight_report.overall_status != "blocked")
        self.stop_button.setEnabled(state.run_active)
        self.planned_output_label.setText(state.planned_output_dir or "n/a")
        self.last_output_label.setText(state.last_output_path or "none")
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
            self.history_list.addItem(item)

    def _update_color_mode(self, state: ExperimentViewState) -> None:
        target_row = 0
        for index, (_label, value) in enumerate(self._COLOR_MODES):
            if value == state.color_mode:
                target_row = index
                break
        with QSignalBlocker(self.color_mode_list):
            self.color_mode_list.setCurrentRow(target_row)

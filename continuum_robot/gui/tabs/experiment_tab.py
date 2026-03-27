"""Experiment tab widget."""

from __future__ import annotations

from pathlib import Path

from PySide6.QtWidgets import (
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QProgressBar,
    QPushButton,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from continuum_robot.gui.controllers.experiment_controller import ExperimentViewState


class ExperimentTab(QWidget):
    """Experiment file loading and execution UI."""

    def __init__(self, controller, parent=None) -> None:
        super().__init__(parent)
        self.controller = controller

        self.file_edit = QLineEdit()
        browse_button = QPushButton("Browse")
        load_button = QPushButton("Load")
        self.run_button = QPushButton("Run")
        self.stop_button = QPushButton("Stop")
        browse_button.clicked.connect(self._browse)
        load_button.clicked.connect(self._load)
        self.run_button.clicked.connect(lambda: self._safe_call(self.controller.run))
        self.stop_button.clicked.connect(lambda: self._safe_call(self.controller.stop))

        file_row = QHBoxLayout()
        file_row.addWidget(self.file_edit)
        file_row.addWidget(browse_button)
        file_row.addWidget(load_button)
        file_row.addWidget(self.run_button)
        file_row.addWidget(self.stop_button)

        self.point_count_label = QLabel()
        self.prereq_label = QLabel()
        self.output_label = QLabel()
        self.progress_bar = QProgressBar()

        summary_box = QGroupBox("Experiment Summary")
        summary_layout = QFormLayout(summary_box)
        summary_layout.addRow("Points", self.point_count_label)
        summary_layout.addRow("Prerequisites", self.prereq_label)
        summary_layout.addRow("Last output", self.output_label)
        summary_layout.addRow("Progress", self.progress_bar)

        self.status_text = QTextEdit()
        self.status_text.setReadOnly(True)

        layout = QVBoxLayout(self)
        layout.addLayout(file_row)
        layout.addWidget(summary_box)
        layout.addWidget(self.status_text)

    def update(self, state: ExperimentViewState) -> None:
        if state.loaded_file:
            self.file_edit.setText(state.loaded_file)
        self.point_count_label.setText(str(state.point_count))
        self.prereq_label.setText(state.prerequisite_message)
        self.output_label.setText(state.last_output_path or "none")
        self.progress_bar.setMaximum(max(1, state.progress_total))
        self.progress_bar.setValue(state.progress_current)
        self.run_button.setEnabled(state.prerequisites_ok and not state.run_active)
        self.stop_button.setEnabled(state.run_active)
        lines = [state.status_message]
        if state.last_error:
            lines.append(f"Error: {state.last_error}")
        self.status_text.setPlainText("\n".join(lines))

    def _browse(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Select Experiment CSV", "", "CSV Files (*.csv)")
        if path:
            self.file_edit.setText(path)

    def _load(self) -> None:
        path = self.file_edit.text().strip()
        if path:
            self._safe_call(lambda: self.controller.load_file(Path(path)))

    def _safe_call(self, fn) -> None:
        try:
            fn()
        except Exception:
            return

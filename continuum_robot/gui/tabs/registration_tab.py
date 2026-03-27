"""Registration tab widget."""

from __future__ import annotations

from PySide6.QtWidgets import (
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from continuum_robot.gui.controllers.registration_controller import RegistrationViewState
from continuum_robot.gui.widgets.registration_plot_widget import RegistrationPlotWidget


class RegistrationTab(QWidget):
    """Guided landmark capture and registration UI."""

    def __init__(self, controller, parent=None) -> None:
        super().__init__(parent)
        self.controller = controller

        self.begin_button = QPushButton("Begin Session")
        self.capture_button = QPushButton("Capture Sample")
        self.finish_button = QPushButton("Solve + Save")
        self.retry_button = QPushButton("Retry")
        self.begin_button.clicked.connect(lambda: self._safe_call(self.controller.begin_session))
        self.capture_button.clicked.connect(lambda: self._safe_call(self.controller.capture_current_label_sample))
        self.finish_button.clicked.connect(lambda: self._safe_call(self.controller.finish_session))
        self.retry_button.clicked.connect(lambda: self._safe_call(self.controller.retry_session))

        self.tool_label = QLabel()
        self.current_label = QLabel()
        self.geometry_label = QLabel()
        self.fre_label = QLabel()
        self.result_path_label = QLabel()

        summary_box = QGroupBox("Registration Summary")
        summary_layout = QFormLayout(summary_box)
        summary_layout.addRow("Capture tool", self.tool_label)
        summary_layout.addRow("Capture geometry", self.geometry_label)
        summary_layout.addRow("Current landmark", self.current_label)
        summary_layout.addRow("FRE (mm)", self.fre_label)
        summary_layout.addRow("Latest result", self.result_path_label)

        buttons = QHBoxLayout()
        buttons.addWidget(self.begin_button)
        buttons.addWidget(self.capture_button)
        buttons.addWidget(self.finish_button)
        buttons.addWidget(self.retry_button)

        self.progress_table = QTableWidget(0, 3)
        self.progress_table.setHorizontalHeaderLabels(["Landmark", "Captures", "Latest sample"])
        self.plot_widget = RegistrationPlotWidget()
        self.status_text = QTextEdit()
        self.status_text.setReadOnly(True)

        layout = QVBoxLayout(self)
        layout.addWidget(summary_box)
        layout.addLayout(buttons)
        layout.addWidget(self.plot_widget)
        layout.addWidget(self.progress_table)
        layout.addWidget(self.status_text)

    def update(self, state: RegistrationViewState) -> None:
        self.tool_label.setText(state.capture_tool_id)
        self.geometry_label.setText(state.capture_geometry_status)
        self.current_label.setText(state.current_label or "complete")
        self.fre_label.setText(f"{state.fre_mm:.3f}" if state.fre_mm is not None else "not computed")
        self.result_path_label.setText(state.last_result_path or "none")
        self.capture_button.setEnabled(state.active and state.current_label is not None)
        self.finish_button.setEnabled(state.active and self.controller.is_ready_to_finish())
        self.retry_button.setEnabled(state.active or state.last_result_path is not None)

        self.progress_table.setRowCount(len(state.landmark_labels))
        captured_plot: dict[str, list[tuple[float, float]]] = {}
        for row, label in enumerate(state.landmark_labels):
            count = state.captured_counts.get(label, 0)
            latest = state.latest_sample_by_label.get(label)
            self.progress_table.setItem(row, 0, QTableWidgetItem(label))
            self.progress_table.setItem(row, 1, QTableWidgetItem(f"{count}/{state.captures_per_landmark}"))
            self.progress_table.setItem(row, 2, QTableWidgetItem(str(latest) if latest else ""))
            if latest:
                captured_plot[label] = [(latest[0], latest[1])]
            else:
                captured_plot[label] = []

        nominal = {
            label: (
                float(self.controller.config.nominal_landmarks_robot_xyz_mm[label][0]),
                float(self.controller.config.nominal_landmarks_robot_xyz_mm[label][1]),
            )
            for label in self.controller.config.landmark_labels
            if label in self.controller.config.nominal_landmarks_robot_xyz_mm
        }
        self.plot_widget.set_data(nominal=nominal, captured=captured_plot)

        lines = [state.status_message]
        if state.last_error:
            lines.append(f"Error: {state.last_error}")
        if state.residuals_by_label:
            lines.append(f"Residuals: {state.residuals_by_label}")
        self.status_text.setPlainText("\n".join(lines))

    def _safe_call(self, fn) -> None:
        try:
            fn()
        except Exception:
            return

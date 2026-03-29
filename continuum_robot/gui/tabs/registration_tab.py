"""Registration tab widget."""

from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMessageBox,
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
    """Guided 4-point robot-body alignment workflow."""

    def __init__(self, controller, parent=None) -> None:
        super().__init__(parent)
        self.controller = controller
        self.setObjectName("registrationWorkspace")
        self.setStyleSheet(
            """
            QWidget#registrationWorkspace QGroupBox {
                border: 1px solid #d0d7de;
                border-radius: 10px;
                margin-top: 14px;
                padding-top: 10px;
                background: #ffffff;
            }
            QWidget#registrationWorkspace QGroupBox::title {
                subcontrol-origin: margin;
                left: 12px;
                padding: 0 6px;
                color: #0f172a;
                font-weight: 600;
            }
            QWidget#registrationWorkspace QPushButton {
                min-height: 34px;
                padding: 0 12px;
            }
            QWidget#registrationWorkspace QLabel[role="hint"] {
                color: #475569;
            }
            QWidget#registrationWorkspace QLabel[role="status"] {
                padding: 8px 10px;
                border-radius: 8px;
                background: #e2e8f0;
                color: #0f172a;
                font-weight: 600;
            }
            """
        )

        self.workflow_hint = QLabel(
            "Use the pen probe on four robot-body landmarks. Capture one or more samples for each point, "
            "mark the point complete, solve, review RMSE/FRE, then save the accepted registration."
        )
        self.workflow_hint.setWordWrap(True)
        self.workflow_hint.setProperty("role", "hint")

        self.begin_button = QPushButton("Begin 4-Point Session")
        self.capture_button = QPushButton("Capture Sample")
        self.complete_button = QPushButton("Mark Point Complete")
        self.solve_button = QPushButton("Solve Registration")
        self.save_button = QPushButton("Save Registration")
        self.retry_button = QPushButton("Restart")
        self.load_button = QPushButton("Load Latest")

        self.begin_button.clicked.connect(lambda: self._safe_call(self.controller.begin_session))
        self.capture_button.clicked.connect(lambda: self._safe_call(self.controller.capture_current_label_sample))
        self.complete_button.clicked.connect(lambda: self._safe_call(self.controller.complete_current_label))
        self.solve_button.clicked.connect(lambda: self._safe_call(self.controller.solve_session))
        self.save_button.clicked.connect(self._save_registration)
        self.retry_button.clicked.connect(lambda: self._safe_call(self.controller.retry_session))
        self.load_button.clicked.connect(lambda: self._safe_call(self.controller.load_latest_result))

        self.session_status_label = QLabel("Idle")
        self.session_status_label.setProperty("role", "status")
        self.tool_label = QLabel()
        self.coil_tool_label = QLabel()
        self.geometry_label = QLabel()
        self.current_label = QLabel()
        self.live_point_label = QLabel()
        self.samples_used_label = QLabel()
        self.fre_label = QLabel()
        self.accepted_label = QLabel()
        self.result_path_label = QLabel()

        summary_box = QGroupBox("Registration Summary")
        summary_layout = QFormLayout(summary_box)
        summary_layout.addRow("Session", self.session_status_label)
        summary_layout.addRow("Capture tool", self.tool_label)
        summary_layout.addRow("Runtime coil", self.coil_tool_label)
        summary_layout.addRow("Capture geometry", self.geometry_label)
        summary_layout.addRow("Current point", self.current_label)
        summary_layout.addRow("Live tracked point", self.live_point_label)
        summary_layout.addRow("Samples captured", self.samples_used_label)
        summary_layout.addRow("RMSE / FRE", self.fre_label)
        summary_layout.addRow("Accepted result", self.accepted_label)
        summary_layout.addRow("Latest file", self.result_path_label)

        buttons = QHBoxLayout()
        buttons.addWidget(self.begin_button)
        buttons.addWidget(self.capture_button)
        buttons.addWidget(self.complete_button)
        buttons.addWidget(self.solve_button)
        buttons.addWidget(self.save_button)
        buttons.addWidget(self.retry_button)
        buttons.addWidget(self.load_button)

        self.points_table = QTableWidget(0, 5)
        self.points_table.setHorizontalHeaderLabels(
            ["Point", "Model Point (mm)", "Samples", "Status", "Latest Measured Point (mm)"]
        )
        self.points_table.horizontalHeader().setStretchLastSection(True)
        self.points_table.verticalHeader().setVisible(False)

        points_box = QGroupBox("4 Required Registration Points")
        points_layout = QVBoxLayout(points_box)
        points_layout.addWidget(self.points_table)

        self.samples_table = QTableWidget(0, 5)
        self.samples_table.setHorizontalHeaderLabels(["Point", "Sample", "X (mm)", "Y (mm)", "Z (mm)"])
        self.samples_table.horizontalHeader().setStretchLastSection(True)
        self.samples_table.verticalHeader().setVisible(False)

        samples_box = QGroupBox("Captured Samples")
        samples_layout = QVBoxLayout(samples_box)
        samples_layout.addWidget(self.samples_table)

        self.plot_widget = RegistrationPlotWidget()
        plot_box = QGroupBox("Registration Preview")
        plot_layout = QVBoxLayout(plot_box)
        plot_layout.addWidget(self.plot_widget)

        self.status_text = QTextEdit()
        self.status_text.setReadOnly(True)
        status_box = QGroupBox("Operator Notes")
        status_layout = QVBoxLayout(status_box)
        status_layout.addWidget(self.status_text)

        top_row = QHBoxLayout()
        top_row.addWidget(summary_box, 2)
        top_row.addWidget(plot_box, 3)

        lower_row = QHBoxLayout()
        lower_row.addWidget(points_box, 3)
        lower_row.addWidget(samples_box, 4)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(14, 14, 14, 14)
        layout.setSpacing(12)
        layout.addWidget(self.workflow_hint)
        layout.addLayout(buttons)
        layout.addLayout(top_row)
        layout.addLayout(lower_row)
        layout.addWidget(status_box)

    def update(self, state: RegistrationViewState) -> None:
        session_status = "Solved and waiting for save" if state.pending_accept else ("Capturing" if state.active else "Idle")
        self.session_status_label.setText(session_status)
        self.tool_label.setText(state.capture_tool_id)
        self.coil_tool_label.setText(state.coil_tool_id)
        self.geometry_label.setText(state.capture_geometry_status)
        self.current_label.setText(state.current_label or "All four points complete")
        self.live_point_label.setText(_format_xyz(state.current_tracked_xyz_mm, state.current_tracking_status, state.current_tracked_frame_id))
        self.samples_used_label.setText(str(self.controller.total_samples_captured()))
        self.fre_label.setText(f"{state.fre_mm:.3f} mm" if state.fre_mm is not None else "Not solved yet")
        self.accepted_label.setText("Yes" if state.accepted_registration_valid else "No")
        self.result_path_label.setText(state.last_result_path or "None")

        self.capture_button.setEnabled(state.active and state.current_label is not None)
        self.complete_button.setEnabled(state.active and self.controller.is_ready_to_complete_current())
        self.solve_button.setEnabled(state.active and self.controller.is_ready_to_solve())
        self.save_button.setEnabled(state.pending_accept)
        self.retry_button.setEnabled(state.active or state.pending_accept)

        self.points_table.setRowCount(len(state.landmark_labels))
        captured_plot: dict[str, list[tuple[float, float]]] = {}
        for row, label in enumerate(state.landmark_labels):
            count = state.captured_counts.get(label, 0)
            latest = state.latest_sample_by_label.get(label)
            truth = state.truth_points_in_sw_by_label.get(label)
            status = _point_status(label, state)
            self.points_table.setItem(row, 0, QTableWidgetItem(label))
            self.points_table.setItem(row, 1, QTableWidgetItem(_render_point(truth)))
            self.points_table.setItem(row, 2, QTableWidgetItem(f"{count} / {state.captures_per_landmark}+"))
            self.points_table.setItem(row, 3, QTableWidgetItem(status))
            self.points_table.setItem(row, 4, QTableWidgetItem(_render_point(latest)))
            captured_plot[label] = [
                (float(sample[0]), float(sample[1]))
                for sample in state.raw_samples_by_label.get(label, [])
                if len(sample) >= 2
            ]

        sample_rows = [
            (label, index + 1, sample)
            for label in state.landmark_labels
            for index, sample in enumerate(state.raw_samples_by_label.get(label, []))
        ]
        self.samples_table.setRowCount(len(sample_rows))
        for row, (label, sample_index, sample) in enumerate(sample_rows):
            self.samples_table.setItem(row, 0, QTableWidgetItem(label))
            self.samples_table.setItem(row, 1, QTableWidgetItem(str(sample_index)))
            self.samples_table.setItem(row, 2, QTableWidgetItem(_fmt_axis(sample, 0)))
            self.samples_table.setItem(row, 3, QTableWidgetItem(_fmt_axis(sample, 1)))
            self.samples_table.setItem(row, 4, QTableWidgetItem(_fmt_axis(sample, 2)))

        nominal = {
            label: (
                float(state.truth_points_in_sw_by_label[label][0]),
                float(state.truth_points_in_sw_by_label[label][1]),
            )
            for label in state.landmark_labels
            if label in state.truth_points_in_sw_by_label
        }
        centroid_map = {
            label: tuple(_mean_xy(samples))
            for label, samples in state.raw_samples_by_label.items()
            if samples
        }
        current_point = tuple(state.current_tracked_xyz_mm[:2]) if state.current_tracked_xyz_mm else None
        self.plot_widget.set_data(
            nominal=nominal,
            captured=captured_plot,
            centroids=centroid_map,
            current_point=current_point,
        )

        lines = [state.status_message]
        if state.overwrite_required and state.overwrite_target_path:
            lines.append(f"Save confirmation required because {state.overwrite_target_path} already exists.")
        if state.last_error:
            lines.append(f"Error: {state.last_error}")
        if state.residuals_by_label:
            lines.append(f"Residuals (mm): {state.residuals_by_label}")
        self.status_text.setPlainText("\n".join(lines))

    def _save_registration(self) -> None:
        try:
            self.controller.save_registration()
        except RuntimeError as exc:
            if "overwrite confirmation" not in str(exc):
                return
            target = self.controller.state.overwrite_target_path or "latest_registration.json"
            response = QMessageBox.question(
                self,
                "Overwrite Registration",
                f"The accepted registration will overwrite:\n\n{target}\n\nContinue?",
            )
            if response == QMessageBox.Yes:
                self._safe_call(lambda: self.controller.save_registration(confirm_overwrite=True))

    def _safe_call(self, fn) -> None:
        try:
            fn()
        except Exception:
            return


def _format_xyz(point: list[float] | None, status: str, frame_id: int | None) -> str:
    if point is None:
        return status
    suffix = f" | frame {frame_id}" if frame_id is not None else ""
    return f"[{point[0]:.2f}, {point[1]:.2f}, {point[2]:.2f}] mm{suffix}"


def _render_point(point: list[float] | None) -> str:
    if point is None:
        return "—"
    return f"[{point[0]:.2f}, {point[1]:.2f}, {point[2]:.2f}]"


def _fmt_axis(point: list[float], index: int) -> str:
    if index >= len(point):
        return "—"
    return f"{float(point[index]):.3f}"


def _mean_xy(samples: list[list[float]]) -> tuple[float, float]:
    xs = [float(sample[0]) for sample in samples if len(sample) >= 2]
    ys = [float(sample[1]) for sample in samples if len(sample) >= 2]
    return (sum(xs) / len(xs), sum(ys) / len(ys))


def _point_status(label: str, state: RegistrationViewState) -> str:
    if label in state.completed_labels:
        return "Complete"
    if label == state.current_label:
        return "Active"
    if state.pending_accept and state.current_label is None:
        return "Captured"
    return "Pending"

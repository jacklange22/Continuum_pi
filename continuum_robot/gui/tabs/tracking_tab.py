"""Tracking tab widget."""

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

from continuum_robot.gui.controllers.tracking_controller import TrackingViewState
from continuum_robot.gui.widgets.tool_plot_widget import ToolPlotWidget


class TrackingTab(QWidget):
    """Live Aurora tool and tip pose UI."""

    def __init__(self, controller, parent=None) -> None:
        super().__init__(parent)
        self.controller = controller

        connect_button = QPushButton("Connect Tracker")
        disconnect_button = QPushButton("Disconnect Tracker")
        connect_button.clicked.connect(self.controller.connect)
        disconnect_button.clicked.connect(self.controller.disconnect)

        self.connection_label = QLabel()
        self.frame_label = QLabel()
        self.tip_status_label = QLabel()
        self.tip_position_label = QLabel()
        self.registration_label = QLabel()

        summary_box = QGroupBox("Tracker Summary")
        summary_layout = QFormLayout(summary_box)
        summary_layout.addRow("Connection", self.connection_label)
        summary_layout.addRow("Latest frame", self.frame_label)
        summary_layout.addRow("Registration file", self.registration_label)
        summary_layout.addRow("Tip status", self.tip_status_label)
        summary_layout.addRow("Tip position (mm)", self.tip_position_label)

        button_row = QHBoxLayout()
        button_row.addWidget(connect_button)
        button_row.addWidget(disconnect_button)

        self.tools_table = QTableWidget(0, 5)
        self.tools_table.setHorizontalHeaderLabels(["Tool", "Valid", "Status", "Translation mm", "Quality"])
        self.plot_widget = ToolPlotWidget()
        self.status_text = QTextEdit()
        self.status_text.setReadOnly(True)

        layout = QVBoxLayout(self)
        layout.addWidget(summary_box)
        layout.addLayout(button_row)
        layout.addWidget(self.plot_widget)
        layout.addWidget(self.tools_table)
        layout.addWidget(self.status_text)

    def update(self, state: TrackingViewState) -> None:
        self.connection_label.setText(
            f"{state.connection_state} | bridge={state.bridge_running} | socket={state.socket_connected}"
        )
        self.frame_label.setText(str(state.latest_frame_number) if state.latest_frame_number is not None else "none")
        self.registration_label.setText(state.registration_path or "missing")
        self.tip_status_label.setText(state.tip_status)
        self.tip_position_label.setText(
            ", ".join(f"{value:.2f}" for value in state.tip_position_mm) if state.tip_position_mm else "unavailable"
        )

        self.tools_table.setRowCount(len(state.tools))
        plot_points: dict[str, tuple[float, float]] = {}
        for row, tool_id in enumerate(sorted(state.tools)):
            tool = state.tools[tool_id]
            self.tools_table.setItem(row, 0, QTableWidgetItem(tool_id))
            self.tools_table.setItem(row, 1, QTableWidgetItem(str(tool["valid"])))
            self.tools_table.setItem(row, 2, QTableWidgetItem(str(tool["status"])))
            self.tools_table.setItem(row, 3, QTableWidgetItem(str(tuple(round(v, 2) for v in tool["translation_mm"]))))
            self.tools_table.setItem(row, 4, QTableWidgetItem(str(tool["quality"])))
            plot_points[tool_id] = (tool["translation_mm"][0], tool["translation_mm"][1])
        if state.tip_position_mm:
            plot_points["tip"] = (state.tip_position_mm[0], state.tip_position_mm[1])
        self.plot_widget.set_points(plot_points)

        lines = [state.last_status_message or "No tracker messages yet."]
        if state.last_error:
            lines.append(f"Error: {state.last_error}")
        self.status_text.setPlainText("\n".join(lines))

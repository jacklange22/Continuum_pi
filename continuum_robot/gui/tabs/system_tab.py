"""System tab widget."""

from __future__ import annotations

from PySide6.QtWidgets import (
    QComboBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from continuum_robot.gui.controllers.system_controller import SystemViewState


class SystemTab(QWidget):
    """System connectivity and troubleshooting UI."""

    def __init__(self, controller, parent=None) -> None:
        super().__init__(parent)
        self.controller = controller

        self.mock_mode_label = QLabel()
        self.tracker_status_label = QLabel()
        self.openrb_status_label = QLabel()
        self.aurora_port_combo = QComboBox()
        self.aurora_port_combo.setEditable(True)
        self.openrb_port_combo = QComboBox()
        self.openrb_port_combo.setEditable(True)

        tracker_connect = QPushButton("Connect Tracker")
        tracker_disconnect = QPushButton("Disconnect Tracker")
        openrb_connect = QPushButton("Connect OpenRB")
        openrb_disconnect = QPushButton("Disconnect OpenRB")
        prepare_button = QPushButton("Prepare OpenRB")
        refresh_button = QPushButton("Rescan Ports")
        refresh_button.clicked.connect(self._rescan_ports)
        tracker_connect.clicked.connect(self._connect_tracker)
        tracker_disconnect.clicked.connect(self.controller.disconnect_tracker)
        openrb_connect.clicked.connect(self._connect_openrb)
        openrb_disconnect.clicked.connect(self.controller.disconnect_openrb)
        prepare_button.clicked.connect(self.controller.prepare_openrb)

        form = QFormLayout()
        form.addRow("Mock mode", self.mock_mode_label)
        form.addRow("Aurora port", self.aurora_port_combo)
        form.addRow("OpenRB port", self.openrb_port_combo)
        form.addRow("Tracker state", self.tracker_status_label)
        form.addRow("OpenRB state", self.openrb_status_label)

        ports_box = QGroupBox("Connections")
        ports_layout = QVBoxLayout(ports_box)
        ports_layout.addLayout(form)

        buttons = QHBoxLayout()
        buttons.addWidget(tracker_connect)
        buttons.addWidget(tracker_disconnect)
        buttons.addWidget(openrb_connect)
        buttons.addWidget(openrb_disconnect)
        buttons.addWidget(prepare_button)
        buttons.addWidget(refresh_button)
        ports_layout.addLayout(buttons)

        self.config_summary = QTextEdit()
        self.config_summary.setReadOnly(True)
        self.status_text = QTextEdit()
        self.status_text.setReadOnly(True)

        summary_box = QGroupBox("Config Summary")
        summary_layout = QVBoxLayout(summary_box)
        summary_layout.addWidget(self.config_summary)

        diagnostics_box = QGroupBox("Diagnostics")
        diagnostics_layout = QVBoxLayout(diagnostics_box)
        diagnostics_layout.addWidget(self.status_text)

        layout = QVBoxLayout(self)
        layout.addWidget(ports_box)
        layout.addWidget(summary_box)
        layout.addWidget(diagnostics_box)

    def update(self, state: SystemViewState) -> None:
        self.mock_mode_label.setText("enabled" if state.mock_mode else "disabled")
        self.tracker_status_label.setText(
            f"{state.tracker_connection_state} | backend={state.tracker_backend_identity} "
            f"| running={state.tracker_backend_running} | connected={state.tracker_backend_connected}"
        )
        self.openrb_status_label.setText(
            f"{state.openrb_status} | bus connected={state.dynamixel_connected}"
        )
        self.config_summary.setPlainText(state.config_summary)
        status_lines = [state.status_message]
        if state.last_error:
            status_lines.append(f"Error: {state.last_error}")
        self.status_text.setPlainText("\n".join(status_lines))
        self._set_combo_items(self.aurora_port_combo, state.available_ports, state.aurora_port)
        self._set_combo_items(self.openrb_port_combo, state.available_ports, state.openrb_port)

    def _set_combo_items(self, combo: QComboBox, ports, selected: str) -> None:
        current = combo.currentText()
        combo.blockSignals(True)
        combo.clear()
        for port in ports:
            combo.addItem(f"{port.device} ({port.description})", port.device)
        combo.setEditText(selected or current)
        combo.blockSignals(False)

    def _rescan_ports(self) -> None:
        self.controller.rescan_ports()

    def _connect_tracker(self) -> None:
        self.controller.set_aurora_port(self._selected_port(self.aurora_port_combo))
        self.controller.connect_tracker()

    def _connect_openrb(self) -> None:
        self.controller.set_openrb_port(self._selected_port(self.openrb_port_combo))
        self.controller.connect_openrb()

    @staticmethod
    def _selected_port(combo: QComboBox) -> str:
        return str(combo.currentData() or combo.currentText()).strip()

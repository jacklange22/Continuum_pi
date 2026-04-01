"""System tab widget."""

from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSpinBox,
    QSplitter,
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
        self.setObjectName("systemWorkspace")
        self.setStyleSheet(
            """
            QWidget#systemWorkspace {
                background: #eef3f8;
                color: #0f172a;
            }
            QWidget#systemWorkspace QGroupBox {
                border: 1px solid #d9e3ec;
                border-radius: 16px;
                margin-top: 16px;
                padding-top: 10px;
                background: #ffffff;
            }
            QWidget#systemWorkspace QGroupBox::title {
                subcontrol-origin: margin;
                left: 12px;
                padding: 0 6px;
                color: #0f172a;
                font-weight: 700;
            }
            QWidget#systemWorkspace QLabel[role="title"] {
                font-size: 24px;
                font-weight: 700;
                color: #0f172a;
            }
            QWidget#systemWorkspace QLabel[role="hint"] {
                color: #475569;
            }
            QWidget#systemWorkspace QLabel[role="status"] {
                padding: 8px 10px;
                border-radius: 8px;
                background: #e2e8f0;
                color: #0f172a;
                font-weight: 700;
            }
            QWidget#systemWorkspace QComboBox,
            QWidget#systemWorkspace QSpinBox,
            QWidget#systemWorkspace QTextEdit {
                border: 1px solid #dbe4ee;
                border-radius: 10px;
                background: #fbfdff;
                color: #0f172a;
            }
            QWidget#systemWorkspace QPushButton {
                min-height: 36px;
                padding: 0 14px;
                border: 1px solid #dbe4ee;
                border-radius: 10px;
                background: #f8fafc;
                color: #0f172a;
                font-weight: 600;
            }
            QWidget#systemWorkspace QPushButton[role="primary"] {
                background: #dbeafe;
                border-color: #93c5fd;
            }
            """
        )

        self.title_label = QLabel("System Workspace")
        self.title_label.setProperty("role", "title")
        self.workflow_hint = QLabel(
            "Confirm tracker readiness and OpenRB status first. Then save any one-servo bring-up parameters "
            "before reconnecting and moving to the Servos tab."
        )
        self.workflow_hint.setProperty("role", "hint")
        self.workflow_hint.setWordWrap(True)

        self.mock_mode_label = QLabel()
        self.mock_mode_label.setProperty("role", "status")
        self.tracker_status_label = QLabel()
        self.tracker_status_label.setWordWrap(True)
        self.openrb_status_label = QLabel()
        self.openrb_status_label.setWordWrap(True)
        self.expected_servo_ids_label = QLabel()
        self.expected_servo_ids_label.setWordWrap(True)
        self.readiness_status_label = QLabel()
        self.readiness_status_label.setWordWrap(True)
        self.bus_status_label = QLabel()
        self.bus_status_label.setWordWrap(True)
        self.external_power_label = QLabel()
        self.external_power_label.setWordWrap(True)
        self.aurora_port_combo = QComboBox()
        self.aurora_port_combo.setEditable(True)
        self.aurora_port_combo.currentIndexChanged.connect(self._sync_aurora_port)
        self.aurora_port_combo.editTextChanged.connect(self._sync_aurora_port)
        self.openrb_port_combo = QComboBox()
        self.openrb_port_combo.setEditable(True)
        self.openrb_port_combo.currentIndexChanged.connect(self._sync_openrb_port)
        self.openrb_port_combo.editTextChanged.connect(self._sync_openrb_port)

        tracker_connect = QPushButton("Connect Tracker")
        tracker_disconnect = QPushButton("Disconnect Tracker")
        openrb_connect = QPushButton("Connect OpenRB")
        openrb_connect.setProperty("role", "primary")
        openrb_disconnect = QPushButton("Disconnect OpenRB")
        prepare_button = QPushButton("Re-Validate OpenRB")
        readiness_button = QPushButton("Refresh Readiness")
        refresh_button = QPushButton("Rescan Ports")
        refresh_button.clicked.connect(self._rescan_ports)
        tracker_connect.clicked.connect(self._connect_tracker)
        tracker_disconnect.clicked.connect(self.controller.disconnect_tracker)
        openrb_connect.clicked.connect(self._connect_openrb)
        openrb_disconnect.clicked.connect(self.controller.disconnect_openrb)
        prepare_button.clicked.connect(self.controller.prepare_openrb)
        readiness_button.clicked.connect(self._refresh_readiness)

        form = QFormLayout()
        form.setLabelAlignment(Qt.AlignLeft)
        form.addRow("Mock mode", self.mock_mode_label)
        form.addRow("Aurora port", self.aurora_port_combo)
        form.addRow("OpenRB port", self.openrb_port_combo)
        form.addRow("Tracker state", self.tracker_status_label)
        form.addRow("OpenRB state", self.openrb_status_label)
        form.addRow("Expected servo IDs", self.expected_servo_ids_label)
        form.addRow("Readiness", self.readiness_status_label)
        form.addRow("Bus response", self.bus_status_label)
        form.addRow("External power", self.external_power_label)

        ports_box = QGroupBox("Connections")
        ports_layout = QVBoxLayout(ports_box)
        ports_layout.addLayout(form)

        button_row_primary = QHBoxLayout()
        button_row_primary.setSpacing(10)
        button_row_primary.addWidget(tracker_connect)
        button_row_primary.addWidget(tracker_disconnect)
        button_row_primary.addWidget(openrb_connect)
        button_row_primary.addWidget(openrb_disconnect)
        ports_layout.addLayout(button_row_primary)

        button_row_secondary = QHBoxLayout()
        button_row_secondary.setSpacing(10)
        button_row_secondary.addWidget(prepare_button)
        button_row_secondary.addWidget(readiness_button)
        button_row_secondary.addWidget(refresh_button)
        button_row_secondary.addStretch(1)
        ports_layout.addLayout(button_row_secondary)

        self.robot_config_combo = QComboBox()
        self.mock_mode_combo = QComboBox()
        self.mock_mode_combo.addItem("Enabled", True)
        self.mock_mode_combo.addItem("Disabled", False)
        self.baudrate_spin = QSpinBox()
        self.baudrate_spin.setRange(9600, 4000000)
        self.fine_jog_spin = QSpinBox()
        self.fine_jog_spin.setRange(1, 512)
        self.coarse_jog_spin = QSpinBox()
        self.coarse_jog_spin.setRange(1, 1024)
        self.min_offset_spin = QSpinBox()
        self.min_offset_spin.setRange(-4096, 0)
        self.max_offset_spin = QSpinBox()
        self.max_offset_spin.setRange(0, 4096)
        self.software_margin_spin = QSpinBox()
        self.software_margin_spin.setRange(0, 1024)
        self.telemetry_freshness_spin = QDoubleSpinBox()
        self.telemetry_freshness_spin.setRange(0.01, 10.0)
        self.telemetry_freshness_spin.setDecimals(2)
        self.telemetry_freshness_spin.setSingleStep(0.05)
        self.threshold_spin = QSpinBox()
        self.threshold_spin.setRange(1, 5000)
        self.tightening_direction_combo = QComboBox()
        self.tightening_direction_combo.addItems(["cw", "ccw"])
        self.save_parameters_button = QPushButton("Save Runtime Parameters")
        self.save_parameters_button.setProperty("role", "primary")
        self.save_parameters_button.clicked.connect(self._save_runtime_parameters)
        self.parameters_hint = QLabel(
            "These values update `config/system.local.yaml`. Restart the app or reconnect hardware before relying on them."
        )
        self.parameters_hint.setProperty("role", "hint")
        self.parameters_hint.setWordWrap(True)
        self.saved_path_label = QLabel("none")
        self.saved_path_label.setWordWrap(True)

        parameters_box = QGroupBox("Bring-Up Parameters")
        parameters_layout = QVBoxLayout(parameters_box)
        parameters_form = QFormLayout()
        parameters_form.setLabelAlignment(Qt.AlignLeft)
        parameters_form.addRow("Mock mode", self.mock_mode_combo)
        parameters_form.addRow("Robot config", self.robot_config_combo)
        parameters_form.addRow("Baudrate", self.baudrate_spin)
        parameters_form.addRow("Fine jog (ticks)", self.fine_jog_spin)
        parameters_form.addRow("Coarse jog (ticks)", self.coarse_jog_spin)
        parameters_form.addRow("Min offset (ticks)", self.min_offset_spin)
        parameters_form.addRow("Max offset (ticks)", self.max_offset_spin)
        parameters_form.addRow("Software margin (ticks)", self.software_margin_spin)
        parameters_form.addRow("Telemetry fresh (s)", self.telemetry_freshness_spin)
        parameters_form.addRow("Pretension threshold (mA)", self.threshold_spin)
        parameters_form.addRow("Tightening direction", self.tightening_direction_combo)
        parameters_form.addRow("Saved overrides", self.saved_path_label)
        parameters_layout.addLayout(parameters_form)
        parameters_layout.addWidget(self.save_parameters_button)
        parameters_layout.addWidget(self.parameters_hint)

        self.config_summary = QTextEdit()
        self.config_summary.setReadOnly(True)
        self.config_summary.setMinimumHeight(220)

        self.status_text = QTextEdit()
        self.status_text.setReadOnly(True)
        self.status_text.setMinimumHeight(220)

        summary_box = QGroupBox("Config Summary")
        summary_layout = QVBoxLayout(summary_box)
        summary_layout.addWidget(self.config_summary)

        diagnostics_box = QGroupBox("Diagnostics")
        diagnostics_layout = QVBoxLayout(diagnostics_box)
        diagnostics_layout.addWidget(self.status_text)

        lower_splitter = QSplitter(Qt.Horizontal)
        lower_splitter.setChildrenCollapsible(False)
        lower_splitter.setHandleWidth(8)
        lower_splitter.addWidget(summary_box)
        lower_splitter.addWidget(diagnostics_box)
        lower_splitter.setStretchFactor(0, 3)
        lower_splitter.setStretchFactor(1, 4)
        lower_splitter.setSizes([420, 560])

        layout = QVBoxLayout(self)
        layout.setContentsMargins(18, 18, 18, 18)
        layout.setSpacing(14)
        layout.addWidget(self.title_label)
        layout.addWidget(self.workflow_hint)
        layout.addWidget(ports_box)
        layout.addWidget(parameters_box)
        layout.addWidget(lower_splitter, 1)

    def update(self, state: SystemViewState) -> None:
        self.mock_mode_label.setText("Enabled" if state.mock_mode else "Disabled")
        self.tracker_status_label.setText(
            f"{state.tracker_connection_state} | backend={state.tracker_backend_identity} "
            f"| running={state.tracker_backend_running} | connected={state.tracker_backend_connected}"
        )
        self.openrb_status_label.setText(
            f"{state.openrb_status} | prepared={state.openrb_prepared} | bus connected={state.dynamixel_connected}"
        )
        self.expected_servo_ids_label.setText(", ".join(str(servo_id) for servo_id in state.expected_servo_ids) or "none")
        self.readiness_status_label.setText(
            f"{state.readiness_message} | motion ready={state.motion_ready}"
        )
        self.bus_status_label.setText("responsive" if state.bus_reachable else "no confirmed servo response yet")
        if state.external_power_ready is None:
            external_power = "not confirmed"
        else:
            external_power = "ready" if state.external_power_ready else "blocked"
        self.external_power_label.setText(external_power)
        self.config_summary.setPlainText(state.config_summary)
        status_lines = [state.status_message]
        if state.last_error:
            status_lines.append(f"Error: {state.last_error}")
        self.status_text.setPlainText("\n".join(status_lines))
        self._set_combo_items(self.aurora_port_combo, state.available_ports, state.aurora_port)
        self._set_combo_items(self.openrb_port_combo, state.available_ports, state.openrb_port)

        if not self.robot_config_combo.hasFocus():
            self.robot_config_combo.blockSignals(True)
            self.robot_config_combo.clear()
            for robot_config in state.available_robot_configs:
                self.robot_config_combo.addItem(robot_config, robot_config)
            index = self.robot_config_combo.findData(state.robot_config)
            if index >= 0:
                self.robot_config_combo.setCurrentIndex(index)
            self.robot_config_combo.blockSignals(False)

        if not self.mock_mode_combo.hasFocus():
            self.mock_mode_combo.blockSignals(True)
            self.mock_mode_combo.setCurrentIndex(0 if state.mock_mode else 1)
            self.mock_mode_combo.blockSignals(False)

        if not self.baudrate_spin.hasFocus():
            self.baudrate_spin.setValue(int(state.baudrate))
        if not self.fine_jog_spin.hasFocus():
            self.fine_jog_spin.setValue(int(state.fine_jog_step_ticks))
        if not self.coarse_jog_spin.hasFocus():
            self.coarse_jog_spin.setValue(int(state.coarse_jog_step_ticks))
        if not self.min_offset_spin.hasFocus():
            self.min_offset_spin.setValue(int(state.position_min_offset_ticks))
        if not self.max_offset_spin.hasFocus():
            self.max_offset_spin.setValue(int(state.position_max_offset_ticks))
        if not self.software_margin_spin.hasFocus():
            self.software_margin_spin.setValue(int(state.software_position_margin_ticks))
        if not self.telemetry_freshness_spin.hasFocus():
            self.telemetry_freshness_spin.setValue(float(state.telemetry_freshness_timeout_s))
        if not self.threshold_spin.hasFocus():
            self.threshold_spin.setValue(int(state.pretension_threshold_ma))
        if not self.tightening_direction_combo.hasFocus():
            direction_index = self.tightening_direction_combo.findText(
                str(state.tightening_direction_default).lower()
            )
            self.tightening_direction_combo.setCurrentIndex(max(0, direction_index))
        self.saved_path_label.setText(state.saved_overrides_path or "none")

    def _set_combo_items(self, combo: QComboBox, ports, selected: str) -> None:
        if combo.hasFocus():
            return
        current = str(combo.currentText()).strip()
        desired = str(selected or current).strip()
        desired_items = [(f"{port.device} ({port.description})", port.device) for port in ports]
        current_items = [
            (str(combo.itemText(index)), str(combo.itemData(index) or ""))
            for index in range(combo.count())
        ]
        if current_items == desired_items and current == desired:
            return
        combo.blockSignals(True)
        combo.clear()
        for port in ports:
            combo.addItem(f"{port.device} ({port.description})", port.device)
        combo.setEditText(desired)
        combo.blockSignals(False)

    def _rescan_ports(self) -> None:
        self.controller.rescan_ports()

    def _sync_aurora_port(self, _value=None) -> None:
        self.controller.set_aurora_port(self._selected_port(self.aurora_port_combo))

    def _sync_openrb_port(self, _value=None) -> None:
        self.controller.set_openrb_port(self._selected_port(self.openrb_port_combo))

    def _connect_tracker(self) -> None:
        self._sync_aurora_port()
        self.controller.connect_tracker()

    def _connect_openrb(self) -> None:
        self._sync_openrb_port()
        self.controller.connect_openrb()

    def _refresh_readiness(self) -> None:
        refresh_fn = getattr(self.controller, "refresh_readiness", None)
        if callable(refresh_fn):
            refresh_fn()
            return
        self.update(self.controller.refresh())

    def _save_runtime_parameters(self) -> None:
        try:
            self.controller.save_runtime_parameters(
                mock_mode=bool(self.mock_mode_combo.currentData()),
                robot_config=str(self.robot_config_combo.currentData() or self.robot_config_combo.currentText()).strip(),
                openrb_port=self._selected_port(self.openrb_port_combo),
                baudrate=int(self.baudrate_spin.value()),
                fine_jog_step_ticks=int(self.fine_jog_spin.value()),
                coarse_jog_step_ticks=int(self.coarse_jog_spin.value()),
                position_min_offset_ticks=int(self.min_offset_spin.value()),
                position_max_offset_ticks=int(self.max_offset_spin.value()),
                software_position_margin_ticks=int(self.software_margin_spin.value()),
                telemetry_freshness_timeout_s=float(self.telemetry_freshness_spin.value()),
                pretension_threshold_ma=int(self.threshold_spin.value()),
                tightening_direction=str(self.tightening_direction_combo.currentText()).strip().lower(),
            )
        except Exception:
            self.update(self.controller.refresh())

    @staticmethod
    def _selected_port(combo: QComboBox) -> str:
        current_text = str(combo.currentText()).strip()
        if not current_text:
            return ""
        current_index = combo.currentIndex()
        if current_index >= 0:
            item_text = str(combo.itemText(current_index)).strip()
            item_data = str(combo.itemData(current_index) or "").strip()
            if item_data and current_text in {item_text, item_data}:
                return item_data
        matched_index = combo.findData(current_text)
        if matched_index >= 0:
            return str(combo.itemData(matched_index) or current_text).strip()
        return current_text

"""System tab widget."""

from __future__ import annotations

from typing import Callable

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QApplication,
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPlainTextEdit,
    QPushButton,
    QScrollArea,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from continuum_robot.gui.controllers.system_controller import SystemViewState


class SystemTab(QWidget):
    """System connectivity and troubleshooting UI."""

    def __init__(
        self,
        controller,
        parent=None,
        *,
        apply_runtime_parameters: Callable[..., None] | None = None,
    ) -> None:
        super().__init__(parent)
        self.controller = controller
        self._apply_runtime_parameters = apply_runtime_parameters
        self._updating_parameter_widgets = False
        self._parameter_dirty = False
        self._applied_parameter_values: dict[str, object] = {}
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
            QWidget#systemWorkspace QDoubleSpinBox,
            QWidget#systemWorkspace QPlainTextEdit {
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
            "Use this page as the operator bring-up summary. Confirm OpenRB readiness, review diagnostics, "
            "adjust only the key bring-up parameters, then Save + Apply to reload the runtime cleanly."
        )
        self.workflow_hint.setProperty("role", "hint")
        self.workflow_hint.setWordWrap(True)

        self.mock_mode_label = QLabel()
        self.mock_mode_label.setProperty("role", "status")
        self.robot_config_label = QLabel()
        self.robot_config_label.setWordWrap(True)
        self.expected_servo_ids_label = QLabel()
        self.expected_servo_ids_label.setWordWrap(True)
        self.refresh_rate_label = QLabel()
        self.refresh_rate_label.setWordWrap(True)
        self.freshness_label = QLabel()
        self.freshness_label.setWordWrap(True)
        self.tracker_status_label = QLabel()
        self.tracker_status_label.setWordWrap(True)
        self.openrb_status_label = QLabel()
        self.openrb_status_label.setWordWrap(True)
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

        self.tracker_connect_button = QPushButton("Connect Tracker")
        self.tracker_disconnect_button = QPushButton("Disconnect Tracker")
        self.openrb_connect_button = QPushButton("Connect OpenRB")
        self.openrb_connect_button.setProperty("role", "primary")
        self.openrb_disconnect_button = QPushButton("Disconnect OpenRB")
        self.prepare_button = QPushButton("Re-Validate OpenRB")
        self.readiness_button = QPushButton("Refresh Readiness")
        self.refresh_button = QPushButton("Rescan Ports")
        self.refresh_button.clicked.connect(self._rescan_ports)
        self.tracker_connect_button.clicked.connect(self._connect_tracker)
        self.tracker_disconnect_button.clicked.connect(self.controller.disconnect_tracker)
        self.openrb_connect_button.clicked.connect(self._connect_openrb)
        self.openrb_disconnect_button.clicked.connect(self.controller.disconnect_openrb)
        self.prepare_button.clicked.connect(self.controller.prepare_openrb)
        self.readiness_button.clicked.connect(self._refresh_readiness)

        summary_box = QGroupBox("Runtime / Connection Summary")
        summary_layout = QVBoxLayout(summary_box)
        summary_form = QFormLayout()
        summary_form.setLabelAlignment(Qt.AlignLeft)
        summary_form.addRow("Mock mode", self.mock_mode_label)
        summary_form.addRow("Robot config", self.robot_config_label)
        summary_form.addRow("Expected servo IDs", self.expected_servo_ids_label)
        summary_form.addRow("GUI refresh", self.refresh_rate_label)
        summary_form.addRow("Freshness limit", self.freshness_label)
        summary_form.addRow("Aurora port", self.aurora_port_combo)
        summary_form.addRow("OpenRB port", self.openrb_port_combo)
        summary_form.addRow("Tracker state", self.tracker_status_label)
        summary_form.addRow("OpenRB state", self.openrb_status_label)
        summary_form.addRow("Readiness", self.readiness_status_label)
        summary_form.addRow("Bus response", self.bus_status_label)
        summary_form.addRow("External power", self.external_power_label)
        summary_layout.addLayout(summary_form)

        summary_primary_buttons = QHBoxLayout()
        summary_primary_buttons.setSpacing(10)
        summary_primary_buttons.addWidget(self.tracker_connect_button)
        summary_primary_buttons.addWidget(self.tracker_disconnect_button)
        summary_primary_buttons.addWidget(self.openrb_connect_button)
        summary_primary_buttons.addWidget(self.openrb_disconnect_button)
        summary_layout.addLayout(summary_primary_buttons)

        summary_secondary_buttons = QHBoxLayout()
        summary_secondary_buttons.setSpacing(10)
        summary_secondary_buttons.addWidget(self.prepare_button)
        summary_secondary_buttons.addWidget(self.readiness_button)
        summary_secondary_buttons.addWidget(self.refresh_button)
        summary_secondary_buttons.addStretch(1)
        summary_layout.addLayout(summary_secondary_buttons)

        self.robot_config_combo = QComboBox()
        self.robot_config_combo.currentIndexChanged.connect(self._mark_parameter_dirty)
        self.mock_mode_combo = QComboBox()
        self.mock_mode_combo.addItem("Enabled", True)
        self.mock_mode_combo.addItem("Disabled", False)
        self.mock_mode_combo.currentIndexChanged.connect(self._mark_parameter_dirty)
        self.baudrate_spin = QSpinBox()
        self.baudrate_spin.setRange(9600, 4000000)
        self.baudrate_spin.valueChanged.connect(self._mark_parameter_dirty)
        self.poll_rate_spin = QSpinBox()
        self.poll_rate_spin.setRange(1, 60)
        self.poll_rate_spin.valueChanged.connect(self._mark_parameter_dirty)
        self.fine_jog_spin = QSpinBox()
        self.fine_jog_spin.setRange(1, 512)
        self.fine_jog_spin.valueChanged.connect(self._mark_parameter_dirty)
        self.coarse_jog_spin = QSpinBox()
        self.coarse_jog_spin.setRange(1, 1024)
        self.coarse_jog_spin.valueChanged.connect(self._mark_parameter_dirty)
        self.telemetry_freshness_spin = QDoubleSpinBox()
        self.telemetry_freshness_spin.setRange(0.01, 10.0)
        self.telemetry_freshness_spin.setDecimals(3)
        self.telemetry_freshness_spin.setSingleStep(0.05)
        self.telemetry_freshness_spin.valueChanged.connect(self._mark_parameter_dirty)
        self.saved_path_label = QLabel("none")
        self.saved_path_label.setWordWrap(True)

        self.save_parameters_button = QPushButton("Save + Apply")
        self.save_parameters_button.setProperty("role", "primary")
        self.save_parameters_button.clicked.connect(self._save_runtime_parameters)
        self.parameters_hint = QLabel(
            "Save + Apply writes `config/system.local.yaml`, then rebuilds the controllers and services "
            "using the saved values. Existing hardware connections are closed during reload."
        )
        self.parameters_hint.setProperty("role", "hint")
        self.parameters_hint.setWordWrap(True)

        parameters_box = QGroupBox("Bring-Up Parameters")
        parameters_layout = QVBoxLayout(parameters_box)
        parameters_form = QFormLayout()
        parameters_form.setLabelAlignment(Qt.AlignLeft)
        parameters_form.addRow("Mock mode", self.mock_mode_combo)
        parameters_form.addRow("Robot config", self.robot_config_combo)
        parameters_form.addRow("Baudrate", self.baudrate_spin)
        parameters_form.addRow("GUI refresh (Hz)", self.poll_rate_spin)
        parameters_form.addRow("Fine jog (ticks)", self.fine_jog_spin)
        parameters_form.addRow("Coarse jog (ticks)", self.coarse_jog_spin)
        parameters_form.addRow("Telemetry stale after (s)", self.telemetry_freshness_spin)
        parameters_form.addRow("Saved overrides", self.saved_path_label)
        parameters_layout.addLayout(parameters_form)
        parameters_layout.addWidget(self.save_parameters_button)
        parameters_layout.addWidget(self.parameters_hint)

        self.config_summary = QPlainTextEdit()
        self.config_summary.setReadOnly(True)
        self.config_summary.setLineWrapMode(QPlainTextEdit.NoWrap)
        self.config_summary.setMinimumHeight(220)
        self.copy_config_button = QPushButton("Copy Effective Config")
        self.copy_config_button.clicked.connect(lambda: self._copy_text(self.config_summary.toPlainText()))

        config_box = QGroupBox("Effective Config")
        config_layout = QVBoxLayout(config_box)
        config_button_row = QHBoxLayout()
        config_button_row.addStretch(1)
        config_button_row.addWidget(self.copy_config_button)
        config_layout.addLayout(config_button_row)
        config_layout.addWidget(self.config_summary)

        self.status_text = QPlainTextEdit()
        self.status_text.setReadOnly(True)
        self.status_text.setLineWrapMode(QPlainTextEdit.NoWrap)
        self.status_text.setMinimumHeight(240)
        self.copy_diagnostics_button = QPushButton("Copy Diagnostics")
        self.copy_diagnostics_button.clicked.connect(lambda: self._copy_text(self.status_text.toPlainText()))

        diagnostics_box = QGroupBox("Diagnostics")
        diagnostics_layout = QVBoxLayout(diagnostics_box)
        diagnostics_button_row = QHBoxLayout()
        diagnostics_button_row.addStretch(1)
        diagnostics_button_row.addWidget(self.copy_diagnostics_button)
        diagnostics_layout.addLayout(diagnostics_button_row)
        diagnostics_layout.addWidget(self.status_text)

        content_widget = QWidget()
        content_layout = QVBoxLayout(content_widget)
        content_layout.setContentsMargins(0, 0, 0, 0)
        content_layout.setSpacing(14)
        content_layout.addWidget(summary_box)
        content_layout.addWidget(parameters_box)
        content_layout.addWidget(diagnostics_box)
        content_layout.addWidget(config_box)

        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setWidget(content_widget)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(18, 18, 18, 18)
        layout.setSpacing(14)
        layout.addWidget(self.title_label)
        layout.addWidget(self.workflow_hint)
        layout.addWidget(self.scroll_area, 1)

    def update(self, state: SystemViewState) -> None:
        self.mock_mode_label.setText("Enabled" if state.mock_mode else "Disabled")
        self.robot_config_label.setText(f"{state.robot_config} | mode={state.robot_mode}")
        self.expected_servo_ids_label.setText(", ".join(str(servo_id) for servo_id in state.expected_servo_ids) or "none")
        self.refresh_rate_label.setText(f"{int(state.poll_rate_hz)} Hz")
        self.freshness_label.setText(f"{float(state.telemetry_freshness_timeout_s):.3f} s")
        self.tracker_status_label.setText(
            f"{state.tracker_connection_state} | backend={state.tracker_backend_identity} "
            f"| running={state.tracker_backend_running} | connected={state.tracker_backend_connected}"
        )
        self.openrb_status_label.setText(
            f"{state.openrb_status} | prepared={state.openrb_prepared} | bus connected={state.dynamixel_connected}"
        )
        self.readiness_status_label.setText(
            state.readiness_message
        )
        if state.expected_servo_ids:
            self.bus_status_label.setText(
                f"responsive={state.bus_reachable} | detected={len(state.detected_servo_ids)}/{len(state.expected_servo_ids)} "
                f"| telemetry={state.telemetry_ready_count}/{len(state.expected_servo_ids)}"
            )
        else:
            self.bus_status_label.setText("no expected servo IDs configured")
        if state.external_power_ready is None:
            external_power = "not confirmed"
        else:
            external_power = "ready" if state.external_power_ready else "blocked"
        self.external_power_label.setText(external_power)
        self._set_plain_text_preserving_view(self.config_summary, state.config_summary)

        status_lines = [state.status_message]
        if state.last_error:
            status_lines.append(f"Error: {state.last_error}")
        if state.bench_debug_text:
            status_lines.extend(["", state.bench_debug_text])
        self._set_plain_text_preserving_view(self.status_text, "\n".join(status_lines))
        self._set_combo_items(self.aurora_port_combo, state.available_ports, state.aurora_port)
        self._set_combo_items(self.openrb_port_combo, state.available_ports, state.openrb_port)
        applied_values = self._parameter_values_from_state(state)
        if not self._parameter_dirty:
            self._apply_parameter_values(state, applied_values)
        elif self._current_parameter_values() == applied_values:
            self._parameter_dirty = False
            self._apply_parameter_values(state, applied_values)
        self._applied_parameter_values = applied_values
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
        parameters = {
            "mock_mode": bool(self.mock_mode_combo.currentData()),
            "robot_config": str(self.robot_config_combo.currentData() or self.robot_config_combo.currentText()).strip(),
            "openrb_port": self._selected_port(self.openrb_port_combo),
            "baudrate": int(self.baudrate_spin.value()),
            "poll_rate_hz": int(self.poll_rate_spin.value()),
            "fine_jog_step_ticks": int(self.fine_jog_spin.value()),
            "coarse_jog_step_ticks": int(self.coarse_jog_spin.value()),
            "telemetry_freshness_timeout_s": float(self.telemetry_freshness_spin.value()),
        }
        handler = self._apply_runtime_parameters or self.controller.save_runtime_parameters
        try:
            handler(**parameters)
            self._parameter_dirty = False
            self._applied_parameter_values = dict(parameters)
        except Exception:
            self.update(self.controller.refresh())

    def _mark_parameter_dirty(self, *_args) -> None:
        if self._updating_parameter_widgets:
            return
        current_values = self._current_parameter_values()
        self._parameter_dirty = current_values != self._applied_parameter_values

    def _apply_parameter_values(self, state: SystemViewState, values: dict[str, object]) -> None:
        self._updating_parameter_widgets = True
        try:
            self.robot_config_combo.blockSignals(True)
            self.robot_config_combo.clear()
            for robot_config in state.available_robot_configs:
                self.robot_config_combo.addItem(robot_config, robot_config)
            index = self.robot_config_combo.findData(values["robot_config"])
            if index >= 0:
                self.robot_config_combo.setCurrentIndex(index)
            self.robot_config_combo.blockSignals(False)

            self.mock_mode_combo.blockSignals(True)
            self.mock_mode_combo.setCurrentIndex(0 if values["mock_mode"] else 1)
            self.mock_mode_combo.blockSignals(False)

            self.baudrate_spin.setValue(int(values["baudrate"]))
            self.poll_rate_spin.setValue(int(values["poll_rate_hz"]))
            self.fine_jog_spin.setValue(int(values["fine_jog_step_ticks"]))
            self.coarse_jog_spin.setValue(int(values["coarse_jog_step_ticks"]))
            self.telemetry_freshness_spin.setValue(float(values["telemetry_freshness_timeout_s"]))
        finally:
            self._updating_parameter_widgets = False

    def _parameter_values_from_state(self, state: SystemViewState) -> dict[str, object]:
        return {
            "mock_mode": bool(state.mock_mode),
            "robot_config": str(state.robot_config),
            "baudrate": int(state.baudrate),
            "poll_rate_hz": int(state.poll_rate_hz),
            "fine_jog_step_ticks": int(state.fine_jog_step_ticks),
            "coarse_jog_step_ticks": int(state.coarse_jog_step_ticks),
            "telemetry_freshness_timeout_s": float(state.telemetry_freshness_timeout_s),
        }

    def _current_parameter_values(self) -> dict[str, object]:
        return {
            "mock_mode": bool(self.mock_mode_combo.currentData()),
            "robot_config": str(self.robot_config_combo.currentData() or self.robot_config_combo.currentText()).strip(),
            "baudrate": int(self.baudrate_spin.value()),
            "poll_rate_hz": int(self.poll_rate_spin.value()),
            "fine_jog_step_ticks": int(self.fine_jog_spin.value()),
            "coarse_jog_step_ticks": int(self.coarse_jog_spin.value()),
            "telemetry_freshness_timeout_s": float(self.telemetry_freshness_spin.value()),
        }

    @staticmethod
    def _set_plain_text_preserving_view(widget: QPlainTextEdit, text: str) -> None:
        new_text = str(text)
        if widget.toPlainText() == new_text:
            return
        v_scroll = widget.verticalScrollBar()
        h_scroll = widget.horizontalScrollBar()
        old_v = v_scroll.value()
        old_h = h_scroll.value()
        was_at_bottom = old_v >= max(0, v_scroll.maximum() - 2)
        widget.setPlainText(new_text)
        if was_at_bottom:
            v_scroll.setValue(v_scroll.maximum())
        else:
            v_scroll.setValue(min(old_v, v_scroll.maximum()))
        h_scroll.setValue(min(old_h, h_scroll.maximum()))

    @staticmethod
    def _copy_text(text: str) -> None:
        QApplication.clipboard().setText(str(text))

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

"""Servos tab widget."""

from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QAbstractItemView,
    QDoubleSpinBox,
    QFormLayout,
    QGridLayout,
    QGroupBox,
    QHeaderView,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QScrollArea,
    QSpinBox,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from continuum_robot.gui.controllers.servos_controller import ServosViewState


class ServosTab(QWidget):
    """Servo scan, one-servo bring-up, calibration, and cautious motion UI."""

    def __init__(self, controller, parent=None) -> None:
        super().__init__(parent)
        self.controller = controller
        self.displacement_inputs: list[QDoubleSpinBox] = []
        self._displacement_labels: list[QLabel] = []

        self.setObjectName("servoWorkspace")
        self.setStyleSheet(
            """
            QWidget#servoWorkspace {
                background: #eef3f8;
                color: #0f172a;
            }
            QWidget#servoWorkspace QGroupBox {
                border: 1px solid #d9e3ec;
                border-radius: 16px;
                margin-top: 16px;
                padding-top: 10px;
                background: #ffffff;
            }
            QWidget#servoWorkspace QGroupBox::title {
                subcontrol-origin: margin;
                left: 12px;
                padding: 0 6px;
                color: #0f172a;
                font-weight: 700;
            }
            QWidget#servoWorkspace QLabel[role="title"] {
                font-size: 24px;
                font-weight: 700;
                color: #0f172a;
            }
            QWidget#servoWorkspace QLabel[role="hint"] {
                color: #475569;
            }
            QWidget#servoWorkspace QLabel[role="status"] {
                padding: 8px 10px;
                border-radius: 8px;
                background: #e2e8f0;
                color: #0f172a;
                font-weight: 700;
            }
            QWidget#servoWorkspace QPushButton {
                min-height: 36px;
                padding: 0 14px;
                border: 1px solid #dbe4ee;
                border-radius: 10px;
                background: #f8fafc;
                color: #0f172a;
                font-weight: 600;
            }
            QWidget#servoWorkspace QPushButton[role="primary"] {
                background: #dbeafe;
                border-color: #93c5fd;
            }
            QWidget#servoWorkspace QPushButton[role="danger"] {
                background: #fee2e2;
                border-color: #fecaca;
            }
            QWidget#servoWorkspace QSpinBox,
            QWidget#servoWorkspace QDoubleSpinBox,
            QWidget#servoWorkspace QTextEdit,
            QWidget#servoWorkspace QTableWidget {
                border: 1px solid #dbe4ee;
                border-radius: 10px;
                background: #fbfdff;
                color: #0f172a;
            }
            """
        )

        self.title_label = QLabel("Servo Workspace")
        self.title_label.setProperty("role", "title")
        self.workflow_hint = QLabel(
            "Use this workspace for servo bring-up only. Refresh readiness, verify the expected servos, "
            "select one servo at a time, and jog it across the raw 0..4095 range."
        )
        self.workflow_hint.setProperty("role", "hint")
        self.workflow_hint.setWordWrap(True)

        self.connection_label = QLabel()
        self.connection_label.setProperty("role", "status")
        self.mode_label = QLabel()
        self.mode_label.setWordWrap(True)
        self.expected_ids_label = QLabel()
        self.expected_ids_label.setWordWrap(True)
        self.detected_ids_label = QLabel()
        self.detected_ids_label.setWordWrap(True)
        self.missing_ids_label = QLabel()
        self.missing_ids_label.setWordWrap(True)
        self.unexpected_ids_label = QLabel()
        self.unexpected_ids_label.setWordWrap(True)
        self.selected_label = QLabel()
        self.selected_label.setWordWrap(True)
        self.discovery_label = QLabel()
        self.discovery_label.setWordWrap(True)
        self.neutral_label = QLabel()
        self.neutral_label.setWordWrap(True)
        self.pretension_label = QLabel()
        self.pretension_label.setWordWrap(True)
        self.blocking_label = QLabel()
        self.blocking_label.setWordWrap(True)

        self.calibration_status_label = QLabel()
        self.calibration_status_label.setProperty("role", "status")
        self.calibration_path_label = QLabel()
        self.calibration_path_label.setWordWrap(True)
        self.calibration_updated_label = QLabel()
        self.calibration_message_label = QLabel()
        self.calibration_message_label.setWordWrap(True)
        self.calibration_message_label.setProperty("role", "hint")

        self.selected_servo_id_value_label = QLabel("—")
        self.selected_servo_torque_label = QLabel("—")
        self.selected_servo_position_label = QLabel("—")
        self.selected_servo_current_draw_label = QLabel("—")
        self.selected_servo_voltage_label = QLabel("—")
        self.selected_servo_temperature_label = QLabel("—")
        self.selected_servo_target_label = QLabel("—")
        self.selected_servo_bounds_label = QLabel("—")
        self.selected_servo_last_read_label = QLabel("—")
        self.selected_servo_age_label = QLabel("—")
        self.selected_servo_freshness_limit_label = QLabel("—")
        self.selected_servo_fresh_label = QLabel("—")
        self.selected_servo_action_label = QLabel("—")
        self.selected_servo_result_label = QLabel("—")
        self.selected_servo_reason_status_label = QLabel("—")
        self.selected_servo_telemetry_label = QLabel("—")
        self.selected_servo_direction_label = QLabel("—")
        self.selected_servo_last_motion_label = QLabel("—")
        self.selected_servo_ready_label = QLabel("—")
        self.selected_servo_reason_status_label.setWordWrap(True)
        self.selected_servo_direction_label.setWordWrap(True)
        self.selected_servo_last_motion_label.setWordWrap(True)

        self.scan_button = QPushButton("Discover / Read Servo")
        self.scan_button.setProperty("role", "primary")
        self.scan_button.clicked.connect(lambda: self._safe_call(self.controller.scan))
        self.refresh_readiness_button = QPushButton("Refresh Readiness")
        self.refresh_readiness_button.clicked.connect(lambda: self._safe_call(self.controller.refresh_readiness))
        self.capture_neutral_button = QPushButton("Capture Neutral")
        self.capture_neutral_button.clicked.connect(lambda: self._safe_call(self.controller.capture_neutral_setpoints))
        self.load_neutral_button = QPushButton("Load Calibration")
        self.load_neutral_button.clicked.connect(lambda: self._safe_call(self.controller.load_neutral_setpoints))

        self.jog_servo_spin = QSpinBox()
        self.jog_servo_spin.setRange(1, 252)
        self.jog_servo_spin.valueChanged.connect(self._sync_servo_selection)
        self.selector_buttons_widget = QWidget()
        self.selector_buttons_layout = QHBoxLayout(self.selector_buttons_widget)
        self.selector_buttons_layout.setContentsMargins(0, 0, 0, 0)
        self.selector_buttons_layout.setSpacing(8)
        self._selector_buttons: dict[int, QPushButton] = {}
        self.fine_minus_button = QPushButton("Loosen Fine")
        self.fine_plus_button = QPushButton("Tighten Fine")
        self.coarse_minus_button = QPushButton("Loosen Coarse")
        self.coarse_plus_button = QPushButton("Tighten Coarse")
        self.fine_minus_button.clicked.connect(lambda: self._jog("fine", -1))
        self.fine_plus_button.clicked.connect(lambda: self._jog("fine", 1))
        self.coarse_minus_button.clicked.connect(lambda: self._jog("coarse", -1))
        self.coarse_plus_button.clicked.connect(lambda: self._jog("coarse", 1))

        self.calibration_servo_spin = QSpinBox()
        self.calibration_servo_spin.setRange(1, 252)
        self.calibration_servo_spin.valueChanged.connect(self._sync_servo_selection)
        self.min_offset_spin = QSpinBox()
        self.min_offset_spin.setRange(-4096, 0)
        self.max_offset_spin = QSpinBox()
        self.max_offset_spin.setRange(0, 4096)
        self.threshold_spin = QSpinBox()
        self.threshold_spin.setRange(1, 5000)
        self.min_offset_spin.setValue(int(self.controller.settings.safety.position_min_offset_ticks))
        self.max_offset_spin.setValue(int(self.controller.settings.safety.position_max_offset_ticks))
        self.threshold_spin.setValue(int(self.controller.state.default_pretension_threshold_ma))
        self.save_startup_button = QPushButton("Save Startup Calibration")
        self.save_startup_button.setProperty("role", "primary")
        self.save_startup_button.clicked.connect(self._save_startup_calibration)

        self.pretension_servo_spin = QSpinBox()
        self.pretension_servo_spin.setRange(1, 252)
        self.pretension_servo_spin.valueChanged.connect(self._sync_servo_selection)
        self.pretension_threshold_spin = QSpinBox()
        self.pretension_threshold_spin.setRange(1, 5000)
        self.pretension_threshold_spin.setValue(int(self.controller.state.default_pretension_threshold_ma))
        self.start_pretension_button = QPushButton("Start Pretension")
        self.start_pretension_button.setProperty("role", "primary")
        self.cancel_pretension_button = QPushButton("Cancel")
        self.cancel_pretension_button.setProperty("role", "danger")
        self.accept_pretension_button = QPushButton("Accept Result")
        self.retry_pretension_button = QPushButton("Retry")
        self.start_pretension_button.clicked.connect(self._start_pretension)
        self.cancel_pretension_button.clicked.connect(lambda: self._safe_call(self.controller.cancel_pretension))
        self.accept_pretension_button.clicked.connect(self._accept_pretension)
        self.retry_pretension_button.clicked.connect(self._start_pretension)
        self.pretension_hint = QLabel(
            "Pretension uses present current as a practical threshold signal, not true tendon tension."
        )
        self.pretension_hint.setProperty("role", "hint")
        self.pretension_hint.setWordWrap(True)

        self.apply_displacement_button = QPushButton("Apply Displacement")
        self.apply_displacement_button.clicked.connect(self._apply_displacement)

        summary_box = QGroupBox("Bench Bring-Up Summary")
        summary_layout = QFormLayout(summary_box)
        summary_layout.setLabelAlignment(Qt.AlignLeft)
        summary_layout.addRow("Connected", self.connection_label)
        summary_layout.addRow("Mode", self.mode_label)
        summary_layout.addRow("Expected servo IDs", self.expected_ids_label)
        summary_layout.addRow("Detected servo IDs", self.detected_ids_label)
        summary_layout.addRow("Missing expected", self.missing_ids_label)
        summary_layout.addRow("Unexpected found", self.unexpected_ids_label)
        summary_layout.addRow("Selected servo", self.selected_label)
        summary_layout.addRow("Discovery", self.discovery_label)
        summary_layout.addRow("Neutral metadata", self.neutral_label)
        summary_layout.addRow("Current safety block", self.blocking_label)
        summary_layout.addRow("Pretension", self.pretension_label)

        calibration_box = QGroupBox("Calibration State")
        calibration_layout = QVBoxLayout(calibration_box)
        calibration_form = QFormLayout()
        calibration_form.addRow("Status", self.calibration_status_label)
        calibration_form.addRow("Artifact", self.calibration_path_label)
        calibration_form.addRow("Updated", self.calibration_updated_label)
        calibration_layout.addLayout(calibration_form)
        calibration_layout.addWidget(self.calibration_message_label)

        self.calibration_table = QTableWidget(0, 7)
        self.calibration_table.setHorizontalHeaderLabels(
            ["Servo", "Neutral", "Bounds", "Threshold", "Tighten", "Pretension", "Status"]
        )
        self.calibration_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.calibration_table.setSelectionMode(QAbstractItemView.NoSelection)
        self.calibration_table.verticalHeader().setVisible(False)
        self.calibration_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeToContents)
        self.calibration_table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeToContents)
        self.calibration_table.horizontalHeader().setSectionResizeMode(2, QHeaderView.Stretch)
        self.calibration_table.horizontalHeader().setSectionResizeMode(3, QHeaderView.ResizeToContents)
        self.calibration_table.horizontalHeader().setSectionResizeMode(4, QHeaderView.ResizeToContents)
        self.calibration_table.horizontalHeader().setSectionResizeMode(5, QHeaderView.Stretch)
        self.calibration_table.horizontalHeader().setSectionResizeMode(6, QHeaderView.ResizeToContents)
        self.calibration_table.setMinimumHeight(180)
        calibration_layout.addWidget(self.calibration_table)

        bringup_actions_box = QGroupBox("Bring-Up Actions")
        bringup_actions_layout = QVBoxLayout(bringup_actions_box)
        bringup_actions = QHBoxLayout()
        bringup_actions.setSpacing(10)
        bringup_actions.addWidget(self.scan_button)
        bringup_actions.addWidget(self.refresh_readiness_button)
        bringup_actions.addWidget(self.capture_neutral_button)
        bringup_actions.addWidget(self.load_neutral_button)
        bringup_actions.addStretch(1)
        bringup_actions_layout.addLayout(bringup_actions)

        jog_box = QGroupBox("Selected Servo Jog")
        jog_layout = QVBoxLayout(jog_box)
        selector_row = QHBoxLayout()
        selector_row.addWidget(QLabel("Selected servo"))
        selector_row.addWidget(self.selector_buttons_widget, 1)
        jog_layout.addLayout(selector_row)
        jog_form = QFormLayout()
        jog_form.addRow("Selected", self.selected_servo_id_value_label)
        jog_form.addRow("Torque", self.selected_servo_torque_label)
        jog_form.addRow("Position (tick)", self.selected_servo_position_label)
        jog_form.addRow("Current draw (mA)", self.selected_servo_current_draw_label)
        jog_form.addRow("Voltage (mV)", self.selected_servo_voltage_label)
        jog_form.addRow("Temperature (C)", self.selected_servo_temperature_label)
        jog_form.addRow("Target", self.selected_servo_target_label)
        jog_form.addRow("Raw range", self.selected_servo_bounds_label)
        jog_form.addRow("Last read", self.selected_servo_last_read_label)
        jog_form.addRow("Telemetry age", self.selected_servo_age_label)
        jog_form.addRow("Freshness limit", self.selected_servo_freshness_limit_label)
        jog_form.addRow("Fresh", self.selected_servo_fresh_label)
        jog_form.addRow("Telemetry", self.selected_servo_telemetry_label)
        jog_form.addRow("Motion ready", self.selected_servo_ready_label)
        jog_form.addRow("Last action", self.selected_servo_action_label)
        jog_form.addRow("Result", self.selected_servo_result_label)
        jog_form.addRow("Reason", self.selected_servo_reason_status_label)
        jog_form.addRow("Last motion", self.selected_servo_last_motion_label)
        jog_layout.addLayout(jog_form)
        jog_layout.addWidget(self.selected_servo_direction_label)
        jog_buttons = QHBoxLayout()
        jog_buttons.setSpacing(10)
        jog_buttons.addWidget(self.fine_minus_button)
        jog_buttons.addWidget(self.fine_plus_button)
        jog_buttons.addWidget(self.coarse_minus_button)
        jog_buttons.addWidget(self.coarse_plus_button)
        jog_layout.addLayout(jog_buttons)

        self.startup_box = QGroupBox("Startup Calibration")
        startup_layout = QFormLayout(self.startup_box)
        startup_layout.addRow("Servo", self.calibration_servo_spin)
        startup_layout.addRow("Min offset", self.min_offset_spin)
        startup_layout.addRow("Max offset", self.max_offset_spin)
        startup_layout.addRow("Pretension threshold (mA)", self.threshold_spin)
        startup_layout.addRow(self.save_startup_button)

        self.pretension_box = QGroupBox("Pretension")
        pretension_layout = QFormLayout(self.pretension_box)
        pretension_layout.addRow("Servo", self.pretension_servo_spin)
        pretension_layout.addRow("Threshold (mA)", self.pretension_threshold_spin)
        pretension_buttons = QHBoxLayout()
        pretension_buttons.setSpacing(10)
        pretension_buttons.addWidget(self.start_pretension_button)
        pretension_buttons.addWidget(self.cancel_pretension_button)
        pretension_buttons.addWidget(self.retry_pretension_button)
        pretension_buttons.addWidget(self.accept_pretension_button)
        pretension_buttons_widget = QWidget()
        pretension_buttons_widget.setLayout(pretension_buttons)
        pretension_layout.addRow(pretension_buttons_widget)
        pretension_layout.addRow(self.pretension_hint)

        left_column = QWidget()
        left_layout = QVBoxLayout(left_column)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.setSpacing(12)
        left_layout.addWidget(jog_box)
        left_layout.addWidget(bringup_actions_box)
        left_layout.addWidget(self.pretension_box)
        left_layout.addWidget(self.startup_box)

        self.displacement_box = QGroupBox("Tendon Displacement Command (cm)")
        self.displacement_layout = QGridLayout(self.displacement_box)
        self.displacement_layout.setHorizontalSpacing(8)
        self.displacement_layout.setVerticalSpacing(8)
        self._rebuild_displacement_inputs(len(self.controller.state.tendon_displacements_cm))

        telemetry_box = QGroupBox("Telemetry")
        telemetry_layout = QVBoxLayout(telemetry_box)
        self.telemetry_table = QTableWidget(0, 11)
        self.telemetry_table.setHorizontalHeaderLabels(
            [
                "ID",
                "State",
                "Torque",
                "Position (tick)",
                "Current (mA)",
                "Voltage (mV)",
                "Temp (C)",
                "HW Err",
                "Age (s)",
                "Motion",
                "Reason",
            ]
        )
        self.telemetry_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.telemetry_table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.telemetry_table.setSelectionMode(QAbstractItemView.SingleSelection)
        self.telemetry_table.verticalHeader().setVisible(False)
        self.telemetry_table.cellClicked.connect(self._select_servo_from_row)
        for column in range(0, 10):
            self.telemetry_table.horizontalHeader().setSectionResizeMode(column, QHeaderView.ResizeToContents)
        self.telemetry_table.horizontalHeader().setSectionResizeMode(10, QHeaderView.Stretch)
        self.telemetry_table.setMinimumHeight(230)
        telemetry_layout.addWidget(self.telemetry_table)

        right_column = QWidget()
        right_layout = QVBoxLayout(right_column)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.setSpacing(12)
        right_layout.addWidget(telemetry_box, 1)
        right_layout.addWidget(calibration_box)
        right_layout.addWidget(self.displacement_box)

        workspace_splitter = QSplitter(Qt.Horizontal)
        workspace_splitter.setChildrenCollapsible(False)
        workspace_splitter.setHandleWidth(8)
        workspace_splitter.addWidget(left_column)
        workspace_splitter.addWidget(right_column)
        workspace_splitter.setStretchFactor(0, 3)
        workspace_splitter.setStretchFactor(1, 4)
        workspace_splitter.setSizes([460, 700])

        self.status_text = QTextEdit()
        self.status_text.setReadOnly(True)
        self.status_text.setMinimumHeight(82)
        self.status_text.setMaximumHeight(180)
        status_box = QGroupBox("Operator Status")
        status_layout = QVBoxLayout(status_box)
        status_layout.addWidget(self.status_text)

        content_widget = QWidget()
        content_layout = QVBoxLayout(content_widget)
        content_layout.setContentsMargins(0, 0, 0, 0)
        content_layout.setSpacing(14)
        content_layout.addWidget(summary_box)
        content_layout.addWidget(workspace_splitter)
        content_layout.addWidget(status_box)

        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setWidget(content_widget)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(18, 18, 18, 18)
        layout.setSpacing(14)
        layout.addWidget(self.title_label)
        layout.addWidget(self.workflow_hint)
        layout.addWidget(self.scroll_area, 1)

    def update(self, state: ServosViewState) -> None:
        selected_servo_id = state.selected_servo_id or (state.servo_ids[0] if state.servo_ids else 1)
        selected_servo = state.telemetry.get(selected_servo_id, {})

        self.connection_label.setText("Connected" if state.connected else "Disconnected")
        self.mode_label.setText(
            "One-servo bring-up"
            if state.single_servo_mode
            else f"{state.robot_mode} individual-jog bring-up"
        )
        self.scan_button.setText(
            "Discover / Read Servo"
            if state.single_servo_mode
            else "Discover / Read Configured Servos"
        )
        self.expected_ids_label.setText(", ".join(str(sid) for sid in state.expected_servo_ids) or "none")
        self.detected_ids_label.setText(", ".join(str(sid) for sid in state.detected_servo_ids) or "none")
        self.missing_ids_label.setText(", ".join(str(sid) for sid in state.missing_servo_ids) or "none")
        self.unexpected_ids_label.setText(", ".join(str(sid) for sid in state.unexpected_servo_ids) or "none")
        self.selected_label.setText(str(selected_servo_id) if selected_servo_id is not None else "none")
        self.discovery_label.setText(state.discovery_message)
        self.neutral_label.setText(
            ", ".join(f"{sid}:{tick}" for sid, tick in sorted(state.neutral_setpoints.items())) or "not saved"
        )
        self.blocking_label.setText(" | ".join(state.blocking_reasons) if state.blocking_reasons else "none")
        self.pretension_label.setText(state.pretension_message)
        self.calibration_status_label.setText(
            "Ready"
            if state.calibration_exists and state.calibration_compatible
            else ("Review Needed" if state.calibration_exists else "Missing")
        )
        self.calibration_path_label.setText(state.calibration_path or "None")
        self.calibration_updated_label.setText(state.calibration_updated_at_utc or "Never")
        self.calibration_message_label.setText(state.calibration_message)

        current_position = state.selected_servo_current_position_tick
        target_position = state.selected_servo_last_target_tick
        requested_target = state.selected_servo_last_unclamped_target_tick
        if target_position is None:
            target_text = "—"
        elif state.selected_servo_last_motion_clamped and requested_target is not None and requested_target != target_position:
            target_text = f"{target_position} (req {requested_target}, clamped)"
        else:
            target_text = str(target_position)
        if state.selected_servo_safe_min_tick is not None and state.selected_servo_safe_max_tick is not None:
            bounds_text = f"[{state.selected_servo_safe_min_tick}, {state.selected_servo_safe_max_tick}]"
        else:
            bounds_text = str(selected_servo.get("safe_bounds", "—"))
        freshness_limit_text = f"{float(state.telemetry_freshness_threshold_s):.3f} s"
        self.selected_servo_position_label.setText(str(current_position if current_position is not None else "—"))
        self.selected_servo_id_value_label.setText(str(selected_servo_id) if selected_servo_id is not None else "—")
        self.selected_servo_torque_label.setText("On" if state.selected_servo_torque_enabled else ("Off" if state.selected_servo_torque_enabled is False else "—"))
        self.selected_servo_current_draw_label.setText(self._display_value(state.selected_servo_current_ma))
        self.selected_servo_voltage_label.setText(self._display_value(state.selected_servo_voltage_mv))
        self.selected_servo_temperature_label.setText(self._display_value(state.selected_servo_temperature_c))
        self.selected_servo_target_label.setText(target_text)
        self.selected_servo_bounds_label.setText(bounds_text)
        self.selected_servo_last_read_label.setText(state.selected_servo_last_read_label or "unknown")
        self.selected_servo_age_label.setText(
            f"{float(state.selected_servo_telemetry_age_s):.3f} s"
            if state.selected_servo_telemetry_age_s is not None
            else "—"
        )
        self.selected_servo_freshness_limit_label.setText(freshness_limit_text)
        if state.selected_servo_telemetry_fresh is None:
            fresh_text = "Unknown"
        else:
            fresh_text = "Yes" if state.selected_servo_telemetry_fresh else "No"
        self.selected_servo_fresh_label.setText(fresh_text)
        self.selected_servo_action_label.setText(state.selected_servo_last_action_label or "—")
        self.selected_servo_result_label.setText(state.selected_servo_last_result_label or "—")
        self.selected_servo_reason_status_label.setText(state.selected_servo_reason_label or "none")
        self.selected_servo_telemetry_label.setText(state.selected_servo_telemetry_status_label or "Unknown")
        self.selected_servo_direction_label.setText(
            f"Convention: {state.position_convention_summary or '—'}"
        )
        self.selected_servo_last_motion_label.setText(state.selected_servo_last_motion_summary)
        self.selected_servo_ready_label.setText("Yes" if state.selected_servo_motion_ready else "No")
        self._rebuild_servo_selector(state.servo_ids or state.expected_servo_ids, selected_servo_id)

        has_neutral = bool(state.neutral_setpoints)
        any_servo = bool(state.servo_ids)
        show_single_servo_advanced = state.single_servo_mode
        motion_allowed = state.connected and any_servo and state.selected_servo_motion_ready
        self.startup_box.setVisible(show_single_servo_advanced)
        self.pretension_box.setVisible(show_single_servo_advanced)
        self.displacement_box.setVisible(False)
        self.scan_button.setEnabled(state.connected)
        self.refresh_readiness_button.setEnabled(state.connected)
        self.capture_neutral_button.setEnabled(state.connected and any_servo)
        self.load_neutral_button.setEnabled(True)
        self.fine_minus_button.setEnabled(motion_allowed)
        self.fine_plus_button.setEnabled(motion_allowed)
        self.coarse_minus_button.setEnabled(motion_allowed)
        self.coarse_plus_button.setEnabled(motion_allowed)
        self.save_startup_button.setEnabled(show_single_servo_advanced and state.connected and any_servo)
        self.start_pretension_button.setEnabled(show_single_servo_advanced and motion_allowed and not state.pretension_running)
        self.cancel_pretension_button.setEnabled(show_single_servo_advanced and state.pretension_running)
        self.retry_pretension_button.setEnabled(show_single_servo_advanced and motion_allowed and not state.pretension_running)
        self.accept_pretension_button.setEnabled(show_single_servo_advanced and state.pretension_result_can_accept)
        self.apply_displacement_button.setEnabled(False)

        self._set_servo_spin_value(self.jog_servo_spin, selected_servo_id)
        self._set_servo_spin_value(self.calibration_servo_spin, selected_servo_id)
        self._set_servo_spin_value(self.pretension_servo_spin, selected_servo_id)
        max_servo_id = max([252, *state.servo_ids]) if state.servo_ids else 252
        for spin in (self.jog_servo_spin, self.calibration_servo_spin, self.pretension_servo_spin):
            spin.setMaximum(max_servo_id)

        if not self.threshold_spin.hasFocus() and self.threshold_spin.value() <= 1:
            self.threshold_spin.setValue(max(1, state.default_pretension_threshold_ma))
        if not self.pretension_threshold_spin.hasFocus() and self.pretension_threshold_spin.value() <= 1:
            self.pretension_threshold_spin.setValue(max(1, state.default_pretension_threshold_ma))

        if len(self.displacement_inputs) != len(state.tendon_displacements_cm):
            self._rebuild_displacement_inputs(len(state.tendon_displacements_cm))
        for spin, value in zip(self.displacement_inputs, state.tendon_displacements_cm):
            spin.blockSignals(True)
            spin.setValue(value)
            spin.blockSignals(False)

        self.calibration_table.setRowCount(len(state.calibration_rows))
        for row, item in enumerate(state.calibration_rows):
            self.calibration_table.setItem(row, 0, QTableWidgetItem(item["servo_id"]))
            self.calibration_table.setItem(row, 1, QTableWidgetItem(item["neutral"]))
            self.calibration_table.setItem(row, 2, QTableWidgetItem(item["bounds"]))
            self.calibration_table.setItem(row, 3, QTableWidgetItem(item["threshold"]))
            self.calibration_table.setItem(row, 4, QTableWidgetItem(item["direction"]))
            self.calibration_table.setItem(row, 5, QTableWidgetItem(item["pretension"]))
            self.calibration_table.setItem(row, 6, QTableWidgetItem(item["status"]))

        self.telemetry_table.setRowCount(len(state.telemetry))
        sorted_servo_ids = sorted(state.telemetry)
        for row, servo_id in enumerate(sorted_servo_ids):
            item = state.telemetry[servo_id]
            self.telemetry_table.setItem(row, 0, self._text_item(servo_id, align=Qt.AlignRight))
            self.telemetry_table.setItem(row, 1, self._text_item(item.get("telemetry_status", "Unknown"), align=Qt.AlignCenter))
            self.telemetry_table.setItem(row, 2, self._text_item(item.get("torque_label", "—"), align=Qt.AlignCenter))
            self.telemetry_table.setItem(row, 3, self._text_item(self._display_value(item.get("position")), align=Qt.AlignRight))
            self.telemetry_table.setItem(row, 4, self._text_item(self._display_value(item.get("current_ma")), align=Qt.AlignRight))
            self.telemetry_table.setItem(row, 5, self._text_item(self._display_value(item.get("voltage_mv")), align=Qt.AlignRight))
            self.telemetry_table.setItem(row, 6, self._text_item(self._display_value(item.get("temperature_c")), align=Qt.AlignRight))
            self.telemetry_table.setItem(row, 7, self._text_item(item.get("hardware_error_text", "—"), align=Qt.AlignCenter))
            self.telemetry_table.setItem(row, 8, self._text_item(self._age_text(item.get("telemetry_age_s")), align=Qt.AlignRight))
            self.telemetry_table.setItem(row, 9, self._text_item("Yes" if item.get("motion_ready") else "No", align=Qt.AlignCenter))
            self.telemetry_table.setItem(row, 10, self._text_item(item.get("block_reason", "ready")))
        if selected_servo_id in sorted_servo_ids:
            self.telemetry_table.selectRow(sorted_servo_ids.index(selected_servo_id))
        else:
            self.telemetry_table.clearSelection()

        lines = [state.status_message]
        if state.last_error:
            lines.append(f"Error: {state.last_error}")
        if state.bench_debug_text:
            lines.extend(["", state.bench_debug_text])
        self.status_text.setPlainText("\n".join(lines))

    def _rebuild_displacement_inputs(self, count: int) -> None:
        while self.displacement_layout.count():
            item = self.displacement_layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()
        self.displacement_inputs = []
        self._displacement_labels = []
        for index in range(count):
            label = QLabel(f"Tendon {index + 1}")
            spin = QDoubleSpinBox()
            spin.setRange(-5.0, 5.0)
            spin.setDecimals(3)
            spin.setSingleStep(0.05)
            spin.valueChanged.connect(self._update_displacement_values)
            self.displacement_layout.addWidget(label, index, 0)
            self.displacement_layout.addWidget(spin, index, 1)
            self.displacement_inputs.append(spin)
            self._displacement_labels.append(label)
        self.displacement_layout.addWidget(self.apply_displacement_button, max(count, 1), 0, 1, 2)

    def _rebuild_servo_selector(self, servo_ids: list[int], selected_servo_id: int | None) -> None:
        desired_ids = [int(servo_id) for servo_id in servo_ids]
        current_ids = sorted(self._selector_buttons)
        if current_ids != desired_ids:
            while self.selector_buttons_layout.count():
                item = self.selector_buttons_layout.takeAt(0)
                widget = item.widget()
                if widget is not None:
                    widget.deleteLater()
            self._selector_buttons = {}
            for servo_id in desired_ids:
                button = QPushButton(f"Servo {servo_id}")
                button.setCheckable(True)
                button.clicked.connect(lambda _checked=False, sid=servo_id: self._select_servo(sid))
                self.selector_buttons_layout.addWidget(button)
                self._selector_buttons[int(servo_id)] = button
            self.selector_buttons_layout.addStretch(1)
        for servo_id, button in self._selector_buttons.items():
            button.blockSignals(True)
            button.setChecked(int(servo_id) == int(selected_servo_id) if selected_servo_id is not None else False)
            button.blockSignals(False)

    def _update_displacement_values(self) -> None:
        self.controller.set_tendon_displacements([spin.value() for spin in self.displacement_inputs])

    def _apply_displacement(self) -> None:
        self._update_displacement_values()
        self._safe_call(self.controller.apply_displacement)

    def _jog(self, mode: str, direction: int) -> None:
        servo_id = int(self.jog_servo_spin.value())
        if mode == "fine":
            self._safe_call(self.controller.fine_jog, servo_id, int(direction))
            return
        self._safe_call(self.controller.coarse_jog, servo_id, int(direction))

    def _save_startup_calibration(self) -> None:
        self._safe_call(
            self.controller.save_startup_calibration,
            servo_id=int(self.calibration_servo_spin.value()),
            min_offset_ticks=int(self.min_offset_spin.value()),
            max_offset_ticks=int(self.max_offset_spin.value()),
            threshold_ma=int(self.threshold_spin.value()),
        )

    def _start_pretension(self) -> None:
        self._safe_call(
            self.controller.start_pretension,
            int(self.pretension_servo_spin.value()),
            int(self.pretension_threshold_spin.value()),
        )

    def _accept_pretension(self) -> None:
        self._safe_call(self.controller.accept_pretension_result, int(self.pretension_servo_spin.value()))

    def _sync_servo_selection(self, servo_id: int) -> None:
        for spin in (self.jog_servo_spin, self.calibration_servo_spin, self.pretension_servo_spin):
            if spin.value() != int(servo_id):
                spin.blockSignals(True)
                spin.setValue(int(servo_id))
                spin.blockSignals(False)
        set_selected = getattr(self.controller, "set_selected_servo", None)
        if callable(set_selected):
            set_selected(int(servo_id))
        else:
            self.controller.state.selected_servo_id = int(servo_id)

    def _select_servo(self, servo_id: int) -> None:
        self._sync_servo_selection(int(servo_id))
        self.update(self.controller.state)

    def _select_servo_from_row(self, row: int, _column: int) -> None:
        item = self.telemetry_table.item(row, 0)
        if item is None:
            return
        try:
            servo_id = int(item.text())
        except (TypeError, ValueError):
            return
        self._select_servo(int(servo_id))

    @staticmethod
    def _set_servo_spin_value(spin: QSpinBox, value: int) -> None:
        spin.blockSignals(True)
        spin.setValue(int(value))
        spin.blockSignals(False)

    def _safe_call(self, fn, *args, **kwargs) -> None:
        try:
            fn(*args, **kwargs)
        except Exception as exc:
            if not getattr(self.controller.state, "last_error", None):
                self.controller.state.last_error = str(exc)
            if not getattr(self.controller.state, "status_message", ""):
                self.controller.state.status_message = str(exc)
        self.update(self.controller.state)

    @staticmethod
    def _display_value(value) -> str:
        return "—" if value is None else str(value)

    @staticmethod
    def _age_text(age_s: float | None) -> str:
        if age_s is None:
            return "—"
        return f"{float(age_s):.3f}"

    @staticmethod
    def _text_item(value, *, align: int = Qt.AlignLeft | Qt.AlignVCenter) -> QTableWidgetItem:
        item = QTableWidgetItem(str(value))
        item.setTextAlignment(int(align | Qt.AlignVCenter))
        return item

"""Dedicated single-servo pretension workspace."""

from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QApplication,
    QAbstractItemView,
    QDoubleSpinBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QPlainTextEdit,
    QPushButton,
    QScrollArea,
    QSpinBox,
    QSplitter,
    QComboBox,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from continuum_robot.servos.servo_service import PretensionParameters
from continuum_robot.gui.theme import grouped_workspace_stylesheet
from continuum_robot.gui.view_utils import (
    ResponsiveSplitterController,
    preserve_scroll_position,
    set_text_document,
)


class PretensionTab(QWidget):
    """Focused operator UI for selected-servo MVP pretension testing."""

    def __init__(self, controller, parent=None) -> None:
        super().__init__(parent)
        self.controller = controller
        self._updating_parameter_widgets = False
        self._parameter_dirty = False
        self._applied_parameter_values: dict[str, object] = {}
        self._last_selected_servo_id: int | None = None
        self.setObjectName("pretensionWorkspace")
        self.setStyleSheet(
            grouped_workspace_stylesheet(
                object_name="pretensionWorkspace",
                input_selectors=["QSpinBox", "QDoubleSpinBox", "QPlainTextEdit", "QTableWidget"],
            )
        )

        self.title_label = QLabel("Pretension Workspace")
        self.title_label.setProperty("role", "title")
        self.workflow_hint = QLabel(
            "Use this tab to pretension one selected servo while the full configured bus stays connected. "
            "Current is treated as a practical engagement proxy, not true tendon tension."
        )
        self.workflow_hint.setProperty("role", "hint")
        self.workflow_hint.setWordWrap(True)

        self.selection_status_label = QLabel("Select one servo and verify readiness.")
        self.selection_status_label.setProperty("role", "status")
        self.servo_table = QTableWidget(0, 5)
        self.servo_table.setHorizontalHeaderLabels(
            ["Servo", "Position", "Current (mA)", "Pretension Ready", "Status"]
        )
        self.servo_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.servo_table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.servo_table.setSelectionMode(QAbstractItemView.SingleSelection)
        self.servo_table.verticalHeader().setVisible(False)
        self.servo_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeToContents)
        self.servo_table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeToContents)
        self.servo_table.horizontalHeader().setSectionResizeMode(2, QHeaderView.ResizeToContents)
        self.servo_table.horizontalHeader().setSectionResizeMode(3, QHeaderView.ResizeToContents)
        self.servo_table.horizontalHeader().setSectionResizeMode(4, QHeaderView.Stretch)
        self.servo_table.setMinimumHeight(160)
        self.servo_table.cellClicked.connect(self._select_row_servo)

        selection_box = QGroupBox("Servo Selection / Health Summary")
        selection_layout = QVBoxLayout(selection_box)
        selection_layout.addWidget(self.selection_status_label)
        selection_layout.addWidget(self.servo_table)

        self.selected_servo_label = QLabel("—")
        self.torque_label = QLabel("—")
        self.telemetry_label = QLabel("—")
        self.motion_ready_label = QLabel("—")
        self.pretension_ready_label = QLabel("—")
        self.position_label = QLabel("—")
        self.current_label = QLabel("—")
        self.current_validity_label = QLabel("—")
        self.filtered_current_label = QLabel("—")
        self.filtered_current_source_label = QLabel("—")
        self.voltage_label = QLabel("—")
        self.temperature_label = QLabel("—")
        self.bounds_label = QLabel("—")
        self.reference_label = QLabel("—")
        self.travel_window_label = QLabel("—")
        self.direction_label = QLabel("—")
        self.hardware_error_label = QLabel("—")
        self.block_reason_label = QLabel("—")
        self.block_reason_label.setWordWrap(True)
        self.saved_summary_label = QLabel("—")
        self.saved_summary_label.setWordWrap(True)

        feedback_box = QGroupBox("Selected Servo Live Feedback")
        feedback_layout = QFormLayout(feedback_box)
        feedback_layout.addRow("Servo ID", self.selected_servo_label)
        feedback_layout.addRow("Torque", self.torque_label)
        feedback_layout.addRow("Telemetry", self.telemetry_label)
        feedback_layout.addRow("Motion ready", self.motion_ready_label)
        feedback_layout.addRow("Pretension ready", self.pretension_ready_label)
        feedback_layout.addRow("Raw position", self.position_label)
        feedback_layout.addRow("Current draw (mA)", self.current_label)
        feedback_layout.addRow("Current validity", self.current_validity_label)
        feedback_layout.addRow("Filtered current", self.filtered_current_label)
        feedback_layout.addRow("Filtered source", self.filtered_current_source_label)
        feedback_layout.addRow("Voltage (mV)", self.voltage_label)
        feedback_layout.addRow("Temperature (C)", self.temperature_label)
        feedback_layout.addRow("Untensioned reference", self.reference_label)
        feedback_layout.addRow("Effective travel window", self.travel_window_label)
        feedback_layout.addRow("Pretension-safe range", self.bounds_label)
        feedback_layout.addRow("Direction", self.direction_label)
        feedback_layout.addRow("Hardware error", self.hardware_error_label)
        feedback_layout.addRow("Block reason", self.block_reason_label)
        feedback_layout.addRow("Saved result", self.saved_summary_label)

        self.untensioned_reference_spin = QSpinBox()
        self.untensioned_reference_spin.setRange(0, 4095)
        self.start_mode_combo = QComboBox()
        self.start_mode_combo.addItem("Current Position", "current_position")
        self.start_mode_combo.addItem("Manual Startup Artifact", "manual_startup_artifact")
        self.start_mode_combo.addItem("Full Release 4095", "full_release_4095")
        self.step_ticks_spin = QSpinBox()
        self.step_ticks_spin.setRange(1, 256)
        self.settle_time_spin = QDoubleSpinBox()
        self.settle_time_spin.setRange(0.0, 5.0)
        self.settle_time_spin.setDecimals(3)
        self.settle_time_spin.setSingleStep(0.01)
        self.baseline_sample_spin = QSpinBox()
        self.baseline_sample_spin.setRange(1, 50)
        self.filter_window_spin = QSpinBox()
        self.filter_window_spin.setRange(1, 25)
        self.current_delta_spin = QSpinBox()
        self.current_delta_spin.setRange(1, 5000)
        self.absolute_trigger_spin = QSpinBox()
        self.absolute_trigger_spin.setRange(0, 5000)
        self.hard_current_spin = QSpinBox()
        self.hard_current_spin.setRange(1, 10000)
        self.max_travel_spin = QSpinBox()
        self.max_travel_spin.setRange(1, 4095)
        self.timeout_spin = QDoubleSpinBox()
        self.timeout_spin.setRange(0.1, 120.0)
        self.timeout_spin.setDecimals(2)
        self.timeout_spin.setSingleStep(0.1)
        self.min_offset_spin = QSpinBox()
        self.min_offset_spin.setRange(-4096, 0)
        self.max_offset_spin = QSpinBox()
        self.max_offset_spin.setRange(0, 4096)
        self.threshold_spin = QSpinBox()
        self.threshold_spin.setRange(1, 5000)
        for widget in (
            self.untensioned_reference_spin,
            self.step_ticks_spin,
            self.settle_time_spin,
            self.baseline_sample_spin,
            self.filter_window_spin,
            self.current_delta_spin,
            self.absolute_trigger_spin,
            self.hard_current_spin,
            self.max_travel_spin,
            self.timeout_spin,
        ):
            signal = getattr(widget, "valueChanged")
            signal.connect(self._mark_parameter_dirty)
        self.start_mode_combo.currentIndexChanged.connect(self._mark_parameter_dirty)

        parameter_box = QGroupBox("Pretension Parameters")
        parameter_layout = QFormLayout(parameter_box)
        parameter_layout.addRow("Untensioned reference", self.untensioned_reference_spin)
        parameter_layout.addRow("Start mode", self.start_mode_combo)
        parameter_layout.addRow("Pretension step (ticks)", self.step_ticks_spin)
        parameter_layout.addRow("Settle time (s)", self.settle_time_spin)
        parameter_layout.addRow("Baseline samples", self.baseline_sample_spin)
        parameter_layout.addRow("Filter window", self.filter_window_spin)
        parameter_layout.addRow("Current delta trigger (mA)", self.current_delta_spin)
        parameter_layout.addRow("Absolute trigger (mA)", self.absolute_trigger_spin)
        parameter_layout.addRow("Hard current stop (mA)", self.hard_current_spin)
        parameter_layout.addRow("Max pretension travel", self.max_travel_spin)
        parameter_layout.addRow("Timeout (s)", self.timeout_spin)
        parameter_layout.addRow(QLabel("Startup calibration"))
        parameter_layout.addRow("Neutral min offset", self.min_offset_spin)
        parameter_layout.addRow("Neutral max offset", self.max_offset_spin)
        parameter_layout.addRow("Saved threshold (mA)", self.threshold_spin)
        self.apply_live_button = QPushButton("Apply Live Parameters")
        self.apply_live_button.setProperty("role", "primary")
        self.save_defaults_button = QPushButton("Save Parameters as Defaults")
        self.save_defaults_button.setProperty("variant", "ghost")
        self.apply_live_button.clicked.connect(self._apply_live_parameters)
        self.save_defaults_button.clicked.connect(self._save_parameter_defaults)
        parameter_buttons = QHBoxLayout()
        parameter_buttons.setSpacing(10)
        parameter_buttons.addWidget(self.apply_live_button)
        parameter_buttons.addWidget(self.save_defaults_button)
        parameter_layout.addRow(parameter_buttons)

        self.refresh_button = QPushButton("Refresh Selected Servo")
        self.refresh_button.setProperty("variant", "ghost")
        self.measure_baseline_button = QPushButton("Baseline Check")
        self.measure_baseline_button.setProperty("variant", "ghost")
        self.move_reference_button = QPushButton("Move to Untensioned Reference")
        self.move_reference_button.setProperty("variant", "ghost")
        self.pretension_button = QPushButton("Pretension Selected Servo")
        self.pretension_button.setProperty("role", "primary")
        self.stop_button = QPushButton("Stop Pretension")
        self.stop_button.setProperty("role", "danger")
        self.save_button = QPushButton("Save Pretension Result")
        self.save_button.setProperty("variant", "ghost")
        self.save_startup_button = QPushButton("Save Startup Calibration")
        self.save_startup_button.setProperty("variant", "ghost")

        self.refresh_button.clicked.connect(lambda: self._safe_call(self.controller.refresh))
        self.measure_baseline_button.clicked.connect(self._measure_baseline)
        self.move_reference_button.clicked.connect(self._move_to_reference)
        self.pretension_button.clicked.connect(self._start_pretension)
        self.stop_button.clicked.connect(lambda: self._safe_call(self.controller.stop_pretension))
        self.save_button.clicked.connect(lambda: self._safe_call(self.controller.save_pretension_result))
        self.save_startup_button.clicked.connect(self._save_startup_calibration)

        actions_box = QGroupBox("Actions")
        actions_layout = QVBoxLayout(actions_box)
        actions_row_one = QHBoxLayout()
        actions_row_one.setSpacing(10)
        actions_row_one.addWidget(self.refresh_button)
        actions_row_one.addWidget(self.move_reference_button)
        actions_row_one.addWidget(self.save_button)
        actions_row_two = QHBoxLayout()
        actions_row_two.setSpacing(10)
        actions_row_two.addWidget(self.pretension_button)
        actions_row_two.addWidget(self.measure_baseline_button)
        actions_row_two.addWidget(self.stop_button)
        actions_row_three = QHBoxLayout()
        actions_row_three.setSpacing(10)
        actions_row_three.addWidget(self.save_startup_button)
        actions_row_three.addStretch(1)
        actions_layout.addLayout(actions_row_one)
        actions_layout.addLayout(actions_row_two)
        actions_layout.addLayout(actions_row_three)

        self.run_state_label = QLabel("Idle")
        self.run_state_label.setProperty("role", "status")
        self.progress_message_label = QLabel("Pretension not started.")
        self.progress_message_label.setWordWrap(True)
        self.baseline_label = QLabel("Not measured.")
        self.start_position_label = QLabel("—")
        self.last_target_label = QLabel("—")
        self.current_position_label = QLabel("—")
        self.trigger_label = QLabel("—")
        self.steps_label = QLabel("0")
        self.elapsed_label = QLabel("0.00 s")
        self.final_position_label = QLabel("—")
        self.stop_reason_label = QLabel("—")
        self.stop_reason_label.setWordWrap(True)
        self.failure_phase_label = QLabel("—")
        self.failure_primary_label = QLabel("—")
        self.failure_primary_label.setWordWrap(True)
        self.failure_detail_label = QLabel("—")
        self.failure_detail_label.setWordWrap(True)

        progress_box = QGroupBox("Pretension Progress / Result")
        progress_layout = QFormLayout(progress_box)
        progress_layout.addRow("State", self.run_state_label)
        progress_layout.addRow("Run message", self.progress_message_label)
        progress_layout.addRow("Baseline", self.baseline_label)
        progress_layout.addRow("Start position", self.start_position_label)
        progress_layout.addRow("Last target", self.last_target_label)
        progress_layout.addRow("Current position", self.current_position_label)
        progress_layout.addRow("Trigger threshold", self.trigger_label)
        progress_layout.addRow("Steps taken", self.steps_label)
        progress_layout.addRow("Elapsed", self.elapsed_label)
        progress_layout.addRow("Final position", self.final_position_label)
        progress_layout.addRow("Stop / block reason", self.stop_reason_label)
        progress_layout.addRow("Failure phase", self.failure_phase_label)
        progress_layout.addRow("Primary reason", self.failure_primary_label)
        progress_layout.addRow("Detail", self.failure_detail_label)

        self.log_text = QPlainTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setLineWrapMode(QPlainTextEdit.WidgetWidth)
        self.copy_log_button = QPushButton("Copy Operator Log")
        self.copy_log_button.clicked.connect(lambda: self._copy_text(self.log_text.toPlainText()))
        log_box = QGroupBox("Copyable Operator Status / Log")
        log_layout = QVBoxLayout(log_box)
        log_button_row = QHBoxLayout()
        log_button_row.addStretch(1)
        log_button_row.addWidget(self.copy_log_button)
        log_layout.addLayout(log_button_row)
        log_layout.addWidget(self.log_text)

        self.comparison_table = QTableWidget(0, 6)
        self.comparison_table.setHorizontalHeaderLabels(
            ["Servo", "Status", "Final Pos", "Baseline (mA)", "Trigger (mA)", "Travel / Reason"]
        )
        self.comparison_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.comparison_table.setSelectionMode(QAbstractItemView.NoSelection)
        self.comparison_table.verticalHeader().setVisible(False)
        self.comparison_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeToContents)
        self.comparison_table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeToContents)
        self.comparison_table.horizontalHeader().setSectionResizeMode(2, QHeaderView.ResizeToContents)
        self.comparison_table.horizontalHeader().setSectionResizeMode(3, QHeaderView.ResizeToContents)
        self.comparison_table.horizontalHeader().setSectionResizeMode(4, QHeaderView.ResizeToContents)
        self.comparison_table.horizontalHeader().setSectionResizeMode(5, QHeaderView.Stretch)
        self.comparison_table.setMinimumHeight(160)
        comparison_box = QGroupBox("Saved Pretension Comparison")
        comparison_layout = QVBoxLayout(comparison_box)
        comparison_layout.addWidget(self.comparison_table)

        left_column = QWidget()
        left_layout = QVBoxLayout(left_column)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.setSpacing(12)
        left_layout.addWidget(selection_box)
        left_layout.addWidget(parameter_box)
        left_layout.addWidget(actions_box)
        left_layout.addWidget(comparison_box)
        left_layout.addStretch(1)

        right_column = QWidget()
        right_layout = QVBoxLayout(right_column)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.setSpacing(12)
        right_layout.addWidget(feedback_box)
        right_layout.addWidget(progress_box)
        right_layout.addWidget(log_box, 1)

        self.workspace_splitter = QSplitter(Qt.Horizontal)
        self.workspace_splitter.setChildrenCollapsible(False)
        self.workspace_splitter.setHandleWidth(8)
        self.workspace_splitter.addWidget(left_column)
        self.workspace_splitter.addWidget(right_column)
        self.workspace_splitter.setStretchFactor(0, 3)
        self.workspace_splitter.setStretchFactor(1, 4)
        self.workspace_splitter.setSizes([460, 760])
        self._workspace_splitter_layout = ResponsiveSplitterController(
            self.workspace_splitter,
            collapse_below_width=1180,
            horizontal_sizes=[460, 760],
            vertical_sizes=[420, 620],
        )

        content = QWidget()
        content_layout = QVBoxLayout(content)
        content_layout.setContentsMargins(0, 0, 0, 0)
        content_layout.setSpacing(14)
        content_layout.addWidget(self.workspace_splitter)

        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setWidget(content)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(18, 18, 18, 18)
        layout.setSpacing(14)
        layout.addWidget(self.title_label)
        layout.addWidget(self.workflow_hint)
        layout.addWidget(self.scroll_area, 1)
        self._apply_responsive_layout()

    def update(self, state) -> None:
        self.selection_status_label.setText(
            "Connected: select one servo for selected-servo pretension."
            if state.connected
            else "Disconnected: connect OpenRB / DYNAMIXEL before pretensioning."
        )
        applied_values = self._parameter_values_from_state(state)
        if self._last_selected_servo_id != state.selected_servo_id:
            self._parameter_dirty = False
            self._apply_parameter_values(applied_values)
            self._last_selected_servo_id = state.selected_servo_id
        elif not self._parameter_dirty:
            self._apply_parameter_values(applied_values)
        elif self._current_parameter_values() == applied_values:
            self._parameter_dirty = False
            self._apply_parameter_values(applied_values)
        self._applied_parameter_values = applied_values
        self._set_spin_if_not_focused(self.min_offset_spin, state.default_min_offset_ticks)
        self._set_spin_if_not_focused(self.max_offset_spin, state.default_max_offset_ticks)
        self._set_spin_if_not_focused(
            self.threshold_spin,
            max(1, int(state.default_absolute_trigger_current_ma or 1)),
        )

        selected_row = None
        def _rebuild_servo_table() -> None:
            nonlocal selected_row
            self.servo_table.setRowCount(len(state.servo_rows))
            for row, item in enumerate(state.servo_rows):
                self.servo_table.setItem(row, 0, self._item(item["servo_id"], align=Qt.AlignRight))
                self.servo_table.setItem(row, 1, self._item(self._display(item["position"]), align=Qt.AlignRight))
                self.servo_table.setItem(row, 2, self._item(self._display(item["current_ma"]), align=Qt.AlignRight))
                self.servo_table.setItem(row, 3, self._item("Yes" if item["pretension_ready"] else "No", align=Qt.AlignCenter))
                self.servo_table.setItem(row, 4, self._item(item["status"]))
                if item.get("selected"):
                    selected_row = row
            if selected_row is not None:
                self.servo_table.selectRow(selected_row)
            else:
                self.servo_table.clearSelection()
        preserve_scroll_position(self.servo_table, _rebuild_servo_table)

        self.selected_servo_label.setText(self._display(state.selected_servo_id))
        torque_text = "—"
        if state.selected_servo_torque_enabled is True:
            torque_text = "On"
        elif state.selected_servo_torque_enabled is False:
            torque_text = "Auto-enable at start" if state.selected_servo_arming_required else "Off"
        self.torque_label.setText(torque_text)
        telemetry_text = (
            f"age {state.selected_servo_telemetry_age_s:.3f} s | "
            f"fresh {'Yes' if state.selected_servo_telemetry_fresh else 'No'}"
            if state.selected_servo_telemetry_age_s is not None and state.selected_servo_telemetry_fresh is not None
            else "—"
        )
        self.telemetry_label.setText(telemetry_text)
        self.motion_ready_label.setText("Yes" if state.selected_servo_motion_ready else "No")
        self.pretension_ready_label.setText("Yes" if state.selected_servo_pretension_ready else "No")
        self.position_label.setText(self._display(state.selected_servo_position_tick))
        self.current_label.setText(self._display(state.selected_servo_current_ma))
        self.current_validity_label.setText(str(state.selected_servo_current_validity or "unknown"))
        self.filtered_current_label.setText(
            "—" if state.selected_servo_filtered_current_ma is None else f"{state.selected_servo_filtered_current_ma:.1f} mA"
        )
        self.filtered_current_source_label.setText(str(state.selected_servo_filtered_current_source or "none"))
        self.voltage_label.setText(self._display(state.selected_servo_voltage_mv))
        self.temperature_label.setText(self._display(state.selected_servo_temperature_c))
        self.reference_label.setText(self._display(state.selected_servo_untensioned_reference_tick))
        if (
            state.selected_servo_effective_min_target_tick is not None
            and state.selected_servo_effective_max_target_tick is not None
        ):
            self.travel_window_label.setText(
                f"[{state.selected_servo_effective_min_target_tick}, {state.selected_servo_effective_max_target_tick}]"
            )
        else:
            self.travel_window_label.setText("—")
        if state.selected_servo_safe_min_tick is not None and state.selected_servo_safe_max_tick is not None:
            self.bounds_label.setText(f"[{state.selected_servo_safe_min_tick}, {state.selected_servo_safe_max_tick}]")
        else:
            self.bounds_label.setText("—")
        self.direction_label.setText(state.selected_servo_direction_summary)
        self.hardware_error_label.setText(state.selected_servo_hardware_error_text)
        self.block_reason_label.setText(state.selected_servo_block_reason)
        self.saved_summary_label.setText(state.selected_servo_saved_summary)

        self.run_state_label.setText(state.run_state_label)
        self.progress_message_label.setText(state.run_state_message)
        self.baseline_label.setText(state.baseline_samples_label)
        self.start_position_label.setText(self._display(state.start_position_tick))
        self.last_target_label.setText(self._display(state.last_commanded_target_tick))
        self.current_position_label.setText(self._display(state.selected_servo_position_tick))
        effective_absolute = (
            f", abs {self.absolute_trigger_spin.value()} mA"
            if self.absolute_trigger_spin.value() > 0
            else ""
        )
        self.trigger_label.setText(
            "—"
            if state.baseline_current_ma is None
            else (
                f"baseline {state.baseline_current_ma:.1f} mA + delta {self.current_delta_spin.value()} mA"
                f"{effective_absolute}, hard stop {self.hard_current_spin.value()} mA"
            )
        )
        self.steps_label.setText(str(state.steps_taken))
        self.elapsed_label.setText(f"{state.elapsed_s:.2f} s")
        self.final_position_label.setText(self._display(state.final_position_tick))
        self.stop_reason_label.setText(state.stop_reason or "—")
        self.failure_phase_label.setText(state.failure_phase or "—")
        self.failure_primary_label.setText(state.failure_primary_reason or "—")
        self.failure_detail_label.setText(state.failure_detail or "—")
        self._set_plain_text_preserving_view(self.log_text, state.log_text)

        self.servo_table.setEnabled(not state.pretension_running)
        self.refresh_button.setEnabled(not state.pretension_running)
        self.measure_baseline_button.setEnabled(state.can_measure_baseline)
        self.move_reference_button.setEnabled(state.can_move_to_reference)
        self.pretension_button.setEnabled(state.can_start)
        self.stop_button.setEnabled(state.can_stop)
        self.save_button.setEnabled(state.can_save)
        self.save_startup_button.setEnabled(
            bool(state.connected and state.selected_servo_id is not None and not state.pretension_running)
        )
        self.apply_live_button.setEnabled(not state.pretension_running)
        self.save_defaults_button.setEnabled(not state.pretension_running)

        def _rebuild_comparison_table() -> None:
            self.comparison_table.setRowCount(len(state.comparison_rows))
            for row, item in enumerate(state.comparison_rows):
                self.comparison_table.setItem(row, 0, self._item(item["servo_id"], align=Qt.AlignRight))
                self.comparison_table.setItem(row, 1, self._item(item["status"]))
                self.comparison_table.setItem(row, 2, self._item(item["final_position"], align=Qt.AlignRight))
                self.comparison_table.setItem(row, 3, self._item(item["baseline_current"], align=Qt.AlignRight))
                self.comparison_table.setItem(row, 4, self._item(item["trigger_current"], align=Qt.AlignRight))
                self.comparison_table.setItem(
                    row,
                    5,
                    self._item(
                        f"{item['travel_used']} / {item['reason']}"
                        if item["travel_used"] != "—"
                        else item["reason"]
                    ),
                )
        preserve_scroll_position(self.comparison_table, _rebuild_comparison_table)

    def _select_row_servo(self, row: int, _column: int) -> None:
        item = self.servo_table.item(row, 0)
        if item is None:
            return
        try:
            servo_id = int(item.text())
        except (TypeError, ValueError):
            return
        self._safe_call(self.controller.set_selected_servo, int(servo_id))

    def _measure_baseline(self) -> None:
        self._safe_call(
            self.controller.measure_baseline,
            sample_count=int(self.baseline_sample_spin.value()),
            filter_window=int(self.filter_window_spin.value()),
        )

    def _move_to_reference(self) -> None:
        self._safe_call(
            self.controller.move_to_untensioned_reference,
            reference_tick=int(self.untensioned_reference_spin.value()),
        )

    def _start_pretension(self) -> None:
        parameters = self._parameters_from_widgets()
        self._safe_call(self.controller.start_pretension, parameters=parameters)

    def _apply_live_parameters(self) -> None:
        self._safe_call(self.controller.apply_live_parameters, parameters=self._parameters_from_widgets())
        self._parameter_dirty = False

    def _save_parameter_defaults(self) -> None:
        self._safe_call(self.controller.save_pretension_defaults, parameters=self._parameters_from_widgets())
        self._parameter_dirty = False

    def _save_startup_calibration(self) -> None:
        self._safe_call(
            self.controller.save_startup_calibration,
            min_offset_ticks=int(self.min_offset_spin.value()),
            max_offset_ticks=int(self.max_offset_spin.value()),
            threshold_ma=int(self.threshold_spin.value()),
        )

    def _safe_call(self, fn, *args, **kwargs) -> None:
        try:
            fn(*args, **kwargs)
        except Exception as exc:
            if not getattr(self.controller.state, "last_error", None):
                self.controller.state.last_error = str(exc)
            self.controller.state.status_message = str(exc)
        self.update(self.controller.state)

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        self._apply_responsive_layout()

    def _apply_responsive_layout(self) -> None:
        available_width = max(self.width(), self.scroll_area.viewport().width())
        self._workspace_splitter_layout.apply(available_width)

    def _mark_parameter_dirty(self, *_args) -> None:
        if self._updating_parameter_widgets:
            return
        self._parameter_dirty = True

    @staticmethod
    def _copy_text(text: str) -> None:
        QApplication.clipboard().setText(str(text))

    @staticmethod
    def _set_plain_text_preserving_view(widget: QPlainTextEdit, text: str) -> None:
        set_text_document(widget, text, stick_to_bottom_if_at_bottom=False)

    def _apply_parameter_values(self, values: dict[str, object]) -> None:
        self._updating_parameter_widgets = True
        try:
            self.untensioned_reference_spin.setValue(int(values["untensioned_reference_tick"]))
            self._set_combo_value(
                self.start_mode_combo,
                str(values["start_mode"]),
            )
            self.step_ticks_spin.setValue(int(values["step_ticks"]))
            self.settle_time_spin.setValue(float(values["settle_time_s"]))
            self.baseline_sample_spin.setValue(int(values["baseline_sample_count"]))
            self.filter_window_spin.setValue(int(values["current_filter_window"]))
            self.current_delta_spin.setValue(int(values["current_delta_threshold_ma"]))
            self.absolute_trigger_spin.setValue(int(values["absolute_trigger_current_ma"] or 0))
            self.hard_current_spin.setValue(int(values["hard_current_stop_ma"]))
            self.max_travel_spin.setValue(int(values["max_travel_ticks"]))
            self.timeout_spin.setValue(float(values["timeout_s"]))
        finally:
            self._updating_parameter_widgets = False

    def _current_parameter_values(self) -> dict[str, object]:
        return {
            "untensioned_reference_tick": int(self.untensioned_reference_spin.value()),
            "start_mode": str(self.start_mode_combo.currentData() or "current_position"),
            "step_ticks": int(self.step_ticks_spin.value()),
            "settle_time_s": float(self.settle_time_spin.value()),
            "baseline_sample_count": int(self.baseline_sample_spin.value()),
            "current_filter_window": int(self.filter_window_spin.value()),
            "current_delta_threshold_ma": int(self.current_delta_spin.value()),
            "absolute_trigger_current_ma": (None if int(self.absolute_trigger_spin.value()) <= 0 else int(self.absolute_trigger_spin.value())),
            "hard_current_stop_ma": int(self.hard_current_spin.value()),
            "max_travel_ticks": int(self.max_travel_spin.value()),
            "timeout_s": float(self.timeout_spin.value()),
        }

    @staticmethod
    def _parameter_values_from_state(state) -> dict[str, object]:
        return {
            "untensioned_reference_tick": int(state.default_untensioned_reference_tick),
            "start_mode": str(state.default_start_mode or "current_position"),
            "step_ticks": int(state.default_step_ticks),
            "settle_time_s": float(state.default_settle_time_s),
            "baseline_sample_count": int(state.default_baseline_sample_count),
            "current_filter_window": int(state.default_filter_window),
            "current_delta_threshold_ma": int(state.default_current_delta_threshold_ma),
            "absolute_trigger_current_ma": (
                None
                if state.default_absolute_trigger_current_ma in (None, 0)
                else int(state.default_absolute_trigger_current_ma)
            ),
            "hard_current_stop_ma": int(state.default_hard_current_stop_ma),
            "max_travel_ticks": int(state.default_max_travel_ticks),
            "timeout_s": float(state.default_timeout_s),
        }

    def _parameters_from_widgets(self) -> PretensionParameters:
        absolute_trigger_value = int(self.absolute_trigger_spin.value())
        return PretensionParameters(
            untensioned_reference_tick=int(self.untensioned_reference_spin.value()),
            start_mode=str(self.start_mode_combo.currentData() or "current_position"),
            step_ticks=int(self.step_ticks_spin.value()),
            settle_time_s=float(self.settle_time_spin.value()),
            baseline_sample_count=int(self.baseline_sample_spin.value()),
            current_filter_window=int(self.filter_window_spin.value()),
            current_delta_threshold_ma=int(self.current_delta_spin.value()),
            absolute_trigger_current_ma=(None if absolute_trigger_value <= 0 else absolute_trigger_value),
            hard_current_stop_ma=int(self.hard_current_spin.value()),
            max_travel_ticks=int(self.max_travel_spin.value()),
            timeout_s=float(self.timeout_spin.value()),
        )

    @staticmethod
    def _set_combo_value(combo: QComboBox, value: str) -> None:
        target = str(value or "").strip().lower()
        index = combo.findData(target)
        if index < 0 and combo.count() > 0:
            index = 0
        if index >= 0 and combo.currentIndex() != index:
            combo.blockSignals(True)
            combo.setCurrentIndex(index)
            combo.blockSignals(False)

    @staticmethod
    def _set_spin_if_not_focused(spin: QSpinBox, value: int) -> None:
        if spin.hasFocus():
            return
        spin.blockSignals(True)
        spin.setValue(int(value))
        spin.blockSignals(False)

    @staticmethod
    def _set_double_if_not_focused(spin: QDoubleSpinBox, value: float) -> None:
        if spin.hasFocus():
            return
        spin.blockSignals(True)
        spin.setValue(float(value))
        spin.blockSignals(False)

    @staticmethod
    def _display(value) -> str:
        return "—" if value is None else str(value)

    @staticmethod
    def _item(value, *, align: int = Qt.AlignLeft | Qt.AlignVCenter) -> QTableWidgetItem:
        item = QTableWidgetItem(str(value))
        item.setTextAlignment(align | Qt.AlignVCenter)
        return item

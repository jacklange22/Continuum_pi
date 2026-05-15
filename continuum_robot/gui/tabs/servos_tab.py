"""Servos tab widget."""

from __future__ import annotations

from typing import Callable

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QAbstractItemView,
    QFormLayout,
    QGroupBox,
    QHeaderView,
    QHBoxLayout,
    QLabel,
    QPlainTextEdit,
    QPushButton,
    QScrollArea,
    QSpinBox,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from continuum_robot.gui.controllers.servos_controller import ServosViewState
from continuum_robot.gui.theme import grouped_workspace_stylesheet
from continuum_robot.gui.view_utils import preserve_scroll_position


class ServosTab(QWidget):
    """Live servo telemetry, jog controls, and pretension (manual + algorithmic)."""

    def __init__(
        self,
        controller,
        parent=None,
        *,
        apply_runtime_parameters: Callable[..., None] | None = None,
    ) -> None:
        super().__init__(parent)
        self.controller = controller
        # apply_runtime_parameters retained for API compatibility; jog-tick settings
        # now live on the System tab, so this callback is unused here.
        self._apply_runtime_parameters = apply_runtime_parameters

        self.setObjectName("servoWorkspace")
        self.setStyleSheet(
            grouped_workspace_stylesheet(
                object_name="servoWorkspace",
                input_selectors=["QSpinBox", "QPlainTextEdit", "QTableWidget"],
            )
        )

        self.title_label = QLabel("Servos")
        self.title_label.setProperty("role", "title")

        self.status_label = QLabel("Disconnected")
        self.status_label.setProperty("role", "status")
        self.status_label.setWordWrap(True)
        self.blocker_label = QLabel("")
        self.blocker_label.setProperty("role", "hint")
        self.blocker_label.setWordWrap(True)
        self.blocker_label.setVisible(False)

        status_box = QGroupBox("Ready State")
        status_layout = QVBoxLayout(status_box)
        status_layout.addWidget(self.status_label)
        status_layout.addWidget(self.blocker_label)

        self.telemetry_table = QTableWidget(0, 9)
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
            ]
        )
        self.telemetry_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.telemetry_table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.telemetry_table.setSelectionMode(QAbstractItemView.SingleSelection)
        self.telemetry_table.verticalHeader().setVisible(False)
        self.telemetry_table.cellClicked.connect(self._select_servo_from_row)
        for column in range(0, 9):
            self.telemetry_table.horizontalHeader().setSectionResizeMode(column, QHeaderView.ResizeToContents)
        self.telemetry_table.horizontalHeader().setStretchLastSection(True)
        self.telemetry_table.setMinimumHeight(190)

        telemetry_box = QGroupBox("Live Telemetry")
        telemetry_layout = QVBoxLayout(telemetry_box)
        telemetry_layout.addWidget(self.telemetry_table)

        self.jog_label = QLabel("Select a servo to jog")
        self.jog_label.setProperty("role", "hint")
        self.fine_minus_button = QPushButton("Loosen Fine")
        self.fine_plus_button = QPushButton("Tighten Fine")
        self.coarse_minus_button = QPushButton("Loosen Coarse")
        self.coarse_plus_button = QPushButton("Tighten Coarse")
        self.fine_minus_button.clicked.connect(lambda: self._jog("fine", -1))
        self.fine_plus_button.clicked.connect(lambda: self._jog("fine", 1))
        self.coarse_minus_button.clicked.connect(lambda: self._jog("coarse", -1))
        self.coarse_plus_button.clicked.connect(lambda: self._jog("coarse", 1))

        jog_box = QGroupBox("Jog Selected Servo")
        jog_layout = QVBoxLayout(jog_box)
        jog_layout.addWidget(self.jog_label)
        jog_buttons = QHBoxLayout()
        jog_buttons.setSpacing(10)
        jog_buttons.addWidget(self.fine_minus_button)
        jog_buttons.addWidget(self.fine_plus_button)
        jog_buttons.addWidget(self.coarse_minus_button)
        jog_buttons.addWidget(self.coarse_plus_button)
        jog_layout.addLayout(jog_buttons)

        self.manual_pretension_summary_label = QLabel("No accepted pretension source.")
        self.manual_pretension_summary_label.setWordWrap(True)
        self.manual_pretension_note_label = QLabel("No manual pretension note saved.")
        self.manual_pretension_note_label.setWordWrap(True)
        self.manual_pretension_note_edit = QPlainTextEdit()
        self.manual_pretension_note_edit.setPlaceholderText("Optional operator note for this manual startup state.")
        self.manual_pretension_note_edit.setMaximumHeight(64)
        self.capture_manual_pretension_button = QPushButton("Capture Current State")
        self.capture_manual_pretension_button.setProperty("role", "primary")
        self.capture_manual_pretension_button.clicked.connect(self._capture_manual_pretension)
        self.accept_manual_pretension_button = QPushButton("Accept")
        self.accept_manual_pretension_button.clicked.connect(
            lambda: self._safe_call(self.controller.accept_manual_pretension)
        )
        self.clear_manual_pretension_button = QPushButton("Clear")
        self.clear_manual_pretension_button.setProperty("variant", "ghost")
        self.clear_manual_pretension_button.clicked.connect(
            lambda: self._safe_call(self.controller.clear_manual_pretension)
        )

        self.manual_pretension_box = QGroupBox("Manual Pretension")
        manual_layout = QFormLayout(self.manual_pretension_box)
        manual_layout.addRow("Active source", self.manual_pretension_summary_label)
        manual_layout.addRow("Saved note", self.manual_pretension_note_label)
        manual_layout.addRow("Operator note", self.manual_pretension_note_edit)
        manual_buttons = QHBoxLayout()
        manual_buttons.setSpacing(10)
        manual_buttons.addWidget(self.capture_manual_pretension_button)
        manual_buttons.addWidget(self.accept_manual_pretension_button)
        manual_buttons.addWidget(self.clear_manual_pretension_button)
        manual_buttons.addStretch(1)
        manual_buttons_widget = QWidget()
        manual_buttons_widget.setLayout(manual_buttons)
        manual_layout.addRow(manual_buttons_widget)

        self.pretension_servo_spin = QSpinBox()
        self.pretension_servo_spin.setRange(1, 252)
        self.pretension_servo_spin.valueChanged.connect(self._sync_servo_selection)
        self.pretension_threshold_spin = QSpinBox()
        self.pretension_threshold_spin.setRange(1, 5000)
        self.pretension_threshold_spin.setValue(int(self.controller.state.default_pretension_threshold_ma))
        self.start_pretension_button = QPushButton("Start")
        self.start_pretension_button.setProperty("role", "primary")
        self.cancel_pretension_button = QPushButton("Cancel")
        self.cancel_pretension_button.setProperty("role", "danger")
        self.accept_pretension_button = QPushButton("Accept")
        self.retry_pretension_button = QPushButton("Retry")
        self.start_pretension_button.clicked.connect(self._start_pretension)
        self.cancel_pretension_button.clicked.connect(
            lambda: self._safe_call(self.controller.cancel_pretension)
        )
        self.accept_pretension_button.clicked.connect(self._accept_pretension)
        self.retry_pretension_button.clicked.connect(self._start_pretension)
        self.pretension_hint = QLabel(
            "Pretension uses present current as a practical threshold signal, not true tendon tension."
        )
        self.pretension_hint.setProperty("role", "hint")
        self.pretension_hint.setWordWrap(True)

        self.pretension_box = QGroupBox("Algorithmic Pretension")
        pretension_layout = QFormLayout(self.pretension_box)
        pretension_layout.addRow("Servo", self.pretension_servo_spin)
        pretension_layout.addRow("Threshold (mA)", self.pretension_threshold_spin)
        pretension_buttons = QHBoxLayout()
        pretension_buttons.setSpacing(10)
        pretension_buttons.addWidget(self.start_pretension_button)
        pretension_buttons.addWidget(self.cancel_pretension_button)
        pretension_buttons.addWidget(self.retry_pretension_button)
        pretension_buttons.addWidget(self.accept_pretension_button)
        pretension_buttons.addStretch(1)
        pretension_buttons_widget = QWidget()
        pretension_buttons_widget.setLayout(pretension_buttons)
        pretension_layout.addRow(pretension_buttons_widget)
        pretension_layout.addRow(self.pretension_hint)

        content_widget = QWidget()
        content_layout = QVBoxLayout(content_widget)
        content_layout.setContentsMargins(0, 0, 0, 0)
        content_layout.setSpacing(14)
        content_layout.addWidget(status_box)
        content_layout.addWidget(telemetry_box)
        content_layout.addWidget(jog_box)
        content_layout.addWidget(self.manual_pretension_box)
        content_layout.addWidget(self.pretension_box)
        content_layout.addStretch(1)

        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setWidget(content_widget)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(18, 18, 18, 18)
        layout.setSpacing(14)
        layout.addWidget(self.title_label)
        layout.addWidget(self.scroll_area, 1)

    def update(self, state: ServosViewState) -> None:
        selected_servo_id = (
            state.selected_servo_id
            if state.selected_servo_id is not None
            else (state.servo_ids[0] if state.servo_ids else None)
        )

        self.status_label.setText(self._format_status(state))
        blocker_text = self._format_blocker(state)
        self.blocker_label.setText(blocker_text)
        self.blocker_label.setVisible(bool(blocker_text))

        sorted_servo_ids = sorted(state.telemetry)

        def _rebuild_telemetry_table() -> None:
            self.telemetry_table.setRowCount(len(state.telemetry))
            for row, servo_id in enumerate(sorted_servo_ids):
                item = state.telemetry[servo_id]
                self.telemetry_table.setItem(row, 0, self._text_item(servo_id, align=Qt.AlignRight))
                self.telemetry_table.setItem(
                    row, 1, self._text_item(item.get("telemetry_status", "Unknown"), align=Qt.AlignCenter)
                )
                self.telemetry_table.setItem(row, 2, self._text_item(item.get("torque_label", "—"), align=Qt.AlignCenter))
                self.telemetry_table.setItem(row, 3, self._text_item(self._display_value(item.get("position")), align=Qt.AlignRight))
                self.telemetry_table.setItem(row, 4, self._text_item(self._display_value(item.get("current_ma")), align=Qt.AlignRight))
                self.telemetry_table.setItem(row, 5, self._text_item(self._display_value(item.get("voltage_mv")), align=Qt.AlignRight))
                self.telemetry_table.setItem(row, 6, self._text_item(self._display_value(item.get("temperature_c")), align=Qt.AlignRight))
                self.telemetry_table.setItem(row, 7, self._text_item(item.get("hardware_error_text", "—"), align=Qt.AlignCenter))
                self.telemetry_table.setItem(row, 8, self._text_item(self._age_text(item.get("telemetry_age_s")), align=Qt.AlignRight))
            if selected_servo_id in sorted_servo_ids:
                self.telemetry_table.selectRow(sorted_servo_ids.index(selected_servo_id))
            else:
                self.telemetry_table.clearSelection()

        preserve_scroll_position(self.telemetry_table, _rebuild_telemetry_table)

        motion_ready = (
            state.connected
            and bool(state.servo_ids)
            and state.selected_servo_motion_ready
            and selected_servo_id is not None
        )
        self.jog_label.setText(
            f"Jog: Servo {selected_servo_id}" if selected_servo_id is not None else "Select a servo to jog"
        )
        self.fine_minus_button.setEnabled(motion_ready)
        self.fine_plus_button.setEnabled(motion_ready)
        self.coarse_minus_button.setEnabled(motion_ready)
        self.coarse_plus_button.setEnabled(motion_ready)

        self.manual_pretension_summary_label.setText(state.pretension_source_summary)
        self.manual_pretension_note_label.setText(
            state.pretension_source_note or "No manual pretension note saved."
        )
        manual_mode_available = (not state.single_servo_mode) and len(state.expected_servo_ids) in {4, 8}
        any_servo = bool(state.servo_ids)
        self.manual_pretension_box.setVisible(manual_mode_available)
        self.capture_manual_pretension_button.setEnabled(
            state.connected and manual_mode_available and any_servo
        )
        self.accept_manual_pretension_button.setEnabled(
            manual_mode_available and state.manual_pretension_can_accept
        )
        self.clear_manual_pretension_button.setEnabled(
            manual_mode_available and state.manual_pretension_can_clear
        )

        self.pretension_box.setVisible(True)
        if selected_servo_id is not None and not self.pretension_servo_spin.hasFocus():
            self.pretension_servo_spin.blockSignals(True)
            self.pretension_servo_spin.setValue(int(selected_servo_id))
            self.pretension_servo_spin.blockSignals(False)
        max_servo_id = max([252, *state.servo_ids]) if state.servo_ids else 252
        self.pretension_servo_spin.setMaximum(max_servo_id)
        if not self.pretension_threshold_spin.hasFocus() and self.pretension_threshold_spin.value() <= 1:
            self.pretension_threshold_spin.setValue(max(1, state.default_pretension_threshold_ma))
        self.start_pretension_button.setEnabled(motion_ready and not state.pretension_running)
        self.cancel_pretension_button.setEnabled(state.pretension_running)
        self.retry_pretension_button.setEnabled(motion_ready and not state.pretension_running)
        self.accept_pretension_button.setEnabled(state.pretension_result_can_accept)

    @staticmethod
    def _format_status(state: ServosViewState) -> str:
        if not state.connected:
            return "Bus disconnected — connect OpenRB on the System tab."
        detected = sorted(int(sid) for sid in state.detected_servo_ids)
        expected = sorted(int(sid) for sid in state.expected_servo_ids)
        detected_text = ", ".join(str(sid) for sid in detected) or "none"
        if expected and detected == expected:
            servo_summary = f"{len(detected)} of {len(expected)} servos: {detected_text}"
        elif expected:
            servo_summary = (
                f"{len(detected)} of {len(expected)} servos detected (expected {', '.join(str(sid) for sid in expected)})"
            )
        else:
            servo_summary = f"Detected: {detected_text}"
        calibration_state = (
            "ready"
            if state.calibration_exists and state.calibration_compatible
            else ("review needed" if state.calibration_exists else "missing")
        )
        pretension_type = (state.pretension_source_type or "none").lower()
        if pretension_type == "none":
            pretension = "none"
        elif "pending" in pretension_type:
            pretension = "pending accept"
        else:
            pretension = "accepted"
        return (
            f"Bus connected  ·  {servo_summary}  ·  Calibration: {calibration_state}  ·  "
            f"Pretension: {pretension}"
        )

    @staticmethod
    def _format_blocker(state: ServosViewState) -> str:
        parts: list[str] = []
        if state.missing_servo_ids:
            parts.append(
                "Missing servos: " + ", ".join(str(sid) for sid in sorted(state.missing_servo_ids))
            )
        if state.unexpected_servo_ids:
            parts.append(
                "Unexpected servos: " + ", ".join(str(sid) for sid in sorted(state.unexpected_servo_ids))
            )
        if state.blocking_reasons:
            parts.append(state.blocking_reasons[0])
        if state.last_error:
            parts.append(f"Error: {state.last_error}")
        return "  ·  ".join(parts)

    def _jog(self, mode: str, direction: int) -> None:
        servo_id = self.controller.state.selected_servo_id
        if servo_id is None:
            return
        action = self.controller.fine_jog if mode == "fine" else self.controller.coarse_jog
        self._safe_call(action, int(servo_id), int(direction))

    def _start_pretension(self) -> None:
        self._safe_call(
            self.controller.start_pretension,
            int(self.pretension_servo_spin.value()),
            int(self.pretension_threshold_spin.value()),
        )

    def _accept_pretension(self) -> None:
        self._safe_call(self.controller.accept_pretension_result, int(self.pretension_servo_spin.value()))

    def _capture_manual_pretension(self) -> None:
        self._safe_call(
            self.controller.capture_manual_pretension,
            self.manual_pretension_note_edit.toPlainText().strip(),
        )

    def _sync_servo_selection(self, servo_id: int) -> None:
        set_selected = getattr(self.controller, "set_selected_servo", None)
        if callable(set_selected):
            set_selected(int(servo_id))
        else:
            self.controller.state.selected_servo_id = int(servo_id)
        self.update(self.controller.state)

    def _select_servo_from_row(self, row: int, _column: int) -> None:
        item = self.telemetry_table.item(row, 0)
        if item is None:
            return
        try:
            servo_id = int(item.text())
        except (TypeError, ValueError):
            return
        self._sync_servo_selection(int(servo_id))

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
        item.setTextAlignment(align | Qt.AlignVCenter)
        return item

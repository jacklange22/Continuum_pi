"""Tracking tab widget."""

from __future__ import annotations

from datetime import datetime

from PySide6.QtCore import QTimer
from PySide6.QtWidgets import (
    QComboBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QTableWidget,
    QTableWidgetItem,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from continuum_robot.gui.controllers.tracker_mvp_controller import TrackerMvpViewState
from continuum_robot.gui.controllers.tracking_controller import TrackingViewState
from continuum_robot.gui.theme import COLORS, grouped_workspace_stylesheet
from continuum_robot.gui.view_utils import set_text_document
from continuum_robot.gui.widgets.tool_plot_widget import ToolPlotWidget


class TrackingTab(QWidget):
    """Canonical tracker and pivot workspace."""

    def __init__(self, live_controller, *, workflow_controller, parent=None) -> None:
        super().__init__(parent)
        self.live_controller = live_controller
        self.workflow_controller = workflow_controller
        self.setObjectName("trackingWorkspace")
        self.setStyleSheet(
            grouped_workspace_stylesheet(
                object_name="trackingWorkspace",
                input_selectors=["QComboBox", "QTextEdit", "QTableWidget"],
            )
        )

        self.title_label = QLabel("Tracking & Pivot")
        self.title_label.setProperty("role", "title")
        self.workflow_hint = QLabel(
            "Connect Aurora, validate tracker health, confirm 0A and 0B visibility and rigid-valid transforms, "
            "then collect, solve, review, and accept the staged 0B pivot calibration before moving to Registration."
        )
        self.workflow_hint.setProperty("role", "hint")
        self.workflow_hint.setWordWrap(True)
        self.status_label = QLabel("Connect tracker to begin.")
        self.status_label.setProperty("role", "status")

        self.tracker_port_combo = QComboBox()
        self.tracker_port_combo.setEditable(True)
        self.tracker_port_combo.currentIndexChanged.connect(self._sync_tracker_port)
        self.tracker_port_combo.editTextChanged.connect(self._sync_tracker_port)
        self.connection_label = QLabel("disconnected")
        self.verdict_label = QLabel("not validated")
        self.tools_label = QLabel("none")
        self.validation_report_label = QLabel("none")
        self.validation_report_label.setWordWrap(True)
        self.handoff_label = QLabel("Registration will unlock after tracker validation and accepted tip review.")
        self.handoff_label.setWordWrap(True)

        self.connect_button = QPushButton("Connect Tracker")
        self.connect_button.setProperty("role", "primary")
        self.disconnect_button = QPushButton("Disconnect Tracker")
        self.disconnect_button.setProperty("variant", "ghost")
        self.validate_button = QPushButton("Run Tracker Diagnostic")
        self.validate_button.setProperty("variant", "ghost")
        self.rescan_button = QPushButton("Rescan Ports")
        self.rescan_button.setProperty("variant", "ghost")
        self.connect_button.clicked.connect(self._connect_tracker)
        self.disconnect_button.clicked.connect(self._disconnect_tracker)
        self.validate_button.clicked.connect(self._validate_tracker)
        self.rescan_button.clicked.connect(self._rescan_ports)

        readiness_box = QGroupBox("Tracker Readiness")
        readiness_layout = QVBoxLayout(readiness_box)
        readiness_form = QFormLayout()
        readiness_form.addRow("Tracker port", self.tracker_port_combo)
        readiness_form.addRow("Connection", self.connection_label)
        readiness_form.addRow("Verdict", self.verdict_label)
        readiness_form.addRow("Tool visibility", self.tools_label)
        readiness_form.addRow("Validation artifact", self.validation_report_label)
        readiness_form.addRow("Registration handoff", self.handoff_label)
        readiness_layout.addLayout(readiness_form)
        readiness_buttons = QHBoxLayout()
        readiness_buttons.addWidget(self.connect_button)
        readiness_buttons.addWidget(self.disconnect_button)
        readiness_buttons.addWidget(self.validate_button)
        readiness_buttons.addWidget(self.rescan_button)
        readiness_buttons.addStretch(1)
        readiness_layout.addLayout(readiness_buttons)

        self.tools_table = QTableWidget(0, 5)
        self.tools_table.setHorizontalHeaderLabels(["Tool", "Valid", "Status", "Translation mm", "Quality"])
        self.tools_table.verticalHeader().setVisible(False)
        self.tools_table.setMinimumHeight(150)
        self.tools_table.setMaximumHeight(210)
        self.plot_widget = ToolPlotWidget()
        self.plot_widget.setMinimumHeight(420)
        self.plot_widget.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        live_box = QGroupBox("Live Tools")
        live_layout = QVBoxLayout(live_box)
        live_layout.addWidget(self.plot_widget, 5)
        live_layout.addWidget(self.tools_table, 2)

        self.pivot_status_label = QLabel("idle")
        self.pivot_motion_label = QLabel("—")
        self.pivot_last_run_label = QLabel("No pivot run recorded yet.")
        self.pivot_last_run_label.setWordWrap(True)
        self.pivot_tip_path_label = QLabel("none")
        self.pivot_tip_path_label.setWordWrap(True)

        self._pivot_start_button = QPushButton("Start 0B Collection")
        self._pivot_start_button.setProperty("role", "primary")
        self._pivot_stop_button = QPushButton("Stop Collection")
        self._pivot_solve_button = QPushButton("Solve Pivot Calibration")
        self._pivot_accept_button = QPushButton("Accept Tip File")
        self._pivot_reset_button = QPushButton("Reset Pivot Review")
        self._pivot_start_button.clicked.connect(self._start_pivot)
        self._pivot_stop_button.clicked.connect(self._stop_pivot)
        self._pivot_solve_button.clicked.connect(self._solve_pivot)
        self._pivot_accept_button.clicked.connect(self._accept_pivot)
        self._pivot_reset_button.clicked.connect(self._reset_pivot)

        pivot_box = QGroupBox("0B Pivot Calibration")
        pivot_layout = QVBoxLayout(pivot_box)
        pivot_form = QFormLayout()
        pivot_form.addRow("Status", self.pivot_status_label)
        pivot_form.addRow("Motion range", self.pivot_motion_label)
        pivot_form.addRow("Last run", self.pivot_last_run_label)
        pivot_form.addRow("Current tip transform", self.pivot_tip_path_label)
        pivot_layout.addLayout(pivot_form)
        pivot_buttons = QHBoxLayout()
        pivot_buttons.addWidget(self._pivot_start_button)
        pivot_buttons.addWidget(self._pivot_stop_button)
        pivot_buttons.addWidget(self._pivot_solve_button)
        pivot_buttons.addWidget(self._pivot_accept_button)
        pivot_buttons.addWidget(self._pivot_reset_button)
        pivot_layout.addLayout(pivot_buttons)

        self.details_text = QTextEdit()
        self.details_text.setReadOnly(True)
        self.details_text.setMinimumHeight(92)
        self.details_text.setMaximumHeight(130)
        details_box = QGroupBox("Details")
        details_layout = QVBoxLayout(details_box)
        details_layout.addWidget(self.details_text)

        content_widget = QWidget()
        content_layout = QVBoxLayout(content_widget)
        content_layout.setContentsMargins(0, 0, 0, 0)
        content_layout.setSpacing(14)
        content_layout.addWidget(readiness_box)
        content_layout.addWidget(live_box)
        content_layout.addWidget(pivot_box)
        content_layout.addWidget(details_box)

        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setWidget(content_widget)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(18, 18, 18, 18)
        layout.setSpacing(14)
        layout.addWidget(self.title_label)
        layout.addWidget(self.workflow_hint)
        layout.addWidget(self.status_label)
        layout.addWidget(self.scroll_area, 1)

    def update(self, workflow_state: TrackerMvpViewState, live_state: TrackingViewState) -> None:
        self.status_label.setText(workflow_state.status_message)
        self.connection_label.setText(
            f"{workflow_state.canonical_state} ({workflow_state.connection_state}) | "
            f"backend={workflow_state.backend_name or workflow_state.backend_identity}"
        )
        self.verdict_label.setText(self._format_verdict(workflow_state))
        self.tools_label.setText(
            f"0A={'tracked' if workflow_state.tool_0a_visible else workflow_state.tool_0a_status}, "
            f"0B={'tracked' if workflow_state.tool_0b_visible else workflow_state.tool_0b_status}"
        )
        self.validation_report_label.setText(workflow_state.validation_report_path or "none")
        if workflow_state.registration_blockers:
            self.handoff_label.setText("Registration blocked: " + " ".join(workflow_state.registration_blockers))
        else:
            self.handoff_label.setText(
                "Registration is ready. Move to the Registration tab to capture points, solve, review, and save."
            )

        self.tools_table.setRowCount(len(live_state.tools))
        for row, tool_id in enumerate(sorted(live_state.tools)):
            tool = live_state.tools[tool_id]
            translation = (
                str(tuple(round(v, 2) for v in tool["translation_mm"]))
                if tool["translation_mm"] is not None
                else "unavailable"
            )
            valid_text = str(tool["valid"]) if tool["validity_known"] else "unknown"
            status = f'{tool["tracking_state"]}: {tool["status"]}'
            self.tools_table.setItem(row, 0, QTableWidgetItem(tool_id))
            self.tools_table.setItem(row, 1, QTableWidgetItem(valid_text))
            self.tools_table.setItem(row, 2, QTableWidgetItem(status))
            self.tools_table.setItem(row, 3, QTableWidgetItem(translation))
            self.tools_table.setItem(row, 4, QTableWidgetItem(str(tool["quality"])))
        self.plot_widget.set_tracking_state(live_state)

        self.pivot_status_label.setText(self._format_pivot_status(workflow_state))
        self.pivot_status_label.setStyleSheet(self._pivot_status_stylesheet(workflow_state))
        self.pivot_motion_label.setText(self._format_pivot_motion(workflow_state))
        self.pivot_last_run_label.setText(self._format_pivot_last_run(workflow_state))
        self.pivot_tip_path_label.setText(self._format_tip_in_use(workflow_state))

        details_lines = [
            workflow_state.validation_summary,
            f"report={workflow_state.validation_report_path or 'none'}",
            f"tip_status={workflow_state.measurement_point_message}",
        ]
        if live_state.warning_messages:
            details_lines.append(f"warnings={live_state.warning_messages}")
        if live_state.last_error:
            details_lines.append(f"last_error={live_state.last_error}")
        self._set_text_preserving_view(self.details_text, "\n".join(details_lines))

        self._set_combo_items(
            self.tracker_port_combo,
            workflow_state.available_ports,
            workflow_state.tracker_port,
        )
        tracker_connected = bool(
            workflow_state.tracker_connected or workflow_state.connection_state in {"connecting", "starting"}
        )
        self.connect_button.setEnabled(not tracker_connected)
        self.disconnect_button.setEnabled(tracker_connected)
        self.validate_button.setEnabled(bool(workflow_state.tracker_connected))
        self.rescan_button.setEnabled(True)
        self._pivot_start_button.setEnabled(workflow_state.pivot_can_start)
        self._pivot_stop_button.setEnabled(workflow_state.pivot_can_stop)
        self._pivot_solve_button.setEnabled(workflow_state.pivot_can_solve)
        self._pivot_accept_button.setEnabled(workflow_state.pivot_can_accept)
        self._pivot_reset_button.setEnabled(
            workflow_state.pivot_collection_active
            or workflow_state.pivot_pending_accept
            or workflow_state.pivot_live_sample_count > 0
            or bool(workflow_state.pivot_run_path)
        )

    @staticmethod
    def _set_text_preserving_view(widget: QTextEdit, text: str) -> None:
        set_text_document(widget, text, stick_to_bottom_if_at_bottom=True)

    @staticmethod
    def _format_verdict(ws: TrackerMvpViewState) -> str:
        verdict = ws.tracker_verdict or ""
        if verdict == "passed":
            return "ready"
        if verdict == "operational_with_warning":
            return "warning"
        if verdict in {"failed"} and ws.tracker_connected:
            return "validation failed"
        if not ws.tracker_connected:
            return "blocked"
        return "connected (not validated)"

    @staticmethod
    def _format_pivot_status(ws: TrackerMvpViewState) -> str:
        if ws.pivot_collection_active:
            return f"Collecting · {ws.pivot_live_sample_count} samples"
        rmse = ws.pivot_rmse_mm
        rmse_text = f" · RMSE {rmse:.3f} mm" if rmse is not None else ""
        if ws.pivot_status == "solve_failed":
            return f"Solve failed{rmse_text} — see Details"
        if ws.pivot_pending_accept:
            return f"Solved{rmse_text} · Accept Tip File to save"
        if ws.pivot_status == "accepted":
            return f"Accepted{rmse_text}"
        if ws.pivot_live_sample_count > 0:
            return f"Collection stopped · {ws.pivot_live_sample_count} samples · Click Solve"
        return "Idle"

    @staticmethod
    def _pivot_status_stylesheet(ws: TrackerMvpViewState) -> str:
        base = "padding: 6px 10px; border-radius: 8px; font-weight: 700;"
        if ws.pivot_collection_active:
            return f"{base} background: {COLORS.info_bg}; color: {COLORS.info_fg};"
        if ws.pivot_status == "solve_failed":
            return f"{base} background: {COLORS.error_bg}; color: {COLORS.error_fg};"
        if ws.pivot_pending_accept or ws.pivot_status == "accepted":
            return f"{base} background: {COLORS.success_bg}; color: {COLORS.success_fg};"
        return f"{base} background: {COLORS.status_bg}; color: {COLORS.status_fg};"

    @staticmethod
    def _format_pivot_motion(ws: TrackerMvpViewState) -> str:
        if ws.pivot_motion_span_deg is None:
            if ws.pivot_collection_active:
                return "move 0B through varied orientations"
            return "—"
        deg = f"{ws.pivot_motion_span_deg:.1f}°"
        return f"{deg} · wide enough" if ws.pivot_motion_ready else f"{deg} · collect wider motion"

    def _format_pivot_last_run(self, ws: TrackerMvpViewState) -> str:
        iso = ws.pivot_last_run_at_iso
        rmse = ws.pivot_rmse_mm
        if iso:
            timestamp = self._format_iso_local(iso)
            if rmse is not None:
                return f"{timestamp}  ·  RMSE = {rmse:.3f} mm"
            return timestamp
        if rmse is not None:
            return f"RMSE = {rmse:.3f} mm"
        return "No pivot run recorded yet."

    def _format_tip_in_use(self, ws: TrackerMvpViewState) -> str:
        path = ws.pivot_tip_path or "none"
        if ws.measurement_point_file_mtime_iso:
            return f"{path}  ·  from {self._format_iso_local(ws.measurement_point_file_mtime_iso)}"
        return path

    @staticmethod
    def _format_iso_local(iso_str: str) -> str:
        try:
            return datetime.fromisoformat(iso_str).astimezone().strftime("%Y-%m-%d %H:%M:%S")
        except (TypeError, ValueError):
            return iso_str

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

    def _sync_tracker_port(self, _value=None) -> None:
        self.workflow_controller.set_tracker_port(self._selected_port(self.tracker_port_combo))
        self.live_controller.set_device_path(self.workflow_controller.state.tracker_port)

    def _rescan_ports(self) -> None:
        self.workflow_controller.rescan_ports()

    AUTO_VALIDATE_INITIAL_DELAY_MS = 1200
    AUTO_VALIDATE_RETRY_DELAY_MS = 2500
    AUTO_VALIDATE_MAX_ATTEMPTS = 3

    def _connect_tracker(self) -> None:
        self._safe_call(self.workflow_controller.connect_tracker)
        self._auto_validate_attempts = 0
        QTimer.singleShot(self.AUTO_VALIDATE_INITIAL_DELAY_MS, self._run_auto_validate)

    def _run_auto_validate(self) -> None:
        state = self.workflow_controller.state
        if not state.tracker_connected:
            return
        if self.workflow_controller._validation_operational:
            return
        self._auto_validate_attempts += 1
        try:
            self.workflow_controller.validate_tracker()
        except Exception:
            pass
        self.update(self.workflow_controller.refresh(), self.live_controller.refresh())
        if (
            not self.workflow_controller._validation_operational
            and self._auto_validate_attempts < self.AUTO_VALIDATE_MAX_ATTEMPTS
        ):
            QTimer.singleShot(self.AUTO_VALIDATE_RETRY_DELAY_MS, self._run_auto_validate)

    def _disconnect_tracker(self) -> None:
        self._safe_call(self.workflow_controller.disconnect_tracker)

    def _validate_tracker(self) -> None:
        self._safe_call(self.workflow_controller.validate_tracker)

    def _start_pivot(self) -> None:
        self._safe_call(self.workflow_controller.start_pivot_collection)

    def _stop_pivot(self) -> None:
        self._safe_call(self.workflow_controller.stop_pivot_collection)

    def _solve_pivot(self) -> None:
        self._safe_call(self.workflow_controller.solve_pivot_collection)

    def _accept_pivot(self) -> None:
        self._safe_call(self.workflow_controller.accept_pivot_tip_file)

    def _reset_pivot(self) -> None:
        self._safe_call(self.workflow_controller.reset_pivot_workflow)

    def _safe_call(self, fn) -> None:
        try:
            fn()
        except Exception:
            pass
        self.update(self.workflow_controller.refresh(), self.live_controller.refresh())

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

"""Focused tracker-first MVP operator tab."""

from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QComboBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from continuum_robot.gui.controllers.registration_controller import RegistrationViewState
from continuum_robot.gui.controllers.tracker_mvp_controller import TrackerMvpViewState
from continuum_robot.gui.tabs.registration_tab import RegistrationTab


class TrackerMvpTab(QWidget):
    """Operator-first workspace for tracker validation, pivot calibration, and 4-point registration."""

    def __init__(self, workflow_controller, registration_controller, parent=None) -> None:
        super().__init__(parent)
        self.controller = workflow_controller
        self.registration_controller = registration_controller
        self.setObjectName("trackerMvpWorkspace")
        self.setStyleSheet(
            """
            QWidget#trackerMvpWorkspace {
                background: #eef3f8;
                color: #0f172a;
            }
            QWidget#trackerMvpWorkspace QGroupBox {
                border: 1px solid #d9e3ec;
                border-radius: 16px;
                margin-top: 16px;
                padding-top: 10px;
                background: #ffffff;
            }
            QWidget#trackerMvpWorkspace QGroupBox::title {
                subcontrol-origin: margin;
                left: 12px;
                padding: 0 6px;
                color: #0f172a;
                font-weight: 700;
            }
            QWidget#trackerMvpWorkspace QLabel[role="title"] {
                font-size: 24px;
                font-weight: 700;
                color: #0f172a;
            }
            QWidget#trackerMvpWorkspace QLabel[role="hint"] {
                color: #526173;
            }
            QWidget#trackerMvpWorkspace QLabel[role="status"] {
                padding: 8px 10px;
                border-radius: 8px;
                background: #e2e8f0;
                color: #0f172a;
                font-weight: 700;
            }
            QWidget#trackerMvpWorkspace QComboBox,
            QWidget#trackerMvpWorkspace QTextEdit,
            QWidget#trackerMvpWorkspace QTableWidget {
                border: 1px solid #dbe4ee;
                border-radius: 10px;
                background: #fbfdff;
                color: #0f172a;
            }
            QWidget#trackerMvpWorkspace QPushButton {
                min-height: 36px;
                padding: 0 14px;
                border: 1px solid #dbe4ee;
                border-radius: 10px;
                background: #f8fafc;
                color: #0f172a;
                font-weight: 600;
            }
            QWidget#trackerMvpWorkspace QPushButton[role="primary"] {
                background: #dbeafe;
                border-color: #93c5fd;
            }
            """
        )

        self.title_label = QLabel("Tracker-First MVP Workspace")
        self.title_label.setProperty("role", "title")
        self.workflow_hint = QLabel(
            "Use this order tomorrow on the Pi: connect tracker, validate tracker health, confirm tool IDs, "
            "run 0B pivot calibration, save the tip file, capture 4-point registration, save the accepted registration, "
            "then confirm live robot-frame pose from 0A."
        )
        self.workflow_hint.setProperty("role", "hint")
        self.workflow_hint.setWordWrap(True)
        self.status_label = QLabel("Connect tracker to begin.")
        self.status_label.setProperty("role", "status")

        self.workflow_table = QTableWidget(0, 4)
        self.workflow_table.setHorizontalHeaderLabels(["Step", "Status", "Gate", "Detail"])
        self.workflow_table.verticalHeader().setVisible(False)

        self.tracker_port_combo = QComboBox()
        self.tracker_port_combo.setEditable(True)
        self.tracker_port_combo.currentIndexChanged.connect(self._sync_tracker_port)
        self.tracker_port_combo.editTextChanged.connect(self._sync_tracker_port)
        self.connection_label = QLabel("disconnected")
        self.health_label = QLabel("not validated")
        self.tools_label = QLabel("none")
        self.validation_report_label = QLabel("none")
        self.tracker_details = QTextEdit()
        self.tracker_details.setReadOnly(True)
        self.tracker_details.setMinimumHeight(120)

        connect_button = QPushButton("Connect Tracker")
        connect_button.setProperty("role", "primary")
        disconnect_button = QPushButton("Disconnect Tracker")
        validate_button = QPushButton("Validate Tracker")
        rescan_button = QPushButton("Rescan Ports")
        connect_button.clicked.connect(self._connect_tracker)
        disconnect_button.clicked.connect(self._disconnect_tracker)
        validate_button.clicked.connect(self._validate_tracker)
        rescan_button.clicked.connect(self._rescan_ports)
        self._validate_button = validate_button
        self._pivot_button = QPushButton("Run 0B Pivot Calibration")
        self._pivot_button.setProperty("role", "primary")
        self._pivot_button.clicked.connect(self._run_pivot)

        tracker_box = QGroupBox("Tracker Readiness")
        tracker_layout = QVBoxLayout(tracker_box)
        tracker_form = QFormLayout()
        tracker_form.addRow("Tracker port", self.tracker_port_combo)
        tracker_form.addRow("Connection", self.connection_label)
        tracker_form.addRow("Tracker health", self.health_label)
        tracker_form.addRow("Tool visibility", self.tools_label)
        tracker_form.addRow("Validation artifact", self.validation_report_label)
        tracker_layout.addLayout(tracker_form)
        tracker_buttons = QHBoxLayout()
        tracker_buttons.addWidget(connect_button)
        tracker_buttons.addWidget(disconnect_button)
        tracker_buttons.addWidget(validate_button)
        tracker_buttons.addWidget(rescan_button)
        tracker_layout.addLayout(tracker_buttons)
        tracker_layout.addWidget(self.tracker_details)

        self.tip_file_label = QLabel("none")
        self.tip_geometry_label = QLabel("not ready")
        self.pivot_metrics_label = QLabel("No pivot run yet.")
        self.pivot_run_path_label = QLabel("none")
        self.tip_preview_text = QTextEdit()
        self.tip_preview_text.setReadOnly(True)
        self.tip_preview_text.setMinimumHeight(88)
        pivot_box = QGroupBox("Pivot Calibration")
        pivot_layout = QVBoxLayout(pivot_box)
        pivot_form = QFormLayout()
        pivot_form.addRow("Tip file", self.tip_file_label)
        pivot_form.addRow("Tip geometry", self.tip_geometry_label)
        pivot_form.addRow("Pivot metrics", self.pivot_metrics_label)
        pivot_form.addRow("Pivot dataset", self.pivot_run_path_label)
        pivot_layout.addLayout(pivot_form)
        pivot_layout.addWidget(self._pivot_button)
        pivot_layout.addWidget(self.tip_preview_text)

        self.transform_summary = QTextEdit()
        self.transform_summary.setReadOnly(True)
        self.transform_summary.setMinimumHeight(120)
        self.live_pose_label = QLabel("not ready")
        self.registration_status_label = QLabel("No accepted registration saved.")
        transform_box = QGroupBox("Registration Dependency Summary")
        transform_layout = QVBoxLayout(transform_box)
        transform_form = QFormLayout()
        transform_form.addRow("Registration status", self.registration_status_label)
        transform_form.addRow("Live robot-frame pose", self.live_pose_label)
        transform_layout.addLayout(transform_form)
        transform_layout.addWidget(self.transform_summary)

        self.registration_gate_label = QLabel(
            "Registration will unlock here after tracker validation passes and a tip file is ready."
        )
        self.registration_gate_label.setProperty("role", "hint")
        self.registration_gate_label.setWordWrap(True)

        self.registration_tab = RegistrationTab(self.registration_controller)
        self.registration_tab.title_label.hide()
        self.registration_tab.workflow_hint.hide()

        top_left = QWidget()
        top_left_layout = QVBoxLayout(top_left)
        top_left_layout.setContentsMargins(0, 0, 0, 0)
        top_left_layout.setSpacing(12)
        workflow_box = QGroupBox("Canonical Workflow")
        workflow_layout = QVBoxLayout(workflow_box)
        workflow_layout.addWidget(self.workflow_table)
        top_left_layout.addWidget(workflow_box)
        top_left_layout.addWidget(tracker_box)

        top_right = QWidget()
        top_right_layout = QVBoxLayout(top_right)
        top_right_layout.setContentsMargins(0, 0, 0, 0)
        top_right_layout.setSpacing(12)
        top_right_layout.addWidget(pivot_box)
        top_right_layout.addWidget(transform_box)

        top_splitter = QSplitter(Qt.Horizontal)
        top_splitter.setChildrenCollapsible(False)
        top_splitter.addWidget(top_left)
        top_splitter.addWidget(top_right)
        top_splitter.setStretchFactor(0, 5)
        top_splitter.setStretchFactor(1, 4)
        top_splitter.setSizes([640, 520])

        registration_container = QWidget()
        registration_layout = QVBoxLayout(registration_container)
        registration_layout.setContentsMargins(0, 0, 0, 0)
        registration_layout.setSpacing(10)
        registration_layout.addWidget(self.registration_gate_label)
        registration_layout.addWidget(self.registration_tab)

        content_splitter = QSplitter(Qt.Vertical)
        content_splitter.setChildrenCollapsible(False)
        content_splitter.addWidget(top_splitter)
        content_splitter.addWidget(registration_container)
        content_splitter.setStretchFactor(0, 2)
        content_splitter.setStretchFactor(1, 5)
        content_splitter.setSizes([280, 720])

        layout = QVBoxLayout(self)
        layout.setContentsMargins(18, 18, 18, 18)
        layout.setSpacing(14)
        layout.addWidget(self.title_label)
        layout.addWidget(self.workflow_hint)
        layout.addWidget(self.status_label)
        layout.addWidget(content_splitter, 1)

    def update(self, workflow_state: TrackerMvpViewState, registration_state: RegistrationViewState) -> None:
        self.status_label.setText(workflow_state.status_message)
        self.connection_label.setText(
            f"{workflow_state.canonical_state} ({workflow_state.connection_state}) | "
            f"backend={workflow_state.backend_name or workflow_state.backend_identity}"
        )
        self.health_label.setText(
            "validated"
            if workflow_state.validation_passed
            else ("healthy" if workflow_state.tracker_healthy else "needs attention")
        )
        self.tools_label.setText(
            f"0A={'tracked' if workflow_state.tool_0a_visible else workflow_state.tool_0a_status}, "
            f"0B={'tracked' if workflow_state.tool_0b_visible else workflow_state.tool_0b_status}"
        )
        self.validation_report_label.setText(workflow_state.validation_report_path or "none")
        self.tracker_details.setPlainText("\n".join(workflow_state.validation_lines))

        self.tip_file_label.setText(workflow_state.pivot_tip_path or "none")
        self.tip_geometry_label.setText(workflow_state.measurement_point_message)
        if workflow_state.pivot_rmse_mm is not None:
            self.pivot_metrics_label.setText(
                f"RMSE={workflow_state.pivot_rmse_mm:.3f} mm | total={workflow_state.pivot_sample_count_total} "
                f"| used={workflow_state.pivot_sample_count_used} | rejected={workflow_state.pivot_sample_count_rejected}"
            )
        else:
            self.pivot_metrics_label.setText(workflow_state.pivot_summary)
        self.pivot_run_path_label.setText(workflow_state.pivot_run_path or "none")
        self.tip_preview_text.setPlainText(workflow_state.pivot_tip_preview)

        self.registration_status_label.setText(workflow_state.latest_registration_status)
        if workflow_state.live_tip_position_mm is not None:
            live_text = (
                f"{workflow_state.live_tip_status} | "
                + ", ".join(f"{value:.2f}" for value in workflow_state.live_tip_position_mm)
            )
        else:
            live_text = workflow_state.live_tip_status
        self.live_pose_label.setText(live_text)
        self.transform_summary.setPlainText("\n".join(workflow_state.transform_summary_lines))

        self.workflow_table.setRowCount(len(workflow_state.workflow_steps))
        for row, step in enumerate(workflow_state.workflow_steps):
            self.workflow_table.setItem(row, 0, QTableWidgetItem(f"{step.index}. {step.label}"))
            self.workflow_table.setItem(row, 1, QTableWidgetItem(step.status))
            gate_text = "ready" if step.status in {"ready", "complete"} else "blocked"
            self.workflow_table.setItem(row, 2, QTableWidgetItem(gate_text))
            self.workflow_table.setItem(row, 3, QTableWidgetItem(step.message))

        self._set_combo_items(self.tracker_port_combo, workflow_state.available_ports, workflow_state.tracker_port)
        self._validate_button.setEnabled(bool(workflow_state.tracker_connected))
        self._pivot_button.setEnabled(bool(workflow_state.validation_passed and workflow_state.tool_0b_visible))

        self.registration_tab.update(registration_state)
        can_start_registration = bool(
            workflow_state.validation_passed
            and workflow_state.measurement_point_ready
            and self.registration_controller.can_begin_session()
        )
        self.registration_tab.begin_button.setEnabled(can_start_registration)
        if can_start_registration:
            self.registration_gate_label.setText(
                "Registration prerequisites passed. Select four landmarks, begin the session, capture all four points, solve, and save."
            )
        else:
            self.registration_gate_label.setText(
                "Registration is gated until tracker validation passes and the 0B tip file is ready. "
                f"Current tip status: {workflow_state.measurement_point_message}"
            )

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
        self.controller.set_tracker_port(self._selected_port(self.tracker_port_combo))

    def _rescan_ports(self) -> None:
        self.controller.rescan_ports()

    def _connect_tracker(self) -> None:
        self._safe_call(self.controller.connect_tracker)

    def _disconnect_tracker(self) -> None:
        self._safe_call(self.controller.disconnect_tracker)

    def _validate_tracker(self) -> None:
        self._safe_call(self.controller.validate_tracker)

    def _run_pivot(self) -> None:
        self._safe_call(self.controller.run_pivot_calibration)

    def _safe_call(self, fn) -> None:
        try:
            fn()
        except Exception:
            pass
        self.update(self.controller.refresh(), self.registration_controller.refresh())

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

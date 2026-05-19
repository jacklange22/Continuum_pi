"""Focused tracker-first MVP operator tab."""

from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QButtonGroup,
    QComboBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QRadioButton,
    QScrollArea,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from continuum_robot.gui.controllers.registration_controller import RegistrationViewState
from continuum_robot.gui.controllers.tracker_mvp_controller import TrackerMvpViewState
from continuum_robot.gui.theme import COLORS, grouped_workspace_stylesheet
from continuum_robot.gui.view_utils import set_text_document
from continuum_robot.gui.tabs.registration_tab import RegistrationTab


class TrackerMvpTab(QWidget):
    """Operator-first workspace for tracker validation, pivot calibration, and 4-point registration."""

    def __init__(self, workflow_controller, registration_controller, parent=None) -> None:
        super().__init__(parent)
        self.controller = workflow_controller
        self.registration_controller = registration_controller
        self.setObjectName("trackerMvpWorkspace")
        self.setStyleSheet(
            grouped_workspace_stylesheet(
                object_name="trackerMvpWorkspace",
                input_selectors=["QComboBox", "QTextEdit", "QTableWidget"],
                extra_rules=(
                    f"""
                    QWidget#trackerMvpWorkspace QLabel[role="legacy-hint"] {{
                        color: {COLORS.warning_fg};
                        background: {COLORS.warning_bg};
                        border: 1px solid {COLORS.surface_border};
                        border-radius: 10px;
                        padding: 8px 10px;
                    }}
                    """
                ),
            )
        )

        self.title_label = QLabel("Legacy Tracker Compatibility Workspace")
        self.title_label.setProperty("role", "title")
        self.workflow_hint = QLabel(
            "The canonical operator workflow now lives in the Tracking and Registration tabs. "
            "Keep this legacy workspace only for compatibility checks or deeper tracker-first diagnostics."
        )
        self.workflow_hint.setProperty("role", "legacy-hint")
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
        disconnect_button.setProperty("variant", "ghost")
        validate_button = QPushButton("Validate Tracker")
        validate_button.setProperty("variant", "ghost")
        rescan_button = QPushButton("Rescan Ports")
        rescan_button.setProperty("variant", "ghost")
        connect_button.clicked.connect(self._connect_tracker)
        disconnect_button.clicked.connect(self._disconnect_tracker)
        validate_button.clicked.connect(self._validate_tracker)
        rescan_button.clicked.connect(self._rescan_ports)
        self._validate_button = validate_button

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
        self.pending_tip_file_label = QLabel("none")
        self.pivot_collection_label = QLabel("not collecting")
        self.pivot_motion_label = QLabel("not measured")
        self.tip_geometry_label = QLabel("not ready")
        self.pivot_metrics_label = QLabel("No pivot run yet.")
        self.pivot_parse_label = QLabel("format not detected")
        self.pivot_capture_dataset_label = QLabel("none")
        self.pivot_run_path_label = QLabel("none")
        self.tip_preview_text = QTextEdit()
        self.tip_preview_text.setReadOnly(True)
        self.tip_preview_text.setMinimumHeight(88)

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

        pivot_box = QGroupBox("Pivot Calibration")
        pivot_layout = QVBoxLayout(pivot_box)
        pivot_form = QFormLayout()
        pivot_form.addRow("Accepted tip file", self.tip_file_label)
        pivot_form.addRow("Pending tip file", self.pending_tip_file_label)
        pivot_form.addRow("Collection", self.pivot_collection_label)
        pivot_form.addRow("Motion diversity", self.pivot_motion_label)
        pivot_form.addRow("Tip geometry", self.tip_geometry_label)
        pivot_form.addRow("Pivot metrics", self.pivot_metrics_label)
        pivot_form.addRow("Input parse", self.pivot_parse_label)
        pivot_form.addRow("Raw capture dataset", self.pivot_capture_dataset_label)
        pivot_form.addRow("Pivot review run", self.pivot_run_path_label)
        pivot_layout.addLayout(pivot_form)

        # Side-by-side solver chooser. Accept promotes whichever solver is
        # checked. The "✓ best" badge highlights the row with the lower RMSE.
        self.pivot_solver_box = QGroupBox("Solver (select before Accept)")
        pivot_solver_layout = QVBoxLayout(self.pivot_solver_box)
        self.pivot_solver_classical_radio = QRadioButton("Classical std-dev")
        self.pivot_solver_ransac_radio = QRadioButton("RANSAC")
        self.pivot_solver_classical_radio.setChecked(True)
        self.pivot_solver_button_group = QButtonGroup(self.pivot_solver_box)
        self.pivot_solver_button_group.addButton(self.pivot_solver_classical_radio, 0)
        self.pivot_solver_button_group.addButton(self.pivot_solver_ransac_radio, 1)
        self.pivot_solver_button_group.buttonClicked.connect(self._on_pivot_solver_selected)
        self.pivot_solver_classical_metrics_label = QLabel("—")
        self.pivot_solver_classical_metrics_label.setWordWrap(True)
        self.pivot_solver_ransac_metrics_label = QLabel("—")
        self.pivot_solver_ransac_metrics_label.setWordWrap(True)
        self.pivot_solver_classical_best_badge = QLabel("")
        self.pivot_solver_ransac_best_badge = QLabel("")
        for badge in (self.pivot_solver_classical_best_badge, self.pivot_solver_ransac_best_badge):
            badge.setStyleSheet("color: #6fcf97; font-weight: 600;")
            badge.setVisible(False)
        for radio, metrics_label, badge in (
            (self.pivot_solver_classical_radio, self.pivot_solver_classical_metrics_label, self.pivot_solver_classical_best_badge),
            (self.pivot_solver_ransac_radio, self.pivot_solver_ransac_metrics_label, self.pivot_solver_ransac_best_badge),
        ):
            row = QHBoxLayout()
            row.setContentsMargins(0, 0, 0, 0)
            row.setSpacing(8)
            row.addWidget(radio)
            row.addWidget(badge)
            row.addStretch(1)
            row.addWidget(metrics_label, 0, Qt.AlignRight)
            pivot_solver_layout.addLayout(row)
        self.pivot_solver_status_label = QLabel("Solve the pivot to enable the comparison.")
        self.pivot_solver_status_label.setWordWrap(True)
        pivot_solver_layout.addWidget(self.pivot_solver_status_label)
        pivot_layout.addWidget(self.pivot_solver_box)

        pivot_buttons = QHBoxLayout()
        pivot_buttons.addWidget(self._pivot_start_button)
        pivot_buttons.addWidget(self._pivot_stop_button)
        pivot_buttons.addWidget(self._pivot_solve_button)
        pivot_buttons.addWidget(self._pivot_accept_button)
        pivot_buttons.addWidget(self._pivot_reset_button)
        pivot_layout.addLayout(pivot_buttons)
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
            "Registration will unlock here after tracker validation passes and an accepted tip file is ready."
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

        content_widget = QWidget()
        content_layout = QVBoxLayout(content_widget)
        content_layout.setContentsMargins(0, 0, 0, 0)
        content_layout.setSpacing(14)
        content_layout.addWidget(top_splitter)
        content_layout.addWidget(registration_container)

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

    def update(self, workflow_state: TrackerMvpViewState, registration_state: RegistrationViewState) -> None:
        self.status_label.setText(workflow_state.status_message)
        self.connection_label.setText(
            f"{workflow_state.canonical_state} ({workflow_state.connection_state}) | "
            f"backend={workflow_state.backend_name or workflow_state.backend_identity}"
        )
        self.health_label.setText(
            "passing"
            if workflow_state.validation_passed
            else (
                "operational with warning"
                if workflow_state.tracker_operational
                else ("connected" if workflow_state.tracker_healthy else "needs attention")
            )
        )
        self.tools_label.setText(
            f"0A={'tracked' if workflow_state.tool_0a_visible else workflow_state.tool_0a_status}, "
            f"0B={'tracked' if workflow_state.tool_0b_visible else workflow_state.tool_0b_status}"
        )
        self.validation_report_label.setText(workflow_state.validation_report_path or "none")
        set_text_document(self.tracker_details, "\n".join(workflow_state.validation_lines), stick_to_bottom_if_at_bottom=True)

        self.tip_file_label.setText(workflow_state.pivot_tip_path or "none")
        self.pending_tip_file_label.setText(workflow_state.pivot_pending_tip_path or "none")
        self.pivot_collection_label.setText(
            f"{workflow_state.pivot_status} | samples={workflow_state.pivot_live_sample_count} | "
            f"0B={workflow_state.pivot_live_tool_status}"
        )
        if workflow_state.pivot_motion_span_deg is not None:
            motion_text = f"{workflow_state.pivot_motion_span_deg:.1f} deg"
            motion_text += " | wide enough" if workflow_state.pivot_motion_ready else " | collect wider motion"
        else:
            motion_text = "Collect more poses to estimate motion span."
        self.pivot_motion_label.setText(motion_text)
        self.tip_geometry_label.setText(workflow_state.measurement_point_message)
        if workflow_state.pivot_rmse_mm is not None:
            self.pivot_metrics_label.setText(
                f"RMSE={workflow_state.pivot_rmse_mm:.3f} mm | total={workflow_state.pivot_sample_count_total} "
                f"| used={workflow_state.pivot_sample_count_used} | rejected={workflow_state.pivot_sample_count_rejected}"
            )
        else:
            self.pivot_metrics_label.setText(workflow_state.pivot_summary)
        rejected_preview = "; ".join(workflow_state.pivot_input_rejected_rows[:3])
        parse_text = (
            f"{workflow_state.pivot_input_format or 'not_detected'} | usable_0B_rows={workflow_state.pivot_input_usable_rows} "
            f"| rejected_rows={workflow_state.pivot_input_rejected_row_count}"
        )
        if rejected_preview:
            parse_text += f" | {rejected_preview}"
        self.pivot_parse_label.setText(parse_text)
        self.pivot_capture_dataset_label.setText(workflow_state.pivot_capture_dataset_path or "none")
        self.pivot_run_path_label.setText(workflow_state.pivot_run_path or "none")
        set_text_document(self.tip_preview_text, workflow_state.pivot_tip_preview)
        self._update_pivot_solver_panel(workflow_state)

        self.registration_status_label.setText(workflow_state.latest_registration_status)
        if workflow_state.live_tip_position_mm is not None:
            live_text = (
                f"{workflow_state.live_tip_status} | "
                + ", ".join(f"{value:.2f}" for value in workflow_state.live_tip_position_mm)
            )
        else:
            live_text = workflow_state.live_tip_status
        self.live_pose_label.setText(live_text)
        set_text_document(self.transform_summary, "\n".join(workflow_state.transform_summary_lines), stick_to_bottom_if_at_bottom=True)

        self.workflow_table.setRowCount(len(workflow_state.workflow_steps))
        for row, step in enumerate(workflow_state.workflow_steps):
            self.workflow_table.setItem(row, 0, QTableWidgetItem(f"{step.index}. {step.label}"))
            self.workflow_table.setItem(row, 1, QTableWidgetItem(step.status))
            gate_text = "ready" if step.status in {"ready", "complete"} else "blocked"
            self.workflow_table.setItem(row, 2, QTableWidgetItem(gate_text))
            self.workflow_table.setItem(row, 3, QTableWidgetItem(step.message))

        self._set_combo_items(self.tracker_port_combo, workflow_state.available_ports, workflow_state.tracker_port)
        self._validate_button.setEnabled(bool(workflow_state.tracker_connected))
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

        self.registration_tab.update(registration_state)
        self.registration_tab.begin_button.setEnabled(bool(workflow_state.registration_ready))
        if workflow_state.registration_blockers:
            self.registration_gate_label.setText(
                "Registration blocked: " + " ".join(workflow_state.registration_blockers)
            )
        elif workflow_state.registration_ready:
            self.registration_gate_label.setText(
                "Registration prerequisites passed. Select four landmarks, begin the session, capture each point, solve, review FRE/residuals, then save."
            )
        elif registration_state.pending_accept:
            self.registration_gate_label.setText(
                "Registration is solved and pending acceptance. Save it to generate the accepted artifact."
            )
        else:
            self.registration_gate_label.setText(
                "Registration is gated until tracker validation passes, the accepted 0B tip file is loaded, "
                "and four landmarks are selected."
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

    def _start_pivot(self) -> None:
        self._safe_call(self.controller.start_pivot_collection)

    def _stop_pivot(self) -> None:
        self._safe_call(self.controller.stop_pivot_collection)

    def _solve_pivot(self) -> None:
        self._safe_call(self.controller.solve_pivot_collection)

    def _accept_pivot(self) -> None:
        self._safe_call(self.controller.accept_pivot_tip_file)

    def _reset_pivot(self) -> None:
        self._safe_call(self.controller.reset_pivot_workflow)

    def _update_pivot_solver_panel(self, state: TrackerMvpViewState) -> None:
        comparison = state.pivot_solver_comparison if isinstance(state.pivot_solver_comparison, dict) else {}
        classical_available = "classical" in state.pivot_solver_choices
        ransac_available = "ransac" in state.pivot_solver_choices
        self.pivot_solver_classical_radio.setEnabled(state.pivot_pending_accept and classical_available)
        self.pivot_solver_ransac_radio.setEnabled(state.pivot_pending_accept and ransac_available)
        self._sync_pivot_solver_radio_state(state)
        classical_text, ransac_text, status_text, classical_best, ransac_best = _pivot_solver_panel_lines(
            comparison=comparison,
            pending_accept=state.pivot_pending_accept,
            active_solver=state.pivot_active_solver,
            classical_available=classical_available,
            ransac_available=ransac_available,
        )
        self.pivot_solver_classical_metrics_label.setText(classical_text)
        self.pivot_solver_ransac_metrics_label.setText(ransac_text)
        self.pivot_solver_status_label.setText(status_text)
        self.pivot_solver_classical_best_badge.setText("✓ best" if classical_best else "")
        self.pivot_solver_classical_best_badge.setVisible(bool(classical_best))
        self.pivot_solver_ransac_best_badge.setText("✓ best" if ransac_best else "")
        self.pivot_solver_ransac_best_badge.setVisible(bool(ransac_best))

    def _on_pivot_solver_selected(self, button) -> None:
        solver_name = "ransac" if button is self.pivot_solver_ransac_radio else "classical"
        # Same anti-bounce pattern as the registration tab: avoid re-entering
        # set_pivot_solver when this came from a programmatic sync.
        if self.controller.state.pivot_active_solver == solver_name:
            return
        if solver_name not in self.controller.state.pivot_solver_choices:
            self._sync_pivot_solver_radio_state(self.controller.state)
            return
        self._safe_call(lambda: self.controller.set_pivot_solver(solver_name))

    def _sync_pivot_solver_radio_state(self, state: TrackerMvpViewState) -> None:
        target_radio = (
            self.pivot_solver_ransac_radio
            if state.pivot_active_solver == "ransac"
            else self.pivot_solver_classical_radio
        )
        if target_radio.isChecked():
            return
        self.pivot_solver_classical_radio.blockSignals(True)
        self.pivot_solver_ransac_radio.blockSignals(True)
        try:
            target_radio.setChecked(True)
        finally:
            self.pivot_solver_classical_radio.blockSignals(False)
            self.pivot_solver_ransac_radio.blockSignals(False)

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


def _pivot_solver_panel_lines(
    *,
    comparison: dict,
    pending_accept: bool,
    active_solver: str,
    classical_available: bool,
    ransac_available: bool,
) -> tuple[str, str, str, bool, bool]:
    """Compute the metrics rows + status line for the pivot solver chooser.

    Returns ``(classical_text, ransac_text, status, classical_is_best,
    ransac_is_best)``.
    """
    if not pending_accept or not comparison:
        return (
            "—",
            "—",
            "Solve the pivot to enable the comparison.",
            False,
            False,
        )
    classical = comparison.get("classical") if isinstance(comparison.get("classical"), dict) else None
    classical_text = "—"
    classical_rmse: float | None = None
    if classical is not None:
        rmse = classical.get("rmse_mm")
        used = classical.get("sample_count_used")
        rejected = classical.get("sample_count_rejected")
        if rmse is not None:
            classical_rmse = float(rmse)
            classical_text = (
                f"RMSE={classical_rmse:.3f} mm, used={int(used or 0)}, rejected={int(rejected or 0)}"
            )
    elif isinstance(comparison.get("classical_failure"), dict):
        classical_text = f"failed: {comparison['classical_failure'].get('message', 'unknown')}"
    ransac_text = "—"
    ransac_rmse: float | None = None
    if "ransac_skipped" in comparison:
        ransac_text = f"unavailable: {comparison['ransac_skipped']}"
    elif isinstance(comparison.get("ransac_failure"), dict):
        ransac_text = f"failed: {comparison['ransac_failure'].get('message', 'unknown')}"
    elif isinstance(comparison.get("ransac"), dict):
        ransac = comparison["ransac"]
        rmse = ransac.get("rmse_mm")
        used = ransac.get("sample_count_used")
        rejected = ransac.get("sample_count_rejected")
        threshold = ransac.get("inlier_threshold_mm")
        if rmse is not None:
            ransac_rmse = float(rmse)
            ransac_text = (
                f"RMSE={ransac_rmse:.3f} mm, used={int(used or 0)}, rejected={int(rejected or 0)}"
            )
        if threshold is not None:
            ransac_text += f" @ {float(threshold):.2f} mm"
    classical_is_best = (
        classical_rmse is not None
        and ransac_rmse is not None
        and classical_rmse < ransac_rmse
    )
    ransac_is_best = (
        classical_rmse is not None
        and ransac_rmse is not None
        and ransac_rmse < classical_rmse
    )
    if not ransac_available:
        status = "RANSAC unavailable for this run. Accept will save the Classical tip file."
    elif not classical_available:
        status = "Classical solve unavailable for this run. Accept will save the RANSAC tip file."
    elif active_solver == "ransac":
        status = "Accept will save the RANSAC tip vector (outlier poses excluded from the fit)."
    else:
        status = "Accept will save the Classical std-dev tip vector. Switch to RANSAC if its RMSE is meaningfully lower."
    return (classical_text, ransac_text, status, classical_is_best, ransac_is_best)

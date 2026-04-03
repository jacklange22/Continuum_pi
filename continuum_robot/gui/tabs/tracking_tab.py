"""Tracking tab widget."""

from __future__ import annotations

from PySide6.QtWidgets import (
    QComboBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QScrollArea,
    QTableWidget,
    QTableWidgetItem,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from continuum_robot.gui.controllers.tracker_mvp_controller import TrackerMvpViewState
from continuum_robot.gui.controllers.tracking_controller import TrackingViewState
from continuum_robot.gui.widgets.tool_plot_widget import ToolPlotWidget


class TrackingTab(QWidget):
    """Canonical tracker and pivot workspace."""

    def __init__(self, live_controller, *, workflow_controller, parent=None) -> None:
        super().__init__(parent)
        self.live_controller = live_controller
        self.workflow_controller = workflow_controller
        self.setObjectName("trackingWorkspace")
        self.setStyleSheet(
            """
            QWidget#trackingWorkspace {
                background: #eef3f8;
                color: #0f172a;
            }
            QWidget#trackingWorkspace QGroupBox {
                border: 1px solid #d9e3ec;
                border-radius: 16px;
                margin-top: 16px;
                padding-top: 10px;
                background: #ffffff;
            }
            QWidget#trackingWorkspace QGroupBox::title {
                subcontrol-origin: margin;
                left: 12px;
                padding: 0 6px;
                color: #0f172a;
                font-weight: 700;
            }
            QWidget#trackingWorkspace QLabel[role="title"] {
                font-size: 24px;
                font-weight: 700;
                color: #0f172a;
            }
            QWidget#trackingWorkspace QLabel[role="hint"] {
                color: #526173;
            }
            QWidget#trackingWorkspace QLabel[role="status"] {
                padding: 8px 10px;
                border-radius: 8px;
                background: #e2e8f0;
                color: #0f172a;
                font-weight: 700;
            }
            QWidget#trackingWorkspace QComboBox,
            QWidget#trackingWorkspace QTextEdit,
            QWidget#trackingWorkspace QTableWidget {
                border: 1px solid #dbe4ee;
                border-radius: 10px;
                background: #fbfdff;
                color: #0f172a;
            }
            QWidget#trackingWorkspace QPushButton {
                min-height: 36px;
                padding: 0 14px;
                border: 1px solid #dbe4ee;
                border-radius: 10px;
                background: #f8fafc;
                color: #0f172a;
                font-weight: 600;
            }
            QWidget#trackingWorkspace QPushButton[role="primary"] {
                background: #dbeafe;
                border-color: #93c5fd;
            }
            """
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
        self.handoff_label = QLabel("Registration will unlock after tracker validation and accepted tip review.")
        self.handoff_label.setWordWrap(True)

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
        readiness_buttons.addWidget(connect_button)
        readiness_buttons.addWidget(disconnect_button)
        readiness_buttons.addWidget(validate_button)
        readiness_buttons.addWidget(rescan_button)
        readiness_layout.addLayout(readiness_buttons)

        self.backend_label = QLabel("unknown")
        self.frame_label = QLabel("none")
        self.fps_label = QLabel("unknown")
        self.freshness_label = QLabel("unknown")
        self.ids_label = QLabel("none")

        health_box = QGroupBox("Operational Summary")
        health_layout = QVBoxLayout(health_box)
        health_form = QFormLayout()
        health_form.addRow("Backend", self.backend_label)
        health_form.addRow("Latest frames", self.frame_label)
        health_form.addRow("Effective FPS", self.fps_label)
        health_form.addRow("Freshness", self.freshness_label)
        health_form.addRow("Tool IDs", self.ids_label)
        health_layout.addLayout(health_form)

        self.tools_table = QTableWidget(0, 5)
        self.tools_table.setHorizontalHeaderLabels(["Tool", "Valid", "Status", "Translation mm", "Quality"])
        self.tools_table.verticalHeader().setVisible(False)
        self.tools_table.setMinimumHeight(200)
        self.plot_widget = ToolPlotWidget()
        self.plot_widget.setMinimumHeight(320)

        live_box = QGroupBox("Live Tools")
        live_layout = QVBoxLayout(live_box)
        live_layout.addWidget(self.plot_widget)
        live_layout.addWidget(self.tools_table)

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

        pivot_box = QGroupBox("0B Pivot Calibration")
        pivot_layout = QVBoxLayout(pivot_box)
        pivot_form = QFormLayout()
        pivot_form.addRow("Accepted tip file", self.tip_file_label)
        pivot_form.addRow("Pending tip file", self.pending_tip_file_label)
        pivot_form.addRow("Collection", self.pivot_collection_label)
        pivot_form.addRow("Motion diversity", self.pivot_motion_label)
        pivot_form.addRow("Tip geometry", self.tip_geometry_label)
        pivot_form.addRow("Review metrics", self.pivot_metrics_label)
        pivot_form.addRow("Input parse", self.pivot_parse_label)
        pivot_form.addRow("Raw capture dataset", self.pivot_capture_dataset_label)
        pivot_form.addRow("Pivot review run", self.pivot_run_path_label)
        pivot_layout.addLayout(pivot_form)
        pivot_buttons = QHBoxLayout()
        pivot_buttons.addWidget(self._pivot_start_button)
        pivot_buttons.addWidget(self._pivot_stop_button)
        pivot_buttons.addWidget(self._pivot_solve_button)
        pivot_buttons.addWidget(self._pivot_accept_button)
        pivot_buttons.addWidget(self._pivot_reset_button)
        pivot_layout.addLayout(pivot_buttons)
        pivot_layout.addWidget(self.tip_preview_text)

        self.details_text = QTextEdit()
        self.details_text.setReadOnly(True)
        self.details_text.setMinimumHeight(110)
        self.details_text.setMaximumHeight(150)
        details_box = QGroupBox("Details")
        details_layout = QVBoxLayout(details_box)
        details_layout.addWidget(self.details_text)

        content_widget = QWidget()
        content_layout = QVBoxLayout(content_widget)
        content_layout.setContentsMargins(0, 0, 0, 0)
        content_layout.setSpacing(14)
        content_layout.addWidget(readiness_box)
        content_layout.addWidget(health_box)
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
        self.verdict_label.setText(
            "passing"
            if workflow_state.validation_passed
            else (
                "operational with warning"
                if workflow_state.tracker_operational
                else ("connected" if workflow_state.tracker_connected else "needs attention")
            )
        )
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

        self.backend_label.setText(
            f"{workflow_state.backend_name or workflow_state.backend_identity} | port={workflow_state.tracker_port or 'unset'}"
        )
        frame_text = str(live_state.latest_frame_number) if live_state.latest_frame_number is not None else "none"
        if workflow_state.unique_frames_observed:
            frame_text += f" | unique={workflow_state.unique_frames_observed}"
        self.frame_label.setText(frame_text)
        self.fps_label.setText(
            f"{workflow_state.effective_frame_rate_hz:.2f} Hz"
            if workflow_state.effective_frame_rate_hz is not None
            else "unknown"
        )
        self.freshness_label.setText(
            f"{workflow_state.tracker_data_age_s:.3f} s"
            if workflow_state.tracker_data_age_s is not None
            else "unknown"
        )
        self.ids_label.setText(
            f"raw={workflow_state.raw_live_tool_ids} | normalized={workflow_state.normalized_live_tool_ids}"
        )

        self.tools_table.setRowCount(len(live_state.tools))
        plot_points: dict[str, tuple[float, float, float]] = {}
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
            if tool["translation_mm"] is not None:
                plot_points[tool_id] = (
                    float(tool["translation_mm"][0]),
                    float(tool["translation_mm"][1]),
                    float(tool["translation_mm"][2]),
                )
        if live_state.tip_position_mm:
            plot_points["tip"] = (
                float(live_state.tip_position_mm[0]),
                float(live_state.tip_position_mm[1]),
                float(live_state.tip_position_mm[2]),
            )
        self.plot_widget.set_points(plot_points)

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
        self.tip_preview_text.setPlainText(workflow_state.pivot_tip_preview)

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

    @staticmethod
    def _set_text_preserving_view(widget: QTextEdit, text: str) -> None:
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

    def _connect_tracker(self) -> None:
        self._safe_call(self.workflow_controller.connect_tracker)

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

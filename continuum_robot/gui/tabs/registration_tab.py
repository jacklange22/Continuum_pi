"""Registration tab widget."""

from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtGui import QColor
from PySide6.QtWidgets import (
    QAbstractItemView,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from continuum_robot.gui.controllers.registration_controller import RegistrationViewState
from continuum_robot.gui.view_utils import set_text_document
from continuum_robot.gui.widgets.registration_landmark_map_widget import RegistrationLandmarkMapWidget
from continuum_robot.gui.widgets.registration_plot_widget import RegistrationPlotWidget


class RegistrationTab(QWidget):
    """Guided 4-point robot-body alignment workflow."""

    def __init__(self, controller, workflow_controller=None, parent=None) -> None:
        super().__init__(parent)
        self.controller = controller
        self.workflow_controller = workflow_controller
        self._selected_slot_labels: list[QLabel] = []
        self.setObjectName("registrationWorkspace")
        self.setStyleSheet(
            """
            QWidget#registrationWorkspace {
                background: #eef3f8;
            }
            QWidget#registrationWorkspace QGroupBox {
                border: 1px solid #d9e3ec;
                border-radius: 16px;
                margin-top: 16px;
                padding-top: 10px;
                background: #ffffff;
            }
            QWidget#registrationWorkspace QGroupBox::title {
                subcontrol-origin: margin;
                left: 12px;
                padding: 0 6px;
                color: #0f172a;
                font-weight: 700;
            }
            QWidget#registrationWorkspace QPushButton {
                min-height: 36px;
                padding: 0 14px;
                border: 1px solid #dbe4ee;
                border-radius: 10px;
                background: #f8fafc;
                color: #0f172a;
                font-weight: 600;
            }
            QWidget#registrationWorkspace QLabel[role="hint"] {
                color: #526173;
            }
            QWidget#registrationWorkspace QLabel[role="status"] {
                padding: 8px 10px;
                border-radius: 8px;
                background: #e2e8f0;
                color: #0f172a;
                font-weight: 700;
            }
            QWidget#registrationWorkspace QTextEdit, QWidget#registrationWorkspace QTableWidget {
                border: 1px solid #dbe4ee;
                border-radius: 10px;
                background: #fbfdff;
                color: #0f172a;
            }
            """
        )

        self.title_label = QLabel("Registration Workspace")
        self.title_label.setStyleSheet("font-size: 24px; font-weight: 700; color: #0f172a;")
        self.workflow_hint = QLabel(
            "Choose four model points, capture one or more pen-probe samples for each, solve, review FRE, then save the accepted registration."
        )
        self.workflow_hint.setWordWrap(True)
        self.workflow_hint.setProperty("role", "hint")

        self.dependency_status_label = QLabel("Waiting for tracker and accepted tip file.")
        self.dependency_status_label.setProperty("role", "status")
        self.tip_file_label = QLabel("none")
        self.tip_geometry_label = QLabel("not ready")
        self.accepted_registration_label = QLabel("No accepted registration saved.")
        self.live_pose_label = QLabel("not ready")
        self.dependency_text = QTextEdit()
        self.dependency_text.setReadOnly(True)
        self.dependency_text.setMinimumHeight(110)
        self.dependency_text.setMaximumHeight(170)

        dependency_box = QGroupBox("Dependencies & Pose")
        dependency_layout = QVBoxLayout(dependency_box)
        dependency_form = QFormLayout()
        dependency_form.addRow("Workflow gate", self.dependency_status_label)
        dependency_form.addRow("Accepted tip file", self.tip_file_label)
        dependency_form.addRow("Tip geometry", self.tip_geometry_label)
        dependency_form.addRow("Accepted registration", self.accepted_registration_label)
        dependency_form.addRow("Live robot-frame pose", self.live_pose_label)
        dependency_layout.addLayout(dependency_form)
        dependency_layout.addWidget(self.dependency_text)

        self.begin_button = QPushButton("Begin Session")
        self.capture_button = QPushButton("Capture Sample")
        self.complete_button = QPushButton("Mark Point Complete")
        self.solve_button = QPushButton("Solve Registration")
        self.save_button = QPushButton("Save Registration")
        self.retry_button = QPushButton("Restart")
        self.load_button = QPushButton("Load Latest")

        self.begin_button.clicked.connect(lambda: self._safe_call(self.controller.begin_session))
        self.capture_button.clicked.connect(lambda: self._safe_call(self.controller.capture_current_label_sample))
        self.complete_button.clicked.connect(lambda: self._safe_call(self.controller.complete_current_label))
        self.solve_button.clicked.connect(lambda: self._safe_call(self.controller.solve_session))
        self.save_button.clicked.connect(self._save_registration)
        self.retry_button.clicked.connect(lambda: self._safe_call(self.controller.retry_session))
        self.load_button.clicked.connect(lambda: self._safe_call(self.controller.load_latest_result))

        self.selection_hint = QLabel("Select exactly four unique model points in capture order.")
        self.selection_hint.setProperty("role", "hint")
        self.selection_hint.setWordWrap(True)
        self.selection_help_label = QLabel("Click the top-view map or a row in the point list to add or remove a point.")
        self.selection_help_label.setProperty("role", "hint")
        self.selection_help_label.setWordWrap(True)
        self.selection_summary_label = QLabel("No model points selected.")
        self.selection_summary_label.setProperty("role", "status")

        self.landmark_map = RegistrationLandmarkMapWidget()
        self.landmark_map.pointToggled.connect(lambda label: self._safe_call(lambda: self.controller.toggle_selected_model_point(label)))

        slot_row = QHBoxLayout()
        slot_row.setContentsMargins(0, 0, 0, 0)
        slot_row.setSpacing(8)
        for index in range(self.controller.REQUIRED_SELECTION_COUNT):
            label = QLabel(f"{index + 1}. Unselected")
            label.setProperty("role", "status")
            label.setMinimumWidth(118)
            self._selected_slot_labels.append(label)
            slot_row.addWidget(label, 1)

        self.available_points_table = QTableWidget(0, 5)
        self.available_points_table.setHorizontalHeaderLabels(
            ["ID", "Label", "Model Coordinates (mm)", "Selected", "Status"]
        )
        self.available_points_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.available_points_table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.available_points_table.setSelectionMode(QAbstractItemView.SingleSelection)
        self.available_points_table.verticalHeader().setVisible(False)
        self.available_points_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeToContents)
        self.available_points_table.horizontalHeader().setSectionResizeMode(1, QHeaderView.Stretch)
        self.available_points_table.horizontalHeader().setSectionResizeMode(2, QHeaderView.Stretch)
        self.available_points_table.horizontalHeader().setSectionResizeMode(3, QHeaderView.ResizeToContents)
        self.available_points_table.horizontalHeader().setSectionResizeMode(4, QHeaderView.ResizeToContents)
        self.available_points_table.setMinimumHeight(180)
        self.available_points_table.cellClicked.connect(self._on_available_point_clicked)

        selection_box = QGroupBox("Model Point Selection")
        selection_layout = QVBoxLayout(selection_box)
        selection_layout.addWidget(self.selection_hint)
        selection_layout.addWidget(self.selection_help_label)
        selection_layout.addWidget(self.landmark_map)
        selection_layout.addLayout(slot_row)
        selection_layout.addWidget(self.selection_summary_label)
        selection_layout.addWidget(self.available_points_table)

        self.session_status_label = QLabel("Idle")
        self.session_status_label.setProperty("role", "status")
        self.tool_label = QLabel()
        self.coil_tool_label = QLabel()
        self.geometry_label = QLabel()
        self.current_label = QLabel()
        self.live_point_label = QLabel()
        self.samples_used_label = QLabel()
        self.fre_label = QLabel()
        self.max_residual_label = QLabel()
        self.result_status_label = QLabel()
        self.result_path_label = QLabel()
        self.selected_points_label = QLabel()
        self.trust_label = QLabel()
        self.live_chain_label = QLabel()
        self.comparison_label = QLabel()
        for label in (self.trust_label, self.live_chain_label, self.comparison_label):
            label.setWordWrap(True)

        summary_box = QGroupBox("Registration Summary")
        summary_layout = QFormLayout(summary_box)
        summary_layout.addRow("Session", self.session_status_label)
        summary_layout.addRow("Selected points", self.selected_points_label)
        summary_layout.addRow("Capture tool", self.tool_label)
        summary_layout.addRow("Runtime coil", self.coil_tool_label)
        summary_layout.addRow("Capture geometry", self.geometry_label)
        summary_layout.addRow("Active point", self.current_label)
        summary_layout.addRow("Live tracked point", self.live_point_label)
        summary_layout.addRow("Samples captured", self.samples_used_label)
        summary_layout.addRow("RMSE / FRE", self.fre_label)
        summary_layout.addRow("Max residual", self.max_residual_label)
        summary_layout.addRow("Trust", self.trust_label)
        summary_layout.addRow("Live chain", self.live_chain_label)
        summary_layout.addRow("Repeated runs", self.comparison_label)
        summary_layout.addRow("Result status", self.result_status_label)
        summary_layout.addRow("Saved file", self.result_path_label)

        button_row_primary = QHBoxLayout()
        button_row_primary.setSpacing(10)
        button_row_primary.addWidget(self.begin_button)
        button_row_primary.addWidget(self.capture_button)
        button_row_primary.addWidget(self.complete_button)
        button_row_primary.addWidget(self.solve_button)
        button_row_primary.addWidget(self.save_button)

        button_row_secondary = QHBoxLayout()
        button_row_secondary.setSpacing(10)
        button_row_secondary.addWidget(self.retry_button)
        button_row_secondary.addWidget(self.load_button)
        button_row_secondary.addStretch(1)

        self.points_table = QTableWidget(0, 6)
        self.points_table.setHorizontalHeaderLabels(
            ["Order", "Point", "Model Point (mm)", "Samples", "Measured Centroid (mm)", "Status"]
        )
        self.points_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.points_table.setSelectionMode(QAbstractItemView.NoSelection)
        self.points_table.verticalHeader().setVisible(False)
        self.points_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeToContents)
        self.points_table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeToContents)
        self.points_table.horizontalHeader().setSectionResizeMode(2, QHeaderView.Stretch)
        self.points_table.horizontalHeader().setSectionResizeMode(3, QHeaderView.ResizeToContents)
        self.points_table.horizontalHeader().setSectionResizeMode(4, QHeaderView.Stretch)
        self.points_table.horizontalHeader().setSectionResizeMode(5, QHeaderView.ResizeToContents)
        self.points_table.setMinimumHeight(220)

        points_box = QGroupBox("Selected Point Mapping")
        points_layout = QVBoxLayout(points_box)
        points_layout.addWidget(self.points_table)

        self.samples_table = QTableWidget(0, 5)
        self.samples_table.setHorizontalHeaderLabels(["Point", "Sample", "X (mm)", "Y (mm)", "Z (mm)"])
        self.samples_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.samples_table.setSelectionMode(QAbstractItemView.NoSelection)
        self.samples_table.verticalHeader().setVisible(False)
        self.samples_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeToContents)
        self.samples_table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeToContents)
        self.samples_table.horizontalHeader().setSectionResizeMode(2, QHeaderView.Stretch)
        self.samples_table.horizontalHeader().setSectionResizeMode(3, QHeaderView.Stretch)
        self.samples_table.horizontalHeader().setSectionResizeMode(4, QHeaderView.Stretch)
        self.samples_table.setMinimumHeight(220)

        samples_box = QGroupBox("Captured Samples")
        samples_layout = QVBoxLayout(samples_box)
        samples_layout.addWidget(self.samples_table)

        self.plot_widget = RegistrationPlotWidget()
        self.plot_widget.setMinimumHeight(280)
        plot_box = QGroupBox("Registration Preview")
        plot_layout = QVBoxLayout(plot_box)
        plot_layout.addWidget(self.plot_widget)

        self.status_text = QTextEdit()
        self.status_text.setReadOnly(True)
        self.status_text.setMinimumHeight(68)
        self.status_text.setMaximumHeight(96)
        status_box = QGroupBox("Operator Status")
        status_layout = QVBoxLayout(status_box)
        status_layout.addWidget(self.status_text)

        self.validation_text = QTextEdit()
        self.validation_text.setReadOnly(True)
        self.validation_text.setMinimumHeight(120)
        self.validation_text.setMaximumHeight(180)
        validation_box = QGroupBox("Validation & Trust")
        validation_layout = QVBoxLayout(validation_box)
        validation_layout.addWidget(self.validation_text)

        top_splitter = QSplitter(Qt.Horizontal)
        top_splitter.setChildrenCollapsible(False)
        top_splitter.setHandleWidth(8)
        top_splitter.addWidget(selection_box)
        top_splitter.addWidget(summary_box)
        top_splitter.addWidget(plot_box)
        top_splitter.setStretchFactor(0, 4)
        top_splitter.setStretchFactor(1, 3)
        top_splitter.setStretchFactor(2, 4)
        top_splitter.setSizes([460, 340, 420])

        lower_splitter = QSplitter(Qt.Horizontal)
        lower_splitter.setChildrenCollapsible(False)
        lower_splitter.setHandleWidth(8)
        lower_splitter.addWidget(points_box)
        lower_splitter.addWidget(samples_box)
        lower_splitter.setStretchFactor(0, 3)
        lower_splitter.setStretchFactor(1, 4)
        lower_splitter.setSizes([420, 540])

        content_widget = QWidget()
        content_layout = QVBoxLayout(content_widget)
        content_layout.setContentsMargins(0, 0, 0, 0)
        content_layout.setSpacing(14)
        content_layout.addWidget(dependency_box)
        content_layout.addLayout(button_row_primary)
        content_layout.addLayout(button_row_secondary)
        content_layout.addWidget(top_splitter, 3)
        content_layout.addWidget(lower_splitter, 2)
        content_layout.addWidget(status_box)
        content_layout.addWidget(validation_box)

        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setWidget(content_widget)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(18, 18, 18, 18)
        layout.setSpacing(14)
        layout.addWidget(self.title_label)
        layout.addWidget(self.workflow_hint)
        layout.addWidget(self.scroll_area, 1)

    def update(self, state: RegistrationViewState, workflow_state=None) -> None:
        session_status = "Solved - ready to save" if state.pending_accept else ("Capturing" if state.active else "Idle")
        self.session_status_label.setText(session_status)
        self._update_dependencies(state, workflow_state)
        if state.selected_model_labels:
            self.selection_summary_label.setText(
                f"{len(state.selected_model_labels)} / 4 selected: {', '.join(state.selected_model_labels)}"
            )
        else:
            self.selection_summary_label.setText("Choose four model points.")
        self.selected_points_label.setText(", ".join(state.selected_model_labels) or "None")
        self.tool_label.setText(state.capture_tool_id)
        self.coil_tool_label.setText(state.coil_tool_id)
        self.geometry_label.setText(state.capture_geometry_status)
        self.current_label.setText(state.current_label or "All selected points complete")
        self.live_point_label.setText(
            _format_xyz(state.current_tracked_xyz_mm, state.current_tracking_status, state.current_tracked_frame_id)
        )
        self.samples_used_label.setText(str(self.controller.total_samples_captured()))
        if state.fre_mm is not None:
            fre_text = f"FRE={state.fre_mm:.3f} mm"
            max_residual = state.validation_metrics.get("max_residual_mm")
            if max_residual is not None:
                fre_text += f" | max residual={float(max_residual):.3f} mm"
            self.fre_label.setText(fre_text)
        else:
            self.fre_label.setText("Not solved yet")
        if state.max_residual_mm is not None:
            self.max_residual_label.setText(f"{state.max_residual_mm:.3f} mm")
        else:
            self.max_residual_label.setText("—")
        self.trust_label.setText(_trust_text(state.trust_state, state.trust_message))
        self.live_chain_label.setText(_trust_text(state.live_chain_state, state.live_chain_message))
        self.comparison_label.setText(state.comparison_message)
        self.result_status_label.setText(state.result_status)
        self.result_path_label.setText(state.last_result_path or "None")

        begin_enabled = self.controller.can_begin_session()
        if workflow_state is not None and not state.active and not state.pending_accept:
            begin_enabled = begin_enabled and bool(getattr(workflow_state, "registration_ready", False))
        self.begin_button.setEnabled(begin_enabled)
        self.capture_button.setEnabled(state.active and state.current_label is not None)
        self.complete_button.setEnabled(state.active and self.controller.is_ready_to_complete_current())
        self.solve_button.setEnabled(state.active and self.controller.is_ready_to_solve())
        self.save_button.setEnabled(state.pending_accept)
        self.retry_button.setEnabled(state.active or state.pending_accept)

        self._update_selection_slots(state)
        self._update_available_points_table(state)
        self.landmark_map.set_landmarks(
            points_by_label=state.available_model_points_by_label,
            display_labels=state.model_point_display_labels,
            enabled_by_label=state.model_point_enabled,
            selected_labels=state.selected_model_labels,
        )

        self.points_table.setRowCount(len(state.landmark_labels))
        captured_plot: dict[str, list[tuple[float, float]]] = {}
        for row, label in enumerate(state.landmark_labels):
            count = state.captured_counts.get(label, 0)
            truth = state.truth_points_in_sw_by_label.get(label) or state.available_model_points_by_label.get(label)
            centroid = state.averaged_points_by_label.get(label)
            status = _point_status(label, state)
            display_name = _display_name(label, state)
            self.points_table.setItem(row, 0, QTableWidgetItem(str(row + 1)))
            self.points_table.setItem(row, 1, QTableWidgetItem(display_name))
            self.points_table.setItem(row, 2, QTableWidgetItem(_render_point(truth)))
            self.points_table.setItem(row, 3, QTableWidgetItem(f"{count} / {state.captures_per_landmark}+"))
            self.points_table.setItem(row, 4, QTableWidgetItem(_render_point(centroid)))
            self.points_table.setItem(row, 5, QTableWidgetItem(status))
            captured_plot[label] = [
                (float(sample[0]), float(sample[1]))
                for sample in state.raw_samples_by_label.get(label, [])
                if len(sample) >= 2
            ]

        sample_rows = [
            (label, index + 1, sample)
            for label in state.landmark_labels
            for index, sample in enumerate(state.raw_samples_by_label.get(label, []))
        ]
        self.samples_table.setRowCount(len(sample_rows))
        for row, (label, sample_index, sample) in enumerate(sample_rows):
            self.samples_table.setItem(row, 0, QTableWidgetItem(label))
            self.samples_table.setItem(row, 1, QTableWidgetItem(str(sample_index)))
            self.samples_table.setItem(row, 2, QTableWidgetItem(_fmt_axis(sample, 0)))
            self.samples_table.setItem(row, 3, QTableWidgetItem(_fmt_axis(sample, 1)))
            self.samples_table.setItem(row, 4, QTableWidgetItem(_fmt_axis(sample, 2)))

        nominal = {
            label: (
                float((state.truth_points_in_sw_by_label.get(label) or state.available_model_points_by_label.get(label))[0]),
                float((state.truth_points_in_sw_by_label.get(label) or state.available_model_points_by_label.get(label))[1]),
            )
            for label in state.landmark_labels
            if (state.truth_points_in_sw_by_label.get(label) or state.available_model_points_by_label.get(label)) is not None
        }
        centroid_map = {
            label: tuple(_mean_xy(samples))
            for label, samples in state.raw_samples_by_label.items()
            if samples
        }
        current_point = tuple(state.current_tracked_xyz_mm[:2]) if state.current_tracked_xyz_mm else None
        self.plot_widget.set_data(
            nominal=nominal,
            captured=captured_plot,
            centroids=centroid_map,
            current_point=current_point,
        )

        lines = [state.status_message]
        if workflow_state is not None and getattr(workflow_state, "registration_blockers", []):
            lines.append("Workflow blockers: " + " ".join(getattr(workflow_state, "registration_blockers", [])))
        elif not self.controller.can_begin_session() and not state.active and not state.pending_accept:
            lines.append(self.controller.begin_session_readiness_message())
        if state.active and not self.controller.is_ready_to_solve():
            lines.append(self.controller.solve_readiness_message())
        if state.pending_accept:
            lines.append("Review FRE / residuals, then save the accepted registration or restart.")
        if state.overwrite_required and state.overwrite_target_path:
            lines.append(f"Save confirmation required: {state.overwrite_target_path}")
        residual_norms = state.validation_metrics.get("residual_norms_mm_by_label", {})
        if isinstance(residual_norms, dict) and residual_norms:
            rendered = ", ".join(
                f"{label}={float(value):.3f} mm"
                for label, value in residual_norms.items()
            )
            lines.append(f"Residual norms: {rendered}")
        if state.last_error:
            lines.append(f"Error: {state.last_error}")
        set_text_document(self.status_text, "\n".join(lines), stick_to_bottom_if_at_bottom=True)
        set_text_document(
            self.validation_text,
            "\n".join(state.validation_lines or [state.trust_message, state.live_chain_message, state.comparison_message]),
            stick_to_bottom_if_at_bottom=True,
        )

    def _update_dependencies(self, state: RegistrationViewState, workflow_state) -> None:
        if workflow_state is None:
            self.dependency_status_label.setText(
                self.controller.begin_session_readiness_message()
                if not state.active and not state.pending_accept
                else "Registration session active."
            )
            self.tip_file_label.setText(state.capture_geometry_status)
            self.tip_geometry_label.setText(state.capture_geometry_status)
            self.accepted_registration_label.setText(state.result_status)
            self.live_pose_label.setText("Load an accepted registration to compute live pose.")
            set_text_document(
                self.dependency_text,
                "\n".join(
                    [
                        "Registration depends on a valid tracker session and accepted pen-probe tip geometry.",
                        f"Capture geometry: {state.capture_geometry_status}",
                    ]
                ),
                stick_to_bottom_if_at_bottom=True,
            )
            return

        blockers = list(getattr(workflow_state, "registration_blockers", []))
        if blockers:
            self.dependency_status_label.setText("Blocked: " + " ".join(blockers))
        elif state.pending_accept:
            self.dependency_status_label.setText("Solved and pending save.")
        elif state.active:
            self.dependency_status_label.setText("Capture in progress.")
        else:
            self.dependency_status_label.setText("Ready for registration workflow.")

        self.tip_file_label.setText(getattr(workflow_state, "pivot_tip_path", "") or "none")
        self.tip_geometry_label.setText(getattr(workflow_state, "measurement_point_message", state.capture_geometry_status))
        self.accepted_registration_label.setText(
            getattr(workflow_state, "latest_registration_status", state.result_status) or state.result_status
        )
        if getattr(workflow_state, "live_tip_position_mm", None) is not None:
            live_pose_text = (
                f"{getattr(workflow_state, 'live_tip_status', 'ok')} | "
                + ", ".join(f"{float(value):.2f}" for value in getattr(workflow_state, "live_tip_position_mm"))
            )
        else:
            live_pose_text = getattr(workflow_state, "live_tip_status", "not ready")
        self.live_pose_label.setText(live_pose_text)

        dependency_lines = []
        if blockers:
            dependency_lines.append("Blocked until: " + " ".join(blockers))
        else:
            dependency_lines.append(
                "Tracker validation, accepted tip geometry, and landmark selection are in place for registration."
            )
        dependency_lines.extend(list(getattr(workflow_state, "transform_summary_lines", [])))
        set_text_document(self.dependency_text, "\n".join(dependency_lines), stick_to_bottom_if_at_bottom=True)

    def _update_selection_slots(self, state: RegistrationViewState) -> None:
        for index, label_widget in enumerate(self._selected_slot_labels):
            if index < len(state.selected_model_labels):
                label = state.selected_model_labels[index]
                label_widget.setText(f"{index + 1}. {_display_name(label, state)}")
                label_widget.setStyleSheet(
                    "padding: 8px 10px; border-radius: 8px; background: #dbeafe; color: #1e3a8a; font-weight: 700;"
                )
            else:
                label_widget.setText(f"{index + 1}. Unselected")
                label_widget.setStyleSheet(
                    "padding: 8px 10px; border-radius: 8px; background: #e2e8f0; color: #475569; font-weight: 700;"
                )
        hint = (
            "Choose four unique enabled model points in capture order."
            if state.selection_editable
            else "This registration mode uses a fixed point set."
        )
        self.selection_hint.setText(hint)

    def _update_available_points_table(self, state: RegistrationViewState) -> None:
        self.available_points_table.setRowCount(len(state.available_model_labels))
        selected = list(state.selected_model_labels)
        selected_lookup = {label: index + 1 for index, label in enumerate(selected)}
        for row, label in enumerate(state.available_model_labels):
            point = state.available_model_points_by_label.get(label)
            display_name = state.model_point_display_labels.get(label, label)
            enabled = state.model_point_enabled.get(label, True)
            selection_text = f"Point {selected_lookup[label]}" if label in selected_lookup else "—"
            status = "Disabled" if not enabled else (_point_status(label, state) if label in state.landmark_labels else "Available")
            self.available_points_table.setItem(row, 0, QTableWidgetItem(label))
            self.available_points_table.setItem(row, 1, QTableWidgetItem(display_name))
            self.available_points_table.setItem(row, 2, QTableWidgetItem(_render_point(point)))
            self.available_points_table.setItem(row, 3, QTableWidgetItem(selection_text))
            self.available_points_table.setItem(row, 4, QTableWidgetItem(status))
            background = QColor("#eff6ff") if label in selected_lookup else (QColor("#f8fafc") if enabled else QColor("#f1f5f9"))
            for column in range(self.available_points_table.columnCount()):
                item = self.available_points_table.item(row, column)
                if item is not None:
                    item.setBackground(background)
                    if not enabled:
                        item.setForeground(QColor("#94a3b8"))
        self.available_points_table.clearSelection()

    def _on_available_point_clicked(self, row: int, _column: int) -> None:
        item = self.available_points_table.item(row, 0)
        if item is None:
            return
        self._safe_call(lambda: self.controller.toggle_selected_model_point(item.text().strip()))

    def _save_registration(self) -> None:
        try:
            self.controller.save_registration()
        except RuntimeError as exc:
            if "overwrite confirmation" not in str(exc):
                self._apply_action_error(exc)
                self.update(self.controller.refresh())
                return
            target = self.controller.state.overwrite_target_path or "latest_registration.json"
            response = QMessageBox.question(
                self,
                "Overwrite Registration",
                f"The accepted registration will overwrite:\n\n{target}\n\nContinue?",
            )
            if response == QMessageBox.Yes:
                self._safe_call(lambda: self.controller.save_registration(confirm_overwrite=True))
            else:
                self.update(self.controller.refresh())

    def _safe_call(self, fn) -> None:
        try:
            fn()
        except Exception as exc:
            self._apply_action_error(exc)
        self.update(self.controller.refresh())

    def _apply_action_error(self, exc: Exception) -> None:
        message = str(exc)
        self.controller.state.last_error = message
        if message not in (self.controller.state.status_message or ""):
            self.controller.state.status_message = f"Action failed: {message}"


def _format_xyz(point: list[float] | None, status: str, frame_id: int | None) -> str:
    if point is None:
        return status
    suffix = f" | frame {frame_id}" if frame_id is not None else ""
    return f"[{point[0]:.2f}, {point[1]:.2f}, {point[2]:.2f}] mm{suffix}"


def _render_point(point: list[float] | None) -> str:
    if point is None:
        return "—"
    return f"[{point[0]:.2f}, {point[1]:.2f}, {point[2]:.2f}]"


def _fmt_axis(point: list[float], index: int) -> str:
    if index >= len(point):
        return "—"
    return f"{float(point[index]):.3f}"


def _mean_xy(samples: list[list[float]]) -> tuple[float, float]:
    xs = [float(sample[0]) for sample in samples if len(sample) >= 2]
    ys = [float(sample[1]) for sample in samples if len(sample) >= 2]
    return (sum(xs) / len(xs), sum(ys) / len(ys))


def _point_status(label: str, state: RegistrationViewState) -> str:
    if label in state.completed_labels:
        return "Complete"
    if label == state.current_label:
        return "Active"
    if state.pending_accept and state.current_label is None:
        return "Captured"
    return "Pending"


def _display_name(label: str, state: RegistrationViewState) -> str:
    display = state.model_point_display_labels.get(label, label)
    return display if display == label else f"{display} ({label})"


def _trust_text(state: str, message: str) -> str:
    state_text = str(state).replace("_", " ").strip()
    if not state_text:
        return message
    prefix = state_text.capitalize()
    return f"{prefix}: {message}"

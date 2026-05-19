"""Registration tab widget."""

from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtGui import QColor
from PySide6.QtWidgets import (
    QAbstractItemView,
    QComboBox,
    QFormLayout,
    QFrame,
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
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from continuum_robot.gui.controllers.registration_controller import RegistrationViewState
from continuum_robot.gui.theme import COLORS, chip_stylesheet, grouped_workspace_stylesheet, semantic_chip_colors
from continuum_robot.gui.view_utils import ResponsiveSplitterController, preserve_scroll_position, set_text_document
from continuum_robot.gui.widgets.registration_landmark_map_widget import RegistrationLandmarkMapWidget
from continuum_robot.gui.widgets.registration_plot_widget import RegistrationPlotWidget


class RegistrationTab(QWidget):
    """Guided 4-point robot-body alignment workflow."""

    def __init__(
        self,
        controller,
        workflow_controller=None,
        open_runtime_tip_calibration=None,
        open_registration_trial=None,
        parent=None,
    ) -> None:
        super().__init__(parent)
        self.controller = controller
        self.workflow_controller = workflow_controller
        self.open_runtime_tip_calibration = open_runtime_tip_calibration
        self.open_registration_trial = open_registration_trial
        self._selected_slot_labels: list[QLabel] = []
        self.setObjectName("registrationWorkspace")
        self.setStyleSheet(
            grouped_workspace_stylesheet(
                object_name="registrationWorkspace",
                input_selectors=["QTextEdit", "QTableWidget"],
            )
        )

        self.title_label = QLabel("Registration")
        self.title_label.setProperty("role", "title")
        self.status_chip = QLabel("Idle")
        self.status_chip.setAlignment(Qt.AlignCenter)
        self.status_chip.setMinimumWidth(100)
        self.workflow_hint = QLabel("")
        self.workflow_hint.setVisible(False)

        self.dependency_status_label = QLabel("Waiting for tracker and accepted tip file.")
        self.dependency_status_label.setProperty("role", "status")
        self.runtime_tip_mode_combo = QComboBox()
        self.runtime_tip_mode_combo.addItem("Latest Accepted", "latest_accepted")
        self.runtime_tip_mode_combo.addItem("Quick 4-Point", "quick_4_point")
        self.runtime_tip_mode_combo.addItem("Coil as Tip (0A Direct)", "coil_as_tip")
        self.runtime_tip_mode_combo.currentIndexChanged.connect(self._on_runtime_tip_mode_changed)
        self.runtime_tip_trust_label = QLabel("missing")
        self.runtime_tip_trust_label.setWordWrap(True)
        self.runtime_tip_mode_message_label = QLabel("No runtime tip mode selected.")
        self.runtime_tip_mode_message_label.setWordWrap(True)
        self.tip_file_label = QLabel("none")
        self.tip_file_label.setWordWrap(True)
        self.tip_geometry_label = QLabel("not ready")
        self.tip_geometry_label.setWordWrap(True)
        self.accepted_registration_label = QLabel("No accepted registration saved.")
        self.accepted_registration_label.setWordWrap(True)
        self.live_pose_label = QLabel("not ready")
        self.live_pose_label.setWordWrap(True)
        self.dependency_text = QTextEdit()
        self.dependency_text.setReadOnly(True)
        self.dependency_text.setMinimumHeight(90)
        self.dependency_text.setMaximumHeight(150)

        dependency_box = QGroupBox("Tip & Registration Source")
        dependency_layout = QVBoxLayout(dependency_box)
        status_row = QHBoxLayout()
        status_row.setContentsMargins(0, 0, 0, 0)
        status_row.setSpacing(10)
        status_row.addWidget(self.status_chip, 0)
        status_row.addWidget(self.dependency_status_label, 1)
        dependency_layout.addLayout(status_row)
        dependency_form = QFormLayout()
        runtime_tip_row = QHBoxLayout()
        runtime_tip_row.setContentsMargins(0, 0, 0, 0)
        runtime_tip_row.setSpacing(8)
        runtime_tip_row.addWidget(self.runtime_tip_mode_combo, 1)
        runtime_tip_row.addWidget(self.runtime_tip_trust_label, 0)
        runtime_tip_widget = QWidget()
        runtime_tip_widget.setLayout(runtime_tip_row)
        dependency_form.addRow("Runtime tip mode", runtime_tip_widget)
        dependency_form.addRow("Tip source", self.runtime_tip_mode_message_label)
        dependency_form.addRow("Accepted tip file", self.tip_file_label)
        dependency_form.addRow("Saved registration", self.accepted_registration_label)
        dependency_form.addRow("Live robot-frame pose", self.live_pose_label)
        dependency_layout.addLayout(dependency_form)

        self.begin_button = QPushButton("Begin Session")
        self.begin_button.setProperty("role", "primary")
        self.capture_button = QPushButton("Capture Sample")
        captures_per_landmark = int(getattr(self.controller.state, "captures_per_landmark", 0) or 0)
        self.capture_batch_button = QPushButton(
            f"Capture Batch (×{captures_per_landmark})" if captures_per_landmark > 0 else "Capture Batch"
        )
        self.capture_batch_button.setToolTip(
            "Capture captures_per_landmark samples for the current point in one click. "
            "Useful when the probe is settled on a fixture and you want the full sample "
            "count without pressing Capture repeatedly."
        )
        self.complete_button = QPushButton("Mark Point Complete")
        self.solve_button = QPushButton("Solve Registration")
        self.save_button = QPushButton("Save Registration")
        self.save_button.setProperty("role", "primary")
        self.retry_button = QPushButton("Restart")
        self.retry_button.setProperty("variant", "ghost")
        self.load_button = QPushButton("Load Latest")
        self.load_button.setProperty("variant", "ghost")
        self.runtime_tip_button = QPushButton("Open Runtime Tip Calibration")
        self.runtime_tip_button.setProperty("variant", "ghost")
        # Promoted to a primary-styled button so the operator can find it on the
        # secondary row at a glance. The previous variant="ghost" sat next to the
        # other ghost buttons and was reported as invisible on a real bench.
        self.trial_mode_button = QPushButton("Run Registration Trial →")
        self.trial_mode_button.setProperty("role", "primary")
        self.trial_mode_button.setToolTip(
            "Capture many samples across many landmarks and run the registration_trial "
            "experiment to compare averaging methods and find the best 4-of-N subset. "
            "Does not affect the production registration session."
        )

        self.begin_button.clicked.connect(lambda: self._safe_call(self.controller.begin_session))
        self.capture_button.clicked.connect(lambda: self._safe_call(self.controller.capture_current_label_sample))
        self.capture_batch_button.clicked.connect(lambda: self._safe_call(self.controller.capture_current_label_batch))
        self.complete_button.clicked.connect(lambda: self._safe_call(self.controller.complete_current_label))
        self.solve_button.clicked.connect(lambda: self._safe_call(self.controller.solve_session))
        self.save_button.clicked.connect(self._save_registration)
        self.retry_button.clicked.connect(lambda: self._safe_call(self.controller.retry_session))
        self.load_button.clicked.connect(lambda: self._safe_call(self.controller.load_latest_result))
        self.runtime_tip_button.clicked.connect(self._open_runtime_tip_calibration)
        self.trial_mode_button.clicked.connect(self._open_registration_trial)
        # Always visible. The opener callback may be None in standalone tests; in that
        # case the click handler is a no-op (no exception raised). This keeps the
        # button discoverable even if the host app shell forgets to wire it.
        self.trial_mode_button.setVisible(True)

        required_count = int(self.controller.REQUIRED_SELECTION_COUNT)
        minimum_count = int(self.controller.MINIMUM_SELECTION_COUNT)
        self.selection_hint = QLabel(
            f"Tap the map or a row to toggle a landmark. Capture order = selection order. "
            f"Min {minimum_count}, up to {required_count}."
        )
        self.selection_hint.setProperty("role", "hint")
        self.selection_hint.setWordWrap(True)

        # Selection summary chip: compact count badge + condensed label list.
        self.selection_count_chip = QLabel("0 / 0")
        self.selection_count_chip.setAlignment(Qt.AlignCenter)
        self.selection_count_chip.setMinimumWidth(70)
        self.selection_summary_label = QLabel("No landmarks selected.")
        self.selection_summary_label.setProperty("role", "hint")
        self.selection_summary_label.setWordWrap(True)
        selection_summary_row = QHBoxLayout()
        selection_summary_row.setContentsMargins(0, 0, 0, 0)
        selection_summary_row.setSpacing(10)
        selection_summary_row.addWidget(self.selection_count_chip, 0)
        selection_summary_row.addWidget(self.selection_summary_label, 1)

        # Capture-order strip: compact numbered dots (works at any N, replaces
        # the old 4-slot chip row which broke past 4 selections).
        self.capture_order_strip = QLabel("")
        self.capture_order_strip.setWordWrap(True)
        self.capture_order_strip.setTextFormat(Qt.RichText)
        self.capture_order_strip.setProperty("role", "hint")

        self.landmark_map = RegistrationLandmarkMapWidget()
        self.landmark_map.pointToggled.connect(lambda label: self._safe_call(lambda: self.controller.toggle_selected_model_point(label)))

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
        self.available_points_table.setMinimumHeight(160)
        self.available_points_table.cellClicked.connect(self._on_available_point_clicked)

        selection_box = QGroupBox("Model Point Selection")
        selection_layout = QVBoxLayout(selection_box)
        selection_layout.setSpacing(8)
        selection_layout.addWidget(self.selection_hint)
        selection_layout.addWidget(self.landmark_map)
        selection_layout.addLayout(selection_summary_row)
        selection_layout.addWidget(self.capture_order_strip)
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
        self.result_path_label.setWordWrap(True)
        self.selected_points_label = QLabel()
        self.trust_label = QLabel()
        self.live_chain_label = QLabel()
        self.comparison_label = QLabel()
        for label in (self.trust_label, self.live_chain_label, self.comparison_label):
            label.setWordWrap(True)

        summary_box = QGroupBox("Session")
        summary_layout = QFormLayout(summary_box)
        summary_layout.addRow("Status", self.session_status_label)
        summary_layout.addRow("Selected points", self.selected_points_label)
        summary_layout.addRow("Active point", self.current_label)
        summary_layout.addRow("Live tracked point", self.live_point_label)
        summary_layout.addRow("Samples captured", self.samples_used_label)
        summary_layout.addRow("RMSE / FRE", self.fre_label)
        summary_layout.addRow("Max residual", self.max_residual_label)

        # Group buttons by workflow phase so 6+ actions don't read as one wall.
        def _phase_group(*buttons: QPushButton) -> QWidget:
            wrapper = QWidget()
            wrap_layout = QHBoxLayout(wrapper)
            wrap_layout.setContentsMargins(0, 0, 0, 0)
            wrap_layout.setSpacing(6)
            for btn in buttons:
                wrap_layout.addWidget(btn)
            return wrapper

        def _phase_separator() -> QFrame:
            sep = QFrame()
            sep.setFrameShape(QFrame.VLine)
            sep.setFrameShadow(QFrame.Plain)
            sep.setStyleSheet(f"color: {COLORS.surface_border}; background: {COLORS.surface_border}; max-width: 1px;")
            sep.setFixedWidth(1)
            return sep

        button_row_primary = QHBoxLayout()
        button_row_primary.setSpacing(14)
        button_row_primary.addWidget(_phase_group(self.begin_button))
        button_row_primary.addWidget(_phase_separator())
        button_row_primary.addWidget(_phase_group(self.capture_button, self.capture_batch_button))
        button_row_primary.addWidget(_phase_separator())
        button_row_primary.addWidget(_phase_group(self.complete_button, self.solve_button, self.save_button))
        button_row_primary.addStretch(1)

        # trial_mode_button stays role=primary (operators couldn't find it
        # under ghost styling — see test_trial_mode_button_uses_primary_style).
        button_row_secondary = QHBoxLayout()
        button_row_secondary.setSpacing(10)
        button_row_secondary.addWidget(self.retry_button)
        button_row_secondary.addWidget(self.load_button)
        button_row_secondary.addWidget(self.runtime_tip_button)
        button_row_secondary.addWidget(self.trial_mode_button)
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
        self.points_table.setMinimumHeight(180)

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
        self.samples_table.setMinimumHeight(180)

        samples_box = QGroupBox("Captured Samples")
        samples_layout = QVBoxLayout(samples_box)
        samples_layout.addWidget(self.samples_table)

        self.plot_widget = RegistrationPlotWidget()
        self.plot_widget.setMinimumHeight(220)
        plot_box = QGroupBox("Registration Preview")
        plot_layout = QVBoxLayout(plot_box)
        plot_layout.addWidget(self.plot_widget)

        self.status_text = QTextEdit()
        self.status_text.setReadOnly(True)
        self.status_text.setMinimumHeight(72)
        self.status_text.setMaximumHeight(110)
        self.validation_text = QTextEdit()
        self.validation_text.setReadOnly(True)
        self.validation_text.setMinimumHeight(96)
        self.validation_text.setMaximumHeight(150)

        details_box = QFrame()
        details_box.setProperty("role", "card")
        details_box_layout = QVBoxLayout(details_box)
        details_box_layout.setContentsMargins(16, 12, 16, 12)
        details_box_layout.setSpacing(8)
        details_header = QHBoxLayout()
        details_header.setContentsMargins(0, 0, 0, 0)
        details_header.setSpacing(8)
        self._details_toggle = QToolButton()
        self._details_toggle.setCheckable(True)
        self._details_toggle.setChecked(False)
        self._details_toggle.setArrowType(Qt.RightArrow)
        self._details_toggle.setToolButtonStyle(Qt.ToolButtonIconOnly)
        self._details_toggle.setAutoRaise(True)
        details_title = QLabel("Transforms, validation & operator log")
        details_title.setProperty("role", "section-title")
        details_title.setCursor(Qt.PointingHandCursor)
        details_title.mousePressEvent = lambda _e: self._details_toggle.setChecked(not self._details_toggle.isChecked())
        details_header.addWidget(self._details_toggle)
        details_header.addWidget(details_title, 1)
        details_box_layout.addLayout(details_header)
        self._details_body = QWidget()
        self._details_body.setVisible(False)
        details_body_layout = QVBoxLayout(self._details_body)
        details_body_layout.setContentsMargins(0, 0, 0, 0)
        details_body_layout.setSpacing(10)

        transforms_label = QLabel("Transforms & dependencies")
        transforms_label.setStyleSheet(f"color: {COLORS.text_secondary}; font-weight: 600;")
        details_body_layout.addWidget(transforms_label)
        details_body_layout.addWidget(self.dependency_text)

        status_label = QLabel("Operator status")
        status_label.setStyleSheet(f"color: {COLORS.text_secondary}; font-weight: 600;")
        details_body_layout.addWidget(status_label)
        details_body_layout.addWidget(self.status_text)

        validation_label = QLabel("Validation & trust")
        validation_label.setStyleSheet(f"color: {COLORS.text_secondary}; font-weight: 600;")
        details_body_layout.addWidget(validation_label)
        details_body_layout.addWidget(self.validation_text)

        details_box_layout.addWidget(self._details_body)
        self._details_toggle.toggled.connect(self._on_details_toggled)

        self.top_splitter = QSplitter(Qt.Horizontal)
        self.top_splitter.setChildrenCollapsible(False)
        self.top_splitter.setHandleWidth(8)
        self.top_splitter.addWidget(selection_box)
        self.top_splitter.addWidget(summary_box)
        self.top_splitter.addWidget(plot_box)
        self.top_splitter.setStretchFactor(0, 4)
        self.top_splitter.setStretchFactor(1, 3)
        self.top_splitter.setStretchFactor(2, 4)
        self.top_splitter.setSizes([460, 340, 420])

        self.lower_splitter = QSplitter(Qt.Horizontal)
        self.lower_splitter.setChildrenCollapsible(False)
        self.lower_splitter.setHandleWidth(8)
        self.lower_splitter.addWidget(points_box)
        self.lower_splitter.addWidget(samples_box)
        self.lower_splitter.setStretchFactor(0, 3)
        self.lower_splitter.setStretchFactor(1, 4)
        self.lower_splitter.setSizes([420, 540])
        self._top_splitter_layout = ResponsiveSplitterController(
            self.top_splitter,
            collapse_below_width=1220,
            horizontal_sizes=[460, 340, 420],
            vertical_sizes=[360, 280, 260],
        )
        self._lower_splitter_layout = ResponsiveSplitterController(
            self.lower_splitter,
            collapse_below_width=1080,
            horizontal_sizes=[420, 540],
            vertical_sizes=[220, 220],
        )

        content_widget = QWidget()
        content_layout = QVBoxLayout(content_widget)
        content_layout.setContentsMargins(0, 0, 0, 0)
        content_layout.setSpacing(14)
        content_layout.addWidget(dependency_box)
        content_layout.addLayout(button_row_primary)
        content_layout.addLayout(button_row_secondary)
        content_layout.addWidget(self.top_splitter, 3)
        content_layout.addWidget(self.lower_splitter, 2)
        content_layout.addWidget(details_box)

        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setWidget(content_widget)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(18, 18, 18, 18)
        layout.setSpacing(14)
        layout.addWidget(self.title_label)
        layout.addWidget(self.workflow_hint)
        layout.addWidget(self.scroll_area, 1)
        self._apply_responsive_layout()

    def update(self, state: RegistrationViewState, workflow_state=None) -> None:
        self.runtime_tip_mode_combo.blockSignals(True)
        mode_index = max(0, self.runtime_tip_mode_combo.findData(state.runtime_tip_mode))
        self.runtime_tip_mode_combo.setCurrentIndex(mode_index)
        self.runtime_tip_mode_combo.blockSignals(False)
        self.runtime_tip_trust_label.setText(str(state.runtime_tip_trust_level).replace("_", " "))
        self.runtime_tip_mode_message_label.setText(state.runtime_tip_mode_message)
        session_status = "Solved - ready to save" if state.pending_accept else ("Capturing" if state.active else "Idle")
        self.session_status_label.setText(session_status)
        self._update_dependencies(state, workflow_state)
        self._update_status_chip(state, workflow_state)
        required_count = int(self.controller.REQUIRED_SELECTION_COUNT)
        selected = list(state.selected_model_labels)
        selected_count = len(selected)
        # Count chip styling reflects whether selection is ready to solve.
        chip_kind = "ok" if selected_count >= int(self.controller.MINIMUM_SELECTION_COUNT) else "warning"
        chip_bg, chip_fg = semantic_chip_colors(chip_kind)
        self.selection_count_chip.setText(f"{selected_count} / {required_count}")
        self.selection_count_chip.setStyleSheet(chip_stylesheet(background=chip_bg, foreground=chip_fg))
        if selected:
            preview = ", ".join(selected[:6])
            if selected_count > 6:
                preview = f"{preview}, … ({selected_count - 6} more)"
            self.selection_summary_label.setText(preview)
        else:
            self.selection_summary_label.setText(
                f"Choose at least {int(self.controller.MINIMUM_SELECTION_COUNT)} model points."
            )
        self.capture_order_strip.setText(self._render_capture_order_strip(state))
        self.selected_points_label.setText(", ".join(state.selected_model_labels) or "None")
        self.tool_label.setText(state.capture_tool_id)
        self.coil_tool_label.setText(state.coil_tool_id)
        self.geometry_label.setText(state.capture_geometry_status)
        self.current_label.setText(state.current_label or "All selected points complete")
        self.live_point_label.setText(
            _format_xyz(state.current_tracked_xyz_mm, state.current_tracking_status, state.current_tracked_frame_id)
        )
        target_per_point = int(state.captures_per_landmark)
        total_done = int(self.controller.total_samples_captured())
        points_complete = sum(
            1
            for label in state.selected_model_labels
            if int(state.captured_counts.get(label, 0)) >= target_per_point
        )
        total_target = target_per_point * len(state.selected_model_labels)
        self.samples_used_label.setText(
            f"{total_done} / {total_target}  ·  {points_complete} / {len(state.selected_model_labels)} points done"
        )
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
        # Batch button: live whenever single Capture is live AND the current
        # point still needs more samples to hit captures_per_landmark.
        batch_remaining = (
            int(state.captures_per_landmark) - int(state.captured_counts.get(state.current_label, 0))
            if state.current_label is not None
            else 0
        )
        self.capture_batch_button.setEnabled(
            state.active and state.current_label is not None and batch_remaining > 0
        )
        self.capture_batch_button.setText(
            f"Capture Batch (×{batch_remaining})" if batch_remaining > 0 else "Capture Batch"
        )
        self.complete_button.setEnabled(state.active and self.controller.is_ready_to_complete_current())
        self.solve_button.setEnabled(state.active and self.controller.is_ready_to_solve())
        self.save_button.setEnabled(state.pending_accept)
        self.retry_button.setEnabled(state.active or state.pending_accept)
        self.runtime_tip_button.setEnabled(True)

        self._update_selection_slots(state)
        self._update_available_points_table(state)
        self.landmark_map.set_landmarks(
            points_by_label=state.available_model_points_by_label,
            display_labels=state.model_point_display_labels,
            enabled_by_label=state.model_point_enabled,
            selected_labels=state.selected_model_labels,
        )

        def _rebuild_points_table() -> None:
            self.points_table.setRowCount(len(state.landmark_labels))
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
        preserve_scroll_position(self.points_table, _rebuild_points_table)

        sample_rows = [
            (label, index + 1, sample)
            for label in state.landmark_labels
            for index, sample in enumerate(state.raw_samples_by_label.get(label, []))
        ]
        def _rebuild_samples_table() -> None:
            self.samples_table.setRowCount(len(sample_rows))
            for row, (label, sample_index, sample) in enumerate(sample_rows):
                self.samples_table.setItem(row, 0, QTableWidgetItem(label))
                self.samples_table.setItem(row, 1, QTableWidgetItem(str(sample_index)))
                self.samples_table.setItem(row, 2, QTableWidgetItem(_fmt_axis(sample, 0)))
                self.samples_table.setItem(row, 3, QTableWidgetItem(_fmt_axis(sample, 1)))
                self.samples_table.setItem(row, 4, QTableWidgetItem(_fmt_axis(sample, 2)))
        preserve_scroll_position(self.samples_table, _rebuild_samples_table)

        nominal = {
            label: tuple(
                float(value)
                for value in (state.truth_points_in_sw_by_label.get(label) or state.available_model_points_by_label.get(label))[0:3]
            )
            for label in state.landmark_labels
            if (state.truth_points_in_sw_by_label.get(label) or state.available_model_points_by_label.get(label)) is not None
        }
        averaged_points = {
            label: tuple(float(value) for value in point[0:3])
            for label, point in state.averaged_points_by_label.items()
            if point is not None and len(point) >= 3
        }
        current_point = tuple(float(value) for value in state.current_tracked_xyz_mm[0:3]) if state.current_tracked_xyz_mm else None
        self.plot_widget.set_data(
            nominal=nominal,
            captured=state.raw_samples_by_label,
            averaged_points=averaged_points,
            completed_labels=state.completed_labels,
            selected_label=state.current_label or (state.selected_model_labels[0] if state.selected_model_labels else None),
            current_point=current_point,
            solved_transform=state.T_robot_aurora,
            solved_residuals_by_label=state.residuals_by_label,
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

    def _render_capture_order_strip(self, state: RegistrationViewState) -> str:
        """Compact HTML dot-strip showing capture-order progress.

        Each dot is the ordinal number, color-coded by phase:
          done    — full sample count reached         (success)
          current — actively being captured            (accent)
          pending — selected, waiting in line          (neutral)
        """
        selected = list(state.selected_model_labels)
        if not selected:
            return ""
        target = max(1, int(state.captures_per_landmark))
        current = state.current_label
        chips: list[str] = []
        for index, label in enumerate(selected):
            count = int(state.captured_counts.get(label, 0))
            if count >= target:
                bg, fg = semantic_chip_colors("ok")
            elif label == current and state.active:
                bg, fg = semantic_chip_colors("accent")
            else:
                bg, fg = semantic_chip_colors("neutral")
            chips.append(
                f'<span style="'
                f'display: inline-block; padding: 2px 8px; margin: 2px 4px 2px 0;'
                f' border-radius: 999px; background: {bg}; color: {fg};'
                f' font-weight: 700; font-size: 11px;">'
                f"{index + 1}&nbsp;{_display_name(label, state)}"
                f"</span>"
            )
        return "".join(chips)

    def _update_selection_slots(self, state: RegistrationViewState) -> None:
        """Kept as a no-op for back-compat; the dot strip handles selection viz now."""
        required_count = int(self.controller.REQUIRED_SELECTION_COUNT)
        minimum_count = int(self.controller.MINIMUM_SELECTION_COUNT)
        hint = (
            f"Tap the map or a row to toggle a landmark. Capture order = selection order. "
            f"Min {minimum_count}, up to {required_count}."
            if state.selection_editable
            else "This registration mode uses a fixed point set."
        )
        self.selection_hint.setText(hint)

    def _update_available_points_table(self, state: RegistrationViewState) -> None:
        selected = list(state.selected_model_labels)
        selected_lookup = {label: index + 1 for index, label in enumerate(selected)}
        def _rebuild() -> None:
            self.available_points_table.setRowCount(len(state.available_model_labels))
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
                background = (
                    QColor(COLORS.selection_bg)
                    if label in selected_lookup
                    else (QColor(COLORS.surface_bg) if enabled else QColor(COLORS.button_bg))
                )
                for column in range(self.available_points_table.columnCount()):
                    item = self.available_points_table.item(row, column)
                    if item is not None:
                        item.setBackground(background)
                        if not enabled:
                            item.setForeground(QColor(COLORS.text_subtle))
            self.available_points_table.clearSelection()
        preserve_scroll_position(self.available_points_table, _rebuild)

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

    def _open_runtime_tip_calibration(self) -> None:
        if callable(self.open_runtime_tip_calibration):
            self.open_runtime_tip_calibration()

    def _on_details_toggled(self, checked: bool) -> None:
        self._details_toggle.setArrowType(Qt.DownArrow if checked else Qt.RightArrow)
        self._details_body.setVisible(bool(checked))

    def _update_status_chip(self, state: RegistrationViewState, workflow_state) -> None:
        if workflow_state is not None and getattr(workflow_state, "registration_blockers", None):
            kind, text = "blocked", "Blocked"
        elif state.pending_accept:
            kind, text = "warning", "Solved · Save"
        elif state.active:
            kind, text = "accent", "Capturing"
        elif state.last_result_path:
            kind, text = "ready", "Loaded"
        elif self.controller.can_begin_session() or (workflow_state is not None and getattr(workflow_state, "registration_ready", False)):
            kind, text = "ready", "Ready"
        else:
            kind, text = "neutral", "Idle"
        bg, fg = semantic_chip_colors(kind)
        self.status_chip.setText(text)
        self.status_chip.setStyleSheet(chip_stylesheet(background=bg, foreground=fg))

    def _open_registration_trial(self) -> None:
        if callable(self.open_registration_trial):
            self.open_registration_trial()

    def _on_runtime_tip_mode_changed(self, _index: int) -> None:
        mode = self.runtime_tip_mode_combo.currentData()
        if mode is None:
            return
        self._safe_call(lambda: self.controller.set_runtime_tip_mode(str(mode)))

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

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        self._apply_responsive_layout()

    def _apply_responsive_layout(self) -> None:
        available_width = max(self.width(), self.scroll_area.viewport().width())
        self._top_splitter_layout.apply(available_width)
        self._lower_splitter_layout.apply(available_width)


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

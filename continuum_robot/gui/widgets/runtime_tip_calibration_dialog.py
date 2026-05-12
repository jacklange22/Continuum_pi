"""Advanced dialog for 0A runtime tip calibration."""

from __future__ import annotations

from PySide6.QtCore import Qt, QTimer
from PySide6.QtWidgets import (
    QAbstractItemView,
    QComboBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QScrollArea,
    QSpinBox,
    QDoubleSpinBox,
    QTableWidget,
    QTableWidgetItem,
    QTextEdit,
    QVBoxLayout,
    QWidget,
    QDialog,
)

from continuum_robot.gui.theme import COLORS
from continuum_robot.gui.view_utils import set_text_document


class RuntimeTipCalibrationDialog(QDialog):
    """Small advanced workspace for hat-based 0A runtime tip calibration."""

    REFRESH_INTERVAL_MS = 200

    def __init__(self, controller, parent=None) -> None:
        super().__init__(parent)
        self.controller = controller
        self.setWindowTitle("Runtime Tip Calibration")
        self.resize(1100, 860)
        self._timer = QTimer(self)
        self._timer.timeout.connect(self.refresh)

        self.title_label = QLabel("Runtime Tip Calibration")
        self.title_label.setStyleSheet(
            f"font-size: 22px; font-weight: 700; color: {COLORS.text_primary};"
        )
        self.description_label = QLabel(
            "Advanced 0A hat-based calibration for the live tip chain. "
            "Use the calibrated 0B pen probe to capture the hat points, collect stationary 0A samples, "
            "then solve T_coil_tip for T_robot_tip = T_robot_aurora @ T_aurora_coil @ T_coil_tip. "
            "If you want direct 0A / no-transform behavior, select Coil as Tip (0A Direct) in Registration; "
            "that mode does not require running this calibration dialog."
        )
        self.description_label.setWordWrap(True)

        self.measurement_point_label = QLabel("unknown")
        self.hat_geometry_label = QLabel("unknown")
        self.runtime_chain_label = QLabel("unknown")
        self.live_runtime_tip_mode_label = QLabel("unknown")
        self.live_runtime_tip_mode_label.setWordWrap(True)
        self.live_runtime_tip_source_label = QLabel("unknown")
        self.live_runtime_tip_source_label.setWordWrap(True)
        self.live_runtime_tip_guidance_label = QLabel("Select a runtime tip mode from Registration.")
        self.live_runtime_tip_guidance_label.setWordWrap(True)
        self.latest_artifact_label = QLabel("none")
        dependency_box = QGroupBox("Dependencies & Runtime Chain")
        dependency_form = QFormLayout(dependency_box)
        dependency_form.addRow("0B measurement point", self.measurement_point_label)
        dependency_form.addRow("Hat truth geometry", self.hat_geometry_label)
        dependency_form.addRow("Active live tip mode", self.live_runtime_tip_mode_label)
        dependency_form.addRow("Active live tip source", self.live_runtime_tip_source_label)
        dependency_form.addRow("Operator note", self.live_runtime_tip_guidance_label)
        dependency_form.addRow("Runtime chain status", self.runtime_chain_label)
        dependency_form.addRow("Latest artifact", self.latest_artifact_label)

        self.session_mode_combo = QComboBox()
        self.session_mode_combo.addItem("Full Accepted Hat Calibration", "full_hat")
        self.session_mode_combo.addItem("Quick 4-Point Override", "quick_4_point")
        self.session_mode_combo.currentIndexChanged.connect(self._on_session_mode_changed)
        self._pending_session_mode = str(controller.state.session_mode or "full_hat")
        self.captures_spin = QSpinBox()
        self.captures_spin.setRange(1, 20)
        self.captures_spin.setValue(max(1, int(controller.state.captures_per_landmark or 3)))
        self.coil_sample_count_spin = QSpinBox()
        self.coil_sample_count_spin.setRange(1, 500)
        self.coil_sample_count_spin.setValue(max(1, int(controller.state.coil_sample_count or 50)))
        self.coil_interval_spin = QDoubleSpinBox()
        self.coil_interval_spin.setDecimals(3)
        self.coil_interval_spin.setSingleStep(0.01)
        self.coil_interval_spin.setRange(0.0, 5.0)
        self.coil_interval_spin.setValue(0.02)
        parameter_box = QGroupBox("Session Parameters")
        parameter_form = QFormLayout(parameter_box)
        parameter_form.addRow("Session mode", self.session_mode_combo)
        parameter_form.addRow("Captures / hat point", self.captures_spin)
        parameter_form.addRow("0A sample count", self.coil_sample_count_spin)
        parameter_form.addRow("0A sample interval (s)", self.coil_interval_spin)

        self.begin_button = QPushButton("Begin Session")
        self.capture_button = QPushButton("Capture Sample")
        self.complete_button = QPushButton("Mark Point Complete")
        self.collect_coil_button = QPushButton("Collect 0A Samples")
        self.solve_button = QPushButton("Solve Calibration")
        self.save_button = QPushButton("Save Calibration")
        self.load_button = QPushButton("Load Latest")

        self.begin_button.clicked.connect(
            lambda: self._safe_call(
                lambda: self.controller.begin_session(
                    captures_per_landmark=self.captures_spin.value(),
                    session_mode=self._selected_session_mode(),
                )
            )
        )
        self.capture_button.clicked.connect(lambda: self._safe_call(self.controller.capture_current_label_sample))
        self.complete_button.clicked.connect(lambda: self._safe_call(self.controller.complete_current_label))
        self.collect_coil_button.clicked.connect(
            lambda: self._safe_call(
                lambda: self.controller.collect_coil_samples(
                    sample_count=self.coil_sample_count_spin.value(),
                    sample_interval_s=float(self.coil_interval_spin.value()),
                )
            )
        )
        self.solve_button.clicked.connect(lambda: self._safe_call(self.controller.solve_calibration))
        self.save_button.clicked.connect(lambda: self._safe_call(self.controller.save_calibration))
        self.load_button.clicked.connect(
            lambda: self._safe_call(
                lambda: self.controller.load_latest_result(
                    session_mode=self._selected_session_mode(),
                )
            )
        )

        button_row_primary = QHBoxLayout()
        for button in (
            self.begin_button,
            self.capture_button,
            self.complete_button,
            self.collect_coil_button,
            self.solve_button,
            self.save_button,
        ):
            button_row_primary.addWidget(button)
        button_row_secondary = QHBoxLayout()
        button_row_secondary.addWidget(self.load_button)
        button_row_secondary.addStretch(1)

        self.current_label_value = QLabel("—")
        self.fit_rmse_value = QLabel("—")
        self.max_residual_value = QLabel("—")
        self.coil_samples_value = QLabel("0")
        self.coil_translation_spread_value = QLabel("—")
        self.coil_rotation_spread_value = QLabel("—")
        self.saved_path_value = QLabel("none")
        summary_box = QGroupBox("Summary")
        summary_form = QFormLayout(summary_box)
        summary_form.addRow("Current hat point", self.current_label_value)
        summary_form.addRow("Hat fit RMSE", self.fit_rmse_value)
        summary_form.addRow("Max hat residual", self.max_residual_value)
        summary_form.addRow("0A samples", self.coil_samples_value)
        summary_form.addRow("0A translation spread", self.coil_translation_spread_value)
        summary_form.addRow("0A rotation spread", self.coil_rotation_spread_value)
        summary_form.addRow("Saved artifact", self.saved_path_value)

        self.points_table = QTableWidget(0, 7)
        self.points_table.setHorizontalHeaderLabels(
            ["Point", "Truth Tip (mm)", "Samples", "Measured Centroid (mm)", "Residual (mm)", "Spread (mm)", "Status"]
        )
        self.points_table.verticalHeader().setVisible(False)
        self.points_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.points_table.setSelectionMode(QAbstractItemView.NoSelection)
        self.points_table.setMinimumHeight(280)
        points_box = QGroupBox("Hat Point Capture")
        points_layout = QVBoxLayout(points_box)
        points_layout.addWidget(self.points_table)

        self.status_text = QTextEdit()
        self.status_text.setReadOnly(True)
        self.status_text.setMinimumHeight(90)
        self.validation_text = QTextEdit()
        self.validation_text.setReadOnly(True)
        self.validation_text.setMinimumHeight(140)
        status_box = QGroupBox("Operator Status")
        status_layout = QVBoxLayout(status_box)
        status_layout.addWidget(self.status_text)
        validation_box = QGroupBox("Validation & Trust")
        validation_layout = QVBoxLayout(validation_box)
        validation_layout.addWidget(self.validation_text)

        content = QWidget()
        content_layout = QVBoxLayout(content)
        content_layout.setContentsMargins(0, 0, 0, 0)
        content_layout.setSpacing(14)
        content_layout.addWidget(dependency_box)
        content_layout.addWidget(parameter_box)
        content_layout.addLayout(button_row_primary)
        content_layout.addLayout(button_row_secondary)
        content_layout.addWidget(summary_box)
        content_layout.addWidget(points_box)
        content_layout.addWidget(status_box)
        content_layout.addWidget(validation_box)

        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setWidget(content)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(18, 18, 18, 18)
        layout.setSpacing(12)
        layout.addWidget(self.title_label)
        layout.addWidget(self.description_label)
        layout.addWidget(self.scroll_area, 1)

        self.update(self.controller.refresh())

    def showEvent(self, event) -> None:  # pragma: no cover - Qt lifecycle
        super().showEvent(event)
        self._timer.start(self.REFRESH_INTERVAL_MS)
        self.refresh()

    def hideEvent(self, event) -> None:  # pragma: no cover - Qt lifecycle
        self._timer.stop()
        super().hideEvent(event)

    def closeEvent(self, event) -> None:  # pragma: no cover - Qt lifecycle
        self._timer.stop()
        super().closeEvent(event)

    def refresh(self) -> None:
        self.update(self.controller.refresh())

    def update(self, state) -> None:
        if state.active or state.pending_accept:
            self._pending_session_mode = str(state.session_mode or "full_hat")
        selected_mode = self._pending_session_mode if not (state.active or state.pending_accept) else str(state.session_mode or "full_hat")
        self.session_mode_combo.blockSignals(True)
        selected_index = max(0, self.session_mode_combo.findData(selected_mode))
        self.session_mode_combo.setCurrentIndex(selected_index)
        self.session_mode_combo.blockSignals(False)
        self.measurement_point_label.setText(state.measurement_point_status)
        self.hat_geometry_label.setText(state.hat_geometry_status)
        self.live_runtime_tip_mode_label.setText(
            f"{str(state.active_runtime_tip_mode).replace('_', ' ')} | "
            f"{str(state.active_runtime_tip_trust_level).replace('_', ' ')}"
        )
        runtime_tip_mode = str(state.active_runtime_tip_mode or "")
        runtime_tip_source_message = str(state.active_runtime_tip_mode_message or "")
        if runtime_tip_mode == "coil_as_tip":
            runtime_tip_source_message = (
                "0A coil pose is shown directly as the tip. "
                f"{runtime_tip_source_message}".strip()
            )
        self.live_runtime_tip_source_label.setText(runtime_tip_source_message)
        self.live_runtime_tip_guidance_label.setText(state.active_runtime_tip_guidance)
        self.runtime_chain_label.setText(f"{state.runtime_chain_state}: {state.runtime_chain_message}")
        self.latest_artifact_label.setText(state.latest_accepted_path or "none")
        self.current_label_value.setText(state.current_label or "All hat points complete")
        self.fit_rmse_value.setText(f"{state.fit_rmse_mm:.3f} mm" if state.fit_rmse_mm is not None else "—")
        self.max_residual_value.setText(
            f"{state.max_residual_mm:.3f} mm" if state.max_residual_mm is not None else "—"
        )
        self.coil_samples_value.setText(f"{state.coil_samples_captured} / {state.coil_sample_count}")
        self.coil_translation_spread_value.setText(_summary_text(state.translation_spread_summary_mm, "mm"))
        self.coil_rotation_spread_value.setText(_summary_text(state.rotation_spread_summary_deg, "deg"))
        self.saved_path_value.setText(state.accepted_output_path or state.latest_accepted_path or "none")

        self.begin_button.setEnabled(self.controller.can_begin_session())
        self.capture_button.setEnabled(self.controller.can_capture())
        self.complete_button.setEnabled(self.controller.can_complete_current())
        self.collect_coil_button.setEnabled(self.controller.can_collect_coil_samples())
        self.solve_button.setEnabled(self.controller.can_solve())
        self.save_button.setEnabled(self.controller.can_save())

        labels = list(state.labels)
        self.points_table.setRowCount(len(labels))
        residual_norms = (state.validation_metrics or {}).get("residual_norms_mm_by_label", {}) or {}
        point_spreads = (state.validation_metrics or {}).get("point_spread_mm_by_label", {}) or {}
        for row, label in enumerate(labels):
            truth = state.truth_points_in_tip_by_label.get(label)
            centroid = state.averaged_points_by_label.get(label)
            count = state.captured_counts.get(label, 0)
            status = _point_status(label, state)
            residual = residual_norms.get(label)
            spread = point_spreads.get(label)
            self.points_table.setItem(row, 0, QTableWidgetItem(label))
            self.points_table.setItem(row, 1, QTableWidgetItem(_render_point(truth)))
            self.points_table.setItem(row, 2, QTableWidgetItem(f"{count} / {state.captures_per_landmark}"))
            self.points_table.setItem(row, 3, QTableWidgetItem(_render_point(centroid)))
            self.points_table.setItem(row, 4, QTableWidgetItem(f"{float(residual):.3f}" if residual is not None else "—"))
            self.points_table.setItem(row, 5, QTableWidgetItem(f"{float(spread):.3f}" if spread is not None else "—"))
            self.points_table.setItem(row, 6, QTableWidgetItem(status))

        status_lines = [state.status_message]
        status_lines.append(f"Session mode: {state.session_mode_summary}")
        if state.pending_accept:
            status_lines.append(
                "Review the residuals and 0A spread metrics, then save this runtime tip result."
            )
        elif state.active and state.current_label is None and state.coil_samples_captured <= 0:
            status_lines.append("Collect stationary 0A samples before solving.")
        if state.health.last_error:
            status_lines.append(f"Error: {state.health.last_error}")
        set_text_document(self.status_text, "\n".join(status_lines), stick_to_bottom_if_at_bottom=True)

        validation_lines = [
            f"Runtime chain: {state.runtime_chain_state}",
            state.runtime_chain_message,
        ]
        if state.fit_rmse_mm is not None:
            validation_lines.append(f"Hat fit RMSE: {state.fit_rmse_mm:.3f} mm")
        if state.max_residual_mm is not None:
            validation_lines.append(f"Max hat residual: {state.max_residual_mm:.3f} mm")
        if state.translation_spread_summary_mm:
            validation_lines.append(
                "0A translation spread: " + _summary_text(state.translation_spread_summary_mm, "mm")
            )
        if state.rotation_spread_summary_deg:
            validation_lines.append(
                "0A rotation spread: " + _summary_text(state.rotation_spread_summary_deg, "deg")
            )
        set_text_document(self.validation_text, "\n".join(validation_lines), stick_to_bottom_if_at_bottom=True)

    def set_preferred_session_mode(self, session_mode: str | None) -> None:
        if session_mode in {"full_hat", "quick_4_point"}:
            self._pending_session_mode = str(session_mode)
        else:
            self._pending_session_mode = "full_hat"
        if not (self.controller.state.active or self.controller.state.pending_accept):
            self.update(self.controller.refresh())

    def _selected_session_mode(self) -> str:
        current = self.session_mode_combo.currentData()
        if current in {"full_hat", "quick_4_point"}:
            return str(current)
        return str(self._pending_session_mode or "full_hat")

    def _on_session_mode_changed(self, _index: int) -> None:
        self._pending_session_mode = self._selected_session_mode()

    def _safe_call(self, fn) -> None:
        try:
            fn()
        except Exception as exc:
            self.controller.state.health.last_error = str(exc)
            self.controller.state.status_message = f"Action failed: {exc}"
        self.update(self.controller.refresh())


def _render_point(point: list[float] | None) -> str:
    if point is None:
        return "—"
    return f"[{float(point[0]):.2f}, {float(point[1]):.2f}, {float(point[2]):.2f}]"


def _summary_text(summary: dict, unit: str) -> str:
    if not summary:
        return "—"
    mean = summary.get("mean")
    max_value = summary.get("max")
    count = int(summary.get("count") or 0)
    if mean is None or max_value is None:
        return f"{count} samples"
    return f"mean={float(mean):.3f} {unit}, max={float(max_value):.3f} {unit}, n={count}"


def _point_status(label: str, state) -> str:
    count = state.captured_counts.get(label, 0)
    if label == state.current_label:
        return "Active"
    if count >= state.captures_per_landmark:
        return "Complete"
    if count > 0:
        return "Partial"
    return "Pending"

"""Custom experiment pages hosted by the Experiment tab shell/router."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Callable

import numpy as np
import yaml
from PySide6.QtCore import QSignalBlocker, Qt, QUrl
from PySide6.QtGui import QColor, QDesktopServices
from PySide6.QtWidgets import (
    QAbstractItemView,
    QBoxLayout,
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QFrame,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QPlainTextEdit,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QSpinBox,
    QTableWidget,
    QTableWidgetItem,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from continuum_robot.experiments.critical_experiments import (
    GridAccuracyPreview,
    GridDefinitionConfig,
    RepeatabilityDatasetConfig,
    RepeatabilityPreview,
    build_grid_accuracy_preview,
    build_grid_truth_catalog,
    build_repeatability_preview,
    capture_grid_measurement_from_snapshot,
    resolve_grid_tip_vector,
)
from continuum_robot.experiments.builtins import PretensionValidationExperimentConfig
from continuum_robot.gui.controllers.experiment_controller import ExperimentViewState
from continuum_robot.gui.view_utils import preserve_scroll_position, set_line_edit_text, set_text_document
from continuum_robot.gui.widgets.experiment_3d_widget import Experiment3DWidget
from continuum_robot.gui.widgets.experiment_preflight_widget import ExperimentPreflightWidget
from continuum_robot.gui.widgets.experiment_results_widget import ExperimentResultsWidget


class ExperimentPageBase(QWidget):
    """Base class for one custom experiment workspace page."""

    show_visualization = False
    page_hint = ""

    def __init__(self, controller, experiment_name: str, parent=None) -> None:
        super().__init__(parent)
        self.controller = controller
        self.experiment_name = experiment_name
        self._history_signature: tuple[tuple[str, str, str, str, str, str], ...] | None = None
        self._responsive_layout_mode = ""

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setFrameShape(QFrame.NoFrame)
        self.scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        layout.addWidget(self.scroll_area)

        page_content = QWidget()
        self.scroll_area.setWidget(page_content)
        content_layout = QVBoxLayout(page_content)
        content_layout.setContentsMargins(0, 0, 0, 0)
        content_layout.setSpacing(14)

        if self.page_hint:
            hint = QLabel(self.page_hint)
            hint.setProperty("role", "body")
            hint.setWordWrap(True)
            content_layout.addWidget(hint)

        self.top_row = QBoxLayout(QBoxLayout.LeftToRight)
        self.top_row.setContentsMargins(0, 0, 0, 0)
        self.top_row.setSpacing(14)

        self.parameter_scroll = QScrollArea()
        self.parameter_scroll.setWidgetResizable(True)
        self.parameter_scroll.setFrameShape(QFrame.NoFrame)
        self.parameter_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.parameter_scroll.setMinimumWidth(300)
        parameter_container = QWidget()
        self.parameter_layout = QVBoxLayout(parameter_container)
        self.parameter_layout.setContentsMargins(0, 0, 0, 0)
        self.parameter_layout.setSpacing(12)
        self._build_parameter_sections()
        self.parameter_layout.addStretch(1)
        self.parameter_scroll.setWidget(parameter_container)

        self.preflight_widget = ExperimentPreflightWidget()
        preflight_card = ExperimentCard("Preflight", "Review the experiment-specific blockers and warnings before running.")
        preflight_card.body_layout.addWidget(self.preflight_widget)

        self.run_button = QPushButton("Run Experiment")
        self.run_button.setProperty("variant", "primary")
        self.stop_button = QPushButton("Stop")
        self.stop_button.setProperty("variant", "danger")
        self.refresh_button = QPushButton("Refresh Checks")
        self.refresh_button.setProperty("variant", "ghost")
        self.run_button.clicked.connect(self._run)
        self.stop_button.clicked.connect(self.controller.stop)
        self.refresh_button.clicked.connect(self.controller.refresh)
        self.progress_bar = QProgressBar()
        self.status_text = QTextEdit()
        self.status_text.setReadOnly(True)
        self.status_text.setMinimumHeight(74)
        self.status_text.setMaximumHeight(110)
        self.output_root_edit = QLineEdit()
        self.output_root_edit.editingFinished.connect(lambda: self.controller.set_output_root(self.output_root_edit.text()))
        self.notes_edit = QPlainTextEdit()
        self.notes_edit.setPlaceholderText("Short operator note stored in metadata.json")
        self.notes_edit.setMinimumHeight(72)
        self.notes_edit.setMaximumHeight(92)
        self.notes_edit.textChanged.connect(lambda: self.controller.set_operator_notes(self.notes_edit.toPlainText()))
        self.destination_widget = KeyValueSummaryWidget()
        self.open_output_button = QPushButton("Open Run Folder")
        self.open_output_button.setProperty("variant", "ghost")
        self.open_output_button.clicked.connect(self._open_current_run_folder)

        run_card = ExperimentCard("Run", "Use the canonical runner and save outputs to the canonical experiment data location.")
        run_form = QFormLayout()
        run_form.setContentsMargins(0, 0, 0, 0)
        run_form.setSpacing(10)
        run_form.addRow("Output Root", self.output_root_edit)
        run_form.addRow("Operator Note", self.notes_edit)
        run_card.body_layout.addLayout(run_form)
        run_buttons = QHBoxLayout()
        run_buttons.setContentsMargins(0, 0, 0, 0)
        run_buttons.setSpacing(10)
        run_buttons.addWidget(self.run_button, 3)
        run_buttons.addWidget(self.stop_button, 2)
        run_buttons.addWidget(self.refresh_button, 2)
        run_card.body_layout.addLayout(run_buttons)
        run_card.body_layout.addWidget(self.progress_bar)
        run_card.body_layout.addWidget(self.destination_widget)
        run_card.body_layout.addWidget(self.status_text)
        run_card.body_layout.addWidget(self.open_output_button, 0, Qt.AlignLeft)

        right_column = QWidget()
        right_layout = QVBoxLayout(right_column)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.setSpacing(12)
        right_layout.addWidget(preflight_card)
        right_layout.addWidget(run_card)
        right_layout.addStretch(1)

        self.top_row.addWidget(self.parameter_scroll, 3)
        self.top_row.addWidget(right_column, 2)
        content_layout.addLayout(self.top_row)

        if self.show_visualization:
            self.viewer_3d = Experiment3DWidget(
                requested_mode=self.controller.settings.runtime.visualization_mode,
                safe_effects=self.controller.settings.runtime.visualization_safe_effects,
            )
            self.viewer_3d.setMinimumHeight(300)
            viz_card = ExperimentCard("Visualization", "Experiment-specific spatial context when the selected run includes positional samples.")
            viz_card.body_layout.addWidget(self.viewer_3d)
            content_layout.addWidget(viz_card)
        else:
            self.viewer_3d = None

        self.result_details_widget = KeyValueSummaryWidget()
        self.results_widget = ExperimentResultsWidget()
        self.export_plot_button = QPushButton("Save Current Plot")
        self.export_plot_button.setProperty("variant", "ghost")
        self.export_plot_button.clicked.connect(self._save_plot_image)
        results_card = ExperimentCard("Results", "Summary metrics, paths, warnings, and plots for the current or loaded run.")
        results_toolbar = QHBoxLayout()
        results_toolbar.setContentsMargins(0, 0, 0, 0)
        results_toolbar.addStretch(1)
        results_toolbar.addWidget(self.export_plot_button)
        results_card.body_layout.addLayout(results_toolbar)
        results_card.body_layout.addWidget(self.result_details_widget)
        results_card.body_layout.addWidget(self.results_widget)

        self.history_list = QListWidget()
        self.history_list.itemDoubleClicked.connect(self._load_selected_history_item)
        self.history_list.setSpacing(8)
        self.history_list.setMinimumHeight(220)
        history_card = ExperimentCard("Recent Runs", "Double-click a previous run to load its summary, output paths, and plots.")
        history_card.body_layout.addWidget(self.history_list)

        self.bottom_row = QBoxLayout(QBoxLayout.LeftToRight)
        self.bottom_row.setContentsMargins(0, 0, 0, 0)
        self.bottom_row.setSpacing(14)
        self.bottom_row.addWidget(results_card, 3)
        self.bottom_row.addWidget(history_card, 2)
        content_layout.addLayout(self.bottom_row)

        self._apply_responsive_layout()

    def set_state(self, state: ExperimentViewState) -> None:
        self._set_line_text(self.output_root_edit, state.output_root)
        self._set_plain_text(self.notes_edit, state.operator_notes)

        self.preflight_widget.set_report(state.preflight_report)
        self.result_details_widget.set_pairs(state.result_details)
        self.progress_bar.setMaximum(max(1, state.progress_total or 1))
        self.progress_bar.setValue(state.progress_current)
        self.run_button.setEnabled(not state.run_active and state.preflight_report.overall_status != "blocked")
        self.stop_button.setEnabled(state.run_active)
        self.open_output_button.setEnabled(bool(state.last_output_path or state.loaded_run_path))
        self.export_plot_button.setEnabled(bool(state.visualization_model.charts))
        self.destination_widget.set_pairs(
            [
                ("Planned Output", state.planned_output_dir or "n/a"),
                ("Current Run", state.last_output_path or state.loaded_run_path or "none"),
            ]
        )
        lines = [state.status_message]
        if state.config_error:
            lines.append(f"Config: {state.config_error}")
        if state.last_error:
            lines.append(f"Error: {state.last_error}")
        set_text_document(self.status_text, "\n".join(lines), stick_to_bottom_if_at_bottom=True)
        self.results_widget.set_model(state.visualization_model)
        if self.viewer_3d is not None:
            self.viewer_3d.set_view_options(show_axes=state.show_axes, show_labels=state.show_labels)
            self.viewer_3d.set_series(state.visualization_model.series_3d)
        self._update_history(state)
        self._sync_parameters_from_state(state)

    def _build_parameter_sections(self) -> None:
        raise NotImplementedError

    def _sync_parameters_from_state(self, state: ExperimentViewState) -> None:
        raise NotImplementedError

    def _run(self) -> None:
        try:
            report = self.controller.refresh().preflight_report
            if report.requires_confirmation:
                details = "\n".join(report.overwrite_targets)
                response = QMessageBox.question(
                    self,
                    "Confirm Overwrite",
                    f"The following output(s) already exist and will be overwritten:\n\n{details}\n\nContinue?",
                )
                if response != QMessageBox.Yes:
                    return
                self.controller.run(confirm_overwrite=True)
            else:
                self.controller.run()
        except Exception:
            return

    def _load_selected_history_item(self, item: QListWidgetItem) -> None:
        raw_path = item.data(Qt.UserRole)
        if raw_path:
            try:
                self.controller.load_run(raw_path)
            except Exception:
                return

    def _open_current_run_folder(self) -> None:
        path = self.controller.state.last_output_path or self.controller.state.loaded_run_path
        if path:
            QDesktopServices.openUrl(QUrl.fromLocalFile(str(path)))

    def _save_plot_image(self) -> None:
        path, _ = QFileDialog.getSaveFileName(self, "Save Plot Image", "experiment_plot.png", "PNG Images (*.png)")
        if path:
            self.results_widget.save_current_view(path)

    def _update_history(self, state: ExperimentViewState) -> None:
        signature = tuple(
            (
                entry.path,
                entry.experiment_name,
                entry.timestamp_utc,
                entry.status,
                entry.label,
                entry.metric_summary,
            )
            for entry in state.history
        )
        if signature == self._history_signature:
            return
        self._history_signature = signature
        selected_path = None
        current_item = self.history_list.currentItem()
        if current_item is not None:
            selected_path = current_item.data(Qt.UserRole)

        def _rebuild() -> None:
            self.history_list.clear()
            selected_row = None
            for row, entry in enumerate(state.history):
                item = QListWidgetItem()
                item.setData(Qt.UserRole, entry.path)
                item.setToolTip(entry.path)
                self.history_list.addItem(item)
                widget = HistoryItemWidget(
                    experiment_name=entry.experiment_name,
                    timestamp_utc=entry.timestamp_utc,
                    status=entry.status,
                    metric_summary=entry.metric_summary,
                    run_path=entry.path,
                )
                item.setSizeHint(widget.sizeHint())
                self.history_list.setItemWidget(item, widget)
                if entry.path == selected_path:
                    selected_row = row
            if selected_row is not None:
                self.history_list.setCurrentRow(selected_row)

        preserve_scroll_position(self.history_list, _rebuild)

    def _set_line_text(self, widget: QLineEdit, value: str) -> None:
        set_line_edit_text(widget, value, skip_if_focused=True, block_signals=True)

    def _set_plain_text(self, widget: QPlainTextEdit, value: str) -> None:
        set_text_document(widget, value, skip_if_focused=True, block_signals=True, stick_to_bottom_if_at_bottom=False)

    def _set_checkbox(self, widget: QCheckBox, value: bool) -> None:
        with QSignalBlocker(widget):
            widget.setChecked(bool(value))

    def _set_spin(self, widget: QSpinBox, value: int) -> None:
        with QSignalBlocker(widget):
            widget.setValue(int(value))

    def _set_double(self, widget: QDoubleSpinBox, value: float) -> None:
        with QSignalBlocker(widget):
            widget.setValue(float(value))

    def _set_combo_value(self, widget: QComboBox, value: str) -> None:
        target = str(value)
        index = widget.findData(target)
        if index < 0:
            index = widget.findText(target)
        if index < 0:
            return
        with QSignalBlocker(widget):
            widget.setCurrentIndex(index)

    def _apply_responsive_layout(self) -> None:
        available_width = self.scroll_area.viewport().width() if self.scroll_area is not None else self.width()
        mode = "stacked" if int(available_width) < 1240 else "wide"
        if mode == self._responsive_layout_mode:
            return
        self._responsive_layout_mode = mode
        if mode == "stacked":
            self.top_row.setDirection(QBoxLayout.TopToBottom)
            self.bottom_row.setDirection(QBoxLayout.TopToBottom)
        else:
            self.top_row.setDirection(QBoxLayout.LeftToRight)
            self.bottom_row.setDirection(QBoxLayout.LeftToRight)
        self.top_row.setStretch(0, 3)
        self.top_row.setStretch(1, 2)
        self.bottom_row.setStretch(0, 3)
        self.bottom_row.setStretch(1, 2)

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        self._apply_responsive_layout()


class RepeatabilityDatasetPage(ExperimentPageBase):
    show_visualization = True
    page_hint = (
        "Run the thesis-facing repeatability dataset by revisiting the same commanded targets from different prior states. "
        "This page focuses on per-target spread, path dependence, and how close the system is to the < 1 mm repeatability goal."
    )

    def __init__(self, controller, experiment_name: str, parent=None) -> None:
        super().__init__(controller, experiment_name, parent)
        self.run_button.setText("Run Repeatability Dataset")

    def _build_parameter_sections(self) -> None:
        mode_card = ExperimentCard(
            "Run Setup",
            "Repeatability uses repeated target revisits. Registration is preferred so the saved metrics land in robot frame; without it the run still saves, but it is marked partial success.",
        )
        mode_form = QFormLayout()
        self.dry_run_check = QCheckBox("Dry Run")
        self.dry_run_check.toggled.connect(lambda value: self.controller.set_config_value("dry_run", bool(value)))
        self.tool_id_edit = QLineEdit()
        self.tool_id_edit.editingFinished.connect(lambda: self.controller.set_config_value("tool_id", self.tool_id_edit.text().strip() or "0A"))
        self.target_set_combo = QComboBox()
        self.target_set_combo.addItem("Single-Segment Ring 17", "single_segment_ring_17")
        self.target_set_combo.addItem("Manual Targets", "manual")
        self.target_set_combo.currentIndexChanged.connect(self._on_target_set_changed)
        self.low_magnitude_spin = QDoubleSpinBox()
        self.low_magnitude_spin.setRange(0.0, 10.0)
        self.low_magnitude_spin.setDecimals(3)
        self.low_magnitude_spin.setSingleStep(0.01)
        self.low_magnitude_spin.valueChanged.connect(
            lambda value: self.controller.set_config_value("schedule.low_magnitude_cm", float(value))
        )
        self.high_magnitude_spin = QDoubleSpinBox()
        self.high_magnitude_spin.setRange(0.0, 10.0)
        self.high_magnitude_spin.setDecimals(3)
        self.high_magnitude_spin.setSingleStep(0.01)
        self.high_magnitude_spin.valueChanged.connect(
            lambda value: self.controller.set_config_value("schedule.high_magnitude_cm", float(value))
        )
        self.seed_spin = QSpinBox()
        self.seed_spin.setRange(0, 999999)
        self.seed_spin.valueChanged.connect(lambda value: self.controller.set_config_value("schedule.seed", int(value)))
        mode_form.addRow("Mode", self.dry_run_check)
        mode_form.addRow("Tool ID", self.tool_id_edit)
        mode_form.addRow("Target Set", self.target_set_combo)
        mode_form.addRow("Low Magnitude (cm)", self.low_magnitude_spin)
        mode_form.addRow("High Magnitude (cm)", self.high_magnitude_spin)
        mode_form.addRow("Random Seed", self.seed_spin)
        mode_card.body_layout.addLayout(mode_form)

        schedule_card = ExperimentCard(
            "Revisit Schedule",
            "Each target is revisited from different prior states. Tune the revisit count, settle time, and sampling depth without editing raw YAML.",
        )
        schedule_form = QFormLayout()
        self.revisit_count_spin = QSpinBox()
        self.revisit_count_spin.setRange(1, 100)
        self.revisit_count_spin.valueChanged.connect(lambda value: self.controller.set_config_value("schedule.revisit_count", int(value)))
        self.samples_per_point_spin = QSpinBox()
        self.samples_per_point_spin.setRange(1, 50)
        self.samples_per_point_spin.valueChanged.connect(lambda value: self.controller.set_config_value("schedule.samples_per_point", int(value)))
        self.settle_time_spin = QDoubleSpinBox()
        self.settle_time_spin.setRange(0.0, 60.0)
        self.settle_time_spin.setDecimals(3)
        self.settle_time_spin.setSingleStep(0.05)
        self.settle_time_spin.valueChanged.connect(lambda value: self.controller.set_config_value("schedule.settle_time_s", float(value)))
        self.randomize_check = QCheckBox("Randomize approach order")
        self.randomize_check.toggled.connect(
            lambda value: self.controller.set_config_value("schedule.randomize_approach_order", bool(value))
        )
        schedule_form.addRow("Revisit Count", self.revisit_count_spin)
        schedule_form.addRow("Samples / Visit", self.samples_per_point_spin)
        schedule_form.addRow("Settle Time (s)", self.settle_time_spin)
        schedule_form.addRow("Approach Order", self.randomize_check)
        schedule_card.body_layout.addLayout(schedule_form)
        target_label = QLabel("Manual Targets (YAML list)")
        target_label.setProperty("role", "muted")
        self.target_points_edit = QPlainTextEdit()
        self.target_points_edit.setMinimumHeight(150)
        self.target_points_edit.textChanged.connect(
            lambda: self.controller.set_parameter_value("schedule.target_points_cm", self.target_points_edit.toPlainText())
        )
        self.manual_target_label = target_label
        schedule_card.body_layout.addWidget(self.manual_target_label)
        schedule_card.body_layout.addWidget(self.target_points_edit)

        summary_card = ExperimentCard(
            "Schedule Preview",
            "Preview the target set, planned revisit count, and grouped target classes before running.",
        )
        self.schedule_summary_widget = KeyValueSummaryWidget()
        summary_card.body_layout.addWidget(self.schedule_summary_widget)

        targets_card = ExperimentCard(
            "Target Catalog",
            "Review the labeled repeatability targets and how many different prior states will feed each revisit cluster.",
        )
        self.target_table = QTableWidget(0, 5)
        self.target_table.setHorizontalHeaderLabels(
            ["Label", "Groups", "Command (cm)", "Planned Visits", "Unique Prior States"]
        )
        self.target_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.target_table.setSelectionMode(QAbstractItemView.NoSelection)
        self.target_table.verticalHeader().setVisible(False)
        self.target_table.horizontalHeader().setStretchLastSection(True)
        self.target_table.setMinimumHeight(220)
        targets_card.body_layout.addWidget(self.target_table)

        self.parameter_layout.addWidget(mode_card)
        self.parameter_layout.addWidget(schedule_card)
        self.parameter_layout.addWidget(summary_card)
        self.parameter_layout.addWidget(targets_card)

    def _sync_parameters_from_state(self, state: ExperimentViewState) -> None:
        _ = state
        config = self._repeatability_config()
        preview = self._current_preview(config=config)
        self._set_checkbox(self.dry_run_check, bool(config.dry_run))
        self._set_line_text(self.tool_id_edit, str(config.tool_id or "0A"))
        self._set_combo_value(self.target_set_combo, str(config.schedule.target_set or "single_segment_ring_17"))
        self._set_double(self.low_magnitude_spin, float(config.schedule.low_magnitude_cm))
        self._set_double(self.high_magnitude_spin, float(config.schedule.high_magnitude_cm))
        self._set_spin(self.seed_spin, int(config.schedule.seed))
        self._set_spin(self.revisit_count_spin, int(config.schedule.revisit_count))
        self._set_spin(self.samples_per_point_spin, int(config.schedule.samples_per_point))
        self._set_double(self.settle_time_spin, float(config.schedule.settle_time_s))
        self._set_checkbox(self.randomize_check, bool(config.schedule.randomize_approach_order))
        self._set_plain_text(self.target_points_edit, _yaml_block(config.schedule.target_points_cm))
        manual_mode = str(config.schedule.target_set or "single_segment_ring_17") == "manual"
        self.manual_target_label.setVisible(manual_mode)
        self.target_points_edit.setVisible(manual_mode)
        self.low_magnitude_spin.setEnabled(not manual_mode)
        self.high_magnitude_spin.setEnabled(not manual_mode)
        self._sync_target_table(preview)
        self._sync_schedule_summary(preview)

    def _on_target_set_changed(self) -> None:
        self.controller.set_config_value("schedule.target_set", str(self.target_set_combo.currentData()))
        try:
            self.set_state(self.controller.refresh())
        except Exception:
            return

    def _repeatability_config(self) -> RepeatabilityDatasetConfig:
        return RepeatabilityDatasetConfig.from_dict(self.controller.config_payload())

    def _current_preview(self, *, config: RepeatabilityDatasetConfig | None = None) -> RepeatabilityPreview:
        config = config or self._repeatability_config()
        try:
            tendon_count = len(self.controller.settings.robot.tendon_to_servo or self.controller.settings.robot.servo_ids)
            return build_repeatability_preview(config, tendon_count=tendon_count)
        except Exception:
            return RepeatabilityPreview(target_catalog=[], visits=[], summary={})

    def _sync_target_table(self, preview: RepeatabilityPreview) -> None:
        visits_by_target = dict(preview.summary.get("visits_by_target", {}) or {})
        approach_counts = dict(preview.summary.get("unique_approach_counts", {}) or {})
        signature = tuple(
            (
                str(entry.get("label", f"T{row + 1:02d}")),
                ", ".join(str(tag) for tag in entry.get("group_tags", []) or []) or str(entry.get("axis_class", "")),
                _render_inline_list(entry.get("tendon_deltas_cm", [])),
                str(int(visits_by_target.get(str(entry.get("target_index")), 0) or 0)),
                str(int(approach_counts.get(str(entry.get("target_index")), 0) or 0)),
            )
            for row, entry in enumerate(preview.target_catalog)
        )
        if getattr(self, "_target_table_signature", None) == signature:
            return
        self._target_table_signature = signature

        def _rebuild() -> None:
            with QSignalBlocker(self.target_table):
                self.target_table.setRowCount(len(preview.target_catalog))
                for row, entry in enumerate(preview.target_catalog):
                    target_key = str(entry.get("target_index"))
                    groups = ", ".join(str(tag) for tag in entry.get("group_tags", []) or []) or str(entry.get("axis_class", ""))
                    cells = [
                        str(entry.get("label", f"T{row + 1:02d}")),
                        groups,
                        _render_inline_list(entry.get("tendon_deltas_cm", [])),
                        str(int(visits_by_target.get(target_key, 0) or 0)),
                        str(int(approach_counts.get(target_key, 0) or 0)),
                    ]
                    for column, text in enumerate(cells):
                        self.target_table.setItem(row, column, QTableWidgetItem(text))

        preserve_scroll_position(self.target_table, _rebuild)

    def _sync_schedule_summary(self, preview: RepeatabilityPreview) -> None:
        axis_counts = dict(preview.summary.get("axis_counts", {}) or {})
        magnitude_counts = dict(preview.summary.get("magnitude_counts", {}) or {})
        self.schedule_summary_widget.set_pairs(
            [
                ("Target Set", str(preview.summary.get("target_set", "n/a")).replace("_", " ")),
                ("Targets", str(int(preview.summary.get("target_count", 0) or 0))),
                ("Planned Revisits", str(int(preview.summary.get("visit_count", 0) or 0))),
                ("Planned Samples", str(int(preview.summary.get("planned_sample_count", 0) or 0))),
                (
                    "Axis Groups",
                    f"on-axis={int(axis_counts.get('on_axis', 0) or 0)}, "
                    f"off-axis={int(axis_counts.get('off_axis', 0) or 0)}",
                ),
                (
                    "Magnitude Groups",
                    f"low={int(magnitude_counts.get('low', 0) or 0)}, "
                    f"high={int(magnitude_counts.get('high', 0) or 0)}",
                ),
                ("Thesis Goal", "< 1.0 mm overall repeatability RMS"),
            ]
        )


class AuroraGridAccuracyPage(ExperimentPageBase):
    show_visualization = True
    page_hint = (
        "Capture labeled 0B points on the physical mesh in any order, then fit the measured centroids "
        "to an ideal truth grid in code. The board's tracker-frame origin does not matter."
    )

    def __init__(self, controller, experiment_name: str, parent=None) -> None:
        self._selected_target_index = 0
        self._capture_log_lines: list[str] = []
        self._point_table_signature: tuple[tuple[str, str, str, str, str, str, str], ...] | None = None
        self._preview_cache_key: str | None = None
        self._preview_cache: GridAccuracyPreview | None = None
        super().__init__(controller, experiment_name, parent)
        self.run_button.setText("Save Dataset")
        self.refresh_button.setText("Refresh Tracker State")
        self.stop_button.hide()

    def _build_parameter_sections(self) -> None:
        params_card = ExperimentCard(
            "Grid Parameters",
            "Set the ideal mesh geometry and the per-point capture settings used for the aligned residual analysis.",
        )
        params_form = QFormLayout()
        self.dry_run_check = QCheckBox("Dry Run")
        self.dry_run_check.toggled.connect(self._on_mode_changed)
        self.tool_id_edit = QLineEdit("0B")
        self.tool_id_edit.setReadOnly(True)
        self.rows_spin = QSpinBox()
        self.rows_spin.setRange(1, 20)
        self.rows_spin.valueChanged.connect(self._apply_grid_geometry)
        self.cols_spin = QSpinBox()
        self.cols_spin.setRange(1, 20)
        self.cols_spin.valueChanged.connect(self._apply_grid_geometry)
        self.spacing_spin = QDoubleSpinBox()
        self.spacing_spin.setRange(0.1, 500.0)
        self.spacing_spin.setDecimals(3)
        self.spacing_spin.setSingleStep(1.0)
        self.spacing_spin.valueChanged.connect(self._apply_grid_geometry)
        self.samples_spin = QSpinBox()
        self.samples_spin.setRange(1, 50)
        self.samples_spin.valueChanged.connect(lambda value: self.controller.set_config_value("samples_per_point", int(value)))
        self.settle_time_spin = QDoubleSpinBox()
        self.settle_time_spin.setRange(0.0, 10.0)
        self.settle_time_spin.setDecimals(3)
        self.settle_time_spin.setSingleStep(0.05)
        self.settle_time_spin.valueChanged.connect(lambda value: self.controller.set_config_value("settle_time_s", float(value)))
        self.outlier_threshold_spin = QDoubleSpinBox()
        self.outlier_threshold_spin.setRange(0.05, 25.0)
        self.outlier_threshold_spin.setDecimals(3)
        self.outlier_threshold_spin.setSingleStep(0.1)
        self.outlier_threshold_spin.valueChanged.connect(
            lambda value: self.controller.set_config_value("outlier_threshold_mm", float(value))
        )
        self.use_tip_check = QCheckBox("Use tip calibration")
        self.use_tip_check.toggled.connect(lambda value: self.controller.set_config_value("use_tip_calibration", bool(value)))
        self.allow_fallback_check = QCheckBox("Allow coil-origin fallback")
        self.allow_fallback_check.toggled.connect(
            lambda value: self.controller.set_config_value("allow_coil_origin_fallback", bool(value))
        )
        self.tip_vector_edit = QLineEdit()
        self.tip_vector_edit.editingFinished.connect(
            lambda: _apply_inline_list_edit(self.controller, "tip_vector_mm", self.tip_vector_edit.text())
        )
        params_form.addRow("Mode", self.dry_run_check)
        params_form.addRow("Tool ID", self.tool_id_edit)
        params_form.addRow("Rows", self.rows_spin)
        params_form.addRow("Cols", self.cols_spin)
        params_form.addRow("Spacing (mm)", self.spacing_spin)
        params_form.addRow("Samples / Point", self.samples_spin)
        params_form.addRow("Settle Time (s)", self.settle_time_spin)
        params_form.addRow("Outlier Threshold (mm)", self.outlier_threshold_spin)
        params_form.addRow("Tip Calibration", self.use_tip_check)
        params_form.addRow("Fallback", self.allow_fallback_check)
        params_form.addRow("Tip Vector (mm)", self.tip_vector_edit)
        params_card.body_layout.addLayout(params_form)

        capture_card = ExperimentCard(
            "Labeled Point Capture",
            "Select any truth-grid label, capture a batch of 0B samples, and review centroid/spread status before saving the dataset.",
        )
        selection_row = QHBoxLayout()
        selection_row.setContentsMargins(0, 0, 0, 0)
        selection_row.setSpacing(10)
        self.selected_point_label = QLabel("Selected Point: P01")
        self.selected_point_label.setProperty("role", "section-title")
        selection_row.addWidget(self.selected_point_label, 1)
        self.capture_selected_button = QPushButton("Capture Selected Point")
        self.capture_selected_button.setProperty("variant", "primary")
        self.capture_selected_button.clicked.connect(self.capture_selected_point)
        self.clear_selected_button = QPushButton("Clear Selected Point")
        self.clear_selected_button.setProperty("variant", "ghost")
        self.clear_selected_button.clicked.connect(self.clear_selected_point)
        self.clear_all_button = QPushButton("Clear All Points")
        self.clear_all_button.setProperty("variant", "ghost")
        self.clear_all_button.clicked.connect(self.clear_all_points)
        selection_row.addWidget(self.capture_selected_button)
        selection_row.addWidget(self.clear_selected_button)
        selection_row.addWidget(self.clear_all_button)
        capture_card.body_layout.addLayout(selection_row)

        self.selected_point_summary_widget = KeyValueSummaryWidget()
        capture_card.body_layout.addWidget(self.selected_point_summary_widget)

        self.point_table = QTableWidget(0, 7)
        self.point_table.setHorizontalHeaderLabels(
            ["Label", "Truth XY (mm)", "Samples", "Accepted", "Spread", "Residual", "Status"]
        )
        self.point_table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.point_table.setSelectionMode(QAbstractItemView.SingleSelection)
        self.point_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.point_table.verticalHeader().setVisible(False)
        self.point_table.horizontalHeader().setStretchLastSection(True)
        self.point_table.itemSelectionChanged.connect(self._on_point_selection_changed)
        self.point_table.setMinimumHeight(250)
        capture_card.body_layout.addWidget(self.point_table)

        self.capture_summary_widget = KeyValueSummaryWidget()
        capture_card.body_layout.addWidget(self.capture_summary_widget)

        self.capture_status_text = QPlainTextEdit()
        self.capture_status_text.setReadOnly(True)
        self.capture_status_text.setMinimumHeight(96)
        self.capture_status_text.setMaximumHeight(120)
        self.capture_status_text.setPlaceholderText("Capture status and session notes will appear here.")
        capture_card.body_layout.addWidget(self.capture_status_text)

        self.parameter_layout.addWidget(params_card)
        self.parameter_layout.addWidget(capture_card)

    def _sync_parameters_from_state(self, state: ExperimentViewState) -> None:
        _ = state
        config = self._grid_config()
        dims = list(config.dimensions) or [3, 3]
        if len(dims) == 1:
            dims = [dims[0], 1]
        if len(dims) == 2:
            cols, rows = int(dims[0]), int(dims[1])
        else:
            cols, rows = int(dims[0]), int(dims[1])
        self._set_checkbox(self.dry_run_check, bool(config.dry_run))
        self._set_line_text(self.tool_id_edit, str(config.tool_id or "0B"))
        self._set_spin(self.rows_spin, rows)
        self._set_spin(self.cols_spin, cols)
        self._set_double(self.spacing_spin, float(config.spacing_mm))
        self._set_spin(self.samples_spin, int(config.samples_per_point))
        self._set_double(self.settle_time_spin, float(config.settle_time_s))
        self._set_double(self.outlier_threshold_spin, float(config.outlier_threshold_mm))
        self._set_checkbox(self.use_tip_check, bool(config.use_tip_calibration))
        self._set_checkbox(self.allow_fallback_check, bool(config.allow_coil_origin_fallback))
        self._set_line_text(
            self.tip_vector_edit,
            _render_inline_list(config.tip_vector_mm if config.tip_vector_mm is not None else [0.0, 0.0, 125.0]),
        )
        preview = self._current_preview(config=config)
        self._sync_point_table(preview, expected_samples=int(config.samples_per_point))
        self._sync_selected_point_summary(preview, expected_samples=int(config.samples_per_point))
        self._sync_capture_summary(preview, expected_samples=int(config.samples_per_point))
        self._sync_capture_status()
        self.capture_selected_button.setEnabled(not state.run_active)
        self.clear_selected_button.setEnabled(not state.run_active)
        self.clear_all_button.setEnabled(not state.run_active)

    def _on_mode_changed(self, value: bool) -> None:
        self.controller.set_config_value("dry_run", bool(value))
        self._refresh_now()

    def _apply_grid_geometry(self) -> None:
        dims = [int(self.cols_spin.value()), int(self.rows_spin.value())]
        previous_dims = [int(value) for value in (self.controller.get_config_value("dimensions", [3, 3]) or [3, 3])]
        previous_spacing = float(self.controller.get_config_value("spacing_mm", 25.4))
        spacing = float(self.spacing_spin.value())
        geometry_changed = dims != previous_dims or not np.isclose(spacing, previous_spacing)
        self.controller.set_config_value("dimensions", dims)
        self.controller.set_config_value("spacing_mm", spacing)
        self.controller.set_config_value("truth_points_file", None)
        self.controller.set_config_value("truth_points_mm", [])
        if geometry_changed and self.controller.get_config_value("captured_points", []):
            self.controller.set_config_value("captured_points", [])
            self._selected_target_index = 0
            self._append_capture_log("Grid geometry changed. Cleared previously captured points.")
        self._refresh_now()

    def _on_point_selection_changed(self) -> None:
        selected_items = self.point_table.selectedItems()
        if not selected_items:
            return
        row = selected_items[0].row()
        target_index = self.point_table.item(row, 0).data(Qt.UserRole)
        if target_index is None:
            return
        self._selected_target_index = int(target_index)
        preview = self._current_preview()
        expected_samples = max(1, int(self._grid_config().samples_per_point))
        self._update_selected_point_label(truth_catalog=preview.truth_catalog)
        self._sync_selected_point_summary(preview, expected_samples=expected_samples)
        self._sync_capture_summary(preview, expected_samples=expected_samples)

    def capture_selected_point(self) -> None:
        config = self._grid_config()
        preview = self._current_preview(config=config)
        truth_entry = self._selected_truth_entry(preview.truth_catalog)
        if truth_entry is None:
            return
        had_existing_capture = any(
            isinstance(record, dict) and int(record.get("target_index", -1)) == int(truth_entry["target_index"])
            for record in (self.controller.get_config_value("captured_points", []) or [])
        )
        try:
            raw_samples = self._collect_point_samples(config=config, truth_entry=truth_entry)
        except Exception as exc:
            self._append_capture_log(f"{truth_entry['label']}: capture failed: {exc}")
            self._refresh_now()
            return
        captured_points = [
            dict(record)
            for record in (self.controller.get_config_value("captured_points", []) or [])
            if isinstance(record, dict)
        ]
        captured_points = [record for record in captured_points if int(record.get("target_index", -1)) != int(truth_entry["target_index"])]
        captured_points.append(
            {
                "label": str(truth_entry["label"]),
                "target_index": int(truth_entry["target_index"]),
                "truth_point_mm": [float(value) for value in truth_entry["truth_point_mm"]],
                "raw_samples": raw_samples,
            }
        )
        captured_points.sort(key=lambda record: int(record.get("target_index", 0)))
        self.controller.set_config_value("captured_points", captured_points)
        updated_config = self._grid_config()
        updated_preview = self._current_preview(config=updated_config)
        point_metrics = (updated_preview.metrics.get("per_point_metrics", {}) or {}).get(str(truth_entry["label"]), {})
        capture_action = "recaptured" if had_existing_capture else "captured"
        capture_message = (
            f"{truth_entry['label']}: {capture_action} {len(raw_samples)} raw sample(s), "
            f"{int(point_metrics.get('accepted_sample_count', 0) or 0)} accepted, "
            f"spread {_fmt_metric(point_metrics.get('sample_spread_rms_mm'))} mm, "
            f"residual {_fmt_metric(point_metrics.get('residual_mm'))} mm."
        )
        if not had_existing_capture:
            next_target_index = self._next_incomplete_target_index(
                updated_preview,
                expected_samples=max(1, int(updated_config.samples_per_point)),
                start_after=int(truth_entry["target_index"]),
            )
            if next_target_index is not None and next_target_index != int(truth_entry["target_index"]):
                self._selected_target_index = int(next_target_index)
                next_entry = updated_preview.truth_catalog[next_target_index]
                capture_message += f" Next suggested point: {next_entry['label']}."
        self._append_capture_log(capture_message)
        self._refresh_now()

    def clear_selected_point(self) -> None:
        target_index = int(self._selected_target_index)
        label = f"P{target_index + 1:02d}"
        captured_points = [
            dict(record)
            for record in (self.controller.get_config_value("captured_points", []) or [])
            if isinstance(record, dict)
        ]
        kept_points = [record for record in captured_points if int(record.get("target_index", -1)) != target_index]
        if len(kept_points) == len(captured_points):
            return
        self.controller.set_config_value("captured_points", kept_points)
        self._append_capture_log(f"{label}: cleared captured samples.")
        self._refresh_now()

    def clear_all_points(self) -> None:
        if not self.controller.get_config_value("captured_points", []):
            return
        self.controller.set_config_value("captured_points", [])
        self._selected_target_index = 0
        self._append_capture_log("Cleared all captured grid points.")
        self._refresh_now()

    def _collect_point_samples(self, *, config: GridDefinitionConfig, truth_entry: dict[str, object]) -> list[dict[str, object]]:
        tip_vector_mm, tip_available = resolve_grid_tip_vector(config, project_root=self.controller.project_root)
        if config.use_tip_calibration and not tip_available and not config.allow_coil_origin_fallback:
            raise RuntimeError("Tip calibration is required for this grid capture.")
        sample_count = max(1, int(config.samples_per_point))
        raw_samples: list[dict[str, object]] = []
        for sample_index in range(sample_count):
            if config.dry_run:
                raw_sample = self._synthetic_grid_sample(config=config, truth_entry=truth_entry, sample_index=sample_index)
            else:
                snapshot = self.controller.tracking_service.get_snapshot()
                raw_sample = capture_grid_measurement_from_snapshot(
                    snapshot,
                    tool_id=str(config.tool_id or "0B"),
                    tip_vector_mm=tip_vector_mm if tip_available else None,
                    require_tip_calibration=bool(config.use_tip_calibration),
                    allow_coil_origin_fallback=bool(config.allow_coil_origin_fallback),
                )
            raw_sample["monotonic_time_s"] = float(self.controller.experiment_runner.monotonic_fn())
            raw_sample["wall_time_utc"] = datetime.now(timezone.utc).isoformat()
            raw_samples.append(raw_sample)
            if sample_index + 1 < sample_count and float(config.settle_time_s) > 0.0:
                self.controller.experiment_runner.sleep_fn(float(config.settle_time_s))
        return raw_samples

    def _synthetic_grid_sample(
        self,
        *,
        config: GridDefinitionConfig,
        truth_entry: dict[str, object],
        sample_index: int,
    ) -> dict[str, object]:
        if config.use_tip_calibration and config.tip_vector_mm is None and not config.allow_coil_origin_fallback:
            raise RuntimeError("Tip calibration is required for this grid capture.")
        truth_point = np.asarray(truth_entry["truth_point_mm"], dtype=float)
        rng = np.random.default_rng(int(config.seed) + (int(truth_entry["target_index"]) * 100) + sample_index)
        position = truth_point + np.asarray(config.synthetic_bias_mm, dtype=float) + rng.normal(
            0.0,
            float(config.synthetic_noise_std_mm),
            size=3,
        )
        return {
            "position_mm": [float(value) for value in position.tolist()],
            "tool_translation_mm": [float(value) for value in position.tolist()],
            "quaternion_wxyz": [1.0, 0.0, 0.0, 0.0],
            "tracker_frame_id": sample_index,
            "freshness_s": 0.0,
            "tracking_state": "valid",
            "status_flags": ["dry_run"],
            "position_source": "tip" if config.tip_vector_mm is not None else "coil_origin",
        }

    def _current_preview(self, *, config: GridDefinitionConfig | None = None):
        config = config or self._grid_config()
        cache_key = repr(config)
        if self._preview_cache_key == cache_key and self._preview_cache is not None:
            return self._preview_cache
        try:
            preview = build_grid_accuracy_preview(config, project_root=self.controller.project_root)
        except Exception as exc:
            if not self._capture_log_lines or "preview failed" not in self._capture_log_lines[-1]:
                self._append_capture_log(f"Grid preview failed: {exc}")
            preview = GridAccuracyPreview(
                truth_catalog=build_grid_truth_catalog(config, project_root=self.controller.project_root),
                samples=[],
                metrics={},
            )
        self._preview_cache_key = cache_key
        self._preview_cache = preview
        return preview

    def _grid_config(self) -> GridDefinitionConfig:
        return GridDefinitionConfig.from_dict(self.controller.config_payload())

    def _selected_truth_entry(self, truth_catalog: list[dict[str, object]]) -> dict[str, object] | None:
        if not truth_catalog:
            return None
        self._selected_target_index = max(0, min(int(self._selected_target_index), len(truth_catalog) - 1))
        return truth_catalog[self._selected_target_index]

    def _sync_point_table(self, preview, *, expected_samples: int) -> None:
        truth_catalog = list(preview.truth_catalog)
        metric_rows = preview.metrics.get("per_point_metrics", {}) or {}
        captured_rows = {
            int(record.get("target_index", -1)): record
            for record in (self.controller.get_config_value("captured_points", []) or [])
            if isinstance(record, dict)
        }
        rows_signature = tuple(
            self._point_row_signature(
                entry=entry,
                point_metrics=metric_rows.get(str(entry["label"]), {}),
                point_record=captured_rows.get(int(entry["target_index"]), {}),
                expected_samples=expected_samples,
            )
            for entry in truth_catalog
        )
        if rows_signature == self._point_table_signature:
            if truth_catalog:
                with QSignalBlocker(self.point_table):
                    self.point_table.selectRow(self._selected_target_index)
            self._update_selected_point_label(truth_catalog=truth_catalog)
            return
        self._point_table_signature = rows_signature

        def _rebuild() -> None:
            with QSignalBlocker(self.point_table):
                self.point_table.setRowCount(len(truth_catalog))
                for row, entry in enumerate(truth_catalog):
                    label = str(entry["label"])
                    point_metrics = metric_rows.get(label, {})
                    point_record = captured_rows.get(int(entry["target_index"]), {})
                    sample_count = len(point_record.get("raw_samples", []) or [])
                    accepted_count = int(point_metrics.get("accepted_sample_count", 0) or 0)
                    rejected_count = int(point_metrics.get("outlier_count", 0) or 0)
                    spread_text = _fmt_metric(point_metrics.get("sample_spread_rms_mm"))
                    residual_text = _fmt_metric(point_metrics.get("residual_mm"))
                    status = self._point_capture_status(
                        sample_count=sample_count,
                        expected_samples=expected_samples,
                        rejected_count=rejected_count,
                    )
                    truth_xy = entry.get("truth_point_mm", [0.0, 0.0, 0.0])
                    cells = [
                        (label, entry["target_index"]),
                        (f"{truth_xy[0]:.1f}, {truth_xy[1]:.1f}", None),
                        (str(sample_count), None),
                        (str(accepted_count), None),
                        (spread_text, None),
                        (residual_text, None),
                        (status, None),
                    ]
                    for column, (text, user_data) in enumerate(cells):
                        item = QTableWidgetItem(str(text))
                        if user_data is not None:
                            item.setData(Qt.UserRole, user_data)
                        self._style_point_table_item(item, status=status)
                        self.point_table.setItem(row, column, item)
                if truth_catalog:
                    self.point_table.selectRow(self._selected_target_index)

        preserve_scroll_position(self.point_table, _rebuild)
        self._update_selected_point_label(truth_catalog=truth_catalog)

    def _sync_capture_summary(self, preview, *, expected_samples: int) -> None:
        metrics = preview.metrics
        captured_points = [
            record
            for record in (self.controller.get_config_value("captured_points", []) or [])
            if isinstance(record, dict)
        ]
        complete_points = sum(
            1 for record in captured_points if len(record.get("raw_samples", []) or []) >= expected_samples
        )
        partial_points = sum(
            1 for record in captured_points if 0 < len(record.get("raw_samples", []) or []) < expected_samples
        )
        selected = self._selected_truth_entry(preview.truth_catalog)
        selected_label = str(selected["label"]) if selected is not None else "n/a"
        next_target_index = self._next_incomplete_target_index(
            preview,
            expected_samples=expected_samples,
            start_after=int(self._selected_target_index),
        )
        next_label = (
            str(preview.truth_catalog[next_target_index]["label"])
            if next_target_index is not None and preview.truth_catalog
            else "All complete"
        )
        self.capture_summary_widget.set_pairs(
            [
                ("Selected Point", selected_label),
                ("Coverage", f"{complete_points} complete, {partial_points} partial, {len(preview.truth_catalog) - complete_points - partial_points} not started"),
                ("Solve Ready", "Yes" if metrics.get("alignment_ready") else "No"),
                ("Suggested Next", next_label),
                ("Raw Samples", str(int(metrics.get("raw_sample_count", 0) or 0))),
                ("Accepted Samples", str(int(metrics.get("accepted_sample_count", 0) or 0))),
                ("Rejected Samples", str(int(metrics.get("rejected_sample_count", metrics.get("outlier_count", 0)) or 0))),
                ("RMS Residual", _fmt_metric(metrics.get("overall_rms_residual_mm"))),
                ("Max Residual", _fmt_metric(metrics.get("max_residual_mm"))),
                ("Mean Spread", _fmt_metric(metrics.get("mean_within_point_spread_mm"))),
                ("Max Spread", _fmt_metric(metrics.get("max_within_point_spread_mm"))),
            ]
        )

    def _sync_selected_point_summary(self, preview, *, expected_samples: int) -> None:
        selected = self._selected_truth_entry(preview.truth_catalog)
        if selected is None:
            self.selected_point_summary_widget.set_pairs([("Selected Point", "n/a")])
            return
        label = str(selected["label"])
        target_index = int(selected["target_index"])
        point_metrics = (preview.metrics.get("per_point_metrics", {}) or {}).get(label, {})
        captured_rows = {
            int(record.get("target_index", -1)): record
            for record in (self.controller.get_config_value("captured_points", []) or [])
            if isinstance(record, dict)
        }
        point_record = captured_rows.get(target_index, {})
        raw_samples = list(point_record.get("raw_samples", []) or [])
        sample_count = len(raw_samples)
        accepted_count = int(point_metrics.get("accepted_sample_count", 0) or 0)
        rejected_count = int(point_metrics.get("outlier_count", 0) or 0)
        position_sources = sorted(
            {
                str(sample.get("position_source", "tracker_tool"))
                for sample in raw_samples
                if isinstance(sample, dict)
            }
        )
        next_target_index = self._next_incomplete_target_index(
            preview,
            expected_samples=expected_samples,
            start_after=target_index,
        )
        suggested_next = (
            str(preview.truth_catalog[next_target_index]["label"])
            if next_target_index is not None and next_target_index != target_index
            else "None"
        )
        truth_point = selected.get("truth_point_mm", [0.0, 0.0, 0.0])
        self.selected_point_summary_widget.set_pairs(
            [
                ("Selected Point", label),
                ("Truth Point", f"{truth_point[0]:.1f}, {truth_point[1]:.1f}, {truth_point[2]:.1f} mm"),
                (
                    "Capture Status",
                    self._point_capture_status(
                        sample_count=sample_count,
                        expected_samples=expected_samples,
                        rejected_count=rejected_count,
                    ),
                ),
                ("Raw Samples", str(sample_count)),
                ("Accepted Samples", str(accepted_count)),
                ("Rejected Samples", str(rejected_count)),
                ("Within-Point Spread", _fmt_metric(point_metrics.get("sample_spread_rms_mm"))),
                ("Aligned Residual", _fmt_metric(point_metrics.get("residual_mm"))),
                ("Position Source", ", ".join(position_sources) or "n/a"),
                ("Suggested Next", suggested_next),
            ]
        )

    def _sync_capture_status(self) -> None:
        lines = self._capture_log_lines or [
            "Select any grid label and capture a full sample batch. The aligned residual summary will update once at least three points are complete."
        ]
        set_text_document(
            self.capture_status_text,
            "\n".join(lines[-8:]),
            block_signals=True,
            stick_to_bottom_if_at_bottom=True,
        )

    def _append_capture_log(self, message: str) -> None:
        timestamp = datetime.now(timezone.utc).strftime("%H:%M:%S")
        self._capture_log_lines.append(f"[{timestamp}Z] {message}")
        if len(self._capture_log_lines) > 100:
            self._capture_log_lines = self._capture_log_lines[-100:]

    def _refresh_now(self) -> None:
        try:
            self.set_state(self.controller.refresh())
        except Exception:
            return

    def _update_selected_point_label(self, *, truth_catalog: list[dict[str, object]] | None = None) -> None:
        truth_catalog = truth_catalog or self._current_preview().truth_catalog
        selected = self._selected_truth_entry(truth_catalog)
        if selected is None:
            self.selected_point_label.setText("Selected Point: n/a")
            return
        truth_point = selected.get("truth_point_mm", [0.0, 0.0, 0.0])
        self.selected_point_label.setText(
            f"Selected Point: {selected['label']}  •  truth=({truth_point[0]:.1f}, {truth_point[1]:.1f}, {truth_point[2]:.1f}) mm"
        )

    @staticmethod
    def _point_capture_status(*, sample_count: int, expected_samples: int, rejected_count: int) -> str:
        if sample_count <= 0:
            return "Not started"
        if sample_count < expected_samples:
            return f"Partial ({sample_count}/{expected_samples})"
        if rejected_count > 0:
            return f"Complete ({rejected_count} rejected)"
        return "Complete"

    @staticmethod
    def _style_point_table_item(item: QTableWidgetItem, *, status: str) -> None:
        normalized = status.lower()
        if normalized.startswith("complete") and "rejected" not in normalized:
            background = "#dcfce7"
            foreground = "#166534"
        elif normalized.startswith("partial") or "rejected" in normalized:
            background = "#fef3c7"
            foreground = "#92400e"
        else:
            background = "#f8fafc"
            foreground = "#475569"
        item.setBackground(QColor(background))
        item.setForeground(QColor(foreground))

    def _point_row_signature(
        self,
        *,
        entry: dict[str, object],
        point_metrics: dict[str, object],
        point_record: dict[str, object],
        expected_samples: int,
    ) -> tuple[str, str, str, str, str, str, str]:
        sample_count = len(point_record.get("raw_samples", []) or [])
        accepted_count = int(point_metrics.get("accepted_sample_count", 0) or 0)
        rejected_count = int(point_metrics.get("outlier_count", 0) or 0)
        truth_xy = entry.get("truth_point_mm", [0.0, 0.0, 0.0])
        status = self._point_capture_status(
            sample_count=sample_count,
            expected_samples=expected_samples,
            rejected_count=rejected_count,
        )
        return (
            str(entry["label"]),
            f"{truth_xy[0]:.1f}, {truth_xy[1]:.1f}",
            str(sample_count),
            str(accepted_count),
            _fmt_metric(point_metrics.get("sample_spread_rms_mm")),
            _fmt_metric(point_metrics.get("residual_mm")),
            status,
        )

    def _next_incomplete_target_index(
        self,
        preview,
        *,
        expected_samples: int,
        start_after: int,
    ) -> int | None:
        truth_catalog = list(preview.truth_catalog)
        if not truth_catalog:
            return None
        captured_rows = {
            int(record.get("target_index", -1)): record
            for record in (self.controller.get_config_value("captured_points", []) or [])
            if isinstance(record, dict)
        }
        total = len(truth_catalog)
        for offset in range(1, total + 1):
            candidate_index = (int(start_after) + offset) % total
            record = captured_rows.get(candidate_index, {})
            sample_count = len(record.get("raw_samples", []) or [])
            if sample_count < expected_samples:
                return candidate_index
        return None


class PretensionValidationPage(ExperimentPageBase):
    page_hint = (
        "Validate pretension response against commanded travel for one selected servo. "
        "Current is treated conservatively as an engagement proxy only, and optional tracker displacement "
        "can be captured alongside the servo trace when live tracking is available."
    )

    def __init__(self, controller, experiment_name: str, parent=None) -> None:
        super().__init__(controller, experiment_name, parent)
        self.run_button.setText("Run Pretension Validation")

    def _build_parameter_sections(self) -> None:
        setup_card = ExperimentCard(
            "Validation Setup",
            "Keep this page focused on one servo at a time. Use the Pretension tab for tuning; use this page for canonical response-vs-travel datasets.",
        )
        setup_form = QFormLayout()
        self.servo_combo = QComboBox()
        configured_ids = list(self.controller.settings.robot.servo_ids or [])
        if configured_ids:
            for servo_id in configured_ids:
                self.servo_combo.addItem(f"Servo {int(servo_id)}", str(int(servo_id)))
        else:
            self.servo_combo.addItem("Servo 1", "1")
        self.servo_combo.currentIndexChanged.connect(self._on_servo_changed)
        self.include_tracker_check = QCheckBox("Include tracker-side displacement when available")
        self.include_tracker_check.toggled.connect(
            lambda value: self.controller.set_config_value("include_tracker_displacement", bool(value))
        )
        setup_form.addRow("Servo", self.servo_combo)
        setup_form.addRow("Tracker Metric", self.include_tracker_check)
        setup_card.body_layout.addLayout(setup_form)
        note = QLabel(
            "Servo convention: raw XC330 position uses 0..4095 ticks, 4095 is untensioned, and tightening lowers raw counts."
        )
        note.setProperty("role", "muted")
        note.setWordWrap(True)
        setup_card.body_layout.addWidget(note)

        params_card = ExperimentCard(
            "Pretension Parameters",
            "These values mirror the current live pretension parameters. Leave them on the resolved defaults unless you are intentionally validating a changed threshold or travel limit.",
        )
        params_form = QFormLayout()
        self.untensioned_spin = QSpinBox()
        self.untensioned_spin.setRange(0, 4095)
        self.untensioned_spin.valueChanged.connect(
            lambda value: self.controller.set_config_value("untensioned_reference_tick", int(value))
        )
        self.step_ticks_spin = QSpinBox()
        self.step_ticks_spin.setRange(1, 512)
        self.step_ticks_spin.valueChanged.connect(
            lambda value: self.controller.set_config_value("step_ticks", int(value))
        )
        self.settle_time_spin = QDoubleSpinBox()
        self.settle_time_spin.setRange(0.0, 60.0)
        self.settle_time_spin.setDecimals(3)
        self.settle_time_spin.setSingleStep(0.01)
        self.settle_time_spin.valueChanged.connect(
            lambda value: self.controller.set_config_value("settle_time_s", float(value))
        )
        self.baseline_samples_spin = QSpinBox()
        self.baseline_samples_spin.setRange(1, 100)
        self.baseline_samples_spin.valueChanged.connect(
            lambda value: self.controller.set_config_value("baseline_sample_count", int(value))
        )
        self.filter_window_spin = QSpinBox()
        self.filter_window_spin.setRange(1, 25)
        self.filter_window_spin.valueChanged.connect(
            lambda value: self.controller.set_config_value("current_filter_window", int(value))
        )
        self.delta_trigger_spin = QSpinBox()
        self.delta_trigger_spin.setRange(1, 5000)
        self.delta_trigger_spin.valueChanged.connect(
            lambda value: self.controller.set_config_value("current_delta_threshold_ma", int(value))
        )
        self.absolute_trigger_spin = QSpinBox()
        self.absolute_trigger_spin.setRange(0, 5000)
        self.absolute_trigger_spin.setSpecialValueText("Disabled")
        self.absolute_trigger_spin.valueChanged.connect(
            lambda value: self.controller.set_config_value(
                "absolute_trigger_current_ma",
                None if int(value) <= 0 else int(value),
            )
        )
        self.hard_stop_spin = QSpinBox()
        self.hard_stop_spin.setRange(1, 5000)
        self.hard_stop_spin.valueChanged.connect(
            lambda value: self.controller.set_config_value("hard_current_stop_ma", int(value))
        )
        self.max_travel_spin = QSpinBox()
        self.max_travel_spin.setRange(1, 4095)
        self.max_travel_spin.valueChanged.connect(
            lambda value: self.controller.set_config_value("max_travel_ticks", int(value))
        )
        self.timeout_spin = QDoubleSpinBox()
        self.timeout_spin.setRange(0.1, 120.0)
        self.timeout_spin.setDecimals(2)
        self.timeout_spin.setSingleStep(0.1)
        self.timeout_spin.valueChanged.connect(
            lambda value: self.controller.set_config_value("timeout_s", float(value))
        )
        params_form.addRow("Untensioned Reference", self.untensioned_spin)
        params_form.addRow("Step Ticks", self.step_ticks_spin)
        params_form.addRow("Settle Time (s)", self.settle_time_spin)
        params_form.addRow("Baseline Samples", self.baseline_samples_spin)
        params_form.addRow("Filter Window", self.filter_window_spin)
        params_form.addRow("Current Delta Trigger (mA)", self.delta_trigger_spin)
        params_form.addRow("Absolute Trigger (mA)", self.absolute_trigger_spin)
        params_form.addRow("Hard Current Stop (mA)", self.hard_stop_spin)
        params_form.addRow("Max Travel (ticks)", self.max_travel_spin)
        params_form.addRow("Timeout (s)", self.timeout_spin)
        params_card.body_layout.addLayout(params_form)

        summary_card = ExperimentCard(
            "What This Run Produces",
            "Each run writes a canonical time-ordered sample series plus a static response plot and compact summary note under data/experiments/....",
        )
        self.parameter_summary_widget = KeyValueSummaryWidget()
        summary_card.body_layout.addWidget(self.parameter_summary_widget)

        self.parameter_layout.addWidget(setup_card)
        self.parameter_layout.addWidget(params_card)
        self.parameter_layout.addWidget(summary_card)

    def _sync_parameters_from_state(self, state: ExperimentViewState) -> None:
        _ = state
        config = PretensionValidationExperimentConfig.from_dict(self.controller.config_payload())
        defaults = self._resolved_defaults(config.servo_id)
        self._set_combo_value(self.servo_combo, str(int(config.servo_id)))
        self._set_checkbox(self.include_tracker_check, bool(config.include_tracker_displacement))
        self._set_spin(
            self.untensioned_spin,
            int(config.untensioned_reference_tick if config.untensioned_reference_tick is not None else defaults.untensioned_reference_tick),
        )
        self._set_spin(
            self.step_ticks_spin,
            int(config.step_ticks if config.step_ticks is not None else defaults.step_ticks),
        )
        self._set_double(
            self.settle_time_spin,
            float(config.settle_time_s if config.settle_time_s is not None else defaults.settle_time_s),
        )
        self._set_spin(
            self.baseline_samples_spin,
            int(
                config.baseline_sample_count
                if config.baseline_sample_count is not None
                else defaults.baseline_sample_count
            ),
        )
        self._set_spin(
            self.filter_window_spin,
            int(
                config.current_filter_window
                if config.current_filter_window is not None
                else defaults.current_filter_window
            ),
        )
        self._set_spin(
            self.delta_trigger_spin,
            int(
                config.current_delta_threshold_ma
                if config.current_delta_threshold_ma is not None
                else defaults.current_delta_threshold_ma
            ),
        )
        self._set_spin(
            self.absolute_trigger_spin,
            int(
                config.absolute_trigger_current_ma
                if config.absolute_trigger_current_ma is not None
                else (defaults.absolute_trigger_current_ma or 0)
            ),
        )
        self._set_spin(
            self.hard_stop_spin,
            int(config.hard_current_stop_ma if config.hard_current_stop_ma is not None else defaults.hard_current_stop_ma),
        )
        self._set_spin(
            self.max_travel_spin,
            int(config.max_travel_ticks if config.max_travel_ticks is not None else defaults.max_travel_ticks),
        )
        self._set_double(
            self.timeout_spin,
            float(config.timeout_s if config.timeout_s is not None else defaults.timeout_s),
        )
        self.parameter_summary_widget.set_pairs(
            [
                ("Travel Axis", "Travel is plotted from the untensioned reference toward lower raw counts."),
                ("Current Metric", "Current is saved as an engagement proxy, not a tendon-force estimate."),
                (
                    "Tracker Metric",
                    (
                        "Tracker displacement is sampled alongside the servo trace when available."
                        if bool(config.include_tracker_displacement)
                        else "Tracker displacement capture is disabled for this run."
                    ),
                ),
            ]
        )

    def _on_servo_changed(self) -> None:
        raw_servo_id = self.servo_combo.currentData()
        servo_id = int(raw_servo_id) if raw_servo_id not in (None, "") else 1
        self.controller.set_config_value("servo_id", servo_id)
        self._refresh_now()

    def _resolved_defaults(self, servo_id: int):
        try:
            return self.controller.servo_service.default_pretension_parameters(int(servo_id))
        except Exception:
            return self.controller.servo_service.default_pretension_parameters(1)

    def _refresh_now(self) -> None:
        try:
            self.set_state(self.controller.refresh())
        except Exception:
            return


class CommandScheduleValidationPage(ExperimentPageBase):
    page_hint = "Use this page to validate generated sweep, grid, trajectory, or babble command schedules before using them elsewhere."

    def _build_parameter_sections(self) -> None:
        schedule_card = ExperimentCard("Schedule", "Configure the generated command schedule you want to validate.")
        schedule_form = QFormLayout()
        self.kind_combo = QComboBox()
        for label, value in (
            ("Sweep", "sweep"),
            ("Grid", "grid"),
            ("Trajectory", "trajectory"),
            ("Babble", "babble"),
        ):
            self.kind_combo.addItem(label, value)
        self.kind_combo.currentIndexChanged.connect(
            lambda _index: self.controller.set_config_value("schedule.kind", str(self.kind_combo.currentData()))
        )
        self.dimensions_spin = QSpinBox()
        self.dimensions_spin.setRange(1, 16)
        self.dimensions_spin.valueChanged.connect(lambda value: self.controller.set_config_value("schedule.dimensions", int(value)))
        self.amplitude_spin = QDoubleSpinBox()
        self.amplitude_spin.setRange(0.0, 10.0)
        self.amplitude_spin.setDecimals(3)
        self.amplitude_spin.setSingleStep(0.05)
        self.amplitude_spin.valueChanged.connect(lambda value: self.controller.set_config_value("schedule.amplitude_cm", float(value)))
        self.steps_spin = QSpinBox()
        self.steps_spin.setRange(1, 100)
        self.steps_spin.valueChanged.connect(lambda value: self.controller.set_config_value("schedule.steps_per_axis", int(value)))
        self.repeats_spin = QSpinBox()
        self.repeats_spin.setRange(1, 100)
        self.repeats_spin.valueChanged.connect(lambda value: self.controller.set_config_value("schedule.repeats", int(value)))
        self.babble_spin = QSpinBox()
        self.babble_spin.setRange(1, 10000)
        self.babble_spin.valueChanged.connect(lambda value: self.controller.set_config_value("schedule.babble_count", int(value)))
        schedule_form.addRow("Kind", self.kind_combo)
        schedule_form.addRow("Dimensions", self.dimensions_spin)
        schedule_form.addRow("Amplitude (cm)", self.amplitude_spin)
        schedule_form.addRow("Steps / Axis", self.steps_spin)
        schedule_form.addRow("Repeats", self.repeats_spin)
        schedule_form.addRow("Babble Count", self.babble_spin)
        schedule_card.body_layout.addLayout(schedule_form)
        trajectory_label = QLabel("Trajectory Points (YAML list)")
        trajectory_label.setProperty("role", "muted")
        self.trajectory_edit = QPlainTextEdit()
        self.trajectory_edit.setMinimumHeight(140)
        self.trajectory_edit.textChanged.connect(
            lambda: self.controller.set_parameter_value("schedule.trajectory_points_cm", self.trajectory_edit.toPlainText())
        )
        schedule_card.body_layout.addWidget(trajectory_label)
        schedule_card.body_layout.addWidget(self.trajectory_edit)
        self.parameter_layout.addWidget(schedule_card)

    def _sync_parameters_from_state(self, state: ExperimentViewState) -> None:
        _ = state
        self._set_combo_value(self.kind_combo, str(self.controller.get_config_value("schedule.kind", "sweep")))
        self._set_spin(self.dimensions_spin, int(self.controller.get_config_value("schedule.dimensions", 4)))
        self._set_double(self.amplitude_spin, float(self.controller.get_config_value("schedule.amplitude_cm", 0.2)))
        self._set_spin(self.steps_spin, int(self.controller.get_config_value("schedule.steps_per_axis", 5)))
        self._set_spin(self.repeats_spin, int(self.controller.get_config_value("schedule.repeats", 1)))
        self._set_spin(self.babble_spin, int(self.controller.get_config_value("schedule.babble_count", 20)))
        self._set_plain_text(
            self.trajectory_edit,
            _yaml_block(self.controller.get_config_value("schedule.trajectory_points_cm", [])),
        )


class CollectPoseCommandDatasetPage(ExperimentPageBase):
    show_visualization = True
    page_hint = "Use this page to collect pose-command datasets from a generated command schedule. This stays separate from routine servo tuning or pretension."

    def _build_parameter_sections(self) -> None:
        collection_card = ExperimentCard("Collection", "Configure the data-collection mode and per-point sampling behavior.")
        collection_form = QFormLayout()
        self.dry_run_check = QCheckBox("Dry Run")
        self.dry_run_check.toggled.connect(lambda value: self.controller.set_config_value("dry_run", bool(value)))
        self.samples_spin = QSpinBox()
        self.samples_spin.setRange(1, 100)
        self.samples_spin.valueChanged.connect(lambda value: self.controller.set_config_value("sample_count_per_point", int(value)))
        self.settle_time_spin = QDoubleSpinBox()
        self.settle_time_spin.setRange(0.0, 60.0)
        self.settle_time_spin.setDecimals(3)
        self.settle_time_spin.setSingleStep(0.05)
        self.settle_time_spin.valueChanged.connect(lambda value: self.controller.set_config_value("settle_time_s", float(value)))
        collection_form.addRow("Mode", self.dry_run_check)
        collection_form.addRow("Samples / Point", self.samples_spin)
        collection_form.addRow("Settle Time (s)", self.settle_time_spin)
        collection_card.body_layout.addLayout(collection_form)

        schedule_card = ExperimentCard("Command Schedule", "Configure the generated command schedule used for dataset collection.")
        schedule_form = QFormLayout()
        self.kind_combo = QComboBox()
        for label, value in (
            ("Sweep", "sweep"),
            ("Grid", "grid"),
            ("Trajectory", "trajectory"),
            ("Babble", "babble"),
        ):
            self.kind_combo.addItem(label, value)
        self.kind_combo.currentIndexChanged.connect(
            lambda _index: self.controller.set_config_value("command_schedule.kind", str(self.kind_combo.currentData()))
        )
        self.dimensions_spin = QSpinBox()
        self.dimensions_spin.setRange(1, 16)
        self.dimensions_spin.valueChanged.connect(lambda value: self.controller.set_config_value("command_schedule.dimensions", int(value)))
        self.amplitude_spin = QDoubleSpinBox()
        self.amplitude_spin.setRange(0.0, 10.0)
        self.amplitude_spin.setDecimals(3)
        self.amplitude_spin.setSingleStep(0.05)
        self.amplitude_spin.valueChanged.connect(lambda value: self.controller.set_config_value("command_schedule.amplitude_cm", float(value)))
        self.steps_spin = QSpinBox()
        self.steps_spin.setRange(1, 100)
        self.steps_spin.valueChanged.connect(lambda value: self.controller.set_config_value("command_schedule.steps_per_axis", int(value)))
        self.repeats_spin = QSpinBox()
        self.repeats_spin.setRange(1, 100)
        self.repeats_spin.valueChanged.connect(lambda value: self.controller.set_config_value("command_schedule.repeats", int(value)))
        schedule_form.addRow("Kind", self.kind_combo)
        schedule_form.addRow("Dimensions", self.dimensions_spin)
        schedule_form.addRow("Amplitude (cm)", self.amplitude_spin)
        schedule_form.addRow("Steps / Axis", self.steps_spin)
        schedule_form.addRow("Repeats", self.repeats_spin)
        schedule_card.body_layout.addLayout(schedule_form)
        trajectory_label = QLabel("Trajectory Points (YAML list)")
        trajectory_label.setProperty("role", "muted")
        self.trajectory_edit = QPlainTextEdit()
        self.trajectory_edit.setMinimumHeight(140)
        self.trajectory_edit.textChanged.connect(
            lambda: self.controller.set_parameter_value("command_schedule.trajectory_points_cm", self.trajectory_edit.toPlainText())
        )
        schedule_card.body_layout.addWidget(trajectory_label)
        schedule_card.body_layout.addWidget(self.trajectory_edit)
        self.parameter_layout.addWidget(collection_card)
        self.parameter_layout.addWidget(schedule_card)

    def _sync_parameters_from_state(self, state: ExperimentViewState) -> None:
        _ = state
        self._set_checkbox(self.dry_run_check, bool(self.controller.get_config_value("dry_run", True)))
        self._set_spin(self.samples_spin, int(self.controller.get_config_value("sample_count_per_point", 1)))
        self._set_double(self.settle_time_spin, float(self.controller.get_config_value("settle_time_s", 0.0)))
        self._set_combo_value(self.kind_combo, str(self.controller.get_config_value("command_schedule.kind", "sweep")))
        self._set_spin(self.dimensions_spin, int(self.controller.get_config_value("command_schedule.dimensions", 4)))
        self._set_double(self.amplitude_spin, float(self.controller.get_config_value("command_schedule.amplitude_cm", 0.2)))
        self._set_spin(self.steps_spin, int(self.controller.get_config_value("command_schedule.steps_per_axis", 5)))
        self._set_spin(self.repeats_spin, int(self.controller.get_config_value("command_schedule.repeats", 1)))
        self._set_plain_text(
            self.trajectory_edit,
            _yaml_block(self.controller.get_config_value("command_schedule.trajectory_points_cm", [])),
        )


class ReplayRunnerPage(ExperimentPageBase):
    show_visualization = True
    page_hint = "Use this page to reload an existing canonical dataset bundle and run the replay/analysis path without touching live setup workflows."

    def _build_parameter_sections(self) -> None:
        replay_card = ExperimentCard("Replay Source", "Choose an existing canonical experiment run directory to replay through the shared analysis path.")
        replay_form = QFormLayout()
        self.dataset_path_edit = QLineEdit()
        self.dataset_path_edit.editingFinished.connect(
            lambda: self.controller.set_config_value("dataset_path", self.dataset_path_edit.text().strip())
        )
        browse_button = QPushButton("Browse")
        browse_button.setProperty("variant", "ghost")
        browse_button.clicked.connect(self._browse_dataset_path)
        path_row = QWidget()
        path_layout = QHBoxLayout(path_row)
        path_layout.setContentsMargins(0, 0, 0, 0)
        path_layout.setSpacing(8)
        path_layout.addWidget(self.dataset_path_edit, 1)
        path_layout.addWidget(browse_button)
        replay_form.addRow("Dataset Path", path_row)
        replay_card.body_layout.addLayout(replay_form)
        self.parameter_layout.addWidget(replay_card)

    def _browse_dataset_path(self) -> None:
        path = QFileDialog.getExistingDirectory(self, "Select Replay Dataset", "")
        if path:
            self.controller.set_config_value("dataset_path", path)

    def _sync_parameters_from_state(self, state: ExperimentViewState) -> None:
        _ = state
        self._set_line_text(self.dataset_path_edit, str(self.controller.get_config_value("dataset_path", "")))


class EmptyExperimentWorkspace(QWidget):
    """Empty state shown before the operator selects an experiment."""

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addStretch(1)
        card = ExperimentCard(
            "Select An Experiment",
            "Choose a validation or data-generation workflow from the dropdown above. "
            "The selected experiment page will load here with only its relevant controls and outputs.",
        )
        hint = QLabel(
            "Operational setup workflows stay in their dedicated tabs. The Experiments tab is for less-frequent validation runs, dataset generation, and structured comparisons."
        )
        hint.setProperty("role", "body")
        hint.setWordWrap(True)
        card.body_layout.addWidget(hint)
        layout.addWidget(card)
        layout.addStretch(1)


class ExperimentCard(QFrame):
    """Simple shared card container for experiment shell pages."""

    def __init__(self, title: str | None = None, subtitle: str | None = None, parent=None) -> None:
        super().__init__(parent)
        self.setProperty("role", "card")
        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(12)
        if title:
            title_label = QLabel(title)
            title_label.setProperty("role", "section-title")
            layout.addWidget(title_label)
        if subtitle:
            subtitle_label = QLabel(subtitle)
            subtitle_label.setProperty("role", "body")
            subtitle_label.setWordWrap(True)
            layout.addWidget(subtitle_label)
        self.body_layout = layout


class KeyValueSummaryWidget(QWidget):
    """Shared key/value summary rows."""

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._pairs_signature: tuple[tuple[str, str], ...] | None = None
        self.layout_ = QVBoxLayout(self)
        self.layout_.setContentsMargins(0, 0, 0, 0)
        self.layout_.setSpacing(8)

    def set_pairs(self, pairs: list[tuple[str, str]]) -> None:
        signature = tuple((str(label), str(value)) for label, value in pairs)
        if signature == self._pairs_signature:
            return
        self._pairs_signature = signature
        while self.layout_.count():
            item = self.layout_.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()
        for label, value in signature:
            row = QFrame()
            row.setStyleSheet("background: #f8fafc; border: 1px solid #e2e8f0; border-radius: 10px;")
            row_layout = QHBoxLayout(row)
            row_layout.setContentsMargins(10, 8, 10, 8)
            row_layout.setSpacing(12)
            key = QLabel(label)
            key.setProperty("role", "muted")
            val = QLabel(value)
            val.setWordWrap(True)
            val.setStyleSheet("color: #0f172a; font-weight: 600;")
            row_layout.addWidget(key, 1)
            row_layout.addWidget(val, 2)
            self.layout_.addWidget(row)
        self.layout_.addStretch(1)


class HistoryItemWidget(QWidget):
    """Shared recent-run history row widget."""

    def __init__(self, *, experiment_name: str, timestamp_utc: str, status: str, metric_summary: str, run_path: str, parent=None) -> None:
        super().__init__(parent)
        title = QLabel(experiment_name.replace("_", " ").title())
        title.setStyleSheet("font-weight: 700; color: #0f172a;")
        chip = QLabel(_status_label(status))
        chip.setStyleSheet(
            f"padding: 4px 8px; border-radius: 999px; background: {_status_bg(status)}; color: {_status_fg(status)}; font-weight: 700;"
        )
        stamp = QLabel(timestamp_utc.replace("T", " ").replace("+00:00", "Z"))
        stamp.setStyleSheet("color: #64748b;")
        metric = QLabel(metric_summary or "No metric summary")
        metric.setWordWrap(True)
        metric.setStyleSheet("color: #334155;")
        run_name = QLabel(run_path.split("/")[-1])
        run_name.setStyleSheet("color: #94a3b8;")
        title_row = QHBoxLayout()
        title_row.setContentsMargins(0, 0, 0, 0)
        title_row.addWidget(title, 1)
        title_row.addWidget(chip)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 10, 12, 10)
        layout.setSpacing(4)
        layout.addLayout(title_row)
        layout.addWidget(stamp)
        layout.addWidget(metric)
        layout.addWidget(run_name)
        self.setStyleSheet("background: #f8fafc; border: 1px solid #e2e8f0; border-radius: 12px;")


def build_experiment_page(controller, experiment_name: str) -> ExperimentPageBase:
    """Return the custom page widget for one supported experiment."""
    factories: dict[str, Callable[[object], ExperimentPageBase]] = {
        "repeatability_dataset": lambda ctrl: RepeatabilityDatasetPage(ctrl, "repeatability_dataset"),
        "aurora_grid_accuracy": lambda ctrl: AuroraGridAccuracyPage(ctrl, "aurora_grid_accuracy"),
        "pretension_validation": lambda ctrl: PretensionValidationPage(ctrl, "pretension_validation"),
        "command_schedule_validation": lambda ctrl: CommandScheduleValidationPage(ctrl, "command_schedule_validation"),
        "collect_pose_command_dataset": lambda ctrl: CollectPoseCommandDatasetPage(ctrl, "collect_pose_command_dataset"),
        "replay_runner": lambda ctrl: ReplayRunnerPage(ctrl, "replay_runner"),
    }
    if experiment_name not in factories:
        raise KeyError(f"No custom experiment page is registered for {experiment_name}")
    return factories[experiment_name](controller)


def _yaml_block(value) -> str:
    rendered = yaml.safe_dump(value if value is not None else [], sort_keys=False)
    return str(rendered or "").strip()


def _render_inline_list(value) -> str:
    if value is None:
        return ""
    if isinstance(value, (list, tuple)):
        return ", ".join(str(item) for item in value)
    return str(value)


def _apply_inline_list_edit(controller, key: str, text: str) -> None:
    raw = str(text).strip()
    if not raw:
        controller.set_config_value(key, [])
        return
    normalized = raw if raw.startswith("[") else f"[{raw}]"
    controller.set_parameter_value(key, normalized)


def _status_label(status: str) -> str:
    mapping = {
        "success": "Pass",
        "partial_success": "Partial",
        "invalid_due_to_missing_registration": "Needs Registration",
        "invalid_due_to_missing_tip_cal": "Needs Tip Cal",
        "invalid_due_to_insufficient_samples": "Too Few Samples",
        "invalid_due_to_invalid_transforms": "Invalid",
    }
    return mapping.get(status, status.replace("_", " ").title())


def _fmt_metric(value) -> str:
    if value is None:
        return "n/a"
    return f"{float(value):.3f}"


def _status_bg(status: str) -> str:
    if status == "success":
        return "#dcfce7"
    if status == "partial_success":
        return "#fef3c7"
    return "#fee2e2"


def _status_fg(status: str) -> str:
    if status == "success":
        return "#166534"
    if status == "partial_success":
        return "#92400e"
    return "#991b1b"

"""Custom experiment pages hosted by the Experiment tab shell/router."""

from __future__ import annotations

from datetime import datetime, timezone
import json
import logging
from pathlib import Path
import threading
import time
from typing import Callable

import numpy as np
import yaml
from shiboken6 import isValid as _qt_is_valid
from PySide6.QtCore import QCoreApplication, QEvent, QSignalBlocker, Qt, QThread, QTimer, QUrl, Signal
from PySide6.QtGui import QColor, QDesktopServices
from PySide6.QtWidgets import (
    QAbstractItemView,
    QApplication,
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
    QToolButton,
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
from continuum_robot.experiments.grid_accuracy_outputs import build_grid_accuracy_summary_pairs
from continuum_robot.experiments.builtins import (
    PretensionValidationExperimentConfig,
    ServoTrackerSyncValidationConfig,
    TrackerTimingValidationConfig,
)
from continuum_robot.experiments.penprobe_chasing_demo import (
    MAPPING_LEGACY_POLYNOMIAL_WORKSPACE,
    MAPPING_PAIRED_XY_PROPORTIONAL,
    PenprobeChasingDemoConfig,
)
from continuum_robot.experiments.two_segment_collect_pose_dataset import (
    TwoSegmentCollectPoseDatasetConfig,
)
from continuum_robot.gui.widgets.no_wheel_combo_box import NoWheelComboBox
from continuum_robot.experiments.calibration_validation import (
    list_pivot_validation_candidates,
    list_registration_validation_candidates,
)
from continuum_robot.experiments.single_segment_repeatability import (
    SingleSegmentRepeatabilityConfig,
    build_legacy_17_point_targets,
    generate_legacy_revisit_sequence,
    repeatability_ring_tick_defaults,
)
from continuum_robot.gui.controllers.experiment_controller import ExperimentViewState
from continuum_robot.gui.theme import COLORS, chip_stylesheet, semantic_chip_colors
from continuum_robot.gui.view_utils import editable_update_blocked, preserve_scroll_position, set_line_edit_text, set_spinbox_value, set_text_document
from continuum_robot.gui.widgets.experiment_3d_widget import Experiment3DWidget, VIS_MODE_PROJECTION
from continuum_robot.gui.widgets.grid_accuracy_preview_widget import GridAccuracyPreviewWidget
from continuum_robot.gui.widgets.experiment_preflight_widget import ExperimentPreflightWidget
from continuum_robot.gui.widgets.experiment_results_widget import ExperimentResultsWidget


LOG = logging.getLogger(__name__)


class ExperimentPageBase(QWidget):
    """Base class for one custom experiment workspace page."""

    show_visualization = False
    refresh_policy = "live"
    parameter_panel_scrollable = True
    page_hint = ""
    defer_visualization_until_data = False
    visualization_mode_override: str | None = None

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

        parameter_container = QWidget()
        parameter_container.setMinimumWidth(300)
        self.parameter_layout = QVBoxLayout(parameter_container)
        self.parameter_layout.setContentsMargins(0, 0, 0, 0)
        self.parameter_layout.setSpacing(12)
        self._build_parameter_sections()
        self.parameter_layout.addStretch(1)
        self.parameter_container = parameter_container
        if self.parameter_panel_scrollable:
            self.parameter_scroll = QScrollArea()
            self.parameter_scroll.setWidgetResizable(True)
            self.parameter_scroll.setFrameShape(QFrame.NoFrame)
            self.parameter_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
            self.parameter_scroll.setMinimumWidth(300)
            self.parameter_scroll.setWidget(parameter_container)
            parameter_panel = self.parameter_scroll
        else:
            self.parameter_scroll = None
            parameter_panel = self.parameter_container
        self.parameter_panel = parameter_panel

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

        self.top_row.addWidget(self.parameter_panel, 3)
        self.top_row.addWidget(right_column, 2)
        content_layout.addLayout(self.top_row)

        self.viewer_3d = None
        self.viewer_placeholder = None
        self._viewer_requested_mode = (
            self.visualization_mode_override or self.controller.settings.runtime.visualization_mode
        )
        self._viewer_safe_effects = self.controller.settings.runtime.visualization_safe_effects
        self._visualization_card = None
        if self.show_visualization:
            self._visualization_card = ExperimentCard(
                "Visualization",
                "Experiment-specific spatial context when the selected run includes positional samples.",
            )
            self.viewer_placeholder = QLabel(
                "3D visualization is deferred until a run or loaded result provides spatial samples."
            )
            self.viewer_placeholder.setWordWrap(True)
            self.viewer_placeholder.setProperty("role", "muted")
            self.viewer_placeholder.setMinimumHeight(72)
            self._visualization_card.body_layout.addWidget(self.viewer_placeholder)
            content_layout.addWidget(self._visualization_card)
            if not self.defer_visualization_until_data:
                self._ensure_viewer_3d()

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
        if self.show_visualization and self.viewer_3d is None:
            should_create_viewer = (not self.defer_visualization_until_data) or bool(state.visualization_model.series_3d)
            if should_create_viewer:
                self._ensure_viewer_3d()
        if self.viewer_3d is not None:
            self.viewer_3d.set_view_options(show_axes=state.show_axes, show_labels=state.show_labels)
            self.viewer_3d.set_series(state.visualization_model.series_3d)
        elif self.viewer_placeholder is not None:
            self.viewer_placeholder.setVisible(True)
        self._update_history(state)
        self._sync_parameters_from_state(state)

    def state_fingerprint(self, state: ExperimentViewState) -> tuple[object, ...]:
        preflight_signature = (
            state.preflight_report.overall_status,
            tuple(
                (check.key, check.status, check.message)
                for check in state.preflight_report.checks
            ),
            tuple(state.preflight_report.overwrite_targets),
            state.preflight_report.planned_output_dir,
        )
        history_signature = tuple(
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
        return (
            state.selected_experiment,
            state.experiment_title,
            state.experiment_description,
            tuple(state.experiment_badges),
            state.output_root,
            state.operator_notes,
            state.config_error or "",
            state.status_message,
            state.last_error or "",
            state.last_output_path or "",
            state.loaded_run_path or "",
            state.run_active,
            state.progress_current,
            state.progress_total,
            state.planned_output_dir,
            preflight_signature,
            tuple(state.result_details),
            tuple(state.result_summary_lines),
            history_signature,
            state.history_loading,
            self._parameter_state_fingerprint(state),
        )

    def _build_parameter_sections(self) -> None:
        raise NotImplementedError

    def _sync_parameters_from_state(self, state: ExperimentViewState) -> None:
        raise NotImplementedError

    def _parameter_state_fingerprint(self, state: ExperimentViewState) -> tuple[object, ...]:
        _ = state
        return ()

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
        except Exception as exc:
            LOG.exception("Experiment page run action failed")
            set_text_document(self.status_text, f"Run action failed: {exc}", stick_to_bottom_if_at_bottom=True)
            return

    def _load_selected_history_item(self, item: QListWidgetItem) -> None:
        raw_path = item.data(Qt.UserRole)
        if raw_path:
            try:
                self.controller.load_run(raw_path)
            except Exception as exc:
                LOG.exception("Experiment history load failed | path=%s", raw_path)
                set_text_document(
                    self.status_text,
                    f"History load failed for {raw_path}: {exc}",
                    stick_to_bottom_if_at_bottom=True,
                )
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
        signature_entries = list(
            tuple(
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
        )
        if state.history_loading:
            signature_entries.append(("__loading__", "", "", "", "", ""))
        signature = tuple(signature_entries)
        if signature == self._history_signature:
            return
        self._history_signature = signature
        selected_path = None
        current_item = self.history_list.currentItem()
        if current_item is not None:
            selected_path = current_item.data(Qt.UserRole)

        def _rebuild() -> None:
            self.history_list.clear()
            if state.history_loading and not state.history:
                item = QListWidgetItem("Loading recent runs...")
                item.setFlags(Qt.NoItemFlags)
                self.history_list.addItem(item)
                return
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

    def _ensure_viewer_3d(self) -> None:
        if not self.show_visualization or self.viewer_3d is not None or self._visualization_card is None:
            return
        started = time.monotonic()
        viewer = Experiment3DWidget(
            requested_mode=self._viewer_requested_mode,
            safe_effects=self._viewer_safe_effects,
        )
        viewer.setMinimumHeight(300)
        self.viewer_3d = viewer
        self._visualization_card.body_layout.insertWidget(0, viewer)
        if self.viewer_placeholder is not None:
            self.viewer_placeholder.hide()
        duration_ms = (time.monotonic() - started) * 1000.0
        LOG.debug(
            "Created experiment visualization widget for %s in %.1f ms using backend %s",
            self.experiment_name,
            duration_ms,
            getattr(viewer, "backend_mode", "unknown"),
        )

    def _set_line_text(self, widget: QLineEdit, value: str) -> None:
        set_line_edit_text(widget, value, skip_if_focused=True, block_signals=True)

    def _set_plain_text(self, widget: QPlainTextEdit, value: str) -> None:
        set_text_document(widget, value, skip_if_focused=True, block_signals=True, stick_to_bottom_if_at_bottom=False)

    def _set_checkbox(self, widget: QCheckBox, value: bool) -> None:
        with QSignalBlocker(widget):
            widget.setChecked(bool(value))

    def _set_spin(self, widget: QSpinBox, value: int) -> None:
        set_spinbox_value(widget, int(value), skip_if_editing=True, block_signals=True)

    def _set_double(self, widget: QDoubleSpinBox, value: float) -> None:
        set_spinbox_value(widget, float(value), skip_if_editing=True, block_signals=True)

    def _set_combo_value(self, widget: QComboBox, value: str) -> None:
        target = str(value)
        index = widget.findData(target)
        if index < 0:
            index = widget.findText(target)
        if index < 0:
            return
        if editable_update_blocked(widget):
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
        self.target_set_combo = NoWheelComboBox()
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


class SingleSegmentRepeatabilityPage(ExperimentPageBase):
    show_visualization = True
    refresh_policy = "manual"
    defer_visualization_until_data = True
    visualization_mode_override = VIS_MODE_PROJECTION
    page_hint = (
        "Live thesis repeatability protocol: the legacy 17-target single-segment pattern, "
        "with each desired target revisited after approaching from every other target. "
        "This page requires accepted registration, accepted 0A runtime tip calibration, connected servos, "
        "and accepted pretension state."
    )

    def __init__(self, controller, experiment_name: str, parent=None) -> None:
        super().__init__(controller, experiment_name, parent)
        self.run_button.setText("Run 17-Target Repeatability")
        self._target_table_signature: tuple[tuple[str, str, str, str, str, str], ...] | None = None

    def _build_parameter_sections(self) -> None:
        protocol_card = ExperimentCard(
            "Protocol",
            "Fixed legacy protocol: 17 targets, 272 approach/repeat visits, 544 planned captures. "
            "Repeatability metrics use the repeat captures after returning to the desired target.",
        )
        self.protocol_summary_widget = KeyValueSummaryWidget()
        protocol_card.body_layout.addWidget(self.protocol_summary_widget)

        comparison_card = ExperimentCard(
            "Baseline Comparison",
            "Optionally compare this run against a prior single-segment repeatability run. "
            "The saved summary will report deltas for overall RMS, max deviation, path-dependence, ring groups, and per-target RMSE.",
        )
        comparison_form = QFormLayout()
        self.baseline_path_edit = QLineEdit()
        self.baseline_path_edit.setPlaceholderText("Optional prior run folder or summary.json")
        self.baseline_path_edit.editingFinished.connect(
            lambda: self.controller.set_config_value("baseline_run_path", self.baseline_path_edit.text().strip())
        )
        browse_button = QPushButton("Browse")
        browse_button.setProperty("variant", "ghost")
        browse_button.clicked.connect(self._browse_baseline_run)
        clear_button = QPushButton("Clear")
        clear_button.setProperty("variant", "ghost")
        clear_button.clicked.connect(lambda: self.controller.set_config_value("baseline_run_path", ""))
        baseline_row = QWidget()
        baseline_layout = QHBoxLayout(baseline_row)
        baseline_layout.setContentsMargins(0, 0, 0, 0)
        baseline_layout.setSpacing(8)
        baseline_layout.addWidget(self.baseline_path_edit, 1)
        baseline_layout.addWidget(browse_button)
        baseline_layout.addWidget(clear_button)
        comparison_form.addRow("Baseline Run", baseline_row)
        comparison_card.body_layout.addLayout(comparison_form)
        self.comparison_summary_widget = KeyValueSummaryWidget()
        comparison_card.body_layout.addWidget(self.comparison_summary_widget)

        config_card = ExperimentCard(
            "Run Parameters",
            "Only timing and gating parameters are exposed here. Target geometry and sequence structure are intentionally fixed.",
        )
        form = QFormLayout()
        self.tool_id_edit = QLineEdit()
        self.tool_id_edit.editingFinished.connect(
            lambda: self.controller.set_config_value("tool_id", self.tool_id_edit.text().strip().upper() or "0A")
        )
        self.settle_time_spin = QDoubleSpinBox()
        self.settle_time_spin.setRange(0.0, 60.0)
        self.settle_time_spin.setDecimals(3)
        self.settle_time_spin.setSingleStep(0.25)
        self.settle_time_spin.valueChanged.connect(
            lambda value: self.controller.set_config_value("settle_time_s", float(value))
        )
        self.capture_timeout_spin = QDoubleSpinBox()
        self.capture_timeout_spin.setRange(0.0, 10.0)
        self.capture_timeout_spin.setDecimals(3)
        self.capture_timeout_spin.setSingleStep(0.05)
        self.capture_timeout_spin.valueChanged.connect(
            lambda value: self.controller.set_config_value("capture_timeout_s", float(value))
        )
        self.capture_poll_spin = QDoubleSpinBox()
        self.capture_poll_spin.setRange(0.001, 1.0)
        self.capture_poll_spin.setDecimals(3)
        self.capture_poll_spin.setSingleStep(0.005)
        self.capture_poll_spin.valueChanged.connect(
            lambda value: self.controller.set_config_value("capture_poll_interval_s", float(value))
        )
        self.max_age_spin = QDoubleSpinBox()
        self.max_age_spin.setRange(0.0, 5.0)
        self.max_age_spin.setDecimals(3)
        self.max_age_spin.setSingleStep(0.05)
        self.max_age_spin.valueChanged.connect(
            lambda value: self.controller.set_config_value("max_tracker_age_s", float(value))
        )
        self.seed_spin = QSpinBox()
        self.seed_spin.setRange(0, 999999)
        self.seed_spin.valueChanged.connect(lambda value: self.controller.set_config_value("random_seed", int(value)))
        self.run_label_edit = QLineEdit()
        self.run_label_edit.editingFinished.connect(
            lambda: self.controller.set_config_value("run_label", self.run_label_edit.text().strip())
        )
        self.return_center_check = QCheckBox("Return to center on finalize")
        self.return_center_check.toggled.connect(
            lambda value: self.controller.set_config_value("return_to_center_on_finalize", bool(value))
        )
        self.fail_rejected_check = QCheckBox("Abort on first rejected capture")
        self.fail_rejected_check.toggled.connect(
            lambda value: self.controller.set_config_value("fail_on_rejected_capture", bool(value))
        )
        self.debug_coil_as_tip_check = QCheckBox("Allow lower-trust runtime tip override")
        self.debug_coil_as_tip_check.setToolTip(
            "Compatibility flag for lower-trust runtime tip modes. Coil-as-tip itself is governed by the shared runtime tip policy."
        )
        self.debug_coil_as_tip_check.toggled.connect(
            lambda value: self.controller.set_config_value("allow_debug_coil_as_tip", bool(value))
        )
        form.addRow("Runtime Tool", self.tool_id_edit)
        form.addRow("Settle Time (s)", self.settle_time_spin)
        form.addRow("Capture Timeout (s)", self.capture_timeout_spin)
        form.addRow("Capture Poll (s)", self.capture_poll_spin)
        form.addRow("Max Tracker Age (s)", self.max_age_spin)
        form.addRow("Random Seed", self.seed_spin)
        form.addRow("Run Label", self.run_label_edit)
        form.addRow("Finalize", self.return_center_check)
        form.addRow("Rejected Captures", self.fail_rejected_check)
        form.addRow("Debug Runtime Tip", self.debug_coil_as_tip_check)
        config_card.body_layout.addLayout(form)

        target_card = ExperimentCard(
            "Fixed Target Catalog",
            "Target 0 is center; targets 1-8 use the configured inner ring and targets 9-16 use the configured outer ring.",
        )
        self.target_table = QTableWidget(0, 6)
        self.target_table.setHorizontalHeaderLabels(
            ["Target", "Ring", "Angle", "Command (mm)", "Command (ticks)", "Approaches"]
        )
        self.target_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.target_table.setSelectionMode(QAbstractItemView.NoSelection)
        self.target_table.verticalHeader().setVisible(False)
        self.target_table.horizontalHeader().setStretchLastSection(True)
        self.target_table.setMinimumHeight(260)
        target_card.body_layout.addWidget(self.target_table)

        self.parameter_layout.addWidget(protocol_card)
        self.parameter_layout.addWidget(comparison_card)
        self.parameter_layout.addWidget(config_card)
        self.parameter_layout.addWidget(target_card)

    def _sync_parameters_from_state(self, state: ExperimentViewState) -> None:
        _ = state
        config = SingleSegmentRepeatabilityConfig.from_dict(self.controller.config_payload())
        self._set_line_text(self.tool_id_edit, str(config.tool_id or "0A"))
        self._set_double(self.settle_time_spin, float(config.settle_time_s))
        self._set_double(self.capture_timeout_spin, float(config.capture_timeout_s))
        self._set_double(self.capture_poll_spin, float(config.capture_poll_interval_s))
        self._set_double(self.max_age_spin, float(config.max_tracker_age_s))
        self._set_spin(self.seed_spin, int(config.random_seed))
        self._set_line_text(self.run_label_edit, str(config.run_label or ""))
        self._set_checkbox(self.return_center_check, bool(config.return_to_center_on_finalize))
        self._set_checkbox(self.fail_rejected_check, bool(config.fail_on_rejected_capture))
        self._set_checkbox(self.debug_coil_as_tip_check, bool(config.allow_debug_coil_as_tip))
        self._set_line_text(self.baseline_path_edit, str(config.baseline_run_path or ""))
        self._sync_protocol_summary(config)
        self._sync_comparison_summary(config)
        self._sync_target_table(config)

    def _sync_protocol_summary(self, config: SingleSegmentRepeatabilityConfig) -> None:
        targets = build_legacy_17_point_targets(
            inner_ring_radius_mm=float(config.inner_ring_radius_mm),
            outer_ring_radius_mm=float(config.outer_ring_radius_mm),
        )
        visits = generate_legacy_revisit_sequence(targets=targets, seed=int(config.random_seed))
        planned_captures = len(visits) * 2
        total_time_min = (planned_captures * float(config.settle_time_s)) / 60.0
        ring_ticks = repeatability_ring_tick_defaults(
            mapper=self.controller.servo_service.mapper,
            inner_ring_radius_mm=float(config.inner_ring_radius_mm),
            outer_ring_radius_mm=float(config.outer_ring_radius_mm),
        )
        spool_diameter_mm = float(self.controller.settings.robot.spool_diameter_cm) * 10.0
        self.protocol_summary_widget.set_pairs(
            [
                ("Protocol", "Legacy 17-target single segment"),
                ("Targets", "17 (center + 8 inner + 8 outer)"),
                (
                    "Ring Radii",
                    (
                        f"inner {float(config.inner_ring_radius_mm):.1f} mm "
                        f"({int(ring_ticks['inner_ring_radius_ticks'])} ticks), "
                        f"outer {float(config.outer_ring_radius_mm):.1f} mm "
                        f"({int(ring_ticks['outer_ring_radius_ticks'])} ticks)"
                    ),
                ),
                ("Spool Diameter", f"{spool_diameter_mm:.1f} mm"),
                (
                    "Repeatability Amplitude Cap",
                    f"{int(config.max_target_tick_delta_from_startup)} ticks max from startup",
                ),
                ("Approach Visits", str(len(visits))),
                ("Planned Captures", str(planned_captures)),
                ("Repeat Captures Used For RMSE", str(len(visits))),
                (
                    "Run Validity Threshold",
                    f">= {int(config.min_repeat_captures_per_target)} repeats/target "
                    f"and >= {float(config.min_repeat_capture_fraction) * 100.0:.0f}% total repeat coverage",
                ),
                (
                    "Rejected Capture Limit",
                    f"<= {float(config.max_rejected_capture_fraction) * 100.0:.0f}% of all captures",
                ),
                ("Estimated Settle Time", f"{total_time_min:.1f} min"),
                ("Required Pose Frame", "robot/base frame via 0A runtime tip"),
                ("Run Trust", "Blocked unless registration, runtime tip calibration, pretension, and servos are ready"),
            ]
        )

    def _sync_comparison_summary(self, config: SingleSegmentRepeatabilityConfig) -> None:
        baseline = str(config.baseline_run_path or "").strip()
        if not baseline:
            self.comparison_summary_widget.set_pairs(
                [
                    ("Mode", "No baseline selected"),
                    ("Comparison Output", "Current run will still save full repeatability metrics and figures."),
                ]
            )
            return
        self.comparison_summary_widget.set_pairs(
            [
                ("Mode", "Compare current run to selected baseline"),
                ("Baseline", baseline),
                ("Deltas", "overall RMS, max deviation, path RMS, ring means, rejected captures, per-target RMSE"),
            ]
        )

    def _browse_baseline_run(self) -> None:
        path = QFileDialog.getExistingDirectory(self, "Select Baseline Repeatability Run", "")
        if path:
            self.controller.set_config_value("baseline_run_path", path)

    def _sync_target_table(self, config: SingleSegmentRepeatabilityConfig) -> None:
        targets = build_legacy_17_point_targets(
            inner_ring_radius_mm=float(config.inner_ring_radius_mm),
            outer_ring_radius_mm=float(config.outer_ring_radius_mm),
        )
        visits = generate_legacy_revisit_sequence(targets, seed=int(config.random_seed))
        approaches_by_target: dict[int, int] = {}
        for visit in visits:
            approaches_by_target[int(visit.target_index)] = approaches_by_target.get(int(visit.target_index), 0) + 1
        mapper = self.controller.servo_service.mapper
        signature = tuple(
            (
                target.label,
                target.ring,
                f"{target.angle_deg:.0f}",
                _render_inline_list(target.cable_deltas_mm),
                _render_inline_list([int(mapper.displacement_cm_to_ticks(float(value))) for value in target.cable_deltas_cm]),
                str(approaches_by_target.get(int(target.target_index), 0)),
            )
            for target in targets
        )
        if self._target_table_signature == signature:
            return
        self._target_table_signature = signature

        def _rebuild() -> None:
            with QSignalBlocker(self.target_table):
                self.target_table.setRowCount(len(targets))
                for row, target in enumerate(targets):
                    cells = [
                        target.label,
                        target.ring,
                        f"{target.angle_deg:.0f} deg",
                        _render_inline_list(target.cable_deltas_mm),
                        _render_inline_list(
                            [
                                int(mapper.displacement_cm_to_ticks(float(value)))
                                for value in target.cable_deltas_cm
                            ]
                        ),
                        str(approaches_by_target.get(int(target.target_index), 0)),
                    ]
                    for column, text in enumerate(cells):
                        self.target_table.setItem(row, column, QTableWidgetItem(text))

        preserve_scroll_position(self.target_table, _rebuild)


class AuroraGridAccuracyPage(ExperimentPageBase):
    show_visualization = False
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
        self._capture_settings_locked = False
        super().__init__(controller, experiment_name, parent)
        self.run_button.setText("Save Grid Validation Run")
        self.refresh_button.setText("Refresh Tracker State")
        self.stop_button.hide()

    def _build_parameter_sections(self) -> None:
        params_card = ExperimentCard(
            "Grid Parameters",
            "Set the ideal truth-grid geometry and the per-point capture settings used for the aligned residual analysis.",
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
        params_card.body_layout.addLayout(params_form)
        self.capture_settings_notice = QLabel(
            "Capture settings are locked after the first point is recorded. Use Restart Run to change grid geometry or sampling."
        )
        self.capture_settings_notice.setProperty("role", "muted")
        self.capture_settings_notice.setWordWrap(True)
        self.capture_settings_notice.setVisible(False)
        params_card.body_layout.addWidget(self.capture_settings_notice)
        self.advanced_options = CollapsibleSection("Advanced Capture Options")
        advanced_form = QFormLayout()
        advanced_form.setContentsMargins(0, 0, 0, 0)
        advanced_form.setSpacing(10)
        advanced_form.addRow("Fallback", self.allow_fallback_check)
        advanced_form.addRow("Tip Vector (mm)", self.tip_vector_edit)
        advanced_hint = QLabel(
            "These options change how the 0B point position is derived. Leave them alone during a real capture session unless you intentionally restart the run."
        )
        advanced_hint.setProperty("role", "muted")
        advanced_hint.setWordWrap(True)
        self.advanced_options.body_layout.addLayout(advanced_form)
        self.advanced_options.body_layout.addWidget(advanced_hint)
        params_card.body_layout.addWidget(self.advanced_options)

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
        self.next_incomplete_button = QPushButton("Next Incomplete")
        self.next_incomplete_button.setProperty("variant", "ghost")
        self.next_incomplete_button.clicked.connect(self.select_next_incomplete_point)
        self.clear_selected_button = QPushButton("Clear Selected Point")
        self.clear_selected_button.setProperty("variant", "ghost")
        self.clear_selected_button.clicked.connect(self.clear_selected_point)
        self.clear_all_button = QPushButton("Restart Run")
        self.clear_all_button.setProperty("variant", "ghost")
        self.clear_all_button.clicked.connect(self.clear_all_points)
        selection_row.addWidget(self.capture_selected_button)
        selection_row.addWidget(self.next_incomplete_button)
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
        self.capture_status_text.setMinimumHeight(84)
        self.capture_status_text.setMaximumHeight(112)
        self.capture_status_text.setPlaceholderText("Capture status and session notes will appear here.")
        capture_card.body_layout.addWidget(self.capture_status_text)

        preview_card = ExperimentCard(
            "Live Alignment Preview",
            "Display-only view in the local grid frame. Truth points stay fixed; aligned centroids and residual vectors appear as soon as enough labeled points exist.",
        )
        self.grid_preview_widget = GridAccuracyPreviewWidget()
        preview_card.body_layout.addWidget(self.grid_preview_widget)

        self.parameter_layout.addWidget(params_card)
        self.parameter_layout.addWidget(capture_card)
        self.parameter_layout.addWidget(preview_card)

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
        has_captured_points = bool(self.controller.get_config_value("captured_points", []))
        self._set_capture_settings_locked(has_captured_points)
        self._sync_point_table(preview, expected_samples=int(config.samples_per_point))
        self._sync_selected_point_summary(preview, expected_samples=int(config.samples_per_point))
        self._sync_capture_summary(preview, expected_samples=int(config.samples_per_point))
        self._sync_capture_status()
        self.grid_preview_widget.set_preview(
            truth_catalog=preview.truth_catalog,
            metrics=preview.metrics,
            selected_target_index=self._selected_target_index,
            expected_samples=int(config.samples_per_point),
        )
        selected = self._selected_truth_entry(preview.truth_catalog)
        point_metrics = (
            (preview.metrics.get("per_point_metrics", {}) or {}).get(str(selected["label"]), {})
            if selected is not None
            else {}
        )
        selected_has_capture = int(point_metrics.get("raw_sample_count", 0) or 0) > 0
        self.capture_selected_button.setEnabled(not state.run_active)
        self.next_incomplete_button.setEnabled(not state.run_active and bool(preview.truth_catalog))
        self.clear_selected_button.setEnabled(not state.run_active and selected_has_capture)
        self.clear_all_button.setEnabled(not state.run_active and has_captured_points)

    def _on_mode_changed(self, value: bool) -> None:
        self.controller.set_config_value("dry_run", bool(value))
        self._refresh_now()

    def _apply_grid_geometry(self) -> None:
        if self._capture_settings_locked:
            self._append_capture_log("Restart the run before changing grid geometry or capture settings.")
            self._refresh_now()
            return
        dims = [int(self.cols_spin.value()), int(self.rows_spin.value())]
        spacing = float(self.spacing_spin.value())
        self.controller.set_config_value("dimensions", dims)
        self.controller.set_config_value("spacing_mm", spacing)
        self.controller.set_config_value("truth_points_file", None)
        self.controller.set_config_value("truth_points_mm", [])
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
        self.grid_preview_widget.set_preview(
            truth_catalog=preview.truth_catalog,
            metrics=preview.metrics,
            selected_target_index=self._selected_target_index,
            expected_samples=expected_samples,
        )

    def select_next_incomplete_point(self) -> None:
        preview = self._current_preview()
        expected_samples = max(1, int(self._grid_config().samples_per_point))
        next_target_index = self._next_incomplete_target_index(
            preview,
            expected_samples=expected_samples,
            start_after=int(self._selected_target_index),
        )
        if next_target_index is None:
            self._append_capture_log("All grid points are complete.")
            self._refresh_now()
            return
        self._selected_target_index = int(next_target_index)
        self._refresh_now()

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
        response = QMessageBox.question(
            self,
            "Restart Grid Run",
            "Clear all captured points and restart this grid-validation run?\n\nThis does not delete saved runs on disk.",
        )
        if response != QMessageBox.Yes:
            return
        self.controller.set_config_value("captured_points", [])
        self._selected_target_index = 0
        self._append_capture_log("Restarted the grid-validation run and cleared all captured points.")
        self._refresh_now()

    def _collect_point_samples(self, *, config: GridDefinitionConfig, truth_entry: dict[str, object]) -> list[dict[str, object]]:
        tip_vector_mm, tip_available = resolve_grid_tip_vector(config, project_root=self.controller.project_root)
        if config.use_tip_calibration and not tip_available and not config.allow_coil_origin_fallback:
            raise RuntimeError("Tip calibration is required for this grid capture.")
        sample_count = max(1, int(config.samples_per_point))
        raw_samples: list[dict[str, object]] = []
        for sample_index in range(sample_count):
            if config.dry_run:
                raw_sample = self._synthetic_grid_sample(
                    config=config,
                    truth_entry=truth_entry,
                    sample_index=sample_index,
                    tip_available=tip_available,
                )
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
        tip_available: bool,
    ) -> dict[str, object]:
        if config.use_tip_calibration and not tip_available and not config.allow_coil_origin_fallback:
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
            "position_source": "tip" if tip_available else "coil_origin",
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
            self._update_selected_point_label(
                truth_catalog=truth_catalog,
                metrics=preview.metrics,
                expected_samples=expected_samples,
            )
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
        self._update_selected_point_label(
            truth_catalog=truth_catalog,
            metrics=preview.metrics,
            expected_samples=expected_samples,
        )

    def _sync_capture_summary(self, preview, *, expected_samples: int) -> None:
        metrics = preview.metrics
        summary_lookup = dict(
            build_grid_accuracy_summary_pairs(
                metrics=metrics,
                config_used=self.controller.config_payload(),
            )
        )
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
                ("Coverage", summary_lookup.get("Point Status", "n/a")),
                (
                    "Solve Ready",
                    "Yes" if str(summary_lookup.get("Solve Ready", "")).lower() == "yes" else "No",
                ),
                ("Suggested Next", next_label),
                ("Aligned Points", summary_lookup.get("Aligned Points", "0")),
                ("Raw / Accepted / Rejected", _render_sample_triplet(metrics)),
                ("RMS Residual", summary_lookup.get("Overall RMS Residual", "n/a")),
                ("Max Residual", summary_lookup.get("Max Residual", "n/a")),
                ("Mean Spread", summary_lookup.get("Mean Within-Point Spread", "n/a")),
                ("Max Spread", summary_lookup.get("Max Within-Point Spread", "n/a")),
                ("Tip Calibration Used", summary_lookup.get("Tip Calibration Used", "n/a")),
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
        position_source_summary = str(point_metrics.get("position_source_summary") or "n/a")
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
        remaining_samples = max(0, int(expected_samples) - int(sample_count))
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
                ("Raw / Accepted / Rejected", f"{sample_count} / {accepted_count} / {rejected_count}"),
                ("Samples Needed", str(remaining_samples)),
                ("Within-Point Spread", _fmt_metric(point_metrics.get("sample_spread_rms_mm"))),
                ("Aligned Residual", _fmt_metric(point_metrics.get("residual_mm"))),
                ("Position Sources", position_source_summary),
                ("Suggested Next", suggested_next),
                ("Solve Note", str(preview.metrics.get("alignment_ready_reason", "n/a"))),
            ]
        )

    def _sync_capture_status(self) -> None:
        lines = self._capture_log_lines or [
            "Select a grid label, capture the full sample batch with tool 0B, and review the aligned residual preview once at least three labeled points are complete."
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

    def _update_selected_point_label(
        self,
        *,
        truth_catalog: list[dict[str, object]] | None = None,
        metrics: dict[str, object] | None = None,
        expected_samples: int | None = None,
    ) -> None:
        truth_catalog = truth_catalog or self._current_preview().truth_catalog
        selected = self._selected_truth_entry(truth_catalog)
        if selected is None:
            self.selected_point_label.setText("Selected Point: n/a")
            return
        metrics = dict(metrics or {})
        point_metrics = (metrics.get("per_point_metrics", {}) or {}).get(str(selected["label"]), {})
        truth_point = selected.get("truth_point_mm", [0.0, 0.0, 0.0])
        sample_count = int(point_metrics.get("raw_sample_count", 0) or 0)
        expected = max(1, int(expected_samples or self._grid_config().samples_per_point))
        remaining = max(0, expected - sample_count)
        self.selected_point_label.setText(
            f"Selected Point: {selected['label']}  •  truth=({truth_point[0]:.1f}, {truth_point[1]:.1f}, {truth_point[2]:.1f}) mm  •  "
            f"{sample_count}/{expected} samples ({remaining} needed)"
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

    def _set_capture_settings_locked(self, locked: bool) -> None:
        self._capture_settings_locked = bool(locked)
        for widget in (
            self.dry_run_check,
            self.rows_spin,
            self.cols_spin,
            self.spacing_spin,
            self.samples_spin,
            self.settle_time_spin,
            self.use_tip_check,
            self.allow_fallback_check,
            self.tip_vector_edit,
        ):
            widget.setEnabled(not self._capture_settings_locked)
        self.capture_settings_notice.setVisible(self._capture_settings_locked)
        self.advanced_options.setEnabled(not self._capture_settings_locked)
        if self._capture_settings_locked:
            self.advanced_options.toggle.setChecked(False)

    @staticmethod
    def _style_point_table_item(item: QTableWidgetItem, *, status: str) -> None:
        normalized = status.lower()
        if normalized.startswith("complete") and "rejected" not in normalized:
            background, foreground = semantic_chip_colors("ready")
        elif normalized.startswith("partial") or "rejected" in normalized:
            background, foreground = semantic_chip_colors("warning")
        else:
            background = COLORS.surface_bg
            foreground = COLORS.text_muted
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


class TrackerTimingValidationPage(ExperimentPageBase):
    page_hint = (
        "Benchmark the active Python Aurora backend acquisition path. This page measures backend timing and "
        "duplicate-frame behavior, not GUI refresh rate."
    )

    def __init__(self, controller, experiment_name: str, parent=None) -> None:
        super().__init__(controller, experiment_name, parent)
        self.run_button.setText("Run Timing Diagnostic")

    def _build_parameter_sections(self) -> None:
        scope_card = ExperimentCard(
            "Diagnostic Scope",
            "Keep this run focused on the backend path actually used by the app. Use Tracking for live monitoring; use this page for saved timing evidence and figures.",
        )
        scope_form = QFormLayout()
        self.tool_mode_combo = NoWheelComboBox()
        self.tool_mode_combo.addItem("0A and 0B", "both")
        self.tool_mode_combo.addItem("0A only", "0A")
        self.tool_mode_combo.addItem("0B only", "0B")
        self.tool_mode_combo.currentIndexChanged.connect(self._on_tool_mode_changed)
        self.enable_servo_logging_check = QCheckBox("Log servo telemetry for timestamp alignment")
        self.enable_servo_logging_check.toggled.connect(
            lambda value: self.controller.set_config_value("enable_servo_logging", bool(value))
        )
        self.run_label_edit = QLineEdit()
        self.run_label_edit.editingFinished.connect(
            lambda: self.controller.set_config_value("run_label", self.run_label_edit.text().strip())
        )
        scope_form.addRow("Requested Tools", self.tool_mode_combo)
        scope_form.addRow("Servo Sync", self.enable_servo_logging_check)
        scope_form.addRow("Run Label", self.run_label_edit)
        scope_card.body_layout.addLayout(scope_form)

        params_card = ExperimentCard(
            "Timing Parameters",
            "Use duration for a fixed observation window or set analyzed-sample target to stop after a specific number of fresh post-warmup samples.",
        )
        params_form = QFormLayout()
        self.duration_spin = QDoubleSpinBox()
        self.duration_spin.setRange(0.1, 300.0)
        self.duration_spin.setDecimals(2)
        self.duration_spin.setSingleStep(0.5)
        self.duration_spin.valueChanged.connect(
            lambda value: self.controller.set_config_value("run_duration_s", float(value))
        )
        self.sample_target_spin = QSpinBox()
        self.sample_target_spin.setRange(0, 200000)
        self.sample_target_spin.setSpecialValueText("Disabled")
        self.sample_target_spin.valueChanged.connect(
            lambda value: self.controller.set_config_value(
                "sample_count_target",
                None if int(value) <= 0 else int(value),
            )
        )
        self.warmup_spin = QSpinBox()
        self.warmup_spin.setRange(0, 10000)
        self.warmup_spin.valueChanged.connect(
            lambda value: self.controller.set_config_value("warmup_samples", int(value))
        )
        self.timeout_spin = QDoubleSpinBox()
        self.timeout_spin.setRange(0.2, 600.0)
        self.timeout_spin.setDecimals(2)
        self.timeout_spin.setSingleStep(0.5)
        self.timeout_spin.valueChanged.connect(
            lambda value: self.controller.set_config_value("timeout_s", float(value))
        )
        params_form.addRow("Run Duration (s)", self.duration_spin)
        params_form.addRow("Analyzed Sample Target", self.sample_target_spin)
        params_form.addRow("Warmup Samples", self.warmup_spin)
        params_form.addRow("Timeout (s)", self.timeout_spin)
        params_card.body_layout.addLayout(params_form)

        summary_card = ExperimentCard(
            "What This Run Saves",
            "Each run writes canonical timing samples, a summary JSON, a text note, and static figures for histogram, stage breakdown, and time-series timing behavior.",
        )
        self.parameter_summary_widget = KeyValueSummaryWidget()
        summary_card.body_layout.addWidget(self.parameter_summary_widget)

        self.parameter_layout.addWidget(scope_card)
        self.parameter_layout.addWidget(params_card)
        self.parameter_layout.addWidget(summary_card)

    def _sync_parameters_from_state(self, state: ExperimentViewState) -> None:
        _ = state
        config = TrackerTimingValidationConfig.from_dict(self.controller.config_payload())
        tool_mode = "both"
        if config.requested_tool_ids == ["0A"]:
            tool_mode = "0A"
        elif config.requested_tool_ids == ["0B"]:
            tool_mode = "0B"
        self._set_combo_value(self.tool_mode_combo, tool_mode)
        self._set_checkbox(self.enable_servo_logging_check, bool(config.enable_servo_logging))
        self._set_line_text(self.run_label_edit, str(config.run_label or ""))
        self._set_double(self.duration_spin, float(config.run_duration_s))
        self._set_spin(self.sample_target_spin, int(config.sample_count_target or 0))
        self._set_spin(self.warmup_spin, int(config.warmup_samples))
        self._set_double(self.timeout_spin, float(config.timeout_s))
        stop_mode = (
            f"{int(config.sample_count_target)} analyzed samples"
            if config.sample_count_target is not None
            else f"{float(config.run_duration_s):.1f} s duration"
        )
        self.parameter_summary_widget.set_pairs(
            [
                ("Benchmark Truth", "Uses backend acquisition timing and device-frame freshness. GUI refresh rate is not part of the metric."),
                ("Stop Condition", stop_mode),
                (
                    "Servo Sync",
                    (
                        "Servo telemetry timestamps will be saved and compared against tracker samples."
                        if bool(config.enable_servo_logging)
                        else "Servo sync logging is disabled for this run."
                    ),
                ),
            ]
        )

    def _on_tool_mode_changed(self) -> None:
        mode = str(self.tool_mode_combo.currentData() or "both")
        requested_tool_ids = ["0A", "0B"] if mode == "both" else [mode]
        self.controller.set_config_value("requested_tool_ids", requested_tool_ids)


class ServoTrackerSyncValidationPage(ExperimentPageBase):
    page_hint = (
        "Validate host-time alignment between tracker frames, servo commands, and servo telemetry during a "
        "small scripted motion sequence. This page measures logging usability for later motion experiments, "
        "not closed-loop control quality."
    )

    def __init__(self, controller, experiment_name: str, parent=None) -> None:
        super().__init__(controller, experiment_name, parent)
        self.run_button.setText("Run Sync Validation")

    def _build_parameter_sections(self) -> None:
        motion_card = ExperimentCard(
            "Motion Scope",
            "Use a small alternating step schedule around the current live servo position. Keep the motion bounded and reproducible so the run measures stream alignment rather than ambitious robot behavior.",
        )
        motion_form = QFormLayout()
        self.servo_ids_edit = QLineEdit()
        self.servo_ids_edit.setPlaceholderText("1 or 1,2")
        self.servo_ids_edit.editingFinished.connect(self._on_servo_ids_changed)
        self.tool_mode_combo = NoWheelComboBox()
        self.tool_mode_combo.addItem("0A only", "0A")
        self.tool_mode_combo.addItem("0B only", "0B")
        self.tool_mode_combo.addItem("0A and 0B", "both")
        self.tool_mode_combo.currentIndexChanged.connect(self._on_tool_mode_changed)
        self.include_tip_pose_check = QCheckBox("Include robot-frame tip pose when the live transform chain supports it")
        self.include_tip_pose_check.toggled.connect(
            lambda value: self.controller.set_config_value("include_robot_frame_tip_pose", bool(value))
        )
        self.run_label_edit = QLineEdit()
        self.run_label_edit.editingFinished.connect(
            lambda: self.controller.set_config_value("run_label", self.run_label_edit.text().strip())
        )
        motion_form.addRow("Servo IDs", self.servo_ids_edit)
        motion_form.addRow("Tracked Tools", self.tool_mode_combo)
        motion_form.addRow("Robot-Frame Tip", self.include_tip_pose_check)
        motion_form.addRow("Run Label", self.run_label_edit)
        motion_card.body_layout.addLayout(motion_form)

        params_card = ExperimentCard(
            "Timing And Motion Parameters",
            "These settings bound the motion window and the logging cadence. The experiment uses one host monotonic clock for tracker, servo-command, and servo-telemetry timestamps.",
        )
        params_form = QFormLayout()
        self.duration_spin = QDoubleSpinBox()
        self.duration_spin.setRange(0.5, 300.0)
        self.duration_spin.setDecimals(2)
        self.duration_spin.setSingleStep(0.5)
        self.duration_spin.valueChanged.connect(
            lambda value: self.controller.set_config_value("run_duration_s", float(value))
        )
        self.warmup_spin = QDoubleSpinBox()
        self.warmup_spin.setRange(0.0, 60.0)
        self.warmup_spin.setDecimals(2)
        self.warmup_spin.setSingleStep(0.1)
        self.warmup_spin.valueChanged.connect(
            lambda value: self.controller.set_config_value("warmup_duration_s", float(value))
        )
        self.amplitude_spin = QSpinBox()
        self.amplitude_spin.setRange(1, 512)
        self.amplitude_spin.valueChanged.connect(
            lambda value: self.controller.set_config_value("command_amplitude_ticks", int(value))
        )
        self.step_period_spin = QDoubleSpinBox()
        self.step_period_spin.setRange(0.05, 30.0)
        self.step_period_spin.setDecimals(3)
        self.step_period_spin.setSingleStep(0.05)
        self.step_period_spin.valueChanged.connect(
            lambda value: self.controller.set_config_value("step_period_s", float(value))
        )
        self.telemetry_poll_spin = QDoubleSpinBox()
        self.telemetry_poll_spin.setRange(0.005, 5.0)
        self.telemetry_poll_spin.setDecimals(3)
        self.telemetry_poll_spin.setSingleStep(0.005)
        self.telemetry_poll_spin.valueChanged.connect(
            lambda value: self.controller.set_config_value("telemetry_poll_interval_s", float(value))
        )
        self.timeout_spin = QDoubleSpinBox()
        self.timeout_spin.setRange(0.5, 600.0)
        self.timeout_spin.setDecimals(2)
        self.timeout_spin.setSingleStep(0.5)
        self.timeout_spin.valueChanged.connect(
            lambda value: self.controller.set_config_value("timeout_s", float(value))
        )
        params_form.addRow("Run Duration (s)", self.duration_spin)
        params_form.addRow("Warmup Duration (s)", self.warmup_spin)
        params_form.addRow("Command Amplitude (ticks)", self.amplitude_spin)
        params_form.addRow("Step Period (s)", self.step_period_spin)
        params_form.addRow("Telemetry Poll (s)", self.telemetry_poll_spin)
        params_form.addRow("Timeout (s)", self.timeout_spin)
        params_card.body_layout.addLayout(params_form)

        summary_card = ExperimentCard(
            "What This Run Saves",
            "Each run writes canonical tracker/servo samples, summary JSON, a text note, and static figures for offsets, motion traces, and validity/match-rate summaries.",
        )
        self.parameter_summary_widget = KeyValueSummaryWidget()
        summary_card.body_layout.addWidget(self.parameter_summary_widget)

        self.parameter_layout.addWidget(motion_card)
        self.parameter_layout.addWidget(params_card)
        self.parameter_layout.addWidget(summary_card)

    def _sync_parameters_from_state(self, state: ExperimentViewState) -> None:
        _ = state
        config = ServoTrackerSyncValidationConfig.from_dict(self.controller.config_payload())
        self._set_line_text(self.servo_ids_edit, ",".join(str(servo_id) for servo_id in config.servo_ids))
        tool_mode = "both"
        if config.requested_tool_ids == ["0A"]:
            tool_mode = "0A"
        elif config.requested_tool_ids == ["0B"]:
            tool_mode = "0B"
        self._set_combo_value(self.tool_mode_combo, tool_mode)
        self._set_checkbox(self.include_tip_pose_check, bool(config.include_robot_frame_tip_pose))
        self._set_line_text(self.run_label_edit, str(config.run_label or ""))
        self._set_double(self.duration_spin, float(config.run_duration_s))
        self._set_double(self.warmup_spin, float(config.warmup_duration_s))
        self._set_spin(self.amplitude_spin, int(config.command_amplitude_ticks))
        self._set_double(self.step_period_spin, float(config.step_period_s))
        self._set_double(self.telemetry_poll_spin, float(config.telemetry_poll_interval_s))
        self._set_double(self.timeout_spin, float(config.timeout_s))
        self.parameter_summary_widget.set_pairs(
            [
                (
                    "Motion Protocol",
                    (
                        f"Alternating bounded step on servo IDs {', '.join(str(value) for value in config.servo_ids)} "
                        f"with amplitude {int(config.command_amplitude_ticks)} ticks and {float(config.step_period_s):.3f} s cadence."
                    ),
                ),
                (
                    "Benchmark Truth",
                    "Uses backend timing records plus servo command/telemetry host timestamps. GUI refresh rate is not part of the metric.",
                ),
                (
                    "Pose Metric",
                    (
                        "Prefer robot-frame tip pose when the loaded transform chain supports it."
                        if bool(config.include_robot_frame_tip_pose)
                        else "Use raw tool pose only for tracker-side motion metrics."
                    ),
                ),
            ]
        )

    def _on_servo_ids_changed(self) -> None:
        raw_text = self.servo_ids_edit.text().strip()
        parsed = ServoTrackerSyncValidationConfig.from_dict({"servo_ids": raw_text}).servo_ids
        self.controller.set_config_value("servo_ids", parsed)
        self._set_line_text(self.servo_ids_edit, ",".join(str(servo_id) for servo_id in parsed))

    def _on_tool_mode_changed(self) -> None:
        mode = str(self.tool_mode_combo.currentData() or "0A")
        requested_tool_ids = ["0A", "0B"] if mode == "both" else [mode]
        self.controller.set_config_value("requested_tool_ids", requested_tool_ids)


class TwoSegmentStartupValidationPage(ExperimentPageBase):
    show_visualization = False
    refresh_policy = "manual"
    page_hint = (
        "Manual dual-segment startup scaffold. Use the Servos page to jog and pretension; "
        "each checkpoint records all-8 telemetry without commanding motion."
    )

    def __init__(self, controller, experiment_name: str, parent=None) -> None:
        self.stage_summary_widget = None
        self.workflow_summary_widget = None
        super().__init__(controller, experiment_name, parent)
        self.run_button.setText("Capture Full Stage Sequence")
        self.stop_button.hide()

    def _build_parameter_sections(self) -> None:
        workflow_card = ExperimentCard(
            "Manual Stages",
            "Capture one checkpoint after each manual operator action, then use Final / Accept Startup to write the all-8 artifact.",
        )
        self.stage_summary_widget = KeyValueSummaryWidget()
        workflow_card.body_layout.addWidget(self.stage_summary_widget)
        for stage, label in [
            ("baseline", "Capture Baseline"),
            ("segment_a_pretensioned", "Capture Segment A Pretensioned"),
            ("segment_b_pretensioned", "Capture Segment B Pretensioned"),
            ("segment_a_recheck", "Capture Segment A Recheck"),
            ("final_accept", "Capture Final / Accept Startup"),
        ]:
            button = QPushButton(label)
            button.clicked.connect(lambda checked=False, stage=stage: self._capture_stage(stage))
            workflow_card.body_layout.addWidget(button)
        self.parameter_layout.addWidget(workflow_card)

        options_card = ExperimentCard("Capture Options", "Settings used by each manual checkpoint capture.")
        options_form = QFormLayout()
        self.workflow_state_edit = QLineEdit("data/two_segment_startup_validation/current_workflow_state.json")
        self.workflow_state_edit.editingFinished.connect(
            lambda: self.controller.set_config_value("workflow_state_path", self.workflow_state_edit.text())
        )
        self.allow_missing_current_check = QCheckBox("Allow missing current estimate")
        self.allow_missing_current_check.toggled.connect(
            lambda value: self.controller.set_config_value("allow_missing_current", bool(value))
        )
        self.capture_tracker_check = QCheckBox("Capture tracker snapshot when available")
        self.capture_tracker_check.toggled.connect(
            lambda value: self.controller.set_config_value("capture_tracker_snapshot", bool(value))
        )
        self.workflow_summary_widget = KeyValueSummaryWidget()
        options_form.addRow("Workflow State", self.workflow_state_edit)
        options_form.addRow("Current", self.allow_missing_current_check)
        options_form.addRow("Tracking", self.capture_tracker_check)
        options_card.body_layout.addLayout(options_form)
        options_card.body_layout.addWidget(self.workflow_summary_widget)
        self.parameter_layout.addWidget(options_card)

    def _sync_parameters_from_state(self, state: ExperimentViewState) -> None:
        _ = state
        self._set_line_text(
            self.workflow_state_edit,
            str(self.controller.get_config_value("workflow_state_path", "data/two_segment_startup_validation/current_workflow_state.json")),
        )
        self._set_checkbox(self.allow_missing_current_check, bool(self.controller.get_config_value("allow_missing_current", True)))
        self._set_checkbox(self.capture_tracker_check, bool(self.controller.get_config_value("capture_tracker_snapshot", True)))
        context = self.controller.settings.robot.operating_context()
        segments = dict(context.metadata().get("segments", {}) or {})
        stage_completion = self._stage_completion_from_state_path()
        self.stage_summary_widget.set_pairs(
            [
                ("Baseline", _stage_done(stage_completion, "baseline")),
                ("Segment A Pretensioned", _stage_done(stage_completion, "segment_a_pretensioned")),
                ("Segment B Pretensioned", _stage_done(stage_completion, "segment_b_pretensioned")),
                ("Segment A Recheck", _stage_done(stage_completion, "segment_a_recheck")),
                ("Final Accept", _stage_done(stage_completion, "final_accept")),
            ]
        )
        self.workflow_summary_widget.set_pairs(
            [
                ("Operating Mode", str(context.operating_mode)),
                ("Segment A", _segment_display(segments.get("segment_a"))),
                ("Segment B", _segment_display(segments.get("segment_b"))),
                ("Motion", "manual jog only; no automatic two-segment pretension/control"),
            ]
        )

    def _parameter_state_fingerprint(self, state: ExperimentViewState) -> tuple[object, ...]:
        _ = state
        return (
            str(self.controller.get_config_value("workflow_state_path", "")),
            bool(self.controller.get_config_value("allow_missing_current", True)),
            bool(self.controller.get_config_value("capture_tracker_snapshot", True)),
            str(self.controller.get_config_value("capture_stage", "")),
        )

    def _capture_stage(self, stage: str) -> None:
        self.controller.set_config_value("capture_stage", stage)
        self.controller.set_config_value("reset_workflow_state", stage == "baseline")
        self.controller.set_config_value("final_accept", stage == "final_accept")
        self._run()

    def _stage_completion_from_state_path(self) -> dict[str, bool]:
        raw_path = str(self.controller.get_config_value("workflow_state_path", "data/two_segment_startup_validation/current_workflow_state.json") or "")
        path = Path(raw_path)
        if not path.is_absolute():
            path = Path(self.controller.project_root) / path
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return {}
        return {str(key): bool(value) for key, value in dict(payload.get("stage_completion", {}) or {}).items()}


class TwoSegmentCollectPoseDatasetPage(ExperimentPageBase):
    show_visualization = False
    refresh_policy = "manual"
    page_hint = (
        "Collect conservative two-segment command/pose data in dual_segment mode. "
        "Servo-only or dry-run output is explicitly lower-trust and not model-training valid."
    )

    def __init__(self, controller, experiment_name: str, parent=None) -> None:
        self.summary_widget = None
        super().__init__(controller, experiment_name, parent)
        self.run_button.setText("Run Two-Segment Dataset")

    def _build_parameter_sections(self) -> None:
        schedule_card = ExperimentCard("Schedule", "Bounded open-loop command patterns around the accepted all-8 startup state.")
        form = QFormLayout()
        self.schedule_combo = NoWheelComboBox()
        for label, value in [
            ("Single Axis Micro", "single_axis_micro"),
            ("Zero", "zero"),
            ("Segment Isolation", "segment_isolation"),
            ("Small Combined", "small_combined"),
        ]:
            self.schedule_combo.addItem(label, value)
        self.schedule_combo.currentIndexChanged.connect(
            lambda _index: self.controller.set_config_value("schedule_type", self.schedule_combo.currentData())
        )
        self.max_disp_spin = QDoubleSpinBox()
        self.max_disp_spin.setRange(0.0, 5.0)
        self.max_disp_spin.setDecimals(3)
        self.max_disp_spin.setSingleStep(0.1)
        self.max_disp_spin.valueChanged.connect(
            lambda value: self.controller.set_config_value("max_segment_displacement_mm", float(value))
        )
        self.max_tick_spin = QSpinBox()
        self.max_tick_spin.setRange(1, 200)
        self.max_tick_spin.valueChanged.connect(lambda value: self.controller.set_config_value("max_tick_delta_from_startup", int(value)))
        self.samples_spin = QSpinBox()
        self.samples_spin.setRange(1, 20)
        self.samples_spin.valueChanged.connect(lambda value: self.controller.set_config_value("samples_per_pattern", int(value)))
        self.repeats_spin = QSpinBox()
        self.repeats_spin.setRange(1, 20)
        self.repeats_spin.valueChanged.connect(lambda value: self.controller.set_config_value("capture_repeats", int(value)))
        self.settle_spin = QDoubleSpinBox()
        self.settle_spin.setRange(0.0, 10.0)
        self.settle_spin.setDecimals(3)
        self.settle_spin.setSingleStep(0.05)
        self.settle_spin.valueChanged.connect(lambda value: self.controller.set_config_value("settle_time_s", float(value)))
        form.addRow("Schedule Type", self.schedule_combo)
        form.addRow("Max Displacement (mm)", self.max_disp_spin)
        form.addRow("Max Tick Delta", self.max_tick_spin)
        form.addRow("Samples / Pattern", self.samples_spin)
        form.addRow("Repeats", self.repeats_spin)
        form.addRow("Settle Time (s)", self.settle_spin)
        schedule_card.body_layout.addLayout(form)
        self.parameter_layout.addWidget(schedule_card)

        trust_card = ExperimentCard("Trust", "Trusted hardware runs require startup and tracking readiness; servo-only runs remain explicitly not trainable.")
        trust_form = QFormLayout()
        self.dry_run_check = QCheckBox("Dry run")
        self.dry_run_check.toggled.connect(lambda value: self.controller.set_config_value("dry_run", bool(value)))
        self.servo_only_check = QCheckBox("Allow servo-only test run")
        self.servo_only_check.toggled.connect(lambda value: self.controller.set_config_value("allow_servo_only_test_run", bool(value)))
        self.run_trust_combo = NoWheelComboBox()
        for label, value in [("Servo Only", "servo_only"), ("Lower Trust", "lower_trust"), ("Thesis Trusted", "thesis_trusted")]:
            self.run_trust_combo.addItem(label, value)
        self.run_trust_combo.currentIndexChanged.connect(
            lambda _index: self.controller.set_config_value("run_trust_mode", self.run_trust_combo.currentData())
        )
        self.tool_roles_edit = QPlainTextEdit()
        self.tool_roles_edit.setPlaceholderText("0A: distal_tip\n0C: intermediate")
        self.tool_roles_edit.setMinimumHeight(72)
        self.tool_roles_edit.setMaximumHeight(96)
        self.tool_roles_edit.textChanged.connect(self._on_tool_roles_changed)
        self.summary_widget = KeyValueSummaryWidget()
        trust_form.addRow("Mode", self.dry_run_check)
        trust_form.addRow("Servo Only", self.servo_only_check)
        trust_form.addRow("Run Trust", self.run_trust_combo)
        trust_form.addRow("Tool Roles", self.tool_roles_edit)
        trust_card.body_layout.addLayout(trust_form)
        trust_card.body_layout.addWidget(self.summary_widget)
        self.parameter_layout.addWidget(trust_card)

    def _sync_parameters_from_state(self, state: ExperimentViewState) -> None:
        _ = state
        config = TwoSegmentCollectPoseDatasetConfig.from_dict(self.controller.config_payload())
        self._set_combo_value(self.schedule_combo, config.schedule_type)
        self._set_double(self.max_disp_spin, float(config.max_segment_displacement_mm))
        self._set_spin(self.max_tick_spin, int(config.max_tick_delta_from_startup))
        self._set_spin(self.samples_spin, int(config.samples_per_pattern))
        self._set_spin(self.repeats_spin, int(config.capture_repeats))
        self._set_double(self.settle_spin, float(config.settle_time_s))
        self._set_checkbox(self.dry_run_check, bool(config.dry_run))
        self._set_checkbox(self.servo_only_check, bool(config.allow_servo_only_test_run))
        self._set_combo_value(self.run_trust_combo, config.run_trust_mode)
        self._set_plain_text(self.tool_roles_edit, _yaml_block(config.requested_tool_roles or {}))
        context = self.controller.settings.robot.operating_context()
        role_configs = getattr(self.controller.settings.registration, "two_segment_tracking_roles", {}) or {}
        role_summary = "; ".join(
            f"{role}={getattr(config, 'tool_id', '') or 'unconfigured'} "
            f"({'required' if getattr(config, 'required_for_two_segment_model_training', False) else 'optional'}, "
            f"{getattr(config, 'pose_kind', 'coil_origin')})"
            for role, config in sorted(dict(role_configs).items())
        )
        self.summary_widget.set_pairs(
            [
                ("Operating Mode", str(context.operating_mode)),
                ("Commanded IDs", str(list(context.commanded_servo_ids))),
                ("Tracking Roles", role_summary or "n/a"),
                ("Schedule", str(config.schedule_type)),
                ("Training Validity", "false unless accepted startup and real pose labels are present"),
            ]
        )

    def _parameter_state_fingerprint(self, state: ExperimentViewState) -> tuple[object, ...]:
        _ = state
        config = TwoSegmentCollectPoseDatasetConfig.from_dict(self.controller.config_payload())
        return (
            config.schedule_type,
            config.dry_run,
            config.max_segment_displacement_mm,
            config.max_tick_delta_from_startup,
            config.samples_per_pattern,
            config.capture_repeats,
            config.settle_time_s,
            config.allow_servo_only_test_run,
            config.run_trust_mode,
            tuple(sorted(config.requested_tool_roles.items())),
        )

    def _on_tool_roles_changed(self) -> None:
        try:
            parsed = yaml.safe_load(self.tool_roles_edit.toPlainText()) or {}
        except Exception:
            return
        if isinstance(parsed, dict):
            self.controller.set_config_value("requested_tool_roles", {str(key): str(value) for key, value in parsed.items()})


class PretensionValidationPage(ExperimentPageBase):
    refresh_policy = "manual"
    page_hint = (
        "Run single-servo onset traces, current characterization, or conservative 4-servo startup. "
        "Current is treated as a relative load proxy only; tracker XY centering is the primary startup metric."
    )

    def __init__(self, controller, experiment_name: str, parent=None) -> None:
        super().__init__(controller, experiment_name, parent)
        self.run_button.setText("Run Pretension Validation")

    def _build_parameter_sections(self) -> None:
        setup_card = ExperimentCard(
            "Validation Setup",
            "Use single-servo trace mode for detailed load-onset traces, or staged 4-servo mode for repeatable startup-state characterization.",
        )
        setup_form = QFormLayout()
        self.mode_combo = NoWheelComboBox()
        self.mode_combo.addItem("Single-Servo Trace", "single_servo_trace")
        self.mode_combo.addItem("Characterization / Identification", "single_segment_characterization")
        self.mode_combo.addItem("Conservative 4-Servo Startup", "single_segment_staged")
        self.mode_combo.currentIndexChanged.connect(self._on_mode_changed)
        self.strategy_combo = NoWheelComboBox()
        self.strategy_combo.addItem("Conservative Startup", "conservative_startup")
        self.strategy_combo.addItem("Characterization", "characterization")
        self.strategy_combo.addItem("Legacy Threshold Routine", "legacy")
        self.strategy_combo.currentIndexChanged.connect(
            lambda _index: self.controller.set_config_value("staged_strategy", str(self.strategy_combo.currentData()))
        )
        self.servo_combo = NoWheelComboBox()
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
        self.allow_current_only_check = QCheckBox("Allow current-only fallback when tracker tip pose is unavailable")
        self.allow_current_only_check.toggled.connect(
            lambda value: self.controller.set_config_value("allow_current_only_when_tracker_missing", bool(value))
        )
        self.allow_no_tracker_check = QCheckBox("Allow no-tracker current-only test run")
        self.allow_no_tracker_check.toggled.connect(
            lambda value: self.controller.set_config_value("allow_no_tracker_test_run", bool(value))
        )
        self.staged_servo_ids_edit = QLineEdit()
        self.staged_servo_ids_edit.setPlaceholderText("1,2,3,4")
        self.staged_servo_ids_edit.editingFinished.connect(self._on_staged_servo_ids_changed)
        self.repeat_runs_spin = QSpinBox()
        self.repeat_runs_spin.setRange(1, 50)
        self.repeat_runs_spin.valueChanged.connect(
            lambda value: self.controller.set_config_value("repeat_runs", int(value))
        )
        self.enable_tip_centering_check = QCheckBox("Enable staged tip XY centering (robot frame)")
        self.enable_tip_centering_check.toggled.connect(
            lambda value: self.controller.set_config_value("enable_tip_centering", bool(value))
        )
        setup_form.addRow("Servo", self.servo_combo)
        setup_form.addRow("Mode", self.mode_combo)
        setup_form.addRow("Staged Strategy", self.strategy_combo)
        setup_form.addRow("Staged Servo IDs", self.staged_servo_ids_edit)
        setup_form.addRow("Repeat Runs", self.repeat_runs_spin)
        setup_form.addRow("Tracker Metric", self.include_tracker_check)
        setup_form.addRow("Tracker Fallback", self.allow_current_only_check)
        setup_form.addRow("No-Tracker Test", self.allow_no_tracker_check)
        setup_form.addRow("Tip Centering", self.enable_tip_centering_check)
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
        self.pretension_start_mode_combo = NoWheelComboBox()
        self.pretension_start_mode_combo.addItem("Current Position", "current_position")
        self.pretension_start_mode_combo.addItem("Manual Startup Artifact", "manual_startup_artifact")
        self.pretension_start_mode_combo.addItem("Release 200 From Current", "release_200_from_current")
        self.pretension_start_mode_combo.addItem("Full Release 4095", "full_release_4095")
        self.pretension_start_mode_combo.currentIndexChanged.connect(
            lambda _index: self.controller.set_config_value(
                "pretension_start_mode",
                str(self.pretension_start_mode_combo.currentData()),
            )
        )
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
        params_form.addRow("Pretension Start Mode", self.pretension_start_mode_combo)
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

        staged_card = ExperimentCard(
            "Staged 4-Servo Parameters",
            "These staged controls apply only in staged mode and stay hidden in single-servo trace mode.",
        )
        staged_form = QFormLayout()
        self.load_balance_tolerance_spin = QDoubleSpinBox()
        self.load_balance_tolerance_spin.setRange(0.0, 5000.0)
        self.load_balance_tolerance_spin.setDecimals(1)
        self.load_balance_tolerance_spin.setSingleStep(5.0)
        self.load_balance_tolerance_spin.valueChanged.connect(
            lambda value: self.controller.set_config_value("load_balance_tolerance_ma", float(value))
        )
        self.pair_balance_tolerance_spin = QDoubleSpinBox()
        self.pair_balance_tolerance_spin.setRange(0.0, 5000.0)
        self.pair_balance_tolerance_spin.setDecimals(1)
        self.pair_balance_tolerance_spin.setSingleStep(5.0)
        self.pair_balance_tolerance_spin.valueChanged.connect(
            lambda value: self.controller.set_config_value("pair_balance_tolerance_ma", float(value))
        )
        self.tip_center_tolerance_spin = QDoubleSpinBox()
        self.tip_center_tolerance_spin.setRange(0.0, 500.0)
        self.tip_center_tolerance_spin.setDecimals(2)
        self.tip_center_tolerance_spin.setSingleStep(0.5)
        self.tip_center_tolerance_spin.valueChanged.connect(
            lambda value: self.controller.set_config_value("tip_center_tolerance_mm", float(value))
        )
        self.equalization_max_iterations_spin = QSpinBox()
        self.equalization_max_iterations_spin.setRange(0, 500)
        self.equalization_max_iterations_spin.valueChanged.connect(
            lambda value: self.controller.set_config_value("equalization_max_iterations", int(value))
        )
        self.conservative_step_spin = QSpinBox()
        self.conservative_step_spin.setRange(1, 25)
        self.conservative_step_spin.valueChanged.connect(
            lambda value: self.controller.set_config_value("conservative_step_ticks", int(value))
        )
        self.conservative_iterations_spin = QSpinBox()
        self.conservative_iterations_spin.setRange(0, 500)
        self.conservative_iterations_spin.valueChanged.connect(
            lambda value: self.controller.set_config_value("conservative_max_iterations", int(value))
        )
        self.travel_budget_spin = QSpinBox()
        self.travel_budget_spin.setRange(0, 4095)
        self.travel_budget_spin.setSpecialValueText("Disabled")
        self.travel_budget_spin.valueChanged.connect(
            lambda value: self.controller.set_config_value("conservative_max_cumulative_travel_ticks", int(value))
        )
        self.current_noise_samples_spin = QSpinBox()
        self.current_noise_samples_spin.setRange(2, 500)
        self.current_noise_samples_spin.valueChanged.connect(
            lambda value: self.controller.set_config_value("current_characterization_sample_count", int(value))
        )
        self.min_meaningful_current_delta_spin = QDoubleSpinBox()
        self.min_meaningful_current_delta_spin.setRange(0.0, 500.0)
        self.min_meaningful_current_delta_spin.setDecimals(1)
        self.min_meaningful_current_delta_spin.setSingleStep(1.0)
        self.min_meaningful_current_delta_spin.valueChanged.connect(
            lambda value: self.controller.set_config_value("min_meaningful_current_delta_ma", float(value))
        )
        staged_form.addRow("Load Balance Tolerance (mA)", self.load_balance_tolerance_spin)
        staged_form.addRow("Pair Balance Tolerance (mA)", self.pair_balance_tolerance_spin)
        staged_form.addRow("Tip XY Tolerance (mm)", self.tip_center_tolerance_spin)
        staged_form.addRow("Equalization Max Iterations", self.equalization_max_iterations_spin)
        staged_form.addRow("Conservative Step (ticks)", self.conservative_step_spin)
        staged_form.addRow("Conservative Iterations", self.conservative_iterations_spin)
        staged_form.addRow("Cumulative Travel Budget", self.travel_budget_spin)
        staged_form.addRow("Current Noise Samples", self.current_noise_samples_spin)
        staged_form.addRow("Minimum Useful Current Delta (mA)", self.min_meaningful_current_delta_spin)
        staged_card.body_layout.addLayout(staged_form)
        self.staged_mode_card = staged_card

        summary_card = ExperimentCard(
            "What This Run Produces",
            "Each run writes a canonical time-ordered sample series plus a static response plot and compact summary note under data/experiments/....",
        )
        self.parameter_summary_widget = KeyValueSummaryWidget()
        summary_card.body_layout.addWidget(self.parameter_summary_widget)

        self.parameter_layout.addWidget(setup_card)
        self.parameter_layout.addWidget(params_card)
        self.parameter_layout.addWidget(staged_card)
        self.parameter_layout.addWidget(summary_card)

    def _sync_parameters_from_state(self, state: ExperimentViewState) -> None:
        _ = state
        config = PretensionValidationExperimentConfig.from_dict(self.controller.config_payload())
        defaults = self._resolved_defaults(config.servo_id)
        mode = str(config.mode or "single_servo_trace")
        self._set_combo_value(self.mode_combo, mode)
        self._set_combo_value(self.strategy_combo, str(config.staged_strategy or "conservative_startup"))
        self._set_combo_value(
            self.pretension_start_mode_combo,
            str(config.pretension_start_mode or defaults.start_mode),
        )
        self._set_combo_value(self.servo_combo, str(int(config.servo_id)))
        self._set_checkbox(self.include_tracker_check, bool(config.include_tracker_displacement))
        self._set_checkbox(self.allow_current_only_check, bool(config.allow_current_only_when_tracker_missing))
        self._set_checkbox(self.allow_no_tracker_check, bool(getattr(config, "allow_no_tracker_test_run", False)))
        self._set_checkbox(self.enable_tip_centering_check, bool(config.enable_tip_centering))
        self._set_line_text(
            self.staged_servo_ids_edit,
            ",".join(str(int(value)) for value in (config.servo_ids or self.controller.settings.robot.servo_ids)),
        )
        self._set_spin(self.repeat_runs_spin, int(config.repeat_runs))
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
        self._set_double(self.load_balance_tolerance_spin, float(config.load_balance_tolerance_ma))
        self._set_double(self.pair_balance_tolerance_spin, float(config.pair_balance_tolerance_ma))
        self._set_double(self.tip_center_tolerance_spin, float(config.tip_center_tolerance_mm))
        self._set_spin(self.equalization_max_iterations_spin, int(config.equalization_max_iterations))
        self._set_spin(self.conservative_step_spin, int(config.conservative_step_ticks))
        self._set_spin(self.conservative_iterations_spin, int(config.conservative_max_iterations))
        self._set_spin(self.travel_budget_spin, int(config.conservative_max_cumulative_travel_ticks))
        self._set_spin(self.current_noise_samples_spin, int(config.current_characterization_sample_count))
        self._set_double(self.min_meaningful_current_delta_spin, float(config.min_meaningful_current_delta_ma))
        self._sync_mode_visibility(mode)
        strategy_label = str(config.staged_strategy or "conservative_startup").replace("_", " ")
        self.parameter_summary_widget.set_pairs(
            [
                (
                    "Mode",
                    (
                        "Single-servo trace mode captures fine-grained current/position onset behavior."
                        if mode == "single_servo_trace"
                        else "Advanced staged mode characterizes current noise, uses bounded paired moves, checks tip XY, settles, and scores the run."
                    ),
                ),
                ("Staged Strategy", strategy_label),
                (
                    "Pretension Start",
                    str(config.pretension_start_mode or defaults.start_mode).replace("_", " "),
                ),
                ("Target XY", f"{float(config.tip_target_xy_mm[0]):.2f}, {float(config.tip_target_xy_mm[1]):.2f} mm"),
                ("Quality Score", "Saved as quality_score_0_100 plus component scores for each staged run."),
                ("Debug Bundle", "Each staged run exports pretension_debug_bundle with summary, CSV, JSONL, and plots."),
                ("Travel Axis", "Travel is measured from the untensioned reference toward lower raw counts (ticks / mm)."),
                ("Current Metric", "Current is saved as an engagement/load proxy, not a tendon-force estimate."),
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

    def _on_mode_changed(self) -> None:
        mode = str(self.mode_combo.currentData() or "single_servo_trace")
        self.controller.set_config_value("mode", mode)
        if mode == "single_segment_characterization":
            self._set_combo_value(self.strategy_combo, "characterization")
            self.controller.set_config_value("staged_strategy", "characterization")
        elif mode == "single_segment_staged" and str(self.strategy_combo.currentData() or "") == "characterization":
            self._set_combo_value(self.strategy_combo, "conservative_startup")
            self.controller.set_config_value("staged_strategy", "conservative_startup")
        self._sync_mode_visibility(mode)
        self._refresh_now()

    def _on_staged_servo_ids_changed(self) -> None:
        raw_text = self.staged_servo_ids_edit.text().strip()
        parsed = PretensionValidationExperimentConfig.from_dict({"servo_ids": raw_text}).servo_ids
        self.controller.set_config_value("servo_ids", parsed)
        self._set_line_text(self.staged_servo_ids_edit, ",".join(str(value) for value in parsed))

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

    def _sync_mode_visibility(self, mode: str) -> None:
        staged_mode = str(mode or "").strip().lower() in {
            "single_segment_staged",
            "staged",
            "four_servo_staged",
            "single_segment_characterization",
            "pretension_characterization",
            "characterization",
        }
        self.servo_combo.setEnabled(not staged_mode)
        self.strategy_combo.setEnabled(staged_mode)
        self.staged_servo_ids_edit.setEnabled(staged_mode)
        self.enable_tip_centering_check.setEnabled(staged_mode)
        self.repeat_runs_spin.setEnabled(staged_mode)
        self.staged_mode_card.setVisible(staged_mode)


class CommandScheduleValidationPage(ExperimentPageBase):
    page_hint = "Use this page to validate generated sweep, grid, trajectory, or babble command schedules before using them elsewhere."

    def _build_parameter_sections(self) -> None:
        schedule_card = ExperimentCard("Schedule", "Configure the generated command schedule you want to validate.")
        schedule_form = QFormLayout()
        self.kind_combo = NoWheelComboBox()
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
    defer_visualization_until_data = True
    visualization_mode_override = VIS_MODE_PROJECTION
    page_hint = (
        "Use this workspace to collect single-segment Motor Babble modeling datasets with trusted registration, runtime tip, "
        "pretension provenance, and ordered command/pose samples for later offline training."
    )

    def __init__(self, controller, experiment_name: str, parent=None) -> None:
        super().__init__(controller, experiment_name, parent)
        self.run_button.setText("Run Motor Babble Dataset")
        self._training_window = None

    def _build_parameter_sections(self) -> None:
        collection_card = ExperimentCard("Dataset Mode", "Choose the dataset family you want to collect. All modes preserve ordered command history and explicit accepted/rejected capture state.")
        collection_form = QFormLayout()
        self.dataset_mode_combo = NoWheelComboBox()
        for label, value in (
            ("Workspace Coverage", "workspace_coverage"),
            ("Hysteresis / Path Dependence", "hysteresis_path_dependence"),
            ("Repeatability Linked", "repeatability_linked"),
        ):
            self.dataset_mode_combo.addItem(label, value)
        self.dataset_mode_combo.currentIndexChanged.connect(
            lambda _index: self.controller.set_config_value("dataset_mode", str(self.dataset_mode_combo.currentData()))
        )
        self.dry_run_check = QCheckBox("Dry Run")
        self.dry_run_check.toggled.connect(lambda value: self.controller.set_config_value("dry_run", bool(value)))
        self.run_label_edit = QLineEdit()
        self.run_label_edit.editingFinished.connect(lambda: self.controller.set_config_value("run_label", self.run_label_edit.text().strip()))
        self.dataset_tag_edit = QLineEdit()
        self.dataset_tag_edit.editingFinished.connect(lambda: self.controller.set_config_value("dataset_tag", self.dataset_tag_edit.text().strip()))
        self.sample_target_spin = QSpinBox()
        self.sample_target_spin.setRange(1, 50000)
        self.sample_target_spin.valueChanged.connect(lambda value: self.controller.set_config_value("sample_count_target", int(value)))
        self.samples_per_command_spin = QSpinBox()
        self.samples_per_command_spin.setRange(1, 20)
        self.samples_per_command_spin.valueChanged.connect(lambda value: self.controller.set_config_value("samples_per_command", int(value)))
        self.settle_time_spin = QDoubleSpinBox()
        self.settle_time_spin.setRange(0.0, 60.0)
        self.settle_time_spin.setDecimals(3)
        self.settle_time_spin.setSingleStep(0.05)
        self.settle_time_spin.valueChanged.connect(lambda value: self.controller.set_config_value("settle_time_s", float(value)))
        self.capture_timeout_spin = QDoubleSpinBox()
        self.capture_timeout_spin.setRange(0.05, 30.0)
        self.capture_timeout_spin.setDecimals(3)
        self.capture_timeout_spin.setSingleStep(0.05)
        self.capture_timeout_spin.valueChanged.connect(lambda value: self.controller.set_config_value("capture_timeout_s", float(value)))
        self.tracker_age_spin = QDoubleSpinBox()
        self.tracker_age_spin.setRange(0.01, 5.0)
        self.tracker_age_spin.setDecimals(3)
        self.tracker_age_spin.setSingleStep(0.01)
        self.tracker_age_spin.valueChanged.connect(lambda value: self.controller.set_config_value("max_tracker_age_s", float(value)))
        collection_form.addRow("Dataset Mode", self.dataset_mode_combo)
        collection_form.addRow("Dry Run", self.dry_run_check)
        collection_form.addRow("Run Label", self.run_label_edit)
        collection_form.addRow("Dataset Tag", self.dataset_tag_edit)
        collection_form.addRow("Target Accepted Samples", self.sample_target_spin)
        collection_form.addRow("Samples / Command", self.samples_per_command_spin)
        collection_form.addRow("Settle Time (s)", self.settle_time_spin)
        collection_form.addRow("Capture Timeout (s)", self.capture_timeout_spin)
        collection_form.addRow("Max Tracker Age (s)", self.tracker_age_spin)
        collection_card.body_layout.addLayout(collection_form)

        protocol_card = ExperimentCard("Command Protocol", "Tune the safe pair-command envelope and the mode-specific sequence size without editing raw YAML.")
        protocol_form = QFormLayout()
        self.workspace_amplitude_spin = QDoubleSpinBox()
        self.workspace_amplitude_spin.setRange(0.05, 5.0)
        self.workspace_amplitude_spin.setDecimals(3)
        self.workspace_amplitude_spin.setSingleStep(0.05)
        self.workspace_amplitude_spin.valueChanged.connect(lambda value: self.controller.set_config_value("workspace_amplitude_cm", float(value)))
        self.envelope_utilization_spin = QDoubleSpinBox()
        self.envelope_utilization_spin.setRange(0.05, 1.0)
        self.envelope_utilization_spin.setDecimals(2)
        self.envelope_utilization_spin.setSingleStep(0.05)
        self.envelope_utilization_spin.valueChanged.connect(lambda value: self.controller.set_config_value("envelope_utilization", float(value)))
        self.quasi_random_check = QCheckBox("Use quasi-random coverage")
        self.quasi_random_check.toggled.connect(lambda value: self.controller.set_config_value("quasi_random", bool(value)))
        self.seed_spin = QSpinBox()
        self.seed_spin.setRange(0, 999999)
        self.seed_spin.valueChanged.connect(lambda value: self.controller.set_config_value("random_seed", int(value)))
        self.hysteresis_targets_spin = QSpinBox()
        self.hysteresis_targets_spin.setRange(1, 1000)
        self.hysteresis_targets_spin.valueChanged.connect(lambda value: self.controller.set_config_value("hysteresis_target_count", int(value)))
        self.hysteresis_cycles_spin = QSpinBox()
        self.hysteresis_cycles_spin.setRange(1, 100)
        self.hysteresis_cycles_spin.valueChanged.connect(lambda value: self.controller.set_config_value("hysteresis_cycle_count", int(value)))
        self.hysteresis_prior_spin = QSpinBox()
        self.hysteresis_prior_spin.setRange(2, 8)
        self.hysteresis_prior_spin.valueChanged.connect(lambda value: self.controller.set_config_value("hysteresis_prior_family_count", int(value)))
        self.repeatability_blocks_spin = QSpinBox()
        self.repeatability_blocks_spin.setRange(1, 100)
        self.repeatability_blocks_spin.valueChanged.connect(lambda value: self.controller.set_config_value("repeatability_block_count", int(value)))
        protocol_form.addRow("Workspace Amplitude (cm)", self.workspace_amplitude_spin)
        protocol_form.addRow("Envelope Utilization", self.envelope_utilization_spin)
        protocol_form.addRow("Coverage Sampling", self.quasi_random_check)
        protocol_form.addRow("Random Seed", self.seed_spin)
        protocol_form.addRow("Hysteresis Targets", self.hysteresis_targets_spin)
        protocol_form.addRow("Hysteresis Cycles", self.hysteresis_cycles_spin)
        protocol_form.addRow("Prior-State Families", self.hysteresis_prior_spin)
        protocol_form.addRow("Repeatability Blocks", self.repeatability_blocks_spin)
        protocol_card.body_layout.addLayout(protocol_form)

        stability_card = ExperimentCard("Run Stability", "Bounded telemetry retry and warning-only load reporting for long modeling runs.")
        stability_form = QFormLayout()
        self.post_write_settle_spin = QDoubleSpinBox()
        self.post_write_settle_spin.setRange(0.0, 1.0)
        self.post_write_settle_spin.setDecimals(3)
        self.post_write_settle_spin.setSingleStep(0.01)
        self.post_write_settle_spin.valueChanged.connect(lambda value: self.controller.set_config_value("post_write_settle_s", float(value)))
        self.telemetry_retry_count_spin = QSpinBox()
        self.telemetry_retry_count_spin.setRange(0, 5)
        self.telemetry_retry_count_spin.valueChanged.connect(lambda value: self.controller.set_config_value("telemetry_retry_count", int(value)))
        self.telemetry_retry_delay_spin = QDoubleSpinBox()
        self.telemetry_retry_delay_spin.setRange(0.0, 0.5)
        self.telemetry_retry_delay_spin.setDecimals(3)
        self.telemetry_retry_delay_spin.setSingleStep(0.01)
        self.telemetry_retry_delay_spin.valueChanged.connect(lambda value: self.controller.set_config_value("telemetry_retry_delay_s", float(value)))
        self.allow_recovered_packet_errors_check = QCheckBox("Continue after recovered packet errors")
        self.allow_recovered_packet_errors_check.toggled.connect(
            lambda value: self.controller.set_config_value("allow_recovered_packet_errors", bool(value))
        )
        self.max_recovered_packet_errors_spin = QSpinBox()
        self.max_recovered_packet_errors_spin.setRange(0, 10000)
        self.max_recovered_packet_errors_spin.valueChanged.connect(
            lambda value: self.controller.set_config_value(
                "max_recovered_packet_errors_per_run",
                None if int(value) <= 0 else int(value),
            )
        )
        self.max_current_warning_spin = QSpinBox()
        self.max_current_warning_spin.setRange(0, 3000)
        self.max_current_warning_spin.valueChanged.connect(
            lambda value: self.controller.set_config_value("max_current_warning_ma", None if int(value) <= 0 else int(value))
        )
        stability_form.addRow("Post-Write Settle (s)", self.post_write_settle_spin)
        stability_form.addRow("Telemetry Retries", self.telemetry_retry_count_spin)
        stability_form.addRow("Retry Delay (s)", self.telemetry_retry_delay_spin)
        stability_form.addRow("Recovered Packet Errors", self.allow_recovered_packet_errors_check)
        stability_form.addRow("Max Recovered / Run (0=off)", self.max_recovered_packet_errors_spin)
        stability_form.addRow("Current Warning mA (0=off)", self.max_current_warning_spin)
        stability_card.body_layout.addLayout(stability_form)

        trust_card = ExperimentCard("Trust Policy", "Default behavior blocks low-trust modeling runs. Only enable overrides deliberately, and the saved summary will record them as lower-trust data.")
        trust_form = QFormLayout()
        self.allow_lower_trust_runtime_tip_check = QCheckBox("Allow quick runtime-tip override / coil-as-tip")
        self.allow_lower_trust_runtime_tip_check.toggled.connect(lambda value: self.controller.set_config_value("allow_lower_trust_runtime_tip", bool(value)))
        self.allow_lower_trust_pretension_check = QCheckBox("Allow missing or lower-trust pretension source")
        self.allow_lower_trust_pretension_check.toggled.connect(lambda value: self.controller.set_config_value("allow_lower_trust_pretension", bool(value)))
        self.allow_no_tracker_test_check = QCheckBox("Allow no-tracker servo-only hardware test")
        self.allow_no_tracker_test_check.toggled.connect(
            lambda value: self.controller.set_config_value("allow_no_tracker_test_run", bool(value))
        )
        self.export_legacy_dat_check = QCheckBox("Write legacy-compatible .dat export")
        self.export_legacy_dat_check.toggled.connect(lambda value: self.controller.set_config_value("export_legacy_dat", bool(value)))
        trust_form.addRow("Runtime Tip Override", self.allow_lower_trust_runtime_tip_check)
        trust_form.addRow("Pretension Override", self.allow_lower_trust_pretension_check)
        trust_form.addRow("No-Tracker Test", self.allow_no_tracker_test_check)
        trust_form.addRow("Legacy Export", self.export_legacy_dat_check)
        trust_card.body_layout.addLayout(trust_form)

        summary_card = ExperimentCard("Collection Summary", "Review the current trust state, planned sample volume, and active runtime dependencies before running.")
        self.collection_summary_widget = KeyValueSummaryWidget()
        summary_card.body_layout.addWidget(self.collection_summary_widget)
        self.open_training_button = QPushButton("Open ANN Training Popout")
        self.open_training_button.setProperty("variant", "ghost")
        self.open_training_button.clicked.connect(self._open_training_window)
        summary_card.body_layout.addWidget(self.open_training_button, 0, Qt.AlignLeft)

        self.parameter_layout.addWidget(collection_card)
        self.parameter_layout.addWidget(protocol_card)
        self.parameter_layout.addWidget(stability_card)
        self.parameter_layout.addWidget(trust_card)
        self.parameter_layout.addWidget(summary_card)

    def _sync_parameters_from_state(self, state: ExperimentViewState) -> None:
        _ = state
        mode = str(self.controller.get_config_value("dataset_mode", "workspace_coverage") or "workspace_coverage")
        self._set_combo_value(self.dataset_mode_combo, mode)
        self._set_checkbox(self.dry_run_check, bool(self.controller.get_config_value("dry_run", False)))
        self._set_line_text(self.run_label_edit, str(self.controller.get_config_value("run_label", "")))
        self._set_line_text(self.dataset_tag_edit, str(self.controller.get_config_value("dataset_tag", "")))
        self._set_spin(self.sample_target_spin, int(self.controller.get_config_value("sample_count_target", 120)))
        self._set_spin(
            self.samples_per_command_spin,
            int(self.controller.get_config_value("samples_per_command", self.controller.get_config_value("sample_count_per_point", 1))),
        )
        self._set_double(self.settle_time_spin, float(self.controller.get_config_value("settle_time_s", 0.15)))
        self._set_double(self.capture_timeout_spin, float(self.controller.get_config_value("capture_timeout_s", 1.0)))
        self._set_double(self.tracker_age_spin, float(self.controller.get_config_value("max_tracker_age_s", 0.15)))
        self._set_double(self.workspace_amplitude_spin, float(self.controller.get_config_value("workspace_amplitude_cm", 1.0)))
        self._set_double(self.envelope_utilization_spin, float(self.controller.get_config_value("envelope_utilization", 0.75)))
        self._set_checkbox(self.quasi_random_check, bool(self.controller.get_config_value("quasi_random", True)))
        self._set_spin(self.seed_spin, int(self.controller.get_config_value("random_seed", 0)))
        self._set_spin(self.hysteresis_targets_spin, int(self.controller.get_config_value("hysteresis_target_count", 8)))
        self._set_spin(self.hysteresis_cycles_spin, int(self.controller.get_config_value("hysteresis_cycle_count", 2)))
        self._set_spin(self.hysteresis_prior_spin, int(self.controller.get_config_value("hysteresis_prior_family_count", 4)))
        self._set_spin(self.repeatability_blocks_spin, int(self.controller.get_config_value("repeatability_block_count", 3)))
        self._set_double(self.post_write_settle_spin, float(self.controller.get_config_value("post_write_settle_s", 0.0)))
        self._set_spin(self.telemetry_retry_count_spin, int(self.controller.get_config_value("telemetry_retry_count", 2)))
        self._set_double(self.telemetry_retry_delay_spin, float(self.controller.get_config_value("telemetry_retry_delay_s", 0.03)))
        self._set_checkbox(
            self.allow_recovered_packet_errors_check,
            bool(self.controller.get_config_value("allow_recovered_packet_errors", True)),
        )
        max_recovered = self.controller.get_config_value("max_recovered_packet_errors_per_run", None)
        self._set_spin(self.max_recovered_packet_errors_spin, 0 if max_recovered in (None, "") else int(max_recovered))
        current_warning = self.controller.get_config_value("max_current_warning_ma", None)
        self._set_spin(self.max_current_warning_spin, 0 if current_warning in (None, "") else int(current_warning))
        self._set_checkbox(self.allow_lower_trust_runtime_tip_check, bool(self.controller.get_config_value("allow_lower_trust_runtime_tip", False)))
        self._set_checkbox(self.allow_lower_trust_pretension_check, bool(self.controller.get_config_value("allow_lower_trust_pretension", False)))
        self._set_checkbox(self.allow_no_tracker_test_check, bool(self.controller.get_config_value("allow_no_tracker_test_run", False)))
        self._set_checkbox(self.export_legacy_dat_check, bool(self.controller.get_config_value("export_legacy_dat", True)))

        is_hysteresis = mode == "hysteresis_path_dependence"
        is_repeatability_linked = mode == "repeatability_linked"
        self.hysteresis_targets_spin.setEnabled(is_hysteresis)
        self.hysteresis_cycles_spin.setEnabled(is_hysteresis)
        self.hysteresis_prior_spin.setEnabled(is_hysteresis)
        self.repeatability_blocks_spin.setEnabled(is_repeatability_linked)
        self.sample_target_spin.setEnabled(not is_hysteresis)

        self._sync_collection_summary(state=state, mode=mode)

    def _sync_collection_summary(self, *, state: ExperimentViewState, mode: str) -> None:
        tracking_service = getattr(self.controller, "tracking_service", None)
        tracking_snapshot = tracking_service.get_snapshot() if tracking_service is not None else None
        try:
            pretension_source = self.controller.servo_service.pretension_source_summary(list(self.controller.settings.robot.servo_ids))
            pretension_label = f"{pretension_source.source_type} ({'accepted' if pretension_source.accepted else 'not accepted'})"
        except Exception:
            pretension_label = "unavailable"
        planned_commands, planned_captures, duration_label = self._plan_preview(mode=mode)
        self.collection_summary_widget.set_pairs(
            [
                ("Mode Summary", self._mode_blurb(mode)),
                (
                    "Runtime Tip",
                    (
                        f"{tracking_snapshot.runtime_tip_mode} ({tracking_snapshot.runtime_tip_trust_level})"
                        if tracking_snapshot is not None
                        else "unavailable (servo-only possible only with explicit override)"
                    ),
                ),
                ("Pretension Source", pretension_label),
                ("Preflight", state.preflight_report.summary),
                ("Planned Commands", str(planned_commands)),
                ("Planned Captures", str(planned_captures)),
                ("Estimated Duration", duration_label),
                ("Output Root", state.planned_output_dir or "n/a"),
            ]
        )

    def _open_training_window(self) -> None:
        from continuum_robot.gui.controllers.ann_training_controller import AnnTrainingController
        from continuum_robot.gui.widgets.ann_training_window import AnnTrainingWindow

        output_root = self.controller._resolve_repo_path(self.controller.state.output_root)
        if self._training_window is None:
            training_controller = AnnTrainingController(
                project_root=self.controller.project_root,
                dataset_output_root=output_root,
            )
            self._training_window = AnnTrainingWindow(training_controller, parent=self.window())
        else:
            self._training_window.controller.set_dataset_output_root(output_root)
        preferred_dataset = self.controller.state.loaded_run_path or self.controller.state.last_output_path
        if preferred_dataset:
            self._training_window.controller.select_dataset(str(preferred_dataset))
        self._training_window.show()
        self._training_window.raise_()
        self._training_window.activateWindow()

    def _plan_preview(self, *, mode: str) -> tuple[int, int, str]:
        samples_per_command = max(1, int(self.controller.get_config_value("samples_per_command", 1)))
        settle_time_s = float(self.controller.get_config_value("settle_time_s", 0.15))
        capture_timeout_s = float(self.controller.get_config_value("capture_timeout_s", 1.0))
        if mode == "hysteresis_path_dependence":
            planned_commands = (
                max(1, int(self.controller.get_config_value("hysteresis_target_count", 8)))
                * max(1, int(self.controller.get_config_value("hysteresis_cycle_count", 2)))
                * max(2, int(self.controller.get_config_value("hysteresis_prior_family_count", 4)))
                * 2
            )
        elif mode == "repeatability_linked":
            planned_commands = (
                max(1, int(self.controller.get_config_value("repeatability_block_count", 3)))
                * max(1, int((int(self.controller.get_config_value("sample_count_target", 120)) + samples_per_command - 1) / samples_per_command))
            )
        else:
            planned_commands = max(1, int((int(self.controller.get_config_value("sample_count_target", 120)) + samples_per_command - 1) / samples_per_command))
        planned_captures = int(planned_commands * samples_per_command + 2)
        estimated_s = float(planned_commands) * (settle_time_s + min(capture_timeout_s, 0.25)) + (2.0 * min(capture_timeout_s, 0.25))
        return planned_commands, planned_captures, f"{estimated_s:.1f} s"

    @staticmethod
    def _mode_blurb(mode: str) -> str:
        if mode == "hysteresis_path_dependence":
            return "Revisits similar commands from different prior states for future state-aware modeling."
        if mode == "repeatability_linked":
            return "Collects repeated trusted startup blocks for robot/system revision comparison."
        return "Bounded workspace exploration for first-pass forward-model training."


class PenprobeChasingDemoPage(ExperimentPageBase):
    show_visualization = True
    defer_visualization_until_data = True
    visualization_mode_override = VIS_MODE_PROJECTION
    page_hint = "MVP demo: chase live 0B penprobe XY target with the active single-segment 0A coil-as-tip. Commands stay bounded from the accepted startup reference."

    @staticmethod
    def _penprobe_demo_travel_warning_text(max_tick_delta_from_startup: int) -> str:
        parts: list[str] = []
        if int(max_tick_delta_from_startup) > 300:
            parts.append(
                "Large penprobe demo travel. Confirm spine is clear and tendon routing is safe."
            )
        if int(max_tick_delta_from_startup) >= 700:
            parts.append(
                "Near GUI max; ensure hard_max_tick_delta_from_startup in YAML allows this value."
            )
        return " ".join(parts).strip()

    def __init__(self, controller, experiment_name: str, parent=None) -> None:
        super().__init__(controller, experiment_name, parent)
        self.run_button.setText("Start Demo")
        nominal_hz = float(PenprobeChasingDemoConfig.from_dict(self.controller.config_payload()).gui_status_nominal_hz)
        interval_ms = int(max(100, min(200, round(1000.0 / max(nominal_hz, 1.0)))))
        self._chase_status_timer = QTimer(self)
        self._chase_status_timer.setInterval(interval_ms)
        self._chase_status_timer.timeout.connect(self._pulse_chasing_hud)
        self._chase_status_timer.start()

    def shutdown(self) -> None:
        timer = getattr(self, "_chase_status_timer", None)
        if timer is not None and timer.isActive():
            timer.stop()

    def _pulse_chasing_hud(self) -> None:
        self.controller.note_penprobe_gui_pulse()
        text = str(getattr(self.controller.state, "penprobe_chase_live_summary", "") or "").strip()
        if hasattr(self, "loop_status_label") and text:
            self.loop_status_label.setText(text)

    def _build_parameter_sections(self) -> None:
        control_card = ExperimentCard("Demo Controls", "Keep the envelope conservative while validating 0A-to-0B chasing behavior.")
        form = QFormLayout()
        self.max_tick_delta_spin = QSpinBox()
        self.max_tick_delta_spin.setRange(1, 800)
        self.max_tick_delta_spin.valueChanged.connect(
            lambda value: self.controller.set_config_value("max_tick_delta_from_startup", int(value))
        )
        self.loop_period_spin = QDoubleSpinBox()
        self.loop_period_spin.setRange(0.02, 1.0)
        self.loop_period_spin.setDecimals(3)
        self.loop_period_spin.setSingleStep(0.025)
        self.loop_period_spin.valueChanged.connect(lambda value: self.controller.set_config_value("loop_period_s", float(value)))
        self.duration_spin = QDoubleSpinBox()
        self.duration_spin.setRange(0.0, 300.0)
        self.duration_spin.setDecimals(1)
        self.duration_spin.setSingleStep(5.0)
        self.duration_spin.valueChanged.connect(lambda value: self.controller.set_config_value("max_duration_s", float(value)))
        self.max_iterations_spin = QSpinBox()
        self.max_iterations_spin.setRange(1, 20000)
        self.max_iterations_spin.valueChanged.connect(lambda value: self.controller.set_config_value("max_iterations", int(value)))
        self.mapping_mode_combo = NoWheelComboBox()
        self.mapping_mode_combo.addItem("Paired XY Proportional", MAPPING_PAIRED_XY_PROPORTIONAL)
        self.mapping_mode_combo.addItem("Legacy Polynomial Workspace", MAPPING_LEGACY_POLYNOMIAL_WORKSPACE)
        self.mapping_mode_combo.currentIndexChanged.connect(
            lambda _index: self.controller.set_config_value("mapping_mode", str(self.mapping_mode_combo.currentData()))
        )
        self.max_radius_spin = QDoubleSpinBox()
        self.max_radius_spin.setRange(0.0, 300.0)
        self.max_radius_spin.setDecimals(1)
        self.max_radius_spin.setSingleStep(5.0)
        self.max_radius_spin.valueChanged.connect(lambda value: self.controller.set_config_value("max_target_radius_mm", float(value)))
        self.gain_spin = QDoubleSpinBox()
        self.gain_spin.setRange(0.0, 50.0)
        self.gain_spin.setDecimals(1)
        self.gain_spin.setSingleStep(1.0)
        self.gain_spin.valueChanged.connect(lambda value: self.controller.set_config_value("proportional_gain_ticks_per_mm", float(value)))
        self.max_step_spin = QSpinBox()
        self.max_step_spin.setRange(1, 100)
        self.max_step_spin.valueChanged.connect(lambda value: self.controller.set_config_value("max_tick_step_per_cycle", int(value)))
        self.demo_travel_warning = QLabel("")
        self.demo_travel_warning.setWordWrap(True)
        self.demo_travel_warning.setProperty("role", "warning")
        self.max_tick_delta_spin.valueChanged.connect(
            lambda value: self.demo_travel_warning.setText(
                self._penprobe_demo_travel_warning_text(int(value))
            )
        )
        self.deadband_spin = QDoubleSpinBox()
        self.deadband_spin.setRange(0.0, 20.0)
        self.deadband_spin.setDecimals(1)
        self.deadband_spin.setSingleStep(0.5)
        self.deadband_spin.valueChanged.connect(lambda value: self.controller.set_config_value("xy_deadband_mm", float(value)))
        self.stale_timeout_spin = QDoubleSpinBox()
        self.stale_timeout_spin.setRange(0.02, 2.0)
        self.stale_timeout_spin.setDecimals(2)
        self.stale_timeout_spin.setSingleStep(0.05)
        self.stale_timeout_spin.valueChanged.connect(lambda value: self.controller.set_config_value("stale_tracker_timeout_s", float(value)))
        self.max_write_hz_spin = QDoubleSpinBox()
        self.max_write_hz_spin.setRange(1.0, 60.0)
        self.max_write_hz_spin.setDecimals(1)
        self.max_write_hz_spin.setSingleStep(5.0)
        self.max_write_hz_spin.valueChanged.connect(lambda value: self.controller.set_config_value("max_servo_write_hz", float(value)))
        self.flip_x_checkbox = QCheckBox("Flip X")
        self.flip_x_checkbox.toggled.connect(lambda value: self.controller.set_config_value("flip_x", bool(value)))
        self.flip_y_checkbox = QCheckBox("Flip Y")
        self.flip_y_checkbox.toggled.connect(lambda value: self.controller.set_config_value("flip_y", bool(value)))
        form.addRow("Max Delta From Startup (ticks)", self.max_tick_delta_spin)
        form.addRow("", self.demo_travel_warning)
        form.addRow("Loop Period (s)", self.loop_period_spin)
        form.addRow("Max Duration (s)", self.duration_spin)
        form.addRow("Max Iterations", self.max_iterations_spin)
        form.addRow("Mapping Mode", self.mapping_mode_combo)
        form.addRow("Max Target Radius (mm)", self.max_radius_spin)
        form.addRow("Gain (ticks/mm)", self.gain_spin)
        form.addRow("Max Step / Cycle (ticks)", self.max_step_spin)
        form.addRow("XY Deadband (mm)", self.deadband_spin)
        form.addRow("Tracker Stale Timeout (s)", self.stale_timeout_spin)
        form.addRow("Max Servo Write Hz", self.max_write_hz_spin)
        form.addRow("Sign Override", self.flip_x_checkbox)
        form.addRow("", self.flip_y_checkbox)
        control_card.body_layout.addLayout(form)
        self.demo_travel_warning.setText(
            self._penprobe_demo_travel_warning_text(int(self.max_tick_delta_spin.value()))
        )

        status_card = ExperimentCard("Live Status", "Last received sample and active scope for the running demo.")
        self.loop_status_label = QLabel(
            "Control: — Hz, Servo writes: — Hz, Tracker age: — ms (running updates appear here)."
        )
        self.loop_status_label.setWordWrap(True)
        self.loop_status_label.setProperty("role", "body")
        status_card.body_layout.addWidget(self.loop_status_label)
        self.demo_status_widget = KeyValueSummaryWidget()
        status_card.body_layout.addWidget(self.demo_status_widget)
        self.parameter_layout.addWidget(control_card)
        self.parameter_layout.addWidget(status_card)

    def _sync_parameters_from_state(self, state: ExperimentViewState) -> None:
        _ = state
        config = PenprobeChasingDemoConfig.from_dict(self.controller.config_payload())
        self._set_spin(self.max_tick_delta_spin, int(config.max_tick_delta_from_startup))
        if hasattr(self, "demo_travel_warning"):
            self.demo_travel_warning.setText(
                self._penprobe_demo_travel_warning_text(int(config.max_tick_delta_from_startup))
            )
        self._set_double(self.loop_period_spin, float(config.loop_period_s))
        self._set_double(self.duration_spin, float(config.max_duration_s))
        self._set_spin(self.max_iterations_spin, int(config.max_iterations))
        self._set_combo_value(self.mapping_mode_combo, str(config.mapping_mode))
        self._set_double(self.max_radius_spin, float(config.max_target_radius_mm))
        self._set_double(self.gain_spin, float(config.proportional_gain_ticks_per_mm))
        self._set_spin(self.max_step_spin, int(config.max_tick_step_per_cycle))
        self._set_double(self.deadband_spin, float(config.xy_deadband_mm))
        self._set_double(self.stale_timeout_spin, float(config.stale_tracker_timeout_s))
        self._set_double(self.max_write_hz_spin, float(config.max_servo_write_hz))
        with QSignalBlocker(self.flip_x_checkbox):
            self.flip_x_checkbox.setChecked(bool(config.flip_x))
        with QSignalBlocker(self.flip_y_checkbox):
            self.flip_y_checkbox.setChecked(bool(config.flip_y))
        self._sync_demo_status(state=state)

    def _sync_demo_status(self, *, state: ExperimentViewState) -> None:
        context = self.controller.settings.robot.operating_context()
        last_sample = None
        samples = getattr(self.controller, "_live_samples", [])
        if samples:
            last_sample = samples[-1]
        extra = dict(getattr(last_sample, "extra", {}) or {}) if last_sample is not None else {}
        hud = str(getattr(state, "penprobe_chase_live_summary", "") or "").strip()
        if hasattr(self, "loop_status_label"):
            self.loop_status_label.setText(
                hud
                or "Control: — Hz, Servo writes: — Hz, Tracker age: — ms (when the demo runs, live Hz fills in here)."
            )
        tip_xy = extra.get("tip_xy_mm")
        target_xy = extra.get("target_xy_mm")
        error_norm = extra.get("xy_error_norm_mm")
        max_delta = extra.get("max_tick_delta_used")
        step_by_servo = extra.get("commanded_tick_step_by_servo")
        goal_by_servo = getattr(last_sample, "commanded_motor_values", {}) if last_sample is not None else {}
        mapping_debug = dict(extra.get("mapping_debug", {}) or {})
        distance_to_cap = mapping_debug.get("distance_to_cap_ticks")
        write_rate = extra.get("actual_servo_write_hz")
        control_hz = extra.get("actual_control_hz")
        gui_hz = extra.get("actual_gui_update_hz")
        pairs = context.active_pairs if context.operating_mode == "single_segment" else {}
        self.demo_status_widget.set_pairs(
            [
                ("Operating Mode", str(context.operating_mode)),
                ("Active Segment", f"{context.active_segment_label or 'n/a'} ({context.active_segment_key or 'n/a'})"),
                ("Servo IDs", ", ".join(str(value) for value in context.active_segment_servo_ids) or "n/a"),
                ("Pairs", str(pairs or "n/a")),
                ("Tip 0A XY", _format_xy_pair(tip_xy)),
                ("Target 0B XY", _format_xy_pair(target_xy)),
                ("XY Error", f"{float(error_norm):.2f} mm" if error_norm is not None else "n/a"),
                ("Commanded Step", str(step_by_servo or "n/a")),
                ("Goal Ticks", str(goal_by_servo or "n/a")),
                ("Max Delta Used", f"{int(max_delta)} ticks" if max_delta is not None else "n/a"),
                ("Distance To Cap", f"{int(distance_to_cap)} ticks" if distance_to_cap is not None else "n/a"),
                ("Tracker Freshness", f"{float(getattr(last_sample, 'freshness_s', 0.0)):.3f} s" if last_sample is not None and getattr(last_sample, "freshness_s", None) is not None else "n/a"),
                ("Control Hz (instant)", f"{float(control_hz):.1f} Hz" if isinstance(control_hz, (int, float)) else "n/a"),
                ("Servo writes Hz", f"{float(write_rate):.1f} Hz" if isinstance(write_rate, (int, float)) else "n/a"),
                ("GUI Hz probe", f"{float(gui_hz):.1f} Hz" if isinstance(gui_hz, (int, float)) else "n/a"),
                ("Preflight", state.preflight_report.summary),
            ]
        )

    def _parameter_state_fingerprint(self, state: ExperimentViewState) -> tuple[object, ...]:
        cfg_key = json.dumps(self.controller.config_payload(), sort_keys=True, default=str)
        if bool(state.run_active) and str(state.selected_experiment or "") == str(self.experiment_name):
            return (
                cfg_key,
                True,
                str(state.preflight_report.overall_status),
            )
        samples = getattr(self.controller, "_live_samples", [])
        last_sample = samples[-1] if samples else None
        return (
            cfg_key,
            getattr(last_sample, "sample_index", None),
            state.preflight_report.overall_status,
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


class _ValidationSelectionPage(ExperimentPageBase):
    """Shared offline multi-run selection page for thesis validation analyses."""

    candidates_loaded = Signal(int, object, object)
    TABLE_APPLY_BATCH_SIZE = 16

    show_visualization = True
    refresh_policy = "manual"
    parameter_panel_scrollable = False
    defer_visualization_until_data = True
    visualization_mode_override = VIS_MODE_PROJECTION

    selection_title = "Source Runs"
    selection_subtitle = "Choose the saved runs to analyze."
    run_button_text = "Run Validation"
    table_headers: tuple[str, ...] = ()

    def __init__(self, controller, experiment_name: str, parent=None) -> None:
        self._table_signature: tuple[tuple[str, ...], ...] | None = None
        self._candidate_cache: list[object] = []
        self._candidate_load_generation = 0
        self._candidate_loading = False
        self._candidate_cache_loaded = False
        self._candidate_error: str | None = None
        self._candidate_cache_key: tuple[str, str] | None = None
        self._pending_candidates: list[object] = []
        self._pending_selected: set[str] = set()
        self._table_apply_generation = 0
        self._table_apply_row = 0
        self._table_apply_complete = True
        self._table_apply_started_s = 0.0
        self._table_apply_token = 0
        self._page_closing = False
        super().__init__(controller, experiment_name, parent)
        self.run_button.setText(self.run_button_text)
        self._install_interaction_instrumentation()
        self.candidates_loaded.connect(self._apply_loaded_candidates)
        self.destroyed.connect(lambda *_args: self.shutdown())

    def shutdown(self) -> None:
        self._page_closing = True
        self._candidate_load_generation += 1
        self._table_apply_token += 1
        self._candidate_loading = False

    def _page_is_valid(self) -> bool:
        return bool(not self._page_closing and _qt_is_valid(self))

    def _build_parameter_sections(self) -> None:
        selection_card = ExperimentCard(self.selection_title, self.selection_subtitle)
        selection_card.setObjectName(f"{self.experiment_name}SelectionCard")
        toolbar = QHBoxLayout()
        toolbar.setContentsMargins(0, 0, 0, 0)
        toolbar.setSpacing(8)
        refresh_button = QPushButton("Refresh")
        refresh_button.setProperty("variant", "ghost")
        refresh_button.clicked.connect(self._refresh_sources)
        select_all_button = QPushButton("Select All")
        select_all_button.setProperty("variant", "ghost")
        select_all_button.clicked.connect(lambda: self._set_all_selected(True))
        clear_button = QPushButton("Clear")
        clear_button.setProperty("variant", "ghost")
        clear_button.clicked.connect(lambda: self._set_all_selected(False))
        toolbar.addWidget(refresh_button)
        toolbar.addWidget(select_all_button)
        toolbar.addWidget(clear_button)
        toolbar.addStretch(1)
        selection_card.body_layout.addLayout(toolbar)

        self.selection_summary_widget = KeyValueSummaryWidget()
        selection_card.body_layout.addWidget(self.selection_summary_widget)

        self.loading_label = QLabel("Loading saved runs...")
        self.loading_label.setObjectName(f"{self.experiment_name}LoadingLabel")
        self.loading_label.setProperty("role", "muted")
        self.loading_label.setWordWrap(True)
        self.loading_label.setEnabled(False)
        self.loading_label.setAttribute(Qt.WA_TransparentForMouseEvents, True)
        self.loading_label.hide()
        selection_card.body_layout.addWidget(self.loading_label)

        self.run_table = QTableWidget(0, len(self.table_headers))
        self.run_table.setObjectName(f"{self.experiment_name}RunTable")
        self.run_table.viewport().setObjectName(f"{self.experiment_name}RunTableViewport")
        self.run_table.setHorizontalHeaderLabels(list(self.table_headers))
        self.run_table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.run_table.setSelectionMode(QAbstractItemView.SingleSelection)
        self.run_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.run_table.setFocusPolicy(Qt.ClickFocus)
        self.run_table.verticalHeader().setVisible(False)
        self.run_table.horizontalHeader().setStretchLastSection(True)
        self.run_table.setMinimumHeight(280)
        self.run_table.itemChanged.connect(self._on_item_changed)
        selection_card.body_layout.addWidget(self.run_table)

        help_card = ExperimentCard(
            "Saved Outputs",
            "Each validation run writes summary.json, metrics.csv, config_snapshot.yaml, a concise summary text file, and thesis-ready PNG plots under the canonical experiment output folder.",
        )
        self.parameter_layout.addWidget(selection_card)
        self.parameter_layout.addWidget(help_card)

    def _sync_parameters_from_state(self, state: ExperimentViewState) -> None:
        selected = {str(value) for value in (self.controller.get_config_value("run_paths", []) or [])}
        started = time.monotonic()
        self._log_stage(
            "set_state",
            "start",
            candidate_count=len(self._candidate_cache),
            selected_count=len(selected),
            loading=self._candidate_loading,
            table_rows=self.run_table.rowCount(),
        )
        _ = state
        self._ensure_candidates_loaded()
        candidates = list(self._candidate_cache)
        self._sync_table(candidates, selected)
        self.selection_summary_widget.set_pairs(self._summary_pairs(candidates, selected))
        self._sync_loading_state(candidates)
        self._log_interaction_snapshot("set_state_end")
        self._log_stage(
            "set_state",
            "end",
            started_s=started,
            candidate_count=len(candidates),
            selected_count=len(selected),
            loading=self._candidate_loading,
            table_rows=self.run_table.rowCount(),
        )

    def _refresh_sources(self) -> None:
        started = time.monotonic()
        self._log_stage("explicit_refresh", "start", candidate_count=len(self._candidate_cache))
        self._ensure_candidates_loaded(force=True)
        selected = {str(value) for value in (self.controller.get_config_value("run_paths", []) or [])}
        self._sync_loading_state(list(self._candidate_cache))
        self.selection_summary_widget.set_pairs(self._summary_pairs(list(self._candidate_cache), selected))
        self._log_stage(
            "explicit_refresh",
            "end",
            started_s=started,
            candidate_count=len(self._candidate_cache),
            loading=self._candidate_loading,
        )

    def _set_all_selected(self, selected: bool) -> None:
        candidates = list(self._candidate_cache)
        run_paths = [candidate.path for candidate in candidates] if selected else []
        self.controller.set_config_value("run_paths", run_paths)
        selected_paths = {str(value) for value in run_paths}
        self._sync_table(candidates, selected_paths)
        self.selection_summary_widget.set_pairs(self._summary_pairs(candidates, selected_paths))

    def _on_item_changed(self, item: QTableWidgetItem) -> None:
        if item.column() != 0:
            return
        started = time.monotonic()
        self._log_stage("selection_change", "start", candidate_count=len(self._candidate_cache))
        selected_paths: list[str] = []
        for row in range(self.run_table.rowCount()):
            path_item = self.run_table.item(row, 0)
            if path_item is None:
                continue
            if path_item.checkState() == Qt.Checked:
                raw_path = path_item.data(Qt.UserRole)
                if raw_path:
                    selected_paths.append(str(raw_path))
        known = {candidate.path for candidate in self._candidate_cache}
        selected_paths = [path for path in selected_paths if path in known]
        self.controller.set_config_value("run_paths", selected_paths)
        self._log_stage(
            "selection_change",
            "end",
            started_s=started,
            selected_count=len(selected_paths),
            candidate_count=len(self._candidate_cache),
        )

    def _sync_table(self, candidates, selected: set[str]) -> None:
        signature = tuple(self._row_signature(candidate, selected) for candidate in candidates)
        if not candidates and self.run_table.rowCount() == 0 and self._table_signature in {None, ()}:
            self._table_signature = signature
            self._table_apply_complete = True
            return
        if signature == self._table_signature:
            if not self._table_apply_complete and self.controller.state.selected_experiment == self.experiment_name:
                self._schedule_table_apply(self._table_apply_generation)
            return
        self._table_signature = signature
        self._table_apply_generation += 1
        self._table_apply_row = 0
        self._table_apply_complete = False
        self._table_apply_started_s = time.monotonic()
        self._pending_candidates = list(candidates)
        self._pending_selected = set(selected)
        self._log_stage(
            "table_rebuild",
            "start",
            generation=self._table_apply_generation,
            candidate_count=len(candidates),
        )

        def _initialize_table() -> None:
            with QSignalBlocker(self.run_table):
                self.run_table.clearContents()
                self.run_table.setRowCount(len(candidates))

        preserve_scroll_position(self.run_table, _initialize_table)
        self._schedule_table_apply(self._table_apply_generation)

    def _row_signature(self, candidate, selected: set[str]) -> tuple[str, ...]:
        return (*self._row_values(candidate), "1" if candidate.path in selected else "0")

    def _summary_pairs(self, candidates, selected: set[str]) -> list[tuple[str, str]]:
        included = [candidate for candidate in candidates if candidate.path in selected]
        return [
            ("Available Runs", str(len(candidates))),
            ("Included", str(len(included))),
            ("Selection", self._selection_summary_text(included)),
        ]

    def _selection_summary_text(self, included) -> str:
        if not included:
            return "Select at least 2 saved runs."
        return f"{len(included)} run(s) selected."

    def _parameter_state_fingerprint(self, state: ExperimentViewState) -> tuple[object, ...]:
        _ = state
        selected = tuple(str(value) for value in (self.controller.get_config_value("run_paths", []) or []))
        return (
            selected,
            len(self._candidate_cache),
            self._candidate_loading,
            self._candidate_error or "",
            self._candidate_cache_loaded,
            self._table_apply_complete,
            self.run_table.rowCount(),
        )

    def _discover_candidates(self):
        raise NotImplementedError

    def _row_values(self, candidate) -> tuple[str, ...]:
        raise NotImplementedError

    def _ensure_candidates_loaded(self, *, force: bool = False) -> None:
        key = (str(self.controller.project_root), self.experiment_name)
        if force or self._candidate_cache_key != key:
            self._candidate_cache_key = key
            self._candidate_cache = []
            self._candidate_cache_loaded = False
            self._table_signature = None
        if self._candidate_loading:
            self._log_stage(
                "async_discovery",
                "skip",
                reason="already_loading",
                generation=self._candidate_load_generation,
            )
            return
        if self._candidate_cache_loaded and not force:
            self._log_stage("async_discovery", "skip", reason="cache_hit", candidate_count=len(self._candidate_cache))
            return
        self._candidate_loading = True
        self._candidate_error = None
        self._candidate_load_generation += 1
        generation = self._candidate_load_generation
        started = time.monotonic()
        self._log_stage("async_discovery", "start", generation=generation, force=force)

        def _worker() -> None:
            try:
                candidates = list(self._discover_candidates())
                error = None
            except Exception as exc:
                LOG.exception(
                    "Validation source discovery failed | experiment=%s | error=%s",
                    self.experiment_name,
                    exc,
                )
                candidates = []
                error = str(exc)
            elapsed_ms = (time.monotonic() - started) * 1000.0
            self._log_stage(
                "async_discovery",
                "end",
                elapsed_ms=elapsed_ms,
                generation=generation,
                candidate_count=len(candidates),
                error=error or "",
            )
            if self._page_closing:
                return
            try:
                self.candidates_loaded.emit(generation, candidates, error)
            except RuntimeError:
                LOG.debug(
                    "Validation source discovery result dropped after page shutdown | experiment=%s",
                    self.experiment_name,
                )

        threading.Thread(
            target=_worker,
            name=f"{self.experiment_name}-source-scan",
            daemon=True,
        ).start()

    def _apply_loaded_candidates(self, generation: int, candidates, error: object) -> None:
        if not self._page_is_valid():
            return
        started = time.monotonic()
        self._log_stage(
            "candidate_apply",
            "start",
            generation=generation,
            candidate_count=len(candidates or []),
        )
        if int(generation) != int(self._candidate_load_generation):
            self._log_stage("candidate_apply", "skip", reason="stale_generation", generation=generation)
            return
        self._candidate_loading = False
        self._candidate_error = str(error) if error else None
        self._candidate_cache = list(candidates or [])
        self._candidate_cache_loaded = True
        self._table_signature = None
        if self.controller.state.selected_experiment != self.experiment_name:
            self._log_stage(
                "candidate_apply",
                "defer",
                started_s=started,
                generation=generation,
                candidate_count=len(self._candidate_cache),
                reason="page_inactive",
            )
            return
        selected = {str(value) for value in (self.controller.get_config_value("run_paths", []) or [])}
        self._sync_table(self._candidate_cache, selected)
        self.selection_summary_widget.set_pairs(self._summary_pairs(self._candidate_cache, selected))
        self._sync_loading_state(self._candidate_cache)
        self._log_interaction_snapshot("candidate_apply_end")
        self._log_stage(
            "candidate_apply",
            "end",
            started_s=started,
            generation=generation,
            candidate_count=len(self._candidate_cache),
        )

    def _sync_loading_state(self, candidates) -> None:
        has_candidates = bool(candidates)
        if self._candidate_loading:
            self.loading_label.setText("Loading saved runs..." if not has_candidates else "Refreshing saved runs...")
            self.loading_label.setVisible(not has_candidates)
            self.loading_label.setEnabled(not has_candidates)
            self.run_table.setEnabled(has_candidates)
            self.run_table.setVisible(has_candidates)
            self._log_surface_state("loading_state")
            return
        if self._candidate_error:
            self.loading_label.setText(f"Could not load saved runs: {self._candidate_error}")
            self.loading_label.setVisible(not has_candidates)
            self.loading_label.setEnabled(not has_candidates)
            self.run_table.setEnabled(has_candidates)
            self.run_table.setVisible(has_candidates)
            self._log_surface_state("error_state")
            return
        self.loading_label.setVisible(not has_candidates)
        self.loading_label.setEnabled(not has_candidates)
        if not has_candidates:
            self.loading_label.setText("No saved runs found yet.")
        self.run_table.setVisible(has_candidates)
        self.run_table.setEnabled(has_candidates)
        self._log_surface_state("loaded_state")

    def _schedule_table_apply(self, generation: int) -> None:
        if not self._page_is_valid():
            return
        self._table_apply_token += 1
        token = self._table_apply_token
        QTimer.singleShot(0, lambda current_generation=generation, current_token=token: self._apply_table_batch(current_generation, current_token))

    def _apply_table_batch(self, generation: int, token: int) -> None:
        if not self._page_is_valid():
            return
        if int(token) != int(self._table_apply_token):
            self._log_stage("table_rebuild", "skip", reason="stale_callback", generation=generation, token=token)
            return
        if int(generation) != int(self._table_apply_generation):
            self._log_stage("table_rebuild", "skip", reason="stale_generation", generation=generation)
            return
        if self.controller.state.selected_experiment != self.experiment_name:
            self._log_stage(
                "page_visibility",
                "defer",
                generation=generation,
                candidate_count=len(self._pending_candidates),
                reason="page_inactive_during_table_apply",
            )
            return
        start_row = int(self._table_apply_row)
        end_row = min(start_row + self.TABLE_APPLY_BATCH_SIZE, len(self._pending_candidates))
        if start_row >= end_row:
            self._table_apply_complete = True
            self._log_stage(
                "table_rebuild",
                "end",
                started_s=self._table_apply_started_s,
                generation=generation,
                candidate_count=len(self._pending_candidates),
            )
            return
        with QSignalBlocker(self.run_table):
            for row in range(start_row, end_row):
                candidate = self._pending_candidates[row]
                row_values = self._row_values(candidate)
                for column, text in enumerate(row_values):
                    item = QTableWidgetItem(text)
                    if column == 0:
                        item.setFlags(item.flags() | Qt.ItemIsUserCheckable | Qt.ItemIsEnabled | Qt.ItemIsSelectable)
                        item.setCheckState(Qt.Checked if candidate.path in self._pending_selected else Qt.Unchecked)
                        item.setData(Qt.UserRole, candidate.path)
                    self.run_table.setItem(row, column, item)
        self._table_apply_row = end_row
        if end_row < len(self._pending_candidates):
            self._schedule_table_apply(generation)
            return
        self._table_apply_complete = True
        self._log_interaction_snapshot("table_apply_end")
        self._log_stage(
            "table_rebuild",
            "end",
            started_s=self._table_apply_started_s,
            generation=generation,
            candidate_count=len(self._pending_candidates),
        )

    def _log_stage(self, stage: str, event: str, *, started_s: float | None = None, elapsed_ms: float | None = None, **fields) -> None:
        details = dict(fields)
        if elapsed_ms is None and started_s is not None:
            elapsed_ms = (time.monotonic() - float(started_s)) * 1000.0
        if elapsed_ms is not None:
            details["elapsed_ms"] = f"{float(elapsed_ms):.1f}"
        details["gui_thread"] = self._is_gui_thread()
        details["experiment"] = self.experiment_name
        rendered = " ".join(f"{key}={value}" for key, value in details.items())
        LOG.info("ValidationPage[%s] %s | %s", stage, event, rendered)

    def _install_interaction_instrumentation(self) -> None:
        if self.viewer_placeholder is not None:
            self.viewer_placeholder.setObjectName(f"{self.experiment_name}ViewerPlaceholder")
            self.viewer_placeholder.setAttribute(Qt.WA_TransparentForMouseEvents, True)
        watched_widgets = [
            self,
            self.scroll_area,
            self.scroll_area.viewport(),
            self.parameter_container,
            self.loading_label,
            self.run_table,
            self.run_table.viewport(),
            self.viewer_placeholder,
        ]
        for widget in watched_widgets:
            if widget is not None:
                widget.installEventFilter(self)

    def eventFilter(self, watched, event) -> bool:  # pragma: no cover - log-only Pi diagnostics
        event_type = event.type()
        if event_type in {
            QEvent.Type.MouseButtonPress,
            QEvent.Type.Wheel,
            QEvent.Type.FocusIn,
            QEvent.Type.FocusOut,
        }:
            self._log_stage(
                "interaction",
                _event_type_name(event_type),
                source=_widget_debug_name(watched),
                visible=getattr(watched, "isVisible", lambda: False)(),
                enabled=getattr(watched, "isEnabled", lambda: False)(),
                table_visible=self.run_table.isVisible(),
                table_enabled=self.run_table.isEnabled(),
                loading_visible=self.loading_label.isVisible(),
                loading_enabled=self.loading_label.isEnabled(),
                viewer_placeholder_visible=(
                    self.viewer_placeholder.isVisible() if self.viewer_placeholder is not None else False
                ),
                viewer_backend=getattr(self.viewer_3d, "backend_mode", "none"),
            )
        return super().eventFilter(watched, event)

    def _log_surface_state(self, event: str) -> None:
        self._log_stage(
            "surface_state",
            event,
            loading_visible=self.loading_label.isVisible(),
            loading_enabled=self.loading_label.isEnabled(),
            loading_transparent=self.loading_label.testAttribute(Qt.WA_TransparentForMouseEvents),
            table_visible=self.run_table.isVisible(),
            table_enabled=self.run_table.isEnabled(),
            table_rows=self.run_table.rowCount(),
            table_viewport_enabled=self.run_table.viewport().isEnabled(),
            table_focus_policy=str(self.run_table.focusPolicy()),
            viewer_placeholder_visible=(
                self.viewer_placeholder.isVisible() if self.viewer_placeholder is not None else False
            ),
            viewer_placeholder_transparent=(
                self.viewer_placeholder.testAttribute(Qt.WA_TransparentForMouseEvents)
                if self.viewer_placeholder is not None
                else False
            ),
            viewer_backend=getattr(self.viewer_3d, "backend_mode", "none"),
            viewer_container_present=bool(getattr(self.viewer_3d, "_container", None)),
        )

    def _log_interaction_snapshot(self, event: str, *, selector: QWidget | None = None) -> None:
        selector_top = _widget_at_center(selector) if selector is not None else "n/a"
        table_top = _widget_at_center(self.run_table) if self.run_table.isVisible() else "n/a"
        self._log_stage(
            "interaction_snapshot",
            event,
            page_visible=self.isVisible(),
            page_enabled=self.isEnabled(),
            scroll_visible=self.scroll_area.isVisible(),
            scroll_enabled=self.scroll_area.isEnabled(),
            selector_top=selector_top,
            table_top=table_top,
            loading_rect=_global_rect_text(self.loading_label),
            table_rect=_global_rect_text(self.run_table),
            scroll_rect=_global_rect_text(self.scroll_area),
            viewer_placeholder_rect=_global_rect_text(self.viewer_placeholder),
            viewer_placeholder_visible=(
                self.viewer_placeholder.isVisible() if self.viewer_placeholder is not None else False
            ),
            viewer_backend=getattr(self.viewer_3d, "backend_mode", "none"),
            viewer_container_present=bool(getattr(self.viewer_3d, "_container", None)),
        )

    @staticmethod
    def _is_gui_thread() -> bool:
        app = QCoreApplication.instance()
        if app is None:
            return False
        return bool(QThread.currentThread() == app.thread())


class RegistrationValidationPage(_ValidationSelectionPage):
    page_hint = (
        "Analyze repeated saved registration solves as one thesis-facing validation bundle. "
        "Select multiple prior registration records, quantify FRE and transform spread, and save citeable figures without touching the live Registration workflow."
    )
    selection_title = "Saved Registration Runs"
    selection_subtitle = "Choose the accepted registration records to include in the repeated-run stability analysis."
    run_button_text = "Run Registration Validation"
    table_headers = ("Use", "Saved", "FRE (mm)", "Landmarks", "Tool", "Run")

    def _discover_candidates(self):
        return list_registration_validation_candidates(self.controller.project_root)

    def _row_values(self, candidate) -> tuple[str, ...]:
        metadata = dict(candidate.metadata or {})
        return (
            "",
            str(candidate.timestamp_utc).replace("T", " ").replace("+00:00", "Z"),
            _fmt_float(metadata.get("fre_mm")),
            str(int(metadata.get("landmark_count", 0) or 0)),
            str(metadata.get("measurement_tool_id", "unknown") or "unknown"),
            str(candidate.path),
        )

    def _selection_summary_text(self, included) -> str:
        if not included:
            return "Select at least 2 saved registration records."
        fre_values = [
            float(candidate.metadata.get("fre_mm"))
            for candidate in included
            if candidate.metadata.get("fre_mm") is not None
        ]
        if not fre_values:
            return f"{len(included)} run(s) selected."
        return f"{len(included)} selected | mean FRE {_fmt_float(float(np.mean(fre_values)))} mm"


class PivotValidationPage(_ValidationSelectionPage):
    page_hint = (
        "Analyze repeated saved pivot-calibration runs as one thesis-facing validation bundle. "
        "Select multiple prior pivot solves, quantify per-axis offset spread and solve-quality variation, and save citeable figures without reopening the live pivot workflow."
    )
    selection_title = "Saved Pivot Runs"
    selection_subtitle = "Choose the hidden canonical pivot-calibration runs to include in the repeated-run tip-offset analysis."
    run_button_text = "Run Pivot Validation"
    table_headers = ("Use", "Saved", "RMSE (mm)", "Used / Rejected", "Tool", "Run")

    def _discover_candidates(self):
        return list_pivot_validation_candidates(self.controller.project_root)

    def _row_values(self, candidate) -> tuple[str, ...]:
        metadata = dict(candidate.metadata or {})
        return (
            "",
            str(candidate.timestamp_utc).replace("T", " ").replace("+00:00", "Z"),
            _fmt_float(metadata.get("rmse_mm")),
            f"{int(metadata.get('sample_count_used', 0) or 0)} / {int(metadata.get('sample_count_rejected', 0) or 0)}",
            str(metadata.get("tool_id", "0B") or "0B"),
            str(candidate.path),
        )

    def _selection_summary_text(self, included) -> str:
        if not included:
            return "Select at least 2 saved pivot-calibration runs."
        rmse_values = [
            float(candidate.metadata.get("rmse_mm"))
            for candidate in included
            if candidate.metadata.get("rmse_mm") is not None
        ]
        if not rmse_values:
            return f"{len(included)} run(s) selected."
        return f"{len(included)} selected | mean RMSE {_fmt_float(float(np.mean(rmse_values)))} mm"


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


class CollapsibleSection(QWidget):
    """Small inline collapsible section for secondary controls."""

    def __init__(self, title: str, parent=None) -> None:
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)
        self.toggle = QToolButton()
        self.toggle.setText(title)
        self.toggle.setCheckable(True)
        self.toggle.setChecked(False)
        self.toggle.setToolButtonStyle(Qt.ToolButtonTextBesideIcon)
        self.toggle.setArrowType(Qt.RightArrow)
        self.toggle.setStyleSheet(f"color: {COLORS.text_secondary}; font-weight: 600;")
        self.toggle.toggled.connect(self._set_open)
        layout.addWidget(self.toggle, 0, Qt.AlignLeft)
        self.body = QWidget()
        self.body.setVisible(False)
        self.body_layout = QVBoxLayout(self.body)
        self.body_layout.setContentsMargins(0, 0, 0, 0)
        self.body_layout.setSpacing(8)
        layout.addWidget(self.body)

    def _set_open(self, open_: bool) -> None:
        self.toggle.setArrowType(Qt.DownArrow if open_ else Qt.RightArrow)
        self.body.setVisible(bool(open_))


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
            row.setStyleSheet(
                f"background: {COLORS.surface_bg}; border: 1px solid {COLORS.surface_border}; border-radius: 10px;"
            )
            row_layout = QHBoxLayout(row)
            row_layout.setContentsMargins(10, 8, 10, 8)
            row_layout.setSpacing(12)
            key = QLabel(label)
            key.setProperty("role", "muted")
            val = QLabel(value)
            val.setWordWrap(True)
            val.setStyleSheet(f"color: {COLORS.text_primary}; font-weight: 600;")
            row_layout.addWidget(key, 1)
            row_layout.addWidget(val, 2)
            self.layout_.addWidget(row)
        self.layout_.addStretch(1)


class HistoryItemWidget(QWidget):
    """Shared recent-run history row widget."""

    def __init__(self, *, experiment_name: str, timestamp_utc: str, status: str, metric_summary: str, run_path: str, parent=None) -> None:
        super().__init__(parent)
        title = QLabel(experiment_name.replace("_", " ").title())
        title.setStyleSheet(f"font-weight: 700; color: {COLORS.text_primary};")
        chip = QLabel(_status_label(status))
        chip.setStyleSheet(chip_stylesheet(background=_status_bg(status), foreground=_status_fg(status)))
        stamp = QLabel(timestamp_utc.replace("T", " ").replace("+00:00", "Z"))
        stamp.setStyleSheet(f"color: {COLORS.text_muted};")
        metric = QLabel(metric_summary or "No metric summary")
        metric.setWordWrap(True)
        metric.setStyleSheet(f"color: {COLORS.text_secondary};")
        run_name = QLabel(run_path.split("/")[-1])
        run_name.setStyleSheet(f"color: {COLORS.text_subtle};")
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
        self.setStyleSheet(
            f"background: {COLORS.surface_bg}; border: 1px solid {COLORS.surface_border}; border-radius: 12px;"
        )


def build_experiment_page(controller, experiment_name: str) -> ExperimentPageBase:
    """Return the custom page widget for one supported experiment."""
    factories: dict[str, Callable[[object], ExperimentPageBase]] = {
        "single_segment_repeatability": lambda ctrl: SingleSegmentRepeatabilityPage(ctrl, "single_segment_repeatability"),
        "registration_validation": lambda ctrl: RegistrationValidationPage(ctrl, "registration_validation"),
        "pivot_validation": lambda ctrl: PivotValidationPage(ctrl, "pivot_validation"),
        "aurora_grid_accuracy": lambda ctrl: AuroraGridAccuracyPage(ctrl, "aurora_grid_accuracy"),
        "tracker_timing_validation": lambda ctrl: TrackerTimingValidationPage(ctrl, "tracker_timing_validation"),
        "servo_tracker_sync_validation": lambda ctrl: ServoTrackerSyncValidationPage(ctrl, "servo_tracker_sync_validation"),
        "two_segment_startup_validation": lambda ctrl: TwoSegmentStartupValidationPage(ctrl, "two_segment_startup_validation"),
        "two_segment_collect_pose_command_dataset": lambda ctrl: TwoSegmentCollectPoseDatasetPage(ctrl, "two_segment_collect_pose_command_dataset"),
        "pretension_validation": lambda ctrl: PretensionValidationPage(ctrl, "pretension_validation"),
        "penprobe_chasing_demo": lambda ctrl: PenprobeChasingDemoPage(ctrl, "penprobe_chasing_demo"),
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


def _stage_done(stage_completion: dict[str, bool], stage: str) -> str:
    return "captured" if bool(stage_completion.get(stage)) else "pending"


def _segment_display(value) -> str:
    data = dict(value or {})
    label = str(data.get("segment_label") or data.get("label") or "")
    role = str(data.get("segment_role") or "")
    ids = data.get("servo_ids") or []
    pairs = data.get("pairs") or data.get("pair_mapping") or {}
    pieces = [part for part in [label, role, str(ids), str(pairs)] if part]
    return " | ".join(pieces) if pieces else "n/a"


def _format_xy_pair(value) -> str:
    if not isinstance(value, (list, tuple)) or len(value) < 2:
        return "n/a"
    try:
        return f"{float(value[0]):.2f}, {float(value[1]):.2f} mm"
    except Exception:
        return "n/a"


def _event_type_name(event_type) -> str:
    names = {
        QEvent.Type.MouseButtonPress: "mouse_press",
        QEvent.Type.Wheel: "wheel",
        QEvent.Type.FocusIn: "focus_in",
        QEvent.Type.FocusOut: "focus_out",
    }
    return names.get(event_type, str(event_type))


def _widget_debug_name(widget) -> str:
    if widget is None:
        return "none"
    object_name = str(getattr(widget, "objectName", lambda: "")() or "")
    cls_name = widget.__class__.__name__
    return f"{cls_name}#{object_name}" if object_name else cls_name


def _global_rect_text(widget: QWidget | None) -> str:
    if widget is None:
        return "none"
    rect = widget.rect()
    top_left = widget.mapToGlobal(rect.topLeft())
    return f"{top_left.x()},{top_left.y()},{rect.width()}x{rect.height()}"


def _widget_at_center(widget: QWidget | None) -> str:
    if widget is None or not widget.isVisible():
        return "none"
    center = widget.rect().center()
    top = QApplication.widgetAt(widget.mapToGlobal(center))
    return _widget_debug_name(top)


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


def _render_sample_triplet(metrics: dict[str, object]) -> str:
    raw_count = int(metrics.get("raw_sample_count", 0) or 0)
    accepted_count = int(metrics.get("accepted_sample_count", 0) or 0)
    rejected_count = int(metrics.get("rejected_sample_count", metrics.get("outlier_count", 0)) or 0)
    return f"{raw_count} / {accepted_count} / {rejected_count}"


def _fmt_float(value: object) -> str:
    if value is None:
        return "n/a"
    return f"{float(value):.3f}"


def _status_label(status: str) -> str:
    mapping = {
        "success": "Pass",
        "partial_success": "Partial",
        "invalid_due_to_missing_registration": "Needs Registration",
        "invalid_due_to_missing_tip_cal": "Needs Tip Cal",
        "invalid_due_to_insufficient_samples": "Too Few Samples",
        "invalid_due_to_repeatability_coverage": "Invalid Coverage",
        "invalid_due_to_invalid_transforms": "Invalid",
    }
    return mapping.get(status, status.replace("_", " ").title())


def _fmt_metric(value) -> str:
    if value is None:
        return "n/a"
    return f"{float(value):.3f}"


def _status_bg(status: str) -> str:
    if status == "success":
        return semantic_chip_colors("ready")[0]
    if status == "partial_success":
        return semantic_chip_colors("warning")[0]
    return semantic_chip_colors("blocked")[0]


def _status_fg(status: str) -> str:
    if status == "success":
        return semantic_chip_colors("ready")[1]
    if status == "partial_success":
        return semantic_chip_colors("warning")[1]
    return semantic_chip_colors("blocked")[1]

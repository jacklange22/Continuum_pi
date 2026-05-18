"""Dialog for the Registration tab "Trial Mode" button.

Single-button entry: operator opens the dialog, picks landmarks and capture
count, then walks through them one by one. Each landmark fills its quota of
captures from the live tracker (via ``sample_measurement_point_capture``)
without touching the production registration session. When the operator
finishes capturing, the dialog hands the data to ``RegistrationTrialController``
which runs the ``registration_trial`` experiment and reports back the run
folder and ``trial_report.md`` path.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

from PySide6.QtCore import Qt, QTimer
from PySide6.QtWidgets import (
    QCheckBox,
    QDialog,
    QDialogButtonBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QSpinBox,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)

from continuum_robot.gui.widgets.registration_landmark_map_widget import (
    RegistrationLandmarkMapWidget,
)


class RegistrationTrialDialog(QDialog):
    """Three-phase dialog: setup → capture → result."""

    AUTO_CAPTURE_INTERVAL_MS = 60
    """Interval between automatic captures; just above the tracker poll rate."""

    def __init__(
        self,
        trial_controller,
        *,
        candidate_labels: Iterable[str],
        candidate_points_by_label: dict[str, list[float]] | None = None,
        candidate_display_labels: dict[str, str] | None = None,
        parent=None,
    ) -> None:
        super().__init__(parent)
        self.trial_controller = trial_controller
        self.candidate_labels = list(candidate_labels)
        # XYZ + display labels are optional so existing call sites (and tests)
        # that pass only labels keep working; the capture-phase map is only
        # drawn when both are provided.
        self._candidate_points_by_label: dict[str, list[float]] = {
            str(label): [float(value) for value in coords]
            for label, coords in (candidate_points_by_label or {}).items()
            if isinstance(coords, (list, tuple)) and len(coords) >= 2
        }
        self._candidate_display_labels: dict[str, str] = {
            str(label): str(text)
            for label, text in (candidate_display_labels or {}).items()
        }
        # Fixed numeric labels (1..N) follow the candidate-list order so the
        # operator can always identify "this is point 7" regardless of which
        # landmarks they selected for this trial.
        self._fixed_numeric_labels: dict[str, int] = {
            label: index + 1 for index, label in enumerate(self.candidate_labels)
        }
        self.setWindowTitle("Registration Trial Mode")
        self.resize(720, 620)

        self._auto_timer = QTimer(self)
        self._auto_timer.setInterval(self.AUTO_CAPTURE_INTERVAL_MS)
        self._auto_timer.timeout.connect(self._on_auto_tick)

        self._stack = QStackedWidget(self)
        self._setup_widget = self._build_setup_widget()
        self._capture_widget = self._build_capture_widget()
        self._result_widget = self._build_result_widget()
        self._stack.addWidget(self._setup_widget)
        self._stack.addWidget(self._capture_widget)
        self._stack.addWidget(self._result_widget)

        layout = QVBoxLayout(self)
        layout.addWidget(self._stack)

        self._show_phase("setup")

    # --- phase builders -------------------------------------------------

    def _build_setup_widget(self) -> QWidget:
        widget = QWidget(self)
        layout = QVBoxLayout(widget)
        intro = QLabel(
            "Trial Mode captures many samples across many landmarks and runs the "
            "registration_trial experiment to compare averaging methods and find "
            "the best landmark subset. Captures do not touch the production "
            "registration session. Position the pen probe on each landmark when "
            "prompted, then click Capture."
        )
        intro.setWordWrap(True)
        layout.addWidget(intro)

        self.landmark_list = QListWidget(widget)
        self.landmark_list.setSelectionMode(QListWidget.NoSelection)
        for label in self.candidate_labels:
            item = QListWidgetItem(label)
            item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
            item.setCheckState(Qt.Checked)
            self.landmark_list.addItem(item)
        landmark_box = QGroupBox("Landmarks to capture", widget)
        landmark_layout = QVBoxLayout(landmark_box)
        landmark_layout.addWidget(self.landmark_list)
        bulk_row = QHBoxLayout()
        select_all = QPushButton("Select All")
        select_all.clicked.connect(self._select_all_landmarks)
        select_none = QPushButton("Select None")
        select_none.clicked.connect(self._select_no_landmarks)
        bulk_row.addWidget(select_all)
        bulk_row.addWidget(select_none)
        bulk_row.addStretch(1)
        landmark_layout.addLayout(bulk_row)
        layout.addWidget(landmark_box)

        form = QFormLayout()
        self.captures_spin = QSpinBox(widget)
        self.captures_spin.setRange(5, 500)
        self.captures_spin.setValue(50)
        self.captures_spin.setSingleStep(5)
        form.addRow("Captures per landmark", self.captures_spin)
        layout.addLayout(form)

        buttons = QDialogButtonBox()
        self._start_button = QPushButton("Start Trial")
        self._start_button.clicked.connect(self._on_start_clicked)
        cancel = QPushButton("Cancel")
        cancel.clicked.connect(self.reject)
        buttons.addButton(self._start_button, QDialogButtonBox.AcceptRole)
        buttons.addButton(cancel, QDialogButtonBox.RejectRole)
        layout.addWidget(buttons)
        return widget

    def _build_capture_widget(self) -> QWidget:
        widget = QWidget(self)
        layout = QVBoxLayout(widget)

        self.current_label_label = QLabel("(none)", widget)
        self.current_label_label.setStyleSheet("font-size: 28px; font-weight: 700;")
        self.progress_label = QLabel("0 / 0", widget)
        self.status_label = QLabel("Position probe and capture.", widget)
        self.status_label.setWordWrap(True)
        self.overall_progress = QProgressBar(widget)
        self.overall_progress.setRange(0, 1)
        self.overall_progress.setValue(0)

        layout.addWidget(QLabel("Current landmark:"))
        layout.addWidget(self.current_label_label)
        layout.addWidget(self.progress_label)
        layout.addWidget(self.overall_progress)
        layout.addWidget(self.status_label)

        # The capture-phase map shows all candidate landmarks with their fixed
        # 1..N labels and highlights the active one. Only drawn when the caller
        # provided XYZ coordinates -- the dialog still works without them
        # (used by old call sites and headless tests).
        self.capture_map: RegistrationLandmarkMapWidget | None = None
        if self._candidate_points_by_label:
            map_box = QGroupBox("Landmark map", widget)
            map_layout = QVBoxLayout(map_box)
            self.capture_map = RegistrationLandmarkMapWidget(widget)
            self.capture_map.setMinimumHeight(280)
            map_legend = QLabel(
                "Active landmark is highlighted. Completed landmarks are dimmed; "
                "candidates not in this trial appear faded for spatial reference.",
                widget,
            )
            map_legend.setWordWrap(True)
            map_legend.setProperty("role", "hint")
            map_layout.addWidget(self.capture_map)
            map_layout.addWidget(map_legend)
            layout.addWidget(map_box, stretch=1)

        button_row = QHBoxLayout()
        self.capture_one_button = QPushButton("Capture One")
        self.capture_one_button.clicked.connect(self._on_capture_one_clicked)
        self.capture_batch_button = QPushButton("Capture Batch")
        self.capture_batch_button.clicked.connect(self._on_capture_batch_clicked)
        self.skip_button = QPushButton("Skip Landmark")
        self.skip_button.clicked.connect(self._on_skip_clicked)
        self.next_button = QPushButton("Next Landmark")
        self.next_button.clicked.connect(self._on_next_clicked)
        button_row.addWidget(self.capture_one_button)
        button_row.addWidget(self.capture_batch_button)
        button_row.addWidget(self.skip_button)
        button_row.addWidget(self.next_button)
        layout.addLayout(button_row)

        finish_row = QHBoxLayout()
        finish_row.addStretch(1)
        self.stop_button = QPushButton("Stop & Analyze")
        self.stop_button.clicked.connect(self._on_stop_clicked)
        cancel = QPushButton("Cancel Trial")
        cancel.clicked.connect(self.reject)
        finish_row.addWidget(self.stop_button)
        finish_row.addWidget(cancel)
        layout.addLayout(finish_row)
        return widget

    def _build_result_widget(self) -> QWidget:
        widget = QWidget(self)
        layout = QVBoxLayout(widget)
        header = QLabel("Trial complete")
        header.setStyleSheet("font-size: 22px; font-weight: 700;")
        layout.addWidget(header)
        self.result_path_label = QLabel("(unknown)")
        self.result_path_label.setWordWrap(True)
        layout.addWidget(self.result_path_label)
        self.result_summary_label = QLabel("")
        self.result_summary_label.setWordWrap(True)
        layout.addWidget(self.result_summary_label)
        close_row = QHBoxLayout()
        close_row.addStretch(1)
        close = QPushButton("Close")
        close.clicked.connect(self.accept)
        close_row.addWidget(close)
        layout.addLayout(close_row)
        return widget

    # --- phase transitions ---------------------------------------------

    def _show_phase(self, phase: str) -> None:
        order = {"setup": 0, "capture": 1, "result": 2}
        if phase not in order:
            raise ValueError(f"Unknown phase: {phase}")
        self._stack.setCurrentIndex(order[phase])

    def _select_all_landmarks(self) -> None:
        for index in range(self.landmark_list.count()):
            self.landmark_list.item(index).setCheckState(Qt.Checked)

    def _select_no_landmarks(self) -> None:
        for index in range(self.landmark_list.count()):
            self.landmark_list.item(index).setCheckState(Qt.Unchecked)

    def _selected_landmark_labels(self) -> list[str]:
        labels: list[str] = []
        for index in range(self.landmark_list.count()):
            item = self.landmark_list.item(index)
            if item.checkState() == Qt.Checked:
                labels.append(item.text())
        return labels

    # --- setup phase ---------------------------------------------------

    def _on_start_clicked(self) -> None:
        labels = self._selected_landmark_labels()
        if len(labels) < 3:
            QMessageBox.warning(
                self, "Trial Mode", "Select at least 3 landmarks before starting."
            )
            return
        captures = int(self.captures_spin.value())
        try:
            self.trial_controller.start(labels, captures_per_landmark=captures)
        except Exception as exc:
            QMessageBox.critical(self, "Trial Mode", f"Could not start trial: {exc}")
            return
        self._show_phase("capture")
        self._refresh_capture_view()

    # --- capture phase --------------------------------------------------

    def _refresh_capture_view(self) -> None:
        state = self.trial_controller.state
        self.current_label_label.setText(state.current_label or "Done")
        if state.current_label:
            target = int(state.captures_per_landmark)
            count = int(state.captured_counts_by_label.get(state.current_label, 0))
            self.progress_label.setText(f"{count} / {target}")
        else:
            self.progress_label.setText("All landmarks captured")
        total_target = self.trial_controller.target_total()
        total_captured = self.trial_controller.total_captured()
        self.overall_progress.setRange(0, max(1, total_target))
        self.overall_progress.setValue(total_captured)
        self.status_label.setText(state.status_message)
        # Capture controls only make sense while we still have a current landmark
        has_current = state.current_label is not None and not state.is_complete
        self.capture_one_button.setEnabled(has_current)
        self.capture_batch_button.setEnabled(has_current)
        self.skip_button.setEnabled(has_current)
        # "Next" is meaningful when the operator wants to advance even before the target.
        self.next_button.setEnabled(has_current)
        # Stop is allowed any time captures exist.
        self.stop_button.setEnabled(total_captured > 0)
        self._refresh_capture_map(state)

    def _refresh_capture_map(self, state) -> None:
        """Push the controller's progress into the capture-phase landmark map.

        ``completed_labels`` is computed locally rather than tracked on the
        controller so this view stays decoupled from controller state shape
        changes -- a landmark is "completed" once its captured-count reaches
        the configured per-landmark quota.
        """
        if self.capture_map is None:
            return
        target = int(state.captures_per_landmark)
        captured_counts = dict(state.captured_counts_by_label or {})
        completed = [
            label
            for label, count in captured_counts.items()
            if int(count) >= target > 0
        ]
        self.capture_map.set_trial_capture_state(
            points_by_label=self._candidate_points_by_label,
            display_labels=self._candidate_display_labels,
            fixed_numeric_labels=self._fixed_numeric_labels,
            trial_labels=list(state.landmark_labels),
            active_label=state.current_label,
            completed_labels=completed,
        )

    def _on_capture_one_clicked(self) -> None:
        try:
            self.trial_controller.capture_one()
        except Exception as exc:
            QMessageBox.critical(self, "Trial Mode", f"Capture failed: {exc}")
        self._refresh_capture_view()

    def _on_capture_batch_clicked(self) -> None:
        # If a batch is already running, stop it.
        if self._auto_timer.isActive():
            self._auto_timer.stop()
            self.capture_batch_button.setText("Capture Batch")
            self._refresh_capture_view()
            return
        self.capture_batch_button.setText("Stop Batch")
        self._auto_timer.start()

    def _on_auto_tick(self) -> None:
        if self.trial_controller.remaining_for_current_label() <= 0:
            self._auto_timer.stop()
            self.capture_batch_button.setText("Capture Batch")
            self._refresh_capture_view()
            return
        try:
            self.trial_controller.capture_one()
        except Exception as exc:
            self._auto_timer.stop()
            self.capture_batch_button.setText("Capture Batch")
            QMessageBox.critical(self, "Trial Mode", f"Capture failed: {exc}")
        self._refresh_capture_view()

    def _on_skip_clicked(self) -> None:
        if self._auto_timer.isActive():
            self._auto_timer.stop()
            self.capture_batch_button.setText("Capture Batch")
        self.trial_controller.skip_current_label()
        self._refresh_capture_view()

    def _on_next_clicked(self) -> None:
        if self._auto_timer.isActive():
            self._auto_timer.stop()
            self.capture_batch_button.setText("Capture Batch")
        self.trial_controller.advance_to_next_label()
        self._refresh_capture_view()

    def _on_stop_clicked(self) -> None:
        if self._auto_timer.isActive():
            self._auto_timer.stop()
            self.capture_batch_button.setText("Capture Batch")
        if self.trial_controller.total_captured() <= 0:
            QMessageBox.warning(self, "Trial Mode", "No captures recorded yet.")
            return
        try:
            state = self.trial_controller.run_analysis()
        except Exception as exc:
            QMessageBox.critical(
                self, "Trial Mode", f"Analysis failed: {exc}"
            )
            return
        report = state.last_trial_report_md
        out = state.last_run_output_dir
        self.result_path_label.setText(
            f"Run folder: {out}\nReport: {report or '(missing)'}"
        )
        summary_lines: list[str] = []
        method_summary = state.last_run_summary.get("method_summary", {})
        best_method = method_summary.get("best_method")
        best_fre = method_summary.get("best_fre_mm")
        if best_method is not None and best_fre is not None:
            summary_lines.append(
                f"Best averaging method: {best_method} at {float(best_fre):.4f} mm"
            )
        subset_summary = state.last_run_summary.get("subset_search_summary", {})
        global_best = subset_summary.get("global_best") if isinstance(subset_summary, dict) else None
        if isinstance(global_best, dict):
            summary_lines.append(
                "Best subset: size={size}, labels={labels}, FRE={fre:.4f} mm".format(
                    size=global_best.get("size"),
                    labels=global_best.get("labels"),
                    fre=float(global_best.get("fre_mm") or 0.0),
                )
            )
        for rec in state.last_run_summary.get("trial_recommendations") or []:
            summary_lines.append(f"• {rec}")
        self.result_summary_label.setText("\n\n".join(summary_lines) or "(no summary)")
        self._show_phase("result")

"""Servos tab widget."""

from __future__ import annotations

import time
from datetime import datetime
from typing import Callable

from PySide6.QtCore import Qt, QTimer
from PySide6.QtWidgets import (
    QAbstractItemView,
    QDoubleSpinBox,
    QFormLayout,
    QGridLayout,
    QGroupBox,
    QHeaderView,
    QHBoxLayout,
    QLabel,
    QPlainTextEdit,
    QPushButton,
    QScrollArea,
    QSpinBox,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from continuum_robot.gui.controllers.servos_controller import ServosViewState
from continuum_robot.gui.theme import grouped_workspace_stylesheet
from continuum_robot.gui.view_utils import preserve_scroll_position
from continuum_robot.gui.widgets.xy_joystick_widget import XyJoystickWidget


class ServosTab(QWidget):
    """Live servo telemetry, jog controls, and pretension (manual + algorithmic)."""

    def __init__(
        self,
        controller,
        parent=None,
        *,
        apply_runtime_parameters: Callable[..., None] | None = None,
        pretension_trial_controller=None,
    ) -> None:
        super().__init__(parent)
        self.controller = controller
        # apply_runtime_parameters retained for API compatibility; jog-tick settings
        # now live on the System tab, so this callback is unused here.
        self._apply_runtime_parameters = apply_runtime_parameters
        self.pretension_trial_controller = pretension_trial_controller

        self.setObjectName("servoWorkspace")
        self.setStyleSheet(
            grouped_workspace_stylesheet(
                object_name="servoWorkspace",
                input_selectors=["QSpinBox", "QPlainTextEdit", "QTableWidget"],
            )
        )

        self.title_label = QLabel("Servos")
        self.title_label.setProperty("role", "title")

        self.status_label = QLabel("Disconnected")
        self.status_label.setProperty("role", "status")
        self.status_label.setWordWrap(True)
        self.blocker_label = QLabel("")
        self.blocker_label.setProperty("role", "hint")
        self.blocker_label.setWordWrap(True)
        self.blocker_label.setVisible(False)

        status_box = QGroupBox("Ready State")
        status_layout = QVBoxLayout(status_box)
        status_layout.addWidget(self.status_label)
        status_layout.addWidget(self.blocker_label)

        self.telemetry_table = QTableWidget(0, 9)
        self.telemetry_table.setHorizontalHeaderLabels(
            [
                "ID",
                "State",
                "Torque",
                "Position (tick)",
                "Current (mA)",
                "Voltage (mV)",
                "Temp (C)",
                "HW Err",
                "Age (s)",
            ]
        )
        self.telemetry_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.telemetry_table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.telemetry_table.setSelectionMode(QAbstractItemView.SingleSelection)
        self.telemetry_table.verticalHeader().setVisible(False)
        self.telemetry_table.cellClicked.connect(self._select_servo_from_row)
        for column in range(0, 9):
            self.telemetry_table.horizontalHeader().setSectionResizeMode(column, QHeaderView.ResizeToContents)
        self.telemetry_table.horizontalHeader().setStretchLastSection(True)
        self.telemetry_table.setMinimumHeight(190)

        telemetry_box = QGroupBox("Live Telemetry")
        telemetry_layout = QVBoxLayout(telemetry_box)
        telemetry_layout.addWidget(self.telemetry_table)

        self.jog_label = QLabel("Select a servo to jog")
        self.jog_label.setProperty("role", "hint")
        self.fine_minus_button = QPushButton("Loosen Fine")
        self.fine_plus_button = QPushButton("Tighten Fine")
        self.coarse_minus_button = QPushButton("Loosen Coarse")
        self.coarse_plus_button = QPushButton("Tighten Coarse")
        self.fine_minus_button.clicked.connect(lambda: self._jog("fine", -1))
        self.fine_plus_button.clicked.connect(lambda: self._jog("fine", 1))
        self.coarse_minus_button.clicked.connect(lambda: self._jog("coarse", -1))
        self.coarse_plus_button.clicked.connect(lambda: self._jog("coarse", 1))

        jog_box = QGroupBox("Jog Selected Servo")
        jog_layout = QVBoxLayout(jog_box)
        jog_layout.addWidget(self.jog_label)
        jog_buttons_grid = QGridLayout()
        jog_buttons_grid.setHorizontalSpacing(10)
        jog_buttons_grid.setVerticalSpacing(10)
        jog_buttons_grid.addWidget(self.fine_minus_button, 0, 0)
        jog_buttons_grid.addWidget(self.fine_plus_button, 0, 1)
        jog_buttons_grid.addWidget(self.coarse_minus_button, 1, 0)
        jog_buttons_grid.addWidget(self.coarse_plus_button, 1, 1)
        jog_layout.addLayout(jog_buttons_grid)
        jog_layout.addStretch(1)

        # --- Whole-segment XY drive ----------------------------------------
        # Round drag-pad that maps a bounded XY vector to antagonistic tendon
        # displacements on the active 4-tendon segment (matches the convention
        # used by the penprobe chasing demo: axis_a = [s0, s2], axis_b = [s1, s3]).
        self.joystick_widget = XyJoystickWidget(radius_cm=1.0)
        self.joystick_radius_spin = QDoubleSpinBox()
        self.joystick_radius_spin.setRange(0.1, 3.0)
        self.joystick_radius_spin.setSingleStep(0.1)
        self.joystick_radius_spin.setDecimals(2)
        self.joystick_radius_spin.setSuffix(" cm")
        self.joystick_radius_spin.setValue(1.0)
        self.joystick_radius_spin.valueChanged.connect(self._on_joystick_radius_changed)
        self.joystick_readout_label = QLabel("X = +0.000 cm   Y = +0.000 cm")
        self.joystick_readout_label.setProperty("role", "hint")
        self.joystick_readout_label.setWordWrap(True)
        self.joystick_per_servo_label = QLabel("Tendon deltas: —")
        self.joystick_per_servo_label.setProperty("role", "hint")
        self.joystick_per_servo_label.setWordWrap(True)
        self.joystick_last_action_label = QLabel("No XY drive command sent yet.")
        self.joystick_last_action_label.setProperty("role", "hint")
        self.joystick_last_action_label.setWordWrap(True)
        self.joystick_center_button = QPushButton("Return to Center")
        self.joystick_center_button.clicked.connect(self._center_joystick)
        self.joystick_blocker_label = QLabel("")
        self.joystick_blocker_label.setProperty("role", "hint")
        self.joystick_blocker_label.setWordWrap(True)
        self.joystick_blocker_label.setVisible(False)

        joystick_box = QGroupBox("Whole-Segment XY Drive")
        joystick_layout = QVBoxLayout(joystick_box)
        joystick_layout.setSpacing(8)
        joystick_layout.addWidget(self.joystick_widget, 0, Qt.AlignHCenter)
        joystick_radius_row = QHBoxLayout()
        joystick_radius_row.setSpacing(10)
        joystick_radius_row.addWidget(QLabel("Bounded radius"))
        joystick_radius_row.addWidget(self.joystick_radius_spin)
        joystick_radius_row.addStretch(1)
        joystick_layout.addLayout(joystick_radius_row)
        joystick_layout.addWidget(self.joystick_readout_label)
        joystick_layout.addWidget(self.joystick_per_servo_label)
        joystick_layout.addWidget(self.joystick_last_action_label)
        joystick_layout.addWidget(self.joystick_blocker_label)
        joystick_layout.addWidget(self.joystick_center_button, 0, Qt.AlignLeft)
        joystick_layout.addStretch(1)

        # Side-by-side: jog buttons left, XY drive right.
        jog_row_widget = QWidget()
        jog_row_layout = QHBoxLayout(jog_row_widget)
        jog_row_layout.setContentsMargins(0, 0, 0, 0)
        jog_row_layout.setSpacing(14)
        jog_row_layout.addWidget(jog_box, 1)
        jog_row_layout.addWidget(joystick_box, 1)

        # Tight-loop dispatch state. We bypass controller.apply_displacement
        # (which calls a heavy refresh() in finally) and talk to servo_service
        # directly with chase_tight_loop_writes + skip_post_command_telemetry +
        # prevalidated telemetry — the same pattern the penprobe chasing demo
        # uses for live control. Cache neutral_ticks and minimal telemetry so
        # most sends are a single bus write.
        self._joystick_pending_xy_cm: tuple[float, float] | None = None
        self._joystick_last_sent_xy_cm: tuple[float, float] | None = None
        self._joystick_send_in_flight = False
        self._joystick_neutral_ticks_cache: list[int] | None = None
        self._joystick_telemetry_cache: dict = {}
        self._joystick_cache_at = 0.0
        self._joystick_cache_key: tuple | None = None
        self._joystick_cache_ttl_s = 3.0
        # Backstop timer in case position_changed somehow doesn't fire the
        # post-send chain (lost wakeup, exception path); fires only as a
        # safety net at low frequency.
        self._joystick_send_timer = QTimer(self)
        self._joystick_send_timer.setInterval(150)
        self._joystick_send_timer.timeout.connect(self._maybe_send_joystick_command)
        self._joystick_send_timer.start()
        self.joystick_widget.position_changed.connect(self._on_joystick_position_changed)
        # On mouse release, flush the final position immediately — a fast drag
        # that ends right after a send completes shouldn't have to wait for the
        # backstop timer to push the final pixel.
        self.joystick_widget.drag_released.connect(self._on_joystick_drag_released)

        self.manual_pretension_summary_label = QLabel("No accepted pretension source.")
        self.manual_pretension_summary_label.setWordWrap(True)
        self.manual_pretension_note_label = QLabel("No manual pretension note saved.")
        self.manual_pretension_note_label.setWordWrap(True)
        self.manual_pretension_note_edit = QPlainTextEdit()
        self.manual_pretension_note_edit.setPlaceholderText("Optional operator note for this manual startup state.")
        self.manual_pretension_note_edit.setMaximumHeight(64)
        self.capture_manual_pretension_button = QPushButton("Capture Current State")
        self.capture_manual_pretension_button.setProperty("role", "primary")
        self.capture_manual_pretension_button.clicked.connect(self._capture_manual_pretension)
        self.accept_manual_pretension_button = QPushButton("Accept")
        self.accept_manual_pretension_button.clicked.connect(
            lambda: self._safe_call(self.controller.accept_manual_pretension)
        )
        self.clear_manual_pretension_button = QPushButton("Clear")
        self.clear_manual_pretension_button.setProperty("variant", "ghost")
        self.clear_manual_pretension_button.clicked.connect(
            lambda: self._safe_call(self.controller.clear_manual_pretension)
        )

        self.manual_pretension_box = QGroupBox("Manual Pretension")
        manual_layout = QFormLayout(self.manual_pretension_box)
        manual_layout.addRow("Active source", self.manual_pretension_summary_label)
        manual_layout.addRow("Saved note", self.manual_pretension_note_label)
        manual_layout.addRow("Operator note", self.manual_pretension_note_edit)
        manual_buttons = QHBoxLayout()
        manual_buttons.setSpacing(10)
        manual_buttons.addWidget(self.capture_manual_pretension_button)
        manual_buttons.addWidget(self.accept_manual_pretension_button)
        manual_buttons.addWidget(self.clear_manual_pretension_button)
        manual_buttons.addStretch(1)
        manual_buttons_widget = QWidget()
        manual_buttons_widget.setLayout(manual_buttons)
        manual_layout.addRow(manual_buttons_widget)

        # --- Segment Pretension Trial (4-servo, one-click) ---------------
        # Drives the pretension_validation experiment with the saved config so
        # the operator can pretension the whole active segment from one place.
        # Tuning still lives on the Experiments tab pretension page.
        self.segment_pretension_status_label = QLabel(
            "Pretension trial idle. Tune knobs on the Experiments tab pretension page."
        )
        self.segment_pretension_status_label.setProperty("role", "status")
        self.segment_pretension_status_label.setWordWrap(True)
        self.segment_pretension_manual_count_label = QLabel("Manual baselines recorded: 0")
        self.segment_pretension_manual_count_label.setProperty("role", "hint")
        self.record_manual_baseline_button = QPushButton("Record Manual Baseline")
        self.record_manual_baseline_button.setToolTip(
            "Hand-tension the spine first, then click this to snapshot the current state. "
            "Repeat 5 times to build a manual repeatability baseline that the trial compares against."
        )
        self.clear_manual_baselines_button = QPushButton("Clear Baselines")
        self.clear_manual_baselines_button.setProperty("variant", "ghost")
        self.run_segment_pretension_button = QPushButton("Run Pretension Trial")
        self.run_segment_pretension_button.setProperty("role", "primary")
        self.run_segment_pretension_button.setToolTip(
            "Runs the saved pretension_validation experiment on the active segment. "
            "Uses recorded manual baselines (if any) for the algorithm-vs-manual comparison report."
        )
        self.record_manual_baseline_button.clicked.connect(self._record_manual_baseline)
        self.clear_manual_baselines_button.clicked.connect(self._clear_manual_baselines)
        self.run_segment_pretension_button.clicked.connect(self._run_segment_pretension)

        self.segment_pretension_box = QGroupBox("Segment Pretension Trial (4-servo)")
        segment_pretension_layout = QVBoxLayout(self.segment_pretension_box)
        segment_pretension_layout.addWidget(self.segment_pretension_status_label)
        segment_pretension_layout.addWidget(self.segment_pretension_manual_count_label)
        segment_pretension_button_row = QHBoxLayout()
        segment_pretension_button_row.setSpacing(10)
        segment_pretension_button_row.addWidget(self.record_manual_baseline_button)
        segment_pretension_button_row.addWidget(self.clear_manual_baselines_button)
        segment_pretension_button_row.addWidget(self.run_segment_pretension_button)
        segment_pretension_button_row.addStretch(1)
        segment_pretension_button_widget = QWidget()
        segment_pretension_button_widget.setLayout(segment_pretension_button_row)
        segment_pretension_layout.addWidget(segment_pretension_button_widget)
        # The whole section is hidden unless an experiment runner was wired in.
        self.segment_pretension_box.setVisible(self.pretension_trial_controller is not None)
        self._refresh_segment_pretension_state()

        content_widget = QWidget()
        content_layout = QVBoxLayout(content_widget)
        content_layout.setContentsMargins(0, 0, 0, 0)
        content_layout.setSpacing(14)
        content_layout.addWidget(status_box)
        content_layout.addWidget(telemetry_box)
        content_layout.addWidget(jog_row_widget)
        content_layout.addWidget(self.manual_pretension_box)
        content_layout.addWidget(self.segment_pretension_box)
        content_layout.addStretch(1)

        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setWidget(content_widget)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(18, 18, 18, 18)
        layout.setSpacing(14)
        layout.addWidget(self.title_label)
        layout.addWidget(self.scroll_area, 1)

    def update(self, state: ServosViewState) -> None:
        selected_servo_id = (
            state.selected_servo_id
            if state.selected_servo_id is not None
            else (state.servo_ids[0] if state.servo_ids else None)
        )

        self.status_label.setText(self._format_status(state))
        blocker_text = self._format_blocker(state)
        self.blocker_label.setText(blocker_text)
        self.blocker_label.setVisible(bool(blocker_text))

        sorted_servo_ids = sorted(state.telemetry)

        def _rebuild_telemetry_table() -> None:
            self.telemetry_table.setRowCount(len(state.telemetry))
            for row, servo_id in enumerate(sorted_servo_ids):
                item = state.telemetry[servo_id]
                self.telemetry_table.setItem(row, 0, self._text_item(servo_id, align=Qt.AlignRight))
                self.telemetry_table.setItem(
                    row, 1, self._text_item(item.get("telemetry_status", "Unknown"), align=Qt.AlignCenter)
                )
                self.telemetry_table.setItem(row, 2, self._text_item(item.get("torque_label", "—"), align=Qt.AlignCenter))
                self.telemetry_table.setItem(row, 3, self._text_item(self._display_value(item.get("position")), align=Qt.AlignRight))
                self.telemetry_table.setItem(row, 4, self._text_item(self._display_value(item.get("current_ma")), align=Qt.AlignRight))
                self.telemetry_table.setItem(row, 5, self._text_item(self._display_value(item.get("voltage_mv")), align=Qt.AlignRight))
                self.telemetry_table.setItem(row, 6, self._text_item(self._display_value(item.get("temperature_c")), align=Qt.AlignRight))
                self.telemetry_table.setItem(row, 7, self._text_item(item.get("hardware_error_text", "—"), align=Qt.AlignCenter))
                self.telemetry_table.setItem(row, 8, self._text_item(self._age_text(item.get("telemetry_age_s")), align=Qt.AlignRight))
            if selected_servo_id in sorted_servo_ids:
                self.telemetry_table.selectRow(sorted_servo_ids.index(selected_servo_id))
            else:
                self.telemetry_table.clearSelection()

        preserve_scroll_position(self.telemetry_table, _rebuild_telemetry_table)

<<<<<<< Updated upstream
        motion_ready = (
            state.connected
            and bool(state.servo_ids)
            and state.selected_servo_motion_ready
            and selected_servo_id is not None
        )
        self.jog_label.setText(
            f"Jog: Servo {selected_servo_id}" if selected_servo_id is not None else "Select a servo to jog"
        )
        self.fine_minus_button.setEnabled(motion_ready)
        self.fine_plus_button.setEnabled(motion_ready)
        self.coarse_minus_button.setEnabled(motion_ready)
        self.coarse_plus_button.setEnabled(motion_ready)

        self.manual_pretension_summary_label.setText(state.pretension_source_summary)
        self.manual_pretension_note_label.setText(
            state.pretension_source_note or "No manual pretension note saved."
        )
        manual_mode_available = (not state.single_servo_mode) and len(state.expected_servo_ids) in {4, 8}
        any_servo = bool(state.servo_ids)
        self.manual_pretension_box.setVisible(manual_mode_available)
        self.capture_manual_pretension_button.setEnabled(
            state.connected and manual_mode_available and any_servo
        )
        self.accept_manual_pretension_button.setEnabled(
            manual_mode_available and state.manual_pretension_can_accept
        )
        self.clear_manual_pretension_button.setEnabled(
            manual_mode_available and state.manual_pretension_can_clear
        )

        self._sync_joystick_availability(state)

    # --- XY drive (joystick) handlers ------------------------------------

    def _sync_joystick_availability(self, state: ServosViewState) -> None:
        servo_ids = self._joystick_dispatch_servo_ids(state)
        block_reason = ""
        if not state.connected:
            block_reason = "Bus disconnected — connect on the System tab."
        elif len(servo_ids) != 4:
            block_reason = "XY drive needs the active 4-tendon segment commanded on the bus."
        elif (state.pretension_source_type or "none").lower() in {"none", ""}:
            block_reason = "Accept a pretension source first (Manual Pretension or Pretension Trial)."
        elif "pending" in (state.pretension_source_type or "").lower():
            block_reason = "Accept the pending pretension source before driving the segment."
        motion_enabled = not bool(block_reason)
        self.joystick_widget.set_motion_enabled(motion_enabled)
        self.joystick_blocker_label.setText(block_reason)
        self.joystick_blocker_label.setVisible(bool(block_reason))
        self.joystick_radius_spin.setEnabled(motion_enabled)
        x_cm, y_cm = self.joystick_widget.position_cm()
        self.joystick_center_button.setEnabled(motion_enabled and (x_cm != 0.0 or y_cm != 0.0))
        self._refresh_joystick_readout(state)

    def _on_joystick_radius_changed(self, value: float) -> None:
        self.joystick_widget.set_radius_cm(float(value))
        self._refresh_joystick_readout(self.controller.state)

    def _on_joystick_position_changed(self, x_cm: float, y_cm: float) -> None:
        self._joystick_pending_xy_cm = (float(x_cm), float(y_cm))
        self.joystick_center_button.setEnabled(x_cm != 0.0 or y_cm != 0.0)
        self._refresh_joystick_readout(self.controller.state)
        # Fire immediately — don't wait for the backstop timer. The in-flight
        # guard inside _maybe_send_joystick_command coalesces fast bursts.
        self._maybe_send_joystick_command()

    def _on_joystick_drag_released(self, x_cm: float, y_cm: float) -> None:
        self._joystick_pending_xy_cm = (float(x_cm), float(y_cm))
        self._maybe_send_joystick_command()

    def _center_joystick(self) -> None:
        self.joystick_widget.center()
        self._joystick_pending_xy_cm = (0.0, 0.0)
        self._maybe_send_joystick_command()

    def _invalidate_joystick_caches(self) -> None:
        self._joystick_neutral_ticks_cache = None
        self._joystick_telemetry_cache = {}
        self._joystick_cache_at = 0.0
        self._joystick_cache_key = None

    def _refresh_joystick_readout(self, state: ServosViewState) -> None:
        x_cm, y_cm = self.joystick_widget.position_cm()
        self.joystick_readout_label.setText(
            f"X = {x_cm:+.3f} cm   Y = {y_cm:+.3f} cm"
        )
        deltas = self._joystick_servo_deltas_cm(state, x_cm, y_cm)
        if not deltas:
            self.joystick_per_servo_label.setText("Tendon deltas: —")
            return
        delta_text = "   ".join(f"{servo_id}: {value:+.3f} cm" for servo_id, value in deltas)
        self.joystick_per_servo_label.setText(f"Tendon deltas: {delta_text}")

    def _maybe_send_joystick_command(self) -> None:
        pending = self._joystick_pending_xy_cm
        if pending is None or self._joystick_send_in_flight:
            return
        if pending == self._joystick_last_sent_xy_cm:
            return
        state = self.controller.state
        if not self._joystick_motion_allowed(state):
            return
        servo_ids = self._joystick_dispatch_servo_ids(state)
        if len(servo_ids) != 4:
            return
        displacements = self._joystick_displacement_vector_for_dispatch(servo_ids, pending[0], pending[1])
        if displacements is None:
            self.joystick_last_action_label.setText(
                "XY drive: could not build displacement vector — check operating mode."
            )
            return

        self._joystick_send_in_flight = True
        timestamp = datetime.now().strftime("%H:%M:%S")
        try:
            neutral_ticks, prevalidated = self._ensure_joystick_caches(state, servo_ids)
            # Direct fast-path call: chase_tight_loop_writes + prevalidated
            # telemetry + skip post-command telemetry. Same pattern the
            # penprobe chasing demo uses for live control. Bypasses the
            # controller's heavy refresh() in apply_displacement's finally.
            self.controller.servo_service.command_displacement(
                tendon_displacements_cm=displacements,
                neutral_ticks=neutral_ticks,
                servo_ids=servo_ids,
                motion_workflow="experiment_motion",
                chase_tight_loop_writes=True,
                prevalidated_telemetry_by_id=dict(prevalidated),
                skip_post_command_telemetry=True,
            )
            self._joystick_last_sent_xy_cm = pending
            # Keep the controller's state.tendon_displacements_cm in sync for
            # debug consistency, but skip the refresh() it normally chases.
            self.controller.state.tendon_displacements_cm = list(displacements)
            self.joystick_last_action_label.setText(
                f"Sent X={pending[0]:+.3f} cm  Y={pending[1]:+.3f} cm  at {timestamp}"
            )
        except Exception as exc:  # noqa: BLE001 — surface but never crash the GUI
            self._invalidate_joystick_caches()
            self.joystick_last_action_label.setText(f"XY drive error at {timestamp}: {exc}")
        finally:
            self._joystick_send_in_flight = False
            # If new pending arrived during the send, schedule another send
            # ASAP. QTimer.singleShot(0) yields one Qt event-loop tick so the
            # GUI stays responsive between writes.
            if (
                self._joystick_pending_xy_cm is not None
                and self._joystick_pending_xy_cm != self._joystick_last_sent_xy_cm
            ):
                QTimer.singleShot(0, self._maybe_send_joystick_command)

    def _ensure_joystick_caches(
        self,
        state: ServosViewState,
        servo_ids: list[int],
    ) -> tuple[list[int], dict]:
        """Return cached neutral ticks and prevalidated telemetry for fast writes.

        Invalidated when: pretension source changes, active servo set changes,
        or `_joystick_cache_ttl_s` has elapsed since the last refresh. Most
        joystick sends hit the cached path.
        """
        now = time.monotonic()
        cache_key = (
            state.pretension_source_type,
            state.pretension_source_updated_at_utc,
            tuple(servo_ids),
        )
        cache_stale = (now - self._joystick_cache_at) > self._joystick_cache_ttl_s
        if (
            self._joystick_neutral_ticks_cache is None
            or self._joystick_cache_key != cache_key
            or cache_stale
            or not self._joystick_telemetry_cache
        ):
            reference = self.controller.servo_service.resolve_startup_reference_ticks(list(servo_ids))
            missing = [sid for sid in servo_ids if sid not in reference.ticks_by_servo]
            if missing:
                raise RuntimeError(
                    f"Startup reference ticks missing for servo IDs: {missing}. "
                    "Accept pretension or recapture neutral first."
                )
            self._joystick_neutral_ticks_cache = [
                int(reference.ticks_by_servo[sid]) for sid in servo_ids
            ]
            self._joystick_telemetry_cache = self.controller.servo_service.read_minimal_telemetry(
                list(servo_ids)
            )
            self._joystick_cache_at = now
            self._joystick_cache_key = cache_key
        return self._joystick_neutral_ticks_cache, self._joystick_telemetry_cache

    @staticmethod
    def _joystick_motion_allowed(state: ServosViewState) -> bool:
        if not state.connected:
            return False
        pretension_type = (state.pretension_source_type or "none").lower()
        if pretension_type in {"none", ""} or "pending" in pretension_type:
            return False
        return True

    @staticmethod
    def _joystick_dispatch_servo_ids(state: ServosViewState) -> list[int]:
        """Servo IDs in the order command_displacement consumes them.

        Mirrors what ServosController.apply_displacement passes: state.servo_ids
        (the currently commanded set). Joystick displacement vectors must be
        positional against this list, not against settings.robot.tendon_to_servo
        — the two can diverge after discovery or operator overrides.
        """
        return [int(value) for value in (state.servo_ids or [])]

    @staticmethod
    def _joystick_displacement_vector_for_dispatch(
        servo_ids: list[int],
        x_cm: float,
        y_cm: float,
    ) -> list[float] | None:
        """Translate an XY puck position into a tendon displacement vector.

        Uses the antagonistic-pair convention from the penprobe chasing demo:
        axis_a = [servo_ids[0], servo_ids[2]] responds to X, axis_b =
        [servo_ids[1], servo_ids[3]] responds to Y; the lower position in each
        pair loosens, the higher tightens. The output is positional against the
        provided servo_ids list, which command_displacement consumes directly.
        """
        if len(servo_ids) != 4:
            return None
        displacements = [0.0] * 4
        # X axis on pair (0, 2)
        displacements[0] = -float(x_cm)
        displacements[2] = float(x_cm)
        # Y axis on pair (1, 3)
        displacements[1] = -float(y_cm)
        displacements[3] = float(y_cm)
        return displacements

    def _joystick_servo_deltas_cm(
        self,
        state: ServosViewState,
        x_cm: float,
        y_cm: float,
    ) -> list[tuple[int, float]]:
        servo_ids = self._joystick_dispatch_servo_ids(state)
        displacements = self._joystick_displacement_vector_for_dispatch(servo_ids, x_cm, y_cm)
        if displacements is None:
            return []
        return [(servo_ids[i], displacements[i]) for i in range(len(displacements))]

    @staticmethod
    def _format_status(state: ServosViewState) -> str:
        if not state.connected:
            return "Bus disconnected — connect OpenRB on the System tab."
        detected = sorted(int(sid) for sid in state.detected_servo_ids)
        expected = sorted(int(sid) for sid in state.expected_servo_ids)
        detected_text = ", ".join(str(sid) for sid in detected) or "none"
        if expected and detected == expected:
            servo_summary = f"{len(detected)} of {len(expected)} servos: {detected_text}"
        elif expected:
            servo_summary = (
                f"{len(detected)} of {len(expected)} servos detected (expected {', '.join(str(sid) for sid in expected)})"
            )
        else:
            servo_summary = f"Detected: {detected_text}"
        calibration_state = (
            "ready"
            if state.calibration_exists and state.calibration_compatible
            else ("review needed" if state.calibration_exists else "missing")
        )
        pretension_type = (state.pretension_source_type or "none").lower()
        if pretension_type == "none":
            pretension = "none"
        elif "pending" in pretension_type:
            pretension = "pending accept"
        else:
            pretension = "accepted"
        return (
            f"Bus connected  ·  {servo_summary}  ·  Calibration: {calibration_state}  ·  "
            f"Pretension: {pretension}"
        )

    @staticmethod
    def _format_blocker(state: ServosViewState) -> str:
        parts: list[str] = []
        if state.missing_servo_ids:
            parts.append(
                "Missing servos: " + ", ".join(str(sid) for sid in sorted(state.missing_servo_ids))
            )
        if state.unexpected_servo_ids:
            parts.append(
                "Unexpected servos: " + ", ".join(str(sid) for sid in sorted(state.unexpected_servo_ids))
            )
        if state.blocking_reasons:
            parts.append(state.blocking_reasons[0])
=======
        operator_lines = [
            f"Servo: {selected_servo_id if selected_servo_id is not None else 'none'}",
            f"Motion ready: {'Yes' if state.selected_servo_motion_ready else 'No'}",
            f"Torque: {self.selected_servo_torque_label.text()}",
            f"Telemetry: {self.selected_servo_telemetry_label.text()} | age {self.selected_servo_age_label.text()} | fresh {self.selected_servo_fresh_label.text()}",
            f"Position: {self.selected_servo_position_label.text()} | Target: {self.selected_servo_target_label.text()}",
            f"Raw hard bounds: {self.selected_servo_bounds_label.text()}",
            f"Last action: {self.selected_servo_action_label.text()} | Result: {self.selected_servo_result_label.text()}",
        ]
        if not state.single_servo_mode:
            if state.robot_mode in {"dual_segment", "parallel_single"} and state.all_8_readiness_summary:
                operator_lines.append(state.all_8_readiness_summary)
            if state.segment_readiness_summary:
                operator_lines.append(f"Segments: {state.segment_readiness_summary}")
            if state.single_segment_readiness_summary:
                operator_lines.append(state.single_segment_readiness_summary)
            operator_lines.append(f"Active pretension source: {state.pretension_source_summary}")
            if state.single_segment_reference_summary:
                operator_lines.append(f"Experiment reference: {state.single_segment_reference_summary}")
            if state.single_segment_motion_config_summary:
                operator_lines.append(f"Single-segment motion config: {state.single_segment_motion_config_summary}")
            if state.single_segment_enforced_bounds_summary:
                operator_lines.append(f"Enforced experiment bounds: {state.single_segment_enforced_bounds_summary}")
            if state.single_segment_characterization_summary:
                operator_lines.append(f"Diagnostic pair travel: {state.single_segment_characterization_summary}")
            if state.last_displacement_summary:
                operator_lines.append(f"Last displacement: {state.last_displacement_summary}")
            for line in state.last_displacement_debug_lines:
                operator_lines.append(line)
        if state.selected_servo_reason_label and state.selected_servo_reason_label != "none":
            operator_lines.append(f"Reason: {state.selected_servo_reason_label}")
>>>>>>> Stashed changes
        if state.last_error:
            parts.append(f"Error: {state.last_error}")
        return "  ·  ".join(parts)

    def _jog(self, mode: str, direction: int) -> None:
        servo_id = self.controller.state.selected_servo_id
        if servo_id is None:
            return
        action = self.controller.fine_jog if mode == "fine" else self.controller.coarse_jog
        self._safe_call(action, int(servo_id), int(direction))

    def _capture_manual_pretension(self) -> None:
        self._safe_call(
            self.controller.capture_manual_pretension,
            self.manual_pretension_note_edit.toPlainText().strip(),
        )

    def _sync_servo_selection(self, servo_id: int) -> None:
        set_selected = getattr(self.controller, "set_selected_servo", None)
        if callable(set_selected):
            set_selected(int(servo_id))
        else:
            self.controller.state.selected_servo_id = int(servo_id)
        self.update(self.controller.state)

    def _select_servo_from_row(self, row: int, _column: int) -> None:
        item = self.telemetry_table.item(row, 0)
        if item is None:
            return
        try:
            servo_id = int(item.text())
        except (TypeError, ValueError):
            return
        self._sync_servo_selection(int(servo_id))

    def _safe_call(self, fn, *args, **kwargs) -> None:
        try:
            fn(*args, **kwargs)
        except Exception as exc:
            if not getattr(self.controller.state, "last_error", None):
                self.controller.state.last_error = str(exc)
            if not getattr(self.controller.state, "status_message", ""):
                self.controller.state.status_message = str(exc)
        self.update(self.controller.state)

    # --- Segment pretension trial handlers -------------------------------

    def _record_manual_baseline(self) -> None:
        if self.pretension_trial_controller is None:
            return
        try:
            self.pretension_trial_controller.record_manual_baseline()
        except Exception as exc:
            self.controller.state.last_error = str(exc)
            self.controller.state.status_message = f"Manual baseline capture failed: {exc}"
        self._refresh_segment_pretension_state()
        self.update(self.controller.state)

    def _clear_manual_baselines(self) -> None:
        if self.pretension_trial_controller is None:
            return
        try:
            self.pretension_trial_controller.clear_manual_baselines()
        except Exception as exc:
            self.controller.state.last_error = str(exc)
            self.controller.state.status_message = f"Clearing baselines failed: {exc}"
        self._refresh_segment_pretension_state()
        self.update(self.controller.state)

    def _run_segment_pretension(self) -> None:
        if self.pretension_trial_controller is None:
            return
        # Disable buttons during the run; matplotlib output writing is
        # synchronous and the experiment itself blocks on the worker.
        self.run_segment_pretension_button.setEnabled(False)
        self.record_manual_baseline_button.setEnabled(False)
        self.segment_pretension_status_label.setText("Pretension trial running…")
        try:
            self.pretension_trial_controller.run_pretension_trial()
        except Exception as exc:
            self.controller.state.last_error = str(exc)
            self.controller.state.status_message = f"Pretension trial failed: {exc}"
        finally:
            self.run_segment_pretension_button.setEnabled(True)
            self.record_manual_baseline_button.setEnabled(True)
            self._refresh_segment_pretension_state()
            self.update(self.controller.state)

    def _refresh_segment_pretension_state(self) -> None:
        if self.pretension_trial_controller is None:
            return
        state = self.pretension_trial_controller.state
        self.segment_pretension_status_label.setText(state.last_status)
        self.segment_pretension_manual_count_label.setText(
            f"Manual baselines recorded: {int(state.manual_baseline_count)}"
            + (f" (file: {state.manual_baseline_path})" if state.manual_baseline_path else "")
        )

    @staticmethod
    def _display_value(value) -> str:
        return "—" if value is None else str(value)

    @staticmethod
    def _age_text(age_s: float | None) -> str:
        if age_s is None:
            return "—"
        return f"{float(age_s):.3f}"

    @staticmethod
    def _text_item(value, *, align: int = Qt.AlignLeft | Qt.AlignVCenter) -> QTableWidgetItem:
        item = QTableWidgetItem(str(value))
        item.setTextAlignment(align | Qt.AlignVCenter)
        return item

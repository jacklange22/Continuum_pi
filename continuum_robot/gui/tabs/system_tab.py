"""System tab widget."""

from __future__ import annotations

from pathlib import Path
from typing import Callable

from PySide6.QtCore import Qt, QUrl
from PySide6.QtGui import QDesktopServices
from PySide6.QtWidgets import (
    QApplication,
    QComboBox,
    QDoubleSpinBox,
    QFrame,
    QFormLayout,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPlainTextEdit,
    QPushButton,
    QScrollArea,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from continuum_robot.config.schemas import RobotConfig, RobotSegmentConfig
from continuum_robot.gui.controllers.system_controller import SystemViewState
from continuum_robot.gui.theme import chip_stylesheet, grouped_workspace_stylesheet, semantic_chip_colors
from continuum_robot.gui.view_utils import editable_update_blocked, set_combo_value, set_spinbox_value, set_text_document
from continuum_robot.gui.widgets.no_wheel_combo_box import NoWheelComboBox


class SystemTab(QWidget):
    """System connectivity and troubleshooting UI."""

    def __init__(
        self,
        controller,
        parent=None,
        *,
        apply_runtime_parameters: Callable[..., None] | None = None,
    ) -> None:
        super().__init__(parent)
        self.controller = controller
        self._apply_runtime_parameters = apply_runtime_parameters
        self._updating_parameter_widgets = False
        self._parameter_dirty = False
        self._applied_parameter_values: dict[str, object] = {}
        self._last_state: SystemViewState | None = None
        self.setObjectName("systemWorkspace")
        self.setStyleSheet(
            grouped_workspace_stylesheet(
                object_name="systemWorkspace",
                input_selectors=["QComboBox", "QSpinBox", "QDoubleSpinBox", "QPlainTextEdit"],
                extra_rules="""
                QWidget#systemWorkspace QFrame[role="statusHeader"] {
                    background: #141b22;
                    border: 1px solid #384859;
                    border-radius: 18px;
                }
                QWidget#systemWorkspace QFrame[role="statusCard"] {
                    background: #19222b;
                    border: 1px solid #384859;
                    border-radius: 16px;
                }
                QWidget#systemWorkspace QFrame[role="connectionCard"] {
                    background: #19222b;
                    border: 1px solid #384859;
                    border-radius: 16px;
                }
                QWidget#systemWorkspace QLabel[role="section-kicker"] {
                    color: #b1bcc7;
                    font-size: 11px;
                    font-weight: 700;
                    text-transform: uppercase;
                    letter-spacing: 0.08em;
                }
                QWidget#systemWorkspace QLabel[role="status-title"] {
                    color: #dce4eb;
                    font-size: 14px;
                    font-weight: 700;
                }
                QWidget#systemWorkspace QLabel[role="status-detail"] {
                    color: #b1bcc7;
                }
                QWidget#systemWorkspace QLabel[role="status-pill"] {
                    padding: 7px 12px;
                    border-radius: 999px;
                    font-weight: 700;
                }
                QWidget#systemWorkspace QLabel[role="blockerBanner"] {
                    background: #552d32;
                    color: #fbe8ea;
                    border: 1px solid #956569;
                    border-radius: 12px;
                    padding: 10px 12px;
                    font-weight: 700;
                }
                QWidget#systemWorkspace QWidget[role="advancedPanel"] {
                    background: #141b22;
                    border: 1px solid #384859;
                    border-radius: 12px;
                }
                """,
            )
        )

        self.title_label = QLabel("System")
        self.title_label.setProperty("role", "title")
        self.workflow_hint = QLabel(
            "Check bring-up truth first, connect hardware second, adjust the few startup settings only when needed, and copy session diagnostics fast."
        )
        self.workflow_hint.setProperty("role", "hint")
        self.workflow_hint.setWordWrap(True)

        self.mode_label = self._build_status_pill()
        self.robot_label = self._build_status_pill()
        self.tracker_header_label = self._build_status_pill()
        self.openrb_header_label = self._build_status_pill()
        self.overall_header_label = self._build_status_pill()
        self.blocker_label = QLabel()
        self.blocker_label.setProperty("role", "blockerBanner")
        self.blocker_label.setWordWrap(True)
        self.blocker_label.hide()

        self.tracker_status_label = QLabel()
        self.tracker_status_label.setProperty("role", "status-detail")
        self.tracker_status_label.setWordWrap(True)
        self.openrb_status_label = QLabel()
        self.openrb_status_label.setProperty("role", "status-detail")
        self.openrb_status_label.setWordWrap(True)
        self.saved_path_label = QLabel("none")
        self.saved_path_label.setProperty("role", "status-detail")
        self.saved_path_label.setWordWrap(True)
        self.session_log_label = QLabel("unset")
        self.session_log_label.setProperty("role", "status-detail")
        self.session_log_label.setWordWrap(True)

        self.aurora_port_combo = NoWheelComboBox()
        self.aurora_port_combo.setEditable(True)
        self.aurora_port_combo.currentIndexChanged.connect(self._sync_aurora_port)
        self.aurora_port_combo.editTextChanged.connect(self._sync_aurora_port)
        self.openrb_port_combo = NoWheelComboBox()
        self.openrb_port_combo.setEditable(True)
        self.openrb_port_combo.currentIndexChanged.connect(self._sync_openrb_port)
        self.openrb_port_combo.editTextChanged.connect(self._sync_openrb_port)

        self.tracker_connect_button = QPushButton("Connect Tracker")
        self.tracker_connect_button.setProperty("role", "primary")
        self.tracker_disconnect_button = QPushButton("Disconnect Tracker")
        self.tracker_disconnect_button.setProperty("variant", "ghost")
        self.openrb_connect_button = QPushButton("Connect OpenRB")
        self.openrb_connect_button.setProperty("role", "primary")
        self.openrb_disconnect_button = QPushButton("Disconnect OpenRB")
        self.openrb_disconnect_button.setProperty("variant", "ghost")
        self.prepare_button = QPushButton("Re-Prepare OpenRB Pass-Through")
        self.prepare_button.setProperty("variant", "ghost")
        self.tracker_rescan_button = QPushButton("Rescan")
        self.tracker_rescan_button.setProperty("variant", "ghost")
        self.openrb_rescan_button = QPushButton("Rescan")
        self.openrb_rescan_button.setProperty("variant", "ghost")
        self.tracker_rescan_button.clicked.connect(self._rescan_ports)
        self.openrb_rescan_button.clicked.connect(self._rescan_ports)
        self.tracker_connect_button.clicked.connect(self._connect_tracker)
        self.tracker_disconnect_button.clicked.connect(self.controller.disconnect_tracker)
        self.openrb_connect_button.clicked.connect(self._connect_openrb)
        self.openrb_disconnect_button.clicked.connect(self.controller.disconnect_openrb)
        self.prepare_button.clicked.connect(self.controller.prepare_openrb)

        header_box = QFrame()
        header_box.setProperty("role", "statusHeader")
        header_layout = QVBoxLayout(header_box)
        header_layout.setContentsMargins(18, 18, 18, 18)
        header_layout.setSpacing(12)
        kicker = QLabel("Bring-Up Status")
        kicker.setProperty("role", "section-kicker")
        header_layout.addWidget(kicker)

        card_grid = QGridLayout()
        card_grid.setHorizontalSpacing(12)
        card_grid.setVerticalSpacing(12)
        card_grid.addWidget(self._build_status_card("Mode", self.mode_label), 0, 0)
        card_grid.addWidget(self._build_status_card("Robot Layout", self.robot_label), 0, 1)
        card_grid.addWidget(self._build_status_card("Overall", self.overall_header_label), 0, 2)
        card_grid.addWidget(self._build_status_card("Tracker", self.tracker_header_label), 1, 0, 1, 2)
        card_grid.addWidget(self._build_status_card("OpenRB", self.openrb_header_label), 1, 2)
        header_layout.addLayout(card_grid)
        header_layout.addWidget(self.blocker_label)

        connections_box = QGroupBox("Connections")
        connections_layout = QVBoxLayout(connections_box)
        connections_layout.addWidget(
            self._build_connection_card(
                title="Tracker",
                detail_label=self.tracker_status_label,
                port_combo=self.aurora_port_combo,
                rescan_button=self.tracker_rescan_button,
                connect_button=self.tracker_connect_button,
                disconnect_button=self.tracker_disconnect_button,
            )
        )
        connections_layout.addWidget(
            self._build_connection_card(
                title="OpenRB",
                detail_label=self.openrb_status_label,
                port_combo=self.openrb_port_combo,
                rescan_button=self.openrb_rescan_button,
                connect_button=self.openrb_connect_button,
                disconnect_button=self.openrb_disconnect_button,
            )
        )
        self.connection_advanced_toggle = QPushButton("Show Advanced")
        self.connection_advanced_toggle.setCheckable(True)
        self.connection_advanced_toggle.setProperty("variant", "ghost")
        self.connection_advanced_toggle.toggled.connect(
            lambda checked: self._toggle_advanced_section(
                self.connection_advanced_toggle,
                self.connection_advanced_panel,
                checked,
            )
        )
        self.connection_advanced_panel = QWidget()
        self.connection_advanced_panel.setProperty("role", "advancedPanel")
        self.connection_advanced_panel.hide()
        connection_advanced_layout = QVBoxLayout(self.connection_advanced_panel)
        connection_advanced_layout.setContentsMargins(12, 12, 12, 12)
        connection_advanced_layout.setSpacing(10)
        advanced_hint = QLabel("Use this only if OpenRB needs its DYNAMIXEL pass-through prepared again after reconnects.")
        advanced_hint.setProperty("role", "hint")
        advanced_hint.setWordWrap(True)
        prepare_row = QHBoxLayout()
        prepare_row.addWidget(self.prepare_button)
        prepare_row.addStretch(1)
        connection_advanced_layout.addWidget(advanced_hint)
        connection_advanced_layout.addLayout(prepare_row)
        connections_layout.addWidget(self.connection_advanced_toggle, alignment=Qt.AlignLeft)
        connections_layout.addWidget(self.connection_advanced_panel)

        self.robot_config_combo = NoWheelComboBox()
        self.robot_config_combo.currentIndexChanged.connect(self._mark_parameter_dirty)
        self.robot_config_combo.currentIndexChanged.connect(self._sync_segment_options_for_robot_profile)
        self.operating_mode_combo = NoWheelComboBox()
        for label, value in (
            ("1 Servo", "one_servo"),
            ("Single Segment", "single_segment"),
            ("Dual Segment", "dual_segment"),
            ("Parallel Single", "parallel_single"),
        ):
            self.operating_mode_combo.addItem(label, value)
        self.operating_mode_combo.currentIndexChanged.connect(self._mark_parameter_dirty)
        self.operating_mode_combo.currentIndexChanged.connect(self._sync_operating_mode_visibility)
        self.selected_servo_combo = NoWheelComboBox()
        for servo_id in range(1, 9):
            self.selected_servo_combo.addItem(f"Servo {servo_id}", int(servo_id))
        self.selected_servo_combo.currentIndexChanged.connect(self._mark_parameter_dirty)
        self.selected_servo_combo.currentIndexChanged.connect(self._sync_operating_context_summary)
        self.active_segment_combo = NoWheelComboBox()
        self.active_segment_combo.currentIndexChanged.connect(self._mark_parameter_dirty)
        self.active_segment_combo.currentIndexChanged.connect(self._sync_operating_context_summary)
        self.mock_mode_combo = NoWheelComboBox()
        self.mock_mode_combo.addItem("Enabled", True)
        self.mock_mode_combo.addItem("Disabled", False)
        self.mock_mode_combo.currentIndexChanged.connect(self._mark_parameter_dirty)
        self.baudrate_spin = QSpinBox()
        self.baudrate_spin.setRange(9600, 4000000)
        self.baudrate_spin.valueChanged.connect(self._mark_parameter_dirty)
        self.poll_rate_spin = QSpinBox()
        self.poll_rate_spin.setRange(1, 60)
        self.poll_rate_spin.valueChanged.connect(self._mark_parameter_dirty)
        self.telemetry_freshness_spin = QDoubleSpinBox()
        self.telemetry_freshness_spin.setRange(0.01, 10.0)
        self.telemetry_freshness_spin.setDecimals(3)
        self.telemetry_freshness_spin.setSingleStep(0.05)
        self.telemetry_freshness_spin.valueChanged.connect(self._mark_parameter_dirty)
        self.figure_quality_combo = NoWheelComboBox()
        for label, value in (
            ("Low (120 dpi)", "low"),
            ("Medium (200 dpi)", "medium"),
            ("Production (300 dpi)", "production"),
        ):
            self.figure_quality_combo.addItem(label, value)
        self.figure_quality_combo.currentIndexChanged.connect(self._mark_parameter_dirty)

        self.save_parameters_button = QPushButton("Save + Apply")
        self.save_parameters_button.setProperty("role", "primary")
        self.save_parameters_button.clicked.connect(self._save_runtime_parameters)
        self.parameters_hint = QLabel("Save only the startup settings that should persist into the next launch.")
        self.parameters_hint.setProperty("role", "hint")
        self.parameters_hint.setWordWrap(True)
        self.hardware_profile_hint = QLabel(
            "Hardware profile defines the available servos, segments, geometry, and pair mappings. "
            "Most normal operation should use the full 8-servo platform profile."
        )
        self.hardware_profile_hint.setProperty("role", "hint")
        self.hardware_profile_hint.setWordWrap(True)
        self.operating_context_summary_label = QLabel()
        self.operating_context_summary_label.setProperty("role", "hint")
        self.operating_context_summary_label.setWordWrap(True)

        parameters_box = QGroupBox("Startup Settings")
        parameters_layout = QVBoxLayout(parameters_box)
        self.parameters_form = QFormLayout()
        self.parameters_form.setLabelAlignment(Qt.AlignLeft)
        self.parameters_form.addRow("Mock mode", self.mock_mode_combo)
        self.parameters_form.addRow("Operating mode", self.operating_mode_combo)
        self.parameters_form.addRow("Selected servo", self.selected_servo_combo)
        self.parameters_form.addRow("Active segment", self.active_segment_combo)
        self.parameters_form.addRow("Resolved scope", self.operating_context_summary_label)
        self.parameters_form.addRow("Baudrate", self.baudrate_spin)
        parameters_layout.addLayout(self.parameters_form)
        save_row = QHBoxLayout()
        save_row.addWidget(self.save_parameters_button)
        save_row.addStretch(1)
        parameters_layout.addLayout(save_row)
        parameters_layout.addWidget(self.parameters_hint)
        self.settings_advanced_toggle = QPushButton("Show Advanced")
        self.settings_advanced_toggle.setCheckable(True)
        self.settings_advanced_toggle.setProperty("variant", "ghost")
        self.settings_advanced_toggle.toggled.connect(
            lambda checked: self._toggle_advanced_section(
                self.settings_advanced_toggle,
                self.settings_advanced_panel,
                checked,
            )
        )
        self.settings_advanced_panel = QWidget()
        self.settings_advanced_panel.setProperty("role", "advancedPanel")
        self.settings_advanced_panel.hide()
        settings_advanced_layout = QVBoxLayout(self.settings_advanced_panel)
        settings_advanced_layout.setContentsMargins(12, 12, 12, 12)
        settings_advanced_layout.addWidget(self.hardware_profile_hint)
        settings_advanced_form = QFormLayout()
        settings_advanced_form.setLabelAlignment(Qt.AlignLeft)
        settings_advanced_form.addRow("Hardware profile", self.robot_config_combo)
        settings_advanced_form.addRow("Figure export quality", self.figure_quality_combo)
        settings_advanced_form.addRow("GUI refresh (Hz)", self.poll_rate_spin)
        settings_advanced_form.addRow("Telemetry stale after (s)", self.telemetry_freshness_spin)
        settings_advanced_form.addRow("Saved overrides", self.saved_path_label)
        settings_advanced_layout.addLayout(settings_advanced_form)
        parameters_layout.addWidget(self.settings_advanced_toggle, alignment=Qt.AlignLeft)
        parameters_layout.addWidget(self.settings_advanced_panel)

        self.status_text = QPlainTextEdit()
        self.status_text.setReadOnly(True)
        self.status_text.setLineWrapMode(QPlainTextEdit.WidgetWidth)
        self.status_text.setMinimumHeight(170)
        self.copy_diagnostics_button = QPushButton("Copy Session Diagnostics")
        self.copy_diagnostics_button.clicked.connect(self._copy_session_diagnostics)
        self.copy_log_path_button = QPushButton("Copy Log Path")
        self.copy_log_path_button.setProperty("variant", "ghost")
        self.copy_log_path_button.clicked.connect(lambda: self._copy_text(self.session_log_label.text()))
        self.open_log_button = QPushButton("Open Log")
        self.open_log_button.setProperty("variant", "ghost")
        self.open_log_button.clicked.connect(self._open_session_log)
        self.open_logs_folder_button = QPushButton("Open Logs Folder")
        self.open_logs_folder_button.setProperty("variant", "ghost")
        self.open_logs_folder_button.clicked.connect(self._open_logs_folder)

        diagnostics_box = QGroupBox("Session Diagnostics")
        diagnostics_layout = QVBoxLayout(diagnostics_box)
        diagnostics_form = QFormLayout()
        diagnostics_form.setLabelAlignment(Qt.AlignLeft)
        diagnostics_form.addRow("Current session log", self.session_log_label)
        diagnostics_layout.addLayout(diagnostics_form)
        diagnostics_button_row = QHBoxLayout()
        diagnostics_button_row.addWidget(self.copy_diagnostics_button)
        diagnostics_button_row.addWidget(self.copy_log_path_button)
        diagnostics_button_row.addWidget(self.open_log_button)
        diagnostics_button_row.addWidget(self.open_logs_folder_button)
        diagnostics_button_row.addStretch(1)
        diagnostics_layout.addLayout(diagnostics_button_row)
        diagnostics_preview_hint = QLabel("Latest session activity")
        diagnostics_preview_hint.setProperty("role", "section-kicker")
        diagnostics_layout.addWidget(diagnostics_preview_hint)
        diagnostics_layout.addWidget(self.status_text)

        content_widget = QWidget()
        content_layout = QVBoxLayout(content_widget)
        content_layout.setContentsMargins(0, 0, 0, 0)
        content_layout.setSpacing(14)
        content_layout.addWidget(header_box)
        content_layout.addWidget(connections_box)
        content_layout.addWidget(parameters_box)
        content_layout.addWidget(diagnostics_box)

        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setWidget(content_widget)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(18, 18, 18, 18)
        layout.setSpacing(14)
        layout.addWidget(self.title_label)
        layout.addWidget(self.workflow_hint)
        layout.addWidget(self.scroll_area, 1)

    def update(self, state: SystemViewState) -> None:
        self._last_state = state
        diagnostics_preview = state.diagnostics_preview or "No session activity captured yet."
        self._set_status_pill(self.mode_label, state.mode_display, "accent" if state.mock_mode else "info")
        self._set_status_pill(self.robot_label, state.robot_layout_display, "neutral")
        self._set_status_pill(self.tracker_header_label, state.tracker_status_label, state.tracker_status_kind)
        self._set_status_pill(self.openrb_header_label, state.openrb_status_label, state.openrb_status_kind)
        self._set_status_pill(self.overall_header_label, state.overall_status_label, state.overall_status_kind)
        self.blocker_label.setText(f"Blocking reason: {state.primary_blocker}")
        self.blocker_label.setVisible(bool(str(state.primary_blocker).strip()))
        self.tracker_status_label.setText(state.tracker_truth_summary)
        self.openrb_status_label.setText(state.openrb_truth_summary)
        self.session_log_label.setText(state.session_log_summary or "unset")
        self._set_plain_text_preserving_view(self.status_text, diagnostics_preview)
        self._set_combo_items(self.aurora_port_combo, state.available_ports, state.aurora_port)
        self._set_combo_items(self.openrb_port_combo, state.available_ports, state.openrb_port)
        applied_values = self._parameter_values_from_state(state)
        if self._startup_parameter_popup_open():
            self._sync_operating_mode_visibility()
        elif not self._parameter_dirty:
            self._apply_parameter_values(state, applied_values)
        elif self._current_parameter_values() == applied_values:
            self._parameter_dirty = False
            self._apply_parameter_values(state, applied_values)
        else:
            self._sync_operating_mode_visibility()
        self._applied_parameter_values = applied_values
        self.saved_path_label.setText(state.saved_overrides_path or "none")
        tracker_connected = bool(
            state.tracker_backend_connected or state.tracker_connection_state in {"starting", "connecting"}
        )
        openrb_connected = bool(state.openrb_connected or state.dynamixel_connected)
        self.tracker_connect_button.setEnabled(not tracker_connected)
        self.tracker_disconnect_button.setEnabled(tracker_connected)
        self.openrb_connect_button.setEnabled(not openrb_connected)
        self.openrb_disconnect_button.setEnabled(openrb_connected)
        self.prepare_button.setEnabled(bool(state.openrb_connected))
        session_log_path = self._session_log_path()
        has_session_log = bool(session_log_path is not None and session_log_path.exists())
        self.copy_log_path_button.setEnabled(has_session_log)
        self.open_log_button.setEnabled(has_session_log)
        self.open_logs_folder_button.setEnabled(has_session_log)

    def _build_status_pill(self) -> QLabel:
        label = QLabel()
        label.setProperty("role", "status-pill")
        label.setAlignment(Qt.AlignCenter)
        label.setWordWrap(True)
        return label

    def _build_status_card(self, title: str, value_label: QLabel) -> QFrame:
        card = QFrame()
        card.setProperty("role", "statusCard")
        layout = QVBoxLayout(card)
        layout.setContentsMargins(14, 14, 14, 14)
        layout.setSpacing(10)
        title_label = QLabel(title)
        title_label.setProperty("role", "status-title")
        layout.addWidget(title_label)
        layout.addWidget(value_label)
        layout.addStretch(1)
        return card

    def _build_connection_card(
        self,
        *,
        title: str,
        detail_label: QLabel,
        port_combo: QComboBox,
        rescan_button: QPushButton,
        connect_button: QPushButton,
        disconnect_button: QPushButton,
    ) -> QFrame:
        card = QFrame()
        card.setProperty("role", "connectionCard")
        layout = QVBoxLayout(card)
        layout.setContentsMargins(14, 14, 14, 14)
        layout.setSpacing(10)

        title_label = QLabel(title)
        title_label.setProperty("role", "status-title")
        layout.addWidget(title_label)
        layout.addWidget(detail_label)

        port_row = QHBoxLayout()
        port_row.addWidget(port_combo, 1)
        port_row.addWidget(rescan_button)
        layout.addLayout(port_row)

        action_row = QHBoxLayout()
        action_row.setSpacing(10)
        action_row.addWidget(connect_button)
        action_row.addWidget(disconnect_button)
        action_row.addStretch(1)
        layout.addLayout(action_row)
        return card

    def _toggle_advanced_section(self, button: QPushButton, panel: QWidget, checked: bool) -> None:
        panel.setVisible(bool(checked))
        button.setText("Hide Advanced" if checked else "Show Advanced")

    @staticmethod
    def _set_status_pill(label: QLabel, text: str, kind: str) -> None:
        background, foreground = semantic_chip_colors(kind)
        label.setText(text)
        label.setStyleSheet(chip_stylesheet(background=background, foreground=foreground))

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

    def _rescan_ports(self) -> None:
        self.controller.rescan_ports()

    def _sync_aurora_port(self, _value=None) -> None:
        self.controller.set_aurora_port(self._selected_port(self.aurora_port_combo))

    def _sync_openrb_port(self, _value=None) -> None:
        self.controller.set_openrb_port(self._selected_port(self.openrb_port_combo))

    def _connect_tracker(self) -> None:
        self._sync_aurora_port()
        self.controller.connect_tracker()

    def _connect_openrb(self) -> None:
        self._sync_openrb_port()
        self.controller.connect_openrb()

    def _save_runtime_parameters(self) -> None:
        parameters = {
            "mock_mode": bool(self.mock_mode_combo.currentData()),
            "robot_config": str(self.robot_config_combo.currentData() or self.robot_config_combo.currentText()).strip(),
            "operating_mode": str(self.operating_mode_combo.currentData() or "single_segment"),
            "selected_servo_id": int(self.selected_servo_combo.currentData() or 1),
            "active_segment": str(self.active_segment_combo.currentData() or "").strip(),
            "openrb_port": self._selected_port(self.openrb_port_combo),
            "baudrate": int(self.baudrate_spin.value()),
            "poll_rate_hz": int(self.poll_rate_spin.value()),
            "figure_output_quality": str(self.figure_quality_combo.currentData() or "production"),
            "telemetry_freshness_timeout_s": float(self.telemetry_freshness_spin.value()),
        }
        handler = self._apply_runtime_parameters or self.controller.save_runtime_parameters
        try:
            handler(**parameters)
            self._parameter_dirty = False
            self._applied_parameter_values = dict(parameters)
        except Exception:
            self.update(self.controller.refresh())

    def _copy_session_diagnostics(self) -> None:
        builder = getattr(self.controller, "build_session_diagnostics_document", None)
        if callable(builder):
            self._copy_text(builder())
            return
        self._copy_text(self.status_text.toPlainText())

    def _open_session_log(self) -> None:
        path = self._session_log_path()
        if path is None or not path.exists():
            return
        QDesktopServices.openUrl(QUrl.fromLocalFile(str(path)))

    def _open_logs_folder(self) -> None:
        path = self._session_log_path()
        if path is None or not path.exists():
            return
        logs_root = path.parent.parent if path.parent.name == "current" else path.parent
        QDesktopServices.openUrl(QUrl.fromLocalFile(str(logs_root)))

    def _mark_parameter_dirty(self, *_args) -> None:
        if self._updating_parameter_widgets:
            return
        current_values = self._current_parameter_values()
        self._parameter_dirty = current_values != self._applied_parameter_values
        self._sync_operating_context_summary()

    def _sync_segment_options_for_robot_profile(self, *_args) -> None:
        if self._updating_parameter_widgets:
            return
        loader = getattr(self.controller, "config_loader", None)
        if loader is None:
            return
        robot_config = str(self.robot_config_combo.currentData() or self.robot_config_combo.currentText()).strip()
        try:
            robot = loader.load_robot_config(robot_config)
        except Exception:
            return
        previous = str(self.active_segment_combo.currentData() or "")
        self._set_active_segment_options_from_robot(robot, preferred_key=previous)
        self._sync_operating_mode_visibility()
        self._mark_parameter_dirty()

    def _set_active_segment_options_from_robot(self, robot: RobotConfig, *, preferred_key: str = "") -> None:
        self.active_segment_combo.blockSignals(True)
        self.active_segment_combo.clear()
        for key, segment in robot.segment_map().items():
            servo_ids = [int(value) for value in segment.servo_ids]
            display = f"{segment.label} ({', '.join(str(value) for value in servo_ids)})"
            self.active_segment_combo.addItem(display, str(key))
        index = self.active_segment_combo.findData(str(preferred_key or ""))
        if index < 0:
            index = self.active_segment_combo.findData(robot.active_segment_key())
        if index >= 0:
            self.active_segment_combo.setCurrentIndex(index)
        self.active_segment_combo.blockSignals(False)

    def _sync_operating_mode_visibility(self, *_args) -> None:
        self._maybe_promote_full_platform_profile()
        mode = str(self.operating_mode_combo.currentData() or "single_segment")
        self._set_parameter_row_visible(self.selected_servo_combo, mode == "one_servo")
        self._set_parameter_row_visible(self.active_segment_combo, mode == "single_segment")
        self.selected_servo_combo.setEnabled(mode == "one_servo")
        self.active_segment_combo.setEnabled(mode == "single_segment")
        self._sync_operating_context_summary()

    def _maybe_promote_full_platform_profile(self) -> None:
        if self._updating_parameter_widgets or self.settings_advanced_panel.isVisible():
            return
        full_profile = "robot_8servo.yaml"
        full_index = self.robot_config_combo.findData(full_profile)
        if full_index < 0:
            return
        current_profile = str(self.robot_config_combo.currentData() or self.robot_config_combo.currentText()).strip()
        if current_profile == full_profile:
            return
        loader = getattr(self.controller, "config_loader", None)
        if loader is None:
            return
        try:
            current_robot = loader.load_robot_config(current_profile)
            full_robot = loader.load_robot_config(full_profile)
        except Exception:
            return
        current_is_full = len(current_robot.all_segment_servo_ids()) >= 8 and len(current_robot.segment_map()) >= 2
        full_is_capable = len(full_robot.all_segment_servo_ids()) >= 8 and len(full_robot.segment_map()) >= 2
        if current_is_full or not full_is_capable:
            return
        previous_segment = str(self.active_segment_combo.currentData() or "")
        self.robot_config_combo.blockSignals(True)
        self.robot_config_combo.setCurrentIndex(full_index)
        self.robot_config_combo.blockSignals(False)
        self._set_active_segment_options_from_robot(full_robot, preferred_key=previous_segment)
        self._mark_parameter_dirty()

    def _sync_operating_context_summary(self, *_args) -> None:
        if not hasattr(self, "operating_context_summary_label"):
            return
        try:
            robot = self._pending_robot_config()
            context = self._pending_operating_context(robot)
            lines = self._format_operating_context_summary(context)
            lines.extend(self._format_profile_mode_warnings(robot, context))
        except Exception as exc:
            lines = [f"Could not resolve pending servo scope: {exc}"]
        self.operating_context_summary_label.setText("\n".join(lines))

    def _pending_operating_context(self, robot: RobotConfig | None = None):
        robot = robot or self._pending_robot_config()
        robot.mode = str(self.operating_mode_combo.currentData() or "single_segment")
        robot.selected_servo_id = int(self.selected_servo_combo.currentData() or 1)
        active_segment = str(self.active_segment_combo.currentData() or "").strip()
        if active_segment:
            robot.active_segment = active_segment
        return robot.operating_context()

    def _pending_robot_config(self) -> RobotConfig:
        robot_config = str(self.robot_config_combo.currentData() or self.robot_config_combo.currentText()).strip()
        loader = getattr(self.controller, "config_loader", None)
        if loader is not None and robot_config:
            try:
                return loader.load_robot_config(robot_config)
            except Exception:
                pass
        state = self._last_state or getattr(self.controller, "state", None)
        segments: dict[str, RobotSegmentConfig] = {}
        for segment in list(getattr(state, "available_segments", []) or []):
            key = str(segment.get("key", "") or "").strip()
            if not key:
                continue
            servo_ids = [int(value) for value in list(segment.get("servo_ids", []) or [])]
            pairs = {
                str(pair_key): [int(value) for value in values]
                for pair_key, values in dict(segment.get("pairs", {}) or {}).items()
            }
            segments[key] = RobotSegmentConfig(
                key=key,
                label=str(segment.get("label", key) or key),
                servo_ids=servo_ids,
                pairs=pairs,
            )
        if not segments:
            ids = [int(value) for value in list(getattr(state, "expected_servo_ids", []) or [1, 2, 3, 4])]
            if len(ids) >= 8:
                segments = {
                    "segment_a": RobotSegmentConfig(
                        key="segment_a",
                        label="Spine 1",
                        servo_ids=ids[:4],
                        pairs={"axis_a": [ids[0], ids[2]], "axis_b": [ids[1], ids[3]]},
                    ),
                    "segment_b": RobotSegmentConfig(
                        key="segment_b",
                        label="Spine 2",
                        servo_ids=ids[4:8],
                        pairs={"axis_a": [ids[4], ids[6]], "axis_b": [ids[5], ids[7]]},
                    ),
                }
            else:
                ids = (ids + [1, 2, 3, 4])[:4]
                segments = {
                    "segment_a": RobotSegmentConfig(
                        key="segment_a",
                        label="Spine 1",
                        servo_ids=ids[:4],
                        pairs={"axis_a": [ids[0], ids[2]], "axis_b": [ids[1], ids[3]]},
                    )
                }
        all_ids: list[int] = []
        for segment in segments.values():
            for servo_id in segment.servo_ids:
                sid = int(servo_id)
                if sid not in all_ids:
                    all_ids.append(sid)
        return RobotConfig(
            mode=str(getattr(state, "operating_mode", "single_segment") or "single_segment"),
            servo_ids=all_ids or [1, 2, 3, 4],
            tendon_to_servo=all_ids or [1, 2, 3, 4],
            active_segment=str(getattr(state, "active_segment_key", "segment_a") or "segment_a"),
            selected_servo_id=int(getattr(state, "selected_servo_id", 1) or 1),
            segments=segments,
        )

    @staticmethod
    def _format_operating_context_summary(context) -> list[str]:
        mode = str(context.operating_mode)
        if mode == "one_servo":
            return [f"Expected IDs: {list(context.expected_servo_ids)}"]
        if mode == "single_segment":
            pairs = SystemTab._format_pairs(context.active_pairs)
            return [
                f"{context.active_segment_label}: {list(context.active_segment_servo_ids)}",
                f"Pairs: {pairs or 'not configured'}",
            ]
        if mode == "dual_segment":
            lines = [f"Expected IDs: {list(context.expected_servo_ids)}"]
            lines.extend(SystemTab._format_segment_lines(context.segments))
            return lines
        if mode == "parallel_single":
            mirror = ", ".join(
                f"{int(source)}->{int(target)}"
                for source, target in sorted(dict(context.mirror_pairs or {}).items())
            )
            return [
                f"Expected IDs: {list(context.expected_servo_ids)}",
                f"Mirror mapping: {mirror or 'not configured'}",
                "Mirrored single-segment commands, not full two-segment kinematics.",
            ]
        return [f"Expected IDs: {list(context.expected_servo_ids)}"]

    @staticmethod
    def _format_segment_lines(segments: dict[str, RobotSegmentConfig]) -> list[str]:
        lines: list[str] = []
        for key, segment in sorted(dict(segments or {}).items()):
            label = str(segment.label or key)
            lines.append(f"{label}: {[int(value) for value in segment.servo_ids]}")
        return lines

    @staticmethod
    def _format_pairs(pairs: dict[str, list[int]]) -> str:
        return ", ".join(
            "-".join(str(int(value)) for value in values)
            for _key, values in sorted(dict(pairs or {}).items())
            if values
        )

    @staticmethod
    def _format_profile_mode_warnings(robot: RobotConfig, context) -> list[str]:
        mode = str(context.operating_mode)
        segments = robot.segment_map()
        warnings: list[str] = []
        if mode == "one_servo" and context.selected_servo_id not in context.all_configured_servo_ids:
            warnings.append(
                "Warning: selected servo is not in the current hardware profile. "
                "Use the 8-servo hardware profile to select any servo 1-8."
            )
        if mode == "single_segment" and len(segments) < 2:
            warnings.append(
                "Warning: current hardware profile only defines Segment A. "
                "Use the 8-servo hardware profile to select Segment B."
            )
        if mode in {"dual_segment", "parallel_single"} and len(context.expected_servo_ids) != 8:
            warnings.append(
                "Warning: this operating mode requires an 8-servo hardware profile; "
                f"current profile resolves expected IDs {list(context.expected_servo_ids)}."
            )
        if mode == "parallel_single" and len(context.mirror_pairs) != 4:
            warnings.append(
                "Warning: parallel_single requires four mirror pairs. "
                f"Current hardware profile resolves {dict(context.mirror_pairs)}."
            )
        return warnings

    def _set_parameter_row_visible(self, widget: QWidget, visible: bool) -> None:
        widget.setVisible(bool(visible))
        label = self.parameters_form.labelForField(widget)
        if label is not None:
            label.setVisible(bool(visible))

    def _startup_parameter_popup_open(self) -> bool:
        combos = (
            self.mock_mode_combo,
            self.robot_config_combo,
            self.operating_mode_combo,
            self.selected_servo_combo,
            self.active_segment_combo,
            self.figure_quality_combo,
        )
        return any(bool(getattr(combo, "popup_open", False)) for combo in combos)

    def _apply_parameter_values(self, state: SystemViewState, values: dict[str, object]) -> None:
        self._updating_parameter_widgets = True
        try:
            self.robot_config_combo.blockSignals(True)
            self.robot_config_combo.clear()
            for robot_config in state.available_robot_configs:
                self.robot_config_combo.addItem(robot_config, robot_config)
            if not editable_update_blocked(self.robot_config_combo):
                index = self.robot_config_combo.findData(values["robot_config"])
                if index >= 0:
                    self.robot_config_combo.setCurrentIndex(index)
            self.robot_config_combo.blockSignals(False)

            self.operating_mode_combo.blockSignals(True)
            set_combo_value(self.operating_mode_combo, str(values["operating_mode"]), block_signals=False)
            self.operating_mode_combo.blockSignals(False)

            self.selected_servo_combo.blockSignals(True)
            set_combo_value(self.selected_servo_combo, int(values["selected_servo_id"]), block_signals=False)
            self.selected_servo_combo.blockSignals(False)

            self.active_segment_combo.blockSignals(True)
            self.active_segment_combo.clear()
            for segment in state.available_segments:
                self.active_segment_combo.addItem(
                    str(segment.get("display", segment.get("key", ""))),
                    str(segment.get("key", "")),
                )
            if not editable_update_blocked(self.active_segment_combo):
                segment_index = self.active_segment_combo.findData(values["active_segment"])
                if segment_index >= 0:
                    self.active_segment_combo.setCurrentIndex(segment_index)
            self.active_segment_combo.blockSignals(False)

            set_combo_value(self.mock_mode_combo, bool(values["mock_mode"]), block_signals=True)

            set_spinbox_value(self.baudrate_spin, int(values["baudrate"]), block_signals=True)
            set_spinbox_value(self.poll_rate_spin, int(values["poll_rate_hz"]), block_signals=True)
            self.figure_quality_combo.blockSignals(True)
            set_combo_value(self.figure_quality_combo, str(values["figure_output_quality"]), block_signals=False)
            self.figure_quality_combo.blockSignals(False)
            set_spinbox_value(self.telemetry_freshness_spin, float(values["telemetry_freshness_timeout_s"]), block_signals=True)
            self._sync_operating_mode_visibility()
        finally:
            self._updating_parameter_widgets = False
        self._sync_operating_mode_visibility()

    def _parameter_values_from_state(self, state: SystemViewState) -> dict[str, object]:
        return {
            "mock_mode": bool(state.mock_mode),
            "robot_config": str(state.robot_config),
            "operating_mode": str(state.operating_mode),
            "selected_servo_id": int(state.selected_servo_id),
            "active_segment": str(state.active_segment_key),
            "baudrate": int(state.baudrate),
            "poll_rate_hz": int(state.poll_rate_hz),
            "figure_output_quality": str(state.figure_output_quality),
            "telemetry_freshness_timeout_s": float(state.telemetry_freshness_timeout_s),
        }

    def _current_parameter_values(self) -> dict[str, object]:
        return {
            "mock_mode": bool(self.mock_mode_combo.currentData()),
            "robot_config": str(self.robot_config_combo.currentData() or self.robot_config_combo.currentText()).strip(),
            "operating_mode": str(self.operating_mode_combo.currentData() or "single_segment"),
            "selected_servo_id": int(self.selected_servo_combo.currentData() or 1),
            "active_segment": str(self.active_segment_combo.currentData() or "").strip(),
            "baudrate": int(self.baudrate_spin.value()),
            "poll_rate_hz": int(self.poll_rate_spin.value()),
            "figure_output_quality": str(self.figure_quality_combo.currentData() or "production"),
            "telemetry_freshness_timeout_s": float(self.telemetry_freshness_spin.value()),
        }

    @staticmethod
    def _set_plain_text_preserving_view(widget: QPlainTextEdit, text: str) -> None:
        set_text_document(widget, text, stick_to_bottom_if_at_bottom=True)

    @staticmethod
    def _copy_text(text: str) -> None:
        QApplication.clipboard().setText(str(text))

    def _session_log_path(self) -> Path | None:
        text = str(self.session_log_label.text()).strip()
        return Path(text) if text and text.lower() != "unset" else None

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

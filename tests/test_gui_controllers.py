from __future__ import annotations
import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from pathlib import Path
import json
import logging
import threading
import time
from types import SimpleNamespace

import numpy as np
import pytest

pytestmark = [pytest.mark.gui]

pytest.importorskip("PySide6")

from PySide6.QtCore import Qt
from PySide6.QtGui import QPalette
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication, QBoxLayout, QMessageBox, QScrollArea

from continuum_robot.app.bootstrap import AppContext
from continuum_robot.app.service_registry import ServiceRegistry
from continuum_robot.config.schemas import RegistrationLandmarkConfig, RegistrationWorkflowConfig
from continuum_robot.config.config_loader import ConfigLoader
from continuum_robot.config.settings import Settings
from continuum_robot.config.schemas import (
    CalibrationConfig,
    ExperimentConfig,
    RobotConfig,
    RobotSegmentConfig,
    RuntimeConfig,
    SafetyConfig,
    SerialConfig,
    TwoSegmentTrackingRoleConfig,
)
from continuum_robot.gui.app_window import AppWindow
from continuum_robot.gui.controllers.registration_controller import RegistrationController
from continuum_robot.gui.controllers.servos_controller import ServosController
from continuum_robot.gui.controllers.system_controller import SystemController, SystemViewState
from continuum_robot.gui.controllers.experiment_controller import ExperimentController
from continuum_robot.gui.controllers.tracker_mvp_controller import TrackerMvpController, TrackerMvpViewState
from continuum_robot.gui.experiment_visualization import ChartModel, ScatterSeries3D, VisualizationModel
from continuum_robot.gui.theme import COLORS
from continuum_robot.gui.tabs.experiment_tab import ExperimentTab
from continuum_robot.gui.tabs.registration_tab import RegistrationTab
from continuum_robot.gui.tabs.servos_tab import ServosTab
from continuum_robot.gui.tabs.system_tab import SystemTab
from continuum_robot.gui.tabs.tracker_mvp_tab import TrackerMvpTab
from continuum_robot.gui.tabs.tracking_tab import TrackingTab
from continuum_robot.gui.widgets.experiment_results_widget import ExperimentResultsWidget
from continuum_robot.gui.widgets.experiment_3d_widget import BACKEND_NATIVE_3D, BACKEND_PLACEHOLDER, BACKEND_PROJECTION
from continuum_robot.gui.widgets import experiment_pages as experiment_pages_module
from continuum_robot.gui.widgets.runtime_tip_calibration_dialog import RuntimeTipCalibrationDialog
from continuum_robot.gui.widgets.tool_plot_widget import ToolPlotWidget
from continuum_robot.experiments.experiment_loader import ExperimentLoader
from continuum_robot.experiments.experiment_runner import ExperimentRunner
from continuum_robot.experiments.calibration_validation import ValidationRunCandidate
from continuum_robot.hardware.mock_dxl_bus import MockDxlBus
from continuum_robot.hardware.mock_openrb_client import MockOpenRbClient
from continuum_robot.hardware.serial_ports import SerialPortInfo
from continuum_robot.registration.repository import RegistrationRepository
from continuum_robot.registration.rigid_solver import RigidRegistrationSolver
from continuum_robot.registration.runtime_tip_repository import RuntimeTipCalibrationRepository
from continuum_robot.services.registration_service import RegistrationService
from continuum_robot.services.runtime_tip_calibration_service import RuntimeTipCalibrationService
from continuum_robot.services.tracking_service import TrackingService
from continuum_robot.gui.controllers.runtime_tip_calibration_controller import RuntimeTipCalibrationController
from continuum_robot.servos.displacement_mapper import TendonDisplacementMapper
from continuum_robot.servos.neutral_calibration_service import (
    NeutralCalibrationService,
    ServoCalibrationContext,
)
from continuum_robot.servos.pretension_validation_service import PretensionValidationService
from continuum_robot.servos.safety_guard import SafetyGuard
from continuum_robot.servos.servo_service import PretensionParameters, ServoCommandResult, ServoService
from continuum_robot.hardware.dxl_bus import ServoTelemetry
from continuum_robot.tracking.mock_tracker_manager import MockTrackerManager


def _app() -> QApplication:
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app


def _settings() -> Settings:
    return Settings(
        runtime=RuntimeConfig(mock_mode=True, poll_rate_hz=10, robot_config="robot_4servo.yaml"),
        robot=RobotConfig(
            mode="4-servo",
            spool_diameter_cm=1.2,
            ticks_per_revolution=4096,
            servo_ids=[1, 2, 3, 4],
            tightening_rotation_by_servo={1: "cw", 2: "cw", 3: "cw", 4: "cw"},
        ),
        serial=SerialConfig(aurora_port="/dev/mock-aurora", openrb_port="/dev/mock-openrb", baudrate=115200),
        safety=SafetyConfig(
            position_min_offset_ticks=-600,
            position_max_offset_ticks=600,
            max_current_ma=850,
            default_pretension_current_threshold_ma=220,
            pretension_current_balance_tolerance_ma=120,
            fine_jog_step_ticks=5,
            coarse_jog_step_ticks=25,
            software_position_margin_ticks=64,
            telemetry_stale_after_s=0.25,
            pretension_step_ticks=2,
            pretension_timeout_s=2.0,
            pretension_settle_time_s=0.0,
            max_temperature_c=70,
        ),
        registration=RegistrationWorkflowConfig(
            landmark_labels=["L1", "L2", "L3", "L4"],
            captures_per_landmark=1,
            nominal_landmarks_robot_xyz_mm={
                "L1": [0.0, 0.0, 0.0],
                "L2": [30.0, 0.0, 0.0],
                "L3": [0.0, 30.0, 0.0],
                "L4": [0.0, 0.0, 30.0],
            },
            candidate_landmarks=[
                RegistrationLandmarkConfig(id="L1", xyz_mm=[0.0, 0.0, 0.0], display_label="Front Left"),
                RegistrationLandmarkConfig(id="L2", xyz_mm=[30.0, 0.0, 0.0], display_label="Front Right"),
                RegistrationLandmarkConfig(id="L3", xyz_mm=[0.0, 30.0, 0.0], display_label="Rear Left"),
                RegistrationLandmarkConfig(id="L4", xyz_mm=[0.0, 0.0, 30.0], display_label="Rear Upper"),
            ],
            capture_tool_id="0B",
            max_fre_mm=None,
        ),
        experiment=ExperimentConfig(default_settle_time_s=0.0, sample_count_per_point=1, output_dir="data/experiments"),
        calibration=CalibrationConfig(
            neutral_setpoints_path="config/neutral_setpoints.json",
            latest_registration_path="data/registrations/latest_registration.json",
        ),
    )


def _servo_service(tmp_path: Path, *, dxl_bus=None) -> ServoService:
    return ServoService(
        dxl_bus=dxl_bus or MockDxlBus([1, 2, 3, 4]),
        mapper=TendonDisplacementMapper(spool_diameter_cm=1.2),
        safety_guard=SafetyGuard(
            min_offset_ticks=-600,
            max_offset_ticks=600,
            max_current_ma=850,
            default_pretension_current_threshold_ma=220,
            fine_jog_step_ticks=5,
            coarse_jog_step_ticks=25,
            software_position_margin_ticks=64,
            telemetry_stale_after_s=0.25,
            pretension_step_ticks=2,
            pretension_timeout_s=2.0,
            pretension_settle_time_s=0.0,
            max_temperature_c=70,
            time_fn=time.monotonic,
        ),
        neutral_calibration=NeutralCalibrationService(
            path=tmp_path / "neutral.json",
            context=ServoCalibrationContext(
                robot_mode="4-servo",
                servo_ids=[1, 2, 3, 4],
                tendon_to_servo=[1, 2, 3, 4],
                position_min_offset_ticks=-600,
                position_max_offset_ticks=600,
                default_pretension_current_threshold_ma=220,
                tightening_rotation_by_servo={1: "cw", 2: "cw", 3: "cw", 4: "cw"},
            ),
        ),
        pretension_validation=PretensionValidationService(),
    )


def _pretension_service(tmp_path: Path, *, dxl_bus=None) -> ServoService:
    return ServoService(
        dxl_bus=dxl_bus or _MultiServoPretensionBus(current_sequences={2: [180, 230]}),
        mapper=TendonDisplacementMapper(spool_diameter_cm=1.2),
        safety_guard=SafetyGuard(
            min_offset_ticks=-600,
            max_offset_ticks=600,
            max_current_ma=850,
            default_pretension_current_threshold_ma=220,
            fine_jog_step_ticks=5,
            coarse_jog_step_ticks=25,
            software_position_margin_ticks=64,
            telemetry_stale_after_s=0.25,
            pretension_step_ticks=2,
            pretension_timeout_s=2.0,
            pretension_settle_time_s=0.0,
            pretension_baseline_sample_count=3,
            pretension_current_filter_window=1,
            pretension_current_delta_threshold_ma=60,
            pretension_absolute_trigger_current_ma=500,
            pretension_max_travel_ticks=320,
            max_temperature_c=70,
            time_fn=time.monotonic,
        ),
        neutral_calibration=NeutralCalibrationService(
            path=tmp_path / "pretension_neutral.json",
            context=ServoCalibrationContext(
                robot_mode="4-servo",
                servo_ids=[1, 2, 3, 4],
                tendon_to_servo=[1, 2, 3, 4],
                position_min_offset_ticks=-600,
                position_max_offset_ticks=600,
                default_pretension_current_threshold_ma=220,
                tightening_rotation_by_servo={1: "cw", 2: "cw", 3: "cw", 4: "cw"},
            ),
        ),
        pretension_validation=PretensionValidationService(),
        sleep_fn=lambda _seconds: None,
        time_fn=time.monotonic,
    )


class _MultiServoPretensionBus(MockDxlBus):
    def __init__(self, *, current_sequences: dict[int, list[int | None]]) -> None:
        super().__init__([1, 2, 3, 4])
        self._current_sequences = {int(key): list(values) for key, values in current_sequences.items()}
        for telemetry in self._state.values():
            telemetry.torque_enabled = True
            telemetry.present_position = 4031
            telemetry.present_current_ma = 150

    def write_goal_positions(self, positions_by_id: dict[int, int]) -> None:
        super().write_goal_positions(positions_by_id)
        for servo_id, _goal in positions_by_id.items():
            sequence = self._current_sequences.get(int(servo_id))
            if sequence:
                self._state[int(servo_id)].present_current_ma = sequence.pop(0)


class _RuntimeReadCountingBus(_MultiServoPretensionBus):
    def __init__(self) -> None:
        super().__init__(current_sequences={})
        self.full_read_count = 0
        self.minimal_read_count = 0

    def read_telemetry(self, servo_ids: list[int], **kwargs):
        self.full_read_count += 1
        return super().read_telemetry(servo_ids, **kwargs)

    def read_minimal_telemetry(self, servo_ids: list[int]):
        self.minimal_read_count += 1
        telemetry = MockDxlBus.read_telemetry(
            self,
            servo_ids,
            include_reported_id=False,
            include_identity=False,
            include_limits=False,
        )
        for item in telemetry.values():
            item.present_voltage_raw_unit = None
            item.present_voltage_mv = None
            item.present_temperature_c = None
        return telemetry


class _TorqueEnableFailureMultiServoPretensionBus(_MultiServoPretensionBus):
    def __init__(self, *, current_sequences: dict[int, list[int | None]], fail_message: str = "mock torque enable failure") -> None:
        super().__init__(current_sequences=current_sequences)
        self.fail_message = fail_message
        self.torque_enable_calls: list[tuple[int, bool]] = []

    def write_torque_enable(self, servo_id: int, enabled: bool) -> None:
        self.torque_enable_calls.append((int(servo_id), bool(enabled)))
        raise RuntimeError(self.fail_message)


def _tracking_service(settings: Settings, tmp_path: Path) -> TrackingService:
    return TrackingService(
        live_backend=MockTrackerManager(poll_hz=10),
        port=settings.serial.aurora_port,
        registration_path=tmp_path / "latest_registration.json",
        runtime_tip_calibration_path=tmp_path / "latest_runtime_tip_calibration.json",
        config_source="test",
        runtime_coil_tool_id=settings.registration.coil_tool_id,
        registration_tool_id=settings.registration.capture_tool_id,
    )


def _registration_service(
    settings: Settings,
    tmp_path: Path,
    tracking_service: TrackingService,
    *,
    nominal_points: dict[str, list[float]] | None = None,
    captures_per_landmark: int = 1,
) -> RegistrationService:
    nominal_points = nominal_points or {
        "L1": [5.0, 5.0, 0.0],
        "L2": [23.0, 5.0, 0.0],
        "L3": [5.0, 23.0, 0.0],
        "L4": [5.0, 5.0, 18.0],
    }
    config_path = tmp_path / "registration.yaml"
    lines = [
        f"landmark_labels: [{', '.join(nominal_points.keys())}]",
        f"captures_per_landmark: {captures_per_landmark}",
        "capture_tool_id: \"0B\"",
        "coil_tool_id: \"0A\"",
        "nominal_landmarks_robot_xyz_mm:",
    ]
    for label, point in nominal_points.items():
        lines.append(f"  {label}: [{point[0]}, {point[1]}, {point[2]}]")
    config_path.write_text(
        "\n".join(lines),
        encoding="utf-8",
    )
    return RegistrationService(
        tracking_service=tracking_service,
        repository=RegistrationRepository(root_dir=tmp_path),
        solver=RigidRegistrationSolver(),
        config_path=config_path,
        config_source=str(config_path),
    )


def _runtime_tip_calibration_service(
    settings: Settings,
    tmp_path: Path,
    tracking_service: TrackingService,
    registration_service: RegistrationService,
) -> RuntimeTipCalibrationService:
    return RuntimeTipCalibrationService(
        tracking_service=tracking_service,
        registration_service=registration_service,
        repository=RuntimeTipCalibrationRepository(latest_path=tmp_path / "latest_runtime_tip_calibration.json"),
        solver=RigidRegistrationSolver(),
        registration_config=settings.registration,
        config_path=tmp_path / "registration.yaml",
        config_source="test-registration",
        sleep_fn=lambda _seconds: None,
    )


def _experiment_runner(settings: Settings, tmp_path: Path, tracking_service: TrackingService, servo_service: ServoService, registration_path: Path) -> ExperimentRunner:
    return ExperimentRunner(
        project_root=Path(__file__).resolve().parents[1],
        settings=settings,
        tracking_service=tracking_service,
        servo_service=servo_service,
        output_dir=tmp_path / "data" / "experiments",
        default_settle_time_s=0.0,
        registration_path=registration_path,
        sleep_fn=lambda _seconds: None,
    )


def _experiment_controller(tmp_path: Path) -> ExperimentController:
    settings = _settings()
    servo_service = _servo_service(tmp_path)
    tracking_service = _tracking_service(settings, tmp_path)
    registration_path = tmp_path / "latest_registration.json"
    runner = _experiment_runner(settings, tmp_path, tracking_service, servo_service, registration_path)
    return ExperimentController(
        experiment_loader=ExperimentLoader(),
        experiment_runner=runner,
        registration_path=registration_path,
        servo_service=servo_service,
        tracking_service=tracking_service,
    )


def test_tracker_mvp_tab_wraps_full_workflow_in_scroll_area(tmp_path: Path) -> None:
    _app()
    settings = _settings()
    tracking_service = _tracking_service(settings, tmp_path)
    registration_service = _registration_service(settings, tmp_path, tracking_service)
    registration_controller = RegistrationController(
        registration_service=registration_service,
        registration_config=settings.registration,
    )
    experiment_runner = _experiment_runner(
        settings,
        tmp_path,
        tracking_service,
        _servo_service(tmp_path),
        tmp_path / "latest_registration.json",
    )
    tracker_mvp_controller = TrackerMvpController(
        tracking_service=tracking_service,
        registration_service=registration_service,
        registration_controller=registration_controller,
        experiment_runner=experiment_runner,
        settings=settings,
        project_root=tmp_path,
    )
    tab = TrackerMvpTab(tracker_mvp_controller, registration_controller)

    assert isinstance(tab.scroll_area, QScrollArea)
    assert tab.scroll_area.widget() is not None
    assert tab.scroll_area.widget().findChild(RegistrationTab) is tab.registration_tab
    assert tab.title_label.text() == "Legacy Tracker Compatibility Workspace"
    assert COLORS.workspace_bg in tab.styleSheet()
    assert "#eef3f8" not in tab.styleSheet()


def test_tracking_tab_wraps_workspace_in_scroll_area(tmp_path: Path) -> None:
    _app()
    settings = _settings()
    tracking_service = _tracking_service(settings, tmp_path)
    registration_service = _registration_service(settings, tmp_path, tracking_service)
    registration_controller = RegistrationController(
        registration_service=registration_service,
        registration_config=settings.registration,
    )
    experiment_runner = _experiment_runner(
        settings,
        tmp_path,
        tracking_service,
        _servo_service(tmp_path),
        tmp_path / "latest_registration.json",
    )
    tracker_mvp_controller = TrackerMvpController(
        tracking_service=tracking_service,
        registration_service=registration_service,
        registration_controller=registration_controller,
        experiment_runner=experiment_runner,
        settings=settings,
        project_root=tmp_path,
    )
    tracking_controller = None
    try:
        from continuum_robot.gui.controllers.tracking_controller import TrackingController

        tracking_controller = TrackingController(
            tracking_service=tracking_service,
            settings=settings,
            registration_path=tmp_path / "latest_registration.json",
        )
        tab = TrackingTab(tracking_controller, workflow_controller=tracker_mvp_controller)
        assert isinstance(tab.scroll_area, QScrollArea)
        assert tab.scroll_area.widget() is not None
        assert tab.plot_widget.minimumHeight() >= 420
        assert tab.plot_widget.minimumHeight() > tab.tools_table.minimumHeight()
    finally:
        if tracking_controller is not None:
            tracking_controller.shutdown()


def test_tracking_tab_demotes_disconnect_and_diagnostic_actions(tmp_path: Path) -> None:
    _app()
    settings = _settings()
    tracking_service = _tracking_service(settings, tmp_path)
    registration_service = _registration_service(settings, tmp_path, tracking_service)
    registration_controller = RegistrationController(
        registration_service=registration_service,
        registration_config=settings.registration,
    )
    experiment_runner = _experiment_runner(
        settings,
        tmp_path,
        tracking_service,
        _servo_service(tmp_path),
        tmp_path / "latest_registration.json",
    )
    workflow_controller = TrackerMvpController(
        tracking_service=tracking_service,
        registration_service=registration_service,
        registration_controller=registration_controller,
        experiment_runner=experiment_runner,
        settings=settings,
        project_root=tmp_path,
    )
    from continuum_robot.gui.controllers.tracking_controller import TrackingController

    live_controller = TrackingController(
        tracking_service=tracking_service,
        settings=settings,
        registration_path=tmp_path / "latest_registration.json",
    )
    try:
        tab = TrackingTab(live_controller, workflow_controller=workflow_controller)
        tab.update(workflow_controller.refresh(), live_controller.refresh())

        assert tab.connect_button.isEnabled() is True
        assert tab.disconnect_button.isEnabled() is False
        assert tab.validate_button.isEnabled() is False
        assert tab.disconnect_button.property("variant") == "ghost"
        assert tab.validate_button.property("variant") == "ghost"
        assert tab.validate_button.text() == "Run Tracker Diagnostic"

        workflow_controller.connect_tracker()
        tab.update(workflow_controller.refresh(), live_controller.refresh())

        assert tab.connect_button.isEnabled() is False
        assert tab.disconnect_button.isEnabled() is True
        assert tab.validate_button.isEnabled() is True
    finally:
        live_controller.shutdown()


def test_registration_tab_uses_workflow_state_to_gate_begin_button(tmp_path: Path) -> None:
    _app()
    settings = _settings()
    tracking_service = _tracking_service(settings, tmp_path)
    registration_service = _registration_service(settings, tmp_path, tracking_service)
    controller = RegistrationController(
        registration_service=registration_service,
        registration_config=settings.registration,
    )
    tab = RegistrationTab(controller)
    workflow_state = TrackerMvpViewState(
        tracker_port="/dev/mock-aurora",
        registration_ready=False,
        registration_blockers=["Accept the staged pivot tip file before registration."],
        pivot_tip_path="data/pivot_calibration/generated_penprobe_tip.csv",
        measurement_point_message="Pen-probe tip file loaded from data/pivot_calibration/generated_penprobe_tip.csv.",
        latest_registration_status="No accepted registration saved.",
        live_tip_status="missing_registration",
    )

    tab.update(controller.refresh(), workflow_state)

    assert tab.begin_button.isEnabled() is False
    assert "Blocked:" in tab.dependency_status_label.text()
    assert "Accept the staged pivot tip file" in tab.dependency_text.toPlainText()


def test_registration_tab_wraps_workspace_in_scroll_area(tmp_path: Path) -> None:
    _app()
    settings = _settings()
    tracking_service = _tracking_service(settings, tmp_path)
    registration_service = _registration_service(settings, tmp_path, tracking_service)
    controller = RegistrationController(
        registration_service=registration_service,
        registration_config=settings.registration,
    )
    tab = RegistrationTab(controller)

    assert isinstance(tab.scroll_area, QScrollArea)
    assert tab.scroll_area.widget() is not None


def test_registration_tab_launches_runtime_tip_calibration_dialog_from_app_window(tmp_path: Path) -> None:
    _app()
    window = AppWindow(_app_context(tmp_path))
    try:
        window._refresh_timer.stop()
        window.show()
        window.tab_widget.setCurrentWidget(window.registration_tab)
        QTest.mouseClick(window.registration_tab.runtime_tip_button, Qt.LeftButton)

        assert window.runtime_tip_calibration_dialog.isVisible() is True
    finally:
        window.shutdown()


def test_registration_tab_runtime_tip_mode_selector_updates_tracking_service(tmp_path: Path) -> None:
    _app()
    settings = _settings()
    tracking_service = _tracking_service(settings, tmp_path)
    registration_service = _registration_service(settings, tmp_path, tracking_service)
    controller = RegistrationController(
        registration_service=registration_service,
        registration_config=settings.registration,
    )
    tab = RegistrationTab(controller)

    tab.update(controller.refresh())
    tab.runtime_tip_mode_combo.setCurrentIndex(tab.runtime_tip_mode_combo.findData("coil_as_tip"))
    tab.update(controller.refresh())

    assert tracking_service.get_snapshot().runtime_tip_mode == "coil_as_tip"
    assert controller.state.runtime_tip_mode == "coil_as_tip"
    assert tab.runtime_tip_mode_combo.currentData() == "coil_as_tip"


def test_runtime_tip_calibration_dialog_preserves_manual_quick_mode_selection_across_refresh(tmp_path: Path) -> None:
    _app()
    settings = _settings()
    tracking_service = _tracking_service(settings, tmp_path)
    registration_service = _registration_service(settings, tmp_path, tracking_service)
    runtime_tip_service = _runtime_tip_calibration_service(
        settings,
        tmp_path,
        tracking_service,
        registration_service,
    )
    controller = RuntimeTipCalibrationController(runtime_tip_service)
    dialog = RuntimeTipCalibrationDialog(controller)
    try:
        dialog.update(controller.refresh())
        dialog.session_mode_combo.setCurrentIndex(dialog.session_mode_combo.findData("quick_4_point"))
        dialog.refresh()

        assert dialog.session_mode_combo.currentData() == "quick_4_point"
    finally:
        dialog.close()


def test_runtime_tip_calibration_dialog_prefers_quick_mode_from_registration_selection(tmp_path: Path) -> None:
    _app()
    window = AppWindow(_app_context(tmp_path))
    try:
        window._refresh_timer.stop()
        window.registration_tab.update(window.registration_controller.refresh())
        window.registration_tab.runtime_tip_mode_combo.setCurrentIndex(
            window.registration_tab.runtime_tip_mode_combo.findData("quick_4_point")
        )

        window._open_runtime_tip_calibration()

        assert window.runtime_tip_calibration_dialog.session_mode_combo.currentData() == "quick_4_point"
        assert "quick 4 point" in window.runtime_tip_calibration_dialog.live_runtime_tip_mode_label.text().lower()
        assert "quick 4-point live mode is selected" in window.runtime_tip_calibration_dialog.live_runtime_tip_guidance_label.text().lower()
    finally:
        window.shutdown()


def test_runtime_tip_calibration_dialog_surfaces_explicit_coil_as_tip_truth(tmp_path: Path) -> None:
    _app()
    window = AppWindow(_app_context(tmp_path))
    try:
        window._refresh_timer.stop()
        window.registration_tab.update(window.registration_controller.refresh())
        window.registration_tab.runtime_tip_mode_combo.setCurrentIndex(
            window.registration_tab.runtime_tip_mode_combo.findData("coil_as_tip")
        )

        window._open_runtime_tip_calibration()

        assert "coil as tip" in window.runtime_tip_calibration_dialog.live_runtime_tip_mode_label.text().lower()
        assert "0a coil pose is shown directly as the tip" in window.runtime_tip_calibration_dialog.live_runtime_tip_source_label.text().lower()
        assert "direct 0a / no-transform mode is active" in window.runtime_tip_calibration_dialog.live_runtime_tip_guidance_label.text().lower()
    finally:
        window.shutdown()


def test_tool_plot_widget_accepts_xyz_points() -> None:
    _app()
    widget = ToolPlotWidget()

    widget.set_points({"0A": (10.0, 20.0, 30.0), "tip": (5.0, 6.0)})

    assert widget._points["0A"] == (10.0, 20.0, 30.0)
    assert widget._points["tip"] == (5.0, 6.0, 0.0)


def test_app_window_promotes_tracking_and_registration_before_legacy(tmp_path: Path) -> None:
    _app()
    window = AppWindow(_app_context(tmp_path))
    try:
        labels = [window.tab_widget.tabText(index) for index in range(window.tab_widget.count())]
        assert labels == ["System", "Tracking", "Registration", "Servos", "Pretension", "Experiment", "Modeling", "Data"]
    finally:
        window.shutdown()


def test_servos_tab_wraps_workspace_in_scroll_area(tmp_path: Path) -> None:
    _app()
    controller = ServosController(_servo_service(tmp_path), _settings())
    tab = ServosTab(controller)

    assert isinstance(tab.scroll_area, QScrollArea)
    assert tab.scroll_area.widget() is not None


def test_servos_tab_telemetry_table_reflects_controller_state(tmp_path: Path) -> None:
    _app()
    service = _servo_service(tmp_path)
    service.connect("/dev/mock-openrb", 115200)
    controller = ServosController(service, _settings())
    controller.set_selected_servo(2)
    controller.fine_jog(2, 1)
    tab = ServosTab(controller)

    tab.update(controller.state)

    sorted_servo_ids = sorted(controller.state.telemetry)
    row = sorted_servo_ids.index(2)
    assert tab.telemetry_table.item(row, 0).text() == "2"
    assert tab.telemetry_table.item(row, 3).text() == str(service.dxl_bus._state[2].present_position)
    assert tab.telemetry_table.item(row, 4).text() == str(service.dxl_bus._state[2].present_current_ma)


def test_servos_tab_status_line_summarises_connection(tmp_path: Path) -> None:
    _app()
    service = _servo_service(tmp_path)
    service.connect("/dev/mock-openrb", 115200)
    controller = ServosController(service, _settings())
    controller.set_selected_servo(3)
    tab = ServosTab(controller)

    tab.update(controller.state)

    assert "Bus connected" in tab.status_label.text()
    assert tab.jog_label.text() == "Jog: Servo 3"


def test_servos_tab_drops_discover_and_id_assignment_buttons(tmp_path: Path) -> None:
    _app()
    tab = ServosTab(ServosController(_servo_service(tmp_path), _settings()))

    assert not hasattr(tab, "scan_button")
    assert not hasattr(tab, "refresh_readiness_button")
    assert not hasattr(tab, "assign_button")


def test_system_tab_save_jog_settings_includes_fine_and_coarse_ticks(tmp_path: Path) -> None:
    _app()
    settings = _settings()
    servo_service = _servo_service(tmp_path)
    system_controller = SystemController(
        tracking_service=_tracking_service(settings, tmp_path),
        openrb_client=MockOpenRbClient(),
        servo_service=servo_service,
        settings=settings,
    )
    received: list[dict] = []
    tab = SystemTab(system_controller, apply_runtime_parameters=lambda **kwargs: received.append(dict(kwargs)))

    tab.update(system_controller.refresh())
    tab.fine_jog_step_spin.setValue(7)
    tab.coarse_jog_step_spin.setValue(31)
    tab.save_parameters_button.click()

    assert received, "expected save callback to fire"
    assert received[-1]["fine_jog_step_ticks"] == 7
    assert received[-1]["coarse_jog_step_ticks"] == 31


def test_servos_controller_refresh_selected_servo_preserves_cached_state_when_bus_busy(tmp_path: Path) -> None:
    settings = _settings()
    service = _pretension_service(tmp_path, dxl_bus=_MultiServoPretensionBus(current_sequences={2: [180, 230]}))
    service.connect("/dev/mock-openrb", 115200)
    controller = ServosController(servo_service=service, settings=settings)
    controller.set_selected_servo(2)
    controller.refresh_selected_servo()
    cached_position = controller.state.selected_servo_current_position_tick
    ready = threading.Event()
    release = threading.Event()

    def _owner() -> None:
        with service.exclusive_bus_operation(
            owner="pretension run",
            servo_id=2,
            reason="selected-servo pretension",
        ):
            ready.set()
            release.wait(timeout=1.0)

    thread = threading.Thread(target=_owner, daemon=True)
    thread.start()
    assert ready.wait(timeout=1.0)
    try:
        state = controller.refresh_selected_servo()
        assert state.selected_servo_current_position_tick == cached_position
        assert "owned by active pretension run on servo 2" in state.status_message
        assert state.last_error is None
    finally:
        release.set()
        thread.join(timeout=1.0)


def test_system_controller_refresh_readiness_preserves_counts_when_bus_busy(tmp_path: Path) -> None:
    settings = _settings()
    service = _pretension_service(tmp_path, dxl_bus=_MultiServoPretensionBus(current_sequences={}))
    service.connect("/dev/mock-openrb", 115200)
    controller = SystemController(
        tracking_service=_tracking_service(settings, tmp_path),
        openrb_client=MockOpenRbClient(),
        servo_service=service,
        settings=settings,
    )
    snapshot = service.build_runtime_servo_snapshot([1, 2, 3, 4], selected_servo_id=2)
    controller.sync_servo_runtime_snapshot(snapshot)
    ready = threading.Event()
    release = threading.Event()

    def _owner() -> None:
        with service.exclusive_bus_operation(
            owner="pretension run",
            servo_id=2,
            reason="selected-servo pretension",
        ):
            ready.set()
            release.wait(timeout=1.0)

    thread = threading.Thread(target=_owner, daemon=True)
    thread.start()
    assert ready.wait(timeout=1.0)
    try:
        state = controller.refresh_readiness()
        assert state.motion_ready_count == snapshot.motion_ready_count
        assert state.telemetry_ready_count == snapshot.telemetry_ready_count
        assert "owned by pretension run" in state.readiness_message
    finally:
        release.set()
        thread.join(timeout=1.0)


def test_system_controller_refresh_readiness_uses_runtime_snapshot_counts(tmp_path: Path) -> None:
    settings = _settings()
    service = _pretension_service(tmp_path, dxl_bus=_MultiServoPretensionBus(current_sequences={}))
    service.connect("/dev/mock-openrb", 115200)
    controller = SystemController(
        tracking_service=_tracking_service(settings, tmp_path),
        openrb_client=MockOpenRbClient(),
        servo_service=service,
        settings=settings,
    )

    state = controller.refresh_readiness()

    assert state.telemetry_ready_count == 4
    assert state.motion_ready_count == 4
    assert state.motion_ready is True
    assert "Packet read 4/4" in state.readiness_message
    assert "Experiments use fresh pre-motion read" in state.readiness_message


def test_system_controller_normal_refresh_uses_minimal_runtime_telemetry(tmp_path: Path) -> None:
    settings = _settings()
    bus = _RuntimeReadCountingBus()
    service = _pretension_service(tmp_path, dxl_bus=bus)
    service.connect("/dev/mock-openrb", 57600)
    controller = SystemController(
        tracking_service=_tracking_service(settings, tmp_path),
        openrb_client=MockOpenRbClient(),
        servo_service=service,
        settings=settings,
    )

    state = controller.refresh_readiness(include_scan=False)

    assert state.telemetry_ready_count == 4
    assert bus.minimal_read_count == 1
    assert bus.full_read_count == 0


def test_system_controller_refresh_readiness_uses_cache_for_repeated_timer_refresh(tmp_path: Path, monkeypatch) -> None:
    settings = _settings()
    service = _pretension_service(tmp_path, dxl_bus=_MultiServoPretensionBus(current_sequences={}))
    service.connect("/dev/mock-openrb", 115200)
    controller = SystemController(
        tracking_service=_tracking_service(settings, tmp_path),
        openrb_client=MockOpenRbClient(),
        servo_service=service,
        settings=settings,
    )
    live_calls = {"count": 0}
    original = service.build_runtime_servo_snapshot

    def _counting_live(*args, **kwargs):
        live_calls["count"] += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(service, "build_runtime_servo_snapshot", _counting_live)

    controller.refresh_readiness(include_scan=False)
    controller.refresh_readiness(include_scan=False)
    controller.refresh_readiness(include_scan=False)

    assert live_calls["count"] == 1


def test_system_controller_explicit_scan_bypasses_readiness_cache(tmp_path: Path, monkeypatch) -> None:
    settings = _settings()
    service = _pretension_service(tmp_path, dxl_bus=_MultiServoPretensionBus(current_sequences={}))
    service.connect("/dev/mock-openrb", 115200)
    controller = SystemController(
        tracking_service=_tracking_service(settings, tmp_path),
        openrb_client=MockOpenRbClient(),
        servo_service=service,
        settings=settings,
    )
    live_calls = {"count": 0}
    original = service.build_runtime_servo_snapshot

    def _counting_live(*args, **kwargs):
        live_calls["count"] += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(service, "build_runtime_servo_snapshot", _counting_live)

    controller.refresh_readiness(include_scan=True)
    controller.refresh_readiness(include_scan=True)

    assert live_calls["count"] == 2


def _app_context(tmp_path: Path) -> AppContext:
    settings = _settings()
    tracking_service = _tracking_service(settings, tmp_path)
    servo_service = _servo_service(tmp_path)
    registration_service = _registration_service(settings, tmp_path, tracking_service)
    runtime_tip_calibration_service = _runtime_tip_calibration_service(
        settings,
        tmp_path,
        tracking_service,
        registration_service,
    )
    registration_path = tmp_path / "latest_registration.json"
    experiment_runner = _experiment_runner(
        settings,
        tmp_path,
        tracking_service,
        servo_service,
        registration_path,
    )
    services = ServiceRegistry()
    services.register("tracking_service", tracking_service)
    services.register("registration_service", registration_service)
    services.register("runtime_tip_calibration_service", runtime_tip_calibration_service)
    services.register("servo_service", servo_service)
    services.register("openrb_client", MockOpenRbClient())
    services.register("experiment_loader", ExperimentLoader())
    services.register("experiment_runner", experiment_runner)
    return AppContext(
        project_root=tmp_path,
        settings=settings,
        config_loader=ConfigLoader(),
        services=services,
    )


class _PortSelectionController:
    def __init__(self) -> None:
        self.save_calls: list[dict] = []
        self.state = SystemViewState(
            mock_mode=True,
            aurora_port="/dev/mock-aurora",
            openrb_port="/dev/mock-openrb",
            baudrate=115200,
            poll_rate_hz=20,
            robot_config="robot_4servo.yaml",
            robot_mode="4-servo",
            expected_servo_ids=[1, 2, 3, 4],
            telemetry_freshness_timeout_s=0.25,
            available_robot_configs=["robot_1servo.yaml", "robot_4servo.yaml"],
            available_ports=[
                SerialPortInfo(device="/dev/mock-aurora", description="Mock Aurora"),
                SerialPortInfo(device="/dev/mock-openrb", description="Mock OpenRB"),
                SerialPortInfo(device="/dev/ttyUSB_TRACKER", description="Tracker"),
                SerialPortInfo(device="/dev/ttyUSB_OPENRB", description="OpenRB"),
            ],
        )

    def set_aurora_port(self, port: str) -> None:
        self.state.aurora_port = port

    def set_openrb_port(self, port: str) -> None:
        self.state.openrb_port = port

    def connect_tracker(self) -> None:
        pass

    def disconnect_tracker(self) -> None:
        pass

    def connect_openrb(self) -> None:
        pass

    def disconnect_openrb(self) -> None:
        pass

    def prepare_openrb(self) -> None:
        pass

    def rescan_ports(self) -> SystemViewState:
        return self.state

    def save_runtime_parameters(self, **_kwargs) -> None:
        self.save_calls.append(dict(_kwargs))

    def refresh(self) -> SystemViewState:
        return self.state


class _PretensionConfigLoader:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.saved_overrides: dict | None = None

    def save_system_local_overrides(self, overrides: dict) -> Path:
        self.saved_overrides = dict(overrides)
        return self.path


def test_system_controller_connects_mock_tracker_and_openrb(tmp_path: Path) -> None:
    settings = _settings()
    tracking_service = _tracking_service(settings, tmp_path)
    servo_service = _servo_service(tmp_path)
    controller = SystemController(
        tracking_service=tracking_service,
        openrb_client=MockOpenRbClient(),
        servo_service=servo_service,
        settings=settings,
    )
    try:
        controller.connect_tracker()
        controller.connect_openrb()
        state = controller.refresh()

        assert any(port.device == "/dev/mock-aurora" for port in state.available_ports)
        assert state.tracker_connection_state == "tracking"
        assert state.tracker_backend_identity == "mock_tracker_manager"
        assert state.openrb_connected is True
        assert state.openrb_prepared is True
        assert state.dynamixel_connected is True
    finally:
        controller.disconnect_tracker()
        controller.disconnect_openrb()


def test_system_controller_openrb_connect_falls_back_when_configured_port_is_stale(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    class _FlakyOpenRbClient(MockOpenRbClient):
        def __init__(self) -> None:
            super().__init__()
            self.attempted_ports: list[str] = []

        def connect(self, port: str, baudrate: int = 115200) -> None:
            self.attempted_ports.append(str(port))
            if str(port) == "/dev/ttyACM1":
                raise RuntimeError("configured port not available")
            super().connect(port, baudrate)

    settings = _settings()
    settings.runtime.mock_mode = False
    settings.serial.aurora_port = "/dev/ttyUSB_TRACKER"
    settings.serial.openrb_port = "/dev/ttyACM1"
    tracking_service = _tracking_service(settings, tmp_path)
    servo_service = _servo_service(tmp_path)
    openrb_client = _FlakyOpenRbClient()
    controller = SystemController(
        tracking_service=tracking_service,
        openrb_client=openrb_client,
        servo_service=servo_service,
        settings=settings,
    )
    monkeypatch.setattr(
        controller,
        "_refresh_available_ports_snapshot",
        lambda: [
            SerialPortInfo(device="/dev/ttyACM1", description="OpenRB stale mapping"),
            SerialPortInfo(device="/dev/ttyACM0", description="OpenRB USB serial"),
            SerialPortInfo(device="/dev/ttyUSB_TRACKER", description="Tracker"),
        ],
    )
    controller.state.openrb_port = "/dev/ttyACM1"
    controller.state.aurora_port = "/dev/ttyUSB_TRACKER"
    try:
        controller.connect_openrb()
        assert openrb_client.attempted_ports[:2] == ["/dev/ttyACM1", "/dev/ttyACM0"]
        assert controller.state.openrb_port == "/dev/ttyACM0"
        assert controller.state.dynamixel_connected is True
    finally:
        controller.disconnect_openrb()


def test_system_controller_openrb_connect_skips_onboard_uart_fallback_by_default(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    class _RecordingOpenRbClient(MockOpenRbClient):
        def __init__(self) -> None:
            super().__init__()
            self.attempted_ports: list[str] = []

        def connect(self, port: str, baudrate: int = 115200) -> None:
            self.attempted_ports.append(str(port))
            super().connect(port, baudrate)

    settings = _settings()
    settings.runtime.mock_mode = False
    settings.serial.aurora_port = "/dev/ttyUSB_TRACKER"
    settings.serial.openrb_port = "/dev/ttyACM9"
    settings.serial.openrb_settings = {}
    controller = SystemController(
        tracking_service=_tracking_service(settings, tmp_path),
        openrb_client=_RecordingOpenRbClient(),
        servo_service=_servo_service(tmp_path),
        settings=settings,
    )
    monkeypatch.setattr(
        controller,
        "_refresh_available_ports_snapshot",
        lambda: [
            SerialPortInfo(device="/dev/ttyAMA0", description="Broadcom UART"),
            SerialPortInfo(device="/dev/ttyACM0", description="OpenRB USB serial"),
            SerialPortInfo(device="/dev/ttyUSB_TRACKER", description="Tracker"),
        ],
    )
    controller.state.openrb_port = "/dev/ttyACM9"
    controller.state.aurora_port = "/dev/ttyUSB_TRACKER"
    try:
        controller.connect_openrb()
        attempted = controller.openrb_client.attempted_ports
        assert attempted
        assert attempted[0] == "/dev/ttyACM0"
        assert "/dev/ttyAMA0" not in attempted
        assert controller.state.openrb_port == "/dev/ttyACM0"
    finally:
        controller.disconnect_openrb()


def test_system_controller_openrb_connect_allows_onboard_uart_when_explicitly_enabled(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    class _RecordingOpenRbClient(MockOpenRbClient):
        def __init__(self) -> None:
            super().__init__()
            self.attempted_ports: list[str] = []

        def connect(self, port: str, baudrate: int = 115200) -> None:
            self.attempted_ports.append(str(port))
            super().connect(port, baudrate)

    settings = _settings()
    settings.runtime.mock_mode = False
    settings.serial.aurora_port = "/dev/ttyUSB_TRACKER"
    settings.serial.openrb_port = "/dev/ttyAMA0"
    settings.serial.openrb_settings = {"allow_onboard_uart_fallback": True}
    controller = SystemController(
        tracking_service=_tracking_service(settings, tmp_path),
        openrb_client=_RecordingOpenRbClient(),
        servo_service=_servo_service(tmp_path),
        settings=settings,
    )
    monkeypatch.setattr(
        controller,
        "_refresh_available_ports_snapshot",
        lambda: [
            SerialPortInfo(device="/dev/ttyAMA0", description="Broadcom UART"),
            SerialPortInfo(device="/dev/ttyUSB_TRACKER", description="Tracker"),
        ],
    )
    controller.state.openrb_port = "/dev/ttyAMA0"
    controller.state.aurora_port = "/dev/ttyUSB_TRACKER"
    try:
        controller.connect_openrb()
        attempted = controller.openrb_client.attempted_ports
        assert attempted == ["/dev/ttyAMA0"]
        assert controller.state.openrb_port == "/dev/ttyAMA0"
    finally:
        controller.disconnect_openrb()


def test_system_controller_openrb_candidate_filter_excludes_tracker_port(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    settings = _settings()
    settings.runtime.mock_mode = False
    settings.serial.aurora_port = "/dev/ttyUSB0"
    settings.serial.openrb_port = "/dev/ttyUSB0"
    controller = SystemController(
        tracking_service=_tracking_service(settings, tmp_path),
        openrb_client=MockOpenRbClient(),
        servo_service=_servo_service(tmp_path),
        settings=settings,
    )
    monkeypatch.setattr(
        controller,
        "_refresh_available_ports_snapshot",
        lambda: [
            SerialPortInfo(device="/dev/ttyUSB0", description="Tracker"),
            SerialPortInfo(device="/dev/ttyACM0", description="OpenRB USB serial"),
        ],
    )
    controller.state.aurora_port = "/dev/ttyUSB0"
    controller.state.openrb_port = "/dev/ttyUSB0"

    candidates, skipped = controller._openrb_port_candidates()

    assert all(candidate_port != "/dev/ttyUSB0" for candidate_port, _reason in candidates)
    assert any(
        row["port"] == "/dev/ttyUSB0"
        and "tracker" in row.get("detail", "")
        for row in skipped
    )


def test_system_controller_reports_openrb_stage_when_bus_responds_but_configured_servos_missing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    settings = _settings()
    settings.runtime.mock_mode = False
    settings.serial.aurora_port = "/dev/ttyUSB_TRACKER"
    settings.serial.openrb_port = "/dev/ttyACM0"
    tracker_service = _tracking_service(settings, tmp_path)
    servo_service = ServoService(
        dxl_bus=MockDxlBus([9]),
        mapper=TendonDisplacementMapper(spool_diameter_cm=1.2),
        safety_guard=SafetyGuard(
            min_offset_ticks=-600,
            max_offset_ticks=600,
            max_current_ma=850,
            default_pretension_current_threshold_ma=220,
            fine_jog_step_ticks=5,
            coarse_jog_step_ticks=25,
            software_position_margin_ticks=64,
            telemetry_stale_after_s=0.25,
            pretension_step_ticks=2,
            pretension_timeout_s=2.0,
            pretension_settle_time_s=0.0,
            max_temperature_c=70,
        ),
        neutral_calibration=NeutralCalibrationService(
            path=tmp_path / "missing_ids_neutral.json",
            context=ServoCalibrationContext(
                robot_mode="4-servo",
                servo_ids=[1, 2, 3, 4],
                tendon_to_servo=[1, 2, 3, 4],
                position_min_offset_ticks=-600,
                position_max_offset_ticks=600,
                default_pretension_current_threshold_ma=220,
                tightening_rotation_by_servo={1: "cw", 2: "cw", 3: "cw", 4: "cw"},
            ),
        ),
        pretension_validation=PretensionValidationService(),
    )
    controller = SystemController(
        tracking_service=tracker_service,
        openrb_client=MockOpenRbClient(),
        servo_service=servo_service,
        settings=settings,
    )
    monkeypatch.setattr(
        controller,
        "_refresh_available_ports_snapshot",
        lambda: [
            SerialPortInfo(device="/dev/ttyACM0", description="OpenRB USB serial"),
            SerialPortInfo(device="/dev/ttyUSB_TRACKER", description="Tracker"),
        ],
    )
    try:
        controller.connect_tracker()
        controller.connect_openrb()
        state = controller.refresh()
        assert state.openrb_status_label == "Degraded"
        assert "Missing: [1, 2, 3, 4]" in state.openrb_truth_summary
        assert state.primary_blocker == "Configured servos missing on DYNAMIXEL bus: [1, 2, 3, 4]."
    finally:
        controller.disconnect_tracker()
        controller.disconnect_openrb()


def test_system_controller_reports_openrb_stage_when_serial_connected_but_bus_not_responding(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    settings = _settings()
    settings.runtime.mock_mode = False
    settings.serial.aurora_port = "/dev/ttyUSB_TRACKER"
    settings.serial.openrb_port = "/dev/ttyACM0"
    tracker_service = _tracking_service(settings, tmp_path)
    controller = SystemController(
        tracking_service=tracker_service,
        openrb_client=MockOpenRbClient(),
        servo_service=_servo_service(tmp_path, dxl_bus=MockDxlBus([])),
        settings=settings,
    )
    monkeypatch.setattr(
        controller,
        "_refresh_available_ports_snapshot",
        lambda: [
            SerialPortInfo(device="/dev/ttyACM0", description="OpenRB USB serial"),
            SerialPortInfo(device="/dev/ttyUSB_TRACKER", description="Tracker"),
        ],
    )
    try:
        controller.connect_tracker()
        controller.connect_openrb()
        state = controller.refresh()
        assert state.openrb_status_label == "Degraded"
        assert "no configured servos responded" in state.openrb_truth_summary.lower()
        assert state.primary_blocker == "Configured servos are not responding on the DYNAMIXEL bus."
    finally:
        controller.disconnect_tracker()
        controller.disconnect_openrb()


def test_system_controller_disconnects_servo_bus_before_openrb_client(tmp_path: Path) -> None:
    order: list[str] = []

    class _ServoServiceStub:
        is_connected = True

        def disconnect(self) -> None:
            order.append("servo")

    class _OpenRbStub(MockOpenRbClient):
        def disconnect(self) -> None:
            order.append("openrb")
            super().disconnect()

    settings = _settings()
    controller = SystemController(
        tracking_service=_tracking_service(settings, tmp_path),
        openrb_client=_OpenRbStub(),
        servo_service=_ServoServiceStub(),
        settings=settings,
    )

    controller.disconnect_openrb()

    assert order == ["servo", "openrb"]


def test_system_controller_saves_runtime_parameters(tmp_path: Path) -> None:
    config_dir = tmp_path / "config"
    config_dir.mkdir()
    (config_dir / "system.yaml").write_text('robot_config: "robot_1servo.yaml"\n', encoding="utf-8")
    (config_dir / "robot_1servo.yaml").write_text(
        "\n".join(
            [
                'mode: "1-servo"',
                "servo_ids: [1]",
                "tendon_to_servo: [1]",
                "tightening_rotation_by_servo: {1: cw}",
            ]
        ),
        encoding="utf-8",
    )
    loader = ConfigLoader(base_dir=config_dir)
    settings = _settings()
    settings.runtime.robot_config = "robot_1servo.yaml"
    settings.robot.mode = "1-servo"
    settings.robot.servo_ids = [1]
    settings.robot.tendon_to_servo = [1]
    settings.robot.tightening_rotation_by_servo = {1: "cw"}
    controller = SystemController(
        tracking_service=_tracking_service(settings, tmp_path),
        openrb_client=MockOpenRbClient(),
        servo_service=_servo_service(tmp_path),
        settings=settings,
        config_loader=loader,
    )

    controller.save_runtime_parameters(
        mock_mode=False,
        robot_config="robot_1servo.yaml",
        openrb_port="/dev/ttyUSB_TEST",
        baudrate=57600,
        poll_rate_hz=20,
        fine_jog_step_ticks=3,
        coarse_jog_step_ticks=15,
        position_min_offset_ticks=-120,
        position_max_offset_ticks=140,
        software_position_margin_ticks=32,
        telemetry_freshness_timeout_s=0.3,
        pretension_threshold_ma=210,
        tightening_direction="ccw",
    )

    saved = loader.load_system_local_overrides()
    assert saved["openrb_port"] == "/dev/ttyUSB_TEST"
    assert saved["baudrate"] == 57600
    assert saved["safety_overrides"]["fine_jog_step_ticks"] == 3
    assert saved["safety_overrides"]["position_min_offset_ticks"] == -120
    assert "servo_ids" not in saved["robot_overrides"]
    assert "tendon_to_servo" not in saved["robot_overrides"]
    assert saved["robot_overrides"]["tightening_rotation_by_servo"]["1"] == "ccw"


def test_system_controller_builds_comprehensive_session_diagnostics_document(tmp_path: Path) -> None:
    settings = _settings()
    session_log = tmp_path / "data" / "logs" / "current" / "operator_gui_test.log"
    session_log.parent.mkdir(parents=True)
    session_log.write_text("first event\nsecond event\n", encoding="utf-8")
    controller = SystemController(
        tracking_service=_tracking_service(settings, tmp_path),
        openrb_client=MockOpenRbClient(),
        servo_service=_servo_service(tmp_path),
        settings=settings,
        session_log_path=str(session_log),
    )

    controller.refresh()
    document = controller.build_session_diagnostics_document()

    assert "System Session Diagnostics" in document
    assert "Session log:" in document
    assert "second event" in document
    assert "Effective config:" in document


def test_system_tab_preserves_selected_ports_between_refreshes() -> None:
    _app()
    controller = _PortSelectionController()
    tab = SystemTab(controller)

    tab.update(controller.state)
    tab.aurora_port_combo.setCurrentIndex(tab.aurora_port_combo.findData("/dev/ttyUSB_TRACKER"))
    tab.openrb_port_combo.setCurrentIndex(tab.openrb_port_combo.findData("/dev/ttyUSB_OPENRB"))

    assert controller.state.aurora_port == "/dev/ttyUSB_TRACKER"
    assert controller.state.openrb_port == "/dev/ttyUSB_OPENRB"

    tab.update(controller.state)

    assert tab._selected_port(tab.aurora_port_combo) == "/dev/ttyUSB_TRACKER"
    assert tab._selected_port(tab.openrb_port_combo) == "/dev/ttyUSB_OPENRB"


def test_system_tab_prefers_custom_port_text_for_editable_combo() -> None:
    _app()
    controller = _PortSelectionController()
    tab = SystemTab(controller)

    tab.update(controller.state)
    tab.aurora_port_combo.setEditText("/dev/custom-tracker")
    tab.openrb_port_combo.setEditText("/dev/custom-openrb")

    assert controller.state.aurora_port == "/dev/custom-tracker"
    assert controller.state.openrb_port == "/dev/custom-openrb"
    assert tab._selected_port(tab.aurora_port_combo) == "/dev/custom-tracker"
    assert tab._selected_port(tab.openrb_port_combo) == "/dev/custom-openrb"


def test_system_tab_wraps_workspace_in_scroll_area() -> None:
    _app()
    controller = _PortSelectionController()
    tab = SystemTab(controller)

    assert isinstance(tab.scroll_area, QScrollArea)
    assert tab.scroll_area.widget() is not None


def test_system_tab_surfaces_operator_bring_up_truth() -> None:
    _app()
    controller = _PortSelectionController()
    controller.state.mode_display = "Mock"
    controller.state.robot_layout_display = "1 Segment"
    controller.state.tracker_status_label = "Connected"
    controller.state.tracker_status_kind = "ready"
    controller.state.openrb_status_label = "Connected"
    controller.state.openrb_status_kind = "ready"
    controller.state.overall_status_label = "Ready"
    controller.state.overall_status_kind = "ready"
    controller.state.primary_blocker = ""
    controller.state.tracker_truth_summary = "Healthy on ndi."
    controller.state.openrb_truth_summary = "OpenRB and the DYNAMIXEL bus are ready."
    controller.state.session_log_summary = "data/logs/current/operator_gui_test.log"
    controller.state.diagnostics_preview = "first event\nsecond event"
    tab = SystemTab(controller)

    tab.update(controller.state)

    assert tab.mode_label.text() == "Mock"
    assert tab.robot_label.text() == "1 Segment"
    assert tab.overall_header_label.text() == "Ready"
    assert tab.blocker_label.isHidden()
    assert "Healthy on ndi." in tab.tracker_status_label.text()
    assert tab.session_log_label.text().endswith("operator_gui_test.log")
    assert "second event" in tab.status_text.toPlainText()


def test_system_tab_shows_blocker_banner_when_blocked() -> None:
    _app()
    controller = _PortSelectionController()
    controller.state.mode_display = "Hardware"
    controller.state.robot_layout_display = "2 Segments"
    controller.state.tracker_status_label = "Degraded"
    controller.state.tracker_status_kind = "warning"
    controller.state.openrb_status_label = "Not Connected"
    controller.state.openrb_status_kind = "blocked"
    controller.state.overall_status_label = "Blocked"
    controller.state.overall_status_kind = "blocked"
    controller.state.primary_blocker = "OpenRB / DYNAMIXEL is not connected."
    tab = SystemTab(controller)

    tab.update(controller.state)

    assert not tab.blocker_label.isHidden()
    assert "OpenRB / DYNAMIXEL is not connected." in tab.blocker_label.text()


def test_system_tab_save_apply_prefers_callback_when_available() -> None:
    _app()
    controller = _PortSelectionController()
    received: list[dict] = []
    tab = SystemTab(controller, apply_runtime_parameters=lambda **kwargs: received.append(dict(kwargs)))

    tab.update(controller.state)
    tab.poll_rate_spin.setValue(24)
    tab.telemetry_freshness_spin.setValue(0.3)
    tab.save_parameters_button.click()

    assert len(received) == 1
    assert received[0]["poll_rate_hz"] == 24
    assert received[0]["telemetry_freshness_timeout_s"] == 0.3
    assert controller.save_calls == []


def test_system_tab_preserves_unsaved_parameter_edits_across_refresh() -> None:
    _app()
    controller = _PortSelectionController()
    tab = SystemTab(controller)

    tab.update(controller.state)
    tab.poll_rate_spin.setValue(24)
    tab.telemetry_freshness_spin.setValue(0.5)
    tab.update(controller.state)

    assert tab.poll_rate_spin.value() == 24
    assert tab.telemetry_freshness_spin.value() == 0.5


def test_system_tab_keeps_hardware_profile_in_advanced_settings() -> None:
    _app()
    controller = _PortSelectionController()
    tab = SystemTab(controller)

    tab.update(controller.state)

    assert tab.settings_advanced_panel.isHidden() is True
    assert tab.robot_config_combo.isHidden() is True

    tab.settings_advanced_toggle.click()

    assert tab.settings_advanced_panel.isHidden() is False
    assert tab.robot_config_combo.isHidden() is False


def _controller_with_dual_segment_options() -> _PortSelectionController:
    controller = _PortSelectionController()
    controller.state.robot_config = "robot_8servo.yaml"
    controller.state.available_robot_configs = ["robot_4servo.yaml", "robot_8servo.yaml"]
    controller.state.operating_mode = "single_segment"
    controller.state.robot_mode = "single_segment"
    controller.state.selected_servo_id = 1
    controller.state.active_segment_key = "segment_a"
    controller.state.active_segment_label = "Spine 1"
    controller.state.active_segment_servo_ids = [1, 2, 3, 4]
    controller.state.active_segment_pairs = {"axis_a": [1, 3], "axis_b": [2, 4]}
    controller.state.expected_servo_ids = [1, 2, 3, 4]
    controller.state.available_segments = [
        {
            "key": "segment_a",
            "label": "Spine 1",
            "servo_ids": [1, 2, 3, 4],
            "pairs": {"axis_a": [1, 3], "axis_b": [2, 4]},
            "display": "Spine 1 (1, 2, 3, 4)",
        },
        {
            "key": "segment_b",
            "label": "Spine 2",
            "servo_ids": [5, 6, 7, 8],
            "pairs": {"axis_a": [5, 7], "axis_b": [6, 8]},
            "display": "Spine 2 (5, 6, 7, 8)",
        },
    ]
    return controller


class _RobotProfileLoader:
    def __init__(self) -> None:
        self._robots = {
            "robot_4servo.yaml": RobotConfig(
                mode="single_segment",
                servo_ids=[1, 2, 3, 4],
                tendon_to_servo=[1, 2, 3, 4],
                segments={
                    "segment_a": RobotSegmentConfig(
                        key="segment_a",
                        label="Spine 1",
                        servo_ids=[1, 2, 3, 4],
                        pairs={"axis_a": [1, 3], "axis_b": [2, 4]},
                    )
                },
            ),
            "robot_8servo.yaml": RobotConfig(
                mode="single_segment",
                servo_ids=[1, 2, 3, 4, 5, 6, 7, 8],
                tendon_to_servo=[1, 2, 3, 4, 5, 6, 7, 8],
                segments={
                    "segment_a": RobotSegmentConfig(
                        key="segment_a",
                        label="Spine 1",
                        servo_ids=[1, 2, 3, 4],
                        pairs={"axis_a": [1, 3], "axis_b": [2, 4]},
                    ),
                    "segment_b": RobotSegmentConfig(
                        key="segment_b",
                        label="Spine 2",
                        servo_ids=[5, 6, 7, 8],
                        pairs={"axis_a": [5, 7], "axis_b": [6, 8]},
                    ),
                },
            ),
        }

    def load_robot_config(self, name: str) -> RobotConfig:
        return self._robots[str(name)]


def test_system_tab_auto_selects_full_profile_for_normal_operator_scope() -> None:
    _app()
    controller = _PortSelectionController()
    controller.config_loader = _RobotProfileLoader()
    controller.state.robot_config = "robot_4servo.yaml"
    controller.state.available_robot_configs = ["robot_4servo.yaml", "robot_8servo.yaml"]
    tab = SystemTab(controller)

    tab.update(controller.state)

    assert tab.robot_config_combo.currentData() == "robot_8servo.yaml"
    assert tab.active_segment_combo.findData("segment_b") >= 0
    assert controller.save_calls == []


def test_system_tab_one_servo_mode_only_shows_selected_servo_scope() -> None:
    _app()
    controller = _controller_with_dual_segment_options()
    tab = SystemTab(controller)

    tab.update(controller.state)
    tab.operating_mode_combo.setCurrentIndex(tab.operating_mode_combo.findData("one_servo"))
    tab.selected_servo_combo.setCurrentIndex(tab.selected_servo_combo.findData(8))

    assert tab.selected_servo_combo.isHidden() is False
    assert tab.active_segment_combo.isHidden() is True
    assert "Expected IDs: [8]" in tab.operating_context_summary_label.text()
    assert controller.save_calls == []


def test_system_tab_single_segment_mode_only_shows_active_segment_scope() -> None:
    _app()
    controller = _controller_with_dual_segment_options()
    tab = SystemTab(controller)

    tab.update(controller.state)
    tab.operating_mode_combo.setCurrentIndex(tab.operating_mode_combo.findData("single_segment"))
    tab.active_segment_combo.setCurrentIndex(tab.active_segment_combo.findData("segment_b"))

    assert tab.selected_servo_combo.isHidden() is True
    assert tab.active_segment_combo.isHidden() is False
    summary = tab.operating_context_summary_label.text()
    assert "Spine 2: [5, 6, 7, 8]" in summary
    assert "5-7, 6-8" in summary
    assert controller.save_calls == []


def test_system_tab_single_segment_exposes_segment_b_on_full_hardware_profile() -> None:
    _app()
    controller = _controller_with_dual_segment_options()
    tab = SystemTab(controller)

    tab.update(controller.state)
    tab.operating_mode_combo.setCurrentIndex(tab.operating_mode_combo.findData("single_segment"))

    assert tab.active_segment_combo.findData("segment_a") >= 0
    assert tab.active_segment_combo.findData("segment_b") >= 0


def test_system_tab_dual_segment_mode_hides_specific_selectors_and_summarizes_all_ids() -> None:
    _app()
    controller = _controller_with_dual_segment_options()
    tab = SystemTab(controller)

    tab.update(controller.state)
    tab.operating_mode_combo.setCurrentIndex(tab.operating_mode_combo.findData("dual_segment"))

    assert tab.selected_servo_combo.isHidden() is True
    assert tab.active_segment_combo.isHidden() is True
    summary = tab.operating_context_summary_label.text()
    assert "Expected IDs: [1, 2, 3, 4, 5, 6, 7, 8]" in summary
    assert "Spine 1: [1, 2, 3, 4]" in summary
    assert "Spine 2: [5, 6, 7, 8]" in summary
    assert tab.bottom_segment_combo.isHidden() is False
    assert tab.top_segment_combo.isHidden() is False
    assert tab.assembly_confirm_check.isHidden() is False


def test_system_tab_dual_segment_bottom_top_confirmation_is_saved() -> None:
    _app()
    controller = _controller_with_dual_segment_options()
    received: list[dict] = []
    tab = SystemTab(controller, apply_runtime_parameters=lambda **kwargs: received.append(dict(kwargs)))

    tab.update(controller.state)
    tab.operating_mode_combo.setCurrentIndex(tab.operating_mode_combo.findData("dual_segment"))
    tab.bottom_segment_combo.setCurrentIndex(tab.bottom_segment_combo.findData("segment_b"))
    tab.top_segment_combo.setCurrentIndex(tab.top_segment_combo.findData("segment_a"))
    tab.assembly_confirm_check.setChecked(True)
    tab.save_parameters_button.click()

    assert received
    assert received[-1]["bottom_segment_key"] == "segment_b"
    assert received[-1]["top_segment_key"] == "segment_a"
    assert received[-1]["physical_assembly_confirmed_by_operator"] is True


def test_system_tab_parallel_single_mode_hides_specific_selectors_and_summarizes_mirroring() -> None:
    _app()
    controller = _controller_with_dual_segment_options()
    tab = SystemTab(controller)

    tab.update(controller.state)
    tab.operating_mode_combo.setCurrentIndex(tab.operating_mode_combo.findData("parallel_single"))

    assert tab.selected_servo_combo.isHidden() is True
    assert tab.active_segment_combo.isHidden() is True
    summary = tab.operating_context_summary_label.text()
    assert "Expected IDs: [1, 2, 3, 4, 5, 6, 7, 8]" in summary
    assert "Mirror mapping: 1->5, 2->6, 3->7, 4->8" in summary
    assert "not full two-segment kinematics" in summary


class _TwoSegmentDatasetPageController:
    def __init__(self, tmp_path: Path) -> None:
        self.project_root = tmp_path
        self._payload = {
            "schedule_type": "workspace_coverage",
            "dry_run": False,
            "allow_servo_only_test_run": False,
            "run_trust_mode": "thesis_trusted",
            "max_segment_displacement_cm": 0.25,
            "requested_tool_roles": {},
        }
        segments = {
            "segment_a": RobotSegmentConfig(
                key="segment_a",
                label="Segment A",
                servo_ids=[1, 2, 3, 4],
                pairs={"axis_a": [1, 3], "axis_b": [2, 4]},
            ),
            "segment_b": RobotSegmentConfig(
                key="segment_b",
                label="Segment B",
                servo_ids=[5, 6, 7, 8],
                pairs={"axis_a": [5, 7], "axis_b": [6, 8]},
            ),
        }
        self.settings = Settings(
            runtime=RuntimeConfig(mock_mode=True, robot_config="robot_8servo.yaml"),
            robot=RobotConfig(
                mode="dual_segment",
                servo_ids=[1, 2, 3, 4, 5, 6, 7, 8],
                tendon_to_servo=[1, 2, 3, 4, 5, 6, 7, 8],
                segments=segments,
            ),
            serial=SerialConfig(aurora_port="/dev/mock-aurora", openrb_port="/dev/mock-openrb", baudrate=1_000_000),
            safety=SafetyConfig(),
            experiment=ExperimentConfig(output_dir=str(tmp_path / "data" / "experiments")),
            calibration=CalibrationConfig(neutral_setpoints_path=str(tmp_path / "neutral_setpoints.json")),
            registration=RegistrationWorkflowConfig(
                two_segment_tracking_roles={
                    "distal_tip": TwoSegmentTrackingRoleConfig(
                        role_name="distal_tip",
                        tool_id="0A",
                        required_for_two_segment_model_training=True,
                        enabled=True,
                    ),
                    "intermediate_segment": TwoSegmentTrackingRoleConfig(
                        role_name="intermediate_segment",
                        tool_id="",
                        required_for_two_segment_model_training=False,
                        enabled=False,
                    ),
                }
            ),
        )
        self.tracking_service = SimpleNamespace(get_snapshot=lambda: SimpleNamespace(tools={"0A": object(), "0C": object()}))

    def config_payload(self) -> dict:
        return dict(self._payload)

    def set_config_value(self, key: str, value) -> None:
        self._payload[str(key)] = value

    def get_config_value(self, key: str, default=None):
        return self._payload.get(str(key), default)

    def stop(self) -> None:
        pass

    def refresh(self):
        return None

    def set_output_root(self, _value: str) -> None:
        pass

    def set_operator_notes(self, _value: str) -> None:
        pass


def test_two_segment_collect_pose_page_selects_tracker_roles_and_range(tmp_path: Path) -> None:
    _app()
    controller = _TwoSegmentDatasetPageController(tmp_path)
    page = experiment_pages_module.TwoSegmentCollectPoseDatasetPage(
        controller,
        "two_segment_collect_pose_command_dataset",
    )

    page._sync_parameters_from_state(SimpleNamespace())
    page.distal_tool_combo.setEditText("0A")
    page.intermediate_tool_combo.setEditText("0C")
    page.range_preset_combo.setCurrentIndex(page.range_preset_combo.findData(0.75))
    page.continue_valid_check.setChecked(True)
    page.target_valid_spin.setValue(5000)
    page.assembly_confirm_check.setChecked(True)

    assert controller.config_payload()["requested_tool_roles"]["0A"] == "distal_tip"
    assert controller.config_payload()["requested_tool_roles"]["0C"] == "intermediate_segment"
    assert controller.config_payload()["max_segment_displacement_cm"] == 0.75
    assert controller.config_payload()["continue_until_valid_samples"] is True
    assert controller.config_payload()["target_valid_sample_count"] == 5000
    assert controller.config_payload()["physical_assembly_confirmed_by_operator"] is True


def test_system_tab_warns_when_legacy_profile_cannot_support_parallel_single() -> None:
    _app()
    controller = _PortSelectionController()
    controller.state.operating_mode = "single_segment"
    controller.state.available_segments = [
        {
            "key": "segment_a",
            "label": "Spine 1",
            "servo_ids": [1, 2, 3, 4],
            "pairs": {"axis_a": [1, 3], "axis_b": [2, 4]},
            "display": "Spine 1 (1, 2, 3, 4)",
        }
    ]
    tab = SystemTab(controller)

    tab.update(controller.state)
    tab.operating_mode_combo.setCurrentIndex(tab.operating_mode_combo.findData("parallel_single"))

    summary = tab.operating_context_summary_label.text()
    assert "requires an 8-servo hardware profile" in summary


def test_system_tab_refresh_does_not_overwrite_dirty_operating_mode_edit() -> None:
    _app()
    controller = _controller_with_dual_segment_options()
    tab = SystemTab(controller)

    tab.update(controller.state)
    tab.operating_mode_combo.setCurrentIndex(tab.operating_mode_combo.findData("parallel_single"))
    controller.state.operating_mode = "single_segment"
    tab.update(controller.state)

    assert tab.operating_mode_combo.currentData() == "parallel_single"
    assert controller.save_calls == []


def test_system_tab_save_apply_is_required_to_persist_operating_mode_edits() -> None:
    _app()
    controller = _controller_with_dual_segment_options()
    tab = SystemTab(controller)

    tab.update(controller.state)
    tab.operating_mode_combo.setCurrentIndex(tab.operating_mode_combo.findData("dual_segment"))

    assert controller.save_calls == []

    tab.save_parameters_button.click()

    assert len(controller.save_calls) == 1
    assert controller.save_calls[0]["operating_mode"] == "dual_segment"


def test_system_tab_opening_dropdown_does_not_change_selection() -> None:
    _app()
    controller = _controller_with_dual_segment_options()
    tab = SystemTab(controller)

    tab.update(controller.state)
    initial = tab.operating_mode_combo.currentData()
    tab.operating_mode_combo.showPopup()
    QTest.qWait(10)
    tab.operating_mode_combo.hidePopup()

    assert tab.operating_mode_combo.currentData() == initial
    assert controller.save_calls == []


def test_system_tab_refresh_does_not_overwrite_open_startup_combo() -> None:
    _app()
    controller = _controller_with_dual_segment_options()
    tab = SystemTab(controller)

    tab.update(controller.state)
    initial = tab.operating_mode_combo.currentData()
    tab.operating_mode_combo.showPopup()
    controller.state.operating_mode = "dual_segment"
    tab.update(controller.state)
    tab.operating_mode_combo.hidePopup()

    assert tab.operating_mode_combo.currentData() == initial
    assert controller.save_calls == []


def test_system_tab_preserves_scrolled_diagnostics_position_on_update() -> None:
    _app()
    controller = _PortSelectionController()
    controller.state.diagnostics_preview = "\n".join(f"line {index}" for index in range(160))
    tab = SystemTab(controller)
    tab.resize(900, 700)
    tab.show()

    tab.update(controller.state)
    QTest.qWait(20)
    scroll_bar = tab.status_text.verticalScrollBar()
    scroll_bar.setValue(max(1, scroll_bar.maximum() // 2))
    previous_value = scroll_bar.value()

    controller.state.diagnostics_preview += "\nline 161"
    tab.update(controller.state)
    QTest.qWait(20)

    assert tab.status_text.verticalScrollBar().value() >= previous_value - 2


def test_system_tab_copy_session_diagnostics_prefers_controller_document_builder() -> None:
    _app()
    controller = _PortSelectionController()
    controller.build_session_diagnostics_document = lambda: "full diagnostics document"
    tab = SystemTab(controller)

    tab.copy_diagnostics_button.click()

    assert QApplication.clipboard().text() == "full diagnostics document"


def test_experiment_results_widget_preserves_summary_scroll_on_update() -> None:
    _app()
    widget = ExperimentResultsWidget()
    widget.resize(900, 700)
    widget.show()

    widget.set_model(
        VisualizationModel(
            summary_lines=[f"line {index}" for index in range(120)],
            charts=[
                ChartModel(
                    kind="bar",
                    title="Spread",
                    x_title="Target",
                    y_title="mm",
                    categories=["A", "B"],
                    values=[0.4, 0.6],
                )
            ],
        )
    )
    QTest.qWait(20)
    scroll_bar = widget.summary_text.verticalScrollBar()
    scroll_bar.setValue(max(1, scroll_bar.maximum() // 2))
    previous_value = scroll_bar.value()

    widget.set_model(
        VisualizationModel(
            summary_lines=[f"line {index}" for index in range(121)],
            charts=[
                ChartModel(
                    kind="bar",
                    title="Spread",
                    x_title="Target",
                    y_title="mm",
                    categories=["A", "B"],
                    values=[0.5, 0.7],
                )
            ],
        )
    )
    QTest.qWait(20)

    assert widget.summary_text.verticalScrollBar().value() >= previous_value - 2


def test_app_window_skips_hidden_tracking_and_registration_scene_updates(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _app()
    window = AppWindow(_app_context(tmp_path))
    try:
        window._refresh_timer.stop()
        counts = {"tracking_scene": 0, "registration_scene": 0}

        def _count_tracking_scene(*args, **kwargs):
            counts["tracking_scene"] += 1

        def _count_registration_scene(*args, **kwargs):
            counts["registration_scene"] += 1

        monkeypatch.setattr(window.tracking_tab.plot_widget, "set_tracking_state", _count_tracking_scene)
        monkeypatch.setattr(window.registration_tab.plot_widget, "set_data", _count_registration_scene)

        window.tab_widget.setCurrentWidget(window.servos_tab)
        counts = {"tracking_scene": 0, "registration_scene": 0}
        window.refresh()
        assert counts == {"tracking_scene": 0, "registration_scene": 0}

        window.tab_widget.setCurrentWidget(window.tracking_tab)
        counts = {"tracking_scene": 0, "registration_scene": 0}
        window.refresh()
        assert counts == {"tracking_scene": 1, "registration_scene": 0}

        window.tab_widget.setCurrentWidget(window.registration_tab)
        counts = {"tracking_scene": 0, "registration_scene": 0}
        window.refresh()
        assert counts == {"tracking_scene": 0, "registration_scene": 1}
    finally:
        window.shutdown()


def test_app_window_keeps_hidden_tracking_and_registration_scenes_idle_across_repeated_refreshes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _app()
    window = AppWindow(_app_context(tmp_path))
    try:
        window._refresh_timer.stop()
        counts = {"tracking_scene": 0, "registration_scene": 0}

        def _count_tracking_scene(*args, **kwargs):
            counts["tracking_scene"] += 1

        def _count_registration_scene(*args, **kwargs):
            counts["registration_scene"] += 1

        monkeypatch.setattr(window.tracking_tab.plot_widget, "set_tracking_state", _count_tracking_scene)
        monkeypatch.setattr(window.registration_tab.plot_widget, "set_data", _count_registration_scene)

        window.tab_widget.setCurrentWidget(window.experiment_tab)
        window.refresh()
        window.refresh()
        assert counts == {"tracking_scene": 0, "registration_scene": 0}

        window.tab_widget.setCurrentWidget(window.system_tab)
        window.refresh()
        window.refresh()
        assert counts == {"tracking_scene": 0, "registration_scene": 0}
    finally:
        window.shutdown()


def test_servos_controller_captures_neutral_and_applies_displacement(tmp_path: Path) -> None:
    controller = ServosController(_servo_service(tmp_path), _settings())
    controller.servo_service.connect("/dev/mock-openrb", 115200)

    neutral = controller.capture_neutral_setpoints()
    controller.save_neutral_setpoints()
    controller.set_tendon_displacements([0.0, 0.1, -0.1, 0.0])
    controller.apply_displacement()

    assert neutral
    assert "Commanded" in controller.state.status_message
    assert controller.state.telemetry[2]["position"] is not None
    assert controller.state.calibration_exists is True
    assert controller.state.calibration_compatible is True
    assert controller.state.calibration_rows[0]["bounds"] != "missing"


def test_servos_controller_uses_manual_pretension_as_experiment_reference(tmp_path: Path) -> None:
    controller = ServosController(_servo_service(tmp_path), _settings())
    controller.servo_service.connect("/dev/mock-openrb", 115200)

    controller.capture_neutral_setpoints()
    for servo_id in [1, 2, 3, 4]:
        controller.servo_service.dxl_bus._state[servo_id].present_position = 3010 + servo_id
        controller.servo_service.dxl_bus._state[servo_id].present_current_ma = 220 + servo_id
    controller.servo_service.capture_manual_pretension_state(note="manual experiment reference")
    controller.servo_service.accept_manual_pretension_state()
    controller.set_tendon_displacements([0.0, 0.0, 0.0, 0.0])
    controller.apply_displacement()

    assert "accepted manual pretension/startup reference positions" in controller.state.single_segment_reference_summary
    assert controller.state.telemetry[1]["position"] == 3011
    assert "3011" in controller.state.last_displacement_summary or "3011" in "\n".join(controller.state.last_displacement_debug_lines)


def test_servos_controller_surfaces_single_segment_motion_diagnostics(tmp_path: Path) -> None:
    controller = ServosController(_servo_service(tmp_path), _settings())
    controller.servo_service.connect("/dev/mock-openrb", 115200)

    controller.capture_neutral_setpoints()
    controller.refresh()

    assert "experiment motion" in controller.state.single_segment_motion_config_summary
    assert "Position Control" in controller.state.single_segment_motion_config_summary
    assert "raw 0..4095" in controller.state.single_segment_enforced_bounds_summary
    assert "saved neutral reference positions" in controller.state.single_segment_reference_summary
    assert "Display-only diagnostic pair travel" in controller.state.single_segment_characterization_summary
    assert "artifact only" in controller.state.calibration_rows[0]["bounds"]


def test_servos_controller_supports_fine_and_coarse_jog_steps(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    service = _servo_service(tmp_path)
    service.connect("/dev/mock-openrb", 115200)
    controller = ServosController(service, _settings())
    seen: list[tuple[int, str]] = []

    class _FakeDirectionalResult:
        def __init__(self, servo_id: int, action: str) -> None:
            self.servo_id = servo_id
            self.command_direction = "tighten" if action.startswith("tighten") else "loosen"
            self.message = "ok"
            self.success = True
            self.blocked = False
            self.delta_ticks = 25 if action == "loosen_coarse" else -5
            self.goal_tick = 2048
            self.current_position_tick = 2053
            self.unclamped_goal_tick = 2048
            self.safe_min_tick = 1948
            self.safe_max_tick = 2148
            self.clamped = False

    def _fake_directional(*, servo_id: int, action: str):
        seen.append((servo_id, action))
        return _FakeDirectionalResult(servo_id, action)

    monkeypatch.setattr(service, "jog_servo_action", _fake_directional)

    controller.fine_jog(1, 1)
    controller.coarse_jog(1, -1)

    assert seen == [(1, "tighten_fine"), (1, "loosen_coarse")]


def test_servos_controller_and_tab_support_manual_pretension_capture_and_accept(tmp_path: Path) -> None:
    _app()
    service = _servo_service(tmp_path)
    service.connect("/dev/mock-openrb", 115200)
    for servo_id in [1, 2, 3, 4]:
        service.set_servo_torque_enabled(servo_id, True)
    controller = ServosController(service, _settings())
    tab = ServosTab(controller)

    controller.capture_neutral_setpoints()
    controller.capture_manual_pretension("bench startup state")
    tab.update(controller.state)

    assert controller.state.manual_pretension_can_accept is True
    assert "Pending manual pretension capture" in tab.manual_pretension_summary_label.text()
    assert "bench startup state" in tab.manual_pretension_note_label.text()

    controller.accept_manual_pretension()
    tab.update(controller.state)

    assert controller.state.pretension_source_type == "manual"
    assert "Accepted manual pretension" in tab.manual_pretension_summary_label.text()


def test_servos_controller_blocks_displacement_when_calibration_is_incompatible(tmp_path: Path) -> None:
    compatible_service = _servo_service(tmp_path)
    compatible_service.connect("/dev/mock-openrb", 115200)
    compatible_service.save_neutral_setpoints({1: 2048, 2: 2048, 3: 2048, 4: 2048})

    mismatched_service = ServoService(
        dxl_bus=MockDxlBus([1, 2, 3, 4]),
        mapper=TendonDisplacementMapper(spool_diameter_cm=1.2),
        safety_guard=SafetyGuard(min_offset_ticks=-600, max_offset_ticks=600, max_current_ma=850),
        neutral_calibration=NeutralCalibrationService(
            path=tmp_path / "neutral.json",
            context=ServoCalibrationContext(
                robot_mode="8-servo",
                servo_ids=[1, 2, 3, 4],
                tendon_to_servo=[1, 2, 3, 4],
                position_min_offset_ticks=-600,
                position_max_offset_ticks=600,
                default_pretension_current_threshold_ma=850,
            ),
        ),
        pretension_validation=PretensionValidationService(),
    )
    mismatched_service.connect("/dev/mock-openrb", 115200)
    settings = _settings()
    settings.robot.mode = "8-servo"
    controller = ServosController(mismatched_service, settings)
    controller.load_neutral_setpoints()
    controller.set_tendon_displacements([0.0, 0.1, -0.1, 0.0])

    with pytest.raises(RuntimeError, match="does not match the current robot configuration"):
        controller.apply_displacement()


def test_registration_controller_guides_capture_and_save(tmp_path: Path) -> None:
    settings = _settings()
    tracking_service = _tracking_service(settings, tmp_path)
    registration_service = _registration_service(settings, tmp_path, tracking_service)
    tracking_service.start()
    try:
        controller = RegistrationController(
            registration_service=registration_service,
            registration_config=settings.registration,
        )

        controller.begin_session()
        controller.capture_current_label_sample()
        controller.complete_current_label()
        controller.capture_current_label_sample()
        controller.complete_current_label()
        controller.capture_current_label_sample()
        controller.complete_current_label()
        controller.capture_current_label_sample()
        controller.complete_current_label()
        controller.solve_session()
        result = controller.save_registration(confirm_overwrite=True)
    finally:
        tracking_service.stop()

    assert result.output_path.exists()
    assert controller.state.fre_mm is not None
    assert controller.state.current_label is None
    payload = json.loads(result.output_path.read_text(encoding="utf-8"))
    assert payload["landmark_labels"] == ["L1", "L2", "L3", "L4"]
    assert set(payload["raw_captured_landmarks_robot_xyz"]) == {"L1", "L2", "L3", "L4"}
    assert payload["config_used"]["registration_mode"] == "simple"


def test_registration_controller_supports_selectable_four_point_session(tmp_path: Path) -> None:
    settings = _settings()
    tracking_service = _tracking_service(settings, tmp_path)
    registration_service = _registration_service(
        settings,
        tmp_path,
        tracking_service,
        nominal_points={
            "L1": [0.0, 0.0, 0.0],
            "L2": [10.0, 0.0, 0.0],
            "L3": [0.0, 10.0, 0.0],
            "L4": [0.0, 0.0, 10.0],
            "L5": [10.0, 10.0, 0.0],
            "L6": [10.0, 0.0, 10.0],
        },
    )
    controller = RegistrationController(
        registration_service=registration_service,
        registration_config=settings.registration,
    )

    controller.set_selected_model_point(1, "L5")
    controller.set_selected_model_point(2, "L6")
    controller.set_selected_model_point(3, "L3")
    controller.set_selected_model_point(0, "L2")
    controller.begin_session()

    assert controller.state.landmark_labels == ["L2", "L5", "L6", "L3"]
    assert controller.state.selected_model_labels == ["L2", "L5", "L6", "L3"]
    assert controller.state.current_label == "L2"


def test_registration_controller_prevents_duplicate_point_selection(tmp_path: Path) -> None:
    settings = _settings()
    tracking_service = _tracking_service(settings, tmp_path)
    registration_service = _registration_service(
        settings,
        tmp_path,
        tracking_service,
        nominal_points={
            "L1": [0.0, 0.0, 0.0],
            "L2": [10.0, 0.0, 0.0],
            "L3": [0.0, 10.0, 0.0],
            "L4": [0.0, 0.0, 10.0],
            "L5": [10.0, 10.0, 0.0],
        },
    )
    controller = RegistrationController(
        registration_service=registration_service,
        registration_config=settings.registration,
    )

    with pytest.raises(RuntimeError, match="unique model point"):
        controller.set_selected_model_point(1, controller.state.selected_model_labels[0])


def test_registration_controller_toggle_selection_limits_to_four_unique_points(tmp_path: Path) -> None:
    settings = _settings()
    settings.registration.candidate_landmarks = [
        RegistrationLandmarkConfig(id="L1", xyz_mm=[0.0, 0.0, 0.0]),
        RegistrationLandmarkConfig(id="L2", xyz_mm=[10.0, 0.0, 0.0]),
        RegistrationLandmarkConfig(id="L3", xyz_mm=[0.0, 10.0, 0.0]),
        RegistrationLandmarkConfig(id="L4", xyz_mm=[10.0, 10.0, 0.0]),
        RegistrationLandmarkConfig(id="L5", xyz_mm=[20.0, 10.0, 5.0]),
    ]
    settings.registration.landmark_labels = ["L1", "L2", "L3", "L4"]
    tracking_service = _tracking_service(settings, tmp_path)
    registration_service = _registration_service(
        settings,
        tmp_path,
        tracking_service,
        nominal_points={
            "L1": [0.0, 0.0, 0.0],
            "L2": [10.0, 0.0, 0.0],
            "L3": [0.0, 10.0, 0.0],
            "L4": [10.0, 10.0, 0.0],
            "L5": [20.0, 10.0, 5.0],
        },
    )
    controller = RegistrationController(
        registration_service=registration_service,
        registration_config=settings.registration,
    )

    controller.toggle_selected_model_point("L2")
    assert controller.state.selected_model_labels == ["L1", "L3", "L4"]

    controller.toggle_selected_model_point("L5")
    assert controller.state.selected_model_labels == ["L1", "L3", "L4", "L5"]

    with pytest.raises(RuntimeError, match="Only four model points"):
        controller.toggle_selected_model_point("L2")


def test_registration_controller_blocks_solve_until_selected_points_are_complete(tmp_path: Path) -> None:
    settings = _settings()
    tracking_service = _tracking_service(settings, tmp_path)
    registration_service = _registration_service(settings, tmp_path, tracking_service)
    tracking_service.start()
    try:
        controller = RegistrationController(
            registration_service=registration_service,
            registration_config=settings.registration,
        )
        controller.begin_session()
        controller.capture_current_label_sample()
        controller.complete_current_label()

        assert controller.is_ready_to_solve() is False
        with pytest.raises(RuntimeError, match="not ready to solve"):
            controller.solve_session()
    finally:
        tracking_service.stop()


def test_registration_controller_requires_overwrite_confirmation(tmp_path: Path) -> None:
    settings = _settings()
    tracking_service = _tracking_service(settings, tmp_path)
    registration_service = _registration_service(settings, tmp_path, tracking_service)
    latest = registration_service.repository.root_dir / "latest_registration.json"
    latest.write_text("{}", encoding="utf-8")
    tracking_service.start()
    try:
        controller = RegistrationController(
            registration_service=registration_service,
            registration_config=settings.registration,
        )
        controller.begin_session()
        for _index in range(4):
            controller.capture_current_label_sample()
            controller.complete_current_label()
        controller.solve_session()
        try:
            controller.save_registration()
        except RuntimeError as exc:
            assert "overwrite confirmation" in str(exc)
        else:
            raise AssertionError("Expected overwrite confirmation error.")
    finally:
        tracking_service.stop()


def test_registration_controller_load_latest_populates_saved_result_state(tmp_path: Path) -> None:
    settings = _settings()
    tracking_service = _tracking_service(settings, tmp_path)
    registration_service = _registration_service(settings, tmp_path, tracking_service)
    tracking_service.start()
    try:
        controller = RegistrationController(
            registration_service=registration_service,
            registration_config=settings.registration,
        )
        controller.begin_session()
        for _index in range(4):
            controller.capture_current_label_sample()
            controller.complete_current_label()
        controller.solve_session()
        saved = controller.save_registration(confirm_overwrite=True)
        assert saved.output_path.exists()

        reloaded = RegistrationController(
            registration_service=registration_service,
            registration_config=settings.registration,
        )
        reloaded.load_latest_result()
    finally:
        tracking_service.stop()

    assert reloaded.state.fre_mm is not None
    assert reloaded.state.result_status == "Accepted"
    assert reloaded.state.last_result_path is not None
    assert reloaded.state.captured_counts == {"L1": 1, "L2": 1, "L3": 1, "L4": 1}
    assert reloaded.state.selected_model_labels == ["L1", "L2", "L3", "L4"]
    assert reloaded.state.trust_state == "trusted"
    assert "FRE" in reloaded.state.trust_message
    assert reloaded.state.comparison_message.startswith("This is the first saved registration")


def test_experiment_workspace_selection_binds_example_config(tmp_path: Path) -> None:
    controller = _experiment_controller(tmp_path)

    empty_state = controller.refresh()
    assert empty_state.selected_experiment == ""

    controller.select_experiment("command_schedule_validation")
    state = controller.refresh()

    assert state.selected_experiment == "command_schedule_validation"
    assert state.experiment_title == "Command Schedule Validation"
    assert "schedule:" in state.config_text


def test_experiment_workspace_hides_operational_pivot_workflow_from_selector(tmp_path: Path) -> None:
    controller = _experiment_controller(tmp_path)

    option_names = [option.name for option in controller.refresh().experiment_options]

    assert "pivot_calibration" not in option_names
    assert "repeatability_dataset" not in option_names
    assert "command_schedule_validation" not in option_names
    assert "replay_runner" not in option_names
    assert "single_segment_repeatability" in option_names
    assert "registration_validation" in option_names
    assert "pivot_validation" in option_names
    assert "tracker_timing_validation" in option_names
    assert "servo_tracker_sync_validation" in option_names
    assert "pretension_validation" in option_names
    assert "collect_pose_command_dataset" in option_names


def test_experiment_workspace_filters_options_by_operating_mode(tmp_path: Path) -> None:
    controller = _experiment_controller(tmp_path)

    single_options = [option.name for option in controller.refresh().experiment_options]
    assert "pretension_validation" in single_options
    assert "single_segment_repeatability" in single_options
    assert "collect_pose_command_dataset" in single_options
    assert "penprobe_chasing_demo" in single_options
    assert "two_segment_startup_validation" not in single_options
    assert "two_segment_collect_pose_command_dataset" not in single_options
    assert "single_segment workflows" in controller.state.experiment_filter_summary

    controller.settings.robot.mode = "dual_segment"
    dual_options = [option.name for option in controller.refresh().experiment_options]
    assert "two_segment_startup_validation" in dual_options
    assert "two_segment_collect_pose_command_dataset" in dual_options
    assert "pretension_validation" not in dual_options
    assert "single_segment_repeatability" not in dual_options
    assert "penprobe_chasing_demo" not in dual_options

    controller.settings.robot.mode = "parallel_single"
    parallel_options = [option.name for option in controller.refresh().experiment_options]
    assert "collect_pose_command_dataset" in parallel_options
    assert "pretension_validation" not in parallel_options
    assert "penprobe_chasing_demo" not in parallel_options


def test_experiment_workspace_clears_hidden_selection_after_mode_change(tmp_path: Path) -> None:
    controller = _experiment_controller(tmp_path)
    controller.select_experiment("pretension_validation")
    assert controller.refresh().selected_experiment == "pretension_validation"

    controller.settings.robot.mode = "dual_segment"
    state = controller.refresh()

    assert state.selected_experiment == ""
    assert "hidden for operating_mode=dual_segment" in state.status_message


def test_experiment_workspace_blocks_single_segment_repeatability_in_mock_mode(tmp_path: Path) -> None:
    controller = _experiment_controller(tmp_path)
    controller.select_experiment("single_segment_repeatability")

    state = controller.refresh()

    assert state.preflight_report.overall_status == "blocked"
    assert any("Disable mock mode" in message for message in state.preflight_report.blocking_messages)


def test_experiment_workspace_loads_single_segment_repeatability_defaults(tmp_path: Path) -> None:
    controller = _experiment_controller(tmp_path)
    controller.select_experiment("single_segment_repeatability")

    state = controller.refresh()

    assert state.selected_experiment == "single_segment_repeatability"
    assert "baseline_run_path" in state.config_text
    assert "min_repeat_captures_per_target" in state.config_text
    assert "max_rejected_capture_fraction" in state.config_text


def test_experiment_workspace_loads_motor_babble_page_and_summary(tmp_path: Path) -> None:
    _app()
    controller = _experiment_controller(tmp_path)
    tab = ExperimentTab(controller)
    controller.select_experiment("collect_pose_command_dataset")

    state = controller.refresh()
    tab.update(state)
    page = tab._page_for("collect_pose_command_dataset")

    assert state.selected_experiment == "collect_pose_command_dataset"
    assert state.experiment_title == "Random Data Collection"
    assert page.dataset_mode_combo.count() == 4
    assert page.dataset_mode_combo.currentData() == "workspace_coverage"
    assert page.run_button.text() == "Run Random Data Collection"
    assert page.telemetry_retry_count_spin.value() == 2
    assert page.allow_recovered_packet_errors_check.isChecked() is True
    assert page.open_training_button.text() == "Open ANN Training Popout"
    assert page.collection_summary_widget._pairs_signature is not None
    assert page.viewer_3d is None


def test_experiment_tab_skips_unchanged_penprobe_full_page_updates(tmp_path: Path) -> None:
    _app()
    controller = _experiment_controller(tmp_path)
    tab = ExperimentTab(controller)
    controller.select_experiment("penprobe_chasing_demo")

    state = controller.refresh()
    tab.update(state)
    page = tab._page_for("penprobe_chasing_demo")
    original_set_state = page.set_state
    calls = {"count": 0}

    def _counting_set_state(next_state):
        calls["count"] += 1
        return original_set_state(next_state)

    page.set_state = _counting_set_state

    assert tab.update(state) is False
    assert calls["count"] == 0


def test_experiment_workspace_loads_registration_validation_page(tmp_path: Path) -> None:
    _app()
    controller = _experiment_controller(tmp_path)
    tab = ExperimentTab(controller)
    controller.select_experiment("registration_validation")

    state = controller.refresh()
    tab.update(state)
    page = tab._page_for("registration_validation")

    assert state.selected_experiment == "registration_validation"
    assert state.experiment_title == "Registration Validation"
    assert page.run_button.text() == "Run Registration Validation"
    assert page.run_table.columnCount() == 6


def test_registration_validation_page_deferred_loads_candidates_without_blocking_ui(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _app()
    controller = _experiment_controller(tmp_path)
    tab = ExperimentTab(controller)
    release = threading.Event()

    def _slow_candidates(_project_root):
        release.wait(timeout=1.0)
        return []

    monkeypatch.setattr(experiment_pages_module, "list_registration_validation_candidates", _slow_candidates)
    controller.select_experiment("registration_validation")

    state = controller.refresh()
    started = time.monotonic()
    tab.update(state)
    elapsed = time.monotonic() - started
    page = tab._page_for("registration_validation")

    assert elapsed < 0.1
    assert page.loading_label.isVisible()
    assert "Loading saved runs" in page.loading_label.text()
    assert page.run_table.isEnabled() is False

    release.set()
    QTest.qWait(50)


def test_validation_page_shutdown_drops_late_candidate_results(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _app()
    controller = _experiment_controller(tmp_path)
    tab = ExperimentTab(controller)
    release = threading.Event()

    def _slow_candidates(_project_root):
        release.wait(timeout=1.0)
        return [
            ValidationRunCandidate(
                path="data/registrations/late.json",
                timestamp_utc="2026-01-01T00:00:00+00:00",
                label="late.json",
            )
        ]

    monkeypatch.setattr(experiment_pages_module, "list_registration_validation_candidates", _slow_candidates)
    controller.select_experiment("registration_validation")
    tab.update(controller.refresh())
    page = tab._page_for("registration_validation")
    assert page._candidate_loading is True

    page.shutdown()
    release.set()
    QTest.qWait(80)

    assert page._candidate_loading is False
    assert page._candidate_cache == []


def test_registration_validation_page_does_not_recurse_during_table_rebuild(tmp_path: Path) -> None:
    _app()
    controller = _experiment_controller(tmp_path)
    tab = ExperimentTab(controller)
    controller.select_experiment("registration_validation")
    page = tab._page_for("registration_validation")
    calls: list[tuple[str, object]] = []
    controller.set_config_value = lambda key, value: calls.append((key, value))
    page._candidate_load_generation = 1

    page._apply_loaded_candidates(
        1,
        [
            ValidationRunCandidate(
                path="data/registrations/registration_a.json",
                timestamp_utc="2026-01-01T00:00:00+00:00",
                label="registration_a.json",
            )
        ],
        None,
    )

    assert calls == []


def test_registration_validation_page_caches_empty_discovery_results(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _app()
    controller = _experiment_controller(tmp_path)
    tab = ExperimentTab(controller)
    calls = {"count": 0}

    def _empty_candidates(_project_root):
        calls["count"] += 1
        return []

    monkeypatch.setattr(experiment_pages_module, "list_registration_validation_candidates", _empty_candidates)
    controller.select_experiment("registration_validation")
    state = controller.refresh()
    tab.update(state)
    page = tab._page_for("registration_validation")
    QTest.qWait(50)

    tab.update(state)

    assert calls["count"] == 1
    assert page.loading_label.text() == "No saved runs found yet."


def test_registration_validation_page_skips_duplicate_concurrent_load_requests(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _app()
    controller = _experiment_controller(tmp_path)
    tab = ExperimentTab(controller)
    release = threading.Event()
    calls = {"count": 0}

    def _slow_candidates(_project_root):
        calls["count"] += 1
        release.wait(timeout=1.0)
        return []

    monkeypatch.setattr(experiment_pages_module, "list_registration_validation_candidates", _slow_candidates)
    controller.select_experiment("registration_validation")
    state = controller.refresh()
    tab.update(state)
    page = tab._page_for("registration_validation")

    page._ensure_candidates_loaded()
    page._ensure_candidates_loaded()

    release.set()
    QTest.qWait(50)

    assert calls["count"] == 1


def test_registration_validation_page_discards_stale_async_results(tmp_path: Path) -> None:
    _app()
    controller = _experiment_controller(tmp_path)
    tab = ExperimentTab(controller)
    controller.select_experiment("registration_validation")
    page = tab._page_for("registration_validation")
    page._candidate_load_generation = 2
    page._candidate_cache = [
        ValidationRunCandidate(
            path="data/registrations/current.json",
            timestamp_utc="2026-01-01T00:00:00+00:00",
            label="current.json",
        )
    ]

    page._apply_loaded_candidates(
        1,
        [
            ValidationRunCandidate(
                path="data/registrations/stale.json",
                timestamp_utc="2026-01-01T00:00:00+00:00",
                label="stale.json",
            )
        ],
        None,
    )

    assert [candidate.path for candidate in page._candidate_cache] == ["data/registrations/current.json"]


def test_registration_validation_selection_changes_do_not_trigger_reload(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _app()
    controller = _experiment_controller(tmp_path)
    tab = ExperimentTab(controller)
    calls = {"count": 0}

    def _candidates(_project_root):
        calls["count"] += 1
        return [
            ValidationRunCandidate(
                path="data/registrations/registration_a.json",
                timestamp_utc="2026-01-01T00:00:00+00:00",
                label="registration_a.json",
            )
        ]

    monkeypatch.setattr(experiment_pages_module, "list_registration_validation_candidates", _candidates)
    controller.select_experiment("registration_validation")
    state = controller.refresh()
    tab.update(state)
    page = tab._page_for("registration_validation")
    QTest.qWait(50)

    item = page.run_table.item(0, 0)
    assert item is not None
    item.setCheckState(Qt.Checked)
    QTest.qWait(20)

    assert calls["count"] == 1


def test_registration_validation_table_apply_is_batched(tmp_path: Path) -> None:
    _app()
    controller = _experiment_controller(tmp_path)
    tab = ExperimentTab(controller)
    controller.select_experiment("registration_validation")
    page = tab._page_for("registration_validation")
    candidates = [
        ValidationRunCandidate(
            path=f"data/registrations/run_{index}.json",
            timestamp_utc="2026-01-01T00:00:00+00:00",
            label=f"run_{index}.json",
        )
        for index in range(80)
    ]

    page._sync_table(candidates, set())

    assert page.run_table.rowCount() == 80
    assert page._table_apply_complete is False

    QTest.qWait(80)

    assert page._table_apply_complete is True
    assert page.run_table.item(79, 0) is not None


def test_registration_validation_page_uses_single_scroll_shell(tmp_path: Path) -> None:
    _app()
    controller = _experiment_controller(tmp_path)
    tab = ExperimentTab(controller)
    controller.select_experiment("registration_validation")
    page = tab._page_for("registration_validation")

    assert page.parameter_scroll is None
    assert page.parameter_panel is page.parameter_container


@pytest.mark.parametrize("experiment_name", ["registration_validation", "pivot_validation"])
def test_validation_page_scroll_hierarchy_has_single_outer_scroll_shell(
    tmp_path: Path, experiment_name: str
) -> None:
    _app()
    controller = _experiment_controller(tmp_path)
    tab = ExperimentTab(controller)
    controller.select_experiment(experiment_name)
    tab.update(controller.refresh())
    page = tab._page_for(experiment_name)

    scroll_areas = page.findChildren(QScrollArea)

    assert page.parameter_scroll is None
    assert page.parameter_panel is page.parameter_container
    assert page.scroll_area in scroll_areas
    assert len(scroll_areas) == 1
    assert page.scroll_area.widgetResizable() is True
    assert page.run_table.parent() is not page.scroll_area


def test_registration_validation_page_slow_row_formatting_stays_responsive(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _app()
    controller = _experiment_controller(tmp_path)
    tab = ExperimentTab(controller)
    controller.select_experiment("registration_validation")
    page = tab._page_for("registration_validation")
    candidates = [
        ValidationRunCandidate(
            path=f"data/registrations/run_{index}.json",
            timestamp_utc="2026-01-01T00:00:00+00:00",
            label=f"run_{index}.json",
        )
        for index in range(48)
    ]
    monkeypatch.setattr(experiment_pages_module, "list_registration_validation_candidates", lambda _project_root: candidates)
    original_row_values = page._row_values

    def _slow_row_values(candidate):
        time.sleep(0.002)
        return original_row_values(candidate)

    monkeypatch.setattr(page, "_row_values", _slow_row_values)

    started = time.monotonic()
    tab.update(controller.refresh())
    elapsed = time.monotonic() - started

    assert elapsed < 0.1
    assert page.loading_label.isVisible() is True

    QTest.qWait(180)

    assert page._table_apply_complete is True
    assert page.run_table.item(47, 0) is not None


def test_registration_validation_placeholder_hides_after_candidates_load(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _app()
    controller = _experiment_controller(tmp_path)
    tab = ExperimentTab(controller)
    controller.select_experiment("registration_validation")
    monkeypatch.setattr(
        experiment_pages_module,
        "list_registration_validation_candidates",
        lambda _project_root: [
            ValidationRunCandidate(
                path="data/registrations/registration_a.json",
                timestamp_utc="2026-01-01T00:00:00+00:00",
                label="registration_a.json",
            )
        ],
    )

    tab.update(controller.refresh())
    page = tab._page_for("registration_validation")
    QTest.qWait(60)

    assert page.loading_label.isVisible() is False
    assert page.loading_label.isEnabled() is False
    assert page.loading_label.testAttribute(Qt.WA_TransparentForMouseEvents) is True
    assert page.run_table.isVisible() is True
    assert page.run_table.isEnabled() is True
    assert page.run_table.rowCount() == 1


@pytest.mark.parametrize(
    ("experiment_name", "candidate_factory", "candidate_attr"),
    [
        (
            "registration_validation",
            lambda: [
                ValidationRunCandidate(
                    path="data/registrations/registration_a.json",
                    timestamp_utc="2026-01-01T00:00:00+00:00",
                    label="registration_a.json",
                )
            ],
            "list_registration_validation_candidates",
        ),
        (
            "pivot_validation",
            lambda: [
                ValidationRunCandidate(
                    path="data/pivot_calibrations/pivot_a",
                    timestamp_utc="2026-01-01T00:00:00+00:00",
                    label="pivot_a",
                )
            ],
            "list_pivot_validation_candidates",
        ),
    ],
)
def test_validation_pages_use_exclusive_non_intercepting_candidate_surfaces(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    experiment_name: str,
    candidate_factory,
    candidate_attr: str,
) -> None:
    _app()
    controller = _experiment_controller(tmp_path)
    tab = ExperimentTab(controller)
    monkeypatch.setattr(experiment_pages_module, candidate_attr, lambda _project_root: candidate_factory())
    controller.select_experiment(experiment_name)

    tab.update(controller.refresh())
    page = tab._page_for(experiment_name)
    QTest.qWait(80)

    assert page.loading_label.isVisible() is False
    assert page.loading_label.isEnabled() is False
    assert page.loading_label.testAttribute(Qt.WA_TransparentForMouseEvents) is True
    assert page.run_table.isVisible() is True
    assert page.run_table.isEnabled() is True
    assert page.run_table.viewport().isEnabled() is True
    assert page.run_table.rowCount() == 1


@pytest.mark.parametrize(
    ("experiment_name", "candidate_attr"),
    [
        ("registration_validation", "list_registration_validation_candidates"),
        ("pivot_validation", "list_pivot_validation_candidates"),
    ],
)
def test_validation_page_table_remains_clickable_after_candidate_apply(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    experiment_name: str,
    candidate_attr: str,
) -> None:
    _app()
    controller = _experiment_controller(tmp_path)
    tab = ExperimentTab(controller)
    candidate_path = f"data/{experiment_name}/source_a"
    monkeypatch.setattr(
        experiment_pages_module,
        candidate_attr,
        lambda _project_root: [
            ValidationRunCandidate(
                path=candidate_path,
                timestamp_utc="2026-01-01T00:00:00+00:00",
                label="source_a",
            )
        ],
    )
    controller.select_experiment(experiment_name)
    tab.update(controller.refresh())
    page = tab._page_for(experiment_name)
    QTest.qWait(80)

    item = page.run_table.item(0, 0)
    assert item is not None
    assert item.flags() & Qt.ItemIsUserCheckable
    assert item.flags() & Qt.ItemIsEnabled

    item.setCheckState(Qt.Checked)
    QTest.qWait(20)

    assert controller.get_config_value("run_paths", []) == [candidate_path]


@pytest.mark.parametrize("experiment_name", ["registration_validation", "pivot_validation"])
def test_validation_pages_defer_and_force_non_native_visualization(tmp_path: Path, experiment_name: str) -> None:
    _app()
    controller = _experiment_controller(tmp_path)
    tab = ExperimentTab(controller)
    controller.select_experiment(experiment_name)
    state = controller.refresh()
    tab.update(state)
    page = tab._page_for(experiment_name)

    assert page.defer_visualization_until_data is True
    assert page.visualization_mode_override == experiment_pages_module.VIS_MODE_PROJECTION
    assert page.viewer_3d is None
    assert page.viewer_placeholder is not None
    assert page.viewer_placeholder.testAttribute(Qt.WA_TransparentForMouseEvents) is True

    state.visualization_model = VisualizationModel(
        series_3d=[
            ScatterSeries3D(
                name="Loaded Validation Points",
                color_hex="#2563eb",
                points_xyz=[(0.0, 0.0, 0.0), (1.0, 2.0, 3.0)],
            )
        ]
    )
    page.set_state(state)

    assert page.viewer_3d is not None
    assert page.viewer_3d.backend_mode in {BACKEND_PROJECTION, BACKEND_PLACEHOLDER}
    assert page.viewer_3d.backend_mode != BACKEND_NATIVE_3D
    assert getattr(page.viewer_3d, "_container", None) is None


def test_registration_validation_candidate_reload_replaces_old_rows_cleanly(
    tmp_path: Path,
) -> None:
    _app()
    controller = _experiment_controller(tmp_path)
    tab = ExperimentTab(controller)
    controller.select_experiment("registration_validation")
    page = tab._page_for("registration_validation")

    first_candidates = [
        ValidationRunCandidate(
            path="data/registrations/registration_a.json",
            timestamp_utc="2026-01-01T00:00:00+00:00",
            label="registration_a.json",
        ),
        ValidationRunCandidate(
            path="data/registrations/registration_b.json",
            timestamp_utc="2026-01-02T00:00:00+00:00",
            label="registration_b.json",
        ),
    ]
    second_candidates = [
        ValidationRunCandidate(
            path="data/registrations/registration_c.json",
            timestamp_utc="2026-01-03T00:00:00+00:00",
            label="registration_c.json",
        )
    ]

    page._candidate_load_generation = 1
    page._apply_loaded_candidates(1, first_candidates, None)
    QTest.qWait(80)
    assert page.run_table.rowCount() == 2

    page._candidate_load_generation = 2
    page._apply_loaded_candidates(2, second_candidates, None)
    QTest.qWait(80)

    assert page.run_table.rowCount() == 1
    item = page.run_table.item(0, 5)
    assert item is not None
    assert item.text() == "data/registrations/registration_c.json"


def test_registration_validation_stale_table_callbacks_do_not_duplicate_end_logs(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    _app()
    controller = _experiment_controller(tmp_path)
    tab = ExperimentTab(controller)
    controller.select_experiment("registration_validation")
    page = tab._page_for("registration_validation")
    candidates = [
        ValidationRunCandidate(
            path=f"data/registrations/run_{index}.json",
            timestamp_utc="2026-01-01T00:00:00+00:00",
            label=f"run_{index}.json",
        )
        for index in range(6)
    ]

    caplog.set_level(logging.INFO)
    page._sync_table(candidates, set())
    page._schedule_table_apply(page._table_apply_generation)
    QTest.qWait(80)

    end_messages = [
        record.getMessage()
        for record in caplog.records
        if "ValidationPage[table_rebuild] end" in record.getMessage()
        and "generation=1" in record.getMessage()
    ]
    assert len(end_messages) == 1


def test_registration_validation_page_emits_timing_stage_logs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    _app()
    controller = _experiment_controller(tmp_path)
    tab = ExperimentTab(controller)
    controller.select_experiment("registration_validation")
    monkeypatch.setattr(experiment_pages_module, "list_registration_validation_candidates", lambda _project_root: [])

    caplog.set_level(logging.INFO)
    tab.update(controller.refresh())
    QTest.qWait(80)

    messages = [record.getMessage() for record in caplog.records]
    assert any("ValidationPage[set_state] start" in message for message in messages)
    assert any("ValidationPage[async_discovery] start" in message for message in messages)
    assert any("ValidationPage[async_discovery] end" in message for message in messages)
    assert any("ValidationPage[candidate_apply] end" in message for message in messages)


def test_registration_validation_manual_page_skips_unchanged_set_state_updates(tmp_path: Path) -> None:
    _app()
    controller = _experiment_controller(tmp_path)
    tab = ExperimentTab(controller)
    controller.select_experiment("registration_validation")
    page = tab._page_for("registration_validation")
    calls = {"count": 0}
    original_set_state = page.set_state

    def _counting_set_state(state):
        calls["count"] += 1
        return original_set_state(state)

    page.set_state = _counting_set_state
    state = controller.refresh()

    assert tab.update(state) is True
    assert tab.update(state) is False
    assert calls["count"] == 1


def test_pivot_validation_manual_page_skips_unchanged_set_state_updates(tmp_path: Path) -> None:
    _app()
    controller = _experiment_controller(tmp_path)
    tab = ExperimentTab(controller)
    controller.select_experiment("pivot_validation")
    page = tab._page_for("pivot_validation")
    calls = {"count": 0}
    original_set_state = page.set_state

    def _counting_set_state(state):
        calls["count"] += 1
        return original_set_state(state)

    page.set_state = _counting_set_state
    state = controller.refresh()

    assert tab.update(state) is True
    assert tab.update(state) is False
    assert calls["count"] == 1


def test_manual_validation_pages_disable_idle_periodic_refresh(tmp_path: Path) -> None:
    controller = _experiment_controller(tmp_path)
    controller.select_experiment("registration_validation")
    controller.refresh()
    controller.state.history_loading = False
    controller._history_dirty = False
    controller._visualization_dirty = False
    controller._preflight_cache_report = controller.state.preflight_report

    assert controller.refresh_policy_for() == "manual"
    assert controller.should_periodically_refresh_selected_experiment() is False


def test_live_experiment_pages_keep_periodic_refresh_enabled(tmp_path: Path) -> None:
    controller = _experiment_controller(tmp_path)
    controller.select_experiment("tracker_timing_validation")
    controller.refresh()
    controller.state.history_loading = False
    controller._history_dirty = False
    controller._visualization_dirty = False
    controller._preflight_cache_report = controller.state.preflight_report

    assert controller.refresh_policy_for() == "live"
    assert controller.should_periodically_refresh_selected_experiment() is True


def test_repeatability_page_uses_manual_refresh_policy_when_idle(tmp_path: Path) -> None:
    controller = _experiment_controller(tmp_path)
    controller.select_experiment("single_segment_repeatability")
    controller.refresh()
    controller.state.history_loading = False
    controller._history_dirty = False
    controller._visualization_dirty = False
    controller._preflight_cache_report = controller.state.preflight_report

    assert controller.refresh_policy_for() == "manual"
    assert controller.should_periodically_refresh_selected_experiment() is False


def test_pretension_validation_page_uses_manual_refresh_policy_when_idle(tmp_path: Path) -> None:
    controller = _experiment_controller(tmp_path)
    controller.select_experiment("pretension_validation")
    controller.refresh()
    controller.state.history_loading = False
    controller._history_dirty = False
    controller._visualization_dirty = False
    controller._preflight_cache_report = controller.state.preflight_report

    assert controller.refresh_policy_for() == "manual"
    assert controller.should_periodically_refresh_selected_experiment() is False


def test_repeatability_manual_page_skips_unchanged_set_state_updates(tmp_path: Path) -> None:
    _app()
    controller = _experiment_controller(tmp_path)
    tab = ExperimentTab(controller)
    controller.select_experiment("single_segment_repeatability")
    page = tab._page_for("single_segment_repeatability")
    calls = {"count": 0}
    original_set_state = page.set_state

    def _counting_set_state(state):
        calls["count"] += 1
        return original_set_state(state)

    page.set_state = _counting_set_state
    state = controller.refresh()

    assert tab.update(state) is True
    assert tab.update(state) is False
    assert calls["count"] == 1


def test_registration_validation_explicit_refresh_still_forces_candidate_reload(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _app()
    controller = _experiment_controller(tmp_path)
    tab = ExperimentTab(controller)
    calls = {"count": 0}

    def _candidates(_project_root):
        calls["count"] += 1
        return []

    monkeypatch.setattr(experiment_pages_module, "list_registration_validation_candidates", _candidates)
    controller.select_experiment("registration_validation")
    state = controller.refresh()
    tab.update(state)
    page = tab._page_for("registration_validation")
    QTest.qWait(40)

    page._refresh_sources()
    QTest.qWait(40)

    assert calls["count"] >= 2


def test_experiment_workspace_can_switch_away_from_registration_validation_while_loading(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _app()
    controller = _experiment_controller(tmp_path)
    tab = ExperimentTab(controller)
    release = threading.Event()

    def _slow_registration_candidates(_project_root):
        release.wait(timeout=1.0)
        return []

    monkeypatch.setattr(experiment_pages_module, "list_registration_validation_candidates", _slow_registration_candidates)
    monkeypatch.setattr(experiment_pages_module, "list_pivot_validation_candidates", lambda _project_root: [])

    controller.select_experiment("registration_validation")
    tab.update(controller.refresh())

    started = time.monotonic()
    controller.select_experiment("pivot_validation")
    pivot_state = controller.refresh()
    tab.update(pivot_state)
    elapsed = time.monotonic() - started

    assert elapsed < 0.1
    assert tab.page_stack.currentWidget() is tab._page_for("pivot_validation")

    release.set()
    QTest.qWait(50)


@pytest.mark.parametrize(
    ("experiment_name", "candidate_attr"),
    [
        ("registration_validation", "list_registration_validation_candidates"),
        ("pivot_validation", "list_pivot_validation_candidates"),
    ],
)
def test_experiment_selector_switches_away_after_validation_candidates_apply(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    experiment_name: str,
    candidate_attr: str,
) -> None:
    _app()
    controller = _experiment_controller(tmp_path)
    tab = ExperimentTab(controller)
    monkeypatch.setattr(
        experiment_pages_module,
        candidate_attr,
        lambda _project_root: [
            ValidationRunCandidate(
                path=f"data/{experiment_name}/source_a",
                timestamp_utc="2026-01-01T00:00:00+00:00",
                label="source_a",
            )
        ],
    )
    controller.select_experiment(experiment_name)
    tab.update(controller.refresh())
    page = tab._page_for(experiment_name)
    QTest.qWait(80)
    assert page.run_table.isVisible() is True

    target_index = tab.experiment_combo.findData("command_schedule_validation")
    assert target_index >= 0
    tab.experiment_combo.setCurrentIndex(target_index)
    QTest.qWait(20)
    tab.update(controller.refresh())

    assert controller.state.selected_experiment == "command_schedule_validation"
    assert tab.page_stack.currentWidget() is tab._page_for("command_schedule_validation")


def test_experiment_workspace_loads_pivot_validation_page(tmp_path: Path) -> None:
    _app()
    controller = _experiment_controller(tmp_path)
    tab = ExperimentTab(controller)
    controller.select_experiment("pivot_validation")

    state = controller.refresh()
    tab.update(state)
    page = tab._page_for("pivot_validation")

    assert state.selected_experiment == "pivot_validation"
    assert state.experiment_title == "Pivot Validation"
    assert page.run_button.text() == "Run Pivot Validation"
    assert page.run_table.columnCount() == 6


def test_app_window_skips_periodic_controller_refresh_for_stable_manual_experiment_pages(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _app()
    window = AppWindow(_app_context(tmp_path))
    try:
        window._refresh_timer.stop()
        controller = window.experiment_controller
        controller.select_experiment("registration_validation")
        state = controller.refresh()
        window.experiment_tab.update(state)
        controller.state.history_loading = False
        controller._history_dirty = False
        controller._visualization_dirty = False
        controller._preflight_cache_report = controller.state.preflight_report
        window.tab_widget.setCurrentWidget(window.experiment_tab)

        counts = {"refresh_prerequisites": 0, "update": 0}
        original_update = window.experiment_tab.update

        def _count_refresh_prerequisites():
            counts["refresh_prerequisites"] += 1
            return controller.state

        def _count_update(state):
            counts["update"] += 1
            return original_update(state)

        monkeypatch.setattr(controller, "refresh_prerequisites", _count_refresh_prerequisites)
        monkeypatch.setattr(window.experiment_tab, "update", _count_update)

        window.refresh()
        window.refresh()

        assert counts == {"refresh_prerequisites": 0, "update": 0}
    finally:
        window.shutdown()


def test_app_window_keeps_periodic_refresh_for_live_experiment_pages(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _app()
    window = AppWindow(_app_context(tmp_path))
    try:
        window._refresh_timer.stop()
        controller = window.experiment_controller
        controller.select_experiment("tracker_timing_validation")
        controller.refresh()
        window.tab_widget.setCurrentWidget(window.experiment_tab)

        counts = {"refresh_prerequisites": 0}

        def _count_refresh_prerequisites():
            counts["refresh_prerequisites"] += 1
            return controller.state

        monkeypatch.setattr(controller, "refresh_prerequisites", _count_refresh_prerequisites)

        window.refresh()
        window.refresh()

        assert counts["refresh_prerequisites"] == 2
    finally:
        window.shutdown()


def test_app_window_skips_periodic_refresh_for_idle_repeatability_page(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _app()
    window = AppWindow(_app_context(tmp_path))
    try:
        window._refresh_timer.stop()
        controller = window.experiment_controller
        controller.select_experiment("single_segment_repeatability")
        state = controller.refresh()
        window.experiment_tab.update(state)
        controller.state.history_loading = False
        controller._history_dirty = False
        controller._visualization_dirty = False
        controller._preflight_cache_report = controller.state.preflight_report
        window.tab_widget.setCurrentWidget(window.experiment_tab)

        counts = {"refresh_prerequisites": 0, "update": 0}
        original_update = window.experiment_tab.update

        def _count_refresh_prerequisites():
            counts["refresh_prerequisites"] += 1
            return controller.state

        def _count_update(current_state):
            counts["update"] += 1
            return original_update(current_state)

        monkeypatch.setattr(controller, "refresh_prerequisites", _count_refresh_prerequisites)
        monkeypatch.setattr(window.experiment_tab, "update", _count_update)

        window.refresh()
        window.refresh()

        assert counts == {"refresh_prerequisites": 0, "update": 0}
    finally:
        window.shutdown()


def test_experiment_workspace_loads_motor_babble_run_result_details(tmp_path: Path) -> None:
    controller = _experiment_controller(tmp_path)
    result = controller.experiment_runner.run_experiment(
        "collect_pose_command_dataset",
        config={
            "dataset_mode": "workspace_coverage",
            "dry_run": True,
            "sample_count_target": 3,
            "samples_per_command": 1,
            "require_robot_frame_tip": False,
            "allow_lower_trust_runtime_tip": True,
            "allow_lower_trust_pretension": True,
            "export_legacy_dat": True,
        },
        output_dir=tmp_path / "data" / "experiments",
        output_dir_name="motor_babble_run",
    )

    assert result.paths.output_dir.joinpath("modeling_dataset_summary.txt").exists()
    assert result.paths.output_dir.joinpath("modeling_dataset_export.jsonl").exists()

    controller.load_run(result.paths.output_dir)
    loaded = controller.refresh()
    labels = {label for label, _value in loaded.result_details}

    assert loaded.selected_experiment == "collect_pose_command_dataset"
    assert "Workspace Plot" in labels
    assert "Command Plot" in labels
    assert "Export JSONL" in labels
    assert "Summary Note" in labels


def test_experiment_workspace_blocks_grid_accuracy_without_tip_calibration(tmp_path: Path) -> None:
    controller = _experiment_controller(tmp_path)
    controller.select_experiment("aurora_grid_accuracy")
    controller.set_config_text(
        "\n".join(
            [
                "dry_run: true",
                "dimensions: [2, 2]",
                "repetitions_per_point: 1",
                "samples_per_point: 1",
                "tool_id: \"0B\"",
                "truth_frame: \"tracker\"",
                "use_tip_calibration: true",
                "allow_coil_origin_fallback: false",
            ]
        )
    )

    state = controller.refresh()

    assert state.preflight_report.overall_status == "blocked"
    assert any("Tip calibration is required" in message for message in state.preflight_report.blocking_messages)


def test_grid_accuracy_page_captures_labeled_points_and_updates_preview(tmp_path: Path) -> None:
    _app()
    controller = _experiment_controller(tmp_path)
    tab = ExperimentTab(controller)
    controller.select_experiment("aurora_grid_accuracy")
    state = controller.refresh()
    tab.update(state)
    page = tab._page_for("aurora_grid_accuracy")

    page.rows_spin.setValue(2)
    page.cols_spin.setValue(2)
    page.samples_spin.setValue(2)
    page.capture_selected_point()
    page._selected_target_index = 1
    page.capture_selected_point()
    page._selected_target_index = 2
    page.capture_selected_point()

    refreshed = controller.refresh()
    tab.update(refreshed)

    assert refreshed.preflight_report.overall_status == "ok_with_warning"
    assert any("synthetic dry-run data" in message.lower() for message in refreshed.preflight_report.warning_messages)
    assert len(controller.get_config_value("captured_points", [])) == 3
    assert page.point_table.item(0, 2).text() == "2"
    assert "P04" in page.selected_point_label.text()
    assert any(label == "Coverage" for label, _value in refreshed.result_details)
    assert page.capture_summary_widget._pairs_signature is not None


def test_grid_accuracy_live_capture_accepts_tracked_tool_state(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Regression: the tracking service reports visible tools as ``tracked``.

    Preflight accepted that state, but the page-side Capture Selected Point
    path used to reject it as not valid, so the button appeared to do nothing
    during real Aurora grid runs.
    """
    from continuum_robot.services.models import ServiceHealthSnapshot, ToolTrackingSnapshot, TrackingSnapshot

    _app()
    controller = _experiment_controller(tmp_path)
    tab = ExperimentTab(controller)
    controller.select_experiment("aurora_grid_accuracy")
    controller.set_config_value("dry_run", False)
    controller.set_config_value("use_tip_calibration", False)
    controller.set_config_value("dimensions", [2, 2])
    controller.set_config_value("samples_per_point", 2)

    snapshot = TrackingSnapshot(
        health=ServiceHealthSnapshot(name="tracking", health="healthy", state="tracking", status="ok"),
        connection_state="connected",
        canonical_state="streaming_healthy",
        tracker_data_age_s=0.01,
        tracker_data_stale=False,
        tools={
            "0B": ToolTrackingSnapshot(
                tool_id="0B",
                present=True,
                valid=True,
                tracking_state="tracked",
                translation_mm=(10.0, 20.0, 30.0),
                quaternion_wxyz=(1.0, 0.0, 0.0, 0.0),
                frame_number=123,
            )
        },
    )
    monkeypatch.setattr(controller.tracking_service, "get_snapshot", lambda: snapshot)

    tab.update(controller.refresh())
    page = tab._page_for("aurora_grid_accuracy")
    page.capture_selected_point()

    captured_points = controller.get_config_value("captured_points", [])
    assert len(captured_points) == 1
    assert len(captured_points[0]["raw_samples"]) == 2
    assert captured_points[0]["raw_samples"][0]["tracking_state"] == "tracked"
    assert captured_points[0]["raw_samples"][0]["capture_mode"] == "live_tracker"
    assert "capture failed" not in page.capture_status_text.toPlainText().lower()


def test_grid_accuracy_dry_run_manual_capture_is_marked_synthetic_and_not_reused(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _app()
    controller = _experiment_controller(tmp_path)
    tab = ExperimentTab(controller)
    controller.select_experiment("aurora_grid_accuracy")
    controller.set_config_value("dry_run", True)
    controller.set_config_value("dimensions", [2, 2])
    controller.set_config_value("samples_per_point", 2)
    state = controller.refresh()
    tab.update(state)
    page = tab._page_for("aurora_grid_accuracy")
    truth_entry = page._current_preview().truth_catalog[0]

    seeds = iter([111, 222])
    monkeypatch.setattr(experiment_pages_module.secrets, "randbits", lambda _bits: next(seeds))

    first_batch = page._collect_point_samples(config=page._grid_config(), truth_entry=truth_entry)
    second_batch = page._collect_point_samples(config=page._grid_config(), truth_entry=truth_entry)

    assert first_batch[0]["capture_mode"] == "synthetic_dry_run"
    assert "synthetic_capture" in first_batch[0]["status_flags"]
    assert first_batch[0]["synthetic_seed_used"] == 111
    assert second_batch[0]["synthetic_seed_used"] == 222
    assert first_batch[0]["position_mm"] != second_batch[0]["position_mm"]


def test_grid_accuracy_preflight_blocks_synthetic_captures_in_live_mode(tmp_path: Path) -> None:
    _app()
    controller = _experiment_controller(tmp_path)
    tab = ExperimentTab(controller)
    controller.select_experiment("aurora_grid_accuracy")
    controller.set_config_value("dry_run", False)
    controller.set_config_value("dimensions", [2, 2])
    controller.set_config_value("samples_per_point", 1)
    controller.set_config_value(
        "captured_points",
        [
            {
                "label": "P01",
                "target_index": 0,
                "raw_samples": [
                    {
                        "position_mm": [0.0, 0.0, 0.0],
                        "quaternion_wxyz": [1.0, 0.0, 0.0, 0.0],
                        "tracking_state": "valid",
                        "position_source": "synthetic_tip",
                        "capture_mode": "synthetic_dry_run",
                        "status_flags": ["dry_run", "synthetic_capture"],
                    }
                ],
            }
        ],
    )

    refreshed = controller.refresh()
    tab.update(refreshed)

    assert refreshed.preflight_report.overall_status == "blocked"
    assert any("synthetic" in message.lower() for message in refreshed.preflight_report.blocking_messages)


def test_grid_accuracy_page_shows_partial_status_and_selected_point_summary(tmp_path: Path) -> None:
    _app()
    controller = _experiment_controller(tmp_path)
    tab = ExperimentTab(controller)
    controller.select_experiment("aurora_grid_accuracy")
    controller.set_config_value("dimensions", [2, 2])
    controller.set_config_value("samples_per_point", 3)
    controller.set_config_value(
        "captured_points",
        [
            {
                "label": "P01",
                "target_index": 0,
                "raw_samples": [
                    {
                        "position_mm": [0.1, 0.0, 0.0],
                        "quaternion_wxyz": [1.0, 0.0, 0.0, 0.0],
                        "tracking_state": "valid",
                        "position_source": "tip",
                    }
                ],
            }
        ],
    )
    state = controller.refresh()
    tab.update(state)
    page = tab._page_for("aurora_grid_accuracy")

    assert page.point_table.item(0, 6).text().startswith("Partial")
    assert page.selected_point_summary_widget._pairs_signature is not None
    assert any(label == "Solve Ready" and value == "No" for label, value in page.capture_summary_widget._pairs_signature)
    assert "1/3 samples" in page.selected_point_label.text()


def test_grid_accuracy_page_locks_capture_settings_until_restart(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _app()
    controller = _experiment_controller(tmp_path)
    tab = ExperimentTab(controller)
    controller.select_experiment("aurora_grid_accuracy")
    state = controller.refresh()
    tab.update(state)
    page = tab._page_for("aurora_grid_accuracy")

    page.rows_spin.setValue(2)
    page.cols_spin.setValue(2)
    page.samples_spin.setValue(2)
    page.capture_selected_point()

    refreshed = controller.refresh()
    tab.update(refreshed)

    assert controller.get_config_value("captured_points", [])
    assert page.rows_spin.isEnabled() is False
    assert page.samples_spin.isEnabled() is False
    assert page.capture_settings_notice.isHidden() is False

    monkeypatch.setattr(
        "continuum_robot.gui.widgets.experiment_pages.QMessageBox.question",
        lambda *args, **kwargs: QMessageBox.Yes,
    )
    page.clear_all_points()
    tab.update(controller.refresh())

    assert controller.get_config_value("captured_points", []) == []
    assert page.rows_spin.isEnabled() is True
    assert page.samples_spin.isEnabled() is True
    assert page.capture_settings_notice.isHidden() is True


def test_grid_accuracy_page_recapture_replaces_existing_batch_without_duplicate_rows(tmp_path: Path) -> None:
    _app()
    controller = _experiment_controller(tmp_path)
    tab = ExperimentTab(controller)
    controller.select_experiment("aurora_grid_accuracy")
    state = controller.refresh()
    tab.update(state)
    page = tab._page_for("aurora_grid_accuracy")

    page.rows_spin.setValue(2)
    page.cols_spin.setValue(2)
    page.samples_spin.setValue(2)
    page.capture_selected_point()
    page._selected_target_index = 0
    page.capture_selected_point()

    captured_points = controller.get_config_value("captured_points", [])
    assert len(captured_points) == 1
    assert len(captured_points[0]["raw_samples"]) == 2
    assert "recaptured" in page.capture_status_text.toPlainText().lower()


def test_single_segment_repeatability_page_shows_fixed_target_catalog_and_baseline_controls(tmp_path: Path) -> None:
    _app()
    controller = _experiment_controller(tmp_path)
    tab = ExperimentTab(controller)
    controller.select_experiment("single_segment_repeatability")
    state = controller.refresh()
    tab.update(state)
    page = tab._page_for("single_segment_repeatability")

    assert page.target_table.rowCount() == 17
    assert page.target_table.item(0, 0).text() == "T00"
    assert "272 visits" in page.protocol_preview_label.text()
    assert page.comparison_summary_widget._pairs_signature is not None
    assert "baseline_run_path" in controller.refresh().config_text


def test_repeatability_page_open_stays_responsive_during_history_scan(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _app()
    controller = _experiment_controller(tmp_path)
    tab = ExperimentTab(controller)

    def _slow_scan(_output_root: Path, _experiment_name: str):
        time.sleep(0.25)
        return []

    monkeypatch.setattr(controller, "_scan_run_history", _slow_scan)
    controller.select_experiment("single_segment_repeatability")

    started = time.monotonic()
    state = controller.refresh()
    tab.update(state)
    elapsed_s = time.monotonic() - started

    assert elapsed_s < 0.20
    assert tab._page_for("single_segment_repeatability") is not None


def test_repeatability_refresh_prerequisites_uses_cached_state_while_history_load_runs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    controller = _experiment_controller(tmp_path)

    def _slow_scan(_output_root: Path, _experiment_name: str):
        time.sleep(0.25)
        return []

    monkeypatch.setattr(controller, "_scan_run_history", _slow_scan)
    controller.select_experiment("single_segment_repeatability")
    initial_state = controller.refresh()
    assert initial_state.selected_experiment == "single_segment_repeatability"

    monkeypatch.setattr(
        controller,
        "refresh",
        lambda: (_ for _ in ()).throw(AssertionError("refresh() should not be called while cached preflight is valid")),
    )

    cached_state = controller.refresh_prerequisites()

    assert cached_state is controller.state
    assert cached_state.selected_experiment == "single_segment_repeatability"


def test_repeatability_page_baseline_browse_does_not_force_synchronous_refresh(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _app()
    controller = _experiment_controller(tmp_path)
    tab = ExperimentTab(controller)
    controller.select_experiment("single_segment_repeatability")
    tab.update(controller.refresh())
    page = tab._page_for("single_segment_repeatability")

    monkeypatch.setattr(
        "continuum_robot.gui.widgets.experiment_pages.QFileDialog.getExistingDirectory",
        lambda *args, **kwargs: str(tmp_path),
    )
    monkeypatch.setattr(
        controller,
        "refresh",
        lambda: (_ for _ in ()).throw(AssertionError("baseline browse should not force a synchronous refresh")),
    )

    page._browse_baseline_run()

    assert controller.get_config_value("baseline_run_path") == str(tmp_path)


def test_experiment_tab_can_switch_away_while_repeatability_history_scan_runs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _app()
    controller = _experiment_controller(tmp_path)
    tab = ExperimentTab(controller)

    def _slow_scan(_output_root: Path, _experiment_name: str):
        time.sleep(0.25)
        return []

    monkeypatch.setattr(controller, "_scan_run_history", _slow_scan)
    controller.select_experiment("single_segment_repeatability")
    tab.update(controller.refresh())

    controller.select_experiment("aurora_grid_accuracy")
    started = time.monotonic()
    tab.update(controller.refresh())
    elapsed_s = time.monotonic() - started

    assert elapsed_s < 0.20
    assert tab.page_stack.currentWidget() is tab._page_for("aurora_grid_accuracy")


def test_experiment_workspace_parameter_edit_updates_serialized_config(tmp_path: Path) -> None:
    controller = _experiment_controller(tmp_path)
    controller.select_experiment("command_schedule_validation")

    controller.set_config_value("schedule.kind", "trajectory")
    controller.set_parameter_value("schedule.trajectory_points_cm", "- [0.0, 0.0, 0.0, 0.0]\n- [0.1, 0.0, 0.0, 0.0]")
    state = controller.refresh()

    assert state.config_error is None
    assert "trajectory" in state.config_text
    assert "trajectory_points_cm" in state.config_text
    assert controller.get_config_value("schedule.kind") == "trajectory"


def test_experiment_workspace_loads_prior_run_and_history(tmp_path: Path) -> None:
    controller = _experiment_controller(tmp_path)
    result = controller.experiment_runner.run_experiment(
        "command_schedule_validation",
        config={
            "schedule": {
                "kind": "babble",
                "dimensions": 4,
                "amplitude_cm": 0.1,
                "babble_count": 5,
            },
        },
        output_dir=tmp_path / "data" / "experiments",
        output_dir_name="saved_schedule_run",
    )
    assert result.success is True

    controller.select_experiment("command_schedule_validation")
    state = controller.refresh()
    assert any("saved_schedule_run" in entry.path for entry in state.history)

    controller.load_run(result.paths.output_dir)
    loaded = controller.refresh()
    assert loaded.loaded_run_path == str(result.paths.output_dir)
    assert loaded.selected_experiment == "command_schedule_validation"
    assert loaded.visualization_model.summary_lines
    assert any(label == "Run ID" for label, _value in loaded.result_details)


def test_experiment_workspace_plans_output_under_experiment_type_folder(tmp_path: Path) -> None:
    controller = _experiment_controller(tmp_path)
    controller.select_experiment("command_schedule_validation")

    state = controller.refresh()

    assert Path(state.planned_output_dir).parent == (
        tmp_path / "data" / "experiments" / "command_schedule_validation"
    )


def test_experiment_workspace_tab_updates_without_crashing_in_mock_mode(tmp_path: Path) -> None:
    _app()
    controller = _experiment_controller(tmp_path)
    tab = ExperimentTab(controller)
    try:
        tab.update(controller.refresh())
        assert tab.page_stack.currentWidget() is tab.empty_page
        controller.select_experiment("aurora_grid_accuracy")
        tab.update(controller.refresh())
        page = tab._page_for("aurora_grid_accuracy")
        assert tab.page_stack.currentWidget() is page
        assert page.viewer_3d is None
        assert page.grid_preview_widget is not None
    finally:
        controller.shutdown()


def test_repeatability_page_wraps_workspace_in_scroll_area_and_stacks_on_narrow_width(tmp_path: Path) -> None:
    _app()
    controller = _experiment_controller(tmp_path)
    tab = ExperimentTab(controller)
    controller.select_experiment("single_segment_repeatability")
    tab.resize(920, 760)
    tab.show()
    tab.update(controller.refresh())
    QTest.qWait(20)
    page = tab._page_for("single_segment_repeatability")

    assert isinstance(page.scroll_area, QScrollArea)
    assert page.scroll_area.widget() is not None
    assert page.top_row.direction() == QBoxLayout.TopToBottom
    assert page.bottom_row.direction() == QBoxLayout.TopToBottom

    tab.resize(1480, 900)
    QTest.qWait(20)

    assert page.top_row.direction() == QBoxLayout.LeftToRight
    assert page.bottom_row.direction() == QBoxLayout.LeftToRight


def test_repeatability_page_defers_3d_viewer_construction_until_data_exists(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _app()
    controller = _experiment_controller(tmp_path)
    tab = ExperimentTab(controller)
    controller.select_experiment("single_segment_repeatability")

    def _fail_viewer_construction(*args, **kwargs):
        raise AssertionError("repeatability page should not build the 3D viewer during initial page activation")

    monkeypatch.setattr(
        "continuum_robot.gui.widgets.experiment_pages.Experiment3DWidget",
        _fail_viewer_construction,
    )

    tab.update(controller.refresh())
    page = tab._page_for("single_segment_repeatability")

    assert page.viewer_3d is None
    assert page.viewer_placeholder is not None
    assert page.viewer_placeholder.isVisible()


def test_repeatability_page_preserves_target_table_scroll_on_benign_refresh(tmp_path: Path) -> None:
    _app()
    controller = _experiment_controller(tmp_path)
    tab = ExperimentTab(controller)
    controller.select_experiment("single_segment_repeatability")
    tab.resize(1100, 860)
    tab.show()
    tab.update(controller.refresh())
    QTest.qWait(20)
    page = tab._page_for("single_segment_repeatability")

    scroll_bar = page.target_table.verticalScrollBar()
    scroll_bar.setValue(max(1, scroll_bar.maximum() // 2))
    previous_value = scroll_bar.value()

    tab.update(controller.refresh())
    QTest.qWait(20)

    assert page.target_table.verticalScrollBar().value() >= previous_value - 2


def test_experiment_shell_header_stays_compact_and_tracks_selection(tmp_path: Path) -> None:
    _app()
    controller = _experiment_controller(tmp_path)
    tab = ExperimentTab(controller)

    tab.update(controller.refresh())
    assert tab.load_defaults_button.isEnabled() is False
    assert tab.selected_status_chip.text() == "No Selection"
    assert tab.selected_badges_label.isHidden() is True

    controller.select_experiment("aurora_grid_accuracy")
    tab.update(controller.refresh())

    assert tab.load_defaults_button.isEnabled() is True
    assert tab.selected_experiment_title.text() == "Aurora Grid Accuracy"
    assert "align" in tab.selected_experiment_description.text().lower()
    assert tab.selected_badges_label.isHidden() is False


def test_registration_and_experiment_tabs_expose_resizable_layout_defaults(tmp_path: Path) -> None:
    _app()
    settings = _settings()
    tracking_service = _tracking_service(settings, tmp_path)
    servo_service = _servo_service(tmp_path)
    registration_service = _registration_service(settings, tmp_path, tracking_service)
    registration_controller = RegistrationController(
        registration_service=registration_service,
        registration_config=settings.registration,
    )
    system_controller = SystemController(
        tracking_service=tracking_service,
        openrb_client=MockOpenRbClient(),
        servo_service=servo_service,
        settings=settings,
    )
    servos_controller = ServosController(servo_service, settings)
    registration_tab = RegistrationTab(registration_controller)
    experiment_controller = _experiment_controller(tmp_path)
    experiment_tab = ExperimentTab(experiment_controller)
    experiment_controller.select_experiment("single_segment_repeatability")
    experiment_tab.update(experiment_controller.refresh())
    repeatability_page = experiment_tab._page_for("single_segment_repeatability")
    servos_tab = ServosTab(servos_controller)
    system_tab = SystemTab(system_controller)

    assert registration_tab.points_table.minimumHeight() >= 180
    assert registration_tab.samples_table.minimumHeight() >= 180
    assert registration_tab.landmark_map.minimumHeight() >= 200
    assert experiment_tab.experiment_combo.minimumHeight() >= 36
    assert repeatability_page.parameter_scroll.minimumWidth() >= 300
    assert repeatability_page.viewer_3d is None
    assert repeatability_page.viewer_placeholder is not None
    assert servos_tab.telemetry_table.minimumHeight() >= 190
    assert system_tab.status_text.minimumHeight() >= 160


def test_experiment_shell_routes_selection_to_custom_pages(tmp_path: Path) -> None:
    _app()
    controller = _experiment_controller(tmp_path)
    tab = ExperimentTab(controller)

    tab.update(controller.refresh())
    assert tab.page_stack.currentWidget() is tab.empty_page

    controller.select_experiment("replay_runner")
    tab.update(controller.refresh())
    replay_page = tab._page_for("replay_runner")

    assert tab.page_stack.currentWidget() is replay_page
    assert replay_page.experiment_name == "replay_runner"


def test_experiment_shell_routes_pretension_validation_to_custom_page(tmp_path: Path) -> None:
    _app()
    controller = _experiment_controller(tmp_path)
    tab = ExperimentTab(controller)

    controller.select_experiment("pretension_validation")
    tab.update(controller.refresh())
    pretension_page = tab._page_for("pretension_validation")

    assert tab.page_stack.currentWidget() is pretension_page
    assert pretension_page.experiment_name == "pretension_validation"
    assert pretension_page.run_button.text() == "Run Pretension Validation"


def test_experiment_pretension_validation_page_exposes_staged_mode_controls(tmp_path: Path) -> None:
    _app()
    controller = _experiment_controller(tmp_path)
    tab = ExperimentTab(controller)

    controller.select_experiment("pretension_validation")
    tab.update(controller.refresh())
    page = tab._page_for("pretension_validation")

    mode_options = {str(page.mode_combo.itemData(index)) for index in range(page.mode_combo.count())}
    assert "single_servo_trace" in mode_options
    assert "single_segment_characterization" in mode_options
    assert "single_segment_staged" in mode_options
    strategy_options = {str(page.strategy_combo.itemData(index)) for index in range(page.strategy_combo.count())}
    assert {"conservative_startup", "characterization", "legacy"}.issubset(strategy_options)
    assert page.staged_mode_card.isHidden() is False
    assert controller.refresh_policy_for("pretension_validation") == "manual"

    staged_index = page.mode_combo.findData("single_segment_staged")
    assert staged_index >= 0
    page.mode_combo.setCurrentIndex(staged_index)
    QTest.qWait(20)

    assert controller.config_payload().get("mode") == "single_segment_staged"
    assert page.staged_mode_card.isHidden() is False
    assert page.servo_combo.isEnabled() is False
    assert page.staged_servo_ids_edit.isEnabled() is True


def test_pretension_validation_page_staged_servo_ids_defaults_to_active_segment(tmp_path: Path) -> None:
    """When the experiment config leaves ``servo_ids`` empty, the staged
    servo-IDs line edit must display the ACTIVE SEGMENT's IDs, not the union
    of every segment's IDs.

    Regression: with robot_8servo.yaml as the active robot config and segment_a
    selected, the legacy fallback used ``settings.robot.servo_ids`` (which is
    ``[1,2,3,4,5,6,7,8]``) and displayed all 8 IDs on a page that only
    operates on the 4-servo active segment. The fix routes the fallback
    through ``active_segment_servo_ids()`` so the field reflects what the
    experiment will actually use at runtime."""
    _app()
    controller = _experiment_controller(tmp_path)
    tab = ExperimentTab(controller)

    controller.select_experiment("pretension_validation")
    tab.update(controller.refresh())
    page = tab._page_for("pretension_validation")

    # Confirm the experiment config has empty servo_ids (the YAML default).
    payload = controller.config_payload()
    assert not payload.get("servo_ids"), (
        "PretensionValidation config should default to empty servo_ids so the "
        "active segment fallback is exercised by the GUI."
    )

    # The line edit must show the active segment IDs only.
    displayed = page.staged_servo_ids_edit.text()
    active_ids = [int(v) for v in controller.settings.robot.active_segment_servo_ids()]
    expected = ",".join(str(v) for v in active_ids)
    assert displayed == expected, (
        f"Staged servo IDs field should show the active segment {active_ids}, "
        f"not the union of all segment IDs. Got {displayed!r}."
    )

    # The placeholder must mention the active segment label so the operator
    # knows what an empty field will inherit.
    placeholder = page.staged_servo_ids_edit.placeholderText()
    assert "active segment" in placeholder.lower() or "1,2,3,4" in placeholder

    # The pretension_start_mode combo must default to (or at least offer)
    # soft_release_to_zero_current as the recommended starting condition.
    start_mode_values = {
        str(page.pretension_start_mode_combo.itemData(index))
        for index in range(page.pretension_start_mode_combo.count())
    }
    assert "soft_release_to_zero_current" in start_mode_values


def test_experiment_shell_routes_tracker_timing_validation_to_custom_page(tmp_path: Path) -> None:
    _app()
    controller = _experiment_controller(tmp_path)
    tab = ExperimentTab(controller)

    controller.select_experiment("tracker_timing_validation")
    tab.update(controller.refresh())
    timing_page = tab._page_for("tracker_timing_validation")

    assert tab.page_stack.currentWidget() is timing_page
    assert timing_page.experiment_name == "tracker_timing_validation"
    assert timing_page.run_button.text() == "Run Timing Diagnostic"
    assert timing_page.tool_mode_combo.currentData() == "both"


def test_experiment_shell_routes_servo_tracker_sync_validation_to_custom_page(tmp_path: Path) -> None:
    _app()
    controller = _experiment_controller(tmp_path)
    tab = ExperimentTab(controller)

    controller.select_experiment("servo_tracker_sync_validation")
    tab.update(controller.refresh())
    sync_page = tab._page_for("servo_tracker_sync_validation")

    assert tab.page_stack.currentWidget() is sync_page
    assert sync_page.experiment_name == "servo_tracker_sync_validation"
    assert sync_page.run_button.text() == "Run Sync Validation"
    assert sync_page.tool_mode_combo.currentData() == "0A"


def test_app_window_status_bar_tracks_active_servo_and_experiment_messages(tmp_path: Path) -> None:
    _app()
    window = AppWindow(_app_context(tmp_path))
    try:
        window._refresh_timer.stop()
        window.servos_controller.state.status_message = "Servo workspace ready"
        window.experiment_controller.state.status_message = "Experiment workspace ready"

        window.tab_widget.setCurrentWidget(window.servos_tab)
        window.refresh()
        assert window.statusBar().currentMessage() == "Servo workspace ready"

        window.tab_widget.setCurrentWidget(window.experiment_tab)
        window.refresh()
        assert window.statusBar().currentMessage() == "Experiment workspace ready"
    finally:
        window.shutdown()


def test_app_window_applies_dark_theme_palette_without_breaking_key_labels(tmp_path: Path) -> None:
    app = _app()
    window = AppWindow(_app_context(tmp_path))
    try:
        window._refresh_timer.stop()
        palette = app.palette()
        assert palette.color(QPalette.Window).lightness() < 40
        assert palette.color(QPalette.WindowText).lightness() > 180
        window.system_tab.update(window.system_controller.refresh())
        assert window.system_tab.title_label.text() == "System"
        assert window.system_tab.tracker_status_label.text() != ""
        assert window.experiment_tab.selected_status_chip.text() != ""
    finally:
        window.shutdown()


def test_registration_tab_capture_button_records_sample_into_service_session(tmp_path: Path) -> None:
    _app()
    settings = _settings()
    tracking_service = _tracking_service(settings, tmp_path)
    registration_service = _registration_service(settings, tmp_path, tracking_service)
    controller = RegistrationController(
        registration_service=registration_service,
        registration_config=settings.registration,
    )
    tab = RegistrationTab(controller)
    tab.resize(1200, 900)
    tab.show()
    tracking_service.start()
    try:
        tab.update(controller.refresh())
        QTest.mouseClick(tab.begin_button, Qt.LeftButton)
        active_label = controller.state.current_label
        assert active_label is not None

        QTest.mouseClick(tab.capture_button, Qt.LeftButton)
        snapshot = registration_service.get_snapshot()
    finally:
        tracking_service.stop()

    assert len(snapshot.raw_points_by_label[active_label]) == 1
    assert controller.state.captured_counts[active_label] == 1
    # samples_table now shows one row per landmark with a median + capture
    # count (not one row per raw capture) — the perf fix that keeps the
    # GUI snappy at 12-pt × 20-capture configs. Find the active_label's
    # row and confirm its capture-count cell reads "1 captures".
    row_index = next(
        (
            i
            for i in range(tab.samples_table.rowCount())
            if tab.samples_table.item(i, 0) is not None
            and tab.samples_table.item(i, 0).text() == active_label
        ),
        None,
    )
    assert row_index is not None
    assert tab.samples_table.item(row_index, 1).text() == "1 captures"


def test_registration_tab_load_latest_without_file_reports_status(tmp_path: Path) -> None:
    _app()
    settings = _settings()
    tracking_service = _tracking_service(settings, tmp_path)
    registration_service = _registration_service(settings, tmp_path, tracking_service)
    controller = RegistrationController(
        registration_service=registration_service,
        registration_config=settings.registration,
    )
    tab = RegistrationTab(controller)
    tab.resize(1200, 900)
    tab.show()

    QTest.mouseClick(tab.load_button, Qt.LeftButton)

    assert "No accepted registration file was found." in tab.status_text.toPlainText()


def test_registration_tab_map_click_toggles_selection(tmp_path: Path) -> None:
    _app()
    settings = _settings()
    settings.registration.candidate_landmarks = [
        RegistrationLandmarkConfig(id="L1", xyz_mm=[0.0, 0.0, 0.0], display_label="Front Left"),
        RegistrationLandmarkConfig(id="L2", xyz_mm=[20.0, 0.0, 0.0], display_label="Front Right"),
        RegistrationLandmarkConfig(id="L3", xyz_mm=[0.0, 20.0, 0.0], display_label="Rear Left"),
        RegistrationLandmarkConfig(id="L4", xyz_mm=[20.0, 20.0, 0.0], display_label="Rear Right"),
        RegistrationLandmarkConfig(id="L5", xyz_mm=[10.0, 10.0, 8.0], display_label="Center"),
    ]
    settings.registration.landmark_labels = ["L1", "L2", "L3", "L4"]
    tracking_service = _tracking_service(settings, tmp_path)
    registration_service = _registration_service(
        settings,
        tmp_path,
        tracking_service,
        nominal_points={
            "L1": [0.0, 0.0, 0.0],
            "L2": [20.0, 0.0, 0.0],
            "L3": [0.0, 20.0, 0.0],
            "L4": [20.0, 20.0, 0.0],
            "L5": [10.0, 10.0, 8.0],
        },
    )
    controller = RegistrationController(
        registration_service=registration_service,
        registration_config=settings.registration,
    )
    tab = RegistrationTab(controller)
    tab.resize(1200, 900)
    tab.show()
    tab.update(controller.refresh())

    point_l2 = tab.landmark_map.point_center_for_label("L2")
    point_l5 = tab.landmark_map.point_center_for_label("L5")
    assert point_l2 is not None
    assert point_l5 is not None

    QTest.mouseClick(tab.landmark_map, Qt.LeftButton, Qt.NoModifier, point_l2)
    tab.update(controller.refresh())
    assert controller.state.selected_model_labels == ["L1", "L3", "L4"]

    QTest.mouseClick(tab.landmark_map, Qt.LeftButton, Qt.NoModifier, point_l5)
    tab.update(controller.refresh())
    assert controller.state.selected_model_labels == ["L1", "L3", "L4", "L5"]


def test_registration_tab_table_click_toggles_selection(tmp_path: Path) -> None:
    _app()
    settings = _settings()
    settings.registration.candidate_landmarks = [
        RegistrationLandmarkConfig(id="L1", xyz_mm=[0.0, 0.0, 0.0], display_label="Front Left"),
        RegistrationLandmarkConfig(id="L2", xyz_mm=[20.0, 0.0, 0.0], display_label="Front Right"),
        RegistrationLandmarkConfig(id="L3", xyz_mm=[0.0, 20.0, 0.0], display_label="Rear Left"),
        RegistrationLandmarkConfig(id="L4", xyz_mm=[20.0, 20.0, 0.0], display_label="Rear Right"),
        RegistrationLandmarkConfig(id="L5", xyz_mm=[10.0, 10.0, 8.0], display_label="Center"),
    ]
    settings.registration.landmark_labels = ["L1", "L2", "L3", "L4"]
    tracking_service = _tracking_service(settings, tmp_path)
    registration_service = _registration_service(
        settings,
        tmp_path,
        tracking_service,
        nominal_points={
            "L1": [0.0, 0.0, 0.0],
            "L2": [20.0, 0.0, 0.0],
            "L3": [0.0, 20.0, 0.0],
            "L4": [20.0, 20.0, 0.0],
            "L5": [10.0, 10.0, 8.0],
        },
    )
    controller = RegistrationController(
        registration_service=registration_service,
        registration_config=settings.registration,
    )
    tab = RegistrationTab(controller)
    tab.resize(1200, 900)
    tab.show()
    tab.update(controller.refresh())

    row = controller.state.available_model_labels.index("L2")
    index = tab.available_points_table.model().index(row, 0)
    rect = tab.available_points_table.visualRect(index)
    QTest.mouseClick(tab.available_points_table.viewport(), Qt.LeftButton, Qt.NoModifier, rect.center())
    tab.update(controller.refresh())
    assert controller.state.selected_model_labels == ["L1", "L3", "L4"]


def test_servos_controller_routes_jog_through_servo_service(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    service = _servo_service(tmp_path)
    service.connect("/dev/mock-openrb", 115200)
    controller = ServosController(service, _settings())
    seen: dict[str, tuple[int, int]] = {}

    def _fake_jog(servo_id: int, delta_ticks: int) -> ServoCommandResult:
        seen["jog"] = (servo_id, delta_ticks)
        return ServoCommandResult(
            positions_by_id={servo_id: 2073},
            telemetry_by_id={
                servo_id: ServoTelemetry(
                    servo_id=servo_id,
                    present_position=2073,
                    present_current_ma=160,
                    present_voltage_mv=12000,
                )
            },
            message="Routed through servo service.",
        )

    monkeypatch.setattr(service, "jog_servo", _fake_jog)

    controller.jog_servo(2, 25)

    assert seen["jog"] == (2, 25)
    assert controller.state.status_message == "Routed through servo service."


def test_servos_controller_routes_displacement_through_servo_service(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    service = _servo_service(tmp_path)
    service.connect("/dev/mock-openrb", 115200)
    controller = ServosController(service, _settings())
    controller.state.neutral_setpoints = {1: 2048, 2: 2048, 3: 2048, 4: 2048}
    controller.set_tendon_displacements([0.0, 0.1, -0.1, 0.0])
    seen: dict[str, object] = {}

    def _fake_command(*, tendon_displacements_cm: list[float], neutral_ticks: list[int], servo_ids: list[int]) -> ServoCommandResult:
        seen["payload"] = {
            "tendon_displacements_cm": list(tendon_displacements_cm),
            "neutral_ticks": list(neutral_ticks),
            "servo_ids": list(servo_ids),
        }
        return ServoCommandResult(
            positions_by_id={1: 2048, 2: 2059, 3: 2037, 4: 2048},
            telemetry_by_id={},
            message="Displacement routed through servo service.",
        )

    monkeypatch.setattr(service, "command_displacement", _fake_command)

    controller.apply_displacement()

    assert seen["payload"] == {
        "tendon_displacements_cm": [0.0, 0.1, -0.1, 0.0],
        "neutral_ticks": [2048, 2048, 2048, 2048],
        "servo_ids": [1, 2, 3, 4],
    }
    assert controller.state.status_message == "Displacement routed through servo service."

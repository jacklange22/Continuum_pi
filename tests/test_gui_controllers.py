import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from pathlib import Path
import json
import threading
import time

import numpy as np
import pytest
from PySide6.QtCore import Qt
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication, QBoxLayout, QScrollArea

from continuum_robot.app.bootstrap import AppContext
from continuum_robot.app.service_registry import ServiceRegistry
from continuum_robot.config.schemas import RegistrationLandmarkConfig, RegistrationWorkflowConfig
from continuum_robot.config.config_loader import ConfigLoader
from continuum_robot.config.settings import Settings
from continuum_robot.config.schemas import (
    CalibrationConfig,
    ExperimentConfig,
    RobotConfig,
    RuntimeConfig,
    SafetyConfig,
    SerialConfig,
)
from continuum_robot.gui.app_window import AppWindow
from continuum_robot.gui.controllers.pretension_controller import PretensionController
from continuum_robot.gui.controllers.registration_controller import RegistrationController
from continuum_robot.gui.controllers.servos_controller import ServosController
from continuum_robot.gui.controllers.system_controller import SystemController, SystemViewState
from continuum_robot.gui.controllers.experiment_controller import ExperimentController
from continuum_robot.gui.controllers.tracker_mvp_controller import TrackerMvpController, TrackerMvpViewState
from continuum_robot.gui.experiment_visualization import ChartModel, VisualizationModel
from continuum_robot.gui.tabs.experiment_tab import ExperimentTab
from continuum_robot.gui.tabs.pretension_tab import PretensionTab
from continuum_robot.gui.tabs.registration_tab import RegistrationTab
from continuum_robot.gui.tabs.servos_tab import ServosTab
from continuum_robot.gui.tabs.system_tab import SystemTab
from continuum_robot.gui.tabs.tracker_mvp_tab import TrackerMvpTab
from continuum_robot.gui.tabs.tracking_tab import TrackingTab
from continuum_robot.gui.widgets.experiment_results_widget import ExperimentResultsWidget
from continuum_robot.gui.widgets.tool_plot_widget import ToolPlotWidget
from continuum_robot.experiments.experiment_loader import ExperimentLoader
from continuum_robot.experiments.experiment_runner import ExperimentRunner
from continuum_robot.hardware.mock_dxl_bus import MockDxlBus
from continuum_robot.hardware.mock_openrb_client import MockOpenRbClient
from continuum_robot.hardware.serial_ports import SerialPortInfo
from continuum_robot.registration.repository import RegistrationRepository
from continuum_robot.registration.rigid_solver import RigidRegistrationSolver
from continuum_robot.registration.runtime_tip_repository import RuntimeTipCalibrationRepository
from continuum_robot.services.registration_service import RegistrationService
from continuum_robot.services.runtime_tip_calibration_service import RuntimeTipCalibrationService
from continuum_robot.services.tracking_service import TrackingService
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
            neutral_setpoints_path="data/calibrations/neutral_setpoints.json",
            latest_registration_path="data/registrations/latest_registration.json",
        ),
    )


def _servo_service(tmp_path: Path) -> ServoService:
    return ServoService(
        dxl_bus=MockDxlBus([1, 2, 3, 4]),
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
        pivot_tip_path="data/tip_cals/generated_penprobe_tip.csv",
        measurement_point_message="Pen-probe tip file loaded from data/tip_cals/generated_penprobe_tip.csv.",
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
        assert labels == ["System", "Tracking", "Registration", "Servos", "Pretension", "Experiment"]
    finally:
        window.shutdown()


def test_servos_tab_wraps_workspace_in_scroll_area(tmp_path: Path) -> None:
    _app()
    controller = ServosController(_servo_service(tmp_path), _settings())
    tab = ServosTab(controller)

    assert isinstance(tab.scroll_area, QScrollArea)
    assert tab.scroll_area.widget() is not None


def test_servos_tab_selected_servo_panel_reflects_controller_state(tmp_path: Path) -> None:
    _app()
    service = _servo_service(tmp_path)
    service.connect("/dev/mock-openrb", 115200)
    controller = ServosController(service, _settings())
    controller.set_selected_servo(2)
    controller.fine_jog(2, 1)
    tab = ServosTab(controller)

    tab.update(controller.state)

    assert tab.selected_servo_id_value_label.text() == "2"
    assert tab.selected_servo_action_label.text() == "Tighten Fine"
    assert tab.selected_servo_result_label.text() == "Sent"
    assert tab.selected_servo_bounds_label.text() == "[0, 4095]"
    assert tab.selected_servo_freshness_limit_label.text() == "0.250 s"
    assert tab.selected_servo_position_label.text() == str(service.dxl_bus._state[2].present_position)
    assert tab.selected_servo_current_draw_label.text() == str(service.dxl_bus._state[2].present_current_ma)


def test_servos_tab_hides_inactive_issue_rows_and_marks_selected_button(tmp_path: Path) -> None:
    _app()
    service = _servo_service(tmp_path)
    service.connect("/dev/mock-openrb", 115200)
    controller = ServosController(service, _settings())
    controller.set_selected_servo(3)
    tab = ServosTab(controller)

    tab.update(controller.state)

    assert tab.missing_ids_label.isHidden() is True
    assert tab.unexpected_ids_label.isHidden() is True
    assert tab._selector_buttons[3].isChecked() is True
    assert tab._selector_buttons[1].isChecked() is False


def test_servos_tab_hides_id_assignment_controls_from_operator_surface(tmp_path: Path) -> None:
    _app()
    tab = ServosTab(ServosController(_servo_service(tmp_path), _settings()))

    assert hasattr(tab, "scan_button")
    assert not hasattr(tab, "assign_button")


def test_pretension_tab_wraps_workspace_in_scroll_area(tmp_path: Path) -> None:
    _app()
    service = _pretension_service(tmp_path)
    controller = PretensionController(servo_service=service, settings=_settings())
    tab = PretensionTab(controller)

    assert isinstance(tab.scroll_area, QScrollArea)
    assert tab.scroll_area.widget() is not None


def test_pretension_tab_preserves_unsaved_parameter_edits_across_refresh(tmp_path: Path) -> None:
    _app()
    service = _pretension_service(tmp_path)
    service.connect("/dev/mock-openrb", 115200)
    controller = PretensionController(servo_service=service, settings=_settings())
    tab = PretensionTab(controller)

    tab.update(controller.state)
    tab.step_ticks_spin.setValue(7)
    tab.current_delta_spin.setValue(95)
    tab.update(controller.refresh())

    assert tab.step_ticks_spin.value() == 7
    assert tab.current_delta_spin.value() == 95


def test_pretension_controller_runs_only_on_selected_servo_and_persists_result(tmp_path: Path) -> None:
    settings = _settings()
    bus = _MultiServoPretensionBus(current_sequences={2: [180, 230]})
    service = _pretension_service(tmp_path, dxl_bus=bus)
    service.connect("/dev/mock-openrb", 115200)
    before = {servo_id: telemetry.present_position for servo_id, telemetry in bus._state.items()}
    controller = PretensionController(servo_service=service, settings=settings)

    controller.set_selected_servo(2)
    controller.measure_baseline(sample_count=3, filter_window=1)
    controller.start_pretension(
        parameters=PretensionParameters(
            untensioned_reference_tick=4095,
            step_ticks=2,
            settle_time_s=0.0,
            baseline_sample_count=3,
            current_filter_window=1,
            current_delta_threshold_ma=60,
            absolute_trigger_current_ma=500,
            hard_current_stop_ma=850,
            max_travel_ticks=320,
            timeout_s=2.0,
        )
    )
    controller._pretension_thread.join(timeout=1.0)
    controller.save_pretension_result()

    assert bus._state[2].present_position < before[2]
    assert bus._state[1].present_position == before[1]
    assert bus._state[3].present_position == before[3]
    assert bus._state[4].present_position == before[4]
    summary = service.get_calibration_summary()
    assert summary.servo_entries[2].pretension_result_status == "accepted"
    assert summary.servo_entries[2].latest_pretension_run is not None
    assert summary.servo_entries[2].latest_pretension_run["status"] == "threshold_reached"
    assert any(row["servo_id"] == "2" and row["status"] == "accepted" for row in controller.state.comparison_rows)


def test_pretension_controller_blocks_selected_servo_when_torque_is_off(tmp_path: Path) -> None:
    settings = _settings()
    service = _pretension_service(tmp_path, dxl_bus=MockDxlBus([1, 2, 3, 4]))
    service.connect("/dev/mock-openrb", 115200)
    controller = PretensionController(servo_service=service, settings=settings)

    controller.set_selected_servo(3)
    controller.start_pretension(parameters=service.default_pretension_parameters(3))

    assert controller.state.run_state == "blocked"
    assert "Torque must be enabled" in controller.state.run_state_message


def test_pretension_controller_applies_live_parameters_without_runtime_reload(tmp_path: Path) -> None:
    settings = _settings()
    service = _pretension_service(tmp_path)
    controller = PretensionController(servo_service=service, settings=settings)

    controller.apply_live_parameters(
        parameters=PretensionParameters(
            untensioned_reference_tick=4010,
            step_ticks=4,
            settle_time_s=0.1,
            baseline_sample_count=6,
            current_filter_window=4,
            current_delta_threshold_ma=75,
            absolute_trigger_current_ma=260,
            hard_current_stop_ma=600,
            max_travel_ticks=220,
            timeout_s=6.0,
        )
    )

    assert service.safety_guard.pretension_untensioned_reference_tick == 4010
    assert service.safety_guard.pretension_step_ticks == 4
    assert service.safety_guard.pretension_hard_current_stop_ma == 600
    assert controller.state.default_max_travel_ticks == 220
    assert "Hardware reconnect is not required" in controller.state.status_message


def test_servos_and_pretension_controllers_share_fresh_selected_servo_state(tmp_path: Path) -> None:
    settings = _settings()
    service = _pretension_service(tmp_path, dxl_bus=_MultiServoPretensionBus(current_sequences={2: [180, 230]}))
    service.connect("/dev/mock-openrb", 115200)
    servos_controller = ServosController(service, settings)
    pretension_controller = PretensionController(servo_service=service, settings=settings)

    servos_controller.set_selected_servo(2)
    servos_state = servos_controller.refresh_selected_servo()
    pretension_controller.set_selected_servo(2)
    pretension_state = pretension_controller.refresh()

    assert servos_state.selected_servo_telemetry_fresh is True
    assert pretension_state.selected_servo_telemetry_fresh is True
    assert servos_state.selected_servo_motion_ready is True
    assert pretension_state.selected_servo_pretension_ready is True
    assert servos_state.selected_servo_telemetry_age_s is not None
    assert pretension_state.selected_servo_telemetry_age_s is not None
    assert servos_state.selected_servo_telemetry_age_s < settings.safety.telemetry_stale_after_s
    assert pretension_state.selected_servo_telemetry_age_s < settings.safety.telemetry_stale_after_s


def test_pretension_controller_refresh_uses_cached_state_while_run_owns_bus(tmp_path: Path) -> None:
    settings = _settings()
    service = _pretension_service(tmp_path, dxl_bus=_MultiServoPretensionBus(current_sequences={2: [180, 230]}))
    service.connect("/dev/mock-openrb", 115200)
    controller = PretensionController(servo_service=service, settings=settings)
    controller.set_selected_servo(2)
    controller.state.pretension_running = True
    controller.state.selected_servo_position_tick = 4010
    controller.state.selected_servo_current_ma = 188
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
        state = controller.refresh()
        assert state.selected_servo_position_tick == 4010
        assert state.selected_servo_current_ma == 188
        assert "background refresh is paused" in state.selected_servo_block_reason
        assert state.can_stop is True
        assert state.can_start is False
    finally:
        release.set()
        thread.join(timeout=1.0)


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
        assert "owned by active pretension run on servo 2" in state.readiness_message
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
    assert "Motion ready 4/4" in state.readiness_message


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
    assert saved["robot_overrides"]["tightening_rotation_by_servo"]["1"] == "ccw"


def test_pretension_controller_saves_defaults_without_runtime_rebuild(tmp_path: Path) -> None:
    settings = _settings()
    service = _pretension_service(tmp_path)
    loader = _PretensionConfigLoader(tmp_path / "system.local.yaml")
    controller = PretensionController(servo_service=service, settings=settings, config_loader=loader)

    saved_path = controller.save_pretension_defaults(
        parameters=PretensionParameters(
            untensioned_reference_tick=4025,
            step_ticks=3,
            settle_time_s=0.15,
            baseline_sample_count=7,
            current_filter_window=5,
            current_delta_threshold_ma=80,
            absolute_trigger_current_ma=240,
            hard_current_stop_ma=610,
            max_travel_ticks=260,
            timeout_s=8.0,
        )
    )

    assert saved_path.endswith("system.local.yaml")
    assert loader.saved_overrides is not None
    assert loader.saved_overrides["safety_overrides"]["pretension_step_ticks"] == 3
    assert loader.saved_overrides["safety_overrides"]["pretension_hard_current_stop_ma"] == 610
    assert service.safety_guard.pretension_current_delta_threshold_ma == 80


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


def test_system_tab_surfaces_registration_runtime_tip_and_live_tip_summaries() -> None:
    _app()
    controller = _PortSelectionController()
    controller.state.registration_summary = "loaded | latest_registration.json | FRE=0.400 mm"
    controller.state.runtime_tip_summary = "accepted | latest_runtime_tip_calibration.json"
    controller.state.live_tip_summary = "ready | tip_pose_status=ok"
    tab = SystemTab(controller)

    tab.update(controller.state)

    assert "FRE=0.400 mm" in tab.registration_status_label.text()
    assert "latest_runtime_tip_calibration.json" in tab.runtime_tip_status_label.text()
    assert "tip_pose_status=ok" in tab.live_tip_status_label.text()


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
    tab.fine_jog_spin.setValue(9)
    tab.update(controller.state)

    assert tab.poll_rate_spin.value() == 24
    assert tab.fine_jog_spin.value() == 9


def test_system_tab_preserves_scrolled_diagnostics_position_on_update() -> None:
    _app()
    controller = _PortSelectionController()
    controller.state.status_message = "\n".join(f"line {index}" for index in range(80))
    controller.state.bench_debug_text = "\n".join(f"debug {index}" for index in range(80))
    tab = SystemTab(controller)
    tab.resize(900, 700)
    tab.show()

    tab.update(controller.state)
    QTest.qWait(20)
    scroll_bar = tab.status_text.verticalScrollBar()
    scroll_bar.setValue(max(1, scroll_bar.maximum() // 2))
    previous_value = scroll_bar.value()

    controller.state.status_message += "\nline 81"
    tab.update(controller.state)
    QTest.qWait(20)

    assert tab.status_text.verticalScrollBar().value() >= previous_value - 2


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


def test_app_window_refreshes_only_the_active_tab(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _app()
    window = AppWindow(_app_context(tmp_path))
    try:
        window._refresh_timer.stop()
        counts = {
            "tracker_mvp": 0,
            "system": 0,
            "servos": 0,
            "servos_selected": 0,
            "pretension": 0,
            "tracking": 0,
            "registration": 0,
            "experiment": 0,
        }

        def _wrap(name: str, result):
            def _inner():
                counts[name] += 1
                return result

            return _inner

        monkeypatch.setattr(
            window.tracker_mvp_controller,
            "refresh",
            _wrap("tracker_mvp", window.tracker_mvp_controller.state),
        )
        monkeypatch.setattr(window.system_controller, "refresh", _wrap("system", window.system_controller.state))
        monkeypatch.setattr(window.servos_controller, "refresh", _wrap("servos", window.servos_controller.state))
        monkeypatch.setattr(window.tracking_controller, "refresh", _wrap("tracking", window.tracking_controller.state))
        monkeypatch.setattr(
            window.registration_controller,
            "refresh",
            _wrap("registration", window.registration_controller.state),
        )
        monkeypatch.setattr(
            window.servos_controller,
            "refresh_selected_servo",
            _wrap("servos_selected", window.servos_controller.state),
        )
        monkeypatch.setattr(
            window.experiment_controller,
            "refresh_prerequisites",
            _wrap("experiment", window.experiment_controller.state),
        )
        monkeypatch.setattr(
            window.pretension_controller,
            "refresh",
            _wrap("pretension", window.pretension_controller.state),
        )

        window.tab_widget.setCurrentWidget(window.tracking_tab)
        counts = {key: 0 for key in counts}
        window.refresh()
        assert counts == {
            "tracker_mvp": 1,
            "system": 1,
            "servos": 0,
            "servos_selected": 0,
            "pretension": 0,
            "tracking": 1,
            "registration": 0,
            "experiment": 0,
        }

        window.tab_widget.setCurrentWidget(window.registration_tab)
        counts = {key: 0 for key in counts}
        window.refresh()
        assert counts == {
            "tracker_mvp": 1,
            "system": 1,
            "servos": 0,
            "servos_selected": 0,
            "pretension": 0,
            "tracking": 0,
            "registration": 1,
            "experiment": 0,
        }

        window.tab_widget.setCurrentWidget(window.servos_tab)
        counts = {key: 0 for key in counts}
        window.refresh()
        assert counts == {
            "tracker_mvp": 0,
            "system": 1,
            "servos": 0,
            "servos_selected": 1,
            "pretension": 0,
            "tracking": 0,
            "registration": 0,
            "experiment": 0,
        }

        window.tab_widget.setCurrentWidget(window.pretension_tab)
        counts = {key: 0 for key in counts}
        window.refresh()
        assert counts == {
            "tracker_mvp": 0,
            "system": 1,
            "servos": 0,
            "servos_selected": 0,
            "pretension": 1,
            "tracking": 0,
            "registration": 0,
            "experiment": 0,
        }

        window.tab_widget.setCurrentWidget(window.experiment_tab)
        counts = {key: 0 for key in counts}
        window.refresh()
        assert counts == {
            "tracker_mvp": 0,
            "system": 1,
            "servos": 0,
            "servos_selected": 0,
            "pretension": 0,
            "tracking": 0,
            "registration": 0,
            "experiment": 1,
        }
    finally:
        window.shutdown()


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


def test_servos_controller_saves_startup_calibration_and_accepts_pretension(tmp_path: Path) -> None:
    settings = _settings()
    settings.runtime.robot_config = "robot_1servo.yaml"
    settings.robot.mode = "1-servo"
    settings.robot.servo_ids = [1]
    settings.robot.tendon_to_servo = [1]
    settings.robot.tightening_rotation_by_servo = {1: "cw"}
    service = ServoService(
        dxl_bus=MockDxlBus([1]),
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
            path=tmp_path / "neutral_single.json",
            context=ServoCalibrationContext(
                robot_mode="1-servo",
                servo_ids=[1],
                tendon_to_servo=[1],
                position_min_offset_ticks=-600,
                position_max_offset_ticks=600,
                default_pretension_current_threshold_ma=220,
                tightening_rotation_by_servo={1: "cw"},
            ),
        ),
        pretension_validation=PretensionValidationService(),
    )
    service.connect("/dev/mock-openrb", 115200)
    service.dxl_bus._state[1].torque_enabled = True
    service.dxl_bus._state[1].present_position = 4031
    controller = ServosController(service, settings)

    controller.save_startup_calibration(
        servo_id=1,
        min_offset_ticks=-100,
        max_offset_ticks=120,
        threshold_ma=230,
    )
    controller.start_pretension(1, 120)
    controller._pretension_thread.join(timeout=1.0)
    controller.refresh()
    controller.accept_pretension_result(1)

    assert controller.state.calibration_rows[0]["threshold"] == "120"
    assert controller.state.pretension_result_can_accept is False
    assert "Accepted pretension result" in controller.state.status_message


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
    assert "repeatability_dataset" in option_names
    assert "pretension_validation" in option_names
    assert "command_schedule_validation" in option_names
    assert "collect_pose_command_dataset" in option_names


def test_experiment_workspace_preflight_warns_for_repeatability_without_registration(tmp_path: Path) -> None:
    controller = _experiment_controller(tmp_path)
    controller.select_experiment("repeatability_dataset")

    state = controller.refresh()

    assert state.preflight_report.overall_status == "ok_with_warning"
    assert any("Registration file is missing" in message for message in state.preflight_report.warning_messages)


def test_experiment_workspace_preflight_reads_registration_and_neutral_from_canonical_paths(tmp_path: Path) -> None:
    controller = _experiment_controller(tmp_path)
    controller.servo_service.save_neutral_setpoints({1: 2048, 2: 2048, 3: 2048, 4: 2048})
    controller.registration_path.write_text(
        json.dumps(
            {
                "fre_mm": 0.4,
                "validation_metrics": {
                    "overall_fre_mm": 0.4,
                    "max_residual_mm": 0.6,
                    "worst_landmark_label": "L2",
                    "worst_landmark_residual_mm": 0.6,
                    "configured_max_fre_mm": 1.0,
                },
            }
        ),
        encoding="utf-8",
    )
    controller.select_experiment("repeatability_dataset")

    state = controller.refresh()
    checks = {check.key: check for check in state.preflight_report.checks}

    assert checks["neutral_setpoints"].status == "ok"
    assert checks["registration"].status == "ok"
    assert "FRE=0.400 mm" in checks["registration"].message


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

    assert refreshed.preflight_report.overall_status == "ok_to_run"
    assert len(controller.get_config_value("captured_points", [])) == 3
    assert page.point_table.item(0, 2).text() == "2"
    assert "P04" in page.selected_point_label.text()
    assert any(label == "Coverage" for label, _value in refreshed.result_details)
    assert page.capture_summary_widget._pairs_signature is not None


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


def test_repeatability_page_uses_custom_schedule_preview_and_target_catalog(tmp_path: Path) -> None:
    _app()
    controller = _experiment_controller(tmp_path)
    tab = ExperimentTab(controller)
    controller.select_experiment("repeatability_dataset")
    state = controller.refresh()
    tab.update(state)
    page = tab._page_for("repeatability_dataset")

    assert page.target_set_combo.currentData() == "single_segment_ring_17"
    assert page.manual_target_label.isHidden() is True
    assert page.target_table.rowCount() == 17
    assert page.target_table.item(0, 0).text() == "Home"
    assert "single_segment_ring_17" in controller.refresh().config_text

    page.target_set_combo.setCurrentIndex(page.target_set_combo.findData("manual"))
    page.target_points_edit.setPlainText("- [0.0, 0.0, 0.0, 0.0]\n- [0.1, 0.0, 0.0, 0.0]")
    tab.update(controller.refresh())

    assert page.manual_target_label.isHidden() is False
    assert controller.get_config_value("schedule.target_set") == "manual"
    assert page.target_table.rowCount() == 2


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
        assert page.viewer_3d is not None
        assert page.viewer_3d.backend_mode == "placeholder"
    finally:
        controller.shutdown()


def test_repeatability_page_wraps_workspace_in_scroll_area_and_stacks_on_narrow_width(tmp_path: Path) -> None:
    _app()
    controller = _experiment_controller(tmp_path)
    tab = ExperimentTab(controller)
    controller.select_experiment("repeatability_dataset")
    tab.resize(920, 760)
    tab.show()
    tab.update(controller.refresh())
    QTest.qWait(20)
    page = tab._page_for("repeatability_dataset")

    assert isinstance(page.scroll_area, QScrollArea)
    assert page.scroll_area.widget() is not None
    assert page.top_row.direction() == QBoxLayout.TopToBottom
    assert page.bottom_row.direction() == QBoxLayout.TopToBottom

    tab.resize(1480, 900)
    QTest.qWait(20)

    assert page.top_row.direction() == QBoxLayout.LeftToRight
    assert page.bottom_row.direction() == QBoxLayout.LeftToRight


def test_repeatability_page_preserves_target_table_scroll_on_benign_refresh(tmp_path: Path) -> None:
    _app()
    controller = _experiment_controller(tmp_path)
    tab = ExperimentTab(controller)
    controller.select_experiment("repeatability_dataset")
    controller.set_config_value("schedule.target_set", "manual")
    controller.set_parameter_value(
        "schedule.target_points_cm",
        "\n".join(f"- [{0.01 * index:.3f}, 0.0, 0.0, 0.0]" for index in range(30)),
    )
    tab.resize(1100, 860)
    tab.show()
    tab.update(controller.refresh())
    QTest.qWait(20)
    page = tab._page_for("repeatability_dataset")

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
    experiment_controller.select_experiment("repeatability_dataset")
    experiment_tab.update(experiment_controller.refresh())
    repeatability_page = experiment_tab._page_for("repeatability_dataset")
    servos_tab = ServosTab(servos_controller)
    system_tab = SystemTab(system_controller)

    assert registration_tab.points_table.minimumHeight() >= 180
    assert registration_tab.samples_table.minimumHeight() >= 180
    assert registration_tab.landmark_map.minimumHeight() >= 200
    assert experiment_tab.experiment_combo.minimumHeight() >= 36
    assert repeatability_page.parameter_scroll.minimumWidth() >= 300
    assert repeatability_page.viewer_3d is not None
    assert repeatability_page.viewer_3d.minimumHeight() >= 280
    assert servos_tab.telemetry_table.minimumHeight() >= 190
    assert servos_tab.calibration_table.minimumHeight() >= 160
    assert system_tab.config_summary.minimumHeight() >= 170


def test_registration_servos_and_pretension_tabs_stack_splitters_on_narrow_width(tmp_path: Path) -> None:
    _app()
    settings = _settings()
    tracking_service = _tracking_service(settings, tmp_path)
    servo_service = _servo_service(tmp_path)
    registration_service = _registration_service(settings, tmp_path, tracking_service)
    registration_controller = RegistrationController(
        registration_service=registration_service,
        registration_config=settings.registration,
    )
    servos_controller = ServosController(servo_service, settings)
    pretension_controller = PretensionController(servo_service=servo_service, settings=settings)

    registration_tab = RegistrationTab(registration_controller)
    servos_tab = ServosTab(servos_controller)
    pretension_tab = PretensionTab(pretension_controller)

    registration_tab.resize(920, 860)
    registration_tab.show()
    servos_tab.resize(920, 860)
    servos_tab.show()
    pretension_tab.resize(920, 860)
    pretension_tab.show()
    QTest.qWait(20)

    assert registration_tab.top_splitter.orientation() == Qt.Vertical
    assert registration_tab.lower_splitter.orientation() == Qt.Vertical
    assert servos_tab.workspace_splitter.orientation() == Qt.Vertical
    assert pretension_tab.workspace_splitter.orientation() == Qt.Vertical

    registration_tab.resize(1500, 900)
    servos_tab.resize(1500, 900)
    pretension_tab.resize(1500, 900)
    QTest.qWait(20)

    assert registration_tab.top_splitter.orientation() == Qt.Horizontal
    assert registration_tab.lower_splitter.orientation() == Qt.Horizontal
    assert servos_tab.workspace_splitter.orientation() == Qt.Horizontal
    assert pretension_tab.workspace_splitter.orientation() == Qt.Horizontal


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


def test_servos_and_pretension_share_selected_servo_torque_truth(tmp_path: Path) -> None:
    settings = _settings()
    service = _servo_service(tmp_path)
    service.connect("/dev/mock-openrb", 115200)
    service.dxl_bus._state[2].torque_enabled = False

    servos_controller = ServosController(service, settings)
    pretension_controller = PretensionController(servo_service=service, settings=settings)
    servos_controller.set_selected_servo(2)
    pretension_controller.set_selected_servo(2)

    assert servos_controller.state.selected_servo_torque_enabled is False
    assert pretension_controller.state.selected_servo_torque_enabled is False

    service.dxl_bus._state[2].torque_enabled = True
    servos_controller.refresh_selected_servo()
    pretension_controller.refresh()

    assert servos_controller.state.selected_servo_torque_enabled is True
    assert pretension_controller.state.selected_servo_torque_enabled is True


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
    assert tab.samples_table.rowCount() == 1


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

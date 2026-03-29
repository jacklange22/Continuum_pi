import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from pathlib import Path
import json

import numpy as np
from PySide6.QtWidgets import QApplication

from continuum_robot.config.schemas import RegistrationWorkflowConfig
from continuum_robot.config.settings import Settings
from continuum_robot.config.schemas import (
    CalibrationConfig,
    ExperimentConfig,
    RobotConfig,
    RuntimeConfig,
    SafetyConfig,
    SerialConfig,
)
from continuum_robot.gui.controllers.registration_controller import RegistrationController
from continuum_robot.gui.controllers.servos_controller import ServosController
from continuum_robot.gui.controllers.system_controller import SystemController
from continuum_robot.gui.controllers.experiment_controller import ExperimentController
from continuum_robot.gui.tabs.experiment_tab import ExperimentTab
from continuum_robot.experiments.experiment_loader import ExperimentLoader
from continuum_robot.experiments.experiment_runner import ExperimentRunner
from continuum_robot.hardware.mock_dxl_bus import MockDxlBus
from continuum_robot.hardware.mock_openrb_client import MockOpenRbClient
from continuum_robot.registration.repository import RegistrationRepository
from continuum_robot.registration.rigid_solver import RigidRegistrationSolver
from continuum_robot.services.registration_service import RegistrationService
from continuum_robot.services.tracking_service import TrackingService
from continuum_robot.servos.displacement_mapper import TendonDisplacementMapper
from continuum_robot.servos.neutral_calibration_service import NeutralCalibrationService
from continuum_robot.servos.pretension_validation_service import PretensionValidationService
from continuum_robot.servos.safety_guard import SafetyGuard
from continuum_robot.servos.servo_service import ServoService
from continuum_robot.tracking.mock_tracker_manager import MockTrackerManager


def _app() -> QApplication:
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app


def _settings() -> Settings:
    return Settings(
        runtime=RuntimeConfig(mock_mode=True, poll_rate_hz=10, robot_config="robot_4servo.yaml"),
        robot=RobotConfig(mode="4-servo", spool_diameter_cm=1.2, ticks_per_revolution=4096, servo_ids=[1, 2, 3, 4]),
        serial=SerialConfig(aurora_port="/dev/mock-aurora", openrb_port="/dev/mock-openrb", baudrate=115200),
        safety=SafetyConfig(
            position_min_offset_ticks=-600,
            position_max_offset_ticks=600,
            max_current_ma=850,
            pretension_current_balance_tolerance_ma=120,
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
            capture_tool_id="0B",
            max_fre_mm=None,
        ),
        experiment=ExperimentConfig(default_settle_time_s=0.0, sample_count_per_point=1, output_dir="data/runs"),
        calibration=CalibrationConfig(
            neutral_setpoints_path="data/calibrations/neutral_setpoints.json",
            latest_registration_path="data/registrations/latest_registration.json",
        ),
    )


def _servo_service(tmp_path: Path) -> ServoService:
    return ServoService(
        dxl_bus=MockDxlBus([1, 2, 3, 4]),
        mapper=TendonDisplacementMapper(spool_diameter_cm=1.2),
        safety_guard=SafetyGuard(min_offset_ticks=-600, max_offset_ticks=600, max_current_ma=850),
        neutral_calibration=NeutralCalibrationService(path=tmp_path / "neutral.json"),
        pretension_validation=PretensionValidationService(),
    )


def _tracking_service(settings: Settings, tmp_path: Path) -> TrackingService:
    return TrackingService(
        live_backend=MockTrackerManager(poll_hz=10),
        port=settings.serial.aurora_port,
        registration_path=tmp_path / "latest_registration.json",
        config_source="test",
        runtime_coil_tool_id=settings.registration.coil_tool_id,
        registration_tool_id=settings.registration.capture_tool_id,
    )


def _registration_service(settings: Settings, tmp_path: Path, tracking_service: TrackingService) -> RegistrationService:
    config_path = tmp_path / "registration.yaml"
    config_path.write_text(
        "\n".join(
            [
                "landmark_labels: [L1, L2, L3, L4]",
                "captures_per_landmark: 1",
                "capture_tool_id: \"0B\"",
                "coil_tool_id: \"0A\"",
                "nominal_landmarks_robot_xyz_mm:",
                "  L1: [5.0, 5.0, 0.0]",
                "  L2: [23.0, 5.0, 0.0]",
                "  L3: [5.0, 23.0, 0.0]",
                "  L4: [5.0, 5.0, 18.0]",
            ]
        ),
        encoding="utf-8",
    )
    return RegistrationService(
        tracking_service=tracking_service,
        repository=RegistrationRepository(root_dir=tmp_path),
        solver=RigidRegistrationSolver(),
        config_path=config_path,
        config_source=str(config_path),
    )


def _experiment_runner(settings: Settings, tmp_path: Path, tracking_service: TrackingService, servo_service: ServoService, registration_path: Path) -> ExperimentRunner:
    return ExperimentRunner(
        project_root=Path(__file__).resolve().parents[1],
        settings=settings,
        tracking_service=tracking_service,
        servo_service=servo_service,
        output_dir=tmp_path / "runs",
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
        assert state.dynamixel_connected is True
    finally:
        controller.disconnect_tracker()
        controller.disconnect_openrb()


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


def test_experiment_workspace_selection_binds_example_config(tmp_path: Path) -> None:
    controller = _experiment_controller(tmp_path)

    controller.select_experiment("pivot_calibration")
    state = controller.refresh()

    assert state.selected_experiment == "pivot_calibration"
    assert "input_path" in state.config_text
    assert state.experiment_title == "Pivot Calibration"


def test_experiment_workspace_preflight_warns_for_repeatability_without_registration(tmp_path: Path) -> None:
    controller = _experiment_controller(tmp_path)
    controller.select_experiment("repeatability_dataset")

    state = controller.refresh()

    assert state.preflight_report.overall_status == "ok_with_warning"
    assert any("Registration file is missing" in message for message in state.preflight_report.warning_messages)


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


def test_experiment_workspace_requires_confirmation_before_pivot_overwrite(tmp_path: Path) -> None:
    controller = _experiment_controller(tmp_path)
    input_csv = tmp_path / "pivot.csv"
    input_csv.write_text(
        "\n".join(
            [
                "0B,1,0,0,0,15,-30,-60",
                "0B,0,1,0,0,15,10,140",
                "0B,0,0,1,0,35,-30,140",
                "0B,0,0,0,1,35,10,-60",
                "0B,0.70710678,0.70710678,0,0,15,90,20",
                "0B,0.70710678,-0.70710678,0,0,15,-110,60",
                "0B,0.70710678,0,0.70710678,0,-75,-30,50",
                "0B,0.70710678,0,-0.70710678,0,125,-30,30",
            ]
        ),
        encoding="utf-8",
    )
    existing_tip = tmp_path / "tip.csv"
    existing_tip.write_text("0,0,100\n", encoding="utf-8")
    controller.select_experiment("pivot_calibration")
    controller.set_config_text(
        "\n".join(
            [
                "tool_id: \"0B\"",
                f"input_path: \"{input_csv}\"",
                f"output_tip_file: \"{existing_tip}\"",
                "min_samples: 8",
            ]
        )
    )

    state = controller.refresh()

    assert state.preflight_report.overall_status == "ok_with_warning"
    assert state.preflight_report.requires_confirmation is True
    try:
        controller.run()
    except RuntimeError as exc:
        assert "overwrite confirmation" in str(exc)
    else:
        raise AssertionError("Expected overwrite confirmation error.")


def test_experiment_workspace_loads_prior_run_and_history(tmp_path: Path) -> None:
    controller = _experiment_controller(tmp_path)
    result = controller.experiment_runner.run_experiment(
        "pivot_calibration",
        config={
            "tool_id": "0B",
            "input_path": "data/examples/pivot_calibration_sample.csv",
            "output_tip_file": str(tmp_path / "generated_tip.csv"),
            "min_samples": 8,
        },
        output_dir=tmp_path / "runs",
        output_dir_name="saved_pivot_run",
    )
    assert result.success is True

    state = controller.refresh()
    assert any("saved_pivot_run" in entry.path for entry in state.history)

    controller.load_run(result.paths.output_dir)
    loaded = controller.refresh()
    assert loaded.loaded_run_path == str(result.paths.output_dir)
    assert loaded.selected_experiment == "pivot_calibration"
    assert loaded.visualization_model.summary_lines


def test_experiment_workspace_tab_updates_without_crashing_in_mock_mode(tmp_path: Path) -> None:
    _app()
    controller = _experiment_controller(tmp_path)
    tab = ExperimentTab(controller)
    try:
        tab.update(controller.refresh())
        controller.select_experiment("aurora_grid_accuracy")
        tab.update(controller.refresh())
        assert tab.viewer_3d.backend_mode == "placeholder"
    finally:
        controller.shutdown()

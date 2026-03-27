from pathlib import Path
import json

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
from continuum_robot.experiments.dat_writer import DatRunWriter
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
import numpy as np


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
            landmark_labels=["L1", "L2", "L3"],
            captures_per_landmark=1,
            nominal_landmarks_robot_xyz_mm={
                "L1": [0.0, 0.0, 0.0],
                "L2": [30.0, 0.0, 0.0],
                "L3": [0.0, 30.0, 0.0],
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
                "landmark_labels: [L1, L2, L3]",
                "captures_per_landmark: 1",
                "capture_tool_id: \"0B\"",
                "coil_tool_id: \"0A\"",
                "nominal_landmarks_robot_xyz_mm:",
                "  L1: [5.0, 5.0, 0.0]",
                "  L2: [23.0, 5.0, 0.0]",
                "  L3: [5.0, 23.0, 0.0]",
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
        controller.capture_current_label_sample()
        controller.capture_current_label_sample()
        result = controller.finish_session()
    finally:
        tracking_service.stop()

    assert result.output_path.exists()
    assert controller.state.fre_mm is not None
    assert controller.state.current_label is None


def test_experiment_controller_requires_tracker_and_servo_ready(tmp_path: Path) -> None:
    settings = _settings()
    servo_service = _servo_service(tmp_path)
    tracker_manager = MockTrackerManager(poll_hz=10)
    registration_path = tmp_path / "latest_registration.json"
    registration_path.write_text(
        json.dumps({"T_robot_aurora": np.eye(4).tolist(), "T_coil_tip": np.eye(4).tolist()}),
        encoding="utf-8",
    )
    servo_service.save_neutral_setpoints({1: 2048, 2: 2048, 3: 2048, 4: 2048})
    experiment_csv = tmp_path / "points.csv"
    experiment_csv.write_text("index,dl_1,dl_2,dl_3,dl_4\n0,0,0,0,0\n", encoding="utf-8")
    controller = ExperimentController(
        experiment_loader=ExperimentLoader(),
        experiment_runner=ExperimentRunner(
            servo_service=servo_service,
            tracker_manager=tracker_manager,
            dat_writer=DatRunWriter(tmp_path / "runs"),
            neutral_servo_ids=[1, 2, 3, 4],
            default_settle_time_s=0.0,
            registration_path=registration_path,
            sleep_fn=lambda _seconds: None,
        ),
        registration_path=registration_path,
        servo_service=servo_service,
        tracker_manager=tracker_manager,
    )
    controller.load_file(experiment_csv)

    state = controller.refresh_prerequisites()
    assert state.prerequisites_ok is False
    assert "OpenRB/DYNAMIXEL connection" in state.prerequisite_message
    assert "tracker connection" in state.prerequisite_message

    servo_service.connect("/dev/mock-openrb", 115200)
    tracker_manager.start()
    try:
        state = controller.refresh_prerequisites()
        assert state.prerequisites_ok is True
    finally:
        tracker_manager.stop()

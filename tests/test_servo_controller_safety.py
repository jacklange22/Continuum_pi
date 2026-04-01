from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from continuum_robot.config.schemas import (
    CalibrationConfig,
    ExperimentConfig,
    RegistrationWorkflowConfig,
    RobotConfig,
    RuntimeConfig,
    SafetyConfig,
    SerialConfig,
)
from continuum_robot.config.settings import Settings
from continuum_robot.gui.controllers.servos_controller import ServosController
from continuum_robot.gui.controllers.system_controller import SystemController
from continuum_robot.hardware.mock_dxl_bus import MockDxlBus
from continuum_robot.hardware.mock_openrb_client import MockOpenRbClient
from continuum_robot.servos.displacement_mapper import TendonDisplacementMapper
from continuum_robot.servos.neutral_calibration_service import (
    NeutralCalibrationService,
    ServoCalibrationContext,
)
from continuum_robot.servos.pretension_validation_service import PretensionValidationService
from continuum_robot.servos.safety_guard import SafetyGuard
from continuum_robot.servos.servo_service import ServoService


class _TrackingStub:
    def __init__(self) -> None:
        self._port = "/dev/mock-aurora"

    def set_port(self, port: str) -> None:
        self._port = port

    def start(self, _port: str | None = None) -> None:
        return None

    def stop(self) -> None:
        return None

    def get_snapshot(self):
        return SimpleNamespace(
            connection_state="disconnected",
            backend_identity="mock_tracker_manager",
            backend_running=False,
            bridge_running=False,
            backend_connected=False,
            socket_connected=False,
            last_error=None,
        )


def _settings() -> Settings:
    return Settings(
        runtime=RuntimeConfig(mock_mode=True, robot_config="robot_1servo.yaml"),
        robot=RobotConfig(
            mode="1-servo",
            servo_ids=[1],
            tendon_to_servo=[1],
            tightening_rotation_by_servo={1: "cw"},
        ),
        serial=SerialConfig(
            aurora_port="/dev/mock-aurora",
            openrb_port="/dev/mock-openrb",
            baudrate=115200,
        ),
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
            min_input_voltage_mv=4000,
        ),
        registration=RegistrationWorkflowConfig(),
        experiment=ExperimentConfig(),
        calibration=CalibrationConfig(
            neutral_setpoints_path="data/calibrations/neutral_setpoints.json",
            latest_registration_path="data/registrations/latest_registration.json",
        ),
    )


def _servo_service(tmp_path: Path) -> ServoService:
    return ServoService(
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
            min_input_voltage_mv=4000,
            time_fn=lambda: 0.0,
        ),
        neutral_calibration=NeutralCalibrationService(
            path=tmp_path / "neutral.json",
            context=ServoCalibrationContext(
                robot_mode="1-servo",
                robot_config_name="robot_1servo.yaml",
                servo_ids=[1],
                tendon_to_servo=[1],
                position_min_offset_ticks=-600,
                position_max_offset_ticks=600,
                default_pretension_current_threshold_ma=220,
                tightening_rotation_by_servo={1: "cw"},
            ),
        ),
        pretension_validation=PretensionValidationService(),
        sleep_fn=lambda _seconds: None,
        time_fn=lambda: 0.0,
    )


def test_system_controller_separates_bus_readiness_from_motion_readiness(tmp_path: Path) -> None:
    settings = _settings()
    servo_service = _servo_service(tmp_path)
    controller = SystemController(
        tracking_service=_TrackingStub(),
        openrb_client=MockOpenRbClient(),
        servo_service=servo_service,
        settings=settings,
    )

    controller.connect_openrb()

    assert controller.state.bus_reachable is True
    assert controller.state.motion_ready is False
    assert "saved safe bounds" in controller.state.readiness_message

    servo_service.capture_and_save_neutral_setpoints([1])
    controller.refresh_readiness()

    assert controller.state.bus_reachable is True
    assert controller.state.motion_ready is True
    assert "Calibrated motion: Ready for cautious motion." in controller.state.readiness_message


def test_servos_controller_reports_motion_blocking_until_neutral_capture(tmp_path: Path) -> None:
    settings = _settings()
    servo_service = _servo_service(tmp_path)
    servo_service.connect("/dev/mock-openrb", 115200)
    controller = ServosController(servo_service, settings)

    assert any("saved safe bounds" in reason for reason in controller.state.blocking_reasons)
    assert controller.state.telemetry[1]["ready"] != "ready"

    controller.capture_neutral_setpoints()
    controller.refresh()

    assert controller.state.blocking_reasons == []
    assert controller.state.telemetry[1]["ready"] == "ready"

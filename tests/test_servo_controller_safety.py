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
            baudrate=57600,
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


class _ExplodingReadBus(MockDxlBus):
    def read_telemetry(self, servo_ids: list[int]):
        raise RuntimeError("mock telemetry read failure")


def _servo_service(tmp_path: Path, *, dxl_bus=None, context_servo_ids: list[int] | None = None, robot_mode: str = "1-servo", calibration_path: Path | None = None) -> ServoService:
    if context_servo_ids is None:
        context_servo_ids = [1]
    if dxl_bus is None:
        dxl_bus = MockDxlBus(list(context_servo_ids))
    return ServoService(
        dxl_bus=dxl_bus,
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
            path=calibration_path or (tmp_path / "neutral.json"),
            context=ServoCalibrationContext(
                robot_mode=robot_mode,
                robot_config_name=f"robot_{robot_mode}.yaml",
                servo_ids=list(context_servo_ids),
                tendon_to_servo=list(context_servo_ids),
                position_min_offset_ticks=-600,
                position_max_offset_ticks=600,
                default_pretension_current_threshold_ma=220,
                tightening_rotation_by_servo={servo_id: "cw" for servo_id in context_servo_ids},
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
    assert "Motion status: ready" in controller.state.readiness_message


def test_servos_controller_reports_motion_blocking_until_neutral_capture(tmp_path: Path) -> None:
    settings = _settings()
    servo_service = _servo_service(tmp_path)
    servo_service.connect("/dev/mock-openrb", 57600)
    controller = ServosController(servo_service, settings)

    assert any("saved safe bounds" in reason for reason in controller.state.blocking_reasons)
    assert controller.state.telemetry[1]["ready"] != "ready"

    controller.capture_neutral_setpoints()
    controller.refresh()

    assert controller.state.blocking_reasons == []
    assert controller.state.telemetry[1]["ready"] == "ready"


def test_servo_service_bench_snapshot_marks_ping_only_when_readback_fails(tmp_path: Path) -> None:
    service = _servo_service(tmp_path, dxl_bus=_ExplodingReadBus([1]))
    service.connect("/dev/mock-openrb", 57600)

    snapshot = service.build_bench_debug_snapshot(1)

    assert snapshot.ping_ok is True
    assert snapshot.identity_read_ok is False
    assert snapshot.telemetry_read_ok is False
    assert snapshot.status == "ping_only"
    assert "mock telemetry read failure" in snapshot.message


def test_servos_controller_keeps_failure_row_when_readback_fails(tmp_path: Path) -> None:
    settings = _settings()
    service = _servo_service(tmp_path, dxl_bus=_ExplodingReadBus([1]))
    service.connect("/dev/mock-openrb", 57600)
    controller = ServosController(service, settings)

    assert list(controller.state.telemetry) == [1]
    assert controller.state.telemetry[1]["error"] is not None
    assert "mock telemetry read failure" in controller.state.telemetry[1]["error"]
    assert "telemetry_read_ok=False" in controller.state.bench_debug_text


def test_servos_controller_ignores_incompatible_multi_servo_calibration_in_one_servo_mode(tmp_path: Path) -> None:
    calibration_path = tmp_path / "neutral.json"
    four_servo_service = _servo_service(
        tmp_path,
        dxl_bus=MockDxlBus([1, 2, 3, 4]),
        context_servo_ids=[1, 2, 3, 4],
        robot_mode="4-servo",
        calibration_path=calibration_path,
    )
    four_servo_service.connect("/dev/mock-openrb", 57600)
    four_servo_service.capture_and_save_neutral_setpoints([1, 2, 3, 4])

    one_servo_service = _servo_service(
        tmp_path,
        dxl_bus=MockDxlBus([1]),
        context_servo_ids=[1],
        robot_mode="1-servo",
        calibration_path=calibration_path,
    )
    one_servo_service.connect("/dev/mock-openrb", 57600)
    controller = ServosController(one_servo_service, _settings())

    assert controller.state.neutral_setpoints == {}
    assert "multi-servo calibration artifact" in controller.state.calibration_message
    assert "calibration_entries_loaded=[1, 2, 3, 4]" in controller.state.bench_debug_text


def test_servos_controller_exposes_current_target_and_clamp_state_after_jog(tmp_path: Path) -> None:
    settings = _settings()
    service = _servo_service(tmp_path)
    service.connect("/dev/mock-openrb", 57600)
    service.dxl_bus._state[1].present_position = 2050
    service.save_startup_calibration(
        servo_id=1,
        min_offset_ticks=-10,
        max_offset_ticks=5,
        pretension_current_threshold_ma=220,
    )
    controller = ServosController(service, settings)

    controller.coarse_jog(1, -1)

    assert controller.state.selected_servo_current_position_tick == 2055
    assert controller.state.selected_servo_last_target_tick == 2055
    assert controller.state.selected_servo_last_unclamped_target_tick == 2075
    assert controller.state.selected_servo_last_motion_clamped is True
    assert "sent loosen" in controller.state.selected_servo_last_motion_summary

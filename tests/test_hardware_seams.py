from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from continuum_robot.app.bootstrap import build_app_context
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
from continuum_robot.hardware.dxl_bus import DxlBus, DxlBusConfig, ServoTelemetry
from continuum_robot.hardware.openrb_client import OpenRbClient
from continuum_robot.services.tracking_service import TrackingService
from continuum_robot.servos.displacement_mapper import TendonDisplacementMapper
from continuum_robot.servos.neutral_calibration_service import (
    NeutralCalibrationService,
    ServoCalibrationContext,
)
from continuum_robot.servos.pretension_validation_service import PretensionValidationService
from continuum_robot.servos.safety_guard import SafetyGuard
from continuum_robot.servos.servo_service import ServoService
from continuum_robot.tracking.mock_tracker_manager import MockTrackerManager


def _hardware_settings() -> Settings:
    return Settings(
        runtime=RuntimeConfig(mock_mode=False, poll_rate_hz=10, robot_config="robot_4servo.yaml"),
        robot=RobotConfig(
            mode="4-servo",
            spool_diameter_cm=1.2,
            ticks_per_revolution=4096,
            servo_ids=[1, 2, 3, 4],
            tightening_rotation_by_servo={1: "cw", 2: "cw", 3: "cw", 4: "cw"},
        ),
        serial=SerialConfig(
            aurora_port="/dev/mock-aurora",
            openrb_port="/dev/ttyUSB_OPENRB",
            baudrate=115200,
            openrb_settings={
                "connect_timeout_s": 0.01,
                "port_settle_time_s": 0.0,
            },
            dynamixel_settings={
                "protocol_version": 2.0,
                "auto_torque_enable_on_write": True,
                "control_table": {
                    "servo_id": 7,
                    "torque_enable": 64,
                    "hardware_error_status": 70,
                    "goal_position": 116,
                    "present_current": 126,
                    "present_position": 132,
                    "present_input_voltage": 144,
                },
            },
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
        ),
        registration=RegistrationWorkflowConfig(capture_tool_id="0B", coil_tool_id="0A", max_fre_mm=None),
        experiment=ExperimentConfig(default_settle_time_s=0.0, sample_count_per_point=1, output_dir="data/experiments"),
        calibration=CalibrationConfig(
            neutral_setpoints_path="config/neutral_setpoints.json",
            latest_registration_path="data/registrations/latest_registration.json",
        ),
    )


class _FakeSerial:
    def __init__(self, **kwargs) -> None:
        self.kwargs = dict(kwargs)
        self.closed = False

    def reset_input_buffer(self) -> None:
        return None

    def reset_output_buffer(self) -> None:
        return None

    def close(self) -> None:
        self.closed = True


def _make_fake_sdk() -> SimpleNamespace:
    state = {
        1: {
            0: 1240,
            6: 53,
            7: 1,
            11: 3,
            38: 2352,
            48: 4095,
            52: 0,
            64: 0,
            70: 0,
            98: 0,
            116: 2048,
            126: 150,
            132: 2048,
            144: 120,
            146: 32,
        },
        2: {
            0: 1240,
            6: 53,
            7: 2,
            11: 3,
            38: 2352,
            48: 4095,
            52: 0,
            64: 0,
            70: 0,
            98: 0,
            116: 2060,
            126: 165,
            132: 2060,
            144: 119,
            146: 33,
        },
    }
    writes: list[tuple[int, int, int]] = []

    class FakePortHandler:
        def __init__(self, port_name: str) -> None:
            self.port_name = port_name
            self.is_open = False
            self.baudrate = None

        def openPort(self) -> bool:
            self.is_open = True
            return True

        def setBaudRate(self, baudrate: int) -> bool:
            self.baudrate = int(baudrate)
            return True

        def closePort(self) -> None:
            self.is_open = False

    class FakePacketHandler:
        def __init__(self, protocol_version: float) -> None:
            self.protocol_version = protocol_version

        def ping(self, _port_handler, servo_id: int):
            if servo_id in state:
                return 1060, 0, 0
            return 0, 1, 0

        def read1ByteTxRx(self, _port_handler, servo_id: int, address: int):
            return self._read(servo_id, address)

        def read2ByteTxRx(self, _port_handler, servo_id: int, address: int):
            return self._read(servo_id, address)

        def read4ByteTxRx(self, _port_handler, servo_id: int, address: int):
            return self._read(servo_id, address)

        def write1ByteTxRx(self, _port_handler, servo_id: int, address: int, value: int):
            if servo_id not in state:
                return 1, 0
            writes.append((servo_id, address, int(value)))
            if address == 7:
                if value in state:
                    return 0, 1
                registers = dict(state.pop(servo_id))
                registers[7] = int(value)
                state[int(value)] = registers
                return 0, 0
            state[servo_id][address] = int(value)
            return 0, 0

        def write2ByteTxRx(self, _port_handler, servo_id: int, address: int, value: int):
            if servo_id not in state:
                return 1, 0
            writes.append((servo_id, address, int(value)))
            state[servo_id][address] = int(value)
            return 0, 0

        def write4ByteTxRx(self, _port_handler, servo_id: int, address: int, value: int):
            if servo_id not in state:
                return 1, 0
            writes.append((servo_id, address, int(value)))
            state[servo_id][address] = int(value)
            if address == 116:
                state[servo_id][132] = int(value)
                state[servo_id][126] = 180
                state[servo_id][144] = 118
                state[servo_id][146] = 35
            return 0, 0

        @staticmethod
        def getTxRxResult(comm_result: int) -> str:
            return f"comm_result={comm_result}"

        @staticmethod
        def getRxPacketError(packet_error: int) -> str:
            return f"packet_error={packet_error}"

        @staticmethod
        def _read(servo_id: int, address: int):
            if servo_id not in state or address not in state[servo_id]:
                return 0, 1, 0
            return state[servo_id][address], 0, 0

    return SimpleNamespace(
        COMM_SUCCESS=0,
        PortHandler=FakePortHandler,
        PacketHandler=lambda protocol_version: FakePacketHandler(protocol_version),
        _state=state,
        _writes=writes,
    )


def _servo_service(tmp_path: Path, sdk) -> ServoService:
    return ServoService(
        dxl_bus=DxlBus(config=_hardware_settings().serial.dynamixel_settings, sdk_loader=lambda: sdk),
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


def _tracking_service(settings: Settings, tmp_path: Path) -> TrackingService:
    return TrackingService(
        live_backend=MockTrackerManager(poll_hz=10),
        port=settings.serial.aurora_port,
        registration_path=tmp_path / "latest_registration.json",
        config_source="test",
        runtime_coil_tool_id=settings.registration.coil_tool_id,
        registration_tool_id=settings.registration.capture_tool_id,
    )


def test_real_dxl_bus_requires_sdk_for_hardware_access() -> None:
    def _missing_sdk():
        raise RuntimeError("Install dynamixel-sdk first.")

    bus = DxlBus(sdk_loader=_missing_sdk)
    with pytest.raises(RuntimeError, match="Install dynamixel-sdk"):
        bus.connect("/dev/ttyUSB0", 115200)


def test_dxl_bus_config_preserves_single_segment_defaults_when_payload_is_partial() -> None:
    config = DxlBusConfig.from_dict({"protocol_version": 2.0})

    assert config.single_segment_experiment_preferred_operating_mode == 3
    assert config.single_segment_experiment_allowed_operating_modes == [3]
    assert config.single_segment_experiment_default_goal_current_ma is None
    assert config.single_segment_experiment_default_profile_velocity is None
    assert config.single_segment_experiment_default_profile_acceleration is None
    assert config.single_segment_current_aware_preferred_operating_mode == 5
    assert config.single_segment_current_aware_allowed_operating_modes == [3, 5]
    assert config.single_segment_current_aware_default_goal_current_ma == 850
    assert config.single_segment_current_aware_default_profile_velocity == 80
    assert config.single_segment_current_aware_default_profile_acceleration == 20


def test_dxl_bus_config_allows_explicit_null_single_segment_motion_defaults() -> None:
    config = DxlBusConfig.from_dict(
        {
            "single_segment_current_aware_default_goal_current_ma": None,
            "single_segment_current_aware_default_profile_velocity": None,
            "single_segment_current_aware_default_profile_acceleration": None,
        }
    )

    assert config.single_segment_current_aware_default_goal_current_ma is None
    assert config.single_segment_current_aware_default_profile_velocity is None
    assert config.single_segment_current_aware_default_profile_acceleration is None


def test_dxl_bus_config_preserves_legacy_single_segment_aliases_for_current_aware_profile() -> None:
    config = DxlBusConfig.from_dict(
        {
            "single_segment_preferred_operating_mode": 5,
            "single_segment_allowed_operating_modes": [3, 5],
            "single_segment_default_goal_current_ma": 700,
            "single_segment_default_profile_velocity": 90,
            "single_segment_default_profile_acceleration": 30,
        }
    )

    assert config.single_segment_current_aware_preferred_operating_mode == 5
    assert config.single_segment_current_aware_allowed_operating_modes == [3, 5]
    assert config.single_segment_current_aware_default_goal_current_ma == 700
    assert config.single_segment_current_aware_default_profile_velocity == 90
    assert config.single_segment_current_aware_default_profile_acceleration == 30
    assert config.single_segment_experiment_preferred_operating_mode == 3


def test_openrb_client_validates_port_and_reports_status() -> None:
    client = OpenRbClient(
        config={"connect_timeout_s": 0.01, "port_settle_time_s": 0.0},
        serial_factory=_FakeSerial,
        sleep_fn=lambda _seconds: None,
    )

    client.connect("/dev/ttyUSB_OPENRB", 115200)
    prepared = client.prepare_for_dynamixel_use()

    assert client.is_connected is True
    assert prepared is True
    assert "DYNAMIXEL pass-through" in client.last_status


def test_dxl_bus_fake_sdk_scans_reads_and_writes() -> None:
    sdk = _make_fake_sdk()
    bus = DxlBus(
        config={"voltage_scale_mv_per_unit": 100.0, "current_scale_ma_per_unit": 1.0},
        sdk_loader=lambda: sdk,
    )

    bus.connect("/dev/ttyUSB_OPENRB", 115200)
    assert bus.scan_ids(1, 5) == [1, 2]

    telemetry = bus.read_telemetry([1, 2, 99])
    assert telemetry[1].model_number == 1240
    assert telemetry[1].firmware_version == 53
    assert telemetry[1].operating_mode == 3
    assert telemetry[1].present_position == 2048
    assert telemetry[1].present_current_ma == 150
    assert telemetry[1].present_voltage_mv == 12000
    assert telemetry[1].present_temperature_c == 32
    assert telemetry[1].hardware_error is None
    assert telemetry[2].present_voltage_mv == 11900
    assert telemetry[99].hardware_error is not None

    bus.write_goal_positions({1: 2100})
    assert bus.read_telemetry([1])[1].present_position == 2100

    bus.write_operating_mode(1, 5)
    bus.write_goal_current_ma(1, 850)
    bus.write_profile_velocity(1, 80)
    bus.write_profile_acceleration(1, 20)
    assert bus.read_telemetry([1])[1].operating_mode == 5

    bus.write_servo_id(2, 5)
    assert bus.scan_ids(1, 6) == [1, 5]
    assert (2, 64, 0) in sdk._writes
    assert (2, 7, 5) in sdk._writes
    assert (1, 11, 5) in sdk._writes
    assert (1, 102, 850) in sdk._writes
    assert (1, 112, 80) in sdk._writes
    assert (1, 108, 20) in sdk._writes


def test_dxl_bus_goal_write_pushes_default_motion_profile_registers() -> None:
    sdk = _make_fake_sdk()
    bus = DxlBus(
        config={
            "default_profile_velocity": 3,
            "default_profile_acceleration": 1,
        },
        sdk_loader=lambda: sdk,
    )

    bus.connect("/dev/ttyUSB_OPENRB", 115200)
    bus.write_goal_positions({1: 2100})

    assert (1, 108, 1) in sdk._writes
    assert (1, 112, 3) in sdk._writes
    assert (1, 116, 2100) in sdk._writes
    assert sdk._writes.index((1, 108, 1)) < sdk._writes.index((1, 112, 3))
    assert sdk._writes.index((1, 112, 3)) < sdk._writes.index((1, 116, 2100))
    assert bus.read_profile_acceleration(1) == 1
    assert bus.read_profile_velocity(1) == 3


def test_system_and_servo_controllers_use_hardware_seams(tmp_path: Path) -> None:
    settings = _hardware_settings()
    sdk = _make_fake_sdk()
    tracking_service = _tracking_service(settings, tmp_path)
    servo_service = _servo_service(tmp_path, sdk)
    openrb_client = OpenRbClient(
        config=settings.serial.openrb_settings,
        serial_factory=_FakeSerial,
        sleep_fn=lambda _seconds: None,
    )
    system_controller = SystemController(
        tracking_service=tracking_service,
        openrb_client=openrb_client,
        servo_service=servo_service,
        settings=settings,
    )
    servos_controller = ServosController(servo_service=servo_service, settings=settings)

    system_controller.connect_openrb()
    state = system_controller.refresh()
    assert state.openrb_connected is True
    assert state.openrb_prepared is True
    assert state.dynamixel_connected is True
    assert "DYNAMIXEL pass-through" in state.openrb_status

    scanned = servos_controller.scan()
    assert scanned == [1, 2]

    telemetry = servo_service.read_telemetry(scanned)
    assert telemetry[1].present_current_ma == 150
    assert telemetry[1].present_voltage_mv == 12000

    servos_controller.assign_servo_id(2, 5)
    assert servos_controller.state.servo_ids == [1, 5]

    servos_controller.coarse_jog(1, 1)
    assert servos_controller.state.telemetry[1]["position"] == 2023
    assert "Sent tighten_coarse for servo 1" in servos_controller.state.status_message


def test_build_app_context_uses_absolute_registration_path() -> None:
    context = build_app_context()
    registration_path = context.project_root / context.settings.calibration.latest_registration_path
    assert registration_path.is_absolute()

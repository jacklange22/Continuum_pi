from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from continuum_robot.config.config_loader import ConfigLoader
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


def _settings_4servo() -> Settings:
    return Settings(
        runtime=RuntimeConfig(mock_mode=True, robot_config="robot_4servo.yaml"),
        robot=RobotConfig(
            mode="4-servo",
            servo_ids=[1, 2, 3, 4],
            tendon_to_servo=[1, 2, 3, 4],
            tightening_rotation_by_servo={1: "cw", 2: "cw", 3: "cw", 4: "cw"},
        ),
        serial=SerialConfig(
            aurora_port="/dev/mock-aurora",
            openrb_port="/dev/ttyACM0",
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
    def read_telemetry(self, servo_ids: list[int], **kwargs):
        raise RuntimeError("mock telemetry read failure")


class _TrackingReadBus(MockDxlBus):
    def __init__(self, servo_ids: list[int]) -> None:
        super().__init__(servo_ids)
        self.live_read_calls: list[list[int]] = []
        self.read_calls: list[tuple[list[int], dict]] = []

    def read_live_telemetry(self, servo_ids: list[int]):
        self.live_read_calls.append([int(servo_id) for servo_id in servo_ids])
        return super().read_live_telemetry(servo_ids)

    def read_telemetry(self, servo_ids: list[int], **kwargs):
        self.read_calls.append(([int(servo_id) for servo_id in servo_ids], dict(kwargs)))
        return super().read_telemetry(servo_ids, **kwargs)


class _StaleControllerBus(MockDxlBus):
    def read_live_telemetry(self, servo_ids: list[int]):
        result = super().read_live_telemetry(servo_ids)
        for servo_id in servo_ids:
            result[int(servo_id)].last_read_monotonic_s = self._state[int(servo_id)].last_read_monotonic_s
        return result

    def read_telemetry(self, servo_ids: list[int], **kwargs):
        result = super().read_telemetry(servo_ids, **kwargs)
        for servo_id in servo_ids:
            result[int(servo_id)].last_read_monotonic_s = self._state[int(servo_id)].last_read_monotonic_s
        return result


def _servo_service(
    tmp_path: Path,
    *,
    dxl_bus=None,
    context_servo_ids: list[int] | None = None,
    robot_mode: str = "1-servo",
    calibration_path: Path | None = None,
    time_fn=None,
    telemetry_stale_after_s: float = 0.25,
) -> ServoService:
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
            telemetry_stale_after_s=telemetry_stale_after_s,
            pretension_step_ticks=2,
            pretension_timeout_s=2.0,
            pretension_settle_time_s=0.0,
            max_temperature_c=70,
            min_input_voltage_mv=4000,
            time_fn=time_fn or (lambda: 0.0),
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
        time_fn=time_fn or (lambda: 0.0),
    )


def test_system_controller_reports_motion_ready_in_single_servo_bench_mode_without_neutral_capture(tmp_path: Path) -> None:
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
    assert controller.state.motion_ready is True
    assert "Active range: 0..4095" in controller.state.readiness_message


def test_servos_controller_allows_raw_range_bench_motion_without_neutral_capture(tmp_path: Path) -> None:
    settings = _settings()
    servo_service = _servo_service(tmp_path)
    servo_service.connect("/dev/mock-openrb", 57600)
    controller = ServosController(servo_service, settings)

    assert controller.state.blocking_reasons == []
    assert controller.state.telemetry[1]["ready"] == "ready"
    assert controller.state.selected_servo_motion_ready is True
    assert controller.state.selected_servo_safe_min_tick == 0
    assert controller.state.selected_servo_safe_max_tick == 4095


def test_servos_controller_capture_neutral_status_explains_metadata_scope(tmp_path: Path) -> None:
    settings = _settings_4servo()
    servo_service = _servo_service(tmp_path, context_servo_ids=[1, 2, 3, 4], robot_mode="4-servo")
    servo_service.connect("/dev/mock-openrb", 57600)
    controller = ServosController(servo_service, settings)

    controller.capture_neutral_setpoints()

    assert "Captured neutral metadata" in controller.state.status_message
    assert "raw 0..4095 range" in controller.state.status_message
    assert controller.state.neutral_setpoints


def test_system_controller_syncs_servo_summary_from_servos_state(tmp_path: Path) -> None:
    settings = _settings_4servo()
    servo_service = _servo_service(tmp_path, context_servo_ids=[1, 2, 3, 4], robot_mode="4-servo")
    servo_service.connect("/dev/ttyACM0", 57600)
    servos_controller = ServosController(servo_service, settings)
    system_controller = SystemController(
        tracking_service=_TrackingStub(),
        openrb_client=MockOpenRbClient(),
        servo_service=servo_service,
        settings=settings,
    )

    system_controller.sync_servo_bringup_state(servos_controller.state)

    expected_motion_ready_count = sum(
        1 for row in servos_controller.state.telemetry.values() if row.get("motion_ready")
    )
    assert system_controller.state.detected_servo_ids == [1, 2, 3, 4]
    assert system_controller.state.telemetry_ready_count == 4
    assert system_controller.state.motion_ready_count == expected_motion_ready_count
    assert "Detected 4/4" in system_controller.state.readiness_message
    assert f"Motion ready {expected_motion_ready_count}/4" in system_controller.state.readiness_message


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
    service.dxl_bus._state[1].present_position = 4090
    controller = ServosController(service, settings)

    controller.coarse_jog(1, -1)

    assert controller.state.selected_servo_current_position_tick == 4095
    assert controller.state.selected_servo_last_target_tick == 4095
    assert controller.state.selected_servo_last_unclamped_target_tick == 4115
    assert controller.state.selected_servo_last_motion_clamped is True
    assert controller.state.selected_servo_last_result_label == "Clamped"
    assert controller.state.selected_servo_safe_min_tick == 0
    assert controller.state.selected_servo_safe_max_tick == 4095
    assert "sent loosen" in controller.state.selected_servo_last_motion_summary


def test_servos_controller_preserves_last_blocked_reason_for_boundary_jog(tmp_path: Path) -> None:
    settings = _settings()
    service = _servo_service(tmp_path)
    service.connect("/dev/mock-openrb", 57600)
    service.dxl_bus._state[1].present_position = 0
    controller = ServosController(service, settings)

    try:
        controller.fine_jog(1, 1)
    except RuntimeError:
        pass

    assert controller.state.selected_servo_last_result_label == "Blocked"
    assert "active minimum raw position 0" in controller.state.selected_servo_reason_label


def test_system_controller_reports_configured_4servo_readiness(tmp_path: Path) -> None:
    settings = _settings_4servo()
    servo_service = _servo_service(tmp_path, context_servo_ids=[1, 2, 3, 4], robot_mode="4-servo")
    controller = SystemController(
        tracking_service=_TrackingStub(),
        openrb_client=MockOpenRbClient(),
        servo_service=servo_service,
        settings=settings,
    )

    controller.connect_openrb()

    assert controller.state.bus_reachable is True
    assert controller.state.motion_ready is True
    assert "expected=[1, 2, 3, 4]" in controller.state.readiness_message
    assert "discovered_ids=[1, 2, 3, 4]" in controller.state.bench_debug_text


def test_servos_controller_reports_4servo_missing_and_unexpected_ids(tmp_path: Path) -> None:
    settings = _settings_4servo()
    service = _servo_service(
        tmp_path,
        dxl_bus=MockDxlBus([1, 2, 4, 9]),
        context_servo_ids=[1, 2, 3, 4],
        robot_mode="4-servo",
    )
    service.connect("/dev/ttyACM0", 57600)
    controller = ServosController(service, settings)

    controller.refresh_readiness()

    assert controller.state.detected_servo_ids == [1, 2, 4, 9]
    assert controller.state.missing_servo_ids == [3]
    assert controller.state.unexpected_servo_ids == [9]
    assert controller.state.telemetry[3]["error"] is not None

    controller.refresh()

    assert controller.state.unexpected_servo_ids == [9]
    assert controller.state.detected_servo_ids == [1, 2, 4, 9]


def test_servos_controller_four_servo_selection_and_jog_only_moves_selected_servo(tmp_path: Path) -> None:
    settings = _settings_4servo()
    service = _servo_service(
        tmp_path,
        dxl_bus=MockDxlBus([1, 2, 3, 4]),
        context_servo_ids=[1, 2, 3, 4],
        robot_mode="4-servo",
    )
    service.connect("/dev/ttyACM0", 57600)
    controller = ServosController(service, settings)
    before = {servo_id: telemetry.present_position for servo_id, telemetry in service.dxl_bus._state.items()}

    controller.set_selected_servo(3)
    controller.fine_jog(3, 1)

    assert controller.state.selected_servo_id == 3
    assert controller.state.selected_servo_last_action_label == "Tighten Fine"
    assert controller.state.selected_servo_last_result_label == "Sent"
    assert controller.state.selected_servo_safe_min_tick == 0
    assert controller.state.selected_servo_safe_max_tick == 4095
    assert service.dxl_bus._state[3].present_position == before[3] - 5
    assert service.dxl_bus._state[1].present_position == before[1]
    assert service.dxl_bus._state[2].present_position == before[2]
    assert service.dxl_bus._state[4].present_position == before[4]


def test_servos_controller_exposes_selected_servo_freshness_fields(tmp_path: Path) -> None:
    settings = _settings_4servo()
    service = _servo_service(
        tmp_path,
        dxl_bus=MockDxlBus([1, 2, 3, 4]),
        context_servo_ids=[1, 2, 3, 4],
        robot_mode="4-servo",
    )
    service.connect("/dev/ttyACM0", 57600)
    controller = ServosController(service, settings)

    controller.set_selected_servo(2)

    assert controller.state.selected_servo_id == 2
    assert controller.state.selected_servo_torque_enabled is False
    assert controller.state.selected_servo_telemetry_age_s is not None
    assert controller.state.selected_servo_last_read_label.endswith("ago")
    assert controller.state.selected_servo_telemetry_fresh is True
    assert controller.state.telemetry[2]["telemetry_age_s"] is not None
    assert controller.state.telemetry[2]["freshness_threshold_s"] == 0.25


def test_servos_controller_keeps_last_motion_state_per_servo(tmp_path: Path) -> None:
    settings = _settings_4servo()
    service = _servo_service(
        tmp_path,
        dxl_bus=MockDxlBus([1, 2, 3, 4]),
        context_servo_ids=[1, 2, 3, 4],
        robot_mode="4-servo",
    )
    service.connect("/dev/ttyACM0", 57600)
    controller = ServosController(service, settings)

    controller.set_selected_servo(3)
    controller.fine_jog(3, 1)
    controller.set_selected_servo(1)

    assert controller.state.selected_servo_id == 1
    assert controller.state.selected_servo_last_action_label == "None"
    assert controller.state.selected_servo_last_result_label == "None"
    assert controller.state.selected_servo_last_target_tick is None
    assert controller.state.selected_servo_last_motion_summary == "No jog command sent yet."


def test_servos_controller_uses_live_read_path_for_4servo_refresh(tmp_path: Path) -> None:
    settings = _settings_4servo()
    bus = _TrackingReadBus([1, 2, 3, 4])
    service = _servo_service(
        tmp_path,
        dxl_bus=bus,
        context_servo_ids=[1, 2, 3, 4],
        robot_mode="4-servo",
    )
    service.connect("/dev/ttyACM0", 57600)

    controller = ServosController(service, settings)

    assert bus.live_read_calls[-1] == [1, 2, 3, 4]
    assert all(call[1].get("include_reported_id") is False for call in bus.read_calls)
    assert all(call[1].get("include_identity") is False for call in bus.read_calls)
    assert all(call[1].get("include_limits") is False for call in bus.read_calls)


def test_servos_controller_formats_stale_telemetry_reason_honestly(tmp_path: Path) -> None:
    settings = _settings_4servo()
    bus = _StaleControllerBus([1, 2, 3, 4])
    for servo_id in [1, 2, 3, 4]:
        bus._state[servo_id].last_read_monotonic_s = 0.0
    service = _servo_service(
        tmp_path,
        dxl_bus=bus,
        context_servo_ids=[1, 2, 3, 4],
        robot_mode="4-servo",
        time_fn=lambda: 1.0,
        telemetry_stale_after_s=0.25,
    )
    service.connect("/dev/ttyACM0", 57600)

    controller = ServosController(service, settings)

    assert controller.state.selected_servo_telemetry_status_label == "Stale"
    assert controller.state.selected_servo_telemetry_fresh is False
    assert controller.state.selected_servo_telemetry_age_s == 1.0
    assert controller.state.selected_servo_reason_label == "Telemetry is stale (1.000 s > 0.250 s)."
    assert controller.state.telemetry[1]["block_reason"] == "Telemetry is stale (1.000 s > 0.250 s)."


def test_servos_controller_jog_does_not_trigger_full_table_refresh_reads(tmp_path: Path) -> None:
    settings = _settings_4servo()
    bus = _TrackingReadBus([1, 2, 3, 4])
    service = _servo_service(
        tmp_path,
        dxl_bus=bus,
        context_servo_ids=[1, 2, 3, 4],
        robot_mode="4-servo",
    )
    service.connect("/dev/ttyACM0", 57600)
    controller = ServosController(service, settings)
    bus.live_read_calls.clear()
    bus.read_calls.clear()

    controller.fine_jog(3, 1)

    assert bus.live_read_calls == [[3], [3]]
    assert len(bus.read_calls) == 2
    assert all(call[0] == [3] for call in bus.read_calls)
    assert all(call[1].get("include_reported_id") is False for call in bus.read_calls)
    assert all(call[1].get("include_identity") is False for call in bus.read_calls)
    assert all(call[1].get("include_limits") is False for call in bus.read_calls)


def test_servos_controller_refresh_selected_servo_uses_selected_only_live_read(tmp_path: Path) -> None:
    settings = _settings_4servo()
    bus = _TrackingReadBus([1, 2, 3, 4])
    service = _servo_service(
        tmp_path,
        dxl_bus=bus,
        context_servo_ids=[1, 2, 3, 4],
        robot_mode="4-servo",
    )
    service.connect("/dev/ttyACM0", 57600)
    controller = ServosController(service, settings)
    controller.set_selected_servo(4)
    bus.live_read_calls.clear()
    bus.read_calls.clear()

    controller.refresh_selected_servo()

    assert bus.live_read_calls == [[4]]
    assert len(bus.read_calls) == 1
    assert bus.read_calls[0][0] == [4]
    assert bus.read_calls[0][1].get("include_reported_id") is False
    assert bus.read_calls[0][1].get("include_identity") is False
    assert bus.read_calls[0][1].get("include_limits") is False


def test_system_controller_save_runtime_parameters_persists_poll_rate(tmp_path: Path) -> None:
    config_dir = tmp_path / "config"
    config_dir.mkdir()
    (config_dir / "system.yaml").write_text('robot_config: "robot_4servo.yaml"\npoll_rate_hz: 10\n', encoding="utf-8")
    (config_dir / "robot_4servo.yaml").write_text(
        "\n".join(
            [
                'mode: "4-servo"',
                "servo_ids: [1, 2, 3, 4]",
                "tendon_to_servo: [1, 2, 3, 4]",
                "tightening_rotation_by_servo: {1: cw, 2: cw, 3: cw, 4: cw}",
            ]
        ),
        encoding="utf-8",
    )
    loader = ConfigLoader(base_dir=config_dir)
    settings = _settings_4servo()
    controller = SystemController(
        tracking_service=_TrackingStub(),
        openrb_client=MockOpenRbClient(),
        servo_service=_servo_service(tmp_path, context_servo_ids=[1, 2, 3, 4], robot_mode="4-servo"),
        settings=settings,
        config_loader=loader,
    )

    saved_path = controller.save_runtime_parameters(
        mock_mode=False,
        robot_config="robot_4servo.yaml",
        openrb_port="/dev/ttyACM0",
        baudrate=57600,
        poll_rate_hz=20,
        fine_jog_step_ticks=4,
        coarse_jog_step_ticks=20,
        telemetry_freshness_timeout_s=0.2,
    )

    saved = loader.load_system_local_overrides()

    assert saved_path.endswith("system.local.yaml")
    assert saved["poll_rate_hz"] == 20
    assert saved["openrb_port"] == "/dev/ttyACM0"
    assert saved["safety_overrides"]["fine_jog_step_ticks"] == 4
    assert saved["safety_overrides"]["telemetry_stale_after_s"] == 0.2

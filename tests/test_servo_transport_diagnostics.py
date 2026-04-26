from __future__ import annotations

import json
from pathlib import Path
import threading

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
from continuum_robot.experiments.experiment_runner import ExperimentRunner
from continuum_robot.hardware.dxl_bus import ServoTelemetry
from continuum_robot.hardware.mock_dxl_bus import MockDxlBus
from continuum_robot.gui.controllers.servos_controller import ServosController
from continuum_robot.servos.displacement_mapper import TendonDisplacementMapper
from continuum_robot.servos.neutral_calibration_service import NeutralCalibrationService, ServoCalibrationContext
from continuum_robot.servos.pretension_validation_service import PretensionValidationService
from continuum_robot.servos.safety_guard import SafetyGuard
from continuum_robot.servos.servo_service import ServoService
from continuum_robot.servos.transport_diagnostics import (
    SOAK_COORDINATED,
    SOAK_ONE_SERVO,
    SOAK_STATIC,
    ServoTransportSoakConfig,
    classify_packet_failure,
    run_servo_transport_soak,
)


class _CountingBus(MockDxlBus):
    def __init__(self, servo_ids: list[int] | None = None) -> None:
        super().__init__(servo_ids or [1, 2, 3, 4])
        self.live_read_count = 0
        self.full_read_count = 0

    def read_live_telemetry(self, servo_ids: list[int]) -> dict[int, ServoTelemetry]:
        self.live_read_count += 1
        return super().read_live_telemetry(servo_ids)

    def read_telemetry(self, servo_ids: list[int], **kwargs) -> dict[int, ServoTelemetry]:
        self.full_read_count += 1
        return super().read_telemetry(servo_ids, **kwargs)


def _settings(tmp_path: Path, servo_ids: list[int] | None = None) -> Settings:
    ids = servo_ids or [1, 2, 3, 4]
    return Settings(
        runtime=RuntimeConfig(mock_mode=True, poll_rate_hz=5, robot_config="robot_4servo.yaml"),
        robot=RobotConfig(mode="4-servo", spool_diameter_cm=1.2, servo_ids=list(ids), tendon_to_servo=list(ids)),
        serial=SerialConfig(openrb_port="/dev/mock-openrb", baudrate=115200),
        safety=SafetyConfig(telemetry_stale_after_s=10.0),
        registration=RegistrationWorkflowConfig(),
        experiment=ExperimentConfig(output_dir=str(tmp_path / "data" / "experiments")),
        calibration=CalibrationConfig(neutral_setpoints_path=str(tmp_path / "neutral.json")),
    )


def _service(tmp_path: Path, *, bus: MockDxlBus | None = None, servo_ids: list[int] | None = None) -> ServoService:
    ids = servo_ids or [1, 2, 3, 4]
    service = ServoService(
        dxl_bus=bus or MockDxlBus(ids),
        mapper=TendonDisplacementMapper(spool_diameter_cm=1.2),
        safety_guard=SafetyGuard(
            min_offset_ticks=-600,
            max_offset_ticks=600,
            max_current_ma=850,
            telemetry_stale_after_s=10.0,
            min_input_voltage_mv=4000,
        ),
        neutral_calibration=NeutralCalibrationService(
            path=tmp_path / "neutral.json",
            context=ServoCalibrationContext(
                robot_mode="4-servo",
                robot_config_name="robot_4servo.yaml",
                servo_ids=list(ids),
                tendon_to_servo=list(ids),
                position_min_offset_ticks=-600,
                position_max_offset_ticks=600,
                tightening_rotation_by_servo={servo_id: "cw" for servo_id in ids},
            ),
        ),
        pretension_validation=PretensionValidationService(),
        sleep_fn=lambda _seconds: None,
    )
    service.connect("/dev/mock-openrb", 115200)
    service.capture_neutral_setpoints(list(ids))
    return service


def test_packet_failure_classification_is_explicit() -> None:
    assert classify_packet_failure(message="[TxRxResult] Incorrect status packet!") == "incorrect_status_packet"
    assert classify_packet_failure(message="[TxRxResult] There is no status packet!") == "no_status_packet"
    assert classify_packet_failure(telemetry=ServoTelemetry(servo_id=1, present_position=None), field="position") == "missing_position"
    assert classify_packet_failure(telemetry=ServoTelemetry(servo_id=1, present_current_ma=None), field="current") == "missing_current"
    assert classify_packet_failure(telemetry=ServoTelemetry(servo_id=1, present_voltage_mv=None), field="voltage") == "missing_voltage"
    assert classify_packet_failure(message="target outside active limit") == "safety_limit_rejected"


def test_static_soak_writes_clean_summary_bundle(tmp_path: Path) -> None:
    service = _service(tmp_path)
    result = run_servo_transport_soak(
        service,
        ServoTransportSoakConfig(mode=SOAK_STATIC, servo_ids=[1, 2, 3, 4], duration_s=0.0),
        output_root=tmp_path / "diagnostics",
        sleep_fn=lambda _seconds: None,
    )

    summary = json.loads((result.output_dir / "summary.json").read_text(encoding="utf-8"))
    assert summary["success"] is True
    assert summary["status"] == "success"
    assert summary["mode"] == SOAK_STATIC
    assert (result.output_dir / "metrics.csv").exists()
    assert (result.output_dir / "samples.jsonl").exists()
    assert (result.output_dir / "servo_transport_summary.txt").exists()
    assert (result.output_dir / "manifest.json").exists()
    assert (result.output_dir / "plots" / "failure_type_histogram.svg").exists()
    assert "bench supply current is not measured" in (result.output_dir / "servo_transport_summary.txt").read_text(encoding="utf-8")


def test_one_servo_motion_soak_summary_generation(tmp_path: Path) -> None:
    service = _service(tmp_path)
    result = run_servo_transport_soak(
        service,
        ServoTransportSoakConfig(mode=SOAK_ONE_SERVO, servo_ids=[1, 2, 3, 4], selected_servo_id=2, cycles=2, step_ticks=4),
        output_root=tmp_path / "diagnostics",
        sleep_fn=lambda _seconds: None,
    )

    summary = json.loads((result.output_dir / "summary.json").read_text(encoding="utf-8"))
    assert summary["mode"] == SOAK_ONE_SERVO
    assert summary["success"] is True
    assert summary["status"] == "success"
    assert summary["sample_count"] >= 2


def test_coordinated_micro_motion_soak_summary_generation(tmp_path: Path) -> None:
    service = _service(tmp_path)
    result = run_servo_transport_soak(
        service,
        ServoTransportSoakConfig(mode=SOAK_COORDINATED, servo_ids=[1, 2, 3, 4], cycles=2, coordinated_delta_ticks=2),
        output_root=tmp_path / "diagnostics",
        sleep_fn=lambda _seconds: None,
    )

    summary = json.loads((result.output_dir / "summary.json").read_text(encoding="utf-8"))
    assert summary["mode"] == SOAK_COORDINATED
    assert summary["status"] == "success"
    assert (result.output_dir / "plots" / "commanded_vs_measured_position.svg").exists()


def test_non_owner_live_bus_reads_are_blocked_during_active_soak(tmp_path: Path) -> None:
    bus = _CountingBus([1, 2, 3, 4])
    service = _service(tmp_path, bus=bus)
    service.read_live_telemetry([1, 2, 3, 4])
    before = bus.live_read_count
    acquired = threading.Event()
    release = threading.Event()

    def _owner() -> None:
        with service.exclusive_bus_operation(owner="test_soak", reason="unit_test"):
            acquired.set()
            release.wait(timeout=2.0)

    thread = threading.Thread(target=_owner)
    thread.start()
    acquired.wait(timeout=2.0)
    try:
        snapshot = service.build_cached_runtime_servo_snapshot([1, 2, 3, 4])
        assert snapshot.detected_servo_ids == [1, 2, 3, 4]
        assert bus.live_read_count == before
    finally:
        release.set()
        thread.join(timeout=2.0)


def test_servos_controller_uses_cached_state_when_bus_owned_by_soak(tmp_path: Path) -> None:
    bus = _CountingBus([1, 2, 3, 4])
    settings = _settings(tmp_path)
    service = _service(tmp_path, bus=bus)
    service.read_live_telemetry([1, 2, 3, 4])
    controller = ServosController(service, settings)
    before = bus.live_read_count + bus.full_read_count
    acquired = threading.Event()
    release = threading.Event()

    def _owner() -> None:
        with service.exclusive_bus_operation(owner="servo_transport_soak", reason="unit_test"):
            acquired.set()
            release.wait(timeout=2.0)

    thread = threading.Thread(target=_owner)
    thread.start()
    acquired.wait(timeout=2.0)
    try:
        controller.refresh()
        after = bus.live_read_count + bus.full_read_count
        assert after == before
        assert "cached" in controller.latest_runtime_snapshot.message.lower()
    finally:
        release.set()
        thread.join(timeout=2.0)


def test_experiment_summary_status_cannot_claim_success_when_run_failed(tmp_path: Path) -> None:
    service = _service(tmp_path)
    runner = ExperimentRunner(
        project_root=tmp_path,
        settings=_settings(tmp_path),
        tracking_service=None,
        servo_service=service,
        output_dir=tmp_path / "data" / "experiments",
        registration_path=tmp_path / "missing_registration.json",
    )

    result = runner.run_experiment("collect_pose_command_dataset", config={"dry_run": False, "sample_count_target": 1})
    assert result.success is False
    assert result.summary.success is False
    assert result.summary.status != "success"

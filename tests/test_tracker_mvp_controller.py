from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
import json
import time

import pytest

from continuum_robot.config.schemas import (
    CalibrationConfig,
    ExperimentConfig,
    RegistrationLandmarkConfig,
    RegistrationWorkflowConfig,
    RobotConfig,
    RuntimeConfig,
    SafetyConfig,
    SerialConfig,
)
from continuum_robot.config.settings import Settings
from continuum_robot.experiments.experiment_runner import ExperimentRunner
from continuum_robot.gui.controllers.registration_controller import RegistrationController
from continuum_robot.gui.controllers.tracker_mvp_controller import TrackerMvpController
from continuum_robot.hardware.mock_dxl_bus import MockDxlBus
from continuum_robot.registration.repository import RegistrationRepository
from continuum_robot.registration.rigid_solver import RigidRegistrationSolver
from continuum_robot.services.registration_service import RegistrationService
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


def _settings(project_root: Path, *, penprobe_file: str) -> Settings:
    return Settings(
        runtime=RuntimeConfig(mock_mode=True, poll_rate_hz=20, robot_config="robot_4servo.yaml"),
        robot=RobotConfig(
            mode="4-servo",
            spool_diameter_cm=1.2,
            ticks_per_revolution=4096,
            servo_ids=[1, 2, 3, 4],
            tendon_to_servo=[1, 2, 3, 4],
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
            coil_tool_id="0A",
            penprobe_file=penprobe_file,
            max_fre_mm=None,
        ),
        experiment=ExperimentConfig(
            default_settle_time_s=0.0,
            sample_count_per_point=1,
            output_dir=str(project_root / "runs"),
        ),
        calibration=CalibrationConfig(
            neutral_setpoints_path=str(project_root / "neutral.json"),
            latest_registration_path=str(project_root / "latest_registration.json"),
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


def _registration_config_path(project_root: Path, *, penprobe_file: str) -> Path:
    path = project_root / "registration.yaml"
    path.write_text(
        "\n".join(
            [
                "landmark_labels: [L1, L2, L3, L4]",
                "captures_per_landmark: 1",
                'capture_tool_id: "0B"',
                'coil_tool_id: "0A"',
                f'penprobe_file: "{penprobe_file}"',
                "nominal_landmarks_robot_xyz_mm:",
                "  L1: [0.0, 0.0, 0.0]",
                "  L2: [30.0, 0.0, 0.0]",
                "  L3: [0.0, 30.0, 0.0]",
                "  L4: [0.0, 0.0, 30.0]",
            ]
        ),
        encoding="utf-8",
    )
    return path


def _tracking_service(settings: Settings, tmp_path: Path) -> TrackingService:
    return TrackingService(
        live_backend=MockTrackerManager(poll_hz=20),
        port=settings.serial.aurora_port,
        registration_path=tmp_path / "latest_registration.json",
        config_source="test",
        runtime_coil_tool_id=settings.registration.coil_tool_id,
        registration_tool_id=settings.registration.capture_tool_id,
    )


def _build_runtime(tmp_path: Path, *, penprobe_file: str) -> tuple[TrackerMvpController, RegistrationController, TrackingService]:
    settings = _settings(tmp_path, penprobe_file=penprobe_file)
    tracking_service = _tracking_service(settings, tmp_path)
    registration_service = RegistrationService(
        tracking_service=tracking_service,
        repository=RegistrationRepository(root_dir=tmp_path / "registrations"),
        solver=RigidRegistrationSolver(),
        config_path=_registration_config_path(tmp_path, penprobe_file=penprobe_file),
        config_source="test",
    )
    registration_controller = RegistrationController(
        registration_service=registration_service,
        registration_config=settings.registration,
    )
    experiment_runner = ExperimentRunner(
        project_root=tmp_path,
        settings=settings,
        tracking_service=tracking_service,
        servo_service=_servo_service(tmp_path),
        output_dir=tmp_path / "runs",
        registration_path=tmp_path / "latest_registration.json",
        default_settle_time_s=0.0,
        sleep_fn=lambda _seconds: None,
    )
    controller = TrackerMvpController(
        tracking_service=tracking_service,
        registration_service=registration_service,
        registration_controller=registration_controller,
        experiment_runner=experiment_runner,
        settings=settings,
        project_root=tmp_path,
    )
    return controller, registration_controller, tracking_service


@dataclass
class _FakeValidationReport:
    tracker_ready: bool = True

    def to_dict(self) -> dict:
        return {"tracker_ready": self.tracker_ready}


def test_registration_begin_requires_ready_tip_file(tmp_path: Path) -> None:
    _controller, registration_controller, _tracking_service = _build_runtime(
        tmp_path,
        penprobe_file="data/tip_cals/generated_penprobe_tip.csv",
    )

    with pytest.raises(RuntimeError, match="tip file"):
        registration_controller.begin_session()


def test_tracker_mvp_validation_creates_artifact_and_updates_state(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    controller, _registration_controller, tracking_service = _build_runtime(
        tmp_path,
        penprobe_file="data/tip_cals/generated_penprobe_tip.csv",
    )
    tracking_service.start()

    monkeypatch.setattr(
        "continuum_robot.gui.controllers.tracker_mvp_controller.build_tracking_diagnostics_report",
        lambda *args, **kwargs: _FakeValidationReport(tracker_ready=True),
    )

    report_path = controller.validate_tracker()
    state = controller.refresh()

    assert report_path.exists()
    assert json.loads(report_path.read_text(encoding="utf-8"))["tracker_ready"] is True
    assert state.validation_passed is True
    assert state.tool_0b_visible is True


def test_tracker_mvp_blocks_pivot_without_tool_0b_visibility(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    controller, _registration_controller, tracking_service = _build_runtime(
        tmp_path,
        penprobe_file="data/tip_cals/generated_penprobe_tip.csv",
    )
    tracking_service.start()
    monkeypatch.setattr(
        "continuum_robot.gui.controllers.tracker_mvp_controller.build_tracking_diagnostics_report",
        lambda *args, **kwargs: _FakeValidationReport(tracker_ready=True),
    )
    controller.validate_tracker()
    snapshot = tracking_service.get_snapshot()
    snapshot.tools.pop("0B", None)
    snapshot.normalized_live_tool_ids = ["0A"]
    snapshot.raw_live_tool_ids = ["0A"]
    monkeypatch.setattr(tracking_service, "get_snapshot", lambda: snapshot)

    controller.refresh()

    with pytest.raises(RuntimeError, match="Tool 0B"):
        controller.run_pivot_calibration()


def test_tracker_mvp_pivot_run_updates_tip_geometry_and_status(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    controller, registration_controller, tracking_service = _build_runtime(
        tmp_path,
        penprobe_file="data/tip_cals/generated_penprobe_tip.csv",
    )
    tracking_service.start()
    monkeypatch.setattr(
        "continuum_robot.gui.controllers.tracker_mvp_controller.build_tracking_diagnostics_report",
        lambda *args, **kwargs: _FakeValidationReport(tracker_ready=True),
    )
    controller.validate_tracker()

    tip_path = tmp_path / "data" / "tip_cals" / "generated_penprobe_tip.csv"
    run_path = tmp_path / "runs" / "pivot_run"
    run_path.mkdir(parents=True, exist_ok=True)

    def _fake_run_experiment(*args, **kwargs):
        tip_path.parent.mkdir(parents=True, exist_ok=True)
        tip_path.write_text("+1.0000,+2.0000,+3.0000", encoding="utf-8")
        return SimpleNamespace(
            success=True,
            message="ok",
            paths=SimpleNamespace(output_dir=run_path),
            summary=SimpleNamespace(
                experiment_metrics={
                    "status": "success",
                    "rmse_mm": 0.42,
                    "sample_count_total": 80,
                    "sample_count_used": 76,
                    "sample_count_rejected": 4,
                }
            ),
        )

    monkeypatch.setattr(controller.experiment_runner, "run_experiment", _fake_run_experiment)

    output_dir = controller.run_pivot_calibration()
    state = controller.refresh()

    assert output_dir == run_path
    assert tip_path.exists()
    assert state.measurement_point_ready is True
    assert state.pivot_tip_preview == "+1.0000,+2.0000,+3.0000"
    assert state.pivot_rmse_mm == pytest.approx(0.42)
    registration_controller.begin_session()


def test_tracker_mvp_guided_pivot_collection_requires_accept_before_registration(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    controller, registration_controller, tracking_service = _build_runtime(
        tmp_path,
        penprobe_file="data/tip_cals/generated_penprobe_tip.csv",
    )
    tracking_service.start()
    monkeypatch.setattr(
        "continuum_robot.gui.controllers.tracker_mvp_controller.build_tracking_diagnostics_report",
        lambda *args, **kwargs: _FakeValidationReport(tracker_ready=True),
    )
    controller.validate_tracker()
    controller.PIVOT_MIN_SAMPLES = 2
    controller.PIVOT_SAMPLE_PERIOD_S = 0.001
    controller.start_pivot_collection()
    time.sleep(0.08)
    collecting_state = controller.refresh()
    assert collecting_state.pivot_collection_active is True
    assert collecting_state.pivot_live_sample_count >= 2

    controller.stop_pivot_collection()
    run_path = tmp_path / "runs" / "pivot_review"

    def _fake_run_experiment(*args, **kwargs):
        pending_tip_path = Path(kwargs["config"]["output_tip_file"])
        pending_tip_path.parent.mkdir(parents=True, exist_ok=True)
        pending_tip_path.write_text("+4.0000,+5.0000,+6.0000", encoding="utf-8")
        run_path.mkdir(parents=True, exist_ok=True)
        return SimpleNamespace(
            success=True,
            message="ok",
            paths=SimpleNamespace(output_dir=run_path),
            summary=SimpleNamespace(
                experiment_metrics={
                    "status": "success",
                    "rmse_mm": 0.33,
                    "sample_count_total": 12,
                    "sample_count_used": 11,
                    "sample_count_rejected": 1,
                }
            ),
        )

    monkeypatch.setattr(controller.experiment_runner, "run_experiment", _fake_run_experiment)
    controller.solve_pivot_collection()
    pending_state = controller.refresh()

    assert pending_state.pivot_pending_accept is True
    assert pending_state.registration_ready is False
    assert any("Accept or reset the staged pivot tip file" in blocker for blocker in pending_state.registration_blockers)

    controller.accept_pivot_tip_file()
    accepted_state = controller.refresh()
    assert accepted_state.pivot_pending_accept is False
    assert accepted_state.measurement_point_ready is True
    assert accepted_state.registration_ready is True

    registration_controller.begin_session()


def test_tracker_mvp_pivot_parse_failure_preserves_parse_report_for_operator_review(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    controller, _registration_controller, tracking_service = _build_runtime(
        tmp_path,
        penprobe_file="data/tip_cals/generated_penprobe_tip.csv",
    )
    tracking_service.start()
    monkeypatch.setattr(
        "continuum_robot.gui.controllers.tracker_mvp_controller.build_tracking_diagnostics_report",
        lambda *args, **kwargs: _FakeValidationReport(tracker_ready=True),
    )
    controller.validate_tracker()
    controller.PIVOT_MIN_SAMPLES = 2
    controller.PIVOT_SAMPLE_PERIOD_S = 0.001
    controller.start_pivot_collection()
    time.sleep(0.08)
    controller.stop_pivot_collection()

    failed_run_path = tmp_path / "runs" / "pivot_failed"

    def _fake_failed_run(*args, **kwargs):
        failed_run_path.mkdir(parents=True, exist_ok=True)
        return SimpleNamespace(
            success=False,
            message="Experiment pivot_calibration failed: CSV header detected but required columns missing: qz",
            paths=SimpleNamespace(output_dir=failed_run_path),
            summary=SimpleNamespace(
                experiment_metrics={
                    "pivot_input_format": "canonical_headered_csv",
                    "pivot_input_usable_rows": 0,
                    "pivot_input_rejected_row_count": 1,
                    "pivot_input_rejected_rows": [{"row": 2, "reason": "missing values for qz"}],
                }
            ),
        )

    monkeypatch.setattr(controller.experiment_runner, "run_experiment", _fake_failed_run)

    with pytest.raises(RuntimeError, match="required columns missing: qz"):
        controller.solve_pivot_collection()

    state = controller.refresh()
    assert state.pivot_status == "solve_failed"
    assert state.pivot_input_format == "canonical_headered_csv"
    assert state.pivot_input_usable_rows == 0
    assert state.pivot_input_rejected_row_count == 1
    assert "row 2: missing values for qz" in state.pivot_input_rejected_rows[0]

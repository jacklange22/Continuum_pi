from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from continuum_robot.config.config_loader import ConfigLoader
from continuum_robot.config.schemas import (
    CalibrationConfig,
    ExperimentConfig,
    RegistrationWorkflowConfig,
    RobotConfig,
    RobotSegmentConfig,
    RuntimeConfig,
    SafetyConfig,
    SerialConfig,
)
from continuum_robot.config.settings import Settings
from continuum_robot.experiments.builtins import (
    PretensionValidationExperiment,
    PretensionValidationExperimentConfig,
    _configured_collect_pose_servo_ids,
    _mirror_parallel_single_displacements,
)
from continuum_robot.experiments.single_segment_repeatability import _configured_single_segment_servo_ids
from continuum_robot.gui.experiment_preflight import evaluate_preflight
from continuum_robot.gui.controllers.system_controller import SystemController
from continuum_robot.servos.neutral_calibration_service import NeutralCalibrationService, ServoCalibrationContext


class _TrackingStub:
    def get_snapshot(self):
        return SimpleNamespace(
            connection_state="disconnected",
            backend_identity="none",
            backend_running=False,
            backend_connected=False,
            bridge_running=False,
            socket_connected=False,
            registration_state="missing_registration",
            runtime_tip_calibration_state="missing_runtime_tip_calibration",
            runtime_tip_mode="latest_accepted",
            runtime_tip_trust_level="missing",
            tip_pose_status="missing_registration",
            last_error=None,
        )


class _OpenRbStub:
    def get_status_snapshot(self):
        return SimpleNamespace(connected=False, prepared=False, message="disconnected")


class _ServoStub:
    is_connected = False

    def __init__(self, neutral_calibration: NeutralCalibrationService) -> None:
        self.neutral_calibration = neutral_calibration

    def bus_ownership_status(self):
        return SimpleNamespace(active=False, held_by_current_thread=True)

    def get_calibration_summary(self):
        return self.neutral_calibration.get_calibration_summary()

    @staticmethod
    def raw_position_range():
        return (0, 4095)


def _settings(active_segment: str = "segment_a", *, mode: str = "single_segment", selected_servo_id: int = 1) -> Settings:
    robot = RobotConfig(
        mode=mode,
        spool_diameter_cm=1.2,
        ticks_per_revolution=4096,
        servo_ids=[1, 2, 3, 4, 5, 6, 7, 8],
        tendon_to_servo=[1, 2, 3, 4, 5, 6, 7, 8],
        active_segment=active_segment,
        selected_servo_id=selected_servo_id,
        segments={
            "segment_a": RobotSegmentConfig(
                key="segment_a",
                label="Spine 1",
                segment_label="Segment A",
                segment_role="proximal",
                segment_order_index=0,
                servo_ids=[1, 2, 3, 4],
                pairs={"axis_a": [1, 3], "axis_b": [2, 4]},
            ),
            "segment_b": RobotSegmentConfig(
                key="segment_b",
                label="Spine 2",
                segment_label="Segment B",
                segment_role="distal",
                segment_order_index=1,
                servo_ids=[5, 6, 7, 8],
                pairs={"axis_a": [5, 7], "axis_b": [6, 8]},
            ),
        },
    )
    return Settings(
        runtime=RuntimeConfig(mock_mode=True, poll_rate_hz=20, robot_config="robot_8servo.yaml"),
        robot=robot,
        serial=SerialConfig(),
        safety=SafetyConfig(),
        registration=RegistrationWorkflowConfig(),
        experiment=ExperimentConfig(),
        calibration=CalibrationConfig(),
    )


def _session(settings: Settings):
    return SimpleNamespace(context=SimpleNamespace(settings=settings))


def test_active_segment_b_resolves_servo_ids_and_pairs() -> None:
    settings = _settings("segment_b")

    assert settings.robot.active_segment_key() == "segment_b"
    assert settings.robot.active_segment_label() == "Spine 2"
    assert settings.robot.active_segment_servo_ids() == [5, 6, 7, 8]
    assert settings.robot.active_segment_pairs() == {"axis_a": [5, 7], "axis_b": [6, 8]}


def test_operating_context_resolves_all_modes() -> None:
    assert _settings(mode="one_servo", selected_servo_id=1).robot.operating_context().expected_servo_ids == [1]
    assert _settings(mode="one_servo", selected_servo_id=8).robot.operating_context().expected_servo_ids == [8]
    assert _settings("segment_a", mode="single_segment").robot.operating_context().expected_servo_ids == [1, 2, 3, 4]
    assert _settings("segment_b", mode="single_segment").robot.operating_context().expected_servo_ids == [5, 6, 7, 8]
    assert _settings(mode="dual_segment").robot.operating_context().expected_servo_ids == [1, 2, 3, 4, 5, 6, 7, 8]
    parallel = _settings(mode="parallel_single").robot.operating_context()
    assert parallel.expected_servo_ids == [1, 2, 3, 4, 5, 6, 7, 8]
    assert parallel.mirror_pairs == {1: 5, 2: 6, 3: 7, 4: 8}


def test_dual_segment_context_records_segment_order_roles_and_pairs() -> None:
    context = _settings(mode="dual_segment").robot.operating_context()
    metadata = context.metadata()

    assert context.expected_servo_ids == [1, 2, 3, 4, 5, 6, 7, 8]
    assert context.commanded_servo_ids == [1, 2, 3, 4, 5, 6, 7, 8]
    assert context.segment_order == ["segment_a", "segment_b"]
    assert metadata["segments"]["segment_a"]["segment_label"] == "Segment A"
    assert metadata["segments"]["segment_a"]["segment_role"] == "proximal"
    assert metadata["segments"]["segment_a"]["segment_order_index"] == 0
    assert metadata["segments"]["segment_a"]["servo_ids"] == [1, 2, 3, 4]
    assert metadata["segments"]["segment_a"]["pairs"] == {"axis_a": [1, 3], "axis_b": [2, 4]}
    assert metadata["segments"]["segment_b"]["segment_label"] == "Segment B"
    assert metadata["segments"]["segment_b"]["segment_role"] == "distal"
    assert metadata["segments"]["segment_b"]["segment_order_index"] == 1
    assert metadata["segments"]["segment_b"]["servo_ids"] == [5, 6, 7, 8]
    assert metadata["segments"]["segment_b"]["pairs"] == {"axis_a": [5, 7], "axis_b": [6, 8]}
    assert metadata["mode_profile"] == "all_8_readiness_manual_startup_foundation"
    assert metadata["mode_capabilities"]["manual_startup_capture"] is True
    assert metadata["mode_capabilities"]["two_segment_kinematics_control"] is False


def test_single_segment_workflow_helpers_use_active_segment_ids() -> None:
    settings = _settings("segment_b")
    session = _session(settings)

    assert _configured_single_segment_servo_ids(session) == [5, 6, 7, 8]
    assert _configured_collect_pose_servo_ids(session) == [5, 6, 7, 8]

    pretension = PretensionValidationExperiment(
        PretensionValidationExperimentConfig.from_dict({"mode": "single_segment_staged"})
    )
    assert pretension._staged_servo_ids(session) == [5, 6, 7, 8]


def test_parallel_single_collect_pose_uses_all_ids_and_mirrors_displacements() -> None:
    settings = _settings("segment_a", mode="parallel_single")
    session = _session(settings)
    context = settings.robot.operating_context()

    assert _configured_collect_pose_servo_ids(session) == [1, 2, 3, 4, 5, 6, 7, 8]
    assert _mirror_parallel_single_displacements(
        [0.1, 0.2, -0.1, -0.2],
        servo_ids=context.commanded_servo_ids,
        context=context,
    ) == [0.1, 0.2, -0.1, -0.2, 0.1, 0.2, -0.1, -0.2]


def test_manual_startup_artifact_is_segment_scoped(tmp_path: Path) -> None:
    path = tmp_path / "neutral.json"
    context_a = ServoCalibrationContext(
        robot_mode="8-servo",
        robot_config_name="robot_8servo.yaml",
        servo_ids=[1, 2, 3, 4, 5, 6, 7, 8],
        tendon_to_servo=[1, 2, 3, 4, 5, 6, 7, 8],
        active_segment_key="segment_a",
        active_segment_label="Spine 1",
        active_segment_servo_ids=[1, 2, 3, 4],
        active_segment_pairs={"axis_a": [1, 3], "axis_b": [2, 4]},
    )
    service_a = NeutralCalibrationService(path=path, context=context_a)
    service_a.save_manual_pretension_state(
        states_by_servo={
            servo_id: {"servo_id": servo_id, "measured_position_tick": 2000 + servo_id, "measured_current_ma": 10}
            for servo_id in [1, 2, 3, 4]
        },
        note="segment A startup",
        accepted=True,
    )

    summary_a = service_a.get_calibration_summary().pretension_source_summary([1, 2, 3, 4])
    assert summary_a.accepted is True
    latest_run = service_a.load_calibration_artifact().servos[1].latest_pretension_run
    assert latest_run["active_segment_key"] == "segment_a"
    assert latest_run["active_segment_servo_ids"] == [1, 2, 3, 4]

    context_b = ServoCalibrationContext(
        robot_mode="8-servo",
        robot_config_name="robot_8servo.yaml",
        servo_ids=[1, 2, 3, 4, 5, 6, 7, 8],
        tendon_to_servo=[1, 2, 3, 4, 5, 6, 7, 8],
        active_segment_key="segment_b",
        active_segment_label="Spine 2",
        active_segment_servo_ids=[5, 6, 7, 8],
        active_segment_pairs={"axis_a": [5, 7], "axis_b": [6, 8]},
    )
    service_b = NeutralCalibrationService(path=path, context=context_b)
    summary_b = service_b.get_calibration_summary()
    assert summary_b.compatible is False
    assert "5" in summary_b.message


def test_dual_segment_manual_startup_artifact_records_all_8_scope(tmp_path: Path) -> None:
    path = tmp_path / "neutral.json"
    settings = _settings(mode="dual_segment")
    operating_context = settings.robot.operating_context()
    context = ServoCalibrationContext(
        robot_mode="dual_segment",
        robot_config_name="robot_8servo.yaml",
        servo_ids=[1, 2, 3, 4, 5, 6, 7, 8],
        tendon_to_servo=[1, 2, 3, 4, 5, 6, 7, 8],
        segments=settings.robot.segment_metadata(),
        segment_order=settings.robot.segment_order(),
        expected_servo_ids=list(operating_context.expected_servo_ids),
        commanded_servo_ids=list(operating_context.commanded_servo_ids),
        mode_profile=operating_context.mode_profile,
        mode_capabilities=operating_context.mode_capabilities,
        mode_notes=operating_context.mode_notes,
    )
    service = NeutralCalibrationService(path=path, context=context)
    service.save_manual_pretension_state(
        states_by_servo={
            servo_id: {
                "servo_id": servo_id,
                "measured_position_tick": 2100 + servo_id,
                "measured_current_ma": 20 + servo_id,
            }
            for servo_id in range(1, 9)
        },
        note="all-8 startup",
        accepted=True,
    )

    summary = service.get_calibration_summary()
    assert summary.compatible is True
    source = summary.pretension_source_summary([1, 2, 3, 4, 5, 6, 7, 8])
    assert source.usable is True
    artifact = service.load_calibration_artifact()
    assert sorted(artifact.servos) == [1, 2, 3, 4, 5, 6, 7, 8]
    assert artifact.robot["robot_mode"] == "dual_segment"
    assert artifact.robot["startup_artifact_scope"] == "dual_segment_all_8_manual_startup_foundation"
    assert artifact.robot["expected_servo_ids"] == [1, 2, 3, 4, 5, 6, 7, 8]
    assert artifact.robot["commanded_servo_ids"] == [1, 2, 3, 4, 5, 6, 7, 8]
    assert artifact.robot["segment_order"] == ["segment_a", "segment_b"]
    assert artifact.robot["segments"]["segment_a"]["segment_role"] == "proximal"
    assert artifact.robot["segments"]["segment_b"]["segment_role"] == "distal"
    assert artifact.servos[8].last_measured_current_ma == 28
    latest_run = artifact.servos[1].latest_pretension_run
    assert latest_run["startup_artifact_scope"] == "dual_segment_all_8_manual_startup_foundation"
    assert latest_run["segment_order"] == ["segment_a", "segment_b"]
    assert latest_run["segments"]["segment_a"]["pairs"] == {"axis_a": [1, 3], "axis_b": [2, 4]}

    single_context = ServoCalibrationContext(
        robot_mode="single_segment",
        robot_config_name="robot_8servo.yaml",
        servo_ids=[1, 2, 3, 4, 5, 6, 7, 8],
        tendon_to_servo=[1, 2, 3, 4, 5, 6, 7, 8],
        active_segment_key="segment_a",
        active_segment_label="Spine 1",
        active_segment_servo_ids=[1, 2, 3, 4],
        active_segment_pairs={"axis_a": [1, 3], "axis_b": [2, 4]},
        expected_servo_ids=[1, 2, 3, 4],
        commanded_servo_ids=[1, 2, 3, 4],
    )
    single_summary = NeutralCalibrationService(path=path, context=single_context).get_calibration_summary()
    assert single_summary.compatible is False
    assert "does not match current mode single_segment" in single_summary.message


def test_system_controller_applies_active_segment_to_live_servo_context(tmp_path) -> None:
    settings = _settings("segment_b")
    context = ServoCalibrationContext(
        robot_mode="4-servo",
        robot_config_name="robot_4servo.yaml",
        servo_ids=[1, 2, 3, 4],
        tendon_to_servo=[1, 2, 3, 4],
        active_segment_key="segment_a",
        active_segment_label="Spine 1",
        active_segment_servo_ids=[1, 2, 3, 4],
        active_segment_pairs={"axis_a": [1, 3], "axis_b": [2, 4]},
    )
    servo_service = SimpleNamespace(
        neutral_calibration=NeutralCalibrationService(path=tmp_path / "neutral.json", context=context)
    )
    controller = SystemController.__new__(SystemController)
    controller.servo_service = servo_service

    controller._apply_robot_config_to_servo_context(settings.robot)

    assert context.robot_mode == "single_segment"
    assert context.servo_ids == [1, 2, 3, 4, 5, 6, 7, 8]
    assert context.expected_servo_ids == [5, 6, 7, 8]
    assert context.commanded_servo_ids == [5, 6, 7, 8]
    assert context.active_segment_key == "segment_b"
    assert context.active_segment_label == "Spine 2"
    assert context.active_segment_servo_ids == [5, 6, 7, 8]
    assert context.active_segment_pairs == {"axis_a": [5, 7], "axis_b": [6, 8]}
    assert context.segment_order == ["segment_a", "segment_b"]
    assert context.segments["segment_a"]["segment_role"] == "proximal"
    assert context.segments["segment_b"]["segment_role"] == "distal"


def test_system_controller_saves_mode_scope_without_stale_servo_membership(tmp_path: Path) -> None:
    config_dir = tmp_path / "config"
    config_dir.mkdir()
    (config_dir / "system.yaml").write_text('robot_config: "robot_8servo.yaml"\nmock_mode: true\n', encoding="utf-8")
    (config_dir / "robot_8servo.yaml").write_text(
        "\n".join(
            [
                "mode: single_segment",
                "servo_ids: [1, 2, 3, 4, 5, 6, 7, 8]",
                "tendon_to_servo: [1, 2, 3, 4, 5, 6, 7, 8]",
                "active_segment: segment_a",
                "segments:",
                "  segment_a: {label: Spine 1, servo_ids: [1, 2, 3, 4], pairs: {axis_a: [1, 3], axis_b: [2, 4]}}",
                "  segment_b: {label: Spine 2, servo_ids: [5, 6, 7, 8], pairs: {axis_a: [5, 7], axis_b: [6, 8]}}",
                "tightening_rotation_by_servo: {1: cw, 2: cw, 3: cw, 4: cw, 5: cw, 6: cw, 7: cw, 8: cw}",
            ]
        ),
        encoding="utf-8",
    )
    loader = ConfigLoader(base_dir=config_dir)
    settings = loader.load_settings()
    neutral = NeutralCalibrationService(path=tmp_path / "neutral.json")
    controller = SystemController(
        tracking_service=_TrackingStub(),
        openrb_client=_OpenRbStub(),
        servo_service=_ServoStub(neutral),
        settings=settings,
        config_loader=loader,
    )

    controller.save_runtime_parameters(
        mock_mode=True,
        robot_config="robot_8servo.yaml",
        operating_mode="single_segment",
        selected_servo_id=8,
        active_segment="segment_b",
        openrb_port="/dev/mock-openrb",
        baudrate=57600,
        poll_rate_hz=20,
        telemetry_freshness_timeout_s=0.25,
    )

    saved = loader.load_system_local_overrides()
    assert saved["robot_config"] == "robot_8servo.yaml"
    assert saved["robot_overrides"]["mode"] == "single_segment"
    assert saved["robot_overrides"]["selected_servo_id"] == 8
    assert saved["robot_overrides"]["active_segment"] == "segment_b"
    assert "servo_ids" not in saved["robot_overrides"]
    assert "tendon_to_servo" not in saved["robot_overrides"]
    assert controller.state.expected_servo_ids == [5, 6, 7, 8]


def test_segment_b_pretension_preflight_names_active_ids_without_tracker(tmp_path: Path) -> None:
    settings = _settings("segment_b")
    snapshot = SimpleNamespace(
        selected_backend_name="none",
        backend_identity="none",
        canonical_state="disconnected",
    )

    report = evaluate_preflight(
        experiment_name="pretension_validation",
        config_payload={
            "mode": "single_segment_staged",
            "servo_ids": [],
            "include_tracker_displacement": True,
            "allow_no_tracker_test_run": True,
            "run_trust_mode": "current_only",
        },
        config_error=None,
        settings=settings,
        tracking_snapshot=snapshot,
        servo_connected=True,
        neutral_setpoints={5: 2048, 6: 2048, 7: 2048, 8: 2048},
        registration_path=tmp_path / "latest_registration.json",
        output_root=tmp_path / "data" / "experiments",
        planned_output_dir=tmp_path / "data" / "experiments" / "pretension_validation" / "run",
        project_root=tmp_path,
        servo_calibration_summary=None,
    )

    active_check = next(check for check in report.checks if check.key == "active_segment")
    assert active_check.status == "ok"
    assert "Spine 2" in active_check.message
    assert "[5, 6, 7, 8]" in active_check.message
    tracking_check = next(check for check in report.checks if check.key == "tracking_state")
    assert tracking_check.status == "warning"
    assert "current-only" in tracking_check.message


def test_dual_segment_pretension_preflight_blocks_automatic_two_segment_pretension(tmp_path: Path) -> None:
    settings = _settings(mode="dual_segment")
    snapshot = SimpleNamespace(
        selected_backend_name="none",
        backend_identity="none",
        canonical_state="disconnected",
    )

    report = evaluate_preflight(
        experiment_name="pretension_validation",
        config_payload={
            "mode": "single_segment_staged",
            "servo_ids": [],
            "include_tracker_displacement": False,
            "allow_no_tracker_test_run": True,
            "allow_current_only_when_tracker_missing": True,
            "run_trust_mode": "current_only",
        },
        config_error=None,
        settings=settings,
        tracking_snapshot=snapshot,
        servo_connected=True,
        neutral_setpoints={servo_id: 2048 for servo_id in range(1, 9)},
        registration_path=tmp_path / "latest_registration.json",
        output_root=tmp_path / "data" / "experiments",
        planned_output_dir=tmp_path / "data" / "experiments" / "pretension_validation" / "run",
        project_root=tmp_path,
        servo_calibration_summary=None,
    )

    active_check = next(check for check in report.checks if check.key == "active_segment")
    assert active_check.status == "blocked"
    assert "all-8 readiness and manual startup capture" in active_check.message
    assert "Automatic two-segment pretension/control is not implemented yet" in active_check.message

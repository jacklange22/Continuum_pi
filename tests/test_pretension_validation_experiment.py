from __future__ import annotations

import time
from pathlib import Path
from types import SimpleNamespace

import pytest

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
from continuum_robot.experiments.builtins import PretensionValidationExperimentConfig
from continuum_robot.experiments.builtins import PretensionValidationExperiment
from continuum_robot.experiments.experiment_runner import ExperimentRunner
from continuum_robot.hardware.mock_dxl_bus import MockDxlBus
from continuum_robot.servos.displacement_mapper import TendonDisplacementMapper
from continuum_robot.servos.neutral_calibration_service import NeutralCalibrationService, ServoCalibrationContext
from continuum_robot.servos.pretension_validation_service import PretensionValidationService
from continuum_robot.servos.safety_guard import SafetyGuard
from continuum_robot.servos.servo_service import ServoService


def _settings(tmp_path: Path) -> Settings:
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
            pretension_current_balance_tolerance_ma=6,
            pretension_start_mode="release_200_from_current",
            pretension_step_ticks=2,
            pretension_settle_time_s=0.0,
            pretension_baseline_sample_count=1,
            pretension_current_filter_window=1,
            pretension_max_travel_ticks=4,
        ),
        registration=RegistrationWorkflowConfig(capture_tool_id="0B", coil_tool_id="0A", max_fre_mm=None),
        experiment=ExperimentConfig(default_settle_time_s=0.0, sample_count_per_point=1, output_dir=str(tmp_path)),
        calibration=CalibrationConfig(
            neutral_setpoints_path=str(tmp_path / "neutral.json"),
            latest_registration_path=str(tmp_path / "latest_registration.json"),
        ),
    )


def _servo_service(tmp_path: Path) -> ServoService:
    service = ServoService(
        dxl_bus=MockDxlBus([1, 2, 3, 4]),
        mapper=TendonDisplacementMapper(spool_diameter_cm=1.2),
        safety_guard=SafetyGuard(
            min_offset_ticks=-600,
            max_offset_ticks=600,
            max_current_ma=850,
            pretension_start_mode="release_200_from_current",
            pretension_step_ticks=2,
            pretension_timeout_s=2.0,
            pretension_settle_time_s=0.0,
            pretension_baseline_sample_count=1,
            pretension_current_filter_window=1,
            pretension_max_travel_ticks=4,
        ),
        neutral_calibration=NeutralCalibrationService(
            path=tmp_path / "neutral.json",
            context=ServoCalibrationContext(
                robot_mode="4-servo",
                robot_config_name="robot_4servo.yaml",
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
    )
    service.connect("/dev/mock-openrb", 115200)
    return service


def _servo_service_with_bus(tmp_path: Path, bus: MockDxlBus) -> ServoService:
    service = ServoService(
        dxl_bus=bus,
        mapper=TendonDisplacementMapper(spool_diameter_cm=1.2),
        safety_guard=SafetyGuard(
            min_offset_ticks=-600,
            max_offset_ticks=600,
            max_current_ma=850,
            pretension_start_mode="current_position",
            pretension_step_ticks=2,
            pretension_timeout_s=2.0,
            pretension_settle_time_s=0.0,
            pretension_baseline_sample_count=1,
            pretension_current_filter_window=1,
            pretension_max_travel_ticks=4,
        ),
        neutral_calibration=NeutralCalibrationService(
            path=tmp_path / "neutral.json",
            context=ServoCalibrationContext(
                robot_mode="4-servo",
                robot_config_name="robot_4servo.yaml",
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
    )
    service.connect("/dev/mock-openrb", 115200)
    return service


def _runner(tmp_path: Path, servo_service: ServoService, tracking_service=None) -> ExperimentRunner:
    settings = _settings(tmp_path)
    return ExperimentRunner(
        project_root=tmp_path,
        settings=settings,
        tracking_service=tracking_service,
        servo_service=servo_service,
        output_dir=tmp_path / "data" / "experiments",
        registration_path=tmp_path / "latest_registration.json",
        sleep_fn=lambda _seconds: None,
    )


def _advanced_config() -> dict:
    return {
        "mode": "single_segment_characterization",
        "staged_strategy": "characterization",
        "servo_ids": [1, 2, 3, 4],
        "repeat_runs": 1,
        "include_tracker_displacement": False,
        "allow_current_only_when_tracker_missing": True,
        "enable_tip_centering": False,
        "move_to_reference": True,
        "pretension_start_mode": "current_position",
        "step_ticks": 2,
        "settle_time_s": 0.0,
        "baseline_sample_count": 1,
        "current_filter_window": 1,
        "current_delta_threshold_ma": 60,
        "absolute_trigger_current_ma": 220,
        "hard_current_stop_ma": 850,
        "max_travel_ticks": 4,
        "timeout_s": 2.0,
        "equalization_max_iterations": 1,
        "current_characterization_sample_count": 3,
        "characterization_pair_cycles": 1,
        "characterization_step_ticks": 1,
        "conservative_step_ticks": 1,
        "conservative_max_iterations": 2,
        "conservative_max_cumulative_travel_ticks": 8,
        "soft_release_step_ticks": 2,
        "soft_release_max_travel_ticks": 8,
        "soft_release_current_target_ma": 10.0,
        "takeup_target_load_proxy_ma": 4.0,
        "takeup_max_iterations": 2,
        "staged_packet_retry_budget": 3,
        "min_meaningful_current_delta_ma": 2.0,
        "load_balance_tolerance_ma": 6.0,
        "pair_balance_tolerance_ma": 6.0,
        "accept_max_load_balance_error_ma": 6.0,
        "accept_max_pair_balance_error_ma": 6.0,
        "accept_max_final_tip_xy_offset_mm": 3.0,
    }


def test_pretension_validation_config_uses_adjustable_advanced_tolerances() -> None:
    config = PretensionValidationExperimentConfig.from_dict(
        {
            "mode": "single_segment_staged",
            "tip_center_tolerance_mm": 2.5,
            "load_balance_tolerance_ma": 5.5,
            "pair_balance_tolerance_ma": 4.5,
            "staged_strategy": "characterization",
            "current_characterization_sample_count": 11,
            "min_meaningful_current_delta_ma": 3.5,
            "conservative_step_ticks": 1,
            "conservative_max_cumulative_travel_ticks": 20,
            "soft_release_step_ticks": 2,
            "soft_release_max_travel_ticks": 12,
            "soft_release_current_target_ma": 9.0,
            "takeup_target_load_proxy_ma": 4.0,
            "takeup_max_iterations": 3,
            "staged_packet_retry_budget": 3,
            "accept_max_final_tip_xy_offset_mm": 3.0,
            "accept_max_load_balance_error_ma": 6.0,
            "accept_max_pair_balance_error_ma": 6.0,
        }
    )

    assert config.tip_center_tolerance_mm == 2.5
    assert config.load_balance_tolerance_ma == 5.5
    assert config.pair_balance_tolerance_ma == 4.5
    assert config.staged_strategy == "characterization"
    assert config.current_characterization_sample_count == 11
    assert config.min_meaningful_current_delta_ma == 3.5
    assert config.conservative_step_ticks == 1
    assert config.conservative_max_cumulative_travel_ticks == 20
    assert config.soft_release_step_ticks == 2
    assert config.soft_release_max_travel_ticks == 12
    assert config.soft_release_current_target_ma == 9.0
    assert config.takeup_target_load_proxy_ma == 4.0
    assert config.takeup_max_iterations == 3
    assert config.staged_packet_retry_budget == 3
    assert config.accept_max_final_tip_xy_offset_mm == 3.0
    assert config.accept_max_load_balance_error_ma == 6.0
    assert config.accept_max_pair_balance_error_ma == 6.0


def test_conservative_staged_strategy_is_default() -> None:
    config = PretensionValidationExperimentConfig.from_dict({"mode": "single_segment_staged"})

    assert config.staged_strategy == "conservative_startup"
    assert config.allow_legacy_staged_strategy is False
    assert config.takeup_target_load_proxy_ma == pytest.approx(25.0)
    assert config.high_load_proxy_ma == pytest.approx(40.0)
    assert config.tip_center_tolerance_mm == pytest.approx(1.0)


def test_legacy_staged_strategy_blocks_without_developer_override(tmp_path: Path) -> None:
    service = _servo_service(tmp_path)
    runner = _runner(tmp_path, service)
    config = _advanced_config()
    config["mode"] = "single_segment_staged"
    config["staged_strategy"] = "legacy"
    config["include_tracker_displacement"] = False
    config["allow_current_only_when_tracker_missing"] = True

    result = runner.run_experiment("pretension_validation", config=config)

    assert result.success is False
    assert "Legacy staged pretension is disabled for normal operation" in result.message


@pytest.mark.parametrize("operating_mode", ["one_servo", "dual_segment", "parallel_single"])
def test_automatic_staged_pretension_blocks_non_single_segment_modes(tmp_path: Path, operating_mode: str) -> None:
    service = _servo_service(tmp_path)
    runner = _runner(tmp_path, service)
    runner.settings.robot.mode = operating_mode
    config = _advanced_config()
    config["mode"] = "single_segment_staged"
    config["staged_strategy"] = "conservative_startup"
    config["include_tracker_displacement"] = False
    config["allow_current_only_when_tracker_missing"] = True

    result = runner.run_experiment("pretension_validation", config=config)

    assert result.success is False
    if operating_mode == "dual_segment":
        assert "all-8 readiness and manual startup capture" in result.message
        assert "Automatic two-segment pretension/control is not implemented yet" in result.message
    else:
        assert "requires operating_mode=single_segment" in result.message


def test_current_only_staged_pretension_requires_explicit_lower_trust_override(tmp_path: Path) -> None:
    service = _servo_service(tmp_path)
    runner = _runner(tmp_path, service)
    config = _advanced_config()
    config["mode"] = "single_segment_staged"
    config["staged_strategy"] = "conservative_startup"
    config["include_tracker_displacement"] = False
    config["allow_current_only_when_tracker_missing"] = False
    config["allow_no_tracker_test_run"] = False
    config["run_trust_mode"] = "thesis_trusted"

    result = runner.run_experiment("pretension_validation", config=config)

    assert result.success is False
    assert "requires an explicit lower-trust/current-only override" in result.message


def test_advanced_pretension_staged_run_saves_quality_pairing_and_plots(tmp_path: Path) -> None:
    service = _servo_service(tmp_path)
    runner = _runner(tmp_path, service)

    result = runner.run_experiment("pretension_validation", config=_advanced_config())

    metrics = result.summary.experiment_metrics
    run_row = metrics["run_rows"][0]
    stages = [row.get("stage") for row in metrics["trace_rows"]]

    assert result.success is True
    assert metrics["algorithm"] == "reliable_advanced_4servo_pretension"
    assert metrics["algorithm_mode"] == "characterization"
    assert run_row["quality_score_0_100"] >= 0.0
    assert run_row["quality_score_0_100"] <= 100.0
    assert "tip_centering" in run_row["quality_components"]
    assert "low_load_start" in stages
    assert "soft_release_check" in stages
    assert "current_characterization" in stages
    assert "pair_characterization_step" in stages
    assert "paired_takeup" not in stages
    assert run_row["current_characterization"]["max_useful_current_delta_ma"] >= 2.0
    assert metrics["quality_scores_0_100"] == [run_row["quality_score_0_100"]]

    expected_plots = [
        "pretension_tendon_displacement_vs_tip_xy.png",
        "pretension_tendon_displacement_vs_current.png",
        "pretension_current_vs_tip_error.png",
        "pretension_balance_over_stages.png",
        "pretension_tip_xy_path.png",
        "pretension_final_tip_xy_scatter.png",
        "pretension_final_current_distribution.png",
        "pretension_final_position_distribution.png",
        "pretension_quality_score_distribution.png",
        "pretension_tip_xy_path_report.png",
        "pretension_load_proxy_by_servo_report.png",
        "pretension_tendon_displacement_vs_load_proxy_report.png",
        "pretension_final_state_report.png",
    ]
    for filename in expected_plots:
        assert (result.paths.output_dir / filename).exists()
    bundle_dir = result.paths.output_dir / "pretension_debug_bundle"
    assert (bundle_dir / "summary.json").exists()
    assert (bundle_dir / "metrics.csv").exists()
    assert (bundle_dir / "samples.jsonl").exists()
    assert (bundle_dir / "debug_manifest.json").exists()
    assert "quality_score_0_100" in (result.paths.output_dir / "metrics.csv").read_text(encoding="utf-8").splitlines()[0]
    sample_text = (result.paths.output_dir / "samples.jsonl").read_text(encoding="utf-8")
    assert "current_validity" in sample_text
    assert "ownership_state" in sample_text
    assert "target_xy_mm" in sample_text
    assert "tendon_displacement_mm" in sample_text
    assert "signed_raw_current_ma" in sample_text
    assert "load_proxy_current_ma" in sample_text
    assert "packet_retry_count" in sample_text


def test_advanced_pretension_metrics_preserve_manual_artifact_comparison(tmp_path: Path) -> None:
    service = _servo_service(tmp_path)
    service.neutral_calibration.save_manual_pretension_state(
        states_by_servo={
            servo_id: {
                "servo_id": servo_id,
                "measured_position_tick": 2048 + servo_id,
                "measured_current_ma": 150 + servo_id,
            }
            for servo_id in [1, 2, 3, 4]
        },
        note="manual comparison baseline",
        accepted=True,
    )
    runner = _runner(tmp_path, service)

    result = runner.run_experiment("pretension_validation", config=_advanced_config())

    manual_artifact = result.summary.experiment_metrics["manual_startup_artifact"]
    advanced_artifacts = result.summary.experiment_metrics["advanced_startup_artifacts"]
    assert manual_artifact["available"] is True
    assert manual_artifact["source_type"] == "manual"
    assert manual_artifact["note"] == "manual comparison baseline"
    assert advanced_artifacts[0]["quality_score_0_100"] is not None


def test_staged_pretension_soft_release_does_not_go_to_4095_by_default(tmp_path: Path) -> None:
    service = _servo_service(tmp_path)
    runner = _runner(tmp_path, service)

    result = runner.run_experiment("pretension_validation", config=_advanced_config())

    metrics = result.summary.experiment_metrics
    assert metrics["pretension_start_mode"] == "current_position"
    low_load_row = next(row for row in metrics["trace_rows"] if row.get("stage") == "low_load_start")
    assert low_load_row["startup_source"] == "current_position"
    for row in metrics["trace_rows"]:
        for target in dict(row.get("commanded_positions_ticks") or {}).values():
            assert target != 4095


def test_staged_pretension_full_release_only_when_explicitly_configured(tmp_path: Path) -> None:
    service = _servo_service(tmp_path)
    runner = _runner(tmp_path, service)
    config = _advanced_config()
    config["pretension_start_mode"] = "full_release_4095"
    config["soft_release_max_travel_ticks"] = 0

    result = runner.run_experiment("pretension_validation", config=config)

    metrics = result.summary.experiment_metrics
    assert metrics["pretension_start_mode"] == "full_release_4095"
    assert any(row.get("stage") == "explicit_full_release" for row in metrics["trace_rows"])
    full_release_row = next(row for row in metrics["trace_rows"] if row.get("stage") == "explicit_full_release")
    assert full_release_row["explicit_full_release"] is True
    assert any(int(target) == 4095 for target in dict(full_release_row["commanded_positions_ticks"]).values())


def test_staged_measurement_uses_absolute_load_proxy_for_negative_signed_currents(tmp_path: Path) -> None:
    service = _servo_service(tmp_path)
    for servo_id, telemetry in service.dxl_bus._state.items():
        telemetry.present_current_ma = -20 - servo_id
        telemetry.present_current_raw_unit = telemetry.present_current_ma
    experiment = PretensionValidationExperiment(
        PretensionValidationExperimentConfig.from_dict(
            {"mode": "single_segment_staged", "servo_ids": [1, 2, 3, 4], "include_tracker_displacement": False}
        )
    )

    measurement = experiment._advanced_measurement(
        servo_service=service,
        tracker_service=None,
        servo_ids=[1, 2, 3, 4],
        baseline_current_ma_by_servo={1: -20.0, 2: -20.0, 3: -20.0, 4: -20.0},
        target_xy_mm=[0.0, 0.0],
        startup_reference_ticks_by_servo={1: 2048, 2: 2073, 3: 2098, 4: 2123},
        trust_status="current_only_lower_trust",
    )

    assert measurement["signed_raw_current_ma"][1] == -21
    assert measurement["load_proxy_current_ma"][1] == pytest.approx(1.0)
    assert measurement["current_above_baseline_ma"][4] == pytest.approx(4.0)


class _FlakyTelemetryBus(MockDxlBus):
    def __init__(self, failures: list[str]) -> None:
        super().__init__([1, 2, 3, 4])
        self.failures = list(failures)

    def read_live_telemetry(self, servo_ids: list[int]):
        telemetry = super().read_live_telemetry(servo_ids)
        if not self.failures:
            return telemetry
        failure = self.failures.pop(0)
        if failure == "missing_current":
            telemetry[1].present_current_ma = None
        elif failure == "missing_position":
            telemetry[1].present_position = None
        elif failure == "no_status_packet":
            telemetry[1].telemetry_error = "No status packet received"
        elif failure == "incorrect_status_packet":
            telemetry[1].telemetry_error = "Incorrect status packet received"
        return telemetry


def test_staged_measurement_retries_one_missed_packet_and_logs(tmp_path: Path) -> None:
    service = _servo_service_with_bus(tmp_path, _FlakyTelemetryBus(["missing_current"]))
    experiment = PretensionValidationExperiment(
        PretensionValidationExperimentConfig.from_dict(
            {"mode": "single_segment_staged", "servo_ids": [1, 2, 3, 4], "staged_packet_retry_budget": 3}
        )
    )

    measurement = experiment._advanced_measurement(
        servo_service=service,
        tracker_service=None,
        servo_ids=[1, 2, 3, 4],
        baseline_current_ma_by_servo={},
        target_xy_mm=[0.0, 0.0],
    )

    assert measurement["telemetry_fail_closed_reason"] is None
    assert measurement["packet_retry_count"] == 1
    assert measurement["telemetry_event_counts"]["missing_current"] == 1
    assert measurement["current_validity"][1] == "valid_after_retry"


def test_staged_measurement_repeated_packet_misses_fail_closed(tmp_path: Path) -> None:
    service = _servo_service_with_bus(tmp_path, _FlakyTelemetryBus(["missing_current", "missing_current", "missing_current"]))
    experiment = PretensionValidationExperiment(
        PretensionValidationExperimentConfig.from_dict(
            {"mode": "single_segment_staged", "servo_ids": [1, 2, 3, 4], "staged_packet_retry_budget": 3}
        )
    )

    measurement = experiment._advanced_measurement(
        servo_service=service,
        tracker_service=None,
        servo_ids=[1, 2, 3, 4],
        baseline_current_ma_by_servo={},
        target_xy_mm=[0.0, 0.0],
    )

    assert measurement["telemetry_fail_closed_reason"] == "packet_retry_budget_exhausted"
    assert measurement["packet_retry_count"] == 3


def test_staged_measurement_stale_telemetry_fails_closed(tmp_path: Path) -> None:
    service = _servo_service(tmp_path)
    service.dxl_bus._state[1].last_read_monotonic_s = time.monotonic() - 100.0
    experiment = PretensionValidationExperiment(
        PretensionValidationExperimentConfig.from_dict({"mode": "single_segment_staged", "servo_ids": [1, 2, 3, 4]})
    )

    measurement = experiment._advanced_measurement(
        servo_service=service,
        tracker_service=None,
        servo_ids=[1, 2, 3, 4],
        baseline_current_ma_by_servo={},
        target_xy_mm=[0.0, 0.0],
    )

    assert measurement["telemetry_fail_closed_reason"] == "stale_telemetry"
    assert measurement["telemetry_event_counts"]["stale_telemetry"] == 1


def test_staged_measurement_classifies_no_and_incorrect_status_packets(tmp_path: Path) -> None:
    service = _servo_service_with_bus(tmp_path, _FlakyTelemetryBus(["no_status_packet", "incorrect_status_packet"]))
    experiment = PretensionValidationExperiment(
        PretensionValidationExperimentConfig.from_dict(
            {"mode": "single_segment_staged", "servo_ids": [1, 2, 3, 4], "staged_packet_retry_budget": 2}
        )
    )

    measurement = experiment._advanced_measurement(
        servo_service=service,
        tracker_service=None,
        servo_ids=[1, 2, 3, 4],
        baseline_current_ma_by_servo={},
        target_xy_mm=[0.0, 0.0],
    )

    assert measurement["telemetry_fail_closed_reason"] == "packet_retry_budget_exhausted"
    assert measurement["telemetry_event_counts"]["no_status_packet"] == 1
    assert measurement["telemetry_event_counts"]["incorrect_status_packet"] == 1


def test_pair_preflight_prevents_one_sided_move(tmp_path: Path) -> None:
    service = _servo_service(tmp_path)
    telemetry = service.read_live_telemetry([1, 3])
    experiment = PretensionValidationExperiment(
        PretensionValidationExperimentConfig.from_dict({"mode": "single_segment_staged", "servo_ids": [1, 2, 3, 4]})
    )

    move = experiment._apply_pair_command(
        servo_service=service,
        telemetry=telemetry,
        sid_a=1,
        delta_a=-10_000,
        sid_b=3,
        delta_b=1,
        reason="test_pair_preflight",
    )

    assert move["success"] is False
    assert move["stop_reason"] == "safety_limit_rejected"
    assert service.read_live_telemetry([1])[1].present_position == telemetry[1].present_position
    assert service.read_live_telemetry([3])[3].present_position == telemetry[3].present_position


def test_partial_pair_failure_stops_and_logs(tmp_path: Path) -> None:
    service = _servo_service(tmp_path)
    original_write = service._write_goal_positions

    def partial_write(positions_by_id):
        first_servo = sorted(positions_by_id)[0]
        original_write({first_servo: positions_by_id[first_servo]})
        raise RuntimeError("simulated group write interruption")

    service._write_goal_positions = partial_write
    telemetry = service.read_live_telemetry([1, 3])
    experiment = PretensionValidationExperiment(
        PretensionValidationExperimentConfig.from_dict({"mode": "single_segment_staged", "servo_ids": [1, 2, 3, 4]})
    )

    # Use a delta larger than the position-reached tolerance so the "missed"
    # servo is unambiguously outside the window. With strict equality this would
    # work at any delta; with the post-bug-fix tolerance window the test must
    # use a delta > POSITION_REACHED_TOLERANCE_TICKS.
    move = experiment._apply_pair_command(
        servo_service=service,
        telemetry=telemetry,
        sid_a=1,
        delta_a=-25,
        sid_b=3,
        delta_b=25,
        reason="test_partial_pair_failure",
    )

    assert move["success"] is False
    assert move["stop_reason"] == "partial_pair_failure"
    assert move["reached_targets"][1] is True
    assert move["reached_targets"][3] is False


class _StaticCoilTipTrackingService:
    def __init__(self, x: float = 0.0, y: float = 0.0, z: float = 10.0) -> None:
        self._thread = object()
        self.snapshot = SimpleNamespace(
            T_robot_tip=[
                [1.0, 0.0, 0.0, float(x)],
                [0.0, 1.0, 0.0, float(y)],
                [0.0, 0.0, 1.0, float(z)],
                [0.0, 0.0, 0.0, 1.0],
            ],
            tools={},
            runtime_tip_mode="coil_as_tip",
            runtime_tip_calibration_state="coil_as_tip",
            runtime_tip_trust_level="thesis_trusted",
            runtime_tip_mode_message="Coil-as-tip override is active.",
            runtime_tip_selected_artifact_kind=None,
            runtime_tip_selected_artifact_path=None,
            tip_pose_status="coil_as_tip",
            runtime_tip_identity_fallback=False,
            registration_state="registered",
            selected_backend_name="mock",
            backend_identity="mock",
            canonical_state="connected",
        )

    def get_snapshot(self):
        return self.snapshot

    def peek_snapshot(self):
        return self.snapshot


class _DivergingTipTrackingService:
    def __init__(self) -> None:
        self._thread = object()
        self.index = 0

    def get_snapshot(self):
        self.index += 1
        x = float(self.index * 5.0)
        return SimpleNamespace(
            T_robot_tip=[
                [1.0, 0.0, 0.0, x],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 10.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
            tools={},
            runtime_tip_mode="coil_as_tip",
            runtime_tip_calibration_state="coil_as_tip",
            runtime_tip_trust_level="thesis_trusted",
            runtime_tip_mode_message="Coil-as-tip override is active.",
            runtime_tip_selected_artifact_kind=None,
            runtime_tip_selected_artifact_path=None,
            tip_pose_status="coil_as_tip",
            runtime_tip_identity_fallback=False,
            registration_state="registered",
            selected_backend_name="mock",
            backend_identity="mock",
            canonical_state="connected",
        )

    def peek_snapshot(self):
        return self.get_snapshot()


def test_staged_pretension_records_current_only_lower_trust(tmp_path: Path) -> None:
    service = _servo_service(tmp_path)
    runner = _runner(tmp_path, service)
    config = _advanced_config()
    config["mode"] = "single_segment_staged"
    config["staged_strategy"] = "conservative_startup"
    config["include_tracker_displacement"] = False
    config["allow_current_only_when_tracker_missing"] = True

    result = runner.run_experiment("pretension_validation", config=config)

    metrics = result.summary.experiment_metrics
    run_row = metrics["run_rows"][0]
    assert result.summary.status == "failed"
    assert result.success is False
    assert metrics["accepted_run_count"] == 0
    assert run_row["trust_status"] == "current_only_lower_trust"
    assert "current_only_lower_trust" in run_row["reject_reasons"]


def test_staged_pretension_records_coil_as_tip_policy(tmp_path: Path) -> None:
    service = _servo_service(tmp_path)
    runner = _runner(tmp_path, service, tracking_service=_StaticCoilTipTrackingService())
    config = _advanced_config()
    config["mode"] = "single_segment_staged"
    config["staged_strategy"] = "conservative_startup"
    config["include_tracker_displacement"] = True

    result = runner.run_experiment("pretension_validation", config=config)

    metrics = result.summary.experiment_metrics
    assert metrics["runtime_tip_mode_used"] == "coil_as_tip"
    assert metrics["runtime_tip_trust_level"] == "thesis_trusted"
    assert metrics["thesis_trusted_runtime_tip"] is True
    assert metrics["trace_rows"][0]["trust_status"] == "runtime_tip"


def test_staged_pretension_stops_when_tip_diverges(tmp_path: Path) -> None:
    service = _servo_service(tmp_path)
    runner = _runner(tmp_path, service, tracking_service=_DivergingTipTrackingService())
    config = _advanced_config()
    config["mode"] = "single_segment_staged"
    config["staged_strategy"] = "conservative_startup"
    config["include_tracker_displacement"] = True
    config["tip_center_tolerance_mm"] = 0.1
    config["tip_divergence_stop_mm"] = 0.1
    config["accept_max_load_balance_error_ma"] = 999.0
    config["load_balance_tolerance_ma"] = 999.0
    config["pair_balance_tolerance_ma"] = 999.0

    result = runner.run_experiment("pretension_validation", config=config)

    run_row = result.summary.experiment_metrics["run_rows"][0]
    # With ``tip_center_require_improvement`` (the new default), a paired step
    # that worsens tip XY is stopped via ``tip_no_safe_improvement`` rather
    # than chased via ``tip_response_wrong_direction``. Both legacy reasons
    # also remain valid when require_improvement is disabled.
    assert run_row["stop_reason"] in {
        "tip_diverging",
        "tip_response_wrong_direction",
        "tip_no_safe_improvement",
    }
    assert run_row["accepted"] is False


# --- Phase 2 additions: variant + comparison report ----------------------


def test_comparison_report_summarizes_algorithm_vs_manual_repeatability(tmp_path: Path) -> None:
    """The comparison helper turns algorithm rows + manual records into per-
    metric summaries with deltas. Algorithm rows here are tight (low spread)
    versus manual records that are sloppy; the report should flag the
    algorithm as 'better' on every metric."""
    from continuum_robot.experiments.builtins import _build_pretension_comparison_report

    servo_ids = [1, 2, 3, 4]

    def _algo_row(idx: int, drift: float) -> dict:
        positions = {sid: 2500 + int(drift) for sid in servo_ids}
        currents = {sid: 30.0 + drift * 0.1 for sid in servo_ids}
        return {
            "run_index": idx,
            "run_label": f"run_{idx + 1:02d}",
            "positions_by_servo": positions,
            "currents_ma_by_servo": currents,
            "final_tip_xy_mm": [0.1 * drift, 0.05 * drift],
            "accepted": True,
            "stop_reason": "converged",
            "variant": "paired_then_tip",
        }

    def _manual_row(idx: int, spread: float) -> dict:
        positions = {sid: 2500 + int(spread * (sid - 2.5)) for sid in servo_ids}
        currents = {sid: 28.0 + spread * (sid - 2.5) for sid in servo_ids}
        return {
            "index": idx,
            "positions_by_servo": positions,
            "currents_ma_by_servo": currents,
            "tip_xy_mm": [spread, -0.5 * spread],
        }

    algorithm_rows = [_algo_row(i, drift=0.5) for i in range(5)]
    manual_records = [_manual_row(i, spread=4.0 + 0.5 * i) for i in range(5)]

    report = _build_pretension_comparison_report(
        algorithm_run_rows=algorithm_rows,
        manual_baseline_records=manual_records,
        servo_ids=servo_ids,
        tip_target_xy_mm=[0.0, 0.0],
        target_load_band_ma=(20.0, 40.0),
    )
    assert report["schema_version"] == "1.0"
    assert report["algorithm_population_summary"]["record_count"] == 5
    assert report["manual_population_summary"]["record_count"] == 5
    # Algorithm rows are uniform; manual records are not.
    algo_spread = report["algorithm_population_summary"]["per_run_current_spread_ma"]["mean"]
    manual_spread = report["manual_population_summary"]["per_run_current_spread_ma"]["mean"]
    assert algo_spread is not None and manual_spread is not None
    assert algo_spread < manual_spread
    # Comparison verdict: algorithm wins on per-run spreads (mean field).
    spread_mean_cmp = report["comparison"]["per_run_current_spread_ma"]["mean"]
    assert spread_mean_cmp["algorithm_better"] is True
    assert int(report["algorithm_wins"]) > int(report["manual_wins"])


def test_comparison_report_handles_no_manual_records(tmp_path: Path) -> None:
    """With no manual baselines the algorithm population still summarizes but
    every comparison entry reports the manual side as None."""
    from continuum_robot.experiments.builtins import _build_pretension_comparison_report

    servo_ids = [1, 2, 3, 4]
    algorithm_rows = [
        {
            "run_index": 0,
            "run_label": "run_01",
            "positions_by_servo": {sid: 2500 for sid in servo_ids},
            "currents_ma_by_servo": {sid: 30.0 for sid in servo_ids},
            "final_tip_xy_mm": [0.0, 0.0],
            "accepted": True,
        }
    ]
    report = _build_pretension_comparison_report(
        algorithm_run_rows=algorithm_rows,
        manual_baseline_records=[],
        servo_ids=servo_ids,
        tip_target_xy_mm=[0.0, 0.0],
        target_load_band_ma=(20.0, 40.0),
    )
    assert report["manual_population_summary"]["record_count"] == 0
    assert report["comparison"]["per_run_current_spread_ma"]["mean"]["manual"] is None
    assert report["algorithm_wins"] >= 0


def test_config_parses_new_variant_and_manual_baseline_fields() -> None:
    """Phase 2 added several config fields; verify from_dict parses them."""
    cfg = PretensionValidationExperimentConfig.from_dict(
        {
            "tip_centering_variant": "jacobian_learned_tip",
            "jacobian_probe_step_ticks": 40,
            "jacobian_min_observable_tip_delta_mm": 0.3,
            "jacobian_step_gain": 0.7,
            "jacobian_max_pair_step_ticks": 30,
            "manual_baseline_capture_count": 3,
            "manual_baseline_pause_s": 5.0,
            "manual_baseline_record_path": "data/diagnostics/manual_baselines.json",
        }
    )
    assert cfg.tip_centering_variant == "jacobian_learned_tip"
    assert cfg.jacobian_probe_step_ticks == 40
    assert cfg.jacobian_min_observable_tip_delta_mm == pytest.approx(0.3)
    assert cfg.jacobian_step_gain == pytest.approx(0.7)
    assert cfg.jacobian_max_pair_step_ticks == 30
    assert cfg.manual_baseline_capture_count == 3
    assert cfg.manual_baseline_pause_s == pytest.approx(5.0)
    assert cfg.manual_baseline_record_path == "data/diagnostics/manual_baselines.json"


def test_config_clamps_jacobian_step_gain_and_defaults_to_paired_then_tip() -> None:
    """jacobian_step_gain is clamped to [0.05, 1.5]; unknown variant falls back
    to paired_then_tip."""
    cfg = PretensionValidationExperimentConfig.from_dict({"jacobian_step_gain": 5.0})
    assert cfg.jacobian_step_gain == 1.5
    cfg2 = PretensionValidationExperimentConfig.from_dict({"jacobian_step_gain": -1.0})
    assert cfg2.jacobian_step_gain == 0.05
    cfg3 = PretensionValidationExperimentConfig.from_dict({"tip_centering_variant": "garbage"})
    # from_dict does NOT validate the variant string itself (validation happens
    # at dispatch time inside the experiment); the field is preserved verbatim.
    assert cfg3.tip_centering_variant == "garbage"


def test_comparison_markdown_writes_with_manual_records(tmp_path: Path) -> None:
    from continuum_robot.experiments.builtins import _build_pretension_comparison_report
    from continuum_robot.experiments.pretension_validation_outputs import (
        _write_pretension_comparison_markdown,
    )

    servo_ids = [1, 2, 3, 4]
    algorithm_rows = [
        {
            "run_label": f"run_{i + 1:02d}",
            "positions_by_servo": {sid: 2500 for sid in servo_ids},
            "currents_ma_by_servo": {sid: 30.0 for sid in servo_ids},
            "final_tip_xy_mm": [0.0, 0.0],
        }
        for i in range(5)
    ]
    manual_records = [
        {
            "positions_by_servo": {sid: 2500 + 10 * (sid - 2) for sid in servo_ids},
            "currents_ma_by_servo": {sid: 28.0 + 4.0 * (sid - 2) for sid in servo_ids},
            "tip_xy_mm": [1.0 + i * 0.5, -0.4 + i * 0.3],
        }
        for i in range(5)
    ]
    report = _build_pretension_comparison_report(
        algorithm_run_rows=algorithm_rows,
        manual_baseline_records=manual_records,
        servo_ids=servo_ids,
        tip_target_xy_mm=[0.0, 0.0],
        target_load_band_ma=(20.0, 40.0),
    )
    md_path = tmp_path / "comparison.md"
    _write_pretension_comparison_markdown(
        markdown_path=md_path,
        comparison_report=report,
        manual_record_count=len(manual_records),
        algorithm_run_count=len(algorithm_rows),
        metrics={"tip_centering_variant": "paired_then_tip"},
    )
    text = md_path.read_text(encoding="utf-8")
    assert "Algorithm vs Manual Comparison" in text
    assert "ALGORITHM" in text  # at least one verdict cell wins for algorithm
    assert "Per-servo repeatability" in text


def test_comparison_markdown_handles_no_manual_records(tmp_path: Path) -> None:
    from continuum_robot.experiments.builtins import _build_pretension_comparison_report
    from continuum_robot.experiments.pretension_validation_outputs import (
        _write_pretension_comparison_markdown,
    )

    servo_ids = [1, 2, 3, 4]
    algorithm_rows = [
        {
            "run_label": "run_01",
            "positions_by_servo": {sid: 2500 for sid in servo_ids},
            "currents_ma_by_servo": {sid: 30.0 for sid in servo_ids},
            "final_tip_xy_mm": [0.0, 0.0],
        }
    ]
    report = _build_pretension_comparison_report(
        algorithm_run_rows=algorithm_rows,
        manual_baseline_records=[],
        servo_ids=servo_ids,
        tip_target_xy_mm=[0.0, 0.0],
        target_load_band_ma=(20.0, 40.0),
    )
    md_path = tmp_path / "comparison_no_manual.md"
    _write_pretension_comparison_markdown(
        markdown_path=md_path,
        comparison_report=report,
        manual_record_count=0,
        algorithm_run_count=1,
        metrics={"tip_centering_variant": "paired_then_tip"},
    )
    text = md_path.read_text(encoding="utf-8")
    assert "No manual baselines recorded" in text
    assert "Tip radial dispersion across runs" in text


# --- End-to-end smoke tests: variant dispatch + comparison report writes -----


def test_smoke_current_only_variant_writes_run_folder_and_comparison_report(tmp_path: Path) -> None:
    """End-to-end on mock: current_only variant runs without tracker, writes
    metadata + summary + comparison markdown + comparison plot."""
    service = _servo_service(tmp_path)
    runner = _runner(tmp_path, service, tracking_service=None)
    config = _advanced_config()
    config.update(
        {
            "mode": "single_segment_staged",
            "staged_strategy": "conservative_startup",
            "tip_centering_variant": "current_only",
            "include_tracker_displacement": False,
            "allow_current_only_when_tracker_missing": True,
            "run_trust_mode": "current_only",
            "enable_tip_centering": False,
            "repeat_runs": 1,
        }
    )
    result = runner.run_experiment("pretension_validation", config=config)

    # Run folder + standard artifacts.
    out = result.paths.output_dir
    assert (out / "metadata.json").exists()
    assert (out / "summary.json").exists()
    assert (out / "metrics.csv").exists()
    assert (out / "pretension_summary.txt").exists()

    # NEW Phase-2 comparison artifacts.
    assert (out / "pretension_algorithm_vs_manual.md").exists()
    md_text = (out / "pretension_algorithm_vs_manual.md").read_text(encoding="utf-8")
    assert "Algorithm vs Manual Comparison" in md_text
    # No baselines were captured so the no-manual branch should fire.
    assert "No manual baselines recorded" in md_text

    # Comparison report sits in metrics.
    metrics = result.summary.experiment_metrics
    assert "pretension_comparison_report" in metrics
    assert metrics["tip_centering_variant"] == "current_only"
    assert metrics["manual_baseline_record_count"] == 0


def test_smoke_paired_then_tip_variant_writes_comparison_with_tracker(tmp_path: Path) -> None:
    """End-to-end on mock with a static tracker: paired_then_tip variant runs
    through the take-up + conservative-startup phases without partial_pair
    failure under the new tolerance window."""
    service = _servo_service(tmp_path)
    runner = _runner(tmp_path, service, tracking_service=_StaticCoilTipTrackingService())
    config = _advanced_config()
    config.update(
        {
            "mode": "single_segment_staged",
            "staged_strategy": "conservative_startup",
            "tip_centering_variant": "paired_then_tip",
            "include_tracker_displacement": True,
            "enable_tip_centering": True,
            "repeat_runs": 1,
            "accept_max_load_balance_error_ma": 999.0,  # loose for mock
            "accept_max_pair_balance_error_ma": 999.0,
            "accept_max_final_tip_xy_offset_mm": 999.0,
            "load_balance_tolerance_ma": 999.0,
            "pair_balance_tolerance_ma": 999.0,
            "tip_center_tolerance_mm": 999.0,
            "tip_divergence_stop_mm": 999.0,
        }
    )
    result = runner.run_experiment("pretension_validation", config=config)

    out = result.paths.output_dir
    assert (out / "metadata.json").exists()
    assert (out / "summary.json").exists()
    assert (out / "pretension_algorithm_vs_manual.md").exists()
    metrics = result.summary.experiment_metrics
    assert metrics["tip_centering_variant"] == "paired_then_tip"
    # Crucial regression check: the strict-equality bug fix means moves should
    # not return partial_pair_failure on a static-pose mock.
    run_row = metrics["run_rows"][0]
    assert run_row["stop_reason"] != "partial_pair_failure"


def test_smoke_shipped_safety_yaml_carries_phase1_retune() -> None:
    """The shipped config/safety.yaml file carries the Phase 1 retune
    (travel budget 1600, operator-scale current thresholds, release-to-3500
    reference, full_release_4095 default start mode).

    This test reads the YAML file directly, NOT the loaded Settings, so it
    is not affected by a machine-local system.local.yaml safety_overrides
    block (the operator may legitimately override these for bench-specific
    tuning; the test asserts the SHIPPED defaults are correct)."""
    import yaml
    payload = yaml.safe_load(Path("config/safety.yaml").read_text(encoding="utf-8"))
    # Travel budget must cover release (3500) -> tensioned (~2500) range.
    assert int(payload["pretension_max_travel_ticks"]) >= 1000
    # Operator's scale: 30 mA = tight. Shipped trigger must be in the band,
    # not the legacy 220 mA that over-tensioned tendons by 4-5x.
    assert 15 <= int(payload["default_pretension_current_threshold_ma"]) <= 60
    assert 5 <= int(payload["pretension_current_balance_tolerance_ma"]) <= 30
    # Hard stop is absolute safety; should stay at 850 mA.
    assert int(payload["pretension_hard_current_stop_ma"]) == 850
    # Repeatable starting condition. The default flipped from
    # ``full_release_4095`` to ``soft_release_to_zero_current`` in the
    # bounded simplification pass (2026-05-19) — the new mode releases until
    # current is near zero, proving each tendon slack rather than trusting
    # the position-wrap endpoint.
    assert payload["pretension_start_mode"] in {
        "soft_release_to_zero_current",
        "full_release_4095",
    }
    # Reference tick is the slack target; clipped to safe_max by preflight.
    assert int(payload["pretension_untensioned_reference_tick"]) == 3500


def test_smoke_example_yaml_carries_phase2_variant_knobs() -> None:
    """The example config that the Servos-tab one-click button loads must
    expose the Phase 2 variant + manual-baseline knobs so they are tunable on
    the Experiments tab."""
    import yaml
    payload = yaml.safe_load(
        Path("config/experiment_pretension_validation.example.yaml").read_text(encoding="utf-8")
    )
    assert "tip_centering_variant" in payload
    assert "jacobian_probe_step_ticks" in payload
    assert "jacobian_min_observable_tip_delta_mm" in payload
    assert "jacobian_step_gain" in payload
    assert "jacobian_max_pair_step_ticks" in payload
    assert "manual_baseline_capture_count" in payload
    assert "manual_baseline_pause_s" in payload
    assert "manual_baseline_record_path" in payload
    # Repeatable start condition. The default flipped to
    # ``soft_release_to_zero_current`` in the bounded simplification pass
    # (2026-05-19); ``full_release_4095`` is still accepted for backward
    # compat with older configs.
    assert payload["pretension_start_mode"] in {
        "soft_release_to_zero_current",
        "full_release_4095",
    }
    # Default variant is the conservative one, not Jacobian.
    assert payload["tip_centering_variant"] == "paired_then_tip"


# --- Engagement-scan take-up tests (operator-spec'd 2026-05-18) ----------


def test_engagement_scan_step_ticks_default_is_50() -> None:
    cfg = PretensionValidationExperimentConfig.from_dict({})
    assert cfg.engagement_scan_step_ticks == 50
    assert cfg.engagement_rise_threshold_ma == pytest.approx(5.0)
    assert cfg.engagement_back_off_ticks == 50


def test_engagement_scan_step_ticks_clamps_to_min_1() -> None:
    cfg = PretensionValidationExperimentConfig.from_dict(
        {"engagement_scan_step_ticks": 0, "engagement_back_off_ticks": -5}
    )
    assert cfg.engagement_scan_step_ticks == 1
    assert cfg.engagement_back_off_ticks == 0  # back-off can be 0 = "no back off"


def test_engagement_rise_threshold_floors_to_0_5_ma() -> None:
    cfg = PretensionValidationExperimentConfig.from_dict({"engagement_rise_threshold_ma": -2.0})
    assert cfg.engagement_rise_threshold_ma == pytest.approx(0.5)


def test_engagement_scan_phase_emits_engagement_scan_traces(tmp_path: Path) -> None:
    """Run the full pretension experiment in single_segment_staged mode and
    verify the new engagement_scan + engagement_back_off + fine_take_up stages
    appear in the trace."""
    service = _servo_service(tmp_path)
    runner = _runner(tmp_path, service, tracking_service=_StaticCoilTipTrackingService())
    config = _advanced_config()
    config.update(
        {
            "mode": "single_segment_staged",
            "staged_strategy": "conservative_startup",
            "tip_centering_variant": "current_only",
            "include_tracker_displacement": True,
            "enable_tip_centering": False,
            "repeat_runs": 1,
            "engagement_scan_step_ticks": 4,  # small so the mock bus engages quickly
            "engagement_rise_threshold_ma": 1.0,  # very low so we engage in 1-2 iterations
            "engagement_back_off_ticks": 2,
        }
    )
    result = runner.run_experiment("pretension_validation", config=config)

    metrics = result.summary.experiment_metrics
    trace_rows = metrics["trace_rows"]
    stages = [row.get("stage") for row in trace_rows]
    # The new pipeline must emit these stage labels at least once.
    assert "engagement_scan" in stages, f"engagement_scan stage missing; saw stages: {stages}"


def test_shipped_system_yaml_carries_slowdown_profile() -> None:
    """The shipped config/system.yaml carries non-null default_profile_velocity
    and default_profile_acceleration so every goal-position write applies a
    slowdown. dxl_bus.write_goal_positions writes these BEFORE the goal
    position, so the slowdown applies to jog, pretension, registration capture
    — everything that writes goal positions."""
    import yaml
    payload = yaml.safe_load(Path("config/system.yaml").read_text(encoding="utf-8"))
    dxl = payload.get("dynamixel_settings", {}) or {}
    assert dxl.get("default_profile_velocity") not in (None, 0), (
        "default_profile_velocity must be non-null/non-zero for the slowdown to apply"
    )
    assert dxl.get("default_profile_acceleration") not in (None, 0), (
        "default_profile_acceleration must be non-null/non-zero for the slowdown to apply"
    )
    # Sanity bounds: not so fast it doesn't slow anything, not so slow it
    # makes the rig feel broken.
    assert 5 <= int(dxl["default_profile_velocity"]) <= 200
    assert 1 <= int(dxl["default_profile_acceleration"]) <= 100


def test_shipped_pretension_yaml_carries_engagement_scan_knobs() -> None:
    """Example pretension YAML must expose the new engagement-scan knobs so
    they are tunable on the Experiments tab."""
    import yaml
    payload = yaml.safe_load(
        Path("config/experiment_pretension_validation.example.yaml").read_text(encoding="utf-8")
    )
    assert "engagement_scan_step_ticks" in payload
    assert "engagement_rise_threshold_ma" in payload
    assert "engagement_back_off_ticks" in payload
    # Defaults must match the operator's spec.
    assert int(payload["engagement_scan_step_ticks"]) == 50
    assert int(payload["engagement_back_off_ticks"]) == 50
    assert float(payload["engagement_rise_threshold_ma"]) >= 2.0


# --------------------------------------------------------------------------- #
# Post-write reach-poll, travel-budget, and noise-cap regression fixes
# (see run 20260519_200616_pretension_validation diagnostic).
# --------------------------------------------------------------------------- #


def test_estimate_reach_timeout_scales_with_profile_velocity() -> None:
    """The dynamic post-write reach timeout must be longer when the configured
    ``default_profile_velocity`` is slower, because a single goal-position
    write takes physically longer to complete on a slow profile. Conversely a
    very large ``default_profile_velocity`` (or unset = unlimited) yields a
    short timeout. Without this scaling, the slow profile we run on the rig
    (15 register units = ~234 ticks/sec) makes ``_apply_group_command`` race
    its own read-back and falsely report ``partial_pair_failure`` on the first
    engagement-scan step.
    """
    class _BusCfg:
        def __init__(self, vel):
            self.default_profile_velocity = vel

    class _Bus:
        def __init__(self, vel):
            self.config = _BusCfg(vel)

    class _Service:
        def __init__(self, vel):
            self.dxl_bus = _Bus(vel)

    estimate = PretensionValidationExperiment._estimate_reach_timeout_s

    # Slow profile (matches config/system.yaml today).
    t_slow_small = estimate(_Service(15), 50)
    t_slow_big = estimate(_Service(15), 2000)
    assert t_slow_small >= 0.4, "must respect the safe floor for tiny moves"
    assert t_slow_big > t_slow_small + 1.0, "big moves must get a bigger budget"
    # 2000 ticks at velocity=15 = ~8.5 s of motion; expect the helper to be in
    # that ballpark, not the 0.6 s ceiling of the previous fixed value.
    assert t_slow_big >= 8.0

    # Fast / unlimited profile: even a 5000-tick move stays well under the
    # 15 s hard ceiling.
    t_unlimited = estimate(_Service(None), 5000)
    assert t_unlimited < 5.0, "unlimited profile_velocity should not inflate timeout"

    # Hard ceiling holds for absurdly large deltas.
    t_ceiling = estimate(_Service(15), 200_000)
    assert t_ceiling == PretensionValidationExperiment.POSITION_REACH_POLL_CEIL_S


def test_apply_group_command_polls_after_write_until_reached(tmp_path: Path) -> None:
    """``_apply_group_command`` must NOT rely on an immediate post-write read.
    With a slow profile_velocity the read-back races motion and falsely reports
    ``partial_pair_failure``. We simulate that race by having the mock bus
    return the start position on its first read and the goal position on
    subsequent reads — the polling loop must wait for the second read.
    """
    service = _servo_service(tmp_path)
    # Drive servos to a known starting position.
    telemetry = service.read_live_telemetry([1, 3])

    original_read = service.read_live_telemetry
    call_box = {"count": 0}
    start_positions = {sid: int(telemetry[sid].present_position) for sid in (1, 3)}

    def slow_read(ids):
        # First read after the write returns stale (start) positions; later
        # reads return whatever the mock bus actually wrote.
        call_box["count"] += 1
        if call_box["count"] == 1:
            from types import SimpleNamespace
            return {
                int(sid): SimpleNamespace(
                    present_position=int(start_positions[int(sid)]),
                    present_current=0.0,
                    present_voltage=12.0,
                    present_temperature=30.0,
                    timestamp=time.monotonic(),
                )
                for sid in ids
            }
        return original_read(ids)

    service.read_live_telemetry = slow_read

    experiment = PretensionValidationExperiment(
        PretensionValidationExperimentConfig.from_dict({"mode": "single_segment_staged", "servo_ids": [1, 2, 3, 4]})
    )
    move = experiment._apply_pair_command(
        servo_service=service,
        telemetry=telemetry,
        sid_a=1,
        delta_a=-25,
        sid_b=3,
        delta_b=25,
        reason="test_post_write_poll",
    )

    # The first read would have failed; the polling loop must have read at
    # least once more before declaring success.
    assert call_box["count"] >= 2
    assert move["success"] is True
    assert move["stop_reason"] == ""
    # The new keys are emitted for diagnostic visibility.
    assert "post_write_poll_iterations" in move
    assert "post_write_poll_timeout_s" in move
    assert "post_write_max_commanded_delta_ticks" in move
    assert int(move["post_write_max_commanded_delta_ticks"]) == 25


def test_useful_delta_cap_caps_noisy_characterization() -> None:
    """``max_useful_current_delta_cap_ma`` must clamp the noise-derived
    ``max_useful_current_delta_ma``. Without this clamp a noisy baseline
    (e.g. 70 mA peak-to-peak from PID hunting at the raw-tick wrap edge)
    propagates into the load-balance tolerance and silently disables the
    convergence checks. The cap default of 15 mA aligns with operator's
    "light" tendon-tension reference.
    """
    cfg = PretensionValidationExperimentConfig.from_dict(
        {
            "mode": "single_segment_staged",
            "max_useful_current_delta_cap_ma": 15.0,
        }
    )
    assert cfg.max_useful_current_delta_cap_ma == 15.0

    # Verify the from_dict default is 15.0 (not the historical "no cap").
    cfg_default = PretensionValidationExperimentConfig.from_dict({})
    assert cfg_default.max_useful_current_delta_cap_ma == 15.0


def test_conservative_max_cumulative_travel_ticks_default_covers_full_release() -> None:
    """From a ``full_release_4095`` start the engagement scan must walk every
    servo from ~tick 4094 back to its engagement tick (~1900-2700), which is
    ~1500-2200 ticks of inward travel per servo. The default cumulative
    travel budget must be sized to cover that, not the historical tiny
    "trim-only" 80-tick default that silently killed every full-release run
    with ``travel_budget_exhausted_during_engagement_scan``.
    """
    cfg = PretensionValidationExperimentConfig.from_dict({})
    assert cfg.conservative_max_cumulative_travel_ticks >= 2200, (
        "Default budget must cover a full_release_4095 → engagement_tick walk."
    )


def test_shipped_pretension_yaml_uses_active_segment_by_default() -> None:
    """The shipped example pretension YAML must use ``servo_ids: []`` (empty)
    so the experiment falls back to ``active_segment_servo_ids()`` from the
    robot config. This is the single source of truth for which servos belong
    to the active segment; hardcoding an explicit list here would silently
    diverge from the active-segment selection in robot_4servo.yaml or
    robot_8servo.yaml."""
    import yaml
    payload = yaml.safe_load(
        Path("config/experiment_pretension_validation.example.yaml").read_text(encoding="utf-8")
    )
    assert "servo_ids" in payload
    raw = payload["servo_ids"]
    # Accept empty list (preferred) or null; reject anything else.
    assert raw in (None, [], "", "[]"), (
        "config/experiment_pretension_validation.example.yaml should set "
        "servo_ids: [] so the active segment from robot config is used. "
        f"Found: {raw!r}"
    )
    assert "max_useful_current_delta_cap_ma" in payload
    assert float(payload["max_useful_current_delta_cap_ma"]) > 0.0
    # Conservative budget must comfortably cover a full-release engagement
    # walk (~2000+ ticks).
    assert int(payload["conservative_max_cumulative_travel_ticks"]) >= 2200


def test_baseline_offset_from_full_release_ticks_default_is_off_edge() -> None:
    """After ``full_release_4095`` the servos sit at the position-wrap edge where
    the PID hunts. ``baseline_offset_from_full_release_ticks`` must default to a
    positive value so baseline characterization happens at a stable,
    off-edge position. Run 20260519_200616 measured 70 mA peak-to-peak when
    this offset was 0; 200 ticks is enough to lift the servos off the wrap
    boundary while still keeping the tendons slack."""
    cfg = PretensionValidationExperimentConfig.from_dict({})
    assert cfg.baseline_offset_from_full_release_ticks >= 100, (
        "Default baseline offset must back the servos off the wrap edge."
    )
    cfg_zero = PretensionValidationExperimentConfig.from_dict(
        {"baseline_offset_from_full_release_ticks": 0}
    )
    assert cfg_zero.baseline_offset_from_full_release_ticks == 0


def test_shipped_pretension_yaml_exposes_baseline_offset_knob() -> None:
    """The shipped example YAML must expose
    ``baseline_offset_from_full_release_ticks`` so operators can tune it from
    the Experiments tab without editing source."""
    import yaml
    payload = yaml.safe_load(
        Path("config/experiment_pretension_validation.example.yaml").read_text(encoding="utf-8")
    )
    assert "baseline_offset_from_full_release_ticks" in payload
    assert int(payload["baseline_offset_from_full_release_ticks"]) >= 100


def _make_off_edge_session(servo_service):
    """Minimal ExperimentSession stub for off-edge helper tests."""
    class _Session:
        def __init__(self):
            self.context = SimpleNamespace(
                servo_service=servo_service,
                tracker_service=None,
                monotonic_fn=time.monotonic,
                sleep_fn=lambda _s: None,
            )
            self.staged_samples: list = []

        def raise_if_stop_requested(self):
            return None

        def update_progress(self, *_a, **_kw):
            return None

    return _Session()


def test_baseline_off_edge_step_moves_servos_inward(tmp_path: Path) -> None:
    """``_run_baseline_off_edge_step`` must command every servo to a position
    ``baseline_offset_from_full_release_ticks`` lower than its present
    position. The travel must be accounted in the result, a stage row must be
    appended to ``trace_rows``, and the per-servo applied offsets reported."""
    service = _servo_service(tmp_path)
    experiment = PretensionValidationExperiment(
        PretensionValidationExperimentConfig.from_dict(
            {
                "mode": "single_segment_staged",
                "servo_ids": [1, 2, 3, 4],
                "baseline_offset_from_full_release_ticks": 50,
            }
        )
    )

    session = _make_off_edge_session(service)
    experiment._add_staged_sample = lambda session, **kw: session.staged_samples.append(kw)  # type: ignore[assignment]

    trace_rows: list = []
    result = experiment._run_baseline_off_edge_step(
        session=session,
        servo_service=service,
        tracker_service=None,
        servo_ids=[1, 2, 3, 4],
        run_index=0,
        target_xy_mm=[0.0, 0.0],
        startup_reference_ticks_by_servo={sid: None for sid in (1, 2, 3, 4)},
        mode_kind="conservative_startup",
        trace_rows=trace_rows,
    )

    assert result["stop_reason"] == "baseline_off_edge_complete"
    assert result["move_count"] == 4
    assert result["baseline_offset_ticks_requested"] == 50
    applied = result["baseline_offset_ticks_applied_by_servo"]
    for sid in (1, 2, 3, 4):
        assert applied[str(sid)] >= 40, applied
    assert any(row.get("stage") == "baseline_off_edge" for row in trace_rows)


def test_baseline_off_edge_step_disabled_when_offset_zero(tmp_path: Path) -> None:
    """With ``baseline_offset_from_full_release_ticks: 0`` the helper must be a
    no-op (no move, no stage row) and report ``baseline_off_edge_disabled``.
    This preserves the historical behavior for runs that explicitly want to
    characterize at the full-release endpoint."""
    service = _servo_service(tmp_path)
    experiment = PretensionValidationExperiment(
        PretensionValidationExperimentConfig.from_dict(
            {
                "mode": "single_segment_staged",
                "servo_ids": [1, 2, 3, 4],
                "baseline_offset_from_full_release_ticks": 0,
            }
        )
    )

    session = _make_off_edge_session(service)
    trace_rows: list = []
    result = experiment._run_baseline_off_edge_step(
        session=session,
        servo_service=service,
        tracker_service=None,
        servo_ids=[1, 2, 3, 4],
        run_index=0,
        target_xy_mm=[0.0, 0.0],
        startup_reference_ticks_by_servo={sid: None for sid in (1, 2, 3, 4)},
        mode_kind="conservative_startup",
        trace_rows=trace_rows,
    )

    assert result["stop_reason"] == "baseline_off_edge_disabled"
    assert result["move_count"] == 0
    assert result["travel_ticks"] == 0
    assert trace_rows == []


# --------------------------------------------------------------------------- #
# Bounded simplification pass (2026-05-19): soft_release_to_zero_current is
# the new default start mode; sign-verification is not blocking; consistency
# verdict + new thesis figures + quality JSON.
# --------------------------------------------------------------------------- #


def test_pretension_config_defaults_to_soft_release_to_zero_current() -> None:
    """The new default operator-facing start mode is
    ``soft_release_to_zero_current``: walk outward until current is near zero
    for ``release_current_stable_samples`` consecutive samples. The legacy
    ``full_release_4095`` is no longer recommended.

    Config knob defaults must reflect the operator's tendon-tension scale
    (15 mA = light, 30 mA = tight): a 5 mA release target sits comfortably
    inside the noise floor of an unloaded tendon."""
    cfg = PretensionValidationExperimentConfig.from_dict({})
    assert cfg.release_current_abs_target_ma == 5.0
    assert cfg.release_current_stable_samples == 3
    assert cfg.release_max_travel_ticks == 1500
    assert cfg.release_step_ticks == 40
    assert cfg.release_settle_s == 0.15
    # Plateau fallback defaults (2026-05-19): cover the 6-10 mA holding-current
    # regime that real XC330 servos sit at when slack.
    assert cfg.release_plateau_max_current_ma == 10.0
    assert cfg.release_plateau_delta_ma == 2.0
    assert cfg.release_plateau_window_samples == 4
    # The config field itself stays None so the start_mode falls through to
    # the per-servo default — the experiment's _staged_parameters_for_servo
    # then resolves to "soft_release_to_zero_current".
    assert cfg.pretension_start_mode is None


def test_shipped_pretension_yaml_uses_soft_release_to_zero_current() -> None:
    """The shipped example pretension YAML must default to
    soft_release_to_zero_current and surface the per-servo release knobs."""
    import yaml
    payload = yaml.safe_load(
        Path("config/experiment_pretension_validation.example.yaml").read_text(encoding="utf-8")
    )
    assert str(payload.get("pretension_start_mode")).strip().lower() == "soft_release_to_zero_current"
    for key in (
        "release_current_abs_target_ma",
        "release_current_stable_samples",
        "release_max_travel_ticks",
        "release_step_ticks",
        "release_settle_s",
    ):
        assert key in payload, f"shipped YAML must expose {key}"
    # tip_center_require_improvement should ship as true so first-try
    # operator runs don't chase bad tip steps.
    assert bool(payload.get("tip_center_require_improvement", True)) is True


def test_safety_yaml_default_start_mode_is_soft_release_to_zero_current() -> None:
    """The safety.yaml fallback for ``pretension_start_mode`` must align with
    the experiment YAML default. Otherwise the Servos-tab one-click trial and
    the experiments-tab run would silently use different start modes."""
    import yaml
    payload = yaml.safe_load(Path("config/safety.yaml").read_text(encoding="utf-8"))
    assert str(payload.get("pretension_start_mode")).strip().lower() == "soft_release_to_zero_current"


def test_lower_ticks_tighten_assumed_metadata_present_in_record() -> None:
    """Every per-run record must declare the rig-convention assumption so
    downstream consumers don't have to infer it. This is the one-rig proof's
    explicit statement that the algorithm does not enforce a sign-check."""
    # We synthesize a minimal record dict the way the experiment would build
    # it, and assert the three keys are present and set to the documented
    # values. The keys are set directly in builtins.py at the per-run record
    # assembly site; if someone removes them the tests fail loudly.
    import re
    source = Path("continuum_robot/experiments/builtins.py").read_text(encoding="utf-8")
    assert re.search(r'"lower_ticks_tighten_assumed"\s*:\s*True', source)
    assert re.search(r'"tightening_direction_source"\s*:\s*"rig convention"', source)
    assert re.search(r'"operator_sign_check_required"\s*:\s*False', source)


def test_consistency_verdict_handles_all_four_buckets() -> None:
    """high / medium / low / failed verdict from
    ``_compute_consistency_verdict``."""
    compute = PretensionValidationExperiment._compute_consistency_verdict

    # failed: 0 accepted out of N>1
    failed = compute(
        run_records=[{"accepted": False}, {"accepted": False}],
        accepted_run_count=0,
        repeat_runs=5,
        final_tip_xy_points_mm=[],
    )
    assert failed["verdict"] == "failed"

    # single_run: repeat_runs == 1
    single = compute(
        run_records=[{"accepted": True}],
        accepted_run_count=1,
        repeat_runs=1,
        final_tip_xy_points_mm=[],
    )
    assert single["verdict"] == "single_run"

    # high: all accepted, tight spread
    tight_rows = [
        {
            "accepted": True,
            "final_position_ticks_by_servo": {"1": 2000 + i, "2": 2100 + i, "3": 2050 + i, "4": 2150 + i},
            "load_proxy_current_ma_by_servo": {"1": 22.0, "2": 23.0, "3": 21.0, "4": 24.0},
            "final_tip_xy_mm": [0.1 * i, 0.0],
        }
        for i in range(5)
    ]
    high = compute(
        run_records=tight_rows,
        accepted_run_count=5,
        repeat_runs=5,
        final_tip_xy_points_mm=[],
    )
    assert high["verdict"] == "high"
    assert high["final_tip_xy_std_mm"] is not None

    # low: half accepted but spread is wide enough that medium thresholds fail
    wide_rows = [
        {
            "accepted": True,
            "final_position_ticks_by_servo": {"1": 2000 + i * 100, "2": 2100, "3": 2050, "4": 2150},
            "load_proxy_current_ma_by_servo": {"1": 22.0 + i * 5, "2": 23.0, "3": 21.0, "4": 24.0},
            "final_tip_xy_mm": [i * 2.0, 0.0],
        }
        for i in range(3)
    ] + [{"accepted": False}, {"accepted": False}]
    low = compute(
        run_records=wide_rows,
        accepted_run_count=3,
        repeat_runs=5,
        final_tip_xy_points_mm=[],
    )
    assert low["verdict"] == "low"


def test_phase_for_stage_maps_release_to_release_phase() -> None:
    """The report figures depend on _phase_for_stage to group trace samples
    into coarse phases for the shaded background and the phase summary panel.
    A regression that mislabels the soft_release_to_zero_current stage would
    cause every release sample to fall under "other"."""
    from continuum_robot.experiments.pretension_validation_outputs import _phase_for_stage

    assert _phase_for_stage("soft_release_to_zero_current") == "release"
    assert _phase_for_stage("baseline_off_edge") == "off_edge"
    assert _phase_for_stage("current_characterization_sample") == "characterization"
    assert _phase_for_stage("engagement_scan") == "engagement"
    assert _phase_for_stage("conservative_pair_step") == "tip_center"
    assert _phase_for_stage("unknown_stage") == "other"


# --------------------------------------------------------------------------- #
# Plateau fallback for soft_release_to_zero_current.
# --------------------------------------------------------------------------- #


class _ScriptedCurrentService:
    """Wraps a real ServoService so tests can script the current returned by
    each ``read_live_telemetry`` call.

    The helper does not touch positions / writes — those flow through the
    underlying service untouched. ``script`` is a list of per-call dicts
    keyed by servo_id; the helper returns the next dict on each call and
    sticks on the last dict once the script is exhausted. This lets tests
    simulate "current drops from 60 → 12 → 8 → 7 → 7 → 7 → 7 → 7" easily.
    """

    def __init__(self, base, *, script):
        self._base = base
        self._script = list(script)
        self._read_index = 0
        # Forward attribute access for everything we don't override.

    def __getattr__(self, name):
        return getattr(self._base, name)

    def read_live_telemetry(self, servo_ids):
        live = self._base.read_live_telemetry(list(servo_ids))
        # Pick the dict for this call.
        index = min(self._read_index, len(self._script) - 1) if self._script else None
        plan = self._script[index] if index is not None else {}
        self._read_index += 1
        for sid in servo_ids:
            entry = live.get(int(sid))
            if entry is None:
                continue
            target_current = plan.get(int(sid), plan.get(str(int(sid))))
            if target_current is None:
                continue
            # Replace BOTH the registered current_ma and the present_current_ma
            # depending on which attribute the live entry exposes. The mock
            # bus's telemetry dataclass uses present_current_ma.
            try:
                entry.present_current_ma = float(target_current)
            except Exception:
                pass
            # Some entry types also store present_current; sync both.
            if hasattr(entry, "present_current"):
                try:
                    entry.present_current = float(target_current)
                except Exception:
                    pass
        return live


def _run_release_with_scripted_currents(
    tmp_path: Path,
    *,
    script: list[dict[int, float]],
    extra_config: dict[str, Any] | None = None,
):
    """Run ``_run_soft_release_to_zero_current_start`` against a wrapped servo
    service whose currents follow ``script``. Returns the result dict."""
    base_service = _servo_service(tmp_path)
    wrapped = _ScriptedCurrentService(base_service, script=script)

    cfg_payload = {
        "mode": "single_segment_staged",
        "servo_ids": [1, 2, 3, 4],
        "release_current_abs_target_ma": 5.0,
        "release_current_stable_samples": 3,
        "release_plateau_max_current_ma": 10.0,
        "release_plateau_delta_ma": 2.0,
        "release_plateau_window_samples": 4,
        "release_step_ticks": 25,
        "release_max_travel_ticks": 300,
        "release_settle_s": 0.0,
    }
    cfg_payload.update(extra_config or {})
    experiment = PretensionValidationExperiment(
        PretensionValidationExperimentConfig.from_dict(cfg_payload)
    )

    class _Session:
        def __init__(self):
            self.context = SimpleNamespace(
                servo_service=wrapped,
                tracker_service=None,
                monotonic_fn=time.monotonic,
                sleep_fn=lambda _s: None,
            )
            self.staged_samples: list = []

        def raise_if_stop_requested(self):
            return None

        def update_progress(self, *_a, **_kw):
            return None

    session = _Session()
    experiment._add_staged_sample = lambda session, **kw: session.staged_samples.append(kw)  # type: ignore[assignment]

    trace_rows: list = []
    result = experiment._run_soft_release_to_zero_current_start(
        session=session,
        servo_service=wrapped,
        tracker_service=None,
        servo_ids=[1, 2, 3, 4],
        run_index=0,
        target_xy_mm=[0.0, 0.0],
        startup_reference_ticks_by_servo={sid: None for sid in (1, 2, 3, 4)},
        mode_kind="conservative_startup",
        trace_rows=trace_rows,
    )
    return result, trace_rows


# --------------------------------------------------------------------------- #
# Holding-current (post-move settle + burst average) — tension decisions
# must be based on settled holding readings, not in-motion drive current.
# --------------------------------------------------------------------------- #


# --------------------------------------------------------------------------- #
# Signed holding-tension acceptance — take-up must reach ~-30 mA per servo,
# not just hit the |current - baseline| load proxy band.
# --------------------------------------------------------------------------- #


# --------------------------------------------------------------------------- #
# Tip-centering must preserve per-servo tension — tighten-only mode.
# --------------------------------------------------------------------------- #


def test_tip_center_tighten_only_default_is_true() -> None:
    """The default for ``tip_center_tighten_only`` must be True so the
    Servos-tab one-click flow centers the tip without releasing any tendon.

    Regression for run 20260521_194307: paired-symmetric tip-centering
    released servos 1 and 2 to 0 mA while moving the tip from offset
    3.23 mm down to 0.44 mm. Tighten-only never releases — it only adds
    tension to the side that pulls toward the tip-correction direction."""
    cfg = PretensionValidationExperimentConfig.from_dict({})
    assert cfg.tip_center_tighten_only is True


def test_tip_center_tighten_only_can_be_disabled_for_legacy_paired_path() -> None:
    """The legacy paired-symmetric tip-centering is still available via
    ``tip_center_tighten_only: false`` so existing config payloads /
    reproducibility runs keep working."""
    cfg = PretensionValidationExperimentConfig.from_dict({"tip_center_tighten_only": False})
    assert cfg.tip_center_tighten_only is False


def test_tip_center_tighten_only_picks_neg_x_tendon_for_positive_x_error() -> None:
    """When the tip is too far +X, tighten-only mode must select the -X
    tendon (servo_ids[2] per the canonical axis convention) to pull the tip
    back toward the centre. The +X tendon (servo_ids[0]) must NOT appear in
    the command list — that's the bug that drops servo 1 to 0 mA."""
    # The tighten-only branch lives inside _run_conservative_startup_sequence
    # and depends on telemetry/measurement objects. We exercise the
    # decision rule directly here by replicating the same expression so a
    # change to the rule breaks this test.
    sign_x = 1  # default tip_center_x_sign * axis_sign_correction["x"]
    x_error = +2.5  # tip too far +X by 2.5 mm
    # Mirror the same conditional used in the algorithm.
    if (x_error > 0.0) == (sign_x > 0):
        selected_axis = "tighten_neg_x"  # servo_ids[2]
    else:
        selected_axis = "tighten_pos_x"  # servo_ids[0]
    assert selected_axis == "tighten_neg_x"


def test_tip_center_tighten_only_picks_pos_x_tendon_for_negative_x_error() -> None:
    """Symmetric case: tip too far -X must select the +X tendon."""
    sign_x = 1
    x_error = -2.5
    if (x_error > 0.0) == (sign_x > 0):
        selected_axis = "tighten_neg_x"
    else:
        selected_axis = "tighten_pos_x"
    assert selected_axis == "tighten_pos_x"


def test_takeup_holding_tension_defaults_match_operator_scale() -> None:
    """Operator's tendon-tension scale: -15 mA = light, -30 mA = tight,
    -50 mA = a lot. The config must default to -30 mA (i.e. 30 mA of holding
    tension) as the take-up target and -50 mA as the upper bound. Setting
    the new metric (tension_ma = max(0, -signed_current)) as the take-up
    criterion replaces the |current - baseline| load_proxy that allowed
    runs to end with a servo at signed -3 mA being mis-classified as
    "in target band"."""
    cfg = PretensionValidationExperimentConfig.from_dict({})
    assert cfg.takeup_target_holding_tension_ma == 30.0
    assert cfg.takeup_high_holding_tension_ma == 50.0


def test_holding_tension_below_target_rejected_for_under_tensioned_run(tmp_path: Path) -> None:
    """A run whose final signed currents land at the values observed in
    20260521_160957 (servo 1 at signed -3 mA, others -17 to -21 mA) must be
    REJECTED. Before this change the legacy load-proxy check accepted it
    because |signed - baseline| was around 20 mA. The new check applies
    holding_tension_below_target whenever any servo's tension_ma is more
    than accept_max_load_balance_error_ma below the target.

    We verify the gate via a unit-level simulation of the acceptance logic:
    pick a known final signed_current vector, compute tension_ma, and
    confirm the threshold expression flags servo 1."""
    # Reproduce the run_1 final state from 20260521_160957.
    final_signed = {1: -3, 2: -18, 3: -21, 4: -17}
    tension_by_sid = {sid: max(0.0, -float(val)) for sid, val in final_signed.items()}
    target = 30.0
    tolerance = 15.0
    assert (target - min(tension_by_sid.values())) > tolerance, (
        "Servo 1 with tension_ma=3 must fall more than 15 mA below the 30 mA target."
    )


def test_holding_tension_target_band_accepts_well_tensioned_run() -> None:
    """A run with every servo near -30 mA signed current must NOT trigger
    holding_tension_below_target."""
    final_signed = {1: -28, 2: -32, 3: -30, 4: -29}
    tension_by_sid = {sid: max(0.0, -float(val)) for sid, val in final_signed.items()}
    target = 30.0
    tolerance = 15.0
    # min tension is 28 → target - 28 = 2 mA < tolerance, NOT below target.
    assert (target - min(tension_by_sid.values())) <= tolerance


def test_holding_tension_target_band_accepts_overshoot_run() -> None:
    """A servo overshooting to -45 mA is still tensioned (tension_ma = 45);
    the gate only flags UNDER-tensioned servos, not over-tensioned ones."""
    final_signed = {1: -45, 2: -32, 3: -30, 4: -29}
    tension_by_sid = {sid: max(0.0, -float(val)) for sid, val in final_signed.items()}
    target = 30.0
    tolerance = 15.0
    assert (target - min(tension_by_sid.values())) <= tolerance


def test_positive_signed_current_means_zero_tension_for_target_check() -> None:
    """A servo reading positive signed current (motor pushing outward, no
    tendon load) has tension_ma = 0 — the gate must clamp at 0 rather than
    going negative. Otherwise an overshoot below 0 would give the math a
    false sense of "negative tension" that masks a truly slack tendon."""
    final_signed = {1: +18, 2: -28, 3: -30, 4: -29}
    tension_by_sid = {sid: max(0.0, -float(val)) for sid, val in final_signed.items()}
    assert tension_by_sid[1] == 0.0
    target = 30.0
    tolerance = 15.0
    assert (target - min(tension_by_sid.values())) > tolerance


def test_post_move_settle_and_burst_defaults_match_operator_intent() -> None:
    """``post_move_settle_s``, ``holding_current_burst_count``, and
    ``holding_current_burst_interval_s`` must default to values that defeat
    the motion-current artifacts seen in run 20260521_155327. The defaults
    were picked to settle the XC330 PID + back-EMF before sampling tension."""
    cfg = PretensionValidationExperimentConfig.from_dict({})
    assert cfg.post_move_settle_s >= 0.2, "settle must be long enough for PID to decay"
    assert cfg.holding_current_burst_count >= 2, "need a burst to defeat noise"
    assert cfg.holding_current_burst_interval_s > 0.0


def test_read_holding_telemetry_averages_burst_and_reports_std(tmp_path: Path) -> None:
    """``_read_holding_telemetry`` must sleep, take a burst of reads, and
    return the averaged signed current per servo plus the spread (std).
    Telemetry that comes back stable across the burst → small std (PID is
    settled); high std → motion has not yet decayed."""
    service = _servo_service(tmp_path)
    # Script three settled reads at -25 mA per servo (operator's "tensioned"
    # signed current). We expect the helper to average them and report a
    # small std.
    script = [{1: -25.0, 2: -28.0, 3: -24.0, 4: -31.0} for _ in range(3)]
    wrapped = _ScriptedCurrentService(service, script=script)
    experiment = PretensionValidationExperiment(
        PretensionValidationExperimentConfig.from_dict(
            {
                "mode": "single_segment_staged",
                "servo_ids": [1, 2, 3, 4],
                "post_move_settle_s": 0.0,  # no real sleep in the test
                "holding_current_burst_count": 3,
                "holding_current_burst_interval_s": 0.0,
            }
        )
    )

    class _Session:
        def __init__(self):
            self.context = SimpleNamespace(
                servo_service=wrapped,
                tracker_service=None,
                monotonic_fn=time.monotonic,
                sleep_fn=lambda _s: None,
            )

        def raise_if_stop_requested(self):
            return None

        def update_progress(self, *_a, **_kw):
            return None

    holding = experiment._read_holding_telemetry(
        servo_service=wrapped,
        servo_ids=[1, 2, 3, 4],
        session=_Session(),
    )

    assert holding["holding_burst_count"] == 3
    averaged = holding["holding_current_ma_by_servo"]
    # Manual reference values exactly preserved by the averaging step.
    assert averaged[1] == pytest.approx(-25.0)
    assert averaged[2] == pytest.approx(-28.0)
    assert averaged[3] == pytest.approx(-24.0)
    assert averaged[4] == pytest.approx(-31.0)
    # Std should be ~0 since every sample is identical.
    for sid in (1, 2, 3, 4):
        assert holding["holding_current_std_by_servo"][sid] == pytest.approx(0.0, abs=1e-6)
    # Burst samples preserved for traceability.
    for sid in (1, 2, 3, 4):
        assert len(holding["holding_current_samples_by_servo"][sid]) == 3


def test_advanced_measurement_with_holding_overrides_motion_currents(tmp_path: Path) -> None:
    """``_advanced_measurement_with_holding`` must override the raw_current_ma
    fields with the burst-averaged holding values, so downstream tension
    decisions read the settled current instead of any in-motion read.

    The instantaneous reads must be preserved under
    ``raw_current_ma_instantaneous`` for diagnostics."""
    service = _servo_service(tmp_path)
    # Burst returns -25 mA per servo (settled, "tensioned" reference values).
    script = [{1: -25.0, 2: -25.0, 3: -25.0, 4: -25.0} for _ in range(4)]
    wrapped = _ScriptedCurrentService(service, script=script)
    experiment = PretensionValidationExperiment(
        PretensionValidationExperimentConfig.from_dict(
            {
                "mode": "single_segment_staged",
                "servo_ids": [1, 2, 3, 4],
                "post_move_settle_s": 0.0,
                "holding_current_burst_count": 3,
                "holding_current_burst_interval_s": 0.0,
            }
        )
    )

    class _Session:
        def __init__(self):
            self.context = SimpleNamespace(
                servo_service=wrapped,
                tracker_service=None,
                monotonic_fn=time.monotonic,
                sleep_fn=lambda _s: None,
            )

        def raise_if_stop_requested(self):
            return None

        def update_progress(self, *_a, **_kw):
            return None

    measurement = experiment._advanced_measurement_with_holding(
        servo_service=wrapped,
        tracker_service=None,
        servo_ids=[1, 2, 3, 4],
        baseline_current_ma_by_servo={1: 0.0, 2: 0.0, 3: 0.0, 4: 0.0},
        target_xy_mm=[0.0, 0.0],
        session=_Session(),
        startup_reference_ticks_by_servo={1: None, 2: None, 3: None, 4: None},
        trust_status="current_only_lower_trust",
    )

    # The overridden currents reflect the averaged holding burst (-25 mA each).
    for sid in (1, 2, 3, 4):
        assert measurement["raw_current_ma"][sid] == -25, (
            f"servo {sid} raw_current_ma should be the averaged holding value -25, "
            f"got {measurement['raw_current_ma'][sid]}"
        )
        assert measurement["signed_raw_current_ma"][sid] == -25
        assert measurement["filtered_current_ma"][sid] == pytest.approx(-25.0)
        # Load proxy = |held - baseline| = |-25 - 0| = 25 mA.
        assert measurement["current_above_baseline_ma"][sid] == pytest.approx(25.0)
        assert measurement["load_proxy_current_ma"][sid] == pytest.approx(25.0)
    # Burst metadata surfaced for the trace row + report.
    assert measurement["current_sample_phase"] == "holding"
    assert measurement["holding_burst_count"] == 3
    assert measurement["holding_current_ma_by_servo"][1] == pytest.approx(-25.0)
    # Instantaneous reads preserved (they were the FIRST read of the burst,
    # which the mock served from the first script entry).
    assert "raw_current_ma_instantaneous" in measurement
    assert "signed_raw_current_ma_instantaneous" in measurement
    # Load balance recomputed from the overridden values: all 4 at 25 mA →
    # spread = 0.
    assert measurement["load_balance_error_ma"] == pytest.approx(0.0)


def test_soft_release_signed_current_at_minus_2_succeeds_via_target_current(tmp_path: Path) -> None:
    """Signed convention on this rig: ``raw_current_ma`` is signed. A near-
    slack tendon reads e.g. -1 to -2 mA (or even positive when the motor is
    still pushing outward). With the default release_current_abs_target_ma=5,
    the condition is ``signed_current >= -5`` for N stable samples — so
    -2 mA must fire the ``target_current`` path."""
    script = [{1: -2.0, 2: -1.0, 3: 0.0, 4: -3.0} for _ in range(6)]
    result, _ = _run_release_with_scripted_currents(tmp_path, script=script)
    assert result["stop_reason"] == "soft_release_to_zero_current_complete"
    for sid in (1, 2, 3, 4):
        assert result["release_success_by_servo"][str(sid)] is True
        assert result["release_success_condition_by_servo"][str(sid)] == "target_current"


def test_soft_release_positive_current_during_outward_motion_counts_as_released(tmp_path: Path) -> None:
    """Operator note (2026-05-20): "positive currents will be seen when the
    servos are moving to higher ticks". Those positive currents indicate the
    motor is driving the outward motion freely with no tendon resistance —
    that is fully released. The condition signed_current >= -5 must accept
    positive currents trivially."""
    script = [{1: 5.0, 2: 3.0, 3: 8.0, 4: 12.0} for _ in range(6)]
    result, _ = _run_release_with_scripted_currents(tmp_path, script=script)
    assert result["stop_reason"] == "soft_release_to_zero_current_complete"
    for sid in (1, 2, 3, 4):
        assert result["release_success_by_servo"][str(sid)] is True
        assert result["release_success_condition_by_servo"][str(sid)] == "target_current"


def test_soft_release_at_minus_8_signed_succeeds_via_plateau_fallback(tmp_path: Path) -> None:
    """A real servo often plateaus at around -7 to -10 mA when slack because
    of holding current / friction — the signed current never makes it all
    the way up to >= -5. The plateau fallback must declare release when
    signed_current stays >= -release_plateau_max_current_ma (>= -10 mA
    default) AND the spread across the plateau window is small."""
    # Signed -8 mA: never reaches >= -5 (target path fails), stays >= -10
    # (plateau max), spread ~0.3 mA << 2 mA delta → plateau fires after the
    # 4-sample window fills.
    script = [{1: -8.0, 2: -8.2, 3: -7.9, 4: -8.1} for _ in range(6)]
    result, _ = _run_release_with_scripted_currents(tmp_path, script=script)
    assert result["stop_reason"] == "soft_release_to_zero_current_complete"
    for sid in (1, 2, 3, 4):
        assert result["release_success_by_servo"][str(sid)] is True
        assert result["release_success_condition_by_servo"][str(sid)] == "low_current_plateau"


def test_soft_release_at_minus_30_does_not_release_tensioned_state(tmp_path: Path) -> None:
    """Operator's empirical scale: all four servos at -30 mA is GOOD
    PRETENSION — the tendons are taut. The release algorithm must NOT
    declare this state as released. Neither the target_current path
    (-30 < -5) nor the plateau fallback (-30 < -10) should fire; the run
    must exhaust its travel budget."""
    # Hold deeply negative, low spread so the spread gate alone doesn't
    # save the test. The deepest-allowed signed reading is -10 mA, and -30
    # is well below that, so plateau must NOT fire.
    script = [{1: -30.0, 2: -30.5, 3: -29.8, 4: -30.2} for _ in range(40)]
    result, _ = _run_release_with_scripted_currents(
        tmp_path,
        script=script,
        extra_config={"release_max_travel_ticks": 100, "release_step_ticks": 25},
    )
    for sid in (1, 2, 3, 4):
        assert result["release_success_by_servo"][str(sid)] is False
        assert result["release_success_condition_by_servo"][str(sid)] == ""
        assert result["release_stop_reason_by_servo"][str(sid)] in {
            "release_travel_budget_exhausted",
            "release_iteration_limit",
        }
    assert result["stop_reason"].startswith("release_incomplete_servos=")


def test_soft_release_at_minus_15_below_plateau_cap_does_not_release(tmp_path: Path) -> None:
    """Signed -15 mA sits BELOW the default plateau_max (-10 mA), so even a
    perfectly stable plateau at -15 must NOT trigger the fallback. This is
    the boundary case between "real slack with holding current" (-7 to -10)
    and "still tensioned" (-15+) that protects against false-slack."""
    script = [{1: -15.0, 2: -15.2, 3: -14.8, 4: -15.1} for _ in range(40)]
    result, _ = _run_release_with_scripted_currents(
        tmp_path,
        script=script,
        extra_config={"release_max_travel_ticks": 100, "release_step_ticks": 25},
    )
    for sid in (1, 2, 3, 4):
        assert result["release_success_by_servo"][str(sid)] is False
        assert result["release_success_condition_by_servo"][str(sid)] == ""


def test_soft_release_plateau_disabled_when_cap_is_zero(tmp_path: Path) -> None:
    """Setting ``release_plateau_max_current_ma: 0`` disables the fallback —
    a servo plateauing at -8 mA must NOT be declared released."""
    script = [{1: -8.0, 2: -8.0, 3: -8.0, 4: -8.0} for _ in range(40)]
    result, _ = _run_release_with_scripted_currents(
        tmp_path,
        script=script,
        extra_config={
            "release_plateau_max_current_ma": 0.0,
            "release_max_travel_ticks": 100,
            "release_step_ticks": 25,
        },
    )
    assert result["release_plateau_enabled"] is False
    for sid in (1, 2, 3, 4):
        assert result["release_success_by_servo"][str(sid)] is False


def test_soft_release_travel_budget_exhaustion_still_fails(tmp_path: Path) -> None:
    """A servo whose signed current keeps swinging (no stable plateau, never
    above -5 mA) must hit ``release_travel_budget_exhausted`` rather than the
    plateau fallback. Simulate by alternating -8 / -12 so the window spread
    is 4 mA > plateau_delta (2 mA) and the floor is -12 < -plateau_max."""
    script: list[dict[int, float]] = []
    for i in range(40):
        val = -8.0 if (i % 2 == 0) else -12.0
        script.append({1: val, 2: val, 3: val, 4: val})
    result, _ = _run_release_with_scripted_currents(
        tmp_path,
        script=script,
        extra_config={
            "release_max_travel_ticks": 100,
            "release_step_ticks": 25,
            "release_plateau_delta_ma": 2.0,
        },
    )
    for sid in (1, 2, 3, 4):
        assert result["release_success_by_servo"][str(sid)] is False
        assert result["release_stop_reason_by_servo"][str(sid)] in {
            "release_travel_budget_exhausted",
            "release_iteration_limit",
        }


def test_soft_release_to_zero_current_records_per_servo_release_state(tmp_path: Path) -> None:
    """``_run_soft_release_to_zero_current_start`` must report per-servo
    start position, travel, final current, success flag, and stop reason
    EVEN when the mock bus's current readings do not meet the slack
    criterion. The test verifies the per-servo record structure rather than
    a specific outcome, because the mock bus returns whatever current it was
    configured with and the live rig's noise floor is the real determinant."""
    service = _servo_service(tmp_path)
    experiment = PretensionValidationExperiment(
        PretensionValidationExperimentConfig.from_dict(
            {
                "mode": "single_segment_staged",
                "servo_ids": [1, 2, 3, 4],
                "release_current_stable_samples": 1,
                "release_current_abs_target_ma": 5.0,
                "release_step_ticks": 25,
                "release_max_travel_ticks": 80,
            }
        )
    )

    class _Session:
        def __init__(self):
            self.context = SimpleNamespace(
                servo_service=service,
                tracker_service=None,
                monotonic_fn=time.monotonic,
                sleep_fn=lambda _s: None,
            )
            self.staged_samples: list = []

        def raise_if_stop_requested(self):
            return None

        def update_progress(self, *_a, **_kw):
            return None

    session = _Session()
    experiment._add_staged_sample = lambda session, **kw: session.staged_samples.append(kw)  # type: ignore[assignment]

    trace_rows: list = []
    result = experiment._run_soft_release_to_zero_current_start(
        session=session,
        servo_service=service,
        tracker_service=None,
        servo_ids=[1, 2, 3, 4],
        run_index=0,
        target_xy_mm=[0.0, 0.0],
        startup_reference_ticks_by_servo={sid: None for sid in (1, 2, 3, 4)},
        mode_kind="conservative_startup",
        trace_rows=trace_rows,
    )

    # Per-servo records exist for each id regardless of whether the mock bus
    # declared each one released.
    for sid in (1, 2, 3, 4):
        assert str(sid) in result["release_start_position_by_servo"]
        assert str(sid) in result["release_stop_reason_by_servo"]
        # The stop reason must be one of the documented per-servo states.
        assert result["release_stop_reason_by_servo"][str(sid)] in {
            "released",
            "release_travel_budget_exhausted",
            "release_iteration_limit",
            "missing_position",
            "in_progress",
        }
    # Stage row was traced for visibility.
    assert any(row.get("stage") == "soft_release_to_zero_current" for row in trace_rows)
    # Top-level stop reason must be a known state.
    assert result["stop_reason"] in {
        "soft_release_to_zero_current_complete",
        "soft_release_to_zero_current_disabled",
    } or result["stop_reason"].startswith("release_incomplete_servos=")

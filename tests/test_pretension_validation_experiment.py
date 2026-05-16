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
    assert run_row["stop_reason"] in {"tip_diverging", "tip_response_wrong_direction"}
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
    # Repeatable starting condition.
    assert payload["pretension_start_mode"] == "full_release_4095"
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
    # Repeatable start condition.
    assert payload["pretension_start_mode"] == "full_release_4095"
    # Default variant is the conservative one, not Jacobian.
    assert payload["tip_centering_variant"] == "paired_then_tip"

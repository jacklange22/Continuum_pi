from __future__ import annotations

from pathlib import Path

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


def _runner(tmp_path: Path, servo_service: ServoService) -> ExperimentRunner:
    settings = _settings(tmp_path)
    return ExperimentRunner(
        project_root=tmp_path,
        settings=settings,
        tracking_service=None,
        servo_service=servo_service,
        output_dir=tmp_path / "data" / "experiments",
        registration_path=tmp_path / "latest_registration.json",
        sleep_fn=lambda _seconds: None,
    )


def _advanced_config() -> dict:
    return {
        "mode": "single_segment_staged",
        "servo_ids": [1, 2, 3, 4],
        "repeat_runs": 1,
        "include_tracker_displacement": False,
        "allow_current_only_when_tracker_missing": True,
        "enable_tip_centering": False,
        "move_to_reference": True,
        "pretension_start_mode": "release_200_from_current",
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
            "accept_max_final_tip_xy_offset_mm": 3.0,
            "accept_max_load_balance_error_ma": 6.0,
            "accept_max_pair_balance_error_ma": 6.0,
        }
    )

    assert config.tip_center_tolerance_mm == 2.5
    assert config.load_balance_tolerance_ma == 5.5
    assert config.pair_balance_tolerance_ma == 4.5
    assert config.accept_max_final_tip_xy_offset_mm == 3.0
    assert config.accept_max_load_balance_error_ma == 6.0
    assert config.accept_max_pair_balance_error_ma == 6.0


def test_advanced_pretension_staged_run_saves_quality_pairing_and_plots(tmp_path: Path) -> None:
    service = _servo_service(tmp_path)
    runner = _runner(tmp_path, service)

    result = runner.run_experiment("pretension_validation", config=_advanced_config())

    metrics = result.summary.experiment_metrics
    run_row = metrics["run_rows"][0]
    stages = [row.get("stage") for row in metrics["trace_rows"]]
    pair_labels = {
        str(row.get("pair_label"))
        for row in metrics["trace_rows"]
        if row.get("stage") == "paired_takeup"
    }

    assert result.success is True
    assert metrics["algorithm"] == "advanced_4servo_pretension"
    assert run_row["quality_score_0_100"] >= 0.0
    assert run_row["quality_score_0_100"] <= 100.0
    assert "tip_centering" in run_row["quality_components"]
    assert "custom_start" in stages
    assert "paired_takeup" in stages
    assert pair_labels == {"1", "2"}
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
    ]
    for filename in expected_plots:
        assert (result.paths.output_dir / filename).exists()
    assert "quality_score_0_100" in (result.paths.output_dir / "metrics.csv").read_text(encoding="utf-8").splitlines()[0]


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

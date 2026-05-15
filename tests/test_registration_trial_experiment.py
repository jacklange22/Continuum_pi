from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import yaml

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
from continuum_robot.experiments.framework import (
    ExperimentContext,
    ExperimentSession,
)
from continuum_robot.experiments.registration_trial import (
    CAPTURES_SESSION_KEY,
    RegistrationTrialConfig,
    RegistrationTrialExperiment,
    load_truth_landmarks,
)
from continuum_robot.experiments.schemas import ExperimentMetadata


def _settings() -> Settings:
    return Settings(
        runtime=RuntimeConfig(mock_mode=True, poll_rate_hz=20, robot_config="robot_4servo.yaml"),
        robot=RobotConfig(mode="4-servo", spool_diameter_cm=1.2, ticks_per_revolution=4096, servo_ids=[1, 2, 3, 4]),
        serial=SerialConfig(aurora_port="/dev/mock-aurora", openrb_port="/dev/mock-openrb", baudrate=115200),
        safety=SafetyConfig(
            position_min_offset_ticks=-600,
            position_max_offset_ticks=600,
            max_current_ma=850,
            pretension_current_balance_tolerance_ma=120,
        ),
        registration=RegistrationWorkflowConfig(capture_tool_id="0B", coil_tool_id="0A", max_fre_mm=None),
        experiment=ExperimentConfig(default_settle_time_s=0.0, sample_count_per_point=1, output_dir="data/experiments"),
        calibration=CalibrationConfig(
            neutral_setpoints_path="config/neutral_setpoints.json",
            latest_registration_path="data/registrations/latest_registration.json",
        ),
    )


def _runner(project_root: Path) -> ExperimentRunner:
    return ExperimentRunner(
        project_root=project_root,
        settings=_settings(),
        tracking_service=None,
        servo_service=None,
        output_dir=project_root / "data" / "experiments",
        registration_path=project_root / "data" / "registrations" / "latest_registration.json",
        sleep_fn=lambda _seconds: None,
    )


def _write_truth_yaml(project_root: Path, *, non_coplanar: bool = True) -> Path:
    config_dir = project_root / "config"
    config_dir.mkdir(parents=True, exist_ok=True)
    truth_path = config_dir / "registration.yaml"
    if non_coplanar:
        candidates = [
            {"id": "L1", "xyz_mm": [0.0, 35.0, -5.0]},
            {"id": "L2", "xyz_mm": [-35.0, 0.0, -5.0]},
            {"id": "L3", "xyz_mm": [0.0, -35.0, -5.0]},
            {"id": "L4", "xyz_mm": [35.0, 0.0, -5.0]},
            {"id": "L5", "xyz_mm": [0.0, 25.0, 10.0]},
            {"id": "L6", "xyz_mm": [-25.0, 0.0, 10.0]},
            {"id": "L7", "xyz_mm": [0.0, -25.0, 10.0]},
            {"id": "L8", "xyz_mm": [25.0, 0.0, 10.0]},
        ]
    else:
        candidates = [
            {"id": "L1", "xyz_mm": [0.0, 35.0, -5.0]},
            {"id": "L2", "xyz_mm": [-35.0, 0.0, -5.0]},
            {"id": "L3", "xyz_mm": [0.0, -35.0, -5.0]},
            {"id": "L4", "xyz_mm": [35.0, 0.0, -5.0]},
        ]
    truth_path.write_text(yaml.safe_dump({"candidate_landmarks": candidates}), encoding="utf-8")
    return truth_path


def _synth_captures(
    truth: dict[str, list[float]],
    *,
    captures_per_label: int,
    noise_scale: float = 0.05,
    seed: int = 0,
    bias_label: str | None = None,
    bias_vector: tuple[float, float, float] = (1.5, 0.0, 0.0),
) -> dict[str, list[list[float]]]:
    rng = np.random.default_rng(seed)
    T_robot_aurora = np.eye(4, dtype=float)
    T_robot_aurora[0:3, 3] = [10.0, 5.0, 2.0]
    T_aurora_robot = np.linalg.inv(T_robot_aurora)
    captures: dict[str, list[list[float]]] = {}
    for label, xyz in truth.items():
        truth_h = np.append(np.asarray(xyz, dtype=float), 1.0)
        center_aurora = (T_aurora_robot @ truth_h)[:3]
        noise = rng.normal(scale=noise_scale, size=(captures_per_label, 3))
        if bias_label == label:
            noise = noise + np.asarray(bias_vector)
        captures[label] = (center_aurora + noise).tolist()
    return captures


def _write_raw_captures_record(path: Path, captures: dict[str, list[list[float]]]) -> Path:
    path.write_text(
        json.dumps({"raw_captured_landmarks_aurora_xyz": captures}), encoding="utf-8"
    )
    return path


def test_load_truth_landmarks_reads_yaml(tmp_path: Path) -> None:
    truth_path = _write_truth_yaml(tmp_path)
    truth = load_truth_landmarks(truth_path)
    assert set(truth.keys()) == {f"L{i}" for i in range(1, 9)}


def test_config_from_dict_picks_up_defaults() -> None:
    cfg = RegistrationTrialConfig.from_dict({})
    assert cfg.captures_per_landmark == 50
    assert cfg.subset_sizes == [4, 5, 6, 7, 8]
    assert cfg.averaging_methods == ["mean", "median", "trimmed_mean", "mad_filtered_mean"]


def test_config_from_dict_honors_overrides() -> None:
    cfg = RegistrationTrialConfig.from_dict(
        {
            "captures_per_landmark": 20,
            "landmark_labels": ["L1", "L2", "L5"],
            "subset_sizes": [3, 4, 5],
            "averaging_methods": ["mean", "median"],
        }
    )
    assert cfg.captures_per_landmark == 20
    assert cfg.landmark_labels == ["L1", "L2", "L5"]
    assert cfg.subset_sizes == [3, 4, 5]


def test_experiment_replay_runs_via_runner_and_writes_report(tmp_path: Path) -> None:
    truth_path = _write_truth_yaml(tmp_path)
    truth = load_truth_landmarks(truth_path)
    captures = _synth_captures(truth, captures_per_label=25, seed=1)
    record_path = _write_raw_captures_record(
        tmp_path / "synth_captures.json", captures
    )
    runner = _runner(tmp_path)
    result = runner.run_experiment(
        "registration_trial",
        config={
            "source_record_path": str(record_path.relative_to(tmp_path)),
            "registration_yaml_path": str(truth_path.relative_to(tmp_path)),
            "subset_sizes": [4, 5, 6],
            "captures_per_landmark": 25,
        },
    )
    assert result.success is True
    out = result.paths.output_dir
    assert out.name.endswith("_registration_trial")
    assert (out / "metadata.json").exists()
    assert (out / "summary.json").exists()
    assert (out / "trial_report.md").exists()
    assert (out / "trial_report.json").exists()
    payload = json.loads((out / "trial_report.json").read_text(encoding="utf-8"))
    assert set(payload["method_sweep"].keys()) == {
        "mean",
        "median",
        "trimmed_mean",
        "mad_filtered_mean",
    }
    assert payload["subset_search_count"] > 0
    md = (out / "trial_report.md").read_text(encoding="utf-8")
    assert "## Method comparison" in md
    assert "## Subset search" in md


def _build_session(project_root: Path, *, config_payload: dict | None = None) -> ExperimentSession:
    """Hand-roll a minimal session so execute() can be unit-tested directly."""
    metadata = ExperimentMetadata(
        schema_version="1.0",
        experiment_name="registration_trial",
        run_id="testrun",
        timestamp_utc="2026-05-15T00:00:00+00:00",
        git_commit=None,
        backend_info={},
        registration_info={},
        config_used=dict(config_payload or {}),
    )
    context = ExperimentContext(
        project_root=project_root,
        settings=_settings(),
        tracking_service=None,
        servo_service=None,
        registration_path=project_root / "data" / "registrations" / "latest_registration.json",
        output_root=project_root / "data" / "experiments",
        run_output_dir=project_root / "data" / "experiments" / "registration_trial" / "testrun",
        monotonic_fn=lambda: 0.0,
        sleep_fn=lambda _seconds: None,
    )
    context.run_output_dir.mkdir(parents=True, exist_ok=True)
    return ExperimentSession(context=context, metadata=metadata)


def test_experiment_live_mode_uses_session_metrics_captures(tmp_path: Path) -> None:
    """When source_record_path is empty, captures come from session.metrics."""
    truth_path = _write_truth_yaml(tmp_path)
    truth = load_truth_landmarks(truth_path)
    captures = _synth_captures(truth, captures_per_label=15, seed=2, bias_label="L3")
    config = {
        "registration_yaml_path": str(truth_path.relative_to(tmp_path)),
        "subset_sizes": [4, 5],
    }
    experiment = RegistrationTrialExperiment.from_dict(config)
    session = _build_session(tmp_path, config_payload=config)
    session.metrics[CAPTURES_SESSION_KEY] = captures

    experiment.precheck(session)
    experiment.execute(session)
    # The biased L3 should be excluded from the best subset since it makes FRE worse.
    subset_summary = session.metrics["subset_search_summary"]
    assert subset_summary["global_best"] is not None
    assert "L3" not in subset_summary["global_best"]["labels"]


def test_experiment_errors_when_no_captures_supplied(tmp_path: Path) -> None:
    truth_path = _write_truth_yaml(tmp_path)
    runner = _runner(tmp_path)
    result = runner.run_experiment(
        "registration_trial",
        config={
            "registration_yaml_path": str(truth_path.relative_to(tmp_path)),
            "source_record_path": "",
        },
    )
    assert result.success is False
    failed_stages = [stage for stage, status in result.summary.stage_pass_fail.items() if status == "failed"]
    assert "execute" in failed_stages


def test_experiment_errors_when_source_record_missing(tmp_path: Path) -> None:
    truth_path = _write_truth_yaml(tmp_path)
    runner = _runner(tmp_path)
    result = runner.run_experiment(
        "registration_trial",
        config={
            "registration_yaml_path": str(truth_path.relative_to(tmp_path)),
            "source_record_path": "nonexistent.json",
        },
    )
    assert result.success is False
    failed_stages = [stage for stage, status in result.summary.stage_pass_fail.items() if status == "failed"]
    assert "precheck" in failed_stages


def test_experiment_recommendations_include_coplanar_warning(tmp_path: Path) -> None:
    truth_path = _write_truth_yaml(tmp_path, non_coplanar=False)
    truth = load_truth_landmarks(truth_path)
    captures = _synth_captures(truth, captures_per_label=10, seed=3)
    record_path = _write_raw_captures_record(
        tmp_path / "coplanar_captures.json", captures
    )
    runner = _runner(tmp_path)
    result = runner.run_experiment(
        "registration_trial",
        config={
            "source_record_path": str(record_path.relative_to(tmp_path)),
            "registration_yaml_path": str(truth_path.relative_to(tmp_path)),
            "subset_sizes": [4],
        },
    )
    md = (result.paths.output_dir / "trial_report.md").read_text(encoding="utf-8")
    assert "rank-deficient" in md or "rank=2" in md


def test_experiment_registered_in_builtins() -> None:
    from continuum_robot.experiments.registry import ExperimentRegistry
    from continuum_robot.experiments.builtins import register_builtin_experiments

    registry = ExperimentRegistry()
    register_builtin_experiments(registry)
    names = [descriptor.name for descriptor in registry.list_descriptors()]
    assert "registration_trial" in names
    descriptor = registry.get("registration_trial")
    assert descriptor.title == "Registration Trial"
    assert "Registration" in descriptor.tags

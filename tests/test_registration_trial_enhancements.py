"""Tests for the registration_trial enhancements added 2026-05-15.

These tests cover:

1. ``samples_per_point_study``: synthetic data, FRE drops as k grows on
   noisy data; works at k=K; capped correctly at the captured pool size.
2. ``recommend_samples_per_point``: picks the smallest k within epsilon
   of the best FRE in the ladder.
3. The experiment surfaces ``samples_per_point_summary`` and
   ``samples_per_point_recommendation`` in session metrics, and adds a
   recommendation line that mentions samples per landmark.
4. ``write_registration_trial_outputs`` produces the 4 PNG plots when the
   summary has the new fields.
5. ``promote_registration_trial`` happy-path + error cases (missing
   report, bad subset, dry-run).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from continuum_robot.registration.trial_analysis import (
    aggregate_samples_per_point,
    recommend_samples_per_point,
    samples_per_point_study,
)


def _truth_landmarks() -> dict[str, list[float]]:
    return {
        "L1": [0.0, 35.0, -5.0],
        "L2": [-35.0, 0.0, -5.0],
        "L3": [0.0, -35.0, -5.0],
        "L4": [35.0, 0.0, -5.0],
        "L5": [0.0, 25.0, -5.0],
    }


def _known_T_aurora_robot() -> np.ndarray:
    theta = 0.1
    c, s = float(np.cos(theta)), float(np.sin(theta))
    R = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=float)
    t = np.asarray([10.0, -5.0, 200.0], dtype=float)
    T = np.eye(4)
    T[0:3, 0:3] = R
    T[0:3, 3] = t
    return T


def _noisy_captures(truth: dict[str, list[float]], *, captures_per_label: int, noise_std: float, seed: int) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    T = _known_T_aurora_robot()
    out: dict[str, np.ndarray] = {}
    for label, xyz_robot in truth.items():
        aurora = (T[0:3, 0:3] @ np.asarray(xyz_robot, dtype=float)) + T[0:3, 3]
        out[label] = aurora + rng.normal(scale=noise_std, size=(captures_per_label, 3))
    return out


# ---------------------------------------------------------------------------
# samples_per_point_study math
# ---------------------------------------------------------------------------


def test_samples_per_point_study_returns_rows_per_k_iteration() -> None:
    truth = _truth_landmarks()
    captures = _noisy_captures(truth, captures_per_label=50, noise_std=0.4, seed=11)
    rows = samples_per_point_study(
        captures,
        truth,
        k_values=[1, 5, 20],
        bootstrap_iterations=8,
        random_seed=3,
    )
    ks = {int(row["k"]) for row in rows}
    assert ks == {1, 5, 20}
    counts = {k: sum(1 for row in rows if int(row["k"]) == k) for k in ks}
    for k, n in counts.items():
        assert n == 8, f"k={k} produced {n} iterations, expected 8"


def test_samples_per_point_study_caps_k_at_pool_size() -> None:
    truth = _truth_landmarks()
    captures = _noisy_captures(truth, captures_per_label=10, noise_std=0.4, seed=12)
    rows = samples_per_point_study(
        captures,
        truth,
        k_values=[1, 5, 10, 50, 100],
        bootstrap_iterations=4,
        random_seed=4,
    )
    ks_seen = sorted({int(row["k"]) for row in rows})
    # min pool is 10 so k=50 and k=100 must be capped — and once we reach k=10 we
    # stop (every further k would return the same draw).
    assert max(ks_seen) == 10
    assert 1 in ks_seen and 5 in ks_seen and 10 in ks_seen


def test_more_samples_per_point_lowers_mean_fre_on_noisy_data() -> None:
    truth = _truth_landmarks()
    captures = _noisy_captures(truth, captures_per_label=40, noise_std=0.5, seed=21)
    rows = samples_per_point_study(
        captures,
        truth,
        k_values=[1, 5, 20],
        bootstrap_iterations=30,
        random_seed=5,
    )
    summary = aggregate_samples_per_point(rows)
    by_k = {row["k"]: row["fre_mean_mm"] for row in summary}
    assert by_k[1] > by_k[5] > by_k[20]


def test_recommend_samples_per_point_returns_smallest_within_epsilon() -> None:
    summary = [
        {"k": 1, "fre_mean_mm": 0.80},
        {"k": 5, "fre_mean_mm": 0.42},
        {"k": 10, "fre_mean_mm": 0.40},
        {"k": 20, "fre_mean_mm": 0.40},
    ]
    rec = recommend_samples_per_point(summary, epsilon_mm=0.05)
    # k=5 lands within 0.05 mm of the best (0.40 mm).
    assert rec["recommended_k"] == 5
    assert rec["best_fre_mean_mm"] == pytest.approx(0.40)


def test_recommend_samples_per_point_empty_input_explains_itself() -> None:
    rec = recommend_samples_per_point([])
    assert rec["recommended_k"] is None
    assert "Insufficient" in rec["rationale"]


# ---------------------------------------------------------------------------
# Experiment integration
# ---------------------------------------------------------------------------


def test_registration_trial_experiment_surfaces_samples_per_point_in_metrics(tmp_path: Path) -> None:
    """End-to-end: run the experiment in replay mode and confirm the new fields land."""
    from continuum_robot.experiments.registration_trial import RegistrationTrialExperiment
    from continuum_robot.experiments.framework import ExperimentContext, ExperimentSession
    from continuum_robot.experiments.schemas import ExperimentMetadata
    from continuum_robot.config.settings import (
        CalibrationConfig, ExperimentConfig, RegistrationWorkflowConfig,
        RobotConfig, RuntimeConfig, SafetyConfig, SerialConfig, Settings,
    )

    # Write a registration.yaml truth file the experiment can load.
    truth = _truth_landmarks()
    cfg_dir = tmp_path / "config"
    cfg_dir.mkdir(parents=True, exist_ok=True)
    cfg_path = cfg_dir / "registration.yaml"
    lines = ["candidate_landmarks:"]
    for label, xyz in truth.items():
        lines.append(f"  - id: {label}")
        lines.append(f"    xyz_mm: {xyz}")
        lines.append("    enabled: true")
    cfg_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    # Build a small but useful capture pool.
    captures = _noisy_captures(truth, captures_per_label=20, noise_std=0.3, seed=33)
    captures_payload = {label: pool.tolist() for label, pool in captures.items()}

    experiment = RegistrationTrialExperiment.from_dict(
        {
            "registration_yaml_path": "config/registration.yaml",
            "captures_per_landmark": 20,
            "samples_per_point_ladder": [1, 5, 20],
            "samples_per_point_bootstrap_iterations": 8,
            "samples_per_point_epsilon_mm": 0.05,
        }
    )
    settings = Settings(
        runtime=RuntimeConfig(mock_mode=True, robot_config="robot_8servo.yaml"),
        robot=RobotConfig(mode="single_segment", servo_ids=[1, 2, 3, 4], tendon_to_servo=[1, 2, 3, 4]),
        serial=SerialConfig(aurora_port="/dev/mock-aurora", openrb_port="/dev/mock-openrb", baudrate=57600),
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
            neutral_setpoints_path="config/neutral_setpoints.json",
            latest_registration_path="data/registrations/latest_registration.json",
        ),
    )
    context = ExperimentContext(
        project_root=tmp_path,
        settings=settings,
        tracking_service=None,
        servo_service=None,
        registration_path=tmp_path / "registration.json",
        output_root=tmp_path,
        run_output_dir=tmp_path,
        sleep_fn=lambda _: None,
    )
    metadata = ExperimentMetadata(
        schema_version="1.0",
        experiment_name=experiment.name,
        run_id="test_run",
        timestamp_utc="2026-05-15T00:00:00Z",
        git_commit=None,
        backend_info={},
        registration_info={},
        config_used={},
    )
    session = ExperimentSession(context=context, metadata=metadata)
    # Inject captures via the canonical live-mode key.
    from continuum_robot.experiments.registration_trial import CAPTURES_SESSION_KEY
    session.metrics[CAPTURES_SESSION_KEY] = captures_payload
    experiment.precheck(session)
    experiment.execute(session)

    assert "samples_per_point_summary" in session.metrics
    assert "samples_per_point_recommendation" in session.metrics
    spp_summary = session.metrics["samples_per_point_summary"]
    ks = sorted({row["k"] for row in spp_summary})
    assert ks == [1, 5, 20]
    recommendation = session.metrics["samples_per_point_recommendation"]
    assert recommendation["recommended_k"] in {1, 5, 20}
    # Recommendation must show up in the human-facing trial_recommendations list.
    recs = session.metrics["trial_recommendations"]
    assert any("Samples per landmark" in line for line in recs)


# ---------------------------------------------------------------------------
# Plot writer smoke test (best-effort)
# ---------------------------------------------------------------------------


def test_write_registration_trial_outputs_attempts_plots(tmp_path: Path) -> None:
    """write_registration_trial_outputs must emit md + json. Plot generation
    is best-effort and depends on matplotlib availability; we only assert the
    core text outputs survive."""
    from continuum_robot.experiments.registration_trial_outputs import (
        REPORT_FIGURES,
        write_registration_trial_outputs,
    )
    from continuum_robot.experiments.schemas import ExperimentMetadata, ExperimentSummary

    metadata = ExperimentMetadata(
        schema_version="1.0",
        experiment_name="registration_trial",
        run_id="test_run",
        timestamp_utc="2026-05-15T00:00:00Z",
        git_commit=None,
        backend_info={},
        registration_info={},
        config_used={},
    )
    metrics = {
        "method_summary": {"method_rows": [{"method": "mean", "fre_mm": 0.42, "loo_max_minus_keep_mm": 0.05}], "best_method": "mean", "best_fre_mm": 0.42},
        "subset_search_summary": {
            "per_size_best": {
                4: {"size": 4, "subset_count": 5, "best_fre_mm": 0.5, "best_max_residual_mm": 0.6, "best_labels": ["L1", "L2", "L3", "L4"], "best_geometry_rank": 3, "best_geometry_condition_number": 1.5},
                5: {"size": 5, "subset_count": 1, "best_fre_mm": 0.45, "best_max_residual_mm": 0.55, "best_labels": ["L1", "L2", "L3", "L4", "L5"], "best_geometry_rank": 3, "best_geometry_condition_number": 1.4},
            },
            "global_best": {"size": 5, "labels": ["L1", "L2", "L3", "L4", "L5"], "fre_mm": 0.45},
        },
        "samples_per_point_summary": [
            {"k": 1, "iteration_count": 8, "fre_mean_mm": 0.9, "fre_std_mm": 0.1, "fre_min_mm": 0.7, "fre_max_mm": 1.1, "fre_p95_mm": 1.05},
            {"k": 5, "iteration_count": 8, "fre_mean_mm": 0.5, "fre_std_mm": 0.05, "fre_min_mm": 0.42, "fre_max_mm": 0.58, "fre_p95_mm": 0.55},
        ],
        "samples_per_point_recommendation": {"recommended_k": 5, "recommended_fre_mean_mm": 0.5, "best_fre_mean_mm": 0.5, "epsilon_mm": 0.05},
        "landmark_labels_captured": ["L1", "L2", "L3", "L4", "L5"],
        "captures_per_landmark_target": 20,
        "trial_recommendations": ["Best averaging method on this dataset: mean (FRE=0.4200 mm)."],
        "truth_by_label": {label: [0.0, 0.0, 0.0] for label in ["L1", "L2", "L3", "L4", "L5"]},
        "raw_captures_by_label": {label: [[0.0, 0.0, 0.0], [0.1, 0.1, 0.0]] for label in ["L1", "L2", "L3", "L4", "L5"]},
    }
    summary = ExperimentSummary(
        schema_version="1.0",
        experiment_name="registration_trial",
        run_id="test_run",
        success=True,
        sample_counts={"total": 0},
        dropped_frames=0,
        invalid_transforms=0,
        stage_pass_fail={},
        experiment_metrics=metrics,
    )
    out_dir = tmp_path / "trial_out"
    write_registration_trial_outputs(output_dir=out_dir, metadata=metadata, summary=summary)
    assert (out_dir / "trial_report.md").exists()
    payload = json.loads((out_dir / "trial_report.json").read_text(encoding="utf-8"))
    assert payload["schema_version"] == "1.1"
    assert payload["samples_per_point_summary"]
    assert payload["samples_per_point_recommendation"]["recommended_k"] == 5
    # Confirm REPORT_FIGURES advertisement matches what we tried to emit.
    assert set(payload["report_figures"]) == set(REPORT_FIGURES)


# ---------------------------------------------------------------------------
# Promote tool
# ---------------------------------------------------------------------------


def _write_trial_report(
    run_dir: Path,
    *,
    chosen_labels: list[str] | None = None,
    valid: bool = True,
) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    truth = _truth_landmarks()
    chosen = chosen_labels or sorted(truth.keys())
    # Make raw captures by applying the known T inverse and adding tiny noise.
    rng = np.random.default_rng(101)
    T = _known_T_aurora_robot()
    raw: dict[str, list[list[float]]] = {}
    for label in chosen:
        truth_xyz = np.asarray(truth[label], dtype=float)
        aurora = (T[0:3, 0:3] @ truth_xyz) + T[0:3, 3]
        pool = aurora + rng.normal(scale=0.02, size=(10, 3))
        raw[label] = pool.tolist()
    payload = {
        "schema_version": "1.1" if valid else "0.0",
        "raw_captures_by_label": raw,
        "truth_by_label": {label: truth[label] for label in chosen},
        "subset_search_summary": {
            "global_best": {"size": len(chosen), "labels": chosen, "fre_mm": 0.05},
        },
    }
    (run_dir / "trial_report.json").write_text(json.dumps(payload), encoding="utf-8")


def test_promote_registration_trial_happy_path(tmp_path: Path) -> None:
    from continuum_robot.data.promote_registration_trial import promote_registration_trial

    run_dir = tmp_path / "trial_run"
    _write_trial_report(run_dir)
    registrations_root = tmp_path / "registrations"

    report = promote_registration_trial(
        run_dir=run_dir,
        registrations_root=registrations_root,
        operator_note="test promote",
    )
    assert report["dry_run"] is False
    assert Path(report["active_path"]).exists()
    payload = json.loads((registrations_root / "latest_registration.json").read_text(encoding="utf-8"))
    assert payload["schema_version"] == "registration_promoted_from_trial_v1"
    assert payload["operator_note"] == "test promote"
    assert payload["fre_mm"] == pytest.approx(report["fre_mm"])
    # Legacy key preserved.
    assert "raw_captured_landmarks_robot_xyz" in payload


def test_promote_registration_trial_backs_up_previous(tmp_path: Path) -> None:
    from continuum_robot.data.promote_registration_trial import promote_registration_trial

    run_dir = tmp_path / "trial_run"
    _write_trial_report(run_dir)
    registrations_root = tmp_path / "registrations"
    registrations_root.mkdir()
    (registrations_root / "latest_registration.json").write_text(
        json.dumps({"timestamp_utc": "OLD", "fre_mm": 9.9}), encoding="utf-8"
    )

    report = promote_registration_trial(run_dir=run_dir, registrations_root=registrations_root)
    backup = report["backup_path"]
    assert backup is not None and Path(backup).exists()
    backup_payload = json.loads(Path(backup).read_text(encoding="utf-8"))
    assert backup_payload["timestamp_utc"] == "OLD"


def test_promote_registration_trial_dry_run_writes_nothing(tmp_path: Path) -> None:
    from continuum_robot.data.promote_registration_trial import promote_registration_trial

    run_dir = tmp_path / "trial_run"
    _write_trial_report(run_dir)
    registrations_root = tmp_path / "registrations"
    registrations_root.mkdir()
    (registrations_root / "latest_registration.json").write_text(
        json.dumps({"timestamp_utc": "OLD", "fre_mm": 9.9}), encoding="utf-8"
    )

    report = promote_registration_trial(run_dir=run_dir, registrations_root=registrations_root, dry_run=True)
    assert report["dry_run"] is True
    active = json.loads((registrations_root / "latest_registration.json").read_text(encoding="utf-8"))
    assert active["timestamp_utc"] == "OLD"


def test_promote_registration_trial_explicit_subset(tmp_path: Path) -> None:
    from continuum_robot.data.promote_registration_trial import promote_registration_trial

    run_dir = tmp_path / "trial_run"
    _write_trial_report(run_dir)
    registrations_root = tmp_path / "registrations"

    report = promote_registration_trial(
        run_dir=run_dir,
        registrations_root=registrations_root,
        subset_labels=["L1", "L2", "L3", "L4"],
    )
    assert report["chosen_labels"] == ["L1", "L2", "L3", "L4"]


def test_promote_registration_trial_refuses_missing_report(tmp_path: Path) -> None:
    from continuum_robot.data.promote_registration_trial import promote_registration_trial

    run_dir = tmp_path / "trial_run"
    run_dir.mkdir()
    with pytest.raises(FileNotFoundError):
        promote_registration_trial(run_dir=run_dir, registrations_root=tmp_path / "registrations")


def test_promote_registration_trial_refuses_subset_with_unknown_label(tmp_path: Path) -> None:
    from continuum_robot.data.promote_registration_trial import promote_registration_trial

    run_dir = tmp_path / "trial_run"
    _write_trial_report(run_dir)
    with pytest.raises(ValueError, match="no captures"):
        promote_registration_trial(
            run_dir=run_dir,
            registrations_root=tmp_path / "registrations",
            subset_labels=["L1", "L2", "L99"],
        )

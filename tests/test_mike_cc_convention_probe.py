"""Tests for the Mike constant-curvature convention probe.

The probe is evidence-only: it never auto-flips the convention flag. These
tests pin the math + reporting behavior so the CLI cannot regress silently.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pytest

from continuum_robot.modeling.two_segment.physics import (
    MikeCCConventionReport,
    two_segment_constant_curvature_prediction,
    validate_mike_cc_conventions,
)
from continuum_robot.modeling.two_segment.validate_mike_cc import main as probe_main


def _config_with_design_geometry() -> dict:
    """Return a minimal Mike CC config sufficient for a prediction."""
    return {
        "physics_models": {
            "global": {
                "tendon_displacement_sign_convention": {
                    "value": "positive_shortens_tendon",
                    "source": "design",
                },
            },
            "segments": {
                "segment_a": {
                    "segment_length_mm": {"value": 66.0, "units": "mm", "source": "design"},
                    "tendon_positions_mm": {
                        "value": [[4.12, 0.0], [0.0, 4.12], [-4.12, 0.0], [0.0, -4.12]],
                        "units": "mm",
                        "source": "design",
                    },
                },
                "segment_b": {
                    "segment_length_mm": {"value": 66.0, "units": "mm", "source": "design"},
                    "tendon_positions_mm": {
                        "value": [[4.12, 0.0], [0.0, 4.12], [-4.12, 0.0], [0.0, -4.12]],
                        "units": "mm",
                        "source": "design",
                    },
                },
            },
        }
    }


@dataclass
class _StubSample:
    """Minimal sample stub matching the duck-type used by the probe."""

    feature_mm: np.ndarray
    distal_position_mm: np.ndarray | None
    intermediate_position_mm: np.ndarray | None = None
    sample_index: int = 0


def _truthful_samples_from_config(config: dict, feature_vectors: list[list[float]]) -> list[_StubSample]:
    """Synthesize measurements that exactly match Mike CC predictions.

    This is the "best-case" path — gives the probe samples where the conventions
    are already correct so it should return ``conventions_consistent_with_evidence``.
    """

    samples: list[_StubSample] = []
    for index, feature in enumerate(feature_vectors):
        prediction = two_segment_constant_curvature_prediction(np.asarray(feature, dtype=float), config)
        samples.append(
            _StubSample(
                feature_mm=np.asarray(feature, dtype=float),
                distal_position_mm=np.asarray(prediction["distal_xyz"], dtype=float),
                intermediate_position_mm=np.asarray(prediction["intermediate_xyz"], dtype=float),
                sample_index=int(index),
            )
        )
    return samples


def test_validate_mike_cc_conventions_reports_config_incomplete_when_predictions_fail() -> None:
    """An empty (no physics_models) config causes every prediction to raise.

    The probe must surface this as ``config_incomplete_prediction_failed`` so
    the operator immediately sees that the issue is YAML, not hardware.
    """
    # Build a couple of samples with measured distal data so the "no measured distal"
    # case doesn't trigger; the prediction call itself should KeyError.
    samples = [
        _StubSample(
            feature_mm=np.asarray([0.5, 0.0, -0.5, 0.0, 0.0, 0.0, 0.0, 0.0]),
            distal_position_mm=np.asarray([1.0, 2.0, 30.0]),
            sample_index=0,
        ),
        _StubSample(
            feature_mm=np.asarray([0.0, 0.5, 0.0, -0.5, 0.0, 0.0, 0.0, 0.0]),
            distal_position_mm=np.asarray([2.0, 1.0, 31.0]),
            sample_index=1,
        ),
    ]

    report = validate_mike_cc_conventions(samples=samples, config={})

    assert report.recommendation == "config_incomplete_prediction_failed"
    assert any("modeling config is missing required fields" in note for note in report.notes)
    assert any("segment_length_mm" in note for note in report.notes)
    # Each sample row should have a prediction_error explaining what was missing.
    assert all("prediction_error" in row for row in report.residuals_by_sample)


def test_validate_mike_cc_conventions_returns_no_samples_provided_when_empty() -> None:
    config = _config_with_design_geometry()

    report = validate_mike_cc_conventions(samples=[], config=config)

    assert report.recommendation == "no_samples_provided"
    assert report.sample_count == 0


def test_validate_mike_cc_conventions_recommends_safe_to_confirm_when_perfect_match() -> None:
    config = _config_with_design_geometry()
    samples = _truthful_samples_from_config(
        config,
        [
            [0.5, 0.0, -0.5, 0.0, 0.0, 0.0, 0.0, 0.0],  # bottom-only +X
            [0.0, 0.5, 0.0, -0.5, 0.0, 0.0, 0.0, 0.0],  # bottom-only +Y
            [0.0, 0.0, 0.0, 0.0, 0.5, 0.0, -0.5, 0.0],  # top-only +X
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.5, 0.0, -0.5],  # top-only +Y
        ],
    )

    report = validate_mike_cc_conventions(samples=samples, config=config)

    assert report.sample_count == 4
    assert report.recommendation == "conventions_consistent_with_evidence_safe_to_confirm"
    # Distal residual should be zero (synthesized samples == predictions).
    assert report.distal_residual_norm_mean_mm == pytest.approx(0.0, abs=1e-9)
    assert report.distal_residual_norm_max_mm == pytest.approx(0.0, abs=1e-9)
    assert all(report.per_axis_distal_signs_match.values())
    assert report.largest_distal_sign_disagreement_axis is None


def test_validate_mike_cc_conventions_flags_sign_disagreement_when_x_is_flipped() -> None:
    config = _config_with_design_geometry()
    samples = _truthful_samples_from_config(
        config,
        [
            [0.5, 0.0, -0.5, 0.0, 0.0, 0.0, 0.0, 0.0],
            [-0.5, 0.0, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.3, 0.0, -0.3, 0.0, 0.0, 0.0, 0.0, 0.0],
        ],
    )
    # Flip measured distal X sign on every sample to simulate a frame/sign mistake.
    for sample in samples:
        flipped = sample.distal_position_mm.copy()
        flipped[0] = -flipped[0]
        sample.distal_position_mm = flipped

    report = validate_mike_cc_conventions(samples=samples, config=config)

    assert report.per_axis_distal_signs_match["x"] is False
    assert report.largest_distal_sign_disagreement_axis == "x"
    assert report.recommendation == "sign_convention_likely_flipped_axis_x"


def test_validate_mike_cc_conventions_flags_intermediate_sign_flip_when_distal_signs_are_ok() -> None:
    """Distal looks correct but the intermediate has a flipped axis — operator must see this."""
    config = _config_with_design_geometry()
    samples = _truthful_samples_from_config(
        config,
        [
            [0.5, 0.0, -0.5, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.5, 0.0, -0.5, 0.0, 0.0, 0.0, 0.0],
            [-0.5, 0.0, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0],
        ],
    )
    for sample in samples:
        # Flip the intermediate X sign on every sample; leave distal alone.
        flipped = sample.intermediate_position_mm.copy()
        flipped[0] = -flipped[0]
        sample.intermediate_position_mm = flipped

    report = validate_mike_cc_conventions(samples=samples, config=config)

    assert all(report.per_axis_distal_signs_match.values()), "Distal signs should still match (we only flipped intermediate)"
    assert report.per_axis_intermediate_signs_match["x"] is False
    assert report.largest_intermediate_sign_disagreement_axis == "x"
    assert report.recommendation == "intermediate_sign_convention_likely_flipped_axis_x"
    assert any("Intermediate predicted sign disagrees" in note for note in report.notes)


def test_validate_mike_cc_conventions_emits_magnitude_recommendation_when_signs_ok_but_residuals_high() -> None:
    config = _config_with_design_geometry()
    samples = _truthful_samples_from_config(
        config,
        [
            [0.5, 0.0, -0.5, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.5, 0.0, -0.5, 0.0, 0.0, 0.0, 0.0],
        ],
    )
    # Offset every measured distal Z by 25 mm (signs unchanged) → magnitude error.
    for sample in samples:
        offset = sample.distal_position_mm.copy()
        offset[2] += 25.0
        sample.distal_position_mm = offset

    report = validate_mike_cc_conventions(samples=samples, config=config)

    assert all(report.per_axis_distal_signs_match.values())
    assert report.recommendation == "magnitude_off_but_signs_ok_inspect_geometry"
    assert report.distal_residual_norm_max_mm > 15.0


def test_validate_mike_cc_cli_returns_zero_on_safe_to_confirm_and_writes_report(tmp_path: Path) -> None:
    """End-to-end CLI smoke test using fixture samples and synthetic config."""

    # Build a tiny synthetic dataset run on disk so the loader sees a real layout.
    config = _config_with_design_geometry()
    feature_vectors = [
        [0.5, 0.0, -0.5, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.5, 0.0, -0.5, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.5, 0.0, -0.5, 0.0],
    ]
    run_dir = tmp_path / "data" / "experiments" / "two_segment_collect_pose_command_dataset" / "run"
    run_dir.mkdir(parents=True)
    metadata = {
        "experiment_name": "two_segment_collect_pose_command_dataset",
        "run_id": run_dir.name,
        "trust_info": {"run_trust_mode": "thesis_trusted"},
        "provenance_info": {"operating_mode": "dual_segment"},
    }
    summary = {
        "experiment_name": "two_segment_collect_pose_command_dataset",
        "run_id": run_dir.name,
        "success": True,
        "status": "success",
        "experiment_metrics": {
            "dataset_type": "two_segment_collect_pose_command_dataset",
            "startup_artifact_provenance": {"accepted_all_8_startup": True},
        },
    }
    (run_dir / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
    (run_dir / "summary.json").write_text(json.dumps(summary), encoding="utf-8")
    with (run_dir / "samples.jsonl").open("w", encoding="utf-8") as handle:
        for index, feature in enumerate(feature_vectors):
            prediction = two_segment_constant_curvature_prediction(np.asarray(feature, dtype=float), config)
            distal = prediction["distal_xyz"].tolist()
            matrix = [
                [1.0, 0.0, 0.0, float(distal[0])],
                [0.0, 1.0, 0.0, float(distal[1])],
                [0.0, 0.0, 1.0, float(distal[2])],
                [0.0, 0.0, 0.0, 1.0],
            ]
            payload = {
                "wall_time_utc": f"2026-05-15T12:00:{index:02d}+00:00",
                "phase": "probe",
                "step_index": index,
                "sample_index": index,
                "two_segment_command": {
                    "schema_version": "two_segment_command_v1",
                    "units": "cm",
                    "segments": {
                        "segment_a": [float(value) / 10.0 for value in feature[:4]],
                        "segment_b": [float(value) / 10.0 for value in feature[4:]],
                    },
                    "commanded_servo_ids": [1, 2, 3, 4, 5, 6, 7, 8],
                    "flat_command_cm": [float(value) / 10.0 for value in feature],
                },
                "pose_in_robot_frame": {"roles": {"distal_tip": {"T_robot_tip": matrix}}},
                "two_segment_pose": {"frame": "robot", "distal_tip_pose": {"T_robot_tip": matrix}},
                "extra": {
                    "record_kind": "two_segment_dataset_capture",
                    "run_trust_mode": "thesis_trusted",
                    "capture_accepted": True,
                    "command_success": True,
                    "valid_for_two_segment_model_training": True,
                    "command_units": "cm",
                    "commanded_servo_ids": [1, 2, 3, 4, 5, 6, 7, 8],
                    "startup_artifact_provenance": {"accepted_all_8_startup": True},
                    "available_pose_roles": ["distal_tip"],
                    "missing_required_pose_roles": [],
                    "distal_only": True,
                    "includes_intermediate_pose": False,
                    "segment_order": ["segment_a", "segment_b"],
                    "measured_servo_feedback": {
                        str(servo_id): {"servo_id": int(servo_id), "position_tick": 2048 + int(servo_id) + index}
                        for servo_id in range(1, 9)
                    },
                },
            }
            handle.write(json.dumps(payload) + "\n")

    config_path = tmp_path / "modeling_two_segment.yaml"
    import yaml as _yaml

    config_path.write_text(_yaml.safe_dump(config), encoding="utf-8")
    out_dir = tmp_path / "probe_output"

    rc = probe_main(
        [
            "--runs",
            str(run_dir),
            "--config",
            str(config_path),
            "--output-dir",
            str(out_dir),
        ]
    )

    assert rc == 0
    report_json = json.loads((out_dir / "mike_cc_convention_report.json").read_text(encoding="utf-8"))
    assert report_json["recommendation"] == "conventions_consistent_with_evidence_safe_to_confirm"
    text = (out_dir / "mike_cc_convention_report.txt").read_text(encoding="utf-8")
    assert "recommendation: conventions_consistent_with_evidence_safe_to_confirm" in text
    assert "evidence-only" in text


def test_validate_mike_cc_cli_emits_friendly_error_when_runs_directory_missing(tmp_path: Path, capsys) -> None:
    """First-time operator typing the wrong --runs path should get a clear hint."""
    out_dir = tmp_path / "probe_output"
    nonexistent_run = tmp_path / "nope" / "this-run-does-not-exist"

    rc = probe_main(["--runs", str(nonexistent_run), "--output-dir", str(out_dir)])

    assert rc == 2
    captured = capsys.readouterr()
    assert "missing or do not contain samples.jsonl" in captured.err
    assert "Hint:" in captured.err


def test_validate_mike_cc_cli_emits_friendly_error_when_latest_finds_nothing(tmp_path: Path, capsys) -> None:
    """`--latest` with no two-segment runs should be friendly, not stack-trace-y."""
    out_dir = tmp_path / "probe_output"

    rc = probe_main(["--latest", "--project-root", str(tmp_path), "--output-dir", str(out_dir)])

    assert rc == 2
    captured = capsys.readouterr()
    assert "No runs found" in captured.err
    assert "Hint:" in captured.err


def test_validate_mike_cc_cli_emits_friendly_error_when_config_file_missing(tmp_path: Path, capsys) -> None:
    """Operator typo on --config should be friendly."""
    out_dir = tmp_path / "probe_output"
    missing_config = tmp_path / "missing-config.yaml"
    # We still need at least one valid run dir to skip past the runs-missing check;
    # the config check comes BEFORE dataset loading, so a bare-bones dir works.
    fake_run = tmp_path / "fake_run"
    fake_run.mkdir()
    (fake_run / "samples.jsonl").write_text("", encoding="utf-8")

    rc = probe_main(["--runs", str(fake_run), "--config", str(missing_config), "--output-dir", str(out_dir)])

    assert rc == 2
    captured = capsys.readouterr()
    assert "config file not found" in captured.err


def test_validate_mike_cc_cli_returns_nonzero_when_no_samples(tmp_path: Path) -> None:
    """Empty dataset run → probe should refuse to recommend confirm."""

    run_dir = tmp_path / "data" / "experiments" / "two_segment_collect_pose_command_dataset" / "empty"
    run_dir.mkdir(parents=True)
    (run_dir / "metadata.json").write_text("{}", encoding="utf-8")
    (run_dir / "summary.json").write_text("{}", encoding="utf-8")
    (run_dir / "samples.jsonl").write_text("", encoding="utf-8")
    out_dir = tmp_path / "probe_output"

    rc = probe_main(["--runs", str(run_dir), "--output-dir", str(out_dir)])

    assert rc == 2
    report_json = json.loads((out_dir / "mike_cc_convention_report.json").read_text(encoding="utf-8"))
    assert report_json["recommendation"] in {"no_samples_provided", "no_predictions_generated"}

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import yaml

from continuum_robot.registration.trial_cli import (
    load_captures_from_registration_record,
    load_truth_points,
    main,
    render_markdown,
    run_trial_for_record,
)


def _write_yaml(path: Path, payload: dict) -> None:
    path.write_text(yaml.safe_dump(payload), encoding="utf-8")


def _make_truth_yaml(tmp_path: Path) -> Path:
    path = tmp_path / "registration.yaml"
    _write_yaml(
        path,
        {
            "candidate_landmarks": [
                {"id": "L1", "xyz_mm": [0.0, 35.0, -5.0]},
                {"id": "L2", "xyz_mm": [-35.0, 0.0, -5.0]},
                {"id": "L3", "xyz_mm": [0.0, -35.0, -5.0]},
                {"id": "L4", "xyz_mm": [35.0, 0.0, -5.0]},
            ]
        },
    )
    return path


def _make_record(tmp_path: Path, *, captures_per_label: int, noise_scale: float, seed: int = 0) -> Path:
    """Synthesize a saved-registration JSON with raw captures we can re-solve."""
    truth = {
        "L1": np.array([0.0, 35.0, -5.0]),
        "L2": np.array([-35.0, 0.0, -5.0]),
        "L3": np.array([0.0, -35.0, -5.0]),
        "L4": np.array([35.0, 0.0, -5.0]),
    }
    T_robot_aurora = np.eye(4, dtype=float)
    T_robot_aurora[0:3, 3] = [10.0, 5.0, 2.0]
    T_aurora_robot = np.linalg.inv(T_robot_aurora)
    rng = np.random.default_rng(seed)
    raw = {}
    for label, xyz in truth.items():
        center = (T_aurora_robot @ np.append(xyz, 1.0))[:3]
        noise = rng.normal(scale=noise_scale, size=(captures_per_label, 3))
        raw[label] = (center + noise).tolist()
    record_path = tmp_path / "synth_registration.json"
    record_path.write_text(json.dumps({"raw_captured_landmarks_aurora_xyz": raw}), encoding="utf-8")
    return record_path


def test_load_truth_points_reads_candidate_landmarks(tmp_path: Path) -> None:
    truth = load_truth_points(_make_truth_yaml(tmp_path))
    assert truth == {
        "L1": [0.0, 35.0, -5.0],
        "L2": [-35.0, 0.0, -5.0],
        "L3": [0.0, -35.0, -5.0],
        "L4": [35.0, 0.0, -5.0],
    }


def test_load_truth_points_errors_when_empty(tmp_path: Path) -> None:
    path = tmp_path / "empty.yaml"
    _write_yaml(path, {})
    with pytest.raises(RuntimeError):
        load_truth_points(path)


def test_load_truth_points_uses_nominal_block_if_present(tmp_path: Path) -> None:
    path = tmp_path / "registration.yaml"
    _write_yaml(
        path,
        {
            "nominal_landmarks_robot_xyz_mm": {
                "L1": [0.0, 0.0, 0.0],
                "L2": [10.0, 0.0, 0.0],
            }
        },
    )
    truth = load_truth_points(path)
    assert truth["L1"] == [0.0, 0.0, 0.0]
    assert truth["L2"] == [10.0, 0.0, 0.0]


def test_load_captures_extracts_raw_per_label(tmp_path: Path) -> None:
    record = _make_record(tmp_path, captures_per_label=8, noise_scale=0.05)
    captures = load_captures_from_registration_record(record)
    assert set(captures.keys()) == {"L1", "L2", "L3", "L4"}
    assert all(len(points) == 8 for points in captures.values())


def test_load_captures_errors_when_no_raw_block(tmp_path: Path) -> None:
    path = tmp_path / "missing_raw.json"
    path.write_text(json.dumps({"fre_mm": 0.5}), encoding="utf-8")
    with pytest.raises(RuntimeError):
        load_captures_from_registration_record(path)


def test_run_trial_for_record_produces_all_methods(tmp_path: Path) -> None:
    record = _make_record(tmp_path, captures_per_label=20, noise_scale=0.05, seed=1)
    truth = load_truth_points(_make_truth_yaml(tmp_path))
    results, summary = run_trial_for_record(record, truth)
    assert set(results.keys()) == {"mean", "median", "trimmed_mean", "mad_filtered_mean"}
    assert summary["captures_min"] == 20
    assert summary["captures_max"] == 20
    assert float(summary["best_fre_mm"]) < 0.1  # 0.05mm noise, 20 samples → tiny FRE


def test_run_trial_errors_when_truth_missing_for_label(tmp_path: Path) -> None:
    record = _make_record(tmp_path, captures_per_label=8, noise_scale=0.05)
    truth = {"L1": [0.0, 35.0, -5.0]}  # missing L2-L4
    with pytest.raises(RuntimeError):
        run_trial_for_record(record, truth)


def test_render_markdown_includes_recommendation_for_coplanar_truth(tmp_path: Path) -> None:
    record = _make_record(tmp_path, captures_per_label=10, noise_scale=0.05)
    truth = load_truth_points(_make_truth_yaml(tmp_path))
    results, summary = run_trial_for_record(record, truth)
    md = render_markdown(record, truth, results, summary)
    assert "Truth landmarks are coplanar" in md
    assert "## Recommendations" in md
    assert "rank=2" in md or "rank = 2" in md


def test_render_markdown_picks_up_outlier_landmark(tmp_path: Path) -> None:
    """If one landmark is biased, LOO should flag it as worth recapturing."""
    record_path = _make_record(tmp_path, captures_per_label=15, noise_scale=0.05, seed=42)
    # Bias L3 to make it the bottleneck.
    payload = json.loads(record_path.read_text())
    raw = payload["raw_captured_landmarks_aurora_xyz"]
    biased = (np.asarray(raw["L3"]) + np.array([1.5, 0.0, 0.0])).tolist()
    raw["L3"] = biased
    record_path.write_text(json.dumps(payload), encoding="utf-8")
    truth = load_truth_points(_make_truth_yaml(tmp_path))
    results, summary = run_trial_for_record(record_path, truth)
    md = render_markdown(record_path, truth, results, summary)
    assert "**L3**" in md  # L3 should be the recommended re-capture target


def test_main_cli_writes_outputs(tmp_path: Path) -> None:
    record = _make_record(tmp_path, captures_per_label=15, noise_scale=0.05, seed=7)
    yaml_path = _make_truth_yaml(tmp_path)
    output_dir = tmp_path / "out"
    exit_code = main(
        [
            str(record),
            "--registration-config",
            str(yaml_path),
            "--output-dir",
            str(output_dir),
        ]
    )
    assert exit_code == 0
    assert (output_dir / "trial_report.md").exists()
    assert (output_dir / "trial_report.json").exists()
    payload = json.loads((output_dir / "trial_report.json").read_text())
    assert set(payload["results_by_method"].keys()) == {
        "mean",
        "median",
        "trimmed_mean",
        "mad_filtered_mean",
    }

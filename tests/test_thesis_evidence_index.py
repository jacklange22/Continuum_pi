from __future__ import annotations

import json
from pathlib import Path

from continuum_robot.data.build_thesis_evidence_index import build_thesis_evidence_index


def _write_run(root: Path, name: str, *, review_status: str, trust_mode: str = "thesis_trusted", mock: bool = False) -> Path:
    run_dir = root / "data" / "experiments" / "single_segment_repeatability" / name
    run_dir.mkdir(parents=True)
    provenance = {"operating_mode": "single_segment", "mock_mode": bool(mock)}
    trust = {
        "run_trust_mode": trust_mode,
        "valid_for_model_training": False,
        "valid_for_thesis_repeatability": trust_mode == "thesis_trusted" and not mock,
    }
    (run_dir / "metadata.json").write_text(
        json.dumps(
            {
                "experiment_name": "single_segment_repeatability",
                "run_id": name,
                "provenance_info": provenance,
                "trust_info": trust,
            }
        ),
        encoding="utf-8",
    )
    (run_dir / "summary.json").write_text(
        json.dumps(
            {
                "experiment_name": "single_segment_repeatability",
                "run_id": name,
                "success": True,
                "status": "success",
                "experiment_metrics": {
                    "run_provenance": provenance,
                    "run_trust": trust,
                    "run_trust_mode": trust_mode,
                    "valid_for_model_training": trust["valid_for_model_training"],
                    "valid_for_thesis_repeatability": trust["valid_for_thesis_repeatability"],
                    "mock_mode": bool(mock),
                },
            }
        ),
        encoding="utf-8",
    )
    (run_dir / "run_review.json").write_text(
        json.dumps({"review_status": review_status, "include_in_evidence_index": False}),
        encoding="utf-8",
    )
    return run_dir


def _included_run_ids(index_dir: Path) -> set[str]:
    payload = json.loads((index_dir / "thesis_evidence_index.json").read_text(encoding="utf-8"))
    runs = payload["experiments"].get("single_segment_repeatability", [])
    return {str(run["run_id"]) for run in runs}


def test_evidence_index_includes_thesis_and_advisor_reviews_only_by_default(tmp_path: Path) -> None:
    _write_run(tmp_path, "debug-run", review_status="debug")
    _write_run(tmp_path, "thesis-run", review_status="thesis_candidate")
    _write_run(tmp_path, "advisor-run", review_status="advisor_share")

    output_dir = build_thesis_evidence_index(project_root=tmp_path)

    assert _included_run_ids(output_dir) == {"thesis-run", "advisor-run"}


def test_evidence_index_excludes_mock_even_when_reviewed_by_default(tmp_path: Path) -> None:
    _write_run(tmp_path, "mock-run", review_status="thesis_candidate", trust_mode="mock", mock=True)

    output_dir = build_thesis_evidence_index(project_root=tmp_path)

    assert _included_run_ids(output_dir) == set()


def _write_two_segment_run(
    root: Path,
    *,
    experiment_name: str,
    name: str,
    metrics: dict,
    review_status: str = "thesis_candidate",
) -> Path:
    run_dir = root / "data" / "experiments" / experiment_name / name
    run_dir.mkdir(parents=True)
    trust = {
        "run_trust_mode": "thesis_trusted",
        "valid_for_model_training": False,
        "valid_for_thesis_repeatability": False,
        "valid_for_two_segment_model_training": True,
    }
    provenance = {"operating_mode": "dual_segment", "mock_mode": False}
    (run_dir / "metadata.json").write_text(
        json.dumps({"experiment_name": experiment_name, "run_id": name, "provenance_info": provenance, "trust_info": trust}),
        encoding="utf-8",
    )
    summary_metrics = dict(metrics)
    summary_metrics.setdefault("run_trust", trust)
    summary_metrics.setdefault("run_trust_mode", "thesis_trusted")
    summary_metrics.setdefault("mock_mode", False)
    summary_metrics.setdefault("valid_for_model_training", False)
    summary_metrics.setdefault("valid_for_thesis_repeatability", False)
    summary_metrics.setdefault("run_provenance", provenance)
    (run_dir / "summary.json").write_text(
        json.dumps({"experiment_name": experiment_name, "run_id": name, "success": True, "status": "success", "experiment_metrics": summary_metrics}),
        encoding="utf-8",
    )
    (run_dir / "run_review.json").write_text(
        json.dumps({"review_status": review_status, "include_in_evidence_index": False}),
        encoding="utf-8",
    )
    return run_dir


def test_evidence_index_surfaces_two_segment_collect_pose_metadata(tmp_path: Path) -> None:
    _write_two_segment_run(
        tmp_path,
        experiment_name="two_segment_collect_pose_command_dataset",
        name="20260515_120000_two_segment_collect_pose_command_dataset",
        metrics={
            "dataset_type": "two_segment_collect_pose_command_dataset",
            "schedule_type": "random_babble",
            "bottom_segment_key": "segment_b",
            "top_segment_key": "segment_a",
            "accepted_sample_count": 250,
            "rejected_sample_count": 4,
            "command_failure_count": 1,
            "long_run_health": {
                "stop_reason": "target_valid_sample_count_reached",
                "transport_failures": 1,
                "continue_until_valid_samples": True,
                "target_valid_sample_count": 250,
            },
            "valid_for_two_segment_model_training": True,
        },
    )

    output_dir = build_thesis_evidence_index(project_root=tmp_path)
    payload = json.loads((output_dir / "thesis_evidence_index.json").read_text(encoding="utf-8"))
    runs = payload["experiments"]["two_segment_collect_pose_command_dataset"]
    assert runs, "Expected at least one collect-pose run in the index"
    metrics = runs[0]["key_metrics"]
    assert "two_segment_collect_pose" in metrics
    tsc = metrics["two_segment_collect_pose"]
    assert tsc["bottom_segment_key"] == "segment_b"
    assert tsc["top_segment_key"] == "segment_a"
    assert tsc["stop_reason"] == "target_valid_sample_count_reached"
    assert tsc["transport_failures"] == 1
    assert tsc["continue_until_valid_samples"] is True


def test_evidence_index_surfaces_two_segment_repeatability_metadata(tmp_path: Path) -> None:
    _write_two_segment_run(
        tmp_path,
        experiment_name="two_segment_repeatability",
        name="20260515_120500_two_segment_repeatability",
        metrics={
            "dataset_type": "two_segment_repeatability",
            "bottom_segment_key": "segment_a",
            "top_segment_key": "segment_b",
            "target_count": 13,
            "repeat_visits": 3,
            "scatter_metrics": {"aggregate_distal_rms_mm": 2.4, "aggregate_intermediate_rms_mm": 1.7},
            "target_distal_rms_mm": 1.0,
        },
    )

    output_dir = build_thesis_evidence_index(project_root=tmp_path)
    payload = json.loads((output_dir / "thesis_evidence_index.json").read_text(encoding="utf-8"))
    runs = payload["experiments"]["two_segment_repeatability"]
    assert runs, "Expected at least one repeatability run in the index"
    metrics = runs[0]["key_metrics"]
    assert "two_segment_repeatability" in metrics
    tsr = metrics["two_segment_repeatability"]
    assert tsr["aggregate_distal_rms_mm"] == 2.4
    assert tsr["target_distal_rms_mm"] == 1.0


def test_evidence_index_surfaces_two_segment_startup_validation(tmp_path: Path) -> None:
    _write_two_segment_run(
        tmp_path,
        experiment_name="two_segment_startup_validation",
        name="20260515_120800_two_segment_startup_validation",
        metrics={
            "startup_type": "manual_two_segment_startup",
            "bottom_segment_key": "segment_a",
            "top_segment_key": "segment_b",
            "final_accepted": True,
            "tracker_available": False,
            "stage_order": ["baseline", "bottom_pretensioned", "top_pretensioned", "bottom_recheck", "final_accept"],
        },
    )

    output_dir = build_thesis_evidence_index(project_root=tmp_path)
    payload = json.loads((output_dir / "thesis_evidence_index.json").read_text(encoding="utf-8"))
    runs = payload["experiments"]["two_segment_startup_validation"]
    assert runs
    metrics = runs[0]["key_metrics"]
    assert "two_segment_startup" in metrics
    startup = metrics["two_segment_startup"]
    assert startup["final_accepted"] is True
    assert startup["bottom_segment_key"] == "segment_a"


def test_evidence_index_debug_and_mock_flags_are_explicit(tmp_path: Path) -> None:
    _write_run(tmp_path, "debug-run", review_status="debug", trust_mode="debug")
    _write_run(tmp_path, "mock-run", review_status="thesis_candidate", trust_mode="mock", mock=True)

    output_dir = build_thesis_evidence_index(
        project_root=tmp_path,
        include_debug=True,
        include_mock=True,
    )

    assert _included_run_ids(output_dir) == {"debug-run", "mock-run"}


def _write_run_with_thesis_validity(
    root: Path,
    name: str,
    *,
    review_status: str,
    valid_for_thesis_repeatability: bool,
    valid_for_model_training: bool = False,
    trust_mode: str = "thesis_trusted",
    experiment_name: str = "single_segment_repeatability",
) -> Path:
    run_dir = root / "data" / "experiments" / experiment_name / name
    run_dir.mkdir(parents=True)
    provenance = {"operating_mode": "single_segment", "mock_mode": False}
    trust = {
        "run_trust_mode": trust_mode,
        "valid_for_model_training": bool(valid_for_model_training),
        "valid_for_thesis_repeatability": bool(valid_for_thesis_repeatability),
    }
    (run_dir / "metadata.json").write_text(
        json.dumps(
            {
                "experiment_name": experiment_name,
                "run_id": name,
                "provenance_info": provenance,
                "trust_info": trust,
            }
        ),
        encoding="utf-8",
    )
    (run_dir / "summary.json").write_text(
        json.dumps(
            {
                "experiment_name": experiment_name,
                "run_id": name,
                "success": True,
                "status": "success",
                "experiment_metrics": {
                    "run_provenance": provenance,
                    "run_trust": trust,
                    "run_trust_mode": trust_mode,
                    "valid_for_model_training": trust["valid_for_model_training"],
                    "valid_for_thesis_repeatability": trust["valid_for_thesis_repeatability"],
                    "mock_mode": False,
                },
            }
        ),
        encoding="utf-8",
    )
    (run_dir / "run_review.json").write_text(
        json.dumps({"review_status": review_status, "include_in_evidence_index": False}),
        encoding="utf-8",
    )
    return run_dir


def _payload(index_dir: Path) -> dict:
    return json.loads((index_dir / "thesis_evidence_index.json").read_text(encoding="utf-8"))


def _run_entries(payload: dict, experiment: str) -> list[dict]:
    return list(payload.get("experiments", {}).get(experiment, []))


def test_thesis_candidate_with_failing_validity_is_included_as_lower_trust(tmp_path: Path) -> None:
    _write_run_with_thesis_validity(
        tmp_path,
        "candidate-failing-validity",
        review_status="thesis_candidate",
        valid_for_thesis_repeatability=False,
    )

    output_dir = build_thesis_evidence_index(project_root=tmp_path)

    payload = _payload(output_dir)
    entries = _run_entries(payload, "single_segment_repeatability")
    assert len(entries) == 1, "lower-trust run must still be included per operator preference"
    entry = entries[0]
    assert entry["run_id"] == "candidate-failing-validity"
    assert entry["trust_level"] == "lower_trust"
    assert entry["trust_warnings"], "the inconsistency must be surfaced on the run entry"
    assert any("valid_for_thesis_repeatability=False" in str(item) for item in entry["trust_warnings"])
    assert payload["run_count_lower_trust"] == 1
    assert payload["warnings"], "top-level warnings list must include the lower-trust run"
    assert any("candidate-failing-validity" in str(item) for item in payload["warnings"])


def test_thesis_candidate_with_passing_validity_is_thesis_trusted(tmp_path: Path) -> None:
    _write_run_with_thesis_validity(
        tmp_path,
        "candidate-passing-validity",
        review_status="thesis_candidate",
        valid_for_thesis_repeatability=True,
    )

    output_dir = build_thesis_evidence_index(project_root=tmp_path)

    payload = _payload(output_dir)
    entries = _run_entries(payload, "single_segment_repeatability")
    assert len(entries) == 1
    entry = entries[0]
    assert entry["trust_level"] == "thesis_trusted"
    assert not entry["trust_warnings"]
    assert payload["run_count_lower_trust"] == 0
    assert not payload["warnings"]


def _write_registration_sampling_study_run(
    root: Path,
    *,
    name: str,
    review_status: str,
    recommendation_valid: bool,
) -> Path:
    run_dir = root / "data" / "experiments" / "registration_sampling_study" / name
    run_dir.mkdir(parents=True)
    (run_dir / "metadata.json").write_text(
        json.dumps(
            {
                "experiment_name": "registration_sampling_study",
                "run_id": name,
                "provenance_info": {"operating_mode": "single_segment", "mock_mode": False},
                "trust_info": {
                    "run_trust_mode": "thesis_trusted",
                    "valid_for_model_training": False,
                    "valid_for_thesis_repeatability": False,
                },
            }
        ),
        encoding="utf-8",
    )
    (run_dir / "summary.json").write_text(
        json.dumps(
            {
                "experiment_name": "registration_sampling_study",
                "run_id": name,
                "success": True,
                "status": "success",
                "experiment_metrics": {
                    "run_trust_mode": "thesis_trusted",
                    "valid_for_model_training": False,
                    "valid_for_thesis_repeatability": False,
                    "valid_for_registration_protocol_recommendation": bool(recommendation_valid),
                    "candidate_registration_fre_mm": 0.42,
                    "recommended_protocol": {
                        "recommended_subset_size": 12,
                        "recommended_samples_per_point": 20,
                        "recommended_averaging_method": "mean",
                        "rationale": "test",
                    },
                    "run_provenance": {"operating_mode": "single_segment", "mock_mode": False},
                },
            }
        ),
        encoding="utf-8",
    )
    (run_dir / "run_review.json").write_text(
        json.dumps({"review_status": review_status, "include_in_evidence_index": False}),
        encoding="utf-8",
    )
    return run_dir


def test_evidence_index_registration_study_recommendation_valid_is_thesis_trusted(tmp_path: Path) -> None:
    _write_registration_sampling_study_run(
        tmp_path,
        name="rsstudy-good",
        review_status="thesis_candidate",
        recommendation_valid=True,
    )

    output_dir = build_thesis_evidence_index(project_root=tmp_path)

    payload = _payload(output_dir)
    entries = _run_entries(payload, "registration_sampling_study")
    assert len(entries) == 1
    entry = entries[0]
    assert entry["trust_level"] == "thesis_trusted"
    assert not entry["trust_warnings"]
    # The study's recommendation should surface in the per-run key_metrics.
    metrics = entry.get("key_metrics", {})
    assert metrics.get("candidate_registration_fre_mm") == 0.42
    assert metrics.get("recommended_protocol", {}).get("recommended_subset_size") == 12


def test_evidence_index_registration_study_recommendation_invalid_is_lower_trust(tmp_path: Path) -> None:
    _write_registration_sampling_study_run(
        tmp_path,
        name="rsstudy-bad",
        review_status="thesis_candidate",
        recommendation_valid=False,
    )

    output_dir = build_thesis_evidence_index(project_root=tmp_path)

    payload = _payload(output_dir)
    entries = _run_entries(payload, "registration_sampling_study")
    assert len(entries) == 1
    entry = entries[0]
    assert entry["trust_level"] == "lower_trust"
    assert any("valid_for_registration_protocol_recommendation=False" in str(item) for item in entry["trust_warnings"])


def test_modeling_candidate_with_failing_model_training_is_lower_trust(tmp_path: Path) -> None:
    # Modeling-experiment naming triggers an additional gate on valid_for_model_training.
    _write_run_with_thesis_validity(
        tmp_path,
        "modeling-candidate-failing",
        review_status="advisor_share",
        valid_for_thesis_repeatability=True,
        valid_for_model_training=False,
        experiment_name="collect_pose_command_dataset",
    )

    output_dir = build_thesis_evidence_index(project_root=tmp_path)

    payload = _payload(output_dir)
    entries = _run_entries(payload, "collect_pose_command_dataset")
    assert len(entries) == 1
    entry = entries[0]
    assert entry["trust_level"] == "lower_trust"
    assert any("valid_for_model_training=False" in str(item) for item in entry["trust_warnings"])

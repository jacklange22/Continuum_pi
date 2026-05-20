"""Tests for the thesis data integrity audit script + the dry-run safety
plumbing it depends on (not_thesis_evidence stamp + classifier).

The audit script never modifies experiment outputs; it only reads existing
run folders and writes a diagnostic report. These tests build small
synthetic run folders inside ``tmp_path`` and verify classification.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from continuum_robot.experiments.audit_thesis_data_integrity import (
    CLASS_DEBUG_OR_SYNTHETIC,
    CLASS_MISSING_METADATA,
    CLASS_NEEDS_REVIEW,
    CLASS_THESIS_CANDIDATE,
    audit_project,
    audit_run_folder,
)


def _write_run_folder(
    root: Path,
    *,
    experiment_name: str,
    timestamp: str,
    metadata: dict | None,
    summary: dict | None,
    config_snapshot: dict | None,
    review: dict | None = None,
    include_transform_chain: bool = True,
) -> Path:
    run_dir = root / experiment_name / f"{timestamp}_{experiment_name}"
    run_dir.mkdir(parents=True, exist_ok=True)
    if metadata is not None:
        (run_dir / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
    if summary is not None:
        (run_dir / "summary.json").write_text(json.dumps(summary), encoding="utf-8")
    if config_snapshot is not None:
        (run_dir / "config_snapshot.yaml").write_text(yaml.safe_dump(config_snapshot), encoding="utf-8")
    if review is not None:
        (run_dir / "run_review.json").write_text(json.dumps(review), encoding="utf-8")
    if include_transform_chain:
        (run_dir / "transform_chain_summary.json").write_text(
            json.dumps({"placeholder": True}), encoding="utf-8"
        )
    return run_dir


def _thesis_grade_run_fixture(tmp_path: Path, *, name: str = "single_segment_repeatability") -> Path:
    return _write_run_folder(
        tmp_path / "data" / "experiments",
        experiment_name=name,
        timestamp="20260520_010000",
        metadata={
            "experiment_name": name,
            "timestamp_utc": "2026-05-20T01:00:00+00:00",
            "provenance_info": {
                "mock_mode": False,
                "tracker_backend": "ndi",
            },
            "registration_info": {
                "runtime_tip_mode": "coil_as_tip",
                "runtime_tip_trust_level": "thesis_trusted",
            },
            "trust_info": {
                "run_trust_mode": "thesis_trusted",
                "not_thesis_evidence": False,
                "not_thesis_evidence_reasons": [],
            },
        },
        summary={
            "experiment_metrics": {
                "status": "success",
                "not_thesis_evidence": False,
            }
        },
        config_snapshot={"dry_run": False, "tool_id": "0A"},
    )


class TestAuditClassifications:
    def test_clean_run_is_thesis_candidate(self, tmp_path: Path) -> None:
        run_dir = _thesis_grade_run_fixture(tmp_path)
        result = audit_run_folder(run_dir)
        assert result["classification"] == CLASS_THESIS_CANDIDATE
        assert result["synthetic_or_mock"] is False
        assert result["provenance_gaps"] == []

    def test_dry_run_config_flags_synthetic(self, tmp_path: Path) -> None:
        run_dir = _write_run_folder(
            tmp_path / "data" / "experiments",
            experiment_name="aurora_grid_accuracy",
            timestamp="20260520_010100",
            metadata={
                "experiment_name": "aurora_grid_accuracy",
                "provenance_info": {"mock_mode": False, "tracker_backend": "ndi"},
                "registration_info": {
                    "runtime_tip_mode": "coil_as_tip",
                    "runtime_tip_trust_level": "thesis_trusted",
                },
                "trust_info": {"run_trust_mode": "thesis_trusted"},
            },
            summary={"experiment_metrics": {"synthetic_seed_used": 12345}},
            config_snapshot={"dry_run": True},
        )
        result = audit_run_folder(run_dir)
        assert result["classification"] == CLASS_DEBUG_OR_SYNTHETIC
        assert result["synthetic_or_mock"] is True
        # Both the dry_run config flag and the synthetic_seed_used metric
        # should be visible in the reasons list.
        reasons = result["synthetic_reasons"]
        assert any("dry_run" in r for r in reasons)
        assert any("synthetic_seed_used" in r for r in reasons)

    def test_mock_mode_flags_synthetic(self, tmp_path: Path) -> None:
        run_dir = _write_run_folder(
            tmp_path / "data" / "experiments",
            experiment_name="collect_pose_command_dataset",
            timestamp="20260520_010200",
            metadata={
                "experiment_name": "collect_pose_command_dataset",
                "provenance_info": {"mock_mode": True, "tracker_backend": "mock_tracker_manager"},
                "registration_info": {
                    "runtime_tip_mode": "coil_as_tip",
                    "runtime_tip_trust_level": "thesis_trusted",
                },
                "trust_info": {
                    "run_trust_mode": "mock",
                    "not_thesis_evidence": True,
                    "not_thesis_evidence_reasons": ["mock_mode"],
                },
            },
            summary={"experiment_metrics": {"not_thesis_evidence": True}},
            config_snapshot={"dry_run": False},
        )
        result = audit_run_folder(run_dir)
        assert result["classification"] == CLASS_DEBUG_OR_SYNTHETIC
        assert result["synthetic_or_mock"] is True
        # We should see multiple reasons, not just one.
        assert len(result["synthetic_reasons"]) >= 2

    def test_mock_experiments_tree_flags_synthetic(self, tmp_path: Path) -> None:
        # Path-based detection: runs under data/mock_experiments/ are
        # categorically synthetic regardless of metadata content.
        run_dir = _write_run_folder(
            tmp_path / "data" / "mock_experiments",
            experiment_name="collect_pose_command_dataset",
            timestamp="20260520_010300",
            metadata={
                "experiment_name": "collect_pose_command_dataset",
                "provenance_info": {"mock_mode": False, "tracker_backend": "ndi"},
                "registration_info": {"runtime_tip_mode": "coil_as_tip", "runtime_tip_trust_level": "thesis_trusted"},
                "trust_info": {"run_trust_mode": "thesis_trusted"},
            },
            summary={"experiment_metrics": {}},
            config_snapshot={"dry_run": False},
        )
        result = audit_run_folder(run_dir)
        assert result["classification"] == CLASS_DEBUG_OR_SYNTHETIC
        assert any("mock_experiments" in r for r in result["synthetic_reasons"])

    def test_missing_provenance_is_needs_review(self, tmp_path: Path) -> None:
        # No transform chain, no runtime tip status — but not synthetic.
        run_dir = _write_run_folder(
            tmp_path / "data" / "experiments",
            experiment_name="custom_experiment",
            timestamp="20260520_010400",
            metadata={
                "experiment_name": "custom_experiment",
                "provenance_info": {"tracker_backend": "ndi"},
                "registration_info": {},  # missing runtime_tip_mode + trust
                "trust_info": {"run_trust_mode": "thesis_trusted"},
            },
            summary={"experiment_metrics": {}},
            config_snapshot={"dry_run": False},
            include_transform_chain=False,
        )
        result = audit_run_folder(run_dir)
        assert result["classification"] == CLASS_NEEDS_REVIEW
        gaps = result["provenance_gaps"]
        assert "missing_transform_chain_summary" in gaps
        assert "missing_runtime_tip_mode" in gaps

    def test_missing_metadata_is_classified(self, tmp_path: Path) -> None:
        # Run folder exists but lacks metadata.json AND summary.json.
        run_dir = tmp_path / "data" / "experiments" / "empty_experiment" / "20260520_010500_run"
        run_dir.mkdir(parents=True)
        (run_dir / "stray_file.txt").write_text("hello", encoding="utf-8")
        result = audit_run_folder(run_dir)
        assert result["classification"] == CLASS_MISSING_METADATA


class TestAuditProjectReport:
    def test_audit_project_writes_three_artifacts_and_counts_match(self, tmp_path: Path) -> None:
        # Build a project with one thesis_candidate + one dry-run synthetic.
        _thesis_grade_run_fixture(tmp_path)
        _write_run_folder(
            tmp_path / "data" / "experiments",
            experiment_name="aurora_grid_accuracy",
            timestamp="20260520_020000",
            metadata={
                "experiment_name": "aurora_grid_accuracy",
                "provenance_info": {"mock_mode": False, "tracker_backend": "ndi"},
                "registration_info": {
                    "runtime_tip_mode": "coil_as_tip",
                    "runtime_tip_trust_level": "thesis_trusted",
                },
                "trust_info": {"run_trust_mode": "thesis_trusted"},
            },
            summary={"experiment_metrics": {"synthetic_seed_used": 99}},
            config_snapshot={"dry_run": True},
        )
        result = audit_project(project_root=tmp_path, include_mock_tree=True, include_archived=False)
        paths = result["paths"]
        assert paths["summary_path"].exists()
        assert paths["csv_path"].exists()
        assert paths["txt_path"].exists()

        summary = json.loads(paths["summary_path"].read_text(encoding="utf-8"))
        assert summary["run_count_total"] == 2
        assert summary["classification_counts"][CLASS_THESIS_CANDIDATE] == 1
        assert summary["classification_counts"][CLASS_DEBUG_OR_SYNTHETIC] == 1
        # The synthetic run must be explicitly listed at the top of the
        # synthetic_or_mock_runs array so the operator sees it immediately.
        assert len(summary["synthetic_or_mock_runs"]) == 1
        assert "aurora_grid_accuracy" in summary["synthetic_or_mock_runs"][0]

        # TXT report must also call the synthetic run out by name.
        txt = paths["txt_path"].read_text(encoding="utf-8")
        assert "Synthetic / mock / dry-run runs" in txt
        assert "aurora_grid_accuracy" in txt

    def test_audit_project_never_modifies_run_data(self, tmp_path: Path) -> None:
        run_dir = _thesis_grade_run_fixture(tmp_path)
        meta_path = run_dir / "metadata.json"
        original = meta_path.read_text(encoding="utf-8")
        audit_project(project_root=tmp_path, include_mock_tree=False, include_archived=False)
        # Audit must leave the run folder bit-identical.
        assert meta_path.read_text(encoding="utf-8") == original

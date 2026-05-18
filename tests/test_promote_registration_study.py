"""Tests for the registration_sampling_study promote tool.

The promote tool is the only explicit path to update the active registration
artifact based on a study run. It must:

1. Refuse to promote when the candidate is missing or malformed.
2. Back up the current active registration before overwriting.
3. Preserve the legacy field names in latest_registration.json.
4. Support a dry-run that writes nothing.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from continuum_robot.data.promote_registration_study import (
    ACTIVE_FILENAME,
    CANDIDATE_FILENAME,
    promote_registration_study,
)


def _write_candidate(
    run_dir: Path,
    *,
    schema_version: str = "registration_sampling_study_candidate_v1",
    fre_mm: float | None = 0.42,
    transform_valid: bool = True,
    labels: list[str] | None = None,
) -> Path:
    run_dir.mkdir(parents=True, exist_ok=True)
    labels = labels if labels is not None else ["L1", "L2", "L3", "L4"]
    candidate = {
        "schema_version": schema_version,
        "timestamp_utc": "2026-05-14T22:00:00Z",
        "source_run_id": "test_run_id",
        "candidate_kind": "registration_sampling_study_full_subset_mean_centers",
        "landmark_labels": labels,
        "raw_captured_landmarks_aurora_xyz": {
            label: [[float(i), 0.0, -200.0]] for i, label in enumerate(labels, start=1)
        },
        "averaged_landmarks_aurora_xyz": {
            label: [float(i), 0.0, -200.0] for i, label in enumerate(labels, start=1)
        },
        "truth_points_in_robot_xyz": {
            label: [float(i), 0.0, -5.0] for i, label in enumerate(labels, start=1)
        },
        "fre_mm": fre_mm,
        "max_residual_mm": 0.6,
        "residuals_robot_xyz_mm_per_point": [0.4 for _ in labels],
        "T_robot_aurora": (
            [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 195.0],
                [0.0, 0.0, 0.0, 1.0],
            ]
            if transform_valid
            else None
        ),
        "promote_warning": "test",
    }
    path = run_dir / CANDIDATE_FILENAME
    path.write_text(json.dumps(candidate, indent=2), encoding="utf-8")
    return path


def test_promote_creates_active_and_backs_up_previous(tmp_path: Path) -> None:
    run_dir = tmp_path / "data" / "experiments" / "registration_sampling_study" / "20260514_220000_registration_sampling_study"
    _write_candidate(run_dir)
    registrations_root = tmp_path / "registrations"
    registrations_root.mkdir()
    # Pre-existing active registration must be backed up, not lost.
    previous = {"timestamp_utc": "OLD", "landmark_labels": ["A", "B", "C", "D"], "fre_mm": 9.9, "T_robot_aurora": [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]}
    (registrations_root / ACTIVE_FILENAME).write_text(json.dumps(previous), encoding="utf-8")

    report = promote_registration_study(
        run_dir=run_dir,
        registrations_root=registrations_root,
        operator_note="study X confirms lower FRE",
    )

    assert report["dry_run"] is False
    assert report["backup_path"] is not None
    backup = Path(report["backup_path"])
    assert backup.exists()
    backup_payload = json.loads(backup.read_text(encoding="utf-8"))
    assert backup_payload["timestamp_utc"] == "OLD"  # previous content preserved
    active_payload = json.loads((registrations_root / ACTIVE_FILENAME).read_text(encoding="utf-8"))
    assert active_payload["schema_version"] == "registration_promoted_from_sampling_study_v1"
    assert active_payload["fre_mm"] == pytest.approx(0.42)
    assert active_payload["landmark_labels"] == ["L1", "L2", "L3", "L4"]
    # Legacy field names preserved for existing readers.
    assert "raw_captured_landmarks_robot_xyz" in active_payload
    assert active_payload["raw_captured_landmarks_robot_xyz"]["L1"] == [[1.0, 0.0, -200.0]]
    assert active_payload["operator_note"] == "study X confirms lower FRE"


def test_promote_creates_active_when_no_previous_exists(tmp_path: Path) -> None:
    run_dir = tmp_path / "study_run"
    _write_candidate(run_dir)
    registrations_root = tmp_path / "registrations"

    report = promote_registration_study(
        run_dir=run_dir,
        registrations_root=registrations_root,
    )

    assert report["backup_path"] is None
    assert Path(report["active_path"]).exists()


def test_promote_dry_run_writes_nothing(tmp_path: Path) -> None:
    run_dir = tmp_path / "study_run"
    _write_candidate(run_dir)
    registrations_root = tmp_path / "registrations"
    registrations_root.mkdir()
    (registrations_root / ACTIVE_FILENAME).write_text(json.dumps({"timestamp_utc": "OLD", "landmark_labels": [], "fre_mm": 9.9, "T_robot_aurora": []}), encoding="utf-8")

    report = promote_registration_study(
        run_dir=run_dir,
        registrations_root=registrations_root,
        dry_run=True,
    )

    assert report["dry_run"] is True
    # Active file is unchanged.
    active = json.loads((registrations_root / ACTIVE_FILENAME).read_text(encoding="utf-8"))
    assert active["timestamp_utc"] == "OLD"


def test_promote_refuses_when_schema_version_unknown(tmp_path: Path) -> None:
    run_dir = tmp_path / "study_run"
    _write_candidate(run_dir, schema_version="evil_schema")
    registrations_root = tmp_path / "registrations"
    with pytest.raises(ValueError, match="schema_version"):
        promote_registration_study(run_dir=run_dir, registrations_root=registrations_root)


def test_promote_refuses_when_transform_missing(tmp_path: Path) -> None:
    run_dir = tmp_path / "study_run"
    _write_candidate(run_dir, transform_valid=False)
    registrations_root = tmp_path / "registrations"
    with pytest.raises(ValueError, match="T_robot_aurora"):
        promote_registration_study(run_dir=run_dir, registrations_root=registrations_root)


def test_promote_refuses_when_fre_missing(tmp_path: Path) -> None:
    run_dir = tmp_path / "study_run"
    _write_candidate(run_dir, fre_mm=None)
    registrations_root = tmp_path / "registrations"
    with pytest.raises(ValueError, match="fre_mm"):
        promote_registration_study(run_dir=run_dir, registrations_root=registrations_root)


def test_promote_refuses_when_candidate_missing(tmp_path: Path) -> None:
    run_dir = tmp_path / "study_run"
    run_dir.mkdir()
    registrations_root = tmp_path / "registrations"
    with pytest.raises(FileNotFoundError):
        promote_registration_study(run_dir=run_dir, registrations_root=registrations_root)

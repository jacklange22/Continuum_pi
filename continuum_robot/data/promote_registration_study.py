"""Promote a `registration_sampling_study` candidate to the active registration.

This is the explicit operator-driven follow-up to a study run. It NEVER runs
automatically. It reads `registration_candidate.json` from one study run
directory, back up the current `latest_registration.json` to a timestamped
sibling, and writes a properly-shaped registration JSON into the active
slot.

Usage (CLI):

    python -m continuum_robot.data.promote_registration_study \
        --run-dir data/experiments/registration_sampling_study/<run> \
        [--registrations-root data/registrations] \
        [--operator-note "study X confirmed lower FRE"]

The script exits non-zero on any failure (missing candidate, missing
T_robot_aurora, FRE not present, etc) and refuses to overwrite the active
slot when `--dry-run` is set.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


CANDIDATE_FILENAME = "registration_candidate.json"
ACTIVE_FILENAME = "latest_registration.json"


def promote_registration_study(
    *,
    run_dir: Path,
    registrations_root: Path,
    operator_note: str = "",
    dry_run: bool = False,
) -> dict[str, Any]:
    """Promote one study's candidate registration into the active slot.

    Returns a small report dict describing what happened. Raises on
    structural errors (missing files, missing transforms, etc).
    """
    run_dir = Path(run_dir).resolve()
    if not run_dir.is_dir():
        raise FileNotFoundError(f"Study run directory does not exist: {run_dir}")
    candidate_path = run_dir / CANDIDATE_FILENAME
    if not candidate_path.exists():
        raise FileNotFoundError(
            f"{CANDIDATE_FILENAME} not found in {run_dir}. Was this a registration_sampling_study run?"
        )
    candidate = json.loads(candidate_path.read_text(encoding="utf-8"))
    _validate_candidate(candidate)
    registrations_root = Path(registrations_root).resolve()
    registrations_root.mkdir(parents=True, exist_ok=True)
    active_path = registrations_root / ACTIVE_FILENAME
    backup_path: Path | None = None
    if active_path.exists():
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
        backup_path = registrations_root / f"latest_registration_backup_{timestamp}.json"
    payload = _build_active_payload(candidate, operator_note=operator_note, source_run_dir=run_dir)
    if dry_run:
        return {
            "dry_run": True,
            "candidate_path": str(candidate_path),
            "active_path": str(active_path),
            "would_backup_to": str(backup_path) if backup_path is not None else None,
            "candidate_fre_mm": float(candidate.get("fre_mm")),
            "candidate_label_count": int(len(candidate.get("landmark_labels") or [])),
            "operator_note": operator_note,
        }
    if backup_path is not None:
        shutil.copy2(active_path, backup_path)
    active_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return {
        "dry_run": False,
        "candidate_path": str(candidate_path),
        "active_path": str(active_path),
        "backup_path": str(backup_path) if backup_path is not None else None,
        "candidate_fre_mm": float(candidate.get("fre_mm")),
        "candidate_label_count": int(len(candidate.get("landmark_labels") or [])),
        "operator_note": operator_note,
    }


def _validate_candidate(candidate: dict[str, Any]) -> None:
    if not isinstance(candidate, dict):
        raise ValueError("registration_candidate.json is not a JSON object.")
    if candidate.get("schema_version") != "registration_sampling_study_candidate_v1":
        raise ValueError(
            "registration_candidate.json has unexpected schema_version "
            f"{candidate.get('schema_version')!r}; refusing to promote."
        )
    T = candidate.get("T_robot_aurora")
    if T is None:
        raise ValueError("Candidate has no T_robot_aurora; refusing to promote.")
    rows = list(T)
    if len(rows) != 4 or any(len(list(row)) != 4 for row in rows):
        raise ValueError("Candidate T_robot_aurora is not a 4x4 matrix; refusing to promote.")
    if candidate.get("fre_mm") is None:
        raise ValueError("Candidate has no fre_mm; refusing to promote.")
    if not candidate.get("landmark_labels"):
        raise ValueError("Candidate has no landmark_labels; refusing to promote.")


def _build_active_payload(
    candidate: dict[str, Any],
    *,
    operator_note: str,
    source_run_dir: Path,
) -> dict[str, Any]:
    """Construct an active `latest_registration.json` payload from a candidate.

    The legacy field names (`raw_captured_landmarks_robot_xyz`) historically
    store aurora-frame captures; we preserve that convention so existing
    readers do not break.
    """
    labels = list(candidate.get("landmark_labels") or [])
    aurora_samples_by_label = dict(candidate.get("raw_captured_landmarks_aurora_xyz") or {})
    averaged_aurora = dict(candidate.get("averaged_landmarks_aurora_xyz") or {})
    truth_in_robot = dict(candidate.get("truth_points_in_robot_xyz") or {})
    residuals = list(candidate.get("residuals_robot_xyz_mm_per_point") or [])
    residuals_by_label = {
        label: [float(residuals[index])] if index < len(residuals) else []
        for index, label in enumerate(labels)
    }
    return {
        "schema_version": "registration_promoted_from_sampling_study_v1",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "promoted_from_study_run_dir": str(source_run_dir),
        "promoted_from_study_run_id": candidate.get("source_run_id", ""),
        "operator_note": operator_note,
        "landmark_labels": labels,
        # Legacy aliases: these names appear in the historical registration
        # artifact; preserve them so existing readers continue to work.
        "raw_captured_landmarks_robot_xyz": aurora_samples_by_label,
        "raw_captured_landmarks_aurora_xyz": aurora_samples_by_label,
        "averaged_landmarks_robot_xyz": averaged_aurora,
        "averaged_landmarks_aurora_xyz": averaged_aurora,
        "truth_points_in_robot_xyz": truth_in_robot,
        "residuals_robot_xyz_mm": residuals_by_label,
        "fre_mm": float(candidate.get("fre_mm")),
        "T_robot_aurora": [
            [float(value) for value in row] for row in list(candidate.get("T_robot_aurora"))
        ],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Promote a registration_sampling_study candidate into the active registration slot.")
    parser.add_argument("--run-dir", required=True, help="Path to a registration_sampling_study run directory.")
    parser.add_argument(
        "--registrations-root",
        default="data/registrations",
        help="Where to write latest_registration.json (default: data/registrations).",
    )
    parser.add_argument("--operator-note", default="", help="One-line note recorded inside the promoted artifact.")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate the candidate and report what would happen without writing anything.",
    )
    args = parser.parse_args(argv)
    try:
        report = promote_registration_study(
            run_dir=Path(args.run_dir),
            registrations_root=Path(args.registrations_root),
            operator_note=str(args.operator_note or ""),
            dry_run=bool(args.dry_run),
        )
    except (FileNotFoundError, ValueError) as exc:
        sys.stderr.write(f"Refusing to promote: {exc}\n")
        return 2
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

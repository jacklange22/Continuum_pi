"""Promote a `registration_trial` candidate subset to the active registration.

This is the explicit operator-driven follow-up to a trial run. It NEVER runs
automatically. It reads a `registration_trial` run directory's
`trial_report.json` to find:

- The captured aurora points by label.
- The truth body-frame coordinates by label.
- The best subset (or an operator-specified subset).

Then it solves the registration for that subset using mean averaging,
backs up the current `latest_registration.json` to a timestamped sibling,
and writes a new active registration. The promote step is the only path
that affects the production registration.

Usage (CLI):

    python -m continuum_robot.data.promote_registration_trial \
        --run-dir data/experiments/registration_trial/<run> \
        [--subset L1,L2,L4,L7]    # default: report's global best subset
        [--averaging mean]        # default: mean
        [--registrations-root data/registrations] \
        [--operator-note "Trial T3 shows L4/L7 lower FRE by 0.4 mm"]
        [--dry-run]

Exits non-zero on any structural failure (missing report, missing subset,
fewer than 3 shared labels, etc).
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from continuum_robot.registration.rigid_solver import RigidRegistrationSolver
from continuum_robot.registration.trial_analysis import average_captures


TRIAL_REPORT_FILENAME = "trial_report.json"
ACTIVE_FILENAME = "latest_registration.json"


def promote_registration_trial(
    *,
    run_dir: Path,
    registrations_root: Path,
    subset_labels: list[str] | None = None,
    averaging_method: str = "mean",
    trimmed_fraction: float = 0.2,
    mad_k: float = 3.5,
    operator_note: str = "",
    dry_run: bool = False,
) -> dict[str, Any]:
    """Promote one trial's chosen subset into the active registration slot.

    Returns a small report dict describing what happened. Raises on
    structural errors (missing files, missing labels, fewer than 3 shared
    labels, etc).
    """
    run_dir = Path(run_dir).resolve()
    if not run_dir.is_dir():
        raise FileNotFoundError(f"Trial run directory does not exist: {run_dir}")
    report_path = run_dir / TRIAL_REPORT_FILENAME
    if not report_path.exists():
        raise FileNotFoundError(
            f"{TRIAL_REPORT_FILENAME} not found in {run_dir}. Was this a registration_trial run?"
        )
    report = json.loads(report_path.read_text(encoding="utf-8"))

    raw_captures_by_label = dict(report.get("raw_captures_by_label") or {})
    truth_by_label = dict(report.get("truth_by_label") or {})
    if not raw_captures_by_label or not truth_by_label:
        raise ValueError(
            "trial_report.json does not contain raw_captures_by_label and truth_by_label; "
            "this trial run cannot be promoted."
        )

    # Choose the subset of labels.
    chosen_labels = _resolve_subset_labels(report, subset_labels, raw_captures_by_label, truth_by_label)

    # Average each chosen label's captures with the requested method.
    averaged_by_label: dict[str, list[float]] = {}
    for label in chosen_labels:
        captures = raw_captures_by_label[label]
        if not captures:
            raise ValueError(f"Label {label} has no captures in trial_report.json; cannot promote.")
        result = average_captures(
            captures,
            method=str(averaging_method),
            trimmed_fraction=float(trimmed_fraction),
            mad_k=float(mad_k),
        )
        averaged_by_label[label] = list(result.averaged_xyz_mm)

    # Solve the registration for the chosen subset.
    measured = np.asarray([averaged_by_label[label] for label in chosen_labels], dtype=float)
    truth = np.asarray([truth_by_label[label] for label in chosen_labels], dtype=float)
    if measured.shape[0] < 3 or measured.shape != truth.shape:
        raise ValueError(
            f"Need at least 3 shared labels with matching shapes to solve a registration; "
            f"measured shape={measured.shape}, truth shape={truth.shape}."
        )
    solver = RigidRegistrationSolver()
    solve = solver.solve_alignment(measured, truth)
    T_robot_aurora = np.asarray(solve["transform"], dtype=float)
    if T_robot_aurora.shape != (4, 4):
        raise RuntimeError("Solver did not return a 4x4 transform; refusing to promote.")
    fre_mm = float(solve.get("rmse_mm") or 0.0)
    residuals = np.asarray(solve.get("residuals")).T if "residuals" in solve else np.zeros((measured.shape[0], 3))
    if residuals.shape == (3, measured.shape[0]):
        residuals = residuals.T
    per_point = np.linalg.norm(residuals, axis=1) if residuals.size else np.zeros(0)

    registrations_root = Path(registrations_root).resolve()
    registrations_root.mkdir(parents=True, exist_ok=True)
    active_path = registrations_root / ACTIVE_FILENAME
    backup_path: Path | None = None
    if active_path.exists():
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
        backup_path = registrations_root / f"latest_registration_backup_{timestamp}.json"

    payload = _build_active_payload(
        chosen_labels=chosen_labels,
        raw_captures_by_label=raw_captures_by_label,
        averaged_by_label=averaged_by_label,
        truth_by_label=truth_by_label,
        T_robot_aurora=T_robot_aurora,
        fre_mm=fre_mm,
        per_point_residual_mm=per_point.tolist(),
        averaging_method=str(averaging_method),
        operator_note=operator_note,
        source_run_dir=run_dir,
    )

    if dry_run:
        return {
            "dry_run": True,
            "chosen_labels": chosen_labels,
            "averaging_method": str(averaging_method),
            "fre_mm": fre_mm,
            "max_residual_mm": float(np.max(per_point)) if per_point.size else 0.0,
            "active_path": str(active_path),
            "would_backup_to": str(backup_path) if backup_path is not None else None,
            "operator_note": operator_note,
        }

    if backup_path is not None:
        shutil.copy2(active_path, backup_path)
    active_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return {
        "dry_run": False,
        "chosen_labels": chosen_labels,
        "averaging_method": str(averaging_method),
        "fre_mm": fre_mm,
        "max_residual_mm": float(np.max(per_point)) if per_point.size else 0.0,
        "active_path": str(active_path),
        "backup_path": str(backup_path) if backup_path is not None else None,
        "operator_note": operator_note,
    }


def _resolve_subset_labels(
    report: dict[str, Any],
    requested: list[str] | None,
    raw_captures_by_label: dict[str, list[list[float]]],
    truth_by_label: dict[str, list[float]],
) -> list[str]:
    """Decide which labels to use for the promoted registration."""
    if requested:
        chosen = [label.strip() for label in requested if label.strip()]
    else:
        global_best = (report.get("subset_search_summary") or {}).get("global_best") if isinstance(report.get("subset_search_summary"), dict) else None
        if isinstance(global_best, dict) and global_best.get("labels"):
            chosen = [str(label) for label in global_best["labels"]]
        else:
            # Fall back to every captured label that has truth coordinates.
            chosen = sorted(
                set(raw_captures_by_label.keys()) & set(truth_by_label.keys())
            )
    missing_captures = [label for label in chosen if label not in raw_captures_by_label]
    if missing_captures:
        raise ValueError(f"Subset references labels with no captures in the trial: {missing_captures}")
    missing_truth = [label for label in chosen if label not in truth_by_label]
    if missing_truth:
        raise ValueError(f"Subset references labels with no truth coordinates: {missing_truth}")
    if len(chosen) < 3:
        raise ValueError(f"At least 3 labels are required; got {len(chosen)}: {chosen}")
    return chosen


def _build_active_payload(
    *,
    chosen_labels: list[str],
    raw_captures_by_label: dict[str, list[list[float]]],
    averaged_by_label: dict[str, list[float]],
    truth_by_label: dict[str, list[float]],
    T_robot_aurora: np.ndarray,
    fre_mm: float,
    per_point_residual_mm: list[float],
    averaging_method: str,
    operator_note: str,
    source_run_dir: Path,
) -> dict[str, Any]:
    """Build the active-slot registration payload.

    The legacy field names ``raw_captured_landmarks_robot_xyz`` historically
    store aurora-frame captures; we preserve that convention so existing
    readers (registration_service, GUI tabs, evidence index) keep working.
    """
    raw_for_chosen = {label: [list(map(float, point)) for point in raw_captures_by_label[label]] for label in chosen_labels}
    averaged_for_chosen = {label: list(map(float, averaged_by_label[label])) for label in chosen_labels}
    truth_for_chosen = {label: list(map(float, truth_by_label[label])) for label in chosen_labels}
    residuals = {
        label: [float(per_point_residual_mm[idx])] if idx < len(per_point_residual_mm) else []
        for idx, label in enumerate(chosen_labels)
    }
    return {
        "schema_version": "registration_promoted_from_trial_v1",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "promoted_from_trial_run_dir": str(source_run_dir),
        "operator_note": operator_note,
        "averaging_method": averaging_method,
        "landmark_labels": list(chosen_labels),
        "raw_captured_landmarks_robot_xyz": raw_for_chosen,
        "raw_captured_landmarks_aurora_xyz": raw_for_chosen,
        "averaged_landmarks_robot_xyz": averaged_for_chosen,
        "averaged_landmarks_aurora_xyz": averaged_for_chosen,
        "truth_points_in_robot_xyz": truth_for_chosen,
        "residuals_robot_xyz_mm": residuals,
        "fre_mm": float(fre_mm),
        "T_robot_aurora": [[float(value) for value in row] for row in T_robot_aurora.tolist()],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Promote a registration_trial run's chosen subset into the active registration slot."
    )
    parser.add_argument("--run-dir", required=True, help="Path to a registration_trial run directory.")
    parser.add_argument(
        "--subset",
        default=None,
        help="Comma-separated landmark labels to promote (e.g. L1,L2,L4,L7). Defaults to the report's global_best subset.",
    )
    parser.add_argument(
        "--averaging",
        default="mean",
        help="Averaging method: mean | median | trimmed_mean | mad_filtered_mean (default: mean).",
    )
    parser.add_argument(
        "--registrations-root",
        default="data/registrations",
        help="Where to write latest_registration.json (default: data/registrations).",
    )
    parser.add_argument("--operator-note", default="", help="One-line note recorded inside the promoted artifact.")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate inputs and report what would happen without writing anything.",
    )
    parser.add_argument("--trimmed-fraction", type=float, default=0.2)
    parser.add_argument("--mad-k", type=float, default=3.5)
    args = parser.parse_args(argv)
    subset = (
        [s.strip() for s in str(args.subset).split(",") if s.strip()]
        if args.subset is not None
        else None
    )
    try:
        report = promote_registration_trial(
            run_dir=Path(args.run_dir),
            registrations_root=Path(args.registrations_root),
            subset_labels=subset,
            averaging_method=str(args.averaging or "mean"),
            trimmed_fraction=float(args.trimmed_fraction),
            mad_k=float(args.mad_k),
            operator_note=str(args.operator_note or ""),
            dry_run=bool(args.dry_run),
        )
    except (FileNotFoundError, ValueError, RuntimeError) as exc:
        sys.stderr.write(f"Refusing to promote: {exc}\n")
        return 2
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

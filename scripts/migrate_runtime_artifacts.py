#!/usr/bin/env python3
"""Migrate legacy runtime artifacts into the canonical data/ layout.

This script retires the old repo-root ``runs/`` location and the older flat
``data/pivot_captures/`` bucket. It preserves existing bundles, moves them into
the canonical pivot experiment subtrees, and rewrites moved bundle references so
they still point at the migrated capture or staged-tip artifacts.
"""

from __future__ import annotations

import json
from pathlib import Path
import shutil
from typing import Any

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OLD_RUN_ROOT = PROJECT_ROOT / "runs"
NEW_PIVOT_RUN_ROOT = PROJECT_ROOT / "data" / "experiments" / "pivot" / "runs"
OLD_PIVOT_CAPTURE_ROOT = PROJECT_ROOT / "data" / "pivot_captures"
NEW_PIVOT_CAPTURE_ROOT = PROJECT_ROOT / "data" / "experiments" / "pivot" / "captures"
TIP_ROOT = PROJECT_ROOT / "data" / "tip_cals"
STAGED_TIP_ROOT = TIP_ROOT / "staged"
LEGACY_DATA_RUN_ROOT = PROJECT_ROOT / "data" / "runs"


def _repo_relative(path: Path) -> str:
    return path.relative_to(PROJECT_ROOT).as_posix()


def _canonical_runtime_path_string(raw: Any) -> Any:
    if raw in (None, ""):
        return raw
    text = str(raw)
    candidate_name = Path(text).name

    capture_candidate = NEW_PIVOT_CAPTURE_ROOT / candidate_name
    if candidate_name and capture_candidate.exists():
        old_capture_markers = {
            "pivot_captures",
            _repo_relative(OLD_PIVOT_CAPTURE_ROOT),
            str(OLD_PIVOT_CAPTURE_ROOT),
        }
        if any(marker in text for marker in old_capture_markers):
            return _repo_relative(capture_candidate)

    staged_tip_candidate = STAGED_TIP_ROOT / candidate_name
    if candidate_name and "tip_cals" in text and "staged" in text:
        return _repo_relative(staged_tip_candidate)

    accepted_tip_candidate = TIP_ROOT / candidate_name
    if candidate_name and "tip_cals" in text:
        return _repo_relative(accepted_tip_candidate)

    registration_candidate = PROJECT_ROOT / "data" / "registrations" / candidate_name
    if candidate_name and "data/registrations" in text:
        return _repo_relative(registration_candidate)

    return raw


def _rewrite_mapping_paths(payload: Any) -> Any:
    if isinstance(payload, dict):
        rewritten: dict[str, Any] = {}
        for key, value in payload.items():
            if key in {
                "input_path",
                "pivot_input_path",
                "output_tip_file",
                "accepted_tip_file",
                "tip_output_file",
                "path",
            }:
                rewritten[str(key)] = _canonical_runtime_path_string(value)
            else:
                rewritten[str(key)] = _rewrite_mapping_paths(value)
        return rewritten
    if isinstance(payload, list):
        return [_rewrite_mapping_paths(item) for item in payload]
    return payload


def _move_children(src_root: Path, dst_root: Path) -> list[tuple[Path, Path]]:
    moves: list[tuple[Path, Path]] = []
    if not src_root.exists():
        return moves
    dst_root.mkdir(parents=True, exist_ok=True)
    for child in sorted(src_root.iterdir()):
        destination = dst_root / child.name
        if destination.exists():
            raise RuntimeError(f"Refusing to overwrite existing artifact destination: {destination}")
        shutil.move(str(child), str(destination))
        moves.append((child, destination))
    return moves


def _rewrite_run_bundle(run_dir: Path) -> bool:
    changed = False

    metadata_path = run_dir / "metadata.json"
    if metadata_path.exists():
        payload = json.loads(metadata_path.read_text(encoding="utf-8"))
        rewritten = _rewrite_mapping_paths(payload)
        if rewritten != payload:
            metadata_path.write_text(json.dumps(rewritten, indent=2) + "\n", encoding="utf-8")
            changed = True

    summary_path = run_dir / "summary.json"
    if summary_path.exists():
        payload = json.loads(summary_path.read_text(encoding="utf-8"))
        rewritten = _rewrite_mapping_paths(payload)
        if rewritten != payload:
            summary_path.write_text(json.dumps(rewritten, indent=2) + "\n", encoding="utf-8")
            changed = True

    config_snapshot_path = run_dir / "config_snapshot.yaml"
    if config_snapshot_path.exists():
        payload = yaml.safe_load(config_snapshot_path.read_text(encoding="utf-8")) or {}
        rewritten = _rewrite_mapping_paths(payload)
        if rewritten != payload:
            dumped = yaml.safe_dump(rewritten, sort_keys=False) or "{}\n"
            config_snapshot_path.write_text(dumped, encoding="utf-8")
            changed = True

    return changed


def _cleanup_empty_dir(path: Path) -> None:
    if not path.exists():
        return
    if any(path.iterdir()):
        return
    path.rmdir()


def main() -> int:
    moved: list[str] = []
    rewritten: list[str] = []

    for source, destination in _move_children(OLD_PIVOT_CAPTURE_ROOT, NEW_PIVOT_CAPTURE_ROOT):
        moved.append(f"{_repo_relative(source)} -> {_repo_relative(destination)}")

    for source, destination in _move_children(OLD_RUN_ROOT, NEW_PIVOT_RUN_ROOT):
        moved.append(f"{_repo_relative(source)} -> {_repo_relative(destination)}")

    if NEW_PIVOT_RUN_ROOT.exists():
        for run_dir in sorted(path for path in NEW_PIVOT_RUN_ROOT.iterdir() if path.is_dir()):
            if _rewrite_run_bundle(run_dir):
                rewritten.append(_repo_relative(run_dir))

    if LEGACY_DATA_RUN_ROOT.exists():
        for child in list(LEGACY_DATA_RUN_ROOT.iterdir()):
            child.unlink()
        _cleanup_empty_dir(LEGACY_DATA_RUN_ROOT)

    _cleanup_empty_dir(OLD_RUN_ROOT)
    _cleanup_empty_dir(OLD_PIVOT_CAPTURE_ROOT)

    if moved:
        print("Moved artifacts:")
        for item in moved:
            print(f"  - {item}")
    else:
        print("No legacy runtime artifacts needed moving.")

    if rewritten:
        print("Rewrote bundle references:")
        for item in rewritten:
            print(f"  - {item}")

    print(f"Canonical pivot runs: {_repo_relative(NEW_PIVOT_RUN_ROOT)}")
    print(f"Canonical pivot captures: {_repo_relative(NEW_PIVOT_CAPTURE_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

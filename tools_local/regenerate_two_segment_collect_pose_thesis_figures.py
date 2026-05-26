"""Re-render the two-segment collect-pose thesis figure set against existing runs.

Reads ``metadata.json``, ``summary.json``, ``samples.jsonl`` (and the optional
``sample_failure_events.jsonl``) from each run directory and invokes the
canonical thesis-figure writer in-place. The legacy ``two_segment_*_report.png``
files are left untouched; only the ``thesis_0N`` PNG set is regenerated.

Usage:

    .venv/bin/python tools_local/regenerate_two_segment_collect_pose_thesis_figures.py \\
        data/experiments/two_segment_collect_pose_command_dataset/<run_folder>

Pass ``--latest`` to regenerate the most recent run by mtime, or
``--all`` to walk every run folder under
``data/experiments/two_segment_collect_pose_command_dataset/``.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Iterable

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from continuum_robot.experiments.schemas import ExperimentTimeseriesSample  # noqa: E402
from continuum_robot.experiments.two_segment_modeling_dataset_outputs import (  # noqa: E402
    THESIS_FIGURE_NAMES,
    write_two_segment_thesis_figures,
)


DEFAULT_RUN_ROOT = (
    Path(__file__).resolve().parents[1]
    / "data" / "experiments" / "two_segment_collect_pose_command_dataset"
)


def _load_samples(path: Path) -> list[ExperimentTimeseriesSample]:
    samples: list[ExperimentTimeseriesSample] = []
    with path.open("r", encoding="utf-8") as handle:
        for raw in handle:
            line = raw.strip()
            if not line:
                continue
            payload = json.loads(line)
            samples.append(
                ExperimentTimeseriesSample(
                    monotonic_time_s=float(payload.get("monotonic_time_s", 0.0) or 0.0),
                    wall_time_utc=str(payload.get("wall_time_utc", "")),
                    phase=str(payload.get("phase", "")),
                    step_index=int(payload.get("step_index", 0) or 0),
                    sample_index=int(payload.get("sample_index", 0) or 0),
                    cycle_index=payload.get("cycle_index"),
                    target_index=payload.get("target_index"),
                    revisit_index=payload.get("revisit_index"),
                    approach_index=payload.get("approach_index"),
                    commanded_motor_values=dict(payload.get("commanded_motor_values", {}) or {}),
                    commanded_cable_deltas_cm=list(payload.get("commanded_cable_deltas_cm", []) or []),
                    two_segment_command=dict(payload.get("two_segment_command", {}) or {}),
                    tracker_frame_id=(
                        int(payload["tracker_frame_id"])
                        if payload.get("tracker_frame_id") is not None
                        else None
                    ),
                    tool_ids_seen=list(payload.get("tool_ids_seen", []) or []),
                    transform_validity=dict(payload.get("transform_validity", {}) or {}),
                    pose_in_tracker_frame=dict(payload.get("pose_in_tracker_frame", {}) or {}),
                    pose_in_robot_frame=dict(payload.get("pose_in_robot_frame", {}) or {}),
                    two_segment_pose=dict(payload.get("two_segment_pose", {}) or {}),
                    freshness_s=payload.get("freshness_s"),
                    latency_s=payload.get("latency_s"),
                    status_flags=list(payload.get("status_flags", []) or []),
                    backend_health=dict(payload.get("backend_health", {}) or {}),
                    extra=dict(payload.get("extra", {}) or {}),
                )
            )
    return samples


def _load_failure_events(path: Path) -> list[dict]:
    if not path.exists():
        return []
    events: list[dict] = []
    with path.open("r", encoding="utf-8") as handle:
        for raw in handle:
            line = raw.strip()
            if not line:
                continue
            try:
                events.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return events


def _regenerate(run_dir: Path) -> dict[str, Path]:
    metadata_path = run_dir / "metadata.json"
    summary_path = run_dir / "summary.json"
    samples_path = run_dir / "samples.jsonl"
    if not all(p.exists() for p in (metadata_path, summary_path, samples_path)):
        raise SystemExit(
            f"missing one of metadata.json/summary.json/samples.jsonl in {run_dir}"
        )
    summary_payload = json.loads(summary_path.read_text(encoding="utf-8"))
    metadata_payload = json.loads(metadata_path.read_text(encoding="utf-8"))
    metrics = dict(summary_payload.get("experiment_metrics") or {})
    metrics.setdefault("config_used", dict(metadata_payload.get("config_used") or {}))
    samples = _load_samples(samples_path)
    failure_events = _load_failure_events(run_dir / "sample_failure_events.jsonl")
    return write_two_segment_thesis_figures(
        output_dir=run_dir,
        metrics=metrics,
        samples=samples,
        sample_failure_events=failure_events,
    )


def _resolve_run_dirs(args: argparse.Namespace) -> list[Path]:
    if args.all:
        if not DEFAULT_RUN_ROOT.exists():
            raise SystemExit(f"run root not found: {DEFAULT_RUN_ROOT}")
        return sorted(
            (p for p in DEFAULT_RUN_ROOT.iterdir() if p.is_dir() and (p / "samples.jsonl").exists()),
            key=lambda p: p.name,
        )
    if args.latest:
        if not DEFAULT_RUN_ROOT.exists():
            raise SystemExit(f"run root not found: {DEFAULT_RUN_ROOT}")
        candidates = [p for p in DEFAULT_RUN_ROOT.iterdir() if p.is_dir() and (p / "samples.jsonl").exists()]
        if not candidates:
            raise SystemExit(f"no candidate runs under {DEFAULT_RUN_ROOT}")
        return [max(candidates, key=lambda p: p.stat().st_mtime)]
    if not args.run_dirs:
        raise SystemExit("pass one or more run directories, or --latest / --all")
    return [Path(p) for p in args.run_dirs]


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dirs", nargs="*", type=Path)
    parser.add_argument("--latest", action="store_true", help="regenerate the most recent run by mtime")
    parser.add_argument("--all", action="store_true", help="regenerate every run under the canonical root")
    args = parser.parse_args(list(argv) if argv is not None else None)
    targets = _resolve_run_dirs(args)
    for run_dir in targets:
        print(f"regenerating: {run_dir}")
        outputs = _regenerate(run_dir)
        for key, path in outputs.items():
            marker = "ok" if Path(path).exists() else "missing"
            size_kb = Path(path).stat().st_size / 1024.0 if Path(path).exists() else 0.0
            print(f"  {marker:>7} {key}: {path.name}  ({size_kb:.1f} KB)")
    missing_total = 0
    for run_dir in targets:
        for filename in THESIS_FIGURE_NAMES:
            if not (run_dir / filename).exists():
                print(f"  WARN missing: {run_dir / filename}")
                missing_total += 1
    return 0 if missing_total == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())

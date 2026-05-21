"""Re-run the tracker-timing and servo-tracker-sync figure writers against
existing run directories without re-running the underlying experiments.

Reads metadata.json / summary.json / samples.jsonl from each run directory
and invokes the canonical writer, overwriting the PNGs in place.
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

from continuum_robot.experiments.schemas import (
    ExperimentMetadata,
    ExperimentSummary,
    ExperimentTimeseriesSample,
)
from continuum_robot.experiments.servo_tracker_sync_outputs import (
    write_servo_tracker_sync_outputs,
)
from continuum_robot.experiments.tracker_timing_outputs import (
    write_tracker_timing_outputs,
)


def _load_metadata(path: Path) -> ExperimentMetadata:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return ExperimentMetadata(
        schema_version=str(payload.get("schema_version", "1.0")),
        experiment_name=str(payload.get("experiment_name", "")),
        run_id=str(payload.get("run_id", "")),
        timestamp_utc=str(payload.get("timestamp_utc", "")),
        git_commit=str(payload.get("git_commit", "")),
        backend_info=dict(payload.get("backend_info", {}) or {}),
        registration_info=dict(payload.get("registration_info", {}) or {}),
        config_used=dict(payload.get("config_used", {}) or {}),
        operator_notes=str(payload.get("operator_notes", "") or ""),
        provenance_info=dict(payload.get("provenance_info", {}) or {}),
        trust_info=dict(payload.get("trust_info", {}) or {}),
    )


def _load_summary(path: Path) -> ExperimentSummary:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return ExperimentSummary(
        schema_version=str(payload.get("schema_version", "1.0")),
        experiment_name=str(payload.get("experiment_name", "")),
        run_id=str(payload.get("run_id", "")),
        success=bool(payload.get("success", False)),
        sample_counts=dict(payload.get("sample_counts", {}) or {}),
        dropped_frames=int(payload.get("dropped_frames", 0) or 0),
        invalid_transforms=int(payload.get("invalid_transforms", 0) or 0),
        stage_pass_fail=dict(payload.get("stage_pass_fail", {}) or {}),
        status=str(payload.get("status", "")),
        experiment_metrics=dict(payload.get("experiment_metrics", {}) or {}),
        warning_messages=list(payload.get("warning_messages", []) or []),
        error_messages=list(payload.get("error_messages", []) or []),
    )


def _load_samples(path: Path) -> list[ExperimentTimeseriesSample]:
    samples: list[ExperimentTimeseriesSample] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
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
                    tracker_frame_id=int(payload.get("tracker_frame_id", -1) or -1),
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


def _regenerate(run_dir: Path) -> None:
    metadata = _load_metadata(run_dir / "metadata.json")
    summary = _load_summary(run_dir / "summary.json")
    samples = _load_samples(run_dir / "samples.jsonl")
    if metadata.experiment_name == "tracker_timing_validation":
        outputs = write_tracker_timing_outputs(
            output_dir=run_dir, metadata=metadata, summary=summary, samples=samples,
        )
    elif metadata.experiment_name == "servo_tracker_sync_validation":
        outputs = write_servo_tracker_sync_outputs(
            output_dir=run_dir, metadata=metadata, summary=summary, samples=samples,
        )
    else:
        raise SystemExit(f"unsupported experiment: {metadata.experiment_name}")
    for key, path in outputs.items():
        marker = "ok" if Path(path).exists() else "missing"
        print(f"  {marker:>7} {key}: {path}")


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dirs", nargs="+", type=Path)
    args = parser.parse_args(list(argv) if argv is not None else None)
    for run_dir in args.run_dirs:
        print(f"regenerating: {run_dir}")
        _regenerate(run_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

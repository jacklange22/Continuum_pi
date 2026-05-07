"""Canonical experiment metadata, timeseries, and summary schemas."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class ExperimentMetadata:
    """Metadata stored once per experiment run."""

    schema_version: str
    experiment_name: str
    run_id: str
    timestamp_utc: str
    git_commit: str | None
    backend_info: dict[str, Any]
    registration_info: dict[str, Any]
    config_used: dict[str, Any]
    operator_notes: str = ""
    provenance_info: dict[str, Any] = field(default_factory=dict)
    trust_info: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert metadata into a JSON-serializable dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "ExperimentMetadata":
        """Construct metadata from a dictionary payload."""
        return cls(**payload)


@dataclass
class ExperimentTimeseriesSample:
    """One canonical timeseries sample recorded during a run."""

    monotonic_time_s: float
    wall_time_utc: str
    phase: str
    step_index: int
    sample_index: int
    cycle_index: int | None = None
    target_index: int | None = None
    revisit_index: int | None = None
    approach_index: int | None = None
    commanded_motor_values: dict[str, int | float | None] = field(default_factory=dict)
    commanded_cable_deltas_cm: list[float] = field(default_factory=list)
    two_segment_command: dict[str, Any] = field(default_factory=dict)
    tracker_frame_id: int | None = None
    tool_ids_seen: list[str] = field(default_factory=list)
    transform_validity: dict[str, str] = field(default_factory=dict)
    pose_in_tracker_frame: dict[str, Any] = field(default_factory=dict)
    pose_in_robot_frame: dict[str, Any] = field(default_factory=dict)
    two_segment_pose: dict[str, Any] = field(default_factory=dict)
    freshness_s: float | None = None
    latency_s: float | None = None
    status_flags: list[str] = field(default_factory=list)
    backend_health: dict[str, Any] = field(default_factory=dict)
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert one sample into a JSON-serializable dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "ExperimentTimeseriesSample":
        """Construct one sample from a dictionary payload."""
        return cls(**payload)


@dataclass
class ExperimentSummary:
    """Canonical summary stored once per experiment run."""

    schema_version: str
    experiment_name: str
    run_id: str
    success: bool
    sample_counts: dict[str, int]
    dropped_frames: int
    invalid_transforms: int
    stage_pass_fail: dict[str, str]
    status: str = "success"
    experiment_metrics: dict[str, Any] = field(default_factory=dict)
    warning_messages: list[str] = field(default_factory=list)
    error_messages: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Convert summary into a JSON-serializable dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "ExperimentSummary":
        """Construct summary from a dictionary payload."""
        return cls(**payload)


@dataclass
class ExperimentDatasetPaths:
    """Filesystem paths for one written dataset bundle."""

    output_dir: Path
    metadata_path: Path
    samples_path: Path
    summary_path: Path
    config_snapshot_path: Path | None = None


@dataclass
class ExperimentDatasetBundle:
    """Loaded dataset bundle."""

    metadata: ExperimentMetadata
    samples: list[ExperimentTimeseriesSample]
    summary: ExperimentSummary
    paths: ExperimentDatasetPaths


@dataclass
class ExperimentRunResult:
    """High-level result returned by the canonical experiment runner."""

    experiment_name: str
    run_id: str
    success: bool
    message: str
    paths: ExperimentDatasetPaths
    metadata: ExperimentMetadata
    summary: ExperimentSummary
    sample_count: int

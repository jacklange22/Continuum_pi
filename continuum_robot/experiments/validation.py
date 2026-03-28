"""Dataset validation helpers and summary status classification."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from continuum_robot.experiments.schemas import ExperimentMetadata, ExperimentSummary, ExperimentTimeseriesSample

STATUS_SUCCESS = "success"
STATUS_PARTIAL_SUCCESS = "partial_success"
STATUS_INVALID_MISSING_TIP_CAL = "invalid_due_to_missing_tip_cal"
STATUS_INVALID_MISSING_REGISTRATION = "invalid_due_to_missing_registration"
STATUS_INVALID_INSUFFICIENT_SAMPLES = "invalid_due_to_insufficient_samples"
STATUS_INVALID_INVALID_TRANSFORMS = "invalid_due_to_invalid_transforms"

REQUIRED_FIELDS_BY_EXPERIMENT: dict[str, tuple[str, ...]] = {
    "repeatability_dataset": ("phase", "target_index", "revisit_index", "commanded_cable_deltas_cm", "pose_in_tracker_frame"),
    "aurora_grid_accuracy": ("phase", "target_index", "pose_in_tracker_frame"),
    "pivot_calibration": ("phase", "pose_in_tracker_frame"),
}


@dataclass
class DatasetValidationReport:
    """Validation report for one canonical experiment dataset."""

    experiment_name: str
    missing_fields: list[str] = field(default_factory=list)
    sample_count_consistent: bool = True
    registration_available: bool = False
    tip_calibration_available: bool = False
    status: str = STATUS_SUCCESS


def validate_required_fields(
    experiment_name: str,
    samples: list[ExperimentTimeseriesSample],
) -> list[str]:
    """Return missing required field paths for one experiment type."""
    missing: set[str] = set()
    required = REQUIRED_FIELDS_BY_EXPERIMENT.get(experiment_name, ())
    for field_path in required:
        if not any(_field_present(sample, field_path) for sample in samples):
            missing.add(field_path)
    return sorted(missing)


def sample_count_consistent(
    samples: list[ExperimentTimeseriesSample],
    summary: ExperimentSummary,
) -> bool:
    """Return whether summary sample counts match the raw sample list."""
    return int(summary.sample_counts.get("total", -1)) == len(samples)


def build_dataset_validation_report(
    metadata: ExperimentMetadata,
    samples: list[ExperimentTimeseriesSample],
    summary: ExperimentSummary,
) -> DatasetValidationReport:
    """Build a compact dataset validation report."""
    metrics = summary.experiment_metrics if isinstance(summary.experiment_metrics, dict) else {}
    registration_available = bool(metadata.registration_info.get("exists") or metrics.get("registration_available"))
    tip_calibration_available = bool(metrics.get("tip_calibration_available"))
    report = DatasetValidationReport(
        experiment_name=metadata.experiment_name,
        missing_fields=validate_required_fields(metadata.experiment_name, samples),
        sample_count_consistent=sample_count_consistent(samples, summary),
        registration_available=registration_available,
        tip_calibration_available=tip_calibration_available,
        status=summary.status,
    )
    return report


def classify_summary_status(
    *,
    sample_count: int,
    invalid_transform_count: int,
    min_sample_count: int = 1,
    require_registration: bool = False,
    registration_available: bool = False,
    allow_partial_missing_registration: bool = False,
    require_tip_calibration: bool = False,
    tip_calibration_available: bool = False,
    allow_partial_missing_tip_cal: bool = False,
    invalid_transforms_are_fatal: bool = True,
) -> str:
    """Classify summary status using shared project rules."""
    if sample_count < int(min_sample_count):
        return STATUS_INVALID_INSUFFICIENT_SAMPLES
    if invalid_transforms_are_fatal and invalid_transform_count > 0:
        return STATUS_INVALID_INVALID_TRANSFORMS
    if require_tip_calibration and not tip_calibration_available:
        return STATUS_PARTIAL_SUCCESS if allow_partial_missing_tip_cal else STATUS_INVALID_MISSING_TIP_CAL
    if require_registration and not registration_available:
        return STATUS_PARTIAL_SUCCESS if allow_partial_missing_registration else STATUS_INVALID_MISSING_REGISTRATION
    if (
        (not registration_available and allow_partial_missing_registration)
        or (not tip_calibration_available and allow_partial_missing_tip_cal)
    ):
        return STATUS_PARTIAL_SUCCESS
    return STATUS_SUCCESS


def _field_present(sample: ExperimentTimeseriesSample, field_path: str) -> bool:
    root, *rest = field_path.split(".")
    value: Any = getattr(sample, root, None)
    if not rest:
        return value not in (None, [], {}, "")
    for key in rest:
        if not isinstance(value, dict):
            return False
        value = value.get(key)
    return value not in (None, [], {}, "")

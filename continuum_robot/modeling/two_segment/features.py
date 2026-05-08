"""Feature and label builders for two-segment modeling."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from continuum_robot.modeling.two_segment.dataset import TwoSegmentModelingDataset, TwoSegmentModelingSample


FEATURE_NAMES = [
    "segment_a_servo_1_displacement_mm",
    "segment_a_servo_2_displacement_mm",
    "segment_a_servo_3_displacement_mm",
    "segment_a_servo_4_displacement_mm",
    "segment_b_servo_5_displacement_mm",
    "segment_b_servo_6_displacement_mm",
    "segment_b_servo_7_displacement_mm",
    "segment_b_servo_8_displacement_mm",
]
LABEL_NAMES_XYZ = ["distal_tip_x_mm", "distal_tip_y_mm", "distal_tip_z_mm"]
LABEL_NAMES_TANGENT = ["distal_tip_tx", "distal_tip_ty", "distal_tip_tz"]


@dataclass(frozen=True)
class FeatureLabelBundle:
    """Matrices and metadata used for modeling."""

    samples: list[TwoSegmentModelingSample]
    X: np.ndarray
    y: np.ndarray
    y_position: np.ndarray
    y_tangent: np.ndarray | None
    feature_metadata: dict[str, Any] = field(default_factory=dict)
    label_metadata: dict[str, Any] = field(default_factory=dict)


def build_feature_label_bundle(dataset: TwoSegmentModelingDataset) -> FeatureLabelBundle:
    """Build canonical model matrices from accepted dataset samples."""

    samples = list(dataset.samples)
    X = np.stack([sample.feature_mm for sample in samples], axis=0) if samples else np.zeros((0, 8), dtype=float)
    y_position = np.stack([sample.distal_position_mm for sample in samples], axis=0) if samples else np.zeros((0, 3), dtype=float)
    tangent_available = bool(samples) and all(sample.distal_tangent is not None for sample in samples)
    y_tangent = (
        np.stack([sample.distal_tangent for sample in samples], axis=0)  # type: ignore[arg-type]
        if tangent_available
        else None
    )
    y = np.concatenate([y_position, y_tangent], axis=1) if y_tangent is not None else y_position.copy()
    feature_metadata = {
        "schema_version": "two_segment_feature_metadata_v1",
        "feature_names": list(FEATURE_NAMES),
        "feature_units": ["mm"] * len(FEATURE_NAMES),
        "canonical_feature_order": "Segment A servos [1,2,3,4], then Segment B servos [5,6,7,8]",
        "segment_grouping": {
            "segment_a": {"indices": [0, 1, 2, 3], "servo_ids": [1, 2, 3, 4], "role": "proximal"},
            "segment_b": {"indices": [4, 5, 6, 7], "servo_ids": [5, 6, 7, 8], "role": "distal"},
        },
        "normalization_statistics": normalization_stats(X, names=FEATURE_NAMES),
        "sample_count": len(samples),
        "startup_artifacts": sorted({sample.startup_artifact_path for sample in samples if sample.startup_artifact_path}),
        "command_schema_versions": sorted({sample.command_schema_version for sample in samples if sample.command_schema_version}),
    }
    label_metadata = {
        "schema_version": "two_segment_label_metadata_v1",
        "primary_labels": list(LABEL_NAMES_XYZ),
        "primary_units": ["mm", "mm", "mm"],
        "primary_frame": "robot",
        "orientation_available": bool(y_tangent is not None),
        "orientation_labels": list(LABEL_NAMES_TANGENT) if y_tangent is not None else [],
        "orientation_note": "Explicit tangent labels only; missing tangent labels are not fabricated.",
        "includes_intermediate_pose": bool(dataset.includes_intermediate_pose),
        "distal_only": bool(dataset.distal_only),
        "normalization_statistics": normalization_stats(y, names=list(LABEL_NAMES_XYZ) + (list(LABEL_NAMES_TANGENT) if y_tangent is not None else [])),
    }
    return FeatureLabelBundle(
        samples=samples,
        X=X,
        y=y,
        y_position=y_position,
        y_tangent=y_tangent,
        feature_metadata=feature_metadata,
        label_metadata=label_metadata,
    )


def normalization_stats(matrix: np.ndarray, *, names: list[str]) -> dict[str, Any]:
    if matrix.size == 0:
        return {"mean": {}, "std": {}, "min": {}, "max": {}}
    mean = np.mean(matrix, axis=0)
    std = np.std(matrix, axis=0)
    minimum = np.min(matrix, axis=0)
    maximum = np.max(matrix, axis=0)
    return {
        "mean": {name: float(value) for name, value in zip(names, mean)},
        "std": {name: float(value) for name, value in zip(names, std)},
        "min": {name: float(value) for name, value in zip(names, minimum)},
        "max": {name: float(value) for name, value in zip(names, maximum)},
    }


def standardize_train_test(train: np.ndarray, test: np.ndarray) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Standardize by train-set statistics and return serializable metadata."""

    mean = np.mean(train, axis=0) if train.size else np.zeros((train.shape[1],), dtype=float)
    std = np.std(train, axis=0) if train.size else np.ones((train.shape[1],), dtype=float)
    std = np.where(std == 0.0, 1.0, std)
    return (
        (train - mean) / std,
        (test - mean) / std,
        {"mean": mean.tolist(), "std": std.tolist()},
    )

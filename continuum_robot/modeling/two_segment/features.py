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
SERVO_POSITION_FEATURE_NAMES = [f"servo_{i}_present_position_tick" for i in range(1, 9)]
LABEL_MODES = {"auto", "distal_xyz", "distal_pose6", "two_coil_xyz", "two_coil_pose12"}

# Feature source selects what fills the model input matrix X.
FEATURE_SOURCE_COMMANDED = "commanded_tendon_displacement_mm"
FEATURE_SOURCE_SERVO_POSITION = "measured_servo_position_ticks"
SUPPORTED_FEATURE_SOURCES = (FEATURE_SOURCE_COMMANDED, FEATURE_SOURCE_SERVO_POSITION)


class LabelBuildError(ValueError):
    """Raised when a requested label mode is not present in the dataset."""


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


def build_feature_label_bundle(
    dataset: TwoSegmentModelingDataset,
    *,
    label_mode: str = "auto",
    include_orientation_if_available: bool = False,
    feature_source: str = FEATURE_SOURCE_COMMANDED,
) -> FeatureLabelBundle:
    """Build canonical model matrices from accepted dataset samples.

    ``feature_source`` selects the model input X:
    - ``commanded_tendon_displacement_mm`` (default): the 8-tendon commanded
      displacement in mm (legacy behaviour, preserves all existing runs).
    - ``measured_servo_position_ticks``: the 8 measured present servo
      position ticks (servo-ID order 1..8). This is the "8 servo positions
      -> tip XYZ" forward model. Samples that lack a complete measured
      8-vector are dropped from the bundle (and reported in
      ``feature_metadata.dropped_for_missing_servo_positions``).
    """
    feature_source = str(feature_source or FEATURE_SOURCE_COMMANDED)
    if feature_source not in SUPPORTED_FEATURE_SOURCES:
        raise LabelBuildError(
            f"Unsupported feature_source {feature_source!r}; expected one of {SUPPORTED_FEATURE_SOURCES}."
        )

    all_samples = list(dataset.samples)
    dropped_for_missing_servo_positions = 0
    if feature_source == FEATURE_SOURCE_SERVO_POSITION:
        samples = [s for s in all_samples if getattr(s, "servo_position_ticks", None) is not None]
        dropped_for_missing_servo_positions = len(all_samples) - len(samples)
        feature_names = list(SERVO_POSITION_FEATURE_NAMES)
        feature_units = ["ticks"] * len(feature_names)
        X = (
            np.stack([np.asarray(s.servo_position_ticks, dtype=float) for s in samples], axis=0)
            if samples
            else np.zeros((0, 8), dtype=float)
        )
    else:
        samples = all_samples
        feature_names = list(FEATURE_NAMES)
        feature_units = ["mm"] * len(feature_names)
        X = np.stack([sample.feature_mm for sample in samples], axis=0) if samples else np.zeros((0, 8), dtype=float)

    resolved_mode = resolve_label_mode(
        samples,
        requested=label_mode,
        include_orientation_if_available=bool(include_orientation_if_available),
    )
    y, y_position, y_tangent, label_metadata = _build_labels(samples, label_mode=resolved_mode)
    feature_metadata = {
        "schema_version": "two_segment_feature_metadata_v2",
        "feature_source": feature_source,
        "feature_names": list(feature_names),
        "feature_units": list(feature_units),
        "input_units": ("servo_position_ticks" if feature_source == FEATURE_SOURCE_SERVO_POSITION else "tendon_displacement_mm"),
        "model_input_description": (
            "8 measured present servo position ticks (servo IDs 1..8) -> distal tip XYZ"
            if feature_source == FEATURE_SOURCE_SERVO_POSITION
            else "8 commanded tendon displacements (mm) -> distal tip XYZ"
        ),
        "tendon_displacement_positive_direction": "shortening",
        "encoder_tick_direction_for_shortening": "decreasing_ticks",
        "sign_convention_note": "Positive tendon displacement means tendon shortening / more tension; shortening is expected to decrease encoder ticks.",
        "canonical_feature_order": "Segment A entries followed by Segment B entries unless config metadata declares a different segment order.",
        "segment_order": _merged_segment_order(samples),
        "segment_grouping": _segment_grouping(samples),
        "normalization_statistics": normalization_stats(X, names=feature_names),
        "sample_count": len(samples),
        "dropped_for_missing_servo_positions": int(dropped_for_missing_servo_positions),
        "startup_artifacts": sorted({sample.startup_artifact_path for sample in samples if sample.startup_artifact_path}),
        "command_schema_versions": sorted({sample.command_schema_version for sample in samples if sample.command_schema_version}),
    }
    label_metadata["normalization_statistics"] = normalization_stats(y, names=list(label_metadata["label_names"]))
    return FeatureLabelBundle(
        samples=samples,
        X=X,
        y=y,
        y_position=y_position,
        y_tangent=y_tangent,
        feature_metadata=feature_metadata,
        label_metadata=label_metadata,
    )


def resolve_label_mode(
    samples: list[TwoSegmentModelingSample],
    *,
    requested: str = "auto",
    include_orientation_if_available: bool = False,
) -> str:
    mode = str(requested or "auto").strip().lower()
    if mode not in LABEL_MODES:
        raise LabelBuildError(f"Unsupported two-segment label mode: {requested}")
    if mode != "auto":
        _assert_mode_available(samples, mode)
        return mode
    has_intermediate = bool(samples) and all(sample.intermediate_position_mm is not None for sample in samples)
    has_distal_tangent = bool(samples) and all(sample.distal_tangent is not None for sample in samples)
    has_intermediate_tangent = bool(samples) and all(sample.intermediate_tangent is not None for sample in samples)
    if include_orientation_if_available and has_intermediate and has_distal_tangent and has_intermediate_tangent:
        return "two_coil_pose12"
    if include_orientation_if_available and not has_intermediate and has_distal_tangent:
        return "distal_pose6"
    return "two_coil_xyz" if has_intermediate else "distal_xyz"


def _assert_mode_available(samples: list[TwoSegmentModelingSample], mode: str) -> None:
    if mode in {"distal_xyz", "distal_pose6", "two_coil_xyz", "two_coil_pose12"} and not samples:
        raise LabelBuildError("No accepted samples are available for label extraction.")
    if mode in {"distal_pose6", "two_coil_pose12"} and not all(sample.distal_tangent is not None for sample in samples):
        raise LabelBuildError(f"{mode} requires explicit distal_tip tangent/orientation labels.")
    if mode in {"two_coil_xyz", "two_coil_pose12"} and not all(sample.intermediate_position_mm is not None for sample in samples):
        raise LabelBuildError(f"{mode} requires explicit intermediate_segment XYZ labels in robot frame.")
    if mode == "two_coil_pose12" and not all(sample.intermediate_tangent is not None for sample in samples):
        raise LabelBuildError("two_coil_pose12 requires explicit intermediate_segment tangent/orientation labels.")


def _build_labels(
    samples: list[TwoSegmentModelingSample],
    *,
    label_mode: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None, dict[str, Any]]:
    _assert_mode_available(samples, label_mode)
    arrays: list[np.ndarray] = []
    names: list[str] = []
    units: list[str] = []
    slices: dict[str, dict[str, list[int]]] = {}

    def add_xyz(role: str, values: list[np.ndarray]) -> None:
        start = sum(array.shape[1] for array in arrays)
        arrays.append(np.stack(values, axis=0))
        names.extend([f"{role}_x_mm", f"{role}_y_mm", f"{role}_z_mm"])
        units.extend(["mm", "mm", "mm"])
        slices.setdefault(role, {})["position"] = [start, start + 1, start + 2]

    def add_tangent(role: str, values: list[np.ndarray]) -> None:
        start = sum(array.shape[1] for array in arrays)
        arrays.append(np.stack(values, axis=0))
        names.extend([f"{role}_tx", f"{role}_ty", f"{role}_tz"])
        units.extend(["unit_vector", "unit_vector", "unit_vector"])
        slices.setdefault(role, {})["orientation"] = [start, start + 1, start + 2]

    if label_mode in {"two_coil_xyz", "two_coil_pose12"}:
        add_xyz("intermediate_segment", [sample.intermediate_position_mm for sample in samples if sample.intermediate_position_mm is not None])
        if label_mode == "two_coil_pose12":
            add_tangent("intermediate_segment", [sample.intermediate_tangent for sample in samples if sample.intermediate_tangent is not None])
    add_xyz("distal_tip", [sample.distal_position_mm for sample in samples])
    if label_mode in {"distal_pose6", "two_coil_pose12"}:
        add_tangent("distal_tip", [sample.distal_tangent for sample in samples if sample.distal_tangent is not None])
    y = np.concatenate(arrays, axis=1) if arrays else np.zeros((0, 0), dtype=float)
    distal_slice = slices["distal_tip"]["position"]
    y_position = y[:, distal_slice]
    y_tangent = y[:, slices["distal_tip"]["orientation"]] if "orientation" in slices["distal_tip"] else None
    metadata = {
        "schema_version": "two_segment_label_metadata_v2",
        "label_mode": label_mode,
        "label_names": names,
        "label_units": units,
        "label_frame": "robot",
        "label_slices": slices,
        "primary_role": "distal_tip",
        "primary_labels": [names[index] for index in distal_slice],
        "primary_units": ["mm", "mm", "mm"],
        "primary_frame": "robot",
        "orientation_available": any("orientation" in role_slices for role_slices in slices.values()),
        "orientation_available_by_role": {role: "orientation" in role_slices for role, role_slices in slices.items()},
        "orientation_note": "Explicit tangent labels only; missing tangent labels are not fabricated.",
        "includes_intermediate_label": "intermediate_segment" in slices,
        "includes_intermediate_pose": "intermediate_segment" in slices,
        "distal_only": "intermediate_segment" not in slices,
        "direct_global_model": True,
        "structured_composition_model": False,
        "composition_note": (
            "Segment B rides on Segment A. This scaffold fits a direct global forward model "
            "[Segment A tendon displacement, Segment B tendon displacement] -> robot-frame pose labels."
        ),
    }
    return y, y_position, y_tangent, metadata


def _merged_segment_order(samples: list[TwoSegmentModelingSample]) -> list[str]:
    if not samples:
        return ["segment_a", "segment_b"]
    return list(samples[0].segment_order or ["segment_a", "segment_b"])


def _segment_grouping(samples: list[TwoSegmentModelingSample]) -> dict[str, Any]:
    if not samples:
        metadata = {
            "segment_a": {"indices": [0, 1, 2, 3], "servo_ids": [1, 2, 3, 4], "role": "proximal"},
            "segment_b": {"indices": [4, 5, 6, 7], "servo_ids": [5, 6, 7, 8], "role": "distal"},
        }
    else:
        sample = samples[0]
        metadata = {}
        offset = 0
        for segment in sample.segment_order:
            payload = dict(sample.segment_metadata.get(segment, {}) or {})
            servo_ids = [int(value) for value in list(payload.get("servo_ids") or [])]
            width = len(servo_ids) or 4
            metadata[segment] = {
                "indices": list(range(offset, offset + width)),
                "servo_ids": servo_ids,
                "role": payload.get("role") or payload.get("segment_role") or "",
                "role_aliases": list(payload.get("role_aliases") or []),
                "label": payload.get("label") or payload.get("segment_label") or segment,
                "pair_mapping": payload.get("pair_mapping") or [],
                "segment_order_index": int(payload.get("segment_order_index", offset // max(1, width))),
            }
            offset += width
    role_map = {segment: str(payload.get("role") or "").lower() for segment, payload in metadata.items()}
    metadata["lower_segment"] = _role_segment(metadata, role_map, preferred={"proximal", "lower", "base", "bottom"}, fallback_index=0)
    metadata["upper_segment"] = _role_segment(metadata, role_map, preferred={"distal", "upper", "top"}, fallback_index=1)
    return metadata


def _role_segment(
    metadata: dict[str, Any],
    role_map: dict[str, str],
    *,
    preferred: set[str],
    fallback_index: int,
) -> dict[str, Any]:
    for segment, role in role_map.items():
        if role in preferred:
            return {"segment": segment, **dict(metadata.get(segment, {}) or {})}
    ordered = sorted(
        [(segment, payload) for segment, payload in metadata.items() if segment.startswith("segment_")],
        key=lambda item: int(dict(item[1]).get("segment_order_index", 999)),
    )
    if len(ordered) > fallback_index:
        segment, payload = ordered[fallback_index]
        return {"segment": segment, **dict(payload)}
    return {}


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

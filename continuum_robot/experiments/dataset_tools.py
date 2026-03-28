"""Helpers for extracting transforms and positions from canonical datasets."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from continuum_robot.experiments.dataset_io import ExperimentDatasetLoader
from continuum_robot.experiments.schemas import ExperimentTimeseriesSample
from continuum_robot.tracking.transforms import make_transform_A_B


def extract_tip_or_tool_position_mm(
    sample: ExperimentTimeseriesSample,
    *,
    tool_id: str = "0A",
    prefer_robot_frame: bool = True,
) -> tuple[list[float] | None, str | None]:
    """Extract one position vector from a sample."""
    if prefer_robot_frame:
        tip = sample.pose_in_robot_frame.get("tip", {})
        translation = tip.get("translation_mm")
        if isinstance(translation, list) and len(translation) == 3:
            return [float(value) for value in translation], "robot"
    tool_payload = sample.pose_in_tracker_frame.get(tool_id, {})
    translation = tool_payload.get("translation_mm")
    if isinstance(translation, list) and len(translation) == 3:
        return [float(value) for value in translation], "tracker"
    return None, None


def extract_tool_transform_matrix(
    sample: ExperimentTimeseriesSample,
    *,
    tool_id: str,
) -> np.ndarray | None:
    """Extract one tracker-frame transform matrix for a tool sample."""
    payload = sample.pose_in_tracker_frame.get(tool_id, {})
    quaternion = payload.get("quaternion_wxyz")
    translation = payload.get("translation_mm")
    if not (isinstance(quaternion, list) and len(quaternion) == 4):
        return None
    if not (isinstance(translation, list) and len(translation) == 3):
        return None
    return make_transform_A_B(
        tuple(float(value) for value in quaternion),
        tuple(float(value) for value in translation),
    )


def load_dataset_samples(path: Path) -> list[ExperimentTimeseriesSample]:
    """Load canonical experiment samples from a run directory or file path."""
    bundle = ExperimentDatasetLoader().load_dataset(Path(path))
    return list(bundle.samples)


def extract_tool_transforms_from_dataset(
    path: Path,
    *,
    tool_id: str,
) -> list[np.ndarray]:
    """Extract tool transforms from a canonical dataset."""
    transforms: list[np.ndarray] = []
    for sample in load_dataset_samples(Path(path)):
        matrix = extract_tool_transform_matrix(sample, tool_id=tool_id)
        if matrix is not None:
            transforms.append(matrix)
    return transforms

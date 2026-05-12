"""Offline two-segment modeling scaffold."""

from continuum_robot.modeling.two_segment.dataset import (
    TwoSegmentModelingDataset,
    TwoSegmentModelingSample,
    load_two_segment_modeling_dataset,
)
from continuum_robot.modeling.two_segment.features import FeatureLabelBundle, LabelBuildError, build_feature_label_bundle
from continuum_robot.modeling.two_segment.train import (
    TwoSegmentModelingConfig,
    TwoSegmentModelingResult,
    build_train_test_split,
    run_two_segment_modeling,
)

__all__ = [
    "FeatureLabelBundle",
    "LabelBuildError",
    "TwoSegmentModelingConfig",
    "TwoSegmentModelingDataset",
    "TwoSegmentModelingResult",
    "TwoSegmentModelingSample",
    "build_feature_label_bundle",
    "build_train_test_split",
    "load_two_segment_modeling_dataset",
    "run_two_segment_modeling",
]

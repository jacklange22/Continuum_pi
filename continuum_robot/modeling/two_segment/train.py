"""Offline training/evaluation runner for two-segment modeling."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from continuum_robot.modeling.two_segment.dataset import load_two_segment_modeling_dataset
from continuum_robot.modeling.two_segment.evaluate import position_metrics
from continuum_robot.modeling.two_segment.features import build_feature_label_bundle
from continuum_robot.modeling.two_segment.models import default_model_suite
from continuum_robot.modeling.two_segment.outputs import allocate_output_dir, write_output_bundle


@dataclass
class TwoSegmentModelingConfig:
    """Configuration for offline two-segment modeling analysis."""

    allow_lower_trust: bool = False
    test_fraction: float = 0.25
    random_seed: int = 0
    by_run_split: bool | None = None
    model_keys: list[str] = field(default_factory=lambda: ["linear_baseline", "camarillo", "mike_constant_curvature", "ann"])
    model_config: dict[str, Any] = field(default_factory=dict)
    output_root: str | None = None
    figure_quality: str = "production"
    output_dir: str | None = None


@dataclass(frozen=True)
class TwoSegmentModelingResult:
    """Completed modeling scaffold output."""

    output_dir: Path
    dataset: Any
    bundle: Any
    split: dict[str, Any]
    model_results: list[Any]
    output_paths: dict[str, Path]


def run_two_segment_modeling(
    *,
    run_dirs: list[Path],
    project_root: Path,
    config: TwoSegmentModelingConfig | None = None,
) -> TwoSegmentModelingResult:
    """Run offline two-segment model fitting/evaluation on saved dataset runs."""

    config = config or TwoSegmentModelingConfig()
    dataset = load_two_segment_modeling_dataset(run_dirs, allow_lower_trust=bool(config.allow_lower_trust))
    if dataset.accepted_count < 2:
        raise ValueError(
            "Two-segment modeling requires at least 2 accepted labeled samples. "
            f"Accepted={dataset.accepted_count}, rejected={dataset.rejected_count}, reasons={dataset.rejection_counts()}."
        )
    bundle = build_feature_label_bundle(dataset)
    split = build_train_test_split(
        samples=bundle.samples,
        test_fraction=float(config.test_fraction),
        random_seed=int(config.random_seed),
        by_run_split=config.by_run_split,
    )
    X_train = bundle.X[split["train_indices"], :]
    y_train = bundle.y[split["train_indices"], :]
    X_test = bundle.X[split["test_indices"], :]
    y_test = bundle.y[split["test_indices"], :]
    output_dir = allocate_output_dir(
        project_root=Path(project_root),
        output_root=Path(config.output_root) if config.output_root else None,
        output_dir=Path(config.output_dir) if config.output_dir else None,
    )
    config.output_dir = str(output_dir)
    model_results = []
    predictions_rows: list[dict[str, Any]] = []
    for model in default_model_suite(model_keys=list(config.model_keys), config={**dict(config.model_config or {}), "random_seed": int(config.random_seed)}):
        model_dir = output_dir / "models" / model.model_key
        result = model.fit_predict(
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
            y_test=y_test,
            model_dir=model_dir,
        )
        model_results.append(result)
        if result.predictions is not None:
            predictions_rows.extend(
                _prediction_rows(
                    model_key=result.model_key,
                    status=result.status,
                    samples=[bundle.samples[index] for index in split["test_indices"]],
                    y_true=y_test,
                    y_pred=result.predictions,
                )
            )
    output_paths = write_output_bundle(
        output_dir=output_dir,
        config=config,
        dataset=dataset,
        bundle=bundle,
        split=split,
        model_results=model_results,
        predictions_rows=predictions_rows,
        plot_quality=str(config.figure_quality),
    )
    return TwoSegmentModelingResult(
        output_dir=output_dir,
        dataset=dataset,
        bundle=bundle,
        split=split,
        model_results=model_results,
        output_paths=output_paths,
    )


def build_train_test_split(
    *,
    samples: list[Any],
    test_fraction: float = 0.25,
    random_seed: int = 0,
    by_run_split: bool | None = None,
) -> dict[str, Any]:
    """Return reproducible train/test indices, preferring by-run when useful."""

    n = len(samples)
    if n < 2:
        raise ValueError("At least 2 samples are required for train/test split.")
    run_names = [str(sample.run_dir) for sample in samples]
    unique_runs = sorted(set(run_names))
    use_by_run = bool(by_run_split) or (by_run_split is None and len(unique_runs) > 1)
    rng = np.random.default_rng(int(random_seed))
    if use_by_run and len(unique_runs) > 1:
        shuffled_runs = list(unique_runs)
        rng.shuffle(shuffled_runs)
        test_run_count = max(1, min(len(shuffled_runs) - 1, int(round(len(shuffled_runs) * float(test_fraction)))))
        test_runs = set(shuffled_runs[:test_run_count])
        test_indices = [index for index, run in enumerate(run_names) if run in test_runs]
        train_indices = [index for index, run in enumerate(run_names) if run not in test_runs]
        method = "by_run"
        note = "By-run split used; this is preferred for generalization when multiple real runs are available."
    else:
        indices = np.arange(n)
        rng.shuffle(indices)
        test_count = max(1, min(n - 1, int(round(n * float(test_fraction)))))
        test_indices = sorted(int(value) for value in indices[:test_count])
        train_indices = sorted(int(value) for value in indices[test_count:])
        method = "single_run_random"
        note = "Single-run random split can overestimate generalization; use by-run split when multiple real runs are available."
    if not train_indices or not test_indices:
        raise ValueError("Train/test split produced an empty partition.")
    return {
        "schema_version": "two_segment_train_test_split_v1",
        "method": method,
        "note": note,
        "test_fraction": float(test_fraction),
        "random_seed": int(random_seed),
        "train_indices": train_indices,
        "test_indices": test_indices,
        "run_names": run_names,
        "test_runs": sorted({run_names[index] for index in test_indices}),
    }


def _prediction_rows(*, model_key: str, status: str, samples: list[Any], y_true: np.ndarray, y_pred: np.ndarray) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    errors = np.linalg.norm(y_pred[:, :3] - y_true[:, :3], axis=1)
    for index, sample in enumerate(samples):
        row = {
            "model_key": model_key,
            "status": status,
            "run_name": sample.run_name,
            "source_sample_index": sample.sample_index,
            "measured_x_mm": float(y_true[index, 0]),
            "measured_y_mm": float(y_true[index, 1]),
            "measured_z_mm": float(y_true[index, 2]),
            "predicted_x_mm": float(y_pred[index, 0]),
            "predicted_y_mm": float(y_pred[index, 1]),
            "predicted_z_mm": float(y_pred[index, 2]),
            "position_error_mm": float(errors[index]),
            "segment_a_d1_mm": float(sample.feature_mm[0]),
            "segment_a_d2_mm": float(sample.feature_mm[1]),
            "segment_a_d3_mm": float(sample.feature_mm[2]),
            "segment_a_d4_mm": float(sample.feature_mm[3]),
            "segment_b_d5_mm": float(sample.feature_mm[4]),
            "segment_b_d6_mm": float(sample.feature_mm[5]),
            "segment_b_d7_mm": float(sample.feature_mm[6]),
            "segment_b_d8_mm": float(sample.feature_mm[7]),
        }
        if y_true.shape[1] >= 6 and y_pred.shape[1] >= 6:
            row.update(
                {
                    "measured_tx": float(y_true[index, 3]),
                    "measured_ty": float(y_true[index, 4]),
                    "measured_tz": float(y_true[index, 5]),
                    "predicted_tx": float(y_pred[index, 3]),
                    "predicted_ty": float(y_pred[index, 4]),
                    "predicted_tz": float(y_pred[index, 5]),
                }
            )
        rows.append(row)
    _ = position_metrics
    return rows

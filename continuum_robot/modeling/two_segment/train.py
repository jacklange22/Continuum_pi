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
    """Configuration for offline two-segment modeling analysis.

    The three-fold split defaults are ``train 0.60 / val 0.20 / test 0.20``.
    ``test_fraction`` is kept as the legacy knob; ``val_fraction`` was added
    when the training path moved from "test-as-validation" to a true 3-fold
    contract. The validation fold drives per-architecture early stopping and
    cross-architecture model selection; the test fold is the FINAL held-out
    accuracy report and is never touched by training decisions.
    """

    allow_lower_trust: bool = False
    val_fraction: float = 0.20
    test_fraction: float = 0.20
    random_seed: int = 0
    by_run_split: bool | None = None
    model_keys: list[str] = field(default_factory=lambda: ["linear_baseline", "camarillo", "mike_constant_curvature", "ann"])
    model_config: dict[str, Any] = field(default_factory=dict)
    output_root: str | None = None
    figure_quality: str = "production"
    output_dir: str | None = None
    label_mode: str = "auto"
    include_orientation_if_available: bool = False


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
    bundle = build_feature_label_bundle(
        dataset,
        label_mode=str(config.label_mode),
        include_orientation_if_available=bool(config.include_orientation_if_available),
    )
    split = build_train_val_test_split(
        samples=bundle.samples,
        val_fraction=float(config.val_fraction),
        test_fraction=float(config.test_fraction),
        random_seed=int(config.random_seed),
        by_run_split=config.by_run_split,
    )
    X_train = bundle.X[split["train_indices"], :]
    y_train = bundle.y[split["train_indices"], :]
    X_val = bundle.X[split["val_indices"], :]
    y_val = bundle.y[split["val_indices"], :]
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
            X_val=X_val,
            y_val=y_val,
            X_test=X_test,
            y_test=y_test,
            model_dir=model_dir,
            label_metadata=bundle.label_metadata,
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
                    label_metadata=bundle.label_metadata,
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


def build_train_val_test_split(
    *,
    samples: list[Any],
    val_fraction: float = 0.20,
    test_fraction: float = 0.20,
    random_seed: int = 0,
    by_run_split: bool | None = None,
) -> dict[str, Any]:
    """Return reproducible train / val / test indices.

    The three folds are disjoint and exhaustive: every sample appears in
    exactly one of ``train_indices``, ``val_indices``, ``test_indices``. The
    validation fold drives early stopping and architecture/seed selection;
    the test fold is reserved for the FINAL accuracy report on the selected
    model and must NOT influence training decisions.

    By-run mode (preferred when multiple runs are available):
    - When ``len(unique_runs) >= 4`` the three folds are taken as whole runs
      so generalization across runs is honest. The split sizes are computed
      from ``val_fraction`` and ``test_fraction`` with at least one run per
      fold (train, val, test) and the remainder going to train.
    - When ``2 <= len(unique_runs) <= 3`` we can only split runs cleanly
      into train + test (or train + val); to avoid degenerate one-sample
      splits the function falls back to **random-within-the-pooled-data**
      with a logged warning.
    - When ``len(unique_runs) == 1`` we shuffle row indices and take three
      contiguous random fractions, again with a logged warning because
      same-run leakage cannot be ruled out from the data alone.

    Raises ``ValueError`` for non-positive or excessive fractions, for
    fewer than three samples, or if any of the three folds would end up
    empty.
    """

    if not (val_fraction > 0.0):
        raise ValueError("val_fraction must be positive")
    if not (test_fraction > 0.0):
        raise ValueError("test_fraction must be positive")
    if val_fraction + test_fraction >= 1.0:
        raise ValueError("val_fraction + test_fraction must be strictly less than 1.0")

    n = len(samples)
    if n < 3:
        raise ValueError("At least 3 samples are required for a train/val/test split.")
    run_names = [str(sample.run_dir) for sample in samples]
    unique_runs = sorted(set(run_names))
    rng = np.random.default_rng(int(random_seed))
    warnings: list[str] = []

    want_by_run = bool(by_run_split) if by_run_split is not None else True
    can_by_run_three_folds = want_by_run and len(unique_runs) >= 4
    can_by_run_two_folds = want_by_run and 2 <= len(unique_runs) <= 3

    if can_by_run_three_folds:
        shuffled_runs = list(unique_runs)
        rng.shuffle(shuffled_runs)
        n_runs = len(shuffled_runs)
        test_run_count = max(1, int(round(n_runs * float(test_fraction))))
        val_run_count = max(1, int(round(n_runs * float(val_fraction))))
        # Ensure at least one run for train.
        while test_run_count + val_run_count >= n_runs:
            if val_run_count > 1:
                val_run_count -= 1
            elif test_run_count > 1:
                test_run_count -= 1
            else:
                # n_runs must be >= 4 here; this branch only fires if the
                # fractions are pathologically large for the run count.
                raise ValueError("By-run split cannot leave at least one training run after splitting.")
        test_runs = set(shuffled_runs[:test_run_count])
        val_runs = set(shuffled_runs[test_run_count : test_run_count + val_run_count])
        test_indices = sorted(index for index, run in enumerate(run_names) if run in test_runs)
        val_indices = sorted(index for index, run in enumerate(run_names) if run in val_runs)
        train_indices = sorted(
            index
            for index, run in enumerate(run_names)
            if run not in test_runs and run not in val_runs
        )
        method = "by_run"
        note = "By-run train/val/test split used; this is preferred for generalization when several real runs are available."
    else:
        if can_by_run_two_folds:
            warnings.append(
                f"by_run split requested but only {len(unique_runs)} unique runs available; "
                "falling back to random pooled split with same-run leakage risk."
            )
        elif want_by_run and len(unique_runs) == 1:
            warnings.append(
                "Only one unique run present; random pooled split cannot rule out same-run leakage."
            )
        indices = np.arange(n)
        rng.shuffle(indices)
        test_count = max(1, int(round(n * float(test_fraction))))
        val_count = max(1, int(round(n * float(val_fraction))))
        while test_count + val_count >= n:
            if val_count > 1:
                val_count -= 1
            elif test_count > 1:
                test_count -= 1
            else:
                raise ValueError("Random split cannot leave at least one training sample after splitting.")
        test_indices = sorted(int(value) for value in indices[:test_count])
        val_indices = sorted(int(value) for value in indices[test_count : test_count + val_count])
        train_indices = sorted(int(value) for value in indices[test_count + val_count :])
        method = "random_pooled"
        note = (
            "Random pooled split used; consider collecting multiple runs and re-running for a "
            "by-run split that supports a stronger generalization claim."
        )

    if not train_indices or not val_indices or not test_indices:
        raise ValueError("Train/val/test split produced an empty partition.")
    if set(train_indices) & set(val_indices) or set(train_indices) & set(test_indices) or set(val_indices) & set(test_indices):
        raise AssertionError("Train/val/test split produced overlapping folds.")

    return {
        "schema_version": "two_segment_train_val_test_split_v1",
        "method": method,
        "note": note,
        "val_fraction": float(val_fraction),
        "test_fraction": float(test_fraction),
        "random_seed": int(random_seed),
        "train_indices": train_indices,
        "val_indices": val_indices,
        "test_indices": test_indices,
        "run_names": run_names,
        "val_runs": sorted({run_names[index] for index in val_indices}),
        "test_runs": sorted({run_names[index] for index in test_indices}),
        "warnings": list(warnings),
    }


def build_train_test_split(
    *,
    samples: list[Any],
    test_fraction: float = 0.25,
    random_seed: int = 0,
    by_run_split: bool | None = None,
) -> dict[str, Any]:
    """Deprecated -- kept only so external callers reading legacy split manifests still work.

    The thesis-grade training path is :func:`build_train_val_test_split`,
    which produces a third (validation) fold so the test split can be a
    true held-out report. This function will be removed once all callers
    migrate.
    """
    val_fraction = max(0.05, float(test_fraction) * 0.6)
    result = build_train_val_test_split(
        samples=samples,
        val_fraction=val_fraction,
        test_fraction=float(test_fraction),
        random_seed=random_seed,
        by_run_split=by_run_split,
    )
    # Project to legacy two-fold shape: fold val_indices into train_indices so
    # the legacy caller sees the same effective contract (everything that isn't
    # test is "train"), but the underlying split still respects the new policy.
    train_plus_val = sorted(set(result["train_indices"]) | set(result["val_indices"]))
    return {
        "schema_version": "two_segment_train_test_split_v1",
        "method": result["method"],
        "note": (
            result["note"]
            + " (legacy build_train_test_split caller; val fold folded into train for compatibility)"
        ),
        "test_fraction": float(test_fraction),
        "random_seed": int(random_seed),
        "train_indices": train_plus_val,
        "test_indices": list(result["test_indices"]),
        "run_names": list(result["run_names"]),
        "test_runs": list(result["test_runs"]),
    }


def _prediction_rows(
    *,
    model_key: str,
    status: str,
    samples: list[Any],
    y_true: np.ndarray,
    y_pred: np.ndarray,
    label_metadata: dict[str, Any],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    slices = label_metadata.get("label_slices") if isinstance(label_metadata.get("label_slices"), dict) else {}
    distal_slice = _slice_for(slices, "distal_tip", "position") or [0, 1, 2]
    errors = np.linalg.norm(y_pred[:, distal_slice] - y_true[:, distal_slice], axis=1)
    for index, sample in enumerate(samples):
        # Bottom/top command magnitudes: physical-segment role aware. The
        # 8-vector is always in commanded_servo_ids order (servos 1..4 →
        # segment_a, 5..8 → segment_b). The role assignment determines which
        # half maps to bottom vs top. Defaults assume A=bottom / B=top.
        bottom_key, top_key = _resolve_bottom_top_keys(sample)
        if bottom_key == "segment_b":
            bottom_features = sample.feature_mm[4:]
            top_features = sample.feature_mm[:4]
        else:
            bottom_features = sample.feature_mm[:4]
            top_features = sample.feature_mm[4:]
        bottom_magnitude = float(np.linalg.norm(bottom_features))
        top_magnitude = float(np.linalg.norm(top_features))
        # A small dead-band so floating-point near-zeros don't toggle pattern.
        dead_band = 1e-6
        bottom_active = bottom_magnitude > dead_band
        top_active = top_magnitude > dead_band
        if bottom_active and top_active:
            command_pattern = "both"
        elif bottom_active:
            command_pattern = "bottom_only"
        elif top_active:
            command_pattern = "top_only"
        else:
            command_pattern = "zero"
        row = {
            "model_key": model_key,
            "status": status,
            "label_mode": str(label_metadata.get("label_mode") or ""),
            "run_name": sample.run_name,
            "source_sample_index": sample.sample_index,
            "measured_x_mm": float(y_true[index, distal_slice[0]]),
            "measured_y_mm": float(y_true[index, distal_slice[1]]),
            "measured_z_mm": float(y_true[index, distal_slice[2]]),
            "predicted_x_mm": float(y_pred[index, distal_slice[0]]),
            "predicted_y_mm": float(y_pred[index, distal_slice[1]]),
            "predicted_z_mm": float(y_pred[index, distal_slice[2]]),
            "position_error_mm": float(errors[index]),
            "segment_a_d1_mm": float(sample.feature_mm[0]),
            "segment_a_d2_mm": float(sample.feature_mm[1]),
            "segment_a_d3_mm": float(sample.feature_mm[2]),
            "segment_a_d4_mm": float(sample.feature_mm[3]),
            "segment_b_d5_mm": float(sample.feature_mm[4]),
            "segment_b_d6_mm": float(sample.feature_mm[5]),
            "segment_b_d7_mm": float(sample.feature_mm[6]),
            "segment_b_d8_mm": float(sample.feature_mm[7]),
            "bottom_segment_key": bottom_key,
            "top_segment_key": top_key,
            "bottom_command_magnitude_mm": bottom_magnitude,
            "top_command_magnitude_mm": top_magnitude,
            "command_pattern": command_pattern,
        }
        for role in ("intermediate_segment", "distal_tip"):
            position_slice = _slice_for(slices, role, "position")
            if position_slice:
                prefix = "intermediate" if role == "intermediate_segment" else "distal"
                role_error = float(np.linalg.norm(y_pred[index, position_slice] - y_true[index, position_slice]))
                row.update(
                    {
                        f"measured_{prefix}_x_mm": float(y_true[index, position_slice[0]]),
                        f"measured_{prefix}_y_mm": float(y_true[index, position_slice[1]]),
                        f"measured_{prefix}_z_mm": float(y_true[index, position_slice[2]]),
                        f"predicted_{prefix}_x_mm": float(y_pred[index, position_slice[0]]),
                        f"predicted_{prefix}_y_mm": float(y_pred[index, position_slice[1]]),
                        f"predicted_{prefix}_z_mm": float(y_pred[index, position_slice[2]]),
                        f"{prefix}_position_error_mm": role_error,
                    }
                )
            orientation_slice = _slice_for(slices, role, "orientation")
            if orientation_slice:
                prefix = "intermediate" if role == "intermediate_segment" else "distal"
                row.update(
                    {
                        f"measured_{prefix}_tx": float(y_true[index, orientation_slice[0]]),
                        f"measured_{prefix}_ty": float(y_true[index, orientation_slice[1]]),
                        f"measured_{prefix}_tz": float(y_true[index, orientation_slice[2]]),
                        f"predicted_{prefix}_tx": float(y_pred[index, orientation_slice[0]]),
                        f"predicted_{prefix}_ty": float(y_pred[index, orientation_slice[1]]),
                        f"predicted_{prefix}_tz": float(y_pred[index, orientation_slice[2]]),
                    }
                )
                if role == "distal_tip":
                    row.update(
                        {
                            "measured_tx": row[f"measured_{prefix}_tx"],
                            "measured_ty": row[f"measured_{prefix}_ty"],
                            "measured_tz": row[f"measured_{prefix}_tz"],
                            "predicted_tx": row[f"predicted_{prefix}_tx"],
                            "predicted_ty": row[f"predicted_{prefix}_ty"],
                            "predicted_tz": row[f"predicted_{prefix}_tz"],
                        }
                    )
        rows.append(row)
    _ = position_metrics
    return rows


def _resolve_bottom_top_keys(sample: Any) -> tuple[str, str]:
    """Resolve the bottom/top segment keys for a single accepted sample.

    Reads first from the sample's source payload (saved at collection time);
    falls back to the segment_metadata role labels (proximal/distal); finally
    defaults to A=bottom / B=top.
    """
    source_payload = dict(getattr(sample, "source_payload", {}) or {})
    extra = dict(source_payload.get("extra") or {})
    assembly = dict(extra.get("physical_assembly") or {})
    bottom_key = str(assembly.get("bottom_segment_key") or "")
    top_key = str(assembly.get("top_segment_key") or "")
    if bottom_key in {"segment_a", "segment_b"} and top_key in {"segment_a", "segment_b"} and bottom_key != top_key:
        return bottom_key, top_key
    # Fallback: look at segment_metadata roles on the sample object.
    segment_metadata = dict(getattr(sample, "segment_metadata", {}) or {})
    for key, data in segment_metadata.items():
        role = str(data.get("role") or "").strip().lower()
        if role == "proximal":
            bottom_key = str(key)
        elif role == "distal":
            top_key = str(key)
    if bottom_key in {"segment_a", "segment_b"} and top_key in {"segment_a", "segment_b"} and bottom_key != top_key:
        return bottom_key, top_key
    return "segment_a", "segment_b"


def _slice_for(slices: dict[str, Any], role: str, kind: str) -> list[int] | None:
    role_slices = slices.get(role) if isinstance(slices.get(role), dict) else {}
    value = role_slices.get(kind)
    if isinstance(value, list) and len(value) == 3:
        return [int(item) for item in value]
    return None

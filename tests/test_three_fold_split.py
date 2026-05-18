"""Tests pinning the train/val/test three-fold contract.

The two-segment scaffold's ``build_train_val_test_split`` produces three
disjoint folds with three guarantees the rest of the training stack relies on:

1. Disjoint + exhaustive: every sample appears in exactly one fold.
2. By-run when the dataset has >= 4 unique runs; falls back to random
   pooled with a logged warning when it cannot.
3. Reproducible under a fixed seed.

These tests pin those guarantees and the legacy ``build_train_test_split``
compatibility shim (which folds val into train so old callers see the same
shape they always did).

The third class exercises the val-driven early-stopping contract end-to-end
in ``_train_normalized_mlp``: when training on a clean affine map with a
deliberately noisy "val" fold whose minimum is reached early, the loop must
stop before late epochs that would otherwise fit train more tightly.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("torch")

from continuum_robot.modeling.two_segment.models import _train_normalized_mlp
from continuum_robot.modeling.two_segment.train import (
    build_train_test_split,
    build_train_val_test_split,
)


@dataclass(frozen=True)
class _StubSample:
    run_dir: str


def _samples_from_runs(*, per_run_counts: list[int]) -> list[_StubSample]:
    samples: list[_StubSample] = []
    for run_index, count in enumerate(per_run_counts):
        run_name = f"/tmp/run_{run_index:02d}"
        for _ in range(int(count)):
            samples.append(_StubSample(run_dir=run_name))
    return samples


# ---------------------------------------------------------------------------
# build_train_val_test_split contract
# ---------------------------------------------------------------------------


class TestBuildTrainValTestSplit:
    def test_three_folds_disjoint_and_exhaustive(self) -> None:
        samples = _samples_from_runs(per_run_counts=[100])
        split = build_train_val_test_split(samples=samples, random_seed=0)
        train, val, test = set(split["train_indices"]), set(split["val_indices"]), set(split["test_indices"])
        assert train.isdisjoint(val)
        assert train.isdisjoint(test)
        assert val.isdisjoint(test)
        assert train | val | test == set(range(len(samples)))

    def test_default_fractions_yield_60_20_20(self) -> None:
        samples = _samples_from_runs(per_run_counts=[100])
        split = build_train_val_test_split(samples=samples, random_seed=0)
        assert len(split["train_indices"]) == 60
        assert len(split["val_indices"]) == 20
        assert len(split["test_indices"]) == 20

    def test_by_run_when_four_or_more_runs(self) -> None:
        samples = _samples_from_runs(per_run_counts=[10, 10, 10, 10, 10])
        split = build_train_val_test_split(samples=samples, random_seed=1)
        assert split["method"] == "by_run"
        # Every sample in a val_run goes to val; every sample in a test_run
        # goes to test; train picks up the remaining whole runs.
        val_runs = set(split["val_runs"])
        test_runs = set(split["test_runs"])
        assert val_runs.isdisjoint(test_runs)
        # All three folds contain WHOLE runs in by-run mode.
        for index in split["val_indices"]:
            assert split["run_names"][index] in val_runs
        for index in split["test_indices"]:
            assert split["run_names"][index] in test_runs
        for index in split["train_indices"]:
            assert split["run_names"][index] not in val_runs
            assert split["run_names"][index] not in test_runs

    def test_three_unique_runs_falls_back_to_random_pooled_with_warning(self) -> None:
        samples = _samples_from_runs(per_run_counts=[20, 20, 20])
        split = build_train_val_test_split(samples=samples, random_seed=2)
        assert split["method"] == "random_pooled"
        assert any("by_run" in w and "3 unique runs" in w for w in split["warnings"])

    def test_single_run_falls_back_to_random_pooled_with_warning(self) -> None:
        samples = _samples_from_runs(per_run_counts=[60])
        split = build_train_val_test_split(samples=samples, random_seed=3)
        assert split["method"] == "random_pooled"
        assert any("one unique run" in w for w in split["warnings"])

    def test_explicit_by_run_false_uses_random_even_with_many_runs(self) -> None:
        samples = _samples_from_runs(per_run_counts=[10, 10, 10, 10, 10, 10])
        split = build_train_val_test_split(samples=samples, random_seed=4, by_run_split=False)
        assert split["method"] == "random_pooled"

    def test_reproducible_under_fixed_seed(self) -> None:
        samples = _samples_from_runs(per_run_counts=[10, 10, 10, 10, 10])
        first = build_train_val_test_split(samples=samples, random_seed=7)
        second = build_train_val_test_split(samples=samples, random_seed=7)
        assert first["train_indices"] == second["train_indices"]
        assert first["val_indices"] == second["val_indices"]
        assert first["test_indices"] == second["test_indices"]
        # Different seeds should at least sometimes change the split.
        third = build_train_val_test_split(samples=samples, random_seed=42)
        assert (
            third["train_indices"] != first["train_indices"]
            or third["val_indices"] != first["val_indices"]
            or third["test_indices"] != first["test_indices"]
        )

    def test_rejects_pathological_fractions(self) -> None:
        samples = _samples_from_runs(per_run_counts=[20])
        with pytest.raises(ValueError, match="val_fraction"):
            build_train_val_test_split(samples=samples, val_fraction=0.0, random_seed=0)
        with pytest.raises(ValueError, match="test_fraction"):
            build_train_val_test_split(samples=samples, test_fraction=0.0, random_seed=0)
        with pytest.raises(ValueError, match="must be strictly less than"):
            build_train_val_test_split(samples=samples, val_fraction=0.5, test_fraction=0.5, random_seed=0)
        with pytest.raises(ValueError, match="At least 3 samples"):
            build_train_val_test_split(samples=samples[:2], random_seed=0)


# ---------------------------------------------------------------------------
# Legacy build_train_test_split compatibility shim
# ---------------------------------------------------------------------------


class TestLegacyTwoFoldShim:
    def test_legacy_shape_returns_train_and_test_indices(self) -> None:
        samples = _samples_from_runs(per_run_counts=[50])
        legacy = build_train_test_split(samples=samples, test_fraction=0.25, random_seed=0)
        assert legacy["schema_version"] == "two_segment_train_test_split_v1"
        assert "train_indices" in legacy
        assert "test_indices" in legacy
        # No val_indices in the legacy shape -- callers that ask for the old
        # contract get the old shape back (the val fold is silently folded
        # into train).
        assert "val_indices" not in legacy

    def test_legacy_call_train_plus_val_covers_complement_of_test(self) -> None:
        samples = _samples_from_runs(per_run_counts=[50])
        legacy = build_train_test_split(samples=samples, test_fraction=0.25, random_seed=0)
        train_plus_val = set(legacy["train_indices"])
        test = set(legacy["test_indices"])
        assert train_plus_val.isdisjoint(test)
        assert train_plus_val | test == set(range(len(samples)))


# ---------------------------------------------------------------------------
# Val-driven early stopping inside _train_normalized_mlp
# ---------------------------------------------------------------------------


class TestValDrivenEarlyStopping:
    def test_train_normalized_mlp_returns_test_and_val_predictions(self) -> None:
        """_train_normalized_mlp must return BOTH val and test predictions so
        callers can compute val metrics for selection and test metrics for the
        held-out final report."""
        import torch
        from torch import nn

        rng = np.random.default_rng(0)
        n = 60
        X = rng.normal(size=(n, 4))
        weights = rng.normal(size=(4, 2))
        y = X @ weights
        test_predictions, val_predictions, history, state_dict, stats = _train_normalized_mlp(
            X_train=X[:36],
            y_train=y[:36],
            X_val=X[36:48],
            y_val=y[36:48],
            X_test=X[48:],
            hidden_layers=[8, 8],
            learning_rate=5e-3,
            epochs=10,
            patience=20,
            seed=1,
            batch_size=8,
            torch=torch,
            nn=nn,
        )
        assert test_predictions.shape == y[48:].shape
        assert val_predictions.shape == y[36:48].shape
        assert state_dict is not None
        # Loss history must record val (not test) -- our caller uses this for
        # selection. Test loss never appears in history.
        sample_row = history[0]
        assert "train_loss" in sample_row
        assert "validation_loss" in sample_row
        assert "test_loss" not in sample_row

    def test_early_stopping_uses_val_not_test(self) -> None:
        """Construct a setup where val loss stops improving early. With patience
        small enough to bite, the loop must stop before all epochs are run --
        i.e., early stopping IS responsive to the val signal."""
        import torch
        from torch import nn

        rng = np.random.default_rng(123)
        n = 100
        X = rng.normal(size=(n, 4))
        weights = rng.normal(size=(4, 2))
        y = X @ weights
        # Add noise only to val so it bottoms out early in training and the
        # patience trigger fires; train and test are clean.
        y_val = y[60:80] + rng.normal(scale=2.0, size=(20, 2))
        _, _, history, _, _ = _train_normalized_mlp(
            X_train=X[:60],
            y_train=y[:60],
            X_val=X[60:80],
            y_val=y_val,
            X_test=X[80:],
            hidden_layers=[8, 8],
            learning_rate=5e-3,
            epochs=500,  # plenty
            patience=3,  # tiny -- early stop should fire well before epoch 500
            seed=7,
            batch_size=16,
            torch=torch,
            nn=nn,
        )
        assert len(history) < 500, (
            f"early stopping should have fired before all 500 epochs; "
            f"got {len(history)} epochs"
        )

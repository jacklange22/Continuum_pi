"""Conditional ANN training smoke for the two-segment modeling pipeline.

These tests only run when ``torch`` is importable. CI environments without
torch skip them cleanly via ``pytest.importorskip``. On a bench setup where
torch is installed, the tests pin three thesis-critical behaviors:

1. The ANN can train end-to-end on a synthetic affine map and beats predicting
   the per-axis mean by a meaningful margin.
2. The ANN persists weights + loss history to disk so the operator can audit
   what the model actually learned.
3. The loss history contains both ``train_loss`` and ``validation_loss``
   columns and is monotonically improving on average (basic convergence sanity).

If torch is unavailable the conditional `unavailable_when_torch_missing` test
in test_two_segment_modeling.py already covers the graceful fallback.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")  # noqa: F841 — skip the whole module if torch absent

from continuum_robot.modeling.two_segment.models import TorchANNModel


def _random_affine_dataset(
    *,
    sample_count: int,
    feature_dim: int,
    label_dim: int,
    noise_sigma: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Synthetic 60/20/20 train/val/test split of a noisy affine map.

    Returns ``(X_train, y_train, X_val, y_val, X_test, y_test)`` so callers
    pass a true validation fold into ``fit_predict``. The val fold matters
    because the training loop early-stops on it; mixing test into val would
    leak the held-out fold back into training decisions.
    """
    rng = np.random.default_rng(seed)
    weights = rng.normal(scale=2.0, size=(feature_dim, label_dim))
    bias = rng.normal(scale=5.0, size=(label_dim,))
    train_size = max(int(sample_count * 0.60), 1)
    val_size = max(int(sample_count * 0.20), 1)
    X_full = rng.normal(scale=1.0, size=(sample_count, feature_dim))
    y_clean = X_full @ weights + bias
    y_full = y_clean + rng.normal(scale=noise_sigma, size=y_clean.shape)
    return (
        X_full[:train_size],
        y_full[:train_size],
        X_full[train_size : train_size + val_size],
        y_full[train_size : train_size + val_size],
        X_full[train_size + val_size :],
        y_full[train_size + val_size :],
    )


def test_torch_ann_beats_mean_predictor_on_noisy_linear_map(tmp_path: Path) -> None:
    """The ANN trained for a few epochs must clearly beat the constant-mean baseline."""

    X_train, y_train, X_val, y_val, X_test, y_test = _random_affine_dataset(
        sample_count=400, feature_dim=8, label_dim=3, noise_sigma=0.3, seed=11
    )

    model = TorchANNModel(hidden_layers=[32, 32], learning_rate=1e-2, epochs=40, patience=10, seed=11, batch_size=64)
    result = model.fit_predict(
        X_train=X_train, y_train=y_train,
        X_val=X_val, y_val=y_val,
        X_test=X_test, y_test=y_test,
        model_dir=tmp_path / "ann",
    )

    assert result.status == "completed", f"ANN should train on this dataset; status={result.status} reason={result.reason}"
    assert result.predictions is not None
    mean_pred = np.broadcast_to(y_train.mean(axis=0), y_test.shape)
    mean_rmse = float(np.sqrt(np.mean(np.linalg.norm(mean_pred - y_test, axis=1) ** 2)))
    ann_rmse = float(result.metrics["xyz_rmse_mm"])
    # ANN must clearly beat the constant-mean predictor.
    assert ann_rmse < mean_rmse / 2.0, f"ANN RMSE={ann_rmse:.3f} should beat mean RMSE={mean_rmse:.3f} by 2x"


def test_torch_ann_persists_weights_and_loss_history(tmp_path: Path) -> None:
    """Operator must be able to load the saved ANN state and inspect convergence."""

    X_train, y_train, X_val, y_val, X_test, y_test = _random_affine_dataset(
        sample_count=200, feature_dim=8, label_dim=3, noise_sigma=0.1, seed=23
    )

    model_dir = tmp_path / "ann"
    model = TorchANNModel(hidden_layers=[16, 16], learning_rate=5e-3, epochs=30, patience=10, seed=23, batch_size=32)
    result = model.fit_predict(
        X_train=X_train, y_train=y_train,
        X_val=X_val, y_val=y_val,
        X_test=X_test, y_test=y_test,
        model_dir=model_dir,
    )

    assert result.status == "completed"
    ann_model_path = model_dir / "ann_model.pt"
    history_path = model_dir / "ann_loss_history.json"
    assert ann_model_path.exists()
    assert history_path.exists()

    # Loss history shape sanity.
    history = json.loads(history_path.read_text(encoding="utf-8"))
    assert isinstance(history, list)
    assert history, "loss history must not be empty when training completed"
    sample_row = history[0]
    assert "epoch" in sample_row
    assert "train_loss" in sample_row
    assert "validation_loss" in sample_row

    # Saved state can be reloaded with the recorded shapes.
    payload = torch.load(ann_model_path, map_location="cpu", weights_only=False)
    assert payload["input_dim"] == 8
    assert payload["output_dim"] == 3
    assert payload["hidden_layers"] == [16, 16]
    assert isinstance(payload["state_dict"], dict)


def test_torch_ann_loss_history_shows_overall_improvement(tmp_path: Path) -> None:
    """Mean of first 25 % training-loss epochs should clearly exceed mean of last 25 %.

    This is a basic convergence sanity check; it does not require strict
    monotonicity (Adam + small batches is noisy) but catches the case where
    learning rate is so high the loss never settles.
    """

    X_train, y_train, X_val, y_val, X_test, y_test = _random_affine_dataset(
        sample_count=400, feature_dim=8, label_dim=3, noise_sigma=0.2, seed=37
    )

    model_dir = tmp_path / "ann"
    model = TorchANNModel(hidden_layers=[32, 32], learning_rate=5e-3, epochs=40, patience=40, seed=37, batch_size=64)
    result = model.fit_predict(
        X_train=X_train, y_train=y_train,
        X_val=X_val, y_val=y_val,
        X_test=X_test, y_test=y_test,
        model_dir=model_dir,
    )

    assert result.status == "completed"
    history = result.loss_history
    assert len(history) >= 8, f"need at least 8 epochs to compare improvement; got {len(history)}"
    quarter = max(2, len(history) // 4)
    train_losses = [float(row["train_loss"]) for row in history]
    early_mean = float(np.mean(train_losses[:quarter]))
    late_mean = float(np.mean(train_losses[-quarter:]))
    assert late_mean < early_mean, f"late training loss ({late_mean:.4f}) should be below early ({early_mean:.4f})"

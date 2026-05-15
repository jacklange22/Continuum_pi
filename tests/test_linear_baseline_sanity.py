"""Sanity tests for LinearBaselineModel.

These tests pin three thesis-critical behaviors:

1. On a truly linear map, the baseline recovers the underlying weights
   accurately and gets near-zero error on held-out samples.
2. On a noisy linear map, the baseline beats the constant-mean predictor by a
   meaningful margin (this is the minimum bar for any model claim).
3. Ridge regularization is engaged on the weights but NOT on the bias term —
   the bias should reproduce the training-mean even at high ridge alpha.

If any of these regress silently, the linear baseline becomes an empty model
that still reports "completed" — exactly the kind of failure mode the audit
wanted to prevent.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from continuum_robot.modeling.two_segment.models import LinearBaselineModel


def _random_affine_dataset(
    *,
    sample_count: int,
    feature_dim: int,
    label_dim: int,
    noise_sigma: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Build (X_train, y_train, X_test, y_test) for a known affine map y = X @ W + b + noise."""
    rng = np.random.default_rng(seed)
    weights = rng.normal(scale=2.0, size=(feature_dim, label_dim))
    bias = rng.normal(scale=5.0, size=(label_dim,))
    train_size = max(int(sample_count * 0.7), 1)
    X_full = rng.normal(scale=1.0, size=(sample_count, feature_dim))
    y_clean = X_full @ weights + bias
    y_full = y_clean + rng.normal(scale=noise_sigma, size=y_clean.shape)
    return X_full[:train_size], y_full[:train_size], X_full[train_size:], y_full[train_size:]


def test_linear_baseline_recovers_affine_map_to_within_small_tolerance(tmp_path: Path) -> None:
    """No noise → recovered predictions should be ~exact on held-out samples."""
    X_train, y_train, X_test, y_test = _random_affine_dataset(
        sample_count=200, feature_dim=8, label_dim=3, noise_sigma=0.0, seed=7
    )

    model = LinearBaselineModel(ridge_alpha=1e-9)
    result = model.fit_predict(
        X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test, model_dir=tmp_path / "linear"
    )

    assert result.status == "completed"
    assert result.predictions is not None
    residual = np.linalg.norm(result.predictions - y_test, axis=1)
    assert float(np.max(residual)) < 1e-3, f"residual max should be near zero, got {np.max(residual)}"
    assert result.metrics["xyz_rmse_mm"] < 1e-3


def test_linear_baseline_beats_mean_predictor_on_noisy_linear_map(tmp_path: Path) -> None:
    """With moderate noise, the baseline should still substantially outperform predicting the per-axis mean."""
    X_train, y_train, X_test, y_test = _random_affine_dataset(
        sample_count=400, feature_dim=8, label_dim=3, noise_sigma=0.5, seed=13
    )

    model = LinearBaselineModel(ridge_alpha=1e-6)
    result = model.fit_predict(
        X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test, model_dir=tmp_path / "linear"
    )

    mean_prediction = np.broadcast_to(y_train.mean(axis=0), y_test.shape)
    mean_rmse = float(np.sqrt(np.mean(np.linalg.norm(mean_prediction - y_test, axis=1) ** 2)))
    baseline_rmse = float(result.metrics["xyz_rmse_mm"])

    # The baseline must beat the mean predictor by a clear margin (at least 5×) on a noisy linear map.
    assert baseline_rmse < mean_rmse / 5.0, (
        f"linear baseline should beat mean predictor by at least 5x; "
        f"baseline_rmse={baseline_rmse:.3f}, mean_rmse={mean_rmse:.3f}"
    )


def test_linear_baseline_persists_weights_and_intercept_to_disk(tmp_path: Path) -> None:
    """Operator must be able to inspect what the linear baseline actually learned."""
    X_train, y_train, X_test, y_test = _random_affine_dataset(
        sample_count=80, feature_dim=8, label_dim=3, noise_sigma=0.1, seed=21
    )

    model = LinearBaselineModel(ridge_alpha=1e-6)
    model_dir = tmp_path / "linear"
    model.fit_predict(X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test, model_dir=model_dir)

    artifact_path = model_dir / "linear_baseline_weights.json"
    assert artifact_path.exists(), "weights JSON must be written so the operator can audit the fit"
    import json

    payload = json.loads(artifact_path.read_text(encoding="utf-8"))
    assert payload["model_key"] == "linear_baseline"
    weights_array = np.asarray(payload["weights"], dtype=float)
    intercept_array = np.asarray(payload["intercept"], dtype=float)
    assert weights_array.shape == (8, 3)
    assert intercept_array.shape == (3,)


def test_linear_baseline_high_ridge_alpha_does_not_regularize_bias(tmp_path: Path) -> None:
    """Bias must NOT be regularized: with high alpha the bias should still reproduce the training mean.

    If a future refactor accidentally regularizes the bias term (e.g. drops the
    ``reg[-1, -1] = 0.0`` line), the bias will collapse toward zero and the
    test will fail loudly.
    """
    X_train, y_train, X_test, y_test = _random_affine_dataset(
        sample_count=200, feature_dim=8, label_dim=3, noise_sigma=0.2, seed=29
    )

    # Very large ridge alpha → weights driven toward zero, predictions should ≈ y_train.mean(axis=0).
    model = LinearBaselineModel(ridge_alpha=1e12)
    result = model.fit_predict(
        X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test, model_dir=tmp_path / "linear"
    )

    expected_mean = y_train.mean(axis=0)
    actual_intercept = np.asarray(result.predictions, dtype=float).mean(axis=0)
    # Allow some tolerance because of test feature variance.
    assert np.allclose(actual_intercept, expected_mean, atol=0.5), (
        f"expected predictions to collapse to training mean {expected_mean}, "
        f"got {actual_intercept}"
    )

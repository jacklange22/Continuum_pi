"""Model families for offline two-segment modeling analysis."""

from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path
from typing import Any

import numpy as np

from continuum_robot.modeling.two_segment.evaluate import all_metrics


@dataclass(frozen=True)
class ModelFitResult:
    """Predictions, metrics, and artifact metadata for one model."""

    model_key: str
    label: str
    status: str
    reason: str = ""
    predictions: np.ndarray | None = None
    metrics: dict[str, Any] = field(default_factory=dict)
    artifact_paths: dict[str, str] = field(default_factory=dict)
    loss_history: list[dict[str, float]] = field(default_factory=list)


class BaseTwoSegmentModel:
    model_key = "base"
    label = "Base"

    def fit_predict(
        self,
        *,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray,
        model_dir: Path,
    ) -> ModelFitResult:
        raise NotImplementedError


class LinearBaselineModel(BaseTwoSegmentModel):
    """Small ridge/least-squares linear baseline."""

    model_key = "linear_baseline"
    label = "Linear Baseline"

    def __init__(self, *, ridge_alpha: float = 1e-6) -> None:
        self.ridge_alpha = float(ridge_alpha)

    def fit_predict(
        self,
        *,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray,
        model_dir: Path,
    ) -> ModelFitResult:
        model_dir.mkdir(parents=True, exist_ok=True)
        X_aug = np.concatenate([X_train, np.ones((X_train.shape[0], 1), dtype=float)], axis=1)
        reg = float(self.ridge_alpha) * np.eye(X_aug.shape[1], dtype=float)
        reg[-1, -1] = 0.0
        weights = np.linalg.pinv(X_aug.T @ X_aug + reg) @ X_aug.T @ y_train
        test_aug = np.concatenate([X_test, np.ones((X_test.shape[0], 1), dtype=float)], axis=1)
        predictions = test_aug @ weights
        artifact_path = model_dir / "linear_baseline_weights.json"
        artifact_path.write_text(
            json.dumps(
                {
                    "model_key": self.model_key,
                    "ridge_alpha": self.ridge_alpha,
                    "weights": weights[:-1, :].tolist(),
                    "intercept": weights[-1, :].tolist(),
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        return ModelFitResult(
            model_key=self.model_key,
            label=self.label,
            status="completed",
            predictions=predictions,
            metrics=all_metrics(y_test, predictions),
            artifact_paths={"weights": str(artifact_path)},
        )


class UnavailableModel(BaseTwoSegmentModel):
    """Explicit scaffold for model families not yet available in active code."""

    def __init__(self, *, model_key: str, label: str, reason: str) -> None:
        self.model_key = model_key
        self.label = label
        self.reason = reason

    def fit_predict(
        self,
        *,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray,
        model_dir: Path,
    ) -> ModelFitResult:
        _ = X_train, y_train, X_test, y_test
        model_dir.mkdir(parents=True, exist_ok=True)
        metadata_path = model_dir / f"{self.model_key}_unavailable.json"
        metadata_path.write_text(
            json.dumps({"model_key": self.model_key, "status": "unavailable", "reason": self.reason}, indent=2),
            encoding="utf-8",
        )
        return ModelFitResult(
            model_key=self.model_key,
            label=self.label,
            status="unavailable",
            reason=self.reason,
            artifact_paths={"metadata": str(metadata_path)},
        )


class TorchANNModel(BaseTwoSegmentModel):
    """PyTorch MLP scaffold inspired by the legacy [128,128] two-segment ANN."""

    model_key = "ann"
    label = "ANN MLP"

    def __init__(
        self,
        *,
        hidden_layers: list[int] | None = None,
        learning_rate: float = 1e-3,
        epochs: int = 200,
        patience: int = 20,
        seed: int = 0,
    ) -> None:
        self.hidden_layers = list(hidden_layers or [128, 128])
        self.learning_rate = float(learning_rate)
        self.epochs = int(epochs)
        self.patience = int(patience)
        self.seed = int(seed)

    def fit_predict(
        self,
        *,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray,
        model_dir: Path,
    ) -> ModelFitResult:
        try:
            import torch
            from torch import nn
        except Exception as exc:  # pragma: no cover - depends on local optional torch install
            return ModelFitResult(
                model_key=self.model_key,
                label=self.label,
                status="unavailable",
                reason=f"PyTorch unavailable: {exc}",
            )
        if X_train.shape[0] < 2:
            return ModelFitResult(model_key=self.model_key, label=self.label, status="unavailable", reason="ANN requires at least 2 training samples.")

        torch.manual_seed(self.seed)
        model_dir.mkdir(parents=True, exist_ok=True)
        x_mean, x_std = _stats(X_train)
        y_mean, y_std = _stats(y_train)
        x_train = torch.tensor((X_train - x_mean) / x_std, dtype=torch.float32)
        y_train_t = torch.tensor((y_train - y_mean) / y_std, dtype=torch.float32)
        x_test = torch.tensor((X_test - x_mean) / x_std, dtype=torch.float32)
        model = _mlp(input_dim=X_train.shape[1], output_dim=y_train.shape[1], hidden_layers=self.hidden_layers, nn=nn)
        optimizer = torch.optim.Adam(model.parameters(), lr=self.learning_rate)
        loss_fn = nn.MSELoss()
        best_loss = float("inf")
        best_state: dict[str, Any] | None = None
        stale_epochs = 0
        history: list[dict[str, float]] = []
        for epoch in range(1, max(1, self.epochs) + 1):
            model.train()
            optimizer.zero_grad()
            train_loss = loss_fn(model(x_train), y_train_t)
            train_loss.backward()
            optimizer.step()
            model.eval()
            with torch.no_grad():
                val_loss = float(loss_fn(model(x_test), torch.tensor((y_test - y_mean) / y_std, dtype=torch.float32)))
            history.append({"epoch": float(epoch), "train_loss": float(train_loss.detach()), "validation_loss": val_loss})
            if val_loss < best_loss:
                best_loss = val_loss
                best_state = {key: value.detach().clone() for key, value in model.state_dict().items()}
                stale_epochs = 0
            else:
                stale_epochs += 1
            if stale_epochs >= max(1, self.patience):
                break
        if best_state is not None:
            model.load_state_dict(best_state)
        model.eval()
        with torch.no_grad():
            pred_norm = model(x_test).cpu().numpy()
        predictions = pred_norm * y_std + y_mean
        model_path = model_dir / "ann_model.pt"
        torch.save(
            {
                "state_dict": model.state_dict(),
                "input_dim": int(X_train.shape[1]),
                "output_dim": int(y_train.shape[1]),
                "hidden_layers": list(self.hidden_layers),
                "x_mean": x_mean.tolist(),
                "x_std": x_std.tolist(),
                "y_mean": y_mean.tolist(),
                "y_std": y_std.tolist(),
            },
            model_path,
        )
        history_path = model_dir / "ann_loss_history.json"
        history_path.write_text(json.dumps(history, indent=2), encoding="utf-8")
        return ModelFitResult(
            model_key=self.model_key,
            label=self.label,
            status="completed",
            predictions=predictions,
            metrics=all_metrics(y_test, predictions),
            artifact_paths={"model": str(model_path), "loss_history": str(history_path)},
            loss_history=history,
        )


def default_model_suite(*, model_keys: list[str] | None = None, config: dict[str, Any] | None = None) -> list[BaseTwoSegmentModel]:
    """Return the requested default model family objects."""

    requested = list(model_keys or ["linear_baseline", "camarillo", "mike_constant_curvature", "ann"])
    raw = dict(config or {})
    models: list[BaseTwoSegmentModel] = []
    for key in requested:
        if key == "linear_baseline":
            models.append(LinearBaselineModel(ridge_alpha=float(raw.get("ridge_alpha", 1e-6))))
        elif key == "camarillo":
            reason = "Camarillo two-segment active adapter is not configured in this scaffold; no fake physics values are produced."
            if not raw.get("camarillo"):
                models.append(UnavailableModel(model_key="camarillo", label="Camarillo", reason=reason))
            else:
                models.append(UnavailableModel(model_key="camarillo", label="Camarillo", reason="Configured Camarillo parameters are not yet wired to an active two-segment solver."))
        elif key in {"mike", "mike_constant_curvature"}:
            models.append(
                UnavailableModel(
                    model_key="mike_constant_curvature",
                    label="Mike Constant Curvature",
                    reason="No active two-segment Mike-style constant-curvature adapter is available yet.",
                )
            )
        elif key == "ann":
            ann_config = dict(raw.get("ann", {}) or {})
            models.append(
                TorchANNModel(
                    hidden_layers=list(ann_config.get("hidden_layers", [128, 128])),
                    learning_rate=float(ann_config.get("learning_rate", 1e-3)),
                    epochs=int(ann_config.get("epochs", 200)),
                    patience=int(ann_config.get("patience", 20)),
                    seed=int(raw.get("random_seed", ann_config.get("seed", 0))),
                )
            )
    return models


def _stats(matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = np.mean(matrix, axis=0)
    std = np.std(matrix, axis=0)
    return mean, np.where(std == 0.0, 1.0, std)


def _mlp(*, input_dim: int, output_dim: int, hidden_layers: list[int], nn):
    layers = []
    current = int(input_dim)
    for hidden in hidden_layers:
        layers.append(nn.Linear(current, int(hidden)))
        layers.append(nn.ReLU())
        current = int(hidden)
    layers.append(nn.Linear(current, int(output_dim)))
    return nn.Sequential(*layers)

"""Model families for offline two-segment modeling analysis."""

from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path
from typing import Any

import numpy as np

from continuum_robot.modeling.two_segment.evaluate import all_metrics
from continuum_robot.modeling.two_segment.physics import (
    PhysicsAdapterStatus,
    assess_camarillo_status,
    assess_mike_constant_curvature_status,
    fill_prediction_from_role_poses,
    two_segment_constant_curvature_prediction,
)


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
    status_details: dict[str, Any] = field(default_factory=dict)


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
        label_metadata: dict[str, Any] | None = None,
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
        label_metadata: dict[str, Any] | None = None,
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
            metrics=all_metrics(y_test, predictions, label_metadata=label_metadata),
            artifact_paths={"weights": str(artifact_path)},
        )


class UnavailableModel(BaseTwoSegmentModel):
    """Explicit scaffold for model families not yet available in active code."""

    def __init__(self, *, model_key: str, label: str, reason: str, status_details: dict[str, Any] | None = None) -> None:
        self.model_key = model_key
        self.label = label
        self.reason = reason
        self.status_details = dict(status_details or {})

    def fit_predict(
        self,
        *,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray,
        model_dir: Path,
        label_metadata: dict[str, Any] | None = None,
    ) -> ModelFitResult:
        _ = X_train, y_train, X_test, y_test
        model_dir.mkdir(parents=True, exist_ok=True)
        status = str(self.status_details.get("status") or "unavailable")
        metadata_path = model_dir / "model_status.json"
        payload = {"model_key": self.model_key, "status": status, "reason": self.reason, **self.status_details}
        metadata_path.write_text(
            json.dumps(payload, indent=2),
            encoding="utf-8",
        )
        return ModelFitResult(
            model_key=self.model_key,
            label=self.label,
            status=status,
            reason=self.reason,
            artifact_paths={"metadata": str(metadata_path)},
            status_details=payload,
        )


class MikeConstantCurvatureModel(BaseTwoSegmentModel):
    """Config-gated two-segment constant-curvature geometric baseline."""

    model_key = "mike_constant_curvature"
    label = "Mike Constant Curvature"

    def __init__(self, *, config: dict[str, Any]) -> None:
        self.config = dict(config or {})
        self.adapter_status = assess_mike_constant_curvature_status(self.config)

    def fit_predict(
        self,
        *,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray,
        model_dir: Path,
        label_metadata: dict[str, Any] | None = None,
    ) -> ModelFitResult:
        _ = X_train, y_train
        model_dir.mkdir(parents=True, exist_ok=True)
        if self.adapter_status.status != "available":
            status = self.adapter_status.to_dict()
            status_path = model_dir / "model_status.json"
            status_path.write_text(json.dumps(status, indent=2), encoding="utf-8")
            return ModelFitResult(
                model_key=self.model_key,
                label=self.label,
                status=self.adapter_status.status,
                reason=self.adapter_status.reason,
                artifact_paths={"metadata": str(status_path)},
                status_details=status,
            )
        if not label_metadata:
            raise ValueError("Mike constant-curvature predictions require label metadata.")
        predictions = []
        for command in np.asarray(X_test, dtype=float):
            pose = two_segment_constant_curvature_prediction(command, self.config)
            predictions.append(
                fill_prediction_from_role_poses(
                    label_metadata=label_metadata,
                    intermediate_xyz=pose["intermediate_xyz"],
                    distal_xyz=pose["distal_xyz"],
                    intermediate_tangent=pose["intermediate_tangent"],
                    distal_tangent=pose["distal_tangent"],
                )
            )
        y_pred = np.stack(predictions, axis=0) if predictions else np.zeros_like(y_test)
        status = {
            **self.adapter_status.to_dict(),
            "status": "completed",
            "predictions_generated": True,
            "sample_count": int(y_pred.shape[0]),
        }
        status_path = model_dir / "model_status.json"
        status_path.write_text(json.dumps(status, indent=2), encoding="utf-8")
        return ModelFitResult(
            model_key=self.model_key,
            label=self.label,
            status="completed",
            predictions=y_pred,
            metrics=all_metrics(y_test, y_pred, label_metadata=label_metadata),
            artifact_paths={"metadata": str(status_path)},
            status_details=status,
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
        batch_size: int = 64,
    ) -> None:
        self.hidden_layers = list(hidden_layers or [128, 128])
        self.learning_rate = float(learning_rate)
        self.epochs = int(epochs)
        self.patience = int(patience)
        self.seed = int(seed)
        self.batch_size = int(batch_size)

    def fit_predict(
        self,
        *,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray,
        model_dir: Path,
        label_metadata: dict[str, Any] | None = None,
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

        model_dir.mkdir(parents=True, exist_ok=True)
        predictions, history, state_dict, stats = _train_normalized_mlp(
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
            y_test=y_test,
            hidden_layers=self.hidden_layers,
            learning_rate=self.learning_rate,
            epochs=self.epochs,
            patience=self.patience,
            seed=self.seed,
            batch_size=self.batch_size,
            torch=torch,
            nn=nn,
        )
        model_path = model_dir / "ann_model.pt"
        torch.save(
            {
                "state_dict": state_dict,
                "input_dim": int(X_train.shape[1]),
                "output_dim": int(y_train.shape[1]),
                "hidden_layers": list(self.hidden_layers),
                "batch_size": int(self.batch_size),
                "x_mean": stats["x_mean"].tolist(),
                "x_std": stats["x_std"].tolist(),
                "y_mean": stats["y_mean"].tolist(),
                "y_std": stats["y_std"].tolist(),
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
            metrics=all_metrics(y_test, predictions, label_metadata=label_metadata),
            artifact_paths={"model": str(model_path), "loss_history": str(history_path)},
            loss_history=history,
        )


class TorchANNSweepModel(BaseTwoSegmentModel):
    """Optional small ANN architecture sweep. Skips cleanly when torch is absent."""

    model_key = "ann"
    label = "ANN MLP Sweep"

    def __init__(
        self,
        *,
        hidden_layer_options: list[list[int]],
        learning_rate: float = 1e-3,
        epochs: int = 200,
        patience: int = 20,
        seeds: list[int] | None = None,
        batch_size: int = 64,
    ) -> None:
        self.hidden_layer_options = [list(option) for option in (hidden_layer_options or [[32, 32], [64, 64], [128, 128]])]
        self.learning_rate = float(learning_rate)
        self.epochs = int(epochs)
        self.patience = int(patience)
        self.seeds = [int(value) for value in (seeds or [0])]
        self.batch_size = int(batch_size)

    def fit_predict(
        self,
        *,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray,
        model_dir: Path,
        label_metadata: dict[str, Any] | None = None,
    ) -> ModelFitResult:
        candidates: list[ModelFitResult] = []
        sweep_rows: list[dict[str, Any]] = []
        for layers in self.hidden_layer_options:
            for seed in self.seeds:
                key = "_".join(str(value) for value in layers)
                child_dir = model_dir / f"hidden_{key}_seed_{seed}"
                model = TorchANNModel(
                    hidden_layers=layers,
                    learning_rate=self.learning_rate,
                    epochs=self.epochs,
                    patience=self.patience,
                    seed=seed,
                    batch_size=self.batch_size,
                )
                result = model.fit_predict(
                    X_train=X_train,
                    y_train=y_train,
                    X_test=X_test,
                    y_test=y_test,
                    model_dir=child_dir,
                    label_metadata=label_metadata,
                )
                candidates.append(result)
                sweep_rows.append(
                    {
                        "hidden_layers": layers,
                        "seed": int(seed),
                        "status": result.status,
                        "reason": result.reason,
                        "xyz_rmse_mm": result.metrics.get("xyz_rmse_mm"),
                        "distal_xyz_rmse_mm": result.metrics.get("distal_xyz_rmse_mm"),
                        "best_epoch": _best_epoch(result.loss_history),
                    }
                )
        model_dir.mkdir(parents=True, exist_ok=True)
        sweep_path = model_dir / "ann_sweep_summary.json"
        sweep_path.write_text(json.dumps(sweep_rows, indent=2), encoding="utf-8")
        completed = [result for result in candidates if result.status == "completed" and result.predictions is not None]
        if not completed:
            reason = next((result.reason for result in candidates if result.reason), "No ANN sweep candidate completed.")
            return ModelFitResult(
                model_key=self.model_key,
                label=self.label,
                status="unavailable",
                reason=reason,
                artifact_paths={"sweep_summary": str(sweep_path)},
                metrics={"ann_sweep_candidate_count": len(candidates)},
            )
        best = min(completed, key=lambda result: float(result.metrics.get("xyz_rmse_mm", float("inf"))))
        metrics = dict(best.metrics)
        metrics.update(
            {
                "ann_sweep_candidate_count": len(candidates),
                "ann_sweep_completed_count": len(completed),
                "ann_best_epoch": _best_epoch(best.loss_history),
            }
        )
        artifacts = dict(best.artifact_paths)
        artifacts["sweep_summary"] = str(sweep_path)
        return ModelFitResult(
            model_key=self.model_key,
            label=self.label,
            status="completed",
            predictions=best.predictions,
            metrics=metrics,
            artifact_paths=artifacts,
            loss_history=list(best.loss_history),
        )


class HybridResidualModel(BaseTwoSegmentModel):
    """Mike CC base prediction + ANN residual learner.

    The model computes Mike constant-curvature predictions on the training
    inputs, fits an MLP to the residual ``y_train - mike_pred_train``, and at
    inference returns ``mike_pred(X_test) + ann_residual(X_test)``. The
    underlying ANN reuses the same z-score normalization, optimizer, and
    early-stopping machinery as :class:`TorchANNModel` so the comparison is
    apples-to-apples on the same dataset/config.

    Skips cleanly with the same scaffold output as :class:`UnavailableModel`
    when the Mike CC adapter is not available (missing config or
    unvalidated conventions).
    """

    model_key = "hybrid_residual"
    label = "Hybrid (Mike CC + ANN Residual)"

    def __init__(
        self,
        *,
        config: dict[str, Any],
        hidden_layers: list[int] | None = None,
        learning_rate: float = 1e-3,
        epochs: int = 200,
        patience: int = 20,
        seed: int = 0,
        batch_size: int = 64,
    ) -> None:
        self.config = dict(config or {})
        self.hidden_layers = list(hidden_layers or [128, 128])
        self.learning_rate = float(learning_rate)
        self.epochs = int(epochs)
        self.patience = int(patience)
        self.seed = int(seed)
        self.batch_size = int(batch_size)
        self.adapter_status = assess_mike_constant_curvature_status(self.config)

    def fit_predict(
        self,
        *,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray,
        model_dir: Path,
        label_metadata: dict[str, Any] | None = None,
    ) -> ModelFitResult:
        model_dir.mkdir(parents=True, exist_ok=True)
        if self.adapter_status.status != "available":
            status_payload = {
                **self.adapter_status.to_dict(),
                "hybrid_residual_status": self.adapter_status.status,
                "hybrid_residual_reason": (
                    f"Hybrid requires Mike CC base predictions, but {self.adapter_status.reason}"
                ),
            }
            status_path = model_dir / "model_status.json"
            status_path.write_text(json.dumps(status_payload, indent=2), encoding="utf-8")
            return ModelFitResult(
                model_key=self.model_key,
                label=self.label,
                status=self.adapter_status.status,
                reason=str(status_payload["hybrid_residual_reason"]),
                artifact_paths={"metadata": str(status_path)},
                status_details=status_payload,
            )
        if not label_metadata:
            raise ValueError("Hybrid residual model requires label metadata for Mike CC base predictions.")
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
            return ModelFitResult(
                model_key=self.model_key,
                label=self.label,
                status="unavailable",
                reason="Hybrid residual model requires at least 2 training samples.",
            )
        if X_test.shape[0] < 1:
            return ModelFitResult(
                model_key=self.model_key,
                label=self.label,
                status="unavailable",
                reason="Hybrid residual model requires at least 1 test sample.",
            )

        mike_pred_train = _mike_cc_predict_batch(
            X_train, config=self.config, label_metadata=label_metadata
        )
        mike_pred_test = _mike_cc_predict_batch(
            X_test, config=self.config, label_metadata=label_metadata
        )
        residual_train = y_train - mike_pred_train
        residual_test = y_test - mike_pred_test

        ann_residual_pred, history, state_dict, stats = _train_normalized_mlp(
            X_train=X_train,
            y_train=residual_train,
            X_test=X_test,
            y_test=residual_test,
            hidden_layers=self.hidden_layers,
            learning_rate=self.learning_rate,
            epochs=self.epochs,
            patience=self.patience,
            seed=self.seed,
            batch_size=self.batch_size,
            torch=torch,
            nn=nn,
        )
        hybrid_predictions = mike_pred_test + ann_residual_pred

        mike_only_metrics = all_metrics(y_test, mike_pred_test, label_metadata=label_metadata)
        hybrid_metrics = all_metrics(y_test, hybrid_predictions, label_metadata=label_metadata)
        residual_learn_metrics = all_metrics(
            residual_test, ann_residual_pred, label_metadata=label_metadata
        )

        breakdown = {
            "mike_only": dict(mike_only_metrics),
            "ann_residual_only": dict(residual_learn_metrics),
            "hybrid": dict(hybrid_metrics),
        }
        breakdown_path = model_dir / "hybrid_breakdown.json"
        breakdown_path.write_text(json.dumps(breakdown, indent=2), encoding="utf-8")

        model_path = model_dir / "hybrid_residual_model.pt"
        torch.save(
            {
                "state_dict": state_dict,
                "input_dim": int(X_train.shape[1]),
                "output_dim": int(y_train.shape[1]),
                "hidden_layers": list(self.hidden_layers),
                "batch_size": int(self.batch_size),
                "x_mean": stats["x_mean"].tolist(),
                "x_std": stats["x_std"].tolist(),
                "y_mean": stats["y_mean"].tolist(),
                "y_std": stats["y_std"].tolist(),
                "training_target": "mike_cc_residual",
            },
            model_path,
        )
        history_path = model_dir / "hybrid_loss_history.json"
        history_path.write_text(json.dumps(history, indent=2), encoding="utf-8")
        mike_predictions_path = model_dir / "hybrid_mike_predictions_test.npy"
        np.save(mike_predictions_path, mike_pred_test)
        ann_residual_path = model_dir / "hybrid_ann_residual_predictions_test.npy"
        np.save(ann_residual_path, ann_residual_pred)

        improvement = None
        mike_xyz = mike_only_metrics.get("xyz_rmse_mm")
        hybrid_xyz = hybrid_metrics.get("xyz_rmse_mm")
        if isinstance(mike_xyz, (int, float)) and isinstance(hybrid_xyz, (int, float)):
            improvement = float(mike_xyz) - float(hybrid_xyz)

        merged_metrics: dict[str, Any] = dict(hybrid_metrics)
        if mike_xyz is not None:
            merged_metrics["mike_only_xyz_rmse_mm"] = float(mike_xyz)
        if hybrid_xyz is not None:
            merged_metrics["hybrid_xyz_rmse_mm"] = float(hybrid_xyz)
        if improvement is not None:
            merged_metrics["hybrid_xyz_rmse_improvement_over_mike_mm"] = improvement

        return ModelFitResult(
            model_key=self.model_key,
            label=self.label,
            status="completed",
            predictions=hybrid_predictions,
            metrics=merged_metrics,
            artifact_paths={
                "model": str(model_path),
                "loss_history": str(history_path),
                "breakdown": str(breakdown_path),
                "mike_predictions_test": str(mike_predictions_path),
                "ann_residual_predictions_test": str(ann_residual_path),
            },
            loss_history=history,
            status_details={
                "breakdown": breakdown,
                "mike_only_xyz_rmse_mm": mike_xyz,
                "hybrid_xyz_rmse_mm": hybrid_xyz,
                "hybrid_xyz_rmse_improvement_over_mike_mm": improvement,
            },
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
            status = assess_camarillo_status(raw)
            models.append(UnavailableModel(model_key="camarillo", label="Camarillo", reason=status.reason, status_details=status.to_dict()))
        elif key in {"mike", "mike_constant_curvature"}:
            models.append(MikeConstantCurvatureModel(config=raw))
        elif key == "ann":
            ann_config = dict(raw.get("ann", {}) or {})
            if bool(ann_config.get("sweep_enabled", False)):
                models.append(
                    TorchANNSweepModel(
                        hidden_layer_options=[list(value) for value in ann_config.get("hidden_layer_options", [[32, 32], [64, 64], [128, 128]])],
                        learning_rate=float(ann_config.get("learning_rate", 1e-3)),
                        epochs=int(ann_config.get("epochs", 200)),
                        patience=int(ann_config.get("patience", 20)),
                        seeds=[int(value) for value in ann_config.get("seeds", [raw.get("random_seed", ann_config.get("seed", 0))])],
                        batch_size=int(ann_config.get("batch_size", 64)),
                    )
                )
            else:
                models.append(
                    TorchANNModel(
                        hidden_layers=list(ann_config.get("hidden_layers", [128, 128])),
                        learning_rate=float(ann_config.get("learning_rate", 1e-3)),
                        epochs=int(ann_config.get("epochs", 200)),
                        patience=int(ann_config.get("patience", 20)),
                        seed=int(raw.get("random_seed", ann_config.get("seed", 0))),
                        batch_size=int(ann_config.get("batch_size", 64)),
                    )
                )
        elif key in {"hybrid_residual", "hybrid"}:
            hybrid_config = dict(raw.get("hybrid_residual") or raw.get("hybrid") or {})
            ann_defaults = dict(raw.get("ann") or {})
            models.append(
                HybridResidualModel(
                    config=raw,
                    hidden_layers=list(
                        hybrid_config.get("hidden_layers")
                        or ann_defaults.get("hidden_layers")
                        or [128, 128]
                    ),
                    learning_rate=float(
                        hybrid_config.get(
                            "learning_rate", ann_defaults.get("learning_rate", 1e-3)
                        )
                    ),
                    epochs=int(hybrid_config.get("epochs", ann_defaults.get("epochs", 200))),
                    patience=int(hybrid_config.get("patience", ann_defaults.get("patience", 20))),
                    seed=int(
                        hybrid_config.get(
                            "seed",
                            raw.get("random_seed", ann_defaults.get("seed", 0)),
                        )
                    ),
                    batch_size=int(
                        hybrid_config.get("batch_size", ann_defaults.get("batch_size", 64))
                    ),
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


def _batch_train_loss(*, model: Any, x_train: Any, y_train: Any, optimizer: Any, loss_fn: Any, batch_size: int, torch: Any) -> Any:
    if int(batch_size) >= int(x_train.shape[0]):
        train_loss = loss_fn(model(x_train), y_train)
        train_loss.backward()
        optimizer.step()
        return train_loss
    permutation = torch.randperm(x_train.shape[0])
    total = None
    batches = 0
    for start in range(0, int(x_train.shape[0]), int(batch_size)):
        indices = permutation[start : start + int(batch_size)]
        batch_loss = loss_fn(model(x_train[indices]), y_train[indices])
        batch_loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        total = batch_loss.detach() if total is None else total + batch_loss.detach()
        batches += 1
    return total / max(1, batches)


def _train_normalized_mlp(
    *,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    hidden_layers: list[int],
    learning_rate: float,
    epochs: int,
    patience: int,
    seed: int,
    batch_size: int,
    torch: Any,
    nn: Any,
) -> tuple[np.ndarray, list[dict[str, float]], dict[str, Any], dict[str, np.ndarray]]:
    """Train a z-score-normalized MLP on (X_train, y_train) and predict on X_test.

    Returns ``(predictions_in_original_units, loss_history, best_state_dict,
    normalization_stats)``. Validation loss is tracked on the test split (the
    pre-existing convention in this codebase -- we are not adding a second
    held-out fold here, just extracting the duplicated training plumbing so
    the hybrid model can share it). The best-loss state is restored before
    the final prediction pass so early stopping is honored.
    """

    torch.manual_seed(int(seed))
    x_mean, x_std = _stats(X_train)
    y_mean, y_std = _stats(y_train)
    x_train_t = torch.tensor((X_train - x_mean) / x_std, dtype=torch.float32)
    y_train_t = torch.tensor((y_train - y_mean) / y_std, dtype=torch.float32)
    x_test_t = torch.tensor((X_test - x_mean) / x_std, dtype=torch.float32)
    y_test_t = torch.tensor((y_test - y_mean) / y_std, dtype=torch.float32)
    model = _mlp(input_dim=X_train.shape[1], output_dim=y_train.shape[1], hidden_layers=hidden_layers, nn=nn)
    optimizer = torch.optim.Adam(model.parameters(), lr=float(learning_rate))
    loss_fn = nn.MSELoss()
    best_loss = float("inf")
    best_state: dict[str, Any] | None = None
    stale_epochs = 0
    history: list[dict[str, float]] = []
    for epoch in range(1, max(1, int(epochs)) + 1):
        model.train()
        optimizer.zero_grad()
        train_loss = _batch_train_loss(
            model=model,
            x_train=x_train_t,
            y_train=y_train_t,
            optimizer=optimizer,
            loss_fn=loss_fn,
            batch_size=max(1, int(batch_size)),
            torch=torch,
        )
        model.eval()
        with torch.no_grad():
            val_loss = float(loss_fn(model(x_test_t), y_test_t))
        history.append({"epoch": float(epoch), "train_loss": float(train_loss.detach()), "validation_loss": val_loss})
        if val_loss < best_loss:
            best_loss = val_loss
            best_state = {key: value.detach().clone() for key, value in model.state_dict().items()}
            stale_epochs = 0
        else:
            stale_epochs += 1
        if stale_epochs >= max(1, int(patience)):
            break
    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        pred_norm = model(x_test_t).cpu().numpy()
    predictions = pred_norm * y_std + y_mean
    return predictions, history, best_state if best_state is not None else dict(model.state_dict()), {
        "x_mean": x_mean,
        "x_std": x_std,
        "y_mean": y_mean,
        "y_std": y_std,
    }


def _mike_cc_predict_batch(
    X: np.ndarray, *, config: dict[str, Any], label_metadata: dict[str, Any]
) -> np.ndarray:
    """Predict the Mike CC pose for every row in ``X`` and project into label slots.

    Returns an array of shape ``(N, label_dim)`` aligned with ``label_metadata``.
    Returns an empty array of shape ``(0, 0)`` when ``X`` is empty (callers
    should not depend on this empty shape for downstream broadcasting --
    handle the empty case before calling).
    """

    arr = np.asarray(X, dtype=float)
    if arr.size == 0:
        return np.zeros((0, 0), dtype=float)
    predictions: list[np.ndarray] = []
    for command in arr:
        pose = two_segment_constant_curvature_prediction(command, config)
        predictions.append(
            fill_prediction_from_role_poses(
                label_metadata=label_metadata,
                intermediate_xyz=pose["intermediate_xyz"],
                distal_xyz=pose["distal_xyz"],
                intermediate_tangent=pose["intermediate_tangent"],
                distal_tangent=pose["distal_tangent"],
            )
        )
    return np.stack(predictions, axis=0)


def _best_epoch(history: list[dict[str, float]]) -> int | None:
    if not history:
        return None
    best = min(history, key=lambda row: float(row.get("validation_loss", float("inf"))))
    return int(best.get("epoch", 0) or 0)


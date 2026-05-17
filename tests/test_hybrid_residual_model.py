"""Tests for the hybrid Mike-CC-plus-ANN-residual two-segment model.

These exercise:

* The core ``HybridResidualModel`` end-to-end on synthetic data where the
  ground truth is *exactly* Mike CC plus a smooth nonlinear residual. Hybrid
  must beat Mike-alone on the test split, and beat the ANN-alone baseline
  by a meaningful margin too.
* The unavailable path -- the model should refuse to train (status reports the
  Mike adapter reason) when the config has not confirmed conventions, and the
  classical models in the same suite must still complete.
* Sanity: when residuals are zero, the trained ANN component cannot hurt the
  Mike baseline by more than noise.
* The accompanying visualization module: each plotting helper must produce a
  valid PNG, and the bundle function must skip cleanly when an input slice
  is missing.

These tests skip themselves if ``torch`` is not available; the hybrid model
implementation will fall back to a status-only result in that case, which is
already covered by the unavailable-path tests for ``TorchANNModel``.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")  # noqa: F841 — skip the whole module if torch absent

from continuum_robot.modeling.two_segment.models import (
    HybridResidualModel,
    MikeConstantCurvatureModel,
    TorchANNModel,
    default_model_suite,
)
from continuum_robot.modeling.two_segment.physics import two_segment_constant_curvature_prediction
from continuum_robot.modeling.two_segment.visualization import (
    write_hybrid_before_after_scatter,
    write_hybrid_convergence_overlay,
    write_hybrid_improvement_bars,
    write_hybrid_residual_histograms,
    write_hybrid_visualization_bundle,
    write_hybrid_workspace_error_map,
)


# ---------------------------------------------------------------------------
# Test fixtures
# ---------------------------------------------------------------------------


def _wrap(value: object, *, source: str = "design") -> dict[str, object]:
    """Wrap a raw value in the {value, source} convention used by physics_models config."""
    return {"value": value, "source": source}


def _validated_mike_cc_config(
    *,
    segment_a_length_mm: float = 60.0,
    segment_b_length_mm: float = 60.0,
    tendon_radius_mm: float = 4.0,
) -> dict[str, object]:
    """Build a Mike CC config that the adapter reports as ``available``.

    All required keys are present and ``required_conventions_confirmed`` is
    True, so ``assess_mike_constant_curvature_status`` returns ``available``.
    """
    tendon_positions = [
        [tendon_radius_mm, 0.0],
        [0.0, tendon_radius_mm],
        [-tendon_radius_mm, 0.0],
        [0.0, -tendon_radius_mm],
    ]
    return {
        "physics_models": {
            "global": {
                "model_frame_convention": _wrap("z_along_segment"),
                "tendon_displacement_sign_convention": _wrap("positive_shortens_tendon"),
                "output_pose_frame": _wrap("robot"),
                "tangent_representation": _wrap("z_axis"),
                "segment_order_source": _wrap("config"),
            },
            "segments": {
                "segment_a": {
                    "segment_length_mm": _wrap(float(segment_a_length_mm)),
                    "tendon_positions_mm": _wrap(tendon_positions),
                },
                "segment_b": {
                    "segment_length_mm": _wrap(float(segment_b_length_mm)),
                    "tendon_positions_mm": _wrap(tendon_positions),
                },
            },
            "mike_constant_curvature": {
                "curvature_from_tendon_displacement": _wrap("explicit_lstsq_over_positions"),
                "required_conventions_confirmed": True,
                "validated_against_hardware": True,
            },
        }
    }


def _label_metadata_distal_only() -> dict[str, object]:
    """Label metadata for distal-tip XYZ only (the typical Mike CC-only target)."""
    return {
        "label_mode": "distal_xyz",
        "label_names": ["distal_x_mm", "distal_y_mm", "distal_z_mm"],
        "label_slices": {"distal_tip": {"position": [0, 1, 2]}},
        "orientation_available": False,
        "includes_intermediate_label": False,
    }


def _mike_pose_for_command(command_mm: np.ndarray, config: dict[str, object]) -> np.ndarray:
    pose = two_segment_constant_curvature_prediction(command_mm, config)
    return np.asarray(pose["distal_xyz"], dtype=float)


def _build_hybrid_synthetic_dataset(
    *,
    n_train: int,
    n_test: int,
    config: dict[str, object],
    residual_scale_mm: float,
    noise_sigma_mm: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Generate (X_train, y_train, X_test, y_test) where y = mike(X) + residual(X) + noise.

    The residual is a smooth multi-axis sinusoidal function of the command
    -- something the ANN can learn but Mike CC cannot model. ``noise_sigma_mm``
    adds independent Gaussian noise to every label coordinate so the test is
    not perfectly fitted (which would be uninteresting).
    """
    rng = np.random.default_rng(seed)
    n_total = n_train + n_test
    # Commands span ±2 mm displacement across all 8 servos; not all axes need
    # to be excited but mixing both segments gives Mike CC a non-trivial pose.
    commands = rng.uniform(-2.0, 2.0, size=(n_total, 8))
    # Mike's distal prediction for every command:
    mike_distal = np.asarray([_mike_pose_for_command(commands[index], config) for index in range(n_total)], dtype=float)
    # A smooth non-linear residual the ANN can learn:
    residual = np.column_stack(
        [
            residual_scale_mm * np.sin(0.6 * commands[:, 0] + 0.3 * commands[:, 4]),
            residual_scale_mm * np.cos(0.4 * commands[:, 1] - 0.5 * commands[:, 5]),
            residual_scale_mm * np.sin(0.3 * commands[:, 2] + 0.7 * commands[:, 6]) * 0.5,
        ]
    )
    noise = rng.normal(scale=noise_sigma_mm, size=mike_distal.shape)
    y = mike_distal + residual + noise
    return commands[:n_train], y[:n_train], commands[n_train:], y[n_train:]


# ---------------------------------------------------------------------------
# Core HybridResidualModel behavior
# ---------------------------------------------------------------------------


class TestHybridResidualModel:
    def test_hybrid_beats_mike_alone_on_residual_dataset(self, tmp_path: Path) -> None:
        config = _validated_mike_cc_config()
        label_metadata = _label_metadata_distal_only()
        X_train, y_train, X_test, y_test = _build_hybrid_synthetic_dataset(
            n_train=240, n_test=60, config=config, residual_scale_mm=2.0, noise_sigma_mm=0.05, seed=101
        )

        mike_only = MikeConstantCurvatureModel(config=config).fit_predict(
            X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test,
            model_dir=tmp_path / "mike", label_metadata=label_metadata,
        )
        hybrid = HybridResidualModel(
            config=config, hidden_layers=[64, 64], learning_rate=5e-3,
            epochs=200, patience=30, seed=11, batch_size=64,
        ).fit_predict(
            X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test,
            model_dir=tmp_path / "hybrid", label_metadata=label_metadata,
        )

        assert mike_only.status == "completed"
        assert hybrid.status == "completed"
        assert hybrid.predictions is not None
        mike_rmse = float(mike_only.metrics["xyz_rmse_mm"])
        hybrid_rmse = float(hybrid.metrics["xyz_rmse_mm"])
        # The residual is roughly the same magnitude as the noise budget plus
        # the actual sinusoidal signal, so Hybrid should cut the error by ~50%.
        assert hybrid_rmse < mike_rmse * 0.6, (
            f"Hybrid xyz_rmse={hybrid_rmse:.3f} mm should clearly beat Mike-only xyz_rmse={mike_rmse:.3f} mm"
        )
        # The reported improvement field should match the headline metric delta.
        improvement = hybrid.metrics.get("hybrid_xyz_rmse_improvement_over_mike_mm")
        assert improvement is not None
        assert abs(float(improvement) - (mike_rmse - hybrid_rmse)) < 1e-6

    def test_hybrid_beats_plain_ann_when_mike_captures_most_of_the_signal(self, tmp_path: Path) -> None:
        """When Mike CC alone is already a strong baseline, Hybrid should still beat plain ANN.

        Plain ANN has to relearn the geometric mapping from scratch, while
        Hybrid only has to learn the residual on top of Mike's already-good
        pose. With limited training data this difference is large.
        """
        config = _validated_mike_cc_config()
        label_metadata = _label_metadata_distal_only()
        X_train, y_train, X_test, y_test = _build_hybrid_synthetic_dataset(
            n_train=80, n_test=40, config=config, residual_scale_mm=1.0, noise_sigma_mm=0.05, seed=102
        )

        ann_only = TorchANNModel(hidden_layers=[64, 64], learning_rate=5e-3, epochs=200, patience=30, seed=13, batch_size=32).fit_predict(
            X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test,
            model_dir=tmp_path / "ann", label_metadata=label_metadata,
        )
        hybrid = HybridResidualModel(
            config=config, hidden_layers=[64, 64], learning_rate=5e-3,
            epochs=200, patience=30, seed=13, batch_size=32,
        ).fit_predict(
            X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test,
            model_dir=tmp_path / "hybrid", label_metadata=label_metadata,
        )

        assert ann_only.status == "completed"
        assert hybrid.status == "completed"
        ann_rmse = float(ann_only.metrics["xyz_rmse_mm"])
        hybrid_rmse = float(hybrid.metrics["xyz_rmse_mm"])
        assert hybrid_rmse < ann_rmse * 0.85, (
            f"Hybrid xyz_rmse={hybrid_rmse:.3f} mm should beat ANN-only xyz_rmse={ann_rmse:.3f} mm on small data"
        )

    def test_hybrid_does_not_significantly_hurt_when_no_residual_exists(self, tmp_path: Path) -> None:
        """Sanity: when y = mike(X) (no residual), Hybrid should not be much worse than Mike."""
        config = _validated_mike_cc_config()
        label_metadata = _label_metadata_distal_only()
        X_train, y_train, X_test, y_test = _build_hybrid_synthetic_dataset(
            n_train=120, n_test=40, config=config, residual_scale_mm=0.0, noise_sigma_mm=0.01, seed=103
        )

        mike_only = MikeConstantCurvatureModel(config=config).fit_predict(
            X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test,
            model_dir=tmp_path / "mike", label_metadata=label_metadata,
        )
        hybrid = HybridResidualModel(
            config=config, hidden_layers=[32, 32], learning_rate=1e-3,
            epochs=120, patience=20, seed=14, batch_size=32,
        ).fit_predict(
            X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test,
            model_dir=tmp_path / "hybrid", label_metadata=label_metadata,
        )

        assert mike_only.status == "completed"
        assert hybrid.status == "completed"
        mike_rmse = float(mike_only.metrics["xyz_rmse_mm"])
        hybrid_rmse = float(hybrid.metrics["xyz_rmse_mm"])
        # Hybrid is allowed some overhead because the ANN will fit a bit of noise,
        # but it should not be dramatically worse than the pure Mike baseline.
        assert hybrid_rmse < max(mike_rmse * 3.0, 0.1), (
            f"Hybrid xyz_rmse={hybrid_rmse:.3f} mm overshoot vs Mike xyz_rmse={mike_rmse:.3f} mm is too large"
        )

    def test_hybrid_persists_breakdown_and_loss_history(self, tmp_path: Path) -> None:
        config = _validated_mike_cc_config()
        label_metadata = _label_metadata_distal_only()
        X_train, y_train, X_test, y_test = _build_hybrid_synthetic_dataset(
            n_train=60, n_test=24, config=config, residual_scale_mm=0.6, noise_sigma_mm=0.05, seed=104
        )
        model_dir = tmp_path / "hybrid"
        result = HybridResidualModel(
            config=config, hidden_layers=[32, 32], epochs=40, patience=15, seed=15, batch_size=24,
        ).fit_predict(
            X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test,
            model_dir=model_dir, label_metadata=label_metadata,
        )

        assert result.status == "completed"
        assert (model_dir / "hybrid_residual_model.pt").exists()
        assert (model_dir / "hybrid_loss_history.json").exists()
        assert (model_dir / "hybrid_breakdown.json").exists()
        assert (model_dir / "hybrid_mike_predictions_test.npy").exists()
        assert (model_dir / "hybrid_ann_residual_predictions_test.npy").exists()
        breakdown = json.loads((model_dir / "hybrid_breakdown.json").read_text(encoding="utf-8"))
        for key in ("mike_only", "ann_residual_only", "hybrid"):
            assert key in breakdown
            assert "xyz_rmse_mm" in breakdown[key]
        payload = torch.load(model_dir / "hybrid_residual_model.pt", map_location="cpu", weights_only=False)
        assert payload["training_target"] == "mike_cc_residual"
        assert payload["input_dim"] == X_train.shape[1]
        assert payload["output_dim"] == y_train.shape[1]

    def test_hybrid_reports_unavailable_when_mike_config_missing(self, tmp_path: Path) -> None:
        # Empty config -> Mike CC adapter cannot evaluate -> Hybrid must skip cleanly
        result = HybridResidualModel(config={}, hidden_layers=[16, 16], epochs=10, patience=5, seed=0, batch_size=8).fit_predict(
            X_train=np.zeros((4, 8), dtype=float),
            y_train=np.zeros((4, 3), dtype=float),
            X_test=np.zeros((2, 8), dtype=float),
            y_test=np.zeros((2, 3), dtype=float),
            model_dir=tmp_path / "hybrid",
            label_metadata=_label_metadata_distal_only(),
        )

        assert result.status.startswith("unavailable")
        assert result.predictions is None
        assert "Mike" in (result.reason or "")
        assert (tmp_path / "hybrid" / "model_status.json").exists()

    def test_hybrid_requires_label_metadata_when_available(self, tmp_path: Path) -> None:
        config = _validated_mike_cc_config()
        with pytest.raises(ValueError, match="label metadata"):
            HybridResidualModel(config=config).fit_predict(
                X_train=np.zeros((4, 8), dtype=float),
                y_train=np.zeros((4, 3), dtype=float),
                X_test=np.zeros((2, 8), dtype=float),
                y_test=np.zeros((2, 3), dtype=float),
                model_dir=tmp_path / "hybrid",
                label_metadata=None,
            )


# ---------------------------------------------------------------------------
# default_model_suite registration
# ---------------------------------------------------------------------------


class TestHybridModelSuiteRegistration:
    def test_default_suite_includes_hybrid_when_requested(self) -> None:
        suite = default_model_suite(
            model_keys=["linear_baseline", "mike_constant_curvature", "ann", "hybrid_residual"],
            config={"random_seed": 0},
        )
        assert [model.model_key for model in suite] == [
            "linear_baseline",
            "mike_constant_curvature",
            "ann",
            "hybrid_residual",
        ]

    def test_hybrid_picks_up_ann_defaults_when_no_hybrid_section(self) -> None:
        suite = default_model_suite(
            model_keys=["hybrid_residual"],
            config={"random_seed": 7, "ann": {"hidden_layers": [16, 16], "learning_rate": 0.005}},
        )
        hybrid = suite[0]
        assert isinstance(hybrid, HybridResidualModel)
        assert hybrid.hidden_layers == [16, 16]
        assert hybrid.learning_rate == pytest.approx(0.005)
        assert hybrid.seed == 7

    def test_hybrid_section_overrides_ann_defaults(self) -> None:
        suite = default_model_suite(
            model_keys=["hybrid_residual"],
            config={
                "random_seed": 1,
                "ann": {"hidden_layers": [16, 16]},
                "hybrid_residual": {"hidden_layers": [64], "learning_rate": 1e-4, "epochs": 5, "patience": 2, "batch_size": 4, "seed": 99},
            },
        )
        hybrid = suite[0]
        assert isinstance(hybrid, HybridResidualModel)
        assert hybrid.hidden_layers == [64]
        assert hybrid.learning_rate == pytest.approx(1e-4)
        assert hybrid.epochs == 5
        assert hybrid.patience == 2
        assert hybrid.batch_size == 4
        assert hybrid.seed == 99


# ---------------------------------------------------------------------------
# Visualization helpers
# ---------------------------------------------------------------------------


def _png_is_valid(path: Path) -> bool:
    """Return True if the file starts with the PNG magic bytes."""
    if not path.exists():
        return False
    with path.open("rb") as handle:
        return handle.read(8) == b"\x89PNG\r\n\x1a\n"


class TestHybridVisualization:
    def _toy_predictions(self) -> tuple[np.ndarray, dict[str, np.ndarray]]:
        rng = np.random.default_rng(0)
        n = 24
        truth = rng.uniform(-10.0, 10.0, size=(n, 3))
        mike = truth + rng.normal(scale=1.0, size=truth.shape)
        ann = truth + rng.normal(scale=2.0, size=truth.shape)
        hybrid = truth + rng.normal(scale=0.4, size=truth.shape)
        return truth, {
            "mike_constant_curvature": mike,
            "ann": ann,
            "hybrid_residual": hybrid,
        }

    def test_scatter_writes_png(self, tmp_path: Path) -> None:
        truth, preds = self._toy_predictions()
        path = write_hybrid_before_after_scatter(
            output_path=tmp_path / "scatter.png",
            y_true=truth,
            predictions_by_model=preds,
            label_metadata=_label_metadata_distal_only(),
        )
        assert _png_is_valid(path)

    def test_residual_histograms_writes_png(self, tmp_path: Path) -> None:
        truth, preds = self._toy_predictions()
        path = write_hybrid_residual_histograms(
            output_path=tmp_path / "histograms.png",
            y_true=truth,
            predictions_by_model=preds,
            label_metadata=_label_metadata_distal_only(),
        )
        assert _png_is_valid(path)

    def test_workspace_error_map_writes_png(self, tmp_path: Path) -> None:
        truth, preds = self._toy_predictions()
        path = write_hybrid_workspace_error_map(
            output_path=tmp_path / "workspace.png",
            y_true=truth,
            predictions_by_model=preds,
            label_metadata=_label_metadata_distal_only(),
        )
        assert _png_is_valid(path)

    def test_convergence_overlay_writes_png(self, tmp_path: Path) -> None:
        history = [
            {"epoch": float(i), "train_loss": 1.0 / (i + 1), "validation_loss": 1.1 / (i + 1)}
            for i in range(1, 12)
        ]
        path = write_hybrid_convergence_overlay(
            output_path=tmp_path / "convergence.png",
            loss_history=history,
            mike_only_xyz_rmse_mm=1.7,
            hybrid_xyz_rmse_mm=0.8,
        )
        assert _png_is_valid(path)

    def test_improvement_bars_writes_png(self, tmp_path: Path) -> None:
        metrics = {
            "mike_constant_curvature": {"xyz_rmse_mm": 1.7, "p95_error_mm": 2.4, "max_error_mm": 3.5},
            "ann": {"xyz_rmse_mm": 1.3, "p95_error_mm": 1.9, "max_error_mm": 2.7},
            "hybrid_residual": {"xyz_rmse_mm": 0.8, "p95_error_mm": 1.2, "max_error_mm": 1.7},
        }
        path = write_hybrid_improvement_bars(
            output_path=tmp_path / "bars.png",
            metrics_by_model=metrics,
        )
        assert _png_is_valid(path)

    def test_bundle_produces_every_figure_when_inputs_present(self, tmp_path: Path) -> None:
        truth, preds = self._toy_predictions()
        history = [
            {"epoch": float(i), "train_loss": 0.5 / (i + 1), "validation_loss": 0.55 / (i + 1)}
            for i in range(1, 9)
        ]
        metrics = {
            "mike_constant_curvature": {"xyz_rmse_mm": 1.7, "p95_error_mm": 2.4, "max_error_mm": 3.5},
            "ann": {"xyz_rmse_mm": 1.3, "p95_error_mm": 1.9, "max_error_mm": 2.7},
            "hybrid_residual": {"xyz_rmse_mm": 0.8, "p95_error_mm": 1.2, "max_error_mm": 1.7},
        }
        paths = write_hybrid_visualization_bundle(
            output_dir=tmp_path / "bundle",
            y_true=truth,
            predictions_by_model=preds,
            label_metadata=_label_metadata_distal_only(),
            hybrid_loss_history=history,
            mike_only_xyz_rmse_mm=1.7,
            hybrid_xyz_rmse_mm=0.8,
            metrics_by_model=metrics,
        )
        for key in ("scatter", "residual_histograms", "workspace_error_map", "convergence", "improvement_bars"):
            assert key in paths
            assert _png_is_valid(paths[key])

    def test_bundle_skips_figures_with_missing_inputs(self, tmp_path: Path) -> None:
        truth, preds = self._toy_predictions()
        # No loss history and no metrics -> only data-driven figures appear.
        paths = write_hybrid_visualization_bundle(
            output_dir=tmp_path / "bundle",
            y_true=truth,
            predictions_by_model=preds,
            label_metadata=_label_metadata_distal_only(),
            hybrid_loss_history=None,
            metrics_by_model=None,
        )
        assert set(paths.keys()) == {"scatter", "residual_histograms", "workspace_error_map"}
        # And conversely: no predictions -> only the convergence + bars figures.
        paths_no_data = write_hybrid_visualization_bundle(
            output_dir=tmp_path / "bundle_no_data",
            y_true=np.zeros((0, 3), dtype=float),
            predictions_by_model={},
            label_metadata=_label_metadata_distal_only(),
            hybrid_loss_history=[{"epoch": 1.0, "train_loss": 0.1, "validation_loss": 0.12}],
            mike_only_xyz_rmse_mm=1.0,
            metrics_by_model={"mike_constant_curvature": {"xyz_rmse_mm": 1.0, "p95_error_mm": 1.5, "max_error_mm": 2.0}},
        )
        assert set(paths_no_data.keys()) == {"convergence", "improvement_bars"}


# ---------------------------------------------------------------------------
# End-to-end integration via run_two_segment_modeling
# ---------------------------------------------------------------------------


def _write_mike_cc_compatible_dataset_run(
    root: Path,
    *,
    name: str,
    config: dict[str, object],
    n_samples: int,
    residual_scale_mm: float,
    noise_sigma_mm: float,
    seed: int,
) -> Path:
    """Write a synthetic dataset run whose poses are Mike CC + small residual.

    Reuses the on-disk layout the modeling loader expects (same as
    ``tests/test_two_segment_modeling.py::_write_two_segment_dataset_run``)
    but the distal poses are generated from Mike CC rather than a generic
    linear map, so Hybrid (Mike CC + ANN residual) has a meaningful base
    prediction to lean on. Returns the run directory path.
    """
    run_dir = root / "data" / "experiments" / "two_segment_collect_pose_command_dataset" / name
    run_dir.mkdir(parents=True, exist_ok=True)
    metrics = {
        "dataset_type": "two_segment_collect_pose_command_dataset",
        "run_trust_mode": "thesis_trusted",
        "valid_for_model_training": False,
        "valid_for_two_segment_model_training": True,
        "valid_for_thesis_repeatability": False,
        "startup_artifact_provenance": {
            "accepted_all_8_startup": True,
            "artifact_path": str(run_dir / "all8_startup.json"),
            "artifact_sha256": "abc123",
        },
        "pose_label_summary": {
            "available_roles": ["distal_tip"],
            "missing_required_roles": [],
            "distal_pose_sample_count": int(n_samples),
        },
        "run_provenance": {
            "operating_mode": "dual_segment",
            "hardware_profile": "robot_8servo.yaml",
            "two_segment_foundation": {
                "command_schema": {"schema_version": "two_segment_command_v1"},
                "pose_schema": {"schema_version": "two_segment_pose_observation_v1"},
            },
        },
    }
    metadata = {
        "experiment_name": "two_segment_collect_pose_command_dataset",
        "trust_info": {
            "run_trust_mode": metrics["run_trust_mode"],
            "valid_for_model_training": False,
            "valid_for_two_segment_model_training": True,
            "valid_for_thesis_repeatability": False,
        },
        "provenance_info": {
            "operating_mode": "dual_segment",
            "hardware_profile": "robot_8servo.yaml",
            "two_segment_foundation": metrics["run_provenance"]["two_segment_foundation"],
        },
    }
    summary = {
        "experiment_name": "two_segment_collect_pose_command_dataset",
        "success": True,
        "status": "success",
        "sample_counts": {"total": int(n_samples)},
        "experiment_metrics": metrics,
    }
    (run_dir / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
    (run_dir / "summary.json").write_text(json.dumps(summary), encoding="utf-8")
    (run_dir / "config_snapshot.yaml").write_text("robot:\n  mode: dual_segment\n", encoding="utf-8")
    (run_dir / "two_segment_tracking_role_provenance.json").write_text(
        json.dumps({"pose_label_summary": metrics["pose_label_summary"]}), encoding="utf-8"
    )

    rng = np.random.default_rng(seed)
    with (run_dir / "samples.jsonl").open("w", encoding="utf-8") as handle:
        for index in range(int(n_samples)):
            command_mm = rng.uniform(-2.0, 2.0, size=8)
            mike_pose = _mike_pose_for_command(command_mm, config)
            residual = np.asarray(
                [
                    residual_scale_mm * float(np.sin(0.5 * command_mm[0] + 0.3 * command_mm[4])),
                    residual_scale_mm * float(np.cos(0.4 * command_mm[1] - 0.6 * command_mm[5])),
                    residual_scale_mm * 0.5 * float(np.sin(0.7 * command_mm[2])),
                ]
            )
            noise = rng.normal(scale=noise_sigma_mm, size=3)
            position = mike_pose + residual + noise
            matrix = [
                [1.0, 0.0, 0.0, float(position[0])],
                [0.0, 1.0, 0.0, float(position[1])],
                [0.0, 0.0, 1.0, float(position[2])],
                [0.0, 0.0, 0.0, 1.0],
            ]
            sample = {
                "wall_time_utc": f"2026-05-08T12:00:{index:02d}+00:00",
                "phase": "synthetic_hybrid_test",
                "step_index": index,
                "sample_index": index,
                "two_segment_command": {
                    "schema_version": "two_segment_command_v1",
                    "units": "cm",
                    "segments": {
                        "segment_a": [float(value) / 10.0 for value in command_mm[:4]],
                        "segment_b": [float(value) / 10.0 for value in command_mm[4:]],
                    },
                    "commanded_servo_ids": [1, 2, 3, 4, 5, 6, 7, 8],
                    "flat_command_cm": [float(value) / 10.0 for value in command_mm],
                },
                "pose_in_robot_frame": {"roles": {"distal_tip": {"T_robot_tip": matrix}}},
                "two_segment_pose": {"frame": "robot", "distal_tip_pose": {"T_robot_tip": matrix}},
                "extra": {
                    "record_kind": "two_segment_dataset_capture",
                    "run_trust_mode": "thesis_trusted",
                    "capture_accepted": True,
                    "command_success": True,
                    "valid_for_two_segment_model_training": True,
                    "command_units": "cm",
                    "commanded_servo_ids": [1, 2, 3, 4, 5, 6, 7, 8],
                    "startup_artifact_provenance": metrics["startup_artifact_provenance"],
                    "available_pose_roles": ["distal_tip"],
                    "missing_required_pose_roles": [],
                    "distal_only": True,
                    "includes_intermediate_pose": False,
                    "segment_order": ["segment_a", "segment_b"],
                    "segments": {
                        "segment_a": {"label": "Segment A", "role": "proximal", "servo_ids": [1, 2, 3, 4], "pair_mapping": [[1, 3], [2, 4]], "segment_order_index": 0},
                        "segment_b": {"label": "Segment B", "role": "distal", "servo_ids": [5, 6, 7, 8], "pair_mapping": [[5, 7], [6, 8]], "segment_order_index": 1},
                    },
                    "measured_servo_feedback": {
                        str(servo_id): {
                            "servo_id": int(servo_id),
                            "position_tick": 2048 + index + int(servo_id),
                            "signed_raw_current_ma": 50 + int(servo_id),
                            "load_proxy_ma": 50 + int(servo_id),
                        }
                        for servo_id in range(1, 9)
                    },
                },
            }
            handle.write(json.dumps(sample) + "\n")
    return run_dir


class TestHybridEndToEndPipeline:
    def test_run_two_segment_modeling_writes_hybrid_figures_when_hybrid_in_suite(self, tmp_path: Path) -> None:
        from continuum_robot.modeling.two_segment import (
            TwoSegmentModelingConfig,
            run_two_segment_modeling,
        )

        config = _validated_mike_cc_config()
        run_dir = _write_mike_cc_compatible_dataset_run(
            tmp_path,
            name="20260508_140000_two_segment_collect_pose_command_dataset",
            config=config,
            n_samples=40,
            residual_scale_mm=1.0,
            noise_sigma_mm=0.05,
            seed=201,
        )
        result = run_two_segment_modeling(
            run_dirs=[run_dir],
            project_root=tmp_path,
            config=TwoSegmentModelingConfig(
                model_keys=["mike_constant_curvature", "ann", "hybrid_residual"],
                model_config={
                    **config,
                    "ann": {"hidden_layers": [32, 32], "epochs": 40, "patience": 20, "batch_size": 16},
                    "hybrid_residual": {"hidden_layers": [32, 32], "epochs": 40, "patience": 20, "batch_size": 16},
                },
                output_root=str(tmp_path / "data" / "experiments"),
                random_seed=42,
            ),
        )
        summary = json.loads((result.output_dir / "summary.json").read_text(encoding="utf-8"))
        models = summary["experiment_metrics"]["models"]
        # Mike CC must be available and completed -- otherwise the hybrid path
        # would have skipped instead of training.
        assert models["mike_constant_curvature"]["status"] == "completed"
        assert models["hybrid_residual"]["status"] == "completed"
        for figure_name in (
            "two_segment_hybrid_before_after_scatter_report.png",
            "two_segment_hybrid_residual_histograms_report.png",
            "two_segment_hybrid_workspace_error_map_report.png",
            "two_segment_hybrid_convergence_overlay_report.png",
            "two_segment_hybrid_improvement_bars_report.png",
        ):
            assert (result.output_dir / figure_name).exists(), f"missing hybrid figure: {figure_name}"
        # Hybrid breakdown JSON should also be on disk for offline auditing.
        breakdown_path = result.output_dir / "models" / "hybrid_residual" / "hybrid_breakdown.json"
        assert breakdown_path.exists()
        breakdown = json.loads(breakdown_path.read_text(encoding="utf-8"))
        for key in ("mike_only", "ann_residual_only", "hybrid"):
            assert "xyz_rmse_mm" in breakdown[key]
        # On this synthetic dataset Hybrid should beat Mike-only on XYZ RMSE.
        assert float(breakdown["hybrid"]["xyz_rmse_mm"]) <= float(breakdown["mike_only"]["xyz_rmse_mm"])

    def test_run_two_segment_modeling_skips_hybrid_figures_when_mike_unavailable(self, tmp_path: Path) -> None:
        """No physics_models config -> Mike CC reports unavailable, Hybrid skips, no hybrid figures."""
        from tests.test_two_segment_modeling import _write_two_segment_dataset_run
        from continuum_robot.modeling.two_segment import (
            TwoSegmentModelingConfig,
            run_two_segment_modeling,
        )

        run_dir = _write_two_segment_dataset_run(tmp_path)
        result = run_two_segment_modeling(
            run_dirs=[run_dir],
            project_root=tmp_path,
            config=TwoSegmentModelingConfig(
                model_keys=["linear_baseline", "mike_constant_curvature", "hybrid_residual"],
                output_root=str(tmp_path / "data" / "experiments"),
                random_seed=5,
            ),
        )
        summary = json.loads((result.output_dir / "summary.json").read_text(encoding="utf-8"))
        models = summary["experiment_metrics"]["models"]
        assert models["mike_constant_curvature"]["status"].startswith("unavailable")
        assert models["hybrid_residual"]["status"].startswith("unavailable")
        # No hybrid figures should be on disk because the bundle is gated on a completed hybrid result.
        for figure_name in (
            "two_segment_hybrid_before_after_scatter_report.png",
            "two_segment_hybrid_residual_histograms_report.png",
            "two_segment_hybrid_workspace_error_map_report.png",
            "two_segment_hybrid_convergence_overlay_report.png",
            "two_segment_hybrid_improvement_bars_report.png",
        ):
            assert not (result.output_dir / figure_name).exists()

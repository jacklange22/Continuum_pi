"""Tests for the side-by-side ANN comparison backend.

The backend powers the "External Model Comparison" card on the Modeling tab:
operator picks Model A from the artifact dropdown, uploads a .pt for Model B,
and gets two 3D scatter plots colored by error vs the recorded ground-truth
tip. Tests here cover the loader/inference/figure-builder path without GUI.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

# These tests instantiate the real PyTorch model. Skip cleanly when torch isn't
# installed (CI/wheel-free environments).
torch = pytest.importorskip("torch")

from continuum_robot.modeling import ann_training as training_module
from continuum_robot.modeling.model_comparison import (
    LoadedModelHandle,
    ModelComparisonResult,
    _infer_architecture_from_state_dict,
    build_comparison_figure,
    load_model_for_comparison,
    run_side_by_side_comparison,
    save_comparison_png,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _write_modeling_dataset(tmp_path: Path, *, run_name: str = "ds_a", n: int = 64) -> Path:
    """Build a minimal but legal modeling dataset run with N accepted samples.

    The accepted rows carry resolved_cable_command_cm + tip_position_xyz_mm +
    tip_tangent_xyz, which is what prepare_legacy_ann_dataset reads.
    """
    run_dir = tmp_path / "data" / "experiments" / "collect_pose_command_dataset" / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "metadata.json").write_text(
        json.dumps(
            {
                "schema_version": "1.0",
                "experiment_name": "collect_pose_command_dataset",
                "run_id": run_name,
                "timestamp_utc": "2026-05-18T00:00:00+00:00",
            }
        ),
        encoding="utf-8",
    )
    (run_dir / "summary.json").write_text(
        json.dumps(
            {
                "schema_version": "1.0",
                "experiment_name": "collect_pose_command_dataset",
                "run_id": run_name,
                "success": True,
                "sample_counts": {"total": n},
                "status": "success",
                "experiment_metrics": {
                    "dataset_mode": "angular_test_mesh",
                    "accepted_sample_count": n,
                    "rejected_sample_count": 0,
                    "run_trust_mode": "thesis_trusted",
                    "valid_for_model_training": True,
                    "target_valid_sample_count": n,
                    "complete_training_row_count": n,
                    "mock_mode": False,
                },
            }
        ),
        encoding="utf-8",
    )
    # Spread the samples around a small XYZ workspace so the figure has shape.
    with (run_dir / "modeling_dataset_export.jsonl").open("w", encoding="utf-8") as handle:
        for i in range(n):
            angle = (i / max(n - 1, 1)) * 2.0 * np.pi
            cable = [0.5 * np.cos(angle), 0.5 * np.sin(angle), 0.4, 0.3]
            xyz = [10.0 * np.cos(angle), 10.0 * np.sin(angle), 5.0 + 0.1 * i]
            tangent = [0.0, 0.0, 1.0]
            handle.write(
                json.dumps(
                    {
                        "sequence_index": i,
                        "step_index": i // 4,
                        "sample_index": i,
                        "accepted": True,
                        "resolved_cable_command_cm": cable,
                        "tip_position_xyz_mm": xyz,
                        "tip_tangent_xyz": tangent,
                    }
                )
                + "\n"
            )
    return run_dir


def _train_and_save_tiny_artifact(
    tmp_path: Path,
    *,
    name: str = "model_a",
    hidden_layers: list[int] | None = None,
    epochs: int = 2,
) -> Path:
    """Build a deterministic tiny artifact dir for use as Model A.

    We don't actually run training; we construct a randomly-initialized model and
    save its state_dict + a legal training_metadata.json. The artifact loader
    only needs the metadata to declare the architecture.
    """
    layers = list(hidden_layers or [8, 8])
    artifact_dir = tmp_path / "data" / "models" / "ann" / name
    artifact_dir.mkdir(parents=True, exist_ok=True)
    model = training_module._build_legacy_ann_model(
        torch=torch,
        input_dim=4,
        output_dim=3,
        hidden_layers=layers,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )
    torch.save(model.state_dict(), artifact_dir / "model.pt")
    (artifact_dir / "training_metadata.json").write_text(
        json.dumps(
            {
                "artifact_kind": "legacy_ann_xyz_v1",
                "created_at_utc": "2026-05-18T00:00:00+00:00",
                "status": "completed",
                "model": {
                    "input_dim": 4,
                    "output_dim": 3,
                    "hidden_layers": layers,
                    "dtype": "float32",
                    "output_target": "xyz",
                },
                "training": {"epochs_completed": int(epochs), "best_validation_loss": 0.1},
                "dataset": {"run_name": "fake_train_run", "path": str(tmp_path / "fake_run")},
                "backend": {"selected_backend": "cpu"},
                "files": {"model_path": str(artifact_dir / "model.pt")},
            }
        ),
        encoding="utf-8",
    )
    (artifact_dir / "training_config.json").write_text(json.dumps({"epochs": int(epochs)}), encoding="utf-8")
    (artifact_dir / "split_manifest.json").write_text(json.dumps({"test_indices": [1]}), encoding="utf-8")
    (artifact_dir / "loss_history.csv").write_text("epoch,train_loss,validation_loss,elapsed_s\n1,0.5,0.6,0.1\n", encoding="utf-8")
    return artifact_dir


def _save_bare_state_dict(tmp_path: Path, *, name: str, hidden_layers: list[int]) -> Path:
    """Save a state_dict-only .pt with no sidecar metadata — the 'old .pt' path."""
    out_path = tmp_path / "uploads" / f"{name}.pt"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    model = training_module._build_legacy_ann_model(
        torch=torch,
        input_dim=4,
        output_dim=3,
        hidden_layers=hidden_layers,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )
    torch.save(model.state_dict(), out_path)
    return out_path


# ---------------------------------------------------------------------------
# _infer_architecture_from_state_dict
# ---------------------------------------------------------------------------


def test_infer_architecture_recovers_legacy_widths() -> None:
    """Given a state_dict from the standard nn.Sequential(input, hidden*, output)
    layout, the inferrer should recover exact input/hidden/output widths."""
    model = training_module._build_legacy_ann_model(
        torch=torch,
        input_dim=4,
        output_dim=3,
        hidden_layers=[16, 8, 12],
        device=torch.device("cpu"),
        dtype=torch.float32,
    )
    state_dict = {k: v.cpu() for k, v in model.state_dict().items()}
    input_dim, hidden, output_dim = _infer_architecture_from_state_dict(state_dict)
    assert input_dim == 4
    assert hidden == [16, 8, 12]
    assert output_dim == 3


def test_infer_architecture_rejects_unknown_layer_names() -> None:
    """An unfamiliar layer name (e.g. a model not built via _build_legacy_ann_model)
    should raise — silently mis-guessing the architecture would give bad
    predictions and bad plots."""
    state_dict = {"my_custom_layer.weight": torch.zeros(8, 4)}
    with pytest.raises(ValueError, match="legacy nn.Sequential"):
        _infer_architecture_from_state_dict(state_dict)


# ---------------------------------------------------------------------------
# load_model_for_comparison
# ---------------------------------------------------------------------------


def test_load_artifact_directory_returns_full_provenance(tmp_path: Path) -> None:
    """When the operator picks an artifact directory, the loader should use the
    full metadata path — scalers (if any) included, no warning attached."""
    artifact_dir = _train_and_save_tiny_artifact(tmp_path, hidden_layers=[8, 8])
    handle = load_model_for_comparison(artifact_dir, label_prefix="A: ")
    assert handle.label.startswith("A: ")
    assert handle.artifact_details is not None
    assert handle.hidden_layers == [8, 8]
    assert handle.input_dim == 4
    assert handle.output_dim == 3
    assert handle.output_target == "xyz"
    assert handle.warnings == ()


def test_load_bare_pt_infers_architecture_and_warns(tmp_path: Path) -> None:
    """When the operator uploads a raw .pt with no sidecar metadata, the loader
    should infer the architecture from state_dict shapes AND attach a warning
    so the figure carries the caveat."""
    bare_path = _save_bare_state_dict(tmp_path, name="legacy_unscaled", hidden_layers=[24, 24])
    handle = load_model_for_comparison(bare_path, label_prefix="B: ")
    assert handle.label.startswith("B: ")
    assert handle.artifact_details is None
    assert handle.hidden_layers == [24, 24]
    assert handle.input_dim == 4
    assert handle.output_dim == 3
    assert handle.output_target == "xyz"
    assert handle.scalers is None
    assert any("inferred" in w.lower() for w in handle.warnings)


def test_load_bare_pt_with_sibling_metadata_uses_metadata(tmp_path: Path) -> None:
    """If the operator's .pt happens to sit next to its training_metadata.json
    (the normal artifact layout), the loader should promote it to the full
    metadata path. No warning, scalers honored, etc."""
    artifact_dir = _train_and_save_tiny_artifact(tmp_path, name="sib_artifact", hidden_layers=[12, 12])
    # The .pt lives at artifact_dir/model.pt; metadata is the sibling.
    handle = load_model_for_comparison(artifact_dir / "model.pt", label_prefix="B: ")
    assert handle.artifact_details is not None
    assert handle.hidden_layers == [12, 12]
    assert handle.warnings == ()


def test_load_rejects_inverse_models(tmp_path: Path) -> None:
    """Inverse (xyz → cable) artifacts have output_dim=4 input_dim=3; the comparison
    plot is forward cable → tip XYZ and would dim-mismatch on the dataset's cable
    inputs. Reject upstream with a clear message."""
    artifact_dir = tmp_path / "data" / "models" / "ann" / "inverse_v1"
    artifact_dir.mkdir(parents=True)
    model = training_module._build_legacy_ann_model(
        torch=torch,
        input_dim=3,
        output_dim=4,
        hidden_layers=[8, 8],
        device=torch.device("cpu"),
        dtype=torch.float32,
    )
    torch.save(model.state_dict(), artifact_dir / "model.pt")
    (artifact_dir / "training_metadata.json").write_text(
        json.dumps(
            {
                "artifact_kind": "legacy_ann_inverse_xyz_to_cable_v1",
                "created_at_utc": "2026-05-18T00:00:00+00:00",
                "status": "completed",
                "model": {
                    "input_dim": 3,
                    "output_dim": 4,
                    "hidden_layers": [8, 8],
                    "dtype": "float32",
                    "output_target": "cable_from_xyz",
                },
                "training": {"epochs_completed": 1, "best_validation_loss": 0.1},
                "dataset": {"run_name": "fake", "path": str(tmp_path)},
                "backend": {"selected_backend": "cpu"},
                "files": {"model_path": str(artifact_dir / "model.pt")},
            }
        ),
        encoding="utf-8",
    )
    (artifact_dir / "training_config.json").write_text("{}", encoding="utf-8")
    (artifact_dir / "split_manifest.json").write_text("{}", encoding="utf-8")
    (artifact_dir / "loss_history.csv").write_text("epoch,train_loss,validation_loss,elapsed_s\n", encoding="utf-8")
    with pytest.raises(ValueError, match="inverse"):
        load_model_for_comparison(artifact_dir)


def test_load_rejects_missing_path(tmp_path: Path) -> None:
    """Be loud and obvious when the operator points the picker at a stale path."""
    with pytest.raises(FileNotFoundError):
        load_model_for_comparison(tmp_path / "does_not_exist.pt")


# ---------------------------------------------------------------------------
# run_side_by_side_comparison
# ---------------------------------------------------------------------------


def test_comparison_end_to_end_produces_aligned_shapes(tmp_path: Path) -> None:
    """End-to-end: two artifacts + dataset → result with matching N for both
    models and a non-zero shared color max."""
    artifact_a = _train_and_save_tiny_artifact(tmp_path, name="model_a", hidden_layers=[8, 8])
    artifact_b = _train_and_save_tiny_artifact(tmp_path, name="model_b", hidden_layers=[16, 16])
    dataset = _write_modeling_dataset(tmp_path, run_name="comp_ds", n=64)
    result = run_side_by_side_comparison(
        model_a_path=artifact_a,
        model_b_path=artifact_b,
        dataset_path=dataset,
    )
    n = result.actuals_xyz_mm.shape[0]
    assert n == 64
    assert result.a_predictions_xyz_mm.shape == (n, 3)
    assert result.b_predictions_xyz_mm.shape == (n, 3)
    assert result.a_errors_mm.shape == (n,)
    assert result.b_errors_mm.shape == (n,)
    assert result.shared_color_max_mm == pytest.approx(
        max(result.a_errors_mm.max(), result.b_errors_mm.max())
    )
    # Stats sanity: mean ≤ p95 ≤ max for each model.
    for stats in (result.a_stats, result.b_stats):
        assert stats.sample_count == n
        assert stats.mean_mm <= stats.p95_mm + 1e-9
        assert stats.p95_mm <= stats.max_mm + 1e-9


def test_comparison_supports_bare_pt_upload(tmp_path: Path) -> None:
    """Model A from artifact dir + Model B from a bare .pt upload should work
    end-to-end. The bare .pt warning should propagate into result.warnings."""
    artifact_a = _train_and_save_tiny_artifact(tmp_path, name="model_a", hidden_layers=[8, 8])
    bare_b = _save_bare_state_dict(tmp_path, name="legacy_b", hidden_layers=[16, 16])
    dataset = _write_modeling_dataset(tmp_path, run_name="comp_ds_bare", n=32)
    result = run_side_by_side_comparison(
        model_a_path=artifact_a,
        model_b_path=bare_b,
        dataset_path=dataset,
    )
    assert result.model_b.artifact_details is None
    assert any("inferred" in w.lower() for w in result.warnings)


def test_comparison_refuses_tiny_dataset(tmp_path: Path) -> None:
    """The plot is meant to show workspace coverage. A dataset with <10 samples
    can't show coverage, so the backend should refuse rather than producing a
    misleading figure."""
    artifact_a = _train_and_save_tiny_artifact(tmp_path, name="ma", hidden_layers=[8, 8])
    artifact_b = _train_and_save_tiny_artifact(tmp_path, name="mb", hidden_layers=[8, 8])
    dataset = _write_modeling_dataset(tmp_path, run_name="tiny_ds", n=5)
    with pytest.raises(ValueError, match="workspace"):
        run_side_by_side_comparison(
            model_a_path=artifact_a,
            model_b_path=artifact_b,
            dataset_path=dataset,
        )


# ---------------------------------------------------------------------------
# Figure / save
# ---------------------------------------------------------------------------


def test_build_figure_has_two_3d_axes_and_one_colorbar(tmp_path: Path) -> None:
    """The figure must carry exactly two 3D axes (the two model panels) plus a
    shared colorbar axis. Smoke check guards against accidental refactoring."""
    artifact_a = _train_and_save_tiny_artifact(tmp_path, name="ma", hidden_layers=[8, 8])
    artifact_b = _train_and_save_tiny_artifact(tmp_path, name="mb", hidden_layers=[8, 8])
    dataset = _write_modeling_dataset(tmp_path, run_name="fig_ds", n=32)
    result = run_side_by_side_comparison(
        model_a_path=artifact_a, model_b_path=artifact_b, dataset_path=dataset
    )
    figure = build_comparison_figure(result)
    try:
        # Two Axes3D + one colorbar axis (matplotlib classifies the colorbar as a
        # separate Axes).
        three_d = [ax for ax in figure.axes if ax.name == "3d"]
        non_3d = [ax for ax in figure.axes if ax.name != "3d"]
        assert len(three_d) == 2
        assert len(non_3d) >= 1  # the colorbar
    finally:
        import matplotlib.pyplot as plt

        plt.close(figure)


def test_save_comparison_png_writes_a_file(tmp_path: Path) -> None:
    """save_comparison_png writes a PNG at the requested path."""
    artifact_a = _train_and_save_tiny_artifact(tmp_path, name="ma", hidden_layers=[8, 8])
    artifact_b = _train_and_save_tiny_artifact(tmp_path, name="mb", hidden_layers=[8, 8])
    dataset = _write_modeling_dataset(tmp_path, run_name="png_ds", n=32)
    result = run_side_by_side_comparison(
        model_a_path=artifact_a, model_b_path=artifact_b, dataset_path=dataset
    )
    target = tmp_path / "out" / "comparison.png"
    saved = save_comparison_png(result, target, dpi=80)
    assert saved == target
    assert target.exists()
    assert target.stat().st_size > 1024  # nontrivial PNG


def test_figure_colorbar_uses_shared_color_max(tmp_path: Path) -> None:
    """The colorbar's max must equal shared_color_max_mm so that the same color
    on both panels means the same mm of error — the operator's explicit ask."""
    artifact_a = _train_and_save_tiny_artifact(tmp_path, name="ma", hidden_layers=[8, 8])
    artifact_b = _train_and_save_tiny_artifact(tmp_path, name="mb", hidden_layers=[16, 16])
    dataset = _write_modeling_dataset(tmp_path, run_name="cb_ds", n=32)
    result = run_side_by_side_comparison(
        model_a_path=artifact_a, model_b_path=artifact_b, dataset_path=dataset
    )
    figure = build_comparison_figure(result)
    try:
        # The scatter on the first 3D axis carries the norm we want to inspect.
        ax_a = [ax for ax in figure.axes if ax.name == "3d"][0]
        scatter = ax_a.collections[0]
        clim = scatter.get_clim()
        # vmin = 0, vmax = shared_color_max_mm (within float tolerance)
        assert clim[0] == pytest.approx(0.0)
        assert clim[1] == pytest.approx(result.shared_color_max_mm, rel=1e-6)
    finally:
        import matplotlib.pyplot as plt

        plt.close(figure)

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


# ---------------------------------------------------------------------------
# Cable-input unit auto-detection
# ---------------------------------------------------------------------------


def test_autodetect_picks_x10_for_mm_trained_model(tmp_path: Path) -> None:
    """Regression for the May-2024 continuum_jack legacy .pt collapse bug.

    Hand-build a model whose hidden-layer weights are large enough that
    cm-scale (±0.5 unit) inputs all fall in the same near-bias regime — only
    the ×10 scale produces meaningful spread. The loader must auto-detect
    ×10 and the bare-.pt warning must surface that.
    """
    from continuum_robot.modeling.model_comparison import (
        _AUTODETECT_MIN_HEALTHY_SPREAD_MM,
        _autodetect_input_scale,
    )
    from continuum_robot.modeling import ann_training as training_module

    model = training_module._build_legacy_ann_model(
        torch=torch,
        input_dim=4,
        output_dim=3,
        hidden_layers=[16, 16],
        device=torch.device("cpu"),
        dtype=torch.float64,
    )
    # Hand-craft weights so the first Linear has ~unit-scale weights but a
    # large negative bias — cm inputs (±0.5) can't overcome the bias to
    # activate the ReLU; mm-scale inputs (×10 → ±5) can.
    with torch.no_grad():
        model.input.weight.fill_(1.0)
        model.input.bias.fill_(-3.0)  # ReLU dead unless input contributes > 3
        model.hidden1.weight.fill_(0.5)
        model.hidden1.bias.fill_(0.0)
        model.output.weight.fill_(2.0)
        model.output.bias.fill_(0.0)
    model.eval()
    multiplier, info = _autodetect_input_scale(
        model=model, torch=torch, dtype=torch.float64
    )
    assert multiplier == 10.0, f"expected ×10 (smallest healthy), got ×{multiplier}; info: {info}"
    assert info["scale_1.0_spread_mm"] < _AUTODETECT_MIN_HEALTHY_SPREAD_MM
    assert info["scale_10.0_spread_mm"] >= _AUTODETECT_MIN_HEALTHY_SPREAD_MM


def test_autodetect_picks_x1_for_cm_trained_model(tmp_path: Path) -> None:
    """When the model is already trained on cm-scale inputs (the new
    convention), auto-detection should pick ×1 — not over-scale. Hand-build a
    model whose biases are small enough that ±0.5 inputs activate the ReLU."""
    from continuum_robot.modeling.model_comparison import _autodetect_input_scale
    from continuum_robot.modeling import ann_training as training_module

    model = training_module._build_legacy_ann_model(
        torch=torch,
        input_dim=4,
        output_dim=3,
        hidden_layers=[16, 16],
        device=torch.device("cpu"),
        dtype=torch.float64,
    )
    with torch.no_grad():
        # Larger weights but near-zero bias → cm-scale inputs trigger varied ReLU outputs.
        model.input.weight.uniform_(-5.0, 5.0)
        model.input.bias.fill_(0.0)
        model.hidden1.weight.uniform_(-2.0, 2.0)
        model.hidden1.bias.fill_(0.0)
        model.output.weight.uniform_(-1.0, 1.0)
        model.output.bias.fill_(0.0)
    model.eval()
    multiplier, info = _autodetect_input_scale(
        model=model, torch=torch, dtype=torch.float64
    )
    assert multiplier == 1.0, (
        f"expected ×1 (smallest healthy when cm already works), got ×{multiplier}; info: {info}"
    )


def test_loaded_handle_applies_multiplier_in_inference(tmp_path: Path) -> None:
    """End-to-end: a model that needs ×10 scaling must produce non-collapsed
    predictions when ``_predict_with_handle`` is fed cm-scale cable inputs."""
    from continuum_robot.modeling.model_comparison import (
        _predict_with_handle,
        load_model_for_comparison,
    )
    from continuum_robot.modeling import ann_training as training_module

    # Save a hand-crafted "mm-trained" model as a bare .pt.
    model = training_module._build_legacy_ann_model(
        torch=torch,
        input_dim=4,
        output_dim=3,
        hidden_layers=[16, 16],
        device=torch.device("cpu"),
        dtype=torch.float64,
    )
    with torch.no_grad():
        model.input.weight.fill_(1.0)
        model.input.bias.fill_(-3.0)
        model.hidden1.weight.fill_(0.5)
        model.hidden1.bias.fill_(0.0)
        model.output.weight.fill_(2.0)
        model.output.bias.fill_(0.0)
    bare_pt = tmp_path / "uploads" / "mm_trained.pt"
    bare_pt.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), bare_pt)

    handle = load_model_for_comparison(bare_pt, label_prefix="legacy: ")
    assert handle.input_scale_multiplier == 10.0
    # Predictions on cm-scale inputs should now span the workspace.
    cable_cm = np.array(
        [[0.5, 0.0, -0.3, 0.2], [-0.5, 0.0, 0.3, -0.2], [1.5, 0.0, -1.5, 0.0]]
    )
    preds = _predict_with_handle(handle, inputs=cable_cm)
    spread = float(np.sum(np.std(preds[:, :3], axis=0)))
    assert spread >= 5.0, (
        f"predictions still collapsed (spread={spread:.2f} mm) even though "
        f"handle.input_scale_multiplier=×{handle.input_scale_multiplier}"
    )


# ---------------------------------------------------------------------------
# Legacy continuum_jack kinematic_*.dat loader + per-slot test data
# ---------------------------------------------------------------------------


def test_legacy_dat_loader_unit_conversion_and_filtering(tmp_path: Path) -> None:
    """Audit the math: cable_mm → cable_cm is exactly ÷10; xyz unchanged;
    rows whose XYZ are all (near-)zero are dropped as sentinels."""
    from continuum_robot.modeling.model_comparison import _load_legacy_kinematic_dat

    # Synthesize a small .dat with three live rows + one zero-xyz sentinel.
    dat = tmp_path / "kinematic_2024_07_16_21_33_12.dat"
    dat.write_text(
        "DATE: 2024-7-16\n"
        "TIME: 21-33-12\n"
        "NUM_CABLES: 4\n"
        "num_coils: 1\n"
        "NUM_MEASUREMENTS: 4\n"
        "---\n"
        # idx, c0, c1, c2, c3,    px,    py,    pz, tx, ty, tz
        "0, 10.0,  0.0,  0.0,  0.0,  1.0,  2.0,  3.0, 0.0, 0.0, 1.0\n"
        # zero-xyz sentinel — should be dropped:
        "1,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0, 0.0, 0.0, 1.0\n"
        "2,-12.5,  6.25, -3.0, 0.5, -10.0, 5.0, 30.0, 0.0, 0.0, 1.0\n"
        "3,  1.5, -2.5,  0.0, 0.75,  0.5, -1.0, 50.0, 0.0, 0.0, 1.0\n",
        encoding="utf-8",
    )
    cable_cm, xyz_mm, name = _load_legacy_kinematic_dat(dat)
    assert name == "kinematic_2024_07_16_21_33_12"
    # Sentinel dropped: 4 raw rows → 3 valid.
    assert cable_cm.shape == (3, 4)
    assert xyz_mm.shape == (3, 3)
    # Cable mm → cm: exactly ÷10.
    np.testing.assert_allclose(cable_cm[0], np.array([10.0, 0.0, 0.0, 0.0]) / 10.0)
    np.testing.assert_allclose(cable_cm[1], np.array([-12.5, 6.25, -3.0, 0.5]) / 10.0)
    np.testing.assert_allclose(cable_cm[2], np.array([1.5, -2.5, 0.0, 0.75]) / 10.0)
    # XYZ unchanged.
    np.testing.assert_allclose(xyz_mm[0], [1.0, 2.0, 3.0])
    np.testing.assert_allclose(xyz_mm[1], [-10.0, 5.0, 30.0])
    np.testing.assert_allclose(xyz_mm[2], [0.5, -1.0, 50.0])


def test_legacy_dat_loader_rejects_no_separator(tmp_path: Path) -> None:
    """Without the ``---`` header separator we can't tell where data starts."""
    from continuum_robot.modeling.model_comparison import _load_legacy_kinematic_dat

    bad = tmp_path / "bad.dat"
    bad.write_text("DATE: x\n0,0,0,0,0,1,2,3,0,0,1\n", encoding="utf-8")
    with pytest.raises(ValueError, match="separator"):
        _load_legacy_kinematic_dat(bad)


def test_legacy_dat_loader_rejects_all_zero_xyz(tmp_path: Path) -> None:
    """If every row has zero XYZ (failed measurements), we should raise rather
    than return an empty array."""
    from continuum_robot.modeling.model_comparison import _load_legacy_kinematic_dat

    dat = tmp_path / "all_zero.dat"
    dat.write_text(
        "DATE: x\nTIME: x\nNUM_CABLES: 4\nnum_coils: 1\nNUM_MEASUREMENTS: 2\n---\n"
        "0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0\n"
        "1, 2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="no valid rows"):
        _load_legacy_kinematic_dat(dat)


def test_load_comparison_test_data_dispatch_by_extension(tmp_path: Path) -> None:
    """The dispatcher routes ``.dat`` to the legacy loader and other paths
    through ``prepare_legacy_ann_dataset``. Synthesize both and confirm."""
    from continuum_robot.modeling.model_comparison import _load_comparison_test_data

    # Legacy .dat
    dat = tmp_path / "leg.dat"
    dat.write_text(
        "DATE:x\nTIME:x\nNUM_CABLES:4\nnum_coils:1\nNUM_MEASUREMENTS:1\n---\n"
        "0, 5.0, -5.0, 2.5, -2.5, 1.0, 2.0, 3.0, 0.0, 0.0, 1.0\n",
        encoding="utf-8",
    )
    a_cable, a_xyz, a_name = _load_comparison_test_data(dat)
    assert a_cable.shape == (1, 4)
    np.testing.assert_allclose(a_cable[0], [0.5, -0.5, 0.25, -0.25])  # mm → cm
    assert "leg" in a_name


def test_run_side_by_side_uses_per_slot_test_datasets(tmp_path: Path) -> None:
    """Slot A points at a synthetic dataset_A, Slot B at dataset_B with a
    DIFFERENT workspace. Each panel's actuals must match its own dataset; the
    warning must mention the split; the suptitle must say "each model on its
    own test data"; per-panel axis limits must not be globally pooled."""
    from continuum_robot.modeling.model_comparison import (
        run_side_by_side_comparison,
        build_comparison_figure,
    )

    # Build two distinct .dat files: A has tip XYZ near origin, B has it far
    # away. This catches the "shared axis limit" bug: with per-panel limits
    # each panel's bounding box is local; with global pooling, panel A would
    # squash to a corner of a giant box.
    def _write_dat(p: Path, xyz_offset: list[float]) -> None:
        rows = [
            "DATE:x\nTIME:x\nNUM_CABLES:4\nnum_coils:1\nNUM_MEASUREMENTS:20\n---\n"
        ]
        for i in range(20):
            cx = 1.0 + 0.1 * i
            rows.append(
                f"{i},{cx:.3f},{(-cx):.3f},{(cx/2):.3f},{(-cx/2):.3f},"
                f"{xyz_offset[0] + 0.5 * i:.3f},{xyz_offset[1]:.3f},{xyz_offset[2]:.3f},0,0,1\n"
            )
        p.write_text("".join(rows), encoding="utf-8")

    dat_a = tmp_path / "ds_a.dat"
    dat_b = tmp_path / "ds_b.dat"
    _write_dat(dat_a, xyz_offset=[0.0, 0.0, 30.0])
    _write_dat(dat_b, xyz_offset=[1000.0, 1000.0, 1000.0])  # very far away

    # Two distinct models (so the labels differ).
    model_a = _save_bare_state_dict(tmp_path, name="md_a", hidden_layers=[8, 8])
    model_b = _save_bare_state_dict(tmp_path, name="md_b", hidden_layers=[16, 16])

    result = run_side_by_side_comparison(
        model_a_path=model_a,
        model_b_path=model_b,
        dataset_path=dat_a,  # fallback (unused since both test_*_path set)
        test_a_path=dat_a,
        test_b_path=dat_b,
    )
    # Per-slot dataset names recorded.
    assert "ds_a" in result.a_dataset_run_name
    assert "ds_b" in result.b_dataset_run_name
    # Per-panel actuals are NOT the same array.
    assert not np.array_equal(result.a_actuals_xyz_mm, result.b_actuals_xyz_mm)
    # B's actuals are at the far offset; A's are near origin.
    assert result.a_actuals_xyz_mm[:, 0].max() < 100.0
    assert result.b_actuals_xyz_mm[:, 0].min() > 500.0
    # Warning mentions the split.
    assert any("DIFFERENT test data" in w for w in result.warnings)
    # Figure: per-panel axis limits — A's xmax should be << B's xmin.
    fig = build_comparison_figure(result)
    try:
        three_d_axes = [ax for ax in fig.axes if ax.name == "3d"]
        ax_a, ax_b = three_d_axes
        a_xmax = ax_a.get_xlim()[1]
        b_xmax = ax_b.get_xlim()[1]
        # Panel B's actuals span x ∈ [1000, 1010], so b_xmax must reach ≥1000.
        assert b_xmax >= 1000.0, f"Panel B's xmax should ≥1000; got {b_xmax}"
        # Panel A's actuals are <50 mm. Per-panel limits keep A.xmax local.
        # Globally pooled limits would make a_xmax == b_xmax (both ≥1000).
        assert a_xmax < 100.0, (
            f"Panel A's xmax should be local to its own actuals (<100); "
            f"got {a_xmax}. Globally pooled limits would make this fail."
        )
        assert b_xmax - a_xmax > 800.0, (
            f"Per-panel axis limits should be distinct (≥800 mm apart): "
            f"a_xmax={a_xmax}, b_xmax={b_xmax}"
        )
        # Suptitle calls out the split.
        suptitle_text = fig._suptitle.get_text() if fig._suptitle is not None else ""  # noqa: SLF001
        assert "each model on its own test data" in suptitle_text
    finally:
        import matplotlib.pyplot as plt
        plt.close(fig)


def test_run_side_by_side_back_compat_shared_dataset(tmp_path: Path) -> None:
    """When neither test_a_path nor test_b_path is set, both panels read from
    ``dataset_path`` (old behavior preserved). No split warning, no per-panel
    "test data: …" subtitle."""
    from continuum_robot.modeling.model_comparison import (
        run_side_by_side_comparison,
        build_comparison_figure,
    )

    dat = tmp_path / "shared.dat"
    rows = ["DATE:x\nTIME:x\nNUM_CABLES:4\nnum_coils:1\nNUM_MEASUREMENTS:20\n---\n"]
    for i in range(20):
        rows.append(f"{i},1,2,3,4,{0.5*i:.2f},{0.3*i:.2f},{50-i:.2f},0,0,1\n")
    dat.write_text("".join(rows), encoding="utf-8")

    model_a = _save_bare_state_dict(tmp_path, name="bc_a", hidden_layers=[8, 8])
    model_b = _save_bare_state_dict(tmp_path, name="bc_b", hidden_layers=[8, 8])

    result = run_side_by_side_comparison(
        model_a_path=model_a,
        model_b_path=model_b,
        dataset_path=dat,
    )
    assert result.a_dataset_run_name == result.b_dataset_run_name
    # No split warning (only the bare-.pt warnings from auto-detect).
    assert not any("DIFFERENT test data" in w for w in result.warnings)
    # The back-compat ``actuals_xyz_mm`` property works because both panels share data.
    np.testing.assert_array_equal(result.actuals_xyz_mm, result.a_actuals_xyz_mm)
    np.testing.assert_array_equal(result.a_actuals_xyz_mm, result.b_actuals_xyz_mm)


def test_actuals_xyz_property_raises_when_panels_diverge(tmp_path: Path) -> None:
    """The legacy ``result.actuals_xyz_mm`` property must explicitly raise
    when the two panels use different test datasets — otherwise back-compat
    callers would silently get Slot A's array as if it represented both."""
    from continuum_robot.modeling.model_comparison import (
        run_side_by_side_comparison,
    )

    dat_a = tmp_path / "div_a.dat"
    dat_b = tmp_path / "div_b.dat"
    for p, offset in ((dat_a, 0.0), (dat_b, 100.0)):
        rows = ["DATE:x\nTIME:x\nNUM_CABLES:4\nnum_coils:1\nNUM_MEASUREMENTS:20\n---\n"]
        for i in range(20):
            rows.append(f"{i},1,2,3,4,{offset+0.5*i:.2f},0,30,0,0,1\n")
        p.write_text("".join(rows), encoding="utf-8")
    model_a = _save_bare_state_dict(tmp_path, name="div_a_m", hidden_layers=[8, 8])
    model_b = _save_bare_state_dict(tmp_path, name="div_b_m", hidden_layers=[8, 8])
    result = run_side_by_side_comparison(
        model_a_path=model_a,
        model_b_path=model_b,
        dataset_path=dat_a,
        test_a_path=dat_a,
        test_b_path=dat_b,
    )
    with pytest.raises(AttributeError, match="ambiguous"):
        _ = result.actuals_xyz_mm

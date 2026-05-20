"""Tests for the single-panel ANN-error histogram figure.

Covers the slot resolver, the renderer, the high-DPI save path, and the
controller wiring that the GUI button consumes.
"""

from __future__ import annotations

import os
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from continuum_robot.modeling.model_comparison import (
    LoadedModelHandle,
    ModelComparisonResult,
    ModelStats,
    build_ann_error_histogram_figure,
    resolve_ann_histogram_slot,
    save_ann_error_histogram_png,
)


_PLACEHOLDER_PNG_BYTES = 90


def _assert_real_png(path: Path) -> None:
    assert path.exists(), f"{path} not written"
    size = path.stat().st_size
    assert size > _PLACEHOLDER_PNG_BYTES, (
        f"{path} looks like a placeholder ({size} bytes) — render likely raised"
    )
    assert path.read_bytes().startswith(b"\x89PNG")


def _stats(arr: np.ndarray) -> ModelStats:
    return ModelStats(
        mean_mm=float(arr.mean()),
        median_mm=float(np.median(arr)),
        p95_mm=float(np.percentile(arr, 95)),
        max_mm=float(arr.max()),
        sample_count=int(arr.size),
    )


def _make_handle(label: str) -> LoadedModelHandle:
    return LoadedModelHandle(
        label=label,
        source_path=Path(f"/tmp/{label}.pt"),
        artifact_details=None,
        hidden_layers=[64, 64],
        input_dim=4,
        output_dim=3,
        output_target="xyz",
        dtype_name="float32",
        scalers=None,
    )


def _make_result(
    *,
    label_a: str = "ann_64_64",
    label_b: str = "linear_ridge",
    n: int = 120,
    seed: int = 0,
) -> ModelComparisonResult:
    rng = np.random.default_rng(seed)
    a_errors = np.abs(rng.normal(0.6, 0.25, size=n))
    b_errors = np.abs(rng.normal(2.1, 1.4, size=n))
    return ModelComparisonResult(
        model_a=_make_handle(label_a),
        model_b=_make_handle(label_b),
        dataset_run_name="test_run",
        dataset_path=Path("/tmp/dataset"),
        a_actuals_xyz_mm=rng.normal(0, 10, (n, 3)),
        b_actuals_xyz_mm=rng.normal(0, 10, (n, 3)),
        a_cable_inputs_cm=rng.normal(0, 1, (n, 4)),
        b_cable_inputs_cm=rng.normal(0, 1, (n, 4)),
        a_predictions_xyz_mm=rng.normal(0, 10, (n, 3)),
        b_predictions_xyz_mm=rng.normal(0, 10, (n, 3)),
        a_errors_mm=a_errors,
        b_errors_mm=b_errors,
        a_dataset_run_name="test_run",
        b_dataset_run_name="test_run",
        shared_color_max_mm=float(max(a_errors.max(), b_errors.max())),
        a_stats=_stats(a_errors),
        b_stats=_stats(b_errors),
    )


# --------------------------------------------------------------------------- #
# Slot resolver                                                                #
# --------------------------------------------------------------------------- #


def test_resolve_slot_explicit_returns_verbatim() -> None:
    result = _make_result()
    assert resolve_ann_histogram_slot(result, slot="a") == "a"
    assert resolve_ann_histogram_slot(result, slot="b") == "b"


def test_resolve_slot_auto_picks_ann_when_only_one_slot_is_ann() -> None:
    result = _make_result(label_a="ann_64_64", label_b="linear_ridge")
    assert resolve_ann_histogram_slot(result, slot="auto") == "a"
    result_swapped = _make_result(label_a="linear_ridge", label_b="ann_64_64")
    assert resolve_ann_histogram_slot(result_swapped, slot="auto") == "b"


def test_resolve_slot_auto_breaks_ties_to_a() -> None:
    """Both slots are ANN — operator's slot A wins as a stable tiebreak."""
    result = _make_result(label_a="ann_32_32", label_b="ann_64_64")
    assert resolve_ann_histogram_slot(result, slot="auto") == "a"


def test_resolve_slot_auto_falls_back_to_a_when_neither_is_ann() -> None:
    result = _make_result(label_a="kinematic_baseline", label_b="linear_ridge")
    assert resolve_ann_histogram_slot(result, slot="auto") == "a"


def test_resolve_slot_invalid_value_raises() -> None:
    result = _make_result()
    with pytest.raises(ValueError, match="slot must be"):
        resolve_ann_histogram_slot(result, slot="garbage")


# --------------------------------------------------------------------------- #
# Figure builder                                                               #
# --------------------------------------------------------------------------- #


def test_build_histogram_uses_slot_label_in_title() -> None:
    result = _make_result(label_a="ann_64_64", label_b="linear_ridge")
    fig = build_ann_error_histogram_figure(result, slot="auto")
    # suptitle text
    assert "ann_64_64" in (fig._suptitle.get_text() if fig._suptitle else "")
    import matplotlib.pyplot as plt
    plt.close(fig)


def test_build_histogram_forced_slot_b_renders_b_label() -> None:
    result = _make_result(label_a="ann_64_64", label_b="linear_ridge")
    fig = build_ann_error_histogram_figure(result, slot="b")
    assert "linear_ridge" in (fig._suptitle.get_text() if fig._suptitle else "")
    import matplotlib.pyplot as plt
    plt.close(fig)


def test_build_histogram_handles_empty_errors_without_crashing() -> None:
    base = _make_result()
    empty = replace(
        base,
        a_errors_mm=np.asarray([], dtype=float),
        a_stats=ModelStats(mean_mm=0.0, median_mm=0.0, p95_mm=0.0, max_mm=0.0, sample_count=0),
    )
    fig = build_ann_error_histogram_figure(empty, slot="a")
    # Should still produce a figure object (with "empty" text on the axes)
    assert fig is not None
    import matplotlib.pyplot as plt
    plt.close(fig)


# --------------------------------------------------------------------------- #
# Save path                                                                    #
# --------------------------------------------------------------------------- #


def test_save_writes_real_png_at_requested_dpi(tmp_path: Path) -> None:
    result = _make_result()
    out = tmp_path / "ann_error_histogram.png"
    saved = save_ann_error_histogram_png(result, out, slot="auto", dpi=200)
    assert saved == out
    _assert_real_png(out)


def test_save_default_dpi_is_high_resolution() -> None:
    """Default DPI must stay at 600 — the operator explicitly asked for
    high-resolution thesis-bound output and a regression here would silently
    degrade every saved histogram."""
    import inspect
    sig = inspect.signature(save_ann_error_histogram_png)
    assert sig.parameters["dpi"].default == 600


# --------------------------------------------------------------------------- #
# Controller wiring                                                            #
# --------------------------------------------------------------------------- #


def test_controller_method_exists_and_is_callable() -> None:
    pytest.importorskip("PySide6")
    from continuum_robot.gui.controllers.modeling_controller import ModelingController

    assert hasattr(ModelingController, "save_last_ann_error_histogram_png")
    import inspect
    sig = inspect.signature(ModelingController.save_last_ann_error_histogram_png)
    # Must default dpi to 600 (the headline thesis-bound resolution)
    assert sig.parameters["dpi"].default == 600
    # And accept a slot kwarg defaulting to "auto"
    assert sig.parameters["slot"].default == "auto"

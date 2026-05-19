"""Side-by-side comparison of two single-segment forward ANN models.

This module powers the "External Model Comparison" card on the Modeling tab.
The operator picks one model from the existing artifact dropdown (Model A) and
uploads a second .pt file (Model B), then sees two 3D scatter plots side by
side — each plotting the model's predicted tip positions colored by the
Euclidean error to the recorded ground-truth tip. A shared viridis colorbar
makes the two plots directly visually comparable: the same color = the same
mm of error on both panels.

Design constraints (from the audit conversation, 2026-05-18):
  - Single-segment ANN only (4 cables → tip XYZ). Two-segment is a follow-up.
  - Same color scale on both plots (max-of-both, viridis, 0 → max_mm).
  - Continuous colormap with a colorbar, NOT Wolfe-tier discrete thresholds.
  - Output is a side-by-side PNG; no CSV/JSON bundle in this pass.
  - Model A always comes from a fully-discovered artifact (has metadata).
  - Model B is uploaded; the loader tries (in order):
      1. If the path is an artifact directory, load it normally.
      2. If the path is a bare .pt but a sibling ``training_metadata.json``
         exists in the same directory, treat the directory as an artifact.
      3. If the path is a bare .pt with no sidecar metadata, infer the model
         architecture (hidden layer widths, input/output dims) directly from
         the state_dict shapes. No scalers are available in this fallback, so
         we assume the model was trained on raw cable cm + raw XYZ mm (the
         project's pre-scaler-era convention). A warning is surfaced in the
         result so the operator can interpret outliers correctly.
"""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass, field
import json
from pathlib import Path
from typing import Any

import numpy as np

from continuum_robot.modeling.analysis import (
    ArtifactDetails,
    load_trained_artifact_details,
)
# Public re-exports for tests / external probing of the auto-detect contract.
__all__ = [
    "LoadedModelHandle",
    "ModelComparisonResult",
    "ModelStats",
    "build_comparison_figure",
    "load_model_for_comparison",
    "run_side_by_side_comparison",
    "save_comparison_png",
]
from continuum_robot.modeling.ann_training import (
    IoScalers,
    LEGACY_FULL_POSE_INPUT_DIM,
    OUTPUT_TARGET_FULL_POSE,
    OUTPUT_TARGET_XYZ,
    TorchUnavailableError,
    TrainedArtifactSummary,
    _build_legacy_ann_model,
    _require_torch,
    _torch_dtype,
    prepare_legacy_ann_dataset,
)


@dataclass(frozen=True)
class LoadedModelHandle:
    """Resolved Model handle: metadata + the loaded torch module + scalers.

    Carries enough provenance for the figure title strip ("Model A: <name>") and
    lets ``run_side_by_side_comparison`` reuse the same inference path for both
    A and B regardless of whether they came from a discovered artifact or a
    bare-.pt upload.
    """

    label: str  # operator-facing name shown in the figure title
    source_path: Path  # the path the operator actually selected
    artifact_details: ArtifactDetails | None  # None when inferred from bare .pt
    hidden_layers: list[int]
    input_dim: int
    output_dim: int
    output_target: str  # "xyz" or "full_pose"; "cable_from_xyz" is rejected upstream
    dtype_name: str  # "float32" / "float64"
    scalers: IoScalers | None  # None ⇒ raw I/O fallback
    warnings: tuple[str, ...] = ()
    # Multiplier applied to cable inputs (in cm) before they reach the model.
    # 1.0 = no change (model was trained on cm, matching the new project's convention).
    # 10.0 = legacy continuum_jack convention (model was trained on mm; new datasets
    # store cable_cm, so we multiply by 10 to get the mm scale the model expects).
    # Detected automatically for bare-.pt uploads via :func:`_autodetect_input_scale`
    # and persisted in the archived metadata so future re-loads honor it without
    # re-probing.
    input_scale_multiplier: float = 1.0


@dataclass(frozen=True)
class ModelStats:
    """Per-model error stats. Used in the figure title strip and tests."""

    mean_mm: float
    median_mm: float
    p95_mm: float
    max_mm: float
    sample_count: int


@dataclass(frozen=True)
class ModelComparisonResult:
    """Output of :func:`run_side_by_side_comparison`.

    The figure builder consumes this directly; tests can assert against the
    numeric fields without parsing pixels.
    """

    model_a: LoadedModelHandle
    model_b: LoadedModelHandle
    dataset_run_name: str
    dataset_path: Path
    actuals_xyz_mm: np.ndarray  # (N, 3) recorded tip positions
    a_predictions_xyz_mm: np.ndarray  # (N, 3) Model A predictions
    b_predictions_xyz_mm: np.ndarray  # (N, 3) Model B predictions
    a_errors_mm: np.ndarray  # (N,) Euclidean ||A_pred − actual||
    b_errors_mm: np.ndarray  # (N,) Euclidean ||B_pred − actual||
    shared_color_max_mm: float  # max over both error arrays — drives the colorbar
    a_stats: ModelStats
    b_stats: ModelStats
    warnings: tuple[str, ...] = ()


# ---------------------------------------------------------------------------
# model loading
# ---------------------------------------------------------------------------


# Probe cable inputs (cm) used by ``_autodetect_input_scale`` to figure out what
# unit a bare-.pt was actually trained on. Seven vectors spanning the four-cable
# space: cardinal axes, two diagonals, and a stretch. Magnitudes are typical of
# a real workspace acquisition (±1 cm = ±10 mm).
_AUTODETECT_PROBE_CABLE_CM: np.ndarray = np.array(
    [
        [1.0, 0.0, 0.0, 0.0],
        [-1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, -1.0, 0.0, 0.0],
        [1.0, 1.0, -1.0, -1.0],
        [-1.0, -1.0, 1.0, 1.0],
        [1.5, 0.0, -1.5, 0.0],
    ],
    dtype=float,
)


# Candidate input-scale multipliers we try when auto-detecting the unit
# convention of a legacy bare-.pt. Ordered preference: 1.0 first (the new
# convention; matches scaler-trained models when scalers are absent), 10.0 next
# (legacy continuum_jack mm-cable convention), then 100 / 0.1 / 0.01 as
# fallbacks for stranger conventions.
_AUTODETECT_CANDIDATE_SCALES: tuple[float, ...] = (1.0, 10.0, 100.0, 0.1, 0.01)


# A model that produces less than this much spread (mm of std across the probe
# set) at any scale is "collapsing to a point" and gets a strong warning.
_AUTODETECT_MIN_HEALTHY_SPREAD_MM: float = 5.0


def _autodetect_input_scale(*, model, torch, dtype) -> tuple[float, dict[str, float]]:
    """Probe a freshly-loaded legacy model with cable inputs at several scales.

    Returns ``(best_scale_multiplier, diagnostic_info)``. The multiplier is the
    factor that, when applied to cable values in cm, gives the model the input
    range it was actually trained on. ``diagnostic_info`` is a mapping of
    ``"scale_<N>_spread_mm" → <float>`` entries the caller can fold into the
    figure warning text.

    Why this exists: legacy continuum_jack ``.pt`` files (May 2024 vintage)
    were trained on tendon displacement in **mm**, but new Continuum-Pi
    datasets store ``resolved_cable_command_cm``. Feeding cm values to a
    mm-trained model puts every hidden ReLU in its near-bias region and
    collapses every prediction to ~output.bias — visually, all points map to
    a single workspace location.

    Selection rule (refined after observing the May-2024 models): pick the
    SMALLEST scale that produces a "healthy" spread (≥
    ``_AUTODETECT_MIN_HEALTHY_SPREAD_MM``). This avoids over-scaling — the
    legacy mm-trained models cleared the threshold at ×10 but kept growing
    spread at ×100 because the inputs entered the model's extrapolation
    region, giving fake "300 mm workspace" outputs that look impressive but
    are unphysical. Picking the smallest healthy scale gives the most
    in-distribution predictions. Fallback: if no scale is healthy, pick the
    one with maximum spread anyway (best effort).
    """
    info: dict[str, float] = {}
    for scale in _AUTODETECT_CANDIDATE_SCALES:
        try:
            with torch.inference_mode():
                tensor = torch.tensor(
                    _AUTODETECT_PROBE_CABLE_CM * float(scale),
                    dtype=dtype,
                )
                preds = model(tensor).detach().cpu().numpy()
        except Exception:
            continue
        if preds.size == 0:
            continue
        xyz = preds[:, :3] if preds.shape[1] >= 3 else preds
        if not np.all(np.isfinite(xyz)):
            continue
        spread = float(np.sum(np.std(xyz, axis=0)))
        info[f"scale_{scale}_spread_mm"] = round(spread, 4)
    if not info:
        return 1.0, info
    # Pick the first scale in preferred order that produces a healthy spread.
    # Preferred order = candidate order (1.0 first, then 10.0, etc.) so we bias
    # toward the new cm convention when it works, fall to ×10 (legacy mm) next,
    # and only consider sub-unit scales if nothing else fires. This avoids
    # over-scaling a genuinely cm-trained model (where every scale is "healthy"
    # because the response is approximately linear).
    for scale in _AUTODETECT_CANDIDATE_SCALES:
        key = f"scale_{scale}_spread_mm"
        if info.get(key, 0.0) >= _AUTODETECT_MIN_HEALTHY_SPREAD_MM:
            return float(scale), info
    # Nothing healthy — pick whichever maximized spread as a best-effort.
    best_key = max(info, key=lambda k: info[k])
    best_scale = float(best_key.split("_")[1])
    return best_scale, info


def _infer_architecture_from_state_dict(state_dict: dict[str, Any]) -> tuple[int, list[int], int]:
    """Reverse-engineer the legacy ANN's layer widths from a saved state_dict.

    The training pipeline builds models as ``nn.Sequential`` with keys
    ``input``, ``hidden1``, ``hidden2``, ..., ``output``. Each Linear's
    ``weight`` is shape (out_features, in_features). Walk those in order to
    recover input_dim → hidden widths → output_dim.

    Raises ValueError if the keys don't match the legacy layout — we'd rather
    refuse than guess wrong and give silently bad predictions.
    """
    linear_keys: list[tuple[str, tuple[int, int]]] = []
    for key, tensor in state_dict.items():
        if not key.endswith(".weight"):
            continue
        layer_name = key[: -len(".weight")]
        if layer_name not in ("input", "output") and not layer_name.startswith("hidden"):
            raise ValueError(
                f"Unexpected layer '{layer_name}' in state_dict; bare-.pt upload "
                "only supports the legacy nn.Sequential(input, hidden*, output) layout."
            )
        shape = tuple(int(v) for v in tensor.shape)
        if len(shape) != 2:
            raise ValueError(
                f"Layer '{layer_name}' weight has shape {shape}; expected 2D Linear weight."
            )
        linear_keys.append((layer_name, shape))

    if not linear_keys:
        raise ValueError("state_dict contains no Linear weights — not a legacy ANN.")
    # Sort: input first, then hidden1..hiddenK in numeric order, then output.
    def _sort_key(item: tuple[str, tuple[int, int]]) -> tuple[int, int]:
        name, _ = item
        if name == "input":
            return (0, 0)
        if name == "output":
            return (2, 0)
        # hidden<N>
        try:
            return (1, int(name[len("hidden") :]))
        except ValueError as exc:
            raise ValueError(f"Unparseable hidden-layer name '{name}'") from exc

    linear_keys.sort(key=_sort_key)
    # Walk: input_dim = first layer's in_features; each layer's out_features is the
    # next hidden width; output_dim = last layer's out_features.
    input_dim = int(linear_keys[0][1][1])  # (out, IN)
    hidden_layers: list[int] = []
    for name, (out_features, _) in linear_keys[:-1]:
        # Every non-final Linear's out_features is a hidden width.
        hidden_layers.append(int(out_features))
    output_dim = int(linear_keys[-1][1][0])
    return input_dim, hidden_layers, output_dim


def load_model_for_comparison(path: Path, *, label_prefix: str = "") -> LoadedModelHandle:
    """Load either an artifact directory OR a bare .pt for side-by-side inference.

    Strategy (in order):

      1. If ``path`` is a directory containing ``training_metadata.json``, use the
         normal artifact loader. Full provenance + scalers + dtype.
      2. If ``path`` is a ``.pt`` file in a directory containing
         ``training_metadata.json`` (the standard layout when the operator picks
         the model file directly), promote to case 1.
      3. If ``path`` is a bare ``.pt`` with no sidecar metadata, infer the
         architecture from the state_dict shapes. Falls back to raw I/O (no
         scalers) — a warning is attached for the GUI title strip.

    ``label_prefix`` is prepended to the human-readable label (e.g., "A: ", "B: ")
    so the figure title strip distinguishes the two panels.
    """
    candidate = Path(path)
    if not candidate.exists():
        raise FileNotFoundError(f"Model path does not exist: {candidate}")

    artifact_dir: Path | None = None
    if candidate.is_dir() and (candidate / "training_metadata.json").exists():
        artifact_dir = candidate
    elif candidate.is_file() and candidate.suffix == ".pt":
        sibling = candidate.parent / "training_metadata.json"
        if sibling.exists():
            artifact_dir = candidate.parent

    warnings: list[str] = []
    input_scale_multiplier: float = 1.0
    if artifact_dir is not None:
        details = load_trained_artifact_details(artifact_dir)
        metadata = dict(details.metadata or {})
        model_payload = dict(metadata.get("model", {}) or {})
        output_target = str(model_payload.get("output_target") or "").strip().lower()
        artifact_kind = str(metadata.get("artifact_kind", "") or "")
        if not output_target:
            # Back-compat: infer from artifact_kind (same heuristic as analysis._evaluate_ann).
            if "xyz_to_cable" in artifact_kind or "cable_from_xyz" in artifact_kind:
                output_target = "cable_from_xyz"
            elif "xyz" in artifact_kind and "full_pose" not in artifact_kind:
                output_target = OUTPUT_TARGET_XYZ
            else:
                output_target = OUTPUT_TARGET_FULL_POSE
        if output_target == "cable_from_xyz":
            raise ValueError(
                "Selected model is an inverse ANN (xyz → cable); the side-by-side "
                "comparison shows forward cable → tip-XYZ predictions only."
            )
        hidden_layers = [int(v) for v in model_payload.get("hidden_layers", [32, 32]) or [32, 32]]
        input_dim = int(model_payload.get("input_dim", LEGACY_FULL_POSE_INPUT_DIM) or LEGACY_FULL_POSE_INPUT_DIM)
        output_dim = int(model_payload.get("output_dim", 6) or 6)
        dtype_name = str(model_payload.get("dtype", "float64") or "float64")
        scalers: IoScalers | None = None
        scaler_payload = metadata.get("io_scaler")
        if isinstance(scaler_payload, dict):
            try:
                scalers = IoScalers.from_dict(scaler_payload)
            except Exception:
                scalers = None
                warnings.append("Saved I/O scalers could not be parsed; using raw I/O.")
        # Honor any previously-detected input-scale multiplier from the archived
        # upload's metadata (set by ModelingController._persist_uploaded_model).
        # Regular trained artifacts (output_target='xyz' / 'full_pose' with scalers)
        # leave this at 1.0 because they were trained on the same cm convention the
        # new datasets use.
        upload_prov = dict(metadata.get("upload_provenance", {}) or {})
        if "input_scale_multiplier" in upload_prov:
            try:
                input_scale_multiplier = float(upload_prov["input_scale_multiplier"])
            except (TypeError, ValueError):
                input_scale_multiplier = 1.0
            if input_scale_multiplier != 1.0:
                warnings.append(
                    f"Cable inputs scaled by ×{input_scale_multiplier:g} before "
                    "inference (auto-detected when this model was uploaded — the "
                    "model was trained on a different unit convention than the "
                    "current dataset). See training_metadata.json upload_provenance."
                )
        label_name = details.summary.artifact_name
    else:
        # Bare-.pt fallback path. We need PyTorch to peek at the state_dict.
        try:
            torch = _require_torch()
        except TorchUnavailableError as exc:
            raise RuntimeError(
                f"PyTorch is required to load bare .pt files for comparison: {exc}"
            ) from exc
        state_dict = torch.load(candidate, map_location="cpu")
        if not isinstance(state_dict, dict):
            raise ValueError(
                f".pt file did not contain a state_dict; loaded {type(state_dict).__name__}. "
                "Side-by-side comparison expects a legacy ANN state_dict."
            )
        input_dim, hidden_layers, output_dim = _infer_architecture_from_state_dict(state_dict)
        # Output dim 3 ⇒ XYZ-only; 6 ⇒ full-pose. Anything else is unsupported.
        if output_dim == 3:
            output_target = OUTPUT_TARGET_XYZ
        elif output_dim == 6:
            output_target = OUTPUT_TARGET_FULL_POSE
        else:
            raise ValueError(
                f"Inferred output_dim={output_dim}; expected 3 (XYZ) or 6 (full pose). "
                "Side-by-side comparison only supports forward single-segment ANNs."
            )
        dtype_name = "float64"
        scalers = None
        # Auto-detect the cable-input unit convention by probing the freshly-loaded
        # model at several scale multipliers. Legacy continuum_jack models (May
        # 2024 vintage) were trained on tendon displacement in mm, but new
        # datasets store cable_cm — without this probe, predictions collapse to a
        # single point because every hidden ReLU sits in its near-bias region.
        dtype = _torch_dtype(torch, dtype_name)
        probe_model = _build_legacy_ann_model(
            torch=torch,
            input_dim=input_dim,
            output_dim=output_dim,
            hidden_layers=hidden_layers,
            device=torch.device("cpu"),
            dtype=dtype,
        )
        probe_model.load_state_dict(state_dict)
        probe_model.eval()
        input_scale_multiplier, scale_info = _autodetect_input_scale(
            model=probe_model, torch=torch, dtype=dtype
        )
        # Build a tight one-liner summary of what each scale produced — the figure
        # warning text shows this so the operator can sanity-check the auto-pick.
        info_pieces = [f"{k}={v}mm" for k, v in scale_info.items()]
        arch_blurb = (
            f"architecture inferred from state_dict (input_dim={input_dim}, "
            f"hidden_layers={hidden_layers}, output_dim={output_dim})"
        )
        if input_scale_multiplier == 1.0 and max(scale_info.values(), default=0.0) < _AUTODETECT_MIN_HEALTHY_SPREAD_MM:
            warnings.append(
                f"Loaded as bare .pt — {arch_blurb}. Auto-detection could not find "
                f"an input scale that produces healthy output spread (best="
                f"{max(scale_info.values(), default=0.0):.2f} mm std). Predictions "
                f"will likely collapse near the model's output bias. Probe results: "
                f"{', '.join(info_pieces)}."
            )
        elif input_scale_multiplier != 1.0:
            warnings.append(
                f"Loaded as bare .pt — {arch_blurb}. Auto-detected cable input "
                f"scale ×{input_scale_multiplier:g} (model was trained on a different "
                f"unit than the new dataset's resolved_cable_command_cm). Probe "
                f"spreads: {', '.join(info_pieces)}."
            )
        else:
            warnings.append(
                f"Loaded as bare .pt — {arch_blurb}; cable inputs kept at the new "
                f"convention (cm, ×1.0). No I/O scalers applied."
            )
        details = None
        label_name = candidate.name

    label = f"{label_prefix}{label_name}" if label_prefix else label_name
    return LoadedModelHandle(
        label=label,
        source_path=candidate,
        artifact_details=details,
        hidden_layers=list(hidden_layers),
        input_dim=int(input_dim),
        output_dim=int(output_dim),
        output_target=str(output_target),
        dtype_name=str(dtype_name),
        scalers=scalers,
        warnings=tuple(warnings),
        input_scale_multiplier=float(input_scale_multiplier),
    )


# ---------------------------------------------------------------------------
# inference
# ---------------------------------------------------------------------------


def _predict_with_handle(handle: LoadedModelHandle, *, inputs: np.ndarray) -> np.ndarray:
    """Run inference for one loaded model on the dataset's cable commands.

    Returns predictions shaped (N, 3) — the XYZ portion. For full-pose models
    (output_dim=6) we slice off the tangent. Inputs are scaled with the saved
    input_scaler when present (matches the production eval path).
    """
    try:
        torch = _require_torch()
    except TorchUnavailableError as exc:
        raise RuntimeError(f"PyTorch is required for model comparison: {exc}") from exc

    dtype = _torch_dtype(torch, handle.dtype_name)
    device = torch.device("cpu")
    model = _build_legacy_ann_model(
        torch=torch,
        input_dim=handle.input_dim,
        output_dim=handle.output_dim,
        hidden_layers=handle.hidden_layers,
        device=device,
        dtype=dtype,
    )
    # Resolve the state_dict path. Artifact path: ``<dir>/model.pt``. Bare upload:
    # the source_path itself (when source_path is a .pt file).
    if handle.artifact_details is not None:
        model_path = handle.artifact_details.summary.model_path
        if model_path is None or not Path(model_path).exists():
            raise FileNotFoundError(
                f"Artifact '{handle.label}' is missing its model.pt file."
            )
    else:
        model_path = str(handle.source_path)
    state_dict = torch.load(Path(model_path), map_location="cpu")
    model.load_state_dict(state_dict)
    model.eval()

    inputs_arr = np.asarray(inputs, dtype=float)
    # Apply the unit-convention multiplier BEFORE the scaler — the multiplier
    # converts cable_cm (the new dataset convention) into whatever unit this
    # model was trained on (legacy continuum_jack mm = ×10, etc.). Then the
    # scaler — if present — Z-scores the now-correctly-scaled values. Skipping
    # the multiplier was why legacy mm-trained models collapsed every
    # prediction to ~output.bias on cm inputs.
    if handle.input_scale_multiplier != 1.0:
        inputs_arr = inputs_arr * float(handle.input_scale_multiplier)
    if handle.scalers is not None:
        inputs_arr = handle.scalers.input_scaler.transform(inputs_arr)
    with torch.inference_mode():
        tensor_inputs = torch.tensor(inputs_arr, dtype=dtype, device=device)
        predictions = model(tensor_inputs).detach().cpu().numpy()
    predictions = np.asarray(predictions, dtype=float)
    if handle.scalers is not None:
        predictions = handle.scalers.output_scaler.inverse_transform(predictions)
    # Slice to XYZ for the plot. Full-pose models put XYZ in the first 3 cols.
    if predictions.shape[1] >= 3:
        return predictions[:, :3]
    raise ValueError(
        f"Model '{handle.label}' produced predictions with shape {predictions.shape}; "
        "expected at least 3 output columns (XYZ)."
    )


def _stats(errors: np.ndarray) -> ModelStats:
    """Compute the headline error stats shown in the figure title strip."""
    arr = np.asarray(errors, dtype=float).ravel()
    if arr.size == 0:
        return ModelStats(0.0, 0.0, 0.0, 0.0, 0)
    return ModelStats(
        mean_mm=float(np.mean(arr)),
        median_mm=float(np.median(arr)),
        p95_mm=float(np.percentile(arr, 95)),
        max_mm=float(np.max(arr)),
        sample_count=int(arr.size),
    )


def run_side_by_side_comparison(
    *,
    model_a_path: Path,
    model_b_path: Path,
    dataset_path: Path,
) -> ModelComparisonResult:
    """End-to-end: load both models, predict on the dataset, compute errors.

    The two models must both be forward single-segment ANNs (cable → XYZ or
    cable → full pose). The dataset is parsed via
    :func:`prepare_legacy_ann_dataset` with ``output_target='xyz'`` so we get
    cable inputs and recorded tip XYZ side by side. The predictions are sliced
    to XYZ for full-pose models so both columns are directly comparable on the
    same 3D scatter.
    """
    model_a = load_model_for_comparison(Path(model_a_path), label_prefix="A: ")
    model_b = load_model_for_comparison(Path(model_b_path), label_prefix="B: ")

    # Prepare the dataset in xyz-target mode so .inputs is cable (N,4) and .outputs
    # is recorded tip XYZ (N,3). Even if Model A or B is full-pose internally, we
    # only need cable→XYZ for the plot.
    prepared = prepare_legacy_ann_dataset(Path(dataset_path), output_target=OUTPUT_TARGET_XYZ)
    if prepared.inputs.shape[0] == 0:
        raise ValueError(
            "Selected dataset has no accepted samples after the row filter — pick a "
            "fully-collected workspace dataset."
        )
    if prepared.inputs.shape[0] < 10:
        raise ValueError(
            f"Selected dataset has only {prepared.inputs.shape[0]} samples — a workspace "
            "comparison plot needs more coverage. Pick an angular_test_mesh or a fuller "
            "collect_pose_command_dataset run."
        )

    cable_inputs = prepared.inputs  # (N, 4)
    actuals_xyz = prepared.outputs  # (N, 3) since output_target=xyz

    a_preds = _predict_with_handle(model_a, inputs=cable_inputs)
    b_preds = _predict_with_handle(model_b, inputs=cable_inputs)

    a_errors = np.linalg.norm(a_preds - actuals_xyz, axis=1)
    b_errors = np.linalg.norm(b_preds - actuals_xyz, axis=1)
    shared_max = float(max(a_errors.max() if a_errors.size else 0.0, b_errors.max() if b_errors.size else 0.0))

    warnings: list[str] = []
    warnings.extend(model_a.warnings)
    warnings.extend(model_b.warnings)

    return ModelComparisonResult(
        model_a=model_a,
        model_b=model_b,
        dataset_run_name=prepared.summary.run_name,
        dataset_path=Path(dataset_path),
        actuals_xyz_mm=actuals_xyz,
        a_predictions_xyz_mm=a_preds,
        b_predictions_xyz_mm=b_preds,
        a_errors_mm=a_errors,
        b_errors_mm=b_errors,
        shared_color_max_mm=shared_max,
        a_stats=_stats(a_errors),
        b_stats=_stats(b_errors),
        warnings=tuple(warnings),
    )


# ---------------------------------------------------------------------------
# figure
# ---------------------------------------------------------------------------


def _format_stats_line(stats: ModelStats) -> str:
    return (
        f"mean={stats.mean_mm:.2f}  "
        f"med={stats.median_mm:.2f}  "
        f"p95={stats.p95_mm:.2f}  "
        f"max={stats.max_mm:.2f} mm  "
        f"(N={stats.sample_count})"
    )


def build_comparison_figure(
    result: ModelComparisonResult,
    *,
    figsize: tuple[float, float] = (14.0, 6.5),
    dpi: int = 110,
):
    """Build a matplotlib Figure with two 3D scatter axes + shared viridis colorbar.

    Returns a Figure (caller embeds via FigureCanvasQTAgg or saves with savefig).
    """
    # Local import — matplotlib is heavy and tests that don't render shouldn't pay it.
    import matplotlib

    matplotlib.use("Agg", force=False)
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 — registers 3D projection

    figure = plt.figure(figsize=figsize, dpi=dpi)
    figure.suptitle(
        f"Side-by-side prediction error on {result.dataset_run_name}",
        fontsize=12,
        fontweight="bold",
    )

    ax_a = figure.add_subplot(1, 2, 1, projection="3d")
    ax_b = figure.add_subplot(1, 2, 2, projection="3d")

    vmin = 0.0
    vmax = max(result.shared_color_max_mm, 1e-6)  # avoid degenerate colormap

    sc_a = ax_a.scatter(
        result.a_predictions_xyz_mm[:, 0],
        result.a_predictions_xyz_mm[:, 1],
        result.a_predictions_xyz_mm[:, 2],
        c=result.a_errors_mm,
        cmap="viridis",
        vmin=vmin,
        vmax=vmax,
        s=14,
        depthshade=False,
    )
    ax_b.scatter(
        result.b_predictions_xyz_mm[:, 0],
        result.b_predictions_xyz_mm[:, 1],
        result.b_predictions_xyz_mm[:, 2],
        c=result.b_errors_mm,
        cmap="viridis",
        vmin=vmin,
        vmax=vmax,
        s=14,
        depthshade=False,
    )

    # Share the X/Y/Z limits across the two subplots so the geometry is comparable.
    all_points = np.vstack(
        [result.a_predictions_xyz_mm, result.b_predictions_xyz_mm, result.actuals_xyz_mm]
    )
    pad_mm = 2.0
    xmin, ymin, zmin = all_points.min(axis=0) - pad_mm
    xmax, ymax, zmax = all_points.max(axis=0) + pad_mm
    for ax in (ax_a, ax_b):
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)
        ax.set_zlim(zmin, zmax)
        ax.set_xlabel("X (mm)")
        ax.set_ylabel("Y (mm)")
        ax.set_zlabel("Z (mm)")
        ax.tick_params(labelsize=8)

    ax_a.set_title(
        f"{result.model_a.label}\n{_format_stats_line(result.a_stats)}",
        fontsize=9,
    )
    ax_b.set_title(
        f"{result.model_b.label}\n{_format_stats_line(result.b_stats)}",
        fontsize=9,
    )

    # Shared colorbar between the two axes. Positioning: a thin vertical bar at the
    # right margin. Same vmin/vmax on both scatters guarantees the bar maps to both.
    cbar = figure.colorbar(
        sc_a,
        ax=[ax_a, ax_b],
        orientation="vertical",
        fraction=0.025,
        pad=0.04,
        shrink=0.85,
    )
    cbar.set_label("Tip position error |pred − actual| (mm)", fontsize=9)
    cbar.ax.tick_params(labelsize=8)

    if result.warnings:
        # Surface non-fatal warnings (e.g., "bare .pt, no scalers") on the figure
        # itself so the thesis-bound PNG carries the caveat with it.
        warning_text = "\n".join(f"⚠ {w}" for w in result.warnings)
        figure.text(
            0.5,
            0.02,
            warning_text,
            ha="center",
            va="bottom",
            fontsize=7,
            color="#b45309",
            wrap=True,
        )

    return figure


def save_comparison_png(
    result: ModelComparisonResult,
    target_path: Path,
    *,
    dpi: int = 300,
) -> Path:
    """Build the figure and save it at publication DPI. Returns the saved path."""
    figure = build_comparison_figure(result, dpi=dpi)
    target = Path(target_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(target, dpi=dpi, bbox_inches="tight")
    # Free the figure to avoid leaks in long sessions.
    import matplotlib.pyplot as plt

    plt.close(figure)
    return target

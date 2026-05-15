"""Config-gated physics helpers for offline two-segment model comparison."""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Any

import numpy as np


@dataclass(frozen=True)
class PhysicsAdapterStatus:
    """Operator-facing status for a physics/geometric model adapter."""

    model_key: str
    label: str
    status: str
    reason: str = ""
    required_parameters: list[str] = field(default_factory=list)
    missing_parameters: list[str] = field(default_factory=list)
    convention_notes: list[str] = field(default_factory=list)
    can_predict_distal_xyz: bool = False
    can_predict_intermediate_xyz: bool = False
    can_predict_orientation: bool = False
    hardware_validated: bool = False
    predictions_generated: bool = False
    known_design_parameters: list[str] = field(default_factory=list)
    measured_parameters: list[str] = field(default_factory=list)
    estimated_parameters: list[str] = field(default_factory=list)
    unknown_parameters: list[str] = field(default_factory=list)
    hardware_validation_required: list[str] = field(default_factory=list)
    blocking_missing_parameters: list[str] = field(default_factory=list)
    nonblocking_uncertain_parameters: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_key": self.model_key,
            "label": self.label,
            "status": self.status,
            "reason": self.reason,
            "required_parameters": list(self.required_parameters),
            "missing_parameters": list(self.missing_parameters),
            "convention_notes": list(self.convention_notes),
            "can_predict_distal_xyz": bool(self.can_predict_distal_xyz),
            "can_predict_intermediate_xyz": bool(self.can_predict_intermediate_xyz),
            "can_predict_orientation": bool(self.can_predict_orientation),
            "hardware_validated": bool(self.hardware_validated),
            "predictions_generated": bool(self.predictions_generated),
            "known_design_parameters": list(self.known_design_parameters),
            "measured_parameters": list(self.measured_parameters),
            "estimated_parameters": list(self.estimated_parameters),
            "unknown_parameters": list(self.unknown_parameters),
            "hardware_validation_required": list(self.hardware_validation_required),
            "blocking_missing_parameters": list(self.blocking_missing_parameters),
            "nonblocking_uncertain_parameters": list(self.nonblocking_uncertain_parameters),
        }


MIKE_REQUIRED_PARAMETERS = [
    "physics_models.global.model_frame_convention",
    "physics_models.global.tendon_displacement_sign_convention",
    "physics_models.global.output_pose_frame",
    "physics_models.global.tangent_representation",
    "physics_models.global.segment_order_source",
    "physics_models.segments.segment_a.segment_length_mm",
    "physics_models.segments.segment_a.tendon_positions_mm",
    "physics_models.segments.segment_b.segment_length_mm",
    "physics_models.segments.segment_b.tendon_positions_mm",
    "physics_models.mike_constant_curvature.curvature_from_tendon_displacement",
    "physics_models.mike_constant_curvature.required_conventions_confirmed",
]

CAMARILLO_REQUIRED_PARAMETERS = [
    "physics_models.global.model_frame_convention",
    "physics_models.global.tendon_displacement_sign_convention",
    "physics_models.global.output_pose_frame",
    "physics_models.global.segment_order_source",
    "physics_models.camarillo.segment_lengths_mm.segment_a",
    "physics_models.camarillo.segment_lengths_mm.segment_b",
    "physics_models.camarillo.cable_positions_mm.segment_a",
    "physics_models.camarillo.cable_positions_mm.segment_b",
    "physics_models.camarillo.segment_stiffness_values.segment_a",
    "physics_models.camarillo.segment_stiffness_values.segment_b",
    "physics_models.camarillo.cable_stiffness_values.segment_a",
    "physics_models.camarillo.cable_stiffness_values.segment_b",
    "physics_models.camarillo.additional_cable_length_mm",
    "physics_models.camarillo.required_conventions_confirmed",
]


def assess_mike_constant_curvature_status(config: dict[str, Any]) -> PhysicsAdapterStatus:
    physics = _physics(config)
    missing = _missing_parameters(physics, [path.removeprefix("physics_models.") for path in MIKE_REQUIRED_PARAMETERS])
    conventions_confirmed = bool(_nested(physics, "mike_constant_curvature.required_conventions_confirmed"))
    categories = _parameter_categories(physics)
    convention_notes = [
        "Assumes each segment is constant curvature.",
        "Solves curvature from explicit tendon positions and configured displacement sign convention.",
        "Composes Segment A transform with Segment B transform; Segment B rides on Segment A.",
        "Ignores hysteresis, friction, gravity, tendon routing coupling, and load-dependent effects.",
        "Positive tendon displacement is configured as tendon shortening / more tension.",
        "Encoder ticks for shortening are configured as decreasing ticks.",
    ]
    if missing:
        return PhysicsAdapterStatus(
            model_key="mike_constant_curvature",
            label="Mike Constant Curvature",
            status="unavailable_missing_parameters",
            reason="Missing required constant-curvature parameters: " + ", ".join(missing),
            required_parameters=MIKE_REQUIRED_PARAMETERS,
            missing_parameters=["physics_models." + path for path in missing],
            convention_notes=convention_notes,
            blocking_missing_parameters=["physics_models." + path for path in missing],
            **categories,
        )
    if not conventions_confirmed:
        return PhysicsAdapterStatus(
            model_key="mike_constant_curvature",
            label="Mike Constant Curvature",
            status="unavailable_unvalidated_convention",
            reason="Constant-curvature frame/sign conventions are not explicitly confirmed in config.",
            required_parameters=MIKE_REQUIRED_PARAMETERS,
            convention_notes=convention_notes,
            hardware_validation_required=[
                "physics_models.mike_constant_curvature.required_conventions_confirmed",
                "physics_models.mike_constant_curvature.frame_convention_hardware_validated",
                "tracked single-axis sign/frame validation",
            ],
            **categories,
        )
    return PhysicsAdapterStatus(
        model_key="mike_constant_curvature",
        label="Mike Constant Curvature",
        status="available",
        required_parameters=MIKE_REQUIRED_PARAMETERS,
        convention_notes=convention_notes,
        can_predict_distal_xyz=True,
        can_predict_intermediate_xyz=True,
        can_predict_orientation=True,
        hardware_validated=bool(_nested(physics, "mike_constant_curvature.validated_against_hardware")),
        hardware_validation_required=[] if bool(_nested(physics, "mike_constant_curvature.validated_against_hardware")) else ["compare predicted intermediate/distal poses against tracked hardware data"],
        **categories,
    )


def assess_camarillo_status(config: dict[str, Any]) -> PhysicsAdapterStatus:
    physics = _physics(config)
    missing = _missing_parameters(physics, [path.removeprefix("physics_models.") for path in CAMARILLO_REQUIRED_PARAMETERS])
    categories = _parameter_categories(physics)
    explicit_blockers = [
        "measured/fitted segment stiffness",
        "tendon/cable stiffness",
        "additional cable routing/effective length",
        "hardware-validated frame/sign conventions",
    ]
    convention_notes = [
        "Legacy Camarillo math maps tendon length changes through stiffness matrices to Webster parameters.",
        "Two-segment use requires validated stiffness, cable stiffness, routing, sign, and frame conventions.",
        "The active scaffold does not emit Camarillo predictions until those conventions are confirmed against current hardware.",
    ]
    status = "unavailable_missing_parameters" if missing else "unavailable_unvalidated_convention"
    reason = (
        "Missing required Camarillo parameters: " + ", ".join(missing)
        if missing
        else "Camarillo parameters are present, but current two-segment sign/frame/stiffness conventions are not validated in active code."
    )
    return PhysicsAdapterStatus(
        model_key="camarillo",
        label="Camarillo",
        status=status,
        reason=reason,
        required_parameters=CAMARILLO_REQUIRED_PARAMETERS,
        missing_parameters=["physics_models." + path for path in missing],
        convention_notes=convention_notes,
        can_predict_distal_xyz=False,
        can_predict_intermediate_xyz=False,
        can_predict_orientation=False,
        hardware_validated=bool(_nested(physics, "camarillo.validated_against_hardware")),
        predictions_generated=False,
        blocking_missing_parameters=["physics_models." + path for path in missing] + explicit_blockers,
        hardware_validation_required=["fit/validate Camarillo parameters against tracked two-segment hardware data"],
        **categories,
    )


def two_segment_constant_curvature_prediction(command_mm: np.ndarray, config: dict[str, Any]) -> dict[str, np.ndarray]:
    """Predict intermediate/distal transforms for an explicit two-segment CC model."""

    physics = _physics(config)
    sign = str(_param_value(_nested(physics, "global.tendon_displacement_sign_convention")) or "positive_shortens_tendon")
    seg_a = _as_dict(_nested(physics, "segments.segment_a"))
    seg_b = _as_dict(_nested(physics, "segments.segment_b"))
    command = np.asarray(command_mm, dtype=float).reshape((8,))
    T_a = _segment_transform(command[:4], segment=seg_a, sign_convention=sign)
    T_b = _segment_transform(command[4:], segment=seg_b, sign_convention=sign)
    T_distal = T_a @ T_b
    return {
        "T_intermediate": T_a,
        "T_distal": T_distal,
        "intermediate_xyz": T_a[:3, 3].copy(),
        "distal_xyz": T_distal[:3, 3].copy(),
        "intermediate_tangent": _normalize(T_a[:3, 2]),
        "distal_tangent": _normalize(T_distal[:3, 2]),
    }


@dataclass(frozen=True)
class MikeCCConventionReport:
    """Evidence-based check of Mike constant-curvature sign/frame conventions.

    The operator collects a small dataset of single-axis sweeps and runs this
    probe to compare predicted vs measured distal/intermediate XYZ. The report
    surfaces residual statistics + a recommendation. It never auto-flips the
    convention flag in config — that decision stays with the operator after
    looking at the numbers.
    """

    sample_count: int
    distal_residual_norm_mean_mm: float
    distal_residual_norm_median_mm: float
    distal_residual_norm_max_mm: float
    distal_residual_norm_p95_mm: float
    intermediate_residual_norm_mean_mm: float | None
    intermediate_residual_norm_max_mm: float | None
    per_axis_distal_signs_match: dict[str, bool]
    largest_distal_sign_disagreement_axis: str | None
    per_axis_intermediate_signs_match: dict[str, bool]
    largest_intermediate_sign_disagreement_axis: str | None
    intermediate_sample_count: int
    recommendation: str
    notes: list[str]
    residuals_by_sample: list[dict[str, Any]]

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": "mike_cc_convention_report_v1",
            "sample_count": int(self.sample_count),
            "distal_residual_norm_mean_mm": float(self.distal_residual_norm_mean_mm),
            "distal_residual_norm_median_mm": float(self.distal_residual_norm_median_mm),
            "distal_residual_norm_max_mm": float(self.distal_residual_norm_max_mm),
            "distal_residual_norm_p95_mm": float(self.distal_residual_norm_p95_mm),
            "intermediate_residual_norm_mean_mm": (
                float(self.intermediate_residual_norm_mean_mm)
                if self.intermediate_residual_norm_mean_mm is not None
                else None
            ),
            "intermediate_residual_norm_max_mm": (
                float(self.intermediate_residual_norm_max_mm)
                if self.intermediate_residual_norm_max_mm is not None
                else None
            ),
            "per_axis_distal_signs_match": {str(key): bool(value) for key, value in self.per_axis_distal_signs_match.items()},
            "largest_distal_sign_disagreement_axis": self.largest_distal_sign_disagreement_axis,
            "per_axis_intermediate_signs_match": {str(key): bool(value) for key, value in self.per_axis_intermediate_signs_match.items()},
            "largest_intermediate_sign_disagreement_axis": self.largest_intermediate_sign_disagreement_axis,
            "intermediate_sample_count": int(self.intermediate_sample_count),
            "recommendation": str(self.recommendation),
            "notes": [str(item) for item in self.notes],
            "residuals_by_sample": [dict(row) for row in self.residuals_by_sample],
        }


def validate_mike_cc_conventions(
    *,
    samples: list[Any],
    config: dict[str, Any],
    distal_mean_threshold_mm: float = 5.0,
    distal_max_threshold_mm: float = 15.0,
) -> MikeCCConventionReport:
    """Run a non-destructive convention probe against measured samples.

    For each sample with an 8-tendon command and a measured distal_position_mm,
    predict the distal XYZ using the configured Mike CC math and compute the
    residual against the measurement. Sign-match per axis tells the operator
    whether the predicted motion direction agrees with the measured direction
    — a cheap way to catch a flipped tendon-displacement sign convention.

    Thresholds are conservative defaults; the operator can override them via
    the CLI. ``samples`` items are expected to expose ``feature_mm`` (8-vector
    in mm) and ``distal_position_mm`` (3-vector in mm in robot frame); see
    ``TwoSegmentModelingSample``.
    """

    if not samples:
        return MikeCCConventionReport(
            sample_count=0,
            distal_residual_norm_mean_mm=float("nan"),
            distal_residual_norm_median_mm=float("nan"),
            distal_residual_norm_max_mm=float("nan"),
            distal_residual_norm_p95_mm=float("nan"),
            intermediate_residual_norm_mean_mm=None,
            intermediate_residual_norm_max_mm=None,
            per_axis_distal_signs_match={},
            largest_distal_sign_disagreement_axis=None,
            per_axis_intermediate_signs_match={},
            largest_intermediate_sign_disagreement_axis=None,
            intermediate_sample_count=0,
            recommendation="no_samples_provided",
            notes=["Need at least one accepted sample with measured distal_position_mm and 8-tendon feature_mm."],
            residuals_by_sample=[],
        )

    residuals: list[np.ndarray] = []
    intermediate_residuals: list[float] = []
    rows: list[dict[str, Any]] = []
    distal_predicted_signs: list[np.ndarray] = []
    distal_measured_signs: list[np.ndarray] = []
    intermediate_predicted_signs: list[np.ndarray] = []
    intermediate_measured_signs: list[np.ndarray] = []

    for sample in samples:
        feature_mm = np.asarray(getattr(sample, "feature_mm", np.zeros(8)), dtype=float).reshape((8,))
        measured_distal = getattr(sample, "distal_position_mm", None)
        if measured_distal is None:
            continue
        measured_distal_arr = np.asarray(measured_distal, dtype=float).reshape((3,))
        try:
            prediction = two_segment_constant_curvature_prediction(feature_mm, config)
        except Exception as exc:
            rows.append(
                {
                    "sample_index": int(getattr(sample, "sample_index", -1)),
                    "feature_mm": feature_mm.tolist(),
                    "prediction_error": str(exc),
                }
            )
            continue
        predicted_distal = np.asarray(prediction["distal_xyz"], dtype=float).reshape((3,))
        residual = predicted_distal - measured_distal_arr
        residuals.append(residual)
        rows.append(
            {
                "sample_index": int(getattr(sample, "sample_index", -1)),
                "feature_mm": feature_mm.tolist(),
                "measured_distal_mm": measured_distal_arr.tolist(),
                "predicted_distal_mm": predicted_distal.tolist(),
                "residual_mm": residual.tolist(),
                "residual_norm_mm": float(np.linalg.norm(residual)),
            }
        )
        distal_predicted_signs.append(np.sign(predicted_distal))
        distal_measured_signs.append(np.sign(measured_distal_arr))
        measured_intermediate = getattr(sample, "intermediate_position_mm", None)
        if measured_intermediate is not None:
            predicted_intermediate = np.asarray(prediction["intermediate_xyz"], dtype=float).reshape((3,))
            measured_intermediate_arr = np.asarray(measured_intermediate, dtype=float).reshape((3,))
            intermediate_residuals.append(float(np.linalg.norm(predicted_intermediate - measured_intermediate_arr)))
            intermediate_predicted_signs.append(np.sign(predicted_intermediate))
            intermediate_measured_signs.append(np.sign(measured_intermediate_arr))

    if not residuals:
        prediction_errors = [row for row in rows if "prediction_error" in row]
        if prediction_errors:
            distinct_errors = sorted({str(row.get("prediction_error") or "") for row in prediction_errors})
            recommendation = "config_incomplete_prediction_failed"
            notes = [
                "Mike CC prediction raised exceptions on every sample — most likely the modeling config is missing required fields.",
                f"Distinct errors: {distinct_errors}",
                "Check `physics_models.segments.segment_a.segment_length_mm`, `.tendon_positions_mm`, and the same for segment_b.",
                "See config/modeling_two_segment.example.yaml for a complete reference.",
            ]
        else:
            recommendation = "no_predictions_generated"
            notes = [
                "No samples had a measured distal_position_mm in robot frame.",
                "Confirm the dataset includes distal_tip pose labels (run with a connected tracker).",
            ]
        return MikeCCConventionReport(
            sample_count=int(len(samples)),
            distal_residual_norm_mean_mm=float("nan"),
            distal_residual_norm_median_mm=float("nan"),
            distal_residual_norm_max_mm=float("nan"),
            distal_residual_norm_p95_mm=float("nan"),
            intermediate_residual_norm_mean_mm=None,
            intermediate_residual_norm_max_mm=None,
            per_axis_distal_signs_match={},
            largest_distal_sign_disagreement_axis=None,
            per_axis_intermediate_signs_match={},
            largest_intermediate_sign_disagreement_axis=None,
            intermediate_sample_count=0,
            recommendation=recommendation,
            notes=notes,
            residuals_by_sample=rows,
        )

    residual_array = np.stack(residuals, axis=0)
    norms = np.linalg.norm(residual_array, axis=1)
    intermediate_mean = float(np.mean(intermediate_residuals)) if intermediate_residuals else None
    intermediate_max = float(np.max(intermediate_residuals)) if intermediate_residuals else None
    axis_labels = ("x", "y", "z")

    def _per_axis_match_and_largest_disagreement(
        predicted_signs: list[np.ndarray], measured_signs: list[np.ndarray]
    ) -> tuple[dict[str, bool], str | None, dict[str, int], int]:
        if not predicted_signs:
            return {}, None, {}, 0
        predicted_array = np.stack(predicted_signs, axis=0)
        measured_array = np.stack(measured_signs, axis=0)
        per_axis_match: dict[str, bool] = {}
        disagreement_counts: dict[str, int] = {}
        for axis_index, axis_label in enumerate(axis_labels):
            mask = (predicted_array[:, axis_index] != 0) & (measured_array[:, axis_index] != 0)
            disagree = int(
                np.sum((predicted_array[mask, axis_index] != measured_array[mask, axis_index]).astype(int))
            )
            total = int(np.sum(mask.astype(int)))
            match_fraction = 1.0 - (float(disagree) / float(total)) if total > 0 else 1.0
            per_axis_match[axis_label] = bool(match_fraction >= 0.70)
            disagreement_counts[axis_label] = disagree
        largest = max(disagreement_counts, key=lambda key: disagreement_counts[key]) if disagreement_counts else None
        if largest is not None and disagreement_counts[largest] == 0:
            largest = None
        return per_axis_match, largest, disagreement_counts, int(predicted_array.shape[0])

    (
        per_axis_signs_match,
        largest_disagreement_axis,
        sign_disagreement_counts,
        distal_sign_sample_count,
    ) = _per_axis_match_and_largest_disagreement(distal_predicted_signs, distal_measured_signs)
    (
        per_axis_intermediate_signs_match,
        largest_intermediate_disagreement_axis,
        intermediate_disagreement_counts,
        intermediate_sample_count,
    ) = _per_axis_match_and_largest_disagreement(intermediate_predicted_signs, intermediate_measured_signs)

    mean_norm = float(np.mean(norms))
    median_norm = float(np.median(norms))
    max_norm = float(np.max(norms))
    p95_norm = float(np.percentile(norms, 95))

    notes: list[str] = []
    all_distal_signs_ok = all(per_axis_signs_match.values()) if per_axis_signs_match else False
    all_intermediate_signs_ok = (
        all(per_axis_intermediate_signs_match.values()) if per_axis_intermediate_signs_match else True
    )
    if not all_distal_signs_ok and largest_disagreement_axis is not None:
        notes.append(
            f"Distal predicted sign disagrees with measured sign on axis {largest_disagreement_axis} "
            f"({sign_disagreement_counts[largest_disagreement_axis]}/{distal_sign_sample_count} samples). "
            "Re-check `tendon_displacement_sign_convention` and `model_frame_convention`."
        )
    if (
        per_axis_intermediate_signs_match
        and not all_intermediate_signs_ok
        and largest_intermediate_disagreement_axis is not None
    ):
        notes.append(
            f"Intermediate predicted sign disagrees with measured sign on axis {largest_intermediate_disagreement_axis} "
            f"({intermediate_disagreement_counts[largest_intermediate_disagreement_axis]}/{intermediate_sample_count} samples). "
            "If only intermediate signs are off (distal OK), the bottom-segment frame may be flipped relative to top."
        )
    if mean_norm > distal_mean_threshold_mm:
        notes.append(
            f"Distal residual mean {mean_norm:.2f} mm > {distal_mean_threshold_mm:.1f} mm threshold. "
            "Check segment lengths, tendon positions, and frame conventions."
        )
    if max_norm > distal_max_threshold_mm:
        notes.append(
            f"Distal residual max {max_norm:.2f} mm > {distal_max_threshold_mm:.1f} mm threshold. "
            "Inspect outlier samples in residuals_by_sample."
        )

    all_signs_ok = all_distal_signs_ok and all_intermediate_signs_ok
    if all_signs_ok and mean_norm <= distal_mean_threshold_mm and max_norm <= distal_max_threshold_mm:
        recommendation = "conventions_consistent_with_evidence_safe_to_confirm"
        notes.append(
            "Sign-match per axis is OK and residuals are below thresholds. You may set "
            "`physics_models.mike_constant_curvature.required_conventions_confirmed: true` in the modeling config. "
            "Treat the model as 'available' but still report measured-vs-predicted errors in the run summary."
        )
    elif all_signs_ok:
        recommendation = "magnitude_off_but_signs_ok_inspect_geometry"
    elif (
        per_axis_intermediate_signs_match
        and all_distal_signs_ok
        and not all_intermediate_signs_ok
        and largest_intermediate_disagreement_axis is not None
    ):
        recommendation = (
            f"intermediate_sign_convention_likely_flipped_axis_{largest_intermediate_disagreement_axis}"
        )
    elif largest_disagreement_axis is not None:
        recommendation = f"sign_convention_likely_flipped_axis_{largest_disagreement_axis}"
    else:
        recommendation = "ambiguous_collect_more_samples"

    return MikeCCConventionReport(
        sample_count=int(len(residuals)),
        distal_residual_norm_mean_mm=mean_norm,
        distal_residual_norm_median_mm=median_norm,
        distal_residual_norm_max_mm=max_norm,
        distal_residual_norm_p95_mm=p95_norm,
        intermediate_residual_norm_mean_mm=intermediate_mean,
        intermediate_residual_norm_max_mm=intermediate_max,
        per_axis_distal_signs_match=per_axis_signs_match,
        largest_distal_sign_disagreement_axis=largest_disagreement_axis,
        per_axis_intermediate_signs_match=per_axis_intermediate_signs_match,
        largest_intermediate_sign_disagreement_axis=largest_intermediate_disagreement_axis,
        intermediate_sample_count=intermediate_sample_count,
        recommendation=recommendation,
        notes=notes,
        residuals_by_sample=rows,
    )


def fill_prediction_from_role_poses(
    *,
    label_metadata: dict[str, Any],
    intermediate_xyz: np.ndarray,
    distal_xyz: np.ndarray,
    intermediate_tangent: np.ndarray,
    distal_tangent: np.ndarray,
) -> np.ndarray:
    names = list(label_metadata.get("label_names") or [])
    y = np.zeros((len(names),), dtype=float)
    slices = label_metadata.get("label_slices") if isinstance(label_metadata.get("label_slices"), dict) else {}
    _fill_slice(y, slices, "intermediate_segment", "position", intermediate_xyz)
    _fill_slice(y, slices, "distal_tip", "position", distal_xyz)
    _fill_slice(y, slices, "intermediate_segment", "orientation", intermediate_tangent)
    _fill_slice(y, slices, "distal_tip", "orientation", distal_tangent)
    return y


def _segment_transform(command_mm: np.ndarray, *, segment: dict[str, Any], sign_convention: str) -> np.ndarray:
    length = float(_param_value(segment["segment_length_mm"]))
    positions = np.asarray(_param_value(segment["tendon_positions_mm"]), dtype=float)
    if positions.shape != (4, 2):
        raise ValueError("tendon_positions_mm must contain four [x,y] positions per segment.")
    deltas = np.asarray(command_mm, dtype=float).reshape((4,))
    sign = -1.0 if sign_convention == "positive_shortens_tendon" else 1.0
    solution, *_ = np.linalg.lstsq(positions, sign * deltas / max(length, 1e-12), rcond=None)
    kx = float(solution[0])
    ky = float(solution[1])
    return _constant_curvature_transform(length_mm=length, kx=kx, ky=ky)


def _constant_curvature_transform(*, length_mm: float, kx: float, ky: float) -> np.ndarray:
    kappa = math.sqrt(kx * kx + ky * ky)
    if kappa <= 1e-12:
        T = np.eye(4, dtype=float)
        T[2, 3] = float(length_mm)
        return T
    phi = math.atan2(ky, kx)
    return _calculate_transform(np.asarray([float(length_mm), kappa, phi], dtype=float))


def _calculate_transform(webster_params: np.ndarray) -> np.ndarray:
    transform = np.eye(4, dtype=float)
    for theta, d, r, alpha in _dh_parameters(np.asarray(webster_params, dtype=float).reshape((3,))):
        transform = transform @ _dh_transform(theta=theta, d=d, r=r, alpha=alpha)
    return transform


def _dh_parameters(webster_params: np.ndarray) -> list[tuple[float, float, float, float]]:
    length = float(webster_params[0])
    kappa = float(webster_params[1])
    phi = float(webster_params[2])
    if abs(kappa) <= 1e-12:
        return [(0.0, length, 0.0, 0.0)]
    return [
        (phi, 0.0, 0.0, -math.pi / 2.0),
        (kappa * length / 2.0, 0.0, 0.0, math.pi / 2.0),
        (0.0, (2.0 / kappa) * math.sin(kappa * length / 2.0), 0.0, -math.pi / 2.0),
        (kappa * length / 2.0, 0.0, 0.0, math.pi / 2.0),
        (-phi, 0.0, 0.0, 0.0),
    ]


def _dh_transform(*, theta: float, d: float, r: float, alpha: float) -> np.ndarray:
    z_transform = np.asarray(
        [
            [math.cos(theta), -math.sin(theta), 0.0, 0.0],
            [math.sin(theta), math.cos(theta), 0.0, 0.0],
            [0.0, 0.0, 1.0, d],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=float,
    )
    x_transform = np.asarray(
        [
            [1.0, 0.0, 0.0, r],
            [0.0, math.cos(alpha), -math.sin(alpha), 0.0],
            [0.0, math.sin(alpha), math.cos(alpha), 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=float,
    )
    return z_transform @ x_transform


def _fill_slice(y: np.ndarray, slices: dict[str, Any], role: str, kind: str, value: np.ndarray) -> None:
    role_slices = slices.get(role) if isinstance(slices.get(role), dict) else {}
    indices = role_slices.get(kind)
    if isinstance(indices, list) and len(indices) == 3:
        y[[int(index) for index in indices]] = np.asarray(value, dtype=float).reshape((3,))


def _physics(config: dict[str, Any]) -> dict[str, Any]:
    raw = dict(config or {})
    return _as_dict(raw.get("physics_models") or raw.get("mike_constant_curvature") or raw.get("camarillo") or {})


def _missing_parameters(payload: dict[str, Any], paths: list[str]) -> list[str]:
    missing: list[str] = []
    for path in paths:
        value = _param_value(_nested(payload, path))
        if value in (None, "", []):
            missing.append(path)
    return missing


def _nested(payload: dict[str, Any], dotted: str) -> Any:
    current: Any = payload
    for key in dotted.split("."):
        if not isinstance(current, dict):
            return None
        current = current.get(key)
    return current


def _as_dict(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, dict) else {}


def _param_value(value: Any) -> Any:
    if isinstance(value, dict) and "value" in value:
        return value.get("value")
    return value


def _parameter_categories(physics: dict[str, Any]) -> dict[str, list[str]]:
    categories = {
        "known_design_parameters": [],
        "measured_parameters": [],
        "estimated_parameters": [],
        "unknown_parameters": [],
        "nonblocking_uncertain_parameters": [],
    }
    for path, payload in _walk_parameter_wrappers(physics):
        source = str(payload.get("source") or "unknown").lower()
        value = payload.get("value")
        full_path = "physics_models." + path
        if source == "design" and value not in (None, "", []):
            categories["known_design_parameters"].append(full_path)
        elif source == "measured" and value not in (None, "", []):
            categories["measured_parameters"].append(full_path)
        elif source == "estimated" and value not in (None, "", []):
            categories["estimated_parameters"].append(full_path)
        if source == "unknown" or value in (None, "", []):
            categories["unknown_parameters"].append(full_path)
        notes = str(payload.get("notes") or "").lower()
        if "uncertain" in notes or "confirm whether" in notes or "validate" in notes:
            categories["nonblocking_uncertain_parameters"].append(full_path)
    return {key: sorted(set(value)) for key, value in categories.items()}


def _walk_parameter_wrappers(payload: Any, prefix: str = ""):
    if isinstance(payload, dict) and {"value", "source"}.issubset(payload.keys()):
        yield prefix.strip("."), payload
        return
    if isinstance(payload, dict):
        for key, value in payload.items():
            yield from _walk_parameter_wrappers(value, f"{prefix}.{key}" if prefix else str(key))


def _normalize(value: np.ndarray) -> np.ndarray:
    vector = np.asarray(value, dtype=float).reshape((3,))
    norm = float(np.linalg.norm(vector))
    return vector / norm if norm > 0.0 else np.asarray([0.0, 0.0, 1.0], dtype=float)

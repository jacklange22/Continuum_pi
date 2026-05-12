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

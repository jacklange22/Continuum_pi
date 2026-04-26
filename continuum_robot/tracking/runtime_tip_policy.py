"""Shared runtime tip trust policy.

The policy is intentionally conservative and explicit: coil-as-tip is the
current thesis-trusted runtime position path, while calibrated runtime tip
artifacts remain lower-trust until their own validation ladder is complete.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


RUNTIME_TIP_MODE_LATEST_ACCEPTED = "latest_accepted"
RUNTIME_TIP_MODE_QUICK_4_POINT = "quick_4_point"
RUNTIME_TIP_MODE_COIL_AS_TIP = "coil_as_tip"

TRUST_THESIS = "thesis_trusted"
TRUST_LOWER = "lower_trust"
TRUST_DEBUG = "debug_only"
TRUST_UNAVAILABLE = "unavailable"

WORKFLOW_REPEATABILITY = "repeatability"
WORKFLOW_PRETENSION_VALIDATION = "pretension_validation"
WORKFLOW_PENPROBE_CHASING = "penprobe_chasing"
WORKFLOW_MODELING_DATASET = "modeling_dataset"
WORKFLOW_TRANSFORM_CHAIN_VALIDATION = "transform_chain_validation"
WORKFLOW_GENERIC = "generic"

KNOWN_WORKFLOWS = (
    WORKFLOW_REPEATABILITY,
    WORKFLOW_PRETENSION_VALIDATION,
    WORKFLOW_PENPROBE_CHASING,
    WORKFLOW_MODELING_DATASET,
    WORKFLOW_TRANSFORM_CHAIN_VALIDATION,
)


@dataclass(frozen=True)
class RuntimeTipWorkflowResolution:
    """Canonical workflow resolution for policy evaluation."""

    requested_workflow: str
    canonical_workflow: str


_WORKFLOW_ALIASES: dict[str, str] = {
    WORKFLOW_GENERIC: WORKFLOW_GENERIC,
    WORKFLOW_REPEATABILITY: WORKFLOW_REPEATABILITY,
    "single_segment_repeatability": WORKFLOW_REPEATABILITY,
    "single_segment_repeatability_platform": WORKFLOW_REPEATABILITY,
    "repeatability_dataset": WORKFLOW_REPEATABILITY,
    "repeatability_platform": WORKFLOW_REPEATABILITY,
    WORKFLOW_PRETENSION_VALIDATION: WORKFLOW_PRETENSION_VALIDATION,
    "pretension": WORKFLOW_PRETENSION_VALIDATION,
    "pretension_platform": WORKFLOW_PRETENSION_VALIDATION,
    WORKFLOW_PENPROBE_CHASING: WORKFLOW_PENPROBE_CHASING,
    "penprobe": WORKFLOW_PENPROBE_CHASING,
    "penprobe_platform": WORKFLOW_PENPROBE_CHASING,
    WORKFLOW_MODELING_DATASET: WORKFLOW_MODELING_DATASET,
    "collect_pose_command_dataset": WORKFLOW_MODELING_DATASET,
    "motor_babble": WORKFLOW_MODELING_DATASET,
    "modeling": WORKFLOW_MODELING_DATASET,
    "modeling_platform": WORKFLOW_MODELING_DATASET,
    WORKFLOW_TRANSFORM_CHAIN_VALIDATION: WORKFLOW_TRANSFORM_CHAIN_VALIDATION,
    "transform_chain": WORKFLOW_TRANSFORM_CHAIN_VALIDATION,
    "transform_chain_platform": WORKFLOW_TRANSFORM_CHAIN_VALIDATION,
    "registration_validation": WORKFLOW_GENERIC,
    "pivot_validation": WORKFLOW_GENERIC,
    "aurora_grid_accuracy": WORKFLOW_GENERIC,
    "pivot_calibration": WORKFLOW_GENERIC,
    "tracker_pipeline_mock": WORKFLOW_GENERIC,
    "command_schedule_validation": WORKFLOW_GENERIC,
    "dataset_schema_roundtrip": WORKFLOW_GENERIC,
    "replay_runner": WORKFLOW_GENERIC,
    "tracker_timing_validation": WORKFLOW_GENERIC,
    "servo_tracker_sync_validation": WORKFLOW_GENERIC,
}


@dataclass(frozen=True)
class RuntimeTipTrustEvaluation:
    """One normalized trust decision for the active runtime tip path."""

    mode: str
    calibration_state: str
    tip_pose_status: str
    trust_label: str
    thesis_trusted: bool
    requested_workflow: str = WORKFLOW_GENERIC
    workflow: str = WORKFLOW_GENERIC
    allowed_for_workflow: bool = False
    requires_lower_trust_override: bool = False
    uses_coil_as_tip: bool = False
    uses_calibrated_tip_transform: bool = False
    uses_identity_transform: bool = False
    intended_use: str = ""
    status_message: str = ""
    reasons: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    allowed_workflows: dict[str, bool] = field(default_factory=dict)

    @property
    def trust_level(self) -> str:
        """Compatibility alias for older metadata fields."""

        return self.trust_label

    @property
    def canonical_workflow(self) -> str:
        """Compatibility alias for explicit canonical naming."""

        return self.workflow

    def to_dict(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "calibration_state": self.calibration_state,
            "tip_pose_status": self.tip_pose_status,
            "trust_label": self.trust_label,
            "trust_level": self.trust_label,
            "thesis_trusted": bool(self.thesis_trusted),
            "requested_workflow": self.requested_workflow,
            "workflow": self.workflow,
            "canonical_workflow": self.workflow,
            "allowed_for_workflow": bool(self.allowed_for_workflow),
            "requires_lower_trust_override": bool(self.requires_lower_trust_override),
            "uses_coil_as_tip": bool(self.uses_coil_as_tip),
            "uses_calibrated_tip_transform": bool(self.uses_calibrated_tip_transform),
            "uses_identity_transform": bool(self.uses_identity_transform),
            "intended_use": self.intended_use,
            "status_message": self.status_message,
            "reasons": list(self.reasons),
            "warnings": list(self.warnings),
            "allowed_workflows": dict(self.allowed_workflows),
        }


def evaluate_runtime_tip_trust(
    *,
    snapshot: Any | None = None,
    mode: str | None = None,
    calibration_state: str | None = None,
    tip_pose_status: str | None = None,
    has_robot_tip: bool | None = None,
    identity_fallback: bool | None = None,
    workflow: str | None = None,
    allow_lower_trust: bool = False,
) -> RuntimeTipTrustEvaluation:
    """Evaluate the active runtime tip mode using the shared project policy."""

    if snapshot is not None:
        mode = mode if mode is not None else getattr(snapshot, "runtime_tip_mode", None)
        calibration_state = (
            calibration_state
            if calibration_state is not None
            else getattr(snapshot, "runtime_tip_calibration_state", None)
        )
        tip_pose_status = tip_pose_status if tip_pose_status is not None else getattr(snapshot, "tip_pose_status", None)
        if has_robot_tip is None:
            has_robot_tip = getattr(snapshot, "T_robot_tip", None) is not None
        if identity_fallback is None:
            identity_fallback = bool(getattr(snapshot, "runtime_tip_identity_fallback", False))

    normalized_mode = _normalize_mode(mode)
    state = str(calibration_state or "").strip() or "missing_runtime_tip_calibration"
    pose_status = str(tip_pose_status or "").strip() or "missing_runtime_tip"
    workflow_resolution = resolve_runtime_tip_workflow(workflow)
    workflow_name = workflow_resolution.canonical_workflow
    robot_tip_available = bool(has_robot_tip)
    identity = bool(identity_fallback)

    warnings: list[str] = []
    reasons: list[str] = []

    if normalized_mode == RUNTIME_TIP_MODE_COIL_AS_TIP:
        ready = robot_tip_available and pose_status in {"ok", "coil_as_tip"} and state == "coil_as_tip"
        trust_label = TRUST_THESIS if ready else TRUST_UNAVAILABLE
        thesis_trusted = bool(ready)
        allowed_workflows = {
            WORKFLOW_REPEATABILITY: bool(ready),
            WORKFLOW_PRETENSION_VALIDATION: bool(ready),
            WORKFLOW_PENPROBE_CHASING: bool(ready),
            WORKFLOW_MODELING_DATASET: bool(ready),
            WORKFLOW_TRANSFORM_CHAIN_VALIDATION: True,
        }
        intended_use = (
            "Current trusted runtime position path for robot-frame tip repeatability, pretension "
            "centering, modeling datasets, and demos when coil-origin-as-tip is the declared measurement."
        )
        if ready:
            warnings.append(
                "Coil-as-tip uses identity T_coil_tip and reports the 0A coil origin as the tip; "
                "do not interpret it as a validated calibrated tip transform or true tip orientation."
            )
        else:
            reasons.extend(_readiness_reasons(state=state, pose_status=pose_status, has_robot_tip=robot_tip_available))
        return _build_evaluation(
            mode=normalized_mode,
            state=state,
            pose_status=pose_status,
            trust_label=trust_label,
            thesis_trusted=thesis_trusted,
            requested_workflow=workflow_resolution.requested_workflow,
            workflow=workflow_name,
            allowed_workflows=allowed_workflows,
            allow_lower_trust=allow_lower_trust,
            uses_coil_as_tip=True,
            uses_calibrated_tip_transform=False,
            uses_identity_transform=True,
            intended_use=intended_use,
            status_message=(
                "Coil-as-tip is active: identity T_coil_tip, the 0A coil pose is shown directly as the tip."
            ),
            reasons=reasons,
            warnings=warnings,
        )

    if normalized_mode == RUNTIME_TIP_MODE_LATEST_ACCEPTED:
        ready = robot_tip_available and pose_status == "ok" and state == "loaded" and not identity
        trust_label = TRUST_LOWER if ready else TRUST_UNAVAILABLE
        allowed_workflows = {
            WORKFLOW_REPEATABILITY: False,
            WORKFLOW_PRETENSION_VALIDATION: bool(ready),
            WORKFLOW_PENPROBE_CHASING: False,
            WORKFLOW_MODELING_DATASET: bool(ready),
            WORKFLOW_TRANSFORM_CHAIN_VALIDATION: True,
        }
        intended_use = (
            "Accepted runtime tip calibration artifact for validation and lower-trust data collection; "
            "not thesis-trusted until the calibration workflow is independently validated."
        )
        if ready:
            warnings.append(
                "Latest accepted runtime tip calibration is loaded but currently lower-trust than coil-as-tip; "
                "runs must record this explicitly."
            )
        else:
            reasons.extend(_readiness_reasons(state=state, pose_status=pose_status, has_robot_tip=robot_tip_available))
            if identity:
                reasons.append("runtime_tip_identity_fallback")
        return _build_evaluation(
            mode=normalized_mode,
            state=state,
            pose_status=pose_status,
            trust_label=trust_label,
            thesis_trusted=False,
            requested_workflow=workflow_resolution.requested_workflow,
            workflow=workflow_name,
            allowed_workflows=allowed_workflows,
            allow_lower_trust=allow_lower_trust,
            uses_coil_as_tip=False,
            uses_calibrated_tip_transform=bool(ready),
            uses_identity_transform=bool(identity),
            intended_use=intended_use,
            status_message="Latest accepted runtime tip artifact is active but marked lower_trust by policy.",
            reasons=reasons,
            warnings=warnings,
        )

    if normalized_mode == RUNTIME_TIP_MODE_QUICK_4_POINT:
        ready = robot_tip_available and pose_status == "ok" and state == "quick_4_point_loaded" and not identity
        trust_label = TRUST_DEBUG if ready else TRUST_UNAVAILABLE
        allowed_workflows = {
            WORKFLOW_REPEATABILITY: False,
            WORKFLOW_PRETENSION_VALIDATION: bool(ready),
            WORKFLOW_PENPROBE_CHASING: False,
            WORKFLOW_MODELING_DATASET: bool(ready),
            WORKFLOW_TRANSFORM_CHAIN_VALIDATION: True,
        }
        intended_use = "Bench/debug override for rapid runtime tip checks; not thesis-trusted."
        if ready:
            warnings.append("Quick 4-point runtime tip override is debug_only and must not be presented as thesis-trusted.")
        else:
            reasons.extend(_readiness_reasons(state=state, pose_status=pose_status, has_robot_tip=robot_tip_available))
            if identity:
                reasons.append("runtime_tip_identity_fallback")
        return _build_evaluation(
            mode=normalized_mode,
            state=state,
            pose_status=pose_status,
            trust_label=trust_label,
            thesis_trusted=False,
            requested_workflow=workflow_resolution.requested_workflow,
            workflow=workflow_name,
            allowed_workflows=allowed_workflows,
            allow_lower_trust=allow_lower_trust,
            uses_coil_as_tip=False,
            uses_calibrated_tip_transform=bool(ready),
            uses_identity_transform=bool(identity),
            intended_use=intended_use,
            status_message="Quick 4-point runtime tip override is active and debug_only.",
            reasons=reasons,
            warnings=warnings,
        )

    return _build_evaluation(
        mode=normalized_mode,
        state=state,
        pose_status=pose_status,
        trust_label=TRUST_UNAVAILABLE,
        thesis_trusted=False,
        requested_workflow=workflow_resolution.requested_workflow,
        workflow=workflow_name,
        allowed_workflows={workflow: False for workflow in KNOWN_WORKFLOWS},
        allow_lower_trust=allow_lower_trust,
        uses_coil_as_tip=False,
        uses_calibrated_tip_transform=False,
        uses_identity_transform=identity,
        intended_use="Unknown runtime tip mode.",
        status_message=f"Unknown runtime tip mode {normalized_mode}.",
        reasons=["unknown_runtime_tip_mode"],
        warnings=[],
    )


def _build_evaluation(
    *,
    mode: str,
    state: str,
    pose_status: str,
    trust_label: str,
    thesis_trusted: bool,
    requested_workflow: str,
    workflow: str,
    allowed_workflows: dict[str, bool],
    allow_lower_trust: bool,
    uses_coil_as_tip: bool,
    uses_calibrated_tip_transform: bool,
    uses_identity_transform: bool,
    intended_use: str,
    status_message: str,
    reasons: list[str],
    warnings: list[str],
) -> RuntimeTipTrustEvaluation:
    base_allowed = bool(allowed_workflows.get(workflow, False)) if workflow != WORKFLOW_GENERIC else bool(thesis_trusted)
    requires_override = bool(base_allowed and trust_label in {TRUST_LOWER, TRUST_DEBUG} and workflow in {
        WORKFLOW_MODELING_DATASET,
        WORKFLOW_REPEATABILITY,
        WORKFLOW_PENPROBE_CHASING,
    })
    allowed = bool(base_allowed and (not requires_override or allow_lower_trust))
    return RuntimeTipTrustEvaluation(
        mode=mode,
        calibration_state=state,
        tip_pose_status=pose_status,
        trust_label=trust_label,
        thesis_trusted=bool(thesis_trusted),
        requested_workflow=requested_workflow,
        workflow=workflow,
        allowed_for_workflow=allowed,
        requires_lower_trust_override=requires_override,
        uses_coil_as_tip=bool(uses_coil_as_tip),
        uses_calibrated_tip_transform=bool(uses_calibrated_tip_transform),
        uses_identity_transform=bool(uses_identity_transform),
        intended_use=intended_use,
        status_message=status_message,
        reasons=list(reasons),
        warnings=list(warnings),
        allowed_workflows=dict(allowed_workflows),
    )


def _normalize_mode(mode: str | None) -> str:
    value = str(mode or RUNTIME_TIP_MODE_LATEST_ACCEPTED).strip().lower()
    if value in {RUNTIME_TIP_MODE_COIL_AS_TIP, "coil", "identity"}:
        return RUNTIME_TIP_MODE_COIL_AS_TIP
    if value in {RUNTIME_TIP_MODE_QUICK_4_POINT, "quick4", "quick"}:
        return RUNTIME_TIP_MODE_QUICK_4_POINT
    if value in {RUNTIME_TIP_MODE_LATEST_ACCEPTED, "latest", "accepted"}:
        return RUNTIME_TIP_MODE_LATEST_ACCEPTED
    return value


def resolve_runtime_tip_workflow(workflow: str | None) -> RuntimeTipWorkflowResolution:
    """Resolve a concrete workflow/platform name to the canonical policy workflow."""

    requested = str(workflow or "").strip()
    if not requested:
        return RuntimeTipWorkflowResolution(
            requested_workflow=WORKFLOW_GENERIC,
            canonical_workflow=WORKFLOW_GENERIC,
        )

    normalized = _normalize_workflow_key(requested)
    for candidate in _workflow_alias_candidates(normalized):
        canonical = _WORKFLOW_ALIASES.get(candidate)
        if canonical is not None:
            return RuntimeTipWorkflowResolution(
                requested_workflow=requested,
                canonical_workflow=canonical,
            )

    supported = ", ".join(KNOWN_WORKFLOWS)
    aliases = ", ".join(sorted(alias for alias in _WORKFLOW_ALIASES.keys() if alias != WORKFLOW_GENERIC))
    raise ValueError(
        f"Unknown runtime tip workflow/platform '{requested}'. "
        f"Supported canonical workflows: {supported}. "
        f"Known workflow aliases: {aliases}."
    )


def _normalize_workflow_key(value: str) -> str:
    normalized = value.strip().lower().replace("-", "_").replace(" ", "_")
    while "__" in normalized:
        normalized = normalized.replace("__", "_")
    return normalized.strip("_")


def _workflow_alias_candidates(value: str) -> tuple[str, ...]:
    candidates = [value]
    if value.startswith("experiment_"):
        candidates.append(value.removeprefix("experiment_"))
    for suffix in ("_platform", "_workflow", "_experiment"):
        if value.endswith(suffix):
            trimmed = value[: -len(suffix)]
            if trimmed:
                candidates.append(trimmed)
    # Preserve order while de-duplicating.
    return tuple(dict.fromkeys(candidates))


def _readiness_reasons(*, state: str, pose_status: str, has_robot_tip: bool) -> list[str]:
    reasons: list[str] = []
    if state not in {"loaded", "coil_as_tip", "quick_4_point_loaded"}:
        reasons.append(f"runtime_tip_state_{state}")
    if pose_status not in {"ok", "coil_as_tip"}:
        reasons.append(f"tip_pose_status_{pose_status}")
    if not has_robot_tip:
        reasons.append("missing_T_robot_tip")
    return reasons

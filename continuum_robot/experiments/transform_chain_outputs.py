"""Transform-chain summary artifacts shared by experiments and diagnostics."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from continuum_robot.tracking.runtime_tip_policy import (
    WORKFLOW_GENERIC,
    evaluate_runtime_tip_trust,
)
from continuum_robot.utils.time_utils import utc_now_iso


TRANSFORM_CHAIN_EXPRESSION = "T_robot_tip = T_robot_aurora @ T_aurora_coil_0A @ T_coil_tip"


def build_transform_chain_summary(
    *,
    snapshot,
    workflow: str = WORKFLOW_GENERIC,
    allow_lower_trust: bool = False,
    provenance_note: str = "",
) -> dict[str, Any]:
    """Build a compact, thesis-facing summary of the active transform chain."""

    policy = evaluate_runtime_tip_trust(
        snapshot=snapshot,
        workflow=workflow,
        allow_lower_trust=allow_lower_trust,
    )
    registration_path = str(getattr(snapshot, "registration_path", "") or "")
    runtime_tip_path = str(getattr(snapshot, "runtime_tip_selected_artifact_path", None) or getattr(snapshot, "runtime_tip_calibration_path", "") or "")
    warnings = list(policy.warnings)
    warnings.extend(str(message) for message in getattr(snapshot, "warning_messages", []) or [])
    if getattr(snapshot, "last_error", None):
        warnings.append(str(getattr(snapshot, "last_error")))
    return {
        "schema_version": "1.0",
        "generated_at_utc": utc_now_iso(),
        "provenance_note": str(provenance_note or ""),
        "tracker": {
            "backend_identity": str(getattr(snapshot, "backend_identity", "") or ""),
            "configured_backend_name": str(getattr(snapshot, "configured_backend_name", "") or ""),
            "selected_backend_name": str(getattr(snapshot, "selected_backend_name", "") or ""),
            "canonical_state": str(getattr(snapshot, "canonical_state", "") or ""),
            "last_frame_number": getattr(snapshot, "last_frame_number", None),
            "tracker_data_age_s": getattr(snapshot, "tracker_data_age_s", None),
            "runtime_coil_tool_id": str(getattr(snapshot, "runtime_coil_tool_id", "0A") or "0A"),
            "registration_tool_id": str(getattr(snapshot, "registration_tool_id", "0B") or "0B"),
        },
        "registration": {
            "state": str(getattr(snapshot, "registration_state", "") or ""),
            "artifact_path": registration_path,
            "artifact_exists": bool(Path(registration_path).exists()) if registration_path else False,
            "stored_timestamp_utc": getattr(snapshot, "stored_registration_timestamp_utc", None),
            "fre_mm": getattr(snapshot, "stored_registration_fre_mm", None),
            "measurement_tool_id": getattr(snapshot, "stored_registration_measurement_tool_id", None),
            "coil_tool_id": getattr(snapshot, "stored_registration_coil_tool_id", None),
        },
        "runtime_tip": {
            "mode": policy.mode,
            "trust_label": policy.trust_label,
            "thesis_trusted": bool(policy.thesis_trusted),
            "allowed_for_workflow": bool(policy.allowed_for_workflow),
            "workflow": policy.workflow,
            "uses_coil_as_tip": bool(policy.uses_coil_as_tip),
            "uses_calibrated_tip_transform": bool(policy.uses_calibrated_tip_transform),
            "uses_identity_transform": bool(policy.uses_identity_transform),
            "calibration_state": policy.calibration_state,
            "tip_pose_status": policy.tip_pose_status,
            "mode_message": str(getattr(snapshot, "runtime_tip_mode_message", "") or policy.status_message),
            "selected_artifact_kind": getattr(snapshot, "runtime_tip_selected_artifact_kind", None),
            "selected_artifact_path": runtime_tip_path or None,
            "selected_artifact_exists": bool(Path(runtime_tip_path).exists()) if runtime_tip_path else False,
            "stored_timestamp_utc": getattr(snapshot, "stored_runtime_tip_timestamp_utc", None),
            "measurement_tool_id": getattr(snapshot, "stored_runtime_tip_measurement_tool_id", None),
            "coil_tool_id": getattr(snapshot, "stored_runtime_tip_coil_tool_id", None),
            "identity_fallback": bool(getattr(snapshot, "runtime_tip_identity_fallback", False)),
            "policy": policy.to_dict(),
        },
        "transform_chain": {
            "expression": TRANSFORM_CHAIN_EXPRESSION,
            "frame_conventions": {
                "T_A_B": "maps coordinates from frame B into frame A",
                "robot": "registered robot/body frame",
                "aurora": "NDI Aurora tracker frame",
                "coil_0A": "runtime 0A coil frame",
                "tip": "declared runtime tip frame; coil origin when runtime_tip.mode is coil_as_tip",
            },
            "active_transforms": {
                "T_robot_aurora": getattr(snapshot, "T_robot_aurora", None) is not None,
                "T_aurora_coil_0A": "0A" in (getattr(snapshot, "tools", {}) or {}),
                "T_coil_tip": bool(policy.uses_calibrated_tip_transform or policy.uses_identity_transform),
                "T_robot_tip": getattr(snapshot, "T_robot_tip", None) is not None,
            },
        },
        "warnings": warnings,
        "reasons": list(policy.reasons),
    }


def write_transform_chain_outputs(
    *,
    output_dir: Path,
    snapshot,
    workflow: str = WORKFLOW_GENERIC,
    allow_lower_trust: bool = False,
    provenance_note: str = "",
) -> dict[str, Path]:
    """Write the transform-chain summary as JSON. The .txt and overview .png
    duplicate this JSON and have no consumers; they are no longer emitted."""

    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    summary = build_transform_chain_summary(
        snapshot=snapshot,
        workflow=workflow,
        allow_lower_trust=allow_lower_trust,
        provenance_note=provenance_note,
    )
    json_path = output_root / "transform_chain_summary.json"
    json_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    return {"json": json_path}



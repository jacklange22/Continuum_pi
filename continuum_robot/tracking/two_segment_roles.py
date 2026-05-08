"""Two-segment tracking role resolution.

This module maps configured Aurora tools onto semantic two-segment pose-label
roles. It records what is present/missing/stale without changing transform
math or fabricating unavailable robot-frame labels.
"""

from __future__ import annotations

from dataclasses import asdict, is_dataclass
from typing import Any

from continuum_robot.two_segment import TwoSegmentPoseObservation


SUPPORTED_TWO_SEGMENT_TRACKING_ROLES = (
    "registration_probe",
    "distal_tip",
    "intermediate_segment",
    "debug_tool",
)


def role_config_records(role_configs: dict[str, Any] | None) -> dict[str, dict[str, Any]]:
    """Return enabled role config records keyed by role name."""
    records: dict[str, dict[str, Any]] = {}
    for key, raw_config in dict(role_configs or {}).items():
        if is_dataclass(raw_config):
            data = asdict(raw_config)
        elif hasattr(raw_config, "to_dict"):
            data = dict(raw_config.to_dict())
        else:
            data = dict(raw_config or {})
        role_name = str(data.get("role_name") or key)
        data["role_name"] = role_name
        data["tool_id"] = str(data.get("tool_id", "") or "").strip().upper()
        data["enabled"] = bool(data.get("enabled", True))
        if data["enabled"]:
            records[role_name] = data
    return records


def resolve_two_segment_tracking_roles(
    *,
    snapshot,
    role_configs: dict[str, Any] | None,
    require_model_training_roles: bool = False,
) -> dict[str, Any]:
    """Resolve live tracking snapshot tools into two-segment semantic roles."""
    configs = role_config_records(role_configs)
    tools = dict(getattr(snapshot, "tools", {}) or {}) if snapshot is not None else {}
    snapshot_age_s = getattr(snapshot, "tracker_data_age_s", None) if snapshot is not None else None
    snapshot_stale = bool(getattr(snapshot, "tracker_data_stale", False)) if snapshot is not None else True
    role_observations: dict[str, dict[str, Any]] = {}
    available_roles: list[str] = []
    missing_roles: list[str] = []
    stale_roles: list[str] = []
    invalid_roles: list[str] = []
    missing_required_roles: list[str] = []
    pose_trust_by_role: dict[str, str] = {}
    tracker_freshness_by_role: dict[str, Any] = {}
    tool_roles: dict[str, str] = {}
    raw_tool_poses: dict[str, dict[str, Any]] = {}
    robot_frame_role_poses: dict[str, dict[str, Any]] = {}

    for role_name, config in configs.items():
        tool_id = str(config.get("tool_id") or "").strip().upper()
        required_for_training = bool(config.get("required_for_two_segment_model_training", False))
        tool_roles[tool_id] = role_name if tool_id else role_name
        base_record = {
            "role_name": role_name,
            "tool_id": tool_id,
            "physical_meaning": str(config.get("physical_meaning", "") or ""),
            "pose_kind": str(config.get("pose_kind", "coil_origin") or "coil_origin"),
            "required_for_two_segment_model_training": required_for_training,
            "required_for_control": bool(config.get("required_for_control", False)),
            "trust_level": str(config.get("trust_level", "configured") or "configured"),
            "source": str(config.get("source", "config") or "config"),
            "max_age_s": config.get("max_age_s"),
        }
        if not tool_id:
            record = {**base_record, "status": "unconfigured", "available": False, "stale": False, "valid": False}
            role_observations[role_name] = record
            if required_for_training:
                missing_required_roles.append(role_name)
            missing_roles.append(role_name)
            pose_trust_by_role[role_name] = "unconfigured"
            continue
        tool = tools.get(tool_id)
        if tool is None:
            record = {**base_record, "status": "missing_tool", "available": False, "stale": False, "valid": False}
            role_observations[role_name] = record
            missing_roles.append(role_name)
            if required_for_training:
                missing_required_roles.append(role_name)
            pose_trust_by_role[role_name] = "missing"
            continue

        pose = _tool_pose_record(tool)
        max_age = config.get("max_age_s")
        stale = bool(snapshot_stale)
        if max_age not in (None, "") and snapshot_age_s is not None:
            stale = stale or float(snapshot_age_s) > float(max_age)
        valid = bool(getattr(tool, "valid", True) is not False and pose.get("translation_mm"))
        present = bool(getattr(tool, "present", bool(pose.get("translation_mm"))))
        available = bool(present and valid and not stale)
        status = "available" if available else ("stale" if stale and present and valid else ("invalid" if present else "missing_tool"))
        record = {
            **base_record,
            "status": status,
            "available": available,
            "stale": bool(stale),
            "valid": bool(valid),
            "present": bool(present),
            "pose": pose,
            "raw_tool_id": tool_id,
            "tracking_state": getattr(tool, "tracking_state", None),
            "tracker_data_age_s": snapshot_age_s,
        }
        robot_pose = _existing_robot_frame_pose_for_role(snapshot=snapshot, role_name=role_name, tool_id=tool_id)
        if robot_pose:
            record["robot_frame_pose"] = robot_pose
            robot_frame_role_poses[role_name] = robot_pose
        role_observations[role_name] = record
        raw_tool_poses[tool_id] = {**pose, "role_name": role_name, "pose_kind": record["pose_kind"]}
        tracker_freshness_by_role[role_name] = snapshot_age_s
        pose_trust_by_role[role_name] = status
        if available:
            available_roles.append(role_name)
        else:
            if stale:
                stale_roles.append(role_name)
            elif present:
                invalid_roles.append(role_name)
            else:
                missing_roles.append(role_name)
            if required_for_training:
                missing_required_roles.append(role_name)

    distal_available = "distal_tip" in available_roles
    intermediate_available = "intermediate_segment" in available_roles
    dataset_validity = {
        "required_model_training_roles_present": not bool(missing_required_roles),
        "missing_required_roles": sorted(set(missing_required_roles)),
        "distal_only": bool(distal_available and not intermediate_available),
        "includes_intermediate_pose": bool(intermediate_available),
        "valid_for_two_segment_model_training_pose_labels": bool(
            require_model_training_roles and not missing_required_roles
        ),
    }
    observation = TwoSegmentPoseObservation(
        frame="robot" if robot_frame_role_poses else "tracker",
        distal_tip_pose=dict(role_observations.get("distal_tip", {}).get("robot_frame_pose") or role_observations.get("distal_tip", {}).get("pose") or {}),
        intermediate_pose=dict(role_observations.get("intermediate_segment", {}).get("robot_frame_pose") or role_observations.get("intermediate_segment", {}).get("pose") or {}),
        segment_poses={
            role_name: dict(record.get("robot_frame_pose") or record.get("pose") or {})
            for role_name, record in role_observations.items()
            if record.get("available")
        },
        tool_poses=raw_tool_poses,
        tool_roles={tool_id: role for tool_id, role in tool_roles.items() if tool_id},
        status="pose_roles_available" if available_roles else "pose_roles_missing",
    ).to_dict()
    return {
        "schema_version": "two_segment_tracking_roles_v1",
        "role_config": configs,
        "role_observations": role_observations,
        "two_segment_pose": observation,
        "available_roles": sorted(set(available_roles)),
        "missing_roles": sorted(set(missing_roles)),
        "stale_roles": sorted(set(stale_roles)),
        "invalid_roles": sorted(set(invalid_roles)),
        "missing_required_roles": sorted(set(missing_required_roles)),
        "pose_trust_by_role": pose_trust_by_role,
        "tracker_freshness_by_role": tracker_freshness_by_role,
        "tool_ids_seen": sorted(str(key) for key in tools),
        "pose_in_tracker_frame": {"tools": raw_tool_poses, "roles": role_observations},
        "pose_in_robot_frame": {"roles": robot_frame_role_poses},
        "transform_validity": {
            role_name: ("available" if role_name in available_roles else "missing")
            for role_name in configs
        },
        "dataset_validity": dataset_validity,
    }


def two_segment_role_readiness_rows(*, snapshot, role_configs: dict[str, Any] | None) -> list[dict[str, Any]]:
    """Return UI/preflight-friendly role readiness rows."""
    resolved = resolve_two_segment_tracking_roles(
        snapshot=snapshot,
        role_configs=role_configs,
        require_model_training_roles=False,
    )
    return [
        {
            "role": role_name,
            "tool_id": record.get("tool_id", ""),
            "status": record.get("status", "unknown"),
            "physical_meaning": record.get("physical_meaning", ""),
            "pose_kind": record.get("pose_kind", ""),
            "required_for_two_segment_model_training": bool(
                record.get("required_for_two_segment_model_training", False)
            ),
            "required_for_control": bool(record.get("required_for_control", False)),
        }
        for role_name, record in sorted(dict(resolved.get("role_observations", {}) or {}).items())
    ]


def _tool_pose_record(tool) -> dict[str, Any]:
    translation = getattr(tool, "translation_mm", None) or getattr(tool, "last_good_translation_mm", None)
    quaternion = getattr(tool, "quaternion_wxyz", None) or getattr(tool, "last_good_quaternion_wxyz", None)
    pose = {
        "translation_mm": [float(value) for value in translation] if translation is not None else [],
        "quaternion_wxyz": [float(value) for value in quaternion] if quaternion is not None else [],
        "quality": getattr(tool, "quality", None),
        "T_aurora_tool": getattr(tool, "T_aurora_tool", None) or getattr(tool, "last_good_T_aurora_tool", None),
    }
    return pose


def _existing_robot_frame_pose_for_role(*, snapshot, role_name: str, tool_id: str) -> dict[str, Any]:
    if snapshot is None:
        return {}
    if role_name != "distal_tip":
        return {}
    runtime_tool_id = str(getattr(snapshot, "runtime_coil_tool_id", "") or "").strip().upper()
    if tool_id != runtime_tool_id:
        return {}
    T_robot_tip = getattr(snapshot, "T_robot_tip", None)
    if T_robot_tip is None:
        return {}
    return {
        "T_robot_tip": T_robot_tip,
        "pose_kind": "runtime_tip_existing_transform",
        "tip_pose_status": getattr(snapshot, "tip_pose_status", ""),
        "runtime_tip_mode": getattr(snapshot, "runtime_tip_mode", ""),
        "runtime_tip_trust_level": getattr(snapshot, "runtime_tip_trust_level", ""),
    }

"""Shared collect-pose model-training validity checks."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any


TRUSTED_RUN_MODES = {"thesis_trusted", "trusted"}
NON_TRAINING_PHASES = {"initial_neutral", "final_neutral"}


def _as_dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _as_bool(value: Any) -> bool:
    return bool(value)


def _as_int(value: Any, default: int = 0) -> int:
    try:
        if value is None:
            return int(default)
        return int(value)
    except Exception:
        return int(default)


def _tip_xyz_from_sample(sample: Any) -> list[Any]:
    pose_robot = _as_dict(getattr(sample, "pose_in_robot_frame", {}))
    tip = _as_dict(pose_robot.get("tip"))
    xyz = tip.get("translation_mm")
    return list(xyz) if isinstance(xyz, list) else []


def _extra_from_sample(sample: Any) -> dict[str, Any]:
    return _as_dict(getattr(sample, "extra", {}))


def _sample_is_accepted(sample: Any) -> bool:
    return _as_bool(_extra_from_sample(sample).get("capture_accepted"))


def _sample_is_export_excluded(sample: Any) -> bool:
    return _as_bool(_extra_from_sample(sample).get("modeling_export_exclude"))


def sample_has_complete_command_servo_tip(sample: Any) -> bool:
    extra = _extra_from_sample(sample)
    command = extra.get("resolved_cable_command_cm")
    if not (isinstance(command, list) and len(command) == 4):
        command = getattr(sample, "commanded_cable_deltas_cm", None)
    command_ok = isinstance(command, list) and len(command) == 4
    tip_ok = len(_tip_xyz_from_sample(sample)) == 3
    servo_feedback = _as_dict(extra.get("servo_feedback_at_capture"))
    final_goal = _as_dict(extra.get("final_goal_ticks_by_servo"))
    if not servo_feedback or not final_goal:
        return False
    for servo_id in final_goal:
        entry = servo_feedback.get(str(servo_id), servo_feedback.get(servo_id))
        if not isinstance(entry, dict):
            return False
        if entry.get("present_position_ticks") is None:
            return False
    return command_ok and tip_ok


def collect_pose_training_row_stats_from_samples(samples: Sequence[Any]) -> dict[str, Any]:
    export_rows = 0
    excluded_rows = 0
    accepted_training_rows = 0
    accepted_complete_rows = 0
    accepted_tip_rows = 0
    accepted_invalid_transform_rows = 0
    accepted_sample_count = 0
    accepted_workspace_sample_count = 0
    non_training_accepted_row_count = 0
    incomplete_accepted_workspace_row_count = 0
    dropped_quarantined_sample_count = 0
    for sample in samples:
        phase = str(getattr(sample, "phase", "") or "")
        accepted = _sample_is_accepted(sample)
        if accepted:
            accepted_sample_count += 1
            if phase in NON_TRAINING_PHASES:
                non_training_accepted_row_count += 1
            else:
                accepted_workspace_sample_count += 1
        excluded = _sample_is_export_excluded(sample)
        if excluded:
            dropped_quarantined_sample_count += 1
        if excluded:
            excluded_rows += 1
            continue
        # Policy: export rows are complete accepted workspace rows only.
        if phase in NON_TRAINING_PHASES:
            continue
        if not accepted:
            continue
        if not sample_has_complete_command_servo_tip(sample):
            incomplete_accepted_workspace_row_count += 1
            continue
        export_rows += 1
        accepted_training_rows += 1
        accepted_complete_rows += 1
        if len(_tip_xyz_from_sample(sample)) == 3:
            accepted_tip_rows += 1
        transform_validity = _as_dict(getattr(sample, "transform_validity", {}))
        if any(str(state).lower() == "invalid" for state in transform_validity.values()):
            accepted_invalid_transform_rows += 1
    return {
        "accepted_sample_count": int(accepted_sample_count),
        "accepted_workspace_sample_count": int(accepted_workspace_sample_count),
        "non_training_accepted_row_count": int(non_training_accepted_row_count),
        "incomplete_accepted_workspace_row_count": int(incomplete_accepted_workspace_row_count),
        "dropped_quarantined_sample_count": int(dropped_quarantined_sample_count),
        "modeling_export_row_count": int(export_rows),
        "excluded_sample_row_count": int(excluded_rows),
        "accepted_training_row_count": int(accepted_training_rows),
        "complete_training_row_count": int(accepted_complete_rows),
        "accepted_complete_training_row_count": int(accepted_complete_rows),
        "accepted_tip_row_count": int(accepted_tip_rows),
        "accepted_invalid_transform_row_count": int(accepted_invalid_transform_rows),
    }


def evaluate_collect_pose_model_training_validity(
    *,
    success: bool,
    experiment_name: str,
    metrics: Mapping[str, Any],
    row_stats: Mapping[str, Any],
    legacy_row_count: int | None,
) -> dict[str, Any]:
    metrics_map = dict(metrics or {})
    row_map = dict(row_stats or {})
    run_provenance = _as_dict(metrics_map.get("run_provenance"))
    runtime_tip_policy = _as_dict(metrics_map.get("runtime_tip_policy"))
    if not runtime_tip_policy:
        runtime_tip_policy = _as_dict(_as_dict(run_provenance.get("runtime_tip_calibration")).get("policy"))
    run_mode = str(metrics_map.get("run_trust_mode", "") or "").strip().lower()
    dry_run = _as_bool(metrics_map.get("dry_run"))
    mock_mode = _as_bool(metrics_map.get("mock_mode"))
    tracker_connected = _as_bool(metrics_map.get("tracker_connected"))
    parallel_single_demo = _as_bool(metrics_map.get("parallel_single_demo"))
    target_valid = metrics_map.get("target_valid_sample_count")
    complete_training_rows = _as_int(
        row_map.get("complete_training_row_count", row_map.get("accepted_complete_training_row_count", 0))
    )
    if target_valid is None:
        accepted_count_meets_target = complete_training_rows > 0
    else:
        accepted_count_meets_target = complete_training_rows >= _as_int(target_valid, 0)
    dropped_post = _as_int(metrics_map.get("dropped_post_motion_telemetry_samples"))
    dropped_pre = _as_int(metrics_map.get("dropped_pre_motion_telemetry_samples"))
    dropped_total = dropped_post + dropped_pre
    excluded_rows = _as_int(row_map.get("excluded_sample_row_count"))
    dropped_rows_excluded = (dropped_total <= 0) or (excluded_rows >= dropped_total)
    accepted_rows_complete = _as_int(row_map.get("accepted_training_row_count")) == _as_int(
        row_map.get("complete_training_row_count")
    ) and _as_int(row_map.get("accepted_training_row_count")) > 0
    pose_labels_available = _as_int(row_map.get("accepted_training_row_count")) == _as_int(
        row_map.get("accepted_tip_row_count")
    ) and _as_int(row_map.get("accepted_training_row_count")) > 0
    no_invalid_transforms_in_accepted_rows = _as_int(row_map.get("accepted_invalid_transform_row_count")) == 0
    pretension_artifact = _as_dict(run_provenance.get("pretension_artifact"))
    startup_provenance_present = _as_bool(pretension_artifact.get("accepted")) and _as_bool(
        pretension_artifact.get("usable")
    )
    registration_or_coil_tip_policy_present = _as_bool(runtime_tip_policy.get("allowed_for_workflow")) or (
        str(metrics_map.get("runtime_tip_mode_used", "") or "").strip().lower() == "coil_as_tip"
        and str(metrics_map.get("runtime_tip_trust_level", "") or "").strip().lower() in {"thesis_trusted", "trusted"}
    )
    checks = {
        "run_success": _as_bool(success),
        "real_hardware": bool((not dry_run) and (not mock_mode)),
        "trusted_run_mode": run_mode in TRUSTED_RUN_MODES,
        "tracker_connected": tracker_connected,
        "pose_labels_available": pose_labels_available,
        "accepted_rows_complete": accepted_rows_complete,
        "accepted_count_meets_target": accepted_count_meets_target,
        "dropped_rows_excluded": dropped_rows_excluded,
        "no_invalid_transforms_in_accepted_rows": no_invalid_transforms_in_accepted_rows,
        "startup_provenance_present": startup_provenance_present,
        "registration_or_coil_tip_policy_present": registration_or_coil_tip_policy_present,
        "not_parallel_demo": not parallel_single_demo,
        "not_servo_only": run_mode != "servo_only",
        "not_mock": not mock_mode,
    }
    hard_fail_reasons: list[str] = []
    if str(experiment_name or "").strip().lower() != "collect_pose_command_dataset":
        hard_fail_reasons.append("not_collect_pose_command_dataset")
    for check_name, passed in checks.items():
        if not bool(passed):
            hard_fail_reasons.append(check_name)
    warnings: list[str] = []
    if dropped_total > 0:
        warnings.append(
            f"{dropped_total} dropped telemetry samples excluded from training export."
        )
    unrecovered = _as_int(metrics_map.get("unrecovered_packet_error_count"))
    if unrecovered > 0:
        warnings.append(f"{unrecovered} unrecovered packet-error events were recorded and quarantined.")
    rejected = _as_int(metrics_map.get("rejected_sample_count"))
    if rejected > 0:
        warnings.append(f"{rejected} rejected samples were excluded from training rows.")
    incomplete_workspace = _as_int(row_map.get("incomplete_accepted_workspace_row_count"))
    if incomplete_workspace > 0:
        warnings.append(f"{incomplete_workspace} accepted workspace rows were excluded due to incomplete training fields.")
    non_training_accepted = _as_int(row_map.get("non_training_accepted_row_count"))
    if non_training_accepted > 0:
        warnings.append(f"{non_training_accepted} accepted non-training phase rows were excluded from modeling export.")
    if hard_fail_reasons:
        status = "invalid"
        reason = hard_fail_reasons[0]
        valid_for_model_training = False
    elif warnings:
        status = "warning_valid"
        reason = "warnings_present_but_training_ready"
        valid_for_model_training = True
    else:
        status = "valid"
        reason = "all_required_checks_passed"
        valid_for_model_training = True
    if str(experiment_name or "").strip().lower() != "collect_pose_command_dataset":
        status = "not_applicable"
        reason = "not_collect_pose_command_dataset"
        valid_for_model_training = False
    if legacy_row_count is None:
        legacy_rows_value = None
    else:
        legacy_rows_value = int(legacy_row_count)
    return {
        "model_training_validity_status": status,
        "model_training_validity_reason": reason,
        "model_training_validity_checks": checks,
        "model_training_warnings": warnings,
        "model_training_hard_invalidation_reasons": hard_fail_reasons,
        "valid_for_model_training": bool(valid_for_model_training),
        "not_model_training_ready": not bool(valid_for_model_training),
        "dropped_samples_excluded_from_training": bool(dropped_rows_excluded),
        "accepted_rows_complete": bool(accepted_rows_complete),
        "modeling_export_row_count": _as_int(row_map.get("modeling_export_row_count")),
        "accepted_training_row_count": _as_int(row_map.get("accepted_training_row_count")),
        "complete_training_row_count": _as_int(row_map.get("complete_training_row_count")),
        "accepted_workspace_sample_count": _as_int(row_map.get("accepted_workspace_sample_count")),
        "non_training_accepted_row_count": _as_int(row_map.get("non_training_accepted_row_count")),
        "incomplete_accepted_workspace_row_count": _as_int(row_map.get("incomplete_accepted_workspace_row_count")),
        "dropped_quarantined_sample_count": _as_int(row_map.get("dropped_quarantined_sample_count")),
        "modeling_legacy_row_count": legacy_rows_value,
    }

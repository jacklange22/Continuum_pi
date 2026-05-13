from __future__ import annotations

from types import SimpleNamespace

import pytest

from continuum_robot.tracking.runtime_tip_policy import (
    TRUST_DEBUG,
    TRUST_LOWER,
    TRUST_THESIS,
    WORKFLOW_MODELING_DATASET,
    WORKFLOW_PENPROBE_CHASING,
    WORKFLOW_REPEATABILITY,
    evaluate_runtime_tip_trust,
)


def _snapshot(*, mode: str, state: str, tip_status: str = "ok", has_tip: bool = True, fallback: bool = False):
    return SimpleNamespace(
        runtime_tip_mode=mode,
        runtime_tip_calibration_state=state,
        tip_pose_status=tip_status,
        T_robot_tip=[[1.0, 0.0, 0.0, 0.0]] * 4 if has_tip else None,
        runtime_tip_identity_fallback=fallback,
    )


def test_coil_as_tip_is_current_thesis_trusted_repeatability_path() -> None:
    evaluation = evaluate_runtime_tip_trust(
        snapshot=_snapshot(mode="coil_as_tip", state="coil_as_tip", tip_status="coil_as_tip", fallback=True),
        workflow=WORKFLOW_REPEATABILITY,
    )

    assert evaluation.trust_label == TRUST_THESIS
    assert evaluation.thesis_trusted is True
    assert evaluation.allowed_for_workflow is True
    assert evaluation.uses_coil_as_tip is True
    assert evaluation.uses_identity_transform is True
    assert "0A coil origin" in evaluation.warnings[0]


def test_latest_accepted_runtime_tip_is_lower_trust_until_validated() -> None:
    evaluation = evaluate_runtime_tip_trust(
        snapshot=_snapshot(mode="latest_accepted", state="loaded"),
        workflow=WORKFLOW_REPEATABILITY,
    )

    assert evaluation.trust_label == TRUST_LOWER
    assert evaluation.thesis_trusted is False
    assert evaluation.allowed_for_workflow is False
    assert evaluation.uses_calibrated_tip_transform is True


def test_repeatability_rejects_lower_trust_even_with_override_flag() -> None:
    evaluation = evaluate_runtime_tip_trust(
        snapshot=_snapshot(mode="latest_accepted", state="loaded"),
        workflow=WORKFLOW_REPEATABILITY,
        allow_lower_trust=True,
    )

    assert evaluation.trust_label == TRUST_LOWER
    assert evaluation.allowed_for_workflow is False


def test_modeling_dataset_requires_explicit_override_for_lower_trust_runtime_tip() -> None:
    snapshot = _snapshot(mode="latest_accepted", state="loaded")

    blocked = evaluate_runtime_tip_trust(snapshot=snapshot, workflow=WORKFLOW_MODELING_DATASET)
    allowed = evaluate_runtime_tip_trust(
        snapshot=snapshot,
        workflow=WORKFLOW_MODELING_DATASET,
        allow_lower_trust=True,
    )

    assert blocked.trust_label == TRUST_LOWER
    assert blocked.requires_lower_trust_override is True
    assert blocked.allowed_for_workflow is False
    assert allowed.allowed_for_workflow is True


def test_quick_4_point_is_debug_only() -> None:
    evaluation = evaluate_runtime_tip_trust(
        snapshot=_snapshot(mode="quick_4_point", state="quick_4_point_loaded"),
        workflow=WORKFLOW_MODELING_DATASET,
        allow_lower_trust=True,
    )

    assert evaluation.trust_label == TRUST_DEBUG
    assert evaluation.thesis_trusted is False
    assert evaluation.allowed_for_workflow is True


def test_missing_tip_pose_is_unavailable() -> None:
    evaluation = evaluate_runtime_tip_trust(
        snapshot=_snapshot(mode="coil_as_tip", state="coil_as_tip", tip_status="missing", has_tip=False),
        workflow=WORKFLOW_REPEATABILITY,
    )

    assert evaluation.trust_label == "unavailable"
    assert evaluation.allowed_for_workflow is False
    assert "missing_T_robot_tip" in evaluation.reasons


def test_single_segment_repeatability_platform_alias_resolves_to_repeatability() -> None:
    evaluation = evaluate_runtime_tip_trust(
        snapshot=_snapshot(mode="coil_as_tip", state="coil_as_tip", tip_status="coil_as_tip", fallback=True),
        workflow="single_segment_repeatability_platform",
    )

    assert evaluation.requested_workflow == "single_segment_repeatability_platform"
    assert evaluation.workflow == WORKFLOW_REPEATABILITY


def test_penprobe_chasing_demo_alias_resolves_to_penprobe_chasing() -> None:
    evaluation = evaluate_runtime_tip_trust(
        snapshot=_snapshot(mode="coil_as_tip", state="coil_as_tip"),
        workflow="penprobe_chasing_demo",
    )

    assert evaluation.requested_workflow == "penprobe_chasing_demo"
    assert evaluation.workflow == WORKFLOW_PENPROBE_CHASING
    assert evaluation.allowed_for_workflow is True


def test_unknown_workflow_raises_clear_error_with_supported_list() -> None:
    with pytest.raises(ValueError) as exc_info:
        evaluate_runtime_tip_trust(
            snapshot=_snapshot(mode="coil_as_tip", state="coil_as_tip", tip_status="coil_as_tip", fallback=True),
            workflow="unknown_repeatability_platform_v2",
        )
    message = str(exc_info.value)
    assert "Unknown runtime tip workflow/platform 'unknown_repeatability_platform_v2'" in message
    assert "Supported canonical workflows:" in message
    assert "repeatability" in message

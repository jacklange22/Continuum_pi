"""Phase 5: thesis-eligibility helper.

Tests the pure decision function that decides whether a tracker-timing
or servo-tracker-sync run should be marked as thesis evidence. The
helper is intentionally strict (any of: mock_mode, dry_run, non-live
tracker, servo missing-when-required, force_status != success
disqualifies the run).
"""

from __future__ import annotations

import pytest

from continuum_robot.experiments.thesis_eligibility import (
    LABEL_DEBUG_OR_SYNTHETIC,
    LABEL_THESIS,
    compute_thesis_eligibility,
)


# --------------------------------------------------------------------------- #
# Happy path                                                                   #
# --------------------------------------------------------------------------- #


def test_live_aurora_real_servo_success_is_thesis_eligible() -> None:
    verdict = compute_thesis_eligibility(
        mock_mode=False,
        tracker_backend_identity="ndi_tracker_python",
        selected_backend_name="python",
        configured_backend_name="python",
        servo_required=True,
        servo_connected=True,
        servo_mock=False,
        dry_run=False,
        force_status="success",
    )
    assert verdict["eligible"] is True
    assert verdict["label"] == LABEL_THESIS
    assert verdict["reasons"] == []
    assert verdict["runtime_environment"]["tracker_backend_identity"] == "ndi_tracker_python"


def test_live_aurora_without_servo_requirement_is_eligible() -> None:
    """tracker_timing_validation case: servo not required."""
    verdict = compute_thesis_eligibility(
        mock_mode=False,
        tracker_backend_identity="ndi_tracker_python",
        selected_backend_name="python",
        servo_required=False,
        servo_connected=False,  # not required, so absence doesn't disqualify
        servo_mock=False,
        dry_run=False,
        force_status="success",
    )
    assert verdict["eligible"] is True
    assert verdict["runtime_environment"]["servo_required"] is False


def test_live_aurora_alias_aurora_identity_accepted() -> None:
    """A real Aurora identity that doesn't match the explicit allow-list
    but contains 'aurora' should still pass."""
    verdict = compute_thesis_eligibility(
        mock_mode=False,
        tracker_backend_identity="aurora_v3_realtime",
        selected_backend_name="python",
        servo_required=False,
    )
    assert verdict["eligible"] is True


# --------------------------------------------------------------------------- #
# Disqualifying conditions                                                     #
# --------------------------------------------------------------------------- #


def test_mock_mode_disqualifies() -> None:
    verdict = compute_thesis_eligibility(
        mock_mode=True,
        tracker_backend_identity="ndi_tracker_python",
        selected_backend_name="python",
        servo_required=False,
        force_status="success",
    )
    assert verdict["eligible"] is False
    assert verdict["label"] == LABEL_DEBUG_OR_SYNTHETIC
    assert any("mock_mode" in reason for reason in verdict["reasons"])


def test_dry_run_disqualifies() -> None:
    verdict = compute_thesis_eligibility(
        mock_mode=False,
        tracker_backend_identity="ndi_tracker_python",
        selected_backend_name="python",
        servo_required=False,
        dry_run=True,
    )
    assert verdict["eligible"] is False
    assert any("dry_run" in reason for reason in verdict["reasons"])


def test_bridge_backend_disqualifies_even_with_aurora_name() -> None:
    """ndi_bridge / tracker_bridge_json must be rejected — they're the
    legacy synthetic-friendly path."""
    verdict = compute_thesis_eligibility(
        mock_mode=False,
        tracker_backend_identity="tracker_bridge_json",
        selected_backend_name="bridge",
        servo_required=False,
    )
    assert verdict["eligible"] is False
    assert any("tracker backend" in reason for reason in verdict["reasons"])


def test_ndi_bridge_identity_disqualifies() -> None:
    """An identity string containing 'bridge' must be rejected even if
    it also contains 'ndi'."""
    verdict = compute_thesis_eligibility(
        mock_mode=False,
        tracker_backend_identity="ndi_bridge_legacy",
        selected_backend_name="bridge",
        servo_required=False,
    )
    assert verdict["eligible"] is False


def test_empty_tracker_identity_disqualifies() -> None:
    """No tracker identity → no idea what backend ran → not thesis-grade."""
    verdict = compute_thesis_eligibility(
        mock_mode=False,
        tracker_backend_identity=None,
        selected_backend_name=None,
        servo_required=False,
    )
    assert verdict["eligible"] is False


def test_servo_required_but_disconnected_disqualifies() -> None:
    verdict = compute_thesis_eligibility(
        mock_mode=False,
        tracker_backend_identity="ndi_tracker_python",
        selected_backend_name="python",
        servo_required=True,
        servo_connected=False,
    )
    assert verdict["eligible"] is False
    assert any("not connected" in reason for reason in verdict["reasons"])


def test_servo_required_but_mock_disqualifies() -> None:
    verdict = compute_thesis_eligibility(
        mock_mode=False,
        tracker_backend_identity="ndi_tracker_python",
        selected_backend_name="python",
        servo_required=True,
        servo_connected=True,
        servo_mock=True,
    )
    assert verdict["eligible"] is False
    assert any("mock mode" in reason for reason in verdict["reasons"])


def test_non_success_force_status_disqualifies() -> None:
    """A timed-out or insufficient-samples run isn't thesis evidence."""
    verdict = compute_thesis_eligibility(
        mock_mode=False,
        tracker_backend_identity="ndi_tracker_python",
        selected_backend_name="python",
        servo_required=False,
        force_status="partial_success",
    )
    assert verdict["eligible"] is False
    assert any("force_status" in reason for reason in verdict["reasons"])


def test_insufficient_samples_force_status_disqualifies() -> None:
    verdict = compute_thesis_eligibility(
        mock_mode=False,
        tracker_backend_identity="ndi_tracker_python",
        selected_backend_name="python",
        servo_required=False,
        force_status="invalid_due_to_insufficient_samples",
    )
    assert verdict["eligible"] is False


def test_extra_disqualifying_reasons_are_preserved() -> None:
    """Caller-supplied reasons must be merged into the reasons list."""
    verdict = compute_thesis_eligibility(
        mock_mode=False,
        tracker_backend_identity="ndi_tracker_python",
        selected_backend_name="python",
        servo_required=False,
        extra_disqualifying_reasons=["operator marked this run as scratch"],
    )
    assert verdict["eligible"] is False
    assert "operator marked this run as scratch" in verdict["reasons"]


# --------------------------------------------------------------------------- #
# Runtime environment fingerprint                                              #
# --------------------------------------------------------------------------- #


def test_runtime_environment_captures_all_inputs() -> None:
    verdict = compute_thesis_eligibility(
        mock_mode=True,
        tracker_backend_identity="ndi_tracker_python",
        selected_backend_name="python",
        configured_backend_name="auto",
        servo_required=True,
        servo_connected=True,
        servo_mock=False,
        dry_run=True,
        force_status="partial_success",
    )
    env = verdict["runtime_environment"]
    assert env["mock_mode"] is True
    assert env["dry_run"] is True
    assert env["tracker_backend_identity"] == "ndi_tracker_python"
    assert env["tracker_backend_selected"] == "python"
    assert env["tracker_backend_configured"] == "auto"
    assert env["servo_required"] is True
    assert env["servo_connected"] is True
    assert env["servo_mock"] is False
    assert env["force_status"] == "partial_success"


def test_multiple_reasons_all_reported() -> None:
    """Operator running everything wrong should see ALL the reasons,
    not just the first one."""
    verdict = compute_thesis_eligibility(
        mock_mode=True,
        tracker_backend_identity="tracker_bridge_json",
        selected_backend_name="bridge",
        servo_required=True,
        servo_connected=False,
        dry_run=True,
        force_status="invalid_due_to_insufficient_samples",
    )
    assert verdict["eligible"] is False
    # Expect at least 5 reasons: mock_mode, dry_run, non-live tracker,
    # servo not connected, force_status non-success.
    assert len(verdict["reasons"]) >= 5


# --------------------------------------------------------------------------- #
# Stability / determinism                                                      #
# --------------------------------------------------------------------------- #


def test_pure_function_same_inputs_same_outputs() -> None:
    inputs = {
        "mock_mode": False,
        "tracker_backend_identity": "ndi_tracker_python",
        "selected_backend_name": "python",
        "servo_required": True,
        "servo_connected": True,
        "force_status": "success",
    }
    first = compute_thesis_eligibility(**inputs)
    second = compute_thesis_eligibility(**inputs)
    assert first == second


# --------------------------------------------------------------------------- #
# Integration anchors: both timing experiments must stamp the verdict.        #
# We use inspect.getsource on the execute() methods (same trick as            #
# test_servo_tracker_sync_safety_diagnostics) so we don't have to spin up a   #
# full ExperimentSession just to check the wiring is present.                 #
# --------------------------------------------------------------------------- #


def test_tracker_timing_execute_stamps_thesis_eligibility_into_metrics() -> None:
    import inspect

    from continuum_robot.experiments.builtins import TrackerTimingValidationExperiment

    src = inspect.getsource(TrackerTimingValidationExperiment.execute)
    assert "compute_thesis_eligibility" in src, (
        "tracker_timing_validation.execute() must call compute_thesis_eligibility "
        "so the headline figures and downstream consumers can see the verdict."
    )
    assert "\"thesis_eligibility\"" in src or "'thesis_eligibility'" in src, (
        "tracker_timing_validation.execute() must stamp the result under the "
        "'thesis_eligibility' key in metrics."
    )
    # The tracker_timing experiment does NOT require a connected servo.
    assert "servo_required=False" in src, (
        "tracker_timing_validation must call compute_thesis_eligibility with "
        "servo_required=False — its headline metric is the tracker stream and "
        "servo absence is allowed."
    )


def test_servo_tracker_sync_execute_stamps_thesis_eligibility_into_metrics() -> None:
    import inspect

    from continuum_robot.experiments.builtins import ServoTrackerSyncValidationExperiment

    src = inspect.getsource(ServoTrackerSyncValidationExperiment.execute)
    assert "compute_thesis_eligibility" in src, (
        "servo_tracker_sync_validation.execute() must call compute_thesis_eligibility "
        "so the headline figures and downstream consumers can see the verdict."
    )
    assert "\"thesis_eligibility\"" in src or "'thesis_eligibility'" in src, (
        "servo_tracker_sync_validation.execute() must stamp the result under the "
        "'thesis_eligibility' key in metrics."
    )
    # The sync experiment DOES require a connected servo.
    assert "servo_required=True" in src, (
        "servo_tracker_sync_validation must call compute_thesis_eligibility with "
        "servo_required=True — the whole point is correlating real motion."
    )

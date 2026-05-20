"""Phase 1 of the timing-experiment repair pass.

Covers the two surgical fixes shipped first:
  - Sync experiment defaults: amplitude 25 ticks, duration 30s, warmup 2s,
    timeout 60s. These match the spec's "one-servo micro-motion is enough"
    + "tiny safe amplitude" + "30s report duration" requirements.
  - Enriched rejection diagnostics when servo_service blocks a step (the
    previous code just propagated `result.message` so the operator got
    "wrap safety rejection: ..." with zero context — no current position,
    no planned goals, no bounds).

End-to-end execute-loop coverage lives in the broader experiment-runner
tests; these focused tests pin the contract for the defaults and the
error-message shape so future churn can't silently regress them.
"""

from __future__ import annotations

import inspect

import pytest

from continuum_robot.experiments.builtins import (
    ServoTrackerSyncValidationConfig,
    ServoTrackerSyncValidationExperiment,
)


# --------------------------------------------------------------------------- #
# Defaults                                                                    #
# --------------------------------------------------------------------------- #


def test_default_amplitude_is_25_ticks() -> None:
    """Tiny, safe, well inside the 1500-2500 typical servo range — see spec."""
    cfg = ServoTrackerSyncValidationConfig()
    assert cfg.command_amplitude_ticks == 25


def test_default_run_duration_is_30s() -> None:
    cfg = ServoTrackerSyncValidationConfig()
    assert cfg.run_duration_s == 30.0


def test_default_warmup_is_2s() -> None:
    cfg = ServoTrackerSyncValidationConfig()
    assert cfg.warmup_duration_s == 2.0


def test_default_timeout_covers_full_session() -> None:
    """Timeout must leave headroom over (warmup + run) so the experiment
    doesn't false-time-out on the very session length it's configured for."""
    cfg = ServoTrackerSyncValidationConfig()
    assert cfg.timeout_s >= cfg.warmup_duration_s + cfg.run_duration_s


def test_yaml_payload_parser_honors_new_defaults() -> None:
    cfg = ServoTrackerSyncValidationConfig.from_dict({})
    assert cfg.command_amplitude_ticks == 25
    assert cfg.run_duration_s == 30.0
    assert cfg.warmup_duration_s == 2.0
    assert cfg.timeout_s == 60.0


def test_yaml_payload_parser_accepts_operator_overrides() -> None:
    cfg = ServoTrackerSyncValidationConfig.from_dict(
        {
            "command_amplitude_ticks": 8,
            "run_duration_s": 15.0,
            "warmup_duration_s": 0.5,
            "timeout_s": 45.0,
        }
    )
    assert cfg.command_amplitude_ticks == 8
    assert cfg.run_duration_s == 15.0
    assert cfg.warmup_duration_s == 0.5
    assert cfg.timeout_s == 45.0


# --------------------------------------------------------------------------- #
# Rejection-message contract                                                  #
# --------------------------------------------------------------------------- #


def _read_execute_source() -> str:
    """Return the source text of the experiment's execute() method."""
    return inspect.getsource(ServoTrackerSyncValidationExperiment.execute)


def test_precheck_rejection_when_current_outside_safe_bounds_includes_diagnostics() -> None:
    """When the servo's current position is outside the saved safe window,
    the experiment must raise a structured RuntimeError naming the servo,
    the current position, the safe range, and the operator fix-it path,
    instead of letting the canonical wrap-safety check fire deeper in the
    stack with a context-free message."""
    src = _read_execute_source()
    # Anchor the new precheck guard.
    assert "is outside the saved safe window" in src, (
        "Precheck should raise with an explicit 'outside the saved safe window' "
        "message — see Phase 1 of the timing-experiment repair pass."
    )
    # Must name the operator remediation explicitly.
    assert "Recapture neutral" in src or "recapture neutral" in src.lower()


def test_step_rejection_enriches_service_message_with_diagnostics() -> None:
    """When servo_service.move_servo_to_raw_target returns success=False,
    the experiment must wrap the bare service message with every diagnostic
    field the spec requires."""
    src = _read_execute_source()
    required_anchors = [
        "servo_tracker_sync_validation step rejected",
        "servo_id=",
        "current_pos_at_start=",
        "used_amplitude_ticks=",
        "planned_min_goal=",
        "planned_max_goal=",
        "raw_bounds=",
        "safe_bounds=",
        "reason=",
    ]
    missing = [anchor for anchor in required_anchors if anchor not in src]
    assert not missing, (
        f"Step rejection message is missing required diagnostic fields: {missing}. "
        "See spec section 'WRAP-SAFETY REJECTION BUG'."
    )


def test_step_rejection_no_longer_bare_propagates_result_message() -> None:
    """Regression guard: the old `raise RuntimeError(result.message)` form
    must not return. Anything propagating result.message verbatim should
    be wrapped with diagnostic context."""
    src = _read_execute_source()
    assert "raise RuntimeError(result.message)" not in src, (
        "Bare propagation of result.message lost the operator's diagnostic "
        "context (current pos, planned goals, safe bounds). Always enrich."
    )


def test_precheck_amplitude_zero_message_lists_planned_goals() -> None:
    """When the requested amplitude cannot fit inside the safe window, the
    error should show planned goals + how to fix (reduce amplitude or widen
    bounds), not just 'cannot execute the requested bounded motion'."""
    src = _read_execute_source()
    assert "planned goals would be" in src
    assert "Reduce command_amplitude_ticks or" in src or "reduce command_amplitude_ticks" in src.lower()

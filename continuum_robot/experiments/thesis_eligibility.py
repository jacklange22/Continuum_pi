"""Thesis-eligibility stamping for timing / sync experiments.

A single tiny module that decides whether a given run was captured under
conditions we'd defend in the thesis (real Aurora backend, real servo,
live mode, sufficient samples) — and packages the verdict plus the
runtime fingerprint into a dict that the experiment can stamp directly
into ``session.metrics`` (and therefore into ``summary.experiment_metrics``).

Downstream readers (GUI summary, CLI report, debug.json) can show or
omit the figures based on the verdict, and the operator gets one clear
explanation of *why* a run isn't thesis-grade rather than a scattered
list of warnings.

Decision rules (intentionally strict, intentionally conservative):

  - mock_mode is the only sufficient reason to disqualify; everything
    else (no tracker backend, servo disconnected, bad force_status)
    funnels into the same ``debug_or_synthetic`` bucket so the GUI/CLI
    surface a *single* "this isn't thesis evidence" indicator.

  - The tracker_bridge / legacy bridge selection is explicitly
    disqualifying — the upstream prechecks already reject it, but we
    still mark it here so a downstream consumer reading only the JSON
    can tell.

  - ``force_status`` of anything other than ``"success"`` is treated as
    not-thesis-evidence. That keeps partial/timed-out and
    insufficient-samples runs out of the headline plot set, while
    still making the run reviewable.

This module does NOT inspect samples directly; it only sees the small
inputs the experiment already has on hand. Keeps it cheap and easy to
unit-test.
"""

from __future__ import annotations

from typing import Any


LABEL_THESIS = "thesis_evidence"
LABEL_DEBUG_OR_SYNTHETIC = "debug_or_synthetic"

# Backend identities/names that look real (live Aurora). Anything outside
# this allow-list is treated as synthetic/debug.
_LIVE_TRACKER_BACKENDS = frozenset(
    {
        "ndi_tracker_python",
        "ndi_tracker_aurora",
        "aurora",
        "ndi_aurora",
    }
)

# Selected backend names that explicitly indicate a non-thesis path
# (typically the legacy bridge or test/mock backends).
_NON_THESIS_SELECTED_NAMES = frozenset(
    {
        "bridge",
        "tracker_bridge",
        "mock",
        "synthetic",
        "test",
    }
)


def _looks_live_tracker(backend_identity: str | None, selected_backend_name: str | None) -> bool:
    identity = str(backend_identity or "").strip().lower()
    selected = str(selected_backend_name or "").strip().lower()
    if selected in _NON_THESIS_SELECTED_NAMES:
        return False
    if identity in _LIVE_TRACKER_BACKENDS:
        return True
    # Be slightly lenient: any identity that looks like a real Aurora
    # build (contains "aurora" or "ndi") is treated as live. The negative
    # path above already caught the disqualifying selections.
    if identity and ("aurora" in identity or "ndi" in identity):
        # Final guard: a "ndi_bridge" identity must still be rejected.
        if "bridge" in identity:
            return False
        return True
    return False


def compute_thesis_eligibility(
    *,
    mock_mode: bool,
    tracker_backend_identity: str | None,
    selected_backend_name: str | None,
    configured_backend_name: str | None = None,
    servo_required: bool = False,
    servo_connected: bool = False,
    servo_mock: bool = False,
    dry_run: bool = False,
    force_status: str = "success",
    extra_disqualifying_reasons: list[str] | None = None,
) -> dict[str, Any]:
    """Compute thesis-eligibility verdict + runtime environment fingerprint.

    Returns a dict with three top-level keys:

      ``eligible``: bool — quick yes/no.
      ``label``: ``"thesis_evidence"`` or ``"debug_or_synthetic"``.
      ``reasons``: list of human-readable strings explaining
        every reason the run is *not* thesis evidence. Empty list when
        ``eligible`` is True.
      ``runtime_environment``: dict — fingerprint of the run's environment
        (mock_mode, tracker backend identity, selected/configured backend
        name, servo status, dry_run state). Stamped verbatim so the
        downstream consumer never has to look it up again.

    The function is pure: same inputs → same output. No I/O.
    """
    reasons: list[str] = list(extra_disqualifying_reasons or [])

    if bool(mock_mode):
        reasons.append("runtime is in mock_mode (not live hardware)")

    if bool(dry_run):
        reasons.append("experiment was run with dry_run=true (no live motion)")

    if not _looks_live_tracker(tracker_backend_identity, selected_backend_name):
        reasons.append(
            f"tracker backend is not a recognised live Aurora path "
            f"(identity={tracker_backend_identity!r}, selected={selected_backend_name!r})"
        )

    if servo_required:
        if not bool(servo_connected):
            reasons.append("servo service is not connected")
        elif bool(servo_mock):
            reasons.append("servo service is in mock mode")

    if str(force_status or "").strip().lower() != "success":
        reasons.append(f"force_status={force_status!r} (run did not complete cleanly)")

    eligible = not reasons
    label = LABEL_THESIS if eligible else LABEL_DEBUG_OR_SYNTHETIC

    runtime_environment = {
        "mock_mode": bool(mock_mode),
        "dry_run": bool(dry_run),
        "tracker_backend_identity": str(tracker_backend_identity or ""),
        "tracker_backend_selected": str(selected_backend_name or ""),
        "tracker_backend_configured": str(configured_backend_name or ""),
        "servo_required": bool(servo_required),
        "servo_connected": bool(servo_connected),
        "servo_mock": bool(servo_mock),
        "force_status": str(force_status or "success"),
    }

    return {
        "eligible": bool(eligible),
        "label": label,
        "reasons": reasons,
        "runtime_environment": runtime_environment,
    }

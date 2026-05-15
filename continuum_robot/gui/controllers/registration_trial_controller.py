"""Controller for the Registration Trial Mode dialog.

Owns the per-landmark capture state machine and runs the
``registration_trial`` experiment when the operator finishes capturing.

This controller does **not** mutate the production registration session.
Captures flow through ``RegistrationService.sample_measurement_point_capture``,
which is the same code path used by the production tab but without touching
session state. The captured points are stored in this controller's own
buffer and handed off to the experiment runner as live captures via
``session.metrics[CAPTURES_SESSION_KEY]``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from continuum_robot.experiments.registration_trial import CAPTURES_SESSION_KEY


@dataclass
class TrialCaptureState:
    """Snapshot of the trial workflow for the GUI to render."""

    is_active: bool = False
    is_complete: bool = False
    captures_per_landmark: int = 50
    landmark_labels: list[str] = field(default_factory=list)
    current_label: str | None = None
    captured_counts_by_label: dict[str, int] = field(default_factory=dict)
    last_error: str | None = None
    status_message: str = "Press Start to begin a trial capture run."
    last_run_output_dir: Path | None = None
    last_trial_report_md: Path | None = None
    last_trial_report_json: Path | None = None
    last_run_summary: dict[str, Any] = field(default_factory=dict)


class RegistrationTrialController:
    """State machine for trial captures + handoff to the registration_trial experiment."""

    def __init__(
        self,
        *,
        registration_service,
        experiment_runner,
        project_root: Path,
        registration_yaml_path: str = "config/registration.yaml",
    ) -> None:
        self.registration_service = registration_service
        self.experiment_runner = experiment_runner
        self.project_root = Path(project_root)
        self.registration_yaml_path = str(registration_yaml_path)
        self._captures_by_label: dict[str, list[list[float]]] = {}
        self.state = TrialCaptureState()

    # --- query helpers ---------------------------------------------------

    @property
    def captures_by_label(self) -> dict[str, list[list[float]]]:
        """Read-only view of the trial captures collected so far."""
        return {label: list(points) for label, points in self._captures_by_label.items()}

    def total_captured(self) -> int:
        return sum(len(points) for points in self._captures_by_label.values())

    def target_total(self) -> int:
        return len(self.state.landmark_labels) * int(self.state.captures_per_landmark)

    def remaining_for_current_label(self) -> int:
        if not self.state.current_label:
            return 0
        target = int(self.state.captures_per_landmark)
        captured = int(self.state.captured_counts_by_label.get(self.state.current_label, 0))
        return max(0, target - captured)

    # --- workflow --------------------------------------------------------

    def start(
        self,
        landmark_labels: list[str],
        *,
        captures_per_landmark: int = 50,
    ) -> TrialCaptureState:
        """Begin a trial session with the given landmark order."""
        if captures_per_landmark < 1:
            raise ValueError("captures_per_landmark must be >= 1")
        if not landmark_labels:
            raise ValueError("At least one landmark is required for a trial run.")
        if len(landmark_labels) != len(set(landmark_labels)):
            raise ValueError("Landmark labels must be unique within a trial run.")
        self._captures_by_label = {label: [] for label in landmark_labels}
        self.state = TrialCaptureState(
            is_active=True,
            is_complete=False,
            captures_per_landmark=int(captures_per_landmark),
            landmark_labels=list(landmark_labels),
            current_label=landmark_labels[0],
            captured_counts_by_label={label: 0 for label in landmark_labels},
            last_error=None,
            status_message=(
                f"Position probe on {landmark_labels[0]} and capture {captures_per_landmark} samples."
            ),
        )
        return self.state

    def capture_one(self) -> int:
        """Capture one sample for the current landmark. Returns the new count."""
        if not self.state.is_active:
            raise RuntimeError("Trial is not active. Call start() first.")
        label = self.state.current_label
        if label is None:
            raise RuntimeError("No current landmark; trial may already be complete.")
        try:
            capture = self.registration_service.sample_measurement_point_capture()
        except Exception as exc:
            self.state.last_error = str(exc)
            self.state.status_message = f"Capture failed: {exc}"
            raise
        point = list(capture["point_xyz_mm"])
        self._captures_by_label.setdefault(label, []).append([float(v) for v in point])
        count = len(self._captures_by_label[label])
        self.state.captured_counts_by_label[label] = count
        target = int(self.state.captures_per_landmark)
        if count >= target:
            self.state.status_message = (
                f"{label} complete ({count}/{target}). Press Next to advance."
            )
        else:
            self.state.status_message = f"{label}: {count}/{target} captured."
        self.state.last_error = None
        return count

    def advance_to_next_label(self) -> str | None:
        """Move the cursor to the next landmark; returns the new current label or None."""
        if not self.state.is_active:
            raise RuntimeError("Trial is not active.")
        labels = list(self.state.landmark_labels)
        if self.state.current_label is None:
            return None
        try:
            idx = labels.index(self.state.current_label)
        except ValueError:
            self.state.current_label = None
            return None
        next_idx = idx + 1
        if next_idx >= len(labels):
            self.state.current_label = None
            self.state.is_complete = True
            self.state.status_message = (
                "All landmarks captured. Press Run Analysis to compute the trial report."
            )
            return None
        next_label = labels[next_idx]
        self.state.current_label = next_label
        target = int(self.state.captures_per_landmark)
        captured = int(self.state.captured_counts_by_label.get(next_label, 0))
        self.state.status_message = (
            f"Position probe on {next_label}; capture {target - captured} more samples."
        )
        return next_label

    def skip_current_label(self) -> str | None:
        """Drop the current landmark's collected captures and move on."""
        if not self.state.is_active or self.state.current_label is None:
            return None
        label = self.state.current_label
        self._captures_by_label[label] = []
        self.state.captured_counts_by_label[label] = 0
        return self.advance_to_next_label()

    def reset(self) -> None:
        """Discard any in-progress trial and return to the idle state."""
        self._captures_by_label = {}
        self.state = TrialCaptureState()

    def run_analysis(
        self,
        *,
        subset_sizes: list[int] | None = None,
        averaging_methods: list[str] | None = None,
    ) -> TrialCaptureState:
        """Hand the collected captures to the registration_trial experiment runner."""
        if not self._captures_by_label:
            raise RuntimeError("No captures recorded; nothing to analyze.")
        config_payload: dict[str, Any] = {
            "registration_yaml_path": self.registration_yaml_path,
            "captures_per_landmark": int(self.state.captures_per_landmark),
            "landmark_labels": list(self.state.landmark_labels),
        }
        if subset_sizes is not None:
            config_payload["subset_sizes"] = [int(v) for v in subset_sizes]
        if averaging_methods is not None:
            config_payload["averaging_methods"] = [str(v) for v in averaging_methods]

        # The registration_trial experiment supports a "live mode" where captures are
        # injected via session metrics. We hand the runner the captures so the
        # experiment consumes them without re-capturing.
        captures_snapshot = {label: list(points) for label, points in self._captures_by_label.items()}
        result = self.experiment_runner.run_experiment(
            "registration_trial",
            config=config_payload,
            operator_notes="registration_tab_trial_mode",
            inject_session_metrics={CAPTURES_SESSION_KEY: captures_snapshot},
        )

        if not result.success:
            self.state.last_error = result.message
            self.state.status_message = f"Trial analysis failed: {result.message}"
            raise RuntimeError(result.message)

        out_dir = result.paths.output_dir
        report_md = out_dir / "trial_report.md"
        report_json = out_dir / "trial_report.json"
        self.state.last_run_output_dir = out_dir
        self.state.last_trial_report_md = report_md if report_md.exists() else None
        self.state.last_trial_report_json = report_json if report_json.exists() else None
        self.state.last_run_summary = dict(result.summary.experiment_metrics or {})
        self.state.status_message = (
            f"Trial analysis complete. Report: {report_md if report_md.exists() else out_dir}"
        )
        self.state.last_error = None
        return self.state


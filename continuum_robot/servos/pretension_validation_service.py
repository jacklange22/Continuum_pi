"""Startup pretension/centered-state validation service.

This service is intentionally separate from neutral calibration.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any


@dataclass
class PretensionValidationResult:
    """Summary of a current-balance validation attempt."""

    passed: bool
    currents_ma: list[int]
    spread_ma: int | None
    message: str


@dataclass
class PretensionTrackerDisplacementResult:
    """Tracker-side displacement metric for one pretension validation command."""

    metric_frame: str
    before_position_mm: list[float] | None
    after_position_mm: list[float] | None
    displacement_vector_mm: list[float] | None
    displacement_magnitude_mm: float | None
    message: str


class PretensionValidationService:
    """Evaluate whether startup state is sufficiently centered/pretensioned."""

    def validate_current_balance(
        self,
        currents_ma: list[int | None],
        tolerance_ma: int,
    ) -> PretensionValidationResult:
        """Return a rich result describing current balance."""
        values = [int(current) for current in currents_ma if current is not None]
        if not values:
            return PretensionValidationResult(
                passed=False,
                currents_ma=[],
                spread_ma=None,
                message="Current balance unavailable because no servo currents were readable.",
            )

        spread = max(values) - min(values)
        passed = spread <= tolerance_ma
        if passed:
            message = f"Current balance passed with spread {spread} mA."
        else:
            message = f"Current balance failed with spread {spread} mA (limit {tolerance_ma} mA)."
        return PretensionValidationResult(
            passed=passed,
            currents_ma=values,
            spread_ma=spread,
            message=message,
        )

    def compute_tracker_displacement(
        self,
        before_snapshot,
        after_snapshot,
        *,
        tracker_tool_id: str = "0A",
    ) -> PretensionTrackerDisplacementResult:
        """Measure displacement from two tracking snapshots using the best available pose frame."""
        before_metric = self._snapshot_position_metric(before_snapshot, tracker_tool_id=tracker_tool_id)
        after_metric = self._snapshot_position_metric(after_snapshot, tracker_tool_id=tracker_tool_id)
        if before_metric is None or after_metric is None:
            return PretensionTrackerDisplacementResult(
                metric_frame="unavailable",
                before_position_mm=before_metric["position_mm"] if before_metric is not None else None,
                after_position_mm=after_metric["position_mm"] if after_metric is not None else None,
                displacement_vector_mm=None,
                displacement_magnitude_mm=None,
                message=(
                    "Tracker displacement metric is unavailable because the validation pose was not "
                    "readable in either robot frame or tracker frame."
                ),
            )
        if before_metric["frame"] != after_metric["frame"]:
            return PretensionTrackerDisplacementResult(
                metric_frame="frame_mismatch",
                before_position_mm=list(before_metric["position_mm"]),
                after_position_mm=list(after_metric["position_mm"]),
                displacement_vector_mm=None,
                displacement_magnitude_mm=None,
                message=(
                    "Tracker displacement metric is unavailable because the before/after validation "
                    "poses used different frames."
                ),
            )
        vector = [
            float(after_metric["position_mm"][index] - before_metric["position_mm"][index])
            for index in range(3)
        ]
        magnitude = math.sqrt(sum(component * component for component in vector))
        return PretensionTrackerDisplacementResult(
            metric_frame=str(before_metric["frame"]),
            before_position_mm=list(before_metric["position_mm"]),
            after_position_mm=list(after_metric["position_mm"]),
            displacement_vector_mm=vector,
            displacement_magnitude_mm=float(magnitude),
            message=(
                f"Validation displacement in {before_metric['frame']}: "
                f"{magnitude:.3f} mm."
            ),
        )

    def summarize_validation_runs(self, run_records: list[dict[str, Any]]) -> dict[str, Any]:
        """Summarize repeated pretension validation runs for operator comparison."""
        final_positions = [
            int(record["final_position_tick"])
            for record in run_records
            if record.get("final_position_tick") is not None
        ]
        trigger_currents = [
            float(record["trigger_current_ma"])
            for record in run_records
            if record.get("trigger_current_ma") is not None
        ]
        displacements = [
            float(record["validation_displacement_mm"])
            for record in run_records
            if record.get("validation_displacement_mm") is not None
        ]
        return {
            "run_count": len(run_records),
            "successful_run_count": sum(1 for record in run_records if bool(record.get("pretension_success"))),
            "accepted_run_count": sum(1 for record in run_records if bool(record.get("accepted"))),
            "final_position_spread_ticks": (
                max(final_positions) - min(final_positions) if final_positions else None
            ),
            "trigger_current_spread_ma": (
                max(trigger_currents) - min(trigger_currents) if trigger_currents else None
            ),
            "validation_displacement_mean_mm": (
                sum(displacements) / len(displacements) if displacements else None
            ),
            "validation_displacement_spread_mm": (
                max(displacements) - min(displacements) if displacements else None
            ),
            "validation_displacement_rms_mm": self._rms_about_mean(displacements),
        }

    @staticmethod
    def _rms_about_mean(values: list[float]) -> float | None:
        if not values:
            return None
        mean = sum(values) / len(values)
        return math.sqrt(sum((value - mean) ** 2 for value in values) / len(values))

    @staticmethod
    def _snapshot_position_metric(snapshot, *, tracker_tool_id: str) -> dict[str, Any] | None:
        if snapshot is None:
            return None
        matrix = getattr(snapshot, "T_robot_tip", None)
        if matrix is not None:
            return {
                "frame": "robot_tip_mm",
                "position_mm": [
                    float(matrix[0][3]),
                    float(matrix[1][3]),
                    float(matrix[2][3]),
                ],
            }
        tool = getattr(snapshot, "tools", {}).get(str(tracker_tool_id))
        translation = getattr(tool, "translation_mm", None) if tool is not None else None
        tracking_state = getattr(tool, "tracking_state", None) if tool is not None else None
        if translation is None or tracking_state not in {"tracked", "ok", "valid"}:
            return None
        return {
            "frame": f"tracker_{str(tracker_tool_id)}_mm",
            "position_mm": [float(value) for value in translation],
        }

"""Startup pretension/centered-state validation service.

This service is intentionally separate from neutral calibration.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class PretensionValidationResult:
    """Summary of a current-balance validation attempt."""

    passed: bool
    currents_ma: list[int]
    spread_ma: int | None
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

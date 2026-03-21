"""Startup pretension/centered-state validation service.

This service is intentionally separate from neutral calibration.
"""


class PretensionValidationService:
    """Evaluate whether startup state is sufficiently centered/pretensioned."""

    def validate_current_balance(self, currents_ma: list[int], tolerance_ma: int) -> bool:
        """Return True when max-min current spread is within tolerance."""
        if not currents_ma:
            return False
        return max(currents_ma) - min(currents_ma) <= tolerance_ma

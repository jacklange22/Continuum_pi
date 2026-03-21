"""Map tendon displacement to servo position ticks."""

import math


class TendonDisplacementMapper:
    """Converts tendon displacement into position commands around neutral."""

    def __init__(self, spool_diameter_cm: float, ticks_per_rev: int = 4096) -> None:
        self.spool_diameter_cm = spool_diameter_cm
        self.ticks_per_rev = ticks_per_rev

    def displacement_cm_to_ticks(self, displacement_cm: float) -> int:
        """Convert tendon displacement to position delta in ticks."""
        circumference_cm = math.pi * self.spool_diameter_cm
        rev = displacement_cm / circumference_cm
        return int(round(rev * self.ticks_per_rev))

    def to_goal_positions(
        self, tendon_displacements_cm: list[float], neutral_ticks: list[int]
    ) -> list[int]:
        """Add displacement-derived tick deltas to neutral setpoints."""
        if len(tendon_displacements_cm) != len(neutral_ticks):
            raise ValueError("Displacement vector and neutral setpoint length mismatch")
        return [
            neutral + self.displacement_cm_to_ticks(dl)
            for dl, neutral in zip(tendon_displacements_cm, neutral_ticks)
        ]

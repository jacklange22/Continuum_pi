"""Safety checks for motion commands and telemetry."""


class SafetyGuard:
    """Validates goal positions and measured currents against thresholds."""

    def __init__(self, min_offset_ticks: int, max_offset_ticks: int, max_current_ma: int) -> None:
        self.min_offset_ticks = min_offset_ticks
        self.max_offset_ticks = max_offset_ticks
        self.max_current_ma = max_current_ma

    def validate_positions(self, goals: list[int], neutral: list[int]) -> None:
        """Raise ValueError when any goal exceeds configured offset range."""
        for goal, base in zip(goals, neutral):
            delta = goal - base
            if delta < self.min_offset_ticks or delta > self.max_offset_ticks:
                raise ValueError(f"Unsafe position offset: {delta}")

    def validate_currents(self, currents_ma: list[int]) -> None:
        """Raise ValueError when any current exceeds threshold."""
        for current in currents_ma:
            if current > self.max_current_ma:
                raise ValueError(f"Current threshold exceeded: {current} mA")

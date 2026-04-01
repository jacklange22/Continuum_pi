"""Safety checks for motion commands and telemetry."""

from __future__ import annotations

import time


class SafetyGuard:
    """Validates goal positions and measured currents against thresholds."""

    def __init__(
        self,
        min_offset_ticks: int,
        max_offset_ticks: int,
        max_current_ma: int,
        *,
        default_pretension_current_threshold_ma: int = 220,
        fine_jog_step_ticks: int = 5,
        coarse_jog_step_ticks: int = 25,
        software_position_margin_ticks: int = 64,
        telemetry_stale_after_s: float = 0.25,
        pretension_step_ticks: int = 2,
        pretension_timeout_s: float = 10.0,
        pretension_settle_time_s: float = 0.05,
        max_temperature_c: int = 70,
        min_input_voltage_mv: int = 4000,
        max_input_voltage_mv: int | None = None,
        time_fn=time.monotonic,
    ) -> None:
        self.min_offset_ticks = min_offset_ticks
        self.max_offset_ticks = max_offset_ticks
        self.max_current_ma = max_current_ma
        self.default_pretension_current_threshold_ma = default_pretension_current_threshold_ma
        self.fine_jog_step_ticks = fine_jog_step_ticks
        self.coarse_jog_step_ticks = coarse_jog_step_ticks
        self.software_position_margin_ticks = software_position_margin_ticks
        self.telemetry_stale_after_s = telemetry_stale_after_s
        self.pretension_step_ticks = pretension_step_ticks
        self.pretension_timeout_s = pretension_timeout_s
        self.pretension_settle_time_s = pretension_settle_time_s
        self.max_temperature_c = max_temperature_c
        self.min_input_voltage_mv = min_input_voltage_mv
        self.max_input_voltage_mv = max_input_voltage_mv
        self._time_fn = time_fn

    def validate_positions(self, goals: list[int], neutral: list[int]) -> None:
        """Raise ValueError when any goal exceeds configured offset range."""
        for goal, base in zip(goals, neutral):
            delta = goal - base
            if delta < self.min_offset_ticks or delta > self.max_offset_ticks:
                raise ValueError(f"Unsafe position offset: {delta}")

    def validate_currents(self, currents_ma: list[int | None], *, require_present: bool = False) -> None:
        """Raise ValueError when any current exceeds threshold."""
        for current in currents_ma:
            if current is None:
                if require_present:
                    raise ValueError("Current telemetry is unavailable.")
                continue
            if current > self.max_current_ma:
                raise ValueError(f"Current threshold exceeded: {current} mA")

    def validate_temperature(self, temperature_c: int | None, *, require_present: bool = False) -> None:
        """Raise ValueError when temperature telemetry is missing or too high."""
        if temperature_c is None:
            if require_present:
                raise ValueError("Temperature telemetry is unavailable.")
            return
        if temperature_c >= self.max_temperature_c:
            raise ValueError(f"Temperature threshold exceeded: {temperature_c} C")

    def validate_voltage(self, voltage_mv: int | None, *, require_present: bool = False) -> None:
        """Raise ValueError when voltage telemetry is missing or outside configured safe bounds."""
        if voltage_mv is None:
            if require_present:
                raise ValueError("Input voltage telemetry is unavailable.")
            return
        if int(voltage_mv) < int(self.min_input_voltage_mv):
            raise ValueError(
                "Input voltage is below the configured motion minimum: "
                f"{voltage_mv} mV < {self.min_input_voltage_mv} mV."
            )
        if self.max_input_voltage_mv is not None and int(voltage_mv) > int(self.max_input_voltage_mv):
            raise ValueError(
                "Input voltage is above the configured motion maximum: "
                f"{voltage_mv} mV > {self.max_input_voltage_mv} mV."
            )

    def validate_telemetry_freshness(self, read_monotonic_s: float | None) -> None:
        """Raise ValueError when telemetry age is unknown or stale."""
        if read_monotonic_s is None:
            raise ValueError("Telemetry freshness is unknown.")
        age = self._time_fn() - float(read_monotonic_s)
        if age > self.telemetry_stale_after_s:
            raise ValueError(
                f"Telemetry is stale ({age:.3f} s old, limit {self.telemetry_stale_after_s:.3f} s)."
            )

    def validate_jog_delta(self, delta_ticks: int) -> None:
        """Keep manual jogs within the configured bring-up step size."""
        if abs(int(delta_ticks)) > int(self.coarse_jog_step_ticks):
            raise ValueError(
                f"Manual jog {delta_ticks} exceeds the configured coarse jog limit of {self.coarse_jog_step_ticks} ticks."
            )

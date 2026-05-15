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
        servo_model: str = "XC330-M288-T",
        servo_reported_current_hard_limit_ma: int | None = None,
        servo_reported_current_warning_ma: int | None = None,
        current_warning_ma: int | None = 500,
        transient_current_spike_ma: int | None = None,
        sustained_jam_current_ma: int | None = None,
        sustained_jam_cycles: int = 3,
        transient_spike_policy: str = "warn_drop_sample_continue",
        sustained_jam_policy: str = "stop_safely",
        current_spike_resync_enabled: bool = True,
        current_spike_cooldown_s: float = 0.25,
        current_spike_return_to_previous_safe_goal: bool = False,
        current_spike_max_events_per_servo: int = 6,
        current_safety_basis: str = (
            "servo-reported input-current estimate; hard safety uses absolute current magnitude"
        ),
        default_pretension_current_threshold_ma: int = 220,
        fine_jog_step_ticks: int = 5,
        coarse_jog_step_ticks: int = 25,
        software_position_margin_ticks: int = 64,
        telemetry_stale_after_s: float = 0.25,
        torque_off_on_disconnect: bool = False,
        wrap_risk_margin_ticks: int = 128,
        max_raw_jump_without_wrap_risk_ticks: int = 900,
        pretension_untensioned_reference_tick: int = 4095,
        pretension_start_mode: str = "current_position",
        pretension_step_ticks: int = 2,
        pretension_timeout_s: float = 10.0,
        pretension_settle_time_s: float = 0.05,
        pretension_baseline_sample_count: int = 5,
        pretension_current_filter_window: int = 3,
        pretension_current_delta_threshold_ma: int = 60,
        pretension_absolute_trigger_current_ma: int | None = 220,
        pretension_hard_current_stop_ma: int | None = None,
        pretension_max_travel_ticks: int = 320,
        max_temperature_c: int = 70,
        min_input_voltage_mv: int = 4000,
        max_input_voltage_mv: int | None = None,
        time_fn=time.monotonic,
    ) -> None:
        self.min_offset_ticks = min_offset_ticks
        self.max_offset_ticks = max_offset_ticks
        self.servo_model = str(servo_model or "unknown")
        self.servo_reported_current_hard_limit_ma = (
            int(max_current_ma)
            if servo_reported_current_hard_limit_ma in (None, "")
            else int(servo_reported_current_hard_limit_ma)
        )
        self.servo_reported_current_warning_ma = (
            None
            if servo_reported_current_warning_ma in (None, "")
            else int(servo_reported_current_warning_ma)
        )
        self.current_warning_ma = (
            int(current_warning_ma)
            if current_warning_ma not in (None, "")
            else self.servo_reported_current_warning_ma
        )
        self.transient_current_spike_ma = int(
            self.servo_reported_current_hard_limit_ma
            if transient_current_spike_ma in (None, "")
            else transient_current_spike_ma
        )
        self.sustained_jam_current_ma = int(
            self.servo_reported_current_hard_limit_ma
            if sustained_jam_current_ma in (None, "")
            else sustained_jam_current_ma
        )
        self.sustained_jam_cycles = max(1, int(sustained_jam_cycles))
        self.transient_spike_policy = str(transient_spike_policy or "warn_drop_sample_continue").strip().lower()
        self.sustained_jam_policy = str(sustained_jam_policy or "stop_safely").strip().lower()
        self.current_spike_resync_enabled = bool(current_spike_resync_enabled)
        self.current_spike_cooldown_s = max(0.0, float(current_spike_cooldown_s))
        self.current_spike_return_to_previous_safe_goal = bool(current_spike_return_to_previous_safe_goal)
        self.current_spike_max_events_per_servo = max(1, int(current_spike_max_events_per_servo))
        self.current_safety_basis = str(current_safety_basis or "").strip() or (
            "servo-reported input-current estimate; hard safety uses absolute current magnitude"
        )
        self.max_current_ma = self.servo_reported_current_hard_limit_ma
        self.default_pretension_current_threshold_ma = default_pretension_current_threshold_ma
        self.fine_jog_step_ticks = fine_jog_step_ticks
        self.coarse_jog_step_ticks = coarse_jog_step_ticks
        self.software_position_margin_ticks = software_position_margin_ticks
        self.telemetry_stale_after_s = telemetry_stale_after_s
        self.torque_off_on_disconnect = bool(torque_off_on_disconnect)
        self.wrap_risk_margin_ticks = int(wrap_risk_margin_ticks)
        self.max_raw_jump_without_wrap_risk_ticks = int(max_raw_jump_without_wrap_risk_ticks)
        self.pretension_untensioned_reference_tick = pretension_untensioned_reference_tick
        self.pretension_start_mode = str(pretension_start_mode or "current_position").strip().lower()
        self.pretension_step_ticks = pretension_step_ticks
        self.pretension_timeout_s = pretension_timeout_s
        self.pretension_settle_time_s = pretension_settle_time_s
        self.pretension_baseline_sample_count = pretension_baseline_sample_count
        self.pretension_current_filter_window = pretension_current_filter_window
        self.pretension_current_delta_threshold_ma = pretension_current_delta_threshold_ma
        self.pretension_absolute_trigger_current_ma = pretension_absolute_trigger_current_ma
        self.pretension_hard_current_stop_ma = (
            max_current_ma
            if pretension_hard_current_stop_ma in (None, "")
            else int(pretension_hard_current_stop_ma)
        )
        self.pretension_max_travel_ticks = pretension_max_travel_ticks
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
        """Raise ValueError when any current magnitude exceeds the configured hard limit."""
        for current in currents_ma:
            if current is None:
                if require_present:
                    raise ValueError("Current telemetry is unavailable.")
                continue
            current_ma = int(current)
            magnitude_ma = abs(current_ma)
            if magnitude_ma > int(self.servo_reported_current_hard_limit_ma):
                raise ValueError(
                    "Servo-reported current magnitude threshold exceeded: "
                    f"|{current_ma}|={magnitude_ma} mA > "
                    f"{self.servo_reported_current_hard_limit_ma} mA "
                    f"(model={self.servo_model}; basis={self.current_safety_basis})."
                )

    def is_transient_current_spike(self, current_ma: int | None) -> bool:
        """Return True when the sample meets the transient-spike threshold."""
        if current_ma is None:
            return False
        return abs(int(current_ma)) >= int(self.transient_current_spike_ma)

    def is_sustained_jam_current(self, current_ma: int | None) -> bool:
        """Return True when the sample meets the sustained-jam threshold."""
        if current_ma is None:
            return False
        return abs(int(current_ma)) >= int(self.sustained_jam_current_ma)

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
        age = self.telemetry_age_s(read_monotonic_s)
        if age is None:
            raise ValueError("Telemetry freshness is unknown.")
        if age > self.telemetry_stale_after_s:
            raise ValueError(
                f"Telemetry is stale ({age:.3f} s > {self.telemetry_stale_after_s:.3f} s)."
            )

    def telemetry_age_s(self, read_monotonic_s: float | None) -> float | None:
        """Return telemetry age in seconds, or None when the read time is unknown."""
        if read_monotonic_s is None:
            return None
        return max(0.0, float(self._time_fn()) - float(read_monotonic_s))

    def telemetry_is_fresh(self, read_monotonic_s: float | None) -> bool | None:
        """Return whether telemetry is within the configured freshness limit."""
        age = self.telemetry_age_s(read_monotonic_s)
        if age is None:
            return None
        return bool(age <= float(self.telemetry_stale_after_s))

    def validate_jog_delta(self, delta_ticks: int) -> None:
        """Keep manual jogs within the configured bring-up step size."""
        if abs(int(delta_ticks)) > int(self.coarse_jog_step_ticks):
            raise ValueError(
                f"Manual jog {delta_ticks} exceeds the configured coarse jog limit of {self.coarse_jog_step_ticks} ticks."
            )

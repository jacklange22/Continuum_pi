"""Smooth bounded command-space patterns for the two-segment slow motion demo.

This module is **pure** -- it does not touch hardware, the servo bus, or the
tracker. It produces a deterministic sequence of ``TwoSegmentPatternPoint``
values that the demo controller can stream to the bus through the existing
safe command path.

Each pattern is parameterised by an amplitude in tip-target cm (so the same
units as ``pair_axis_convention`` and the existing collect-pose schedule
generator), a cycle duration in seconds, a number of cycles, and a sampling
rate. Patterns are wrapped in a configurable ramp-in / hold / ramp-out
envelope so the robot never starts or stops with a discontinuity at full
amplitude.

The coupling mode determines how the top segment's command relates to the
bottom segment's command. ``phase_shifted`` is the default because it
exercises both segments through the workspace without putting them in
exactly opposing tension states. ``same_direction``, ``opposite_direction``,
``top_scaled``, ``bottom_only``, and ``top_only`` are also supported.

The tendon-displacement convention matches ``pair_axis_convention``: a
pair-space command ``(px, py)`` maps to cable deltas ``[-px, -py, +px, +py]``
in tendon order ``[+X, +Y, -X, -Y]``. The all-8 vector is segment-indexed,
NOT bottom/top-ordered -- the demo controller resolves bottom/top via the
runtime ``RobotOperatingContext``.

This module deliberately does not depend on any GUI, servo, or tracker
imports; the tests can construct trajectories without instantiating a
robot.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
import random
from typing import Sequence


# Canonical pattern names. The list is the source of truth for the GUI dropdown.
PATTERN_FIGURE8 = "figure8"
PATTERN_CIRCLE = "circle"
PATTERN_OVAL = "oval"
PATTERN_SWEEP_X = "sweep_x"
PATTERN_SWEEP_Y = "sweep_y"
PATTERN_RASTER = "raster"
PATTERN_CLOVER = "clover"
PATTERN_NEUTRAL_PULSE = "neutral_pulse"
PATTERN_LISSAJOUS = "custom_lissajous"
# Organic / weird patterns. These are aperiodic (or quasi-periodic) and
# intentionally don't trace a single closed curve — they're designed to
# look "alive" instead of mechanical. Per-axis amplitude is still strictly
# bounded by ``amplitude`` so the soft tick cap still holds.
PATTERN_CREEPY_SQUID = "creepy_squid"
PATTERN_SLOW_DRIFT = "slow_drift"

SUPPORTED_PATTERNS = (
    PATTERN_FIGURE8,
    PATTERN_CIRCLE,
    PATTERN_OVAL,
    PATTERN_SWEEP_X,
    PATTERN_SWEEP_Y,
    PATTERN_RASTER,
    PATTERN_CLOVER,
    PATTERN_NEUTRAL_PULSE,
    PATTERN_LISSAJOUS,
    PATTERN_CREEPY_SQUID,
    PATTERN_SLOW_DRIFT,
)

COUPLING_SAME = "same_direction"
COUPLING_OPPOSITE = "opposite_direction"
COUPLING_PHASE_SHIFTED = "phase_shifted"
COUPLING_TOP_SCALED = "top_scaled"
COUPLING_BOTTOM_ONLY = "bottom_only"
COUPLING_TOP_ONLY = "top_only"

SUPPORTED_COUPLINGS = (
    COUPLING_SAME,
    COUPLING_OPPOSITE,
    COUPLING_PHASE_SHIFTED,
    COUPLING_TOP_SCALED,
    COUPLING_BOTTOM_ONLY,
    COUPLING_TOP_ONLY,
)

# Amplitude presets exposed to the GUI. The demo is presentation-only; the
# operator can pick a preset rather than typing a number, and the presets
# are sorted in increasing aggressiveness.
AMPLITUDE_PRESETS_CM = (0.10, 0.25, 0.50, 0.75, 1.00)
DEFAULT_AMPLITUDE_CM = 0.25
DEFAULT_CYCLE_DURATION_S = 45.0
DEFAULT_UPDATE_RATE_HZ = 3.0
DEFAULT_RAMP_S = 5.0
DEFAULT_HOLD_S = 1.0
DEFAULT_CYCLES = 2

# Phase labels recorded on every pattern point for the demo trace.
PHASE_HOLD_START = "hold_start"
PHASE_RAMP_IN = "ramp_in"
PHASE_PATTERN = "pattern"
PHASE_RAMP_OUT = "ramp_out"
PHASE_HOLD_END = "hold_end"


@dataclass(frozen=True)
class TwoSegmentPatternPoint:
    """One trajectory point emitted by the pattern generator.

    Bottom/top here are the *physical* segment roles. The demo controller
    converts these to segment-indexed ``TwoSegmentCommand`` values using the
    runtime ``RobotOperatingContext`` so the canonical bottom/top mapping
    stays in one place.

    Tendon vectors follow the canonical convention from
    ``pair_axis_convention``: for a pair command ``(px, py)`` the tendon
    delta is ``[-px, -py, +px, +py]`` in tendon order ``[+X, +Y, -X, -Y]``.
    """

    sample_index: int
    elapsed_s: float
    phase_label: str  # one of PHASE_*
    phase_fraction: float  # 0..1 across the full pattern run (hold+ramp+pattern+ramp+hold)
    bottom_x_cm: float
    bottom_y_cm: float
    top_x_cm: float
    top_y_cm: float
    bottom_tendon_cm: tuple[float, float, float, float]
    top_tendon_cm: tuple[float, float, float, float]
    # all_8_tendon_cm is segment-indexed: bottom 4 first, top 4 second.
    # The controller re-maps this to physical servo IDs via the operating
    # context's bottom_servo_ids / top_servo_ids before writing the bus.
    all_8_tendon_cm: tuple[float, float, float, float, float, float, float, float]


def expand_pair_to_tendon_cm(px: float, py: float) -> tuple[float, float, float, float]:
    """Pair command ``(px, py)`` in tip-target cm → tendon delta in cm.

    This mirrors ``continuum_robot.experiments.pair_axis_convention.
    expand_pair_command_to_cable_deltas`` but returns a tuple so the result
    can be stored on a frozen dataclass without further copy.
    """
    px_f = float(px)
    py_f = float(py)
    return (-px_f, -py_f, px_f, py_f)


def _pair_command_for_pattern(
    pattern: str,
    *,
    theta: float,
    amplitude: float,
    lissajous_a: float,
    lissajous_b: float,
    lissajous_phase: float,
    raster_ratio: float,
    seed: int,
) -> tuple[float, float]:
    """Return one ``(px, py)`` pair-space command for the given pattern at angle theta.

    ``theta`` is the *pattern* phase in radians (one cycle == 2π). All
    patterns are clamped so ``hypot(px, py) <= amplitude * sqrt(2)`` at any
    point; circular patterns sit exactly on the amplitude circle.
    """
    name = str(pattern or "").strip().lower()
    if name == PATTERN_FIGURE8:
        # Lissajous (1, 2): smooth figure-8 with the long axis on x.
        return amplitude * math.sin(theta), amplitude * math.sin(2.0 * theta) / 2.0
    if name == PATTERN_CIRCLE:
        return amplitude * math.cos(theta), amplitude * math.sin(theta)
    if name == PATTERN_OVAL:
        return amplitude * math.cos(theta), 0.6 * amplitude * math.sin(theta)
    if name == PATTERN_SWEEP_X:
        return amplitude * math.sin(theta), 0.0
    if name == PATTERN_SWEEP_Y:
        return 0.0, amplitude * math.sin(theta)
    if name == PATTERN_RASTER:
        # Slow-axis sweep on x with a faster sine on y; raster_ratio sets
        # how many y-cycles fit inside one x-cycle. A ratio of 3 looks like
        # 3 stripes across the workspace.
        ratio = max(1.0, float(raster_ratio))
        return amplitude * math.sin(theta), amplitude * math.sin(ratio * theta)
    if name == PATTERN_CLOVER:
        # 4-leaf rose: r = amplitude * sin(2θ).
        r = amplitude * math.sin(2.0 * theta)
        return r * math.cos(theta), r * math.sin(theta)
    if name == PATTERN_NEUTRAL_PULSE:
        # Slow excursion along x with long dwell at neutral. Useful as the
        # smallest possible demo (essentially a thresholded triangle wave).
        return amplitude * math.sin(theta), 0.0
    if name == PATTERN_LISSAJOUS:
        a = max(0.1, float(lissajous_a))
        b = max(0.1, float(lissajous_b))
        return (
            amplitude * math.sin(a * theta + lissajous_phase),
            amplitude * math.sin(b * theta),
        )
    if name == PATTERN_CREEPY_SQUID:
        return _creepy_squid_pair(theta=theta, amplitude=amplitude)
    if name == PATTERN_SLOW_DRIFT:
        return _slow_drift_pair(theta=theta, amplitude=amplitude, seed=seed)
    # Unknown pattern: fall back to a safe figure-8 so we never silently
    # over-command. The controller validates the pattern name upstream so
    # this path is unreachable in practice.
    return amplitude * math.sin(theta), amplitude * math.sin(2.0 * theta) / 2.0


# ----------------------------------------------------------------------------
# Organic / weird pattern generators
# ----------------------------------------------------------------------------


# Incommensurate-frequency Fourier components for the creepy squid pattern.
# Coefficients sum to 1.0 in each axis so the per-axis output is bounded
# strictly to [-amplitude, amplitude]. Frequencies are intentionally
# irrational ratios (golden ratio, e, √2, √5, √11) so the trajectory does
# not close on itself over any reasonable run length — that aperiodicity
# is the "weird, alive" feeling. Phases are off-grid to prevent accidental
# axis alignment that would make the motion look mechanical.
_SQUID_X_COMPONENTS: tuple[tuple[float, float, float], ...] = (
    (1.000, 0.45, 0.00),   # base — slow fundamental
    (1.618, 0.30, 1.10),   # golden ratio
    (2.718, 0.15, 2.30),   # e
    (4.123, 0.10, 0.70),   # √17
)
_SQUID_Y_COMPONENTS: tuple[tuple[float, float, float], ...] = (
    (0.927, 0.40, 1.57),   # nearly the same fundamental as x → slow beat
    (1.414, 0.30, 0.50),   # √2
    (2.236, 0.20, 1.70),   # √5
    (3.318, 0.10, 2.90),   # √11
)


def _creepy_squid_pair(*, theta: float, amplitude: float) -> tuple[float, float]:
    """Squid-tentacle-ish quasi-periodic motion bounded per-axis by amplitude.

    The motion is a sum of incommensurate-ratio sinusoids in x and y plus a
    slow breathing envelope, then scaled by ``amplitude``. By construction
    each component sum is bounded by 1.0 and the envelope is bounded by
    1.0, so ``|px|, |py| <= amplitude`` at every point. The aperiodicity
    is what makes it look alive instead of mechanical — over the demo's
    typical 1–2 cycle run length the path never quite closes back on
    itself.
    """
    x_sum = sum(
        weight * math.sin(freq * theta + phase)
        for (freq, weight, phase) in _SQUID_X_COMPONENTS
    )
    y_sum = sum(
        weight * math.sin(freq * theta + phase)
        for (freq, weight, phase) in _SQUID_Y_COMPONENTS
    )
    # Slow breathing envelope (~3× slower than the fundamental) gives the
    # squid a "pulsing" feel: it expands, contracts, and almost-returns
    # to neutral, then expands again at a slightly different angle.
    envelope = 0.55 + 0.45 * math.sin(theta * 0.30 + 0.20)
    return (amplitude * envelope * x_sum, amplitude * envelope * y_sum)


def _slow_drift_pair(*, theta: float, amplitude: float, seed: int) -> tuple[float, float]:
    """Pseudo-random low-frequency drift — looks like a tentacle wandering.

    The output is a finite Fourier series with seeded random phases and a
    1/f weight spectrum. Visually this looks like organic drift: slow
    bias-changing motion that never quite repeats but also never moves
    quickly. With the demo's typical 25 s cycle and 1 cycle, the operator
    sees a single 25 s journey across the bounded workspace.

    Bounded per-axis by ``amplitude`` because the weights are normalized
    so ``sum(|w|) = 1``. The ``seed`` parameter controls which random
    phases are drawn so the same seed always produces the same drift —
    important for tests and reproducible demos.
    """
    # Use a fixed RNG salt so a seed=0 still produces meaningful variation
    # without colliding with seed=0 elsewhere in the codebase.
    rng = random.Random(int(seed) * 7919 + 99991)
    x = 0.0
    y = 0.0
    weight_total = 0.0
    # 6 harmonics: enough to look organic, few enough to render cleanly.
    for n in range(1, 7):
        weight = 1.0 / n  # 1/f → low frequencies dominate, motion stays slow
        # Frequencies are deliberately fractional and offset by RNG jitter
        # so two consecutive harmonics never end up at integer ratios.
        freq_x = (n * 0.6) + (rng.random() - 0.5) * 0.30
        freq_y = (n * 0.55) + (rng.random() - 0.5) * 0.30
        phase_x = rng.random() * math.tau
        phase_y = rng.random() * math.tau
        x += weight * math.sin(freq_x * theta + phase_x)
        y += weight * math.sin(freq_y * theta + phase_y)
        weight_total += weight
    # Normalize so |x|, |y| <= 1 → output magnitudes <= amplitude.
    x /= weight_total
    y /= weight_total
    return (amplitude * x, amplitude * y)


def _apply_coupling(
    *,
    bottom_xy: tuple[float, float],
    pattern: str,
    theta: float,
    amplitude: float,
    coupling: str,
    top_scale: float,
    phase_offset_rad: float,
    lissajous_a: float,
    lissajous_b: float,
    lissajous_phase: float,
    raster_ratio: float,
    seed: int,
) -> tuple[float, float]:
    """Compute the top-segment ``(px, py)`` pair command from the bottom command.

    Each coupling mode is independent of the pattern function: it always
    re-uses the same pattern but with a different role (mirror, scale,
    phase-offset). This keeps the visual shape of the bottom and top paths
    related so the operator can see what coupling does.
    """
    name = str(coupling or "").strip().lower()
    if name == COUPLING_SAME:
        return bottom_xy
    if name == COUPLING_OPPOSITE:
        return (-bottom_xy[0], -bottom_xy[1])
    if name == COUPLING_TOP_SCALED:
        scale = max(0.0, float(top_scale))
        return (bottom_xy[0] * scale, bottom_xy[1] * scale)
    if name == COUPLING_BOTTOM_ONLY:
        return (0.0, 0.0)
    if name == COUPLING_TOP_ONLY:
        # When the operator picks top_only we generate the pattern in the
        # top channel and zero the bottom. We accomplish this here by
        # returning the pattern at the same theta (it gets mapped onto the
        # top tendons by the caller, which also zeros the bottom).
        return _pair_command_for_pattern(
            pattern,
            theta=theta,
            amplitude=amplitude,
            lissajous_a=lissajous_a,
            lissajous_b=lissajous_b,
            lissajous_phase=lissajous_phase,
            raster_ratio=raster_ratio,
            seed=seed,
        )
    if name == COUPLING_PHASE_SHIFTED:
        return _pair_command_for_pattern(
            pattern,
            theta=theta + float(phase_offset_rad),
            amplitude=amplitude,
            lissajous_a=lissajous_a,
            lissajous_b=lissajous_b,
            lissajous_phase=lissajous_phase,
            raster_ratio=raster_ratio,
            seed=seed,
        )
    # Unknown coupling defaults to same_direction for safety.
    return bottom_xy


@dataclass(frozen=True)
class PatternEnvelope:
    """The hold + ramp + pattern + ramp + hold envelope around the raw shape.

    ``hold_start_s`` and ``hold_end_s`` hold the robot at neutral so the
    operator can see the robot is ready before the motion starts and is back
    at neutral at the end. Ramp segments scale the pattern amplitude from
    zero to full and back, so the first sample is always at neutral and the
    last sample is also at neutral. This is the safety contract every
    pattern obeys.
    """

    hold_start_s: float = DEFAULT_HOLD_S
    ramp_in_s: float = DEFAULT_RAMP_S
    pattern_duration_s: float = DEFAULT_CYCLE_DURATION_S * DEFAULT_CYCLES
    ramp_out_s: float = DEFAULT_RAMP_S
    hold_end_s: float = DEFAULT_HOLD_S

    @property
    def total_duration_s(self) -> float:
        return float(
            max(0.0, self.hold_start_s)
            + max(0.0, self.ramp_in_s)
            + max(0.0, self.pattern_duration_s)
            + max(0.0, self.ramp_out_s)
            + max(0.0, self.hold_end_s)
        )


def _envelope_amplitude_scale(
    *,
    elapsed_s: float,
    envelope: PatternEnvelope,
) -> tuple[str, float]:
    """Return ``(phase_label, amplitude_scale)`` for the given elapsed time.

    ``amplitude_scale`` is in ``[0, 1]``. Within ramp segments it ramps
    linearly; within hold segments it is 0; within the pattern body it is 1.
    """
    if elapsed_s < envelope.hold_start_s:
        return PHASE_HOLD_START, 0.0
    t = elapsed_s - envelope.hold_start_s
    if t < envelope.ramp_in_s:
        if envelope.ramp_in_s <= 0.0:
            return PHASE_RAMP_IN, 1.0
        return PHASE_RAMP_IN, t / envelope.ramp_in_s
    t -= envelope.ramp_in_s
    if t < envelope.pattern_duration_s:
        return PHASE_PATTERN, 1.0
    t -= envelope.pattern_duration_s
    if t < envelope.ramp_out_s:
        if envelope.ramp_out_s <= 0.0:
            return PHASE_RAMP_OUT, 0.0
        return PHASE_RAMP_OUT, max(0.0, 1.0 - t / envelope.ramp_out_s)
    return PHASE_HOLD_END, 0.0


@dataclass(frozen=True)
class PatternRequest:
    """User-facing pattern configuration."""

    pattern: str = PATTERN_FIGURE8
    amplitude_cm: float = DEFAULT_AMPLITUDE_CM
    cycle_duration_s: float = DEFAULT_CYCLE_DURATION_S
    cycles: int = DEFAULT_CYCLES
    update_rate_hz: float = DEFAULT_UPDATE_RATE_HZ
    ramp_in_s: float = DEFAULT_RAMP_S
    ramp_out_s: float = DEFAULT_RAMP_S
    hold_at_start_s: float = DEFAULT_HOLD_S
    hold_at_end_s: float = DEFAULT_HOLD_S
    coupling: str = COUPLING_PHASE_SHIFTED
    top_scale: float = 0.5
    phase_offset_deg: float = 90.0
    lissajous_a: float = 1.0
    lissajous_b: float = 2.0
    lissajous_phase_deg: float = 0.0
    raster_ratio: float = 3.0
    seed: int = 0


def validate_request(request: PatternRequest) -> None:
    """Fail loud on obvious config errors instead of silently degrading."""
    if request.pattern not in SUPPORTED_PATTERNS:
        raise ValueError(
            f"Unsupported pattern {request.pattern!r}; choose from {SUPPORTED_PATTERNS}."
        )
    if request.coupling not in SUPPORTED_COUPLINGS:
        raise ValueError(
            f"Unsupported coupling {request.coupling!r}; choose from {SUPPORTED_COUPLINGS}."
        )
    if request.amplitude_cm < 0.0:
        raise ValueError("amplitude_cm must be non-negative.")
    if request.amplitude_cm > max(AMPLITUDE_PRESETS_CM) + 1e-9:
        raise ValueError(
            f"amplitude_cm={request.amplitude_cm} exceeds the conservative demo cap "
            f"({max(AMPLITUDE_PRESETS_CM)} cm). Raise the cap explicitly if needed."
        )
    if request.cycle_duration_s <= 0.0:
        raise ValueError("cycle_duration_s must be positive.")
    if request.cycles < 1:
        raise ValueError("cycles must be >= 1.")
    if request.update_rate_hz <= 0.0:
        raise ValueError("update_rate_hz must be positive.")
    if request.ramp_in_s < 0.0 or request.ramp_out_s < 0.0:
        raise ValueError("ramp times must be non-negative.")
    if request.hold_at_start_s < 0.0 or request.hold_at_end_s < 0.0:
        raise ValueError("hold times must be non-negative.")
    if request.top_scale < 0.0:
        raise ValueError("top_scale must be non-negative.")


def generate_pattern_trajectory(
    request: PatternRequest,
) -> list[TwoSegmentPatternPoint]:
    """Return the full sampled trajectory for a pattern request.

    Sampling is done at ``request.update_rate_hz`` -- this is intentional:
    the demo controller writes the bus at the same rate, and the trajectory
    is timestamped in real seconds so the controller can pace itself with
    ``time.sleep`` between successive points. The trajectory always starts
    and ends at neutral.
    """
    validate_request(request)
    envelope = PatternEnvelope(
        hold_start_s=float(request.hold_at_start_s),
        ramp_in_s=float(request.ramp_in_s),
        pattern_duration_s=float(request.cycle_duration_s) * int(request.cycles),
        ramp_out_s=float(request.ramp_out_s),
        hold_end_s=float(request.hold_at_end_s),
    )
    total = envelope.total_duration_s
    dt = 1.0 / float(request.update_rate_hz)
    # We include t=0 and t=total so the last point is at the end of the
    # hold_end window. The bus write rate is what governs motion smoothness,
    # so we want a sample at every dt boundary plus one final terminator.
    sample_count = int(math.floor(total / dt)) + 1
    omega = 2.0 * math.pi / float(request.cycle_duration_s)
    phase_offset_rad = math.radians(float(request.phase_offset_deg))
    lissajous_phase = math.radians(float(request.lissajous_phase_deg))

    points: list[TwoSegmentPatternPoint] = []
    for index in range(sample_count + 1):
        elapsed_s = min(total, index * dt)
        phase_label, scale = _envelope_amplitude_scale(elapsed_s=elapsed_s, envelope=envelope)
        if phase_label == PHASE_PATTERN:
            theta = omega * (elapsed_s - envelope.hold_start_s - envelope.ramp_in_s)
        elif phase_label == PHASE_RAMP_IN:
            # Use a small fraction of the pattern so the operator sees the
            # shape building up from neutral, not a fixed-direction creep.
            theta = omega * (elapsed_s - envelope.hold_start_s) * 0.5
        elif phase_label == PHASE_RAMP_OUT:
            # Continue the pattern angle smoothly during ramp-out.
            t_into_rampout = elapsed_s - envelope.hold_start_s - envelope.ramp_in_s - envelope.pattern_duration_s
            theta = omega * (envelope.pattern_duration_s + t_into_rampout * 0.5)
        else:
            theta = 0.0

        scaled_amplitude = float(request.amplitude_cm) * float(scale)
        if scaled_amplitude == 0.0:
            bottom_xy = (0.0, 0.0)
        else:
            bottom_xy = _pair_command_for_pattern(
                request.pattern,
                theta=theta,
                amplitude=scaled_amplitude,
                lissajous_a=request.lissajous_a,
                lissajous_b=request.lissajous_b,
                lissajous_phase=lissajous_phase,
                raster_ratio=request.raster_ratio,
                seed=request.seed,
            )

        if scaled_amplitude == 0.0:
            top_xy = (0.0, 0.0)
        else:
            top_xy = _apply_coupling(
                bottom_xy=bottom_xy,
                pattern=request.pattern,
                theta=theta,
                amplitude=scaled_amplitude,
                coupling=request.coupling,
                top_scale=request.top_scale,
                phase_offset_rad=phase_offset_rad,
                lissajous_a=request.lissajous_a,
                lissajous_b=request.lissajous_b,
                lissajous_phase=lissajous_phase,
                raster_ratio=request.raster_ratio,
                seed=request.seed,
            )
            if request.coupling == COUPLING_TOP_ONLY:
                # top_only zeros the bottom; the coupling helper returned
                # the pattern command which we re-target to the top.
                bottom_xy = (0.0, 0.0)

        bottom_tendon = expand_pair_to_tendon_cm(bottom_xy[0], bottom_xy[1])
        top_tendon = expand_pair_to_tendon_cm(top_xy[0], top_xy[1])
        all_8 = (
            bottom_tendon[0],
            bottom_tendon[1],
            bottom_tendon[2],
            bottom_tendon[3],
            top_tendon[0],
            top_tendon[1],
            top_tendon[2],
            top_tendon[3],
        )
        phase_fraction = elapsed_s / total if total > 0 else 0.0
        points.append(
            TwoSegmentPatternPoint(
                sample_index=int(index),
                elapsed_s=float(elapsed_s),
                phase_label=str(phase_label),
                phase_fraction=float(phase_fraction),
                bottom_x_cm=float(bottom_xy[0]),
                bottom_y_cm=float(bottom_xy[1]),
                top_x_cm=float(top_xy[0]),
                top_y_cm=float(top_xy[1]),
                bottom_tendon_cm=bottom_tendon,
                top_tendon_cm=top_tendon,
                all_8_tendon_cm=all_8,
            )
        )
        if elapsed_s >= total:
            break
    return points


def max_pair_amplitude(points: Sequence[TwoSegmentPatternPoint]) -> float:
    """Return the maximum ``hypot(px, py)`` observed across bottom and top.

    Tests use this to verify a generated trajectory never exceeds the
    configured amplitude cap (plus a small floating-point tolerance).
    """
    largest = 0.0
    for point in points:
        for px, py in ((point.bottom_x_cm, point.bottom_y_cm), (point.top_x_cm, point.top_y_cm)):
            r = math.hypot(float(px), float(py))
            if r > largest:
                largest = r
    return float(largest)


def trajectory_starts_and_ends_at_neutral(
    points: Sequence[TwoSegmentPatternPoint],
    *,
    tolerance_cm: float = 1e-9,
) -> bool:
    """Sanity-check guarantee that the trajectory ramps from and back to neutral."""
    if not points:
        return True
    first = points[0]
    last = points[-1]
    return (
        abs(first.bottom_x_cm) <= tolerance_cm
        and abs(first.bottom_y_cm) <= tolerance_cm
        and abs(first.top_x_cm) <= tolerance_cm
        and abs(first.top_y_cm) <= tolerance_cm
        and abs(last.bottom_x_cm) <= tolerance_cm
        and abs(last.bottom_y_cm) <= tolerance_cm
        and abs(last.top_x_cm) <= tolerance_cm
        and abs(last.top_y_cm) <= tolerance_cm
    )

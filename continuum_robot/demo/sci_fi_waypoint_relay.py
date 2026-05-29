"""Waypoint generator + relay scheduler for the 20-second sci-fi spine demo.

This module is **pure** — it does not touch hardware, the servo bus, or the
tracker. It produces:

- a list of 4D ``(bottom_x, bottom_y, top_x, top_y)`` waypoints in tip-target
  cm, bounded by the requested ``amplitude_cm`` per axis;
- a relay schedule (issue time per waypoint in seconds since run start);
- a speed-class classification + advisory warnings for the operator.

The 4D shape lets the bottom and top spines bend independently — the whole
point of the demo is to show that the two-segment spine can take on
non-obvious shapes that a single-segment robot cannot.

Two waypoint sources:

* ``preset_weird``: a fixed normalized list of 9 waypoints chosen to look
  visually distinct on camera (bottom and top intentionally diverge,
  consecutive waypoints are well separated). Scaled by ``amplitude_cm``.
* ``seeded_maximin``: deterministic per ``seed``. Samples 256 random 4D
  candidates within ``±amplitude_cm`` per axis, then greedily picks the
  next waypoint that maximises the minimum distance to the already-picked
  set. Final pass enforces a ``min_waypoint_separation_norm`` between
  consecutive waypoints; if any pair fails, the auto-selector can re-roll
  with the next seed.

``early_switch_fraction`` is the knob that produces the "the spine starts
the next pose before settling the current one" feel, and it is built into
the trajectory itself — it does NOT rely on guessing a ``profile_velocity``
that happens to leave the servo mid-motion. ``build_relay_trajectory``
densely samples an *overlapping smoothstep blend* through the waypoints:
each waypoint-to-waypoint transition is a smoothstep, and the next
transition starts when the current one is only ``early_switch_fraction``
of the way through. At ``early_switch_fraction == 1.0`` the transitions are
back-to-back so the spine settles on every waypoint; below 1.0 they overlap
so the spine rounds the corners and never fully settles — the deterministic
sci-fi flow. The total motion still spans ``video_duration_s`` regardless of
the fraction, and every sample is clamped to ``±amplitude_cm`` so the
amplitude-derived tick budget always bounds the trajectory.

``build_relay_schedule`` still returns one evenly spaced issue time per
waypoint; it is used for the waypoints/preview/CSV artifacts and as a human
record of nominal waypoint pacing. The dense trajectory is what the demo
controller actually streams to the bus.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
import random
from typing import Sequence


# ---------------------------------------------------------------------------
# Public constants
# ---------------------------------------------------------------------------


WAYPOINT_SOURCE_PRESET = "preset_weird"
WAYPOINT_SOURCE_SEEDED = "seeded_maximin"
SUPPORTED_WAYPOINT_SOURCES = (WAYPOINT_SOURCE_PRESET, WAYPOINT_SOURCE_SEEDED)

# Default waypoint count is 9: matches the preset list length, fits a
# 20-second video at ~2.2 s per segment which reads as "smooth slow drift"
# rather than "rapid sequence of poses".
DEFAULT_WAYPOINT_COUNT = 9
DEFAULT_VIDEO_DURATION_S = 20.0
DEFAULT_EARLY_SWITCH_FRACTION = 0.72
DEFAULT_AMPLITUDE_CM = 0.35
DEFAULT_COMMAND_RATE_HZ = 15.0  # dense blend sample + bus write rate
DEFAULT_MIN_WAYPOINT_SEPARATION_NORM = 0.40  # in units of amplitude_cm
DEFAULT_MAX_WAYPOINT_SEPARATION_NORM = 2.20  # √8 ≈ 2.83 is theoretical max
DEFAULT_PROFILE_VELOCITY = 45  # XC330 units = 0.229 rpm/unit → ~10 rpm
DEFAULT_PROFILE_ACCELERATION = 18  # XC330 units = ~3 800 rpm²

# Preset of 9 normalized 4D waypoints. Each entry is
# ``[bottom_x, bottom_y, top_x, top_y]`` in tip-target convention.
# Hand-picked so that:
#   - consecutive waypoints are at least 0.7 norm apart in 4D
#   - both bottom and top components are non-trivial at every step
#   - bottom and top intentionally bend in opposing directions ≥3 times
#     to make the two-segment articulation visually obvious on camera
PRESET_WEIRD_WAYPOINTS_NORM: tuple[tuple[float, float, float, float], ...] = (
    ( 0.75, -0.20, -0.55,  0.70),
    (-0.65,  0.55,  0.80,  0.15),
    ( 0.15,  0.80, -0.75, -0.45),
    (-0.85, -0.25,  0.35, -0.85),
    ( 0.50,  0.65,  0.65, -0.65),
    (-0.20, -0.90, -0.90,  0.20),
    ( 0.90,  0.10, -0.10, -0.90),
    (-0.55,  0.00,  0.85,  0.55),
    ( 0.25, -0.75,  0.75, -0.25),
)


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SciFiWaypoint:
    """One 4D command-space waypoint in tip-target cm."""

    index: int
    bottom_x_cm: float
    bottom_y_cm: float
    top_x_cm: float
    top_y_cm: float

    def as_tuple(self) -> tuple[float, float, float, float]:
        return (
            float(self.bottom_x_cm),
            float(self.bottom_y_cm),
            float(self.top_x_cm),
            float(self.top_y_cm),
        )


@dataclass(frozen=True)
class SciFiRelayScheduleEntry:
    """One scheduled relay write: waypoint + when to issue it."""

    waypoint_index: int
    issue_time_s: float
    waypoint: SciFiWaypoint


@dataclass(frozen=True)
class SciFiRelayConfig:
    """User-facing config for the sci-fi waypoint relay demo.

    All fields have safe defaults so the 20-second preset can be loaded
    from a single ``SciFiRelayConfig()`` call.
    """

    video_duration_s: float = DEFAULT_VIDEO_DURATION_S
    waypoint_count: int = DEFAULT_WAYPOINT_COUNT
    amplitude_cm: float = DEFAULT_AMPLITUDE_CM
    early_switch_fraction: float = DEFAULT_EARLY_SWITCH_FRACTION
    waypoint_source: str = WAYPOINT_SOURCE_PRESET
    seed: int = 0
    auto_select_seed: bool = True
    min_waypoint_separation_norm: float = DEFAULT_MIN_WAYPOINT_SEPARATION_NORM
    max_waypoint_separation_norm: float = DEFAULT_MAX_WAYPOINT_SEPARATION_NORM
    profile_velocity: int = DEFAULT_PROFILE_VELOCITY
    profile_acceleration: int = DEFAULT_PROFILE_ACCELERATION
    command_rate_hz: float = DEFAULT_COMMAND_RATE_HZ


@dataclass(frozen=True)
class SciFiSpeedAdvisory:
    """Speed-class classification + advisory warnings for one config."""

    speed_class: str  # "slow" | "medium" | "aggressive"
    seconds_per_waypoint: float
    estimated_bus_writes: int
    warnings: tuple[str, ...] = ()


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def validate_relay_config(config: SciFiRelayConfig) -> None:
    """Fail loud on obvious configuration errors before any waypoint work."""
    if config.video_duration_s <= 0.0:
        raise ValueError("video_duration_s must be positive.")
    if config.waypoint_count < 2:
        raise ValueError("waypoint_count must be >= 2.")
    if not (0.0 < config.early_switch_fraction <= 1.0):
        raise ValueError("early_switch_fraction must be in (0, 1].")
    if config.amplitude_cm <= 0.0:
        raise ValueError("amplitude_cm must be positive.")
    if config.amplitude_cm > 1.0 + 1e-9:
        raise ValueError(
            f"amplitude_cm={config.amplitude_cm} exceeds the conservative demo cap "
            "(1.0 cm). Lower it or raise the cap deliberately."
        )
    if config.waypoint_source not in SUPPORTED_WAYPOINT_SOURCES:
        raise ValueError(
            f"Unsupported waypoint_source {config.waypoint_source!r}; "
            f"choose from {SUPPORTED_WAYPOINT_SOURCES}."
        )
    if config.command_rate_hz <= 0.0:
        raise ValueError("command_rate_hz must be positive.")
    if config.min_waypoint_separation_norm < 0.0:
        raise ValueError("min_waypoint_separation_norm must be non-negative.")
    if config.max_waypoint_separation_norm < config.min_waypoint_separation_norm:
        raise ValueError(
            "max_waypoint_separation_norm must be >= min_waypoint_separation_norm."
        )


# ---------------------------------------------------------------------------
# Waypoint generators
# ---------------------------------------------------------------------------


def generate_preset_waypoints(*, amplitude_cm: float, waypoint_count: int) -> list[SciFiWaypoint]:
    """Return the first ``waypoint_count`` preset waypoints, scaled by amplitude.

    Repeats the preset if ``waypoint_count`` exceeds the preset length, but
    keeps the visual distinctness of the first cycle by interleaving the
    repeats with a small rotation. For the recommended ``waypoint_count=9``
    the preset is used verbatim (no repeats).
    """
    if amplitude_cm <= 0.0:
        return []
    count = max(1, int(waypoint_count))
    waypoints: list[SciFiWaypoint] = []
    preset = list(PRESET_WEIRD_WAYPOINTS_NORM)
    for index in range(count):
        norm = preset[index % len(preset)]
        # Slightly rotate the bottom-top pair on each repeat so a 12-waypoint
        # request doesn't re-trace the first 9 verbatim. The rotation angle
        # is small enough to keep the visual character intact.
        repeat_pass = index // len(preset)
        rotation_rad = repeat_pass * math.radians(15.0)
        cos_r = math.cos(rotation_rad)
        sin_r = math.sin(rotation_rad)
        bx, by, tx, ty = norm
        bx_rot = bx * cos_r - by * sin_r
        by_rot = bx * sin_r + by * cos_r
        tx_rot = tx * cos_r - ty * sin_r
        ty_rot = tx * sin_r + ty * cos_r
        waypoints.append(
            SciFiWaypoint(
                index=index,
                bottom_x_cm=float(amplitude_cm * bx_rot),
                bottom_y_cm=float(amplitude_cm * by_rot),
                top_x_cm=float(amplitude_cm * tx_rot),
                top_y_cm=float(amplitude_cm * ty_rot),
            )
        )
    return waypoints


def generate_seeded_maximin_waypoints(
    *,
    amplitude_cm: float,
    waypoint_count: int,
    seed: int,
    min_separation_norm: float = DEFAULT_MIN_WAYPOINT_SEPARATION_NORM,
) -> list[SciFiWaypoint]:
    """Greedy maximin (farthest-point) waypoint selection inside ±amplitude_cm per axis.

    Draws ``candidate_count`` random 4D points (bounded by ``amplitude_cm``
    on each axis), greedily picks the point that maximizes the minimum 4D
    distance to the already-picked set, and biases toward points where the
    bottom and top sub-vectors are not aligned (so the two segments
    visibly bend in different ways).

    ``min_separation_norm`` is enforced post-hoc against consecutive
    waypoints; if any pair is too close, the result is still returned but
    the auto-select wrapper can detect the failure via
    :func:`min_consecutive_separation_norm` and pick the next seed.
    """
    if amplitude_cm <= 0.0 or waypoint_count <= 0:
        return []
    rng = random.Random(int(seed) * 31 + 7)
    candidate_count = max(64, int(waypoint_count) * 32)
    candidates: list[tuple[float, float, float, float]] = []
    for _ in range(candidate_count):
        candidates.append(
            (
                rng.uniform(-amplitude_cm, amplitude_cm),
                rng.uniform(-amplitude_cm, amplitude_cm),
                rng.uniform(-amplitude_cm, amplitude_cm),
                rng.uniform(-amplitude_cm, amplitude_cm),
            )
        )

    picked: list[tuple[float, float, float, float]] = []
    # Seed the picked set with the candidate that has the most opposition
    # between bottom and top segments — guarantees the run starts with a
    # visually distinct two-segment shape.
    def opposition_score(pt: tuple[float, float, float, float]) -> float:
        bx, by, tx, ty = pt
        # Negative dot product → opposing directions → higher score.
        return -(bx * tx + by * ty)

    start = max(candidates, key=opposition_score)
    picked.append(start)

    for _ in range(int(waypoint_count) - 1):
        def _farthest_score(pt: tuple[float, float, float, float]) -> float:
            return min(_norm4(pt, q) for q in picked)
        # Bias 70 % maximin distance, 30 % opposition between bottom and top
        # — pure maximin gives a tidy spread; the opposition term keeps the
        # path looking organic instead of geometric.
        def _score(pt: tuple[float, float, float, float]) -> float:
            return 0.7 * _farthest_score(pt) + 0.3 * abs(opposition_score(pt))

        remaining = [pt for pt in candidates if pt not in picked]
        if not remaining:
            break
        picked.append(max(remaining, key=_score))

    waypoints: list[SciFiWaypoint] = []
    for index, (bx, by, tx, ty) in enumerate(picked):
        waypoints.append(
            SciFiWaypoint(
                index=index,
                bottom_x_cm=float(bx),
                bottom_y_cm=float(by),
                top_x_cm=float(tx),
                top_y_cm=float(ty),
            )
        )
    _ = min_separation_norm  # exposed for auto-select wrapper
    return waypoints


def min_consecutive_separation_norm(
    waypoints: Sequence[SciFiWaypoint],
    *,
    amplitude_cm: float,
) -> float:
    """Return the min 4D distance between consecutive waypoints, normalized by amplitude.

    Used by :func:`auto_select_seed` to score how well-spread a seed's
    waypoint set is. Returns 0.0 for empty or single-waypoint lists.
    """
    if len(waypoints) < 2 or amplitude_cm <= 0.0:
        return 0.0
    smallest = float("inf")
    for prev, cur in zip(waypoints, waypoints[1:]):
        d = _norm4(prev.as_tuple(), cur.as_tuple())
        if d < smallest:
            smallest = float(d)
    return float(smallest / amplitude_cm)


def auto_select_seed(
    *,
    base_seed: int,
    amplitude_cm: float,
    waypoint_count: int,
    min_separation_norm: float = DEFAULT_MIN_WAYPOINT_SEPARATION_NORM,
    max_seeds_to_try: int = 64,
) -> tuple[int, list[SciFiWaypoint]]:
    """Try seeds ``base_seed, base_seed+1, …`` and pick the one whose
    consecutive-waypoint min separation is largest.

    Returns ``(chosen_seed, waypoints)``. The first seed that exceeds
    ``min_separation_norm`` wins; if none qualify, returns the seed with
    the largest separation score so the run never fails to produce
    waypoints — just may have one or two slow segments. Bounded to
    ``max_seeds_to_try`` trials.
    """
    best_seed = int(base_seed)
    best_waypoints = generate_seeded_maximin_waypoints(
        amplitude_cm=amplitude_cm,
        waypoint_count=waypoint_count,
        seed=int(base_seed),
        min_separation_norm=min_separation_norm,
    )
    best_score = min_consecutive_separation_norm(best_waypoints, amplitude_cm=amplitude_cm)
    if best_score >= min_separation_norm:
        return best_seed, best_waypoints

    for offset in range(1, max(1, int(max_seeds_to_try))):
        candidate_seed = int(base_seed) + offset
        waypoints = generate_seeded_maximin_waypoints(
            amplitude_cm=amplitude_cm,
            waypoint_count=waypoint_count,
            seed=candidate_seed,
            min_separation_norm=min_separation_norm,
        )
        score = min_consecutive_separation_norm(waypoints, amplitude_cm=amplitude_cm)
        if score > best_score:
            best_seed = candidate_seed
            best_waypoints = waypoints
            best_score = score
        if best_score >= min_separation_norm:
            break
    return best_seed, best_waypoints


def build_waypoints(config: SciFiRelayConfig) -> tuple[list[SciFiWaypoint], int]:
    """Return ``(waypoints, resolved_seed)`` for the given config.

    Dispatches to the configured ``waypoint_source`` and runs the
    auto-select-seed wrapper when ``auto_select_seed`` is true. The
    resolved seed is what the run records into the summary so the operator
    can reproduce the exact waypoint set later.
    """
    validate_relay_config(config)
    if config.waypoint_source == WAYPOINT_SOURCE_PRESET:
        # Preset isn't seed-driven; return the configured seed verbatim.
        return (
            generate_preset_waypoints(
                amplitude_cm=config.amplitude_cm,
                waypoint_count=config.waypoint_count,
            ),
            int(config.seed),
        )
    # seeded_maximin path
    if bool(config.auto_select_seed):
        chosen_seed, waypoints = auto_select_seed(
            base_seed=int(config.seed),
            amplitude_cm=config.amplitude_cm,
            waypoint_count=config.waypoint_count,
            min_separation_norm=float(config.min_waypoint_separation_norm),
        )
        return waypoints, chosen_seed
    waypoints = generate_seeded_maximin_waypoints(
        amplitude_cm=config.amplitude_cm,
        waypoint_count=config.waypoint_count,
        seed=int(config.seed),
        min_separation_norm=float(config.min_waypoint_separation_norm),
    )
    return waypoints, int(config.seed)


# ---------------------------------------------------------------------------
# Relay scheduler + speed advisor
# ---------------------------------------------------------------------------


def build_relay_schedule(
    config: SciFiRelayConfig,
    *,
    waypoints: Sequence[SciFiWaypoint],
) -> list[SciFiRelayScheduleEntry]:
    """Return a sorted list of relay schedule entries.

    Each entry says "issue waypoint i at issue_time_s seconds since run
    start." The total duration of the schedule equals
    ``config.video_duration_s``: ``waypoint_count`` writes evenly spaced
    across the duration, starting at ``t=0``.
    """
    if not waypoints:
        return []
    n = len(waypoints)
    if n == 1:
        return [SciFiRelayScheduleEntry(0, 0.0, waypoints[0])]
    # We want the LAST waypoint commanded at t ≈ video_duration_s so the
    # video covers the full configured duration. Evenly spaced:
    #     t_i = i * (video_duration_s / (n - 1))   for i in 0..n-1
    # That means the final command lands at exactly video_duration_s,
    # which is the cue to start the return-to-neutral.
    interval = float(config.video_duration_s) / float(n - 1)
    return [
        SciFiRelayScheduleEntry(
            waypoint_index=int(i),
            issue_time_s=float(i) * interval,
            waypoint=waypoints[int(i)],
        )
        for i in range(n)
    ]


def speed_advisory(config: SciFiRelayConfig) -> SciFiSpeedAdvisory:
    """Classify the configured speed and surface advisory warnings.

    - ``slow`` (good for video): profile_velocity ≤ 60, amplitude ≤ 0.40 cm,
      seconds_per_waypoint ≥ 1.5.
    - ``medium``: in between.
    - ``aggressive``: profile_velocity ≥ 120, amplitude ≥ 0.60 cm, or
      seconds_per_waypoint ≤ 0.6.
    """
    warnings: list[str] = []
    seconds_per_waypoint = (
        float(config.video_duration_s) / max(1, int(config.waypoint_count))
    )
    # Bus writes: one per relay schedule entry. The command_rate_hz field
    # is bookkeeping/status only — no extra writes happen between waypoints.
    estimated_bus_writes = int(config.waypoint_count)
    velocity = int(config.profile_velocity or 0)
    acceleration = int(config.profile_acceleration or 0)
    amplitude = float(config.amplitude_cm)

    # Speed-class scoring
    score = 0
    score += 1 if velocity > 60 else 0
    score += 2 if velocity > 120 else 0
    score += 1 if acceleration > 30 else 0
    score += 1 if amplitude > 0.40 else 0
    score += 2 if amplitude > 0.60 else 0
    score += 1 if seconds_per_waypoint < 1.5 else 0
    score += 2 if seconds_per_waypoint < 0.6 else 0
    if score <= 1:
        speed_class = "slow"
    elif score <= 3:
        speed_class = "medium"
    else:
        speed_class = "aggressive"

    # Warnings
    if amplitude > 0.50:
        warnings.append(
            f"amplitude_cm={amplitude:.2f} exceeds the 0.50 cm slide-video recommendation; "
            "step down to 0.35 for the first live run."
        )
    if seconds_per_waypoint < 0.6:
        warnings.append(
            f"seconds_per_waypoint={seconds_per_waypoint:.2f} is < 0.6 s — the spine will "
            "barely have time to move between commands; raise video_duration_s or lower "
            "waypoint_count."
        )
    if float(config.command_rate_hz) > 30.0:
        warnings.append(
            f"command_rate_hz={float(config.command_rate_hz):.1f} exceeds the realistic 30 Hz "
            "DYNAMIXEL ceiling. The relay only sends waypoint_count bus writes regardless of "
            "this rate; raising it just spins the bookkeeping loop."
        )
    if velocity > 120:
        warnings.append(
            f"profile_velocity={velocity} is high (>120). The relay's 'almost reaching' effect "
            "needs slow motion; consider 30–60 for a clear sci-fi look."
        )
    if not (0.4 <= config.early_switch_fraction <= 0.9):
        warnings.append(
            f"early_switch_fraction={config.early_switch_fraction:.2f} is outside the "
            "typical 0.4–0.9 range. It controls how much consecutive waypoint transitions "
            "overlap: 1.0 settles on every waypoint (mechanical), very low values smear the "
            "whole path into one blur. 0.6–0.8 reads best as a sci-fi flow on camera."
        )
    return SciFiSpeedAdvisory(
        speed_class=speed_class,
        seconds_per_waypoint=float(seconds_per_waypoint),
        estimated_bus_writes=int(estimated_bus_writes),
        warnings=tuple(warnings),
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _norm4(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
    return math.sqrt(
        (a[0] - b[0]) ** 2
        + (a[1] - b[1]) ** 2
        + (a[2] - b[2]) ** 2
        + (a[3] - b[3]) ** 2
    )


def _smoothstep(t: float) -> float:
    """Standard smoothstep easing 3t² − 2t³, clamped to [0, 1].

    Zero velocity at both ends, so blended transitions accelerate out of and
    decelerate into each waypoint instead of snapping.
    """
    t = max(0.0, min(1.0, float(t)))
    return float(t * t * (3.0 - 2.0 * t))


def relay_blend_positions(
    key_poses: Sequence[tuple[float, float, float, float]],
    *,
    body_duration_s: float,
    early_switch_fraction: float,
    sample_rate_hz: float,
    amplitude_cm: float,
) -> list[tuple[float, float, float, float, float]]:
    """Dense overlapping-smoothstep blend through ``key_poses``.

    Returns ``(elapsed_s, bottom_x, bottom_y, top_x, top_y)`` samples covering
    ``[0, body_duration_s]`` at ``sample_rate_hz``. The position is the
    additive sum of per-transition smoothsteps:

        pos(t) = K₀ + Σⱼ (Kⱼ₊₁ − Kⱼ)·smoothstep((t − j·stride) / T)

    where ``T`` is the per-transition duration and ``stride = esf·T`` is how
    long after one transition starts the next one begins. ``esf`` (the
    ``early_switch_fraction``) is therefore *built into* the path:

    * ``esf == 1.0`` → ``stride == T`` → transitions are back-to-back, the
      spine reaches every key pose exactly (mechanical relay).
    * ``esf < 1.0``  → transitions overlap → the spine rounds the corners and
      never fully settles (the sci-fi flow). Lower ``esf`` → more overlap.

    ``T`` is chosen so the last transition always ends at ``body_duration_s``
    regardless of ``esf``, so the demo length is stable. Every sample is
    clamped to ``±amplitude_cm`` so the amplitude-derived tick budget bounds
    the trajectory even where overlapping transitions would otherwise add up.
    """
    n_poses = len(key_poses)
    transitions = max(1, n_poses - 1)
    esf = max(0.01, min(1.0, float(early_switch_fraction)))
    body = max(0.0, float(body_duration_s))
    # last transition ends at (M−1)·stride + T = body, with stride = esf·T
    #   ⇒ T = body / (1 + (M−1)·esf)
    t_trans = body / (1.0 + (transitions - 1) * esf) if body > 0.0 else 0.0
    stride = esf * t_trans
    dt = 1.0 / max(0.5, float(sample_rate_hz))
    sample_count = max(1, int(round(body / dt))) if body > 0.0 else 1
    amp = abs(float(amplitude_cm))

    samples: list[tuple[float, float, float, float, float]] = []
    for k in range(sample_count + 1):
        t = min(body, k * dt)
        pos = [float(key_poses[0][a]) for a in range(4)]
        for j in range(transitions):
            s = _smoothstep((t - j * stride) / t_trans) if t_trans > 0.0 else 1.0
            for a in range(4):
                pos[a] += (float(key_poses[j + 1][a]) - float(key_poses[j][a])) * s
        if amp > 0.0:
            pos = [max(-amp, min(amp, v)) for v in pos]
        samples.append((t, pos[0], pos[1], pos[2], pos[3]))
        if t >= body:
            break
    return samples


# ---------------------------------------------------------------------------
# Trajectory builder — emits sparse TwoSegmentPatternPoint values for the demo
# ---------------------------------------------------------------------------


def build_relay_trajectory(
    config: SciFiRelayConfig,
    *,
    hold_at_start_s: float = 1.0,
    hold_at_end_s: float = 1.5,
):
    """Build the dense blended trajectory + schedule for the demo.

    Returns ``(points, waypoints, schedule, resolved_seed)``:

    * ``points`` is a ``list[TwoSegmentPatternPoint]`` sampled at
      ``config.command_rate_hz`` along an overlapping-smoothstep blend
      through ``neutral → waypoint₀ → … → waypointₙ₋₁ → neutral`` (see
      :func:`relay_blend_positions`). The blend is bracketed by a neutral
      hold at ``t=0`` (length ``hold_at_start_s``) and a neutral hold at the
      end (length ``hold_at_end_s``). ``early_switch_fraction`` is baked into
      the blend, so the "almost reaching / changing its mind" motion is
      deterministic instead of depending on a lucky ``profile_velocity``.
      The demo's execute() loop sleeps the difference between consecutive
      ``elapsed_s`` values, so streaming these points reproduces the timing.
    * ``waypoints`` is the resolved waypoint list (preset or seeded).
    * ``schedule`` is the nominal relay schedule (one evenly spaced issue
      time per waypoint), kept for the waypoints/preview/CSV artifacts.
    * ``resolved_seed`` is the seed actually used after auto_select_seed
      runs (always equal to ``config.seed`` for the preset source).

    The function is pure: no servo / bus access. Import deferred so this
    module stays dependency-free at import time.
    """
    from continuum_robot.demo.two_segment_motion_patterns import (
        TwoSegmentPatternPoint,
        expand_pair_to_tendon_cm,
        PHASE_HOLD_START,
        PHASE_HOLD_END,
        PHASE_PATTERN,
    )

    waypoints, resolved_seed = build_waypoints(config)
    schedule = build_relay_schedule(config, waypoints=waypoints)

    # Key poses: neutral → each waypoint → neutral. Bracketing the waypoints
    # with neutral makes the blend start and end at neutral (so the hold
    # windows are pure dwell) and gives a smooth ease out of / back into
    # neutral without a separate ramp stage.
    key_poses: list[tuple[float, float, float, float]] = [(0.0, 0.0, 0.0, 0.0)]
    key_poses.extend(
        (float(w.bottom_x_cm), float(w.bottom_y_cm), float(w.top_x_cm), float(w.top_y_cm))
        for w in waypoints
    )
    key_poses.append((0.0, 0.0, 0.0, 0.0))

    blend = relay_blend_positions(
        key_poses,
        body_duration_s=float(config.video_duration_s),
        early_switch_fraction=float(config.early_switch_fraction),
        sample_rate_hz=float(config.command_rate_hz),
        amplitude_cm=float(config.amplitude_cm),
    )

    total_duration_s = float(hold_at_start_s) + float(config.video_duration_s) + float(hold_at_end_s)

    def _pattern_point(
        *,
        sample_index: int,
        elapsed_s: float,
        phase_label: str,
        bottom_xy: tuple[float, float],
        top_xy: tuple[float, float],
    ) -> "TwoSegmentPatternPoint":
        bottom_tendon = expand_pair_to_tendon_cm(bottom_xy[0], bottom_xy[1])
        top_tendon = expand_pair_to_tendon_cm(top_xy[0], top_xy[1])
        all_8 = (
            bottom_tendon[0], bottom_tendon[1], bottom_tendon[2], bottom_tendon[3],
            top_tendon[0], top_tendon[1], top_tendon[2], top_tendon[3],
        )
        phase_fraction = (
            float(elapsed_s) / float(total_duration_s)
            if total_duration_s > 0.0
            else 0.0
        )
        return TwoSegmentPatternPoint(
            sample_index=int(sample_index),
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

    points: list = []
    sample_index = 0

    # 1) Hold at neutral before the blend starts.
    if hold_at_start_s > 0.0:
        points.append(
            _pattern_point(
                sample_index=sample_index,
                elapsed_s=0.0,
                phase_label=PHASE_HOLD_START,
                bottom_xy=(0.0, 0.0),
                top_xy=(0.0, 0.0),
            )
        )
        sample_index += 1

    # 2) Dense blend body, shifted by hold_at_start_s.
    for (t, bx, by, tx, ty) in blend:
        points.append(
            _pattern_point(
                sample_index=sample_index,
                elapsed_s=float(hold_at_start_s) + float(t),
                phase_label=PHASE_PATTERN,
                bottom_xy=(bx, by),
                top_xy=(tx, ty),
            )
        )
        sample_index += 1

    # 3) Final neutral hold so the spine settles back to startup.
    points.append(
        _pattern_point(
            sample_index=sample_index,
            elapsed_s=total_duration_s,
            phase_label=PHASE_HOLD_END,
            bottom_xy=(0.0, 0.0),
            top_xy=(0.0, 0.0),
        )
    )
    return points, waypoints, schedule, int(resolved_seed)

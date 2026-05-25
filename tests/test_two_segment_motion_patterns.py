"""Pattern generator tests for the two-segment slow motion demo.

These tests are pure: they do not touch hardware, the servo bus, or the
tracker. The pattern module is the single source of truth for the
trajectories the demo controller streams to the bus, so the safety
guarantees (amplitude cap, ramp envelope, coupling correctness) live here.
"""

from __future__ import annotations

import math

import pytest

from continuum_robot.demo.two_segment_motion_patterns import (
    AMPLITUDE_PRESETS_CM,
    COUPLING_BOTTOM_ONLY,
    COUPLING_OPPOSITE,
    COUPLING_PHASE_SHIFTED,
    COUPLING_SAME,
    COUPLING_TOP_ONLY,
    COUPLING_TOP_SCALED,
    DEFAULT_AMPLITUDE_CM,
    DEFAULT_CYCLE_DURATION_S,
    PATTERN_CIRCLE,
    PATTERN_CLOVER,
    PATTERN_FIGURE8,
    PATTERN_LISSAJOUS,
    PATTERN_OVAL,
    PATTERN_RASTER,
    PATTERN_SWEEP_X,
    PATTERN_SWEEP_Y,
    PHASE_HOLD_END,
    PHASE_HOLD_START,
    PHASE_PATTERN,
    PHASE_RAMP_IN,
    PHASE_RAMP_OUT,
    SUPPORTED_PATTERNS,
    PatternRequest,
    expand_pair_to_tendon_cm,
    generate_pattern_trajectory,
    max_pair_amplitude,
    trajectory_starts_and_ends_at_neutral,
)


def _request(**overrides) -> PatternRequest:
    base = dict(
        pattern=PATTERN_FIGURE8,
        amplitude_cm=DEFAULT_AMPLITUDE_CM,
        cycle_duration_s=DEFAULT_CYCLE_DURATION_S,
        cycles=2,
        update_rate_hz=5.0,
        ramp_in_s=2.0,
        ramp_out_s=2.0,
        hold_at_start_s=0.5,
        hold_at_end_s=0.5,
        coupling=COUPLING_PHASE_SHIFTED,
        top_scale=0.5,
        phase_offset_deg=90.0,
        lissajous_a=1.0,
        lissajous_b=2.0,
        lissajous_phase_deg=0.0,
        raster_ratio=3.0,
        seed=0,
    )
    base.update(overrides)
    return PatternRequest(**base)


class TestTendonConvention:
    """The tendon expansion must match pair_axis_convention exactly."""

    def test_expand_pair_uses_minus_minus_plus_plus_convention(self) -> None:
        assert expand_pair_to_tendon_cm(0.25, -0.10) == (-0.25, 0.10, 0.25, -0.10)
        assert expand_pair_to_tendon_cm(0.0, 0.0) == (0.0, 0.0, 0.0, 0.0)

    def test_expand_pair_matches_pair_axis_convention_module(self) -> None:
        """Cross-check against the upstream single-source-of-truth helper."""
        from continuum_robot.experiments.pair_axis_convention import (
            expand_pair_command_to_cable_deltas,
        )

        for px, py in [(0.1, 0.2), (-0.3, 0.4), (0.0, 0.5), (1.0, -1.0)]:
            assert list(expand_pair_to_tendon_cm(px, py)) == expand_pair_command_to_cable_deltas([px, py])


class TestPatternBoundsAndFiniteness:
    """Every supported pattern must produce finite, bounded commands."""

    @pytest.mark.parametrize("pattern", SUPPORTED_PATTERNS)
    def test_pattern_produces_finite_commands(self, pattern: str) -> None:
        points = generate_pattern_trajectory(_request(pattern=pattern, cycle_duration_s=4.0, cycles=1))
        assert len(points) > 0
        for point in points:
            for value in (point.bottom_x_cm, point.bottom_y_cm, point.top_x_cm, point.top_y_cm):
                assert math.isfinite(value)
            for value in point.bottom_tendon_cm + point.top_tendon_cm + point.all_8_tendon_cm:
                assert math.isfinite(value)

    @pytest.mark.parametrize("pattern", SUPPORTED_PATTERNS)
    def test_amplitude_cap_respected(self, pattern: str) -> None:
        amplitude = 0.25
        # Clover hits exactly amplitude on its lobes; the figure-8 / lissajous
        # also touch amplitude. Allow a tiny floating-point slop.
        points = generate_pattern_trajectory(
            _request(pattern=pattern, amplitude_cm=amplitude, cycle_duration_s=4.0, cycles=1)
        )
        observed = max_pair_amplitude(points)
        # max_pair_amplitude returns hypot, which can exceed amplitude only
        # when both x and y are simultaneously at amplitude (e.g. raster).
        # The cap is amplitude * sqrt(2) + tolerance.
        assert observed <= amplitude * math.sqrt(2) + 1e-9, (
            f"pattern {pattern!r} exceeded amplitude cap: {observed} > {amplitude * math.sqrt(2)}"
        )


class TestRampEnvelope:
    """The ramp envelope must start and end at neutral."""

    def test_trajectory_starts_and_ends_at_neutral_for_every_pattern(self) -> None:
        for pattern in SUPPORTED_PATTERNS:
            points = generate_pattern_trajectory(
                _request(pattern=pattern, cycle_duration_s=4.0, cycles=1)
            )
            assert trajectory_starts_and_ends_at_neutral(points), (
                f"pattern {pattern!r} did not start/end at neutral"
            )

    def test_first_and_last_phase_labels_are_holds(self) -> None:
        points = generate_pattern_trajectory(_request(cycle_duration_s=2.0, cycles=1))
        assert points[0].phase_label == PHASE_HOLD_START
        assert points[-1].phase_label == PHASE_HOLD_END

    def test_full_phase_progression(self) -> None:
        points = generate_pattern_trajectory(_request(cycle_duration_s=2.0, cycles=1))
        labels = [p.phase_label for p in points]
        # The label set must include all five phases for the full envelope.
        assert PHASE_HOLD_START in labels
        assert PHASE_RAMP_IN in labels
        assert PHASE_PATTERN in labels
        assert PHASE_RAMP_OUT in labels
        assert PHASE_HOLD_END in labels
        # And the labels must appear in the right order.
        first_index_of = {label: labels.index(label) for label in {PHASE_HOLD_START, PHASE_RAMP_IN, PHASE_PATTERN, PHASE_RAMP_OUT, PHASE_HOLD_END}}
        assert first_index_of[PHASE_HOLD_START] < first_index_of[PHASE_RAMP_IN]
        assert first_index_of[PHASE_RAMP_IN] < first_index_of[PHASE_PATTERN]
        assert first_index_of[PHASE_PATTERN] < first_index_of[PHASE_RAMP_OUT]
        assert first_index_of[PHASE_RAMP_OUT] < first_index_of[PHASE_HOLD_END]


class TestCouplingModes:
    """Each coupling mode must produce the documented bottom/top relationship."""

    def _peak_pattern_point(self, points):
        # Pick a point in the middle of the pattern body to avoid ramp scaling.
        body = [p for p in points if p.phase_label == PHASE_PATTERN]
        assert body, "pattern body should contain points"
        return body[len(body) // 2]

    def test_same_direction_top_equals_bottom(self) -> None:
        points = generate_pattern_trajectory(_request(coupling=COUPLING_SAME, cycle_duration_s=4.0, cycles=1))
        for point in points:
            if point.phase_label != PHASE_PATTERN:
                continue
            assert math.isclose(point.top_x_cm, point.bottom_x_cm, abs_tol=1e-9)
            assert math.isclose(point.top_y_cm, point.bottom_y_cm, abs_tol=1e-9)

    def test_opposite_direction_top_negates_bottom(self) -> None:
        points = generate_pattern_trajectory(_request(coupling=COUPLING_OPPOSITE, cycle_duration_s=4.0, cycles=1))
        for point in points:
            if point.phase_label != PHASE_PATTERN:
                continue
            assert math.isclose(point.top_x_cm, -point.bottom_x_cm, abs_tol=1e-9)
            assert math.isclose(point.top_y_cm, -point.bottom_y_cm, abs_tol=1e-9)

    def test_top_scaled_applies_scale_factor(self) -> None:
        scale = 0.25
        points = generate_pattern_trajectory(
            _request(coupling=COUPLING_TOP_SCALED, top_scale=scale, cycle_duration_s=4.0, cycles=1)
        )
        peak = self._peak_pattern_point(points)
        assert math.isclose(peak.top_x_cm, peak.bottom_x_cm * scale, abs_tol=1e-9)
        assert math.isclose(peak.top_y_cm, peak.bottom_y_cm * scale, abs_tol=1e-9)

    def test_bottom_only_zeros_top(self) -> None:
        points = generate_pattern_trajectory(_request(coupling=COUPLING_BOTTOM_ONLY, cycle_duration_s=4.0, cycles=1))
        for point in points:
            assert point.top_x_cm == 0.0
            assert point.top_y_cm == 0.0

    def test_top_only_zeros_bottom_and_runs_pattern_on_top(self) -> None:
        points = generate_pattern_trajectory(_request(coupling=COUPLING_TOP_ONLY, cycle_duration_s=4.0, cycles=1))
        # Bottom must be zero everywhere, including during ramp / hold.
        for point in points:
            assert point.bottom_x_cm == 0.0
            assert point.bottom_y_cm == 0.0
        # Top must be non-zero somewhere inside the pattern body.
        body = [p for p in points if p.phase_label == PHASE_PATTERN]
        assert any(p.top_x_cm != 0.0 or p.top_y_cm != 0.0 for p in body)

    def test_phase_shifted_top_differs_from_bottom(self) -> None:
        """phase_shifted with 90 deg offset must produce a clearly different
        bottom-vs-top sample at the middle of the pattern body."""
        points = generate_pattern_trajectory(
            _request(
                coupling=COUPLING_PHASE_SHIFTED,
                phase_offset_deg=90.0,
                pattern=PATTERN_CIRCLE,
                cycle_duration_s=4.0,
                cycles=1,
            )
        )
        peak = self._peak_pattern_point(points)
        # Circle at theta and theta+90 deg sit on perpendicular axes, so
        # the bottom and top must NOT be equal under phase_shifted.
        assert not (
            math.isclose(peak.top_x_cm, peak.bottom_x_cm, abs_tol=1e-6)
            and math.isclose(peak.top_y_cm, peak.bottom_y_cm, abs_tol=1e-6)
        )


class TestSegmentIndexedAll8Order:
    """The all-8 tendon vector emitted by the generator is segment-indexed
    (bottom 4 first, top 4 second). The controller resolves physical
    bottom/top via the runtime operating context."""

    def test_first_four_match_bottom_tendon(self) -> None:
        points = generate_pattern_trajectory(_request(coupling=COUPLING_BOTTOM_ONLY, cycle_duration_s=2.0, cycles=1))
        for point in points:
            assert point.all_8_tendon_cm[:4] == point.bottom_tendon_cm

    def test_last_four_match_top_tendon(self) -> None:
        points = generate_pattern_trajectory(_request(coupling=COUPLING_TOP_ONLY, cycle_duration_s=2.0, cycles=1))
        for point in points:
            assert point.all_8_tendon_cm[4:] == point.top_tendon_cm


class TestReproducibility:
    def test_same_request_produces_same_trajectory(self) -> None:
        a = generate_pattern_trajectory(_request())
        b = generate_pattern_trajectory(_request())
        assert len(a) == len(b)
        for point_a, point_b in zip(a, b):
            assert point_a.bottom_tendon_cm == point_b.bottom_tendon_cm
            assert point_a.top_tendon_cm == point_b.top_tendon_cm


class TestValidation:
    """Bad configs must fail loud, not silently degrade."""

    def test_unsupported_pattern_raises(self) -> None:
        with pytest.raises(ValueError, match="Unsupported pattern"):
            generate_pattern_trajectory(_request(pattern="spiral_of_doom"))

    def test_unsupported_coupling_raises(self) -> None:
        with pytest.raises(ValueError, match="Unsupported coupling"):
            generate_pattern_trajectory(_request(coupling="random_top"))

    def test_negative_amplitude_raises(self) -> None:
        with pytest.raises(ValueError, match="amplitude_cm"):
            generate_pattern_trajectory(_request(amplitude_cm=-0.1))

    def test_amplitude_above_cap_raises(self) -> None:
        with pytest.raises(ValueError, match="cap"):
            generate_pattern_trajectory(_request(amplitude_cm=2.5))

    def test_zero_update_rate_raises(self) -> None:
        with pytest.raises(ValueError, match="update_rate_hz"):
            generate_pattern_trajectory(_request(update_rate_hz=0.0))

    def test_zero_cycles_raises(self) -> None:
        with pytest.raises(ValueError, match="cycles"):
            generate_pattern_trajectory(_request(cycles=0))


class TestAmplitudePresetsExposed:
    def test_default_amplitude_is_inside_presets(self) -> None:
        assert DEFAULT_AMPLITUDE_CM in AMPLITUDE_PRESETS_CM

    def test_presets_sorted_increasing(self) -> None:
        assert list(AMPLITUDE_PRESETS_CM) == sorted(AMPLITUDE_PRESETS_CM)

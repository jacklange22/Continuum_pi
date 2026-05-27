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
    PATTERN_CINEMATIC_DRIFT,
    PATTERN_CIRCLE,
    PATTERN_CLOVER,
    PATTERN_CREEPY_SQUID,
    PATTERN_FIGURE8,
    PATTERN_LISSAJOUS,
    PATTERN_OVAL,
    PATTERN_RASTER,
    PATTERN_SCI_FI_WAYPOINT_RELAY,
    PATTERN_SLOW_DRIFT,
    PATTERN_SWEEP_X,
    PATTERN_SWEEP_Y,
    PATTERN_WAYPOINT_DRIFT,
    WAYPOINT_DRIFT_NUM_WAYPOINTS,
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

# The sci-fi relay pattern is registered for GUI discoverability but it can
# only be built through ``build_relay_trajectory`` (it needs
# ``waypoint_count`` / ``early_switch_fraction`` / ``waypoint_source`` which
# do not fit on PatternRequest). Tests that loop ``SUPPORTED_PATTERNS``
# through ``generate_pattern_trajectory`` must skip it.
TRAJECTORY_PATTERNS = [p for p in SUPPORTED_PATTERNS if p != PATTERN_SCI_FI_WAYPOINT_RELAY]


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

    @pytest.mark.parametrize("pattern", TRAJECTORY_PATTERNS)
    def test_pattern_produces_finite_commands(self, pattern: str) -> None:
        points = generate_pattern_trajectory(_request(pattern=pattern, cycle_duration_s=4.0, cycles=1))
        assert len(points) > 0
        for point in points:
            for value in (point.bottom_x_cm, point.bottom_y_cm, point.top_x_cm, point.top_y_cm):
                assert math.isfinite(value)
            for value in point.bottom_tendon_cm + point.top_tendon_cm + point.all_8_tendon_cm:
                assert math.isfinite(value)

    @pytest.mark.parametrize("pattern", TRAJECTORY_PATTERNS)
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
        for pattern in TRAJECTORY_PATTERNS:
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


class TestCreepySquidAndSlowDriftPatterns:
    """Coverage for the organic ``creepy_squid`` and ``slow_drift`` patterns.

    Both patterns are designed to look aperiodic / alive while remaining
    strictly amplitude-bounded. Tests here pin:
      - per-axis amplitude is bounded to ``amplitude_cm`` (the soft tick
        cap downstream depends on this);
      - both patterns are registered in ``SUPPORTED_PATTERNS`` so the GUI
        dropdown picks them up automatically;
      - trajectories ramp from and back to neutral (safety contract);
      - the two patterns produce visibly distinct shapes (no copy-paste);
      - ``slow_drift`` is reproducible for the same seed but does change
        when the seed changes (so re-running can refresh the wandering).
    """

    def test_both_patterns_are_registered(self) -> None:
        assert PATTERN_CREEPY_SQUID in SUPPORTED_PATTERNS
        assert PATTERN_SLOW_DRIFT in SUPPORTED_PATTERNS

    def test_creepy_squid_respects_0_5_cm_per_axis_bound(self) -> None:
        traj = generate_pattern_trajectory(
            _request(
                pattern=PATTERN_CREEPY_SQUID,
                amplitude_cm=0.50,
                cycle_duration_s=25.0,
                cycles=1,
                update_rate_hz=10.0,
            )
        )
        for point in traj:
            assert abs(point.bottom_x_cm) <= 0.50 + 1e-9, "creepy_squid x exceeded 0.5 cm bound"
            assert abs(point.bottom_y_cm) <= 0.50 + 1e-9, "creepy_squid y exceeded 0.5 cm bound"
            assert abs(point.top_x_cm) <= 0.50 + 1e-9
            assert abs(point.top_y_cm) <= 0.50 + 1e-9

    def test_slow_drift_respects_0_5_cm_per_axis_bound(self) -> None:
        traj = generate_pattern_trajectory(
            _request(
                pattern=PATTERN_SLOW_DRIFT,
                amplitude_cm=0.50,
                cycle_duration_s=25.0,
                cycles=1,
                update_rate_hz=10.0,
            )
        )
        for point in traj:
            assert abs(point.bottom_x_cm) <= 0.50 + 1e-9, "slow_drift x exceeded 0.5 cm bound"
            assert abs(point.bottom_y_cm) <= 0.50 + 1e-9, "slow_drift y exceeded 0.5 cm bound"
            assert abs(point.top_x_cm) <= 0.50 + 1e-9
            assert abs(point.top_y_cm) <= 0.50 + 1e-9

    def test_both_patterns_start_and_end_at_neutral(self) -> None:
        for pattern in (PATTERN_CREEPY_SQUID, PATTERN_SLOW_DRIFT):
            traj = generate_pattern_trajectory(
                _request(
                    pattern=pattern,
                    amplitude_cm=0.50,
                    cycle_duration_s=25.0,
                    cycles=1,
                    update_rate_hz=10.0,
                )
            )
            assert trajectory_starts_and_ends_at_neutral(traj), (
                f"{pattern} did not start/end at neutral — ramp envelope broken"
            )

    def test_creepy_squid_produces_aperiodic_motion(self) -> None:
        # Aperiodicity proxy: across one cycle, the visited (px, py) points
        # should be spread out enough that the convex hull (here approximated
        # by max distance from origin) is non-trivial AND no single dominant
        # frequency dictates the shape. The mechanical patterns close back
        # on themselves; the squid does not.
        traj = generate_pattern_trajectory(
            _request(
                pattern=PATTERN_CREEPY_SQUID,
                amplitude_cm=0.50,
                cycle_duration_s=30.0,
                cycles=1,
                update_rate_hz=20.0,
                ramp_in_s=0.0,
                ramp_out_s=0.0,
                hold_at_start_s=0.0,
                hold_at_end_s=0.0,
            )
        )
        midpoints = [(p.bottom_x_cm, p.bottom_y_cm) for p in traj]
        # The midpoint of the trajectory should NOT match the start exactly
        # (a perfectly periodic 1-cycle pattern would have midpoint = π phase
        # offset of the start, but typically wouldn't be close to start).
        first = midpoints[1]  # skip the leading neutral sample
        mid = midpoints[len(midpoints) // 2]
        # Distance between first-pattern-sample and mid-pattern-sample
        # should be a meaningful fraction of amplitude — if it were close to
        # zero the pattern would be barely moving / repeating itself.
        dx = mid[0] - first[0]
        dy = mid[1] - first[1]
        assert math.hypot(dx, dy) > 0.05, (
            "creepy_squid mid-cycle ≈ start: pattern is too periodic / not weird enough"
        )

    def test_slow_drift_is_reproducible_for_same_seed(self) -> None:
        req_kwargs = dict(
            pattern=PATTERN_SLOW_DRIFT,
            amplitude_cm=0.50,
            cycle_duration_s=25.0,
            cycles=1,
            update_rate_hz=10.0,
            ramp_in_s=0.0,
            ramp_out_s=0.0,
            hold_at_start_s=0.0,
            hold_at_end_s=0.0,
            seed=42,
        )
        traj_a = generate_pattern_trajectory(_request(**req_kwargs))
        traj_b = generate_pattern_trajectory(_request(**req_kwargs))
        assert len(traj_a) == len(traj_b)
        for a, b in zip(traj_a, traj_b):
            assert a.bottom_x_cm == pytest.approx(b.bottom_x_cm)
            assert a.bottom_y_cm == pytest.approx(b.bottom_y_cm)

    def test_slow_drift_changes_when_seed_changes(self) -> None:
        common = dict(
            pattern=PATTERN_SLOW_DRIFT,
            amplitude_cm=0.50,
            cycle_duration_s=25.0,
            cycles=1,
            update_rate_hz=10.0,
            ramp_in_s=0.0,
            ramp_out_s=0.0,
            hold_at_start_s=0.0,
            hold_at_end_s=0.0,
        )
        traj_seed_0 = generate_pattern_trajectory(_request(**common, seed=0))
        traj_seed_99 = generate_pattern_trajectory(_request(**common, seed=99))
        # Different seeds should give different drift trajectories — pick the
        # midpoint as a single representative sample.
        mid = len(traj_seed_0) // 2
        delta_x = abs(traj_seed_0[mid].bottom_x_cm - traj_seed_99[mid].bottom_x_cm)
        delta_y = abs(traj_seed_0[mid].bottom_y_cm - traj_seed_99[mid].bottom_y_cm)
        assert delta_x + delta_y > 0.01, (
            "slow_drift midpoint identical across seeds 0 and 99 — seed not "
            "actually steering the random phases"
        )

    def test_creepy_squid_and_slow_drift_produce_distinct_motion(self) -> None:
        # The two patterns should not accidentally produce the same trajectory
        # — they're supposed to feel visibly different (one is jittery /
        # tentacle-like, the other slow drift).
        common = dict(
            amplitude_cm=0.50,
            cycle_duration_s=25.0,
            cycles=1,
            update_rate_hz=10.0,
            ramp_in_s=0.0,
            ramp_out_s=0.0,
            hold_at_start_s=0.0,
            hold_at_end_s=0.0,
            seed=0,
        )
        squid = generate_pattern_trajectory(_request(pattern=PATTERN_CREEPY_SQUID, **common))
        drift = generate_pattern_trajectory(_request(pattern=PATTERN_SLOW_DRIFT, **common))
        diffs = [
            math.hypot(s.bottom_x_cm - d.bottom_x_cm, s.bottom_y_cm - d.bottom_y_cm)
            for s, d in zip(squid, drift)
        ]
        # Average per-sample distance should be meaningful — not the same shape.
        assert sum(diffs) / max(1, len(diffs)) > 0.05

    def test_cinematic_drift_is_registered(self) -> None:
        assert PATTERN_CINEMATIC_DRIFT in SUPPORTED_PATTERNS

    def test_cinematic_drift_respects_per_axis_amplitude_bound(self) -> None:
        traj = generate_pattern_trajectory(
            _request(
                pattern=PATTERN_CINEMATIC_DRIFT,
                amplitude_cm=0.50,
                cycle_duration_s=60.0,
                cycles=1,
                update_rate_hz=30.0,
            )
        )
        for point in traj:
            assert abs(point.bottom_x_cm) <= 0.50 + 1e-9
            assert abs(point.bottom_y_cm) <= 0.50 + 1e-9
            assert abs(point.top_x_cm) <= 0.50 + 1e-9
            assert abs(point.top_y_cm) <= 0.50 + 1e-9

    def test_cinematic_drift_starts_and_ends_at_neutral(self) -> None:
        traj = generate_pattern_trajectory(
            _request(
                pattern=PATTERN_CINEMATIC_DRIFT,
                amplitude_cm=0.50,
                cycle_duration_s=60.0,
                cycles=1,
                update_rate_hz=30.0,
            )
        )
        assert trajectory_starts_and_ends_at_neutral(traj)

    def test_cinematic_drift_is_smooth_at_max_refresh_rate(self) -> None:
        # The whole point of cinematic_drift is that it looks smooth on
        # screen. At 30 Hz with the default 0.5 cm amplitude, the per-tick
        # motion WITHIN the pattern body should be well under 1 mm/frame
        # so the recorded video shows fluid sweeps instead of visible
        # step changes. (The ramp-in / ramp-out boundary points are
        # excluded — those gracefully connect neutral to the pattern body
        # and the demo's normal ramps hide the boundary.)
        traj = generate_pattern_trajectory(
            _request(
                pattern=PATTERN_CINEMATIC_DRIFT,
                amplitude_cm=0.50,
                cycle_duration_s=60.0,
                cycles=1,
                update_rate_hz=30.0,
                ramp_in_s=3.0,
                ramp_out_s=3.0,
                hold_at_start_s=1.0,
                hold_at_end_s=1.0,
            )
        )
        max_step_cm = 0.0
        previous = traj[0]
        for cur in traj[1:]:
            step = math.hypot(
                cur.bottom_x_cm - previous.bottom_x_cm,
                cur.bottom_y_cm - previous.bottom_y_cm,
            )
            if step > max_step_cm:
                max_step_cm = step
            previous = cur
        # 0.5 cm * 2.718 (e, largest freq mult) * 2π/60 * (1/30) ≈ 0.047 cm
        # at the peak of the high-frequency component. Cap at 0.10 cm
        # (1 mm/frame) which is well below the visible-jerk threshold
        # for a 30 Hz screen recording.
        assert max_step_cm < 0.10, (
            f"cinematic_drift stepped {max_step_cm*10:.3f} mm/frame at 30 Hz — "
            "would look jerky on a slide video"
        )

    def test_cinematic_drift_covers_meaningful_workspace_area(self) -> None:
        # The pattern should actually sweep across the workspace, not just
        # wobble in one corner. Count the unique 5 mm² bins the trajectory
        # visits over one cycle and require at least 20 distinct cells.
        traj = generate_pattern_trajectory(
            _request(
                pattern=PATTERN_CINEMATIC_DRIFT,
                amplitude_cm=0.50,
                cycle_duration_s=60.0,
                cycles=1,
                update_rate_hz=30.0,
                ramp_in_s=0.0,
                ramp_out_s=0.0,
                hold_at_start_s=0.0,
                hold_at_end_s=0.0,
            )
        )
        bins = {
            (round(p.bottom_x_cm * 20), round(p.bottom_y_cm * 20))
            for p in traj
        }
        assert len(bins) >= 20, (
            f"cinematic_drift visited only {len(bins)} 5 mm² bins; "
            "expected ≥20 — pattern is not actually exploring the workspace"
        )

    def test_cinematic_drift_does_not_repeat_within_a_single_cycle(self) -> None:
        # Irrational-ratio Lissajous → quasi-periodic. The trajectory
        # should NOT close on itself within one cycle. We compare the
        # one-third-cycle midpoint to the start; if they were nearly equal,
        # the curve would have closed early and the camera would see a
        # visible loop.
        traj = generate_pattern_trajectory(
            _request(
                pattern=PATTERN_CINEMATIC_DRIFT,
                amplitude_cm=0.50,
                cycle_duration_s=60.0,
                cycles=1,
                update_rate_hz=30.0,
                ramp_in_s=0.0,
                ramp_out_s=0.0,
                hold_at_start_s=0.0,
                hold_at_end_s=0.0,
            )
        )
        first = traj[1]
        third = traj[len(traj) // 3]
        gap = math.hypot(third.bottom_x_cm - first.bottom_x_cm, third.bottom_y_cm - first.bottom_y_cm)
        assert gap > 0.05, (
            "cinematic_drift third-cycle position ≈ start: pattern is "
            "closing too early, video will look like a repeating loop"
        )

    def test_waypoint_drift_is_registered(self) -> None:
        assert PATTERN_WAYPOINT_DRIFT in SUPPORTED_PATTERNS

    def test_waypoint_drift_respects_per_axis_amplitude_bound(self) -> None:
        traj = generate_pattern_trajectory(
            _request(
                pattern=PATTERN_WAYPOINT_DRIFT,
                amplitude_cm=0.50,
                cycle_duration_s=60.0,
                cycles=1,
                update_rate_hz=30.0,
                seed=7,
            )
        )
        for point in traj:
            assert abs(point.bottom_x_cm) <= 0.50 + 1e-9
            assert abs(point.bottom_y_cm) <= 0.50 + 1e-9
            assert abs(point.top_x_cm) <= 0.50 + 1e-9
            assert abs(point.top_y_cm) <= 0.50 + 1e-9

    def test_waypoint_drift_starts_and_ends_at_neutral(self) -> None:
        traj = generate_pattern_trajectory(
            _request(
                pattern=PATTERN_WAYPOINT_DRIFT,
                amplitude_cm=0.50,
                cycle_duration_s=60.0,
                cycles=1,
                update_rate_hz=30.0,
                seed=0,
            )
        )
        assert trajectory_starts_and_ends_at_neutral(traj)

    def test_waypoint_drift_reaches_far_waypoints(self) -> None:
        # The whole point of waypoint_drift is far-apart targets. Confirm
        # the trajectory actually visits multiple distinct regions of the
        # workspace by checking the bounding box width on both axes.
        traj = generate_pattern_trajectory(
            _request(
                pattern=PATTERN_WAYPOINT_DRIFT,
                amplitude_cm=0.50,
                cycle_duration_s=60.0,
                cycles=1,
                update_rate_hz=30.0,
                ramp_in_s=0.0,
                ramp_out_s=0.0,
                hold_at_start_s=0.0,
                hold_at_end_s=0.0,
                seed=7,
            )
        )
        xs = [p.bottom_x_cm for p in traj]
        ys = [p.bottom_y_cm for p in traj]
        x_span = max(xs) - min(xs)
        y_span = max(ys) - min(ys)
        # 0.5 cm amplitude means total reachable x-span is up to 1.0 cm.
        # The waypoint generator biases away from the origin so we expect
        # the actual span to use at least 60 % of the available range.
        assert x_span > 0.6, f"waypoint_drift x span = {x_span:.3f} cm; expected > 0.6 cm"
        assert y_span > 0.6, f"waypoint_drift y span = {y_span:.3f} cm; expected > 0.6 cm"

    def test_waypoint_drift_decelerates_near_each_waypoint(self) -> None:
        # Smoothstep guarantees velocity → 0 at each waypoint. Verify that
        # the per-tick step distribution has a clear "low velocity" tail
        # corresponding to the waypoint approach/exit segments.
        traj = generate_pattern_trajectory(
            _request(
                pattern=PATTERN_WAYPOINT_DRIFT,
                amplitude_cm=0.50,
                cycle_duration_s=60.0,
                cycles=1,
                update_rate_hz=30.0,
                ramp_in_s=0.0,
                ramp_out_s=0.0,
                hold_at_start_s=0.0,
                hold_at_end_s=0.0,
                seed=0,
            )
        )
        steps = []
        for prev, cur in zip(traj, traj[1:]):
            steps.append(
                math.hypot(
                    cur.bottom_x_cm - prev.bottom_x_cm,
                    cur.bottom_y_cm - prev.bottom_y_cm,
                )
            )
        # At a waypoint, smoothstep has zero derivative → step ≈ 0.
        # At the midpoint of a segment, derivative peaks → step is largest.
        # Confirm: minimum step is < 25 % of the mean step (i.e. there
        # are clear deceleration zones at waypoints).
        mean_step = sum(steps) / len(steps)
        min_step = min(steps)
        assert min_step < 0.25 * mean_step, (
            f"waypoint_drift never visibly decelerates: min step "
            f"{min_step:.4f} cm vs mean step {mean_step:.4f} cm — "
            "the smoothstep easing is broken"
        )

    def test_waypoint_drift_is_reproducible_for_same_seed(self) -> None:
        common = dict(
            pattern=PATTERN_WAYPOINT_DRIFT,
            amplitude_cm=0.50,
            cycle_duration_s=60.0,
            cycles=1,
            update_rate_hz=30.0,
            seed=42,
        )
        a = generate_pattern_trajectory(_request(**common))
        b = generate_pattern_trajectory(_request(**common))
        assert len(a) == len(b)
        for p_a, p_b in zip(a, b):
            assert p_a.bottom_x_cm == pytest.approx(p_b.bottom_x_cm)
            assert p_a.bottom_y_cm == pytest.approx(p_b.bottom_y_cm)

    def test_waypoint_drift_changes_when_seed_changes(self) -> None:
        common = dict(
            pattern=PATTERN_WAYPOINT_DRIFT,
            amplitude_cm=0.50,
            cycle_duration_s=60.0,
            cycles=1,
            update_rate_hz=30.0,
            ramp_in_s=0.0,
            ramp_out_s=0.0,
            hold_at_start_s=0.0,
            hold_at_end_s=0.0,
        )
        a = generate_pattern_trajectory(_request(**common, seed=0))
        b = generate_pattern_trajectory(_request(**common, seed=99))
        mid = len(a) // 2
        delta = math.hypot(
            a[mid].bottom_x_cm - b[mid].bottom_x_cm,
            a[mid].bottom_y_cm - b[mid].bottom_y_cm,
        )
        assert delta > 0.05, (
            f"waypoint_drift midpoint identical across seeds: {delta:.4f} cm — "
            "seed has no effect on waypoint placement"
        )

    def test_waypoint_drift_num_waypoints_constant_is_sensible(self) -> None:
        # If a future commit accidentally drops the waypoint count to 1 or 0,
        # the pattern degenerates into a straight line / point. Pin the
        # invariant so that regresses loudly.
        assert WAYPOINT_DRIFT_NUM_WAYPOINTS >= 4

    def test_creepy_squid_per_axis_velocity_is_smooth(self) -> None:
        # The motion should be visually smooth (the GUI streams at 5–10 Hz
        # in real-time). Per-sample velocity should not exceed a few mm in
        # one tick at 10 Hz so the operator does not see jerky jumps.
        traj = generate_pattern_trajectory(
            _request(
                pattern=PATTERN_CREEPY_SQUID,
                amplitude_cm=0.50,
                cycle_duration_s=25.0,
                cycles=1,
                update_rate_hz=10.0,
                ramp_in_s=0.0,
                ramp_out_s=0.0,
                hold_at_start_s=0.0,
                hold_at_end_s=0.0,
            )
        )
        previous = traj[0]
        max_step_cm = 0.0
        for point in traj[1:]:
            step = math.hypot(
                point.bottom_x_cm - previous.bottom_x_cm,
                point.bottom_y_cm - previous.bottom_y_cm,
            )
            if step > max_step_cm:
                max_step_cm = step
            previous = point
        # 0.5 cm amplitude, 0.1 s tick, max step ~ 0.5 cm * ω * dt ≈ 0.5 *
        # (max_freq_components * 2π/25) * 0.1 ≈ 0.5 * 4.123 * 2π/25 * 0.1
        # ≈ 0.052 cm. Allow generous 2× headroom for combined-harmonic peaks.
        assert max_step_cm < 0.15, (
            f"creepy_squid stepped {max_step_cm:.3f} cm in one 10 Hz tick — "
            "motion is too jerky"
        )

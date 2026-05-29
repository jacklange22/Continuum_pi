"""Tests for the sci-fi waypoint relay demo: waypoint generators, schedule,
speed advisory, and the trajectory builder consumed by the slow motion
experiment. Pure (no hardware, no Qt)."""

from __future__ import annotations

import math
from pathlib import Path

import pytest

from continuum_robot.demo.sci_fi_waypoint_relay import (
    DEFAULT_AMPLITUDE_CM,
    DEFAULT_EARLY_SWITCH_FRACTION,
    DEFAULT_VIDEO_DURATION_S,
    DEFAULT_WAYPOINT_COUNT,
    PRESET_WEIRD_WAYPOINTS_NORM,
    SUPPORTED_WAYPOINT_SOURCES,
    WAYPOINT_SOURCE_PRESET,
    WAYPOINT_SOURCE_SEEDED,
    SciFiRelayConfig,
    auto_select_seed,
    build_relay_schedule,
    build_relay_trajectory,
    build_waypoints,
    generate_preset_waypoints,
    generate_seeded_maximin_waypoints,
    min_consecutive_separation_norm,
    speed_advisory,
    validate_relay_config,
)


# ---------------------------------------------------------------------------
# Waypoint generators
# ---------------------------------------------------------------------------


class TestPresetGenerator:
    def test_returns_exactly_requested_count(self) -> None:
        wps = generate_preset_waypoints(amplitude_cm=0.5, waypoint_count=9)
        assert len(wps) == 9

    def test_within_amplitude_bound_per_axis(self) -> None:
        wps = generate_preset_waypoints(amplitude_cm=0.35, waypoint_count=9)
        for w in wps:
            assert abs(w.bottom_x_cm) <= 0.35 + 1e-9
            assert abs(w.bottom_y_cm) <= 0.35 + 1e-9
            assert abs(w.top_x_cm) <= 0.35 + 1e-9
            assert abs(w.top_y_cm) <= 0.35 + 1e-9

    def test_bottom_and_top_components_nontrivial(self) -> None:
        # Spec: every preset waypoint uses both bottom and top.
        wps = generate_preset_waypoints(amplitude_cm=0.5, waypoint_count=9)
        for w in wps:
            assert max(abs(w.bottom_x_cm), abs(w.bottom_y_cm)) > 0.05
            assert max(abs(w.top_x_cm), abs(w.top_y_cm)) > 0.05

    def test_scales_with_amplitude(self) -> None:
        small = generate_preset_waypoints(amplitude_cm=0.25, waypoint_count=9)
        large = generate_preset_waypoints(amplitude_cm=0.50, waypoint_count=9)
        # Each large waypoint should be exactly 2× the matching small one.
        for s, l in zip(small, large):
            assert l.bottom_x_cm == pytest.approx(2.0 * s.bottom_x_cm)
            assert l.bottom_y_cm == pytest.approx(2.0 * s.bottom_y_cm)
            assert l.top_x_cm == pytest.approx(2.0 * s.top_x_cm)
            assert l.top_y_cm == pytest.approx(2.0 * s.top_y_cm)

    def test_consecutive_separation_meets_min(self) -> None:
        wps = generate_preset_waypoints(amplitude_cm=0.5, waypoint_count=9)
        # The preset was hand-picked to keep consecutive distance well
        # above the default 0.4 norm. Pin it so a careless edit doesn't
        # silently reduce visual variety.
        assert min_consecutive_separation_norm(wps, amplitude_cm=0.5) >= 0.7

    def test_extra_waypoints_rotate_to_avoid_verbatim_repeat(self) -> None:
        wps = generate_preset_waypoints(amplitude_cm=0.5, waypoint_count=18)
        assert len(wps) == 18
        # The 10th waypoint should not be identical to the 1st (rotated).
        assert (wps[0].bottom_x_cm, wps[0].bottom_y_cm) != (wps[9].bottom_x_cm, wps[9].bottom_y_cm)


class TestSeededMaximinGenerator:
    def test_returns_exactly_requested_count(self) -> None:
        wps = generate_seeded_maximin_waypoints(
            amplitude_cm=0.5, waypoint_count=9, seed=0,
        )
        assert len(wps) == 9

    def test_within_amplitude_bound_per_axis(self) -> None:
        wps = generate_seeded_maximin_waypoints(
            amplitude_cm=0.35, waypoint_count=9, seed=42,
        )
        for w in wps:
            assert abs(w.bottom_x_cm) <= 0.35 + 1e-9
            assert abs(w.bottom_y_cm) <= 0.35 + 1e-9
            assert abs(w.top_x_cm) <= 0.35 + 1e-9
            assert abs(w.top_y_cm) <= 0.35 + 1e-9

    def test_reproducible_per_seed(self) -> None:
        a = generate_seeded_maximin_waypoints(amplitude_cm=0.5, waypoint_count=9, seed=7)
        b = generate_seeded_maximin_waypoints(amplitude_cm=0.5, waypoint_count=9, seed=7)
        assert len(a) == len(b) == 9
        for x, y in zip(a, b):
            assert x.as_tuple() == y.as_tuple()

    def test_different_seeds_produce_different_paths(self) -> None:
        a = generate_seeded_maximin_waypoints(amplitude_cm=0.5, waypoint_count=9, seed=0)
        b = generate_seeded_maximin_waypoints(amplitude_cm=0.5, waypoint_count=9, seed=99)
        # At least one waypoint must differ.
        assert any(x.as_tuple() != y.as_tuple() for x, y in zip(a, b))


class TestAutoSelectSeed:
    def test_chooses_seed_with_acceptable_separation(self) -> None:
        chosen, wps = auto_select_seed(
            base_seed=0, amplitude_cm=0.5, waypoint_count=9, min_separation_norm=0.4,
        )
        assert len(wps) == 9
        sep = min_consecutive_separation_norm(wps, amplitude_cm=0.5)
        # Either the first try met the bar, or a later seed did. In either
        # case the picked separation should be at least as good as the
        # configured min (the helper falls back to the best seed found
        # within the bounded search if none qualifies, but in practice the
        # first few maximin seeds always do).
        assert sep >= 0.4

    def test_improves_or_matches_base_seed_score(self) -> None:
        base_wps = generate_seeded_maximin_waypoints(
            amplitude_cm=0.5, waypoint_count=9, seed=0,
        )
        base_score = min_consecutive_separation_norm(base_wps, amplitude_cm=0.5)
        _, picked_wps = auto_select_seed(
            base_seed=0, amplitude_cm=0.5, waypoint_count=9,
            min_separation_norm=base_score + 0.05,
        )
        picked_score = min_consecutive_separation_norm(picked_wps, amplitude_cm=0.5)
        assert picked_score >= base_score


# ---------------------------------------------------------------------------
# Schedule + advisory
# ---------------------------------------------------------------------------


class TestRelaySchedule:
    def test_evenly_spaced_across_video_duration(self) -> None:
        cfg = SciFiRelayConfig(video_duration_s=20.0, waypoint_count=9)
        wps = generate_preset_waypoints(amplitude_cm=cfg.amplitude_cm, waypoint_count=cfg.waypoint_count)
        sched = build_relay_schedule(cfg, waypoints=wps)
        assert len(sched) == 9
        assert sched[0].issue_time_s == 0.0
        assert sched[-1].issue_time_s == pytest.approx(20.0)
        # Even spacing
        intervals = [sched[i + 1].issue_time_s - sched[i].issue_time_s for i in range(len(sched) - 1)]
        for d in intervals:
            assert d == pytest.approx(20.0 / 8.0)

    def test_uses_fewer_writes_than_continuous_30hz_mode(self) -> None:
        cfg = SciFiRelayConfig(video_duration_s=20.0, waypoint_count=9)
        wps = generate_preset_waypoints(amplitude_cm=cfg.amplitude_cm, waypoint_count=cfg.waypoint_count)
        sched = build_relay_schedule(cfg, waypoints=wps)
        continuous_writes = int(20.0 * 30.0)
        assert len(sched) < continuous_writes
        assert len(sched) == 9


class TestSpeedAdvisory:
    def test_default_config_classifies_slow(self) -> None:
        adv = speed_advisory(SciFiRelayConfig())
        assert adv.speed_class == "slow"
        assert adv.estimated_bus_writes == DEFAULT_WAYPOINT_COUNT
        assert adv.seconds_per_waypoint == pytest.approx(DEFAULT_VIDEO_DURATION_S / DEFAULT_WAYPOINT_COUNT)

    def test_high_amplitude_warns(self) -> None:
        adv = speed_advisory(SciFiRelayConfig(amplitude_cm=0.75))
        assert any("amplitude_cm" in w for w in adv.warnings)

    def test_high_command_rate_warns(self) -> None:
        # Above the ~50 Hz 8-servo sync-write ceiling the demo would fall
        # behind its schedule (slow motion), so the advisory flags it.
        adv = speed_advisory(SciFiRelayConfig(command_rate_hz=80.0))
        assert any("command_rate_hz" in w for w in adv.warnings)

    def test_short_interval_warns_and_pushes_aggressive(self) -> None:
        adv = speed_advisory(
            SciFiRelayConfig(
                video_duration_s=2.0, waypoint_count=10, amplitude_cm=0.8, profile_velocity=200,
            )
        )
        assert adv.speed_class == "aggressive"
        assert any("seconds_per_waypoint" in w for w in adv.warnings)


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


class TestRelayValidation:
    def test_zero_duration_raises(self) -> None:
        with pytest.raises(ValueError, match="video_duration_s"):
            validate_relay_config(SciFiRelayConfig(video_duration_s=0.0))

    def test_one_waypoint_raises(self) -> None:
        with pytest.raises(ValueError, match="waypoint_count"):
            validate_relay_config(SciFiRelayConfig(waypoint_count=1))

    def test_zero_amplitude_raises(self) -> None:
        with pytest.raises(ValueError, match="amplitude_cm"):
            validate_relay_config(SciFiRelayConfig(amplitude_cm=0.0))

    def test_excess_amplitude_raises(self) -> None:
        with pytest.raises(ValueError, match="cap"):
            validate_relay_config(SciFiRelayConfig(amplitude_cm=2.0))

    def test_unsupported_source_raises(self) -> None:
        with pytest.raises(ValueError, match="waypoint_source"):
            validate_relay_config(SciFiRelayConfig(waypoint_source="garbage"))


# ---------------------------------------------------------------------------
# Trajectory builder (the sparse trajectory consumed by the demo)
# ---------------------------------------------------------------------------


class TestRelayTrajectory:
    def test_starts_and_ends_at_neutral(self) -> None:
        points, _, _, _ = build_relay_trajectory(SciFiRelayConfig())
        assert points[0].bottom_x_cm == 0.0
        assert points[0].bottom_y_cm == 0.0
        assert points[-1].bottom_x_cm == 0.0
        assert points[-1].bottom_y_cm == 0.0

    def test_schedule_has_one_entry_per_waypoint(self) -> None:
        cfg = SciFiRelayConfig(waypoint_count=9)
        _points, wps, sched, _ = build_relay_trajectory(cfg)
        # The nominal schedule is still one evenly spaced entry per waypoint
        # (used for the waypoints/preview/CSV artifacts), even though the
        # streamed trajectory is now densely interpolated between them.
        assert len(sched) == len(wps)

    def test_trajectory_is_densely_sampled(self) -> None:
        cfg = SciFiRelayConfig(video_duration_s=20.0, waypoint_count=9, command_rate_hz=15.0)
        points, _, _, _ = build_relay_trajectory(cfg)
        # The relay now streams a dense overlapping-blend path (not one write
        # per waypoint) so early_switch_fraction can actually shape the motion.
        # ~20 s * 15 Hz body + 2 neutral holds.
        assert len(points) > 100
        # Body samples are spaced ~1/command_rate_hz apart.
        body = [p for p in points if p.phase_label == "pattern"]
        gaps = [b.elapsed_s - a.elapsed_s for a, b in zip(body, body[1:])]
        assert gaps and max(gaps) < 0.2

    def test_resolved_seed_returned_for_preset(self) -> None:
        cfg = SciFiRelayConfig(waypoint_source=WAYPOINT_SOURCE_PRESET, seed=42)
        _, _, _, resolved = build_relay_trajectory(cfg)
        # Preset source ignores the seed; the resolved value still matches
        # what the operator passed in so the summary can record it.
        assert resolved == 42

    def test_resolved_seed_returned_for_seeded_maximin(self) -> None:
        cfg = SciFiRelayConfig(
            waypoint_source=WAYPOINT_SOURCE_SEEDED, seed=0, auto_select_seed=True,
        )
        _, _, _, resolved = build_relay_trajectory(cfg)
        # auto_select_seed may roll forward from 0 to find a good seed.
        assert resolved >= 0

    def test_per_axis_amplitude_bound_holds(self) -> None:
        amp = 0.35
        cfg = SciFiRelayConfig(amplitude_cm=amp, waypoint_count=9)
        points, _, _, _ = build_relay_trajectory(cfg)
        for p in points:
            assert abs(p.bottom_x_cm) <= amp + 1e-9
            assert abs(p.bottom_y_cm) <= amp + 1e-9
            assert abs(p.top_x_cm) <= amp + 1e-9
            assert abs(p.top_y_cm) <= amp + 1e-9

    def test_early_switch_fraction_is_built_into_the_path(self) -> None:
        # early_switch_fraction must actually shape the trajectory (it used to
        # be metadata-only). At 1.0 the blend settles on every waypoint; lower
        # values overlap the transitions so the spine rounds the corners and
        # stays farther from the discrete waypoints.
        amp = 0.35

        def avg_distance_to_waypoints(esf: float) -> float:
            cfg = SciFiRelayConfig(
                video_duration_s=20.0, waypoint_count=9, amplitude_cm=amp,
                early_switch_fraction=esf, command_rate_hz=15.0,
            )
            points, wps, _sched, _ = build_relay_trajectory(cfg)
            body = [p for p in points if p.phase_label == "pattern"]

            def nearest(w) -> float:
                target = (w.bottom_x_cm, w.bottom_y_cm, w.top_x_cm, w.top_y_cm)
                return min(
                    math.dist((p.bottom_x_cm, p.bottom_y_cm, p.top_x_cm, p.top_y_cm), target)
                    for p in body
                )

            return sum(nearest(w) for w in wps) / len(wps)

        settle = avg_distance_to_waypoints(1.0)
        rounded = avg_distance_to_waypoints(0.5)
        # esf=1.0 reaches the waypoints essentially exactly...
        assert settle < 1e-6
        # ...and a lower fraction visibly rounds them off (never settles).
        assert rounded > settle + 0.02


# ---------------------------------------------------------------------------
# Output writers
# ---------------------------------------------------------------------------


class TestRelayOutputs:
    def test_writes_required_files(self, tmp_path: Path) -> None:
        from continuum_robot.demo.sci_fi_waypoint_relay_outputs import (
            write_sci_fi_relay_outputs,
        )
        cfg = SciFiRelayConfig()
        points, wps, sched, resolved = build_relay_trajectory(cfg)
        adv = speed_advisory(cfg)
        # Build a minimal "config-like" namespace matching the experiment's
        # TwoSegmentSlowMotionDemoConfig attributes that the writers read.
        from types import SimpleNamespace
        config_ns = SimpleNamespace(
            pattern="sci_fi_waypoint_relay",
            amplitude_cm=cfg.amplitude_cm,
            video_duration_s=cfg.video_duration_s,
            waypoint_count=cfg.waypoint_count,
            early_switch_fraction=cfg.early_switch_fraction,
            waypoint_source=cfg.waypoint_source,
            relay_seed=cfg.seed,
            auto_select_seed=cfg.auto_select_seed,
            profile_velocity=cfg.profile_velocity,
            profile_acceleration=cfg.profile_acceleration,
            command_rate_hz=cfg.command_rate_hz,
            dry_run=True,
        )
        paths = write_sci_fi_relay_outputs(
            output_dir=tmp_path,
            config=config_ns,
            waypoints=wps,
            schedule=sched,
            resolved_seed=resolved,
            advisory=adv,
            profile_restore_success=True,
            profile_restore_failure_reason=None,
            trace=[],
        )
        assert (tmp_path / "waypoints.json").exists()
        assert (tmp_path / "waypoint_schedule.csv").exists()
        assert (tmp_path / "sci_fi_waypoint_relay_summary.txt").exists()
        # Figure writers are best-effort but should succeed in this env.
        assert "preview_png" in paths
        assert (tmp_path / "sci_fi_waypoint_preview.png").exists()

    def test_waypoints_json_includes_resolved_seed_and_advisory(self, tmp_path: Path) -> None:
        from continuum_robot.demo.sci_fi_waypoint_relay_outputs import (
            write_sci_fi_relay_outputs,
        )
        import json
        from types import SimpleNamespace
        cfg = SciFiRelayConfig(waypoint_source=WAYPOINT_SOURCE_SEEDED, seed=7, auto_select_seed=False)
        points, wps, sched, resolved = build_relay_trajectory(cfg)
        adv = speed_advisory(cfg)
        config_ns = SimpleNamespace(
            pattern="sci_fi_waypoint_relay",
            amplitude_cm=cfg.amplitude_cm,
            video_duration_s=cfg.video_duration_s,
            waypoint_count=cfg.waypoint_count,
            early_switch_fraction=cfg.early_switch_fraction,
            waypoint_source=cfg.waypoint_source,
            relay_seed=cfg.seed,
            auto_select_seed=cfg.auto_select_seed,
            profile_velocity=cfg.profile_velocity,
            profile_acceleration=cfg.profile_acceleration,
            command_rate_hz=cfg.command_rate_hz,
            dry_run=True,
        )
        write_sci_fi_relay_outputs(
            output_dir=tmp_path,
            config=config_ns,
            waypoints=wps,
            schedule=sched,
            resolved_seed=resolved,
            advisory=adv,
            profile_restore_success=False,
            profile_restore_failure_reason="bus drop on servo 4",
            trace=[],
        )
        payload = json.loads((tmp_path / "waypoints.json").read_text(encoding="utf-8"))
        assert payload["pattern"] == "sci_fi_waypoint_relay"
        assert payload["resolved_seed"] == resolved
        assert payload["demo_only"] is True
        assert payload["closed_loop_control"] is False
        assert len(payload["waypoints"]) == cfg.waypoint_count

    def test_summary_records_profile_restore_failure(self, tmp_path: Path) -> None:
        from continuum_robot.demo.sci_fi_waypoint_relay_outputs import (
            write_sci_fi_relay_outputs,
        )
        from types import SimpleNamespace
        cfg = SciFiRelayConfig()
        points, wps, sched, resolved = build_relay_trajectory(cfg)
        adv = speed_advisory(cfg)
        config_ns = SimpleNamespace(
            pattern="sci_fi_waypoint_relay",
            amplitude_cm=cfg.amplitude_cm,
            video_duration_s=cfg.video_duration_s,
            waypoint_count=cfg.waypoint_count,
            early_switch_fraction=cfg.early_switch_fraction,
            waypoint_source=cfg.waypoint_source,
            relay_seed=cfg.seed,
            auto_select_seed=cfg.auto_select_seed,
            profile_velocity=cfg.profile_velocity,
            profile_acceleration=cfg.profile_acceleration,
            command_rate_hz=cfg.command_rate_hz,
            dry_run=False,
        )
        write_sci_fi_relay_outputs(
            output_dir=tmp_path,
            config=config_ns,
            waypoints=wps,
            schedule=sched,
            resolved_seed=resolved,
            advisory=adv,
            profile_restore_success=False,
            profile_restore_failure_reason="servo 4: timeout",
            trace=[],
        )
        text = (tmp_path / "sci_fi_waypoint_relay_summary.txt").read_text(encoding="utf-8")
        assert "profile_restore_success" not in text or "success:                 False" in text or "False" in text
        assert "servo 4: timeout" in text
        assert "demo_only: True" in text


# ---------------------------------------------------------------------------
# Pattern registration in two_segment_motion_patterns
# ---------------------------------------------------------------------------


class TestPatternRegistration:
    def test_relay_pattern_in_supported_set(self) -> None:
        from continuum_robot.demo.two_segment_motion_patterns import (
            PATTERN_SCI_FI_WAYPOINT_RELAY, SUPPORTED_PATTERNS,
        )
        assert PATTERN_SCI_FI_WAYPOINT_RELAY in SUPPORTED_PATTERNS

    def test_generate_pattern_trajectory_rejects_relay_directly(self) -> None:
        from continuum_robot.demo.two_segment_motion_patterns import (
            PATTERN_SCI_FI_WAYPOINT_RELAY, PatternRequest, generate_pattern_trajectory,
        )
        with pytest.raises(ValueError, match="build_relay_trajectory"):
            generate_pattern_trajectory(
                PatternRequest(pattern=PATTERN_SCI_FI_WAYPOINT_RELAY)
            )


# ---------------------------------------------------------------------------
# Experiment integration smoke test
# ---------------------------------------------------------------------------


class TestExperimentIntegration:
    def test_dry_run_executes_end_to_end_and_records_relay_summary(self) -> None:
        from continuum_robot.experiments.two_segment_slow_motion_demo import (
            TwoSegmentSlowMotionDemoConfig, TwoSegmentSlowMotionDemoExperiment,
        )
        from types import SimpleNamespace

        class _StubSession:
            def __init__(self):
                self.metrics: dict = {}
                self.stage_pass_fail: dict = {}
                self.samples: list = []
                self.context = SimpleNamespace(
                    settings=None, servo_service=None,
                    sleep_fn=lambda _s: None, monotonic_fn=lambda: 0.0,
                )
            def set_metric(self, k, v): self.metrics[k] = v
            def add_sample(self, s): self.samples.append(s)
            def update_progress(self, *a, **k): pass
            def stop_requested(self): return False
            def set_stage(self, *a, **k): pass

        config = TwoSegmentSlowMotionDemoConfig.from_dict(
            {
                "pattern": "sci_fi_waypoint_relay",
                "video_duration_s": 20.0,
                "waypoint_count": 9,
                "amplitude_cm": 0.35,
                "early_switch_fraction": 0.72,
                "waypoint_source": "preset_weird",
                "profile_velocity": 45,
                "profile_acceleration": 18,
                "dry_run": True,
            }
        )
        experiment = TwoSegmentSlowMotionDemoExperiment(config=config)
        session = _StubSession()
        experiment.setup(session)
        # Trajectory is now a dense overlapping-blend path (1 hold_start +
        # many body samples + 1 hold_end), not one write per waypoint.
        assert len(experiment._trajectory) > 100
        assert len(experiment._relay_waypoints) == 9
        assert len(experiment._relay_schedule) == 9
        experiment.precheck(session)
        experiment.execute(session)
        # Every streamed point produced a trace row and a sample.
        assert len(experiment._trace) == len(experiment._trajectory)
        assert len(session.samples) == len(experiment._trajectory)
        summary = experiment.summarize(session)
        assert summary["pattern"] == "sci_fi_waypoint_relay"
        assert summary["video_duration_s"] == 20.0
        assert summary["waypoint_count"] == 9
        assert summary["early_switch_fraction"] == pytest.approx(0.72)
        assert summary["demo_only"] is True
        assert summary["closed_loop_control"] is False
        assert "relay_speed_class" in summary


# ---------------------------------------------------------------------------
# Validator integration
# ---------------------------------------------------------------------------


class TestValidatorIntegration:
    def _make_valid_relay_run(self, tmp_path: Path) -> Path:
        """Build a minimal valid run folder so we can exercise the validator."""
        import json
        run = tmp_path / "20260601_120000_two_segment_slow_motion_demo"
        run.mkdir()
        (run / "metadata.json").write_text(
            json.dumps(
                {
                    "schema_version": "1.0",
                    "experiment_name": "two_segment_slow_motion_demo",
                    "run_id": "test",
                    "timestamp_utc": "2026-06-01T12:00:00+00:00",
                    "git_commit": None,
                    "backend_info": {"mock_mode": True},
                    "registration_info": {"exists": False},
                    "config_used": {"pattern": "sci_fi_waypoint_relay"},
                }
            ),
            encoding="utf-8",
        )
        (run / "summary.json").write_text(
            json.dumps(
                {
                    "schema_version": "1.0",
                    "experiment_name": "two_segment_slow_motion_demo",
                    "run_id": "test",
                    "success": True,
                    "sample_counts": {"total": 11},
                    "dropped_frames": 0,
                    "invalid_transforms": 0,
                    "stage_pass_fail": {"execute": "passed"},
                    "experiment_metrics": {
                        "pattern": "sci_fi_waypoint_relay",
                        "demo_only": True,
                        "closed_loop_control": False,
                        "valid_for_model_training": False,
                        "valid_for_thesis_repeatability": False,
                        "amplitude_cm": 0.35,
                        "profile_restore_success": True,
                        "run_trust_mode": "demo_only",
                        "run_provenance": {
                            "operating_mode": "dual_segment",
                            "hardware_profile": "mock",
                            "two_segment_foundation": {
                                "command_schema": {"schema_version": "two_segment_command_v1"},
                                "pose_schema": {"schema_version": "two_segment_pose_v1"},
                            },
                        },
                    },
                }
            ),
            encoding="utf-8",
        )
        (run / "samples.jsonl").write_text("", encoding="utf-8")
        (run / "waypoints.json").write_text(
            json.dumps({"schema_version": "sci_fi_waypoint_relay_v1", "waypoints": []}),
            encoding="utf-8",
        )
        (run / "waypoint_schedule.csv").write_text(
            "waypoint_index,issue_time_s,bottom_x_cm,bottom_y_cm,top_x_cm,top_y_cm\n",
            encoding="utf-8",
        )
        (run / "sci_fi_waypoint_relay_summary.txt").write_text("demo_only: True\n", encoding="utf-8")
        return run

    def test_valid_relay_run_passes(self, tmp_path: Path) -> None:
        from continuum_robot.data.validate_run_bundle import validate_run_folder
        run = self._make_valid_relay_run(tmp_path)
        result = validate_run_folder(run)
        assert result.status == "PASS", [str(i) for i in result.issues]

    def test_missing_waypoints_json_fails(self, tmp_path: Path) -> None:
        from continuum_robot.data.validate_run_bundle import validate_run_folder
        run = self._make_valid_relay_run(tmp_path)
        (run / "waypoints.json").unlink()
        result = validate_run_folder(run)
        assert result.status == "FAIL"
        assert any("waypoints.json" in str(issue) for issue in result.issues)

    def test_profile_restore_failure_warns(self, tmp_path: Path) -> None:
        import json
        from continuum_robot.data.validate_run_bundle import validate_run_folder
        run = self._make_valid_relay_run(tmp_path)
        summary_path = run / "summary.json"
        payload = json.loads(summary_path.read_text(encoding="utf-8"))
        payload["experiment_metrics"]["profile_restore_success"] = False
        payload["experiment_metrics"]["profile_restore_failure_reason"] = "bus drop"
        summary_path.write_text(json.dumps(payload), encoding="utf-8")
        result = validate_run_folder(run)
        assert any(
            "profile_restore_success" in str(issue) and issue.level == "WARN"
            for issue in result.issues
        ), [str(i) for i in result.issues]

    def test_high_amplitude_warns(self, tmp_path: Path) -> None:
        import json
        from continuum_robot.data.validate_run_bundle import validate_run_folder
        run = self._make_valid_relay_run(tmp_path)
        summary_path = run / "summary.json"
        payload = json.loads(summary_path.read_text(encoding="utf-8"))
        payload["experiment_metrics"]["amplitude_cm"] = 0.75
        summary_path.write_text(json.dumps(payload), encoding="utf-8")
        result = validate_run_folder(run)
        assert any(
            "amplitude_cm" in str(issue) and issue.level == "WARN"
            for issue in result.issues
        ), [str(i) for i in result.issues]


# ---------------------------------------------------------------------------
# GUI preset wiring
# ---------------------------------------------------------------------------


def test_gui_preset_sets_relay_pattern_and_amplitude(monkeypatch) -> None:
    import os
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PySide6.QtWidgets import QApplication
    import sys
    app = QApplication.instance() or QApplication(sys.argv)
    from unittest.mock import MagicMock
    from continuum_robot.gui.widgets.experiment_pages import TwoSegmentSlowMotionDemoPage

    controller = MagicMock()
    controller.settings.runtime.visualization_mode = "off"
    controller.settings.runtime.visualization_safe_effects = True
    page = TwoSegmentSlowMotionDemoPage(controller, "two_segment_slow_motion_demo")
    page._apply_sci_fi_spine_preset()
    # The preset should have called set_config_value with the relay
    # pattern + 0.35 cm amplitude + 9 waypoints.
    pushed = {call.args[0]: call.args[1] for call in controller.set_config_value.call_args_list}
    assert pushed.get("pattern") == "sci_fi_waypoint_relay"
    assert pushed.get("video_duration_s") == 20.0
    assert pushed.get("waypoint_count") == 9
    assert pushed.get("amplitude_cm") == 0.35
    assert pushed.get("dry_run") is True

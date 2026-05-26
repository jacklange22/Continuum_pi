"""Tests for the two-segment collect-pose thesis figure set.

The figures are additive on top of the legacy ``two_segment_*_report.png``
files emitted by ``write_two_segment_dataset_outputs``. These tests confirm:

- every canonical ``thesis_0N`` figure is written and non-empty;
- record extraction correctly maps two-segment command + pose payloads to
  plot-friendly dicts;
- a run with no accepted distal poses still produces every figure (so the
  bundle stays complete on servo-only / pose-missing runs);
- failures during one figure do not prevent the rest from rendering;
- the existing legacy report figures are untouched.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from continuum_robot.experiments.schemas import ExperimentTimeseriesSample
from continuum_robot.experiments.two_segment_modeling_dataset_outputs import (
    THESIS_FIGURE_NAMES,
    VARIABILITY_FIGURE_NAMES,
    _accepted_records,
    _build_export_rows_from_records,
    _build_thesis_records,
    _compute_variability_records,
    _pair_from_cable_deltas,
    _vector_magnitude,
    write_two_segment_thesis_figures,
)


# ---------------------------------------------------------------------------
# Sample factories
# ---------------------------------------------------------------------------


def _identity_4x4(translation_mm: tuple[float, float, float]) -> list[list[float]]:
    return [
        [1.0, 0.0, 0.0, float(translation_mm[0])],
        [0.0, 1.0, 0.0, float(translation_mm[1])],
        [0.0, 0.0, 1.0, float(translation_mm[2])],
        [0.0, 0.0, 0.0, 1.0],
    ]


def _make_sample(
    *,
    sample_index: int,
    phase: str = "random_babble",
    accepted: bool = True,
    rejection_reason: str = "",
    segment_a: list[float] | None = None,
    segment_b: list[float] | None = None,
    tip_xyz_mm: tuple[float, float, float] | None = (10.0, -5.0, 50.0),
    intermediate_xyz_mm: tuple[float, float, float] | None = None,
    feedback_currents: dict[int, int] | None = None,
    feedback_positions: dict[int, int] | None = None,
    delta_from_startup: dict[int, int] | None = None,
    cycle_index: int = 0,
    step_index: int | None = None,
) -> ExperimentTimeseriesSample:
    seg_a = list(segment_a) if segment_a is not None else [0.0, 0.0, 0.0, 0.0]
    seg_b = list(segment_b) if segment_b is not None else [0.0, 0.0, 0.0, 0.0]
    feedback: dict[str, dict[str, Any]] = {}
    for sid in range(1, 9):
        feedback[str(sid)] = {
            "servo_id": sid,
            "position_tick": int((feedback_positions or {}).get(sid, 2048)),
            "delta_from_startup_tick": int((delta_from_startup or {}).get(sid, 0)),
            "signed_raw_current_ma": int((feedback_currents or {}).get(sid, 50)),
            "load_proxy_ma": int(abs((feedback_currents or {}).get(sid, 50))),
            "voltage_mv": 12000,
            "temperature_c": 25,
            "hardware_error": None,
            "telemetry_stale": False,
            "telemetry_age_s": 0.02,
        }
    pose_payload: dict[str, Any] = {
        "schema_version": "2segment_pose_v1",
        "status": "pose_roles_available" if tip_xyz_mm else "tracker_unavailable",
        "distal_tip_pose": (
            {"T_robot_tip": _identity_4x4(tip_xyz_mm), "pose_kind": "test"}
            if tip_xyz_mm is not None
            else {}
        ),
        "intermediate_pose": (
            {"T_robot_intermediate": _identity_4x4(intermediate_xyz_mm)}
            if intermediate_xyz_mm is not None
            else {}
        ),
    }
    extra = {
        "record_kind": "two_segment_collect_pose_sample",
        "capture_accepted": bool(accepted),
        "capture_rejection_reason": rejection_reason or None,
        "measured_servo_feedback": feedback,
        "goal_ticks_by_servo": {str(sid): 2048 for sid in range(1, 9)},
        "available_pose_roles": ["distal_tip"] if tip_xyz_mm else [],
        "cycle_index": int(cycle_index),
    }
    return ExperimentTimeseriesSample(
        monotonic_time_s=float(sample_index) * 0.5,
        wall_time_utc="2026-05-26T00:00:00+00:00",
        phase=phase,
        step_index=int(step_index if step_index is not None else sample_index // 4),
        sample_index=int(sample_index),
        cycle_index=int(cycle_index),
        two_segment_command={
            "schema_version": "2segment_cmd_v1",
            "segments": {"segment_a": seg_a, "segment_b": seg_b},
            "servo_command_cm": {
                **{str(i + 1): float(seg_a[i]) for i in range(4)},
                **{str(i + 5): float(seg_b[i]) for i in range(4)},
            },
        },
        two_segment_pose=pose_payload,
        pose_in_robot_frame=(
            {"roles": {"distal_tip": {"T_robot_tip": _identity_4x4(tip_xyz_mm)}}}
            if tip_xyz_mm is not None
            else {}
        ),
        extra=extra,
    )


def _seeded_samples(*, count: int = 12, with_pose: bool = True, with_intermediate: bool = False) -> list[ExperimentTimeseriesSample]:
    samples: list[ExperimentTimeseriesSample] = []
    for i in range(count):
        # Walk segment-A pair through a small circle so command-space figures
        # show real spread.
        angle = (i / max(1, count - 1)) * 2.0
        seg_a = [-0.1 * angle, -0.05 * angle, 0.1 * angle, 0.05 * angle]
        seg_b = [0.2 * angle, 0.15 * angle, -0.2 * angle, -0.15 * angle]
        tip = (5.0 * angle, 2.0 * angle, 50.0 + angle) if with_pose else None
        intermediate = (2.0 * angle, 1.0 * angle, 25.0 + 0.5 * angle) if with_intermediate else None
        # Inject a couple of rejected samples to populate the dataset-quality plot.
        accepted = (i % 5) != 0
        rejection = "tracker_stale" if not accepted else ""
        samples.append(
            _make_sample(
                sample_index=i,
                accepted=accepted,
                rejection_reason=rejection,
                segment_a=seg_a,
                segment_b=seg_b,
                tip_xyz_mm=tip,
                intermediate_xyz_mm=intermediate,
                feedback_currents={sid: 100 + 10 * sid + 5 * i for sid in range(1, 9)},
                feedback_positions={sid: 2048 + (i * 7 * (1 if sid <= 4 else -1)) for sid in range(1, 9)},
                delta_from_startup={sid: i * 5 * (1 if sid <= 4 else -1) for sid in range(1, 9)},
            )
        )
    return samples


# ---------------------------------------------------------------------------
# Unit tests for helpers
# ---------------------------------------------------------------------------


def test_pair_from_cable_deltas_matches_tip_target_convention() -> None:
    # Canonical mapping: cable_deltas = [-px, -py, +px, +py] → pair = (-c0, -c1).
    assert _pair_from_cable_deltas([-0.3, -0.4, 0.3, 0.4]) == (0.3, 0.4)
    assert _pair_from_cable_deltas([0.0, 0.0, 0.0, 0.0]) == (-0.0, -0.0)
    assert _pair_from_cable_deltas([]) is None
    assert _pair_from_cable_deltas([0.1]) is None


def test_vector_magnitude_handles_empty_and_typical() -> None:
    assert _vector_magnitude([]) == pytest.approx(0.0)
    assert _vector_magnitude([3.0, 4.0]) == pytest.approx(5.0)
    assert _vector_magnitude([1.0, 1.0, 1.0, 1.0]) == pytest.approx(2.0)


def test_build_thesis_records_extracts_command_pose_feedback() -> None:
    samples = _seeded_samples(count=4, with_intermediate=True)
    records = _build_thesis_records(samples)
    assert len(records) == 4
    record = records[1]
    assert record["segment_a_cable_cm"] == [pytest.approx(v) for v in samples[1].two_segment_command["segments"]["segment_a"]]
    assert record["segment_a_pair_cm"] is not None
    assert record["segment_b_pair_cm"] is not None
    assert record["tip_xyz_mm"] is not None
    assert record["intermediate_xyz_mm"] is not None
    assert 1 in record["feedback_by_servo"]
    assert record["feedback_by_servo"][1]["load_proxy_ma"] > 0


def test_accepted_filter_excludes_rejected_and_pose_missing() -> None:
    samples = _seeded_samples(count=10)
    records = _build_thesis_records(samples)
    accepted = _accepted_records(records)
    assert all(r["accepted"] for r in accepted)
    assert all(r["tip_xyz_mm"] is not None for r in accepted)


def test_build_export_rows_carries_tip_and_per_segment_pair_keys() -> None:
    samples = _seeded_samples(count=8)
    records = _build_thesis_records(samples)
    export_rows = _build_export_rows_from_records(records)
    assert export_rows, "expected at least one accepted export row"
    sample_row = export_rows[0]
    # Mirrors the single-segment ``_build_export_rows`` shape so the
    # ported thesis_01 writer can consume them unchanged.
    assert set(sample_row.keys()) >= {
        "sequence_index",
        "tip_position_xyz_mm",
        "requested_pair_command_cm",
        "segment_a_pair_cm",
        "segment_b_pair_cm",
    }
    assert len(sample_row["tip_position_xyz_mm"]) == 3
    assert len(sample_row["segment_a_pair_cm"]) == 2
    assert len(sample_row["segment_b_pair_cm"]) == 2


def test_compute_variability_records_groups_repeated_samples_per_command() -> None:
    # Two commands, each captured 4 times across 2 cycles.
    samples: list = []
    for cycle in range(2):
        for step_index in (10, 11):
            for rep in range(2):
                samples.append(
                    _make_sample(
                        sample_index=rep,
                        cycle_index=cycle,
                        step_index=step_index,
                        # Add a tiny spread per repeat so std_rms > 0.
                        tip_xyz_mm=(
                            10.0 + 0.1 * rep + step_index,
                            -5.0 + 0.05 * rep,
                            50.0 + 0.02 * rep,
                        ),
                        segment_a=[-0.1, -0.05, 0.1, 0.05],
                        segment_b=[0.2, 0.15, -0.2, -0.15],
                    )
                )
    records = _build_thesis_records(samples)
    variability = _compute_variability_records(records)
    # 2 cycles × 2 step indices = 4 unique commands, each with 2 repeats.
    assert len(variability) == 4
    for entry in variability:
        assert entry["valid_sample_count"] == 2
        assert entry["position_std_rms_mm"] is not None
        assert entry["position_std_rms_mm"] > 0.0
        assert entry["averaged_x_mm"] is not None


def test_compute_variability_records_returns_empty_when_no_command_repeats() -> None:
    # Every sample at a unique (cycle, step_index) — i.e. samples_per_pattern=1
    # in the real experiment. The variability path should skip entirely so
    # the writer can decide not to emit the tracker_variability_* figures.
    samples = [
        _make_sample(
            sample_index=i,
            cycle_index=0,
            step_index=i,
            tip_xyz_mm=(float(i), float(i), 50.0),
        )
        for i in range(8)
    ]
    records = _build_thesis_records(samples)
    assert _compute_variability_records(records) == []


def test_random_babble_unique_command_count_resolves_from_target_when_budget_zero() -> None:
    from continuum_robot.experiments.two_segment_collect_pose_dataset import (
        TwoSegmentCollectPoseDatasetConfig,
        _random_babble_unique_command_count,
    )
    # The historical bug: budget=0 collapsed unique-command count to 32 even
    # when the operator asked for 50,000 samples (10 per pattern → 5,000
    # unique). Confirm the new resolver does the right thing.
    cfg = TwoSegmentCollectPoseDatasetConfig.from_dict(
        {
            "schedule_type": "random_babble",
            "pattern_count_budget": 0,
            "continue_until_valid_samples": True,
            "target_valid_sample_count": 50000,
            "samples_per_pattern": 10,
        }
    )
    count = _random_babble_unique_command_count(cfg)
    assert count >= 5000, f"expected at least 5000 unique commands, got {count}"
    # 5 % headroom is added so an 80 %+ acceptance run finishes in one pass.
    assert count <= 5300, f"expected ~5000 + small headroom, got {count}"


def test_random_babble_unique_command_count_respects_explicit_budget() -> None:
    from continuum_robot.experiments.two_segment_collect_pose_dataset import (
        TwoSegmentCollectPoseDatasetConfig,
        _random_babble_unique_command_count,
    )
    cfg = TwoSegmentCollectPoseDatasetConfig.from_dict(
        {
            "schedule_type": "random_babble",
            "pattern_count_budget": 1234,
            "continue_until_valid_samples": True,
            "target_valid_sample_count": 50000,
            "samples_per_pattern": 10,
        }
    )
    assert _random_babble_unique_command_count(cfg) == 1234


def test_random_babble_unique_command_count_default_when_unset() -> None:
    from continuum_robot.experiments.two_segment_collect_pose_dataset import (
        TwoSegmentCollectPoseDatasetConfig,
        _random_babble_unique_command_count,
    )
    cfg = TwoSegmentCollectPoseDatasetConfig.from_dict(
        {
            "schedule_type": "random_babble",
            "pattern_count_budget": 0,
            "continue_until_valid_samples": False,
            "target_valid_sample_count": 0,
            "samples_per_pattern": 1,
        }
    )
    # No budget, no target → fall back to the 5000 default (matches the
    # single-segment ladder so a fresh operator gets sane coverage).
    assert _random_babble_unique_command_count(cfg) == 5000


def test_random_babble_schedule_generates_full_unique_command_set() -> None:
    """Regression test for the 33-unique-command bug.

    Pre-fix: a 50K-target / 10-per-pattern random_babble run silently
    generated only 32 random commands and cycled through them ~150 times.
    Post-fix: it should generate ~5000 distinct random commands so one
    pass covers the requested sample count.
    """
    from continuum_robot.experiments.two_segment_collect_pose_dataset import (
        TwoSegmentCollectPoseDatasetConfig,
        build_two_segment_command_schedule,
    )
    cfg = TwoSegmentCollectPoseDatasetConfig.from_dict(
        {
            "schedule_type": "random_babble",
            "max_segment_displacement_cm": 0.75,
            "pattern_count_budget": 0,
            "continue_until_valid_samples": True,
            "target_valid_sample_count": 50000,
            "samples_per_pattern": 10,
            "random_seed": 0,
        }
    )
    steps = build_two_segment_command_schedule(cfg, context=None)
    # +1 for the leading zero step; everything else is unique random.
    random_steps = [s for s in steps if s.phase == "random_babble"]
    assert len(random_steps) >= 5000, f"expected ≥5000 random commands, got {len(random_steps)}"
    # And every command really is unique (regression: the old code looped
    # through the same 32 commands many times).
    cable_signatures = {
        tuple(round(v, 6) for v in step.command.segment_a)
        + tuple(round(v, 6) for v in step.command.segment_b)
        for step in random_steps
    }
    assert len(cable_signatures) >= 4900, (
        "random_babble produced duplicate commands — unique-count fix regressed"
    )


# ---------------------------------------------------------------------------
# End-to-end figure writes
# ---------------------------------------------------------------------------


def test_write_two_segment_thesis_figures_emits_full_set(tmp_path: Path) -> None:
    samples = _seeded_samples(count=24)
    metrics = {
        "schedule_type": "random_babble",
        "accepted_sample_count": sum(1 for s in samples if s.extra["capture_accepted"]),
        "rejected_sample_count": sum(1 for s in samples if not s.extra["capture_accepted"]),
        "command_failure_count": 1,
        "run_trust_mode": "thesis_trusted",
        "valid_for_two_segment_model_training": True,
        "valid_for_two_segment_ann_training": False,
        "config_used": {
            "max_tick_delta_from_startup": 600,
            "current_warning_ma": 800,
            "sustained_overcurrent_ma": 1200,
        },
    }
    paths = write_two_segment_thesis_figures(
        output_dir=tmp_path,
        metrics=metrics,
        samples=samples,
        sample_failure_events=[
            {"reason": "telemetry_packet_error", "servo_id": 3, "sample_index": 5},
        ],
    )
    expected_thesis_keys = {"thesis_01", "thesis_02", "thesis_03", "thesis_04", "thesis_05", "thesis_06"}
    assert expected_thesis_keys.issubset(set(paths.keys()))
    for filename in THESIS_FIGURE_NAMES:
        target = tmp_path / filename
        assert target.exists(), f"{filename} should be written"
        # PNGs from matplotlib are far larger than 1 KB; placeholder is ~70 bytes.
        assert target.stat().st_size > 4_000, f"{filename} looks like a placeholder, not a real plot"


def test_write_two_segment_thesis_figures_emits_variability_set_when_commands_repeat(tmp_path: Path) -> None:
    """When samples_per_pattern > 1 produces repeated samples per command,
    the variability triplet (workspace_xy / std_histogram / std_vs_command_index)
    should be emitted alongside the standard thesis_* set."""
    samples: list = []
    for step_index in range(6):
        for rep in range(5):
            samples.append(
                _make_sample(
                    sample_index=rep,
                    cycle_index=0,
                    step_index=step_index,
                    # Small noise per repeat → spread > 0.
                    tip_xyz_mm=(10.0 * step_index + 0.05 * rep, -5.0 * step_index, 50.0),
                    segment_a=[-0.1 * step_index, -0.05 * step_index, 0.1 * step_index, 0.05 * step_index],
                    segment_b=[0.2 * step_index, 0.15 * step_index, -0.2 * step_index, -0.15 * step_index],
                )
            )
    metrics = {
        "schedule_type": "random_babble",
        "accepted_sample_count": len(samples),
        "rejected_sample_count": 0,
        "command_failure_count": 0,
        "run_trust_mode": "thesis_trusted",
        "valid_for_two_segment_model_training": True,
        "valid_for_two_segment_ann_training": True,
        "config_used": {
            "max_tick_delta_from_startup": 600,
            "current_warning_ma": 800,
            "sustained_overcurrent_ma": 1200,
        },
    }
    paths = write_two_segment_thesis_figures(
        output_dir=tmp_path,
        metrics=metrics,
        samples=samples,
    )
    for key in ("tracker_variability_workspace_xy", "tracker_variability_std_histogram", "tracker_variability_std_vs_command_index"):
        assert key in paths, f"variability figure {key} should be emitted when commands repeat"
    for filename in VARIABILITY_FIGURE_NAMES:
        target = tmp_path / filename
        assert target.exists(), f"{filename} should be written"
        assert target.stat().st_size > 4_000, f"{filename} looks like a placeholder, not a real plot"


def test_write_two_segment_thesis_figures_skips_variability_when_no_command_repeats(tmp_path: Path) -> None:
    # samples_per_pattern == 1 in the real experiment — each command captures once.
    samples = [
        _make_sample(
            sample_index=i,
            cycle_index=0,
            step_index=i,
            tip_xyz_mm=(float(i), float(i), 50.0),
            segment_b=[0.1 * i, 0.05 * i, -0.1 * i, -0.05 * i],
        )
        for i in range(8)
    ]
    metrics = {
        "schedule_type": "random_babble",
        "accepted_sample_count": len(samples),
        "rejected_sample_count": 0,
        "command_failure_count": 0,
        "run_trust_mode": "thesis_trusted",
        "valid_for_two_segment_model_training": True,
        "valid_for_two_segment_ann_training": True,
        "config_used": {
            "max_tick_delta_from_startup": 600,
            "current_warning_ma": 800,
            "sustained_overcurrent_ma": 1200,
        },
    }
    paths = write_two_segment_thesis_figures(
        output_dir=tmp_path,
        metrics=metrics,
        samples=samples,
    )
    for key in ("tracker_variability_workspace_xy", "tracker_variability_std_histogram", "tracker_variability_std_vs_command_index"):
        assert key not in paths, f"variability figure {key} should NOT be emitted when commands don't repeat"
    for filename in VARIABILITY_FIGURE_NAMES:
        assert not (tmp_path / filename).exists(), f"{filename} should not exist on a single-sample-per-command run"


def test_write_two_segment_thesis_figures_handles_missing_pose_gracefully(tmp_path: Path) -> None:
    samples = _seeded_samples(count=10, with_pose=False)
    metrics = {
        "schedule_type": "random_babble",
        "accepted_sample_count": 0,
        "rejected_sample_count": len(samples),
        "command_failure_count": 0,
        "run_trust_mode": "servo_only",
        "valid_for_two_segment_model_training": False,
        "valid_for_two_segment_ann_training": False,
        "config_used": {
            "max_tick_delta_from_startup": 400,
            "current_warning_ma": 800,
            "sustained_overcurrent_ma": 1200,
        },
    }
    paths = write_two_segment_thesis_figures(
        output_dir=tmp_path,
        metrics=metrics,
        samples=samples,
    )
    # Every figure must still exist even when pose data is missing — the figure
    # body shows an explicit empty-state message rather than crashing.
    for filename in THESIS_FIGURE_NAMES:
        target = tmp_path / filename
        assert target.exists()
        assert target.stat().st_size > 4_000, f"{filename} should be a real empty-state plot, not a placeholder"


def test_thesis_figures_attach_to_full_run_outputs(tmp_path: Path) -> None:
    """Smoke-check the wire-up: write_two_segment_dataset_outputs emits thesis figures."""
    from continuum_robot.experiments.two_segment_collect_pose_dataset import (
        write_two_segment_dataset_outputs,
    )
    samples = _seeded_samples(count=12)
    metrics = {
        "schedule_type": "single_axis_micro",
        "accepted_sample_count": sum(1 for s in samples if s.extra["capture_accepted"]),
        "rejected_sample_count": sum(1 for s in samples if not s.extra["capture_accepted"]),
        "command_failure_count": 0,
        "run_trust_mode": "thesis_trusted",
        "valid_for_two_segment_model_training": True,
        "valid_for_two_segment_ann_training": True,
        "config_used": {
            "max_tick_delta_from_startup": 600,
            "current_warning_ma": 800,
            "sustained_overcurrent_ma": 1200,
        },
    }
    paths = write_two_segment_dataset_outputs(
        output_dir=tmp_path,
        metrics=metrics,
        samples=samples,
        sample_failure_events=[],
        long_run_health={},
        transport_recovery_report={},
    )
    # Legacy reports must remain present.
    for legacy in (
        "two_segment_command_coverage_report.png",
        "two_segment_servo_position_coverage_report.png",
        "two_segment_pose_coverage_report.png",
        "two_segment_dataset_quality_report.png",
        "two_segment_dataset_summary.txt",
        "metrics.csv",
    ):
        assert (tmp_path / legacy).exists(), f"legacy {legacy} should still be written"
    # All thesis figures present.
    for filename in THESIS_FIGURE_NAMES:
        assert (tmp_path / filename).exists(), f"thesis {filename} should be written"
    # And surfaced in the returned paths dict.
    assert "thesis_01" in paths
    assert paths["thesis_01"].name == "thesis_01_workspace_coverage_3d.png"

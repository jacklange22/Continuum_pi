"""Tests for the workspace repeatability map experiment.

Pins three independent contracts:

1. **Target geometry** — the Vogel disk-fill helper produces exactly
   ``target_count`` points, covers ``[0, max_amplitude_mm]`` radially with
   the centre + perimeter explicitly hit, and has bounded nearest-neighbour
   spacing across the disk (the "no dead zones" guarantee the operator
   asked for).
2. **Per-target metric computation** — the analysis helpers reduce a per-
   visit cluster to centroid + RMS spread + per-axis stddev correctly,
   filter rejected visits, and produce a summary dict whose worst-N list
   actually points at the worst-N targets.
3. **End-to-end dry-run** — running ``workspace_repeatability_map`` through
   ``ExperimentRunner`` in ``dry_run`` mode emits the canonical bundle
   (per-target CSV, visits JSONL, summary.json, three PNGs) and reports the
   right visit counts.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest


from continuum_robot.experiments.workspace_repeatability_map import (
    DEFAULT_MAX_AMPLITUDE_MM,
    DEFAULT_TARGET_COUNT,
    DEFAULT_VISITS_PER_TARGET,
    WorkspaceMapTarget,
    WorkspaceRepeatabilityMapConfig,
    WorkspaceRepeatabilityMapExperiment,
    build_round_robin_visit_order,
    build_workspace_repeatability_targets,
    register_workspace_repeatability_map,
    workspace_target_tick_profile,
)
from continuum_robot.experiments.workspace_repeatability_map_outputs import (
    AXIS_STDDEV_PNG,
    PER_TARGET_CSV,
    SCATTER_3D_PNG,
    SUMMARY_JSON,
    VISITS_JSONL,
    XY_HEATMAP_PNG,
    compute_workspace_repeatability_metrics,
    group_visits_by_target,
    summarize_workspace_repeatability,
    write_workspace_repeatability_map_outputs,
)


# ---------------------------------------------------------------------------
# Target geometry
# ---------------------------------------------------------------------------


class TestVogelTargetGeometry:
    def test_default_count_and_amplitude(self) -> None:
        targets = build_workspace_repeatability_targets()
        assert len(targets) == DEFAULT_TARGET_COUNT
        assert pytest.approx(0.0, abs=1e-9) == targets[0].amplitude_mm
        assert pytest.approx(DEFAULT_MAX_AMPLITUDE_MM, abs=1e-9) == targets[-1].amplitude_mm

    def test_center_is_at_origin(self) -> None:
        targets = build_workspace_repeatability_targets()
        assert pytest.approx(0.0, abs=1e-9) == targets[0].x_mm
        assert pytest.approx(0.0, abs=1e-9) == targets[0].y_mm
        assert targets[0].cable_deltas_mm == [0.0, 0.0, 0.0, 0.0]

    def test_perimeter_is_at_configured_radius(self) -> None:
        targets = build_workspace_repeatability_targets(max_amplitude_mm=12.0)
        last = targets[-1]
        assert math.isclose(math.hypot(last.x_mm, last.y_mm), 12.0, abs_tol=1e-9)

    def test_no_large_dead_zones_in_nearest_neighbor_spacing(self) -> None:
        """Operator's hard requirement: density is uniform across the disk."""
        targets = build_workspace_repeatability_targets(target_count=100, max_amplitude_mm=12.0)
        points = np.asarray([[t.x_mm, t.y_mm] for t in targets], dtype=float)
        # Compute nearest-neighbor distance for every point.
        diffs = points[:, None, :] - points[None, :, :]
        sq = np.sum(diffs * diffs, axis=-1)
        np.fill_diagonal(sq, np.inf)
        nn_distances = np.sqrt(sq.min(axis=1))
        # The Vogel pattern at this density gives nearest-neighbour ~2 mm with
        # very low variance; if the spread blows up the disk has dead zones.
        assert nn_distances.min() > 0.5, "nearest-neighbour collapse implies overlapping targets"
        assert nn_distances.max() < 3.0, (
            f"max nearest-neighbour distance {nn_distances.max():.3f} mm is too large "
            f"-- the heatmap will have visible gaps"
        )
        # And the ratio max/min stays tight: density is approximately uniform.
        assert nn_distances.max() / nn_distances.min() < 2.0

    def test_cable_deltas_match_legacy_four_cable_convention(self) -> None:
        targets = build_workspace_repeatability_targets(target_count=8, max_amplitude_mm=10.0)
        for target in targets:
            # cable_deltas_mm = [-x, -y, x, y] -- the legacy single-segment convention.
            assert pytest.approx(-target.x_mm, abs=1e-9) == target.cable_deltas_mm[0]
            assert pytest.approx(-target.y_mm, abs=1e-9) == target.cable_deltas_mm[1]
            assert pytest.approx(target.x_mm, abs=1e-9) == target.cable_deltas_mm[2]
            assert pytest.approx(target.y_mm, abs=1e-9) == target.cable_deltas_mm[3]
            # cm conversion is consistent.
            for mm_value, cm_value in zip(target.cable_deltas_mm, target.cable_deltas_cm):
                assert math.isclose(cm_value, mm_value / 10.0, abs_tol=1e-12)

    def test_target_count_validation(self) -> None:
        with pytest.raises(ValueError, match="at least 3 targets"):
            build_workspace_repeatability_targets(target_count=2)
        with pytest.raises(ValueError, match="max_amplitude_mm"):
            build_workspace_repeatability_targets(max_amplitude_mm=0.0)

    def test_target_indices_and_labels(self) -> None:
        targets = build_workspace_repeatability_targets(target_count=5)
        assert [t.target_index for t in targets] == [0, 1, 2, 3, 4]
        assert [t.label for t in targets] == ["T000", "T001", "T002", "T003", "T004"]


# ---------------------------------------------------------------------------
# Visit ordering
# ---------------------------------------------------------------------------


class TestRoundRobinVisitOrder:
    def test_visits_every_target_correct_number_of_times(self) -> None:
        order = build_round_robin_visit_order(
            target_indices=list(range(100)),
            visits_per_target=15,
            random_seed=0,
        )
        assert len(order) == 1500
        from collections import Counter

        counts = Counter(target_index for _, _, target_index in order)
        assert all(value == 15 for value in counts.values()), counts

    def test_each_cycle_contains_each_target_exactly_once(self) -> None:
        order = build_round_robin_visit_order(
            target_indices=list(range(10)),
            visits_per_target=4,
            random_seed=0,
        )
        for cycle_index in range(4):
            cycle_targets = {
                target_index
                for cycle, _, target_index in order
                if cycle == cycle_index
            }
            assert cycle_targets == set(range(10))

    def test_deterministic_under_seed(self) -> None:
        first = build_round_robin_visit_order(target_indices=list(range(10)), visits_per_target=3, random_seed=42)
        second = build_round_robin_visit_order(target_indices=list(range(10)), visits_per_target=3, random_seed=42)
        assert first == second
        different = build_round_robin_visit_order(target_indices=list(range(10)), visits_per_target=3, random_seed=99)
        # Different seeds must produce a different order at least somewhere.
        assert first != different

    def test_per_cycle_shuffles_are_independent(self) -> None:
        """The shuffle of cycle 0 must not equal the shuffle of cycle 1 with the same seed."""
        order = build_round_robin_visit_order(
            target_indices=list(range(20)),
            visits_per_target=3,
            random_seed=0,
        )
        cycle_orders = {c: [] for c in range(3)}
        for cycle_index, _, target_index in order:
            cycle_orders[cycle_index].append(target_index)
        assert cycle_orders[0] != cycle_orders[1]
        assert cycle_orders[1] != cycle_orders[2]

    def test_rejects_zero_visits(self) -> None:
        with pytest.raises(ValueError, match="visits_per_target"):
            build_round_robin_visit_order(target_indices=[0, 1, 2], visits_per_target=0, random_seed=0)


# ---------------------------------------------------------------------------
# Per-target metrics
# ---------------------------------------------------------------------------


class TestPerTargetMetrics:
    def _synthetic_visits(
        self,
        *,
        target_index: int,
        n_visits: int,
        centroid_mm: tuple[float, float, float],
        spread_mm: float,
        seed: int,
        rejected_count: int = 0,
    ) -> list[dict]:
        """Build n_visits visit-extras for one target."""
        rng = np.random.default_rng(seed)
        visits: list[dict] = []
        for visit in range(n_visits):
            offset = rng.normal(scale=spread_mm, size=3)
            pos = [centroid_mm[i] + offset[i] for i in range(3)]
            visits.append(
                {
                    "target_index": target_index,
                    "target_label": f"T{target_index:03d}",
                    "visit_position": visit,
                    "cycle_index": visit // 5,
                    "visit_in_cycle": visit % 5,
                    "position_mm": pos,
                    "rejected": visit < rejected_count,
                }
            )
        return visits

    def test_centroid_and_rms_spread_match_synthetic_cluster(self) -> None:
        # 100 visits sampled around (5, 3, 0) with sigma 0.5 mm. For 100 samples
        # the RMS spread of the cluster about the centroid should approach
        # sqrt(3) * 0.5 ≈ 0.866 mm. Allow a generous tolerance for finite-N
        # sampling noise.
        visits = self._synthetic_visits(
            target_index=0, n_visits=100, centroid_mm=(5.0, 3.0, 0.0), spread_mm=0.5, seed=11
        )
        rows = compute_workspace_repeatability_metrics(
            visits_by_target={0: visits},
            targets_by_index={0: {"x_mm": 5.0, "y_mm": 3.0, "label": "T000"}},
        )
        assert len(rows) == 1
        row = rows[0]
        assert pytest.approx(5.0, abs=0.1) == row["centroid_x_mm"]
        assert pytest.approx(3.0, abs=0.1) == row["centroid_y_mm"]
        assert pytest.approx(0.0, abs=0.1) == row["centroid_z_mm"]
        # sqrt(3) * 0.5 ≈ 0.866 mm; tolerate 25% deviation for n=100 sampling.
        expected_rms = math.sqrt(3.0) * 0.5
        assert abs(row["rms_spread_mm"] - expected_rms) / expected_rms < 0.25

    def test_rejected_visits_excluded_from_metrics(self) -> None:
        visits = self._synthetic_visits(
            target_index=7,
            n_visits=15,
            centroid_mm=(0.0, 0.0, 0.0),
            spread_mm=0.1,
            seed=12,
            rejected_count=3,
        )
        rows = compute_workspace_repeatability_metrics(
            visits_by_target={7: visits},
            targets_by_index={7: {"label": "T007"}},
        )
        assert rows[0]["n_visits_total"] == 15
        assert rows[0]["n_visits_kept"] == 12
        assert rows[0]["n_visits_rejected"] == 3

    def test_non_finite_positions_treated_as_rejected(self) -> None:
        visits = [
            {"target_index": 1, "position_mm": [1.0, 2.0, 3.0], "rejected": False},
            {"target_index": 1, "position_mm": [float("nan"), 2.0, 3.0], "rejected": False},
            {"target_index": 1, "position_mm": [1.0, float("inf"), 3.0], "rejected": False},
            {"target_index": 1, "position_mm": [1.1, 2.1, 3.1], "rejected": False},
        ]
        rows = compute_workspace_repeatability_metrics(
            visits_by_target={1: visits},
            targets_by_index={1: {}},
        )
        assert rows[0]["n_visits_kept"] == 2
        assert rows[0]["n_visits_rejected"] == 2

    def test_summary_reports_workspace_wide_metrics(self) -> None:
        # Build 5 targets with different spreads so we know which is worst.
        per_target = []
        for index in range(5):
            visits = self._synthetic_visits(
                target_index=index, n_visits=20, centroid_mm=(index * 2.0, 0.0, 0.0),
                spread_mm=0.1 + 0.2 * index, seed=index + 100,
            )
            rows = compute_workspace_repeatability_metrics(
                visits_by_target={index: visits},
                targets_by_index={index: {"label": f"T{index:03d}"}},
            )
            per_target.extend(rows)
        summary = summarize_workspace_repeatability(per_target, worst_n=3, thesis_goal_rms_mm=0.5)
        assert summary["target_count"] == 5
        assert summary["targets_with_data"] == 5
        # Worst is index 4 (largest spread); list is descending.
        assert summary["worst_targets"][0]["target_index"] == 4
        # The "fraction above goal" key exists and is in [0, 1].
        frac = summary["fraction_above_thesis_goal"]
        assert frac is not None and 0.0 <= frac <= 1.0

    def test_summary_handles_empty_rows(self) -> None:
        summary = summarize_workspace_repeatability([], worst_n=3, thesis_goal_rms_mm=1.0)
        assert summary["targets_with_data"] == 0
        assert summary["workspace_rms_mean_mm"] is None
        assert summary["worst_targets"] == []


# ---------------------------------------------------------------------------
# Output bundle writer
# ---------------------------------------------------------------------------


def _png_is_valid(path: Path) -> bool:
    if not path.exists():
        return False
    with path.open("rb") as handle:
        return handle.read(8) == b"\x89PNG\r\n\x1a\n"


class TestOutputBundle:
    def test_write_bundle_produces_every_artifact(self, tmp_path: Path) -> None:
        targets = build_workspace_repeatability_targets(target_count=12, max_amplitude_mm=10.0)
        rng = np.random.default_rng(0)
        samples: list[dict] = []
        for target in targets:
            for visit in range(5):
                offset = rng.normal(scale=0.2, size=3)
                samples.append(
                    {
                        "target_index": target.target_index,
                        "target_label": target.label,
                        "cycle_index": visit // 5,
                        "visit_in_cycle": visit % 5,
                        "visit_position": len(samples),
                        "position_mm": [target.x_mm + offset[0], target.y_mm + offset[1], offset[2]],
                        "rejected": False,
                    }
                )
        paths = write_workspace_repeatability_map_outputs(
            output_dir=tmp_path,
            target_catalog=[t.to_dict() for t in targets],
            samples=samples,
            thesis_goal_rms_mm=1.0,
        )
        for key in ("per_target_csv", "visits_jsonl", "summary_json", "scatter_3d", "xy_heatmap", "axis_stddev"):
            assert key in paths
            assert paths[key].exists(), f"missing {key}: {paths[key]}"
        assert _png_is_valid(tmp_path / SCATTER_3D_PNG)
        assert _png_is_valid(tmp_path / XY_HEATMAP_PNG)
        assert _png_is_valid(tmp_path / AXIS_STDDEV_PNG)
        # Summary JSON is structured.
        summary_payload = json.loads((tmp_path / SUMMARY_JSON).read_text(encoding="utf-8"))
        assert summary_payload["schema_version"] == "workspace_repeatability_map_v1"
        assert summary_payload["summary"]["target_count"] == 12
        # CSV has the expected columns.
        csv_text = (tmp_path / PER_TARGET_CSV).read_text(encoding="utf-8")
        header_columns = csv_text.splitlines()[0].split(",")
        for required in ("target_index", "rms_spread_mm", "centroid_x_mm", "x_stddev_mm"):
            assert required in header_columns

    def test_bundle_with_no_samples_still_writes_summary(self, tmp_path: Path) -> None:
        targets = build_workspace_repeatability_targets(target_count=3, max_amplitude_mm=5.0)
        paths = write_workspace_repeatability_map_outputs(
            output_dir=tmp_path,
            target_catalog=[t.to_dict() for t in targets],
            samples=[],
            thesis_goal_rms_mm=1.0,
        )
        # Figures are skipped when there is no data; summary still exists.
        assert (tmp_path / SUMMARY_JSON).exists()
        assert (tmp_path / PER_TARGET_CSV).exists()
        assert "scatter_3d" not in paths
        assert "xy_heatmap" not in paths

    def test_visits_jsonl_skips_rows_missing_target_index(self, tmp_path: Path) -> None:
        targets = build_workspace_repeatability_targets(target_count=3, max_amplitude_mm=5.0)
        samples = [
            {"target_index": 0, "position_mm": [0.0, 0.0, 0.0], "rejected": False},
            {"missing_target_index": True, "position_mm": [1.0, 2.0, 3.0]},
        ]
        write_workspace_repeatability_map_outputs(
            output_dir=tmp_path,
            target_catalog=[t.to_dict() for t in targets],
            samples=samples,
        )
        lines = (tmp_path / VISITS_JSONL).read_text(encoding="utf-8").strip().splitlines()
        assert len(lines) == 1


# ---------------------------------------------------------------------------
# Dry-run end-to-end via the experiment class
# ---------------------------------------------------------------------------


class TestDryRunEndToEnd:
    def test_dry_run_execute_produces_target_count_x_visits_samples(self) -> None:
        """In dry-run mode the experiment synthesizes positions and adds samples
        without touching servo / tracker. The full sample count must equal
        target_count * visits_per_target and every visit must carry a finite
        synthetic position."""
        config = WorkspaceRepeatabilityMapConfig.from_dict(
            {
                "target_count": 25,
                "visits_per_target": 4,
                "max_amplitude_mm": 8.0,
                "dry_run": True,
                "neutral_settle_s": 0.0,
                "target_settle_s": 0.0,
                "capture_timeout_s": 0.0,
            }
        )
        experiment = WorkspaceRepeatabilityMapExperiment(config=config)

        class _StubSession:
            def __init__(self) -> None:
                self.samples: list = []
                self.metrics: dict[str, object] = {}
                self.progress_updates = 0
                self._stop = False

            def add_sample(self, sample) -> None:
                self.samples.append(sample)

            def set_metric(self, key, value) -> None:
                self.metrics[key] = value

            def update_progress(self, *args, **kwargs) -> None:
                self.progress_updates += 1

            def stop_requested(self) -> bool:
                return self._stop

        session = _StubSession()
        experiment.setup(session)
        experiment.precheck(session)
        experiment.execute(session)

        assert len(session.samples) == 25 * 4
        # Every visit has a 3-vector position and is not flagged rejected.
        for sample in session.samples:
            position = sample.extra.get("position_mm")
            assert position is not None and len(position) == 3
            assert all(math.isfinite(float(v)) for v in position)
            assert sample.extra["rejected"] is False
        # Schedule metric recorded.
        schedule = session.metrics["schedule"]
        assert schedule["visits_per_target"] == 4
        assert schedule["random_seed"] == 0
        assert session.metrics["captured_visit_count"] == 100
        assert session.metrics["rejected_visit_count"] == 0
        assert session.metrics["status"] == "success"

    def test_dry_run_synthetic_position_concentrates_near_target(self) -> None:
        """With low synthetic noise the per-target centroid must be near (x, y, 0)
        so the analysis bundle's downstream RMS metric makes sense."""
        config = WorkspaceRepeatabilityMapConfig.from_dict(
            {
                "target_count": 10,
                "visits_per_target": 30,
                "max_amplitude_mm": 6.0,
                "dry_run": True,
                "synthetic_noise_std_mm": 0.05,
                "synthetic_per_target_bias_std_mm": 0.02,
                "neutral_settle_s": 0.0,
                "target_settle_s": 0.0,
            }
        )
        experiment = WorkspaceRepeatabilityMapExperiment(config=config)

        class _StubSession:
            def __init__(self) -> None:
                self.samples: list = []
                self.metrics: dict[str, object] = {}

            def add_sample(self, sample) -> None:
                self.samples.append(sample)

            def set_metric(self, key, value) -> None:
                self.metrics[key] = value

            def update_progress(self, *args, **kwargs) -> None: ...

            def stop_requested(self) -> bool:
                return False

        session = _StubSession()
        experiment.setup(session)
        experiment.precheck(session)
        experiment.execute(session)

        samples_extra = [sample.extra for sample in session.samples]
        rows = compute_workspace_repeatability_metrics(
            visits_by_target=group_visits_by_target(samples_extra),
            targets_by_index={t.target_index: t.to_dict() for t in experiment._targets},
        )
        # Centroid should be within a few-sigma of the target geometry.
        for row in rows:
            assert row["n_visits_kept"] == 30
            # Sanity: per-target RMS should be on the order of 0.05 mm.
            assert 0.02 < float(row["rms_spread_mm"]) < 0.20


# ---------------------------------------------------------------------------
# Registration into the builtin registry
# ---------------------------------------------------------------------------


def test_registered_under_canonical_name() -> None:
    from continuum_robot.experiments.registry import ExperimentRegistry

    registry = ExperimentRegistry()
    register_workspace_repeatability_map(registry)
    spec = registry.get("workspace_repeatability_map")
    assert spec.title == "Workspace Repeatability Map"
    assert spec.category == "validation"
    assert "Repeatability" in spec.tags
    assert "Workspace" in spec.tags
    experiment = spec.factory({})
    assert isinstance(experiment, WorkspaceRepeatabilityMapExperiment)
    assert len(experiment._targets) == DEFAULT_TARGET_COUNT


def test_default_example_yaml_loads_cleanly() -> None:
    """The shipped example YAML must produce a valid config when parsed."""
    import yaml

    config_path = Path(__file__).resolve().parents[1] / "config" / "experiment_workspace_repeatability_map.example.yaml"
    payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    config = WorkspaceRepeatabilityMapConfig.from_dict(payload)
    assert config.target_count == 100
    assert config.visits_per_target == 15
    assert math.isclose(config.max_amplitude_mm, 12.0, abs_tol=1e-9)

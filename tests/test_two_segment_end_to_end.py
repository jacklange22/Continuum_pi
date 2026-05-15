"""End-to-end integration smoke for the two-segment foundation.

Wires startup-validation → collect-pose → Mike CC convention probe →
repeatability → evidence index → export bundle. The point is to catch
integration drift between components that pass their own unit tests in
isolation but disagree at the handoff (e.g. a metric name renamed in one
place but not the other).

Mock-only. Does not exercise hardware. If a real-hardware run later behaves
differently, that's a hardware-day issue — but at minimum every artifact
expected by the operator workflow should be reachable on bench day 0.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import yaml

from continuum_robot.data.build_thesis_evidence_index import build_thesis_evidence_index
from continuum_robot.data.export_run_bundle import export_run_bundle
from continuum_robot.modeling.two_segment.validate_mike_cc import main as mike_probe_main

from tests.test_two_segment_collect_pose_dataset import (
    EXPERIMENT_NAME as COLLECT_POSE_NAME,
    _FakeTrackingService,
    _runner as _collect_pose_runner,
    _save_all8_startup,
    _servo_service,
    _settings,
    _tracking_snapshot,
)
from tests.test_two_segment_repeatability import EXPERIMENT_NAME as REPEATABILITY_NAME
from tests.test_two_segment_startup_validation import EXPERIMENT_NAME as STARTUP_NAME


def _mark_runs_as_thesis_candidates(*run_dirs: Path) -> None:
    """Pretend the operator marked each run as advisor-shareable so the evidence index includes them."""
    for run_dir in run_dirs:
        (run_dir / "run_review.json").write_text(
            json.dumps({"review_status": "thesis_candidate", "include_in_evidence_index": True}),
            encoding="utf-8",
        )


def _promote_mock_run_to_real_experiments_tree(run_dir: Path, project_root: Path) -> Path:
    """Copy a mock-mode run from data/mock_experiments/<exp>/<run> to data/experiments/<exp>/<run>.

    The thesis evidence index discovers runs under ``data/experiments/`` only.
    Mock-mode runs land under ``data/mock_experiments/``; promoting them here
    lets the integration smoke confirm the evidence-index handoff works without
    forcing the experiment runner out of mock mode.
    """
    try:
        relative = run_dir.resolve().relative_to((project_root / "data" / "mock_experiments").resolve())
    except ValueError:
        return run_dir
    target = (project_root / "data" / "experiments" / relative).resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        shutil.rmtree(target)
    shutil.copytree(run_dir, target)
    return target


def test_two_segment_pipeline_end_to_end_smoke(tmp_path: Path) -> None:
    settings = _settings()
    service = _servo_service(tmp_path, settings=settings)
    tracking = _FakeTrackingService(_tracking_snapshot(include_distal=True, include_intermediate=False))

    # 1) Startup validation (separate runner instance is fine; they share the same servo service).
    from tests.test_two_segment_startup_validation import _runner as _startup_runner

    startup_runner = _startup_runner(tmp_path, settings=settings)
    # The startup runner builds its own servo service from MockDxlBus; the
    # collect-pose runner needs a service with an accepted all-8 startup
    # artifact. To keep the pipeline aligned, we run startup-validation through
    # its own service then mirror the accepted-startup state into the
    # collect-pose service's neutral calibration.
    startup_result = startup_runner.run_experiment(STARTUP_NAME, config={})
    assert startup_result.success is True
    assert startup_result.summary.experiment_metrics["final_accepted"] is True

    # Ensure the collect-pose / repeatability service has a real all-8 startup
    # artifact saved (use the existing helper).
    _save_all8_startup(service)

    # 2) Collect-pose run on the collect-pose service.
    collect_runner = _collect_pose_runner(tmp_path, settings=settings, service=service, tracking_service=tracking)
    collect_result = collect_runner.run_experiment(
        COLLECT_POSE_NAME,
        config={
            "dry_run": False,
            "allow_servo_only_test_run": True,
            "run_trust_mode": "servo_only",
            "schedule_type": "single_axis_micro",
            "max_segment_displacement_cm": 0.02,
            "max_tick_delta_from_startup": 320,
            "samples_per_pattern": 1,
            "capture_repeats": 1,
            "long_run_recovery_enabled": True,
        },
    )
    assert collect_result.success is True
    collect_dir = collect_result.paths.output_dir
    assert (collect_dir / "samples.jsonl").exists()
    assert (collect_dir / "long_run_health.json").exists()
    assert (collect_dir / "transport_recovery_report.json").exists()
    assert (collect_dir / "sample_failure_events.jsonl").exists()
    long_run = json.loads((collect_dir / "long_run_health.json").read_text(encoding="utf-8"))
    assert long_run["schema_version"] == "two_segment_long_run_health_v1"
    metrics = collect_result.summary.experiment_metrics
    assert metrics["current_load_summary"]["schema_version"] == "two_segment_current_load_summary_v1"

    # 3) Mike CC convention probe consumes the collect-pose run.
    probe_config = {
        "physics_models": {
            "global": {"tendon_displacement_sign_convention": {"value": "positive_shortens_tendon", "source": "design"}},
            "segments": {
                "segment_a": {
                    "segment_length_mm": {"value": 66.0, "units": "mm", "source": "design"},
                    "tendon_positions_mm": {
                        "value": [[4.12, 0.0], [0.0, 4.12], [-4.12, 0.0], [0.0, -4.12]],
                        "units": "mm",
                        "source": "design",
                    },
                },
                "segment_b": {
                    "segment_length_mm": {"value": 66.0, "units": "mm", "source": "design"},
                    "tendon_positions_mm": {
                        "value": [[4.12, 0.0], [0.0, 4.12], [-4.12, 0.0], [0.0, -4.12]],
                        "units": "mm",
                        "source": "design",
                    },
                },
            },
        }
    }
    probe_config_path = tmp_path / "modeling_two_segment.yaml"
    probe_config_path.write_text(yaml.safe_dump(probe_config), encoding="utf-8")
    probe_out_dir = tmp_path / "data" / "experiments" / "two_segment_mike_convention_probe" / "smoke"

    # The collect-pose run was servo-only (mock fixture), so samples are filtered out by the
    # default loader. Pass --allow-lower-trust so the probe operates on the available samples.
    rc = mike_probe_main(
        [
            "--runs",
            str(collect_dir),
            "--config",
            str(probe_config_path),
            "--output-dir",
            str(probe_out_dir),
            "--allow-lower-trust",
        ]
    )
    # We don't pin the recommendation (random mock pose) — only that the report is generated.
    assert rc in {0, 2}
    assert (probe_out_dir / "mike_cc_convention_report.json").exists()
    assert (probe_out_dir / "mike_cc_convention_report.txt").exists()

    # 4) Repeatability run on the same service.
    from tests.test_two_segment_repeatability import _runner as _repeatability_runner

    repeat_runner = _repeatability_runner(tmp_path, settings=settings, service=service, tracking_service=tracking)
    repeat_result = repeat_runner.run_experiment(
        REPEATABILITY_NAME,
        config={
            "bottom_inner_radius_cm": 0.02,
            "bottom_outer_radius_cm": 0.04,
            "top_inner_radius_cm": 0.0,
            "top_outer_radius_cm": 0.0,
            "points_per_ring": 2,
            "repeat_visits": 1,
            "samples_per_visit": 1,
            "settle_time_s": 0.0,
            "allow_servo_only_test_run": True,
            "run_trust_mode": "servo_only",
            "max_tick_delta_from_startup": 320,
        },
    )
    assert repeat_result.success is True
    repeat_dir = repeat_result.paths.output_dir
    assert (repeat_dir / "two_segment_repeatability_summary.txt").exists()
    assert (repeat_dir / "two_segment_repeatability_scatter_metrics.json").exists()
    repeat_metrics = repeat_result.summary.experiment_metrics
    assert repeat_metrics["current_load_summary"]["schema_version"] == "two_segment_current_load_summary_v1"

    # 5) Promote mock-mode runs to the canonical experiments tree the evidence
    # index scans, then mark them as thesis candidates so they survive review
    # filtering. Mock-mode is still recorded honestly in each run's metadata.
    startup_dir_canonical = _promote_mock_run_to_real_experiments_tree(startup_result.paths.output_dir, tmp_path)
    collect_dir_canonical = _promote_mock_run_to_real_experiments_tree(collect_dir, tmp_path)
    repeat_dir_canonical = _promote_mock_run_to_real_experiments_tree(repeat_dir, tmp_path)
    _mark_runs_as_thesis_candidates(startup_dir_canonical, collect_dir_canonical, repeat_dir_canonical)
    # Mock-mode runs are excluded by default; include them for this smoke since
    # the only data we have is mock.
    index_dir = build_thesis_evidence_index(project_root=tmp_path, include_mock=True, include_unreviewed=False)
    index_payload = json.loads((index_dir / "thesis_evidence_index.json").read_text(encoding="utf-8"))
    experiments = index_payload["experiments"]
    assert "two_segment_startup_validation" in experiments
    assert "two_segment_collect_pose_command_dataset" in experiments
    assert "two_segment_repeatability" in experiments
    startup_entry = experiments["two_segment_startup_validation"][0]["key_metrics"]
    assert "two_segment_startup" in startup_entry
    collect_entry = experiments["two_segment_collect_pose_command_dataset"][0]["key_metrics"]
    assert "two_segment_collect_pose" in collect_entry
    repeat_entry = experiments["two_segment_repeatability"][0]["key_metrics"]
    assert "two_segment_repeatability" in repeat_entry

    # 6) Export the collect-pose bundle and confirm long-run files travel.
    export = export_run_bundle(
        run_dir=collect_dir_canonical,
        output_root=tmp_path / "exports",
        project_root=tmp_path,
        include_samples=True,
    )
    exported = {entry.bundle_path for entry in export.entries}
    assert "long_run_health.json" in exported
    assert "transport_recovery_report.json" in exported
    assert "sample_failure_events.jsonl" in exported
    assert "samples.jsonl" in exported

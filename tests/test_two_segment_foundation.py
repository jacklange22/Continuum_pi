from __future__ import annotations

import json
from pathlib import Path

import pytest

from continuum_robot.config.schemas import RobotConfig, RobotSegmentConfig
from continuum_robot.data.run_management import detail_pairs_for_run, summarize_run
from continuum_robot.data.validate_run_bundle import validate_run_folder
from continuum_robot.experiments.schemas import ExperimentTimeseriesSample
from continuum_robot.two_segment import (
    TwoSegmentCommand,
    TwoSegmentPoseObservation,
    build_two_segment_foundation_metadata,
    two_segment_command_schema,
    two_segment_pose_schema,
)


def _segments() -> dict[str, RobotSegmentConfig]:
    return {
        "segment_a": RobotSegmentConfig(
            key="segment_a",
            label="Spine 1",
            segment_label="Segment A",
            segment_role="proximal",
            segment_order_index=0,
            servo_ids=[1, 2, 3, 4],
            pairs={"axis_a": [1, 3], "axis_b": [2, 4]},
        ),
        "segment_b": RobotSegmentConfig(
            key="segment_b",
            label="Spine 2",
            segment_label="Segment B",
            segment_role="distal",
            segment_order_index=1,
            servo_ids=[5, 6, 7, 8],
            pairs={"axis_a": [5, 7], "axis_b": [6, 8]},
        ),
    }


def _dual_context():
    return RobotConfig(mode="dual_segment", segments=_segments()).operating_context()


def test_two_segment_command_round_trips_between_segment_mapping_servo_mapping_and_flat_vector() -> None:
    context = _dual_context()
    command = TwoSegmentCommand.from_mapping(
        {
            "segment_a": [0.1, 0.2, 0.3, 0.4],
            "segment_b": [-0.1, -0.2, -0.3, -0.4],
        }
    )

    assert command.to_segment_mapping() == {
        "segment_a": [0.1, 0.2, 0.3, 0.4],
        "segment_b": [-0.1, -0.2, -0.3, -0.4],
    }
    assert command.to_servo_mapping(context=context) == {
        1: 0.1,
        2: 0.2,
        3: 0.3,
        4: 0.4,
        5: -0.1,
        6: -0.2,
        7: -0.3,
        8: -0.4,
    }
    assert command.to_flat(context=context) == [0.1, 0.2, 0.3, 0.4, -0.1, -0.2, -0.3, -0.4]

    round_trip = TwoSegmentCommand.from_flat(command.to_flat(context=context), context=context)
    assert round_trip.to_segment_mapping() == command.to_segment_mapping()


def test_two_segment_command_requires_dual_segment_context_and_four_values_per_segment() -> None:
    with pytest.raises(ValueError, match="segment_a command requires exactly 4 values"):
        TwoSegmentCommand.from_mapping({"segment_a": [0.0], "segment_b": [0.0, 0.0, 0.0, 0.0]})

    command = TwoSegmentCommand.from_mapping(
        {"segment_a": [0.0, 0.0, 0.0, 0.0], "segment_b": [0.0, 0.0, 0.0, 0.0]}
    )
    single_context = RobotConfig(mode="single_segment", segments=_segments()).operating_context()
    with pytest.raises(ValueError, match="require operating_mode=dual_segment"):
        command.to_flat(context=single_context)


def test_two_segment_schema_metadata_records_command_pose_and_blocked_capabilities() -> None:
    context = _dual_context()

    command_schema = two_segment_command_schema(context)
    pose_schema = two_segment_pose_schema(tool_roles={"0A": "distal_tip", "0C": "intermediate_pose"})
    foundation = build_two_segment_foundation_metadata(
        context,
        tool_roles={"0A": "distal_tip", "0C": "intermediate_pose"},
    )

    assert command_schema["canonical_representation"] == {
        "segment_a": ["a1", "a2", "a3", "a4"],
        "segment_b": ["b1", "b2", "b3", "b4"],
    }
    assert command_schema["commanded_servo_ids"] == [1, 2, 3, 4, 5, 6, 7, 8]
    assert pose_schema["tool_roles"] == {"0A": "distal_tip", "0C": "intermediate_pose"}
    assert foundation["available"] is True
    assert foundation["semantic_scope"] == "data_structures_and_metadata_only"
    assert foundation["segments"]["segment_a"]["segment_role"] == "proximal"
    assert foundation["segments"]["segment_b"]["segment_role"] == "distal"
    assert foundation["blocked_capabilities"]["two_segment_kinematics_control"] is True
    assert foundation["blocked_capabilities"]["automatic_two_segment_pretension"] is True


def test_two_segment_pose_observation_and_timeseries_sample_have_explicit_optional_slots() -> None:
    pose = TwoSegmentPoseObservation(
        distal_tip_pose={"translation_mm": [1.0, 2.0, 3.0]},
        intermediate_pose={"translation_mm": [0.5, 1.0, 1.5]},
        tool_roles={"0A": "distal_tip", "0C": "intermediate_pose"},
        status="future_fixture",
    ).to_dict()
    sample = ExperimentTimeseriesSample(
        monotonic_time_s=1.0,
        wall_time_utc="2026-01-02T00:00:00Z",
        phase="dry_schema",
        step_index=0,
        sample_index=0,
        two_segment_command=TwoSegmentCommand.from_mapping(
            {"segment_a": [0.0, 0.0, 0.0, 0.0], "segment_b": [0.0, 0.0, 0.0, 0.0]}
        ).to_record(),
        two_segment_pose=pose,
    )

    payload = sample.to_dict()

    assert payload["two_segment_command"]["segments"]["segment_a"] == [0.0, 0.0, 0.0, 0.0]
    assert payload["two_segment_pose"]["distal_tip_pose"]["translation_mm"] == [1.0, 2.0, 3.0]


def test_two_segment_collect_pose_run_surfaces_long_run_health_in_detail_pairs(tmp_path: Path) -> None:
    context = _dual_context()
    foundation = build_two_segment_foundation_metadata(context)
    run_dir = (
        tmp_path / "data" / "experiments"
        / "two_segment_collect_pose_command_dataset"
        / "20260515_120000_two_segment_collect_pose_command_dataset"
    )
    run_dir.mkdir(parents=True)
    metadata = {
        "experiment_name": "two_segment_collect_pose_command_dataset",
        "run_id": run_dir.name,
        "trust_info": {"run_trust_mode": "thesis_trusted"},
        "provenance_info": {"operating_mode": "dual_segment", "two_segment_foundation": foundation},
    }
    summary = {
        "experiment_name": "two_segment_collect_pose_command_dataset",
        "run_id": run_dir.name,
        "success": True,
        "status": "success",
        "experiment_metrics": {
            "dataset_type": "two_segment_collect_pose_command_dataset",
            "run_trust_mode": "thesis_trusted",
            "run_provenance": metadata["provenance_info"],
        },
    }
    (run_dir / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
    (run_dir / "summary.json").write_text(json.dumps(summary), encoding="utf-8")
    (run_dir / "long_run_health.json").write_text(
        json.dumps({
            "schema_version": "two_segment_long_run_health_v1",
            "stop_reason": "target_valid_sample_count_reached",
            "cycles_completed": 2,
            "accepted_samples": 100,
            "rejected_samples": 5,
            "command_failures": 1,
            "transport_failures": 2,
            "continue_until_valid_samples": True,
            "target_valid_sample_count": 100,
        }),
        encoding="utf-8",
    )

    managed = summarize_run(run_dir)
    detail = dict(detail_pairs_for_run(managed, project_root=tmp_path))

    assert "Long-Run Health" in detail
    text = detail["Long-Run Health"]
    assert "stop=target_valid_sample_count_reached" in text
    assert "accepted=100" in text
    assert "target=100" in text


def test_two_segment_repeatability_run_surfaces_scatter_metrics_in_detail_pairs(tmp_path: Path) -> None:
    run_dir = tmp_path / "data" / "experiments" / "two_segment_repeatability" / "20260515_120500_two_segment_repeatability"
    run_dir.mkdir(parents=True)
    metadata = {"experiment_name": "two_segment_repeatability", "run_id": run_dir.name, "trust_info": {"run_trust_mode": "thesis_trusted"}, "provenance_info": {"operating_mode": "dual_segment"}}
    summary = {"experiment_name": "two_segment_repeatability", "run_id": run_dir.name, "success": True, "status": "success", "experiment_metrics": {"run_trust_mode": "thesis_trusted"}}
    (run_dir / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
    (run_dir / "summary.json").write_text(json.dumps(summary), encoding="utf-8")
    (run_dir / "two_segment_repeatability_scatter_metrics.json").write_text(
        json.dumps({"aggregate_distal_rms_mm": 2.345, "aggregate_intermediate_rms_mm": 1.234}),
        encoding="utf-8",
    )

    managed = summarize_run(run_dir)
    detail = dict(detail_pairs_for_run(managed, project_root=tmp_path))

    assert "Two-Segment Repeatability Scatter" in detail
    assert "distal_rms_mm=2.345" in detail["Two-Segment Repeatability Scatter"]
    assert "intermediate_rms_mm=1.234" in detail["Two-Segment Repeatability Scatter"]


def test_dual_segment_run_provenance_validator_and_summary_show_foundation_metadata(tmp_path: Path) -> None:
    context = _dual_context()
    foundation = build_two_segment_foundation_metadata(context)
    run_dir = tmp_path / "data" / "experiments" / "manual_startup_review" / "20260102_000000_manual_startup_review"
    run_dir.mkdir(parents=True)
    metadata = {
        "experiment_name": "manual_startup_review",
        "run_id": run_dir.name,
        "timestamp_utc": "2026-01-02T00:00:00Z",
        "trust_info": {
            "run_trust_mode": "debug",
            "valid_for_model_training": False,
            "valid_for_thesis_repeatability": False,
        },
        "provenance_info": {
            "hardware_profile": "robot_8servo.yaml",
            "operating_mode": "dual_segment",
            "two_segment_foundation": foundation,
        },
    }
    summary = {
        "experiment_name": "manual_startup_review",
        "run_id": run_dir.name,
        "success": True,
        "status": "success",
        "experiment_metrics": {
            "run_trust_mode": "debug",
            "valid_for_model_training": False,
            "valid_for_thesis_repeatability": False,
            "run_provenance": metadata["provenance_info"],
        },
    }
    (run_dir / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
    (run_dir / "summary.json").write_text(json.dumps(summary), encoding="utf-8")

    validation = validate_run_folder(run_dir)
    managed = summarize_run(run_dir)
    detail = dict(detail_pairs_for_run(managed, project_root=tmp_path))

    assert validation.status == "PASS"
    assert "Segment A proximal" in managed.two_segment_summary
    assert "Segment B distal" in managed.two_segment_summary
    assert "data_structures_and_metadata_only" in detail["Two-Segment Foundation"]

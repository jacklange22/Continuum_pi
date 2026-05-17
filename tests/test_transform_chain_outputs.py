from __future__ import annotations

import json
from types import SimpleNamespace

from continuum_robot.experiments.transform_chain_outputs import (
    build_transform_chain_summary,
    write_transform_chain_outputs,
)
from continuum_robot.services.models import ServiceHealthSnapshot, ToolTrackingSnapshot, TrackingSnapshot


def _snapshot(tmp_path):
    registration_path = tmp_path / "latest_registration.json"
    registration_path.write_text("{}", encoding="utf-8")
    matrix = [
        [1.0, 0.0, 0.0, 1.0],
        [0.0, 1.0, 0.0, 2.0],
        [0.0, 0.0, 1.0, 3.0],
        [0.0, 0.0, 0.0, 1.0],
    ]
    return TrackingSnapshot(
        health=ServiceHealthSnapshot(name="tracking", health="healthy", state="tracking", status="ok"),
        connection_state="tracking",
        canonical_state="streaming_healthy",
        backend_identity="ndi_tracker_python",
        configured_backend_name="ndi",
        selected_backend_name="ndi",
        tracker_data_age_s=0.01,
        last_frame_number=42,
        tools={
            "0A": ToolTrackingSnapshot(
                tool_id="0A",
                present=True,
                valid=True,
                tracking_state="tracked",
                translation_mm=(1.0, 2.0, 3.0),
            )
        },
        registration_state="loaded",
        registration_path=str(registration_path),
        stored_registration_timestamp_utc="2026-04-25T12:00:00Z",
        stored_registration_fre_mm=0.42,
        T_robot_aurora=matrix,
        runtime_tip_mode="coil_as_tip",
        runtime_tip_trust_level="thesis_trusted",
        runtime_tip_mode_message="Coil-as-tip is active.",
        runtime_tip_calibration_state="coil_as_tip",
        runtime_tip_identity_fallback=True,
        runtime_tip_selected_artifact_kind="coil_as_tip",
        tip_pose_status="coil_as_tip",
        T_robot_tip=matrix,
    )


def test_build_transform_chain_summary_records_policy(tmp_path) -> None:
    summary = build_transform_chain_summary(
        snapshot=_snapshot(tmp_path),
        workflow="single_segment_repeatability",
    )

    assert summary["runtime_tip"]["mode"] == "coil_as_tip"
    assert summary["runtime_tip"]["trust_label"] == "thesis_trusted"
    assert summary["runtime_tip"]["thesis_trusted"] is True
    assert summary["registration"]["fre_mm"] == 0.42
    assert summary["transform_chain"]["expression"].startswith("T_robot_tip")


def test_write_transform_chain_outputs_emits_only_json(tmp_path) -> None:
    paths = write_transform_chain_outputs(
        output_dir=tmp_path / "out",
        snapshot=_snapshot(tmp_path),
        workflow="transform_chain_validation",
    )

    assert paths["json"].exists()
    # .txt and overview .png duplicated the JSON and had no consumers; gone.
    assert "txt" not in paths
    assert "png" not in paths
    assert not (paths["json"].parent / "transform_chain_summary.txt").exists()
    assert not (paths["json"].parent / "transform_chain_overview.png").exists()
    payload = json.loads(paths["json"].read_text(encoding="utf-8"))
    assert payload["runtime_tip"]["policy"]["trust_label"] == "thesis_trusted"

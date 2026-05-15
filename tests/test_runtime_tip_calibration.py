import json
from pathlib import Path

import numpy as np

from continuum_robot.hardware.mock_aurora_client import MockAuroraClient
from continuum_robot.registration.legacy_compat import AuroraPoseSample
from continuum_robot.registration.rigid_solver import RigidRegistrationSolver
from continuum_robot.registration.runtime_tip_calibration import solve_runtime_tip_calibration
from continuum_robot.registration.runtime_tip_repository import (
    RuntimeTipCalibrationRecord,
    RuntimeTipCalibrationRepository,
)
from continuum_robot.services.tracking_service import TrackingService
from tests.fixtures.aurora_samples import build_tool_0A_record, build_transform_frame_from_records


def test_solve_runtime_tip_calibration_matches_thesis_chain_and_sign_aligned_averaging() -> None:
    truth_points_in_tip_by_label = {
        "T01": [0.0, 0.0, 0.0],
        "T02": [10.0, 0.0, 0.0],
        "T03": [0.0, 20.0, 0.0],
        "T04": [0.0, 0.0, 30.0],
    }
    labels = list(truth_points_in_tip_by_label.keys())

    T_aurora_coil = np.eye(4, dtype=float)
    T_aurora_coil[0:3, 3] = np.array([10.0, 0.0, 0.0])
    expected_T_coil_tip = np.eye(4, dtype=float)
    expected_T_coil_tip[0:3, 3] = np.array([0.0, 5.0, 0.0])
    T_aurora_tip = T_aurora_coil @ expected_T_coil_tip

    raw_points_by_label = {
        label: [(T_aurora_tip @ np.array([*point, 1.0], dtype=float))[0:3].tolist()]
        for label, point in truth_points_in_tip_by_label.items()
    }
    coil_samples = [
        AuroraPoseSample(
            tool_id="0A",
            quaternion_wxyz=(1.0, 0.0, 0.0, 0.0),
            translation_mm=(10.0, 0.0, 0.0),
        ),
        AuroraPoseSample(
            tool_id="0A",
            quaternion_wxyz=(-1.0, 0.0, 0.0, 0.0),
            translation_mm=(10.0, 0.0, 0.0),
        ),
    ]

    result = solve_runtime_tip_calibration(
        solver=RigidRegistrationSolver(),
        labels=labels,
        raw_points_by_label=raw_points_by_label,
        truth_points_in_tip_by_label=truth_points_in_tip_by_label,
        coil_samples=coil_samples,
        quaternion_average_method="sign_aligned_mean",
    )

    assert np.allclose(result.T_aurora_coil_avg, T_aurora_coil)
    assert np.allclose(result.T_aurora_tip, T_aurora_tip)
    assert np.allclose(result.T_coil_tip, expected_T_coil_tip)
    assert result.fit_rmse_mm < 1e-9
    assert result.validation_metrics["coil_rotation_spread_deg"]["max"] == 0.0


def test_runtime_tip_calibration_repository_round_trips_latest_payload(tmp_path: Path) -> None:
    repository = RuntimeTipCalibrationRepository(
        latest_path=tmp_path / "runtime_tip" / "latest_runtime_tip_calibration.json"
    )
    record = RuntimeTipCalibrationRecord(
        timestamp_utc="2026-04-10T12:00:00Z",
        calibration_kind="runtime_tip_calibration_hat",
        measurement_tool_id="0B",
        coil_tool_id="0A",
        setup_id="bench-a",
        truth_points_in_sw_by_label={"T01": [1.0, 2.0, 3.0]},
        truth_points_in_tip_by_label={"T01": [4.0, 5.0, 6.0]},
        raw_captured_hat_points_aurora_xyz_by_label={"T01": [[7.0, 8.0, 9.0]]},
        averaged_hat_points_aurora_xyz_by_label={"T01": [7.0, 8.0, 9.0]},
        raw_coil_samples=[],
        residuals_tip_xyz_mm_by_label={"T01": [0.1, 0.0, 0.0]},
        fit_rmse_mm=0.1,
        T_tip_aurora=np.eye(4).tolist(),
        T_aurora_tip=np.eye(4).tolist(),
        T_aurora_coil_avg=np.eye(4).tolist(),
        T_coil_tip=np.eye(4).tolist(),
        config_used={"measurement_tool_id": "0B", "coil_tool_id": "0A"},
        validation_metrics={"hat_fit_rmse_mm": 0.1},
    )

    output_path = repository.save_record(record)
    payload = repository.load_latest_payload()

    assert output_path.exists()
    assert payload is not None
    assert payload["timestamp_utc"] == "2026-04-10T12:00:00Z"
    assert payload["T_tip_2_coil"] == np.eye(4).tolist()


def test_runtime_tip_calibration_repository_separates_quick_override_alias(tmp_path: Path) -> None:
    repository = RuntimeTipCalibrationRepository(
        latest_path=tmp_path / "runtime_tip" / "latest_runtime_tip_calibration.json"
    )
    accepted = RuntimeTipCalibrationRecord(
        timestamp_utc="2026-04-10T12:00:00Z",
        calibration_kind="runtime_tip_calibration_hat",
        measurement_tool_id="0B",
        coil_tool_id="0A",
        setup_id=None,
        truth_points_in_sw_by_label={"T01": [0.0, 0.0, 0.0]},
        truth_points_in_tip_by_label={"T01": [0.0, 0.0, 0.0]},
        raw_captured_hat_points_aurora_xyz_by_label={"T01": [[0.0, 0.0, 0.0]]},
        averaged_hat_points_aurora_xyz_by_label={"T01": [0.0, 0.0, 0.0]},
        raw_coil_samples=[],
        residuals_tip_xyz_mm_by_label={"T01": [0.0, 0.0, 0.0]},
        fit_rmse_mm=0.0,
        T_tip_aurora=np.eye(4).tolist(),
        T_aurora_tip=np.eye(4).tolist(),
        T_aurora_coil_avg=np.eye(4).tolist(),
        T_coil_tip=np.eye(4).tolist(),
        config_used={},
    )
    quick = RuntimeTipCalibrationRecord(
        timestamp_utc="2026-04-10T12:10:00Z",
        calibration_kind="runtime_tip_calibration_quick_4_point",
        measurement_tool_id="0B",
        coil_tool_id="0A",
        setup_id=None,
        truth_points_in_sw_by_label={"T01": [0.0, 0.0, 0.0]},
        truth_points_in_tip_by_label={"T01": [0.0, 0.0, 0.0]},
        raw_captured_hat_points_aurora_xyz_by_label={"T01": [[0.0, 0.0, 0.0]]},
        averaged_hat_points_aurora_xyz_by_label={"T01": [0.0, 0.0, 0.0]},
        raw_coil_samples=[],
        residuals_tip_xyz_mm_by_label={"T01": [0.0, 0.0, 0.0]},
        fit_rmse_mm=0.0,
        T_tip_aurora=np.eye(4).tolist(),
        T_aurora_tip=np.eye(4).tolist(),
        T_aurora_coil_avg=np.eye(4).tolist(),
        T_coil_tip=np.eye(4).tolist(),
        config_used={},
    )

    repository.save_record(accepted)
    repository.save_record(quick, mark_as_latest=False, alias_path=repository.quick_latest_path)

    latest = repository.load_latest_payload()
    quick_latest = repository.load_latest_quick_payload()

    assert latest is not None
    assert latest["calibration_kind"] == "runtime_tip_calibration_hat"
    assert quick_latest is not None
    assert quick_latest["calibration_kind"] == "runtime_tip_calibration_quick_4_point"


def test_tracking_service_uses_separate_runtime_tip_artifact_in_live_chain(tmp_path: Path) -> None:
    registration_path = tmp_path / "registrations" / "latest_registration.json"
    registration_path.parent.mkdir(parents=True, exist_ok=True)
    registration_path.write_text(
        json.dumps(
            {
                "timestamp_utc": "2026-04-10T12:00:00Z",
                "T_robot_aurora": [[1.0, 0.0, 0.0, 1.0], [0.0, 1.0, 0.0, 2.0], [0.0, 0.0, 1.0, 3.0], [0.0, 0.0, 0.0, 1.0]],
                "config_used": {"measurement_tool_id": "0B", "coil_tool_id": "0A"},
            }
        ),
        encoding="utf-8",
    )
    runtime_tip_path = tmp_path / "data" / "runtime_tip_calibration" / "latest_runtime_tip_calibration.json"
    runtime_tip_path.parent.mkdir(parents=True, exist_ok=True)
    runtime_tip_path.write_text(
        json.dumps(
            {
                "timestamp_utc": "2026-04-10T12:05:00Z",
                "measurement_tool_id": "0B",
                "coil_tool_id": "0A",
                "calibration_kind": "runtime_tip_calibration_hat",
                "T_coil_tip": [[1.0, 0.0, 0.0, 0.5], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]],
            }
        ),
        encoding="utf-8",
    )

    service = TrackingService(
        MockAuroraClient(),
        port="/dev/null",
        registration_path=registration_path,
        runtime_tip_calibration_path=runtime_tip_path,
        config_source="test",
    )
    frame = build_transform_frame_from_records(
        frame_number=1,
        records=[build_tool_0A_record(translation_xyz=(10.0, 0.0, 0.0))],
    )
    service.ingest_frame(frame, source="test")
    snapshot = service.get_snapshot()

    assert snapshot.runtime_tip_calibration_state == "loaded"
    assert snapshot.stored_runtime_tip_timestamp_utc == "2026-04-10T12:05:00Z"
    assert snapshot.tip_pose_status == "ok"
    assert snapshot.T_robot_tip is not None
    assert np.allclose([row[3] for row in snapshot.T_robot_tip[:3]], [11.5, 2.0, 3.0])


def test_tracking_service_reports_identity_fallback_when_runtime_tip_artifact_missing(tmp_path: Path) -> None:
    registration_path = tmp_path / "registrations" / "latest_registration.json"
    registration_path.parent.mkdir(parents=True, exist_ok=True)
    registration_path.write_text(
        json.dumps(
            {
                "timestamp_utc": "2026-04-10T12:00:00Z",
                "T_robot_aurora": np.eye(4).tolist(),
                "T_coil_tip": np.eye(4).tolist(),
                "live_pose_tip_transform": {"source": "identity_assumption_simple_registration"},
                "config_used": {"measurement_tool_id": "0B", "coil_tool_id": "0A"},
            }
        ),
        encoding="utf-8",
    )

    service = TrackingService(
        MockAuroraClient(),
        port="/dev/null",
        registration_path=registration_path,
        runtime_tip_calibration_path=tmp_path / "missing_runtime_tip_calibration.json",
        config_source="test",
    )
    frame = build_transform_frame_from_records(
        frame_number=1,
        records=[build_tool_0A_record(translation_xyz=(1.0, 2.0, 3.0))],
    )
    service.ingest_frame(frame, source="test")
    snapshot = service.get_snapshot()

    assert snapshot.runtime_tip_calibration_state == "identity_tip_fallback"
    assert snapshot.runtime_tip_identity_fallback is True
    assert snapshot.tip_pose_status == "identity_tip_fallback"
    assert snapshot.T_robot_tip is not None


def test_tracking_service_supports_explicit_coil_as_tip_mode(tmp_path: Path) -> None:
    registration_path = tmp_path / "registrations" / "latest_registration.json"
    registration_path.parent.mkdir(parents=True, exist_ok=True)
    registration_path.write_text(
        json.dumps(
            {
                "timestamp_utc": "2026-04-10T12:00:00Z",
                "T_robot_aurora": np.eye(4).tolist(),
                "config_used": {"measurement_tool_id": "0B", "coil_tool_id": "0A"},
            }
        ),
        encoding="utf-8",
    )

    service = TrackingService(
        MockAuroraClient(),
        port="/dev/null",
        registration_path=registration_path,
        runtime_tip_calibration_path=tmp_path / "missing_runtime_tip_calibration.json",
        config_source="test",
    )
    service.set_runtime_tip_mode("coil_as_tip")
    frame = build_transform_frame_from_records(
        frame_number=1,
        records=[build_tool_0A_record(translation_xyz=(1.0, 2.0, 3.0))],
    )
    service.ingest_frame(frame, source="test")
    snapshot = service.get_snapshot()

    assert snapshot.runtime_tip_mode == "coil_as_tip"
    assert snapshot.runtime_tip_calibration_state == "coil_as_tip"
    assert snapshot.runtime_tip_trust_level == "thesis_trusted"
    assert "0A coil origin/position path is shown directly" in snapshot.runtime_tip_mode_message
    assert snapshot.tip_pose_status == "coil_as_tip"
    assert snapshot.T_robot_tip is not None
    assert np.allclose([row[3] for row in snapshot.T_robot_tip[:3]], [1.0, 2.0, 3.0])


def test_tracking_service_supports_quick_4_point_runtime_tip_override(tmp_path: Path) -> None:
    registration_path = tmp_path / "registrations" / "latest_registration.json"
    registration_path.parent.mkdir(parents=True, exist_ok=True)
    registration_path.write_text(
        json.dumps(
            {
                "timestamp_utc": "2026-04-10T12:00:00Z",
                "T_robot_aurora": np.eye(4).tolist(),
                "config_used": {"measurement_tool_id": "0B", "coil_tool_id": "0A"},
            }
        ),
        encoding="utf-8",
    )
    runtime_tip_path = tmp_path / "data" / "runtime_tip_calibration" / "latest_runtime_tip_calibration.json"
    runtime_tip_path.parent.mkdir(parents=True, exist_ok=True)
    quick_path = runtime_tip_path.parent / "latest_quick_4_point_runtime_tip.json"
    quick_path.write_text(
        json.dumps(
            {
                "timestamp_utc": "2026-04-10T12:05:00Z",
                "measurement_tool_id": "0B",
                "coil_tool_id": "0A",
                "calibration_kind": "runtime_tip_calibration_quick_4_point",
                "T_coil_tip": [[1.0, 0.0, 0.0, 0.25], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]],
            }
        ),
        encoding="utf-8",
    )

    service = TrackingService(
        MockAuroraClient(),
        port="/dev/null",
        registration_path=registration_path,
        runtime_tip_calibration_path=runtime_tip_path,
        config_source="test",
    )
    service.set_runtime_tip_mode("quick_4_point")
    frame = build_transform_frame_from_records(
        frame_number=1,
        records=[build_tool_0A_record(translation_xyz=(10.0, 0.0, 0.0))],
    )
    service.ingest_frame(frame, source="test")
    snapshot = service.get_snapshot()

    assert snapshot.runtime_tip_mode == "quick_4_point"
    assert snapshot.runtime_tip_calibration_state == "quick_4_point_loaded"
    assert snapshot.runtime_tip_trust_level == "debug_only"
    assert snapshot.runtime_tip_selected_artifact_path == str(quick_path)
    assert snapshot.T_robot_tip is not None
    assert np.allclose([row[3] for row in snapshot.T_robot_tip[:3]], [10.25, 0.0, 0.0])


def test_tracking_service_runtime_tip_messages_make_direct_0a_and_quick_override_explicit(tmp_path: Path) -> None:
    registration_path = tmp_path / "registrations" / "latest_registration.json"
    registration_path.parent.mkdir(parents=True, exist_ok=True)
    registration_path.write_text(
        json.dumps(
            {
                "timestamp_utc": "2026-04-10T12:00:00Z",
                "T_robot_aurora": np.eye(4).tolist(),
                "config_used": {"measurement_tool_id": "0B", "coil_tool_id": "0A"},
            }
        ),
        encoding="utf-8",
    )
    runtime_tip_path = tmp_path / "data" / "runtime_tip_calibration" / "latest_runtime_tip_calibration.json"
    runtime_tip_path.parent.mkdir(parents=True, exist_ok=True)
    quick_path = runtime_tip_path.parent / "latest_quick_4_point_runtime_tip.json"
    quick_path.write_text(
        json.dumps(
            {
                "timestamp_utc": "2026-04-10T12:05:00Z",
                "measurement_tool_id": "0B",
                "coil_tool_id": "0A",
                "calibration_kind": "runtime_tip_calibration_quick_4_point",
                "T_coil_tip": [[1.0, 0.0, 0.0, 0.25], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]],
            }
        ),
        encoding="utf-8",
    )

    service = TrackingService(
        MockAuroraClient(),
        port="/dev/null",
        registration_path=registration_path,
        runtime_tip_calibration_path=runtime_tip_path,
        config_source="test",
    )

    service.set_runtime_tip_mode("coil_as_tip")
    service.ingest_frame(
        build_transform_frame_from_records(
            frame_number=1,
            records=[build_tool_0A_record(translation_xyz=(1.0, 2.0, 3.0))],
        ),
        source="test",
    )
    coil_snapshot = service.get_snapshot()
    assert "0A coil origin/position path is shown directly" in coil_snapshot.runtime_tip_mode_message

    service.set_runtime_tip_mode("quick_4_point")
    service.ingest_frame(
        build_transform_frame_from_records(
            frame_number=2,
            records=[build_tool_0A_record(translation_xyz=(1.0, 2.0, 3.0))],
        ),
        source="test",
    )
    quick_snapshot = service.get_snapshot()
    assert quick_snapshot.runtime_tip_calibration_state == "quick_4_point_loaded"
    assert "Quick 4-point runtime tip override is active" in quick_snapshot.runtime_tip_mode_message

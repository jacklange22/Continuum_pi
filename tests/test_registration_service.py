from __future__ import annotations
import json
from pathlib import Path

import numpy as np
import pytest

from continuum_robot.hardware.mock_aurora_client import MockAuroraClient
from continuum_robot.registration.repository import RegistrationRepository
from continuum_robot.registration.rigid_solver import RigidRegistrationSolver
from continuum_robot.services.registration_service import RegistrationService
from continuum_robot.services.tracking_service import TrackingService
from tests.fixtures.aurora_samples import build_tool_0A_record, build_tool_0B_record, build_transform_frame_from_records


def _make_services(
    tmp_path: Path,
    config_lines: list[str] | None = None,
) -> tuple[TrackingService, RegistrationService]:
    registration_path = tmp_path / "registrations" / "latest_registration.json"
    tracking_service = TrackingService(
        MockAuroraClient(),
        port="/dev/null",
        registration_path=registration_path,
        config_source="test-system",
    )
    config_path = tmp_path / "registration.yaml"
    config_lines = config_lines or [
        "landmark_labels: [L1, L2, L3]",
        "captures_per_landmark: 2",
        "capture_tool_id: \"0B\"",
        "nominal_landmarks_robot_xyz_mm:",
        "  L1: [0.0, 0.0, 0.0]",
        "  L2: [10.0, 0.0, 0.0]",
        "  L3: [0.0, 10.0, 0.0]",
        "validation:",
        "  max_fre_mm: 1.0",
    ]
    config_path.write_text(
        "\n".join(config_lines),
        encoding="utf-8",
    )
    repository = RegistrationRepository(root_dir=tmp_path / "registrations")
    registration_service = RegistrationService(
        tracking_service=tracking_service,
        repository=repository,
        solver=RigidRegistrationSolver(),
        config_path=config_path,
        config_source=str(config_path),
    )
    return tracking_service, registration_service


def _ingest_tool_0b_sample(tracking_service: TrackingService, frame_number: int, xyz: tuple[float, float, float]) -> None:
    frame = build_transform_frame_from_records(frame_number=frame_number, records=[build_tool_0B_record(translation_xyz=xyz)])
    tracking_service.ingest_frame(frame, source="test")


def test_registration_service_solves_and_accepts_registration(tmp_path: Path) -> None:
    tracking_service, registration_service = _make_services(tmp_path)
    snapshot = registration_service.begin_session()
    assert snapshot.capture_tool_id == "0B"

    _ingest_tool_0b_sample(tracking_service, 1, (0.0, 0.0, 0.0))
    registration_service.capture_sample("L1")
    _ingest_tool_0b_sample(tracking_service, 2, (0.0, 0.0, 0.0))
    registration_service.capture_sample("L1")
    registration_service.complete_landmark()

    _ingest_tool_0b_sample(tracking_service, 3, (10.0, 0.0, 0.0))
    registration_service.capture_sample("L2")
    _ingest_tool_0b_sample(tracking_service, 4, (10.0, 0.0, 0.0))
    registration_service.capture_sample("L2")
    registration_service.complete_landmark()

    _ingest_tool_0b_sample(tracking_service, 5, (0.0, 10.0, 0.0))
    registration_service.capture_sample("L3")
    _ingest_tool_0b_sample(tracking_service, 6, (0.0, 10.0, 0.0))
    registration_service.capture_sample("L3")
    registration_service.complete_landmark()

    payload = registration_service.solve_registration()
    assert np.allclose(np.asarray(payload["T_robot_aurora"]), np.eye(4), atol=1e-6)
    output_path = registration_service.accept_registration()
    assert output_path.exists()

    saved = json.loads(output_path.read_text(encoding="utf-8"))
    assert saved["config_used"]["capture_tool_id"] == "0B"

    frame = build_transform_frame_from_records(frame_number=7, records=[build_tool_0A_record(translation_xyz=(1.0, 2.0, 3.0))])
    tracking_service.ingest_frame(frame, source="test")
    tracking_snapshot = tracking_service.get_snapshot()
    assert tracking_snapshot.tip_pose_status == "identity_tip_fallback"
    assert tracking_snapshot.T_robot_tip is not None
    assert np.allclose([row[3] for row in tracking_snapshot.T_robot_tip[:3]], [1.0, 2.0, 3.0])


def test_registration_service_rejects_capture_when_0b_missing(tmp_path: Path) -> None:
    tracking_service, registration_service = _make_services(tmp_path)
    registration_service.begin_session()
    frame = build_transform_frame_from_records(frame_number=1, records=[build_tool_0A_record()])
    tracking_service.ingest_frame(frame, source="test")

    try:
        registration_service.capture_sample("L1")
    except RuntimeError as exc:
        assert "Tool 0B" in str(exc)
    else:
        raise AssertionError("Expected capture_sample to reject missing 0B")


def test_registration_service_loads_candidate_landmarks_from_config(tmp_path: Path) -> None:
    _tracking_service, registration_service = _make_services(
        tmp_path,
        config_lines=[
            "captures_per_landmark: 1",
            "capture_tool_id: \"0B\"",
            "candidate_landmarks:",
            "  - id: L1",
            "    xyz_mm: [0.0, 0.0, 0.0]",
            "    enabled: true",
            "  - id: L2",
            "    xyz_mm: [10.0, 0.0, 0.0]",
            "    enabled: true",
            "  - id: L3",
            "    xyz_mm: [0.0, 10.0, 0.0]",
            "    enabled: true",
            "  - id: L4",
            "    xyz_mm: [10.0, 10.0, 5.0]",
            "    enabled: true",
        ],
    )

    snapshot = registration_service.begin_session()

    assert snapshot.labels == ["L1", "L2", "L3", "L4"]
    assert snapshot.nominal_landmarks_robot_xyz_mm["L4"] == [10.0, 10.0, 5.0]


def test_registration_service_loads_tip_file_relative_to_local_config_path(tmp_path: Path) -> None:
    tip_path = tmp_path / "data" / "pivot_calibration" / "generated_penprobe_tip.csv"
    tip_path.parent.mkdir(parents=True, exist_ok=True)
    tip_path.write_text("1.0,2.0,3.0", encoding="utf-8")

    _tracking_service, registration_service = _make_services(
        tmp_path,
        config_lines=[
            "captures_per_landmark: 1",
            'capture_tool_id: "0B"',
            'penprobe_file: "data/pivot_calibration/generated_penprobe_tip.csv"',
            "nominal_landmarks_robot_xyz_mm:",
            "  L1: [0.0, 0.0, 0.0]",
            "  L2: [10.0, 0.0, 0.0]",
            "  L3: [0.0, 10.0, 0.0]",
            "  L4: [10.0, 10.0, 0.0]",
            "landmark_labels: [L1, L2, L3, L4]",
        ],
    )

    status = registration_service.get_measurement_point_status(refresh=True)

    assert status["ready"] is True
    assert status["path"] == str(tip_path)


def test_registration_service_simple_registration_records_tip_provenance_and_applies_tip_offset(tmp_path: Path) -> None:
    tip_path = tmp_path / "data" / "pivot_calibration" / "generated_penprobe_tip.csv"
    tip_path.parent.mkdir(parents=True, exist_ok=True)
    tip_path.write_text("1.0,2.0,3.0", encoding="utf-8")

    tracking_service, registration_service = _make_services(
        tmp_path,
        config_lines=[
            "landmark_labels: [L1, L2, L3, L4]",
            "captures_per_landmark: 1",
            'capture_tool_id: "0B"',
            'coil_tool_id: "0A"',
            'penprobe_file: "data/pivot_calibration/generated_penprobe_tip.csv"',
            "nominal_landmarks_robot_xyz_mm:",
            "  L1: [1.0, 2.0, 3.0]",
            "  L2: [11.0, 2.0, 3.0]",
            "  L3: [1.0, 12.0, 3.0]",
            "  L4: [1.0, 2.0, 13.0]",
        ],
    )

    registration_service.begin_session()
    for frame_number, label, xyz in [
        (1, "L1", (0.0, 0.0, 0.0)),
        (2, "L2", (10.0, 0.0, 0.0)),
        (3, "L3", (0.0, 10.0, 0.0)),
        (4, "L4", (0.0, 0.0, 10.0)),
    ]:
        _ingest_tool_0b_sample(tracking_service, frame_number, xyz)
        registration_service.capture_sample(label)
        registration_service.complete_landmark()

    payload = registration_service.solve_registration()
    output_path = registration_service.accept_registration()
    saved = json.loads(output_path.read_text(encoding="utf-8"))

    assert payload["raw_captured_landmarks_robot_xyz"]["L1"][0] == [1.0, 2.0, 3.0]
    assert saved["raw_captured_landmarks_robot_xyz"]["L4"][0] == [1.0, 2.0, 13.0]
    assert saved["capture_tip_provenance"]["path"] == str(tip_path)
    assert saved["capture_tip_provenance"]["tip_vector_mm"] == [1.0, 2.0, 3.0]
    assert saved["capture_tip_provenance"]["offset_applied_before_solving"] is True
    assert saved["live_pose_tip_transform"]["source"] == "identity_assumption_simple_registration"
    assert np.allclose(np.asarray(saved["T_coil_tip"], dtype=float), np.eye(4))
    assert saved["config_used"]["capture_tip_offset_applied_before_solving"] is True


def test_registration_service_saves_richer_validation_metrics_and_runtime_application_summary(tmp_path: Path) -> None:
    tracking_service, registration_service = _make_services(
        tmp_path,
        config_lines=[
            "landmark_labels: [L1, L2, L3, L4]",
            "captures_per_landmark: 1",
            'capture_tool_id: "0B"',
            'coil_tool_id: "0A"',
            "nominal_landmarks_robot_xyz_mm:",
            "  L1: [0.0, 0.0, 0.0]",
            "  L2: [10.0, 0.0, 0.0]",
            "  L3: [0.0, 10.0, 0.0]",
            "  L4: [0.0, 0.0, 10.0]",
            "validation:",
            "  max_fre_mm: 1.0",
        ],
    )

    registration_service.begin_session()
    for frame_number, label, xyz in [
        (1, "L1", (0.0, 0.0, 0.0)),
        (2, "L2", (10.0, 0.0, 0.0)),
        (3, "L3", (0.0, 10.0, 0.0)),
        (4, "L4", (0.0, 0.0, 10.0)),
    ]:
        _ingest_tool_0b_sample(tracking_service, frame_number, xyz)
        registration_service.capture_sample(label)
        registration_service.complete_landmark()

    registration_service.solve_registration()
    output_path = registration_service.accept_registration()
    saved = json.loads(output_path.read_text(encoding="utf-8"))
    latest_validation = registration_service.repository.load_latest_validation_summary()
    trust = registration_service.get_registration_trust_summary()

    assert saved["validation_metrics"]["truth_geometry"]["geometry_rank"] == 3
    assert saved["validation_metrics"]["capture_counts_by_label"] == {"L1": 1, "L2": 1, "L3": 1, "L4": 1}
    assert saved["validation_metrics"]["configured_max_fre_mm"] == 1.0
    assert saved["validation_metrics"]["worst_landmark_label"] in {"L1", "L2", "L3", "L4"}
    assert latest_validation is not None
    assert latest_validation["comparison_count"] == 0
    assert latest_validation["runtime_application"]["loaded_latest_registration"] is True
    assert latest_validation["runtime_application"]["timestamp_matches_latest"] is True
    assert trust["trust_state"] == "trusted"
    assert trust["live_chain_state"] in {
        "registration_loaded_waiting_for_live_pose",
        "ok",
        "identity_tip_fallback",
    }


def test_registration_service_repeated_validation_summary_compares_recent_runs(tmp_path: Path) -> None:
    tracking_service, registration_service = _make_services(
        tmp_path,
        config_lines=[
            "landmark_labels: [L1, L2, L3, L4]",
            "captures_per_landmark: 1",
            'capture_tool_id: "0B"',
            'coil_tool_id: "0A"',
            "nominal_landmarks_robot_xyz_mm:",
            "  L1: [0.0, 0.0, 0.0]",
            "  L2: [10.0, 0.0, 0.0]",
            "  L3: [0.0, 10.0, 0.0]",
            "  L4: [0.0, 0.0, 10.0]",
        ],
    )

    def _run_once(offset_mm: float) -> None:
        registration_service.begin_session()
        samples = [
            (1, "L1", (0.0 + offset_mm, 0.0, 0.0)),
            (2, "L2", (10.0 + offset_mm, 0.0, 0.0)),
            (3, "L3", (0.0 + offset_mm, 10.0, 0.0)),
            (4, "L4", (0.0 + offset_mm, 0.0, 10.0)),
        ]
        for frame_number, label, xyz in samples:
            _ingest_tool_0b_sample(tracking_service, frame_number, xyz)
            registration_service.capture_sample(label)
            registration_service.complete_landmark()
        registration_service.solve_registration()
        registration_service.accept_registration()

    _run_once(0.0)
    _run_once(0.5)

    latest_validation = registration_service.repository.load_latest_validation_summary()

    assert latest_validation is not None
    assert latest_validation["comparison_count"] == 1
    assert latest_validation["translation_delta_summary_mm"]["max"] == pytest.approx(0.5)
    assert latest_validation["history_run_count"] == 2


def test_registration_service_rejects_stale_tracker_data_during_capture(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    tracking_service, registration_service = _make_services(tmp_path)
    registration_service.begin_session()
    _ingest_tool_0b_sample(tracking_service, 1, (0.0, 0.0, 0.0))

    stale_snapshot = tracking_service.get_snapshot()
    stale_snapshot.tracker_data_stale = True
    stale_snapshot.tracker_data_age_s = 0.61
    monkeypatch.setattr(tracking_service, "get_snapshot", lambda: stale_snapshot)

    with pytest.raises(RuntimeError, match=r"Tracker data is stale \(0.610 s\)"):
        registration_service.capture_sample("L1")


def test_registration_service_peek_current_measurement_point_reports_stale_tracker_data(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tracking_service, registration_service = _make_services(tmp_path)
    _ingest_tool_0b_sample(tracking_service, 1, (0.0, 0.0, 0.0))

    stale_snapshot = tracking_service.get_snapshot()
    stale_snapshot.tracker_data_stale = True
    stale_snapshot.tracker_data_age_s = 0.33
    monkeypatch.setattr(tracking_service, "get_snapshot", lambda: stale_snapshot)

    point = registration_service.peek_current_measurement_point()

    assert point["available"] is False
    assert "Tracker data is stale (0.330 s)." in str(point["status"])


def test_registration_service_rejects_non_rigid_capture_tool_tip_transform(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="capture_tool_tip_transform\\[0:3,0:3\\] is not orthonormal"):
        _make_services(
            tmp_path,
            config_lines=[
                "landmark_labels: [L1, L2, L3]",
                "captures_per_landmark: 1",
                'capture_tool_id: "0B"',
                "capture_tool_tip_transform:",
                "  - [2.0, 0.0, 0.0, 1.0]",
                "  - [0.0, 1.0, 0.0, 2.0]",
                "  - [0.0, 0.0, 1.0, 3.0]",
                "  - [0.0, 0.0, 0.0, 1.0]",
                "nominal_landmarks_robot_xyz_mm:",
                "  L1: [0.0, 0.0, 0.0]",
                "  L2: [10.0, 0.0, 0.0]",
                "  L3: [0.0, 10.0, 0.0]",
            ],
        )


def test_registration_service_solve_records_solver_comparison_skipped_under_five_landmarks(tmp_path: Path) -> None:
    """With only 4 landmarks, classical fills `solver_comparison.classical` but RANSAC is skipped."""
    tracking_service, registration_service = _make_services(
        tmp_path,
        config_lines=[
            "landmark_labels: [L1, L2, L3, L4]",
            "captures_per_landmark: 1",
            'capture_tool_id: "0B"',
            "nominal_landmarks_robot_xyz_mm:",
            "  L1: [0.0, 0.0, 0.0]",
            "  L2: [10.0, 0.0, 0.0]",
            "  L3: [0.0, 10.0, 0.0]",
            "  L4: [0.0, 0.0, 10.0]",
            "validation:",
            "  max_fre_mm: 1.0",
        ],
    )
    registration_service.begin_session()
    for frame_number, label, xyz in [
        (1, "L1", (0.0, 0.0, 0.0)),
        (2, "L2", (10.0, 0.0, 0.0)),
        (3, "L3", (0.0, 10.0, 0.0)),
        (4, "L4", (0.0, 0.0, 10.0)),
    ]:
        _ingest_tool_0b_sample(tracking_service, frame_number, xyz)
        registration_service.capture_sample(label)
        registration_service.complete_landmark()
    payload = registration_service.solve_registration()
    comparison = payload["validation_metrics"]["solver_comparison"]
    assert "classical" in comparison
    assert comparison["classical"]["fre_mm"] == pytest.approx(0.0, abs=1e-6)
    assert "ransac_skipped" in comparison
    assert "5" in comparison["ransac_skipped"]


def test_registration_service_solve_compares_classical_and_ransac_with_outlier(tmp_path: Path) -> None:
    """With 6 landmarks and one mislabeled outlier, RANSAC inlier FRE beats classical FRE."""
    tracking_service, registration_service = _make_services(
        tmp_path,
        config_lines=[
            "landmark_labels: [L1, L2, L3, L4, L5, L6]",
            "captures_per_landmark: 1",
            'capture_tool_id: "0B"',
            "nominal_landmarks_robot_xyz_mm:",
            "  L1: [0.0, 0.0, 0.0]",
            "  L2: [10.0, 0.0, 0.0]",
            "  L3: [0.0, 10.0, 0.0]",
            "  L4: [0.0, 0.0, 10.0]",
            "  L5: [10.0, 10.0, 0.0]",
            "  L6: [10.0, 0.0, 10.0]",
            "ransac:",
            "  inlier_threshold_mm: 1.0",
            "  seed: 42",
        ],
    )
    registration_service.begin_session()
    # Five clean captures match truth exactly; one (L4) is offset by ~6 mm.
    captures = [
        (1, "L1", (0.0, 0.0, 0.0)),
        (2, "L2", (10.0, 0.0, 0.0)),
        (3, "L3", (0.0, 10.0, 0.0)),
        (4, "L4", (4.0, -3.0, 12.0)),  # outlier
        (5, "L5", (10.0, 10.0, 0.0)),
        (6, "L6", (10.0, 0.0, 10.0)),
    ]
    for frame_number, label, xyz in captures:
        _ingest_tool_0b_sample(tracking_service, frame_number, xyz)
        registration_service.capture_sample(label)
        registration_service.complete_landmark()
    payload = registration_service.solve_registration()
    comparison = payload["validation_metrics"]["solver_comparison"]
    assert "classical" in comparison
    assert "ransac" in comparison
    classical_fre = comparison["classical"]["fre_mm"]
    ransac_inlier_fre = comparison["ransac"]["fre_mm_inliers_only"]
    assert ransac_inlier_fre < classical_fre, (
        f"Expected RANSAC inlier FRE ({ransac_inlier_fre}) to beat classical ({classical_fre})"
    )
    assert "L4" in comparison["ransac"]["rejected_labels"]
    assert comparison["ransac"]["converged"] is True
    assert comparison["delta"]["fre_mm_classical_minus_ransac_inliers"] > 0


def test_registration_service_set_active_solver_switches_pending_transform(tmp_path: Path) -> None:
    """set_active_solver flips the pending T_robot_aurora and FRE to match the chosen solver."""
    tracking_service, registration_service = _make_services(
        tmp_path,
        config_lines=[
            "landmark_labels: [L1, L2, L3, L4, L5, L6]",
            "captures_per_landmark: 1",
            'capture_tool_id: "0B"',
            "nominal_landmarks_robot_xyz_mm:",
            "  L1: [0.0, 0.0, 0.0]",
            "  L2: [10.0, 0.0, 0.0]",
            "  L3: [0.0, 10.0, 0.0]",
            "  L4: [0.0, 0.0, 10.0]",
            "  L5: [10.0, 10.0, 0.0]",
            "  L6: [10.0, 0.0, 10.0]",
            "ransac:",
            "  inlier_threshold_mm: 1.0",
            "  seed: 42",
        ],
    )
    registration_service.begin_session()
    # Same outlier scenario as the comparison test: L4 is ~6 mm off truth.
    captures = [
        (1, "L1", (0.0, 0.0, 0.0)),
        (2, "L2", (10.0, 0.0, 0.0)),
        (3, "L3", (0.0, 10.0, 0.0)),
        (4, "L4", (4.0, -3.0, 12.0)),
        (5, "L5", (10.0, 10.0, 0.0)),
        (6, "L6", (10.0, 0.0, 10.0)),
    ]
    for frame_number, label, xyz in captures:
        _ingest_tool_0b_sample(tracking_service, frame_number, xyz)
        registration_service.capture_sample(label)
        registration_service.complete_landmark()

    payload = registration_service.solve_registration()
    snapshot = registration_service.get_snapshot()
    assert snapshot.active_solver == "classical"
    assert snapshot.solver_choices == ["classical", "ransac"]
    classical_fre = float(payload["fre_mm"])

    ransac_payload = registration_service.set_active_solver("ransac")
    ransac_snapshot = registration_service.get_snapshot()
    assert ransac_snapshot.active_solver == "ransac"
    assert ransac_snapshot.fre_mm is not None
    assert ransac_snapshot.fre_mm < classical_fre, (
        f"RANSAC FRE ({ransac_snapshot.fre_mm}) should be lower than classical ({classical_fre}) "
        "after dropping the outlier."
    )
    assert ransac_payload["T_robot_aurora"] != payload["T_robot_aurora"]

    # Switching back restores the classical fit byte-for-byte.
    classical_payload = registration_service.set_active_solver("classical")
    assert classical_payload["T_robot_aurora"] == payload["T_robot_aurora"]
    assert pytest.approx(classical_payload["fre_mm"], abs=1e-9) == classical_fre


def test_registration_service_set_active_solver_rejects_unavailable_solver(tmp_path: Path) -> None:
    """When only 4 landmarks are solved, set_active_solver('ransac') refuses."""
    tracking_service, registration_service = _make_services(
        tmp_path,
        config_lines=[
            "landmark_labels: [L1, L2, L3, L4]",
            "captures_per_landmark: 1",
            'capture_tool_id: "0B"',
            "nominal_landmarks_robot_xyz_mm:",
            "  L1: [0.0, 0.0, 0.0]",
            "  L2: [10.0, 0.0, 0.0]",
            "  L3: [0.0, 10.0, 0.0]",
            "  L4: [0.0, 0.0, 10.0]",
        ],
    )
    registration_service.begin_session()
    for frame_number, label, xyz in [
        (1, "L1", (0.0, 0.0, 0.0)),
        (2, "L2", (10.0, 0.0, 0.0)),
        (3, "L3", (0.0, 10.0, 0.0)),
        (4, "L4", (0.0, 0.0, 10.0)),
    ]:
        _ingest_tool_0b_sample(tracking_service, frame_number, xyz)
        registration_service.capture_sample(label)
        registration_service.complete_landmark()
    registration_service.solve_registration()
    with pytest.raises(RuntimeError, match="not available"):
        registration_service.set_active_solver("ransac")

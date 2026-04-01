import json
from pathlib import Path

import numpy as np

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
    assert tracking_snapshot.tip_pose_status == "ok"
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
    tip_path = tmp_path / "data" / "tip_cals" / "generated_penprobe_tip.csv"
    tip_path.parent.mkdir(parents=True, exist_ok=True)
    tip_path.write_text("1.0,2.0,3.0", encoding="utf-8")

    _tracking_service, registration_service = _make_services(
        tmp_path,
        config_lines=[
            "captures_per_landmark: 1",
            'capture_tool_id: "0B"',
            'penprobe_file: "data/tip_cals/generated_penprobe_tip.csv"',
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
    tip_path = tmp_path / "data" / "tip_cals" / "generated_penprobe_tip.csv"
    tip_path.parent.mkdir(parents=True, exist_ok=True)
    tip_path.write_text("1.0,2.0,3.0", encoding="utf-8")

    tracking_service, registration_service = _make_services(
        tmp_path,
        config_lines=[
            "landmark_labels: [L1, L2, L3, L4]",
            "captures_per_landmark: 1",
            'capture_tool_id: "0B"',
            'coil_tool_id: "0A"',
            'penprobe_file: "data/tip_cals/generated_penprobe_tip.csv"',
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

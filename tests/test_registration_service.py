import json
from pathlib import Path

import numpy as np

from continuum_robot.hardware.mock_aurora_client import MockAuroraClient
from continuum_robot.registration.repository import RegistrationRepository
from continuum_robot.registration.rigid_solver import RigidRegistrationSolver
from continuum_robot.services.registration_service import RegistrationService
from continuum_robot.services.tracking_service import TrackingService
from tests.fixtures.aurora_samples import build_tool_0A_record, build_tool_0B_record, build_transform_frame_from_records


def _make_services(tmp_path: Path) -> tuple[TrackingService, RegistrationService]:
    registration_path = tmp_path / "registrations" / "latest_registration.json"
    tracking_service = TrackingService(
        MockAuroraClient(),
        port="/dev/null",
        registration_path=registration_path,
        config_source="test-system",
    )
    config_path = tmp_path / "registration.yaml"
    config_path.write_text(
        "\n".join(
            [
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
        ),
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

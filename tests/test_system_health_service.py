from pathlib import Path

from continuum_robot.hardware.mock_aurora_client import MockAuroraClient
from continuum_robot.registration.repository import RegistrationRepository
from continuum_robot.registration.rigid_solver import RigidRegistrationSolver
from continuum_robot.services.registration_service import RegistrationService
from continuum_robot.services.system_health_service import SystemHealthService
from continuum_robot.services.tracking_service import TrackingService


def test_system_health_snapshot_has_tracking_and_registration_sections(tmp_path: Path) -> None:
    tracking_service = TrackingService(
        MockAuroraClient(),
        port="/dev/null",
        registration_path=tmp_path / "latest_registration.json",
        config_source="test-system",
    )
    config_path = tmp_path / "registration.yaml"
    config_path.write_text("landmark_labels: [L1]\ncaptures_per_landmark: 1\nnominal_landmarks_robot_xyz_mm:\n  L1: [0, 0, 0]\n", encoding="utf-8")
    registration_service = RegistrationService(
        tracking_service=tracking_service,
        repository=RegistrationRepository(root_dir=tmp_path / "registrations"),
        solver=RigidRegistrationSolver(),
        config_path=config_path,
        config_source=str(config_path),
    )
    health_service = SystemHealthService(
        tracking_service=tracking_service,
        registration_service=registration_service,
        config_source="test-root",
    )

    snapshot = health_service.get_snapshot()
    assert snapshot.health.name == "system_health_service"
    assert snapshot.tracking.name == "tracking_service"
    assert snapshot.registration.name == "registration_service"
    assert "tracking_faults" in snapshot.summary
    assert "tracking_backend_identity" in snapshot.summary

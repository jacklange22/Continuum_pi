"""Application bootstrap helpers."""

from dataclasses import dataclass
from pathlib import Path

from continuum_robot.app.service_registry import ServiceRegistry
from continuum_robot.config.config_loader import ConfigLoader
from continuum_robot.registration.live_registration_service import LiveRegistrationService
from continuum_robot.registration.repository import RegistrationRepository
from continuum_robot.registration.rigid_solver import RigidRegistrationSolver
from continuum_robot.tracking.tracker_service_manager import TrackerServiceManager


@dataclass
class AppContext:
    """Container for top-level services used by the GUI and scripts."""

    config_loader: ConfigLoader
    services: ServiceRegistry


def build_app_context() -> AppContext:
    """Build and return application context with lightweight service stubs."""
    config_loader = ConfigLoader()
    services = ServiceRegistry()
    settings = config_loader.load_settings()

    project_root = Path(__file__).resolve().parents[2]
    bridge_executable = Path(settings.serial.tracker_bridge_executable)
    if not bridge_executable.is_absolute():
        bridge_executable = project_root / bridge_executable

    tracker_manager = TrackerServiceManager(
        bridge_executable=bridge_executable,
        socket_path=Path(settings.serial.tracker_socket_path),
        aurora_port=settings.serial.aurora_port,
        poll_ms=settings.serial.tracker_poll_ms,
    )
    registration_repository = RegistrationRepository()
    rigid_solver = RigidRegistrationSolver()
    live_registration = LiveRegistrationService(
        tracker_manager=tracker_manager,
        repository=registration_repository,
        solver=rigid_solver,
    )

    services.register("tracker_manager", tracker_manager)
    services.register("registration_repository", registration_repository)
    services.register("rigid_solver", rigid_solver)
    services.register("live_registration", live_registration)

    return AppContext(config_loader=config_loader, services=services)

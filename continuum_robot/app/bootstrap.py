"""Application bootstrap helpers."""

from dataclasses import dataclass
from pathlib import Path

from continuum_robot.app.service_registry import ServiceRegistry
from continuum_robot.config.config_loader import ConfigLoader
from continuum_robot.config.settings import Settings
from continuum_robot.experiments.dat_writer import DatRunWriter
from continuum_robot.experiments.experiment_loader import ExperimentLoader
from continuum_robot.experiments.experiment_runner import ExperimentRunner
from continuum_robot.hardware.dxl_bus import DxlBus
from continuum_robot.hardware.mock_dxl_bus import MockDxlBus
from continuum_robot.hardware.mock_openrb_client import MockOpenRbClient
from continuum_robot.hardware.openrb_client import OpenRbClient
from continuum_robot.registration.live_registration_service import LiveRegistrationService
from continuum_robot.registration.repository import RegistrationRepository
from continuum_robot.registration.rigid_solver import RigidRegistrationSolver
from continuum_robot.servos.displacement_mapper import TendonDisplacementMapper
from continuum_robot.servos.neutral_calibration_service import NeutralCalibrationService
from continuum_robot.servos.pretension_validation_service import PretensionValidationService
from continuum_robot.servos.safety_guard import SafetyGuard
from continuum_robot.servos.servo_service import ServoService
from continuum_robot.tracking.mock_tracker_manager import MockTrackerManager
from continuum_robot.tracking.tracker_service_manager import TrackerServiceManager


@dataclass
class AppContext:
    """Container for top-level services used by the GUI and scripts."""

    project_root: Path
    settings: Settings
    config_loader: ConfigLoader
    services: ServiceRegistry


def _resolve_repo_path(project_root: Path, raw_path: str) -> Path:
    path = Path(raw_path)
    if path.is_absolute():
        return path
    return project_root / path


def build_app_context() -> AppContext:
    """Build and return application context for GUI and scripts."""
    config_loader = ConfigLoader()
    services = ServiceRegistry()
    settings = config_loader.load_settings()

    project_root = Path(__file__).resolve().parents[2]
    registration_path = _resolve_repo_path(project_root, settings.calibration.latest_registration_path)
    neutral_setpoints_path = _resolve_repo_path(project_root, settings.calibration.neutral_setpoints_path)
    experiment_output_dir = _resolve_repo_path(project_root, settings.experiment.output_dir)
    bridge_executable = _resolve_repo_path(project_root, settings.serial.tracker_bridge_executable)
    tracker_socket_path = _resolve_repo_path(project_root, settings.serial.tracker_socket_path)

    if settings.runtime.mock_mode:
        tracker_manager = MockTrackerManager(poll_hz=settings.runtime.poll_rate_hz)
        dxl_bus: DxlBus = MockDxlBus(servo_ids=settings.robot.servo_ids)
        openrb_client = MockOpenRbClient()
    else:
        tracker_manager = TrackerServiceManager(
            bridge_executable=bridge_executable,
            socket_path=tracker_socket_path,
            aurora_port=settings.serial.aurora_port,
            poll_ms=settings.serial.tracker_poll_ms,
        )
        dxl_bus = DxlBus()
        openrb_client = OpenRbClient()

    registration_repository = RegistrationRepository()
    rigid_solver = RigidRegistrationSolver()
    live_registration = LiveRegistrationService(
        tracker_manager=tracker_manager,
        repository=registration_repository,
        solver=rigid_solver,
        capture_tool_tip_transform=settings.registration.capture_tool_tip_transform,
    )
    neutral_calibration = NeutralCalibrationService(
        path=neutral_setpoints_path,
    )
    pretension_validation = PretensionValidationService()
    mapper = TendonDisplacementMapper(
        spool_diameter_cm=settings.robot.spool_diameter_cm,
        ticks_per_rev=settings.robot.ticks_per_revolution,
    )
    safety_guard = SafetyGuard(
        min_offset_ticks=settings.safety.position_min_offset_ticks,
        max_offset_ticks=settings.safety.position_max_offset_ticks,
        max_current_ma=settings.safety.max_current_ma,
    )
    servo_service = ServoService(
        dxl_bus=dxl_bus,
        mapper=mapper,
        safety_guard=safety_guard,
        neutral_calibration=neutral_calibration,
        pretension_validation=pretension_validation,
    )
    experiment_loader = ExperimentLoader()
    dat_writer = DatRunWriter(output_dir=experiment_output_dir)
    experiment_runner = ExperimentRunner(
        servo_service=servo_service,
        tracker_manager=tracker_manager,
        dat_writer=dat_writer,
        neutral_servo_ids=settings.robot.servo_ids,
        default_settle_time_s=settings.experiment.default_settle_time_s,
        registration_path=registration_path,
    )

    services.register("tracker_manager", tracker_manager)
    services.register("registration_repository", registration_repository)
    services.register("rigid_solver", rigid_solver)
    services.register("live_registration", live_registration)
    services.register("dxl_bus", dxl_bus)
    services.register("servo_service", servo_service)
    services.register("openrb_client", openrb_client)
    services.register("experiment_loader", experiment_loader)
    services.register("experiment_runner", experiment_runner)

    return AppContext(
        project_root=project_root,
        settings=settings,
        config_loader=config_loader,
        services=services,
    )

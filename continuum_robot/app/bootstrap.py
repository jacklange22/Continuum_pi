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
from continuum_robot.registration.legacy_compat import RegistrationAssetPaths
from continuum_robot.registration.live_registration_service import LiveRegistrationService
from continuum_robot.registration.repository import RegistrationRepository
from continuum_robot.registration.rigid_solver import RigidRegistrationSolver
from continuum_robot.services.registration_service import RegistrationService
from continuum_robot.services.system_health_service import SystemHealthService
from continuum_robot.services.tracking_service import TrackingService
from continuum_robot.servos.displacement_mapper import TendonDisplacementMapper
from continuum_robot.servos.neutral_calibration_service import NeutralCalibrationService
from continuum_robot.servos.pretension_validation_service import PretensionValidationService
from continuum_robot.servos.safety_guard import SafetyGuard
from continuum_robot.servos.servo_service import ServoService
from continuum_robot.tracking.mock_tracker_manager import MockTrackerManager
from continuum_robot.tracking.ndi_backend import TrackerBackendNDI
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
    registration_config_path = project_root / "config" / "registration.yaml"
    neutral_setpoints_path = _resolve_repo_path(project_root, settings.calibration.neutral_setpoints_path)
    experiment_output_dir = _resolve_repo_path(project_root, settings.experiment.output_dir)
    bridge_executable = _resolve_repo_path(project_root, settings.serial.tracker_bridge_executable)
    tracker_socket_path = _resolve_repo_path(project_root, settings.serial.tracker_socket_path)
    registration_asset_paths = None
    if (
        settings.registration.model_points_file
        and settings.registration.tip_points_file
        and settings.registration.T_sw_2_model_file
        and settings.registration.T_sw_2_tip_file
        and settings.registration.penprobe_file
    ):
        registration_asset_paths = RegistrationAssetPaths(
            model_points_file=_resolve_repo_path(project_root, settings.registration.model_points_file),
            tip_points_file=_resolve_repo_path(project_root, settings.registration.tip_points_file),
            T_sw_2_model_file=_resolve_repo_path(project_root, settings.registration.T_sw_2_model_file),
            T_sw_2_tip_file=_resolve_repo_path(project_root, settings.registration.T_sw_2_tip_file),
            penprobe_file=_resolve_repo_path(project_root, settings.registration.penprobe_file),
        )

    if settings.runtime.mock_mode:
        tracker_manager = MockTrackerManager(poll_hz=settings.runtime.poll_rate_hz)
        dxl_bus: DxlBus = MockDxlBus(servo_ids=settings.robot.servo_ids)
        openrb_client = MockOpenRbClient()
    else:
        tracker_backend_name = settings.serial.tracker_backend.strip().lower()
        if tracker_backend_name == "ndi":
            tracker_manager = TrackerBackendNDI(
                aurora_port=settings.serial.aurora_port,
                tracker_type=settings.serial.tracker_type,
                poll_interval_ms=settings.serial.tracker_poll_ms,
                reconnect_delay_s=settings.serial.reconnect_delay_s,
                ports_to_probe=settings.serial.tracker_ports_to_probe,
                settings_overrides=settings.serial.tracker_settings_overrides,
                tool_id_aliases=settings.serial.tracker_tool_id_aliases,
            )
        elif tracker_backend_name == "bridge":
            tracker_manager = TrackerServiceManager(
                bridge_executable=bridge_executable,
                socket_path=tracker_socket_path,
                aurora_port=settings.serial.aurora_port,
                poll_ms=settings.serial.tracker_poll_ms,
            )
        else:
            raise ValueError(
                f"Unsupported tracker backend {settings.serial.tracker_backend!r}. "
                "Use 'ndi' or 'bridge'."
            )
        dxl_bus = DxlBus()
        openrb_client = OpenRbClient()

    registration_repository = RegistrationRepository()
    rigid_solver = RigidRegistrationSolver()
    tracking_service = TrackingService(
        live_backend=tracker_manager,
        port=settings.serial.aurora_port,
        baudrate=settings.serial.baudrate,
        read_timeout_s=settings.serial.read_timeout_s,
        frame_timeout_s=settings.serial.frame_timeout_s,
        reconnect_delay_s=settings.serial.reconnect_delay_s,
        live_poll_hz=settings.runtime.poll_rate_hz,
        stale_after_s=settings.serial.tracker_freshness_timeout_s,
        registration_path=registration_path,
        runtime_coil_tool_id=settings.registration.coil_tool_id,
        registration_tool_id=settings.registration.capture_tool_id,
        config_source="config/system.yaml + config/registration.yaml",
    )
    registration_service = RegistrationService(
        tracking_service=tracking_service,
        repository=registration_repository,
        solver=rigid_solver,
        config_path=registration_config_path,
        config_source=str(registration_config_path),
    )
    system_health_service = SystemHealthService(
        tracking_service=tracking_service,
        registration_service=registration_service,
        config_source="config/system.yaml + config/registration.yaml",
    )
    live_registration = LiveRegistrationService(
        tracker_manager=tracker_manager,
        repository=registration_repository,
        solver=rigid_solver,
        capture_tool_tip_transform=settings.registration.capture_tool_tip_transform,
        asset_paths=registration_asset_paths,
        measurement_tool_id=settings.registration.capture_tool_id,
        coil_tool_id=settings.registration.coil_tool_id,
        quaternion_average_method=settings.registration.quaternion_average_method,
        model_tre_reference_radius_mm=settings.registration.model_tre_reference_radius_mm,
        tip_tre_reference_radius_mm=settings.registration.tip_tre_reference_radius_mm,
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
    services.register("tracker_backend", tracker_manager)
    services.register("tracking_service", tracking_service)
    services.register("registration_repository", registration_repository)
    services.register("rigid_solver", rigid_solver)
    services.register("registration_service", registration_service)
    services.register("system_health_service", system_health_service)
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

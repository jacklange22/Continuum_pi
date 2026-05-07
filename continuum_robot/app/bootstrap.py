"""Application bootstrap helpers."""
from __future__ import annotations

from dataclasses import dataclass
import logging
from pathlib import Path
import time

from continuum_robot.app.service_registry import ServiceRegistry
from continuum_robot.config.config_loader import ConfigLoader
from continuum_robot.config.settings import Settings
from continuum_robot.experiments.dataset_io import ExperimentDatasetLoader, ExperimentDatasetWriter
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
from continuum_robot.registration.runtime_tip_repository import RuntimeTipCalibrationRepository
from continuum_robot.services.registration_service import RegistrationService
from continuum_robot.services.runtime_tip_calibration_service import RuntimeTipCalibrationService
from continuum_robot.services.system_health_service import SystemHealthService
from continuum_robot.services.tracking_service import TrackingService
from continuum_robot.servos.displacement_mapper import TendonDisplacementMapper
from continuum_robot.servos.neutral_calibration_service import (
    NeutralCalibrationService,
    ServoCalibrationContext,
)
from continuum_robot.servos.pretension_validation_service import PretensionValidationService
from continuum_robot.servos.safety_guard import SafetyGuard
from continuum_robot.servos.servo_service import ServoService
from continuum_robot.tracking.backend_router import TrackingBackendRouter
from continuum_robot.utils.logging_setup import current_session_log_path


LOG = logging.getLogger(__name__)


@dataclass
class AppContext:
    """Container for top-level services used by the GUI and scripts."""

    project_root: Path
    settings: Settings
    config_loader: ConfigLoader
    services: ServiceRegistry
    session_log_path: Path | None = None


def _resolve_repo_path(project_root: Path, raw_path: str) -> Path:
    path = Path(raw_path)
    if path.is_absolute():
        return path
    return project_root / path


def build_app_context(*, session_log_path: Path | None = None) -> AppContext:
    """Build and return application context for GUI and scripts."""
    config_loader = ConfigLoader()
    services = ServiceRegistry()
    settings = config_loader.load_settings()
    resolved_session_log_path = session_log_path or current_session_log_path()

    project_root = Path(__file__).resolve().parents[2]
    registration_path = _resolve_repo_path(project_root, settings.calibration.latest_registration_path)
    runtime_tip_calibration_path = _resolve_repo_path(
        project_root,
        settings.calibration.latest_runtime_tip_calibration_path,
    )
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

    tracker_backend_name = settings.serial.tracker_backend.strip().lower()
    if tracker_backend_name not in {"ndi", "bridge", "disabled"}:
        raise ValueError(
            f"Unsupported tracker backend {settings.serial.tracker_backend!r}. "
            "Use 'ndi' or 'disabled'. 'bridge' is retained only as an explicit legacy compatibility path."
        )
    tracking_backend = TrackingBackendRouter(
        mock_mode=settings.runtime.mock_mode,
        preferred_backend=tracker_backend_name,
        fallback_backend=settings.serial.tracker_fallback_backend,
        fallback_enabled=settings.serial.tracker_fallback_enabled,
        aurora_port=settings.serial.aurora_port,
        tracker_type=settings.serial.tracker_type,
        poll_interval_ms=settings.serial.tracker_poll_ms,
        reconnect_delay_s=settings.serial.reconnect_delay_s,
        ports_to_probe=settings.serial.tracker_ports_to_probe,
        settings_overrides=settings.serial.tracker_settings_overrides,
        tool_id_aliases=settings.serial.tracker_tool_id_aliases,
        bridge_executable=bridge_executable,
        socket_path=tracker_socket_path,
        expected_tool_ids=(settings.registration.coil_tool_id, settings.registration.capture_tool_id),
    )
    if settings.runtime.mock_mode:
        dxl_bus = MockDxlBus(servo_ids=settings.robot.servo_ids)
        openrb_client = MockOpenRbClient()
    else:
        dxl_bus = DxlBus(config=settings.serial.dynamixel_settings)
        openrb_client = OpenRbClient(config=settings.serial.openrb_settings)

    registration_repository = RegistrationRepository()
    rigid_solver = RigidRegistrationSolver()
    tracking_service = TrackingService(
        live_backend=tracking_backend,
        port=settings.serial.aurora_port,
        baudrate=settings.serial.baudrate,
        read_timeout_s=settings.serial.read_timeout_s,
        frame_timeout_s=settings.serial.frame_timeout_s,
        reconnect_delay_s=settings.serial.reconnect_delay_s,
        live_poll_hz=max(
            float(settings.runtime.poll_rate_hz),
            1000.0 / max(1, int(settings.serial.tracker_poll_ms)),
        ),
        stale_after_s=settings.serial.tracker_freshness_timeout_s,
        registration_path=registration_path,
        runtime_tip_calibration_path=runtime_tip_calibration_path,
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
    runtime_tip_repository = RuntimeTipCalibrationRepository(latest_path=runtime_tip_calibration_path)
    runtime_tip_calibration_service = RuntimeTipCalibrationService(
        tracking_service=tracking_service,
        registration_service=registration_service,
        repository=runtime_tip_repository,
        solver=rigid_solver,
        registration_config=settings.registration,
        config_path=registration_config_path,
        config_source=str(registration_config_path),
    )
    live_registration_service = LiveRegistrationService(
        tracker_manager=tracking_backend,
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
    system_health_service = SystemHealthService(
        tracking_service=tracking_service,
        registration_service=registration_service,
        config_source="config/system.yaml + config/registration.yaml",
    )
    neutral_calibration = NeutralCalibrationService(
        path=neutral_setpoints_path,
        context=ServoCalibrationContext(
            robot_mode=settings.robot.operating_mode(),
            robot_config_name=settings.runtime.robot_config,
            servo_ids=list(settings.robot.servo_ids),
            tendon_to_servo=list(settings.robot.tendon_to_servo),
            active_segment_key=settings.robot.active_segment_key(),
            active_segment_label=settings.robot.active_segment_label(),
            active_segment_servo_ids=settings.robot.active_segment_servo_ids(),
            active_segment_pairs=settings.robot.active_segment_pairs(),
            segments=settings.robot.segment_metadata(),
            segment_order=settings.robot.segment_order(),
            selected_servo_id=settings.robot.selected_servo_id,
            expected_servo_ids=settings.robot.expected_servo_ids(),
            commanded_servo_ids=settings.robot.commanded_servo_ids(),
            mirror_pairs=settings.robot.parallel_mirror_pairs(),
            mode_profile=settings.robot.operating_context().mode_profile,
            mode_capabilities=settings.robot.operating_context().mode_capabilities,
            mode_notes=settings.robot.operating_context().mode_notes,
            ticks_per_revolution=settings.robot.ticks_per_revolution,
            spool_diameter_cm=settings.robot.spool_diameter_cm,
            position_min_offset_ticks=settings.safety.position_min_offset_ticks,
            position_max_offset_ticks=settings.safety.position_max_offset_ticks,
            default_pretension_current_threshold_ma=settings.safety.default_pretension_current_threshold_ma,
            tightening_rotation_by_servo=dict(settings.robot.tightening_rotation_by_servo),
        ),
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
        servo_model=settings.safety.servo_model,
        servo_reported_current_hard_limit_ma=settings.safety.servo_reported_current_hard_limit_ma,
        servo_reported_current_warning_ma=settings.safety.servo_reported_current_warning_ma,
        current_safety_basis=settings.safety.current_safety_basis,
        default_pretension_current_threshold_ma=settings.safety.default_pretension_current_threshold_ma,
        fine_jog_step_ticks=settings.safety.fine_jog_step_ticks,
        coarse_jog_step_ticks=settings.safety.coarse_jog_step_ticks,
        software_position_margin_ticks=settings.safety.software_position_margin_ticks,
        telemetry_stale_after_s=settings.safety.telemetry_stale_after_s,
        pretension_step_ticks=settings.safety.pretension_step_ticks,
        pretension_timeout_s=settings.safety.pretension_timeout_s,
        pretension_settle_time_s=settings.safety.pretension_settle_time_s,
        pretension_baseline_sample_count=settings.safety.pretension_baseline_sample_count,
        pretension_current_filter_window=settings.safety.pretension_current_filter_window,
        pretension_current_delta_threshold_ma=settings.safety.pretension_current_delta_threshold_ma,
        pretension_absolute_trigger_current_ma=settings.safety.pretension_absolute_trigger_current_ma,
        pretension_hard_current_stop_ma=settings.safety.pretension_hard_current_stop_ma,
        pretension_max_travel_ticks=settings.safety.pretension_max_travel_ticks,
        max_temperature_c=settings.safety.max_temperature_c,
        min_input_voltage_mv=settings.safety.min_input_voltage_mv,
        time_fn=time.monotonic,
    )
    servo_service = ServoService(
        dxl_bus=dxl_bus,
        mapper=mapper,
        safety_guard=safety_guard,
        neutral_calibration=neutral_calibration,
        pretension_validation=pretension_validation,
    )
    experiment_loader = ExperimentLoader()
    experiment_dataset_writer = ExperimentDatasetWriter(output_root=experiment_output_dir)
    experiment_dataset_loader = ExperimentDatasetLoader()
    experiment_runner = ExperimentRunner(
        project_root=project_root,
        settings=settings,
        tracking_service=tracking_service,
        servo_service=servo_service,
        output_dir=experiment_output_dir,
        dataset_writer=experiment_dataset_writer,
        dataset_loader=experiment_dataset_loader,
        default_settle_time_s=settings.experiment.default_settle_time_s,
        registration_path=registration_path,
    )

    services.register("tracking_backend", tracking_backend)
    services.register("tracking_service", tracking_service)
    services.register("registration_repository", registration_repository)
    services.register("rigid_solver", rigid_solver)
    services.register("registration_service", registration_service)
    services.register("runtime_tip_repository", runtime_tip_repository)
    services.register("runtime_tip_calibration_service", runtime_tip_calibration_service)
    services.register("live_registration", live_registration_service)
    services.register("system_health_service", system_health_service)
    services.register("dxl_bus", dxl_bus)
    services.register("servo_service", servo_service)
    services.register("openrb_client", openrb_client)
    services.register("experiment_loader", experiment_loader)
    services.register("experiment_runner", experiment_runner)
    services.register("experiment_dataset_writer", experiment_dataset_writer)
    services.register("experiment_dataset_loader", experiment_dataset_loader)

    LOG.info(
        "App context ready | mock_mode=%s | robot_mode=%s | servo_ids=%s | tracker_backend=%s | openrb_port=%s | baud=%s | session_log=%s",
        settings.runtime.mock_mode,
        settings.robot.mode,
        settings.robot.servo_ids,
        settings.serial.tracker_backend,
        settings.serial.openrb_port,
        settings.serial.baudrate,
        str(resolved_session_log_path) if resolved_session_log_path is not None else "unset",
    )

    return AppContext(
        project_root=project_root,
        settings=settings,
        config_loader=config_loader,
        services=services,
        session_log_path=resolved_session_log_path,
    )

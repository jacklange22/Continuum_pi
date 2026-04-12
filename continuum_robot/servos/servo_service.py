"""High-level servo command service."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import asdict, dataclass
import threading
import time
from typing import Callable

from continuum_robot.hardware.dxl_bus import DxlBus, ServoTelemetry
from continuum_robot.servos.displacement_mapper import TendonDisplacementMapper
from continuum_robot.servos.neutral_calibration_service import (
    NeutralCalibrationService,
    ServoCalibrationArtifact,
    ServoCalibrationSummary,
)
from continuum_robot.servos.pretension_validation_service import (
    PretensionValidationResult,
    PretensionValidationService,
)
from continuum_robot.servos.safety_guard import SafetyGuard

RAW_POSITION_MIN_TICK = 0
RAW_POSITION_MAX_TICK = 4095
CANONICAL_POSITION_CONVENTION = (
    "Raw XC330 position uses 0..4095 ticks: 0 is more tensioned, 4095 is untensioned, "
    "tighten lowers counts, loosen raises counts."
)


@dataclass
class ServoCommandResult:
    """Summary of a servo command dispatch."""

    positions_by_id: dict[int, int]
    telemetry_by_id: dict[int, ServoTelemetry]
    message: str


@dataclass
class ServoMotionAssessment:
    """Safety/readiness assessment for one live servo action."""

    servo_id: int
    ready: bool
    reason: str
    telemetry: ServoTelemetry
    safe_min_tick: int | None = None
    safe_max_tick: int | None = None
    tightening_direction: str | None = None
    blocking_reasons: tuple[str, ...] = ()
    external_power_required: bool = False
    external_power_ready: bool | None = None


@dataclass
class ServoDiscoverySnapshot:
    """Structured one-servo discovery/readiness snapshot."""

    status: str
    connected: bool
    bus_reachable: bool
    expected_servo_id: int | None
    selected_servo_id: int | None
    discovered_ids: list[int]
    scan_range: tuple[int, int] | None
    telemetry: ServoTelemetry | None
    motion_assessment: ServoMotionAssessment | None
    message: str


@dataclass
class ServoIdAssignmentResult:
    """Outcome of a maintenance-only servo ID assignment."""

    current_id: int
    new_id: int
    success: bool
    blocked: bool
    status: str
    selected_ids: list[int]
    message: str


@dataclass
class ServoJogResult:
    """Outcome of a conservative one-servo jog command."""

    servo_id: int
    command_direction: str
    step_ticks: int
    delta_ticks: int
    success: bool
    blocked: bool
    status: str
    message: str
    goal_tick: int | None
    telemetry: ServoTelemetry | None
    assessment: ServoMotionAssessment | None
    current_position_tick: int | None = None
    unclamped_goal_tick: int | None = None
    safe_min_tick: int | None = None
    safe_max_tick: int | None = None
    clamped: bool = False


@dataclass
class ServoMotionPlan:
    """Canonical one-servo operator motion plan in raw position counts."""

    servo_id: int
    action: str
    current_position_tick: int | None
    step_ticks: int
    delta_ticks: int
    unclamped_target_tick: int | None
    clamped_target_tick: int | None
    safe_min_tick: int | None
    safe_max_tick: int | None
    clamped: bool
    allowed: bool
    block_reason: str
    assessment: ServoMotionAssessment | None


@dataclass
class NeutralCaptureResult:
    """Outcome of capturing and persisting neutral setpoints."""

    servo_ids: list[int]
    setpoints_by_id: dict[int, int]
    safe_bounds_by_id: dict[int, tuple[int, int]]
    artifact_path: str
    message: str


@dataclass
class ServoBenchDebugSnapshot:
    """Compact one-servo bench-debug snapshot."""

    expected_servo_id: int | None
    selected_servo_id: int | None
    selected_port: str | None
    selected_baud: int | None
    bus_connected: bool
    bus_reachable: bool
    ping_ok: bool | None
    ping_message: str
    identity_read_ok: bool | None
    telemetry_read_ok: bool | None
    telemetry: ServoTelemetry | None
    calibration_exists: bool
    calibration_compatible: bool
    calibration_entries_loaded: list[int]
    one_servo_mode_ok: bool
    safe_bounds_loaded: bool
    motion_ready: bool
    motion_block_reason: str
    status: str
    message: str
    motion_assessment: ServoMotionAssessment | None = None


@dataclass
class ConfiguredServoBringupEntry:
    """Bring-up status for one configured servo ID."""

    servo_id: int
    ping_ok: bool
    identity_read_ok: bool
    telemetry_read_ok: bool
    telemetry: ServoTelemetry | None
    motion_assessment: ServoMotionAssessment | None
    status: str
    message: str


@dataclass
class ConfiguredServoBringupSnapshot:
    """Structured configured-servo discovery/readback snapshot."""

    connected: bool
    bus_reachable: bool
    selected_port: str | None
    selected_baud: int | None
    expected_servo_ids: list[int]
    discovered_ids: list[int]
    missing_servo_ids: list[int]
    unexpected_servo_ids: list[int]
    servo_entries: dict[int, ConfiguredServoBringupEntry]
    all_expected_present: bool
    all_expected_identity_ok: bool
    all_expected_telemetry_ok: bool
    all_motion_ready: bool
    status: str
    message: str


@dataclass
class ServoRuntimeStateEntry:
    """Canonical live per-servo runtime state used by multiple GUI surfaces."""

    servo_id: int
    telemetry: ServoTelemetry | None
    identity_read_ok: bool
    telemetry_read_ok: bool
    detected: bool
    telemetry_status: str
    motion_assessment: ServoMotionAssessment | None
    pretension_assessment: ServoMotionAssessment | None
    message: str


@dataclass
class ServoRuntimeStateSnapshot:
    """Canonical multi-servo runtime snapshot shared by System, Servos, and Pretension."""

    connected: bool
    expected_servo_ids: list[int]
    detected_servo_ids: list[int]
    missing_servo_ids: list[int]
    unexpected_servo_ids: list[int]
    entries: dict[int, ServoRuntimeStateEntry]
    telemetry_ready_count: int
    motion_ready_count: int
    pretension_ready_count: int
    all_motion_ready: bool
    selected_servo_id: int | None
    message: str


@dataclass
class PretensionRoutineResult:
    """Outcome of the cautious startup pretension routine."""

    servo_id: int
    status: str
    success: bool
    message: str
    threshold_ma: int
    final_position_tick: int | None
    final_current_ma: int | None
    steps_taken: int
    tightening_direction: str | None
    start_position_tick: int | None = None
    untensioned_reference_tick: int | None = None
    current_position_tick: int | None = None
    last_commanded_target_tick: int | None = None
    baseline_current_ma: float | None = None
    filtered_current_ma: float | None = None
    current_delta_ma: float | None = None
    absolute_trigger_current_ma: int | None = None
    hard_current_stop_ma: int | None = None
    elapsed_s: float = 0.0
    stop_reason: str | None = None
    parameters: dict[str, int | float | None] | None = None


@dataclass
class PretensionBaselineMeasurement:
    """Filtered baseline-current estimate for one selected servo."""

    servo_id: int
    sample_count: int
    samples_ma: list[int]
    baseline_current_ma: float
    filtered_current_ma: float
    position_tick: int | None
    message: str


@dataclass
class PretensionParameters:
    """Operator-facing parameters for selected-servo MVP pretensioning."""

    untensioned_reference_tick: int
    step_ticks: int
    settle_time_s: float
    baseline_sample_count: int
    current_filter_window: int
    current_delta_threshold_ma: int
    absolute_trigger_current_ma: int | None
    hard_current_stop_ma: int
    max_travel_ticks: int
    timeout_s: float


@dataclass
class PretensionWindow:
    """Effective one-servo pretension travel window for the current parameters."""

    servo_id: int
    hardware_safe_min_tick: int
    hardware_safe_max_tick: int
    untensioned_reference_tick: int
    effective_min_target_tick: int
    effective_max_target_tick: int


@dataclass
class ServoBusOwnershipStatus:
    """Structured ownership state for the live DYNAMIXEL bus."""

    active: bool
    owner: str | None
    reason: str | None
    servo_id: int | None
    held_by_current_thread: bool
    started_at_monotonic_s: float | None


class ServoBusBusyError(RuntimeError):
    """Raised when a non-owner thread tries to touch the live bus during an exclusive run."""

    def __init__(
        self,
        message: str,
        *,
        owner: str | None = None,
        reason: str | None = None,
        servo_id: int | None = None,
    ) -> None:
        super().__init__(str(message))
        self.owner = owner
        self.reason = reason
        self.servo_id = servo_id


class ServoService:
    """Coordinates mapping, validation, persistence, and low-level bus writes."""

    def __init__(
        self,
        dxl_bus: DxlBus,
        mapper: TendonDisplacementMapper,
        safety_guard: SafetyGuard,
        neutral_calibration: NeutralCalibrationService,
        pretension_validation: PretensionValidationService,
        *,
        sleep_fn: Callable[[float], None] = time.sleep,
        time_fn: Callable[[], float] = time.monotonic,
    ) -> None:
        self.dxl_bus = dxl_bus
        self.mapper = mapper
        self.safety_guard = safety_guard
        self.neutral_calibration = neutral_calibration
        self.pretension_validation = pretension_validation
        self._sleep_fn = sleep_fn
        self._time_fn = time_fn
        self._bus_state_lock = threading.RLock()
        self._bus_io_lock = threading.RLock()
        self._exclusive_bus_owner: str | None = None
        self._exclusive_bus_reason: str | None = None
        self._exclusive_bus_servo_id: int | None = None
        self._exclusive_bus_thread_id: int | None = None
        self._exclusive_bus_started_at: float | None = None
        self._exclusive_bus_depth: int = 0
        self._last_goal_positions_by_id: dict[int, int] = {}
        self._last_goal_command_monotonic_s: dict[int, float] = {}

    @property
    def is_connected(self) -> bool:
        return self.dxl_bus.is_connected

    def connect(self, port: str, baudrate: int) -> None:
        self._guard_bus_call(
            "connect to OpenRB / DYNAMIXEL",
            lambda: self.dxl_bus.connect(port, baudrate),
        )

    def disconnect(self) -> None:
        self._guard_bus_call(
            "disconnect OpenRB / DYNAMIXEL",
            self.dxl_bus.disconnect,
        )
        self._last_goal_positions_by_id.clear()
        self._last_goal_command_monotonic_s.clear()

    @staticmethod
    def position_convention_summary() -> str:
        return CANONICAL_POSITION_CONVENTION

    @staticmethod
    def raw_position_range() -> tuple[int, int]:
        return (RAW_POSITION_MIN_TICK, RAW_POSITION_MAX_TICK)

    def is_single_servo_bench_mode(self) -> bool:
        return (
            str(self.neutral_calibration.context.robot_mode).strip().lower() == "1-servo"
            and len(self.neutral_calibration.context.servo_ids) == 1
        )

    @staticmethod
    def require_calibrated_bounds_for_individual_motion() -> bool:
        return False

    def scan_ids(self, min_id: int = 1, max_id: int = 20) -> list[int]:
        return self._guard_bus_call(
            "scan configured servo IDs",
            lambda: self.dxl_bus.scan_ids(min_id=min_id, max_id=max_id),
        )

    def assign_servo_id(self, current_id: int, new_id: int) -> None:
        self._write_servo_id(int(current_id), int(new_id))

    def assign_servo_id_safely(self, current_id: int, new_id: int) -> ServoIdAssignmentResult:
        discovery = self.discover_one_servo(expected_servo_id=int(current_id), allow_scan=True)
        if not discovery.connected:
            return ServoIdAssignmentResult(
                current_id=int(current_id),
                new_id=int(new_id),
                success=False,
                blocked=True,
                status="disconnected",
                selected_ids=[],
                message="DYNAMIXEL bus is disconnected. Connect OpenRB before maintenance actions.",
            )
        if discovery.selected_servo_id != int(current_id):
            return ServoIdAssignmentResult(
                current_id=int(current_id),
                new_id=int(new_id),
                success=False,
                blocked=True,
                status=discovery.status,
                selected_ids=list(discovery.discovered_ids),
                message=(
                    "Maintenance ID assignment requires exactly one known target servo. "
                    f"Discovery result: {discovery.message}"
                ),
            )
        try:
            self._write_servo_id(int(current_id), int(new_id))
        except Exception as exc:
            return ServoIdAssignmentResult(
                current_id=int(current_id),
                new_id=int(new_id),
                success=False,
                blocked=True,
                status="blocked",
                selected_ids=[int(current_id)],
                message=str(exc),
            )
        return ServoIdAssignmentResult(
            current_id=int(current_id),
            new_id=int(new_id),
            success=True,
            blocked=False,
            status="assigned",
            selected_ids=[int(new_id)],
            message=f"Servo ID changed from {current_id} to {new_id} and verified by readback.",
        )

    def read_telemetry(self, servo_ids: list[int]) -> dict[int, ServoTelemetry]:
        return self._guard_bus_call(
            "read servo telemetry",
            lambda: self.dxl_bus.read_telemetry(servo_ids),
        )

    def read_live_telemetry(self, servo_ids: list[int]) -> dict[int, ServoTelemetry]:
        return self._guard_bus_call(
            "read live servo telemetry",
            lambda: self.dxl_bus.read_live_telemetry(servo_ids),
        )

    def bus_ownership_status(self) -> ServoBusOwnershipStatus:
        """Return the current exclusive-bus ownership state."""
        with self._bus_state_lock:
            thread_id = threading.get_ident()
            return ServoBusOwnershipStatus(
                active=self._exclusive_bus_thread_id is not None,
                owner=self._exclusive_bus_owner,
                reason=self._exclusive_bus_reason,
                servo_id=self._exclusive_bus_servo_id,
                held_by_current_thread=self._exclusive_bus_thread_id == thread_id,
                started_at_monotonic_s=self._exclusive_bus_started_at,
            )

    def has_exclusive_bus_owner(self) -> bool:
        return bool(self.bus_ownership_status().active)

    def bus_busy_message(self, *, action: str | None = None) -> str:
        status = self.bus_ownership_status()
        return self._format_bus_busy_message(status, action=action)

    @contextmanager
    def exclusive_bus_operation(
        self,
        *,
        owner: str,
        servo_id: int | None = None,
        reason: str | None = None,
    ):
        """Grant one thread exclusive ownership of the live DYNAMIXEL bus."""
        owner_name = str(owner).strip() or "servo operation"
        current_thread_id = threading.get_ident()
        with self._bus_state_lock:
            if self._exclusive_bus_thread_id is None:
                self._exclusive_bus_owner = owner_name
                self._exclusive_bus_reason = str(reason).strip() if reason else None
                self._exclusive_bus_servo_id = int(servo_id) if servo_id is not None else None
                self._exclusive_bus_thread_id = current_thread_id
                self._exclusive_bus_started_at = float(self._time_fn())
                self._exclusive_bus_depth = 1
            elif self._exclusive_bus_thread_id == current_thread_id:
                self._exclusive_bus_depth += 1
            else:
                status = self.bus_ownership_status()
                raise ServoBusBusyError(
                    self._format_bus_busy_message(status, action=owner_name),
                    owner=status.owner,
                    reason=status.reason,
                    servo_id=status.servo_id,
                )
        try:
            yield self.bus_ownership_status()
        finally:
            with self._bus_state_lock:
                if self._exclusive_bus_thread_id != current_thread_id:
                    return
                self._exclusive_bus_depth = max(0, int(self._exclusive_bus_depth) - 1)
                if self._exclusive_bus_depth == 0:
                    self._exclusive_bus_owner = None
                    self._exclusive_bus_reason = None
                    self._exclusive_bus_servo_id = None
                    self._exclusive_bus_thread_id = None
                    self._exclusive_bus_started_at = None

    def telemetry_age_s(self, telemetry: ServoTelemetry | None) -> float | None:
        if telemetry is None:
            return None
        return self.safety_guard.telemetry_age_s(telemetry.last_read_monotonic_s)

    def telemetry_is_fresh(self, telemetry: ServoTelemetry | None) -> bool | None:
        if telemetry is None:
            return None
        return self.safety_guard.telemetry_is_fresh(telemetry.last_read_monotonic_s)

    def telemetry_freshness_threshold_s(self) -> float:
        return float(self.safety_guard.telemetry_stale_after_s)

    def last_goal_positions(self) -> dict[int, int]:
        """Return the most recently written goal positions by servo ID."""
        with self._bus_state_lock:
            return dict(self._last_goal_positions_by_id)

    def last_goal_command_times(self) -> dict[int, float]:
        """Return monotonic timestamps for the most recently written goal positions."""
        with self._bus_state_lock:
            return dict(self._last_goal_command_monotonic_s)

    def _guard_bus_call(self, action: str, fn: Callable[[], object]):
        self._assert_bus_access(action=action)
        with self._bus_io_lock:
            self._assert_bus_access(action=action)
            return fn()

    def _assert_bus_access(self, *, action: str) -> None:
        status = self.bus_ownership_status()
        if status.active and not status.held_by_current_thread:
            raise ServoBusBusyError(
                self._format_bus_busy_message(status, action=action),
                owner=status.owner,
                reason=status.reason,
                servo_id=status.servo_id,
            )

    def _format_bus_busy_message(
        self,
        status: ServoBusOwnershipStatus,
        *,
        action: str | None = None,
    ) -> str:
        if not status.active:
            return "DYNAMIXEL bus is available."
        owner_text = str(status.owner or "servo operation")
        servo_text = (
            f" on servo {int(status.servo_id)}"
            if status.servo_id is not None
            else ""
        )
        reason_text = f" ({status.reason})" if status.reason else ""
        action_text = f"{action} is paused because " if action else ""
        return (
            f"{action_text}the DYNAMIXEL bus is owned by active {owner_text}{servo_text}{reason_text}. "
            "Background refresh is paused until that run ends."
        )

    def _write_goal_positions(self, positions_by_id: dict[int, int]) -> None:
        self._guard_bus_call(
            "write servo goal positions",
            lambda: self.dxl_bus.write_goal_positions(positions_by_id),
        )
        written_at = float(self._time_fn())
        with self._bus_state_lock:
            for servo_id, goal in dict(positions_by_id).items():
                self._last_goal_positions_by_id[int(servo_id)] = int(goal)
                self._last_goal_command_monotonic_s[int(servo_id)] = written_at

    def _write_servo_id(self, current_id: int, new_id: int) -> None:
        self._guard_bus_call(
            "write servo ID",
            lambda: self.dxl_bus.write_servo_id(int(current_id), int(new_id)),
        )

    def _ping_servo(self, servo_id: int) -> bool:
        return bool(
            self._guard_bus_call(
                "ping servo",
                lambda: self.dxl_bus.ping_servo(int(servo_id)),
            )
        )

    def _ping_servo_snapshot(self, servo_id: int):
        return self._guard_bus_call(
            "ping servo",
            lambda: self.dxl_bus.ping_servo_snapshot(int(servo_id)),
        )

    def load_neutral_setpoints(self) -> dict[int, int]:
        return self.neutral_calibration.load_neutral_setpoints()

    def save_neutral_setpoints(self, setpoints_by_id: dict[int, int]) -> None:
        self.neutral_calibration.save_neutral_setpoints(setpoints_by_id)

    def load_calibration_artifact(self) -> ServoCalibrationArtifact:
        return self.neutral_calibration.load_calibration_artifact()

    def get_calibration_summary(self) -> ServoCalibrationSummary:
        return self.neutral_calibration.get_calibration_summary()

    def capture_neutral_setpoints(self, servo_ids: list[int]) -> dict[int, int]:
        result = self.capture_and_save_neutral_setpoints(servo_ids)
        return dict(result.setpoints_by_id)

    def capture_and_save_neutral_setpoints(
        self,
        servo_ids: list[int],
        *,
        capture_source: str = "live_present_position",
    ) -> NeutralCaptureResult:
        if not servo_ids:
            raise ValueError("At least one servo ID is required to capture neutral setpoints.")
        telemetry = self.read_telemetry(servo_ids)
        setpoints: dict[int, int] = {}
        safe_bounds_by_id: dict[int, tuple[int, int]] = {}
        for servo_id in servo_ids:
            self._validate_capture_telemetry(telemetry[int(servo_id)], servo_id=int(servo_id))
            position = telemetry[servo_id].present_position
            if position is None:
                raise RuntimeError(f"Servo {servo_id} position is unavailable")
            neutral_tick = int(position)
            setpoints[servo_id] = neutral_tick
            safe_bounds_by_id[int(servo_id)] = self._safe_bounds_from_neutral(
                servo_id=int(servo_id),
                neutral_tick=neutral_tick,
                telemetry=telemetry[int(servo_id)],
                min_offset_ticks=int(self.safety_guard.min_offset_ticks),
                max_offset_ticks=int(self.safety_guard.max_offset_ticks),
            )
        self.neutral_calibration.save_neutral_setpoints(
            setpoints,
            safe_bounds_by_id=safe_bounds_by_id,
            capture_source=capture_source,
        )
        return NeutralCaptureResult(
            servo_ids=[int(servo_id) for servo_id in servo_ids],
            setpoints_by_id=setpoints,
            safe_bounds_by_id=dict(safe_bounds_by_id),
            artifact_path=str(self.neutral_calibration.path),
            message=(
                f"Captured and saved neutral setpoints for servo IDs {sorted(setpoints)} "
                f"to {self.neutral_calibration.path}."
            ),
        )

    def build_bench_debug_snapshot(self, expected_servo_id: int | None) -> ServoBenchDebugSnapshot:
        summary = self.neutral_calibration.get_calibration_summary()
        calibration_entries_loaded = sorted(summary.servo_entries)
        one_servo_mode_ok = (
            str(self.neutral_calibration.context.robot_mode).strip().lower() == "1-servo"
            and len(self.neutral_calibration.context.servo_ids) == 1
        )
        selected_port = self.dxl_bus.port
        selected_baud = self.dxl_bus.baudrate
        if not self.is_connected:
            return ServoBenchDebugSnapshot(
                expected_servo_id=int(expected_servo_id) if expected_servo_id is not None else None,
                selected_servo_id=None,
                selected_port=selected_port,
                selected_baud=selected_baud,
                bus_connected=False,
                bus_reachable=False,
                ping_ok=None,
                ping_message="DYNAMIXEL bus is disconnected.",
                identity_read_ok=None,
                telemetry_read_ok=None,
                telemetry=None,
                calibration_exists=summary.exists,
                calibration_compatible=summary.compatible,
                calibration_entries_loaded=calibration_entries_loaded,
                one_servo_mode_ok=one_servo_mode_ok,
                safe_bounds_loaded=False,
                motion_ready=False,
                motion_block_reason="DYNAMIXEL bus is disconnected.",
                status="disconnected",
                message="DYNAMIXEL bus is disconnected.",
            )
        if expected_servo_id is None:
            return ServoBenchDebugSnapshot(
                expected_servo_id=None,
                selected_servo_id=None,
                selected_port=selected_port,
                selected_baud=selected_baud,
                bus_connected=True,
                bus_reachable=False,
                ping_ok=None,
                ping_message="No expected servo ID is configured.",
                identity_read_ok=None,
                telemetry_read_ok=None,
                telemetry=None,
                calibration_exists=summary.exists,
                calibration_compatible=summary.compatible,
                calibration_entries_loaded=calibration_entries_loaded,
                one_servo_mode_ok=one_servo_mode_ok,
                safe_bounds_loaded=False,
                motion_ready=False,
                motion_block_reason="No expected servo ID is configured.",
                status="no_expected_id",
                message="No expected servo ID is configured.",
            )

        ping = self._ping_servo_snapshot(int(expected_servo_id))
        if not ping.responded:
            ping_message = ping.error or f"Servo {expected_servo_id} did not respond to ping."
            return ServoBenchDebugSnapshot(
                expected_servo_id=int(expected_servo_id),
                selected_servo_id=None,
                selected_port=selected_port,
                selected_baud=selected_baud,
                bus_connected=True,
                bus_reachable=False,
                ping_ok=False,
                ping_message=ping_message,
                identity_read_ok=False,
                telemetry_read_ok=False,
                telemetry=None,
                calibration_exists=summary.exists,
                calibration_compatible=summary.compatible,
                calibration_entries_loaded=calibration_entries_loaded,
                one_servo_mode_ok=one_servo_mode_ok,
                safe_bounds_loaded=False,
                motion_ready=False,
                motion_block_reason=ping_message,
                status="ping_failed",
                message=(
                    f"Expected servo ID {expected_servo_id} did not respond to ping. {ping_message}"
                ),
            )

        try:
            telemetry = self.read_telemetry([int(expected_servo_id)])[int(expected_servo_id)]
        except Exception as exc:
            message = (
                f"Servo {expected_servo_id} responded to ping, but identity/telemetry read failed: {exc}"
            )
            return ServoBenchDebugSnapshot(
                expected_servo_id=int(expected_servo_id),
                selected_servo_id=int(expected_servo_id),
                selected_port=selected_port,
                selected_baud=selected_baud,
                bus_connected=True,
                bus_reachable=True,
                ping_ok=True,
                ping_message=f"Ping succeeded for servo {expected_servo_id}.",
                identity_read_ok=False,
                telemetry_read_ok=False,
                telemetry=None,
                calibration_exists=summary.exists,
                calibration_compatible=summary.compatible,
                calibration_entries_loaded=calibration_entries_loaded,
                one_servo_mode_ok=one_servo_mode_ok,
                safe_bounds_loaded=False,
                motion_ready=False,
                motion_block_reason=message,
                status="ping_only",
                message=message,
            )

        identity_read_ok = self._identity_read_ok(telemetry)
        telemetry_read_ok = self._telemetry_read_ok(telemetry)
        try:
            motion_assessment = self.assess_motion(
                int(expected_servo_id),
                require_calibrated_bounds=self.require_calibrated_bounds_for_individual_motion(),
                telemetry=telemetry,
            )
        except Exception as exc:
            motion_assessment = None
            motion_block_reason = str(exc)
            motion_ready = False
        else:
            motion_block_reason = "" if motion_assessment.ready else motion_assessment.reason
            motion_ready = bool(motion_assessment.ready)
        safe_bounds_loaded = bool(
            summary.exists
            and summary.compatible
            and (
                entry := summary.servo_entries.get(int(expected_servo_id))
            ) is not None
            and entry.safe_min_tick is not None
            and entry.safe_max_tick is not None
        )
        read_failures = []
        if not identity_read_ok:
            read_failures.append(
                f"identity read incomplete: {telemetry.identity_error or self._missing_identity_fields(telemetry)}"
            )
        if not telemetry_read_ok:
            read_failures.append(
                f"telemetry read incomplete: {telemetry.telemetry_error or self._missing_telemetry_fields(telemetry)}"
            )
        if read_failures:
            status = "ping_only" if not identity_read_ok else "identity_only"
            message = (
                f"Servo {expected_servo_id} responded to ping, but follow-up reads are incomplete: "
                + " | ".join(read_failures)
            )
            motion_ready = False
            motion_block_reason = message
        else:
            status = "telemetry_ready" if motion_ready else "telemetry_read_ok"
            active_range = self.raw_position_range() if self.is_single_servo_bench_mode() else (
                motion_assessment.safe_min_tick,
                motion_assessment.safe_max_tick,
            ) if motion_assessment is not None else (None, None)
            message = (
                f"Servo {expected_servo_id} ping, identity, and telemetry reads succeeded. "
                f"Motion status: {motion_block_reason or 'ready'}. "
                f"Active range: {active_range[0]}..{active_range[1]}"
            )
        return ServoBenchDebugSnapshot(
            expected_servo_id=int(expected_servo_id),
            selected_servo_id=int(expected_servo_id),
            selected_port=selected_port,
            selected_baud=selected_baud,
            bus_connected=True,
            bus_reachable=True,
            ping_ok=True,
            ping_message=f"Ping succeeded for servo {expected_servo_id}.",
            identity_read_ok=identity_read_ok,
            telemetry_read_ok=telemetry_read_ok,
            telemetry=telemetry,
            calibration_exists=summary.exists,
            calibration_compatible=summary.compatible,
            calibration_entries_loaded=calibration_entries_loaded,
            one_servo_mode_ok=one_servo_mode_ok,
            safe_bounds_loaded=safe_bounds_loaded,
            motion_ready=motion_ready,
            motion_block_reason=motion_block_reason,
            status=status,
            message=message,
            motion_assessment=motion_assessment,
        )

    def build_configured_servo_bringup_snapshot(
        self,
        expected_servo_ids: list[int],
        *,
        allow_scan: bool = True,
    ) -> ConfiguredServoBringupSnapshot:
        expected_ids = sorted({int(servo_id) for servo_id in expected_servo_ids})
        selected_port = self.dxl_bus.port
        selected_baud = self.dxl_bus.baudrate
        if not self.is_connected:
            return ConfiguredServoBringupSnapshot(
                connected=False,
                bus_reachable=False,
                selected_port=selected_port,
                selected_baud=selected_baud,
                expected_servo_ids=expected_ids,
                discovered_ids=[],
                missing_servo_ids=list(expected_ids),
                unexpected_servo_ids=[],
                servo_entries={},
                all_expected_present=False,
                all_expected_identity_ok=False,
                all_expected_telemetry_ok=False,
                all_motion_ready=False,
                status="disconnected",
                message="DYNAMIXEL bus is disconnected.",
            )
        if not expected_ids:
            return ConfiguredServoBringupSnapshot(
                connected=True,
                bus_reachable=False,
                selected_port=selected_port,
                selected_baud=selected_baud,
                expected_servo_ids=[],
                discovered_ids=[],
                missing_servo_ids=[],
                unexpected_servo_ids=[],
                servo_entries={},
                all_expected_present=False,
                all_expected_identity_ok=False,
                all_expected_telemetry_ok=False,
                all_motion_ready=False,
                status="no_expected_ids",
                message="No expected servo IDs are configured.",
            )

        if allow_scan:
            discovered_ids = self.scan_ids(
                min_id=int(self.dxl_bus.config.discovery_min_id),
                max_id=int(self.dxl_bus.config.discovery_max_id),
            )
        else:
            discovered_ids = [
                servo_id
                for servo_id in expected_ids
                if self._ping_servo(int(servo_id))
            ]

        telemetry_by_id = self.read_telemetry(list(expected_ids))
        missing_servo_ids = [servo_id for servo_id in expected_ids if servo_id not in discovered_ids]
        unexpected_servo_ids = [
            servo_id for servo_id in discovered_ids if servo_id not in expected_ids
        ]
        servo_entries: dict[int, ConfiguredServoBringupEntry] = {}
        all_expected_identity_ok = True
        all_expected_telemetry_ok = True
        all_motion_ready = True

        for servo_id in expected_ids:
            telemetry = telemetry_by_id.get(int(servo_id))
            ping_ok = int(servo_id) in discovered_ids
            if not ping_ok:
                entry = ConfiguredServoBringupEntry(
                    servo_id=int(servo_id),
                    ping_ok=False,
                    identity_read_ok=False,
                    telemetry_read_ok=False,
                    telemetry=telemetry,
                    motion_assessment=None,
                    status="expected_missing",
                    message=f"Expected servo {servo_id} did not respond to ping.",
                )
                all_expected_identity_ok = False
                all_expected_telemetry_ok = False
                all_motion_ready = False
                servo_entries[int(servo_id)] = entry
                continue

            identity_read_ok = bool(telemetry is not None and self._identity_read_ok(telemetry))
            telemetry_read_ok = bool(telemetry is not None and self._telemetry_read_ok(telemetry))
            motion_assessment = (
                self.assess_motion(
                    int(servo_id),
                    require_calibrated_bounds=self.require_calibrated_bounds_for_individual_motion(),
                    telemetry=telemetry,
                )
                if telemetry is not None
                else None
            )
            if not identity_read_ok:
                message = (
                    f"Servo {servo_id} ping succeeded, but identity read is incomplete: "
                    f"{telemetry.identity_error if telemetry is not None else 'identity unavailable'}"
                )
                status = "identity_read_failed"
                all_expected_identity_ok = False
                all_expected_telemetry_ok = False
                all_motion_ready = False
            elif not telemetry_read_ok:
                message = (
                    f"Servo {servo_id} ping succeeded, but telemetry read is incomplete: "
                    f"{telemetry.telemetry_error if telemetry is not None else 'telemetry unavailable'}"
                )
                status = "telemetry_read_failed"
                all_expected_telemetry_ok = False
                all_motion_ready = False
            elif motion_assessment is not None and not motion_assessment.ready:
                message = (
                    f"Servo {servo_id} telemetry read succeeded, but motion is blocked: "
                    f"{motion_assessment.reason}"
                )
                status = "motion_blocked"
                all_motion_ready = False
            else:
                message = f"Servo {servo_id} telemetry read succeeded and it is ready for individual jog."
                status = "ready"

            servo_entries[int(servo_id)] = ConfiguredServoBringupEntry(
                servo_id=int(servo_id),
                ping_ok=True,
                identity_read_ok=identity_read_ok,
                telemetry_read_ok=telemetry_read_ok,
                telemetry=telemetry,
                motion_assessment=motion_assessment,
                status=status,
                message=message,
            )

        all_expected_present = not missing_servo_ids
        if missing_servo_ids:
            status = "expected_missing"
        elif unexpected_servo_ids:
            status = "unexpected_found"
        elif not all_expected_telemetry_ok:
            status = "telemetry_incomplete"
        elif not all_motion_ready:
            status = "motion_blocked"
        else:
            status = "ready"

        telemetry_ready_count = sum(
            1 for entry in servo_entries.values() if entry.telemetry_read_ok
        )
        motion_ready_count = sum(
            1
            for entry in servo_entries.values()
            if entry.motion_assessment is not None and entry.motion_assessment.ready
        )
        details = [
            f"expected={expected_ids}",
            f"discovered={list(discovered_ids)}",
            f"telemetry_ok={telemetry_ready_count}/{len(expected_ids)}",
            f"motion_ready={motion_ready_count}/{len(expected_ids)}",
        ]
        if missing_servo_ids:
            details.append(f"missing={missing_servo_ids}")
        if unexpected_servo_ids:
            details.append(f"unexpected={unexpected_servo_ids}")
        message = "Configured servo bring-up: " + "; ".join(details) + "."
        return ConfiguredServoBringupSnapshot(
            connected=True,
            bus_reachable=bool(discovered_ids),
            selected_port=selected_port,
            selected_baud=selected_baud,
            expected_servo_ids=expected_ids,
            discovered_ids=list(discovered_ids),
            missing_servo_ids=missing_servo_ids,
            unexpected_servo_ids=unexpected_servo_ids,
            servo_entries=servo_entries,
            all_expected_present=all_expected_present,
            all_expected_identity_ok=all_expected_identity_ok,
            all_expected_telemetry_ok=all_expected_telemetry_ok,
            all_motion_ready=all_motion_ready,
            status=status,
            message=message,
        )

    def build_runtime_servo_snapshot(
        self,
        expected_servo_ids: list[int],
        *,
        selected_servo_id: int | None = None,
        selected_pretension_parameters: PretensionParameters | None = None,
        include_scan: bool = True,
    ) -> ServoRuntimeStateSnapshot:
        """Return one canonical live servo snapshot for GUI readiness surfaces."""
        expected_ids = sorted({int(servo_id) for servo_id in expected_servo_ids})
        if not self.is_connected:
            return ServoRuntimeStateSnapshot(
                connected=False,
                expected_servo_ids=expected_ids,
                detected_servo_ids=[],
                missing_servo_ids=list(expected_ids),
                unexpected_servo_ids=[],
                entries={},
                telemetry_ready_count=0,
                motion_ready_count=0,
                pretension_ready_count=0,
                all_motion_ready=False,
                selected_servo_id=int(selected_servo_id) if selected_servo_id is not None else None,
                message="DYNAMIXEL bus is disconnected.",
            )
        if not expected_ids:
            return ServoRuntimeStateSnapshot(
                connected=True,
                expected_servo_ids=[],
                detected_servo_ids=[],
                missing_servo_ids=[],
                unexpected_servo_ids=[],
                entries={},
                telemetry_ready_count=0,
                motion_ready_count=0,
                pretension_ready_count=0,
                all_motion_ready=False,
                selected_servo_id=int(selected_servo_id) if selected_servo_id is not None else None,
                message="No expected servo IDs are configured.",
            )

        discovered_ids = (
            self.scan_ids(
                min_id=int(self.dxl_bus.config.discovery_min_id),
                max_id=int(self.dxl_bus.config.discovery_max_id),
            )
            if include_scan
            else []
        )
        telemetry_by_id = self.read_telemetry(expected_ids)
        entries: dict[int, ServoRuntimeStateEntry] = {}
        detected_servo_ids: list[int] = []
        telemetry_ready_count = 0
        motion_ready_count = 0
        pretension_ready_count = 0

        for servo_id in expected_ids:
            telemetry = telemetry_by_id.get(int(servo_id))
            identity_read_ok = bool(telemetry is not None and self._identity_read_ok(telemetry))
            telemetry_read_ok = bool(telemetry is not None and self._telemetry_read_ok(telemetry))
            motion_assessment = (
                self.assess_motion(
                    int(servo_id),
                    require_calibrated_bounds=self.require_calibrated_bounds_for_individual_motion(),
                    telemetry=telemetry,
                )
                if telemetry is not None
                else None
            )
            pretension_parameters = (
                selected_pretension_parameters
                if selected_servo_id is not None and int(servo_id) == int(selected_servo_id)
                else None
            )
            pretension_assessment = (
                self.assess_pretension_readiness(
                    int(servo_id),
                    parameters=pretension_parameters,
                    telemetry=telemetry,
                )
                if telemetry is not None
                else None
            )
            telemetry_status = self._runtime_telemetry_status(
                telemetry=telemetry,
                motion_assessment=motion_assessment,
            )
            detected = bool(
                int(servo_id) in discovered_ids
                or
                telemetry_read_ok
                or identity_read_ok
                or (
                    telemetry is not None
                    and (
                        telemetry.present_position is not None
                        or telemetry.present_current_ma is not None
                        or telemetry.reported_servo_id is not None
                    )
                )
            )
            if detected:
                detected_servo_ids.append(int(servo_id))
            if telemetry_status == "Live":
                telemetry_ready_count += 1
            if motion_assessment is not None and motion_assessment.ready:
                motion_ready_count += 1
            if pretension_assessment is not None and pretension_assessment.ready:
                pretension_ready_count += 1
            if telemetry is None:
                message = f"Telemetry is unavailable for servo {servo_id}."
            elif motion_assessment is not None and motion_assessment.ready:
                message = f"Servo {servo_id} is ready for cautious motion."
            elif motion_assessment is not None:
                message = motion_assessment.reason
            else:
                message = f"Servo {servo_id} telemetry is unavailable."
            entries[int(servo_id)] = ServoRuntimeStateEntry(
                servo_id=int(servo_id),
                telemetry=telemetry,
                identity_read_ok=identity_read_ok,
                telemetry_read_ok=telemetry_read_ok,
                detected=detected,
                telemetry_status=telemetry_status,
                motion_assessment=motion_assessment,
                pretension_assessment=pretension_assessment,
                message=message,
            )

        combined_detected_ids = sorted({int(servo_id) for servo_id in detected_servo_ids} | {int(servo_id) for servo_id in discovered_ids})
        unexpected_servo_ids = [
            int(servo_id) for servo_id in sorted(discovered_ids) if int(servo_id) not in expected_ids
        ]
        missing_servo_ids = [servo_id for servo_id in expected_ids if servo_id not in combined_detected_ids]
        total = len(expected_ids)
        details = []
        if include_scan:
            details.extend(
                [
                    f"expected={expected_ids}",
                    f"discovered_ids={sorted(discovered_ids)}",
                ]
            )
        details.extend(
            [
                f"Detected {len(combined_detected_ids)}/{total}",
                f"Telemetry {telemetry_ready_count}/{total}",
                f"Motion ready {motion_ready_count}/{total}",
            ]
        )
        if missing_servo_ids:
            details.append(f"missing={missing_servo_ids}")
        if unexpected_servo_ids:
            details.append(f"unexpected={unexpected_servo_ids}")
        message = " | ".join(details)
        return ServoRuntimeStateSnapshot(
            connected=True,
            expected_servo_ids=expected_ids,
            detected_servo_ids=combined_detected_ids,
            missing_servo_ids=sorted(missing_servo_ids),
            unexpected_servo_ids=unexpected_servo_ids,
            entries=entries,
            telemetry_ready_count=int(telemetry_ready_count),
            motion_ready_count=int(motion_ready_count),
            pretension_ready_count=int(pretension_ready_count),
            all_motion_ready=bool(total > 0 and motion_ready_count == total),
            selected_servo_id=int(selected_servo_id) if selected_servo_id is not None else None,
            message=message,
        )

    def discover_one_servo(
        self,
        *,
        expected_servo_id: int | None = None,
        allow_scan: bool = True,
    ) -> ServoDiscoverySnapshot:
        if not self.is_connected:
            return ServoDiscoverySnapshot(
                status="disconnected",
                connected=False,
                bus_reachable=False,
                expected_servo_id=int(expected_servo_id) if expected_servo_id is not None else None,
                selected_servo_id=None,
                discovered_ids=[],
                scan_range=None,
                telemetry=None,
                motion_assessment=None,
                message="DYNAMIXEL bus is disconnected.",
            )

        if expected_servo_id is not None and self._ping_servo(int(expected_servo_id)):
            debug_snapshot = self.build_bench_debug_snapshot(int(expected_servo_id))
            discovery_status = (
                "expected_id_read_ok"
                if debug_snapshot.identity_read_ok and debug_snapshot.telemetry_read_ok
                else "expected_id_ping_only"
            )
            return ServoDiscoverySnapshot(
                status=discovery_status,
                connected=True,
                bus_reachable=bool(debug_snapshot.bus_reachable),
                expected_servo_id=int(expected_servo_id),
                selected_servo_id=debug_snapshot.selected_servo_id,
                discovered_ids=[int(expected_servo_id)],
                scan_range=None,
                telemetry=debug_snapshot.telemetry,
                motion_assessment=debug_snapshot.motion_assessment,
                message=debug_snapshot.message,
            )

        if not allow_scan:
            return ServoDiscoverySnapshot(
                status="expected_id_missing",
                connected=True,
                bus_reachable=False,
                expected_servo_id=int(expected_servo_id) if expected_servo_id is not None else None,
                selected_servo_id=None,
                discovered_ids=[],
                scan_range=None,
                telemetry=None,
                motion_assessment=None,
                message=(
                    f"Expected servo ID {expected_servo_id} did not respond."
                    if expected_servo_id is not None
                    else "No expected servo ID is configured."
                ),
            )

        scan_range = (
            int(self.dxl_bus.config.discovery_min_id),
            int(self.dxl_bus.config.discovery_max_id),
        )
        discovered_ids = self.scan_ids(min_id=scan_range[0], max_id=scan_range[1])
        if not discovered_ids:
            return ServoDiscoverySnapshot(
                status="not_found",
                connected=True,
                bus_reachable=False,
                expected_servo_id=int(expected_servo_id) if expected_servo_id is not None else None,
                selected_servo_id=None,
                discovered_ids=[],
                scan_range=scan_range,
                telemetry=None,
                motion_assessment=None,
                message=(
                    f"No servos responded in conservative scan range {scan_range[0]}..{scan_range[1]}."
                ),
            )
        if len(discovered_ids) != 1:
            return ServoDiscoverySnapshot(
                status="multiple_found",
                connected=True,
                bus_reachable=True,
                expected_servo_id=int(expected_servo_id) if expected_servo_id is not None else None,
                selected_servo_id=None,
                discovered_ids=list(discovered_ids),
                scan_range=scan_range,
                telemetry=None,
                motion_assessment=None,
                message=(
                    "Conservative discovery found multiple servos "
                    f"{discovered_ids}. One-servo maintenance and motion are blocked."
                ),
            )
        selected_servo_id = int(discovered_ids[0])
        debug_snapshot = self.build_bench_debug_snapshot(selected_servo_id)
        discovery_status = (
            "scan_read_ok"
            if debug_snapshot.identity_read_ok and debug_snapshot.telemetry_read_ok
            else "scan_ping_only"
        )
        return ServoDiscoverySnapshot(
            status=discovery_status,
            connected=True,
            bus_reachable=bool(debug_snapshot.bus_reachable),
            expected_servo_id=int(expected_servo_id) if expected_servo_id is not None else None,
            selected_servo_id=debug_snapshot.selected_servo_id,
            discovered_ids=list(discovered_ids),
            scan_range=scan_range,
            telemetry=debug_snapshot.telemetry,
            motion_assessment=debug_snapshot.motion_assessment,
            message=debug_snapshot.message,
        )

    def get_tightening_direction(self, servo_id: int) -> str | None:
        entry = self.neutral_calibration.entry_by_servo_id(int(servo_id))
        if entry and entry.tightening_rotation:
            return entry.tightening_rotation
        return self.neutral_calibration.context.tightening_rotation_by_servo.get(int(servo_id))

    def default_pretension_parameters(self, servo_id: int) -> PretensionParameters:
        absolute_trigger = self._pretension_threshold_for_servo(int(servo_id))
        configured_absolute = self.safety_guard.pretension_absolute_trigger_current_ma
        if configured_absolute is not None:
            absolute_trigger = int(configured_absolute)
        return PretensionParameters(
            untensioned_reference_tick=int(self.safety_guard.pretension_untensioned_reference_tick),
            step_ticks=int(self.safety_guard.pretension_step_ticks),
            settle_time_s=float(self.safety_guard.pretension_settle_time_s),
            baseline_sample_count=int(self.safety_guard.pretension_baseline_sample_count),
            current_filter_window=int(self.safety_guard.pretension_current_filter_window),
            current_delta_threshold_ma=int(self.safety_guard.pretension_current_delta_threshold_ma),
            absolute_trigger_current_ma=int(absolute_trigger) if absolute_trigger is not None else None,
            hard_current_stop_ma=int(self.safety_guard.pretension_hard_current_stop_ma),
            max_travel_ticks=int(self.safety_guard.pretension_max_travel_ticks),
            timeout_s=float(self.safety_guard.pretension_timeout_s),
        )

    def apply_live_pretension_defaults(self, parameters: PretensionParameters) -> PretensionParameters:
        """Update the active in-memory pretension defaults without rebuilding the runtime."""
        applied = PretensionParameters(
            untensioned_reference_tick=int(parameters.untensioned_reference_tick),
            step_ticks=max(1, int(parameters.step_ticks)),
            settle_time_s=max(0.0, float(parameters.settle_time_s)),
            baseline_sample_count=max(1, int(parameters.baseline_sample_count)),
            current_filter_window=max(1, int(parameters.current_filter_window)),
            current_delta_threshold_ma=max(1, int(parameters.current_delta_threshold_ma)),
            absolute_trigger_current_ma=(
                None
                if parameters.absolute_trigger_current_ma in (None, "")
                else int(parameters.absolute_trigger_current_ma)
            ),
            hard_current_stop_ma=max(1, int(parameters.hard_current_stop_ma)),
            max_travel_ticks=max(1, int(parameters.max_travel_ticks)),
            timeout_s=max(0.01, float(parameters.timeout_s)),
        )
        self.safety_guard.pretension_untensioned_reference_tick = int(applied.untensioned_reference_tick)
        self.safety_guard.pretension_step_ticks = int(applied.step_ticks)
        self.safety_guard.pretension_settle_time_s = float(applied.settle_time_s)
        self.safety_guard.pretension_baseline_sample_count = int(applied.baseline_sample_count)
        self.safety_guard.pretension_current_filter_window = int(applied.current_filter_window)
        self.safety_guard.pretension_current_delta_threshold_ma = int(applied.current_delta_threshold_ma)
        self.safety_guard.pretension_absolute_trigger_current_ma = (
            None if applied.absolute_trigger_current_ma is None else int(applied.absolute_trigger_current_ma)
        )
        self.safety_guard.pretension_hard_current_stop_ma = int(applied.hard_current_stop_ma)
        self.safety_guard.pretension_max_travel_ticks = int(applied.max_travel_ticks)
        self.safety_guard.pretension_timeout_s = float(applied.timeout_s)
        return applied

    def pretension_window_for_servo(
        self,
        *,
        servo_id: int,
        parameters: PretensionParameters | None = None,
        telemetry: ServoTelemetry | None = None,
    ) -> PretensionWindow:
        current = self._telemetry_with_position_limits(
            servo_id=int(servo_id),
            telemetry=telemetry,
        )
        hardware_safe_min = int(current.min_position_limit) + int(self.safety_guard.software_position_margin_ticks)
        hardware_safe_max = int(current.max_position_limit) - int(self.safety_guard.software_position_margin_ticks)
        hardware_safe_min = max(int(RAW_POSITION_MIN_TICK), int(hardware_safe_min))
        hardware_safe_max = min(int(RAW_POSITION_MAX_TICK), int(hardware_safe_max))
        if hardware_safe_min > hardware_safe_max:
            raise ValueError("Software safety margin exceeds the servo hardware position range.")
        config = parameters or self.default_pretension_parameters(int(servo_id))
        reference_tick = min(max(int(config.untensioned_reference_tick), hardware_safe_min), hardware_safe_max)
        effective_min = max(int(hardware_safe_min), int(reference_tick) - int(config.max_travel_ticks))
        effective_max = int(reference_tick)
        if effective_min > effective_max:
            raise ValueError("Pretension travel window is invalid after applying hardware limits.")
        return PretensionWindow(
            servo_id=int(servo_id),
            hardware_safe_min_tick=int(hardware_safe_min),
            hardware_safe_max_tick=int(hardware_safe_max),
            untensioned_reference_tick=int(reference_tick),
            effective_min_target_tick=int(effective_min),
            effective_max_target_tick=int(effective_max),
        )

    def assess_pretension_readiness(
        self,
        servo_id: int,
        *,
        parameters: PretensionParameters | None = None,
        telemetry: ServoTelemetry | None = None,
    ) -> ServoMotionAssessment:
        current = telemetry or self.read_telemetry([int(servo_id)])[int(servo_id)]
        config = parameters or self.default_pretension_parameters(int(servo_id))
        errors: list[str] = []
        if current.present_position is None:
            errors.append("Present Position is unavailable.")
        if current.present_current_ma is None:
            errors.append("Present Current is unavailable.")
        elif int(current.present_current_ma) >= int(config.hard_current_stop_ma):
            errors.append(
                f"Present Current {current.present_current_ma} mA has already reached the configured "
                f"hard stop of {config.hard_current_stop_ma} mA."
            )
        if current.hardware_error_code not in (None, 0):
            errors.append(f"Hardware Error Status is 0x{int(current.hardware_error_code):02X}.")
        if current.hardware_error:
            errors.append(str(current.hardware_error))
        if self.dxl_bus.config.require_fresh_telemetry_for_motion:
            try:
                self.safety_guard.validate_telemetry_freshness(current.last_read_monotonic_s)
            except ValueError as exc:
                errors.append(str(exc))
        if self.dxl_bus.config.require_voltage_for_motion:
            try:
                self.safety_guard.validate_voltage(current.present_voltage_mv, require_present=True)
            except ValueError as exc:
                errors.append(str(exc))
        if self.dxl_bus.config.require_temperature_for_motion:
            try:
                self.safety_guard.validate_temperature(current.present_temperature_c, require_present=True)
            except ValueError as exc:
                errors.append(str(exc))
        if current.operating_mode is None:
            errors.append("Operating Mode is unavailable.")
        elif int(current.operating_mode) not in self.dxl_bus.config.allowed_operating_modes:
            errors.append(
                f"Operating Mode {current.operating_mode} is not allowed. "
                f"Expected one of {self.dxl_bus.config.allowed_operating_modes}."
            )
        if current.torque_enabled is not True:
            errors.append("Torque must be enabled before pretensioning.")
        try:
            window = self.pretension_window_for_servo(
                servo_id=int(servo_id),
                parameters=config,
                telemetry=current,
            )
            safe_min = int(window.effective_min_target_tick)
            safe_max = int(window.effective_max_target_tick)
        except ValueError as exc:
            errors.append(str(exc))
            safe_min = safe_max = None
            window = None
        if not errors and current.present_position is not None and window is not None:
            if not (
                int(window.effective_min_target_tick)
                <= int(current.present_position)
                <= int(window.effective_max_target_tick)
            ):
                errors.append(
                    f"Present Position {current.present_position} is outside the configured pretension window "
                    f"[{window.effective_min_target_tick}, {window.effective_max_target_tick}]. "
                    "Move to the untensioned reference before measuring baseline or starting pretension."
                )
                ready_message = " | ".join(errors)
            else:
                ready_message = "Ready for selected-servo pretension."
        else:
            ready_message = " | ".join(errors) if errors else "Ready for selected-servo pretension."
        return ServoMotionAssessment(
            servo_id=int(servo_id),
            ready=not errors,
            reason=ready_message,
            telemetry=current,
            safe_min_tick=safe_min,
            safe_max_tick=safe_max,
            tightening_direction="smaller raw position counts",
            blocking_reasons=tuple(errors),
            external_power_required=bool(self.dxl_bus.config.require_voltage_for_motion),
            external_power_ready=(
                None
                if current.present_voltage_mv is None
                else bool(int(current.present_voltage_mv) >= int(self.safety_guard.min_input_voltage_mv))
            ),
        )

    def measure_pretension_baseline(
        self,
        *,
        servo_id: int,
        sample_count: int | None = None,
        filter_window: int | None = None,
        parameters: PretensionParameters | None = None,
    ) -> PretensionBaselineMeasurement:
        count = max(1, int(sample_count or self.safety_guard.pretension_baseline_sample_count))
        window = max(1, int(filter_window or self.safety_guard.pretension_current_filter_window))
        samples: list[int] = []
        position_tick: int | None = None
        for _ in range(count):
            telemetry = self.read_telemetry([int(servo_id)])[int(servo_id)]
            assessment = self.assess_pretension_readiness(
                int(servo_id),
                parameters=parameters,
                telemetry=telemetry,
            )
            if not assessment.ready:
                raise RuntimeError(f"Servo {servo_id} is not safe to measure a pretension baseline: {assessment.reason}")
            if telemetry.present_current_ma is None:
                raise RuntimeError(f"Servo {servo_id} current telemetry is unavailable.")
            samples.append(int(telemetry.present_current_ma))
            position_tick = telemetry.present_position
        baseline_current_ma = float(sum(samples) / len(samples))
        filtered_samples = samples[-window:]
        filtered_current_ma = float(sum(filtered_samples) / len(filtered_samples))
        return PretensionBaselineMeasurement(
            servo_id=int(servo_id),
            sample_count=count,
            samples_ma=list(samples),
            baseline_current_ma=baseline_current_ma,
            filtered_current_ma=filtered_current_ma,
            position_tick=position_tick,
            message=(
                f"Measured baseline from {count} sample(s): mean {baseline_current_ma:.1f} mA, "
                f"filtered {filtered_current_ma:.1f} mA."
            ),
        )

    def move_servo_to_pretension_reference(
        self,
        *,
        servo_id: int,
        parameters: PretensionParameters | None = None,
    ) -> ServoJogResult:
        config = parameters or self.default_pretension_parameters(int(servo_id))
        assessment = self.assess_motion(
            int(servo_id),
            require_calibrated_bounds=False,
        )
        if not assessment.ready:
            return ServoJogResult(
                servo_id=int(servo_id),
                command_direction="loosen",
                step_ticks=0,
                delta_ticks=0,
                success=False,
                blocked=True,
                status="blocked",
                message=f"Blocked pretension reference move for servo {servo_id}: {assessment.reason}",
                goal_tick=None,
                telemetry=assessment.telemetry,
                assessment=assessment,
                current_position_tick=assessment.telemetry.present_position,
                safe_min_tick=assessment.safe_min_tick,
                safe_max_tick=assessment.safe_max_tick,
                clamped=False,
            )
        window = self.pretension_window_for_servo(
            servo_id=int(servo_id),
            parameters=config,
            telemetry=assessment.telemetry,
        )
        current_position = assessment.telemetry.present_position
        if current_position is None:
            raise RuntimeError(f"Servo {servo_id} position is unavailable.")
        goal_tick = int(window.untensioned_reference_tick)
        if int(current_position) == int(goal_tick):
            return ServoJogResult(
                servo_id=int(servo_id),
                command_direction="loosen",
                step_ticks=0,
                delta_ticks=0,
                success=True,
                blocked=False,
                status="already_at_reference",
                message=f"Servo {servo_id} is already at pretension reference {goal_tick}.",
                goal_tick=goal_tick,
                telemetry=assessment.telemetry,
                assessment=assessment,
                current_position_tick=int(current_position),
                unclamped_goal_tick=goal_tick,
                safe_min_tick=int(window.effective_min_target_tick),
                safe_max_tick=int(window.effective_max_target_tick),
                clamped=False,
            )
        self._write_goal_positions({int(servo_id): int(goal_tick)})
        if float(config.settle_time_s) > 0.0:
            self._sleep_fn(float(config.settle_time_s))
        updated = self.read_telemetry([int(servo_id)])[int(servo_id)]
        updated_assessment = self.assess_pretension_readiness(
            int(servo_id),
            parameters=config,
            telemetry=updated,
        )
        direction = "loosen" if int(goal_tick) >= int(current_position) else "tighten"
        return ServoJogResult(
            servo_id=int(servo_id),
            command_direction=direction,
            step_ticks=abs(int(goal_tick) - int(current_position)),
            delta_ticks=int(goal_tick) - int(current_position),
            success=True,
            blocked=False,
            status="moved",
            message=(
                f"Moved servo {servo_id} to pretension reference {goal_tick} within "
                f"pretension window [{window.effective_min_target_tick}, {window.effective_max_target_tick}]."
            ),
            goal_tick=int(goal_tick),
            telemetry=updated,
            assessment=updated_assessment,
            current_position_tick=int(current_position),
            unclamped_goal_tick=int(goal_tick),
            safe_min_tick=int(window.effective_min_target_tick),
            safe_max_tick=int(window.effective_max_target_tick),
            clamped=False,
        )

    def move_servo_to_raw_target(
        self,
        *,
        servo_id: int,
        target_tick: int,
        reason: str = "selected_servo_move",
    ) -> ServoJogResult:
        assessment = self.assess_motion(
            int(servo_id),
            require_calibrated_bounds=False,
        )
        if not assessment.ready:
            return ServoJogResult(
                servo_id=int(servo_id),
                command_direction="loosen" if int(target_tick) >= int(assessment.telemetry.present_position or 0) else "tighten",
                step_ticks=abs(int(target_tick) - int(assessment.telemetry.present_position or 0)),
                delta_ticks=int(target_tick) - int(assessment.telemetry.present_position or 0),
                success=False,
                blocked=True,
                status="blocked",
                message=f"Blocked {reason} for servo {servo_id}: {assessment.reason}",
                goal_tick=None,
                telemetry=assessment.telemetry,
                assessment=assessment,
                current_position_tick=assessment.telemetry.present_position,
                safe_min_tick=assessment.safe_min_tick,
                safe_max_tick=assessment.safe_max_tick,
                clamped=False,
            )
        if assessment.telemetry.present_position is None:
            raise RuntimeError(f"Servo {servo_id} position is unavailable.")
        current_position = int(assessment.telemetry.present_position)
        safe_min, safe_max = self._application_safe_bounds_for_servo(
            servo_id=int(servo_id),
            telemetry=assessment.telemetry,
            require_calibrated_bounds=False,
        )
        unclamped_goal = int(target_tick)
        goal_tick = min(max(int(target_tick), int(safe_min)), int(safe_max))
        clamped = goal_tick != unclamped_goal
        if goal_tick == current_position:
            return ServoJogResult(
                servo_id=int(servo_id),
                command_direction="loosen" if goal_tick >= current_position else "tighten",
                step_ticks=abs(goal_tick - current_position),
                delta_ticks=goal_tick - current_position,
                success=False,
                blocked=True,
                status="blocked",
                message=(
                    f"Blocked {reason} for servo {servo_id}: target {goal_tick} would not move the servo."
                ),
                goal_tick=goal_tick,
                telemetry=assessment.telemetry,
                assessment=assessment,
                current_position_tick=current_position,
                unclamped_goal_tick=unclamped_goal,
                safe_min_tick=safe_min,
                safe_max_tick=safe_max,
                clamped=clamped,
            )
        self._write_goal_positions({int(servo_id): int(goal_tick)})
        updated = self.read_live_telemetry([int(servo_id)])[int(servo_id)]
        self._validate_post_motion(updated)
        updated_assessment = self.assess_motion(
            int(servo_id),
            require_calibrated_bounds=False,
            telemetry=updated,
        )
        return ServoJogResult(
            servo_id=int(servo_id),
            command_direction="loosen" if goal_tick >= current_position else "tighten",
            step_ticks=abs(goal_tick - current_position),
            delta_ticks=goal_tick - current_position,
            success=True,
            blocked=False,
            status="moved",
            message=(
                f"Moved servo {servo_id} to {goal_tick} ticks for {reason} "
                f"within pretension-safe range [{safe_min}, {safe_max}]."
            ),
            goal_tick=goal_tick,
            telemetry=updated,
            assessment=updated_assessment,
            current_position_tick=current_position,
            unclamped_goal_tick=unclamped_goal,
            safe_min_tick=safe_min,
            safe_max_tick=safe_max,
            clamped=clamped,
        )

    def assess_motion(
        self,
        servo_id: int,
        *,
        require_calibrated_bounds: bool,
        telemetry: ServoTelemetry | None = None,
    ) -> ServoMotionAssessment:
        if telemetry is not None:
            current = telemetry
        elif require_calibrated_bounds:
            current = self.read_telemetry([servo_id])[int(servo_id)]
        else:
            current = self.read_live_telemetry([servo_id])[int(servo_id)]
        errors: list[str] = []
        safe_min: int | None = None
        safe_max: int | None = None
        external_power_ready: bool | None = None

        if current.present_position is None:
            errors.append("Present Position is unavailable.")
        if current.hardware_error_code not in (None, 0):
            errors.append(f"Hardware Error Status is 0x{int(current.hardware_error_code):02X}.")
        if current.hardware_error:
            errors.append(str(current.hardware_error))
        if self.dxl_bus.config.require_fresh_telemetry_for_motion:
            try:
                self.safety_guard.validate_telemetry_freshness(current.last_read_monotonic_s)
            except ValueError as exc:
                errors.append(str(exc))
        if self.dxl_bus.config.require_current_for_motion:
            try:
                self.safety_guard.validate_currents([current.present_current_ma], require_present=True)
            except ValueError as exc:
                errors.append(str(exc))
        if self.dxl_bus.config.require_voltage_for_motion:
            try:
                self.safety_guard.validate_voltage(current.present_voltage_mv, require_present=True)
                external_power_ready = True
            except ValueError as exc:
                external_power_ready = False
                errors.append(str(exc))
        if self.dxl_bus.config.require_temperature_for_motion:
            try:
                self.safety_guard.validate_temperature(current.present_temperature_c, require_present=True)
            except ValueError as exc:
                errors.append(str(exc))
        if current.operating_mode is None:
            errors.append("Operating Mode is unavailable.")
        elif int(current.operating_mode) not in self.dxl_bus.config.allowed_operating_modes:
            errors.append(
                f"Operating Mode {current.operating_mode} is not allowed. "
                f"Expected one of {self.dxl_bus.config.allowed_operating_modes}."
            )
        if current.torque_enabled is False and not self.dxl_bus.config.auto_torque_enable_on_write:
            errors.append("Torque Enable is 0 and auto torque enable is disabled.")
        try:
            safe_min, safe_max = self._active_motion_bounds_for_servo(
                servo_id=int(servo_id),
                telemetry=current,
                require_calibrated_bounds=require_calibrated_bounds,
            )
        except ValueError as exc:
            errors.append(str(exc))
        if (
            current.present_position is not None
            and safe_min is not None
            and safe_max is not None
            and not (int(safe_min) <= int(current.present_position) <= int(safe_max))
        ):
            errors.append(
                f"Present Position {current.present_position} is outside the active motion range "
                f"[{safe_min}, {safe_max}]. Recover to a safe position before commanding motion."
            )
        return ServoMotionAssessment(
            servo_id=int(servo_id),
            ready=not errors,
            reason=" | ".join(errors) if errors else "Ready for cautious motion.",
            telemetry=current,
            safe_min_tick=safe_min,
            safe_max_tick=safe_max,
            tightening_direction="smaller raw position counts",
            blocking_reasons=tuple(errors),
            external_power_required=bool(self.dxl_bus.config.require_voltage_for_motion),
            external_power_ready=external_power_ready,
        )

    def plan_jog_action(self, *, servo_id: int, action: str) -> ServoMotionPlan:
        action_name = str(action).strip().lower()
        if action_name not in {
            "tighten_fine",
            "tighten_coarse",
            "loosen_fine",
            "loosen_coarse",
        }:
            raise ValueError(
                "action must be one of tighten_fine, tighten_coarse, loosen_fine, or loosen_coarse."
            )
        direction = "tighten" if action_name.startswith("tighten") else "loosen"
        step_ticks = (
            int(self.safety_guard.fine_jog_step_ticks)
            if action_name.endswith("fine")
            else int(self.safety_guard.coarse_jog_step_ticks)
        )
        return self._build_relative_motion_plan(
            servo_id=int(servo_id),
            action=action_name,
            delta_ticks=self._canonical_delta_for_direction(direction, step_ticks),
            step_ticks=step_ticks,
        )

    def jog_servo(self, servo_id: int, delta_ticks: int) -> ServoCommandResult:
        # All live single-servo motion must pass through ServoService so
        # calibration bounds, telemetry refresh, and current checks stay consistent.
        self.safety_guard.validate_jog_delta(delta_ticks)
        assessment = self.assess_motion(
            int(servo_id),
            require_calibrated_bounds=self.require_calibrated_bounds_for_individual_motion(),
        )
        if not assessment.ready:
            raise RuntimeError(f"Servo {servo_id} is not safe to jog: {assessment.reason}")
        if assessment.telemetry.present_position is None:
            raise RuntimeError(f"Servo {servo_id} position is unavailable.")
        goal = int(assessment.telemetry.present_position + delta_ticks)
        self._validate_goal_against_assessment(assessment, goal)
        self._write_goal_positions({int(servo_id): goal})
        updated = self.read_live_telemetry([int(servo_id)])
        self._validate_post_motion(updated[int(servo_id)])
        return ServoCommandResult(
            positions_by_id={int(servo_id): goal},
            telemetry_by_id=updated,
            message=(
                f"Jogged servo {servo_id} to {goal} ticks "
                f"within [{assessment.safe_min_tick}, {assessment.safe_max_tick}]."
            ),
        )

    def jog_servo_directional(
        self,
        *,
        servo_id: int,
        command_direction: str,
        step_ticks: int,
    ) -> ServoJogResult:
        direction = str(command_direction).strip().lower()
        if direction not in {"tighten", "loosen"}:
            raise ValueError("command_direction must be 'tighten' or 'loosen'.")
        if int(step_ticks) <= 0:
            raise ValueError("step_ticks must be positive.")
        action = f"{direction}_{'fine' if int(step_ticks) == int(self.safety_guard.fine_jog_step_ticks) else 'coarse'}"
        plan = self._build_relative_motion_plan(
            servo_id=int(servo_id),
            action=action,
            delta_ticks=self._canonical_delta_for_direction(direction, int(step_ticks)),
            step_ticks=int(step_ticks),
        )
        return self._execute_motion_plan(plan, command_direction=direction)

    def jog_servo_action(self, *, servo_id: int, action: str) -> ServoJogResult:
        plan = self.plan_jog_action(servo_id=int(servo_id), action=action)
        direction = "tighten" if str(action).strip().lower().startswith("tighten") else "loosen"
        return self._execute_motion_plan(plan, command_direction=direction)

    def command_displacement(
        self,
        tendon_displacements_cm: list[float],
        neutral_ticks: list[int],
        servo_ids: list[int],
    ) -> ServoCommandResult:
        """Compute and send safe goal position ticks.

        This is the canonical tendon-length command path used by controllers
        and experiments. Do not bypass it with direct bus writes.
        """
        if len(servo_ids) != len(neutral_ticks):
            raise ValueError("Servo ID list and neutral setpoint list length mismatch")
        goals = self.mapper.to_goal_positions(tendon_displacements_cm, neutral_ticks)
        payload = {sid: goal for sid, goal in zip(servo_ids, goals)}
        assessments = {
            int(servo_id): self.assess_motion(int(servo_id), require_calibrated_bounds=True)
            for servo_id in servo_ids
        }
        for servo_id, assessment in assessments.items():
            if not assessment.ready:
                raise RuntimeError(
                    f"Servo {servo_id} is not safe for displacement control: {assessment.reason}"
                )
            self._validate_goal_against_assessment(assessment, int(payload[servo_id]))
        self._write_goal_positions(payload)
        telemetry = self.read_telemetry(servo_ids)
        for servo_id in servo_ids:
            self._validate_post_motion(telemetry[int(servo_id)])
        return ServoCommandResult(
            positions_by_id=payload,
            telemetry_by_id=telemetry,
            message=f"Commanded {len(payload)} servo(s) from tendon displacement input.",
        )

    def save_startup_calibration(
        self,
        *,
        servo_id: int,
        neutral_setpoint: int | None = None,
        min_offset_ticks: int | None = None,
        max_offset_ticks: int | None = None,
        pretension_current_threshold_ma: int | None = None,
    ):
        full_telemetry = self.read_telemetry([int(servo_id)])[int(servo_id)]
        assessment = self.assess_motion(
            int(servo_id),
            require_calibrated_bounds=False,
            telemetry=full_telemetry,
        )
        if not assessment.ready:
            raise RuntimeError(f"Servo {servo_id} is not safe to calibrate: {assessment.reason}")
        if assessment.telemetry.present_position is None:
            raise RuntimeError(f"Servo {servo_id} position is unavailable.")
        neutral = int(
            assessment.telemetry.present_position if neutral_setpoint is None else neutral_setpoint
        )
        offset_min = (
            int(min_offset_ticks)
            if min_offset_ticks is not None
            else int(self.safety_guard.min_offset_ticks)
        )
        offset_max = (
            int(max_offset_ticks)
            if max_offset_ticks is not None
            else int(self.safety_guard.max_offset_ticks)
        )
        safe_min = neutral + offset_min
        safe_max = neutral + offset_max
        hardware_min, hardware_max = self._application_safe_bounds_for_servo(
            servo_id=int(servo_id),
            telemetry=assessment.telemetry,
            require_calibrated_bounds=False,
        )
        safe_min = max(int(safe_min), int(hardware_min))
        safe_max = min(int(safe_max), int(hardware_max))
        threshold = (
            int(pretension_current_threshold_ma)
            if pretension_current_threshold_ma is not None
            else self._pretension_threshold_for_servo(int(servo_id))
        )
        if threshold >= int(self.safety_guard.max_current_ma):
            raise ValueError(
                f"Pretension threshold {threshold} mA must be below the absolute safety current "
                f"limit of {self.safety_guard.max_current_ma} mA."
            )
        return self.neutral_calibration.save_servo_calibration(
            servo_id=int(servo_id),
            neutral_setpoint=neutral,
            safe_min_tick=int(safe_min),
            safe_max_tick=int(safe_max),
            pretension_current_threshold_ma=int(threshold),
            tightening_rotation=self.get_tightening_direction(int(servo_id)),
            hardware_min_tick=assessment.telemetry.min_position_limit,
            hardware_max_tick=assessment.telemetry.max_position_limit,
            hardware_current_limit_ma=assessment.telemetry.current_limit_ma,
            last_measured_current_ma=assessment.telemetry.present_current_ma,
            status="startup_calibrated",
            valid=True,
        )

    def run_pretension_routine(
        self,
        *,
        servo_id: int,
        parameters: PretensionParameters | None = None,
        threshold_ma: int | None = None,
        stop_requested: Callable[[], bool] | None = None,
        progress_callback: Callable[[PretensionRoutineResult], None] | None = None,
    ) -> PretensionRoutineResult:
        config = parameters or self.default_pretension_parameters(int(servo_id))
        if threshold_ma is not None:
            config = PretensionParameters(
                untensioned_reference_tick=int(config.untensioned_reference_tick),
                step_ticks=int(config.step_ticks),
                settle_time_s=float(config.settle_time_s),
                baseline_sample_count=int(config.baseline_sample_count),
                current_filter_window=int(config.current_filter_window),
                current_delta_threshold_ma=int(config.current_delta_threshold_ma),
                absolute_trigger_current_ma=int(threshold_ma),
                hard_current_stop_ma=int(config.hard_current_stop_ma),
                max_travel_ticks=int(config.max_travel_ticks),
                timeout_s=float(config.timeout_s),
            )
        if int(config.step_ticks) <= 0:
            raise ValueError("Pretension step_ticks must be positive.")
        if int(config.baseline_sample_count) <= 0:
            raise ValueError("Pretension baseline_sample_count must be positive.")
        if int(config.current_filter_window) <= 0:
            raise ValueError("Pretension current_filter_window must be positive.")
        if int(config.current_delta_threshold_ma) <= 0:
            raise ValueError("Pretension current_delta_threshold_ma must be positive.")
        if int(config.max_travel_ticks) <= 0:
            raise ValueError("Pretension max_travel_ticks must be positive.")
        if float(config.timeout_s) <= 0.0:
            raise ValueError("Pretension timeout_s must be positive.")
        if float(config.settle_time_s) < 0.0:
            raise ValueError("Pretension settle_time_s cannot be negative.")
        if int(config.hard_current_stop_ma) <= 0:
            raise ValueError("Pretension hard_current_stop_ma must be positive.")
        if (
            config.absolute_trigger_current_ma is not None
            and int(config.absolute_trigger_current_ma) >= int(config.hard_current_stop_ma)
        ):
            raise ValueError(
                "Pretension absolute trigger current must stay below the hard current stop."
            )
        with self.exclusive_bus_operation(
            owner="pretension run",
            servo_id=int(servo_id),
            reason="selected-servo pretension",
        ):
            return self._run_pretension_routine_with_owned_bus(
                servo_id=int(servo_id),
                config=config,
                stop_requested=stop_requested,
                progress_callback=progress_callback,
            )

    def _run_pretension_routine_with_owned_bus(
        self,
        *,
        servo_id: int,
        config: PretensionParameters,
        stop_requested: Callable[[], bool] | None,
        progress_callback: Callable[[PretensionRoutineResult], None] | None,
    ) -> PretensionRoutineResult:
        started_at = float(self._time_fn())
        assessment = self.assess_pretension_readiness(int(servo_id), parameters=config)
        if not assessment.ready:
            if (
                assessment.telemetry.present_current_ma is not None
                and int(assessment.telemetry.present_current_ma) >= int(config.hard_current_stop_ma)
            ):
                return PretensionRoutineResult(
                    servo_id=int(servo_id),
                    status="overcurrent",
                    success=False,
                    message=(
                        f"Pretension stopped for servo {servo_id}: measured current "
                        f"{assessment.telemetry.present_current_ma} mA reached the hard stop of "
                        f"{config.hard_current_stop_ma} mA before the routine could start."
                    ),
                    threshold_ma=(
                        int(config.absolute_trigger_current_ma)
                        if config.absolute_trigger_current_ma is not None
                        else int(config.current_delta_threshold_ma)
                    ),
                    final_position_tick=assessment.telemetry.present_position,
                    final_current_ma=assessment.telemetry.present_current_ma,
                    steps_taken=0,
                    tightening_direction="decreasing_raw_position",
                    start_position_tick=assessment.telemetry.present_position,
                    untensioned_reference_tick=int(config.untensioned_reference_tick),
                    current_position_tick=assessment.telemetry.present_position,
                    last_commanded_target_tick=None,
                    baseline_current_ma=None,
                    filtered_current_ma=float(assessment.telemetry.present_current_ma),
                    current_delta_ma=None,
                    absolute_trigger_current_ma=config.absolute_trigger_current_ma,
                    hard_current_stop_ma=int(config.hard_current_stop_ma),
                    elapsed_s=max(0.0, float(self._time_fn()) - float(started_at)),
                    stop_reason="hard_current_stop",
                    parameters=asdict(config),
                )
            raise RuntimeError(f"Servo {servo_id} is not safe to pretension: {assessment.reason}")
        if assessment.telemetry.present_position is None:
            raise RuntimeError(f"Servo {servo_id} pretension requires position telemetry.")
        window = self.pretension_window_for_servo(
            servo_id=int(servo_id),
            parameters=config,
            telemetry=assessment.telemetry,
        )
        absolute_trigger = (
            int(config.absolute_trigger_current_ma)
            if config.absolute_trigger_current_ma is not None
            else None
        )
        tightening_direction = "decreasing_raw_position"
        step_delta = -abs(int(config.step_ticks))
        start_position_tick = int(assessment.telemetry.present_position)
        untensioned_reference = int(window.untensioned_reference_tick)
        travel_min_tick = int(window.effective_min_target_tick)
        safe_max = int(window.effective_max_target_tick)
        deadline = started_at + float(config.timeout_s)
        steps_taken = 0
        current_position = start_position_tick
        last_commanded_target_tick: int | None = None
        filter_samples: list[int] = []
        baseline: PretensionBaselineMeasurement | None = None
        baseline_current_ma = 0.0
        baseline_delta_trigger = 0.0
        threshold = absolute_trigger if absolute_trigger is not None else int(config.current_delta_threshold_ma)

        def _build_result(
            *,
            status: str,
            success: bool,
            message: str,
            final_position_tick: int | None,
            final_current_ma: int | None,
            filtered_current_ma: float | None,
            current_delta_ma: float | None,
            stop_reason: str,
        ) -> PretensionRoutineResult:
            return PretensionRoutineResult(
                servo_id=int(servo_id),
                status=str(status),
                success=bool(success),
                message=str(message),
                threshold_ma=int(threshold),
                final_position_tick=final_position_tick,
                final_current_ma=final_current_ma,
                steps_taken=int(steps_taken),
                tightening_direction=tightening_direction,
                start_position_tick=int(start_position_tick),
                untensioned_reference_tick=int(untensioned_reference),
                current_position_tick=final_position_tick,
                last_commanded_target_tick=last_commanded_target_tick,
                baseline_current_ma=float(baseline_current_ma),
                filtered_current_ma=filtered_current_ma,
                current_delta_ma=current_delta_ma,
                absolute_trigger_current_ma=absolute_trigger,
                hard_current_stop_ma=int(config.hard_current_stop_ma),
                elapsed_s=max(0.0, float(self._time_fn()) - float(started_at)),
                stop_reason=str(stop_reason),
                parameters=asdict(config),
            )

        def _persist_and_emit(result: PretensionRoutineResult) -> PretensionRoutineResult:
            run_record = {
                "recorded_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "start_position_tick": result.start_position_tick,
                "untensioned_reference_tick": result.untensioned_reference_tick,
                "baseline_current_ma": result.baseline_current_ma,
                "filtered_current_ma": result.filtered_current_ma,
                "baseline_sample_count": int(config.baseline_sample_count),
                "baseline_samples_ma": (list(baseline.samples_ma) if baseline is not None else []),
                "trigger_current_ma": result.threshold_ma,
                "absolute_trigger_current_ma": result.absolute_trigger_current_ma,
                "current_delta_threshold_ma": int(config.current_delta_threshold_ma),
                "current_filter_window": int(config.current_filter_window),
                "hard_current_stop_ma": int(config.hard_current_stop_ma),
                "final_position_tick": result.final_position_tick,
                "final_current_ma": result.final_current_ma,
                "steps_taken": result.steps_taken,
                "stop_reason": result.stop_reason,
                "status": result.status,
                "elapsed_s": result.elapsed_s,
                "last_commanded_target_tick": result.last_commanded_target_tick,
                "effective_min_target_tick": int(travel_min_tick),
                "effective_max_target_tick": int(safe_max),
                "travel_used_ticks": (
                    None
                    if result.final_position_tick is None
                    else max(0, int(untensioned_reference) - int(result.final_position_tick))
                ),
                "parameters": dict(result.parameters or {}),
            }
            self.neutral_calibration.save_pretension_result(
                servo_id=int(servo_id),
                final_position_tick=result.final_position_tick,
                final_current_ma=result.final_current_ma,
                threshold_ma=result.threshold_ma,
                result_status=result.status,
                run_record=run_record,
            )
            if progress_callback is not None:
                progress_callback(result)
            return result

        def _emit_progress(
            *,
            status: str,
            message: str,
            telemetry: ServoTelemetry,
            filtered_current_ma: float | None,
            current_delta_ma: float | None,
        ) -> None:
            if progress_callback is None:
                return
            progress_callback(
                PretensionRoutineResult(
                    servo_id=int(servo_id),
                    status=str(status),
                    success=False,
                    message=str(message),
                    threshold_ma=int(threshold),
                    final_position_tick=None,
                    final_current_ma=telemetry.present_current_ma,
                    steps_taken=int(steps_taken),
                    tightening_direction=tightening_direction,
                    start_position_tick=int(start_position_tick),
                    untensioned_reference_tick=int(untensioned_reference),
                    current_position_tick=telemetry.present_position,
                    last_commanded_target_tick=last_commanded_target_tick,
                    baseline_current_ma=float(baseline_current_ma),
                    filtered_current_ma=filtered_current_ma,
                    current_delta_ma=current_delta_ma,
                    absolute_trigger_current_ma=absolute_trigger,
                    hard_current_stop_ma=int(config.hard_current_stop_ma),
                    elapsed_s=max(0.0, float(self._time_fn()) - float(started_at)),
                    stop_reason=None,
                    parameters=asdict(config),
                )
            )

        baseline = self.measure_pretension_baseline(
            servo_id=int(servo_id),
            sample_count=int(config.baseline_sample_count),
            filter_window=int(config.current_filter_window),
            parameters=config,
        )
        baseline_current_ma = float(baseline.filtered_current_ma)
        baseline_delta_trigger = float(baseline_current_ma + int(config.current_delta_threshold_ma))
        threshold = (
            int(min(baseline_delta_trigger, float(absolute_trigger)))
            if absolute_trigger is not None
            else int(round(baseline_delta_trigger))
        )
        filter_samples = list(baseline.samples_ma[-max(1, int(config.current_filter_window)) :])

        _emit_progress(
            status="baseline_ready",
            message=baseline.message,
            telemetry=self.read_telemetry([int(servo_id)])[int(servo_id)],
            filtered_current_ma=baseline_current_ma,
            current_delta_ma=0.0,
        )

        while True:
            if stop_requested is not None and stop_requested():
                final = self.read_telemetry([int(servo_id)])[int(servo_id)]
                filtered_current = float(final.present_current_ma) if final.present_current_ma is not None else None
                current_delta = (
                    float(filtered_current - baseline_current_ma)
                    if filtered_current is not None
                    else None
                )
                return _persist_and_emit(
                    _build_result(
                        status="canceled",
                        success=False,
                        message=f"Pretension canceled for servo {servo_id}.",
                        final_position_tick=final.present_position,
                        final_current_ma=final.present_current_ma,
                        filtered_current_ma=filtered_current,
                        current_delta_ma=current_delta,
                        stop_reason="operator_canceled",
                    )
                )
            if self._time_fn() > deadline:
                final = self.read_telemetry([int(servo_id)])[int(servo_id)]
                filtered_current = float(final.present_current_ma) if final.present_current_ma is not None else None
                current_delta = (
                    float(filtered_current - baseline_current_ma)
                    if filtered_current is not None
                    else None
                )
                return _persist_and_emit(
                    _build_result(
                        status="timeout",
                        success=False,
                        message=f"Pretension timed out for servo {servo_id}.",
                        final_position_tick=final.present_position,
                        final_current_ma=final.present_current_ma,
                        filtered_current_ma=filtered_current,
                        current_delta_ma=current_delta,
                        stop_reason="timeout",
                    )
                )

            raw_telemetry = self.read_telemetry([int(servo_id)])[int(servo_id)]
            if (
                raw_telemetry.present_current_ma is not None
                and int(raw_telemetry.present_current_ma) >= int(config.hard_current_stop_ma)
            ):
                return _persist_and_emit(
                    _build_result(
                        status="overcurrent",
                        success=False,
                        message=(
                            f"Pretension stopped for servo {servo_id}: measured current "
                            f"{raw_telemetry.present_current_ma} mA reached the hard stop of "
                            f"{config.hard_current_stop_ma} mA."
                        ),
                        final_position_tick=raw_telemetry.present_position,
                        final_current_ma=int(raw_telemetry.present_current_ma),
                        filtered_current_ma=float(raw_telemetry.present_current_ma),
                        current_delta_ma=float(raw_telemetry.present_current_ma - baseline_current_ma),
                        stop_reason="hard_current_stop",
                    )
                )
            current_assessment = self.assess_pretension_readiness(
                int(servo_id),
                parameters=config,
                telemetry=raw_telemetry,
            )
            if not current_assessment.ready:
                return _persist_and_emit(
                    _build_result(
                        status="invalid_telemetry",
                        success=False,
                        message=f"Pretension stopped for servo {servo_id}: {current_assessment.reason}",
                        final_position_tick=current_assessment.telemetry.present_position,
                        final_current_ma=current_assessment.telemetry.present_current_ma,
                        filtered_current_ma=None,
                        current_delta_ma=None,
                        stop_reason=current_assessment.reason,
                    )
                )
            current_ma = current_assessment.telemetry.present_current_ma
            position = current_assessment.telemetry.present_position
            if current_ma is None or position is None:
                return _persist_and_emit(
                    _build_result(
                        status="invalid_telemetry",
                        success=False,
                        message=f"Pretension stopped for servo {servo_id}: current or position telemetry is unavailable.",
                        final_position_tick=position,
                        final_current_ma=current_ma,
                        filtered_current_ma=None,
                        current_delta_ma=None,
                        stop_reason="missing_current_or_position",
                    )
                )
            filter_samples.append(int(current_ma))
            filter_samples = filter_samples[-max(1, int(config.current_filter_window)) :]
            filtered_current = float(sum(filter_samples) / len(filter_samples))
            current_delta = float(filtered_current - baseline_current_ma)
            current_position = int(position)

            if filtered_current >= float(config.hard_current_stop_ma):
                return _persist_and_emit(
                    _build_result(
                        status="overcurrent",
                        success=False,
                        message=(
                            f"Pretension stopped for servo {servo_id}: filtered current "
                            f"{filtered_current:.1f} mA reached the hard stop of {config.hard_current_stop_ma} mA."
                        ),
                        final_position_tick=current_position,
                        final_current_ma=int(current_ma),
                        filtered_current_ma=filtered_current,
                        current_delta_ma=current_delta,
                        stop_reason="hard_current_stop",
                    )
                )

            threshold_reason = None
            if filtered_current >= baseline_delta_trigger:
                threshold_reason = "baseline_delta_trigger"
            if absolute_trigger is not None and filtered_current >= float(absolute_trigger):
                threshold_reason = "absolute_trigger" if threshold_reason is None else "combined_trigger"
            if threshold_reason is not None:
                return _persist_and_emit(
                    _build_result(
                        status="threshold_reached",
                        success=True,
                        message=(
                            f"Servo {servo_id} reached the pretension trigger at "
                            f"{current_position} ticks / filtered {filtered_current:.1f} mA."
                        ),
                        final_position_tick=current_position,
                        final_current_ma=int(current_ma),
                        filtered_current_ma=filtered_current,
                        current_delta_ma=current_delta,
                        stop_reason=threshold_reason,
                    )
                )

            next_goal = int(current_position + step_delta)
            if next_goal < int(travel_min_tick) or next_goal > int(safe_max):
                return _persist_and_emit(
                    _build_result(
                        status="travel_limit",
                        success=False,
                        message=(
                            f"Pretension stopped for servo {servo_id}: next tightening step would exceed "
                            f"the allowed pretension travel [{travel_min_tick}, {safe_max}]."
                        ),
                        final_position_tick=current_position,
                        final_current_ma=int(current_ma),
                        filtered_current_ma=filtered_current,
                        current_delta_ma=current_delta,
                        stop_reason="travel_limit",
                    )
                )

            last_commanded_target_tick = int(next_goal)
            _emit_progress(
                status="running",
                message=(
                    f"Pretension running on servo {servo_id}: step {steps_taken + 1}, "
                    f"target {last_commanded_target_tick}, filtered current {filtered_current:.1f} mA."
                ),
                telemetry=current_assessment.telemetry,
                filtered_current_ma=filtered_current,
                current_delta_ma=current_delta,
            )
            self._write_goal_positions({int(servo_id): int(next_goal)})
            steps_taken += 1
            self._sleep_fn(float(config.settle_time_s))

    def validate_pretension(
        self,
        servo_ids: list[int],
        tolerance_ma: int,
    ) -> PretensionValidationResult:
        telemetry = self.read_telemetry(servo_ids)
        currents = [telemetry[sid].present_current_ma for sid in servo_ids]
        self.safety_guard.validate_currents(currents)
        return self.pretension_validation.validate_current_balance(currents, tolerance_ma)

    def accept_pretension_result(self, servo_id: int):
        return self.neutral_calibration.mark_pretension_accepted(int(servo_id))

    def _pretension_threshold_for_servo(self, servo_id: int) -> int:
        thresholds = self.neutral_calibration.thresholds_by_servo_id([int(servo_id)])
        if thresholds:
            return int(thresholds[int(servo_id)])
        return int(self.safety_guard.default_pretension_current_threshold_ma)

    @staticmethod
    def _canonical_delta_for_direction(command_direction: str, step_ticks: int) -> int:
        if int(step_ticks) <= 0:
            raise ValueError("step_ticks must be positive.")
        direction = str(command_direction).strip().lower()
        if direction == "tighten":
            return -abs(int(step_ticks))
        if direction == "loosen":
            return abs(int(step_ticks))
        raise ValueError("command_direction must be 'tighten' or 'loosen'.")

    def _active_motion_bounds_for_servo(
        self,
        *,
        servo_id: int,
        telemetry: ServoTelemetry,
        require_calibrated_bounds: bool,
    ) -> tuple[int, int]:
        if not require_calibrated_bounds:
            return self.raw_position_range()
        if self.is_single_servo_bench_mode():
            return self.raw_position_range()
        return self._application_safe_bounds_for_servo(
            servo_id=int(servo_id),
            telemetry=telemetry,
            require_calibrated_bounds=require_calibrated_bounds,
        )

    def _application_safe_bounds_for_servo(
        self,
        *,
        servo_id: int,
        telemetry: ServoTelemetry,
        require_calibrated_bounds: bool,
    ) -> tuple[int, int]:
        telemetry_with_limits = self._telemetry_with_position_limits(
            servo_id=int(servo_id),
            telemetry=telemetry,
        )
        safe_min = int(telemetry_with_limits.min_position_limit) + int(self.safety_guard.software_position_margin_ticks)
        safe_max = int(telemetry_with_limits.max_position_limit) - int(self.safety_guard.software_position_margin_ticks)
        safe_min = max(int(RAW_POSITION_MIN_TICK), int(safe_min))
        safe_max = min(int(RAW_POSITION_MAX_TICK), int(safe_max))
        if safe_min > safe_max:
            raise ValueError("Software safety margin exceeds the servo hardware position range.")

        summary = self.neutral_calibration.get_calibration_summary()
        if summary.exists and not summary.compatible and require_calibrated_bounds:
            raise ValueError(
                "Saved servo calibration does not match the current robot configuration. "
                "Recapture one-servo neutral or compatible calibrated bounds before commanding motion."
            )
        entry = summary.servo_entries.get(int(servo_id)) if summary.exists and summary.compatible else None
        if entry and entry.safe_min_tick is not None and entry.safe_max_tick is not None:
            safe_min = max(safe_min, int(entry.safe_min_tick))
            safe_max = min(safe_max, int(entry.safe_max_tick))
        elif require_calibrated_bounds:
            raise ValueError(
                f"Servo {servo_id} does not have saved safe bounds. "
                "Capture neutral or persist calibrated bounds before commanding this action."
            )
        if safe_min > safe_max:
            raise ValueError(
                f"Servo {servo_id} safe bounds are invalid after applying hardware and software limits."
            )
        return int(safe_min), int(safe_max)

    def _telemetry_with_position_limits(
        self,
        *,
        servo_id: int,
        telemetry: ServoTelemetry | None,
    ) -> ServoTelemetry:
        current = telemetry or self.read_telemetry([int(servo_id)])[int(servo_id)]
        if current.min_position_limit is not None and current.max_position_limit is not None:
            return current
        refreshed = self.read_telemetry([int(servo_id)])[int(servo_id)]
        if refreshed.min_position_limit is None or refreshed.max_position_limit is None:
            raise ValueError("Servo position limits are unavailable.")
        return refreshed

    def _safe_bounds_from_neutral(
        self,
        *,
        servo_id: int,
        neutral_tick: int,
        telemetry: ServoTelemetry,
        min_offset_ticks: int,
        max_offset_ticks: int,
    ) -> tuple[int, int]:
        del servo_id
        del telemetry
        raw_min, raw_max = self.raw_position_range()
        safe_min = max(int(raw_min), int(neutral_tick) + int(min_offset_ticks))
        safe_max = min(int(raw_max), int(neutral_tick) + int(max_offset_ticks))
        safe_min = min(int(safe_min), int(neutral_tick))
        safe_max = max(int(safe_max), int(neutral_tick))
        return int(safe_min), int(safe_max)

    def _build_relative_motion_plan(
        self,
        *,
        servo_id: int,
        action: str,
        delta_ticks: int,
        step_ticks: int,
    ) -> ServoMotionPlan:
        assessment: ServoMotionAssessment | None
        try:
            self.safety_guard.validate_jog_delta(int(delta_ticks))
        except ValueError as exc:
            return ServoMotionPlan(
                servo_id=int(servo_id),
                action=str(action),
                current_position_tick=None,
                step_ticks=int(step_ticks),
                delta_ticks=int(delta_ticks),
                unclamped_target_tick=None,
                clamped_target_tick=None,
                safe_min_tick=None,
                safe_max_tick=None,
                clamped=False,
                allowed=False,
                block_reason=str(exc),
                assessment=None,
            )
        try:
            assessment = self.assess_motion(
                int(servo_id),
                require_calibrated_bounds=self.require_calibrated_bounds_for_individual_motion(),
            )
        except Exception as exc:
            return ServoMotionPlan(
                servo_id=int(servo_id),
                action=str(action),
                current_position_tick=None,
                step_ticks=int(step_ticks),
                delta_ticks=int(delta_ticks),
                unclamped_target_tick=None,
                clamped_target_tick=None,
                safe_min_tick=None,
                safe_max_tick=None,
                clamped=False,
                allowed=False,
                block_reason=str(exc),
                assessment=None,
            )

        current_position = assessment.telemetry.present_position
        safe_min = assessment.safe_min_tick
        safe_max = assessment.safe_max_tick
        if not assessment.ready:
            return ServoMotionPlan(
                servo_id=int(servo_id),
                action=str(action),
                current_position_tick=current_position,
                step_ticks=int(step_ticks),
                delta_ticks=int(delta_ticks),
                unclamped_target_tick=None,
                clamped_target_tick=None,
                safe_min_tick=safe_min,
                safe_max_tick=safe_max,
                clamped=False,
                allowed=False,
                block_reason=assessment.reason,
                assessment=assessment,
            )
        if current_position is None:
            return ServoMotionPlan(
                servo_id=int(servo_id),
                action=str(action),
                current_position_tick=None,
                step_ticks=int(step_ticks),
                delta_ticks=int(delta_ticks),
                unclamped_target_tick=None,
                clamped_target_tick=None,
                safe_min_tick=safe_min,
                safe_max_tick=safe_max,
                clamped=False,
                allowed=False,
                block_reason=f"Servo {servo_id} position is unavailable.",
                assessment=assessment,
            )
        if safe_min is None or safe_max is None:
            return ServoMotionPlan(
                servo_id=int(servo_id),
                action=str(action),
                current_position_tick=int(current_position),
                step_ticks=int(step_ticks),
                delta_ticks=int(delta_ticks),
                unclamped_target_tick=None,
                clamped_target_tick=None,
                safe_min_tick=safe_min,
                safe_max_tick=safe_max,
                clamped=False,
                allowed=False,
                block_reason=f"Servo {servo_id} active motion range is unavailable.",
                assessment=assessment,
            )
        unclamped_target = int(current_position) + int(delta_ticks)
        bounded_min = max(int(RAW_POSITION_MIN_TICK), int(safe_min))
        bounded_max = min(int(RAW_POSITION_MAX_TICK), int(safe_max))
        if bounded_min > bounded_max:
            return ServoMotionPlan(
                servo_id=int(servo_id),
                action=str(action),
                current_position_tick=int(current_position),
                step_ticks=int(step_ticks),
                delta_ticks=int(delta_ticks),
                unclamped_target_tick=unclamped_target,
                clamped_target_tick=None,
                safe_min_tick=bounded_min,
                safe_max_tick=bounded_max,
                clamped=False,
                allowed=False,
                block_reason=(
                    f"Servo {servo_id} active motion range is invalid after applying the canonical raw position range."
                ),
                assessment=assessment,
            )
        clamped_target = min(max(int(unclamped_target), bounded_min), bounded_max)
        clamped = int(clamped_target) != int(unclamped_target)
        if int(clamped_target) == int(current_position):
            boundary = "minimum" if int(delta_ticks) < 0 else "maximum"
            return ServoMotionPlan(
                servo_id=int(servo_id),
                action=str(action),
                current_position_tick=int(current_position),
                step_ticks=int(step_ticks),
                delta_ticks=int(delta_ticks),
                unclamped_target_tick=unclamped_target,
                clamped_target_tick=int(clamped_target),
                safe_min_tick=bounded_min,
                safe_max_tick=bounded_max,
                clamped=clamped,
                allowed=False,
                block_reason=(
                    f"Servo {servo_id} is already at the active {boundary} raw position {clamped_target}; "
                    f"{action} would not move the servo."
                ),
                assessment=assessment,
            )
        return ServoMotionPlan(
            servo_id=int(servo_id),
            action=str(action),
            current_position_tick=int(current_position),
            step_ticks=int(step_ticks),
            delta_ticks=int(delta_ticks),
            unclamped_target_tick=int(unclamped_target),
            clamped_target_tick=int(clamped_target),
            safe_min_tick=bounded_min,
            safe_max_tick=bounded_max,
            clamped=clamped,
            allowed=True,
            block_reason="",
            assessment=assessment,
        )

    def _execute_motion_plan(
        self,
        plan: ServoMotionPlan,
        *,
        command_direction: str,
    ) -> ServoJogResult:
        if not plan.allowed or plan.clamped_target_tick is None:
            return ServoJogResult(
                servo_id=int(plan.servo_id),
                command_direction=str(command_direction),
                step_ticks=int(plan.step_ticks),
                delta_ticks=int(plan.delta_ticks),
                success=False,
                blocked=True,
                status="blocked",
                message=f"Blocked {plan.action} for servo {plan.servo_id}: {plan.block_reason}",
                goal_tick=plan.clamped_target_tick,
                telemetry=plan.assessment.telemetry if plan.assessment is not None else None,
                assessment=plan.assessment,
                current_position_tick=plan.current_position_tick,
                unclamped_goal_tick=plan.unclamped_target_tick,
                safe_min_tick=plan.safe_min_tick,
                safe_max_tick=plan.safe_max_tick,
                clamped=plan.clamped,
            )
        updated: ServoTelemetry | None = None
        updated_assessment: ServoMotionAssessment | None = None
        try:
            self._write_goal_positions({int(plan.servo_id): int(plan.clamped_target_tick)})
            updated = self.read_live_telemetry([int(plan.servo_id)])[int(plan.servo_id)]
            self._validate_post_motion(updated)
            updated_assessment = self.assess_motion(
                int(plan.servo_id),
                require_calibrated_bounds=self.require_calibrated_bounds_for_individual_motion(),
                telemetry=updated,
            )
        except Exception as exc:
            return ServoJogResult(
                servo_id=int(plan.servo_id),
                command_direction=str(command_direction),
                step_ticks=int(plan.step_ticks),
                delta_ticks=int(plan.delta_ticks),
                success=False,
                blocked=True,
                status="blocked",
                message=f"Blocked {plan.action} for servo {plan.servo_id}: {exc}",
                goal_tick=plan.clamped_target_tick,
                telemetry=updated or (plan.assessment.telemetry if plan.assessment is not None else None),
                assessment=updated_assessment or plan.assessment,
                current_position_tick=plan.current_position_tick,
                unclamped_goal_tick=plan.unclamped_target_tick,
                safe_min_tick=plan.safe_min_tick,
                safe_max_tick=plan.safe_max_tick,
                clamped=plan.clamped,
            )
        if plan.clamped:
            detail = (
                f"requested {plan.unclamped_target_tick}, clamped to {plan.clamped_target_tick} "
                f"within active range [{plan.safe_min_tick}, {plan.safe_max_tick}]"
            )
        else:
            detail = (
                f"target {plan.clamped_target_tick} within active range [{plan.safe_min_tick}, {plan.safe_max_tick}]"
            )
        return ServoJogResult(
            servo_id=int(plan.servo_id),
            command_direction=str(command_direction),
            step_ticks=int(plan.step_ticks),
            delta_ticks=int(plan.delta_ticks),
            success=True,
            blocked=False,
            status="moved",
            message=(
                f"Sent {plan.action} for servo {plan.servo_id}: current {plan.current_position_tick}, {detail}."
            ),
            goal_tick=plan.clamped_target_tick,
            telemetry=updated,
            assessment=updated_assessment,
            current_position_tick=plan.current_position_tick,
            unclamped_goal_tick=plan.unclamped_target_tick,
            safe_min_tick=plan.safe_min_tick,
            safe_max_tick=plan.safe_max_tick,
            clamped=plan.clamped,
        )

    @staticmethod
    def _validate_goal_against_assessment(assessment: ServoMotionAssessment, goal_tick: int) -> None:
        if assessment.safe_min_tick is None or assessment.safe_max_tick is None:
            raise RuntimeError(f"Servo {assessment.servo_id} active motion range is unavailable.")
        if int(goal_tick) < int(assessment.safe_min_tick) or int(goal_tick) > int(assessment.safe_max_tick):
            raise ValueError(
                f"Servo {assessment.servo_id} goal {goal_tick} is outside the active motion range "
                f"[{assessment.safe_min_tick}, {assessment.safe_max_tick}]."
            )

    def _validate_post_motion(self, telemetry: ServoTelemetry) -> None:
        if telemetry.hardware_error_code not in (None, 0) or telemetry.hardware_error:
            raise RuntimeError(
                f"Servo {telemetry.servo_id} reported a hardware/status error after motion: "
                f"{telemetry.hardware_error or f'0x{telemetry.hardware_error_code:02X}'}"
            )
        try:
            self.safety_guard.validate_voltage(
                telemetry.present_voltage_mv,
                require_present=self.dxl_bus.config.require_voltage_for_motion,
            )
            self.safety_guard.validate_currents(
                [telemetry.present_current_ma],
                require_present=self.dxl_bus.config.require_current_for_motion,
            )
            self.safety_guard.validate_temperature(
                telemetry.present_temperature_c,
                require_present=self.dxl_bus.config.require_temperature_for_motion,
            )
        except ValueError as exc:
            raise RuntimeError(
                f"Servo {telemetry.servo_id} reported unsafe telemetry after motion: {exc}"
            ) from exc

    def _validate_capture_telemetry(self, telemetry: ServoTelemetry, *, servo_id: int) -> None:
        if telemetry.present_position is None:
            raise RuntimeError(f"Servo {servo_id} position is unavailable.")
        if telemetry.hardware_error_code not in (None, 0):
            raise RuntimeError(
                f"Servo {servo_id} hardware error status is 0x{int(telemetry.hardware_error_code):02X}."
            )
        if telemetry.hardware_error:
            raise RuntimeError(f"Servo {servo_id} reported an error during neutral capture: {telemetry.hardware_error}")
        try:
            if self.dxl_bus.config.require_fresh_telemetry_for_motion:
                self.safety_guard.validate_telemetry_freshness(telemetry.last_read_monotonic_s)
            self.safety_guard.validate_voltage(telemetry.present_voltage_mv, require_present=True)
            self.safety_guard.validate_temperature(telemetry.present_temperature_c, require_present=True)
        except ValueError as exc:
            raise RuntimeError(f"Servo {servo_id} is not safe to capture neutral: {exc}") from exc

    @staticmethod
    def _identity_read_ok(telemetry: ServoTelemetry) -> bool:
        return (
            telemetry.identity_error is None
            and telemetry.reported_servo_id is not None
            and telemetry.model_number is not None
            and telemetry.firmware_version is not None
        )

    @staticmethod
    def _missing_identity_fields(telemetry: ServoTelemetry) -> str:
        missing = []
        if telemetry.reported_servo_id is None:
            missing.append("servo_id")
        if telemetry.model_number is None:
            missing.append("model_number")
        if telemetry.firmware_version is None:
            missing.append("firmware_version")
        return ", ".join(missing) if missing else "unknown identity read error"

    @staticmethod
    def _telemetry_read_ok(telemetry: ServoTelemetry) -> bool:
        return (
            telemetry.telemetry_error is None
            and telemetry.operating_mode is not None
            and telemetry.min_position_limit is not None
            and telemetry.max_position_limit is not None
            and telemetry.present_position is not None
            and telemetry.present_current_ma is not None
            and telemetry.present_voltage_mv is not None
            and telemetry.present_temperature_c is not None
            and telemetry.hardware_error_code is not None
        )

    def _runtime_telemetry_status(
        self,
        *,
        telemetry: ServoTelemetry | None,
        motion_assessment: ServoMotionAssessment | None,
    ) -> str:
        if telemetry is None:
            return "Unreadable"
        fresh = self.telemetry_is_fresh(telemetry)
        if fresh is False:
            return "Stale"
        if motion_assessment is not None:
            for reason in motion_assessment.blocking_reasons:
                if "telemetry is stale" in str(reason).lower():
                    return "Stale"
        if any(
            value is None
            for value in (
                telemetry.present_position,
                telemetry.present_current_ma,
                telemetry.present_voltage_mv,
                telemetry.present_temperature_c,
            )
        ):
            return "Unreadable"
        return "Live"

    @staticmethod
    def _missing_telemetry_fields(telemetry: ServoTelemetry) -> str:
        missing = []
        if telemetry.operating_mode is None:
            missing.append("operating_mode")
        if telemetry.min_position_limit is None:
            missing.append("min_position_limit")
        if telemetry.max_position_limit is None:
            missing.append("max_position_limit")
        if telemetry.present_position is None:
            missing.append("present_position")
        if telemetry.present_current_ma is None:
            missing.append("present_current")
        if telemetry.present_voltage_mv is None:
            missing.append("present_voltage")
        if telemetry.present_temperature_c is None:
            missing.append("present_temperature")
        if telemetry.hardware_error_code is None:
            missing.append("hardware_error_status")
        return ", ".join(missing) if missing else "unknown telemetry read error"

"""Servos tab controller for calibration, telemetry, and manual motion."""

from __future__ import annotations

from dataclasses import dataclass, field
import threading

from continuum_robot.config.settings import Settings
from continuum_robot.servos.servo_service import PretensionRoutineResult, ServoMotionAssessment


@dataclass
class ServosViewState:
    """UI-facing servo control state."""

    connected: bool = False
    robot_mode: str = ""
    single_servo_mode: bool = False
    expected_servo_ids: list[int] = field(default_factory=list)
    servo_ids: list[int] = field(default_factory=list)
    neutral_setpoints: dict[int, int] = field(default_factory=dict)
    calibration_exists: bool = False
    calibration_compatible: bool = False
    calibration_status: str = "missing"
    calibration_message: str = "No servo calibration file found."
    calibration_path: str = ""
    calibration_updated_at_utc: str | None = None
    calibration_rows: list[dict[str, str]] = field(default_factory=list)
    telemetry: dict[int, dict] = field(default_factory=dict)
    tendon_displacements_cm: list[float] = field(default_factory=list)
    fine_jog_step_ticks: int = 5
    coarse_jog_step_ticks: int = 25
    default_pretension_threshold_ma: int = 220
    selected_servo_id: int | None = None
    discovery_status: str = "idle"
    discovery_message: str = "One-servo discovery not run yet."
    position_convention_summary: str = ""
    selected_servo_current_position_tick: int | None = None
    selected_servo_last_target_tick: int | None = None
    selected_servo_last_unclamped_target_tick: int | None = None
    selected_servo_safe_min_tick: int | None = None
    selected_servo_safe_max_tick: int | None = None
    selected_servo_last_motion_clamped: bool = False
    selected_servo_last_motion_summary: str = "No jog command sent yet."
    blocking_reasons: list[str] = field(default_factory=list)
    selected_servo_external_power_ready: bool | None = None
    pretension_running: bool = False
    pretension_result_can_accept: bool = False
    pretension_message: str = "Pretension not checked."
    bench_debug_text: str = ""
    status_message: str = "Servo control idle."
    last_error: str | None = None


class ServosController:
    """Owns manual jog, displacement command, and telemetry display actions."""

    def __init__(self, servo_service, settings: Settings) -> None:
        self.servo_service = servo_service
        self.settings = settings
        self._pretension_thread: threading.Thread | None = None
        self._pretension_stop: threading.Event | None = None
        self._last_pretension_result: PretensionRoutineResult | None = None
        self.state = ServosViewState(
            connected=servo_service.is_connected,
            robot_mode=settings.robot.mode,
            single_servo_mode=(settings.robot.mode == "1-servo" or len(settings.robot.servo_ids) == 1),
            expected_servo_ids=list(settings.robot.servo_ids),
            servo_ids=list(settings.robot.servo_ids),
            tendon_displacements_cm=[0.0] * len(settings.robot.tendon_to_servo),
            fine_jog_step_ticks=settings.safety.fine_jog_step_ticks,
            coarse_jog_step_ticks=settings.safety.coarse_jog_step_ticks,
            default_pretension_threshold_ma=settings.safety.default_pretension_current_threshold_ma,
            selected_servo_id=(int(settings.robot.servo_ids[0]) if settings.robot.servo_ids else None),
            position_convention_summary=servo_service.position_convention_summary(),
            status_message=(
                "Mock servo backend ready."
                if settings.runtime.mock_mode
                else (
                    "Hardware servo mode ready. Use one-servo bring-up first: connect OpenRB, "
                    "prepare the bridge path, scan one servo, verify telemetry, then use fine jog only."
                )
            ),
        )
        self.load_neutral_setpoints()
        self.refresh()

    def refresh(self) -> ServosViewState:
        self.state.connected = self.servo_service.is_connected
        self.state.robot_mode = self.settings.robot.mode
        self.state.expected_servo_ids = list(self.settings.robot.servo_ids)
        self.state.single_servo_mode = self.settings.robot.mode == "1-servo" or len(self.state.servo_ids) == 1
        self._refresh_calibration_summary()
        self._sync_active_neutral_setpoints()
        if not self.state.connected:
            self.state.telemetry = {}
            self.state.selected_servo_external_power_ready = None
            self.state.bench_debug_text = self._build_disconnected_bench_debug_text()
            if self.state.discovery_status == "idle":
                self.state.discovery_message = "Connect OpenRB and the DYNAMIXEL bus before discovery."
            self._sync_selected_servo_motion_state()
            return self.state
        if self.state.single_servo_mode:
            return self._refresh_single_servo_state()
        if not self.state.servo_ids:
            self.state.telemetry = {}
            self.state.blocking_reasons = []
            self.state.selected_servo_external_power_ready = None
            self.state.bench_debug_text = self._build_disconnected_bench_debug_text()
            self._sync_selected_servo_motion_state()
            return self.state
        if self.state.servo_ids:
            try:
                telemetry = self.servo_service.read_telemetry(self.state.servo_ids)
            except Exception as exc:
                self.state.last_error = str(exc)
                self.state.status_message = f"Telemetry refresh failed: {exc}"
                self.state.telemetry = {}
                self.state.bench_debug_text = self._build_disconnected_bench_debug_text(extra_error=str(exc))
                self._sync_selected_servo_motion_state()
                return self.state
            if self.state.selected_servo_id not in telemetry and telemetry:
                self.state.selected_servo_id = sorted(telemetry)[0]
            assessments = {
                int(servo_id): self.servo_service.assess_motion(
                    int(servo_id),
                    require_calibrated_bounds=True,
                    telemetry=item,
                )
                for servo_id, item in telemetry.items()
            }
            self.state.telemetry = {
                servo_id: {
                    "reported_servo_id": item.reported_servo_id,
                    "model_number": item.model_number,
                    "firmware_version": item.firmware_version,
                    "operating_mode": item.operating_mode,
                    "torque_enabled": item.torque_enabled,
                    "position": item.present_position,
                    "current_ma": item.present_current_ma,
                    "voltage_mv": item.present_voltage_mv,
                    "temperature_c": item.present_temperature_c,
                    "min_position_limit": item.min_position_limit,
                    "max_position_limit": item.max_position_limit,
                    "hardware_error": item.hardware_error_code,
                    "error": item.hardware_error,
                    "safe_min_tick": assessments[int(servo_id)].safe_min_tick,
                    "safe_max_tick": assessments[int(servo_id)].safe_max_tick,
                    "safe_bounds": self._assessment_bounds_text(assessments[int(servo_id)]),
                    "ready": self._assessment_status_text(assessments[int(servo_id)]),
                    "position_convention": self.servo_service.position_convention_summary(),
                    "external_power_ready": assessments[int(servo_id)].external_power_ready,
                    "blocking_reasons": list(assessments[int(servo_id)].blocking_reasons),
                }
                for servo_id, item in telemetry.items()
            }
            selected = assessments.get(int(self.state.selected_servo_id)) if self.state.selected_servo_id is not None else None
            if selected is not None:
                self.state.blocking_reasons = list(selected.blocking_reasons)
                self.state.selected_servo_external_power_ready = selected.external_power_ready
            self._sync_selected_servo_motion_state()
        return self.state

    def scan(self) -> list[int]:
        try:
            discovery = self.servo_service.discover_one_servo(
                expected_servo_id=self._expected_servo_id(),
                allow_scan=True,
            )
            self._apply_discovery_snapshot(discovery)
            self.state.last_error = None
            return self.state.servo_ids
        except Exception as exc:
            self.state.last_error = str(exc)
            self.state.status_message = f"Servo scan failed: {exc}"
            raise
        finally:
            self.refresh()

    def refresh_readiness(self) -> ServosViewState:
        discovery = self.servo_service.discover_one_servo(
            expected_servo_id=self._expected_servo_id(),
            allow_scan=False,
        )
        self._apply_discovery_snapshot(discovery)
        return self.refresh()

    def assign_servo_id(self, current_id: int, new_id: int) -> None:
        result = self.servo_service.assign_servo_id_safely(current_id, new_id)
        self.state.status_message = result.message
        self.state.discovery_status = result.status
        if result.success:
            self.state.servo_ids = list(result.selected_ids)
            self.state.selected_servo_id = int(new_id)
            self.state.last_error = None
            self.scan()
            return
        self.state.last_error = result.message
        raise RuntimeError(result.message)

    def jog_servo(self, servo_id: int, delta_ticks: int) -> None:
        try:
            result = self.servo_service.jog_servo(servo_id, delta_ticks)
            self.state.status_message = result.message
            self.state.last_error = None
        except Exception as exc:
            self.state.last_error = str(exc)
            self.state.status_message = f"Jog failed: {exc}"
            raise
        finally:
            self.refresh()

    def fine_jog(self, servo_id: int, direction: int) -> None:
        action = "tighten_fine" if int(direction) > 0 else "loosen_fine"
        self._run_jog_action(int(servo_id), action)

    def coarse_jog(self, servo_id: int, direction: int) -> None:
        action = "tighten_coarse" if int(direction) > 0 else "loosen_coarse"
        self._run_jog_action(int(servo_id), action)

    def set_tendon_displacements(self, values: list[float]) -> None:
        self.state.tendon_displacements_cm = list(values)

    def apply_displacement(self) -> None:
        try:
            neutral = self._neutral_ticks_for_current_ids()
            result = self.servo_service.command_displacement(
                tendon_displacements_cm=self.state.tendon_displacements_cm,
                neutral_ticks=neutral,
                servo_ids=self.state.servo_ids,
            )
            self.state.status_message = result.message
            self.state.last_error = None
        except Exception as exc:
            self.state.last_error = str(exc)
            self.state.status_message = f"Displacement rejected: {exc}"
            raise
        finally:
            self.refresh()

    def capture_neutral_setpoints(self) -> dict[int, int]:
        try:
            servo_ids = self._capture_servo_ids()
            result = self.servo_service.capture_and_save_neutral_setpoints(servo_ids)
            setpoints = result.setpoints_by_id
            self.state.neutral_setpoints = setpoints
            self.state.status_message = result.message
            self.state.last_error = None
            return setpoints
        except Exception as exc:
            self.state.last_error = str(exc)
            self.state.status_message = f"Neutral capture failed: {exc}"
            raise
        finally:
            self.refresh()

    def save_neutral_setpoints(self) -> None:
        self.servo_service.save_neutral_setpoints(self.state.neutral_setpoints)
        self.state.status_message = "Servo calibration artifact saved."
        self.state.last_error = None
        self.refresh()

    def load_neutral_setpoints(self) -> dict[int, int]:
        self._sync_active_neutral_setpoints()
        self._refresh_calibration_summary()
        if self.state.calibration_exists and not self.state.calibration_compatible:
            self.state.status_message = (
                "Saved servo calibration is incompatible with the active one-servo bench configuration. "
                "The artifact is shown for review only and is not being used as active neutral bounds."
            )
            return {}
        self.state.status_message = "Loaded servo calibration artifact."
        return dict(self.state.neutral_setpoints)

    def save_startup_calibration(
        self,
        *,
        servo_id: int,
        min_offset_ticks: int,
        max_offset_ticks: int,
        threshold_ma: int,
    ) -> None:
        try:
            entry = self.servo_service.save_startup_calibration(
                servo_id=int(servo_id),
                min_offset_ticks=int(min_offset_ticks),
                max_offset_ticks=int(max_offset_ticks),
                pretension_current_threshold_ma=int(threshold_ma),
            )
            if entry.neutral_setpoint is not None:
                self.state.neutral_setpoints[int(servo_id)] = int(entry.neutral_setpoint)
            self.state.status_message = (
                f"Saved startup calibration for servo {servo_id}: neutral {entry.neutral_setpoint}, "
                f"bounds {entry.safe_min_tick}..{entry.safe_max_tick}, threshold {entry.pretension_current_threshold_ma} mA."
            )
            self.state.last_error = None
        except Exception as exc:
            self.state.last_error = str(exc)
            self.state.status_message = f"Startup calibration failed: {exc}"
            raise
        finally:
            self.refresh()

    def start_pretension(self, servo_id: int, threshold_ma: int | None = None) -> None:
        if self._pretension_thread and self._pretension_thread.is_alive():
            raise RuntimeError("Pretension is already running.")
        self._pretension_stop = threading.Event()
        self._last_pretension_result = None
        self.state.pretension_running = True
        self.state.pretension_result_can_accept = False
        self.state.pretension_message = f"Pretension running for servo {servo_id}."
        self.state.status_message = self.state.pretension_message
        self.state.last_error = None

        def _worker() -> None:
            try:
                result = self.servo_service.run_pretension_routine(
                    servo_id=int(servo_id),
                    threshold_ma=threshold_ma,
                    stop_requested=(self._pretension_stop.is_set if self._pretension_stop else None),
                )
                self._last_pretension_result = result
                self.state.pretension_message = result.message
                self.state.status_message = result.message
                self.state.last_error = None
                self.state.pretension_result_can_accept = bool(result.success)
            except Exception as exc:
                self.state.last_error = str(exc)
                self.state.pretension_message = f"Pretension failed: {exc}"
                self.state.status_message = self.state.pretension_message
            finally:
                self.state.pretension_running = False

        self._pretension_thread = threading.Thread(target=_worker, name="servo-pretension", daemon=True)
        self._pretension_thread.start()

    def cancel_pretension(self) -> None:
        if self._pretension_stop is not None:
            self._pretension_stop.set()
            self.state.status_message = "Pretension cancel requested."
            self.state.pretension_message = self.state.status_message

    def accept_pretension_result(self, servo_id: int) -> None:
        try:
            self.servo_service.accept_pretension_result(int(servo_id))
            self.state.pretension_result_can_accept = False
            self.state.status_message = f"Accepted pretension result for servo {servo_id}."
            self.state.pretension_message = self.state.status_message
            self.state.last_error = None
        except Exception as exc:
            self.state.last_error = str(exc)
            self.state.status_message = f"Accept pretension failed: {exc}"
            self.state.pretension_message = self.state.status_message
            raise
        finally:
            self.refresh()

    def validate_pretension(self) -> None:
        try:
            result = self.servo_service.validate_pretension(
                servo_ids=self.state.servo_ids,
                tolerance_ma=self.settings.safety.pretension_current_balance_tolerance_ma,
            )
            self.state.pretension_message = result.message
            self.state.status_message = result.message
            self.state.last_error = None
        except Exception as exc:
            self.state.last_error = str(exc)
            self.state.pretension_message = f"Pretension validation failed: {exc}"
            self.state.status_message = self.state.pretension_message
            raise
        finally:
            self.refresh()

    def shutdown(self) -> None:
        self.cancel_pretension()
        if self._pretension_thread is not None:
            self._pretension_thread.join(timeout=0.2)

    def _expected_servo_id(self) -> int | None:
        if self.state.selected_servo_id is not None:
            return int(self.state.selected_servo_id)
        if self.settings.robot.servo_ids:
            return int(self.settings.robot.servo_ids[0])
        return None

    def _capture_servo_ids(self) -> list[int]:
        if self.state.single_servo_mode:
            servo_id = self.state.selected_servo_id or self._expected_servo_id()
            if servo_id is None:
                raise RuntimeError("No servo is selected for one-servo neutral capture.")
            return [int(servo_id)]
        return list(self.state.servo_ids)

    def _run_jog_action(self, servo_id: int, action: str) -> None:
        result = self.servo_service.jog_servo_action(
            servo_id=int(servo_id),
            action=str(action),
        )
        self.state.status_message = result.message
        self.state.last_error = None if result.success else result.message
        self._apply_jog_result(result)
        if result.blocked:
            self.refresh()
            raise RuntimeError(result.message)
        self.refresh()

    def _neutral_ticks_for_current_ids(self) -> list[int]:
        if not self.state.neutral_setpoints:
            raise RuntimeError("Neutral setpoints are missing. Capture or load them first.")
        if self.state.calibration_exists and not self.state.calibration_compatible:
            raise RuntimeError(
                "Saved servo calibration does not match the current robot configuration. "
                "Review the calibration summary and recapture neutral before commanding motion."
            )
        missing = [sid for sid in self.state.servo_ids if sid not in self.state.neutral_setpoints]
        if missing:
            raise RuntimeError(f"Neutral setpoints missing for servo IDs: {missing}")
        return [self.state.neutral_setpoints[sid] for sid in self.state.servo_ids]

    def _apply_discovery_snapshot(self, discovery) -> None:
        self.state.discovery_status = discovery.status
        self.state.discovery_message = discovery.message
        if self.state.single_servo_mode and discovery.expected_servo_id is not None:
            self.state.servo_ids = [int(discovery.selected_servo_id or discovery.expected_servo_id)]
        else:
            self.state.servo_ids = list(discovery.discovered_ids)
        if discovery.selected_servo_id is not None:
            self.state.selected_servo_id = int(discovery.selected_servo_id)
        elif discovery.expected_servo_id is not None and self.state.selected_servo_id is None:
            self.state.selected_servo_id = int(discovery.expected_servo_id)
        if discovery.motion_assessment is not None:
            self.state.blocking_reasons = list(discovery.motion_assessment.blocking_reasons)
            self.state.selected_servo_external_power_ready = discovery.motion_assessment.external_power_ready
        else:
            self.state.blocking_reasons = []
            self.state.selected_servo_external_power_ready = None
        self.state.status_message = discovery.message

    def _refresh_calibration_summary(self) -> None:
        summary = self.servo_service.get_calibration_summary()
        self.state.calibration_exists = summary.exists
        self.state.calibration_compatible = summary.compatible
        self.state.calibration_status = summary.status
        self.state.calibration_message = summary.message
        if (
            self.state.single_servo_mode
            and summary.exists
            and not summary.compatible
            and len(summary.servo_entries) > 1
        ):
            self.state.calibration_message = (
                "Warning: a multi-servo calibration artifact is loaded. "
                "One-servo bench mode is ignoring it until neutral is recaptured."
            )
        self.state.calibration_path = summary.path
        self.state.calibration_updated_at_utc = summary.updated_at_utc
        self.state.calibration_rows = []
        for servo_id in sorted(summary.servo_entries):
            entry = summary.servo_entries[servo_id]
            self.state.calibration_rows.append(
                {
                    "servo_id": str(servo_id),
                    "neutral": str(entry.neutral_setpoint) if entry.neutral_setpoint is not None else "—",
                    "bounds": (
                        f"{entry.safe_min_tick} .. {entry.safe_max_tick}"
                        if entry.safe_min_tick is not None and entry.safe_max_tick is not None
                        else "missing"
                    ),
                    "threshold": (
                        str(entry.pretension_current_threshold_ma)
                        if entry.pretension_current_threshold_ma is not None
                        else "missing"
                    ),
                    "direction": entry.tightening_rotation or "unset",
                    "pretension": (
                        f"{entry.pretension_result_status or '—'} @ {entry.pretension_final_position_tick}"
                        if entry.pretension_final_position_tick is not None
                        else (entry.pretension_result_status or "—")
                    ),
                    "status": "valid" if entry.valid else entry.status,
                }
            )

    def _assessment_bounds_text(self, assessment: ServoMotionAssessment) -> str:
        if assessment.safe_min_tick is None or assessment.safe_max_tick is None:
            return "unavailable"
        return f"{assessment.safe_min_tick} .. {assessment.safe_max_tick}"

    def _assessment_status_text(self, assessment: ServoMotionAssessment) -> str:
        return "ready" if assessment.ready else assessment.reason

    def _refresh_single_servo_state(self) -> ServosViewState:
        expected_servo_id = self._expected_servo_id()
        snapshot = self.servo_service.build_bench_debug_snapshot(expected_servo_id)
        servo_id = int(snapshot.selected_servo_id or snapshot.expected_servo_id or 1)
        self.state.servo_ids = [servo_id] if snapshot.expected_servo_id is not None else []
        self.state.selected_servo_id = servo_id if snapshot.expected_servo_id is not None else None
        self.state.discovery_status = snapshot.status
        self.state.discovery_message = snapshot.message
        self.state.bench_debug_text = self._build_bench_debug_text(snapshot)
        self.state.last_error = (
            snapshot.message
            if snapshot.ping_ok is False or snapshot.identity_read_ok is False or snapshot.telemetry_read_ok is False
            else None
        )
        self.state.selected_servo_external_power_ready = (
            snapshot.motion_assessment.external_power_ready
            if snapshot.motion_assessment is not None
            else None
        )
        self.state.blocking_reasons = [snapshot.motion_block_reason] if snapshot.motion_block_reason else []
        self.state.telemetry = {}
        if snapshot.expected_servo_id is not None:
            self.state.telemetry[servo_id] = self._telemetry_row_from_snapshot(snapshot, servo_id)
        self._sync_selected_servo_motion_state()
        return self.state

    def _telemetry_row_from_snapshot(self, snapshot, servo_id: int) -> dict:
        telemetry = snapshot.telemetry
        assessment = snapshot.motion_assessment
        error = None
        if telemetry is not None:
            error = telemetry.hardware_error or telemetry.identity_error or telemetry.telemetry_error
        if error is None and snapshot.motion_block_reason and not snapshot.motion_ready:
            error = snapshot.motion_block_reason
        ready_text = (
            self._assessment_status_text(assessment)
            if assessment is not None
            else ("ready" if snapshot.motion_ready else (snapshot.motion_block_reason or snapshot.status))
        )
        return {
            "reported_servo_id": telemetry.reported_servo_id if telemetry is not None else None,
            "model_number": telemetry.model_number if telemetry is not None else None,
            "firmware_version": telemetry.firmware_version if telemetry is not None else None,
            "operating_mode": telemetry.operating_mode if telemetry is not None else None,
            "torque_enabled": telemetry.torque_enabled if telemetry is not None else None,
            "position": telemetry.present_position if telemetry is not None else None,
            "current_ma": telemetry.present_current_ma if telemetry is not None else None,
            "voltage_mv": telemetry.present_voltage_mv if telemetry is not None else None,
            "temperature_c": telemetry.present_temperature_c if telemetry is not None else None,
            "min_position_limit": telemetry.min_position_limit if telemetry is not None else None,
            "max_position_limit": telemetry.max_position_limit if telemetry is not None else None,
            "hardware_error": telemetry.hardware_error_code if telemetry is not None else None,
            "error": error,
            "safe_min_tick": assessment.safe_min_tick if assessment is not None else None,
            "safe_max_tick": assessment.safe_max_tick if assessment is not None else None,
            "safe_bounds": self._assessment_bounds_text(assessment) if assessment is not None else "unavailable",
            "ready": ready_text,
            "position_convention": self.servo_service.position_convention_summary(),
            "external_power_ready": snapshot.selected_servo_id is not None and snapshot.motion_assessment is not None and snapshot.motion_assessment.external_power_ready,
            "blocking_reasons": list(assessment.blocking_reasons) if assessment is not None else ([snapshot.motion_block_reason] if snapshot.motion_block_reason else []),
        }

    def _build_bench_debug_text(self, snapshot) -> str:
        telemetry = snapshot.telemetry
        last_hw_error = None
        if telemetry is not None:
            last_hw_error = telemetry.hardware_error or (
                f"0x{telemetry.hardware_error_code:02X}"
                if telemetry.hardware_error_code not in (None, 0)
                else "0"
            )
        return "\n".join(
            [
                "Bench debug:",
                f"bus_connected={self.state.connected}",
                f"selected_port={snapshot.selected_port or 'unset'}",
                f"selected_baud={snapshot.selected_baud}",
                f"expected_servo_id={snapshot.expected_servo_id}",
                f"ping_ok={snapshot.ping_ok}",
                f"identity_read_ok={snapshot.identity_read_ok}",
                f"telemetry_read_ok={snapshot.telemetry_read_ok}",
                f"last_position={telemetry.present_position if telemetry is not None else None}",
                f"last_current={telemetry.present_current_ma if telemetry is not None else None}",
                f"last_voltage={telemetry.present_voltage_mv if telemetry is not None else None}",
                f"last_temperature={telemetry.present_temperature_c if telemetry is not None else None}",
                f"last_hw_error={last_hw_error}",
                f"calibration_entries_loaded={snapshot.calibration_entries_loaded}",
                f"one_servo_mode_ok={snapshot.one_servo_mode_ok}",
                "position_convention=tighten->smaller_counts; loosen->larger_counts",
                f"motion_block_reason={snapshot.motion_block_reason or 'none'}",
            ]
        )

    def _build_disconnected_bench_debug_text(self, *, extra_error: str | None = None) -> str:
        summary = self.servo_service.get_calibration_summary()
        return "\n".join(
            [
                "Bench debug:",
                f"bus_connected={self.state.connected}",
                f"selected_port={self.settings.serial.openrb_port or 'unset'}",
                f"selected_baud={self.settings.serial.baudrate}",
                f"expected_servo_id={self._expected_servo_id()}",
                "ping_ok=None",
                "identity_read_ok=None",
                "telemetry_read_ok=None",
                "last_position=None",
                "last_current=None",
                "last_voltage=None",
                "last_temperature=None",
                "last_hw_error=None",
                f"calibration_entries_loaded={sorted(summary.servo_entries)}",
                f"one_servo_mode_ok={self.state.single_servo_mode}",
                "position_convention=tighten->smaller_counts; loosen->larger_counts",
                f"motion_block_reason={extra_error or 'OpenRB/DYNAMIXEL not ready'}",
            ]
        )

    def _apply_jog_result(self, result) -> None:
        self.state.selected_servo_id = int(result.servo_id)
        self.state.selected_servo_current_position_tick = result.current_position_tick
        self.state.selected_servo_last_target_tick = result.goal_tick
        self.state.selected_servo_last_unclamped_target_tick = result.unclamped_goal_tick
        self.state.selected_servo_safe_min_tick = result.safe_min_tick
        self.state.selected_servo_safe_max_tick = result.safe_max_tick
        self.state.selected_servo_last_motion_clamped = bool(result.clamped)
        if result.success:
            if result.clamped and result.goal_tick is not None:
                self.state.selected_servo_last_motion_summary = (
                    f"sent {result.command_direction} to {result.goal_tick} (requested {result.unclamped_goal_tick}, clamped)"
                )
            elif result.goal_tick is not None:
                self.state.selected_servo_last_motion_summary = (
                    f"sent {result.command_direction} to {result.goal_tick}"
                )
            else:
                self.state.selected_servo_last_motion_summary = f"sent {result.command_direction}"
            return
        self.state.selected_servo_last_motion_summary = result.message

    def _sync_selected_servo_motion_state(self) -> None:
        selected_servo_id = self.state.selected_servo_id
        if selected_servo_id is None:
            self.state.selected_servo_current_position_tick = None
            self.state.selected_servo_safe_min_tick = None
            self.state.selected_servo_safe_max_tick = None
            return
        selected = self.state.telemetry.get(int(selected_servo_id), {})
        self.state.selected_servo_current_position_tick = selected.get("position")
        self.state.selected_servo_safe_min_tick = selected.get("safe_min_tick")
        self.state.selected_servo_safe_max_tick = selected.get("safe_max_tick")

    def _sync_active_neutral_setpoints(self) -> None:
        summary = self.servo_service.get_calibration_summary()
        if not summary.exists or not summary.compatible:
            self.state.neutral_setpoints = {}
            return
        active_setpoints: dict[int, int] = {}
        for servo_id, entry in summary.servo_entries.items():
            if (
                int(servo_id) in self.settings.robot.servo_ids
                and entry.valid
                and entry.neutral_setpoint is not None
            ):
                active_setpoints[int(servo_id)] = int(entry.neutral_setpoint)
        self.state.neutral_setpoints = active_setpoints

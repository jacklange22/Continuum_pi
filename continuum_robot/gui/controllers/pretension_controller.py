"""Dedicated single-servo pretension controller."""

from __future__ import annotations

from dataclasses import dataclass, field
import threading
import time

from continuum_robot.servos.servo_service import (
    PRETENSION_START_MODE_CURRENT_POSITION,
    PretensionBaselineMeasurement,
    PretensionOperationError,
    PretensionParameters,
    PretensionRoutineResult,
    ServoBusBusyError,
    ServoJogResult,
    ServoMotionAssessment,
)


@dataclass
class PretensionViewState:
    """UI-facing state for the dedicated pretension workspace."""

    connected: bool = False
    robot_mode: str = ""
    expected_servo_ids: list[int] = field(default_factory=list)
    selected_servo_id: int | None = None
    servo_rows: list[dict] = field(default_factory=list)
    telemetry_freshness_threshold_s: float = 0.25
    selected_servo_torque_enabled: bool | None = None
    selected_servo_motion_ready: bool = False
    selected_servo_pretension_ready: bool = False
    selected_servo_position_tick: int | None = None
    selected_servo_current_ma: int | None = None
    selected_servo_current_validity: str = "unknown"
    selected_servo_filtered_current_ma: float | None = None
    selected_servo_filtered_current_source: str = "none"
    selected_servo_voltage_mv: int | None = None
    selected_servo_temperature_c: int | None = None
    selected_servo_hardware_error_text: str = "—"
    selected_servo_telemetry_age_s: float | None = None
    selected_servo_telemetry_fresh: bool | None = None
    selected_servo_untensioned_reference_tick: int | None = None
    selected_servo_effective_min_target_tick: int | None = None
    selected_servo_effective_max_target_tick: int | None = None
    selected_servo_safe_min_tick: int | None = None
    selected_servo_safe_max_tick: int | None = None
    selected_servo_tightening_rotation: str | None = None
    selected_servo_direction_summary: str = "tighten lowers counts"
    selected_servo_block_reason: str = "Select a servo to begin."
    selected_servo_arming_required: bool = False
    selected_servo_saved_summary: str = "No pretension result saved yet."
    comparison_rows: list[dict[str, str]] = field(default_factory=list)
    calibration_path: str = ""
    default_untensioned_reference_tick: int = 4095
    default_start_mode: str = PRETENSION_START_MODE_CURRENT_POSITION
    default_step_ticks: int = 2
    default_settle_time_s: float = 0.05
    default_baseline_sample_count: int = 5
    default_filter_window: int = 3
    default_current_delta_threshold_ma: int = 60
    default_absolute_trigger_current_ma: int | None = 220
    default_hard_current_stop_ma: int = 850
    default_max_travel_ticks: int = 320
    default_timeout_s: float = 10.0
    default_min_offset_ticks: int = -600
    default_max_offset_ticks: int = 600
    baseline_current_ma: float | None = None
    baseline_filtered_current_ma: float | None = None
    baseline_samples_label: str = "Not measured."
    run_state: str = "idle"
    run_state_label: str = "Idle"
    run_state_message: str = "Pretension not started."
    pretension_running: bool = False
    can_measure_baseline: bool = False
    can_move_to_reference: bool = False
    can_start: bool = False
    can_stop: bool = False
    can_save: bool = False
    start_position_tick: int | None = None
    last_commanded_target_tick: int | None = None
    steps_taken: int = 0
    elapsed_s: float = 0.0
    final_position_tick: int | None = None
    stop_reason: str = ""
    failure_phase: str = ""
    failure_primary_reason: str = ""
    failure_detail: str = ""
    last_result_status: str = "none"
    last_error: str | None = None
    status_message: str = "Ready."
    log_text: str = "Pretension workspace idle."


class PretensionController:
    """Owns the dedicated selected-servo pretension operator workflow."""

    def __init__(self, *, servo_service, settings, config_loader=None) -> None:
        self.config_loader = config_loader
        self.servo_service = servo_service
        self.settings = settings
        operating_context = settings.robot.operating_context()
        expected_servo_ids = [int(value) for value in operating_context.expected_servo_ids]
        default_servo_id = expected_servo_ids[0] if expected_servo_ids else None
        self.state = PretensionViewState(
            connected=servo_service.is_connected,
            robot_mode=operating_context.operating_mode,
            expected_servo_ids=list(expected_servo_ids),
            selected_servo_id=int(default_servo_id) if default_servo_id is not None else None,
            telemetry_freshness_threshold_s=servo_service.telemetry_freshness_threshold_s(),
            calibration_path=str(servo_service.neutral_calibration.path),
            default_untensioned_reference_tick=int(settings.safety.pretension_untensioned_reference_tick),
            default_start_mode=str(getattr(settings.safety, "pretension_start_mode", PRETENSION_START_MODE_CURRENT_POSITION)),
            default_step_ticks=int(settings.safety.pretension_step_ticks),
            default_settle_time_s=float(settings.safety.pretension_settle_time_s),
            default_baseline_sample_count=int(settings.safety.pretension_baseline_sample_count),
            default_filter_window=int(settings.safety.pretension_current_filter_window),
            default_current_delta_threshold_ma=int(settings.safety.pretension_current_delta_threshold_ma),
            default_absolute_trigger_current_ma=settings.safety.pretension_absolute_trigger_current_ma,
            default_hard_current_stop_ma=int(settings.safety.pretension_hard_current_stop_ma),
            default_max_travel_ticks=int(settings.safety.pretension_max_travel_ticks),
            default_timeout_s=float(settings.safety.pretension_timeout_s),
            default_min_offset_ticks=int(settings.safety.position_min_offset_ticks),
            default_max_offset_ticks=int(settings.safety.position_max_offset_ticks),
            status_message="Select one servo and verify live readiness before pretensioning.",
            log_text="Pretension workspace ready.\nUse this tab to pretension one selected servo while the full bus stays connected.",
        )
        self._pretension_thread: threading.Thread | None = None
        self._pretension_stop = threading.Event()
        self._last_result: PretensionRoutineResult | None = None
        self._selection_changed = True
        self.latest_runtime_snapshot = None
        self.refresh()

    def refresh(self) -> PretensionViewState:
        self.state.connected = self.servo_service.is_connected
        operating_context = self.settings.robot.operating_context()
        self.state.robot_mode = operating_context.operating_mode
        self.state.expected_servo_ids = [int(value) for value in operating_context.expected_servo_ids]
        self.state.telemetry_freshness_threshold_s = self.servo_service.telemetry_freshness_threshold_s()
        if self.state.selected_servo_id not in self.state.expected_servo_ids and self.state.expected_servo_ids:
            self.state.selected_servo_id = int(self.state.expected_servo_ids[0])
            self._selection_changed = True
        if not self.state.connected:
            self.latest_runtime_snapshot = None
            self.state.servo_rows = [
                {
                    "servo_id": int(servo_id),
                    "selected": int(servo_id) == int(self.state.selected_servo_id or -1),
                    "position": None,
                    "current_ma": None,
                    "pretension_ready": False,
                    "status": "Disconnected",
                }
                for servo_id in self.state.expected_servo_ids
            ]
            self._sync_selected_from_disconnected_state()
            return self.state
        if self.state.pretension_running and self.servo_service.has_exclusive_bus_owner():
            self.state.selected_servo_block_reason = (
                "Pretension run owns the DYNAMIXEL bus; background refresh is paused until the run ends."
            )
            self.state.can_measure_baseline = False
            self.state.can_move_to_reference = False
            self.state.can_start = False
            self.state.can_stop = True
            self.state.can_save = False
            self.state.comparison_rows = self._comparison_rows()
            return self.state

        try:
            snapshot = self.servo_service.build_runtime_servo_snapshot(
                list(self.state.expected_servo_ids),
                selected_servo_id=self.state.selected_servo_id,
                selected_pretension_parameters=self._current_parameters(),
            )
            self.latest_runtime_snapshot = snapshot
        except ServoBusBusyError as exc:
            self.latest_runtime_snapshot = None
            self.state.selected_servo_block_reason = str(exc)
            self.state.status_message = (
                self.state.status_message if self.state.pretension_running else str(exc)
            )
            self.state.last_error = None if self.state.pretension_running else str(exc)
            self.state.can_measure_baseline = False
            self.state.can_move_to_reference = False
            self.state.can_start = False
            self.state.can_stop = bool(self.state.pretension_running)
            self.state.can_save = False
            self.state.comparison_rows = self._comparison_rows()
            return self.state
        except Exception as exc:
            self.latest_runtime_snapshot = None
            self.state.last_error = str(exc)
            self.state.status_message = f"Pretension telemetry refresh failed: {exc}"
            self.state.servo_rows = [
                {
                    "servo_id": int(servo_id),
                    "selected": int(servo_id) == int(self.state.selected_servo_id or -1),
                    "position": None,
                    "current_ma": None,
                    "pretension_ready": False,
                    "status": "Telemetry failed",
                }
                for servo_id in self.state.expected_servo_ids
            ]
            self._sync_selected_from_disconnected_state()
            return self.state
        self.state.servo_rows = []
        selected_motion: ServoMotionAssessment | None = None
        selected_pretension: ServoMotionAssessment | None = None
        selected_telemetry = None
        for servo_id in snapshot.expected_servo_ids:
            entry = snapshot.entries.get(int(servo_id))
            telemetry = entry.telemetry if entry is not None else None
            motion = entry.motion_assessment if entry is not None else None
            pretension = entry.pretension_assessment if entry is not None else None
            if telemetry is None:
                self.state.servo_rows.append(
                    {
                        "servo_id": int(servo_id),
                        "selected": int(servo_id) == int(self.state.selected_servo_id or -1),
                        "position": None,
                        "current_ma": None,
                        "pretension_ready": False,
                        "status": "Telemetry unavailable",
                    }
                )
                continue
            if int(servo_id) == int(self.state.selected_servo_id or -1):
                selected_motion = motion
                selected_pretension = pretension
                selected_telemetry = telemetry
            self.state.servo_rows.append(
                {
                    "servo_id": int(servo_id),
                    "selected": int(servo_id) == int(self.state.selected_servo_id or -1),
                    "position": telemetry.present_position,
                    "current_ma": telemetry.present_current_ma,
                    "pretension_ready": bool(pretension.ready) if pretension is not None else False,
                    "status": (
                        "Ready"
                        if pretension is not None and pretension.ready
                        else self._first_reason(pretension.blocking_reasons if pretension is not None else ())
                    ),
                }
            )

        self._sync_selected_servo_details(selected_telemetry, selected_motion, selected_pretension)
        return self.state

    def set_selected_servo(self, servo_id: int) -> PretensionViewState:
        if self.state.pretension_running:
            raise RuntimeError("Cannot change the selected servo while pretension is running.")
        if self.state.selected_servo_id != int(servo_id):
            self._last_result = None
            self.state.can_save = False
            self.state.run_state = "idle"
            self.state.run_state_label = "Idle"
            self.state.run_state_message = "Pretension not started."
            self.state.start_position_tick = None
            self.state.last_commanded_target_tick = None
            self.state.steps_taken = 0
            self.state.elapsed_s = 0.0
            self.state.final_position_tick = None
            self.state.stop_reason = ""
            self._clear_failure_details()
        self.state.selected_servo_id = int(servo_id)
        self._selection_changed = True
        return self.refresh()

    def refresh_selected_servo(self) -> PretensionViewState:
        return self.refresh()

    def measure_baseline(
        self,
        *,
        sample_count: int,
        filter_window: int,
    ) -> PretensionBaselineMeasurement:
        servo_id = self._require_selected_servo_id()
        try:
            baseline = self.servo_service.measure_pretension_baseline(
                servo_id=int(servo_id),
                sample_count=int(sample_count),
                filter_window=int(filter_window),
                parameters=self._current_parameters(),
            )
        except PretensionOperationError as exc:
            self._apply_failure_details(
                phase=exc.phase,
                primary_reason=exc.primary_reason,
                detail=exc.detail_reason,
            )
            raise RuntimeError(self._format_failure_message(
                phase=exc.phase,
                primary_reason=exc.primary_reason,
                detail=exc.detail_reason,
            )) from exc
        self.state.baseline_current_ma = float(baseline.baseline_current_ma)
        self.state.baseline_filtered_current_ma = float(baseline.filtered_current_ma)
        self.state.baseline_samples_label = (
            f"{baseline.sample_count} sample(s): mean {baseline.baseline_current_ma:.1f} mA, "
            f"filtered {baseline.filtered_current_ma:.1f} mA."
        )
        self.state.status_message = baseline.message
        self.state.run_state = "ready"
        self.state.run_state_label = "Ready"
        self.state.run_state_message = baseline.message
        self.state.last_error = None
        self._clear_failure_details()
        self._append_log(
            f"Baseline refreshed for servo {servo_id}: {baseline.sample_count} samples, "
            f"mean {baseline.baseline_current_ma:.1f} mA, filtered {baseline.filtered_current_ma:.1f} mA."
        )
        self.refresh()
        return baseline

    def move_to_untensioned_reference(self, *, reference_tick: int) -> ServoJogResult:
        servo_id = self._require_selected_servo_id()
        result = self.servo_service.move_servo_to_pretension_reference(
            servo_id=int(servo_id),
            parameters=self._current_parameters(reference_tick=int(reference_tick)),
        )
        self.state.status_message = result.message
        self.state.last_error = None if result.success else result.message
        if result.success:
            self._clear_failure_details()
        self._append_log(result.message)
        self.refresh()
        if result.blocked:
            raise RuntimeError(result.message)
        return result

    def save_startup_calibration(
        self,
        *,
        min_offset_ticks: int,
        max_offset_ticks: int,
        threshold_ma: int,
    ):
        servo_id = self._require_selected_servo_id()
        entry = self.servo_service.save_startup_calibration(
            servo_id=int(servo_id),
            min_offset_ticks=int(min_offset_ticks),
            max_offset_ticks=int(max_offset_ticks),
            pretension_current_threshold_ma=int(threshold_ma),
        )
        self.state.status_message = (
            f"Saved startup calibration for servo {servo_id}: neutral {entry.neutral_setpoint}, "
            f"bounds {entry.safe_min_tick}..{entry.safe_max_tick}, threshold {entry.pretension_current_threshold_ma} mA."
        )
        self.state.last_error = None
        self._append_log(self.state.status_message)
        self.refresh()
        return entry

    def apply_live_parameters(self, *, parameters: PretensionParameters) -> PretensionParameters:
        applied = self.servo_service.apply_live_pretension_defaults(parameters)
        self.settings.safety.pretension_untensioned_reference_tick = int(applied.untensioned_reference_tick)
        self.settings.safety.pretension_start_mode = str(applied.start_mode)
        self.settings.safety.pretension_step_ticks = int(applied.step_ticks)
        self.settings.safety.pretension_settle_time_s = float(applied.settle_time_s)
        self.settings.safety.pretension_baseline_sample_count = int(applied.baseline_sample_count)
        self.settings.safety.pretension_current_filter_window = int(applied.current_filter_window)
        self.settings.safety.pretension_current_delta_threshold_ma = int(applied.current_delta_threshold_ma)
        self.settings.safety.pretension_absolute_trigger_current_ma = (
            None if applied.absolute_trigger_current_ma is None else int(applied.absolute_trigger_current_ma)
        )
        self.settings.safety.pretension_hard_current_stop_ma = int(applied.hard_current_stop_ma)
        self.settings.safety.pretension_max_travel_ticks = int(applied.max_travel_ticks)
        self.settings.safety.pretension_timeout_s = float(applied.timeout_s)
        self.state.default_untensioned_reference_tick = int(applied.untensioned_reference_tick)
        self.state.default_start_mode = str(applied.start_mode)
        self.state.default_step_ticks = int(applied.step_ticks)
        self.state.default_settle_time_s = float(applied.settle_time_s)
        self.state.default_baseline_sample_count = int(applied.baseline_sample_count)
        self.state.default_filter_window = int(applied.current_filter_window)
        self.state.default_current_delta_threshold_ma = int(applied.current_delta_threshold_ma)
        self.state.default_absolute_trigger_current_ma = (
            None if applied.absolute_trigger_current_ma is None else int(applied.absolute_trigger_current_ma)
        )
        self.state.default_hard_current_stop_ma = int(applied.hard_current_stop_ma)
        self.state.default_max_travel_ticks = int(applied.max_travel_ticks)
        self.state.default_timeout_s = float(applied.timeout_s)
        self._selection_changed = True
        status_message = "Applied pretension tuning parameters live. Hardware reconnect is not required."
        self.state.last_error = None
        self._append_log(
            "Applied pretension parameters live: "
            f"ref={applied.untensioned_reference_tick}, mode={applied.start_mode}, step={applied.step_ticks}, "
            f"delta={applied.current_delta_threshold_ma} mA, hard stop={applied.hard_current_stop_ma} mA."
        )
        self.refresh()
        self.state.status_message = status_message
        self.state.last_error = None
        return applied

    def save_pretension_defaults(self, *, parameters: PretensionParameters) -> str:
        if self.config_loader is None:
            raise RuntimeError("Config loader is unavailable; pretension defaults cannot be saved from the GUI.")
        applied = self.apply_live_parameters(parameters=parameters)
        path = self.config_loader.save_system_local_overrides(
            {
                "safety_overrides": {
                    "pretension_untensioned_reference_tick": int(applied.untensioned_reference_tick),
                    "pretension_start_mode": str(applied.start_mode),
                    "pretension_step_ticks": int(applied.step_ticks),
                    "pretension_settle_time_s": float(applied.settle_time_s),
                    "pretension_baseline_sample_count": int(applied.baseline_sample_count),
                    "pretension_current_filter_window": int(applied.current_filter_window),
                    "pretension_current_delta_threshold_ma": int(applied.current_delta_threshold_ma),
                    "pretension_absolute_trigger_current_ma": applied.absolute_trigger_current_ma,
                    "pretension_hard_current_stop_ma": int(applied.hard_current_stop_ma),
                    "pretension_max_travel_ticks": int(applied.max_travel_ticks),
                    "pretension_timeout_s": float(applied.timeout_s),
                }
            }
        )
        status_message = f"Saved pretension defaults to {path} and applied them live. Hardware reconnect is not required."
        self.state.last_error = None
        self._append_log(f"Saved pretension defaults to {path}.")
        self.refresh()
        self.state.status_message = status_message
        self.state.last_error = None
        return str(path)

    def start_pretension(self, *, parameters: PretensionParameters) -> None:
        servo_id = self._require_selected_servo_id()
        if self._pretension_thread is not None and self._pretension_thread.is_alive():
            raise RuntimeError("Pretension is already running.")
        self.refresh()
        try:
            readiness = self.servo_service.assess_pretension_readiness(
                int(servo_id),
                parameters=parameters,
            )
        except ServoBusBusyError as exc:
            self.state.run_state = "blocked"
            self.state.run_state_label = "Blocked"
            self.state.run_state_message = f"Pretension blocked: {exc}"
            self.state.status_message = self.state.run_state_message
            self.state.last_error = self.state.run_state_message
            self._append_log(f"Pretension blocked for servo {servo_id}: {exc}")
            return
        if not readiness.ready:
            self.state.run_state = "blocked"
            self.state.run_state_label = "Blocked"
            self._apply_failure_details(
                phase="precheck",
                primary_reason=readiness.primary_reason or readiness.reason,
                detail=readiness.detail_reason,
            )
            self.state.run_state_message = self._format_failure_message(
                phase="precheck",
                primary_reason=readiness.primary_reason or readiness.reason,
                detail=readiness.detail_reason,
            )
            self.state.status_message = self.state.run_state_message
            self.state.last_error = self.state.run_state_message
            self._append_log(f"Pretension blocked for servo {servo_id}: {self.state.run_state_message}")
            return
        self._pretension_stop.clear()
        self._last_result = None
        self.state.pretension_running = True
        self.state.can_save = False
        self.state.run_state = "running"
        self.state.run_state_label = "Running"
        self.state.run_state_message = f"Pretension running for servo {servo_id}."
        self.state.status_message = self.state.run_state_message
        self.state.last_error = None
        self.state.start_position_tick = self.state.selected_servo_position_tick
        self.state.steps_taken = 0
        self.state.elapsed_s = 0.0
        self.state.final_position_tick = None
        self.state.stop_reason = ""
        self._clear_failure_details()
        self._append_log(f"Starting pretension for servo {servo_id}.")

        def _progress(result: PretensionRoutineResult) -> None:
            self.state.start_position_tick = result.start_position_tick
            self.state.last_commanded_target_tick = result.last_commanded_target_tick
            self.state.steps_taken = result.steps_taken
            self.state.elapsed_s = result.elapsed_s
            self.state.baseline_current_ma = result.baseline_current_ma
            self.state.baseline_filtered_current_ma = result.filtered_current_ma
            self.state.selected_servo_filtered_current_ma = result.filtered_current_ma
            self.state.selected_servo_filtered_current_source = (
                "run_filter_proxy" if result.filtered_current_ma is not None else "none"
            )
            if result.current_position_tick is not None:
                self.state.selected_servo_position_tick = int(result.current_position_tick)
            if result.final_current_ma is not None:
                self.state.selected_servo_current_ma = int(result.final_current_ma)
                self.state.selected_servo_current_validity = "valid"
            self.state.selected_servo_telemetry_age_s = 0.0
            self.state.selected_servo_telemetry_fresh = True
            self.state.selected_servo_block_reason = (
                "Pretension run owns the DYNAMIXEL bus; background refresh is paused until the run ends."
            )
            self.state.run_state = str(result.status)
            self.state.run_state_label = self._format_run_state(result.status)
            self.state.run_state_message = result.message
            self.state.status_message = result.message
            self.state.final_position_tick = result.final_position_tick
            self.state.stop_reason = str(result.primary_reason or result.stop_reason or "")
            if result.status in {"running", "baseline_ready", "arming"}:
                self._clear_failure_details()
            else:
                self._apply_result_failure_details(result)
            if result.status in {"running", "baseline_ready", "arming"}:
                return
            self._append_log(
                f"Pretension {result.status} for servo {result.servo_id}: "
                f"final position {result.final_position_tick}, filtered current "
                f"{result.filtered_current_ma if result.filtered_current_ma is not None else '—'} mA, "
                f"reason {result.stop_reason or result.status}."
            )

        def _worker() -> None:
            try:
                result = self.servo_service.run_pretension_routine(
                    servo_id=int(servo_id),
                    parameters=parameters,
                    stop_requested=self._pretension_stop.is_set,
                    progress_callback=_progress,
                )
                self._last_result = result
                self.state.run_state = str(result.status)
                self.state.run_state_label = self._format_run_state(result.status)
                self.state.run_state_message = result.message
                self.state.status_message = result.message
                self.state.last_error = None if result.success else result.message
                self.state.final_position_tick = result.final_position_tick
                self.state.last_result_status = result.status
                self.state.stop_reason = str(result.primary_reason or result.stop_reason or result.status)
                self.state.can_save = bool(result.success)
                if result.success:
                    self._clear_failure_details()
                else:
                    self._apply_result_failure_details(result)
            except Exception as exc:
                self.state.run_state = "fault"
                self.state.run_state_label = "Fault"
                self.state.run_state_message = f"Pretension failed: {exc}"
                self.state.status_message = self.state.run_state_message
                self.state.last_error = str(exc)
                self.state.can_save = False
                self._apply_failure_details(
                    phase="worker",
                    primary_reason="Pretension worker failed.",
                    detail=str(exc),
                )
                self._append_log(self.state.run_state_message)
            finally:
                self.state.pretension_running = False
                self.refresh()

        self._pretension_thread = threading.Thread(
            target=_worker,
            name="selected-servo-pretension",
            daemon=True,
        )
        self._pretension_thread.start()

    def stop_pretension(self) -> None:
        if self._pretension_thread is None or not self._pretension_thread.is_alive():
            self.state.status_message = "Pretension is not running."
            return
        self._pretension_stop.set()
        self.state.status_message = "Pretension stop requested."
        self.state.run_state_message = self.state.status_message
        self._append_log("Pretension stop requested.")

    def save_pretension_result(self) -> None:
        servo_id = self._require_selected_servo_id()
        if not self.state.can_save:
            raise RuntimeError("No successful pretension result is ready to save.")
        self.servo_service.accept_pretension_result(int(servo_id))
        self.state.can_save = False
        self.state.status_message = f"Saved accepted pretension result for servo {servo_id}."
        self.state.run_state = "saved"
        self.state.run_state_label = "Saved"
        self.state.run_state_message = self.state.status_message
        self.state.last_error = None
        self._append_log(self.state.status_message)
        self.refresh()

    def shutdown(self) -> None:
        self.stop_pretension()
        if self._pretension_thread is not None:
            self._pretension_thread.join(timeout=1.0)

    def _sync_selected_from_disconnected_state(self) -> None:
        self.state.selected_servo_torque_enabled = None
        self.state.selected_servo_motion_ready = False
        self.state.selected_servo_pretension_ready = False
        self.state.selected_servo_position_tick = None
        self.state.selected_servo_current_ma = None
        self.state.selected_servo_current_validity = "unknown"
        self.state.selected_servo_filtered_current_ma = None
        self.state.selected_servo_filtered_current_source = "none"
        self.state.selected_servo_voltage_mv = None
        self.state.selected_servo_temperature_c = None
        self.state.selected_servo_hardware_error_text = "Disconnected"
        self.state.selected_servo_telemetry_age_s = None
        self.state.selected_servo_telemetry_fresh = None
        self.state.selected_servo_untensioned_reference_tick = None
        self.state.selected_servo_effective_min_target_tick = None
        self.state.selected_servo_effective_max_target_tick = None
        self.state.selected_servo_safe_min_tick = None
        self.state.selected_servo_safe_max_tick = None
        self.state.selected_servo_block_reason = "OpenRB / DYNAMIXEL bus is disconnected."
        self.state.selected_servo_arming_required = False
        self.state.selected_servo_saved_summary = self._saved_summary_for_selected_servo()
        self.state.comparison_rows = self._comparison_rows()
        self.state.can_measure_baseline = False
        self.state.can_move_to_reference = False
        self.state.can_start = False
        self.state.can_stop = bool(self.state.pretension_running)
        self.state.can_save = False
        if not self.state.pretension_running:
            self.state.status_message = "Connect OpenRB and refresh servo telemetry before pretensioning."

    def _sync_selected_servo_details(
        self,
        telemetry,
        motion_assessment: ServoMotionAssessment | None,
        pretension_assessment: ServoMotionAssessment | None,
    ) -> None:
        self.state.selected_servo_saved_summary = self._saved_summary_for_selected_servo()
        if telemetry is None or self.state.selected_servo_id is None:
            self._sync_selected_from_disconnected_state()
            return
        self.state.selected_servo_torque_enabled = telemetry.torque_enabled
        self.state.selected_servo_motion_ready = bool(motion_assessment.ready if motion_assessment else False)
        self.state.selected_servo_pretension_ready = bool(
            pretension_assessment.ready if pretension_assessment else False
        )
        self.state.selected_servo_arming_required = bool(
            pretension_assessment.torque_arm_required if pretension_assessment else False
        )
        self.state.selected_servo_position_tick = telemetry.present_position
        self.state.selected_servo_current_ma = telemetry.present_current_ma
        self.state.selected_servo_current_validity = self._current_validity_label(telemetry)
        if not self.state.pretension_running:
            if self._last_result is not None and self._last_result.filtered_current_ma is not None:
                self.state.selected_servo_filtered_current_ma = float(self._last_result.filtered_current_ma)
                self.state.selected_servo_filtered_current_source = "run_filter_proxy"
            else:
                self.state.selected_servo_filtered_current_ma = (
                    self.state.baseline_filtered_current_ma
                    if self.state.baseline_filtered_current_ma is not None
                    else (float(telemetry.present_current_ma) if telemetry.present_current_ma is not None else None)
                )
                if self.state.baseline_filtered_current_ma is not None:
                    self.state.selected_servo_filtered_current_source = "baseline_filter_proxy"
                elif telemetry.present_current_ma is not None:
                    self.state.selected_servo_filtered_current_source = "live_raw"
                else:
                    self.state.selected_servo_filtered_current_source = "none"
        self.state.selected_servo_voltage_mv = telemetry.present_voltage_mv
        self.state.selected_servo_temperature_c = telemetry.present_temperature_c
        self.state.selected_servo_hardware_error_text = self._hardware_error_text(
            telemetry.hardware_error_code,
            telemetry.hardware_error,
        )
        self.state.selected_servo_telemetry_age_s = self.servo_service.telemetry_age_s(telemetry)
        self.state.selected_servo_telemetry_fresh = self.servo_service.telemetry_is_fresh(telemetry)
        try:
            window = self.servo_service.pretension_window_for_servo(
                servo_id=int(self.state.selected_servo_id),
                parameters=self._current_parameters(),
                telemetry=telemetry,
            )
        except Exception:
            window = None
        self.state.selected_servo_untensioned_reference_tick = (
            int(window.untensioned_reference_tick) if window is not None else None
        )
        self.state.selected_servo_effective_min_target_tick = (
            int(window.effective_min_target_tick) if window is not None else None
        )
        self.state.selected_servo_effective_max_target_tick = (
            int(window.effective_max_target_tick) if window is not None else None
        )
        self.state.selected_servo_safe_min_tick = (
            pretension_assessment.safe_min_tick if pretension_assessment is not None else None
        )
        self.state.selected_servo_safe_max_tick = (
            pretension_assessment.safe_max_tick if pretension_assessment is not None else None
        )
        tightening_rotation = self.servo_service.get_tightening_direction(int(self.state.selected_servo_id))
        self.state.selected_servo_tightening_rotation = tightening_rotation
        direction_text = tightening_rotation.upper() if tightening_rotation else "unset"
        self.state.selected_servo_direction_summary = (
            f"Tightening rotation: {direction_text}. Raw XC330 counts tighten by lowering the position value."
        )
        self.state.selected_servo_block_reason = (
            (pretension_assessment.reason if pretension_assessment is not None else "Pretension assessment unavailable.")
            if not self.state.selected_servo_pretension_ready
            else (
                pretension_assessment.reason
                if pretension_assessment is not None and pretension_assessment.reason
                else "Ready for selected-servo pretension."
            )
        )
        if self._selection_changed and not self.state.pretension_running:
            defaults = self.servo_service.default_pretension_parameters(int(self.state.selected_servo_id))
            self.state.default_untensioned_reference_tick = int(defaults.untensioned_reference_tick)
            self.state.default_start_mode = str(defaults.start_mode)
            self.state.default_step_ticks = int(defaults.step_ticks)
            self.state.default_settle_time_s = float(defaults.settle_time_s)
            self.state.default_baseline_sample_count = int(defaults.baseline_sample_count)
            self.state.default_filter_window = int(defaults.current_filter_window)
            self.state.default_current_delta_threshold_ma = int(defaults.current_delta_threshold_ma)
            self.state.default_absolute_trigger_current_ma = defaults.absolute_trigger_current_ma
            self.state.default_hard_current_stop_ma = int(defaults.hard_current_stop_ma)
            self.state.default_max_travel_ticks = int(defaults.max_travel_ticks)
            self.state.default_timeout_s = float(defaults.timeout_s)
            self.state.baseline_current_ma = None
            self.state.baseline_filtered_current_ma = None
            self.state.baseline_samples_label = "Not measured."
            self._selection_changed = False
        self.state.comparison_rows = self._comparison_rows()
        self.state.can_measure_baseline = bool(self.state.selected_servo_pretension_ready and not self.state.pretension_running)
        self.state.can_move_to_reference = bool(self.state.selected_servo_motion_ready and not self.state.pretension_running)
        self.state.can_start = bool(self.state.selected_servo_pretension_ready and not self.state.pretension_running)
        self.state.can_stop = bool(self.state.pretension_running)

    def _saved_summary_for_selected_servo(self) -> str:
        servo_id = self.state.selected_servo_id
        if servo_id is None:
            return "No servo selected."
        summary = self.servo_service.get_calibration_summary()
        entry = summary.servo_entries.get(int(servo_id)) if summary.exists else None
        if entry is None:
            return "No saved calibration artifact for this servo."
        if entry.latest_pretension_run:
            record = entry.latest_pretension_run
            status = entry.pretension_result_status or record.get("status") or "saved"
            position = record.get("final_position_tick", entry.pretension_final_position_tick)
            return f"Latest run: {status} @ {position if position is not None else '—'}."
        if entry.pretension_final_position_tick is not None:
            return (
                f"Saved result: {entry.pretension_result_status or 'saved'} @ "
                f"{entry.pretension_final_position_tick}."
            )
        return "No pretension result saved yet."

    def _append_log(self, message: str) -> None:
        stamp = time.strftime("%H:%M:%S", time.localtime())
        line = f"[{stamp}] {message.strip()}"
        existing = self.state.log_text.strip()
        self.state.log_text = f"{existing}\n{line}".strip() if existing else line

    def _require_selected_servo_id(self) -> int:
        if self.state.selected_servo_id is None:
            raise RuntimeError("Select a servo before using the pretension workspace.")
        return int(self.state.selected_servo_id)

    def _current_parameters(self, *, reference_tick: int | None = None) -> PretensionParameters:
        return PretensionParameters(
            untensioned_reference_tick=(
                int(reference_tick)
                if reference_tick is not None
                else int(self.state.default_untensioned_reference_tick)
            ),
            start_mode=str(self.state.default_start_mode or PRETENSION_START_MODE_CURRENT_POSITION),
            step_ticks=int(self.state.default_step_ticks),
            settle_time_s=float(self.state.default_settle_time_s),
            baseline_sample_count=int(self.state.default_baseline_sample_count),
            current_filter_window=int(self.state.default_filter_window),
            current_delta_threshold_ma=int(self.state.default_current_delta_threshold_ma),
            absolute_trigger_current_ma=(
                None
                if self.state.default_absolute_trigger_current_ma in (None, 0)
                else int(self.state.default_absolute_trigger_current_ma)
            ),
            hard_current_stop_ma=int(self.state.default_hard_current_stop_ma),
            max_travel_ticks=int(self.state.default_max_travel_ticks),
            timeout_s=float(self.state.default_timeout_s),
        )

    def _comparison_rows(self) -> list[dict[str, str]]:
        summary = self.servo_service.get_calibration_summary()
        rows: list[dict[str, str]] = []
        for servo_id in self.state.expected_servo_ids:
            entry = summary.servo_entries.get(int(servo_id)) if summary.exists else None
            record = dict(entry.latest_pretension_run or {}) if entry and entry.latest_pretension_run else {}
            rows.append(
                {
                    "servo_id": str(int(servo_id)),
                    "status": str((entry.pretension_result_status if entry else None) or record.get("status") or "—"),
                    "final_position": self._display(
                        record.get("final_position_tick") if record else (entry.pretension_final_position_tick if entry else None)
                    ),
                    "baseline_current": self._display(record.get("baseline_current_ma")),
                    "trigger_current": self._display(record.get("trigger_current_ma")),
                    "travel_used": self._display(record.get("travel_used_ticks")),
                    "reason": str(record.get("stop_reason") or "—"),
                }
            )
        return rows

    @staticmethod
    def _display(value) -> str:
        return "—" if value is None else str(value)

    @staticmethod
    def _format_run_state(status: str) -> str:
        mapping = {
            "idle": "Idle",
            "ready": "Ready",
            "moving_to_reference": "Moving To Reference",
            "arming": "Arming",
            "arming_failed": "Blocked",
            "baseline_ready": "Baseline Ready",
            "baseline_failed": "Blocked",
            "running": "Running",
            "threshold_reached": "Completed",
            "saved": "Saved",
            "timeout": "Stopped",
            "travel_limit": "Stopped",
            "overcurrent": "Fault",
            "invalid_telemetry": "Blocked",
            "canceled": "Stopped",
            "blocked": "Blocked",
            "fault": "Fault",
        }
        return mapping.get(str(status), str(status).replace("_", " ").title())

    @staticmethod
    def _first_reason(reasons) -> str:
        for reason in reasons or ():
            text = str(reason).strip()
            if text:
                return text
        return "ready"

    def _clear_failure_details(self) -> None:
        self.state.failure_phase = ""
        self.state.failure_primary_reason = ""
        self.state.failure_detail = ""

    def _apply_failure_details(self, *, phase: str, primary_reason: str, detail: str | None = None) -> None:
        self.state.failure_phase = str(phase or "").strip()
        self.state.failure_primary_reason = str(primary_reason or "").strip()
        self.state.failure_detail = str(detail or "").strip()

    def _apply_result_failure_details(self, result: PretensionRoutineResult) -> None:
        self._apply_failure_details(
            phase=str(result.failure_phase or result.status or ""),
            primary_reason=str(result.primary_reason or result.stop_reason or result.status or ""),
            detail=str(result.detail_reason or ""),
        )

    @staticmethod
    def _format_failure_message(*, phase: str, primary_reason: str, detail: str | None = None) -> str:
        message = f"Pretension blocked during {phase}: {primary_reason}".strip()
        if detail:
            message += f" Detail: {detail}"
        return message

    @staticmethod
    def _hardware_error_text(code: int | None, error: str | None) -> str:
        if code in (None, 0) and not error:
            return "0"
        if code not in (None, 0) and error:
            return f"0x{int(code):02X} | {error}"
        if code not in (None, 0):
            return f"0x{int(code):02X}"
        return str(error or "—")

    def _current_validity_label(self, telemetry) -> str:
        if telemetry is None:
            return "unknown"
        if telemetry.present_current_ma is None:
            return "missing"
        fresh = self.servo_service.telemetry_is_fresh(telemetry)
        if fresh is False:
            return "stale"
        if telemetry.telemetry_error:
            error_text = str(telemetry.telemetry_error).lower()
            if "present_current" in error_text or "0x7e" in error_text:
                return "invalid"
        return "valid"

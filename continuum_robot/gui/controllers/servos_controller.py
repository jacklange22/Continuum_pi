"""Servos tab controller for calibration, telemetry, and manual motion."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from continuum_robot.config.settings import Settings
from continuum_robot.servos.segment_readiness import evaluate_selected_segment_readiness
from continuum_robot.servos.servo_service import (
    ServoBusBusyError,
    ServoMotionAssessment,
)
from continuum_robot.servos.sign_mapping_check import ServoMappingCheckRepository, configured_axis_mapping


@dataclass
class ServosViewState:
    """UI-facing servo control state."""

    connected: bool = False
    robot_mode: str = ""
    single_servo_mode: bool = False
    expected_servo_ids: list[int] = field(default_factory=list)
    active_segment_key: str = "segment_a"
    active_segment_label: str = "Spine 1"
    active_segment_servo_ids: list[int] = field(default_factory=list)
    active_segment_pairs: dict[str, list[int]] = field(default_factory=dict)
    segment_order: list[str] = field(default_factory=list)
    segment_definitions: dict[str, dict] = field(default_factory=dict)
    segment_readiness_summary: str = ""
    all_8_readiness_summary: str = ""
    single_segment_readiness_summary: str = ""
    selected_servo_segment_label: str = "Unknown segment"
    manual_pretension_sequence_hint: str = ""
    dual_segment_foundation_note: str = ""
    servo_ids: list[int] = field(default_factory=list)
    detected_servo_ids: list[int] = field(default_factory=list)
    missing_servo_ids: list[int] = field(default_factory=list)
    unexpected_servo_ids: list[int] = field(default_factory=list)
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
    telemetry_freshness_threshold_s: float = 0.25
    selected_servo_id: int | None = None
    discovery_status: str = "idle"
    discovery_message: str = "One-servo discovery not run yet."
    position_convention_summary: str = ""
    selected_servo_torque_enabled: bool | None = None
    selected_servo_current_position_tick: int | None = None
    selected_servo_current_ma: int | None = None
    selected_servo_voltage_mv: int | None = None
    selected_servo_temperature_c: int | None = None
    selected_servo_last_target_tick: int | None = None
    selected_servo_last_unclamped_target_tick: int | None = None
    selected_servo_safe_min_tick: int | None = None
    selected_servo_safe_max_tick: int | None = None
    selected_servo_last_motion_clamped: bool = False
    selected_servo_last_motion_summary: str = "No jog command sent yet."
    selected_servo_last_action_label: str = "None"
    selected_servo_last_result_label: str = "None"
    selected_servo_reason_label: str = "none"
    selected_servo_telemetry_status_label: str = "Unknown"
    selected_servo_last_read_label: str = "—"
    selected_servo_telemetry_age_s: float | None = None
    selected_servo_telemetry_fresh: bool | None = None
    selected_servo_motion_ready: bool = False
    blocking_reasons: list[str] = field(default_factory=list)
    selected_servo_external_power_ready: bool | None = None
    # Pretension lives in the Segment Pretension Trial section (driven by
    # PretensionTrialController). The single-servo pretension worker is no
    # longer driven from this controller; the underlying ServoService method
    # is kept for diagnostic / test use only.
    pretension_message: str = "Manual pretension idle."
    pretension_source_summary: str = "No accepted pretension source."
    pretension_source_type: str = "none"
    pretension_source_updated_at_utc: str | None = None
    pretension_source_note: str | None = None
    manual_pretension_can_accept: bool = False
    manual_pretension_can_clear: bool = False
    last_displacement_summary: str = ""
    last_displacement_debug_lines: list[str] = field(default_factory=list)
    single_segment_motion_config_summary: str = ""
    single_segment_reference_summary: str = ""
    single_segment_enforced_bounds_summary: str = ""
    single_segment_characterization_summary: str = ""
    bench_debug_text: str = ""
    status_message: str = "Servo control idle."
    last_error: str | None = None
    configured_sign_mapping_summary: str = ""


class ServosController:
    """Owns manual jog, displacement command, and telemetry display actions."""

    def __init__(self, servo_service, settings: Settings) -> None:
        self.servo_service = servo_service
        self.settings = settings
        self._motion_state_by_servo: dict[int, dict[str, object]] = {}
        self.latest_runtime_snapshot = None
        self._initial_discovery_done = False
        operating_context = settings.robot.operating_context()
        self.state = ServosViewState(
            connected=servo_service.is_connected,
            robot_mode=operating_context.operating_mode,
            single_servo_mode=(operating_context.operating_mode == "one_servo"),
            expected_servo_ids=list(operating_context.expected_servo_ids),
            active_segment_key=settings.robot.active_segment_key(),
            active_segment_label=settings.robot.active_segment_label(),
            active_segment_servo_ids=settings.robot.active_segment_servo_ids(),
            active_segment_pairs=settings.robot.active_segment_pairs(),
            segment_order=list(operating_context.segment_order),
            segment_definitions=dict(operating_context.metadata().get("segments", {})),
            dual_segment_foundation_note=(
                operating_context.mode_notes[0]
                if operating_context.operating_mode == "dual_segment" and operating_context.mode_notes
                else ""
            ),
            servo_ids=list(operating_context.expected_servo_ids),
            tendon_displacements_cm=[0.0] * len(settings.robot.tendon_to_servo),
            fine_jog_step_ticks=settings.safety.fine_jog_step_ticks,
            coarse_jog_step_ticks=settings.safety.coarse_jog_step_ticks,
            default_pretension_threshold_ma=settings.safety.default_pretension_current_threshold_ma,
            telemetry_freshness_threshold_s=settings.safety.telemetry_stale_after_s,
            selected_servo_id=(int(operating_context.selected_servo_id or operating_context.expected_servo_ids[0]) if operating_context.expected_servo_ids else None),
            position_convention_summary=servo_service.position_convention_summary(),
            status_message=(
                "Mock servo backend ready."
                if settings.runtime.mock_mode
                else (
                    "Hardware servo mode ready. Connect OpenRB, refresh readiness, verify the configured servos, "
                    "then jog one selected servo at a time."
                )
            ),
        )
        self.load_neutral_setpoints()
        self.refresh()

    def refresh(self) -> ServosViewState:
        self.state.connected = self.servo_service.is_connected
        operating_context = self.settings.robot.operating_context()
        self._sync_operating_context_fields(operating_context)
        self.state.single_servo_mode = operating_context.operating_mode == "one_servo"
        self._refresh_single_segment_motion_diagnostics()
        self._refresh_calibration_summary()
        self._sync_active_neutral_setpoints()
        if not self.state.connected:
            self.state.telemetry = {}
            self.state.detected_servo_ids = []
            self.state.missing_servo_ids = list(self.state.expected_servo_ids)
            self.state.unexpected_servo_ids = []
            self.state.selected_servo_external_power_ready = None
            self.state.bench_debug_text = self._build_disconnected_bench_debug_text()
            if self.state.discovery_status == "idle":
                self.state.discovery_message = "Connect OpenRB and the DYNAMIXEL bus before discovery."
            self._initial_discovery_done = False
            self._sync_selected_servo_motion_state()
            self._sync_segment_readiness_summary()
            return self.state
        if not self._initial_discovery_done:
            self._initial_discovery_done = True
            try:
                if self.state.single_servo_mode:
                    discovery = self.servo_service.discover_one_servo(
                        expected_servo_id=self._expected_servo_id(),
                        allow_scan=True,
                    )
                    self._apply_discovery_snapshot(discovery)
                elif self.state.expected_servo_ids:
                    snapshot = self.servo_service.build_configured_servo_bringup_snapshot(
                        list(self.state.expected_servo_ids),
                        allow_scan=True,
                    )
                    self._apply_configured_servo_snapshot(snapshot)
            except Exception as exc:
                self.state.last_error = str(exc)
                self.state.status_message = f"Auto-discovery failed: {exc}"
        if self.state.single_servo_mode:
            return self._refresh_single_servo_state()
        self.state.servo_ids = list(self.state.expected_servo_ids)
        if not self.state.servo_ids:
            self.state.telemetry = {}
            self.state.blocking_reasons = []
            self.state.selected_servo_external_power_ready = None
            self.state.bench_debug_text = self._build_disconnected_bench_debug_text()
            self._sync_selected_servo_motion_state()
            self._sync_segment_readiness_summary()
            return self.state
        if self.state.servo_ids:
            try:
                assessments = self._refresh_live_telemetry_rows(self.state.servo_ids, replace=True)
            except ServoBusBusyError as exc:
                self.state.status_message = str(exc)
                self.state.last_error = None
                self.state.bench_debug_text = self._build_multi_servo_bench_debug_text()
                self._sync_selected_servo_motion_state()
                self._sync_segment_readiness_summary()
                return self.state
            except Exception as exc:
                self.state.last_error = str(exc)
                self.state.status_message = f"Telemetry refresh failed: {exc}"
                self.state.telemetry = {}
                self.state.bench_debug_text = self._build_disconnected_bench_debug_text(extra_error=str(exc))
                self._sync_selected_servo_motion_state()
                self._sync_segment_readiness_summary()
                return self.state
            if self.state.selected_servo_id not in self.state.servo_ids and self.state.servo_ids:
                self.state.selected_servo_id = int(self.state.servo_ids[0])
            selected = assessments.get(int(self.state.selected_servo_id)) if self.state.selected_servo_id is not None else None
            if selected is not None:
                self.state.blocking_reasons = list(selected.blocking_reasons)
                self.state.selected_servo_external_power_ready = selected.external_power_ready
            self.state.last_error = None
            self._refresh_single_segment_motion_diagnostics()
            self.state.bench_debug_text = self._build_multi_servo_bench_debug_text()
            self._sync_selected_servo_motion_state()
            self._sync_segment_readiness_summary()
        return self.state

    def scan(self) -> list[int]:
        try:
            if self.state.single_servo_mode:
                discovery = self.servo_service.discover_one_servo(
                    expected_servo_id=self._expected_servo_id(),
                    allow_scan=True,
                )
                self._apply_discovery_snapshot(discovery)
                result_ids = self.state.servo_ids
            else:
                snapshot = self.servo_service.build_configured_servo_bringup_snapshot(
                    list(self.state.expected_servo_ids),
                    allow_scan=True,
                )
                self._apply_configured_servo_snapshot(snapshot)
                result_ids = list(snapshot.discovered_ids)
            self.state.last_error = None
            return result_ids
        except Exception as exc:
            self.state.last_error = str(exc)
            self.state.status_message = f"Servo scan failed: {exc}"
            raise
        finally:
            self.refresh()

    def refresh_readiness(self) -> ServosViewState:
        if self.state.single_servo_mode:
            discovery = self.servo_service.discover_one_servo(
                expected_servo_id=self._expected_servo_id(),
                allow_scan=False,
            )
            self._apply_discovery_snapshot(discovery)
        else:
            snapshot = self.servo_service.build_configured_servo_bringup_snapshot(
                list(self.state.expected_servo_ids),
                allow_scan=False,
            )
            self._apply_configured_servo_snapshot(snapshot)
        return self.refresh()

    def create_sign_mapping_checklist(self, *, operator: str = "", notes: str = "") -> str:
        if self.state.robot_mode != "single_segment":
            raise RuntimeError("Sign/mapping checklist is currently scoped to single_segment bring-up.")
        repo = ServoMappingCheckRepository(self._project_root_for_runtime_artifacts() / "data" / "calibration" / "servo_mapping_checks")
        record = repo.create_checklist(
            operating_mode=self.state.robot_mode,
            active_segment_key=self.state.active_segment_key,
            active_segment_label=self.state.active_segment_label,
            expected_servo_ids=[int(value) for value in self.state.active_segment_servo_ids or self.state.expected_servo_ids],
            operator=operator,
            mock_mode=bool(self.settings.runtime.mock_mode),
            notes=notes,
        )
        self.state.status_message = f"Created servo sign/mapping checklist: {record.artifact_path}"
        self.state.last_error = None
        self.refresh()
        return str(record.artifact_path)

    def confirm_configured_sign_mapping(self, *, operator: str = "", notes: str = "") -> str:
        if self.state.robot_mode != "single_segment":
            raise RuntimeError("Configured sign/mapping confirmation is scoped to single_segment bring-up.")
        active_ids = [int(value) for value in self.state.active_segment_servo_ids or self.state.expected_servo_ids]
        repo = ServoMappingCheckRepository(self._project_root_for_runtime_artifacts() / "data" / "calibration" / "servo_mapping_checks")
        record = repo.confirm_configured_mapping(
            operating_mode=self.state.robot_mode,
            active_segment_key=self.state.active_segment_key,
            active_segment_label=self.state.active_segment_label,
            expected_servo_ids=active_ids,
            active_segment_pairs=self.state.active_segment_pairs,
            operator=operator,
            mock_mode=bool(self.settings.runtime.mock_mode),
            notes=notes,
        )
        self.state.status_message = f"Confirmed configured servo sign/mapping: {record.artifact_path}"
        self.state.last_error = None
        self.refresh()
        return str(record.artifact_path)

    def assign_servo_id(self, current_id: int, new_id: int) -> None:
        result = self.servo_service.assign_servo_id_safely(current_id, new_id)
        self.state.status_message = result.message
        self.state.discovery_status = result.status
        if result.success:
            discovered_ids = self.servo_service.scan_ids(
                min_id=int(self.servo_service.dxl_bus.config.discovery_min_id),
                max_id=int(self.servo_service.dxl_bus.config.discovery_max_id),
            )
            self.state.servo_ids = list(discovered_ids)
            self.state.detected_servo_ids = list(discovered_ids)
            self.state.missing_servo_ids = []
            self.state.unexpected_servo_ids = []
            self.state.selected_servo_id = int(new_id)
            self.state.last_error = None
            self.state.discovery_message = (
                f"{result.message} Re-scan configured servos before returning to normal robot bring-up."
            )
            self.state.status_message = self.state.discovery_message
            self.state.telemetry = {}
            self._sync_selected_servo_motion_state()
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
                motion_workflow="experiment_motion",
            )
            self.state.status_message = result.message
            self.state.last_displacement_summary = result.message
            self.state.last_displacement_debug_lines = self._format_displacement_debug_lines(result)
            self.state.last_error = None
        except Exception as exc:
            self.state.last_error = str(exc)
            self.state.status_message = f"Displacement rejected: {exc}"
            self.state.last_displacement_summary = self.state.status_message
            self.state.last_displacement_debug_lines = []
            raise
        finally:
            self.refresh()

    def _format_displacement_debug_lines(self, result) -> list[str]:
        requested = list(getattr(result, "requested_displacements_cm", []) or [])
        resolved = list(getattr(result, "resolved_displacements_cm", []) or [])
        debug_entries = dict(getattr(result, "debug_entries_by_id", {}) or {})
        lines: list[str] = []
        if requested:
            lines.append("Requested displacement (cm): " + ", ".join(f"{value:.3f}" for value in requested))
        if resolved:
            lines.append("Resolved displacement (cm): " + ", ".join(f"{value:.3f}" for value in resolved))
        for servo_id in sorted(debug_entries):
            entry = debug_entries[int(servo_id)]
            mode_text = (
                self.servo_service.operating_mode_label(entry.operating_mode)
                if entry.operating_mode is not None
                else "—"
            )
            preferred_text = (
                self.servo_service.operating_mode_label(entry.preferred_operating_mode)
                if entry.preferred_operating_mode is not None
                else "—"
            )
            lines.append(
                "Servo "
                f"{int(servo_id)}: pos {entry.present_position_tick if entry.present_position_tick is not None else '—'}, "
                f"current {entry.present_current_ma if entry.present_current_ma is not None else '—'} mA, "
                f"goal {entry.raw_goal_tick if entry.raw_goal_tick is not None else '—'}, "
                f"bounds [{entry.safe_min_tick if entry.safe_min_tick is not None else '—'}, "
                f"{entry.safe_max_tick if entry.safe_max_tick is not None else '—'}], "
                f"mode {mode_text}->{preferred_text}, "
                f"goal current {entry.goal_current_ma if entry.goal_current_ma is not None else '—'} mA, "
                f"profile {entry.profile_velocity if entry.profile_velocity is not None else '—'}/"
                f"{entry.profile_acceleration if entry.profile_acceleration is not None else '—'}, "
                f"limit {entry.limit_source}"
            )
        return lines

    def capture_neutral_setpoints(self) -> dict[int, int]:
        try:
            servo_ids = self._capture_servo_ids()
            result = self.servo_service.capture_and_save_neutral_setpoints(servo_ids)
            setpoints = result.setpoints_by_id
            self.state.neutral_setpoints = setpoints
            captured_ids = sorted(int(servo_id) for servo_id in result.servo_ids)
            self.state.status_message = (
                f"Captured neutral metadata for servo IDs {captured_ids} and saved it to {result.artifact_path}. "
                "Bench jogging still uses the raw 0..4095 range as the active motion range."
            )
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
                "Saved servo calibration is incompatible with the active servo configuration. "
                "The artifact is shown for review only and is not being used as the active bring-up motion range."
            )
            return {}
        self.state.status_message = (
            "Loaded servo calibration artifact. Bring-up jog still uses raw 0..4095 counts as the active range."
        )
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

    # The single-servo "Path A" pretension routine (run_pretension_routine on
    # ServoService) is no longer driven from the Servos tab. Pretension lives
    # in one place: the staged 4-servo experiment, launched from the Servos
    # tab "Segment Pretension Trial" section (PretensionTrialController) or
    # tuned via the Experiments tab pretension_validation page. The underlying
    # ServoService method is kept for diagnostic / test use.

    def capture_manual_pretension(self, note: str = "") -> None:
        try:
            entries = self.servo_service.capture_manual_pretension_state(note=note)
            captured_ids = sorted(int(servo_id) for servo_id in entries)
            self.state.status_message = (
                f"Captured current {len(captured_ids)}-servo state as pending manual pretension/startup for servo IDs "
                + ", ".join(str(value) for value in captured_ids)
                + ". Review and accept it before running experiments."
            )
            self.state.last_error = None
        except Exception as exc:
            self.state.last_error = str(exc)
            self.state.status_message = f"Manual pretension capture failed: {exc}"
            raise
        finally:
            self.refresh()

    def accept_manual_pretension(self) -> None:
        try:
            summary = self.servo_service.accept_manual_pretension_state()
            self.state.status_message = summary.message
            self.state.last_error = None
        except Exception as exc:
            self.state.last_error = str(exc)
            self.state.status_message = f"Accept manual pretension failed: {exc}"
            raise
        finally:
            self.refresh()

    def clear_manual_pretension(self) -> None:
        try:
            cleared = self.servo_service.clear_manual_pretension_state()
            if cleared:
                self.state.status_message = (
                    "Cleared saved manual pretension state for servo IDs "
                    + ", ".join(str(value) for value in sorted(cleared))
                    + "."
                )
            else:
                self.state.status_message = "No saved manual pretension state was present to clear."
            self.state.last_error = None
        except Exception as exc:
            self.state.last_error = str(exc)
            self.state.status_message = f"Clear manual pretension failed: {exc}"
            raise
        finally:
            self.refresh()

    def shutdown(self) -> None:
        """Servos controller shutdown is a no-op now that the single-servo
        pretension worker has been removed from this controller. The Segment
        Pretension Trial path runs synchronously inside the ExperimentRunner."""
        return None

    def _expected_servo_id(self) -> int | None:
        if self.state.selected_servo_id is not None:
            return int(self.state.selected_servo_id)
        expected = self.settings.robot.expected_servo_ids()
        if expected:
            return int(expected[0])
        return None

    def _capture_servo_ids(self) -> list[int]:
        if self.state.single_servo_mode:
            servo_id = self.state.selected_servo_id or self._expected_servo_id()
            if servo_id is None:
                raise RuntimeError("No servo is selected for one-servo neutral capture.")
            return [int(servo_id)]
        if self.state.detected_servo_ids:
            return [int(servo_id) for servo_id in self.state.detected_servo_ids]
        return [int(servo_id) for servo_id in self.state.servo_ids]

    def set_selected_servo(self, servo_id: int) -> ServosViewState:
        self.state.selected_servo_id = int(servo_id)
        self._refresh_selected_servo_live()
        return self.state

    def refresh_selected_servo(self) -> ServosViewState:
        self.state.connected = self.servo_service.is_connected
        operating_context = self.settings.robot.operating_context()
        self._sync_operating_context_fields(operating_context)
        self.state.single_servo_mode = operating_context.operating_mode == "one_servo"
        self._refresh_calibration_summary()
        self._sync_active_neutral_setpoints()
        if not self.state.connected:
            self.state.telemetry = {}
            self.state.detected_servo_ids = []
            self.state.missing_servo_ids = list(self.state.expected_servo_ids)
            self.state.unexpected_servo_ids = []
            self.state.selected_servo_external_power_ready = None
            self.state.bench_debug_text = self._build_disconnected_bench_debug_text()
            self._sync_selected_servo_motion_state()
            self._sync_segment_readiness_summary()
            return self.state
        if self.state.single_servo_mode:
            return self.refresh()
        self.state.servo_ids = list(self.state.expected_servo_ids)
        if self.state.selected_servo_id not in self.state.servo_ids and self.state.servo_ids:
            self.state.selected_servo_id = int(self.state.servo_ids[0])
        self._refresh_selected_servo_live()
        self._sync_segment_readiness_summary()
        return self.state

    def _run_jog_action(self, servo_id: int, action: str) -> None:
        self.state.selected_servo_id = int(servo_id)
        self.state.selected_servo_last_action_label = self._format_action_label(action)
        result = self.servo_service.jog_servo_action(
            servo_id=int(servo_id),
            action=str(action),
        )
        self.state.status_message = result.message
        self.state.last_error = None if result.success else result.message
        self._apply_jog_result(result, action=str(action))
        self._update_selected_row_from_jog_result(result)
        self.state.bench_debug_text = self._build_multi_servo_bench_debug_text()
        self._sync_selected_servo_motion_state()
        if result.blocked:
            raise RuntimeError(result.message)

    def _neutral_ticks_for_current_ids(self) -> list[int]:
        if self.state.calibration_exists and not self.state.calibration_compatible:
            raise RuntimeError(
                "Saved servo calibration does not match the current robot configuration. "
                "Review the calibration summary and recapture neutral before commanding motion."
            )
        reference = self.servo_service.resolve_startup_reference_ticks(list(self.state.servo_ids))
        missing = [sid for sid in self.state.servo_ids if sid not in reference.ticks_by_servo]
        if missing:
            raise RuntimeError(f"Startup reference ticks missing for servo IDs: {missing}")
        return [reference.ticks_by_servo[sid] for sid in self.state.servo_ids]

    def _apply_discovery_snapshot(self, discovery) -> None:
        self.state.discovery_status = discovery.status
        self.state.discovery_message = discovery.message
        self.state.detected_servo_ids = list(discovery.discovered_ids)
        self.state.missing_servo_ids = []
        self.state.unexpected_servo_ids = []
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

    def _apply_configured_servo_snapshot(self, snapshot) -> None:
        self.state.discovery_status = snapshot.status
        self.state.discovery_message = snapshot.message
        self.state.servo_ids = list(snapshot.expected_servo_ids)
        self.state.detected_servo_ids = list(snapshot.discovered_ids)
        self.state.missing_servo_ids = list(snapshot.missing_servo_ids)
        self.state.unexpected_servo_ids = list(snapshot.unexpected_servo_ids)
        if self.state.selected_servo_id not in self.state.servo_ids and self.state.servo_ids:
            self.state.selected_servo_id = int(self.state.servo_ids[0])
        selected_entry = (
            snapshot.servo_entries.get(int(self.state.selected_servo_id))
            if self.state.selected_servo_id is not None
            else None
        )
        if selected_entry is not None and selected_entry.motion_assessment is not None:
            self.state.blocking_reasons = list(selected_entry.motion_assessment.blocking_reasons)
            self.state.selected_servo_external_power_ready = (
                selected_entry.motion_assessment.external_power_ready
            )
        else:
            self.state.blocking_reasons = []
            self.state.selected_servo_external_power_ready = None
        self.state.status_message = snapshot.message
        self.state.last_error = None if snapshot.status == "ready" else snapshot.message

    def _refresh_calibration_summary(self) -> None:
        summary = self.servo_service.get_calibration_summary()
        self.state.calibration_exists = summary.exists
        self.state.calibration_compatible = summary.compatible
        self.state.calibration_status = summary.status
        self.state.calibration_message = summary.message
        self.state.pretension_source_summary = "No accepted pretension source."
        self.state.pretension_source_type = "none"
        self.state.pretension_source_updated_at_utc = None
        self.state.pretension_source_note = None
        self.state.manual_pretension_can_accept = False
        self.state.manual_pretension_can_clear = False
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
        configured_servo_ids = [int(value) for value in self.settings.robot.expected_servo_ids()]
        if not self.state.single_servo_mode and len(configured_servo_ids) in {4, 8} and summary.exists and summary.compatible:
            source_summary = summary.pretension_source_summary(configured_servo_ids)
            self.state.pretension_source_type = source_summary.source_type
            self.state.pretension_source_updated_at_utc = source_summary.updated_at_utc
            self.state.pretension_source_note = source_summary.note
            manual_entries = [
                summary.servo_entries.get(int(servo_id))
                for servo_id in configured_servo_ids
            ]
            manual_notes = [
                str(entry.pretension_note).strip()
                for entry in manual_entries
                if entry is not None and entry.pretension_note not in (None, "")
            ]
            self.state.manual_pretension_can_accept = bool(
                manual_entries
                and all(
                    entry is not None
                    and str(entry.pretension_source or "").strip().lower() == "manual"
                    and entry.pretension_result_status == "manual_captured"
                    for entry in manual_entries
                )
            )
            self.state.manual_pretension_can_clear = any(
                entry is not None
                and str(entry.pretension_source or "").strip().lower() == "manual"
                for entry in manual_entries
            )
            if self.state.pretension_source_note is None and manual_notes and len(set(manual_notes)) == 1:
                self.state.pretension_source_note = manual_notes[0]
            summary_text = source_summary.message
            positions_by_servo = dict(source_summary.positions_by_servo)
            if not positions_by_servo:
                positions_by_servo = {
                    int(servo_id): (
                        int(entry.pretension_final_position_tick)
                        if entry is not None and entry.pretension_final_position_tick is not None
                        else None
                    )
                    for servo_id, entry in zip(configured_servo_ids, manual_entries)
                }
            positions_text = ", ".join(
                f"{servo_id}:{position if position is not None else '—'}"
                for servo_id, position in sorted(positions_by_servo.items())
            )
            if self.state.manual_pretension_can_accept:
                if self.state.robot_mode == "dual_segment":
                    summary_text = (
                        "Pending manual pretension/startup capture is saved for the all-8 dual_segment "
                        f"foundation ({', '.join(str(value) for value in configured_servo_ids)})."
                    )
                else:
                    summary_text = (
                        "Pending manual pretension capture is saved for active segment "
                        f"{self.state.active_segment_label} ({', '.join(str(value) for value in configured_servo_ids)})."
                    )
                self.state.pretension_source_type = "manual_pending"
            elif self.state.manual_pretension_can_clear and not source_summary.usable:
                summary_text = "Manual pretension capture exists, but it is incomplete or not accepted yet."
            if positions_text:
                summary_text = f"{summary_text} Positions {positions_text}."
            updated_at = source_summary.updated_at_utc or summary.updated_at_utc
            self.state.pretension_source_updated_at_utc = updated_at
            if updated_at:
                summary_text = f"{summary_text} Updated {updated_at}."
            self.state.pretension_source_summary = summary_text.strip()
        self.state.calibration_path = summary.path
        self.state.calibration_updated_at_utc = summary.updated_at_utc
        self.state.calibration_rows = []
        for servo_id in sorted(summary.servo_entries):
            entry = summary.servo_entries[servo_id]
            pretension_source = (
                str(entry.pretension_source).strip().lower()
                if entry.pretension_source not in (None, "")
                else str((entry.latest_pretension_run or {}).get("source", "")).strip().lower() or None
            )
            pretension_text = (
                f"{entry.pretension_result_status or '—'} @ {entry.pretension_final_position_tick}"
                if entry.pretension_final_position_tick is not None
                else (entry.pretension_result_status or "—")
            )
            if pretension_source not in (None, ""):
                pretension_text = f"{pretension_text} [{pretension_source}]"
            self.state.calibration_rows.append(
                {
                    "servo_id": str(servo_id),
                    "neutral": str(entry.neutral_setpoint) if entry.neutral_setpoint is not None else "—",
                    "bounds": (
                        f"{entry.safe_min_tick} .. {entry.safe_max_tick} (artifact only)"
                        if entry.safe_min_tick is not None and entry.safe_max_tick is not None
                        else "missing"
                    ),
                    "threshold": (
                        str(entry.pretension_current_threshold_ma)
                        if entry.pretension_current_threshold_ma is not None
                        else "missing"
                    ),
                    "direction": entry.tightening_rotation or "unset",
                    "pretension": pretension_text,
                    "status": "valid" if entry.valid else entry.status,
                }
            )

    def _assessment_bounds_text(self, assessment: ServoMotionAssessment) -> str:
        if assessment.safe_min_tick is None or assessment.safe_max_tick is None:
            return "unavailable"
        return f"{assessment.safe_min_tick} .. {assessment.safe_max_tick}"

    def _assessment_status_text(self, assessment: ServoMotionAssessment) -> str:
        return "ready" if assessment.ready else assessment.reason

    def _refresh_live_telemetry_rows(
        self,
        servo_ids: list[int],
        *,
        replace: bool,
    ) -> dict[int, ServoMotionAssessment]:
        target_ids = (
            list(self.state.expected_servo_ids or servo_ids)
            if replace
            else [int(servo_id) for servo_id in servo_ids]
        )
        ownership = self.servo_service.bus_ownership_status()
        if ownership.active and not ownership.held_by_current_thread:
            snapshot = self.servo_service.build_cached_runtime_servo_snapshot(
                target_ids,
                selected_servo_id=self.state.selected_servo_id,
            )
            self.state.status_message = self.servo_service.bus_busy_message(action="servo telemetry refresh")
        else:
            snapshot = self.servo_service.build_runtime_servo_snapshot(
                target_ids,
                selected_servo_id=self.state.selected_servo_id,
                include_scan=False,
                telemetry_profile="minimal",
            )
        self.latest_runtime_snapshot = snapshot
        existing_rows = dict(self.state.telemetry)
        rows: dict[int, dict] = {} if replace else dict(existing_rows)
        assessments: dict[int, ServoMotionAssessment] = {}
        for servo_id in [int(item) for item in snapshot.expected_servo_ids]:
            entry = snapshot.entries.get(int(servo_id))
            if entry is None or entry.telemetry is None or entry.motion_assessment is None:
                rows[int(servo_id)] = self._missing_telemetry_row(
                    int(servo_id),
                    existing_rows.get(int(servo_id)),
                )
                continue
            assessment = entry.motion_assessment
            assessments[int(servo_id)] = entry.motion_assessment
            row = self._telemetry_row_from_live_item(
                int(servo_id),
                entry.telemetry,
                assessment,
                existing_row=existing_rows.get(int(servo_id)),
                runtime_entry=entry,
            )
            rows[int(servo_id)] = row
        self.state.telemetry = rows
        if replace:
            self.state.detected_servo_ids = list(snapshot.detected_servo_ids)
            self.state.missing_servo_ids = list(snapshot.missing_servo_ids)
            self.state.unexpected_servo_ids = list(snapshot.unexpected_servo_ids)
        return assessments

    def _sync_operating_context_fields(self, operating_context) -> None:
        metadata = operating_context.metadata()
        self.state.robot_mode = operating_context.operating_mode
        self.state.expected_servo_ids = list(operating_context.expected_servo_ids)
        self.state.active_segment_key = self.settings.robot.active_segment_key()
        self.state.active_segment_label = self.settings.robot.active_segment_label()
        self.state.active_segment_servo_ids = self.settings.robot.active_segment_servo_ids()
        self.state.active_segment_pairs = self.settings.robot.active_segment_pairs()
        self.state.segment_order = list(operating_context.segment_order)
        self.state.segment_definitions = dict(metadata.get("segments", {}) or {})
        self.state.dual_segment_foundation_note = (
            operating_context.mode_notes[0]
            if operating_context.operating_mode == "dual_segment" and operating_context.mode_notes
            else ""
        )
        self.state.manual_pretension_sequence_hint = (
            "Manual sequence: pretension Segment A / proximal, pretension Segment B / distal, "
            "re-check Segment A, then save the all-8 startup artifact."
            if operating_context.operating_mode == "dual_segment"
            else ""
        )

    def _sync_segment_readiness_summary(self) -> None:
        summaries = []
        selected_segment_label = "Unknown segment"
        expected = [int(value) for value in self.state.expected_servo_ids]
        detected = {int(value) for value in self.state.detected_servo_ids}
        missing_global = {int(value) for value in self.state.missing_servo_ids}
        if not detected and self.state.telemetry:
            detected = {
                int(servo_id)
                for servo_id, row in self.state.telemetry.items()
                if row.get("position") is not None or row.get("telemetry_status") not in {"Unreadable", "Missing"}
            }
        for key in self.state.segment_order or sorted(self.state.segment_definitions):
            data = dict(self.state.segment_definitions.get(str(key), {}) or {})
            servo_ids = [int(value) for value in data.get("servo_ids", [])]
            relevant_ids = [servo_id for servo_id in servo_ids if servo_id in expected or not expected]
            missing = sorted({servo_id for servo_id in relevant_ids if servo_id in missing_global or servo_id not in detected})
            stale = sorted(
                servo_id
                for servo_id in relevant_ids
                if servo_id in self.state.telemetry and self.state.telemetry[servo_id].get("telemetry_fresh") is False
            )
            errors = sorted(
                servo_id
                for servo_id in relevant_ids
                if servo_id in self.state.telemetry and self.state.telemetry[servo_id].get("error")
            )
            label = str(data.get("segment_label") or data.get("label") or key)
            role = str(data.get("segment_role") or "").strip()
            display = f"{label} / {role}" if role else label
            if self.state.selected_servo_id in relevant_ids:
                selected_segment_label = display
            if not relevant_ids:
                status = "no configured IDs"
            elif missing:
                status = "missing " + ", ".join(str(value) for value in missing)
            elif stale:
                status = "display age warning " + ", ".join(str(value) for value in stale)
            elif errors:
                status = "hardware/telemetry issue " + ", ".join(str(value) for value in errors)
            else:
                status = "ready"
            summaries.append(
                f"{display}: {', '.join(str(value) for value in relevant_ids) or 'none'} ({status})"
            )
        self.state.segment_readiness_summary = "; ".join(summaries)
        self.state.selected_servo_segment_label = selected_segment_label
        if self.state.robot_mode in {"dual_segment", "parallel_single"}:
            missing = sorted(int(value) for value in self.state.missing_servo_ids)
            mode_label = (
                "Parallel-single demo (both spines)"
                if self.state.robot_mode == "parallel_single"
                else "All-8"
            )
            if not self.state.connected:
                self.state.all_8_readiness_summary = f"Disconnected: {mode_label.lower()} expects 8 servos and none are read."
            elif missing:
                self.state.all_8_readiness_summary = f"{mode_label} readiness incomplete; missing " + ", ".join(str(value) for value in missing) + "."
            elif len(expected) == 8:
                self.state.all_8_readiness_summary = f"{mode_label} readiness: all expected servos are readable."
            else:
                self.state.all_8_readiness_summary = f"{self.state.robot_mode} expects 8 IDs; current context has {expected}."
        else:
            self.state.all_8_readiness_summary = ""
        active_ids_for_mapping = [int(value) for value in self.state.active_segment_servo_ids or expected]
        configured_mapping = configured_axis_mapping(
            expected_servo_ids=active_ids_for_mapping,
            active_segment_pairs=self.state.active_segment_pairs,
        )
        self.state.configured_sign_mapping_summary = (
            "Configured mapping: "
            + ", ".join(
                f"{servo_id}={configured_mapping[int(servo_id)].get('tendon', '')}"
                for servo_id in active_ids_for_mapping
            )
            + "; lower ticks = more tension/shortening."
        )
        if self.state.robot_mode == "single_segment":
            active_ids = [int(value) for value in self.state.active_segment_servo_ids or expected]
            summary = self.servo_service.get_calibration_summary()
            sign_mapping = self._latest_sign_mapping_summary(active_ids)
            readiness = evaluate_selected_segment_readiness(
                operating_mode=self.state.robot_mode,
                active_segment_key=self.state.active_segment_key,
                active_segment_label=self.state.active_segment_label,
                expected_servo_ids=active_ids,
                calibration_summary=summary,
                mock_mode=bool(self.settings.runtime.mock_mode),
                servo_connected=bool(self.state.connected),
                runtime_snapshot=self.latest_runtime_snapshot,
                telemetry_rows=self.state.telemetry,
                sign_mapping=sign_mapping,
            )
            self.state.single_segment_readiness_summary = readiness.compact_text()
        else:
            self.state.single_segment_readiness_summary = ""

    def _latest_sign_mapping_summary(self, expected_servo_ids: list[int]):
        try:
            repo = ServoMappingCheckRepository(self._project_root_for_runtime_artifacts() / "data" / "calibration" / "servo_mapping_checks")
            return repo.latest_for_segment(
                active_segment_key=self.state.active_segment_key,
                expected_servo_ids=[int(value) for value in expected_servo_ids],
            )
        except Exception:
            return None

    def _project_root_for_runtime_artifacts(self) -> Path:
        path = Path(self.servo_service.neutral_calibration.path).resolve()
        if path.parent.name == "config":
            return path.parent.parent
        return path.parent

    def _refresh_selected_servo_live(self) -> None:
        if not self.state.connected or self.state.selected_servo_id is None:
            self._sync_selected_servo_motion_state()
            return
        try:
            assessments = self._refresh_live_telemetry_rows([int(self.state.selected_servo_id)], replace=False)
        except ServoBusBusyError as exc:
            self.state.status_message = str(exc)
            self.state.last_error = None
            self.state.bench_debug_text = self._build_multi_servo_bench_debug_text()
            self._sync_selected_servo_motion_state()
            self._sync_segment_readiness_summary()
            return
        except Exception as exc:
            self.state.last_error = str(exc)
            self.state.status_message = f"Selected-servo refresh failed: {exc}"
            self._sync_selected_servo_motion_state()
            self._sync_segment_readiness_summary()
            return
        selected = assessments.get(int(self.state.selected_servo_id))
        if selected is not None:
            self.state.blocking_reasons = list(selected.blocking_reasons)
            self.state.selected_servo_external_power_ready = selected.external_power_ready
        self.state.last_error = None
        self.state.bench_debug_text = self._build_multi_servo_bench_debug_text()
        self._sync_selected_servo_motion_state()
        self._sync_segment_readiness_summary()

    def _telemetry_row_from_live_item(
        self,
        servo_id: int,
        telemetry,
        assessment: ServoMotionAssessment,
        *,
        existing_row: dict | None = None,
        runtime_entry=None,
    ) -> dict:
        existing_row = dict(existing_row or {})
        telemetry_age_s = self.servo_service.telemetry_age_s(telemetry)
        telemetry_fresh = self.servo_service.telemetry_is_fresh(telemetry)
        packet_read_ok = bool(
            getattr(runtime_entry, "packet_read_ok", False)
            if runtime_entry is not None
            else (
                telemetry.present_position is not None
                and telemetry.hardware_error_code in (None, 0)
                and not telemetry.hardware_error
                and not telemetry.telemetry_error
                and not telemetry.identity_error
            )
        )
        required_fields_ok = bool(
            getattr(runtime_entry, "required_fields_ok", False)
            if runtime_entry is not None
            else (
                telemetry.present_position is not None
                and telemetry.operating_mode is not None
                and telemetry.hardware_error_code is not None
                and telemetry.telemetry_error is None
            )
        )
        stale_display_warning = bool(
            getattr(runtime_entry, "stale_display_warning", False)
            if runtime_entry is not None
            else packet_read_ok and telemetry_fresh is False
        )
        experiment_motion_ready = bool(
            getattr(runtime_entry, "experiment_motion_ready", False)
            if runtime_entry is not None
            else assessment.ready
        )
        hardware_error_text = self._hardware_error_text(
            telemetry.hardware_error_code,
            telemetry.hardware_error,
        )
        block_reason = self._first_blocking_reason(list(assessment.blocking_reasons))
        status_payload = {
            "position": telemetry.present_position,
            "current_ma": telemetry.present_current_ma,
            "voltage_mv": telemetry.present_voltage_mv,
            "temperature_c": telemetry.present_temperature_c,
            "blocking_reasons": list(assessment.blocking_reasons),
            "telemetry_fresh": telemetry_fresh,
            "packet_read_ok": packet_read_ok,
            "stale_display_warning": stale_display_warning,
        }
        return {
            "reported_servo_id": telemetry.reported_servo_id if telemetry.reported_servo_id is not None else existing_row.get("reported_servo_id"),
            "model_number": telemetry.model_number if telemetry.model_number is not None else existing_row.get("model_number"),
            "firmware_version": telemetry.firmware_version if telemetry.firmware_version is not None else existing_row.get("firmware_version"),
            "operating_mode": telemetry.operating_mode,
            "torque_enabled": telemetry.torque_enabled,
            "torque_label": self._torque_label(telemetry.torque_enabled),
            "position": telemetry.present_position,
            "current_ma": telemetry.present_current_ma,
            "voltage_mv": telemetry.present_voltage_mv,
            "temperature_c": telemetry.present_temperature_c,
            "hardware_error": telemetry.hardware_error_code,
            "hardware_error_text": hardware_error_text,
            "error": telemetry.hardware_error or telemetry.identity_error or telemetry.telemetry_error,
            "safe_min_tick": assessment.safe_min_tick,
            "safe_max_tick": assessment.safe_max_tick,
            "safe_bounds": self._assessment_bounds_text(assessment),
            "ready": (
                "fresh pre-motion read before command"
                if experiment_motion_ready and not assessment.ready and stale_display_warning
                else self._assessment_status_text(assessment)
            ),
            "motion_ready": bool(experiment_motion_ready),
            "telemetry_status": self._telemetry_status_from_row(status_payload),
            "telemetry_age_s": telemetry_age_s,
            "telemetry_age_label": self._format_age_label(telemetry_age_s),
            "telemetry_fresh": telemetry_fresh,
            "gui_cache_fresh": telemetry_fresh,
            "packet_read_ok": packet_read_ok,
            "required_fields_ok": required_fields_ok,
            "stale_display_warning": stale_display_warning,
            "experiment_motion_ready": experiment_motion_ready,
            "freshness_threshold_s": self.servo_service.telemetry_freshness_threshold_s(),
            "position_convention": self.servo_service.position_convention_summary(),
            "external_power_ready": assessment.external_power_ready,
            "blocking_reasons": list(assessment.blocking_reasons),
            "block_reason": block_reason,
            "read_source": telemetry.read_source,
            "last_valid_packet_monotonic_s": telemetry.last_valid_packet_monotonic_s,
            "last_valid_packet_wall_time": telemetry.last_valid_packet_wall_time,
            "last_read_attempt_monotonic_s": telemetry.last_read_attempt_monotonic_s,
            "last_read_monotonic_s": telemetry.last_read_monotonic_s,
            "read_duration_ms": telemetry.read_duration_ms,
            "packet_age_s": telemetry.packet_age_s,
            "telemetry_error_code": telemetry.telemetry_error_code,
            "telemetry_error_detail": telemetry.telemetry_error_detail,
            "bus_owner": telemetry.bus_owner,
            "read_sequence_index": telemetry.read_sequence_index,
        }

    def _missing_telemetry_row(self, servo_id: int, existing_row: dict | None = None) -> dict:
        existing_row = dict(existing_row or {})
        return {
            "reported_servo_id": existing_row.get("reported_servo_id", servo_id),
            "model_number": existing_row.get("model_number"),
            "firmware_version": existing_row.get("firmware_version"),
            "operating_mode": None,
            "torque_enabled": None,
            "torque_label": "—",
            "position": None,
            "current_ma": None,
            "voltage_mv": None,
            "temperature_c": None,
            "hardware_error": None,
            "hardware_error_text": "read failed",
            "error": "telemetry unavailable",
            "safe_min_tick": None,
            "safe_max_tick": None,
            "safe_bounds": "unavailable",
            "ready": "telemetry unavailable",
            "motion_ready": False,
            "telemetry_status": "Unreadable",
            "telemetry_age_s": None,
            "packet_read_ok": False,
            "required_fields_ok": False,
            "stale_display_warning": False,
            "experiment_motion_ready": False,
            "read_source": existing_row.get("read_source", "unavailable"),
            "telemetry_age_label": "—",
            "telemetry_fresh": None,
            "freshness_threshold_s": self.servo_service.telemetry_freshness_threshold_s(),
            "position_convention": self.servo_service.position_convention_summary(),
            "external_power_ready": None,
            "blocking_reasons": ["Telemetry is unavailable."],
            "block_reason": "Telemetry is unavailable.",
            "last_read_monotonic_s": None,
        }

    def _refresh_single_servo_state(self) -> ServosViewState:
        expected_servo_id = self._expected_servo_id()
        snapshot = self.servo_service.build_bench_debug_snapshot(expected_servo_id)
        servo_id = int(snapshot.selected_servo_id or snapshot.expected_servo_id or 1)
        self.state.servo_ids = [servo_id] if snapshot.expected_servo_id is not None else []
        self.state.detected_servo_ids = [servo_id] if snapshot.ping_ok else []
        self.state.missing_servo_ids = (
            []
            if snapshot.ping_ok or snapshot.expected_servo_id is None
            else [int(snapshot.expected_servo_id)]
        )
        self.state.unexpected_servo_ids = []
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
        self._sync_segment_readiness_summary()
        return self.state

    def _telemetry_row_from_snapshot(self, snapshot, servo_id: int) -> dict:
        telemetry = snapshot.telemetry
        assessment = snapshot.motion_assessment
        telemetry_age_s = self.servo_service.telemetry_age_s(telemetry)
        telemetry_fresh = self.servo_service.telemetry_is_fresh(telemetry)
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
            "torque_label": self._torque_label(telemetry.torque_enabled if telemetry is not None else None),
            "position": telemetry.present_position if telemetry is not None else None,
            "current_ma": telemetry.present_current_ma if telemetry is not None else None,
            "voltage_mv": telemetry.present_voltage_mv if telemetry is not None else None,
            "temperature_c": telemetry.present_temperature_c if telemetry is not None else None,
            "min_position_limit": telemetry.min_position_limit if telemetry is not None else None,
            "max_position_limit": telemetry.max_position_limit if telemetry is not None else None,
            "hardware_error": telemetry.hardware_error_code if telemetry is not None else None,
            "hardware_error_text": self._hardware_error_text(
                telemetry.hardware_error_code if telemetry is not None else None,
                telemetry.hardware_error if telemetry is not None else None,
            ),
            "error": error,
            "safe_min_tick": assessment.safe_min_tick if assessment is not None else None,
            "safe_max_tick": assessment.safe_max_tick if assessment is not None else None,
            "safe_bounds": self._assessment_bounds_text(assessment) if assessment is not None else "unavailable",
            "ready": ready_text,
            "motion_ready": bool(snapshot.motion_ready),
            "telemetry_status": self._telemetry_status_from_snapshot(snapshot),
            "telemetry_age_s": telemetry_age_s,
            "telemetry_age_label": self._format_age_label(telemetry_age_s),
            "telemetry_fresh": telemetry_fresh,
            "freshness_threshold_s": self.servo_service.telemetry_freshness_threshold_s(),
            "position_convention": self.servo_service.position_convention_summary(),
            "external_power_ready": snapshot.selected_servo_id is not None and snapshot.motion_assessment is not None and snapshot.motion_assessment.external_power_ready,
            "blocking_reasons": list(assessment.blocking_reasons) if assessment is not None else ([snapshot.motion_block_reason] if snapshot.motion_block_reason else []),
            "block_reason": self._first_blocking_reason(
                list(assessment.blocking_reasons) if assessment is not None else ([snapshot.motion_block_reason] if snapshot.motion_block_reason else [])
            ),
            "last_read_monotonic_s": telemetry.last_read_monotonic_s if telemetry is not None else None,
        }

    def _build_bench_debug_text(self, snapshot) -> str:
        telemetry = snapshot.telemetry
        last_hw_error = None
        telemetry_age_s = self.servo_service.telemetry_age_s(telemetry)
        telemetry_fresh = self.servo_service.telemetry_is_fresh(telemetry)
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
                f"telemetry_age_s={telemetry_age_s}",
                f"freshness_threshold_s={self.servo_service.telemetry_freshness_threshold_s():.3f}",
                f"telemetry_fresh={telemetry_fresh}",
                f"calibration_entries_loaded={snapshot.calibration_entries_loaded}",
                f"one_servo_mode_ok={snapshot.one_servo_mode_ok}",
                f"active_range={self.servo_service.raw_position_range()[0]}..{self.servo_service.raw_position_range()[1]}",
                f"motion_ready={snapshot.motion_ready}",
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
                f"expected_servo_ids={self.state.expected_servo_ids}",
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
                f"freshness_threshold_s={self.servo_service.telemetry_freshness_threshold_s():.3f}",
                f"active_range={self.servo_service.raw_position_range()[0]}..{self.servo_service.raw_position_range()[1]}",
                "motion_ready=False",
                "position_convention=tighten->smaller_counts; loosen->larger_counts",
                f"motion_block_reason={extra_error or 'OpenRB/DYNAMIXEL not ready'}",
            ]
        )

    def _build_multi_servo_bench_debug_text(self) -> str:
        selected_servo_id = self.state.selected_servo_id
        selected = (
            self.state.telemetry.get(int(selected_servo_id), {})
            if selected_servo_id is not None
            else {}
        )
        return "\n".join(
            [
                "Bench debug:",
                f"bus_connected={self.state.connected}",
                f"selected_port={self.settings.serial.openrb_port or 'unset'}",
                f"selected_baud={self.settings.serial.baudrate}",
                f"expected_servo_ids={self.state.expected_servo_ids}",
                f"detected_servo_ids={self.state.detected_servo_ids}",
                f"missing_servo_ids={self.state.missing_servo_ids}",
                f"unexpected_servo_ids={self.state.unexpected_servo_ids}",
                f"selected_servo_id={selected_servo_id}",
                f"selected_position={selected.get('position')}",
                f"selected_current={selected.get('current_ma')}",
                f"selected_voltage={selected.get('voltage_mv')}",
                f"selected_temperature={selected.get('temperature_c')}",
                f"selected_operating_mode={selected.get('operating_mode')}",
                f"selected_torque={selected.get('torque_label')}",
                f"selected_telemetry_age_s={selected.get('telemetry_age_s')}",
                f"freshness_threshold_s={self.servo_service.telemetry_freshness_threshold_s():.3f}",
                f"selected_telemetry_fresh={selected.get('telemetry_fresh')}",
                f"selected_motion_ready={selected.get('motion_ready')}",
                f"selected_error={selected.get('error') or 'none'}",
                f"active_range={self.servo_service.raw_position_range()[0]}..{self.servo_service.raw_position_range()[1]}",
                f"single_segment_motion_config={self.state.single_segment_motion_config_summary or 'unavailable'}",
                f"single_segment_characterization={self.state.single_segment_characterization_summary or 'unavailable'}",
                "position_convention=tighten->smaller_counts; loosen->larger_counts",
            ]
        )

    def _apply_jog_result(self, result, *, action: str) -> None:
        servo_id = int(result.servo_id)
        self.state.selected_servo_id = servo_id
        telemetry = getattr(result, "telemetry", None)
        current_position_tick = (
            telemetry.present_position
            if telemetry is not None and getattr(telemetry, "present_position", None) is not None
            else result.current_position_tick
        )
        motion_state = {
            "current_position_tick": current_position_tick,
            "last_target_tick": result.goal_tick,
            "last_unclamped_target_tick": result.unclamped_goal_tick,
            "safe_min_tick": result.safe_min_tick,
            "safe_max_tick": result.safe_max_tick,
            "last_motion_clamped": bool(result.clamped),
            "last_action_label": self._format_action_label(action),
            "last_result_label": "Clamped" if result.success and result.clamped else ("Sent" if result.success else "Blocked"),
            "reason_label": "none" if result.success else result.message,
            "motion_summary": "",
        }
        if result.success:
            if result.clamped and result.goal_tick is not None:
                motion_state["motion_summary"] = (
                    f"sent {result.command_direction} to {result.goal_tick} (requested {result.unclamped_goal_tick}, clamped)"
                )
            elif result.goal_tick is not None:
                motion_state["motion_summary"] = (
                    f"sent {result.command_direction} to {result.goal_tick}"
                )
            else:
                motion_state["motion_summary"] = f"sent {result.command_direction}"
        else:
            motion_state["motion_summary"] = result.message
        self._motion_state_by_servo[servo_id] = motion_state

    def _sync_selected_servo_motion_state(self) -> None:
        selected_servo_id = self.state.selected_servo_id
        if selected_servo_id is None:
            self.state.selected_servo_torque_enabled = None
            self.state.selected_servo_current_position_tick = None
            self.state.selected_servo_current_ma = None
            self.state.selected_servo_voltage_mv = None
            self.state.selected_servo_temperature_c = None
            self.state.selected_servo_safe_min_tick = None
            self.state.selected_servo_safe_max_tick = None
            self.state.selected_servo_last_target_tick = None
            self.state.selected_servo_last_unclamped_target_tick = None
            self.state.selected_servo_last_motion_clamped = False
            self.state.selected_servo_last_action_label = "None"
            self.state.selected_servo_last_result_label = "None"
            self.state.selected_servo_last_motion_summary = "No jog command sent yet."
            self.state.selected_servo_motion_ready = False
            self.state.selected_servo_reason_label = "none"
            self.state.selected_servo_telemetry_status_label = "Unknown"
            self.state.selected_servo_last_read_label = "—"
            self.state.selected_servo_telemetry_age_s = None
            self.state.selected_servo_telemetry_fresh = None
            return
        selected = self.state.telemetry.get(int(selected_servo_id), {})
        motion_state = dict(self._motion_state_by_servo.get(int(selected_servo_id), {}))
        self.state.selected_servo_torque_enabled = selected.get("torque_enabled")
        self.state.selected_servo_current_position_tick = selected.get("position")
        self.state.selected_servo_current_ma = selected.get("current_ma")
        self.state.selected_servo_voltage_mv = selected.get("voltage_mv")
        self.state.selected_servo_temperature_c = selected.get("temperature_c")
        raw_min, raw_max = self.servo_service.raw_position_range()
        self.state.selected_servo_safe_min_tick = int(raw_min)
        self.state.selected_servo_safe_max_tick = int(raw_max)
        self.state.selected_servo_last_target_tick = motion_state.get("last_target_tick")
        self.state.selected_servo_last_unclamped_target_tick = motion_state.get("last_unclamped_target_tick")
        self.state.selected_servo_last_motion_clamped = bool(motion_state.get("last_motion_clamped", False))
        self.state.selected_servo_last_action_label = str(motion_state.get("last_action_label", "None"))
        self.state.selected_servo_last_result_label = str(motion_state.get("last_result_label", "None"))
        self.state.selected_servo_last_motion_summary = str(
            motion_state.get("motion_summary", "No jog command sent yet.")
        )
        self.state.selected_servo_telemetry_age_s = selected.get("telemetry_age_s")
        self.state.selected_servo_telemetry_fresh = selected.get("telemetry_fresh")
        self.state.selected_servo_last_read_label = self._format_last_read_label(
            selected.get("telemetry_age_s")
        )
        self.state.selected_servo_motion_ready = bool(selected.get("motion_ready", False))
        self.state.selected_servo_telemetry_status_label = self._telemetry_status_from_row(selected)
        blocking_reasons = [str(reason) for reason in selected.get("blocking_reasons", []) if str(reason).strip()]
        if not blocking_reasons and self.state.blocking_reasons:
            blocking_reasons = [str(reason) for reason in self.state.blocking_reasons if str(reason).strip()]
        if blocking_reasons:
            self.state.selected_servo_reason_label = blocking_reasons[0]
        elif self.state.selected_servo_last_result_label == "Blocked":
            self.state.selected_servo_reason_label = str(motion_state.get("reason_label", "none"))
        else:
            self.state.selected_servo_reason_label = "none"

    def _sync_active_neutral_setpoints(self) -> None:
        summary = self.servo_service.get_calibration_summary()
        if not summary.exists or not summary.compatible:
            self.state.neutral_setpoints = {}
            return
        active_setpoints: dict[int, int] = {}
        for servo_id, entry in summary.servo_entries.items():
            if (
                int(servo_id) in self.settings.robot.expected_servo_ids()
                and entry.valid
                and entry.neutral_setpoint is not None
            ):
                active_setpoints[int(servo_id)] = int(entry.neutral_setpoint)
        self.state.neutral_setpoints = active_setpoints

    def _refresh_single_segment_motion_diagnostics(self) -> None:
        active_ids = list(self.state.active_segment_servo_ids or self.state.expected_servo_ids)
        if self.state.single_servo_mode or len(active_ids) != 4:
            self.state.single_segment_motion_config_summary = ""
            self.state.single_segment_reference_summary = ""
            self.state.single_segment_enforced_bounds_summary = ""
            self.state.single_segment_characterization_summary = ""
            return
        servo_ids = list(active_ids)
        config_summary = self.servo_service.single_segment_motion_configuration_summary(servo_ids)
        self.state.single_segment_motion_config_summary = config_summary.message
        try:
            reference = self.servo_service.resolve_startup_reference_ticks(servo_ids)
        except Exception as exc:
            reference = None
            self.state.single_segment_reference_summary = (
                f"Startup/pretension reference is unavailable: {exc}"
            )
        else:
            self.state.single_segment_reference_summary = reference.message
        raw_min, raw_max = self.servo_service.raw_position_range()
        margin = int(self.settings.safety.software_position_margin_ticks)
        self.state.single_segment_enforced_bounds_summary = (
            f"Experiment motion enforces raw {raw_min}..{raw_max} plus live servo position limits with a "
            f"{margin}-tick software margin. Saved startup artifact bounds are display-only metadata."
        )
        telemetry_by_id = {}
        snapshot = self.latest_runtime_snapshot
        if snapshot is not None:
            for servo_id, entry in dict(getattr(snapshot, "entries", {}) or {}).items():
                if entry is not None and getattr(entry, "telemetry", None) is not None:
                    telemetry_by_id[int(servo_id)] = entry.telemetry
        characterization = self.servo_service.characterize_single_segment_motion(
            servo_ids=servo_ids,
            telemetry_by_id=telemetry_by_id or None,
            neutral_ticks_by_id=(reference.ticks_by_servo if reference is not None else None),
        )
        if characterization.available:
            self.state.single_segment_characterization_summary = (
                "Display-only diagnostic pair travel around the current startup reference: "
                + characterization.message
            )
        else:
            self.state.single_segment_characterization_summary = characterization.message

    @staticmethod
    def _format_action_label(action: str) -> str:
        return " ".join(part.capitalize() for part in str(action).strip().split("_")) or "None"

    @staticmethod
    def _telemetry_status_from_snapshot(snapshot) -> str:
        if snapshot.telemetry_read_ok is not True:
            return "Unreadable"
        assessment = snapshot.motion_assessment
        if assessment is not None and any(
            "telemetry is stale" in str(reason).lower() for reason in assessment.blocking_reasons
        ):
            return "Stale display"
        return "Live"

    @staticmethod
    def _telemetry_status_from_row(row: dict) -> str:
        if not row:
            return "Unknown"
        if row.get("position") is None:
            return "Unreadable"
        if bool(row.get("stale_display_warning")) or row.get("telemetry_fresh") is False:
            return "Stale display" if bool(row.get("packet_read_ok", True)) else "Stale"
        for reason in row.get("blocking_reasons", []):
            if "telemetry is stale" in str(reason).lower():
                return "Stale display" if bool(row.get("packet_read_ok", True)) else "Stale"
        return "Live"

    def _telemetry_status_from_result(self, result) -> str:
        if result.telemetry is None:
            return "Unreadable"
        return self._telemetry_status_from_row(
            {
                "position": result.telemetry.present_position,
                "current_ma": result.telemetry.present_current_ma,
                "voltage_mv": result.telemetry.present_voltage_mv,
                "temperature_c": result.telemetry.present_temperature_c,
                "blocking_reasons": list(
                    result.assessment.blocking_reasons if result.assessment is not None else ()
                ),
            }
        )

    def _update_selected_row_from_jog_result(self, result) -> None:
        servo_id = int(result.servo_id)
        telemetry = getattr(result, "telemetry", None)
        assessment = getattr(result, "assessment", None)
        if telemetry is None or assessment is None:
            return
        existing_row = self.state.telemetry.get(servo_id)
        self.state.telemetry[servo_id] = self._telemetry_row_from_live_item(
            servo_id,
            telemetry,
            assessment,
            existing_row=existing_row,
        )
        if servo_id not in self.state.detected_servo_ids:
            self.state.detected_servo_ids.append(servo_id)
            self.state.detected_servo_ids.sort()
        self.state.missing_servo_ids = [sid for sid in self.state.missing_servo_ids if sid != servo_id]
        self.state.blocking_reasons = list(assessment.blocking_reasons)
        self.state.selected_servo_external_power_ready = assessment.external_power_ready

    @staticmethod
    def _torque_label(value: bool | None) -> str:
        if value is None:
            return "—"
        return "On" if bool(value) else "Off"

    @staticmethod
    def _format_age_label(age_s: float | None) -> str:
        if age_s is None:
            return "—"
        return f"{float(age_s):.3f} s"

    @classmethod
    def _format_last_read_label(cls, age_s: float | None) -> str:
        if age_s is None:
            return "unknown"
        return f"{cls._format_age_label(age_s)} ago"

    @staticmethod
    def _hardware_error_text(code: int | None, error: str | None) -> str:
        if code in (None, 0) and not error:
            return "0"
        if code not in (None, 0) and error:
            return f"0x{int(code):02X} | {error}"
        if code not in (None, 0):
            return f"0x{int(code):02X}"
        return str(error or "—")

    @staticmethod
    def _first_blocking_reason(reasons: list[str]) -> str:
        for reason in reasons:
            text = str(reason).strip()
            if text:
                return text
        return "ready"

    @staticmethod
    def _telemetry_indicates_present(row: dict) -> bool:
        if not row:
            return False
        return any(
            row.get(field) is not None
            for field in (
                "reported_servo_id",
                "position",
                "current_ma",
                "voltage_mv",
                "temperature_c",
            )
        )

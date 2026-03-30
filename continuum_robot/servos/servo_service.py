"""High-level servo command service."""

from __future__ import annotations

from dataclasses import dataclass
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

    @property
    def is_connected(self) -> bool:
        return self.dxl_bus.is_connected

    def connect(self, port: str, baudrate: int) -> None:
        self.dxl_bus.connect(port, baudrate)

    def disconnect(self) -> None:
        self.dxl_bus.disconnect()

    def scan_ids(self, min_id: int = 1, max_id: int = 20) -> list[int]:
        return self.dxl_bus.scan_ids(min_id=min_id, max_id=max_id)

    def assign_servo_id(self, current_id: int, new_id: int) -> None:
        self.dxl_bus.write_servo_id(current_id, new_id)

    def read_telemetry(self, servo_ids: list[int]) -> dict[int, ServoTelemetry]:
        return self.dxl_bus.read_telemetry(servo_ids)

    def load_neutral_setpoints(self) -> dict[int, int]:
        return self.neutral_calibration.load_neutral_setpoints()

    def save_neutral_setpoints(self, setpoints_by_id: dict[int, int]) -> None:
        self.neutral_calibration.save_neutral_setpoints(setpoints_by_id)

    def load_calibration_artifact(self) -> ServoCalibrationArtifact:
        return self.neutral_calibration.load_calibration_artifact()

    def get_calibration_summary(self) -> ServoCalibrationSummary:
        return self.neutral_calibration.get_calibration_summary()

    def capture_neutral_setpoints(self, servo_ids: list[int]) -> dict[int, int]:
        telemetry = self.read_telemetry(servo_ids)
        setpoints: dict[int, int] = {}
        for servo_id in servo_ids:
            position = telemetry[servo_id].present_position
            if position is None:
                raise RuntimeError(f"Servo {servo_id} position is unavailable")
            setpoints[servo_id] = int(position)
        return setpoints

    def get_tightening_direction(self, servo_id: int) -> str | None:
        entry = self.neutral_calibration.entry_by_servo_id(int(servo_id))
        if entry and entry.tightening_rotation:
            return entry.tightening_rotation
        return self.neutral_calibration.context.tightening_rotation_by_servo.get(int(servo_id))

    def assess_motion(
        self,
        servo_id: int,
        *,
        require_calibrated_bounds: bool,
        telemetry: ServoTelemetry | None = None,
    ) -> ServoMotionAssessment:
        current = telemetry or self.read_telemetry([servo_id])[int(servo_id)]
        errors: list[str] = []
        safe_min: int | None = None
        safe_max: int | None = None

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
            if current.present_voltage_mv is None or current.present_voltage_mv <= 0:
                errors.append("Input voltage telemetry is unavailable.")
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
            safe_min, safe_max = self._safe_bounds_for_servo(
                servo_id=int(servo_id),
                telemetry=current,
                require_calibrated_bounds=require_calibrated_bounds,
            )
        except ValueError as exc:
            errors.append(str(exc))
        return ServoMotionAssessment(
            servo_id=int(servo_id),
            ready=not errors,
            reason=" | ".join(errors) if errors else "Ready for cautious motion.",
            telemetry=current,
            safe_min_tick=safe_min,
            safe_max_tick=safe_max,
            tightening_direction=self.get_tightening_direction(int(servo_id)),
        )

    def jog_servo(self, servo_id: int, delta_ticks: int) -> ServoCommandResult:
        # All live single-servo motion must pass through ServoService so
        # calibration bounds, telemetry refresh, and current checks stay consistent.
        self.safety_guard.validate_jog_delta(delta_ticks)
        assessment = self.assess_motion(int(servo_id), require_calibrated_bounds=False)
        if not assessment.ready:
            raise RuntimeError(f"Servo {servo_id} is not safe to jog: {assessment.reason}")
        if assessment.telemetry.present_position is None:
            raise RuntimeError(f"Servo {servo_id} position is unavailable.")
        goal = int(assessment.telemetry.present_position + delta_ticks)
        self._validate_goal_against_assessment(assessment, goal)
        self.dxl_bus.write_goal_positions({int(servo_id): goal})
        updated = self.read_telemetry([int(servo_id)])
        self._validate_post_motion(updated[int(servo_id)])
        return ServoCommandResult(
            positions_by_id={int(servo_id): goal},
            telemetry_by_id=updated,
            message=(
                f"Jogged servo {servo_id} to {goal} ticks "
                f"within [{assessment.safe_min_tick}, {assessment.safe_max_tick}]."
            ),
        )

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
        self.dxl_bus.write_goal_positions(payload)
        telemetry = self.dxl_bus.read_telemetry(servo_ids)
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
        assessment = self.assess_motion(int(servo_id), require_calibrated_bounds=False)
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
        hardware_min, hardware_max = self._safe_bounds_for_servo(
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
        threshold_ma: int | None = None,
        stop_requested: Callable[[], bool] | None = None,
        progress_callback: Callable[[PretensionRoutineResult], None] | None = None,
    ) -> PretensionRoutineResult:
        assessment = self.assess_motion(int(servo_id), require_calibrated_bounds=True)
        if not assessment.ready:
            raise RuntimeError(f"Servo {servo_id} is not safe to pretension: {assessment.reason}")
        threshold = int(threshold_ma) if threshold_ma is not None else self._pretension_threshold_for_servo(int(servo_id))
        if threshold <= 0:
            raise ValueError("Pretension threshold must be positive.")
        if threshold >= int(self.safety_guard.max_current_ma):
            raise ValueError(
                f"Pretension threshold {threshold} mA must be below the absolute safety limit "
                f"{self.safety_guard.max_current_ma} mA."
            )
        tightening_direction = self.get_tightening_direction(int(servo_id))
        step_sign = self._tightening_step_sign(int(servo_id))
        safe_min = int(assessment.safe_min_tick) if assessment.safe_min_tick is not None else None
        safe_max = int(assessment.safe_max_tick) if assessment.safe_max_tick is not None else None
        deadline = self._time_fn() + float(self.safety_guard.pretension_timeout_s)
        steps_taken = 0

        def _emit(result: PretensionRoutineResult) -> PretensionRoutineResult:
            if progress_callback is not None:
                progress_callback(result)
            return result

        while True:
            if stop_requested is not None and stop_requested():
                final = self.read_telemetry([int(servo_id)])[int(servo_id)]
                result = PretensionRoutineResult(
                    servo_id=int(servo_id),
                    status="canceled",
                    success=False,
                    message=f"Pretension canceled for servo {servo_id}.",
                    threshold_ma=threshold,
                    final_position_tick=final.present_position,
                    final_current_ma=final.present_current_ma,
                    steps_taken=steps_taken,
                    tightening_direction=tightening_direction,
                )
                self.neutral_calibration.save_pretension_result(
                    servo_id=int(servo_id),
                    final_position_tick=result.final_position_tick,
                    final_current_ma=result.final_current_ma,
                    threshold_ma=threshold,
                    result_status=result.status,
                )
                return _emit(result)
            if self._time_fn() > deadline:
                final = self.read_telemetry([int(servo_id)])[int(servo_id)]
                result = PretensionRoutineResult(
                    servo_id=int(servo_id),
                    status="timeout",
                    success=False,
                    message=f"Pretension timed out for servo {servo_id}.",
                    threshold_ma=threshold,
                    final_position_tick=final.present_position,
                    final_current_ma=final.present_current_ma,
                    steps_taken=steps_taken,
                    tightening_direction=tightening_direction,
                )
                self.neutral_calibration.save_pretension_result(
                    servo_id=int(servo_id),
                    final_position_tick=result.final_position_tick,
                    final_current_ma=result.final_current_ma,
                    threshold_ma=threshold,
                    result_status=result.status,
                )
                return _emit(result)

            raw_telemetry = self.read_telemetry([int(servo_id)])[int(servo_id)]
            if raw_telemetry.present_current_ma is not None and raw_telemetry.present_current_ma >= int(
                self.safety_guard.max_current_ma
            ):
                result = PretensionRoutineResult(
                    servo_id=int(servo_id),
                    status="overcurrent",
                    success=False,
                    message=(
                        f"Pretension stopped for servo {servo_id}: measured current "
                        f"{raw_telemetry.present_current_ma} mA reached the absolute safety limit."
                    ),
                    threshold_ma=threshold,
                    final_position_tick=raw_telemetry.present_position,
                    final_current_ma=raw_telemetry.present_current_ma,
                    steps_taken=steps_taken,
                    tightening_direction=tightening_direction,
                )
                self.neutral_calibration.save_pretension_result(
                    servo_id=int(servo_id),
                    final_position_tick=result.final_position_tick,
                    final_current_ma=result.final_current_ma,
                    threshold_ma=threshold,
                    result_status=result.status,
                )
                return _emit(result)

            current_assessment = self.assess_motion(
                int(servo_id),
                require_calibrated_bounds=True,
                telemetry=raw_telemetry,
            )
            if not current_assessment.ready:
                result = PretensionRoutineResult(
                    servo_id=int(servo_id),
                    status="invalid_telemetry",
                    success=False,
                    message=f"Pretension stopped for servo {servo_id}: {current_assessment.reason}",
                    threshold_ma=threshold,
                    final_position_tick=current_assessment.telemetry.present_position,
                    final_current_ma=current_assessment.telemetry.present_current_ma,
                    steps_taken=steps_taken,
                    tightening_direction=tightening_direction,
                )
                self.neutral_calibration.save_pretension_result(
                    servo_id=int(servo_id),
                    final_position_tick=result.final_position_tick,
                    final_current_ma=result.final_current_ma,
                    threshold_ma=threshold,
                    result_status=result.status,
                )
                return _emit(result)
            current_ma = current_assessment.telemetry.present_current_ma
            position = current_assessment.telemetry.present_position
            if current_ma is None or position is None:
                raise RuntimeError(f"Servo {servo_id} pretension requires current and position telemetry.")
            if current_ma >= threshold:
                result = PretensionRoutineResult(
                    servo_id=int(servo_id),
                    status="threshold_reached",
                    success=True,
                    message=(
                        f"Servo {servo_id} reached the pretension threshold at "
                        f"{position} ticks / {current_ma} mA."
                    ),
                    threshold_ma=threshold,
                    final_position_tick=position,
                    final_current_ma=current_ma,
                    steps_taken=steps_taken,
                    tightening_direction=tightening_direction,
                )
                self.neutral_calibration.save_pretension_result(
                    servo_id=int(servo_id),
                    final_position_tick=result.final_position_tick,
                    final_current_ma=result.final_current_ma,
                    threshold_ma=threshold,
                    result_status=result.status,
                )
                return _emit(result)

            next_goal = int(position + step_sign * int(self.safety_guard.pretension_step_ticks))
            if safe_min is None or safe_max is None or next_goal < safe_min or next_goal > safe_max:
                result = PretensionRoutineResult(
                    servo_id=int(servo_id),
                    status="travel_limit",
                    success=False,
                    message=(
                        f"Pretension stopped for servo {servo_id}: next tightening step would exceed "
                        f"safe bounds [{safe_min}, {safe_max}]."
                    ),
                    threshold_ma=threshold,
                    final_position_tick=position,
                    final_current_ma=current_ma,
                    steps_taken=steps_taken,
                    tightening_direction=tightening_direction,
                )
                self.neutral_calibration.save_pretension_result(
                    servo_id=int(servo_id),
                    final_position_tick=result.final_position_tick,
                    final_current_ma=result.final_current_ma,
                    threshold_ma=threshold,
                    result_status=result.status,
                )
                return _emit(result)

            self.dxl_bus.write_goal_positions({int(servo_id): int(next_goal)})
            steps_taken += 1
            self._sleep_fn(float(self.safety_guard.pretension_settle_time_s))

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

    def _tightening_step_sign(self, servo_id: int) -> int:
        tightening_rotation = self.get_tightening_direction(int(servo_id))
        if tightening_rotation not in {"cw", "ccw"}:
            raise RuntimeError(
                f"Servo {servo_id} tightening direction is not configured. "
                "Set tightening_rotation_by_servo in the robot config or save startup calibration first."
            )
        positive_tick_rotation = str(self.dxl_bus.config.positive_tick_rotation).strip().lower()
        if positive_tick_rotation not in {"cw", "ccw"}:
            raise RuntimeError(
                f"Unsupported positive_tick_rotation setting {self.dxl_bus.config.positive_tick_rotation!r}."
            )
        return 1 if tightening_rotation == positive_tick_rotation else -1

    def _safe_bounds_for_servo(
        self,
        *,
        servo_id: int,
        telemetry: ServoTelemetry,
        require_calibrated_bounds: bool,
    ) -> tuple[int, int]:
        if telemetry.min_position_limit is None or telemetry.max_position_limit is None:
            raise ValueError("Servo position limits are unavailable.")
        safe_min = int(telemetry.min_position_limit) + int(self.safety_guard.software_position_margin_ticks)
        safe_max = int(telemetry.max_position_limit) - int(self.safety_guard.software_position_margin_ticks)
        if safe_min > safe_max:
            raise ValueError("Software safety margin exceeds the servo hardware position range.")

        summary = self.neutral_calibration.get_calibration_summary()
        if summary.exists and not summary.compatible and require_calibrated_bounds:
            raise ValueError(
                "Saved servo calibration does not match the current robot configuration. "
                "Recapture startup calibration before commanding calibrated motion."
            )
        entry = summary.servo_entries.get(int(servo_id)) if summary.exists and summary.compatible else None
        if entry and entry.safe_min_tick is not None and entry.safe_max_tick is not None:
            safe_min = max(safe_min, int(entry.safe_min_tick))
            safe_max = min(safe_max, int(entry.safe_max_tick))
        elif require_calibrated_bounds:
            raise ValueError(
                f"Servo {servo_id} does not have saved safe bounds. "
                "Capture startup calibration before commanding this action."
            )
        if safe_min > safe_max:
            raise ValueError(
                f"Servo {servo_id} safe bounds are invalid after applying hardware and software limits."
            )
        return int(safe_min), int(safe_max)

    @staticmethod
    def _validate_goal_against_assessment(assessment: ServoMotionAssessment, goal_tick: int) -> None:
        if assessment.safe_min_tick is None or assessment.safe_max_tick is None:
            raise RuntimeError(f"Servo {assessment.servo_id} safe bounds are unavailable.")
        if int(goal_tick) < int(assessment.safe_min_tick) or int(goal_tick) > int(assessment.safe_max_tick):
            raise ValueError(
                f"Servo {assessment.servo_id} goal {goal_tick} is outside safe bounds "
                f"[{assessment.safe_min_tick}, {assessment.safe_max_tick}]."
            )

    def _validate_post_motion(self, telemetry: ServoTelemetry) -> None:
        if telemetry.hardware_error_code not in (None, 0) or telemetry.hardware_error:
            raise RuntimeError(
                f"Servo {telemetry.servo_id} reported a hardware/status error after motion: "
                f"{telemetry.hardware_error or f'0x{telemetry.hardware_error_code:02X}'}"
            )
        if self.dxl_bus.config.require_voltage_for_motion and (
            telemetry.present_voltage_mv is None or telemetry.present_voltage_mv <= 0
        ):
            raise RuntimeError(f"Servo {telemetry.servo_id} input voltage telemetry is unavailable after motion.")
        self.safety_guard.validate_currents(
            [telemetry.present_current_ma],
            require_present=self.dxl_bus.config.require_current_for_motion,
        )
        self.safety_guard.validate_temperature(
            telemetry.present_temperature_c,
            require_present=self.dxl_bus.config.require_temperature_for_motion,
        )

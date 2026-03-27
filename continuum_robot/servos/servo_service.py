"""High-level servo command service."""

from __future__ import annotations

from dataclasses import dataclass

from continuum_robot.hardware.dxl_bus import DxlBus, ServoTelemetry
from continuum_robot.servos.displacement_mapper import TendonDisplacementMapper
from continuum_robot.servos.neutral_calibration_service import NeutralCalibrationService
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


class ServoService:
    """Coordinates mapping, validation, persistence, and low-level bus writes."""

    def __init__(
        self,
        dxl_bus: DxlBus,
        mapper: TendonDisplacementMapper,
        safety_guard: SafetyGuard,
        neutral_calibration: NeutralCalibrationService,
        pretension_validation: PretensionValidationService,
    ) -> None:
        self.dxl_bus = dxl_bus
        self.mapper = mapper
        self.safety_guard = safety_guard
        self.neutral_calibration = neutral_calibration
        self.pretension_validation = pretension_validation

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

    def capture_neutral_setpoints(self, servo_ids: list[int]) -> dict[int, int]:
        telemetry = self.read_telemetry(servo_ids)
        setpoints: dict[int, int] = {}
        for servo_id in servo_ids:
            position = telemetry[servo_id].present_position
            if position is None:
                raise RuntimeError(f"Servo {servo_id} position is unavailable")
            setpoints[servo_id] = int(position)
        return setpoints

    def jog_servo(self, servo_id: int, delta_ticks: int) -> ServoCommandResult:
        telemetry = self.read_telemetry([servo_id])
        current = telemetry[servo_id]
        if current.present_position is None:
            raise RuntimeError(f"Servo {servo_id} position is unavailable")
        goal = int(current.present_position + delta_ticks)
        self.dxl_bus.write_goal_positions({servo_id: goal})
        updated = self.read_telemetry([servo_id])
        self.safety_guard.validate_currents([updated[servo_id].present_current_ma])
        return ServoCommandResult(
            positions_by_id={servo_id: goal},
            telemetry_by_id=updated,
            message=f"Jogged servo {servo_id} to {goal} ticks.",
        )

    def command_displacement(
        self,
        tendon_displacements_cm: list[float],
        neutral_ticks: list[int],
        servo_ids: list[int],
    ) -> ServoCommandResult:
        """Compute and send safe goal position ticks."""
        if len(servo_ids) != len(neutral_ticks):
            raise ValueError("Servo ID list and neutral setpoint list length mismatch")
        goals = self.mapper.to_goal_positions(tendon_displacements_cm, neutral_ticks)
        self.safety_guard.validate_positions(goals, neutral_ticks)
        payload = {sid: goal for sid, goal in zip(servo_ids, goals)}
        self.dxl_bus.write_goal_positions(payload)
        telemetry = self.dxl_bus.read_telemetry(servo_ids)
        self.safety_guard.validate_currents([telemetry[sid].present_current_ma for sid in servo_ids])
        return ServoCommandResult(
            positions_by_id=payload,
            telemetry_by_id=telemetry,
            message=f"Commanded {len(payload)} servo(s) from tendon displacement input.",
        )

    def validate_pretension(
        self,
        servo_ids: list[int],
        tolerance_ma: int,
    ) -> PretensionValidationResult:
        telemetry = self.read_telemetry(servo_ids)
        currents = [telemetry[sid].present_current_ma for sid in servo_ids]
        self.safety_guard.validate_currents(currents)
        return self.pretension_validation.validate_current_balance(currents, tolerance_ma)

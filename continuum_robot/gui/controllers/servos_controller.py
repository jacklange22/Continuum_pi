"""Servos tab controller for calibration, telemetry, and manual motion."""

from __future__ import annotations

from dataclasses import dataclass, field

from continuum_robot.config.settings import Settings


@dataclass
class ServosViewState:
    """UI-facing servo control state."""

    connected: bool = False
    servo_ids: list[int] = field(default_factory=list)
    neutral_setpoints: dict[int, int] = field(default_factory=dict)
    telemetry: dict[int, dict] = field(default_factory=dict)
    tendon_displacements_cm: list[float] = field(default_factory=list)
    pretension_message: str = "Pretension not checked."
    status_message: str = "Servo control idle."
    last_error: str | None = None


class ServosController:
    """Owns manual jog, displacement command, and telemetry display actions."""

    def __init__(self, servo_service, settings: Settings) -> None:
        self.servo_service = servo_service
        self.settings = settings
        self.state = ServosViewState(
            connected=servo_service.is_connected,
            servo_ids=list(settings.robot.servo_ids),
            tendon_displacements_cm=[0.0] * len(settings.robot.tendon_to_servo),
            status_message=(
                "Mock servo backend ready."
                if settings.runtime.mock_mode
                else "Real OpenRB/DYNAMIXEL transport is not implemented yet. Servo controls are mock-validated only."
            ),
        )
        self.load_neutral_setpoints()
        self.refresh()

    def refresh(self) -> ServosViewState:
        self.state.connected = self.servo_service.is_connected
        if not self.state.connected:
            self.state.telemetry = {}
            return self.state
        if self.state.servo_ids:
            try:
                telemetry = self.servo_service.read_telemetry(self.state.servo_ids)
            except Exception as exc:
                self.state.last_error = str(exc)
                self.state.status_message = f"Telemetry refresh failed: {exc}"
                self.state.telemetry = {}
                return self.state
            self.state.telemetry = {
                servo_id: {
                    "position": item.present_position,
                    "current_ma": item.present_current_ma,
                    "voltage_mv": item.present_voltage_mv,
                    "error": item.hardware_error,
                }
                for servo_id, item in telemetry.items()
            }
        return self.state

    def scan(self) -> list[int]:
        try:
            ids = self.servo_service.scan_ids()
            if ids:
                self.state.servo_ids = ids
            self.state.status_message = f"Found servo IDs: {self.state.servo_ids}"
            self.state.last_error = None
            return self.state.servo_ids
        except Exception as exc:
            self.state.last_error = str(exc)
            self.state.status_message = f"Servo scan failed: {exc}"
            raise
        finally:
            self.refresh()

    def assign_servo_id(self, current_id: int, new_id: int) -> None:
        try:
            self.servo_service.assign_servo_id(current_id, new_id)
            self.state.status_message = f"Renamed servo {current_id} to {new_id}."
            self.state.last_error = None
            self.scan()
        except Exception as exc:
            self.state.last_error = str(exc)
            self.state.status_message = f"Servo ID assignment failed: {exc}"
            raise

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
            setpoints = self.servo_service.capture_neutral_setpoints(self.state.servo_ids)
            self.state.neutral_setpoints = setpoints
            self.state.status_message = "Captured neutral setpoints from present positions."
            self.state.last_error = None
            return setpoints
        except Exception as exc:
            self.state.last_error = str(exc)
            self.state.status_message = f"Neutral capture failed: {exc}"
            raise

    def save_neutral_setpoints(self) -> None:
        self.servo_service.save_neutral_setpoints(self.state.neutral_setpoints)
        self.state.status_message = "Neutral setpoints saved."
        self.state.last_error = None

    def load_neutral_setpoints(self) -> dict[int, int]:
        setpoints = self.servo_service.load_neutral_setpoints()
        self.state.neutral_setpoints = setpoints
        return setpoints

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

    def _neutral_ticks_for_current_ids(self) -> list[int]:
        if not self.state.neutral_setpoints:
            raise RuntimeError("Neutral setpoints are missing. Capture or load them first.")
        missing = [sid for sid in self.state.servo_ids if sid not in self.state.neutral_setpoints]
        if missing:
            raise RuntimeError(f"Neutral setpoints missing for servo IDs: {missing}")
        return [self.state.neutral_setpoints[sid] for sid in self.state.servo_ids]

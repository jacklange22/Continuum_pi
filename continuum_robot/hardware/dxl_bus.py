"""Low-level DYNAMIXEL bus abstraction.

This module owns raw DYNAMIXEL protocol communication. OpenRB board bring-up is
handled separately by :mod:`continuum_robot.hardware.openrb_client`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import importlib
import time
from typing import Any, Callable


def load_dynamixel_sdk() -> Any:
    """Import and return the Robotis DYNAMIXEL SDK module."""
    try:
        return importlib.import_module("dynamixel_sdk")
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "The Robotis DYNAMIXEL SDK Python package is required for real servo access. "
            "Install it with `pip install dynamixel-sdk` and rerun bootstrap if needed."
        ) from exc


@dataclass
class ServoTelemetry:
    """Readback values for one servo."""

    servo_id: int
    reported_servo_id: int | None = None
    model_number: int | None = None
    firmware_version: int | None = None
    operating_mode: int | None = None
    torque_enabled: bool | None = None
    current_limit_ma: int | None = None
    min_position_limit: int | None = None
    max_position_limit: int | None = None
    bus_watchdog_value: int | None = None
    present_position: int | None = None
    present_current_ma: int | None = None
    present_voltage_mv: int | None = None
    present_temperature_c: int | None = None
    hardware_error_code: int | None = None
    hardware_error: str | None = None
    identity_error: str | None = None
    telemetry_error: str | None = None
    last_read_monotonic_s: float | None = None


@dataclass
class ServoPingResult:
    """One ping attempt against a servo ID."""

    servo_id: int
    responded: bool
    model_number: int | None = None
    error: str | None = None


@dataclass
class DxlBusConfig:
    """Configurable DYNAMIXEL protocol details for X-series style servos."""

    protocol_version: float = 2.0
    positive_tick_rotation: str = "ccw"
    expected_operating_mode: int = 3
    allowed_operating_modes: list[int] = field(default_factory=lambda: [3])
    require_current_for_motion: bool = True
    require_voltage_for_motion: bool = True
    require_temperature_for_motion: bool = True
    require_fresh_telemetry_for_motion: bool = True
    default_profile_velocity: int | None = None
    default_profile_acceleration: int | None = None
    auto_torque_enable_on_write: bool = True
    torque_disable_for_eeprom_write: bool = True
    discovery_min_id: int = 1
    discovery_max_id: int = 20
    voltage_scale_mv_per_unit: float = 100.0
    current_scale_ma_per_unit: float = 1.0
    control_table: dict[str, int] = field(
        default_factory=lambda: {
            "model_number": 0,
            "firmware_version": 6,
            "servo_id": 7,
            "operating_mode": 11,
            "current_limit": 38,
            "max_position_limit": 48,
            "min_position_limit": 52,
            "torque_enable": 64,
            "hardware_error_status": 70,
            "bus_watchdog": 98,
            "profile_acceleration": 108,
            "profile_velocity": 112,
            "goal_position": 116,
            "present_current": 126,
            "present_position": 132,
            "present_input_voltage": 144,
            "present_temperature": 146,
        }
    )

    @classmethod
    def from_dict(cls, payload: dict[str, Any] | None = None) -> "DxlBusConfig":
        payload = dict(payload or {})
        control_table = dict(cls().control_table)
        control_table.update(
            {
                str(key): int(value)
                for key, value in dict(payload.get("control_table", {}) or {}).items()
            }
        )
        return cls(
            protocol_version=float(payload.get("protocol_version", 2.0)),
            positive_tick_rotation=str(payload.get("positive_tick_rotation", "ccw")).strip().lower(),
            expected_operating_mode=int(payload.get("expected_operating_mode", 3)),
            allowed_operating_modes=[
                int(value) for value in list(payload.get("allowed_operating_modes", [payload.get("expected_operating_mode", 3)]))
            ],
            require_current_for_motion=bool(payload.get("require_current_for_motion", True)),
            require_voltage_for_motion=bool(payload.get("require_voltage_for_motion", True)),
            require_temperature_for_motion=bool(payload.get("require_temperature_for_motion", True)),
            require_fresh_telemetry_for_motion=bool(payload.get("require_fresh_telemetry_for_motion", True)),
            default_profile_velocity=(
                int(payload["default_profile_velocity"]) if payload.get("default_profile_velocity") not in (None, "") else None
            ),
            default_profile_acceleration=(
                int(payload["default_profile_acceleration"])
                if payload.get("default_profile_acceleration") not in (None, "")
                else None
            ),
            auto_torque_enable_on_write=bool(payload.get("auto_torque_enable_on_write", True)),
            torque_disable_for_eeprom_write=bool(payload.get("torque_disable_for_eeprom_write", True)),
            discovery_min_id=int(payload.get("discovery_min_id", 1)),
            discovery_max_id=int(payload.get("discovery_max_id", 20)),
            voltage_scale_mv_per_unit=float(payload.get("voltage_scale_mv_per_unit", 100.0)),
            current_scale_ma_per_unit=float(payload.get("current_scale_ma_per_unit", 1.0)),
            control_table=control_table,
        )


class DxlBus:
    """Low-level DYNAMIXEL communication interface."""

    def __init__(
        self,
        *,
        config: dict[str, Any] | DxlBusConfig | None = None,
        sdk_loader: Callable[[], Any] | None = None,
    ) -> None:
        self.config = config if isinstance(config, DxlBusConfig) else DxlBusConfig.from_dict(config)
        self._sdk_loader = sdk_loader or load_dynamixel_sdk
        self._sdk: Any | None = None
        self._packet_handler: Any | None = None
        self._port_handler: Any | None = None
        self._port: str | None = None
        self._baudrate: int | None = None

    @property
    def is_connected(self) -> bool:
        return self._port_handler is not None and self._port is not None

    @property
    def port(self) -> str | None:
        return self._port

    @property
    def baudrate(self) -> int | None:
        return self._baudrate

    def connect(self, port: str, baudrate: int) -> None:
        """Open the configured serial port through the Robotis SDK."""
        if not port:
            raise RuntimeError("OpenRB/DYNAMIXEL port is empty. Configure openrb_port before connecting.")
        if self.is_connected:
            if self._port == port and self._baudrate == int(baudrate):
                return
            self.disconnect()

        sdk = self._sdk_loader()
        port_handler = sdk.PortHandler(str(port))
        if not port_handler.openPort():
            raise RuntimeError(
                f"Could not open DYNAMIXEL port {port}. "
                "Check the USB serial path, cable, and permissions for the OpenRB device."
            )

        try:
            if not port_handler.setBaudRate(int(baudrate)):
                raise RuntimeError(
                    f"Could not set baudrate {baudrate} on {port}. "
                    "Verify the configured DYNAMIXEL baudrate matches the servos."
                )
        except Exception:
            port_handler.closePort()
            raise

        self._sdk = sdk
        self._packet_handler = sdk.PacketHandler(float(self.config.protocol_version))
        self._port_handler = port_handler
        self._port = str(port)
        self._baudrate = int(baudrate)

    def disconnect(self) -> None:
        """Close the DYNAMIXEL serial bus."""
        if self._port_handler is not None:
            try:
                self._port_handler.closePort()
            finally:
                self._packet_handler = None
                self._port_handler = None
                self._sdk = None
        self._port = None
        self._baudrate = None

    def scan_ids(self, min_id: int = 1, max_id: int = 20) -> list[int]:
        """Return discovered servo IDs via DYNAMIXEL ping."""
        self._require_connected()
        found: list[int] = []
        for servo_id in range(int(min_id), int(max_id) + 1):
            if self.ping_servo(int(servo_id)):
                found.append(servo_id)
        return found

    def ping_servo(self, servo_id: int) -> bool:
        """Return whether one servo ID responds on the live bus."""
        self._require_connected()
        return self.ping_servo_snapshot(int(servo_id)).responded

    def ping_servo_snapshot(self, servo_id: int) -> ServoPingResult:
        """Return raw ping status for one servo ID."""
        self._require_connected()
        try:
            model_number, comm_result, packet_error = self._packet_handler.ping(
                self._port_handler,
                int(servo_id),
            )
        except Exception as exc:
            return ServoPingResult(
                servo_id=int(servo_id),
                responded=False,
                error=f"ping failed: {exc}",
            )
        error = self._packet_error_message(comm_result, packet_error)
        if error is not None:
            return ServoPingResult(
                servo_id=int(servo_id),
                responded=False,
                error=error,
            )
        return ServoPingResult(
            servo_id=int(servo_id),
            responded=True,
            model_number=int(model_number),
            error=None,
        )

    def write_goal_positions(self, positions_by_id: dict[int, int]) -> None:
        """Send goal positions in ticks."""
        self._require_connected()
        goal_address = self.config.control_table["goal_position"]
        for servo_id, goal in positions_by_id.items():
            if self.config.auto_torque_enable_on_write:
                self._write1(servo_id, self.config.control_table["torque_enable"], 1, "torque enable")
            if self.config.default_profile_acceleration is not None:
                self._write4(
                    servo_id,
                    self.config.control_table["profile_acceleration"],
                    _to_uint32(int(self.config.default_profile_acceleration)),
                    "profile acceleration",
                )
            if self.config.default_profile_velocity is not None:
                self._write4(
                    servo_id,
                    self.config.control_table["profile_velocity"],
                    _to_uint32(int(self.config.default_profile_velocity)),
                    "profile velocity",
                )
            self._write4(servo_id, goal_address, _to_uint32(int(goal)), "goal position")

    def write_servo_id(self, current_id: int, new_id: int) -> None:
        """Assign a new servo ID."""
        self._require_connected()
        if int(current_id) == int(new_id):
            raise ValueError("Current servo ID and new servo ID must be different.")
        if int(new_id) <= 0 or int(new_id) > 252:
            raise ValueError("Servo ID must be between 1 and 252.")
        if not self._ping(int(current_id)):
            raise RuntimeError(f"Servo {current_id} did not respond on the bus.")
        if self._ping(int(new_id)):
            raise RuntimeError(f"Servo ID {new_id} is already in use.")
        torque_address = self.config.control_table["torque_enable"]
        torque_enabled_raw, torque_error = self._read1(int(current_id), torque_address)
        if torque_error is not None:
            raise RuntimeError(
                f"Failed to read Torque Enable for servo {current_id} before EEPROM write: {torque_error}"
            )
        if self.config.torque_disable_for_eeprom_write and torque_enabled_raw != 0:
            self._write1(int(current_id), torque_address, 0, "torque disable")
        torque_verify_raw, torque_verify_error = self._read1(int(current_id), torque_address)
        if torque_verify_error is not None:
            raise RuntimeError(
                f"Failed to verify Torque Enable for servo {current_id} before EEPROM write: {torque_verify_error}"
            )
        if torque_verify_raw != 0:
            raise RuntimeError(
                f"Torque Enable must be 0 before writing servo ID for servo {current_id}; "
                f"read back {torque_verify_raw}."
            )
        self._write1(int(current_id), self.config.control_table["servo_id"], int(new_id), "servo ID")
        if not self._ping(int(new_id)):
            raise RuntimeError(f"Servo {new_id} did not respond after ID assignment.")
        readback_id, readback_error = self._read1(int(new_id), self.config.control_table["servo_id"])
        if readback_error is not None:
            raise RuntimeError(
                f"Failed to verify servo ID readback for servo {new_id}: {readback_error}"
            )
        if readback_id != int(new_id):
            raise RuntimeError(
                f"Servo ID readback mismatch after assignment: expected {new_id}, got {readback_id}."
            )

    def read_live_telemetry(self, servo_ids: list[int]) -> dict[int, ServoTelemetry]:
        """Return the lighter-weight runtime telemetry subset used by the GUI bring-up loop."""
        return self.read_telemetry(servo_ids, include_identity=False, include_limits=False)

    def read_telemetry(
        self,
        servo_ids: list[int],
        *,
        include_identity: bool = True,
        include_limits: bool = True,
    ) -> dict[int, ServoTelemetry]:
        """Return telemetry map for requested IDs."""
        if not self.is_connected:
            return {
                sid: ServoTelemetry(servo_id=sid, hardware_error="disconnected")
                for sid in servo_ids
            }

        result: dict[int, ServoTelemetry] = {}
        read_time = time.monotonic()
        for servo_id in servo_ids:
            reported_id_raw, reported_id_error = self._read1(servo_id, self.config.control_table["servo_id"])
            model_raw: int | None = None
            model_error: str | None = None
            firmware_raw: int | None = None
            firmware_error: str | None = None
            if include_identity:
                model_raw, model_error = self._read2(servo_id, self.config.control_table["model_number"])
                firmware_raw, firmware_error = self._read1(servo_id, self.config.control_table["firmware_version"])
            operating_mode_raw, operating_mode_error = self._read1(servo_id, self.config.control_table["operating_mode"])
            torque_enabled_raw, torque_enabled_error = self._read1(servo_id, self.config.control_table["torque_enable"])
            position_raw, position_error = self._read4(servo_id, self.config.control_table["present_position"])
            current_raw, current_error = self._read2(servo_id, self.config.control_table["present_current"])
            voltage_raw, voltage_error = self._read2(servo_id, self.config.control_table["present_input_voltage"])
            temperature_raw, temperature_error = self._read1(servo_id, self.config.control_table["present_temperature"])
            hardware_raw, hardware_status_error = self._read1(
                servo_id, self.config.control_table["hardware_error_status"]
            )
            current_limit_raw: int | None = None
            current_limit_error: str | None = None
            max_limit_raw: int | None = None
            max_limit_error: str | None = None
            min_limit_raw: int | None = None
            min_limit_error: str | None = None
            watchdog_raw: int | None = None
            watchdog_error: str | None = None
            if include_limits:
                current_limit_raw, current_limit_error = self._read2(servo_id, self.config.control_table["current_limit"])
                max_limit_raw, max_limit_error = self._read4(servo_id, self.config.control_table["max_position_limit"])
                min_limit_raw, min_limit_error = self._read4(servo_id, self.config.control_table["min_position_limit"])
                watchdog_raw, watchdog_error = self._read1(servo_id, self.config.control_table["bus_watchdog"])

            identity_messages = [
                message
                for message in (
                    reported_id_error,
                    model_error,
                    firmware_error,
                )
                if message
            ]
            telemetry_messages = [
                message
                for message in (
                    position_error,
                    current_error,
                    voltage_error,
                    temperature_error,
                    operating_mode_error,
                    current_limit_error,
                    max_limit_error,
                    min_limit_error,
                    torque_enabled_error,
                    watchdog_error,
                    hardware_status_error,
                )
                if message
            ]
            if hardware_raw not in (None, 0):
                telemetry_messages.append(f"hardware_status=0x{int(hardware_raw):02X}")
            hardware_messages = [*identity_messages, *telemetry_messages]

            result[int(servo_id)] = ServoTelemetry(
                servo_id=int(servo_id),
                reported_servo_id=int(reported_id_raw) if reported_id_raw is not None else None,
                model_number=int(model_raw) if model_raw is not None else None,
                firmware_version=int(firmware_raw) if firmware_raw is not None else None,
                operating_mode=int(operating_mode_raw) if operating_mode_raw is not None else None,
                torque_enabled=(bool(torque_enabled_raw) if torque_enabled_raw is not None else None),
                current_limit_ma=(
                    int(round(_signed16(current_limit_raw) * self.config.current_scale_ma_per_unit))
                    if current_limit_raw is not None
                    else None
                ),
                min_position_limit=_signed32(min_limit_raw) if min_limit_raw is not None else None,
                max_position_limit=_signed32(max_limit_raw) if max_limit_raw is not None else None,
                bus_watchdog_value=int(watchdog_raw) if watchdog_raw is not None else None,
                present_position=_signed32(position_raw) if position_raw is not None else None,
                present_current_ma=(
                    int(round(_signed16(current_raw) * self.config.current_scale_ma_per_unit))
                    if current_raw is not None
                    else None
                ),
                present_voltage_mv=(
                    int(round(int(voltage_raw) * self.config.voltage_scale_mv_per_unit))
                    if voltage_raw is not None
                    else None
                ),
                present_temperature_c=int(temperature_raw) if temperature_raw is not None else None,
                hardware_error_code=int(hardware_raw) if hardware_raw is not None else None,
                hardware_error=" | ".join(hardware_messages) or None,
                identity_error=" | ".join(identity_messages) or None,
                telemetry_error=" | ".join(telemetry_messages) or None,
                last_read_monotonic_s=read_time,
            )
        return result

    def _require_connected(self) -> None:
        if not self.is_connected:
            raise RuntimeError("DYNAMIXEL bus is not connected")

    def _read1(self, servo_id: int, address: int) -> tuple[int | None, str | None]:
        return self._read("read1ByteTxRx", servo_id, address)

    def _read2(self, servo_id: int, address: int) -> tuple[int | None, str | None]:
        return self._read("read2ByteTxRx", servo_id, address)

    def _read4(self, servo_id: int, address: int) -> tuple[int | None, str | None]:
        return self._read("read4ByteTxRx", servo_id, address)

    def _read(self, method_name: str, servo_id: int, address: int) -> tuple[int | None, str | None]:
        try:
            value, comm_result, packet_error = getattr(self._packet_handler, method_name)(
                self._port_handler,
                int(servo_id),
                int(address),
            )
        except Exception as exc:
            return None, f"read 0x{int(address):02X} failed: {exc}"
        error = self._packet_error_message(comm_result, packet_error)
        return (int(value), None) if error is None else (None, error)

    def _write1(self, servo_id: int, address: int, value: int, label: str) -> None:
        self._write("write1ByteTxRx", servo_id, address, int(value) & 0xFF, label)

    def _write4(self, servo_id: int, address: int, value: int, label: str) -> None:
        self._write("write4ByteTxRx", servo_id, address, int(value) & 0xFFFFFFFF, label)

    def _write(self, method_name: str, servo_id: int, address: int, value: int, label: str) -> None:
        try:
            comm_result, packet_error = getattr(self._packet_handler, method_name)(
                self._port_handler,
                int(servo_id),
                int(address),
                int(value),
            )
        except Exception as exc:
            raise RuntimeError(f"Failed to write {label} for servo {servo_id}: {exc}") from exc
        error = self._packet_error_message(comm_result, packet_error)
        if error is not None:
            raise RuntimeError(f"Failed to write {label} for servo {servo_id}: {error}")

    def _packet_error_message(self, comm_result: int, packet_error: int) -> str | None:
        if comm_result != self._sdk.COMM_SUCCESS:
            return str(self._packet_handler.getTxRxResult(comm_result))
        if packet_error:
            return str(self._packet_handler.getRxPacketError(packet_error))
        return None


def _signed16(value: int) -> int:
    value = int(value) & 0xFFFF
    return value - 0x10000 if value & 0x8000 else value


def _signed32(value: int) -> int:
    value = int(value) & 0xFFFFFFFF
    return value - 0x100000000 if value & 0x80000000 else value


def _to_uint32(value: int) -> int:
    return int(value) & 0xFFFFFFFF

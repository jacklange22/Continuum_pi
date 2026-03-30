"""OpenRB-150 board-specific preparation and utility actions."""

from __future__ import annotations

from dataclasses import dataclass
import time
from typing import Any, Callable


def load_pyserial_class() -> type[Any]:
    """Import and return the pyserial Serial class."""
    try:
        import serial
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "pyserial is required for OpenRB serial validation. "
            "Install dependencies with `scripts/bootstrap.sh`."
        ) from exc
    return serial.Serial


@dataclass
class OpenRbClientConfig:
    """Machine-specific OpenRB validation behavior."""

    connect_timeout_s: float = 0.5
    port_settle_time_s: float = 0.15
    require_usb_to_dynamixel_firmware: bool = True
    require_external_power_for_motion: bool = True

    @classmethod
    def from_dict(cls, payload: dict[str, Any] | None = None) -> "OpenRbClientConfig":
        payload = dict(payload or {})
        return cls(
            connect_timeout_s=float(payload.get("connect_timeout_s", 0.5)),
            port_settle_time_s=float(payload.get("port_settle_time_s", 0.15)),
            require_usb_to_dynamixel_firmware=bool(
                payload.get("require_usb_to_dynamixel_firmware", True)
            ),
            require_external_power_for_motion=bool(payload.get("require_external_power_for_motion", True)),
        )


class OpenRbClient:
    """Validate that the OpenRB-150 serial device is reachable.

    The OpenRB board is expected to expose the Robotis `usb_to_dynamixel`
    firmware or equivalent DYNAMIXEL bridge behavior. This client intentionally
    does not keep the serial port open; the DYNAMIXEL SDK transport owns the
    port when the servo service connects.
    """

    def __init__(
        self,
        *,
        config: dict[str, Any] | OpenRbClientConfig | None = None,
        serial_factory: Callable[..., Any] | None = None,
        sleep_fn: Callable[[float], None] = time.sleep,
    ) -> None:
        self.config = config if isinstance(config, OpenRbClientConfig) else OpenRbClientConfig.from_dict(config)
        self._serial_factory = serial_factory
        self._sleep_fn = sleep_fn
        self._port: str | None = None
        self._baudrate: int | None = None
        self._last_status = "disconnected"
        self._prepared = False

    @property
    def is_connected(self) -> bool:
        return self._port is not None

    @property
    def is_prepared(self) -> bool:
        return self._prepared

    @property
    def last_status(self) -> str:
        return self._last_status

    def connect(self, port: str, baudrate: int = 115200) -> None:
        """Validate that the OpenRB serial device can be opened."""
        if not port:
            raise RuntimeError("OpenRB port is empty. Configure openrb_port before connecting.")
        self._probe_serial_port(port=port, baudrate=baudrate)
        self._port = str(port)
        self._baudrate = int(baudrate)
        self._prepared = False
        self._last_status = (
            f"validated {self._port} @ {self._baudrate} for OpenRB access; "
            f"{self._firmware_note()}. {self._power_note()}"
        )

    def disconnect(self) -> None:
        """Clear any cached status."""
        self._port = None
        self._baudrate = None
        self._prepared = False
        self._last_status = "disconnected"

    def prepare_for_dynamixel_use(self) -> bool:
        """Re-validate the board path and report the DYNAMIXEL pass-through assumption."""
        if not self.is_connected or self._port is None or self._baudrate is None:
            raise RuntimeError("OpenRB-150 is not connected. Connect the board first.")
        self._probe_serial_port(port=self._port, baudrate=self._baudrate)
        self._prepared = True
        self._last_status = (
            f"OpenRB ready for DYNAMIXEL pass-through on {self._port}; "
            "confirm the DXL power LED/switch state, external power path, and expected firmware before moving a servo."
        )
        return True

    def _probe_serial_port(self, *, port: str, baudrate: int) -> None:
        serial_factory = self._serial_factory or load_pyserial_class()
        handle = None
        try:
            handle = serial_factory(
                port=str(port),
                baudrate=int(baudrate),
                timeout=float(self.config.connect_timeout_s),
                write_timeout=float(self.config.connect_timeout_s),
            )
            if hasattr(handle, "reset_input_buffer"):
                handle.reset_input_buffer()
            if hasattr(handle, "reset_output_buffer"):
                handle.reset_output_buffer()
        except PermissionError as exc:
            raise RuntimeError(
                f"Permission denied while opening OpenRB port {port}. "
                "Check serial-device permissions for the current user."
            ) from exc
        except Exception as exc:
            raise RuntimeError(
                f"Could not open OpenRB port {port}: {exc}. "
                "Check the USB cable, selected serial device, and whether another process already owns the port."
            ) from exc
        finally:
            if handle is not None and hasattr(handle, "close"):
                handle.close()
        if self.config.port_settle_time_s > 0:
            self._sleep_fn(float(self.config.port_settle_time_s))

    def _firmware_note(self) -> str:
        if self.config.require_usb_to_dynamixel_firmware:
            return "expects the OpenRB usb_to_dynamixel bridge firmware"
        return "uses the configured OpenRB serial bridge mode"

    def _power_note(self) -> str:
        if self.config.require_external_power_for_motion:
            return "external power is recommended for dynamic servo testing; power state is not directly detectable over the serial bridge"
        return "power state is not directly detectable over the serial bridge"

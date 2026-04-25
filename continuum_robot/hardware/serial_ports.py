"""Serial port discovery helpers."""

from __future__ import annotations

from dataclasses import dataclass

from serial.tools import list_ports


@dataclass
class SerialPortInfo:
    """Simple serial port descriptor for GUI dropdowns."""

    device: str
    description: str = ""
    hwid: str = ""
    manufacturer: str = ""
    product: str = ""
    serial_number: str = ""
    interface: str = ""
    location: str = ""
    vid: int | None = None
    pid: int | None = None


def discover_serial_ports() -> list[SerialPortInfo]:
    """Return available serial ports sorted by device path."""
    ports = [
        SerialPortInfo(
            device=str(port.device),
            description=str(port.description or ""),
            hwid=str(getattr(port, "hwid", "") or ""),
            manufacturer=str(getattr(port, "manufacturer", "") or ""),
            product=str(getattr(port, "product", "") or ""),
            serial_number=str(getattr(port, "serial_number", "") or ""),
            interface=str(getattr(port, "interface", "") or ""),
            location=str(getattr(port, "location", "") or ""),
            vid=(int(port.vid) if getattr(port, "vid", None) is not None else None),
            pid=(int(port.pid) if getattr(port, "pid", None) is not None else None),
        )
        for port in list_ports.comports()
    ]
    return sorted(ports, key=lambda port: port.device)

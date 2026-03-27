"""Serial port discovery helpers."""

from dataclasses import dataclass

from serial.tools import list_ports


@dataclass
class SerialPortInfo:
    """Simple serial port descriptor for GUI dropdowns."""

    device: str
    description: str = ""


def discover_serial_ports() -> list[SerialPortInfo]:
    """Return available serial ports sorted by device path."""
    ports = [
        SerialPortInfo(device=str(port.device), description=str(port.description or ""))
        for port in list_ports.comports()
    ]
    return sorted(ports, key=lambda port: port.device)

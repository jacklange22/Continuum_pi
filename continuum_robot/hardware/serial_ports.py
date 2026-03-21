"""Serial port discovery helpers."""

from dataclasses import dataclass


@dataclass
class SerialPortInfo:
    """Simple serial port descriptor for GUI dropdowns."""

    device: str
    description: str = ""

"""Typed config schemas used across the codebase."""

from dataclasses import dataclass, field


@dataclass
class RobotConfig:
    """Robot hardware configuration."""

    mode: str = "4-servo"
    spool_diameter_cm: float = 1.2
    servo_ids: list[int] = field(default_factory=lambda: [1, 2, 3, 4])
    tendon_to_servo: list[int] = field(default_factory=lambda: [1, 2, 3, 4])


@dataclass
class SerialConfig:
    """Serial defaults for Aurora and OpenRB."""

    aurora_port: str = ""
    openrb_port: str = ""
    baudrate: int = 115200
    tracker_socket_path: str = "/tmp/tracker_bridge.sock"
    tracker_bridge_executable: str = "bin/tracker_bridge"
    tracker_poll_ms: int = 20


@dataclass
class SafetyConfig:
    """Safety limits for servo command validation."""

    position_min_offset_ticks: int = -600
    position_max_offset_ticks: int = 600
    max_current_ma: int = 850

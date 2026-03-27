"""Typed config schemas used across the codebase."""

from dataclasses import dataclass, field


@dataclass
class RuntimeConfig:
    """Application runtime configuration."""

    mock_mode: bool = True
    poll_rate_hz: int = 5
    robot_config: str = "robot_4servo.yaml"


@dataclass
class RobotConfig:
    """Robot hardware configuration."""

    mode: str = "4-servo"
    spool_diameter_cm: float = 1.2
    ticks_per_revolution: int = 4096
    servo_ids: list[int] = field(default_factory=lambda: [1, 2, 3, 4])
    tendon_to_servo: list[int] = field(default_factory=lambda: [1, 2, 3, 4])


@dataclass
class SerialConfig:
    """Serial defaults for Aurora and OpenRB."""

    aurora_port: str = ""
    openrb_port: str = ""
    baudrate: int = 115200
    read_timeout_s: float = 0.05
    frame_timeout_s: float = 0.5
    reconnect_delay_s: float = 1.0
    tracker_socket_path: str = "/tmp/tracker_bridge.sock"
    tracker_bridge_executable: str = "bin/tracker_bridge"
    tracker_poll_ms: int = 20
    packet_capture_dir: str = "data/tracker_captures"


@dataclass
class SafetyConfig:
    """Safety limits for servo command validation."""

    position_min_offset_ticks: int = -600
    position_max_offset_ticks: int = 600
    max_current_ma: int = 850
    pretension_current_balance_tolerance_ma: int = 120


@dataclass
class RegistrationWorkflowConfig:
    """Registration workflow defaults."""

    landmark_labels: list[str] = field(default_factory=lambda: ["L1", "L2", "L3", "L4"])
    captures_per_landmark: int = 5
    nominal_landmarks_robot_xyz_mm: dict[str, list[float]] = field(default_factory=dict)
    capture_tool_id: str = "0B"
    capture_tool_tip_transform: list[list[float]] | None = None
    max_fre_mm: float | None = 2.0


@dataclass
class ExperimentConfig:
    """Experiment execution defaults."""

    default_settle_time_s: float = 2.0
    sample_count_per_point: int = 1
    output_dir: str = "data/runs"


@dataclass
class CalibrationConfig:
    """Persistent calibration file locations."""

    neutral_setpoints_path: str = "data/calibrations/neutral_setpoints.json"
    latest_registration_path: str = "data/registrations/latest_registration.json"

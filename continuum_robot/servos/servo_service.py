"""High-level servo command service."""

from continuum_robot.hardware.dxl_bus import DxlBus
from continuum_robot.servos.displacement_mapper import TendonDisplacementMapper
from continuum_robot.servos.safety_guard import SafetyGuard


class ServoService:
    """Coordinates mapping, validation, and low-level bus writes."""

    def __init__(
        self,
        dxl_bus: DxlBus,
        mapper: TendonDisplacementMapper,
        safety_guard: SafetyGuard,
    ) -> None:
        self.dxl_bus = dxl_bus
        self.mapper = mapper
        self.safety_guard = safety_guard

    def command_displacement(
        self,
        tendon_displacements_cm: list[float],
        neutral_ticks: list[int],
        servo_ids: list[int],
    ) -> dict[int, int]:
        """Compute and send safe goal position ticks."""
        goals = self.mapper.to_goal_positions(tendon_displacements_cm, neutral_ticks)
        self.safety_guard.validate_positions(goals, neutral_ticks)
        payload = {sid: goal for sid, goal in zip(servo_ids, goals)}
        self.dxl_bus.write_goal_positions(payload)
        return payload

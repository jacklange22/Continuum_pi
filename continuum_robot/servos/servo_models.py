"""Servo domain models."""

from dataclasses import dataclass


@dataclass
class ServoCommand:
    """Target command in ticks for a servo ID."""

    servo_id: int
    goal_position_ticks: int

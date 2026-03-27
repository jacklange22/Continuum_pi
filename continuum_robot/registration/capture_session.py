"""Guided registration capture session state."""

from dataclasses import dataclass, field
from typing import Any


@dataclass
class RegistrationSession:
    """Tracks progress through landmark capture workflow."""

    labels: list[str]
    captures_per_landmark: int
    raw_points_by_label: dict[str, list[list[float]]] = field(default_factory=dict)
    raw_measurement_tool_samples_by_label: dict[str, list[dict[str, Any]]] = field(default_factory=dict)
    raw_coil_samples_by_label: dict[str, list[dict[str, Any]]] = field(default_factory=dict)
    group_by_label: dict[str, str] = field(default_factory=dict)
    truth_points_in_sw_by_label: dict[str, list[float]] = field(default_factory=dict)
    measurement_tool_id: str = "0B"
    coil_tool_id: str = "0A"

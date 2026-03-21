"""Guided registration capture session state."""

from dataclasses import dataclass, field


@dataclass
class RegistrationSession:
    """Tracks progress through landmark capture workflow."""

    labels: list[str]
    captures_per_landmark: int
    raw_points_by_label: dict[str, list[list[float]]] = field(default_factory=dict)

"""Aurora tool measurement models."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class AuroraToolMeasurement:
    """Parsed Aurora tool transform sample.

    Convention:
    - ``T_aurora_tool`` transforms coordinates from tool frame into Aurora frame.
    """

    tool_id: str
    quat_wxyz: tuple[float, float, float, float]
    translation_xyz: tuple[float, float, float]
    quality: float | None
    status_byte: int
    valid: bool
    status_text: str

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
    quality: float | None = None
    tool_sn: int | None = None
    status_byte: int | None = None
    valid: bool = True
    status_text: str = "status_not_available_in_transform_record"

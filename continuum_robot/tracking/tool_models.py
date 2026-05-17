"""Aurora tool measurement models.

These models are used by the legacy client-packet compatibility path, not the
primary live Aurora runtime path.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class AuroraToolMeasurement:
    """Parsed Aurora tool transform sample.

    Convention:
    - ``T_aurora_tool`` transforms coordinates from tool frame into Aurora frame.

    Fields:
    - ``frame_number`` is the per-packet frame counter from the parent header,
      propagated onto every tool record from the same packet. It is useful for
      replay/regression debugging and for spotting dropped frames in trace logs.
    - ``status_byte`` is left ``None`` for the legacy DLE/STX/ETX packet format,
      because that protocol does not carry a per-tool status byte. The live NDI
      runtime path has its own ``tracking_state`` field for that purpose.
    - ``valid`` is a best-effort flag derived from ``quality`` (see
      ``aurora_parser.AuroraParser``); ``status_text`` carries a one-line
      human-readable explanation that mirrors the validity decision.
    """

    tool_id: str
    quat_wxyz: tuple[float, float, float, float]
    translation_xyz: tuple[float, float, float]
    quality: float | None = None
    tool_sn: int | None = None
    status_byte: int | None = None
    valid: bool | None = None
    status_text: str = "status_not_available_in_transform_record"
    frame_number: int | None = None

"""Message models and parsing for the legacy tracker_bridge JSON stream."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any
import json


@dataclass
class TrackerStatusMessage:
    """Structured status event emitted by the legacy bridge."""

    timestamp: str
    level: str
    state: str
    message: str
    details: dict[str, Any]


@dataclass
class TrackerTransformMessage:
    """One tool transform sample emitted by the legacy bridge."""

    timestamp: str
    frame_number: int
    tool_id: str
    valid: bool
    status: str
    quaternion: tuple[float, float, float, float]
    translation_mm: tuple[float, float, float]
    quality: float | None


def parse_tracker_json_line(line: str) -> TrackerStatusMessage | TrackerTransformMessage:
    """Parse one line-delimited JSON message from the legacy bridge."""
    raw = json.loads(line)
    if not isinstance(raw, dict):
        raise ValueError("tracker message must be a JSON object")

    msg_type = raw.get("type")
    if msg_type == "status":
        details = raw.get("details", {})
        if not isinstance(details, dict):
            raise ValueError("status.details must be an object")
        return TrackerStatusMessage(
            timestamp=str(raw.get("timestamp", "")),
            level=str(raw.get("level", "info")),
            state=str(raw.get("state", "unknown")),
            message=str(raw.get("message", "")),
            details=details,
        )

    if msg_type == "transform":
        quat = raw.get("quaternion")
        trans = raw.get("translation_mm")
        if not (isinstance(quat, list) and len(quat) == 4):
            raise ValueError("transform.quaternion must be a length-4 array")
        if not (isinstance(trans, list) and len(trans) == 3):
            raise ValueError("transform.translation_mm must be a length-3 array")
        return TrackerTransformMessage(
            timestamp=str(raw.get("timestamp", "")),
            frame_number=int(raw.get("frame_number", 0)),
            tool_id=str(raw.get("tool_id", "")),
            valid=bool(raw.get("valid", False)),
            status=str(raw.get("status", "unknown")),
            quaternion=(float(quat[0]), float(quat[1]), float(quat[2]), float(quat[3])),
            translation_mm=(float(trans[0]), float(trans[1]), float(trans[2])),
            quality=float(raw["quality"]) if raw.get("quality") is not None else None,
        )

    raise ValueError(f"unknown tracker message type: {msg_type}")

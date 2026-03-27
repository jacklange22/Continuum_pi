import json

import pytest

from continuum_robot.tracking.tracker_protocol import (
    TrackerStatusMessage,
    TrackerTransformMessage,
    parse_tracker_json_line,
)


def test_parse_status_message() -> None:
    line = json.dumps(
        {
            "type": "status",
            "timestamp": "2026-01-01T00:00:00.000Z",
            "level": "info",
            "state": "tracking_started",
            "message": "Tracking started",
            "details": {"count": 2},
        }
    )
    msg = parse_tracker_json_line(line)
    assert isinstance(msg, TrackerStatusMessage)
    assert msg.state == "tracking_started"
    assert msg.details["count"] == 2


def test_parse_transform_message() -> None:
    line = json.dumps(
        {
            "type": "transform",
            "timestamp": "2026-01-01T00:00:00.000Z",
            "frame_number": 42,
            "tool_id": "0A",
            "valid": True,
            "status": "tracked",
            "quaternion": [1.0, 0.0, 0.0, 0.0],
            "translation_mm": [10.0, 20.0, 30.0],
            "quality": 0.12,
        }
    )
    msg = parse_tracker_json_line(line)
    assert isinstance(msg, TrackerTransformMessage)
    assert msg.frame_number == 42
    assert msg.translation_mm == (10.0, 20.0, 30.0)


def test_parse_unknown_message_type_raises() -> None:
    with pytest.raises(ValueError, match="unknown tracker message type"):
        parse_tracker_json_line('{"type":"other"}')

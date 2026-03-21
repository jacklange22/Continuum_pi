"""Synthetic Aurora packet fixtures for parser/framer tests."""

from __future__ import annotations

import struct

from continuum_robot.tracking.aurora_packet import (
    TOOL_RECORD_SIZE,
    build_transform_payload,
    frame_payload,
)


def build_tool_record(
    tool_id: str,
    status_byte: int,
    quat_wxyz: tuple[float, float, float, float],
    translation_xyz: tuple[float, float, float],
    quality: float,
) -> bytes:
    if len(tool_id) != 2:
        raise ValueError("tool_id must be exactly two characters")

    values = [*quat_wxyz, *translation_xyz, quality]
    body = bytearray()
    body.extend(tool_id.encode("ascii"))
    body.append(status_byte & 0xFF)
    body.append(0)
    for value in values:
        body.extend(struct.pack("<f", float(value)))

    if len(body) != TOOL_RECORD_SIZE:
        raise AssertionError(f"Expected {TOOL_RECORD_SIZE} bytes, got {len(body)}")
    return bytes(body)


def build_valid_transform_frame(frame_number: int = 123) -> bytes:
    records = b"".join(
        [
            build_tool_record("0A", 0, (1.0, 0.0, 0.0, 0.0), (10.0, 20.0, 30.0), 0.01),
            build_tool_record("0B", 1, (1.0, 0.0, 0.0, 0.0), (11.0, 21.0, 31.0), 0.02),
        ]
    )
    payload = build_transform_payload(frame_number=frame_number, records_blob=records)
    return frame_payload(payload)

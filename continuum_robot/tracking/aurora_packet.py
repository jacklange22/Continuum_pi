"""Aurora packet models and CRC/framing helpers.

Packet assumptions are based on observed behavior from legacy code and may need
adjustment on real hardware captures:
- Framing: DLE STX <stuffed payload> DLE ETX
- Payload format for transform packets:
  - byte 0: packet type (0x01)
  - byte 1: number of tool records
  - bytes 2..5: frame number uint32 little-endian
  - bytes 6..N-2: tool records (36 bytes each)
  - byte N-1: CRC-8 over payload without CRC byte
- Tool record layout (36 bytes):
  - bytes 0..3: tool serial number uint32 little-endian
    - convention used here: low two bytes may decode to display ids like "0A"/"0B"
    - uncertainty: this display-id mapping is an implementation convenience, not a
      guaranteed protocol rule
  - bytes 4..35: 8 float32 little-endian values
    - convention used here: quat[4], translation[3], tracker_error
    - uncertainty: final float naming can vary across legacy scripts ("quality"/"error")
"""

from __future__ import annotations

from dataclasses import dataclass


DLE = 0x10
STX = 0x02
ETX = 0x03
TRANSFORM_PACKET_TYPE = 0x01
TOOL_RECORD_SIZE = 36
SUPPORTED_TOOL_IDS = {"0A", "0B"}


@dataclass
class AuroraPacketHeader:
    """Header fields extracted from Aurora payload."""

    packet_type: int
    tool_count: int
    frame_number: int


@dataclass
class AuroraTransformPayload:
    """Decoded payload with header and raw record bytes."""

    header: AuroraPacketHeader
    raw_records: bytes
    crc_received: int
    crc_computed: int


def crc8(payload: bytes, seed: int = 0) -> int:
    """Compute Aurora CRC-8 using polynomial 0x07 and bit order from legacy code."""
    crc = seed & 0xFF
    for byte in payload:
        for bit in range(8, 0, -1):
            this_bit = (byte >> (bit - 1)) & 1
            do_invert = (this_bit ^ ((crc & 128) >> 7)) == 1
            crc = (crc << 1) & 0xFF
            if do_invert:
                crc ^= 0x07
    return crc


def stuff_dle(payload: bytes) -> bytes:
    """Duplicate DLE bytes for framed payload transport."""
    out = bytearray()
    for byte in payload:
        out.append(byte)
        if byte == DLE:
            out.append(DLE)
    return bytes(out)


def frame_payload(payload: bytes) -> bytes:
    """Build a framed packet from unstuffed payload bytes."""
    return bytes([DLE, STX]) + stuff_dle(payload) + bytes([DLE, ETX])


def build_transform_payload(frame_number: int, records_blob: bytes) -> bytes:
    """Build unstuffed transform payload with CRC.

    Used by synthetic packet tests/fixtures.
    """
    if len(records_blob) % TOOL_RECORD_SIZE != 0:
        raise ValueError("records_blob must be a multiple of 36 bytes")
    tool_count = len(records_blob) // TOOL_RECORD_SIZE
    payload_wo_crc = bytes([TRANSFORM_PACKET_TYPE, tool_count]) + int(frame_number).to_bytes(4, "little") + records_blob
    crc = crc8(payload_wo_crc)
    return payload_wo_crc + bytes([crc])

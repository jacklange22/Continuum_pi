"""Aurora transform packet parser."""

from __future__ import annotations

import struct

from continuum_robot.tracking.aurora_framer import extract_payload_from_frame
from continuum_robot.tracking.aurora_packet import (
    SUPPORTED_TOOL_IDS,
    TOOL_RECORD_SIZE,
    TRANSFORM_PACKET_TYPE,
    AuroraPacketHeader,
    AuroraTransformPayload,
    crc8,
)
from continuum_robot.tracking.tool_models import AuroraToolMeasurement


class AuroraParser:
    """Parse framed Aurora packets into tool measurements for 0A/0B."""

    def parse_transform_packet(self, framed_packet: bytes) -> dict[str, AuroraToolMeasurement]:
        """Parse one framed transform packet.

        Raises ValueError when framing, CRC, or payload layout is invalid.
        """
        payload = self.parse_payload(framed_packet)
        records = self._parse_records(payload.raw_records, payload.header.tool_count)

        filtered: dict[str, AuroraToolMeasurement] = {}
        for tool_id, measurement in records.items():
            if tool_id in SUPPORTED_TOOL_IDS:
                filtered[tool_id] = measurement
        return filtered

    def parse_payload(self, framed_packet: bytes) -> AuroraTransformPayload:
        """Parse framed bytes into a validated transform payload model."""
        payload = extract_payload_from_frame(framed_packet)
        if len(payload) < 7:
            raise ValueError("Payload too short")

        packet_type = payload[0]
        if packet_type != TRANSFORM_PACKET_TYPE:
            raise ValueError(f"Unsupported packet type: 0x{packet_type:02X}")

        tool_count = payload[1]
        frame_number = int.from_bytes(payload[2:6], "little")

        raw_records_end = 6 + tool_count * TOOL_RECORD_SIZE
        if len(payload) != raw_records_end + 1:
            raise ValueError(
                f"Unexpected payload length {len(payload)} for tool_count={tool_count}; expected {raw_records_end + 1}"
            )

        raw_records = payload[6:raw_records_end]
        crc_received = payload[-1]
        crc_computed = crc8(payload[:-1])
        if crc_received != crc_computed:
            raise ValueError(
                f"CRC mismatch: received=0x{crc_received:02X}, computed=0x{crc_computed:02X}"
            )

        header = AuroraPacketHeader(
            packet_type=packet_type,
            tool_count=tool_count,
            frame_number=frame_number,
        )
        return AuroraTransformPayload(
            header=header,
            raw_records=raw_records,
            crc_received=crc_received,
            crc_computed=crc_computed,
        )

    def _parse_records(self, raw_records: bytes, tool_count: int) -> dict[str, AuroraToolMeasurement]:
        output: dict[str, AuroraToolMeasurement] = {}
        for idx in range(tool_count):
            start = idx * TOOL_RECORD_SIZE
            record = raw_records[start : start + TOOL_RECORD_SIZE]
            if len(record) != TOOL_RECORD_SIZE:
                raise ValueError("Incomplete tool record")

            tool_sn = int.from_bytes(record[0:4], "little")
            tool_id = record[0:2].decode("ascii", errors="replace")

            vals = [
                struct.unpack("<f", record[4 + 4 * i : 8 + 4 * i])[0]
                for i in range(8)
            ]
            quat = (float(vals[0]), float(vals[1]), float(vals[2]), float(vals[3]))
            trans = (float(vals[4]), float(vals[5]), float(vals[6]))
            quality = float(vals[7])

            output[tool_id] = AuroraToolMeasurement(
                tool_id=tool_id,
                quat_wxyz=quat,
                translation_xyz=trans,
                quality=quality,
                tool_sn=tool_sn,
                status_byte=None,
                valid=True,
                status_text="status_not_available_in_transform_record",
            )
        return output

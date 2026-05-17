"""Legacy client-packet Aurora transform parser.

This parser is retained for replay, regression tests, and compatibility with
the historical DLE/STX/ETX client packet format. It is not the primary live
Aurora runtime path for the app.
"""

from __future__ import annotations

import math
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


# Default cutoff for treating the per-record quality/tracker-error float as
# "tracked". The legacy packet's 8th float is documented as either "quality" or
# "tracker_error" across historical scripts -- both conventions encode small
# positive values for good tracking. NaN/inf or very large values indicate
# missing or out-of-volume tracking. The default 0.5 (mm-equivalent) is a
# conservative cutoff that matches what the legacy MATLAB scripts used; the
# operator can tighten it per deployment via ``AuroraParser(quality_invalid_threshold=...)``.
DEFAULT_QUALITY_INVALID_THRESHOLD = 0.5


class AuroraParser:
    """Parse legacy framed packets into tool measurements for 0A/0B.

    The legacy packet does not carry a per-tool NDI BX status byte; instead,
    each tool record ends with a single float that historical code variously
    calls "quality" or "tracker_error". Because the semantic of that float
    depends on a convention the operator must confirm, the parser leaves
    ``valid=None`` by default to keep the legacy "validity unknown" contract.

    Pass ``derive_validity_from_quality=True`` to opt into a best-effort
    heuristic that flags records whose quality exceeds
    ``quality_invalid_threshold`` (or is NaN/inf/negative) as ``valid=False``.
    The status_byte field on the resulting measurement is always left ``None``
    because the protocol does not carry one; the ``status_text`` field always
    carries a one-line summary that mirrors the validity decision.
    """

    def __init__(
        self,
        *,
        derive_validity_from_quality: bool = False,
        quality_invalid_threshold: float = DEFAULT_QUALITY_INVALID_THRESHOLD,
    ) -> None:
        if not (quality_invalid_threshold > 0.0):
            raise ValueError("quality_invalid_threshold must be positive")
        self.derive_validity_from_quality = bool(derive_validity_from_quality)
        self.quality_invalid_threshold = float(quality_invalid_threshold)

    def parse_transform_packet(self, framed_packet: bytes) -> dict[str, AuroraToolMeasurement]:
        """Parse one framed transform packet.

        Raises ValueError when framing, CRC, or payload layout is invalid.
        """
        payload = self.parse_payload(framed_packet)
        records = self._parse_records(
            payload.raw_records,
            payload.header.tool_count,
            frame_number=payload.header.frame_number,
        )

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

    def _parse_records(
        self,
        raw_records: bytes,
        tool_count: int,
        *,
        frame_number: int,
    ) -> dict[str, AuroraToolMeasurement]:
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
            if self.derive_validity_from_quality:
                valid, status_text = self._validity_from_quality(quality)
            else:
                valid, status_text = None, "validity_not_available_in_compatibility_packet"

            output[tool_id] = AuroraToolMeasurement(
                tool_id=tool_id,
                quat_wxyz=quat,
                translation_xyz=trans,
                quality=quality,
                tool_sn=tool_sn,
                status_byte=None,
                valid=valid,
                status_text=status_text,
                frame_number=int(frame_number),
            )
        return output

    def _validity_from_quality(self, quality: float) -> tuple[bool, str]:
        """Derive a best-effort ``(valid, status_text)`` from the per-record quality float.

        - NaN or infinite: ``valid=False`` (sensor reported a malformed value).
        - Negative or very large: ``valid=False`` (out of expected range,
          consistent with NDI "out of volume" or "missing").
        - Otherwise: ``valid=True`` with the quality value surfaced for audit.
        """
        if math.isnan(quality) or math.isinf(quality):
            return False, f"invalid_quality_not_finite: {quality!r}"
        if quality < 0.0:
            return False, f"invalid_quality_negative: {quality:.6f}"
        if quality > self.quality_invalid_threshold:
            return (
                False,
                f"invalid_quality_above_threshold: {quality:.6f} > {self.quality_invalid_threshold:.6f}",
            )
        return True, f"tracked_quality_below_threshold: {quality:.6f} <= {self.quality_invalid_threshold:.6f}"

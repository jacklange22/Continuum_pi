"""Aurora stream framer utilities."""

from __future__ import annotations

import time

from continuum_robot.tracking.aurora_packet import DLE, ETX, STX


def unstuff_dle_payload(stuffed_payload: bytes) -> bytes:
    """Remove DLE byte stuffing from payload bytes.

    In payload transport, DLE is encoded as DLE DLE.
    """
    out = bytearray()
    i = 0
    while i < len(stuffed_payload):
        current = stuffed_payload[i]
        out.append(current)
        if current == DLE and i + 1 < len(stuffed_payload) and stuffed_payload[i + 1] == DLE:
            i += 1
        i += 1
    return bytes(out)


def extract_payload_from_frame(frame: bytes) -> bytes:
    """Validate a framed packet and return unstuffed payload."""
    if len(frame) < 4:
        raise ValueError("Frame is too short")
    if frame[0] != DLE or frame[1] != STX:
        raise ValueError("Frame does not start with DLE STX")
    if frame[-2] != DLE or frame[-1] != ETX:
        raise ValueError("Frame does not end with DLE ETX")
    stuffed_payload = frame[2:-2]
    return unstuff_dle_payload(stuffed_payload)


class AuroraFramer:
    """Reads framed Aurora packets from a byte stream callback."""

    def __init__(self) -> None:
        self._buffer = bytearray()

    def read_next_frame(self, read_fn, timeout_s: float) -> bytes:
        """Read bytes until a full DLE/STX..DLE/ETX frame is found."""
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            frame = self.try_pop_frame()
            if frame is not None:
                return frame

            chunk = read_fn(1)
            if chunk:
                self._buffer.extend(chunk)
                continue
            time.sleep(0.001)

        raise TimeoutError(f"Timed out waiting for Aurora frame after {timeout_s:.3f}s")

    def try_pop_frame(self) -> bytes | None:
        """Return one frame from internal buffer when available."""
        data = self._buffer
        n = len(data)
        if n < 4:
            return None

        start = self._find_start_index(data)
        if start is None:
            # Keep at most one trailing DLE as possible start prefix.
            if data and data[-1] == DLE:
                self._buffer = bytearray([DLE])
            else:
                self._buffer = bytearray()
            return None

        # Drop noise before valid start marker.
        if start > 0:
            del data[:start]
            n = len(data)

        # Search for frame terminator DLE ETX, ignoring stuffed DLE DLE.
        i = 2
        while i < n - 1:
            if data[i] == DLE:
                nxt = data[i + 1]
                if nxt == ETX:
                    frame = bytes(data[: i + 2])
                    del data[: i + 2]
                    return frame
                if nxt == DLE:
                    i += 2
                    continue
            i += 1

        return None

    @staticmethod
    def _find_start_index(data: bytearray) -> int | None:
        for i in range(0, len(data) - 1):
            if data[i] == DLE and data[i + 1] == STX:
                return i
        return None

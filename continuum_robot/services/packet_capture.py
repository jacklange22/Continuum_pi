"""Aurora packet capture and replay utilities."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Iterator


@dataclass
class PacketCaptureRecord:
    """One raw Aurora frame plus optional parsed metadata."""

    captured_at_utc: str
    source: str
    frame_hex: str
    summary: dict | None = None
    parse_error: str | None = None

    @property
    def frame_bytes(self) -> bytes:
        return bytes.fromhex(self.frame_hex)


class PacketCaptureWriter:
    """Append-only JSONL writer for raw Aurora frame capture."""

    def __init__(self, path: Path, include_summary: bool = True) -> None:
        self.path = path
        self.include_summary = include_summary
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._handle = self.path.open("a", encoding="utf-8")

    def write_frame(
        self,
        frame: bytes,
        *,
        captured_at_utc: str,
        source: str,
        summary: dict | None = None,
        parse_error: str | None = None,
    ) -> None:
        payload = {
            "captured_at_utc": captured_at_utc,
            "source": source,
            "frame_hex": frame.hex(),
        }
        if self.include_summary and summary is not None:
            payload["summary"] = summary
        if parse_error is not None:
            payload["parse_error"] = parse_error
        self._handle.write(json.dumps(payload) + "\n")
        self._handle.flush()

    def close(self) -> None:
        self._handle.close()


def iter_packet_capture_records(path: Path) -> Iterator[PacketCaptureRecord]:
    """Yield packet capture records from a JSONL capture file."""
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            raw = json.loads(line)
            yield PacketCaptureRecord(
                captured_at_utc=str(raw["captured_at_utc"]),
                source=str(raw.get("source", "unknown")),
                frame_hex=str(raw["frame_hex"]),
                summary=raw.get("summary"),
                parse_error=raw.get("parse_error"),
            )


def load_packet_capture_records(path: Path) -> list[PacketCaptureRecord]:
    """Load all packet capture records from a JSONL capture file."""
    return list(iter_packet_capture_records(path))

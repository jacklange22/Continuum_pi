from pathlib import Path

from continuum_robot.services.packet_capture import PacketCaptureWriter, load_packet_capture_records
from tests.fixtures.aurora_samples import build_valid_transform_frame


def test_packet_capture_writer_round_trips_frame_and_summary(tmp_path: Path) -> None:
    capture_path = tmp_path / "capture.jsonl"
    writer = PacketCaptureWriter(capture_path, include_summary=True)
    frame = build_valid_transform_frame(frame_number=55)
    writer.write_frame(
        frame,
        captured_at_utc="2026-01-01T00:00:00.000Z",
        source="serial",
        summary={"frame_number": 55, "tool_ids": ["0A", "0B"]},
    )
    writer.close()

    records = load_packet_capture_records(capture_path)
    assert len(records) == 1
    assert records[0].frame_bytes == frame
    assert records[0].summary == {"frame_number": 55, "tool_ids": ["0A", "0B"]}

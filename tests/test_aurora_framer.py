from continuum_robot.tracking.aurora_framer import (
    AuroraFramer,
    extract_payload_from_frame,
    unstuff_dle_payload,
)
from continuum_robot.tracking.aurora_packet import DLE, STX, ETX


def test_unstuff_dle_payload_removes_stuffed_dle() -> None:
    stuffed = bytes([0x01, DLE, DLE, 0x02])
    assert unstuff_dle_payload(stuffed) == bytes([0x01, DLE, 0x02])


def test_extract_payload_from_frame_validates_markers() -> None:
    payload = bytes([0xAA, 0xBB])
    frame = bytes([DLE, STX]) + payload + bytes([DLE, ETX])
    assert extract_payload_from_frame(frame) == payload


def test_framer_recovers_frame_after_noise_and_stuffed_dle() -> None:
    framer = AuroraFramer()
    data = bytearray([0x99, 0x88, DLE, STX, 0x01, DLE, DLE, 0x22, DLE, ETX])

    def read_fn(_n: int) -> bytes:
        if not data:
            return b""
        return bytes([data.pop(0)])

    frame = framer.read_next_frame(read_fn, timeout_s=0.5)
    assert frame[0] == DLE
    assert frame[1] == STX
    assert frame[-2] == DLE
    assert frame[-1] == ETX


def test_find_start_index_rejects_even_dle_run_before_stx() -> None:
    data = bytearray([0x99, DLE, DLE, STX, 0x01, DLE, ETX])
    assert AuroraFramer._find_start_index(data) is None


def test_find_start_index_accepts_odd_dle_run_before_stx() -> None:
    data = bytearray([0x99, DLE, DLE, DLE, STX, 0x01, DLE, ETX])
    assert AuroraFramer._find_start_index(data) == 3

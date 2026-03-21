from continuum_robot.tracking.aurora_parser import AuroraParser
from continuum_robot.tracking.aurora_packet import crc8
from tests.fixtures.aurora_samples import build_valid_transform_frame


def test_parser_parses_valid_packet_and_filters_to_0A_0B() -> None:
    parser = AuroraParser()
    frame = build_valid_transform_frame(frame_number=7)
    out = parser.parse_transform_packet(frame)

    assert set(out.keys()) == {"0A", "0B"}
    assert out["0A"].valid is True
    assert out["0B"].valid is False
    assert out["0A"].translation_xyz == (10.0, 20.0, 30.0)


def test_parser_rejects_crc_mismatch() -> None:
    parser = AuroraParser()
    frame = bytearray(build_valid_transform_frame())
    frame[-3] ^= 0xFF
    try:
        parser.parse_transform_packet(bytes(frame))
    except ValueError as exc:
        assert "CRC mismatch" in str(exc)
    else:
        raise AssertionError("Expected CRC mismatch error")


def test_crc8_known_value() -> None:
    payload = bytes([0x01, 0x02, 0x03, 0x04])
    assert crc8(payload) == 0xE3

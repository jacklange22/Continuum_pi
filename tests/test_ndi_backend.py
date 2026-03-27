import numpy as np

from continuum_robot.tracking.ndi_backend import TrackerBackendNDI, normalize_tool_id


def test_normalize_tool_id_maps_expected_handles() -> None:
    assert normalize_tool_id(b"0A\x00") == "0A"
    assert normalize_tool_id("port-0b") == "0B"
    assert normalize_tool_id(" 0A ") == "0A"


def test_ndi_backend_parses_tracked_and_missing_tools() -> None:
    backend = TrackerBackendNDI("/dev/ttyUSB0", tracker_factory=lambda _settings: object())
    frame_payload = (
        ["0A", "Port-0B"],
        [1710000000.0, 1710000000.1],
        [12, 12],
        [
            np.eye(4),
            None,
        ],
        [0.15, 0.22],
    )

    tools, latest_frame = backend._parse_frame_payload(frame_payload, observed_at_utc="2026-01-01T00:00:00.000Z")

    assert latest_frame == 12
    assert tools["0A"].status == "tracked"
    assert tools["0A"].validity_known is False
    assert tools["0A"].quaternion == (1.0, 0.0, 0.0, 0.0)
    assert tools["0B"].status == "missing"
    assert tools["0B"].valid is False
    assert tools["0B"].validity_known is True


def test_ndi_backend_marks_invalid_transform_explicitly() -> None:
    backend = TrackerBackendNDI("/dev/ttyUSB0", tracker_factory=lambda _settings: object())
    invalid = np.eye(4)
    invalid[0, 0] = 2.0
    frame_payload = (
        ["0A"],
        [1710000000.0],
        [7],
        [invalid],
        [0.11],
    )

    tools, latest_frame = backend._parse_frame_payload(frame_payload, observed_at_utc="2026-01-01T00:00:00.000Z")

    assert latest_frame == 7
    assert tools["0A"].valid is False
    assert tools["0A"].validity_known is True
    assert tools["0A"].status.startswith("invalid_transform:")

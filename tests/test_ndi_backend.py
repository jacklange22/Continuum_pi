import numpy as np

from continuum_robot.tracking.ndi_backend import TrackerBackendNDI, normalize_tool_id


def test_normalize_tool_id_maps_expected_handles() -> None:
    assert normalize_tool_id(b"0A\x00")[0] == "0A"
    assert normalize_tool_id("port-0b")[0] == "0B"
    assert normalize_tool_id(" 0A ")[0] == "0A"


def test_normalize_tool_id_maps_observed_live_pi_handles_by_default() -> None:
    normalized, raw_text, mapped = normalize_tool_id("10")
    assert normalized == "0A"
    assert raw_text == "10"
    assert mapped is True

    normalized, raw_text, mapped = normalize_tool_id("11")
    assert normalized == "0B"
    assert raw_text == "11"
    assert mapped is True


def test_normalize_tool_id_applies_alias_mapping() -> None:
    normalized, raw_text, mapped = normalize_tool_id("port-01", tool_id_aliases={"PORT01": "0B"})
    assert normalized == "0B"
    assert raw_text == "port-01"
    assert mapped is True

    normalized, raw_text, mapped = normalize_tool_id("10", tool_id_aliases={"10": "0B"})
    assert normalized == "0B"
    assert raw_text == "10"
    assert mapped is True


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

    tools, latest_frame, debug = backend._parse_frame_payload(
        frame_payload,
        observed_at_utc="2026-01-01T00:00:00.000Z",
    )

    assert latest_frame == 12
    assert tools["0A"].status == "tracked"
    assert tools["0A"].validity_known is False
    assert tools["0A"].quaternion == (1.0, 0.0, 0.0, 0.0)
    assert tools["0B"].status == "missing"
    assert tools["0B"].valid is False
    assert tools["0B"].validity_known is True
    assert debug["runtime_role_mappings"] == {"0A": "0A", "0B": "Port-0B"}


def test_ndi_backend_applies_tool_aliases_to_runtime_roles() -> None:
    backend = TrackerBackendNDI(
        "/dev/ttyUSB0",
        tracker_factory=lambda _settings: object(),
        tool_id_aliases={"PORT01": "0A", "PORT02": "0B"},
    )
    frame_payload = (
        ["port-01", "port-02"],
        [1710000000.0, 1710000000.1],
        [18, 18],
        [np.eye(4), np.eye(4)],
        [0.15, 0.18],
    )

    tools, latest_frame, debug = backend._parse_frame_payload(frame_payload, observed_at_utc="2026-01-01T00:00:00.000Z")

    assert latest_frame == 18
    assert sorted(tools) == ["0A", "0B"]
    assert debug["runtime_role_mappings"] == {"0A": "port-01", "0B": "port-02"}
    assert debug["unmapped_tool_ids"] == []


def test_ndi_backend_maps_observed_live_handles_to_runtime_roles_by_default() -> None:
    backend = TrackerBackendNDI("/dev/ttyUSB0", tracker_factory=lambda _settings: object())
    frame_payload = (
        ["10", "11"],
        [1710000000.0, 1710000000.1],
        [22, 22],
        [np.eye(4), np.eye(4)],
        [0.15, 0.18],
    )

    tools, latest_frame, debug = backend._parse_frame_payload(frame_payload, observed_at_utc="2026-01-01T00:00:00.000Z")

    assert latest_frame == 22
    assert sorted(tools) == ["0A", "0B"]
    assert debug["raw_tool_ids"] == ["10", "11"]
    assert debug["normalized_tool_ids"] == ["0A", "0B"]
    assert debug["tool_id_mapping"] == {"10": "0A", "11": "0B"}
    assert debug["runtime_role_mappings"] == {"0A": "10", "0B": "11"}
    assert debug["unmapped_tool_ids"] == []


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

    tools, latest_frame, _debug = backend._parse_frame_payload(
        frame_payload,
        observed_at_utc="2026-01-01T00:00:00.000Z",
    )

    assert latest_frame == 7
    assert tools["0A"].valid is False
    assert tools["0A"].validity_known is True
    assert tools["0A"].status.startswith("invalid_transform:")

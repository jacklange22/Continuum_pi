"""Tests for the two-segment dataset label-mode resolver."""

from __future__ import annotations

from continuum_robot.tracking.two_segment_roles import (
    LABEL_MODE_AUTO,
    LABEL_MODE_DISTAL_POSE6,
    LABEL_MODE_DISTAL_XYZ,
    LABEL_MODE_TWO_COIL_POSE12,
    LABEL_MODE_TWO_COIL_XYZ,
    resolve_two_segment_label_mode,
)


def test_auto_picks_two_coil_pose12_when_all_roles_present() -> None:
    result = resolve_two_segment_label_mode(
        requested_mode=LABEL_MODE_AUTO,
        distal_available=True,
        intermediate_available=True,
        orientation_available=True,
        intermediate_orientation_available=True,
    )

    assert result["mode"] == LABEL_MODE_TWO_COIL_POSE12
    assert result["available"] is True
    assert result["downgraded"] is False


def test_auto_picks_two_coil_xyz_when_orientation_missing() -> None:
    result = resolve_two_segment_label_mode(
        requested_mode=LABEL_MODE_AUTO,
        distal_available=True,
        intermediate_available=True,
        orientation_available=False,
        intermediate_orientation_available=False,
    )

    assert result["mode"] == LABEL_MODE_TWO_COIL_XYZ
    assert result["available"] is True


def test_auto_picks_distal_pose6_when_intermediate_missing() -> None:
    result = resolve_two_segment_label_mode(
        requested_mode=LABEL_MODE_AUTO,
        distal_available=True,
        intermediate_available=False,
        orientation_available=True,
        intermediate_orientation_available=False,
    )

    assert result["mode"] == LABEL_MODE_DISTAL_POSE6
    assert result["available"] is True


def test_auto_picks_distal_xyz_when_only_distal_position_available() -> None:
    result = resolve_two_segment_label_mode(
        requested_mode=LABEL_MODE_AUTO,
        distal_available=True,
        intermediate_available=False,
        orientation_available=False,
        intermediate_orientation_available=False,
    )

    assert result["mode"] == LABEL_MODE_DISTAL_XYZ
    assert result["available"] is True


def test_auto_marks_unavailable_when_distal_missing() -> None:
    result = resolve_two_segment_label_mode(
        requested_mode=LABEL_MODE_AUTO,
        distal_available=False,
        intermediate_available=True,
        orientation_available=False,
        intermediate_orientation_available=False,
    )

    assert result["available"] is False
    assert "distal_tip_unavailable" in result["reasons"]


def test_requested_two_coil_xyz_downgrades_to_distal_when_intermediate_missing() -> None:
    result = resolve_two_segment_label_mode(
        requested_mode=LABEL_MODE_TWO_COIL_XYZ,
        distal_available=True,
        intermediate_available=False,
        orientation_available=False,
        intermediate_orientation_available=False,
    )

    assert result["downgraded"] is True
    assert result["mode"] == LABEL_MODE_DISTAL_XYZ
    assert "intermediate_unavailable_downgrading_to_distal_xyz" in result["reasons"]


def test_requested_two_coil_pose12_downgrades_to_two_coil_xyz_when_orientation_missing() -> None:
    result = resolve_two_segment_label_mode(
        requested_mode=LABEL_MODE_TWO_COIL_POSE12,
        distal_available=True,
        intermediate_available=True,
        orientation_available=False,
        intermediate_orientation_available=False,
    )

    assert result["mode"] == LABEL_MODE_TWO_COIL_XYZ
    assert result["downgraded"] is True
    assert "orientation_unavailable_for_two_coil_pose12" in result["reasons"]


def test_requested_distal_pose6_downgrades_to_distal_xyz_when_no_orientation() -> None:
    result = resolve_two_segment_label_mode(
        requested_mode=LABEL_MODE_DISTAL_POSE6,
        distal_available=True,
        intermediate_available=False,
        orientation_available=False,
        intermediate_orientation_available=False,
    )

    assert result["mode"] == LABEL_MODE_DISTAL_XYZ
    assert result["downgraded"] is True


def test_unknown_label_mode_falls_back_to_auto_and_flags_reason() -> None:
    result = resolve_two_segment_label_mode(
        requested_mode="garbage_mode",
        distal_available=True,
        intermediate_available=True,
        orientation_available=True,
        intermediate_orientation_available=True,
    )

    assert result["mode"] == LABEL_MODE_AUTO
    assert result["downgraded"] is True
    assert any("unknown_label_mode" in reason for reason in result["reasons"])

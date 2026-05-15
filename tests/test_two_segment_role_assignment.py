"""Tests for the selectable bottom/top role abstraction over segment_a/segment_b."""

from __future__ import annotations

import pytest

from continuum_robot.config.schemas import RobotConfig, RobotSegmentConfig
from continuum_robot.two_segment import (
    TwoSegmentCommand,
    build_two_segment_foundation_metadata,
)


def _segments() -> dict[str, RobotSegmentConfig]:
    return {
        "segment_a": RobotSegmentConfig(
            key="segment_a",
            label="Spine 1",
            segment_label="Segment A",
            segment_role="proximal",
            segment_order_index=0,
            servo_ids=[1, 2, 3, 4],
            pairs={"axis_a": [1, 3], "axis_b": [2, 4]},
        ),
        "segment_b": RobotSegmentConfig(
            key="segment_b",
            label="Spine 2",
            segment_label="Segment B",
            segment_role="distal",
            segment_order_index=1,
            servo_ids=[5, 6, 7, 8],
            pairs={"axis_a": [5, 7], "axis_b": [6, 8]},
        ),
    }


def test_physical_assembly_defaults_to_a_bottom_b_top() -> None:
    robot = RobotConfig(mode="dual_segment", segments=_segments())
    assembly = robot.physical_assembly()

    assert assembly["bottom_segment_key"] == "segment_a"
    assert assembly["top_segment_key"] == "segment_b"
    assert assembly["bottom_servo_ids"] == [1, 2, 3, 4]
    assert assembly["top_servo_ids"] == [5, 6, 7, 8]
    assert assembly["is_valid"] is True
    assert assembly["issues"] == []
    assert assembly["lower_tick_means_tension"] is True
    assert assembly["tendon_axis_convention_per_segment"] == "[+X, +Y, -X, -Y]"


def test_physical_assembly_allows_b_bottom_a_top() -> None:
    robot = RobotConfig(
        mode="dual_segment",
        segments=_segments(),
        bottom_segment_key="segment_b",
        top_segment_key="segment_a",
    )
    assembly = robot.physical_assembly()

    assert assembly["bottom_segment_key"] == "segment_b"
    assert assembly["top_segment_key"] == "segment_a"
    assert assembly["bottom_servo_ids"] == [5, 6, 7, 8]
    assert assembly["top_servo_ids"] == [1, 2, 3, 4]
    assert assembly["is_valid"] is True


def test_physical_assembly_blocks_duplicate_role_assignment() -> None:
    robot = RobotConfig(
        mode="dual_segment",
        segments=_segments(),
        bottom_segment_key="segment_a",
        top_segment_key="segment_a",
    )
    assembly = robot.physical_assembly()

    assert assembly["is_valid"] is False
    assert any("must differ" in issue for issue in assembly["issues"])


def test_physical_assembly_blocks_unknown_segment_key() -> None:
    robot = RobotConfig(
        mode="dual_segment",
        segments=_segments(),
        bottom_segment_key="segment_zeta",
        top_segment_key="segment_b",
    )
    assembly = robot.physical_assembly()

    assert assembly["is_valid"] is False
    assert any("not defined" in issue for issue in assembly["issues"])


def test_operating_context_propagates_bottom_top_roles_for_dual_segment() -> None:
    robot = RobotConfig(
        mode="dual_segment",
        segments=_segments(),
        bottom_segment_key="segment_b",
        top_segment_key="segment_a",
    )
    context = robot.operating_context()

    assert context.operating_mode == "dual_segment"
    assert context.bottom_segment_key == "segment_b"
    assert context.top_segment_key == "segment_a"
    assert context.bottom_servo_ids == [5, 6, 7, 8]
    assert context.top_servo_ids == [1, 2, 3, 4]
    assert context.physical_assembly["bottom_segment_label"] == "Segment B"
    assert context.physical_assembly["top_segment_label"] == "Segment A"
    assert context.physical_assembly_issues == []
    metadata = context.metadata()
    assert metadata["physical_assembly"]["bottom_segment_key"] == "segment_b"
    assert metadata["physical_assembly_issues"] == []
    assert metadata["mode_capabilities"]["bottom_top_role_selection"] is True


def test_operating_context_reports_assembly_issues_when_invalid() -> None:
    robot = RobotConfig(
        mode="dual_segment",
        segments=_segments(),
        bottom_segment_key="segment_a",
        top_segment_key="segment_a",
    )
    context = robot.operating_context()

    assert context.physical_assembly_issues
    assert any("must differ" in issue for issue in context.physical_assembly_issues)


def test_single_segment_context_still_exposes_assembly_metadata_without_breaking() -> None:
    robot = RobotConfig(mode="single_segment", segments=_segments())
    context = robot.operating_context()

    assert context.operating_mode == "single_segment"
    # Even in single-segment mode the assembly metadata is computed for awareness.
    assert context.bottom_segment_key == "segment_a"
    assert context.top_segment_key == "segment_b"
    # Active segment behaviour is unchanged.
    assert context.active_segment_key == "segment_a"
    assert context.active_segment_servo_ids == [1, 2, 3, 4]


def test_two_segment_foundation_metadata_includes_bottom_top_block() -> None:
    robot = RobotConfig(
        mode="dual_segment",
        segments=_segments(),
        bottom_segment_key="segment_b",
        top_segment_key="segment_a",
    )
    context = robot.operating_context()

    foundation = build_two_segment_foundation_metadata(context)

    assert foundation["physical_assembly"]["bottom_segment_key"] == "segment_b"
    assert foundation["physical_assembly_valid"] is True
    assert foundation["bottom_top"]["bottom_segment_key"] == "segment_b"
    assert foundation["bottom_top"]["top_segment_key"] == "segment_a"
    assert foundation["bottom_top"]["bottom_servo_ids"] == [5, 6, 7, 8]
    assert foundation["bottom_top"]["top_servo_ids"] == [1, 2, 3, 4]
    assert foundation["bottom_top"]["segment_role_assignment"] == {
        "segment_b": "proximal",
        "segment_a": "distal",
    }
    assert foundation["enabled_capabilities"]["bottom_top_role_selection"] is True


def test_two_segment_foundation_marks_unavailable_when_assembly_invalid() -> None:
    robot = RobotConfig(
        mode="dual_segment",
        segments=_segments(),
        bottom_segment_key="segment_a",
        top_segment_key="segment_a",
    )
    context = robot.operating_context()

    foundation = build_two_segment_foundation_metadata(context)

    assert foundation["physical_assembly_valid"] is False
    assert foundation["physical_assembly_issues"]
    assert foundation["available"] is False  # invalid assembly blocks readiness


def test_two_segment_command_to_record_exposes_bottom_top_view_when_assembly_present() -> None:
    robot = RobotConfig(
        mode="dual_segment",
        segments=_segments(),
        bottom_segment_key="segment_b",
        top_segment_key="segment_a",
    )
    context = robot.operating_context()
    command = TwoSegmentCommand.from_mapping(
        {
            "segment_a": [0.10, 0.20, 0.30, 0.40],  # this is now the TOP (distal)
            "segment_b": [-0.10, -0.20, -0.30, -0.40],  # this is now the BOTTOM (proximal)
        }
    )

    record = command.to_record(context=context)

    assert record["bottom_top_command"]["bottom_segment_key"] == "segment_b"
    assert record["bottom_top_command"]["top_segment_key"] == "segment_a"
    assert record["bottom_top_command"]["bottom"] == [-0.10, -0.20, -0.30, -0.40]
    assert record["bottom_top_command"]["top"] == [0.10, 0.20, 0.30, 0.40]
    assert record["physical_assembly"]["bottom_segment_key"] == "segment_b"
    assert record["physical_assembly"]["bottom_servo_ids"] == [5, 6, 7, 8]
    assert "T_bottom" in record["composed_frame_note"]


def test_physical_assembly_notes_propagate_into_foundation_metadata() -> None:
    """Operator free-form notes attached to the assembly are carried into the run foundation block.

    The notes field is intended for bench-day context the operator wants to
    capture per-rig (e.g., "top mount loose, retighten before runs >0.5cm").
    """
    robot = RobotConfig(
        mode="dual_segment",
        segments=_segments(),
        bottom_segment_key="segment_b",
        top_segment_key="segment_a",
        physical_assembly_notes="Top mount loose — retighten before high-amplitude runs.",
    )
    assembly = robot.physical_assembly()
    assert assembly["notes"] == "Top mount loose — retighten before high-amplitude runs."
    context = robot.operating_context()
    foundation = build_two_segment_foundation_metadata(context)
    assert foundation["physical_assembly"]["notes"] == assembly["notes"]


def test_two_segment_run_summary_surfaces_operator_notes(tmp_path) -> None:
    """The Data tab's two-segment foundation row includes operator notes when present."""
    from continuum_robot.data.run_management import _format_two_segment_foundation

    foundation = {
        "available": True,
        "semantic_scope": "data_structures_and_metadata_only",
        "segment_order": ["segment_a", "segment_b"],
        "segments": {
            "segment_a": {"segment_label": "Segment A", "segment_role": "proximal", "servo_ids": [1, 2, 3, 4]},
            "segment_b": {"segment_label": "Segment B", "segment_role": "distal", "servo_ids": [5, 6, 7, 8]},
        },
        "bottom_top": {
            "bottom_segment_label": "Segment A",
            "top_segment_label": "Segment B",
            "bottom_servo_ids": [1, 2, 3, 4],
            "top_servo_ids": [5, 6, 7, 8],
            "bottom_segment_key": "segment_a",
            "top_segment_key": "segment_b",
        },
        "physical_assembly": {
            "notes": "Bottom mount epoxy curing — handle gently."
        },
    }

    summary_text = _format_two_segment_foundation(foundation)
    assert "operator_notes=Bottom mount epoxy curing — handle gently." in summary_text


def test_two_segment_run_summary_flattens_multiline_operator_notes_for_data_tab() -> None:
    """Multi-line notes are collapsed into a single line for the Data-tab row.

    Raw text (with newlines) remains in the underlying JSON / foundation
    metadata. The flattened version is for the operator's single-row
    detail view so the GUI table doesn't break.
    """
    from continuum_robot.data.run_management import _format_two_segment_foundation

    foundation = {
        "available": True,
        "segments": {
            "segment_a": {"segment_label": "Segment A", "segment_role": "proximal", "servo_ids": [1, 2, 3, 4]},
            "segment_b": {"segment_label": "Segment B", "segment_role": "distal", "servo_ids": [5, 6, 7, 8]},
        },
        "segment_order": ["segment_a", "segment_b"],
        "physical_assembly": {
            "notes": "Line one\nLine two\n\tIndented line three\n",
        },
    }

    summary_text = _format_two_segment_foundation(foundation)

    # Sanitized single line — no embedded newlines or tabs.
    assert "\n" not in summary_text.split("operator_notes=")[1].split("; ")[0]
    assert "\t" not in summary_text
    # All three notes lines appear in the joined string.
    assert "Line one" in summary_text
    assert "Line two" in summary_text
    assert "Indented line three" in summary_text


def test_two_segment_run_summary_truncates_excessively_long_operator_notes() -> None:
    """Notes >200 chars are truncated with ellipsis; full text stays in the JSON."""
    from continuum_robot.data.run_management import _format_two_segment_foundation

    long_notes = "x" * 500
    foundation = {
        "available": True,
        "segments": {
            "segment_a": {"segment_label": "Segment A", "segment_role": "proximal", "servo_ids": [1, 2, 3, 4]},
            "segment_b": {"segment_label": "Segment B", "segment_role": "distal", "servo_ids": [5, 6, 7, 8]},
        },
        "segment_order": ["segment_a", "segment_b"],
        "physical_assembly": {"notes": long_notes},
    }

    summary_text = _format_two_segment_foundation(foundation)
    notes_segment = summary_text.split("operator_notes=")[1].split("; ")[0]

    assert notes_segment.endswith("...")
    assert len(notes_segment) <= 200


def test_two_segment_command_round_trip_preserves_segment_keys_under_swap() -> None:
    robot = RobotConfig(
        mode="dual_segment",
        segments=_segments(),
        bottom_segment_key="segment_b",
        top_segment_key="segment_a",
    )
    context = robot.operating_context()
    command = TwoSegmentCommand.from_mapping(
        {
            "segment_a": [0.1, 0.2, 0.3, 0.4],
            "segment_b": [-0.1, -0.2, -0.3, -0.4],
        }
    )

    # Flat command order is governed by commanded_servo_ids (always [1..8])
    # so the physical role swap must NOT change the flat servo mapping.
    flat = command.to_flat(context=context)
    assert flat == [0.1, 0.2, 0.3, 0.4, -0.1, -0.2, -0.3, -0.4]
    round_trip = TwoSegmentCommand.from_flat(flat, context=context)
    assert round_trip.to_segment_mapping() == command.to_segment_mapping()

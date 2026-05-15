"""Servo-ID consistency invariants for the two-segment path.

These tests pin the contract that the rest of the system relies on:

1. dual_segment mode always exposes commanded_servo_ids == expected_servo_ids == [1..8].
2. Bottom/top role swap NEVER changes commanded_servo_ids order — only the
   bottom/top servo-id partition flips.
3. TwoSegmentCommand.to_flat() produces values in commanded_servo_ids order
   (servo 1 first, servo 8 last) regardless of the bottom/top role swap.
4. Servo 1-4 always belong to segment_a; servo 5-8 always belong to segment_b.
   The role (proximal/distal) is a label, not a remapping of servo IDs.
5. In single_segment + parallel_single + one_servo, the bottom/top assembly
   metadata is still computed (informational only), but expected/commanded IDs
   reflect the mode-appropriate scope.
"""

from __future__ import annotations

import pytest

from continuum_robot.config.schemas import RobotConfig, RobotSegmentConfig
from continuum_robot.two_segment import TwoSegmentCommand


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


@pytest.mark.parametrize("bottom,top", [("segment_a", "segment_b"), ("segment_b", "segment_a")])
def test_dual_segment_servo_ids_remain_1_through_8_regardless_of_role_swap(bottom: str, top: str) -> None:
    robot = RobotConfig(
        mode="dual_segment",
        segments=_segments(),
        bottom_segment_key=bottom,
        top_segment_key=top,
    )
    context = robot.operating_context()

    assert context.expected_servo_ids == [1, 2, 3, 4, 5, 6, 7, 8]
    assert context.commanded_servo_ids == [1, 2, 3, 4, 5, 6, 7, 8]
    assert context.all_configured_servo_ids == [1, 2, 3, 4, 5, 6, 7, 8]


def test_segment_a_always_owns_servos_1_through_4_segment_b_always_owns_5_through_8() -> None:
    """No matter what the bottom/top assignment is, segment_a's IDs do not move."""
    for bottom in ("segment_a", "segment_b"):
        top = "segment_b" if bottom == "segment_a" else "segment_a"
        robot = RobotConfig(
            mode="dual_segment",
            segments=_segments(),
            bottom_segment_key=bottom,
            top_segment_key=top,
        )
        context = robot.operating_context()
        segments_meta = dict(context.metadata().get("segments", {}) or {})
        assert segments_meta["segment_a"]["servo_ids"] == [1, 2, 3, 4]
        assert segments_meta["segment_b"]["servo_ids"] == [5, 6, 7, 8]


def test_bottom_top_servo_ids_flip_correctly_under_role_swap() -> None:
    default_robot = RobotConfig(mode="dual_segment", segments=_segments())
    default_context = default_robot.operating_context()
    assert default_context.bottom_servo_ids == [1, 2, 3, 4]
    assert default_context.top_servo_ids == [5, 6, 7, 8]

    swapped_robot = RobotConfig(
        mode="dual_segment",
        segments=_segments(),
        bottom_segment_key="segment_b",
        top_segment_key="segment_a",
    )
    swapped_context = swapped_robot.operating_context()
    assert swapped_context.bottom_servo_ids == [5, 6, 7, 8]
    assert swapped_context.top_servo_ids == [1, 2, 3, 4]
    # The flat command order is unchanged by the swap.
    assert swapped_context.commanded_servo_ids == [1, 2, 3, 4, 5, 6, 7, 8]


def test_two_segment_command_to_flat_order_is_commanded_servo_ids_under_role_swap() -> None:
    """to_flat must always produce values in commanded_servo_ids order."""
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

    flat = command.to_flat(context=context)

    # segment_a values are at positions 0..3 (matching servo IDs 1..4 in commanded_servo_ids).
    assert flat[:4] == [0.1, 0.2, 0.3, 0.4]
    # segment_b values are at positions 4..7 (matching servo IDs 5..8 in commanded_servo_ids).
    assert flat[4:] == [-0.1, -0.2, -0.3, -0.4]


def test_two_segment_command_servo_mapping_is_servo_id_keyed_not_role_keyed() -> None:
    """to_servo_mapping must use physical servo IDs as keys (1..8), not bottom/top labels."""
    robot = RobotConfig(
        mode="dual_segment",
        segments=_segments(),
        bottom_segment_key="segment_b",
        top_segment_key="segment_a",
    )
    context = robot.operating_context()
    command = TwoSegmentCommand.from_mapping(
        {
            "segment_a": [1.0, 2.0, 3.0, 4.0],
            "segment_b": [5.0, 6.0, 7.0, 8.0],
        }
    )

    mapping = command.to_servo_mapping(context=context)

    # Keys are physical servo IDs.
    assert sorted(mapping.keys()) == [1, 2, 3, 4, 5, 6, 7, 8]
    # segment_a values go to servos 1..4.
    assert mapping[1] == 1.0
    assert mapping[4] == 4.0
    # segment_b values go to servos 5..8.
    assert mapping[5] == 5.0
    assert mapping[8] == 8.0


def test_one_servo_mode_only_exposes_the_selected_servo_id() -> None:
    robot = RobotConfig(mode="one_servo", segments=_segments(), selected_servo_id=3)
    context = robot.operating_context()

    assert context.expected_servo_ids == [3]
    assert context.commanded_servo_ids == [3]
    assert context.all_configured_servo_ids == [1, 2, 3, 4, 5, 6, 7, 8]
    # Bottom/top metadata is still populated but informational.
    assert context.bottom_servo_ids == [1, 2, 3, 4]
    assert context.top_servo_ids == [5, 6, 7, 8]


def test_single_segment_mode_only_exposes_active_segment_servo_ids() -> None:
    robot = RobotConfig(mode="single_segment", segments=_segments(), active_segment="segment_b")
    context = robot.operating_context()

    assert context.expected_servo_ids == [5, 6, 7, 8]
    assert context.commanded_servo_ids == [5, 6, 7, 8]
    assert context.active_segment_servo_ids == [5, 6, 7, 8]


def test_parallel_single_mode_exposes_all_8_with_mirror_pairs() -> None:
    robot = RobotConfig(mode="parallel_single", segments=_segments(), active_segment="segment_a")
    context = robot.operating_context()

    assert context.expected_servo_ids == [1, 2, 3, 4, 5, 6, 7, 8]
    assert context.commanded_servo_ids == [1, 2, 3, 4, 5, 6, 7, 8]
    # Mirror pairs map active-segment IDs to the mirrored other-segment IDs.
    assert context.mirror_pairs == {1: 5, 2: 6, 3: 7, 4: 8}


def test_two_segment_command_round_trip_preserves_servo_assignment_under_swap() -> None:
    """flat → command → flat must be lossless regardless of the bottom/top role swap."""
    for bottom, top in [("segment_a", "segment_b"), ("segment_b", "segment_a")]:
        robot = RobotConfig(
            mode="dual_segment",
            segments=_segments(),
            bottom_segment_key=bottom,
            top_segment_key=top,
        )
        context = robot.operating_context()
        original_flat = [0.01, -0.02, 0.03, -0.04, 0.05, -0.06, 0.07, -0.08]

        command = TwoSegmentCommand.from_flat(original_flat, context=context)
        round_trip_flat = command.to_flat(context=context)

        assert round_trip_flat == original_flat, (
            f"flat round-trip should be lossless under bottom={bottom}, top={top}; "
            f"got {round_trip_flat}"
        )

"""Tests for PR 3 additions:

* Aurora frame-number plumb-through onto every :class:`AuroraToolMeasurement`.
* Opt-in validity-from-quality heuristic on :class:`AuroraParser`.
* Tracker-displacement gate on pretension validation
  (:meth:`PretensionValidationService.validate_current_and_displacement_balance`
  plus the :class:`ServoService` wrapper).

These exercise the new code paths end-to-end and pin the contract changes
so future edits cannot silently regress them. Each addition is opt-in: the
default behavior of the legacy parser and the existing classical
``validate_pretension`` is preserved.
"""
from __future__ import annotations

import math
import struct
from pathlib import Path

import pytest

from continuum_robot.tracking.aurora_packet import (
    TOOL_RECORD_SIZE,
    build_transform_payload,
    frame_payload,
)
from continuum_robot.tracking.aurora_parser import (
    DEFAULT_QUALITY_INVALID_THRESHOLD,
    AuroraParser,
)
from continuum_robot.tracking.tool_models import AuroraToolMeasurement
from continuum_robot.servos.pretension_validation_service import (
    PretensionValidationResult,
    PretensionValidationService,
)


# ---------------------------------------------------------------------------
# Helpers (kept local so the test file is self-contained)
# ---------------------------------------------------------------------------


def _build_tool_record(
    tool_sn: int,
    quat_wxyz: tuple[float, float, float, float],
    translation_xyz: tuple[float, float, float],
    tracker_error: float,
) -> bytes:
    body = bytearray()
    body.extend(int(tool_sn).to_bytes(4, "little", signed=False))
    for value in (*quat_wxyz, *translation_xyz, tracker_error):
        body.extend(struct.pack("<f", float(value)))
    if len(body) != TOOL_RECORD_SIZE:
        raise AssertionError(f"Expected {TOOL_RECORD_SIZE} bytes, got {len(body)}")
    return bytes(body)


def _build_transform_frame(
    *,
    frame_number: int = 99,
    tool_a_quality: float = 0.01,
    tool_b_quality: float = 0.02,
) -> bytes:
    """Build a complete framed transform packet for 0A + 0B."""
    records = b"".join(
        [
            _build_tool_record(0x00004130, (1.0, 0.0, 0.0, 0.0), (10.0, 20.0, 30.0), tool_a_quality),
            _build_tool_record(0x00004230, (1.0, 0.0, 0.0, 0.0), (11.0, 21.0, 31.0), tool_b_quality),
        ]
    )
    return frame_payload(build_transform_payload(frame_number=frame_number, records_blob=records))


# ---------------------------------------------------------------------------
# Aurora frame_number plumb-through (PR 3a)
# ---------------------------------------------------------------------------


class TestAuroraFrameNumberPlumbThrough:
    def test_frame_number_is_propagated_onto_every_tool_record(self) -> None:
        parser = AuroraParser()
        frame = _build_transform_frame(frame_number=12345)
        out = parser.parse_transform_packet(frame)
        assert out["0A"].frame_number == 12345
        assert out["0B"].frame_number == 12345

    def test_frame_number_matches_parsed_header(self) -> None:
        parser = AuroraParser()
        frame = _build_transform_frame(frame_number=7)
        payload = parser.parse_payload(frame)
        assert payload.header.frame_number == 7
        # Same frame number reaches the per-tool measurements.
        out = parser.parse_transform_packet(frame)
        assert {tool.frame_number for tool in out.values()} == {7}

    def test_default_aurora_tool_measurement_has_no_frame_number(self) -> None:
        """Directly-constructed measurements default to ``frame_number=None`` for backwards compatibility."""
        m = AuroraToolMeasurement(
            tool_id="0A",
            quat_wxyz=(1.0, 0.0, 0.0, 0.0),
            translation_xyz=(0.0, 0.0, 0.0),
        )
        assert m.frame_number is None


# ---------------------------------------------------------------------------
# Validity-from-quality heuristic (PR 3b, opt-in)
# ---------------------------------------------------------------------------


class TestAuroraValidityHeuristic:
    def test_default_parser_keeps_legacy_validity_unknown_contract(self) -> None:
        parser = AuroraParser()
        out = parser.parse_transform_packet(_build_transform_frame())
        # No opt-in -> previous behavior preserved (valid is None).
        assert out["0A"].valid is None
        assert out["0B"].valid is None
        assert out["0A"].status_text == "validity_not_available_in_compatibility_packet"

    def test_opt_in_marks_valid_when_quality_below_threshold(self) -> None:
        parser = AuroraParser(derive_validity_from_quality=True)
        out = parser.parse_transform_packet(
            _build_transform_frame(tool_a_quality=0.05, tool_b_quality=0.1)
        )
        assert out["0A"].valid is True
        assert out["0B"].valid is True
        assert "tracked_quality_below_threshold" in out["0A"].status_text
        assert "tracked_quality_below_threshold" in out["0B"].status_text

    def test_opt_in_marks_invalid_when_quality_above_threshold(self) -> None:
        parser = AuroraParser(
            derive_validity_from_quality=True,
            quality_invalid_threshold=0.2,
        )
        out = parser.parse_transform_packet(
            _build_transform_frame(tool_a_quality=0.05, tool_b_quality=0.4)
        )
        assert out["0A"].valid is True
        assert out["0B"].valid is False
        assert "above_threshold" in out["0B"].status_text

    def test_opt_in_marks_invalid_on_negative_or_nonfinite_quality(self) -> None:
        parser = AuroraParser(derive_validity_from_quality=True)
        negative = parser.parse_transform_packet(_build_transform_frame(tool_a_quality=-1.0))
        assert negative["0A"].valid is False
        assert "negative" in negative["0A"].status_text

        nan_frame = parser.parse_transform_packet(_build_transform_frame(tool_a_quality=float("nan")))
        assert nan_frame["0A"].valid is False
        assert "not_finite" in nan_frame["0A"].status_text

        inf_frame = parser.parse_transform_packet(_build_transform_frame(tool_a_quality=float("inf")))
        assert inf_frame["0A"].valid is False
        assert "not_finite" in inf_frame["0A"].status_text

    def test_threshold_must_be_positive(self) -> None:
        with pytest.raises(ValueError, match="quality_invalid_threshold"):
            AuroraParser(quality_invalid_threshold=0.0)
        with pytest.raises(ValueError, match="quality_invalid_threshold"):
            AuroraParser(quality_invalid_threshold=-1.0)

    def test_status_byte_is_always_none_in_legacy_packet(self) -> None:
        """The legacy protocol does not carry a per-tool status byte; both opt-in and opt-out
        leave the field as ``None`` so callers cannot accidentally read fabricated bits."""
        for parser in (AuroraParser(), AuroraParser(derive_validity_from_quality=True)):
            out = parser.parse_transform_packet(_build_transform_frame())
            assert out["0A"].status_byte is None
            assert out["0B"].status_byte is None

    def test_default_threshold_is_sane(self) -> None:
        # Sanity: the documented default of 0.5 mm-equivalent is what the parser uses.
        assert DEFAULT_QUALITY_INVALID_THRESHOLD == 0.5
        assert AuroraParser().quality_invalid_threshold == 0.5


# ---------------------------------------------------------------------------
# Pretension tracker-displacement gate (PR 3c)
# ---------------------------------------------------------------------------


class TestPretensionTrackerDisplacementGate:
    def test_classic_validate_current_balance_unchanged(self) -> None:
        # The classic method still returns the same shape it always has; the new
        # gate-related fields are None when the classic method is used.
        result = PretensionValidationService().validate_current_balance(
            [-30, 20, -25, 24], tolerance_ma=12
        )
        assert result.passed is True
        assert result.spread_ma == 10
        assert result.current_passed is None
        assert result.tracker_displacement_mm is None
        assert result.min_displacement_mm is None
        assert result.displacement_passed is None

    def test_gate_passes_when_both_current_and_displacement_pass(self) -> None:
        result = PretensionValidationService().validate_current_and_displacement_balance(
            [120, 130, 125, 124],
            tolerance_ma=15,
            tracker_displacement_mm=1.2,
            min_displacement_mm=0.5,
        )
        assert result.passed is True
        assert result.current_passed is True
        assert result.displacement_passed is True
        assert result.tracker_displacement_mm == pytest.approx(1.2)
        assert result.min_displacement_mm == pytest.approx(0.5)
        assert "passed" in result.message

    def test_gate_fails_when_displacement_below_floor_even_if_currents_balanced(self) -> None:
        # Stuck-cable case: currents look balanced but tracker shows no motion.
        result = PretensionValidationService().validate_current_and_displacement_balance(
            [120, 130, 125, 124],
            tolerance_ma=15,
            tracker_displacement_mm=0.1,
            min_displacement_mm=0.5,
        )
        assert result.passed is False
        assert result.current_passed is True
        assert result.displacement_passed is False
        assert "may be stuck" in result.message or "not tensioning" in result.message
        assert "Tracker displacement gate failed" in result.message

    def test_gate_fails_when_currents_unbalanced_even_if_displacement_passes(self) -> None:
        result = PretensionValidationService().validate_current_and_displacement_balance(
            [80, 220, 100, 90],
            tolerance_ma=15,
            tracker_displacement_mm=1.0,
            min_displacement_mm=0.5,
        )
        assert result.passed is False
        assert result.current_passed is False
        assert result.displacement_passed is True
        assert "spread" in result.message
        assert "Tracker displacement gate passed" in result.message

    def test_gate_fails_when_tracker_displacement_is_unavailable(self) -> None:
        result = PretensionValidationService().validate_current_and_displacement_balance(
            [120, 130, 125, 124],
            tolerance_ma=15,
            tracker_displacement_mm=None,
            min_displacement_mm=0.5,
        )
        assert result.passed is False
        assert result.current_passed is True
        assert result.displacement_passed is False
        assert result.tracker_displacement_mm is None
        assert "no usable tracker reading" in result.message

    def test_gate_requires_positive_minimum_displacement(self) -> None:
        service = PretensionValidationService()
        with pytest.raises(ValueError, match="min_displacement_mm"):
            service.validate_current_and_displacement_balance(
                [120, 130],
                tolerance_ma=15,
                tracker_displacement_mm=1.0,
                min_displacement_mm=0.0,
            )
        with pytest.raises(ValueError, match="min_displacement_mm"):
            service.validate_current_and_displacement_balance(
                [120, 130],
                tolerance_ma=15,
                tracker_displacement_mm=1.0,
                min_displacement_mm=-0.5,
            )

    def test_gate_handles_all_none_currents_as_failure(self) -> None:
        # If no currents are readable, the underlying current check fails; the
        # combined gate should fail even if displacement is fine.
        result = PretensionValidationService().validate_current_and_displacement_balance(
            [None, None, None, None],
            tolerance_ma=15,
            tracker_displacement_mm=2.0,
            min_displacement_mm=0.5,
        )
        assert result.passed is False
        assert result.current_passed is False
        assert result.displacement_passed is True


# ---------------------------------------------------------------------------
# ServoService wrapper for the gated pretension check
# ---------------------------------------------------------------------------


class TestServoServiceGatedPretensionWrapper:
    def test_wrapper_threads_displacement_value_through_to_gate(self, tmp_path: Path) -> None:
        # Reuse the existing test helper to build a ServoService against mock hardware.
        from tests.test_servo_service import _build_service

        service = _build_service(tmp_path)
        service.connect("/dev/mock-openrb", 115200)
        # Passing a healthy displacement above the floor should produce a
        # PretensionValidationResult whose gate fields are populated.
        result = service.validate_pretension_with_tracker_displacement_gate(
            [1, 2, 3, 4],
            tolerance_ma=80,
            tracker_displacement_mm=1.2,
            min_displacement_mm=0.5,
        )
        assert isinstance(result, PretensionValidationResult)
        assert result.tracker_displacement_mm == pytest.approx(1.2)
        assert result.min_displacement_mm == pytest.approx(0.5)
        assert result.current_passed is not None
        assert result.displacement_passed is True

    def test_wrapper_fails_when_displacement_below_floor(self, tmp_path: Path) -> None:
        from tests.test_servo_service import _build_service

        service = _build_service(tmp_path)
        service.connect("/dev/mock-openrb", 115200)
        result = service.validate_pretension_with_tracker_displacement_gate(
            [1, 2, 3, 4],
            tolerance_ma=80,
            tracker_displacement_mm=0.05,
            min_displacement_mm=0.5,
        )
        assert result.passed is False
        assert result.displacement_passed is False

    def test_wrapper_propagates_unavailable_displacement(self, tmp_path: Path) -> None:
        from tests.test_servo_service import _build_service

        service = _build_service(tmp_path)
        service.connect("/dev/mock-openrb", 115200)
        result = service.validate_pretension_with_tracker_displacement_gate(
            [1, 2, 3, 4],
            tolerance_ma=80,
            tracker_displacement_mm=None,
            min_displacement_mm=0.5,
        )
        assert result.passed is False
        assert result.displacement_passed is False
        assert result.tracker_displacement_mm is None

    def test_classic_validate_pretension_still_returns_classic_shape(self, tmp_path: Path) -> None:
        """The classic entrypoint must not start populating gate fields."""
        from tests.test_servo_service import _build_service

        service = _build_service(tmp_path)
        service.connect("/dev/mock-openrb", 115200)
        classic = service.validate_pretension([1, 2, 3, 4], tolerance_ma=80)
        # No regression: gate fields remain None for the classic path.
        assert classic.current_passed is None
        assert classic.tracker_displacement_mm is None
        assert classic.min_displacement_mm is None
        assert classic.displacement_passed is None

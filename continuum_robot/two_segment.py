"""Canonical two-segment data structures and provenance helpers.

This module is intentionally data-only. It defines how two physical segments
are named, serialized, and flattened, but it does not implement two-segment
kinematics, control, modeling, or automatic pretension.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


SEGMENT_A = "segment_a"
SEGMENT_B = "segment_b"
TWO_SEGMENT_KEYS = (SEGMENT_A, SEGMENT_B)
TWO_SEGMENT_COMMAND_SCHEMA_VERSION = "two_segment_command_v1"
TWO_SEGMENT_POSE_SCHEMA_VERSION = "two_segment_pose_observation_v1"
TWO_SEGMENT_FOUNDATION_SCHEMA_VERSION = "two_segment_foundation_v1"


@dataclass(frozen=True)
class TwoSegmentCommand:
    """One canonical 8-tendon two-segment displacement command.

    The high-level representation is always segment-indexed:
    ``{"segment_a": [a1, a2, a3, a4], "segment_b": [b1, b2, b3, b4]}``.
    Flattening to an 8-vector is a serialization step that must use the robot
    operating context's segment definitions and commanded servo order.
    """

    segment_a: tuple[float, float, float, float]
    segment_b: tuple[float, float, float, float]
    units: str = "cm"
    command_type: str = "tendon_displacement"
    schema_version: str = TWO_SEGMENT_COMMAND_SCHEMA_VERSION

    @classmethod
    def from_mapping(cls, payload: dict[str, Any]) -> "TwoSegmentCommand":
        """Build a command from the canonical segment-indexed mapping."""
        data = dict(payload or {})
        return cls(
            segment_a=_coerce_four_vector(data.get(SEGMENT_A), label=SEGMENT_A),
            segment_b=_coerce_four_vector(data.get(SEGMENT_B), label=SEGMENT_B),
            units=str(data.get("units", "cm") or "cm"),
            command_type=str(data.get("command_type", "tendon_displacement") or "tendon_displacement"),
        )

    @classmethod
    def from_flat(cls, values: list[float] | tuple[float, ...], *, context) -> "TwoSegmentCommand":
        """Build a command from an 8-vector in the context's commanded servo order."""
        flat = [float(value) for value in list(values or [])]
        if len(flat) != 8:
            raise ValueError(f"Two-segment flat command requires exactly 8 values; found {len(flat)}.")
        _validate_two_segment_context(context)
        commanded_ids = [int(value) for value in getattr(context, "commanded_servo_ids", [])]
        if len(commanded_ids) != 8:
            raise ValueError(f"Two-segment context must expose 8 commanded servo IDs; found {commanded_ids}.")
        by_servo = {int(servo_id): float(value) for servo_id, value in zip(commanded_ids, flat)}
        segment_ids = _segment_servo_ids(context)
        return cls(
            segment_a=tuple(float(by_servo[int(servo_id)]) for servo_id in segment_ids[SEGMENT_A]),  # type: ignore[arg-type]
            segment_b=tuple(float(by_servo[int(servo_id)]) for servo_id in segment_ids[SEGMENT_B]),  # type: ignore[arg-type]
        )

    def to_segment_mapping(self) -> dict[str, list[float]]:
        """Return the canonical user-facing segment-indexed representation."""
        return {
            SEGMENT_A: [float(value) for value in self.segment_a],
            SEGMENT_B: [float(value) for value in self.segment_b],
        }

    def to_flat(self, *, context) -> list[float]:
        """Return an 8-vector in the context's commanded servo order."""
        by_servo = self.to_servo_mapping(context=context)
        commanded_ids = [int(value) for value in getattr(context, "commanded_servo_ids", [])]
        if len(commanded_ids) != 8:
            raise ValueError(f"Two-segment context must expose 8 commanded servo IDs; found {commanded_ids}.")
        return [float(by_servo[int(servo_id)]) for servo_id in commanded_ids]

    def to_servo_mapping(self, *, context) -> dict[int, float]:
        """Return command values indexed by physical servo ID."""
        _validate_two_segment_context(context)
        segment_ids = _segment_servo_ids(context)
        by_servo: dict[int, float] = {}
        for servo_id, value in zip(segment_ids[SEGMENT_A], self.segment_a):
            by_servo[int(servo_id)] = float(value)
        for servo_id, value in zip(segment_ids[SEGMENT_B], self.segment_b):
            by_servo[int(servo_id)] = float(value)
        return by_servo

    def to_record(self, *, context=None) -> dict[str, Any]:
        """Return a provenance-friendly JSON record for this command."""
        record: dict[str, Any] = {
            "schema_version": self.schema_version,
            "command_type": self.command_type,
            "units": self.units,
            "segments": self.to_segment_mapping(),
            "segment_order": list(TWO_SEGMENT_KEYS),
            "control_enabled": False,
        }
        if context is not None:
            record["commanded_servo_ids"] = [int(value) for value in getattr(context, "commanded_servo_ids", [])]
            record["servo_command_cm"] = {str(key): value for key, value in self.to_servo_mapping(context=context).items()}
            record["flat_command_cm"] = self.to_flat(context=context)
            bottom_top = self.to_bottom_top_mapping(context=context)
            if bottom_top:
                record["bottom_top_command"] = bottom_top
                metadata = _context_metadata(context)
                assembly = dict(metadata.get("physical_assembly") or {})
                record["physical_assembly"] = {
                    "bottom_segment_key": str(assembly.get("bottom_segment_key") or ""),
                    "top_segment_key": str(assembly.get("top_segment_key") or ""),
                    "bottom_segment_label": str(assembly.get("bottom_segment_label") or ""),
                    "top_segment_label": str(assembly.get("top_segment_label") or ""),
                    "bottom_servo_ids": [int(value) for value in list(assembly.get("bottom_servo_ids") or [])],
                    "top_servo_ids": [int(value) for value in list(assembly.get("top_servo_ids") or [])],
                    "lower_tick_means_tension": bool(assembly.get("lower_tick_means_tension", True)),
                }
                record["composed_frame_note"] = str(assembly.get("composed_frame_note") or "")
        return record

    def to_bottom_top_mapping(self, *, context) -> dict[str, list[float]]:
        """Return the command grouped by physical role (bottom/top).

        Each value is a 4-vector in tendon order ``[+X, +Y, -X, -Y]`` for that
        physical segment. Returns an empty dict when no physical assembly is
        defined (e.g. single-segment context).
        """
        metadata = _context_metadata(context)
        assembly = dict(metadata.get("physical_assembly") or {})
        bottom_key = str(assembly.get("bottom_segment_key") or "")
        top_key = str(assembly.get("top_segment_key") or "")
        if bottom_key not in TWO_SEGMENT_KEYS or top_key not in TWO_SEGMENT_KEYS:
            return {}
        mapping = self.to_segment_mapping()
        return {
            "bottom": [float(value) for value in list(mapping.get(bottom_key) or [])],
            "top": [float(value) for value in list(mapping.get(top_key) or [])],
            "bottom_segment_key": bottom_key,
            "top_segment_key": top_key,
        }


@dataclass(frozen=True)
class TwoSegmentPoseObservation:
    """Canonical metadata shell for future two-segment pose observations."""

    frame: str = "robot"
    distal_tip_pose: dict[str, Any] | None = None
    intermediate_pose: dict[str, Any] | None = None
    segment_poses: dict[str, dict[str, Any]] = field(default_factory=dict)
    tool_poses: dict[str, dict[str, Any]] = field(default_factory=dict)
    tool_roles: dict[str, str] = field(default_factory=dict)
    status: str = "not_collected"
    schema_version: str = TWO_SEGMENT_POSE_SCHEMA_VERSION

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "frame": str(self.frame),
            "status": str(self.status),
            "distal_tip_pose": dict(self.distal_tip_pose or {}),
            "intermediate_pose": dict(self.intermediate_pose or {}),
            "segment_poses": _clean_mapping(self.segment_poses),
            "tool_poses": _clean_mapping(self.tool_poses),
            "tool_roles": {str(key): str(value) for key, value in dict(self.tool_roles or {}).items()},
        }


def two_segment_command_schema(context=None) -> dict[str, Any]:
    """Return the canonical command schema metadata without creating a command."""
    schema: dict[str, Any] = {
        "schema_version": TWO_SEGMENT_COMMAND_SCHEMA_VERSION,
        "command_type": "tendon_displacement",
        "units": "cm",
        "canonical_representation": {
            SEGMENT_A: ["a1", "a2", "a3", "a4"],
            SEGMENT_B: ["b1", "b2", "b3", "b4"],
        },
        "flat_vector_rule": "Flatten only through the operating context commanded_servo_ids order.",
        "control_enabled": False,
    }
    if context is not None:
        schema["segment_order"] = list(getattr(context, "segment_order", []) or TWO_SEGMENT_KEYS)
        schema["commanded_servo_ids"] = [int(value) for value in getattr(context, "commanded_servo_ids", [])]
        schema["segments"] = _segment_metadata(context)
    return schema


def two_segment_pose_schema(tool_roles: dict[str, str] | None = None) -> dict[str, Any]:
    """Return the canonical pose schema metadata for future datasets."""
    return {
        "schema_version": TWO_SEGMENT_POSE_SCHEMA_VERSION,
        "frame": "robot",
        "fields": {
            "intermediate_pose": "optional pose for the Segment A distal end / Segment B base region",
            "distal_tip_pose": "optional pose for the distal tip after Segment B",
            "segment_poses": "optional segment-indexed pose map",
            "tool_poses": "optional raw or registered tracker-tool pose map",
        },
        "tool_roles": {str(key): str(value) for key, value in dict(tool_roles or {}).items()},
        "tracking_enabled": False,
        "multi_coil_tracking_required": "future_validation",
    }


def build_two_segment_foundation_metadata(context, *, tool_roles: dict[str, str] | None = None) -> dict[str, Any]:
    """Return provenance metadata for two-segment readiness/data foundations."""
    metadata = _context_metadata(context)
    segments = _segment_metadata(context)
    has_two_segments = all(key in segments and len(segments[key].get("servo_ids", [])) == 4 for key in TWO_SEGMENT_KEYS)
    assembly = dict(metadata.get("physical_assembly") or {})
    assembly_issues = list(metadata.get("physical_assembly_issues") or [])
    bottom_top_block = _bottom_top_block(assembly=assembly, segments=segments) if assembly else {}
    return {
        "schema_version": TWO_SEGMENT_FOUNDATION_SCHEMA_VERSION,
        "available": bool(has_two_segments and (not assembly or not assembly_issues)),
        "operating_mode": str(getattr(context, "operating_mode", "")),
        "semantic_scope": "data_structures_and_metadata_only",
        "segment_order": [str(value) for value in metadata.get("segment_order", [])],
        "segments": segments,
        "expected_servo_ids": [int(value) for value in getattr(context, "expected_servo_ids", [])],
        "commanded_servo_ids": [int(value) for value in getattr(context, "commanded_servo_ids", [])],
        "command_schema": two_segment_command_schema(context),
        "pose_schema": two_segment_pose_schema(tool_roles=tool_roles),
        "physical_assembly": dict(assembly),
        "physical_assembly_issues": list(assembly_issues),
        "physical_assembly_valid": not bool(assembly_issues),
        "bottom_top": bottom_top_block,
        "enabled_capabilities": {
            "all_8_readiness": bool(metadata.get("mode_capabilities", {}).get("all_8_readiness", False)),
            "manual_startup_capture": bool(metadata.get("mode_capabilities", {}).get("manual_startup_capture", False)),
            "bottom_top_role_selection": bool(metadata.get("mode_capabilities", {}).get("bottom_top_role_selection", False)),
        },
        "blocked_capabilities": {
            "two_segment_kinematics_control": True,
            "two_segment_modeling": True,
            "two_segment_penprobe_chasing": True,
            "automatic_two_segment_pretension": True,
        },
        "parallel_single_semantics": "mirrored single-segment testing only; not true two-segment control",
    }


def _bottom_top_block(*, assembly: dict[str, Any], segments: dict[str, dict[str, Any]]) -> dict[str, Any]:
    bottom_key = str(assembly.get("bottom_segment_key") or "")
    top_key = str(assembly.get("top_segment_key") or "")
    bottom = dict(segments.get(bottom_key) or {})
    top = dict(segments.get(top_key) or {})
    return {
        "bottom_segment_key": bottom_key,
        "top_segment_key": top_key,
        "bottom_segment_label": str(bottom.get("segment_label") or bottom.get("label") or ""),
        "top_segment_label": str(top.get("segment_label") or top.get("label") or ""),
        "bottom_servo_ids": [int(value) for value in list(bottom.get("servo_ids") or [])],
        "top_servo_ids": [int(value) for value in list(top.get("servo_ids") or [])],
        "bottom_pairs": {str(k): list(v) for k, v in dict(bottom.get("pairs") or {}).items()},
        "top_pairs": {str(k): list(v) for k, v in dict(top.get("pairs") or {}).items()},
        "segment_order_bottom_first": [bottom_key, top_key] if bottom_key and top_key else [],
        "segment_role_assignment": {bottom_key: "proximal", top_key: "distal"} if bottom_key and top_key else {},
        "tendon_axis_convention_per_segment": str(assembly.get("tendon_axis_convention_per_segment") or "[+X, +Y, -X, -Y]"),
        "lower_tick_means_tension": bool(assembly.get("lower_tick_means_tension", True)),
        "composed_frame_note": str(assembly.get("composed_frame_note") or ""),
    }


def _coerce_four_vector(value: Any, *, label: str) -> tuple[float, float, float, float]:
    vector = [float(item) for item in list(value or [])]
    if len(vector) != 4:
        raise ValueError(f"{label} command requires exactly 4 values; found {len(vector)}.")
    return (vector[0], vector[1], vector[2], vector[3])


def _validate_two_segment_context(context) -> None:
    mode = str(getattr(context, "operating_mode", "") or "")
    if mode != "dual_segment":
        raise ValueError(
            "Canonical two-segment commands require operating_mode=dual_segment. "
            f"Current mode is {mode or 'unknown'}."
        )
    segment_ids = _segment_servo_ids(context)
    for key in TWO_SEGMENT_KEYS:
        if len(segment_ids.get(key, [])) != 4:
            raise ValueError(f"{key} must define exactly 4 servo IDs; found {segment_ids.get(key, [])}.")


def _segment_servo_ids(context) -> dict[str, list[int]]:
    segments = _segment_metadata(context)
    return {
        key: [int(value) for value in segments.get(key, {}).get("servo_ids", [])]
        for key in TWO_SEGMENT_KEYS
    }


def _segment_metadata(context) -> dict[str, dict[str, Any]]:
    metadata = _context_metadata(context)
    raw_segments = dict(metadata.get("segments", {}) or {})
    clean: dict[str, dict[str, Any]] = {}
    for key, value in raw_segments.items():
        data = dict(value or {})
        clean[str(key)] = {
            "key": str(data.get("key", key)),
            "label": str(data.get("label", key)),
            "segment_label": str(data.get("segment_label", data.get("label", key))),
            "segment_role": str(data.get("segment_role", "")),
            "segment_order_index": data.get("segment_order_index"),
            "servo_ids": [int(item) for item in list(data.get("servo_ids", []) or [])],
            "pairs": {
                str(pair_key): [int(item) for item in list(pair_values or [])]
                for pair_key, pair_values in dict(data.get("pairs", {}) or {}).items()
            },
        }
    return clean


def _context_metadata(context) -> dict[str, Any]:
    reader = getattr(context, "metadata", None)
    if callable(reader):
        payload = reader()
        return dict(payload or {}) if isinstance(payload, dict) else {}
    return {}


def _clean_mapping(value: dict[str, Any]) -> dict[str, Any]:
    clean: dict[str, Any] = {}
    for key, item in dict(value or {}).items():
        if isinstance(item, dict):
            clean[str(key)] = _clean_mapping(item)
        elif isinstance(item, (list, tuple)):
            clean[str(key)] = [
                child for child in item if child is None or isinstance(child, (str, int, float, bool))
            ]
        elif item is None or isinstance(item, (str, int, float, bool)):
            clean[str(key)] = item
    return clean

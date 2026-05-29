"""Runtime nearest-lookup controller for the two-segment penprobe demo.

Given a workspace lookup map (built offline by
``continuum_robot.demo.two_segment_workspace_lookup``) and a live target XYZ
in robot base frame (typically the 0B penprobe), this controller selects the
nearest reachable map point and returns the all-8 servo goal ticks. It does
NOT touch the bus — the demo experiment is responsible for the actual write.

The controller is deliberately conservative:
- nearest raw sample is the default; interpolation is opt-in
- selected goal ticks must obey the per-servo safe bounds derived from the
  loaded map and operator-configurable absolute tick clamps
- if the tracker is stale, if the map is empty, or if the current bottom/top
  assignment / servo IDs disagree with the map, the controller returns a
  decision with ``command_allowed = False`` and a block reason
- never throws on bad input; every disallow is reported as a decision so the
  demo loop and GUI can surface it
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from continuum_robot.demo.two_segment_workspace_lookup import (
    MAP_SCHEMA_VERSION,
    SUPPORTED_INTERPOLATION_MODES,
    load_workspace_lookup_map,
    nearest_lookup,
    workspace_bounds_mm,
)


DEFAULT_NEAREST_DISTANCE_WARNING_MM = 5.0
DEFAULT_MAX_NEAREST_DISTANCE_MM = 25.0
DEFAULT_ABSOLUTE_MIN_TICK = 0
DEFAULT_ABSOLUTE_MAX_TICK = 4095
COMPENSATION_HEADROOM_FACTOR = 1.25


@dataclass
class LookupControllerConfig:
    """Operator-configurable safety + behavior parameters for the controller."""

    interpolation_mode: str = "nearest"
    knn: int = 1
    nearest_distance_warning_mm: float = DEFAULT_NEAREST_DISTANCE_WARNING_MM
    max_nearest_distance_mm: float = DEFAULT_MAX_NEAREST_DISTANCE_MM
    expected_commanded_servo_ids: list[int] = field(default_factory=lambda: [1, 2, 3, 4, 5, 6, 7, 8])
    expected_bottom_segment_key: str | None = None
    expected_top_segment_key: str | None = None
    # When the map carries no bottom/top assignment (e.g. built from an early
    # servo_only dataset that never recorded `physical_assembly`), treat the
    # assignment as UNKNOWN rather than a conflict. With this flag set, an
    # unknown map assembly is allowed (servo IDs still must match, and a
    # genuine *conflicting* assignment is still hard-blocked). Library default
    # is strict; the demo opts in.
    allow_unknown_map_assembly: bool = False
    absolute_min_tick: int = DEFAULT_ABSOLUTE_MIN_TICK
    absolute_max_tick: int = DEFAULT_ABSOLUTE_MAX_TICK
    extra_per_servo_tick_headroom: int = 0

    def __post_init__(self) -> None:
        if self.interpolation_mode not in SUPPORTED_INTERPOLATION_MODES:
            raise ValueError(
                f"Unsupported interpolation_mode {self.interpolation_mode!r}; "
                f"expected one of {SUPPORTED_INTERPOLATION_MODES}."
            )
        if int(self.knn) < 1:
            raise ValueError("knn must be >= 1")


@dataclass(frozen=True)
class LookupDecision:
    """Per-cycle decision object returned by the controller."""

    target_xyz_robot_mm: list[float]
    selected_point_xyz_robot_mm: list[float] | None
    nearest_distance_mm: float | None
    selected_map_index: int | None
    source_sample_ids: list[str]
    all_8_goal_ticks: dict[str, int] | None
    bottom_top_assignment: dict[str, Any]
    interpolation_mode: str
    knn_used: int
    is_extrapolating: bool
    inside_workspace_hint: bool
    command_allowed: bool
    block_reason: str | None
    safety_limiter_reasons: list[str]

    def to_dict(self) -> dict[str, Any]:
        return {
            "target_xyz_robot_mm": list(self.target_xyz_robot_mm),
            "selected_point_xyz_robot_mm": (
                None if self.selected_point_xyz_robot_mm is None else list(self.selected_point_xyz_robot_mm)
            ),
            "nearest_distance_mm": self.nearest_distance_mm,
            "selected_map_index": self.selected_map_index,
            "source_sample_ids": list(self.source_sample_ids),
            "all_8_goal_ticks": None if self.all_8_goal_ticks is None else dict(self.all_8_goal_ticks),
            "bottom_top_assignment": dict(self.bottom_top_assignment),
            "interpolation_mode": str(self.interpolation_mode),
            "knn_used": int(self.knn_used),
            "is_extrapolating": bool(self.is_extrapolating),
            "inside_workspace_hint": bool(self.inside_workspace_hint),
            "command_allowed": bool(self.command_allowed),
            "block_reason": self.block_reason,
            "safety_limiter_reasons": list(self.safety_limiter_reasons),
        }


def _blocked(*, target, reason: str, assembly: dict[str, Any], mode: str, knn: int) -> LookupDecision:
    return LookupDecision(
        target_xyz_robot_mm=[float(v) for v in (target if target is not None else [0.0, 0.0, 0.0])],
        selected_point_xyz_robot_mm=None,
        nearest_distance_mm=None,
        selected_map_index=None,
        source_sample_ids=[],
        all_8_goal_ticks=None,
        bottom_top_assignment=dict(assembly or {}),
        interpolation_mode=mode,
        knn_used=int(knn),
        is_extrapolating=False,
        inside_workspace_hint=False,
        command_allowed=False,
        block_reason=str(reason),
        safety_limiter_reasons=[],
    )


class TwoSegmentWorkspaceLookupController:
    """Loads a lookup map and selects the nearest reachable command per target."""

    def __init__(
        self,
        *,
        map_payload: dict[str, Any],
        config: LookupControllerConfig | None = None,
    ) -> None:
        if not isinstance(map_payload, dict):
            raise TypeError("map_payload must be a dict (use load_from_path() or pass a parsed map).")
        if map_payload.get("schema_version") != MAP_SCHEMA_VERSION:
            raise ValueError(
                f"Map schema mismatch: got {map_payload.get('schema_version')!r}, expected {MAP_SCHEMA_VERSION!r}."
            )
        self._map_payload = map_payload
        self._map_points: list[dict[str, Any]] = list(map_payload.get("map_points") or [])
        self.config = config or LookupControllerConfig()
        # Pre-compute per-servo tick bounds across the map, with optional
        # operator headroom; the interpolated/nearest command must stay
        # inside this envelope to count as safe.
        self._per_servo_min_tick, self._per_servo_max_tick = self._compute_per_servo_bounds()
        # Bounds + assembly are exposed for the GUI + demo runtime to inspect.
        self.metadata: dict[str, Any] = dict(map_payload.get("metadata") or {})
        self.workspace_bounds = workspace_bounds_mm(self._map_points)

    # ------------------------------------------------------------------
    # Factory / introspection
    # ------------------------------------------------------------------

    @classmethod
    def load_from_path(
        cls, path: Path, *, config: LookupControllerConfig | None = None
    ) -> "TwoSegmentWorkspaceLookupController":
        payload = load_workspace_lookup_map(Path(path))
        return cls(map_payload=payload, config=config)

    @property
    def map_point_count(self) -> int:
        return len(self._map_points)

    @property
    def bottom_top_assignment(self) -> dict[str, Any]:
        return dict(self.metadata.get("bottom_top_assignment") or {})

    @property
    def commanded_servo_ids(self) -> list[int]:
        return [int(v) for v in (self.metadata.get("commanded_servo_ids") or [])]

    @property
    def map_assembly_is_unknown(self) -> bool:
        """True when the map carries no bottom/top segment assignment.

        This happens when the source dataset never recorded ``physical_assembly``
        (typical of early servo_only runs). The map is still usable — the goal
        ticks are keyed by servo ID — but the bottom/top compatibility check
        cannot be verified, so the demo surfaces a loud warning.
        """
        assembly = self.bottom_top_assignment
        return not str(assembly.get("bottom_segment_key") or "") and not str(
            assembly.get("top_segment_key") or ""
        )

    # ------------------------------------------------------------------
    # Compatibility checks (called once per demo run)
    # ------------------------------------------------------------------

    def check_compatibility(self) -> tuple[bool, list[str]]:
        """Return ``(ok, reasons)``: whether the map matches the operator's expected runtime config."""
        reasons: list[str] = []
        if self.map_point_count == 0:
            reasons.append("map_empty")
        expected_ids = sorted(int(v) for v in self.config.expected_commanded_servo_ids)
        map_ids = sorted(self.commanded_servo_ids)
        if expected_ids and map_ids and expected_ids != map_ids:
            reasons.append(
                f"servo_id_mismatch:map={map_ids},runtime={expected_ids}"
            )
        assembly = self.bottom_top_assignment
        allow_unknown = bool(getattr(self.config, "allow_unknown_map_assembly", False))
        map_bottom = str(assembly.get("bottom_segment_key") or "")
        map_top = str(assembly.get("top_segment_key") or "")
        if self.config.expected_bottom_segment_key:
            if not map_bottom:
                # Map never recorded a bottom/top assignment: UNKNOWN, not a
                # conflict. Allowed when the operator opts in; otherwise block
                # with a reason that says it is missing (not mismatched).
                if not allow_unknown:
                    reasons.append(
                        f"bottom_segment_key_unknown_in_map:runtime={self.config.expected_bottom_segment_key!r}"
                    )
            elif map_bottom != str(self.config.expected_bottom_segment_key):
                reasons.append(
                    f"bottom_segment_key_mismatch:map={assembly.get('bottom_segment_key')!r},"
                    f"runtime={self.config.expected_bottom_segment_key!r}"
                )
        if self.config.expected_top_segment_key:
            if not map_top:
                if not allow_unknown:
                    reasons.append(
                        f"top_segment_key_unknown_in_map:runtime={self.config.expected_top_segment_key!r}"
                    )
            elif map_top != str(self.config.expected_top_segment_key):
                reasons.append(
                    f"top_segment_key_mismatch:map={assembly.get('top_segment_key')!r},"
                    f"runtime={self.config.expected_top_segment_key!r}"
                )
        # A map that pooled disagreeing assemblies is a genuine conflict and is
        # always blocked, regardless of the unknown-assembly allowance.
        if assembly.get("mixed_assemblies_detected"):
            reasons.append("map_mixed_assemblies_detected")
        return (not reasons, reasons)

    # ------------------------------------------------------------------
    # Decision per cycle
    # ------------------------------------------------------------------

    def decide(
        self,
        *,
        target_xyz_robot_mm: list[float] | np.ndarray | None,
        tracker_stale: bool = False,
    ) -> LookupDecision:
        """Pick the nearest/interpolated map point for the current target."""
        assembly = self.bottom_top_assignment
        config = self.config
        if self.map_point_count == 0:
            return _blocked(
                target=target_xyz_robot_mm,
                reason="map_empty",
                assembly=assembly,
                mode=config.interpolation_mode,
                knn=config.knn,
            )
        if target_xyz_robot_mm is None:
            return _blocked(
                target=None,
                reason="target_unavailable",
                assembly=assembly,
                mode=config.interpolation_mode,
                knn=config.knn,
            )
        if bool(tracker_stale):
            return _blocked(
                target=target_xyz_robot_mm,
                reason="tracker_stale",
                assembly=assembly,
                mode=config.interpolation_mode,
                knn=config.knn,
            )
        try:
            target = np.asarray(target_xyz_robot_mm, dtype=float).reshape(3)
        except Exception:
            return _blocked(
                target=target_xyz_robot_mm,
                reason="target_xyz_invalid",
                assembly=assembly,
                mode=config.interpolation_mode,
                knn=config.knn,
            )
        compatibility_ok, compatibility_reasons = self.check_compatibility()
        if not compatibility_ok:
            return _blocked(
                target=target,
                reason="map_compatibility_failed",
                assembly=assembly,
                mode=config.interpolation_mode,
                knn=config.knn,
            ).__class__(
                target_xyz_robot_mm=[float(v) for v in target.tolist()],
                selected_point_xyz_robot_mm=None,
                nearest_distance_mm=None,
                selected_map_index=None,
                source_sample_ids=[],
                all_8_goal_ticks=None,
                bottom_top_assignment=dict(assembly or {}),
                interpolation_mode=str(config.interpolation_mode),
                knn_used=int(config.knn),
                is_extrapolating=False,
                inside_workspace_hint=False,
                command_allowed=False,
                block_reason="map_compatibility_failed",
                safety_limiter_reasons=list(compatibility_reasons),
            )
        result = nearest_lookup(
            target,
            map_points=self._map_points,
            knn=int(config.knn),
            interpolation=str(config.interpolation_mode),
        )
        nearest_distance = float(result["nearest_distance_mm"])
        inside_workspace = self._inside_workspace_hint(target)
        is_extrapolating = (
            not inside_workspace and nearest_distance > float(config.nearest_distance_warning_mm)
        )
        # Per-servo bounds safety. Interpolation can push goal ticks outside
        # the observed envelope; clamp and block when the clamp is material.
        ticks = {str(k): int(v) for k, v in result["all_8_goal_ticks"].items()}
        limiter_reasons: list[str] = []
        unsafe_servo_ids: list[int] = []
        for servo_id_str, tick in ticks.items():
            try:
                servo_id = int(servo_id_str)
            except ValueError:
                limiter_reasons.append(f"non_integer_servo_id:{servo_id_str}")
                unsafe_servo_ids.append(-1)
                continue
            lower = self._per_servo_min_tick.get(servo_id, config.absolute_min_tick)
            upper = self._per_servo_max_tick.get(servo_id, config.absolute_max_tick)
            lower = max(int(lower), int(config.absolute_min_tick))
            upper = min(int(upper), int(config.absolute_max_tick))
            if int(tick) < lower or int(tick) > upper:
                limiter_reasons.append(
                    f"servo_{servo_id}_tick_out_of_bounds:tick={int(tick)},bounds=[{lower},{upper}]"
                )
                unsafe_servo_ids.append(servo_id)
        if nearest_distance > float(config.max_nearest_distance_mm):
            limiter_reasons.append(
                f"nearest_distance_above_hard_threshold:{nearest_distance:.3f}>"
                f"{float(config.max_nearest_distance_mm):.3f}_mm"
            )
        command_allowed = not unsafe_servo_ids and (
            nearest_distance <= float(config.max_nearest_distance_mm)
        )
        block_reason = None if command_allowed else "safety_limiter_active"
        return LookupDecision(
            target_xyz_robot_mm=[float(v) for v in target.tolist()],
            selected_point_xyz_robot_mm=[float(v) for v in result["selected_point_xyz_robot_mm"]],
            nearest_distance_mm=nearest_distance,
            selected_map_index=int(result.get("selected_map_index", -1)),
            source_sample_ids=list(result.get("source_sample_ids") or []),
            all_8_goal_ticks=ticks,
            bottom_top_assignment=dict(assembly or {}),
            interpolation_mode=str(config.interpolation_mode),
            knn_used=int(result.get("knn_used", config.knn)),
            is_extrapolating=bool(is_extrapolating),
            inside_workspace_hint=bool(inside_workspace),
            command_allowed=bool(command_allowed),
            block_reason=block_reason,
            safety_limiter_reasons=list(limiter_reasons),
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _compute_per_servo_bounds(self) -> tuple[dict[int, int], dict[int, int]]:
        per_min: dict[int, int] = {}
        per_max: dict[int, int] = {}
        if not self._map_points:
            return per_min, per_max
        for point in self._map_points:
            for sid_str, tick in (point.get("all_8_goal_ticks") or {}).items():
                try:
                    sid = int(sid_str)
                    tick_int = int(tick)
                except (TypeError, ValueError):
                    continue
                if sid not in per_min or tick_int < per_min[sid]:
                    per_min[sid] = tick_int
                if sid not in per_max or tick_int > per_max[sid]:
                    per_max[sid] = tick_int
        headroom = max(0, int(self.config.extra_per_servo_tick_headroom))
        if headroom:
            for sid in list(per_min):
                per_min[sid] = max(int(self.config.absolute_min_tick), per_min[sid] - headroom)
                per_max[sid] = min(int(self.config.absolute_max_tick), per_max[sid] + headroom)
        return per_min, per_max

    def _inside_workspace_hint(self, target: np.ndarray) -> bool:
        bounds = self.workspace_bounds or {}
        lo = bounds.get("min_xyz_mm")
        hi = bounds.get("max_xyz_mm")
        if not isinstance(lo, list) or not isinstance(hi, list):
            return False
        for axis in range(3):
            if float(target[axis]) < float(lo[axis]) or float(target[axis]) > float(hi[axis]):
                return False
        return True

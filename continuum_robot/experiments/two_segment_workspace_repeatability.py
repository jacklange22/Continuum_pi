"""Two-segment workspace repeatability experiment.

Hardware-ready data-collection workflow that mirrors the single-segment
``workspace_repeatability_map`` discipline but commands the stacked
two-segment robot in 4D tendon-space (bottom_x, bottom_y, top_x, top_y).

Protocol:
- Generate ``target_count`` (default 200) two-segment workspace targets.
- For each target visit: neutral → target → settle → capture → neutral.
- Round-robin shuffled visit order: each ``repeats_per_target`` (default 20)
  cycle reshuffles all targets so heating/drift biases cancel out.
- Total planned target captures = target_count × repeats_per_target
  (default 200 × 20 = 4000 accepted target captures).

The experiment reuses the safe command path from
``two_segment_collect_pose_dataset``:
- accepted all-8 startup artifact
- startup-relative commands
- top-routing compensation
- per-servo safe-bound + wrap-guard checks
- current/load proxy + hardware error stops
- transport recovery via ServoService

It is NOT a presentation demo — it produces an experiment-grade run folder
with samples.jsonl, per-target metrics, and thesis-style figures.

Primary repeatability metric:
- distal XYZ scatter at the configured ``distal_tip`` tool (default ``0A``)
  measured in robot base frame.

Intermediate / orientation are recorded when available but never required
for accepted-capture validity.
"""

from __future__ import annotations

import csv
import json
import math
import random
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from continuum_robot.experiments.framework import (
    BaseExperiment,
    ExperimentHardwareRequirements,
    ExperimentSession,
)
from continuum_robot.experiments.schemas import ExperimentTimeseriesSample
from continuum_robot.experiments.two_segment_collect_pose_dataset import (
    _apply_top_routing_compensation,
    _effective_role_configs,
    _read_live_telemetry,
    _resolve_startup_ticks,
    _servo_feedback_payload,
    _two_segment_pose_observation,
    tick_budget_for_segment_amplitude_cm,
)
from continuum_robot.two_segment import (
    TwoSegmentCommand,
    build_two_segment_foundation_metadata,
)


EXPERIMENT_NAME = "two_segment_workspace_repeatability"
REPEATABILITY_SCHEMA_VERSION = "two_segment_workspace_repeatability_v1"

DEFAULT_TARGET_COUNT = 200
DEFAULT_REPEATS_PER_TARGET = 20
DEFAULT_AMPLITUDE_CM = 0.25
DEFAULT_NEUTRAL_SETTLE_S = 1.0
DEFAULT_TARGET_SETTLE_S = 1.0
DEFAULT_CAPTURE_DWELL_S = 0.0
DEFAULT_MAX_TRACKER_AGE_S = 0.25
DEFAULT_MIN_REPEATS_PER_TARGET = 15
DEFAULT_MAX_REJECTED_FRACTION = 0.10
SUPPORTED_TARGET_GENERATOR_MODES = (
    "workspace_latin_hypercube",
    "rings_and_axes",
    "grid_subsample",
)
AMPLITUDE_PRESETS_CM = (0.10, 0.25, 0.50, 0.75, 1.00)

BLOCK_MESSAGE = (
    "Two-segment workspace repeatability requires operating_mode=dual_segment, "
    "all 8 servos, an accepted all-8 manual startup foundation, and a confirmed "
    "bottom/top physical assembly."
)


# ---------------------------------------------------------------------------
# Target data class
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TwoSegmentWorkspaceTarget:
    """One commanded target in the 4D two-segment command space."""

    target_index: int
    target_id: str
    bottom_x_cm: float
    bottom_y_cm: float
    top_x_cm: float
    top_y_cm: float
    bottom_tendon_cm: list[float]
    top_tendon_cm: list[float]
    group_tag: str
    amplitude_cm: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "target_index": int(self.target_index),
            "target_id": str(self.target_id),
            "bottom_x_cm": float(self.bottom_x_cm),
            "bottom_y_cm": float(self.bottom_y_cm),
            "top_x_cm": float(self.top_x_cm),
            "top_y_cm": float(self.top_y_cm),
            "bottom_tendon_cm": [float(v) for v in self.bottom_tendon_cm],
            "top_tendon_cm": [float(v) for v in self.top_tendon_cm],
            "group_tag": str(self.group_tag),
            "amplitude_cm": float(self.amplitude_cm),
        }


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclass
class TwoSegmentWorkspaceRepeatabilityConfig:
    """Operator-configurable parameters for the workspace repeatability run."""

    target_count: int = DEFAULT_TARGET_COUNT
    repeats_per_target: int = DEFAULT_REPEATS_PER_TARGET
    max_segment_displacement_cm: float = DEFAULT_AMPLITUDE_CM
    target_generator_mode: str = "workspace_latin_hypercube"
    random_seed: int = 0
    return_to_neutral_between_visits: bool = True
    neutral_settle_s: float = DEFAULT_NEUTRAL_SETTLE_S
    target_settle_s: float = DEFAULT_TARGET_SETTLE_S
    capture_dwell_s: float = DEFAULT_CAPTURE_DWELL_S
    max_tracker_age_s: float = DEFAULT_MAX_TRACKER_AGE_S
    expected_distal_tool_id: str = "0A"
    distal_tool_role: str = "distal_tip"
    require_distal_tool_visible: bool = True
    max_tick_delta_from_startup: int = 0  # 0 => auto from amplitude
    samples_per_visit: int = 1
    capture_neutral_sample: bool = False
    continue_until_valid_captures: bool = True
    target_valid_capture_count: int = 0  # 0 => target_count × repeats_per_target
    max_failed_visits: int = 0  # 0 => disabled
    max_consecutive_failed_visits: int = 0  # 0 => disabled
    max_rejected_fraction: float = DEFAULT_MAX_REJECTED_FRACTION
    minimum_repeats_per_target: int = DEFAULT_MIN_REPEATS_PER_TARGET
    target_distal_rms_threshold_mm: float | None = None
    top_segment_tendon_routing_compensation: bool = True
    capture_tracker_snapshot: bool = True
    requested_tool_roles: dict[str, str] = field(default_factory=dict)
    allow_servo_only_test_run: bool = False
    run_trust_mode: str = "repeatability_run"
    physical_assembly_confirmed_by_operator: bool = False
    dry_run: bool = False
    run_label: str = ""

    @classmethod
    def from_dict(cls, payload: dict[str, Any] | None) -> "TwoSegmentWorkspaceRepeatabilityConfig":
        payload = dict(payload or {})
        mode = str(payload.get("target_generator_mode", "workspace_latin_hypercube") or "workspace_latin_hypercube")
        mode = mode.strip().lower()
        if mode not in SUPPORTED_TARGET_GENERATOR_MODES:
            mode = "workspace_latin_hypercube"
        requested_roles = payload.get("requested_tool_roles") or payload.get("tool_roles") or {}
        target_threshold = payload.get("target_distal_rms_threshold_mm")
        return cls(
            target_count=max(3, int(payload.get("target_count", DEFAULT_TARGET_COUNT))),
            repeats_per_target=max(1, int(payload.get("repeats_per_target", DEFAULT_REPEATS_PER_TARGET))),
            max_segment_displacement_cm=max(
                0.0, float(payload.get("max_segment_displacement_cm", DEFAULT_AMPLITUDE_CM))
            ),
            target_generator_mode=mode,
            random_seed=int(payload.get("random_seed", 0)),
            return_to_neutral_between_visits=bool(payload.get("return_to_neutral_between_visits", True)),
            neutral_settle_s=max(0.0, float(payload.get("neutral_settle_s", DEFAULT_NEUTRAL_SETTLE_S))),
            target_settle_s=max(0.0, float(payload.get("target_settle_s", DEFAULT_TARGET_SETTLE_S))),
            capture_dwell_s=max(0.0, float(payload.get("capture_dwell_s", DEFAULT_CAPTURE_DWELL_S))),
            max_tracker_age_s=max(0.01, float(payload.get("max_tracker_age_s", DEFAULT_MAX_TRACKER_AGE_S))),
            expected_distal_tool_id=str(payload.get("expected_distal_tool_id", "0A") or "0A").upper(),
            distal_tool_role=str(payload.get("distal_tool_role", "distal_tip") or "distal_tip"),
            require_distal_tool_visible=bool(payload.get("require_distal_tool_visible", True)),
            max_tick_delta_from_startup=max(0, int(payload.get("max_tick_delta_from_startup", 0))),
            samples_per_visit=max(1, int(payload.get("samples_per_visit", 1))),
            capture_neutral_sample=bool(payload.get("capture_neutral_sample", False)),
            continue_until_valid_captures=bool(payload.get("continue_until_valid_captures", True)),
            target_valid_capture_count=max(0, int(payload.get("target_valid_capture_count", 0))),
            max_failed_visits=max(0, int(payload.get("max_failed_visits", 0))),
            max_consecutive_failed_visits=max(0, int(payload.get("max_consecutive_failed_visits", 0))),
            max_rejected_fraction=max(0.0, min(1.0, float(payload.get("max_rejected_fraction", DEFAULT_MAX_REJECTED_FRACTION)))),
            minimum_repeats_per_target=max(1, int(payload.get("minimum_repeats_per_target", DEFAULT_MIN_REPEATS_PER_TARGET))),
            target_distal_rms_threshold_mm=(
                None if target_threshold in (None, "") else float(target_threshold)
            ),
            top_segment_tendon_routing_compensation=bool(
                payload.get("top_segment_tendon_routing_compensation", True)
            ),
            capture_tracker_snapshot=bool(payload.get("capture_tracker_snapshot", True)),
            requested_tool_roles={str(k).upper(): str(v) for k, v in dict(requested_roles or {}).items()},
            allow_servo_only_test_run=bool(payload.get("allow_servo_only_test_run", False)),
            run_trust_mode=str(payload.get("run_trust_mode", "repeatability_run") or "repeatability_run"),
            physical_assembly_confirmed_by_operator=bool(
                payload.get("physical_assembly_confirmed_by_operator", False)
            ),
            dry_run=bool(payload.get("dry_run", False)),
            run_label=str(payload.get("run_label", "") or ""),
        )


# ---------------------------------------------------------------------------
# Target generation (pure functions)
# ---------------------------------------------------------------------------


def _tendon_vector_from_xy(x_cm: float, y_cm: float) -> list[float]:
    """Tendon order is [+X, +Y, -X, -Y]; antagonistic about the centre column."""
    return [float(x_cm), float(y_cm), -float(x_cm), -float(y_cm)]


def _latin_hypercube_4d(
    *,
    target_count: int,
    amplitude_cm: float,
    seed: int,
) -> list[tuple[float, float, float, float]]:
    """Generate ``target_count`` 4D Latin-hypercube samples in [-amp, +amp]^4.

    Standard LHS: stratify each dimension into N bins, pick the midpoint of
    each bin, then independently shuffle each dimension's order. The result
    is space-filling along every 1D projection and approximately uniform
    over the 4D box. Deterministic under ``seed``.
    """
    if target_count < 1:
        return []
    rng = np.random.default_rng(int(seed))
    edges = np.linspace(-1.0, 1.0, target_count + 1)
    midpoints = (edges[:-1] + edges[1:]) / 2.0  # length = target_count
    samples = []
    cols = []
    for _ in range(4):
        col = midpoints.copy()
        rng.shuffle(col)
        cols.append(col)
    cols = np.stack(cols, axis=1)  # shape (target_count, 4)
    cols *= float(amplitude_cm)
    for row in cols:
        samples.append((float(row[0]), float(row[1]), float(row[2]), float(row[3])))
    return samples


def _rings_and_axes_4d(
    *,
    target_count: int,
    amplitude_cm: float,
    seed: int,
) -> list[tuple[float, float, float, float]]:
    """Deterministic rings-and-axes set, optionally padded with LHS to reach target_count.

    Lays out:
    - 1 origin
    - 8 axis points on bottom (4 cardinal × 2 amplitudes)
    - 8 axis points on top  (4 cardinal × 2 amplitudes)
    - 8 combined cardinal points
    The remainder is padded with LHS so the total equals ``target_count``.
    """
    if target_count < 1:
        return []
    half = amplitude_cm * 0.5
    full = amplitude_cm
    samples: list[tuple[float, float, float, float]] = [(0.0, 0.0, 0.0, 0.0)]
    cardinal_xy = [(full, 0.0), (0.0, full), (-full, 0.0), (0.0, -full)]
    half_xy = [(half, 0.0), (0.0, half), (-half, 0.0), (0.0, -half)]
    for x, y in cardinal_xy + half_xy:
        samples.append((float(x), float(y), 0.0, 0.0))
        samples.append((0.0, 0.0, float(x), float(y)))
    for (bx, by), (tx, ty) in zip(cardinal_xy, cardinal_xy):
        samples.append((float(bx), float(by), float(tx), float(ty)))
    samples = samples[: int(target_count)]
    if len(samples) < int(target_count):
        padding = _latin_hypercube_4d(
            target_count=int(target_count - len(samples)),
            amplitude_cm=amplitude_cm,
            seed=seed + 1,
        )
        samples.extend(padding)
    return samples[: int(target_count)]


def _grid_subsample_4d(
    *,
    target_count: int,
    amplitude_cm: float,
    seed: int,
) -> list[tuple[float, float, float, float]]:
    """4D grid then maximin farthest-point downsample to ``target_count``.

    A 4D grid grows quickly; we bound the per-axis count to keep the seed
    set under ~3000 points before downsampling.
    """
    if target_count < 1:
        return []
    per_axis = max(3, int(math.ceil(target_count ** 0.25) * 2))
    coords = np.linspace(-float(amplitude_cm), float(amplitude_cm), per_axis)
    grid = np.array(np.meshgrid(coords, coords, coords, coords, indexing="ij")).reshape(4, -1).T
    if len(grid) > 4000:
        rng = np.random.default_rng(int(seed))
        keep = rng.choice(len(grid), size=4000, replace=False)
        grid = grid[keep]
    if len(grid) <= int(target_count):
        return [tuple(float(v) for v in row) for row in grid]
    rng = np.random.default_rng(int(seed))
    selected = [int(rng.integers(len(grid)))]
    distances = np.linalg.norm(grid - grid[selected[0]], axis=1)
    while len(selected) < int(target_count):
        next_idx = int(np.argmax(distances))
        if distances[next_idx] <= 0:
            break
        selected.append(next_idx)
        new_d = np.linalg.norm(grid - grid[next_idx], axis=1)
        distances = np.minimum(distances, new_d)
    return [tuple(float(v) for v in grid[i]) for i in selected]


def build_two_segment_workspace_targets(
    config: TwoSegmentWorkspaceRepeatabilityConfig,
) -> list[TwoSegmentWorkspaceTarget]:
    """Build the canonical target list for the workspace repeatability run."""
    if config.target_generator_mode == "workspace_latin_hypercube":
        raw = _latin_hypercube_4d(
            target_count=int(config.target_count),
            amplitude_cm=float(config.max_segment_displacement_cm),
            seed=int(config.random_seed),
        )
    elif config.target_generator_mode == "rings_and_axes":
        raw = _rings_and_axes_4d(
            target_count=int(config.target_count),
            amplitude_cm=float(config.max_segment_displacement_cm),
            seed=int(config.random_seed),
        )
    elif config.target_generator_mode == "grid_subsample":
        raw = _grid_subsample_4d(
            target_count=int(config.target_count),
            amplitude_cm=float(config.max_segment_displacement_cm),
            seed=int(config.random_seed),
        )
    else:  # pragma: no cover — guarded in config
        raise ValueError(f"Unsupported target_generator_mode: {config.target_generator_mode!r}")
    amplitude = float(config.max_segment_displacement_cm)
    targets: list[TwoSegmentWorkspaceTarget] = []
    for index, (bx, by, tx, ty) in enumerate(raw):
        target_amplitude = float(math.sqrt(bx * bx + by * by + tx * tx + ty * ty))
        group = _group_tag(bx, by, tx, ty, amplitude=amplitude)
        targets.append(
            TwoSegmentWorkspaceTarget(
                target_index=int(index),
                target_id=f"WS_{index:04d}",
                bottom_x_cm=float(bx),
                bottom_y_cm=float(by),
                top_x_cm=float(tx),
                top_y_cm=float(ty),
                bottom_tendon_cm=_tendon_vector_from_xy(bx, by),
                top_tendon_cm=_tendon_vector_from_xy(tx, ty),
                group_tag=group,
                amplitude_cm=target_amplitude,
            )
        )
    return targets


def _group_tag(bx: float, by: float, tx: float, ty: float, *, amplitude: float) -> str:
    """Coarse group label for grouping per-target metrics."""
    threshold = max(1e-6, amplitude * 0.05)
    bottom_active = (bx * bx + by * by) > threshold * threshold
    top_active = (tx * tx + ty * ty) > threshold * threshold
    if not bottom_active and not top_active:
        return "neutral_or_near_neutral"
    if bottom_active and not top_active:
        return "bottom_only"
    if top_active and not bottom_active:
        return "top_only"
    return "combined"


def build_round_robin_visit_order(
    *,
    target_count: int,
    repeats_per_target: int,
    random_seed: int,
) -> list[tuple[int, int, int]]:
    """``(cycle_index, visit_in_cycle, target_index)`` for the run.

    Each cycle visits every target exactly once, in a fresh shuffle seeded
    by ``random_seed + cycle_index`` for deterministic reproducibility.
    """
    if target_count < 1 or repeats_per_target < 1:
        return []
    sequence: list[tuple[int, int, int]] = []
    for cycle_index in range(int(repeats_per_target)):
        rng = random.Random(int(random_seed) * 1_000_003 + int(cycle_index))
        order = list(range(int(target_count)))
        rng.shuffle(order)
        for visit_in_cycle, target_index in enumerate(order):
            sequence.append((int(cycle_index), int(visit_in_cycle), int(target_index)))
    return sequence


# ---------------------------------------------------------------------------
# Command / capture helpers
# ---------------------------------------------------------------------------


def _build_two_segment_command(
    *,
    target: TwoSegmentWorkspaceTarget,
    context,
) -> TwoSegmentCommand:
    """Lay the target into the per-bench bottom/top assignment.

    The TwoSegmentCommand always uses canonical segment_a / segment_b keys
    (which map 1:1 to servo IDs); this helper consults the physical assembly
    metadata to route ``bottom_*`` and ``top_*`` to the correct segment.
    """
    metadata = context.metadata() if callable(getattr(context, "metadata", None)) else {}
    assembly = dict(metadata.get("physical_assembly") or {})
    bottom_key = str(assembly.get("bottom_segment_key") or "segment_a")
    top_key = str(assembly.get("top_segment_key") or "segment_b")
    mapping: dict[str, list[float]] = {bottom_key: list(target.bottom_tendon_cm), top_key: list(target.top_tendon_cm)}
    mapping.setdefault("segment_a", [0.0, 0.0, 0.0, 0.0])
    mapping.setdefault("segment_b", [0.0, 0.0, 0.0, 0.0])
    return TwoSegmentCommand.from_mapping(
        {"segment_a": mapping["segment_a"], "segment_b": mapping["segment_b"]}
    )


@dataclass
class _VisitResult:
    accepted: bool
    target_index: int
    target_id: str
    cycle_index: int
    visit_in_cycle: int
    visit_position: int
    timestamp_utc: str
    monotonic_time_s: float
    bottom_x_cm: float
    bottom_y_cm: float
    top_x_cm: float
    top_y_cm: float
    ordered_8_displacements_cm: list[float]
    servo_flat_command_cm: list[float]
    all_8_goal_ticks: dict[str, int]
    all_8_present_position_ticks: dict[str, int | None]
    all_8_current_load_proxy_ma: dict[str, float | None]
    distal_xyz_robot_mm: list[float] | None
    intermediate_xyz_robot_mm: list[float] | None
    distal_orientation_wxyz: list[float] | None
    command_success: bool
    reject_reason: str | None
    tracker_age_s: float | None
    servo_telemetry_age_s: float | None
    group_tag: str
    amplitude_cm: float

    def to_capture_row(self) -> dict[str, Any]:
        return {
            "timestamp_utc": self.timestamp_utc,
            "monotonic_time_s": self.monotonic_time_s,
            "target_index": self.target_index,
            "target_id": self.target_id,
            "cycle_index": self.cycle_index,
            "visit_in_cycle": self.visit_in_cycle,
            "visit_position": self.visit_position,
            "bottom_x_cm": self.bottom_x_cm,
            "bottom_y_cm": self.bottom_y_cm,
            "top_x_cm": self.top_x_cm,
            "top_y_cm": self.top_y_cm,
            "ordered_8_displacements_cm": list(self.ordered_8_displacements_cm),
            "servo_flat_command_cm": list(self.servo_flat_command_cm),
            "all_8_goal_ticks": dict(self.all_8_goal_ticks),
            "all_8_present_position_ticks": dict(self.all_8_present_position_ticks),
            "all_8_current_load_proxy_ma": dict(self.all_8_current_load_proxy_ma),
            "distal_xyz_robot_mm": (list(self.distal_xyz_robot_mm) if self.distal_xyz_robot_mm is not None else None),
            "intermediate_xyz_robot_mm": (
                list(self.intermediate_xyz_robot_mm) if self.intermediate_xyz_robot_mm is not None else None
            ),
            "distal_orientation_wxyz": (
                list(self.distal_orientation_wxyz) if self.distal_orientation_wxyz is not None else None
            ),
            "command_success": bool(self.command_success),
            "capture_accepted": bool(self.accepted),
            "reject_reason": self.reject_reason,
            "tracker_age_s": self.tracker_age_s,
            "servo_telemetry_age_s": self.servo_telemetry_age_s,
            "group_tag": self.group_tag,
            "amplitude_cm": self.amplitude_cm,
        }


# ---------------------------------------------------------------------------
# Experiment class
# ---------------------------------------------------------------------------


class TwoSegmentWorkspaceRepeatabilityExperiment(BaseExperiment):
    """200×20 two-segment workspace repeatability — hardware-ready protocol."""

    name = EXPERIMENT_NAME
    description = (
        "Two-segment workspace repeatability. 200 (configurable) commanded targets "
        "in 4D tendon space (bottom_x/y, top_x/y), each visited 20 times with "
        "round-robin shuffled ordering and a neutral return between visits. "
        "Primary metric: distal/tip coil XYZ repeatability."
    )
    hardware_requirements = ExperimentHardwareRequirements(
        tracking_required=True,
        servo_required=True,
        registration_required=True,
        mock_compatible=True,
    )

    def __init__(self, config: TwoSegmentWorkspaceRepeatabilityConfig) -> None:
        super().__init__(config)
        self.config: TwoSegmentWorkspaceRepeatabilityConfig = config
        self._startup_ticks_by_servo: dict[int, int] = {}
        self._startup_provenance: dict[str, Any] = {}
        self._targets: list[TwoSegmentWorkspaceTarget] = []
        self._visit_results: list[_VisitResult] = []
        self._failure_events: list[dict[str, Any]] = []
        self._stop_reason: str | None = None
        self._exclusive_owner: bool = False

    @classmethod
    def from_dict(cls, payload: dict[str, Any] | None = None) -> "TwoSegmentWorkspaceRepeatabilityExperiment":
        return cls(TwoSegmentWorkspaceRepeatabilityConfig.from_dict(payload))

    # ----- Lifecycle -----

    def setup(self, session: ExperimentSession) -> None:
        context = session.context.settings.robot.operating_context()
        # Defer startup-tick resolution to precheck when not in dual_segment
        # so the operator-facing error is the dual_segment-block, not the
        # downstream startup-artifact-missing error.
        if context.operating_mode == "dual_segment":
            self._startup_ticks_by_servo, self._startup_provenance = _resolve_startup_ticks(
                session=session,
                config=self.config,
                expected_ids=[int(v) for v in context.expected_servo_ids],
            )
        self._targets = build_two_segment_workspace_targets(self.config)

    def precheck(self, session: ExperimentSession) -> None:
        context = session.context.settings.robot.operating_context()
        if context.operating_mode != "dual_segment":
            raise RuntimeError(BLOCK_MESSAGE)
        expected_ids = [int(v) for v in context.expected_servo_ids]
        if expected_ids != [1, 2, 3, 4, 5, 6, 7, 8]:
            raise RuntimeError(
                "Two-segment workspace repeatability requires all 8 expected servo IDs [1..8]; "
                f"resolved {expected_ids}."
            )
        assembly_issues = list(getattr(context, "physical_assembly_issues", []) or [])
        if assembly_issues:
            raise RuntimeError("Invalid physical assembly: " + "; ".join(assembly_issues))
        servo_service = session.context.servo_service
        if servo_service is None or (
            not bool(self.config.dry_run) and not bool(getattr(servo_service, "is_connected", False))
        ):
            raise RuntimeError("Two-segment workspace repeatability requires a connected ServoService.")
        if not bool(self.config.allow_servo_only_test_run) and not bool(
            self._startup_provenance.get("accepted_all_8_startup")
        ):
            raise RuntimeError(
                "Two-segment workspace repeatability requires an accepted all-8 manual startup artifact. "
                "Run two_segment_startup_validation first, or set allow_servo_only_test_run=true for a lower-trust smoke."
            )
        # Pre-validate every target's tick budget — fail fast if any target's
        # compensated command would exceed safe per-servo bounds.
        violations = self._target_limit_violations(context=context, session=session)
        if violations:
            raise RuntimeError(
                "Two-segment workspace repeatability schedule exceeds configured tick limits: "
                + "; ".join(violations[:5])
                + (f"... +{len(violations) - 5} more" if len(violations) > 5 else "")
            )
        session.set_stage("precheck", "passed", "two-segment workspace repeatability precheck passed.")

    def execute(self, session: ExperimentSession) -> None:
        context = session.context.settings.robot.operating_context()
        role_configs = _effective_role_configs(session=session, config=self.config)
        foundation = build_two_segment_foundation_metadata(context)
        visit_sequence = build_round_robin_visit_order(
            target_count=len(self._targets),
            repeats_per_target=int(self.config.repeats_per_target),
            random_seed=int(self.config.random_seed),
        )
        total_planned_visits = len(visit_sequence)
        target_capture_budget = int(self.config.target_valid_capture_count) or total_planned_visits
        accepted_count = 0
        rejected_count = 0
        consecutive_rejected = 0

        if self.config.dry_run:
            # Pure scheduling/preview path: walk the sequence without writing
            # to the bus. Useful for GUI plan preview + automated tests.
            for visit_position, (cycle_index, visit_in_cycle, target_index) in enumerate(visit_sequence):
                if session.stop_requested is not None and session.stop_requested():
                    self._stop_reason = "operator_stop"
                    break
                target = self._targets[target_index]
                result = self._synth_dry_run_visit(
                    target=target,
                    cycle_index=cycle_index,
                    visit_in_cycle=visit_in_cycle,
                    visit_position=visit_position,
                )
                self._visit_results.append(result)
                accepted_count += int(result.accepted)
                session.add_sample(self._build_session_sample(visit_result=result, context=context))
                session.update_progress(
                    visit_position + 1,
                    total_planned_visits,
                    {
                        "cycle_index": cycle_index,
                        "visit_in_cycle": visit_in_cycle,
                        "target_index": target_index,
                        "accepted": accepted_count,
                    },
                )
            self._record_metrics(
                session=session,
                context=context,
                planned_visits=total_planned_visits,
                accepted=accepted_count,
                rejected=rejected_count,
                role_configs=role_configs,
                foundation=foundation,
            )
            return

        # Live-hardware path.
        servo_service = session.context.servo_service
        with servo_service.exclusive_bus_operation(
            owner=self.name,
            reason="two-segment workspace repeatability sweep",
        ):
            self._exclusive_owner = True
            try:
                for visit_position, (cycle_index, visit_in_cycle, target_index) in enumerate(visit_sequence):
                    if session.stop_requested is not None and session.stop_requested():
                        self._stop_reason = "operator_stop"
                        break
                    target = self._targets[target_index]
                    result = self._execute_single_visit(
                        session=session,
                        context=context,
                        target=target,
                        cycle_index=cycle_index,
                        visit_in_cycle=visit_in_cycle,
                        visit_position=visit_position,
                    )
                    self._visit_results.append(result)
                    if result.accepted:
                        accepted_count += 1
                        consecutive_rejected = 0
                    else:
                        rejected_count += 1
                        consecutive_rejected += 1
                        self._failure_events.append(
                            {
                                "visit_position": visit_position,
                                "target_index": target_index,
                                "cycle_index": cycle_index,
                                "visit_in_cycle": visit_in_cycle,
                                "reject_reason": result.reject_reason,
                                "command_success": result.command_success,
                                "timestamp_utc": result.timestamp_utc,
                            }
                        )
                    session.add_sample(self._build_session_sample(visit_result=result, context=context))
                    session.update_progress(
                        visit_position + 1,
                        total_planned_visits,
                        {
                            "cycle_index": cycle_index,
                            "visit_in_cycle": visit_in_cycle,
                            "target_index": target_index,
                            "accepted": accepted_count,
                            "rejected": rejected_count,
                        },
                    )
                    if self._should_stop_on_failures(
                        rejected=rejected_count,
                        consecutive_rejected=consecutive_rejected,
                        planned=total_planned_visits,
                    ):
                        break
                    if (
                        bool(self.config.continue_until_valid_captures)
                        and accepted_count >= int(target_capture_budget)
                    ):
                        self._stop_reason = "target_valid_capture_count_reached"
                        break
            finally:
                self._exclusive_owner = False

        if self._stop_reason is None and accepted_count >= len(visit_sequence):
            self._stop_reason = "scheduled_visits_completed"
        elif self._stop_reason is None:
            self._stop_reason = "execution_complete"

        self._record_metrics(
            session=session,
            context=context,
            planned_visits=total_planned_visits,
            accepted=accepted_count,
            rejected=rejected_count,
            role_configs=role_configs,
            foundation=foundation,
        )

    def finalize(self, session: ExperimentSession) -> None:
        if self.config.dry_run or not self._startup_ticks_by_servo:
            return
        try:
            writer = getattr(session.context.servo_service, "_write_goal_positions", None)
            if callable(writer) and session.context.servo_service.is_connected:
                writer({int(sid): int(tick) for sid, tick in self._startup_ticks_by_servo.items()})
        except Exception as exc:
            session.add_warning(f"Failed to return all 8 servos to startup ticks during finalize: {exc}")

    def write_outputs(self, session: ExperimentSession, paths, summary) -> None:
        from continuum_robot.experiments.two_segment_workspace_repeatability_outputs import (
            write_two_segment_workspace_repeatability_outputs,
        )

        write_two_segment_workspace_repeatability_outputs(
            output_dir=paths.output_dir,
            targets=self._targets,
            visit_results=self._visit_results,
            failure_events=self._failure_events,
            metrics=dict(summary.experiment_metrics or {}),
        )

    # ----- Internals -----

    def _target_limit_violations(self, *, context, session: ExperimentSession) -> list[str]:
        violations: list[str] = []
        commanded_ids = [int(v) for v in context.commanded_servo_ids]
        startup_ticks = [int(self._startup_ticks_by_servo[sid]) for sid in commanded_ids]
        mapper = session.context.servo_service.mapper
        tick_budget = int(self.config.max_tick_delta_from_startup) or tick_budget_for_segment_amplitude_cm(
            float(self.config.max_segment_displacement_cm),
            spool_diameter_cm=float(session.context.settings.robot.spool_diameter_cm),
        )
        for target in self._targets:
            command = _build_two_segment_command(target=target, context=context)
            intended_flat = command.to_flat(context=context)
            if bool(self.config.top_segment_tendon_routing_compensation):
                servo_flat, _info = _apply_top_routing_compensation(intended_flat, context=context)
            else:
                servo_flat = list(intended_flat)
            goals = mapper.to_goal_positions(servo_flat, startup_ticks)
            for servo_id, startup, goal in zip(commanded_ids, startup_ticks, goals):
                delta = abs(int(goal) - int(startup))
                if delta > int(tick_budget):
                    violations.append(
                        f"{target.target_id}: servo {servo_id} delta {delta} > {int(tick_budget)} ticks"
                    )
        return violations

    def _execute_single_visit(
        self,
        *,
        session: ExperimentSession,
        context,
        target: TwoSegmentWorkspaceTarget,
        cycle_index: int,
        visit_in_cycle: int,
        visit_position: int,
    ) -> _VisitResult:
        monotonic_time_s = session.elapsed_s()
        timestamp_utc = datetime.now(timezone.utc).isoformat()
        commanded_ids = [int(v) for v in context.commanded_servo_ids]
        startup_ticks = [int(self._startup_ticks_by_servo[sid]) for sid in commanded_ids]

        # 1. Move to neutral (startup-relative zero command).
        if self.config.return_to_neutral_between_visits:
            self._write_neutral(session=session, commanded_ids=commanded_ids, startup_ticks=startup_ticks)
            session.context.sleep_fn(float(self.config.neutral_settle_s))

        # 2. Compute + write the target command (with optional top-routing compensation).
        command = _build_two_segment_command(target=target, context=context)
        intended_flat = command.to_flat(context=context)
        if bool(self.config.top_segment_tendon_routing_compensation):
            servo_flat, _comp_info = _apply_top_routing_compensation(intended_flat, context=context)
        else:
            servo_flat = list(intended_flat)
        goal_ticks = session.context.servo_service.mapper.to_goal_positions(servo_flat, startup_ticks)
        goals_by_servo = {int(sid): int(tick) for sid, tick in zip(commanded_ids, goal_ticks)}
        command_success = True
        reject_reason: str | None = None
        try:
            writer = getattr(session.context.servo_service, "_write_goal_positions", None)
            if not callable(writer):
                raise RuntimeError("ServoService all-8 raw goal writer is unavailable.")
            writer(goals_by_servo)
        except Exception as exc:
            command_success = False
            reject_reason = f"command_write_failed:{exc.__class__.__name__}:{exc}"

        # 3. Target settle.
        if command_success:
            session.context.sleep_fn(float(self.config.target_settle_s))
            if float(self.config.capture_dwell_s) > 0.0:
                session.context.sleep_fn(float(self.config.capture_dwell_s))

        # 4. Capture telemetry + pose.
        telemetry = _read_live_telemetry(session=session, servo_ids=commanded_ids, dry_run=False)
        servo_feedback = _servo_feedback_payload(
            session=session,
            telemetry=telemetry,
            servo_ids=commanded_ids,
            startup_ticks_by_servo=self._startup_ticks_by_servo,
        )
        pose_observation, pose_fields = _two_segment_pose_observation(
            session=session,
            config=self.config,
            run_trust_mode=str(self.config.run_trust_mode),
        )
        distal_xyz = _extract_robot_frame_xyz(pose_fields, role="distal_tip")
        intermediate_xyz = _extract_robot_frame_xyz(pose_fields, role="intermediate_segment")
        distal_orientation = _extract_robot_frame_quaternion(pose_observation, role="distal_tip")
        tracker_age_s = pose_fields.get("freshness_s") if isinstance(pose_fields, dict) else None
        missing_servo_ids = [
            int(sid) for sid in commanded_ids
            if dict(servo_feedback.get(str(int(sid)), {}) or {}).get("position_tick") is None
        ]
        accepted = command_success and reject_reason is None
        if accepted and not missing_servo_ids:
            if distal_xyz is None and bool(self.config.require_distal_tool_visible):
                accepted = False
                reject_reason = "distal_xyz_missing"
        elif missing_servo_ids:
            accepted = False
            reject_reason = f"missing_measured_servo_positions:{missing_servo_ids}"

        # 5. Return to neutral (optional; matches single-segment).
        if self.config.return_to_neutral_between_visits:
            self._write_neutral(session=session, commanded_ids=commanded_ids, startup_ticks=startup_ticks)

        return _VisitResult(
            accepted=bool(accepted),
            target_index=int(target.target_index),
            target_id=str(target.target_id),
            cycle_index=int(cycle_index),
            visit_in_cycle=int(visit_in_cycle),
            visit_position=int(visit_position),
            timestamp_utc=timestamp_utc,
            monotonic_time_s=float(monotonic_time_s),
            bottom_x_cm=float(target.bottom_x_cm),
            bottom_y_cm=float(target.bottom_y_cm),
            top_x_cm=float(target.top_x_cm),
            top_y_cm=float(target.top_y_cm),
            ordered_8_displacements_cm=list(intended_flat),
            servo_flat_command_cm=list(servo_flat),
            all_8_goal_ticks={str(k): int(v) for k, v in goals_by_servo.items()},
            all_8_present_position_ticks={
                str(int(sid)): dict(servo_feedback.get(str(int(sid)), {}) or {}).get("position_tick")
                for sid in commanded_ids
            },
            all_8_current_load_proxy_ma={
                str(int(sid)): dict(servo_feedback.get(str(int(sid)), {}) or {}).get("load_proxy_ma")
                for sid in commanded_ids
            },
            distal_xyz_robot_mm=distal_xyz,
            intermediate_xyz_robot_mm=intermediate_xyz,
            distal_orientation_wxyz=distal_orientation,
            command_success=bool(command_success),
            reject_reason=reject_reason,
            tracker_age_s=float(tracker_age_s) if isinstance(tracker_age_s, (int, float)) else None,
            servo_telemetry_age_s=None,
            group_tag=str(target.group_tag),
            amplitude_cm=float(target.amplitude_cm),
        )

    def _synth_dry_run_visit(
        self,
        *,
        target: TwoSegmentWorkspaceTarget,
        cycle_index: int,
        visit_in_cycle: int,
        visit_position: int,
    ) -> _VisitResult:
        """Dry-run synth visit: never touches the bus; produces a deterministic accepted capture.

        Used by the GUI plan-preview and by tests so the framework can write
        the standard run folder even before hardware is connected.
        """
        # Synthesize a tight cluster of distal positions per target so the
        # metrics path has something realistic to compute on. The seed is
        # derived from target index + cycle so repeats vary slightly.
        rng = np.random.default_rng(
            int(self.config.random_seed) * 1_000_003 + int(target.target_index) * 977 + int(cycle_index)
        )
        commanded_ids = [1, 2, 3, 4, 5, 6, 7, 8]
        ordered_8 = [
            target.bottom_tendon_cm[0],
            target.bottom_tendon_cm[1],
            target.bottom_tendon_cm[2],
            target.bottom_tendon_cm[3],
            target.top_tendon_cm[0],
            target.top_tendon_cm[1],
            target.top_tendon_cm[2],
            target.top_tendon_cm[3],
        ]
        bias = rng.normal(scale=0.05, size=3)
        noise = rng.normal(scale=0.10, size=3)
        commanded_xyz_mm = np.array([target.bottom_x_cm * 10.0, target.bottom_y_cm * 10.0, 100.0])
        distal_xyz = (commanded_xyz_mm + bias + noise).tolist()
        return _VisitResult(
            accepted=True,
            target_index=int(target.target_index),
            target_id=str(target.target_id),
            cycle_index=int(cycle_index),
            visit_in_cycle=int(visit_in_cycle),
            visit_position=int(visit_position),
            timestamp_utc=datetime.now(timezone.utc).isoformat(),
            monotonic_time_s=float(visit_position) * 0.001,
            bottom_x_cm=float(target.bottom_x_cm),
            bottom_y_cm=float(target.bottom_y_cm),
            top_x_cm=float(target.top_x_cm),
            top_y_cm=float(target.top_y_cm),
            ordered_8_displacements_cm=list(ordered_8),
            servo_flat_command_cm=list(ordered_8),
            all_8_goal_ticks={str(sid): 2048 for sid in commanded_ids},
            all_8_present_position_ticks={str(sid): 2048 for sid in commanded_ids},
            all_8_current_load_proxy_ma={str(sid): 100.0 for sid in commanded_ids},
            distal_xyz_robot_mm=list(distal_xyz),
            intermediate_xyz_robot_mm=None,
            distal_orientation_wxyz=None,
            command_success=True,
            reject_reason=None,
            tracker_age_s=0.01,
            servo_telemetry_age_s=0.01,
            group_tag=str(target.group_tag),
            amplitude_cm=float(target.amplitude_cm),
        )

    def _write_neutral(
        self,
        *,
        session: ExperimentSession,
        commanded_ids: list[int],
        startup_ticks: list[int],
    ) -> None:
        try:
            writer = getattr(session.context.servo_service, "_write_goal_positions", None)
            if callable(writer):
                writer({int(sid): int(tick) for sid, tick in zip(commanded_ids, startup_ticks)})
        except Exception:
            # Don't crash; surface the issue when the next sample fails too.
            pass

    def _should_stop_on_failures(
        self,
        *,
        rejected: int,
        consecutive_rejected: int,
        planned: int,
    ) -> bool:
        if int(self.config.max_consecutive_failed_visits) and consecutive_rejected >= int(
            self.config.max_consecutive_failed_visits
        ):
            self._stop_reason = f"max_consecutive_failed_visits_exceeded:{consecutive_rejected}"
            return True
        if int(self.config.max_failed_visits) and rejected >= int(self.config.max_failed_visits):
            self._stop_reason = f"max_failed_visits_exceeded:{rejected}"
            return True
        if planned > 0 and float(self.config.max_rejected_fraction) > 0:
            if rejected / float(planned) > float(self.config.max_rejected_fraction):
                self._stop_reason = (
                    f"max_rejected_fraction_exceeded:{rejected}/{planned}"
                )
                return True
        return False

    def _build_session_sample(self, *, visit_result: _VisitResult, context) -> ExperimentTimeseriesSample:
        flags = ["two_segment_workspace_repeatability_visit"]
        if visit_result.accepted:
            flags.append("capture_accepted")
        else:
            flags.append("capture_rejected")
        return ExperimentTimeseriesSample(
            monotonic_time_s=float(visit_result.monotonic_time_s),
            wall_time_utc=visit_result.timestamp_utc,
            phase="workspace_target",
            step_index=int(visit_result.target_index),
            sample_index=int(visit_result.visit_position),
            cycle_index=int(visit_result.cycle_index),
            commanded_motor_values=dict(visit_result.all_8_goal_ticks),
            commanded_cable_deltas_cm=list(visit_result.ordered_8_displacements_cm),
            tracker_frame_id=None,
            tool_ids_seen=[str(self.config.expected_distal_tool_id)],
            transform_validity={"distal_tip": "available" if visit_result.distal_xyz_robot_mm else "missing"},
            pose_in_tracker_frame={},
            pose_in_robot_frame={
                "roles": {
                    "distal_tip": {
                        "translation_mm": list(visit_result.distal_xyz_robot_mm)
                        if visit_result.distal_xyz_robot_mm is not None
                        else [],
                        "tool_id": str(self.config.expected_distal_tool_id),
                    },
                    **(
                        {
                            "intermediate_segment": {
                                "translation_mm": list(visit_result.intermediate_xyz_robot_mm),
                            }
                        }
                        if visit_result.intermediate_xyz_robot_mm is not None
                        else {}
                    ),
                }
            },
            status_flags=sorted(set(flags)),
            backend_health={
                "tracker_age_s": visit_result.tracker_age_s,
                "servo_telemetry_age_s": visit_result.servo_telemetry_age_s,
            },
            extra={
                "record_kind": "two_segment_workspace_repeatability_visit",
                "target_id": visit_result.target_id,
                "target_index": visit_result.target_index,
                "cycle_index": visit_result.cycle_index,
                "visit_in_cycle": visit_result.visit_in_cycle,
                "visit_position": visit_result.visit_position,
                "bottom_x_cm": visit_result.bottom_x_cm,
                "bottom_y_cm": visit_result.bottom_y_cm,
                "top_x_cm": visit_result.top_x_cm,
                "top_y_cm": visit_result.top_y_cm,
                "ordered_8_displacements_cm": list(visit_result.ordered_8_displacements_cm),
                "servo_flat_command_cm": list(visit_result.servo_flat_command_cm),
                "all_8_goal_ticks": dict(visit_result.all_8_goal_ticks),
                "all_8_present_position_ticks": dict(visit_result.all_8_present_position_ticks),
                "all_8_current_load_proxy_ma": dict(visit_result.all_8_current_load_proxy_ma),
                "capture_accepted": visit_result.accepted,
                "reject_reason": visit_result.reject_reason,
                "command_success": visit_result.command_success,
                "group_tag": visit_result.group_tag,
                "amplitude_cm": visit_result.amplitude_cm,
            },
        )

    def _record_metrics(
        self,
        *,
        session: ExperimentSession,
        context,
        planned_visits: int,
        accepted: int,
        rejected: int,
        role_configs: dict[str, Any],
        foundation: dict[str, Any],
    ) -> None:
        from continuum_robot.experiments.two_segment_workspace_repeatability_outputs import (
            compute_workspace_repeatability_metrics,
            summarize_workspace_repeatability,
        )

        per_target_rows = compute_workspace_repeatability_metrics(
            visit_results=self._visit_results,
            targets=self._targets,
        )
        summary = summarize_workspace_repeatability(
            per_target_rows,
            minimum_repeats_per_target=int(self.config.minimum_repeats_per_target),
            target_distal_rms_threshold_mm=self.config.target_distal_rms_threshold_mm,
        )
        valid_for_thesis = (
            accepted >= int(planned_visits) * float(1.0 - float(self.config.max_rejected_fraction))
            and summary.get("targets_below_minimum_repeats", 0) == 0
            and summary.get("planned_target_count", 0) >= int(self.config.target_count)
            and bool(self._startup_provenance.get("accepted_all_8_startup"))
        )
        session.set_metric("schema_version", REPEATABILITY_SCHEMA_VERSION)
        session.set_metric("experiment_name", EXPERIMENT_NAME)
        session.set_metric("demo_only", False)
        session.set_metric("valid_for_repeatability_analysis", True)
        session.set_metric("valid_for_thesis_repeatability", bool(valid_for_thesis))
        session.set_metric("valid_for_model_training", False)
        session.set_metric("primary_metric", "distal_xyz_repeatability")
        session.set_metric("controlled_point", "distal_tip coil origin in robot base frame")
        session.set_metric("expected_distal_tool_id", str(self.config.expected_distal_tool_id))
        session.set_metric("physical_tip_chasing", False)
        session.set_metric("target_count", len(self._targets))
        session.set_metric("repeats_per_target", int(self.config.repeats_per_target))
        session.set_metric("planned_visits", int(planned_visits))
        session.set_metric("accepted_captures", int(accepted))
        session.set_metric("rejected_captures", int(rejected))
        session.set_metric("stop_reason", self._stop_reason or "unspecified")
        session.set_metric("target_generator_mode", str(self.config.target_generator_mode))
        session.set_metric("random_seed", int(self.config.random_seed))
        session.set_metric("max_segment_displacement_cm", float(self.config.max_segment_displacement_cm))
        session.set_metric("neutral_settle_s", float(self.config.neutral_settle_s))
        session.set_metric("target_settle_s", float(self.config.target_settle_s))
        session.set_metric("return_to_neutral_between_visits", bool(self.config.return_to_neutral_between_visits))
        session.set_metric("bottom_segment_key", context.bottom_segment_key)
        session.set_metric("top_segment_key", context.top_segment_key)
        session.set_metric("bottom_servo_ids", list(context.bottom_servo_ids or []))
        session.set_metric("top_servo_ids", list(context.top_servo_ids or []))
        session.set_metric("physical_assembly", dict(context.metadata().get("physical_assembly") or {}))
        session.set_metric("two_segment_foundation", foundation)
        session.set_metric("two_segment_tracking_role_config", dict(role_configs))
        session.set_metric("startup_artifact_provenance", dict(self._startup_provenance))
        session.set_metric("repeatability_summary", summary)
        session.set_metric("per_target_repeatability_count", len(per_target_rows))
        session.set_metric("failure_event_count", len(self._failure_events))


# ---------------------------------------------------------------------------
# Helpers used by both runtime and outputs
# ---------------------------------------------------------------------------


def _extract_robot_frame_xyz(pose_fields: dict[str, Any], *, role: str) -> list[float] | None:
    if not isinstance(pose_fields, dict):
        return None
    robot_frame = dict(pose_fields.get("pose_in_robot_frame") or {})
    roles = dict(robot_frame.get("roles") or {})
    record = dict(roles.get(role) or {})
    translation = record.get("translation_mm")
    if isinstance(translation, list) and len(translation) >= 3:
        return [float(translation[0]), float(translation[1]), float(translation[2])]
    matrix = record.get("T_robot_tool") or record.get("T_robot_tip")
    if isinstance(matrix, list) and len(matrix) >= 3:
        try:
            return [float(matrix[0][3]), float(matrix[1][3]), float(matrix[2][3])]
        except Exception:
            return None
    return None


def _extract_robot_frame_quaternion(pose_observation: dict[str, Any], *, role: str) -> list[float] | None:
    if not isinstance(pose_observation, dict):
        return None
    if role == "distal_tip":
        record = dict(pose_observation.get("distal_tip_pose") or {})
    else:
        record = dict(dict(pose_observation.get("role_observations") or {}).get(role) or {})
    quat = record.get("quaternion_wxyz")
    if isinstance(quat, list) and len(quat) == 4:
        return [float(v) for v in quat]
    return None

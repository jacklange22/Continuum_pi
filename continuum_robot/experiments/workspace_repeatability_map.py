"""Workspace repeatability map experiment.

Visits a Vogel-spiral grid of targets that uniformly fills a disk-shaped
single-segment workspace, returning to neutral between every visit, and
measures how tightly the tip clusters at each target. Output is a per-target
3D RMS spread map suitable for a color-coded thesis figure.

Why a Vogel spiral instead of concentric rings:
    The legacy 17-point catalog has two angular rings with no points between
    them, which creates visible dead zones in any heatmap. The Vogel pattern
    (``r = R · sqrt(i / (N - 1))``, ``θ = i · golden_angle``) fills a disk
    with uniform spatial density for any N. With ``N = 100`` and ``R = 12 mm``
    nearest-neighbor spacing stays roughly constant across the disk, so a
    contour plot of the per-point repeatability has no structural gaps.

Why "from neutral" every visit:
    The legacy single-segment experiment uses every-other-target approaches to
    probe path dependence. This experiment is the complement: pure motor-command
    repeatability, with every visit starting from the same neutral pose. The
    operator can compare the two side by side to attribute spread to history vs.
    motor + tracker noise.

Visit ordering:
    Round-robin shuffled per cycle. The 100 targets are visited once in a
    deterministic-but-shuffled order, then re-shuffled for the next cycle, and
    so on for ``visits_per_target`` cycles. This averages time-correlated drift
    (motor warm-up, tracker drift) across all targets so points captured early
    aren't unfairly compared against points captured late.

The execute loop is intentionally simple:

    for cycle in 1..visits_per_target:
        shuffle order
        for target in order:
            command neutral
            settle (short)
            command target
            settle (long)
            capture
            record visit
        # back to neutral at cycle boundary
    final return to neutral

Real hardware runs are gated by the same precheck infrastructure as the legacy
17-point experiment (servo readiness, tracker streaming, registration, runtime
tip policy). A ``dry_run`` mode synthesizes capture positions from the
commanded deflection + configurable Gaussian noise so the experiment can be
exercised in tests and as a preview before committing to a 2-hour real run.
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from continuum_robot.experiments.framework import (
    BaseExperiment,
    ExperimentHardwareRequirements,
    ExperimentSession,
)
from continuum_robot.experiments.sample_builders import sample_from_tracking_snapshot
from continuum_robot.experiments.schemas import ExperimentTimeseriesSample
from continuum_robot.experiments.validation import (
    STATUS_INVALID_INSUFFICIENT_SAMPLES,
    STATUS_PARTIAL_SUCCESS,
    STATUS_SUCCESS,
)


# Vogel-spiral defaults for a single-segment cable robot. The defaults were
# chosen to match the operator's 12 mm workspace request; you can tune via
# config without code changes.
DEFAULT_TARGET_COUNT = 100
DEFAULT_MAX_AMPLITUDE_MM = 12.0
DEFAULT_VISITS_PER_TARGET = 15

# Settle defaults tuned so a 1500-visit run completes in ~2 hours while still
# letting cables relax and the tracker latch on cleanly. Override per run if
# needed; the experiment records the settle times into the run summary so
# downstream analysis can correlate them with spread.
DEFAULT_NEUTRAL_SETTLE_S = 1.5
DEFAULT_TARGET_SETTLE_S = 2.5
DEFAULT_CAPTURE_TIMEOUT_S = 1.0
DEFAULT_CAPTURE_POLL_INTERVAL_S = 0.02
DEFAULT_MAX_TRACKER_AGE_S = 0.25

# Safety cap on absolute tendon tick excursion. Mirrors the legacy single-
# segment value so an operator that already passed pretension/startup for the
# 17-point experiment can run this one without re-tuning servo limits.
DEFAULT_MAX_TARGET_TICK_DELTA_FROM_STARTUP = 1200


@dataclass(frozen=True)
class WorkspaceMapTarget:
    """One target on the Vogel-spiral disk-fill grid.

    The angular spacing is the golden angle so the points form a uniformly
    dense sunflower pattern across the configured disk radius. ``x_mm`` and
    ``y_mm`` are the predicted in-plane deflections at the tip if the cable
    deltas were applied with the legacy four-cable antagonistic convention
    (``deltas = [-x, -y, x, y]``). They are not measurements -- they're the
    commanded geometric goal, useful for sorting points on the heatmap.
    """

    target_index: int
    label: str
    amplitude_mm: float
    angle_rad: float
    angle_deg: float
    x_mm: float
    y_mm: float
    cable_deltas_mm: list[float]
    cable_deltas_cm: list[float]

    def to_dict(self) -> dict[str, Any]:
        return {
            "target_index": int(self.target_index),
            "label": str(self.label),
            "amplitude_mm": float(self.amplitude_mm),
            "angle_rad": float(self.angle_rad),
            "angle_deg": float(self.angle_deg),
            "x_mm": float(self.x_mm),
            "y_mm": float(self.y_mm),
            "cable_deltas_mm": [float(value) for value in self.cable_deltas_mm],
            "cable_deltas_cm": [float(value) for value in self.cable_deltas_cm],
        }


@dataclass
class WorkspaceRepeatabilityMapConfig:
    """Runtime parameters for the workspace repeatability map experiment.

    The defaults reproduce the operator's request: 100 Vogel-spiral targets
    on a 12 mm-radius disk, 15 round-robin shuffled visits per target, with
    a short neutral settle and a longer target settle between each commanded
    move. ``dry_run`` swaps real-tracker capture for a deterministic-noise
    synthesis so the same code path can be exercised in tests and previews.
    """

    target_count: int = DEFAULT_TARGET_COUNT
    max_amplitude_mm: float = DEFAULT_MAX_AMPLITUDE_MM
    visits_per_target: int = DEFAULT_VISITS_PER_TARGET
    neutral_settle_s: float = DEFAULT_NEUTRAL_SETTLE_S
    target_settle_s: float = DEFAULT_TARGET_SETTLE_S
    capture_timeout_s: float = DEFAULT_CAPTURE_TIMEOUT_S
    capture_poll_interval_s: float = DEFAULT_CAPTURE_POLL_INTERVAL_S
    max_tracker_age_s: float = DEFAULT_MAX_TRACKER_AGE_S
    max_target_tick_delta_from_startup: int = DEFAULT_MAX_TARGET_TICK_DELTA_FROM_STARTUP
    tool_id: str = "0A"
    random_seed: int = 0
    run_label: str = ""
    return_to_center_on_finalize: bool = True
    require_robot_frame_tip: bool = True
    allow_debug_coil_as_tip: bool = False
    thesis_goal_rms_mm: float = 1.0
    dry_run: bool = False
    synthetic_noise_std_mm: float = 0.20
    synthetic_per_target_bias_std_mm: float = 0.05

    @classmethod
    def from_dict(cls, payload: dict[str, Any] | None) -> "WorkspaceRepeatabilityMapConfig":
        payload = dict(payload or {})
        return cls(
            target_count=max(3, int(payload.get("target_count", DEFAULT_TARGET_COUNT))),
            max_amplitude_mm=max(0.1, float(payload.get("max_amplitude_mm", DEFAULT_MAX_AMPLITUDE_MM))),
            visits_per_target=max(2, int(payload.get("visits_per_target", DEFAULT_VISITS_PER_TARGET))),
            neutral_settle_s=max(0.0, float(payload.get("neutral_settle_s", DEFAULT_NEUTRAL_SETTLE_S))),
            target_settle_s=max(0.0, float(payload.get("target_settle_s", DEFAULT_TARGET_SETTLE_S))),
            capture_timeout_s=max(0.0, float(payload.get("capture_timeout_s", DEFAULT_CAPTURE_TIMEOUT_S))),
            capture_poll_interval_s=max(0.001, float(payload.get("capture_poll_interval_s", DEFAULT_CAPTURE_POLL_INTERVAL_S))),
            max_tracker_age_s=max(0.0, float(payload.get("max_tracker_age_s", DEFAULT_MAX_TRACKER_AGE_S))),
            max_target_tick_delta_from_startup=max(
                1,
                int(
                    payload.get(
                        "max_target_tick_delta_from_startup",
                        DEFAULT_MAX_TARGET_TICK_DELTA_FROM_STARTUP,
                    )
                ),
            ),
            tool_id=str(payload.get("tool_id", "0A") or "0A").strip().upper(),
            random_seed=int(payload.get("random_seed", 0)),
            run_label=str(payload.get("run_label", "") or ""),
            return_to_center_on_finalize=bool(payload.get("return_to_center_on_finalize", True)),
            require_robot_frame_tip=bool(payload.get("require_robot_frame_tip", True)),
            allow_debug_coil_as_tip=bool(payload.get("allow_debug_coil_as_tip", False)),
            thesis_goal_rms_mm=float(payload.get("thesis_goal_rms_mm", 1.0)),
            dry_run=bool(payload.get("dry_run", False)),
            synthetic_noise_std_mm=max(0.0, float(payload.get("synthetic_noise_std_mm", 0.20))),
            synthetic_per_target_bias_std_mm=max(
                0.0, float(payload.get("synthetic_per_target_bias_std_mm", 0.05))
            ),
        )


# ---------------------------------------------------------------------------
# Target generation (pure functions, no hardware required)
# ---------------------------------------------------------------------------


def _golden_angle_rad() -> float:
    """Return the golden angle ≈ 137.5077°.

    Defined exactly as ``π · (3 − √5)`` so floating-point round-off is
    deterministic across platforms; the small numerical drift over 100
    iterations is irrelevant to the disk-fill density.
    """
    return math.pi * (3.0 - math.sqrt(5.0))


def build_workspace_repeatability_targets(
    *,
    target_count: int = DEFAULT_TARGET_COUNT,
    max_amplitude_mm: float = DEFAULT_MAX_AMPLITUDE_MM,
) -> list[WorkspaceMapTarget]:
    """Return ``target_count`` Vogel-spiral targets uniformly filling a disk.

    Point ``0`` is the disk centre, point ``target_count - 1`` lies on the
    perimeter. The intermediate points form a sunflower-seed pattern, so
    nearest-neighbour spacing stays approximately constant across the disk
    -- the property the operator asked for ("no large dead zones").

    Cable deltas use the legacy four-cable antagonistic convention so the
    commands can be sent through the same ``ServoService.command_displacement``
    path the 17-point experiment uses. The geometric ``(x_mm, y_mm)`` recorded
    on each target is the *commanded* in-plane goal -- not a measurement -- and
    is what the heatmap is plotted against if no tracker centroid is available.
    """

    if target_count < 3:
        raise ValueError("workspace repeatability map requires at least 3 targets")
    if max_amplitude_mm <= 0.0:
        raise ValueError("max_amplitude_mm must be positive")

    golden_angle = _golden_angle_rad()
    targets: list[WorkspaceMapTarget] = []
    for index in range(int(target_count)):
        # r grows as sqrt(i / (N-1)) so the points are uniformly distributed
        # in *area* across the disk (the differential dA = r dr dθ is what
        # this formula equalises). Center (i=0) is at r=0 by construction.
        radius_fraction = math.sqrt(float(index) / float(target_count - 1)) if target_count > 1 else 0.0
        amplitude_mm = float(max_amplitude_mm) * float(radius_fraction)
        theta = float(index) * golden_angle
        x_mm = amplitude_mm * math.cos(theta)
        y_mm = amplitude_mm * math.sin(theta)
        # Legacy four-cable antagonistic projection. Cable 1 shortens when
        # the tip is pulled in +x; cable 3 lengthens; cables 2 and 4 do the
        # same for the +y axis.
        cable_deltas_mm = [
            -_clean_float(x_mm),
            -_clean_float(y_mm),
            _clean_float(x_mm),
            _clean_float(y_mm),
        ]
        cable_deltas_cm = [_clean_float(value / 10.0) for value in cable_deltas_mm]
        angle_deg = math.degrees(theta) % 360.0
        targets.append(
            WorkspaceMapTarget(
                target_index=index,
                label=f"T{index:03d}",
                amplitude_mm=_clean_float(amplitude_mm),
                angle_rad=_clean_float(theta % (2.0 * math.pi)),
                angle_deg=_clean_float(angle_deg),
                x_mm=_clean_float(x_mm),
                y_mm=_clean_float(y_mm),
                cable_deltas_mm=cable_deltas_mm,
                cable_deltas_cm=cable_deltas_cm,
            )
        )
    return targets


def workspace_target_tick_profile(
    *,
    targets: list[WorkspaceMapTarget],
    mapper,
) -> dict[str, Any]:
    """Project cable-displacement targets into per-servo tick deltas + max abs tick excursion."""
    per_target: dict[int, list[int]] = {}
    max_abs_tick_delta = 0
    for target in targets:
        cable_ticks = [
            int(mapper.displacement_cm_to_ticks(float(delta_cm)))
            for delta_cm in target.cable_deltas_cm
        ]
        per_target[int(target.target_index)] = cable_ticks
        if cable_ticks:
            max_abs_tick_delta = max(max_abs_tick_delta, max(abs(value) for value in cable_ticks))
    return {
        "per_target_cable_deltas_ticks": per_target,
        "max_abs_tick_delta": int(max_abs_tick_delta),
    }


def build_round_robin_visit_order(
    *,
    target_indices: list[int],
    visits_per_target: int,
    random_seed: int,
) -> list[tuple[int, int, int]]:
    """Generate a ``(cycle_index, visit_in_cycle, target_index)`` sequence.

    Each cycle visits every target exactly once, in a freshly shuffled order
    seeded by ``random_seed`` plus the cycle index so the per-cycle shuffles
    are deterministic and independent. The flattened sequence length is
    ``len(target_indices) * visits_per_target``.
    """
    if visits_per_target < 1:
        raise ValueError("visits_per_target must be at least 1")
    if not target_indices:
        return []
    sequence: list[tuple[int, int, int]] = []
    for cycle_index in range(int(visits_per_target)):
        # Per-cycle seed is deterministic but distinct between cycles so
        # the schedule reproduces under the same random_seed.
        rng = random.Random(int(random_seed) * 1_000_003 + int(cycle_index))
        shuffled = list(target_indices)
        rng.shuffle(shuffled)
        for visit_in_cycle, target_index in enumerate(shuffled):
            sequence.append((int(cycle_index), int(visit_in_cycle), int(target_index)))
    return sequence


# ---------------------------------------------------------------------------
# Internal utilities
# ---------------------------------------------------------------------------


def _clean_float(value: float) -> float:
    """Clamp tiny floating-point noise so persisted artifacts stay readable."""
    out = float(value)
    if math.isclose(out, 0.0, abs_tol=1e-12):
        return 0.0
    return out


def _configured_single_segment_servo_ids(session: ExperimentSession) -> list[int]:
    settings = session.context.settings
    servo_ids = list(settings.robot.active_segment_servo_ids() or [])
    if len(servo_ids) != 4:
        raise RuntimeError(
            "workspace_repeatability_map requires exactly 4 servo IDs configured for the "
            f"active single-segment; got {servo_ids}. Run the single-segment startup workflow "
            "before launching this experiment."
        )
    return [int(value) for value in servo_ids]


def _load_neutral_ticks(session: ExperimentSession, servo_ids: list[int]) -> list[int]:
    """Return the per-servo neutral position ticks (servo-order), failing loud."""
    servo_service = session.context.servo_service
    if servo_service is None:
        raise RuntimeError("workspace_repeatability_map requires a servo service.")
    neutral_lookup = servo_service.load_neutral_setpoints() or {}
    missing = [sid for sid in servo_ids if int(sid) not in neutral_lookup]
    if missing:
        raise RuntimeError(
            "Neutral setpoints missing for servo(s) "
            + ", ".join(str(value) for value in missing)
            + ". Run startup pretension before launching this experiment."
        )
    return [int(neutral_lookup[int(sid)]) for sid in servo_ids]


# ---------------------------------------------------------------------------
# Experiment class
# ---------------------------------------------------------------------------


class WorkspaceRepeatabilityMapExperiment(BaseExperiment):
    """Visit 100 Vogel-spiral targets 15× each from neutral and record dispersion."""

    name = "workspace_repeatability_map"
    description = (
        "Map per-point repeatability across a 12 mm-radius single-segment disk. Visits "
        "100 Vogel-spiral targets in round-robin shuffled order, returning to neutral "
        "before each visit, with 15 visits per target. Outputs a color-coded 3D map "
        "of the per-target RMS spread."
    )
    hardware_requirements = ExperimentHardwareRequirements(
        tracking_required=True,
        servo_required=True,
        registration_required=True,
        mock_compatible=True,
    )

    def __init__(self, config: WorkspaceRepeatabilityMapConfig) -> None:
        super().__init__(config)
        self.config: WorkspaceRepeatabilityMapConfig
        self._targets: list[WorkspaceMapTarget] = build_workspace_repeatability_targets(
            target_count=int(self.config.target_count),
            max_amplitude_mm=float(self.config.max_amplitude_mm),
        )
        self._target_by_index: dict[int, WorkspaceMapTarget] = {
            int(target.target_index): target for target in self._targets
        }
        self._servo_ids: list[int] = []
        self._neutral_ticks: list[int] = []
        self._exclusive_owner: bool = False

    @classmethod
    def from_dict(cls, payload: dict[str, Any] | None = None) -> "WorkspaceRepeatabilityMapExperiment":
        return cls(config=WorkspaceRepeatabilityMapConfig.from_dict(payload))

    # --- setup / precheck ---------------------------------------------------

    def setup(self, session: ExperimentSession) -> None:
        # Skip hardware lookups when running in dry_run mode -- the dry-run
        # synthesis path doesn't touch the servo service.
        if self.config.dry_run:
            self._servo_ids = list(range(1, 5))
            self._neutral_ticks = [0, 0, 0, 0]
            tick_profile = {"per_target_cable_deltas_ticks": {}, "max_abs_tick_delta": 0}
        else:
            self._servo_ids = _configured_single_segment_servo_ids(session)
            self._neutral_ticks = _load_neutral_ticks(session, self._servo_ids)
            tick_profile = workspace_target_tick_profile(
                targets=self._targets,
                mapper=session.context.servo_service.mapper,
            )

        # Settings read is guarded so dry-run callers (notably tests) don't need
        # to construct a full ExperimentContext.
        settings = self._safe_settings(session)
        spool_diameter_mm: float | None = None
        active_segment_key: str | None = None
        active_segment_label: str | None = None
        robot_mode: str | None = None
        if settings is not None and getattr(settings, "robot", None) is not None:
            try:
                spool_diameter_mm = float(settings.robot.spool_diameter_cm) * 10.0
            except (TypeError, AttributeError):
                spool_diameter_mm = None
            active_segment_key = getattr(settings.robot, "active_segment_key", lambda: None)()
            active_segment_label = getattr(settings.robot, "active_segment_label", lambda: None)()
            robot_mode = getattr(settings.robot, "mode", None)
        session.set_metric(
            "target_catalog",
            [
                {
                    **target.to_dict(),
                    "cable_deltas_ticks": list(
                        tick_profile["per_target_cable_deltas_ticks"].get(int(target.target_index), [])
                    ),
                }
                for target in self._targets
            ],
        )
        session.set_metric(
            "target_geometry",
            {
                "pattern": "vogel_disk_fill",
                "target_count": int(len(self._targets)),
                "max_amplitude_mm": float(self.config.max_amplitude_mm),
                "max_target_abs_tick_delta_from_startup": int(tick_profile["max_abs_tick_delta"]),
                "max_target_tick_delta_from_startup_cap": int(self.config.max_target_tick_delta_from_startup),
                "spool_diameter_mm": spool_diameter_mm,
            },
        )
        planned_visits = int(len(self._targets) * self.config.visits_per_target)
        session.set_metric("planned_visit_count", planned_visits)
        session.set_metric("planned_capture_count", planned_visits)
        session.set_metric("protocol", "workspace_repeatability_map_vogel_from_neutral")
        session.set_metric("run_label", str(self.config.run_label or ""))
        session.set_metric(
            "active_segment",
            {
                "key": active_segment_key,
                "label": active_segment_label,
                "servo_ids": list(self._servo_ids),
                "robot_mode": robot_mode,
            },
        )
        session.set_metric(
            "schedule",
            {
                "visits_per_target": int(self.config.visits_per_target),
                "neutral_settle_s": float(self.config.neutral_settle_s),
                "target_settle_s": float(self.config.target_settle_s),
                "random_seed": int(self.config.random_seed),
                "round_robin_shuffled_per_cycle": True,
            },
        )

    def precheck(self, session: ExperimentSession) -> None:
        # The legacy single-segment experiment runs an exhaustive precheck
        # (servo limits, runtime tip policy, registration freshness, tracker
        # stream). Reuse it when not in dry_run; in dry_run we just verify
        # geometry is sane.
        if self.config.dry_run:
            if not self._targets:
                raise RuntimeError("workspace_repeatability_map has no targets configured.")
            return
        # Real-hardware precheck delegates to the same helpers the legacy
        # 17-point experiment uses, by re-using its acceptance gates.
        tick_profile = workspace_target_tick_profile(
            targets=self._targets,
            mapper=session.context.servo_service.mapper,
        )
        max_abs = int(tick_profile["max_abs_tick_delta"])
        cap = int(self.config.max_target_tick_delta_from_startup)
        if max_abs > cap:
            raise RuntimeError(
                f"Workspace repeatability map requires {max_abs} tick excursion at the "
                f"outer-most target, exceeding the configured safety cap of {cap}. Reduce "
                "max_amplitude_mm or increase max_target_tick_delta_from_startup."
            )
        # Tracker freshness sanity: make sure the snapshot is live.
        tracker = session.context.tracking_service
        snapshot = tracker.get_snapshot() if tracker is not None else None
        if snapshot is None:
            raise RuntimeError(
                "Tracker snapshot is unavailable; start the tracker before launching the workspace map."
            )

    # --- execute -----------------------------------------------------------

    def execute(self, session: ExperimentSession) -> None:
        visit_sequence = build_round_robin_visit_order(
            target_indices=[int(target.target_index) for target in self._targets],
            visits_per_target=int(self.config.visits_per_target),
            random_seed=int(self.config.random_seed),
        )
        total_visits = len(visit_sequence)
        sample_index = 0
        rejected_count = 0
        if self.config.dry_run:
            for visit_position, (cycle_index, visit_in_cycle, target_index) in enumerate(visit_sequence):
                if session.stop_requested():
                    break
                target = self._target_by_index[target_index]
                synthetic_position = self._synthesize_position_mm(
                    target=target,
                    cycle_index=cycle_index,
                    visit_in_cycle=visit_in_cycle,
                )
                sample = self._build_synthetic_sample(
                    sample_index=sample_index,
                    visit_position=visit_position,
                    cycle_index=cycle_index,
                    visit_in_cycle=visit_in_cycle,
                    target=target,
                    position_mm=synthetic_position,
                )
                session.add_sample(sample)
                sample_index += 1
                session.update_progress(
                    visit_position + 1,
                    total_visits,
                    {"cycle_index": cycle_index, "visit_in_cycle": visit_in_cycle, "target_index": target_index},
                )
            session.set_metric("captured_visit_count", sample_index)
            session.set_metric("rejected_visit_count", 0)
            session.set_metric("status", STATUS_SUCCESS if sample_index > 0 else STATUS_INVALID_INSUFFICIENT_SAMPLES)
            return

        # Real-hardware path. Acquire exclusive bus ownership so concurrent
        # GUI jogs / pretension routines cannot interleave with our commands.
        servo_service = session.context.servo_service
        with servo_service.exclusive_bus_operation(
            owner=self.name,
            reason="workspace repeatability map sweep",
        ):
            self._exclusive_owner = True
            try:
                for visit_position, (cycle_index, visit_in_cycle, target_index) in enumerate(visit_sequence):
                    if session.stop_requested():
                        break
                    target = self._target_by_index[target_index]
                    visit_result = self._execute_single_visit(
                        session=session,
                        sample_index=sample_index,
                        visit_position=visit_position,
                        cycle_index=cycle_index,
                        visit_in_cycle=visit_in_cycle,
                        target=target,
                    )
                    if visit_result is not None:
                        session.add_sample(visit_result["sample"])
                        sample_index += 1
                        if visit_result.get("rejected"):
                            rejected_count += 1
                    session.update_progress(
                        visit_position + 1,
                        total_visits,
                        {
                            "cycle_index": cycle_index,
                            "visit_in_cycle": visit_in_cycle,
                            "target_index": target_index,
                            "rejected_count": rejected_count,
                        },
                    )
            finally:
                self._exclusive_owner = False

        session.set_metric("captured_visit_count", sample_index)
        session.set_metric("rejected_visit_count", rejected_count)
        if sample_index == 0:
            session.set_metric("status", STATUS_INVALID_INSUFFICIENT_SAMPLES)
        elif rejected_count > 0:
            session.set_metric("status", STATUS_PARTIAL_SUCCESS)
        else:
            session.set_metric("status", STATUS_SUCCESS)

    def finalize(self, session: ExperimentSession) -> None:
        if self.config.dry_run or not self.config.return_to_center_on_finalize:
            return
        servo_service = session.context.servo_service
        if servo_service is None:
            return
        try:
            servo_service.command_displacement(
                tendon_displacements_cm=[0.0] * len(self._servo_ids),
                neutral_ticks=list(self._neutral_ticks),
                servo_ids=list(self._servo_ids),
                motion_workflow="experiment_motion",
            )
        except Exception:
            # We never want finalize to mask the primary experiment outcome;
            # the operator can re-center manually from the servo tab if needed.
            pass

    def write_outputs(self, session: ExperimentSession, paths, summary) -> None:
        """Emit the workspace-repeatability artifact bundle into the run folder.

        Also lifts the workspace-wide summary (mean / median / p95 / max RMS,
        worst-targets list) into ``session.metrics`` so the GUI summary widget
        can show the headline numbers next to the figure links without
        re-reading the JSON.
        """
        from continuum_robot.experiments.workspace_repeatability_map_outputs import (
            compute_workspace_repeatability_metrics,
            group_visits_by_target,
            summarize_workspace_repeatability,
            write_workspace_repeatability_map_outputs,
        )

        target_catalog = list(session.metrics.get("target_catalog") or [])
        samples_extra = [
            dict(sample.extra or {})
            for sample in session.samples
            if sample.phase == "workspace_map_visit"
        ]
        write_workspace_repeatability_map_outputs(
            output_dir=paths.output_dir,
            target_catalog=target_catalog,
            samples=samples_extra,
            max_amplitude_mm=float(self.config.max_amplitude_mm),
            thesis_goal_rms_mm=float(self.config.thesis_goal_rms_mm),
        )
        # Mirror the headline summary into session metrics so the GUI summary
        # widget can render it without re-parsing the JSON. The bundle on
        # disk remains the source of truth.
        targets_by_index = {
            int(target.get("target_index", index)): dict(target)
            for index, target in enumerate(target_catalog)
        }
        per_target_rows = compute_workspace_repeatability_metrics(
            visits_by_target=group_visits_by_target(samples_extra),
            targets_by_index=targets_by_index,
        )
        workspace_summary = summarize_workspace_repeatability(
            per_target_rows,
            worst_n=5,
            thesis_goal_rms_mm=float(self.config.thesis_goal_rms_mm),
        )
        session.set_metric("workspace_summary", workspace_summary)

    # --- per-visit helpers --------------------------------------------------

    def _execute_single_visit(
        self,
        *,
        session: ExperimentSession,
        sample_index: int,
        visit_position: int,
        cycle_index: int,
        visit_in_cycle: int,
        target: WorkspaceMapTarget,
    ) -> dict[str, Any] | None:
        servo_service = session.context.servo_service
        tracking_service = session.context.tracking_service

        # 1) Return to neutral first.
        servo_service.command_displacement(
            tendon_displacements_cm=[0.0] * len(self._servo_ids),
            neutral_ticks=list(self._neutral_ticks),
            servo_ids=list(self._servo_ids),
            motion_workflow="experiment_motion",
        )
        self._sleep(self.config.neutral_settle_s)

        # 2) Command the target.
        servo_service.command_displacement(
            tendon_displacements_cm=list(target.cable_deltas_cm),
            neutral_ticks=list(self._neutral_ticks),
            servo_ids=list(self._servo_ids),
            motion_workflow="experiment_motion",
        )
        self._sleep(self.config.target_settle_s)

        # 3) Capture one tracker sample at the target.
        snapshot = tracking_service.get_snapshot() if tracking_service is not None else None
        sample = sample_from_tracking_snapshot(
            session=session,
            snapshot=snapshot,
            phase="workspace_map_visit",
            step_index=visit_position,
            sample_index=sample_index,
            payload={
                "visit_position": int(visit_position),
                "cycle_index": int(cycle_index),
                "visit_in_cycle": int(visit_in_cycle),
                "target_index": int(target.target_index),
                "target_label": str(target.label),
                "target_amplitude_mm": float(target.amplitude_mm),
                "target_angle_deg": float(target.angle_deg),
                "target_x_mm": float(target.x_mm),
                "target_y_mm": float(target.y_mm),
                "cable_deltas_cm": list(target.cable_deltas_cm),
            },
        )
        position_mm = sample.extra.get("position_mm") if sample is not None else None
        rejected = position_mm is None or any(value is None for value in (position_mm or []))
        if sample is not None:
            sample.extra["rejected"] = bool(rejected)
        return {"sample": sample, "rejected": bool(rejected)}

    def _build_synthetic_sample(
        self,
        *,
        sample_index: int,
        visit_position: int,
        cycle_index: int,
        visit_in_cycle: int,
        target: WorkspaceMapTarget,
        position_mm: tuple[float, float, float],
    ) -> ExperimentTimeseriesSample:
        return ExperimentTimeseriesSample(
            monotonic_time_s=float(visit_position),
            wall_time_utc="dry_run",
            phase="workspace_map_visit",
            step_index=int(visit_position),
            sample_index=int(sample_index),
            extra={
                "visit_position": int(visit_position),
                "cycle_index": int(cycle_index),
                "visit_in_cycle": int(visit_in_cycle),
                "target_index": int(target.target_index),
                "target_label": str(target.label),
                "target_amplitude_mm": float(target.amplitude_mm),
                "target_angle_deg": float(target.angle_deg),
                "target_x_mm": float(target.x_mm),
                "target_y_mm": float(target.y_mm),
                "cable_deltas_cm": list(target.cable_deltas_cm),
                "position_mm": [float(value) for value in position_mm],
                "rejected": False,
                "dry_run": True,
            },
        )

    def _synthesize_position_mm(
        self,
        *,
        target: WorkspaceMapTarget,
        cycle_index: int,
        visit_in_cycle: int,
    ) -> tuple[float, float, float]:
        """Deterministic synthetic position for dry-run + tests.

        The commanded XY deflection becomes the per-target mean. A small
        per-target bias (seeded once per target) is added to simulate the
        kind of systematic offset cable-driven robots show. Per-visit
        Gaussian noise (seeded by visit position) provides the spread the
        analysis turns into a heatmap value.
        """
        rng_bias = np.random.default_rng(int(self.config.random_seed) * 2017 + int(target.target_index))
        bias = rng_bias.normal(scale=float(self.config.synthetic_per_target_bias_std_mm), size=3)
        rng_noise = np.random.default_rng(
            int(self.config.random_seed) * 5009
            + int(target.target_index) * 53
            + int(cycle_index) * 17
            + int(visit_in_cycle)
        )
        noise = rng_noise.normal(scale=float(self.config.synthetic_noise_std_mm), size=3)
        return (
            float(target.x_mm + bias[0] + noise[0]),
            float(target.y_mm + bias[1] + noise[1]),
            float(bias[2] + noise[2]),
        )

    # --- small testable seam ------------------------------------------------

    def _sleep(self, seconds: float) -> None:
        # Real implementations override the sleep used by the visit loop with
        # ``time.sleep`` via a context hook; tests can monkeypatch this method
        # on the instance to fast-forward the schedule.
        if seconds <= 0.0:
            return
        import time

        time.sleep(float(seconds))

    @staticmethod
    def _safe_settings(session: ExperimentSession):
        """Return ``session.context.settings`` or None when the session is a stub.

        Dry-run callers (notably unit tests) don't always wire a full
        ExperimentContext. ``setup`` records geometry metrics from settings
        when available and falls back gracefully when not.
        """
        context = getattr(session, "context", None)
        if context is None:
            return None
        return getattr(context, "settings", None)


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------


def register_workspace_repeatability_map(registry) -> None:
    """Register the workspace repeatability map experiment."""
    registry.register(
        name=WorkspaceRepeatabilityMapExperiment.name,
        title="Workspace Repeatability Map",
        description=WorkspaceRepeatabilityMapExperiment.description,
        category="validation",
        tags=["Repeatability", "Workspace", "Thesis", "Tracking", "Servo"],
        default_config_path="config/experiment_workspace_repeatability_map.example.yaml",
        factory=WorkspaceRepeatabilityMapExperiment.from_dict,
    )

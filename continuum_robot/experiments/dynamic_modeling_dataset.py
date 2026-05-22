"""Continuous dynamic-modeling dataset experiment.

This experiment continuously drives the robot through a bounded random
tendon trajectory while continuously logging synchronized servo command,
servo telemetry, and tracker frames at approximately ``target_sample_rate_hz``
(default 20 Hz). It is isolated from the quasi-static
``collect_pose_command_dataset`` collector and does not modify any existing
experiment behavior.

The trajectory generator emits smooth low-pass-filtered tendon-displacement
commands, bounded by a configured maximum step size, total deviation, and a
hard safety cap. The sync builder aligns the latest command, the nearest
valid tracker frame, and the nearest valid servo telemetry record at each
sample tick, flagging the row invalid when either tracker or servo data is
older than the configured staleness thresholds.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
import bisect
import csv
import gzip
import io
import json
import logging
import math
from pathlib import Path
import random
import threading
import time
from typing import Any, Callable, Iterable

import yaml

from continuum_robot.experiments.framework import (
    BaseExperiment,
    ExperimentHardwareRequirements,
    ExperimentSession,
)
from continuum_robot.experiments.pair_axis_convention import (
    PAIR_AXIS_CONVENTION_VERSION,
    expand_pair_command_to_cable_deltas,
)
from continuum_robot.experiments.schemas import (
    ExperimentDatasetPaths,
    ExperimentSummary,
    ExperimentTimeseriesSample,
)
from continuum_robot.tracking.runtime_tip_policy import (
    WORKFLOW_MODELING_DATASET,
    evaluate_runtime_tip_trust,
)


LOG = logging.getLogger(__name__)


EXPERIMENT_NAME = "dynamic_modeling_dataset"


# Conservative default: lower-trust runtime tip modes (latest_accepted,
# quick_4_point) explicitly require an operator override to enter the dynamic
# modeling dataset path. Coil-as-tip remains the thesis-trusted default.
DEFAULT_TARGET_SAMPLE_RATE_HZ = 20.0
DEFAULT_COMMAND_UPDATE_RATE_HZ = 5.0
DEFAULT_DURATION_S = 30.0
DEFAULT_MAX_TICK_DELTA_FROM_START = 200
DEFAULT_MAX_TICK_DELTA_HARD_CAP = 600
DEFAULT_MAX_STEP_TICKS_PER_UPDATE = 25
DEFAULT_MAX_TRACKER_AGE_MS = 75.0
DEFAULT_MAX_SERVO_AGE_MS = 100.0
DEFAULT_TRACKER_POLL_INTERVAL_S = 0.005
DEFAULT_SERVO_TELEMETRY_INTERVAL_S = 0.03
DEFAULT_TRAJECTORY_SMOOTHING = 0.35


# ----------------------------------------------------------------------------
# Config
# ----------------------------------------------------------------------------


@dataclass
class DynamicModelingDatasetConfig:
    """Operator-facing configuration for the dynamic modeling dataset run."""

    duration_s: float = DEFAULT_DURATION_S
    target_sample_rate_hz: float = DEFAULT_TARGET_SAMPLE_RATE_HZ
    command_update_rate_hz: float = DEFAULT_COMMAND_UPDATE_RATE_HZ
    max_tick_delta_from_start: int = DEFAULT_MAX_TICK_DELTA_FROM_START
    max_tick_delta_hard_cap: int = DEFAULT_MAX_TICK_DELTA_HARD_CAP
    max_step_ticks_per_update: int = DEFAULT_MAX_STEP_TICKS_PER_UPDATE
    random_seed: int = 0
    servo_ids: list[int] = field(default_factory=list)
    require_tracker: bool = True
    return_to_start_at_end: bool = True
    max_tracker_age_ms: float = DEFAULT_MAX_TRACKER_AGE_MS
    max_servo_age_ms: float = DEFAULT_MAX_SERVO_AGE_MS
    tracker_poll_interval_s: float = DEFAULT_TRACKER_POLL_INTERVAL_S
    servo_telemetry_interval_s: float = DEFAULT_SERVO_TELEMETRY_INTERVAL_S
    trajectory_smoothing: float = DEFAULT_TRAJECTORY_SMOOTHING
    tracker_tool_id: str = "0A"
    dry_run: bool = False
    allow_lower_trust_runtime_tip: bool = False
    timeout_s: float = 0.0
    run_label: str = ""

    @classmethod
    def from_dict(cls, payload: dict[str, Any] | None) -> "DynamicModelingDatasetConfig":
        payload = dict(payload or {})
        servo_ids_raw = payload.get("servo_ids", []) or []
        if isinstance(servo_ids_raw, str):
            tokens = [
                segment.strip()
                for segment in servo_ids_raw.replace(";", ",").split(",")
                if segment.strip()
            ]
        else:
            tokens = [str(value).strip() for value in servo_ids_raw if str(value).strip()]
        servo_ids: list[int] = []
        for token in tokens:
            try:
                value = int(token)
            except (TypeError, ValueError):
                continue
            if value > 0 and value not in servo_ids:
                servo_ids.append(value)
        smoothing = float(payload.get("trajectory_smoothing", DEFAULT_TRAJECTORY_SMOOTHING))
        # Clamp smoothing into [0, 1] so callers cannot disable bounding by
        # passing pathological values. 0.0 = no momentum (random-walk steps
        # immediately), 1.0 = nearly stationary command.
        if not math.isfinite(smoothing):
            smoothing = DEFAULT_TRAJECTORY_SMOOTHING
        smoothing = max(0.0, min(1.0, float(smoothing)))
        return cls(
            duration_s=max(0.5, float(payload.get("duration_s", DEFAULT_DURATION_S))),
            target_sample_rate_hz=max(
                1.0,
                float(payload.get("target_sample_rate_hz", DEFAULT_TARGET_SAMPLE_RATE_HZ)),
            ),
            command_update_rate_hz=max(
                0.5,
                float(payload.get("command_update_rate_hz", DEFAULT_COMMAND_UPDATE_RATE_HZ)),
            ),
            max_tick_delta_from_start=max(
                1,
                int(payload.get("max_tick_delta_from_start", DEFAULT_MAX_TICK_DELTA_FROM_START)),
            ),
            max_tick_delta_hard_cap=max(
                1,
                int(payload.get("max_tick_delta_hard_cap", DEFAULT_MAX_TICK_DELTA_HARD_CAP)),
            ),
            max_step_ticks_per_update=max(
                1,
                int(payload.get("max_step_ticks_per_update", DEFAULT_MAX_STEP_TICKS_PER_UPDATE)),
            ),
            random_seed=int(payload.get("random_seed", 0)),
            servo_ids=servo_ids,
            require_tracker=bool(payload.get("require_tracker", True)),
            return_to_start_at_end=bool(payload.get("return_to_start_at_end", True)),
            max_tracker_age_ms=max(
                1.0,
                float(payload.get("max_tracker_age_ms", DEFAULT_MAX_TRACKER_AGE_MS)),
            ),
            max_servo_age_ms=max(
                1.0,
                float(payload.get("max_servo_age_ms", DEFAULT_MAX_SERVO_AGE_MS)),
            ),
            tracker_poll_interval_s=max(
                0.001,
                float(payload.get("tracker_poll_interval_s", DEFAULT_TRACKER_POLL_INTERVAL_S)),
            ),
            servo_telemetry_interval_s=max(
                0.005,
                float(payload.get("servo_telemetry_interval_s", DEFAULT_SERVO_TELEMETRY_INTERVAL_S)),
            ),
            trajectory_smoothing=smoothing,
            tracker_tool_id=str(payload.get("tracker_tool_id", "0A") or "0A").strip().upper() or "0A",
            dry_run=bool(payload.get("dry_run", False)),
            allow_lower_trust_runtime_tip=bool(payload.get("allow_lower_trust_runtime_tip", False)),
            timeout_s=float(payload.get("timeout_s", 0.0) or 0.0),
            run_label=str(payload.get("run_label", "") or "").strip(),
        )


# ----------------------------------------------------------------------------
# Trajectory generator
# ----------------------------------------------------------------------------


@dataclass(frozen=True)
class TrajectoryStep:
    """One commanded step in the smooth bounded random trajectory."""

    update_index: int
    elapsed_s: float
    tick_deltas: tuple[int, ...]
    cable_deltas_cm: tuple[float, ...]
    safety_clamped: bool


@dataclass
class BoundedRandomWalkTrajectory:
    """Smooth bounded random walk for per-servo tendon-tick deltas.

    The state is a per-servo tick delta relative to the captured starting
    position. Each update samples a per-servo Gaussian step, blends it with
    the previous step using ``trajectory_smoothing`` (low-pass to keep the
    motion smooth), then clamps to ``max_step_ticks_per_update``. The new
    delta is clamped to ``max_tick_delta_from_start`` and finally to
    ``max_tick_delta_hard_cap`` so the trajectory cannot escape the safety
    envelope even if the soft bound is misconfigured.
    """

    servo_ids: tuple[int, ...]
    max_tick_delta_from_start: int
    max_tick_delta_hard_cap: int
    max_step_ticks_per_update: int
    trajectory_smoothing: float
    random_seed: int

    def __post_init__(self) -> None:
        if len(self.servo_ids) == 0:
            raise ValueError("BoundedRandomWalkTrajectory requires at least one servo.")
        if self.max_tick_delta_hard_cap <= 0:
            raise ValueError("max_tick_delta_hard_cap must be positive.")
        effective_soft_cap = min(
            int(self.max_tick_delta_from_start),
            int(self.max_tick_delta_hard_cap),
        )
        if effective_soft_cap <= 0:
            raise ValueError("Effective soft tick cap must be positive.")
        self._soft_cap = int(effective_soft_cap)
        self._hard_cap = int(self.max_tick_delta_hard_cap)
        self._max_step = max(1, int(self.max_step_ticks_per_update))
        self._smoothing = max(0.0, min(1.0, float(self.trajectory_smoothing)))
        self._rng = random.Random(int(self.random_seed))
        self._current_deltas = [0.0 for _ in self.servo_ids]
        self._previous_step = [0.0 for _ in self.servo_ids]
        self._update_count = 0

    def next_step(self, *, elapsed_s: float) -> TrajectoryStep:
        """Return the next bounded random-walk step in tick units."""
        safety_clamped = False
        new_deltas: list[float] = []
        new_previous_step: list[float] = []
        for index, _ in enumerate(self.servo_ids):
            # Step magnitude is drawn from a gaussian centered on zero with
            # stdev = max_step / 3 so ~99.7% of raw draws stay below
            # max_step; the explicit clamp guarantees the hard limit.
            raw_step = self._rng.gauss(0.0, self._max_step / 3.0)
            blended = (
                self._smoothing * self._previous_step[index]
                + (1.0 - self._smoothing) * raw_step
            )
            clamped_step = max(-float(self._max_step), min(float(self._max_step), blended))
            tentative = self._current_deltas[index] + clamped_step
            # Soft bound first; if violated, mark this step as clamped and
            # pull the trajectory back inside the configured envelope.
            if tentative > self._soft_cap:
                tentative = float(self._soft_cap)
                safety_clamped = True
            elif tentative < -self._soft_cap:
                tentative = -float(self._soft_cap)
                safety_clamped = True
            if tentative > self._hard_cap:
                tentative = float(self._hard_cap)
                safety_clamped = True
            elif tentative < -self._hard_cap:
                tentative = -float(self._hard_cap)
                safety_clamped = True
            new_deltas.append(tentative)
            new_previous_step.append(tentative - self._current_deltas[index])
        self._current_deltas = new_deltas
        self._previous_step = new_previous_step
        self._update_count += 1
        ticks = tuple(int(round(value)) for value in new_deltas)
        step = TrajectoryStep(
            update_index=int(self._update_count),
            elapsed_s=float(max(0.0, elapsed_s)),
            tick_deltas=ticks,
            cable_deltas_cm=tuple(0.0 for _ in ticks),
            safety_clamped=bool(safety_clamped),
        )
        return step

    def coast_to_zero(self) -> TrajectoryStep:
        """Return a step that linearly pulls every servo back toward zero delta."""
        new_deltas: list[float] = []
        new_step: list[float] = []
        for index, current in enumerate(self._current_deltas):
            step = max(-float(self._max_step), min(float(self._max_step), -current))
            new_value = current + step
            new_deltas.append(new_value)
            new_step.append(step)
        self._current_deltas = new_deltas
        self._previous_step = new_step
        self._update_count += 1
        return TrajectoryStep(
            update_index=int(self._update_count),
            elapsed_s=0.0,
            tick_deltas=tuple(int(round(value)) for value in new_deltas),
            cable_deltas_cm=tuple(0.0 for _ in new_deltas),
            safety_clamped=False,
        )

    def current_delta_ticks(self) -> tuple[int, ...]:
        return tuple(int(round(value)) for value in self._current_deltas)

    @property
    def soft_cap(self) -> int:
        return int(self._soft_cap)

    @property
    def hard_cap(self) -> int:
        return int(self._hard_cap)


# ----------------------------------------------------------------------------
# Stream records and sync builder
# ----------------------------------------------------------------------------


@dataclass
class CommandRecord:
    """One commanded position record from the trajectory generator."""

    command_id: int
    monotonic_ns: int
    host_time_s: float
    cable_deltas_cm: tuple[float, ...]
    goal_ticks_by_servo: dict[int, int]
    delta_ticks_by_servo: dict[int, int]
    safety_status: str = "ok"


@dataclass
class TrackerRecord:
    """One tracker frame observed by the synchronizer."""

    frame_index: int
    monotonic_ns: int
    host_time_s: float
    frame_number: int | None
    frame_time_s: float | None
    tip_xyz_mm: tuple[float, float, float] | None
    tip_quat_wxyz: tuple[float, float, float, float] | None
    valid: bool
    age_ms_at_observation: float | None


@dataclass
class ServoTelemetryRecord:
    """One servo-telemetry batch observed by the synchronizer."""

    sample_index: int
    monotonic_ns: int
    host_time_s: float
    positions_by_servo: dict[int, int | None]
    currents_by_servo: dict[int, int | None]
    voltages_by_servo: dict[int, int | None]
    temperatures_by_servo: dict[int, int | None]
    valid: bool
    age_ms_by_servo: dict[int, float | None]
    failure_code: str | None = None


@dataclass
class SynchronizedSample:
    """One synchronized dynamic dataset row."""

    sample_index: int
    host_time_s: float
    monotonic_ns: int
    command: CommandRecord | None
    tracker: TrackerRecord | None
    servo: ServoTelemetryRecord | None
    command_to_tracker_dt_ms: float | None
    servo_to_tracker_dt_ms: float | None
    sample_valid: bool
    failure_code: str | None


def latest_at_or_before(records: list, monotonic_ns: int):
    """Return the last record with monotonic_ns <= the given timestamp."""
    if not records:
        return None
    timestamps = [int(record.monotonic_ns) for record in records]
    idx = bisect.bisect_right(timestamps, int(monotonic_ns)) - 1
    if idx < 0:
        return None
    return records[idx]


def nearest_record(records: list, monotonic_ns: int):
    """Return the record with the smallest |monotonic_ns - record| distance."""
    if not records:
        return None
    timestamps = [int(record.monotonic_ns) for record in records]
    pos = bisect.bisect_left(timestamps, int(monotonic_ns))
    candidates = []
    if pos < len(records):
        candidates.append(records[pos])
    if pos > 0:
        candidates.append(records[pos - 1])
    return min(candidates, key=lambda record: abs(int(record.monotonic_ns) - int(monotonic_ns)))


def build_synchronized_sample(
    *,
    sample_index: int,
    sample_monotonic_ns: int,
    sample_host_time_s: float,
    commands: list[CommandRecord],
    trackers: list[TrackerRecord],
    servos: list[ServoTelemetryRecord],
    max_tracker_age_ms: float,
    max_servo_age_ms: float,
    require_tracker: bool,
) -> SynchronizedSample:
    """Align the latest command, nearest tracker, and nearest servo telemetry.

    Returns a ``SynchronizedSample`` whose ``sample_valid`` flag is False if
    the tracker or servo telemetry is older than the configured thresholds.
    The raw streams are preserved so downstream auditors can replay alignment
    choices.
    """
    command = latest_at_or_before(commands, sample_monotonic_ns)
    tracker = nearest_record(trackers, sample_monotonic_ns)
    servo = nearest_record(servos, sample_monotonic_ns)

    failure_codes: list[str] = []
    command_to_tracker_dt_ms: float | None = None
    servo_to_tracker_dt_ms: float | None = None

    if tracker is not None and command is not None:
        command_to_tracker_dt_ms = float(
            (int(tracker.monotonic_ns) - int(command.monotonic_ns)) / 1_000_000.0
        )
    if tracker is not None and servo is not None:
        servo_to_tracker_dt_ms = float(
            (int(servo.monotonic_ns) - int(tracker.monotonic_ns)) / 1_000_000.0
        )

    tracker_age_ms = (
        abs(float(int(sample_monotonic_ns) - int(tracker.monotonic_ns)) / 1_000_000.0)
        if tracker is not None
        else None
    )
    servo_age_ms = (
        abs(float(int(sample_monotonic_ns) - int(servo.monotonic_ns)) / 1_000_000.0)
        if servo is not None
        else None
    )

    sample_valid = True
    if command is None:
        sample_valid = False
        failure_codes.append("missing_command")
    if require_tracker:
        if tracker is None:
            sample_valid = False
            failure_codes.append("missing_tracker_frame")
        elif not bool(tracker.valid):
            sample_valid = False
            failure_codes.append("tracker_invalid")
        elif tracker_age_ms is not None and tracker_age_ms > float(max_tracker_age_ms):
            sample_valid = False
            failure_codes.append("tracker_stale")
    if servo is None:
        sample_valid = False
        failure_codes.append("missing_servo_telemetry")
    elif not bool(servo.valid):
        sample_valid = False
        failure_codes.append("servo_invalid")
    elif servo_age_ms is not None and servo_age_ms > float(max_servo_age_ms):
        sample_valid = False
        failure_codes.append("servo_stale")

    return SynchronizedSample(
        sample_index=int(sample_index),
        host_time_s=float(sample_host_time_s),
        monotonic_ns=int(sample_monotonic_ns),
        command=command,
        tracker=tracker,
        servo=servo,
        command_to_tracker_dt_ms=command_to_tracker_dt_ms,
        servo_to_tracker_dt_ms=servo_to_tracker_dt_ms,
        sample_valid=bool(sample_valid),
        failure_code=";".join(failure_codes) if failure_codes else None,
    )


# ----------------------------------------------------------------------------
# CSV.GZ writers
# ----------------------------------------------------------------------------


SYNC_COLUMNS: tuple[str, ...] = (
    "sample_index",
    "host_time_s",
    "monotonic_ns",
    "command_id",
    "command_time_s",
    "cmd_d1_mm",
    "cmd_d2_mm",
    "cmd_d3_mm",
    "cmd_d4_mm",
    "cmd_s1_tick",
    "cmd_s2_tick",
    "cmd_s3_tick",
    "cmd_s4_tick",
    "servo_time_s",
    "s1_pos_tick",
    "s2_pos_tick",
    "s3_pos_tick",
    "s4_pos_tick",
    "s1_current_ma",
    "s2_current_ma",
    "s3_current_ma",
    "s4_current_ma",
    "s1_voltage_v",
    "s2_voltage_v",
    "s3_voltage_v",
    "s4_voltage_v",
    "servo_valid",
    "servo_age_ms",
    "tracker_host_time_s",
    "tracker_frame_number",
    "tracker_frame_time_s",
    "tip_x_mm",
    "tip_y_mm",
    "tip_z_mm",
    "tip_qw",
    "tip_qx",
    "tip_qy",
    "tip_qz",
    "tracker_valid",
    "tracker_age_ms",
    "command_to_tracker_dt_ms",
    "servo_to_tracker_dt_ms",
    "sample_valid",
    "failure_code",
)


COMMAND_COLUMNS: tuple[str, ...] = (
    "command_id",
    "host_time_s",
    "monotonic_ns",
    "cmd_d1_mm",
    "cmd_d2_mm",
    "cmd_d3_mm",
    "cmd_d4_mm",
    "cmd_s1_tick",
    "cmd_s2_tick",
    "cmd_s3_tick",
    "cmd_s4_tick",
    "safety_status",
)


TRACKER_COLUMNS: tuple[str, ...] = (
    "frame_index",
    "host_time_s",
    "monotonic_ns",
    "frame_number",
    "frame_time_s",
    "tip_x_mm",
    "tip_y_mm",
    "tip_z_mm",
    "tip_qw",
    "tip_qx",
    "tip_qy",
    "tip_qz",
    "tracker_valid",
    "tracker_age_ms",
)


SERVO_COLUMNS: tuple[str, ...] = (
    "sample_index",
    "host_time_s",
    "monotonic_ns",
    "s1_pos_tick",
    "s2_pos_tick",
    "s3_pos_tick",
    "s4_pos_tick",
    "s1_current_ma",
    "s2_current_ma",
    "s3_current_ma",
    "s4_current_ma",
    "s1_voltage_v",
    "s2_voltage_v",
    "s3_voltage_v",
    "s4_voltage_v",
    "s1_temperature_c",
    "s2_temperature_c",
    "s3_temperature_c",
    "s4_temperature_c",
    "servo_valid",
    "failure_code",
)


def _fmt(value, *, decimals: int | None = None) -> str:
    if value is None:
        return ""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, float):
        if not math.isfinite(value):
            return ""
        if decimals is not None:
            return f"{value:.{decimals}f}"
        return f"{value:.6f}"
    return str(value)


def _servo_slot_value(records: dict[int, Any] | None, servo_id: int) -> Any:
    if not records or servo_id is None:
        return None
    return records.get(int(servo_id))


def _voltage_v_from_mv(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(int(value)) / 1000.0
    except (TypeError, ValueError):
        return None


def write_sync_csv_gz(
    path: Path,
    samples: list[SynchronizedSample],
    *,
    servo_id_slots: list[int],
) -> None:
    """Write the synchronized samples to ``samples`` as a compressed CSV."""
    slot_ids = list(servo_id_slots) + [None] * max(0, 4 - len(servo_id_slots))
    slot_ids = slot_ids[:4]
    with gzip.open(path, "wt", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(list(SYNC_COLUMNS))
        for sample in samples:
            command = sample.command
            tracker = sample.tracker
            servo = sample.servo
            cmd_mm_values: list[float | None] = [None, None, None, None]
            cmd_tick_values: list[int | None] = [None, None, None, None]
            if command is not None:
                for index in range(4):
                    delta_cm = (
                        command.cable_deltas_cm[index]
                        if index < len(command.cable_deltas_cm)
                        else None
                    )
                    cmd_mm_values[index] = (
                        None if delta_cm is None else float(delta_cm) * 10.0
                    )
                    servo_id = slot_ids[index]
                    if servo_id is not None:
                        cmd_tick_values[index] = command.goal_ticks_by_servo.get(int(servo_id))
            servo_positions: list[int | None] = [None, None, None, None]
            servo_currents: list[int | None] = [None, None, None, None]
            servo_voltages: list[float | None] = [None, None, None, None]
            if servo is not None:
                for index, servo_id in enumerate(slot_ids):
                    if servo_id is None:
                        continue
                    servo_positions[index] = _servo_slot_value(servo.positions_by_servo, int(servo_id))
                    servo_currents[index] = _servo_slot_value(servo.currents_by_servo, int(servo_id))
                    servo_voltages[index] = _voltage_v_from_mv(
                        _servo_slot_value(servo.voltages_by_servo, int(servo_id))
                    )
            tracker_age_ms_value = (
                None
                if tracker is None
                else abs(
                    float(int(sample.monotonic_ns) - int(tracker.monotonic_ns)) / 1_000_000.0
                )
            )
            servo_age_ms_value = (
                None
                if servo is None
                else abs(
                    float(int(sample.monotonic_ns) - int(servo.monotonic_ns)) / 1_000_000.0
                )
            )
            row = [
                _fmt(sample.sample_index),
                _fmt(sample.host_time_s),
                _fmt(sample.monotonic_ns),
                _fmt(command.command_id if command is not None else None),
                _fmt(command.host_time_s if command is not None else None),
                _fmt(cmd_mm_values[0]),
                _fmt(cmd_mm_values[1]),
                _fmt(cmd_mm_values[2]),
                _fmt(cmd_mm_values[3]),
                _fmt(cmd_tick_values[0]),
                _fmt(cmd_tick_values[1]),
                _fmt(cmd_tick_values[2]),
                _fmt(cmd_tick_values[3]),
                _fmt(servo.host_time_s if servo is not None else None),
                _fmt(servo_positions[0]),
                _fmt(servo_positions[1]),
                _fmt(servo_positions[2]),
                _fmt(servo_positions[3]),
                _fmt(servo_currents[0]),
                _fmt(servo_currents[1]),
                _fmt(servo_currents[2]),
                _fmt(servo_currents[3]),
                _fmt(servo_voltages[0]),
                _fmt(servo_voltages[1]),
                _fmt(servo_voltages[2]),
                _fmt(servo_voltages[3]),
                _fmt(servo.valid if servo is not None else None),
                _fmt(servo_age_ms_value),
                _fmt(tracker.host_time_s if tracker is not None else None),
                _fmt(tracker.frame_number if tracker is not None else None),
                _fmt(tracker.frame_time_s if tracker is not None else None),
                _fmt(tracker.tip_xyz_mm[0] if tracker is not None and tracker.tip_xyz_mm is not None else None),
                _fmt(tracker.tip_xyz_mm[1] if tracker is not None and tracker.tip_xyz_mm is not None else None),
                _fmt(tracker.tip_xyz_mm[2] if tracker is not None and tracker.tip_xyz_mm is not None else None),
                _fmt(tracker.tip_quat_wxyz[0] if tracker is not None and tracker.tip_quat_wxyz is not None else None),
                _fmt(tracker.tip_quat_wxyz[1] if tracker is not None and tracker.tip_quat_wxyz is not None else None),
                _fmt(tracker.tip_quat_wxyz[2] if tracker is not None and tracker.tip_quat_wxyz is not None else None),
                _fmt(tracker.tip_quat_wxyz[3] if tracker is not None and tracker.tip_quat_wxyz is not None else None),
                _fmt(tracker.valid if tracker is not None else None),
                _fmt(tracker_age_ms_value),
                _fmt(sample.command_to_tracker_dt_ms),
                _fmt(sample.servo_to_tracker_dt_ms),
                _fmt(sample.sample_valid),
                _fmt(sample.failure_code) if sample.failure_code else "",
            ]
            writer.writerow(row)


def write_commands_csv_gz(
    path: Path,
    commands: list[CommandRecord],
    *,
    servo_id_slots: list[int],
) -> None:
    slot_ids = list(servo_id_slots) + [None] * max(0, 4 - len(servo_id_slots))
    slot_ids = slot_ids[:4]
    with gzip.open(path, "wt", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(list(COMMAND_COLUMNS))
        for command in commands:
            cmd_mm: list[float | None] = [None, None, None, None]
            cmd_tick: list[int | None] = [None, None, None, None]
            for index in range(4):
                delta_cm = (
                    command.cable_deltas_cm[index]
                    if index < len(command.cable_deltas_cm)
                    else None
                )
                cmd_mm[index] = None if delta_cm is None else float(delta_cm) * 10.0
                servo_id = slot_ids[index]
                if servo_id is not None:
                    cmd_tick[index] = command.goal_ticks_by_servo.get(int(servo_id))
            writer.writerow(
                [
                    _fmt(command.command_id),
                    _fmt(command.host_time_s),
                    _fmt(command.monotonic_ns),
                    _fmt(cmd_mm[0]),
                    _fmt(cmd_mm[1]),
                    _fmt(cmd_mm[2]),
                    _fmt(cmd_mm[3]),
                    _fmt(cmd_tick[0]),
                    _fmt(cmd_tick[1]),
                    _fmt(cmd_tick[2]),
                    _fmt(cmd_tick[3]),
                    command.safety_status,
                ]
            )


def write_tracker_csv_gz(path: Path, trackers: list[TrackerRecord]) -> None:
    with gzip.open(path, "wt", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(list(TRACKER_COLUMNS))
        for tracker in trackers:
            tip_xyz = tracker.tip_xyz_mm or (None, None, None)
            tip_quat = tracker.tip_quat_wxyz or (None, None, None, None)
            writer.writerow(
                [
                    _fmt(tracker.frame_index),
                    _fmt(tracker.host_time_s),
                    _fmt(tracker.monotonic_ns),
                    _fmt(tracker.frame_number),
                    _fmt(tracker.frame_time_s),
                    _fmt(tip_xyz[0]),
                    _fmt(tip_xyz[1]),
                    _fmt(tip_xyz[2]),
                    _fmt(tip_quat[0]),
                    _fmt(tip_quat[1]),
                    _fmt(tip_quat[2]),
                    _fmt(tip_quat[3]),
                    _fmt(tracker.valid),
                    _fmt(tracker.age_ms_at_observation),
                ]
            )


def write_servo_csv_gz(
    path: Path,
    telemetry: list[ServoTelemetryRecord],
    *,
    servo_id_slots: list[int],
) -> None:
    slot_ids = list(servo_id_slots) + [None] * max(0, 4 - len(servo_id_slots))
    slot_ids = slot_ids[:4]
    with gzip.open(path, "wt", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(list(SERVO_COLUMNS))
        for record in telemetry:
            positions = [
                _servo_slot_value(record.positions_by_servo, int(servo_id)) if servo_id is not None else None
                for servo_id in slot_ids
            ]
            currents = [
                _servo_slot_value(record.currents_by_servo, int(servo_id)) if servo_id is not None else None
                for servo_id in slot_ids
            ]
            voltages = [
                _voltage_v_from_mv(_servo_slot_value(record.voltages_by_servo, int(servo_id)))
                if servo_id is not None
                else None
                for servo_id in slot_ids
            ]
            temperatures = [
                _servo_slot_value(record.temperatures_by_servo, int(servo_id)) if servo_id is not None else None
                for servo_id in slot_ids
            ]
            writer.writerow(
                [
                    _fmt(record.sample_index),
                    _fmt(record.host_time_s),
                    _fmt(record.monotonic_ns),
                    _fmt(positions[0]),
                    _fmt(positions[1]),
                    _fmt(positions[2]),
                    _fmt(positions[3]),
                    _fmt(currents[0]),
                    _fmt(currents[1]),
                    _fmt(currents[2]),
                    _fmt(currents[3]),
                    _fmt(voltages[0]),
                    _fmt(voltages[1]),
                    _fmt(voltages[2]),
                    _fmt(voltages[3]),
                    _fmt(temperatures[0]),
                    _fmt(temperatures[1]),
                    _fmt(temperatures[2]),
                    _fmt(temperatures[3]),
                    _fmt(record.valid),
                    _fmt(record.failure_code) if record.failure_code else "",
                ]
            )


# ----------------------------------------------------------------------------
# Metrics
# ----------------------------------------------------------------------------


def _percentile(values: list[float], pct: float) -> float | None:
    cleaned = [float(value) for value in values if value is not None and math.isfinite(float(value))]
    if not cleaned:
        return None
    cleaned.sort()
    rank = max(0.0, min(1.0, float(pct) / 100.0))
    position = rank * (len(cleaned) - 1)
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return float(cleaned[lower])
    frac = position - lower
    return float(cleaned[lower] * (1.0 - frac) + cleaned[upper] * frac)


def summarize_dynamic_run(
    *,
    config: DynamicModelingDatasetConfig,
    elapsed_s: float,
    sync_samples: list[SynchronizedSample],
    commands: list[CommandRecord],
    trackers: list[TrackerRecord],
    servo_telemetry: list[ServoTelemetryRecord],
    failure_code: str | None,
    thesis_eligible: bool,
    eligibility_reasons: list[str],
    transform_chain_summary: dict[str, Any] | None,
) -> dict[str, Any]:
    valid_count = sum(1 for sample in sync_samples if sample.sample_valid)
    invalid_count = len(sync_samples) - valid_count
    tracker_ages = []
    servo_ages = []
    sync_offsets = []
    for sample in sync_samples:
        if sample.tracker is not None:
            tracker_ages.append(
                abs(float(int(sample.monotonic_ns) - int(sample.tracker.monotonic_ns)) / 1_000_000.0)
            )
        if sample.servo is not None:
            servo_ages.append(
                abs(float(int(sample.monotonic_ns) - int(sample.servo.monotonic_ns)) / 1_000_000.0)
            )
        if sample.servo_to_tracker_dt_ms is not None:
            sync_offsets.append(abs(float(sample.servo_to_tracker_dt_ms)))
    unique_tracker_frames = {
        record.frame_number for record in trackers if record.frame_number is not None
    }
    unique_tracker_count = len(unique_tracker_frames)
    achieved_sample_rate = (
        float(len(sync_samples)) / float(elapsed_s) if elapsed_s > 0 else 0.0
    )
    tracker_frame_rate = (
        float(len(trackers)) / float(elapsed_s) if elapsed_s > 0 else 0.0
    )
    unique_tracker_rate = (
        float(unique_tracker_count) / float(elapsed_s) if elapsed_s > 0 else 0.0
    )
    servo_rate = (
        float(len(servo_telemetry)) / float(elapsed_s) if elapsed_s > 0 else 0.0
    )
    max_commanded_tick_delta = 0
    for command in commands:
        for value in command.delta_ticks_by_servo.values():
            if value is None:
                continue
            magnitude = abs(int(value))
            if magnitude > max_commanded_tick_delta:
                max_commanded_tick_delta = magnitude
    max_measured_tick_delta = 0
    starting_position_by_servo: dict[int, int] = {}
    for record in servo_telemetry:
        for servo_id, position in (record.positions_by_servo or {}).items():
            if position is None:
                continue
            if servo_id not in starting_position_by_servo:
                starting_position_by_servo[servo_id] = int(position)
            else:
                delta = abs(int(position) - starting_position_by_servo[servo_id])
                if delta > max_measured_tick_delta:
                    max_measured_tick_delta = delta
    max_current_by_servo: dict[int, int] = {}
    for record in servo_telemetry:
        for servo_id, current_value in (record.currents_by_servo or {}).items():
            if current_value is None:
                continue
            existing = max_current_by_servo.get(int(servo_id), 0)
            max_current_by_servo[int(servo_id)] = max(existing, abs(int(current_value)))
    return {
        "duration_requested_s": float(config.duration_s),
        "duration_completed_s": float(elapsed_s),
        "target_sample_rate_hz": float(config.target_sample_rate_hz),
        "achieved_sample_rate_hz": float(achieved_sample_rate),
        "dynamic_sample_row_count": int(len(sync_samples)),
        "command_event_count": int(len(commands)),
        "tracker_unique_frame_count": int(unique_tracker_count),
        "tracker_unique_frame_rate_hz": float(unique_tracker_rate),
        "tracker_observed_frame_rate_hz": float(tracker_frame_rate),
        "servo_telemetry_count": int(len(servo_telemetry)),
        "servo_telemetry_rate_hz": float(servo_rate),
        "valid_sample_count": int(valid_count),
        "invalid_sample_count": int(invalid_count),
        "valid_sample_ratio": (
            float(valid_count) / float(len(sync_samples)) if sync_samples else 0.0
        ),
        "tracker_age_ms_median": _percentile(tracker_ages, 50.0),
        "tracker_age_ms_p95": _percentile(tracker_ages, 95.0),
        "servo_age_ms_median": _percentile(servo_ages, 50.0),
        "servo_age_ms_p95": _percentile(servo_ages, 95.0),
        "servo_to_tracker_offset_ms_median": _percentile(sync_offsets, 50.0),
        "servo_to_tracker_offset_ms_p95": _percentile(sync_offsets, 95.0),
        "max_commanded_tick_delta": int(max_commanded_tick_delta),
        "max_measured_tick_delta": int(max_measured_tick_delta),
        "max_current_ma_by_servo": {str(key): int(value) for key, value in max_current_by_servo.items()},
        "failure_code": failure_code,
        "thesis_eligible": bool(thesis_eligible),
        "thesis_eligibility_reasons": list(eligibility_reasons),
        "pair_axis_convention": PAIR_AXIS_CONVENTION_VERSION,
        "transform_chain_summary": dict(transform_chain_summary or {}),
        "random_seed": int(config.random_seed),
    }


# ----------------------------------------------------------------------------
# Figures
# ----------------------------------------------------------------------------


def _render_default_figures(
    *,
    output_dir: Path,
    sync_samples: list[SynchronizedSample],
    commands: list[CommandRecord],
    trackers: list[TrackerRecord],
    servo_telemetry: list[ServoTelemetryRecord],
    servo_id_slots: list[int],
    sample_validity_threshold_ms: float,
) -> list[str]:
    """Render basic dynamic-modeling figures if matplotlib is available."""
    figure_paths: list[str] = []
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover - environment-dependent
        LOG.warning("matplotlib unavailable; dynamic modeling figures skipped: %s", exc)
        return figure_paths

    base_time_s = sync_samples[0].host_time_s if sync_samples else 0.0

    # 1. Tip XY trajectory colored by time.
    tip_x = []
    tip_y = []
    tip_t = []
    for sample in sync_samples:
        if sample.tracker is not None and sample.tracker.tip_xyz_mm is not None:
            tip_x.append(float(sample.tracker.tip_xyz_mm[0]))
            tip_y.append(float(sample.tracker.tip_xyz_mm[1]))
            tip_t.append(float(sample.host_time_s - base_time_s))
    if tip_x:
        fig, ax = plt.subplots(figsize=(5.0, 5.0))
        scatter = ax.scatter(tip_x, tip_y, c=tip_t, s=6, cmap="viridis")
        ax.set_xlabel("Tip X (mm)")
        ax.set_ylabel("Tip Y (mm)")
        ax.set_title("Tip XY trajectory")
        ax.set_aspect("equal", adjustable="datalim")
        fig.colorbar(scatter, ax=ax, label="time (s)")
        fig.tight_layout()
        target = output_dir / "tip_xy_trajectory.png"
        fig.savefig(target, dpi=150)
        plt.close(fig)
        figure_paths.append(target.name)

    # 2. Tendon command over time.
    cmd_times = [float(command.host_time_s - base_time_s) for command in commands]
    if commands:
        fig, ax = plt.subplots(figsize=(6.0, 3.0))
        for index in range(4):
            values_mm = []
            for command in commands:
                delta_cm = (
                    command.cable_deltas_cm[index]
                    if index < len(command.cable_deltas_cm)
                    else None
                )
                values_mm.append(None if delta_cm is None else float(delta_cm) * 10.0)
            ax.plot(cmd_times, values_mm, label=f"cable {index + 1}", linewidth=0.9)
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Commanded cable delta (mm)")
        ax.set_title("Commanded tendon trajectory")
        ax.legend(loc="best", fontsize=8)
        fig.tight_layout()
        target = output_dir / "commanded_tendon_trajectory.png"
        fig.savefig(target, dpi=150)
        plt.close(fig)
        figure_paths.append(target.name)

    # 3. Servo measured positions over time.
    if servo_telemetry:
        fig, ax = plt.subplots(figsize=(6.0, 3.0))
        times = [float(record.host_time_s - base_time_s) for record in servo_telemetry]
        for servo_id in servo_id_slots:
            values = [
                record.positions_by_servo.get(int(servo_id)) if record.positions_by_servo else None
                for record in servo_telemetry
            ]
            ax.plot(times, values, label=f"servo {servo_id}", linewidth=0.9)
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Position (ticks)")
        ax.set_title("Measured servo positions")
        ax.legend(loc="best", fontsize=8)
        fig.tight_layout()
        target = output_dir / "servo_positions.png"
        fig.savefig(target, dpi=150)
        plt.close(fig)
        figure_paths.append(target.name)

    # 4. Tracker frame rate / unique frame rate.
    if trackers:
        fig, ax = plt.subplots(figsize=(6.0, 3.0))
        host_times = [float(record.host_time_s - base_time_s) for record in trackers]
        unique_seen: set[int] = set()
        observed_count = []
        unique_count = []
        for index, record in enumerate(trackers, start=1):
            observed_count.append(index)
            if record.frame_number is not None and record.frame_number not in unique_seen:
                unique_seen.add(int(record.frame_number))
            unique_count.append(len(unique_seen))
        ax.plot(host_times, observed_count, label="observed frames", linewidth=0.9)
        ax.plot(host_times, unique_count, label="unique frames", linewidth=0.9)
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Cumulative frame count")
        ax.set_title("Tracker frame stream")
        ax.legend(loc="best", fontsize=8)
        fig.tight_layout()
        target = output_dir / "tracker_frame_rate.png"
        fig.savefig(target, dpi=150)
        plt.close(fig)
        figure_paths.append(target.name)

    # 5. Servo telemetry rate over time.
    if servo_telemetry:
        fig, ax = plt.subplots(figsize=(6.0, 3.0))
        times = [float(record.host_time_s - base_time_s) for record in servo_telemetry]
        ax.plot(times, list(range(1, len(servo_telemetry) + 1)), linewidth=0.9)
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Cumulative telemetry sample count")
        ax.set_title("Servo telemetry stream")
        fig.tight_layout()
        target = output_dir / "servo_telemetry_rate.png"
        fig.savefig(target, dpi=150)
        plt.close(fig)
        figure_paths.append(target.name)

    # 6. Tracker/servo sync offset histogram.
    offsets = [
        float(sample.servo_to_tracker_dt_ms)
        for sample in sync_samples
        if sample.servo_to_tracker_dt_ms is not None
    ]
    if offsets:
        fig, ax = plt.subplots(figsize=(6.0, 3.0))
        ax.hist(offsets, bins=30)
        ax.set_xlabel("Servo - tracker offset (ms)")
        ax.set_ylabel("Sync sample count")
        ax.set_title("Sync offset histogram")
        fig.tight_layout()
        target = output_dir / "sync_offset_histogram.png"
        fig.savefig(target, dpi=150)
        plt.close(fig)
        figure_paths.append(target.name)

    # 7. Sample validity timeline.
    if sync_samples:
        fig, ax = plt.subplots(figsize=(6.0, 2.0))
        times = [float(sample.host_time_s - base_time_s) for sample in sync_samples]
        validity = [1 if sample.sample_valid else 0 for sample in sync_samples]
        ax.step(times, validity, where="post")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Valid (1=yes, 0=no)")
        ax.set_title("Sample validity timeline")
        ax.set_ylim(-0.1, 1.1)
        fig.tight_layout()
        target = output_dir / "sample_validity_timeline.png"
        fig.savefig(target, dpi=150)
        plt.close(fig)
        figure_paths.append(target.name)

    # 8. Current per servo over time.
    if servo_telemetry:
        fig, ax = plt.subplots(figsize=(6.0, 3.0))
        times = [float(record.host_time_s - base_time_s) for record in servo_telemetry]
        for servo_id in servo_id_slots:
            values = [
                record.currents_by_servo.get(int(servo_id)) if record.currents_by_servo else None
                for record in servo_telemetry
            ]
            ax.plot(times, values, label=f"servo {servo_id}", linewidth=0.9)
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Current (mA)")
        ax.set_title("Servo current over time")
        ax.legend(loc="best", fontsize=8)
        fig.tight_layout()
        target = output_dir / "servo_current_over_time.png"
        fig.savefig(target, dpi=150)
        plt.close(fig)
        figure_paths.append(target.name)
    _ = sample_validity_threshold_ms  # reserved for future overlay
    return figure_paths


# ----------------------------------------------------------------------------
# Tracker / servo polling helpers
# ----------------------------------------------------------------------------


def _safe_tracker_snapshot(tracking_service) -> Any | None:
    if tracking_service is None:
        return None
    reader = getattr(tracking_service, "peek_snapshot", None)
    try:
        if callable(reader):
            return reader()
        return tracking_service.get_snapshot()
    except Exception as exc:
        LOG.debug("Tracker snapshot read failed: %s", exc)
        return None


def _extract_tracker_record(
    *,
    snapshot,
    tool_id: str,
    frame_index: int,
    monotonic_ns: int,
    host_time_s: float,
    last_frame_number: int | None,
) -> tuple[TrackerRecord | None, int | None]:
    """Build one TrackerRecord from the latest tracker snapshot if it is new.

    Returns ``(record_or_None, observed_frame_number)``. ``record_or_None`` is
    ``None`` when the snapshot has not advanced to a new frame yet.
    """
    if snapshot is None:
        return None, last_frame_number
    frame_number = getattr(snapshot, "last_frame_number", None)
    if frame_number is None and last_frame_number is None:
        # No frame info on this backend; treat each call as a fresh sample so
        # downstream age checks remain meaningful but mark it not unique.
        frame_number_resolved = None
    else:
        if frame_number is None:
            return None, last_frame_number
        if last_frame_number is not None and int(frame_number) == int(last_frame_number):
            return None, last_frame_number
        frame_number_resolved = int(frame_number)
    tip_xyz: tuple[float, float, float] | None = None
    tip_quat: tuple[float, float, float, float] | None = None
    t_robot_tip = getattr(snapshot, "T_robot_tip", None)
    if t_robot_tip is not None:
        try:
            tip_xyz = (
                float(t_robot_tip[0][3]),
                float(t_robot_tip[1][3]),
                float(t_robot_tip[2][3]),
            )
        except (TypeError, IndexError, ValueError):
            tip_xyz = None
    tool_payload = (getattr(snapshot, "tools", {}) or {}).get(str(tool_id).upper())
    valid_flag = True
    if tip_xyz is None and tool_payload is not None:
        translation = getattr(tool_payload, "translation_mm", None)
        if translation is not None:
            try:
                tip_xyz = (
                    float(translation[0]),
                    float(translation[1]),
                    float(translation[2]),
                )
            except (TypeError, IndexError, ValueError):
                tip_xyz = None
    if tool_payload is not None:
        quat = getattr(tool_payload, "quaternion_wxyz", None)
        if quat is not None:
            try:
                tip_quat = (
                    float(quat[0]),
                    float(quat[1]),
                    float(quat[2]),
                    float(quat[3]),
                )
            except (TypeError, IndexError, ValueError):
                tip_quat = None
        valid_flag = bool(getattr(tool_payload, "valid", True))
    tracker_age_s = getattr(snapshot, "tracker_data_age_s", None)
    age_ms = None
    if tracker_age_s is not None:
        try:
            age_ms = float(tracker_age_s) * 1000.0
        except (TypeError, ValueError):
            age_ms = None
    record = TrackerRecord(
        frame_index=int(frame_index),
        monotonic_ns=int(monotonic_ns),
        host_time_s=float(host_time_s),
        frame_number=frame_number_resolved,
        frame_time_s=None,
        tip_xyz_mm=tip_xyz,
        tip_quat_wxyz=tip_quat,
        valid=bool(valid_flag and tip_xyz is not None),
        age_ms_at_observation=age_ms,
    )
    return record, frame_number_resolved if frame_number_resolved is not None else last_frame_number


def _make_servo_record(
    *,
    telemetry_by_id: dict[int, Any] | None,
    sample_index: int,
    monotonic_ns: int,
    host_time_s: float,
    servo_ids: list[int],
    failure_code: str | None,
    servo_service,
) -> ServoTelemetryRecord:
    positions: dict[int, int | None] = {}
    currents: dict[int, int | None] = {}
    voltages: dict[int, int | None] = {}
    temperatures: dict[int, int | None] = {}
    age_ms_by_servo: dict[int, float | None] = {}
    valid = bool(telemetry_by_id) and failure_code is None
    for servo_id in servo_ids:
        telemetry = (telemetry_by_id or {}).get(int(servo_id))
        if telemetry is None:
            positions[int(servo_id)] = None
            currents[int(servo_id)] = None
            voltages[int(servo_id)] = None
            temperatures[int(servo_id)] = None
            age_ms_by_servo[int(servo_id)] = None
            valid = False
            continue
        positions[int(servo_id)] = (
            int(telemetry.present_position)
            if getattr(telemetry, "present_position", None) is not None
            else None
        )
        currents[int(servo_id)] = (
            int(telemetry.present_current_ma)
            if getattr(telemetry, "present_current_ma", None) is not None
            else None
        )
        voltages[int(servo_id)] = (
            int(telemetry.present_voltage_mv)
            if getattr(telemetry, "present_voltage_mv", None) is not None
            else None
        )
        temperatures[int(servo_id)] = (
            int(telemetry.present_temperature_c)
            if getattr(telemetry, "present_temperature_c", None) is not None
            else None
        )
        age_s = None
        if servo_service is not None:
            try:
                age_s = servo_service.telemetry_age_s(telemetry)
            except Exception:
                age_s = None
        age_ms_by_servo[int(servo_id)] = (
            float(age_s) * 1000.0 if age_s is not None else None
        )
        if positions[int(servo_id)] is None:
            valid = False
    return ServoTelemetryRecord(
        sample_index=int(sample_index),
        monotonic_ns=int(monotonic_ns),
        host_time_s=float(host_time_s),
        positions_by_servo=positions,
        currents_by_servo=currents,
        voltages_by_servo=voltages,
        temperatures_by_servo=temperatures,
        valid=bool(valid),
        age_ms_by_servo=age_ms_by_servo,
        failure_code=failure_code,
    )


# ----------------------------------------------------------------------------
# Experiment
# ----------------------------------------------------------------------------


class DynamicModelingDatasetExperiment(BaseExperiment):
    """Continuous bounded random trajectory + synchronized dynamic dataset."""

    name = EXPERIMENT_NAME
    description = (
        "Continuously drive the robot through a bounded random tendon trajectory while "
        "logging synchronized servo command, servo telemetry, and tracker frames at "
        "approximately 20 Hz for dynamic modeling."
    )
    hardware_requirements = ExperimentHardwareRequirements(
        tracking_required=True,
        servo_required=True,
        mock_compatible=True,
    )

    def __init__(self, config: DynamicModelingDatasetConfig) -> None:
        super().__init__(config=config)
        self._tracking_started_here = False
        self._sync_samples: list[SynchronizedSample] = []
        self._commands: list[CommandRecord] = []
        self._trackers: list[TrackerRecord] = []
        self._servo_telemetry: list[ServoTelemetryRecord] = []
        self._servo_id_slots: list[int] = []
        self._neutral_ticks_by_servo: dict[int, int] = {}
        self._starting_position_by_servo: dict[int, int] = {}
        self._failure_code: str | None = None
        self._thesis_eligibility_reasons: list[str] = []
        self._thesis_eligible: bool = True

    @classmethod
    def from_dict(cls, payload: dict[str, Any] | None = None) -> "DynamicModelingDatasetExperiment":
        return cls(config=DynamicModelingDatasetConfig.from_dict(payload))

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def setup(self, session: ExperimentSession) -> None:
        tracking_service = session.context.tracking_service
        if tracking_service is not None and getattr(tracking_service, "_thread", None) is None:
            try:
                tracking_service.start()
                self._tracking_started_here = True
            except Exception as exc:
                LOG.debug("Tracking service start in setup raised: %s", exc)

    def precheck(self, session: ExperimentSession) -> None:
        config = self.config
        servo_service = session.context.servo_service
        tracking_service = session.context.tracking_service
        settings = session.context.settings
        if servo_service is None:
            raise RuntimeError(
                f"{self.name} requires a servo service for continuous command and telemetry."
            )
        if not bool(getattr(servo_service, "is_connected", False)) and not bool(config.dry_run):
            raise RuntimeError(
                f"{self.name} requires a connected servo service. Enable dry_run only for "
                "synthetic/debug runs (these are marked not_thesis_evidence automatically)."
            )
        if config.require_tracker and tracking_service is None:
            raise RuntimeError(
                f"{self.name} requires tracking_service when require_tracker is true."
            )
        if bool(settings.runtime.mock_mode) and not bool(config.dry_run):
            session.add_warning(
                "mock_mode is active; this run will be marked not_thesis_evidence automatically."
            )
        servo_ids = self._resolved_servo_ids(session)
        if not servo_ids:
            raise RuntimeError(
                f"{self.name} could not resolve servo IDs to command. Configure servo_ids in "
                "the experiment payload or select a robot mode with active segment servos."
            )
        self._servo_id_slots = list(servo_ids)
        neutral_map: dict[int, int] = {}
        try:
            neutral_map = dict(servo_service.load_neutral_setpoints())
        except Exception as exc:
            LOG.debug("Neutral setpoint load failed: %s", exc)
        for servo_id in self._servo_id_slots:
            if int(servo_id) in neutral_map:
                self._neutral_ticks_by_servo[int(servo_id)] = int(neutral_map[int(servo_id)])
        # Evaluate runtime tip policy (workflow=modeling_dataset). This sets
        # an early eligibility flag the executor refines after the run.
        if tracking_service is not None:
            snapshot = _safe_tracker_snapshot(tracking_service)
            try:
                evaluation = evaluate_runtime_tip_trust(
                    snapshot=snapshot,
                    workflow=WORKFLOW_MODELING_DATASET,
                    allow_lower_trust=bool(config.allow_lower_trust_runtime_tip),
                )
            except ValueError:
                evaluation = evaluate_runtime_tip_trust(
                    snapshot=snapshot,
                    allow_lower_trust=bool(config.allow_lower_trust_runtime_tip),
                )
            if not evaluation.thesis_trusted:
                self._thesis_eligibility_reasons.append(
                    "runtime_tip_not_thesis_trusted"
                )
                if not evaluation.allowed_for_workflow and not bool(config.allow_lower_trust_runtime_tip):
                    raise RuntimeError(
                        "Runtime tip mode is lower-trust for the modeling_dataset workflow. "
                        "Enable allow_lower_trust_runtime_tip to run anyway (the result will "
                        "be marked not_thesis_evidence)."
                    )

    def execute(self, session: ExperimentSession) -> None:  # noqa: C901 - lifecycle script
        config = self.config
        servo_service = session.context.servo_service
        tracking_service = session.context.tracking_service
        servo_ids = list(self._servo_id_slots)
        target_sample_rate_hz = float(config.target_sample_rate_hz)
        command_update_rate_hz = float(config.command_update_rate_hz)
        duration_s = float(config.duration_s)
        sync_period_s = 1.0 / target_sample_rate_hz
        command_period_s = 1.0 / command_update_rate_hz
        tracker_poll_period_s = float(config.tracker_poll_interval_s)
        servo_telemetry_period_s = float(config.servo_telemetry_interval_s)
        max_tracker_age_ms = float(config.max_tracker_age_ms)
        max_servo_age_ms = float(config.max_servo_age_ms)
        require_tracker = bool(config.require_tracker)
        run_start_ns = time.monotonic_ns()
        run_start_wall_s = time.time()
        deadline_ns = run_start_ns + int(round(duration_s * 1_000_000_000.0))
        if config.timeout_s and config.timeout_s > duration_s:
            deadline_ns = max(deadline_ns, run_start_ns + int(round(float(config.timeout_s) * 1_000_000_000.0)))

        if not self._neutral_ticks_by_servo:
            # No neutral file available — fall back to the live position so the
            # trajectory still operates relative to a known starting point.
            try:
                initial_telemetry = servo_service.read_live_telemetry(servo_ids)
            except Exception as exc:
                raise RuntimeError(
                    f"Initial servo telemetry read failed and no neutral setpoints are available: {exc}"
                ) from exc
            for servo_id in servo_ids:
                telemetry = initial_telemetry.get(int(servo_id))
                if telemetry is None or telemetry.present_position is None:
                    raise RuntimeError(
                        f"Servo {servo_id} present position is unavailable; cannot start dynamic trajectory."
                    )
                self._neutral_ticks_by_servo[int(servo_id)] = int(telemetry.present_position)

        # Capture the starting tick position for each servo so the trajectory
        # can drive bounded deltas relative to it.
        try:
            startup_telemetry = servo_service.read_live_telemetry(servo_ids)
        except Exception as exc:
            raise RuntimeError(
                f"Initial servo telemetry read failed before dynamic dataset run started: {exc}"
            ) from exc
        for servo_id in servo_ids:
            telemetry = startup_telemetry.get(int(servo_id))
            if telemetry is None or telemetry.present_position is None:
                raise RuntimeError(
                    f"Servo {servo_id} present position is unavailable at run start; cannot continue."
                )
            self._starting_position_by_servo[int(servo_id)] = int(telemetry.present_position)

        trajectory = BoundedRandomWalkTrajectory(
            servo_ids=tuple(int(servo_id) for servo_id in servo_ids),
            max_tick_delta_from_start=int(config.max_tick_delta_from_start),
            max_tick_delta_hard_cap=int(config.max_tick_delta_hard_cap),
            max_step_ticks_per_update=int(config.max_step_ticks_per_update),
            trajectory_smoothing=float(config.trajectory_smoothing),
            random_seed=int(config.random_seed),
        )

        next_command_ns = run_start_ns
        next_sample_ns = run_start_ns + int(round(sync_period_s * 1_000_000_000.0))
        next_tracker_poll_ns = run_start_ns
        next_servo_poll_ns = run_start_ns
        last_tracker_frame_number: int | None = None
        tracker_frame_index = 0
        servo_sample_index = 0
        command_id = 0
        elapsed_s = 0.0
        coast_mode = False

        owner_label = f"{self.name}:{int(config.random_seed)}"
        bus_owner_ctx = servo_service.exclusive_bus_operation(
            owner=owner_label,
            reason="dynamic modeling dataset continuous trajectory",
        )
        try:
            with bus_owner_ctx:
                # Seed an initial servo telemetry record so sync rows always
                # have a baseline to fall back on.
                self._servo_telemetry.append(
                    _make_servo_record(
                        telemetry_by_id=startup_telemetry,
                        sample_index=servo_sample_index,
                        monotonic_ns=run_start_ns,
                        host_time_s=run_start_wall_s,
                        servo_ids=servo_ids,
                        failure_code=None,
                        servo_service=servo_service,
                    )
                )
                servo_sample_index += 1

                while True:
                    session.raise_if_stop_requested()
                    now_ns = time.monotonic_ns()
                    elapsed_s = max(0.0, float(now_ns - run_start_ns) / 1_000_000_000.0)
                    end_motion = now_ns >= deadline_ns

                    if not coast_mode and end_motion:
                        if config.return_to_start_at_end:
                            coast_mode = True
                            # Schedule one final command series that drives the
                            # trajectory back toward zero delta.
                            next_command_ns = now_ns
                        else:
                            break

                    if coast_mode:
                        deltas = trajectory.current_delta_ticks()
                        if all(int(value) == 0 for value in deltas):
                            break

                    if now_ns >= next_tracker_poll_ns:
                        snapshot = _safe_tracker_snapshot(tracking_service)
                        host_time_s = run_start_wall_s + float(now_ns - run_start_ns) / 1_000_000_000.0
                        record, last_tracker_frame_number = _extract_tracker_record(
                            snapshot=snapshot,
                            tool_id=config.tracker_tool_id,
                            frame_index=tracker_frame_index,
                            monotonic_ns=now_ns,
                            host_time_s=host_time_s,
                            last_frame_number=last_tracker_frame_number,
                        )
                        if record is not None:
                            self._trackers.append(record)
                            tracker_frame_index += 1
                        next_tracker_poll_ns = now_ns + int(round(tracker_poll_period_s * 1_000_000_000.0))

                    if now_ns >= next_servo_poll_ns:
                        host_time_s = run_start_wall_s + float(now_ns - run_start_ns) / 1_000_000_000.0
                        servo_failure_code: str | None = None
                        try:
                            telemetry_by_id = servo_service.read_live_telemetry(servo_ids)
                        except Exception as exc:
                            telemetry_by_id = None
                            servo_failure_code = f"telemetry_read_error:{exc}"
                            session.add_warning(servo_failure_code)
                        self._servo_telemetry.append(
                            _make_servo_record(
                                telemetry_by_id=telemetry_by_id,
                                sample_index=servo_sample_index,
                                monotonic_ns=now_ns,
                                host_time_s=host_time_s,
                                servo_ids=servo_ids,
                                failure_code=servo_failure_code,
                                servo_service=servo_service,
                            )
                        )
                        servo_sample_index += 1
                        if servo_failure_code is not None:
                            self._failure_code = "servo_telemetry_failure"
                            break
                        next_servo_poll_ns = now_ns + int(round(servo_telemetry_period_s * 1_000_000_000.0))

                    if now_ns >= next_command_ns:
                        step = (
                            trajectory.coast_to_zero()
                            if coast_mode
                            else trajectory.next_step(elapsed_s=elapsed_s)
                        )
                        cable_deltas_cm: list[float] = []
                        host_time_s = run_start_wall_s + float(now_ns - run_start_ns) / 1_000_000_000.0
                        cmd_failure: str | None = None
                        try:
                            cable_deltas_cm = self._command_step(
                                servo_service=servo_service,
                                step=step,
                                servo_ids=servo_ids,
                                config=config,
                            )
                        except Exception as exc:
                            cmd_failure = f"command_failure:{exc}"
                            session.add_error(cmd_failure)
                            self._failure_code = "command_safety_event"
                        goal_ticks = {
                            int(servo_id): self._starting_position_by_servo[int(servo_id)] + int(step.tick_deltas[index])
                            for index, servo_id in enumerate(servo_ids)
                        }
                        delta_ticks = {
                            int(servo_id): int(step.tick_deltas[index])
                            for index, servo_id in enumerate(servo_ids)
                        }
                        command = CommandRecord(
                            command_id=int(command_id),
                            monotonic_ns=int(now_ns),
                            host_time_s=float(host_time_s),
                            cable_deltas_cm=tuple(cable_deltas_cm)
                            if cable_deltas_cm
                            else tuple(0.0 for _ in servo_ids),
                            goal_ticks_by_servo=goal_ticks,
                            delta_ticks_by_servo=delta_ticks,
                            safety_status="ok" if cmd_failure is None else cmd_failure,
                        )
                        self._commands.append(command)
                        command_id += 1
                        if cmd_failure is not None:
                            break
                        next_command_ns = now_ns + int(round(command_period_s * 1_000_000_000.0))

                    if now_ns >= next_sample_ns:
                        sample_host_time_s = run_start_wall_s + float(now_ns - run_start_ns) / 1_000_000_000.0
                        sync = build_synchronized_sample(
                            sample_index=len(self._sync_samples),
                            sample_monotonic_ns=now_ns,
                            sample_host_time_s=sample_host_time_s,
                            commands=self._commands,
                            trackers=self._trackers,
                            servos=self._servo_telemetry,
                            max_tracker_age_ms=max_tracker_age_ms,
                            max_servo_age_ms=max_servo_age_ms,
                            require_tracker=require_tracker,
                        )
                        self._sync_samples.append(sync)
                        self._publish_sample(session, sync)
                        next_sample_ns = now_ns + int(round(sync_period_s * 1_000_000_000.0))

                    progress_total = max(1, int(round(duration_s * 1000.0)))
                    progress_current = min(progress_total, int(round(elapsed_s * 1000.0)))
                    session.update_progress(
                        progress_current,
                        progress_total,
                        {
                            "phase": "coast_to_start" if coast_mode else "dynamic",
                            "dynamic_samples": len(self._sync_samples),
                            "command_events": len(self._commands),
                            "tracker_frames": len(self._trackers),
                            "servo_samples": len(self._servo_telemetry),
                        },
                    )

                    if not coast_mode and end_motion and not config.return_to_start_at_end:
                        break
                    if elapsed_s > duration_s + 10.0:
                        # Defensive timeout if something stalls the loop.
                        self._failure_code = self._failure_code or "loop_timeout"
                        break
                    time.sleep(0.001)
        except Exception as exc:
            self._failure_code = self._failure_code or f"execute_failure:{exc}"
            raise
        finally:
            elapsed_s_final = max(0.0, float(time.monotonic_ns() - run_start_ns) / 1_000_000_000.0)
            transform_chain_summary = None
            if tracking_service is not None:
                snapshot = _safe_tracker_snapshot(tracking_service)
                if snapshot is not None:
                    transform_chain_summary = {
                        "selected_backend_name": str(getattr(snapshot, "selected_backend_name", "") or ""),
                        "backend_identity": str(getattr(snapshot, "backend_identity", "") or ""),
                        "runtime_tip_mode": str(getattr(snapshot, "runtime_tip_mode", "") or ""),
                        "runtime_tip_trust_level": str(getattr(snapshot, "runtime_tip_trust_level", "") or ""),
                        "registration_state": str(getattr(snapshot, "registration_state", "") or ""),
                        "tip_pose_status": str(getattr(snapshot, "tip_pose_status", "") or ""),
                    }
            self._record_run_metrics(
                session=session,
                elapsed_s=elapsed_s_final,
                transform_chain_summary=transform_chain_summary,
            )

    def finalize(self, session: ExperimentSession) -> None:
        if self._tracking_started_here:
            try:
                session.context.tracking_service.stop()
            except Exception as exc:
                LOG.debug("Tracking service stop failed: %s", exc)
            self._tracking_started_here = False

    def summarize(self, session: ExperimentSession) -> dict[str, Any]:
        return dict(session.metrics)

    def write_outputs(
        self,
        session: ExperimentSession,
        paths: ExperimentDatasetPaths,
        summary: ExperimentSummary,
    ) -> None:
        output_dir = Path(paths.output_dir)
        servo_id_slots = list(self._servo_id_slots)
        # Compact CSV.GZ exports.
        write_sync_csv_gz(
            output_dir / "dynamic_samples.csv.gz",
            self._sync_samples,
            servo_id_slots=servo_id_slots,
        )
        write_commands_csv_gz(
            output_dir / "commands.csv.gz",
            self._commands,
            servo_id_slots=servo_id_slots,
        )
        write_tracker_csv_gz(output_dir / "tracker_frames.csv.gz", self._trackers)
        write_servo_csv_gz(
            output_dir / "servo_telemetry.csv.gz",
            self._servo_telemetry,
            servo_id_slots=servo_id_slots,
        )

        summary_payload = dict(session.metrics)
        summary_payload["sample_validity_threshold_ms"] = {
            "max_tracker_age_ms": float(self.config.max_tracker_age_ms),
            "max_servo_age_ms": float(self.config.max_servo_age_ms),
        }
        summary_payload["servo_id_slots"] = list(servo_id_slots)
        summary_payload["starting_position_by_servo"] = {
            str(key): int(value) for key, value in self._starting_position_by_servo.items()
        }
        summary_payload["neutral_position_by_servo"] = {
            str(key): int(value) for key, value in self._neutral_ticks_by_servo.items()
        }
        # Do not overwrite the runner-managed canonical summary.json. The
        # canonical file carries trust info, stage status, and the
        # experiment_metrics blob (which already includes the dynamic
        # metrics). Write the dynamic-experiment-specific rollup alongside it
        # so consumers have a single richer view without losing the
        # canonical structure.
        (output_dir / "dynamic_modeling_summary.json").write_text(
            json.dumps(summary_payload, indent=2, sort_keys=True),
            encoding="utf-8",
        )

        text_lines = _build_text_summary(
            metrics=summary_payload,
            config=self.config,
            sync_samples=self._sync_samples,
        )
        (output_dir / "dynamic_modeling_summary.txt").write_text(
            "\n".join(text_lines) + "\n", encoding="utf-8"
        )

        # Config snapshot (in addition to the runner's config_snapshot.yaml).
        (output_dir / "config_snapshot.yaml").write_text(
            yaml.safe_dump(asdict(self.config), sort_keys=False), encoding="utf-8"
        )

        manifest = {
            "experiment_name": self.name,
            "schema_version": "dynamic_modeling_dataset_v1",
            "run_id": session.metadata.run_id,
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "thesis_eligible": bool(self._thesis_eligible),
            "thesis_eligibility_reasons": list(self._thesis_eligibility_reasons),
            "outputs": {
                "sync_samples": "dynamic_samples.csv.gz",
                "commands": "commands.csv.gz",
                "tracker_frames": "tracker_frames.csv.gz",
                "servo_telemetry": "servo_telemetry.csv.gz",
                "summary_json": "summary.json",
                "dynamic_summary_json": "dynamic_modeling_summary.json",
                "summary_text": "dynamic_modeling_summary.txt",
                "config_snapshot": "config_snapshot.yaml",
            },
            "columns": {
                "sync_samples": list(SYNC_COLUMNS),
                "commands": list(COMMAND_COLUMNS),
                "tracker_frames": list(TRACKER_COLUMNS),
                "servo_telemetry": list(SERVO_COLUMNS),
            },
        }

        figures = _render_default_figures(
            output_dir=output_dir,
            sync_samples=self._sync_samples,
            commands=self._commands,
            trackers=self._trackers,
            servo_telemetry=self._servo_telemetry,
            servo_id_slots=servo_id_slots,
            sample_validity_threshold_ms=float(self.config.max_servo_age_ms),
        )
        manifest["figures"] = figures
        (output_dir / "manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _resolved_servo_ids(self, session: ExperimentSession) -> list[int]:
        configured = list(self.config.servo_ids)
        if configured:
            return [int(value) for value in configured]
        settings = session.context.settings
        try:
            operating_context = settings.robot.operating_context()
            commanded = [int(value) for value in operating_context.commanded_servo_ids or []]
        except Exception:
            commanded = []
        if commanded:
            return commanded
        return [int(value) for value in (settings.robot.servo_ids or [])]

    def _command_step(
        self,
        *,
        servo_service,
        step: TrajectoryStep,
        servo_ids: list[int],
        config: DynamicModelingDatasetConfig,
    ) -> list[float]:
        """Convert the bounded random step to displacement and issue a command.

        Returns the per-cable displacement in centimeters used for the command.
        """
        # Convert per-servo tick deltas to per-cable displacement in
        # centimeters using the existing mapper. We avoid changing transform
        # math: we feed the displacement directly to ``command_displacement``
        # so the standard antagonistic-pair projection, safety guard, and
        # post-motion telemetry checks all run.
        mapper = getattr(servo_service, "mapper", None)
        if mapper is None:
            raise RuntimeError("Servo service has no displacement mapper available.")
        displacements_cm: list[float] = []
        for index, _ in enumerate(servo_ids):
            tick_delta = int(step.tick_deltas[index])
            displacement_mm = mapper.ticks_to_displacement_mm(tick_delta)
            displacements_cm.append(float(displacement_mm) / 10.0)
        if config.dry_run:
            return displacements_cm
        neutral_ticks = [self._neutral_ticks_by_servo.get(int(servo_id), 0) for servo_id in servo_ids]
        servo_service.command_displacement(
            displacements_cm,
            neutral_ticks,
            list(servo_ids),
            telemetry_retry_count=0,
            allow_recovered_packet_errors=False,
            skip_post_command_telemetry=True,
        )
        return displacements_cm

    def _publish_sample(self, session: ExperimentSession, sync: SynchronizedSample) -> None:
        """Forward one synchronized row as an ``ExperimentTimeseriesSample``.

        We publish a lightweight summary so the canonical samples.jsonl stays
        small even on long runs (the heavy data lives in CSV.GZ). Each entry
        carries the synchronized timestamps and per-stream identifiers so the
        canonical visualization pipeline still works.
        """
        extra: dict[str, Any] = {
            "record_kind": "dynamic_modeling_sample",
            "sample_valid": bool(sync.sample_valid),
            "failure_code": sync.failure_code or "",
            "command_to_tracker_dt_ms": sync.command_to_tracker_dt_ms,
            "servo_to_tracker_dt_ms": sync.servo_to_tracker_dt_ms,
        }
        if sync.command is not None:
            extra["command_id"] = int(sync.command.command_id)
            extra["command_cable_deltas_cm"] = [
                float(value) for value in sync.command.cable_deltas_cm
            ]
            extra["command_delta_ticks"] = {
                str(key): int(value) for key, value in sync.command.delta_ticks_by_servo.items()
            }
        if sync.tracker is not None:
            extra["tracker_frame_number"] = sync.tracker.frame_number
            extra["tracker_age_ms_at_observation"] = sync.tracker.age_ms_at_observation
        if sync.servo is not None:
            extra["servo_positions"] = {
                str(key): value for key, value in sync.servo.positions_by_servo.items()
            }
            extra["servo_currents_ma"] = {
                str(key): value for key, value in sync.servo.currents_by_servo.items()
            }
        if bool(self.config.dry_run):
            extra["dry_run"] = True
            extra["capture_mode"] = "synthetic_dry_run"
        status_flags = ["dynamic_modeling_sample"]
        if not sync.sample_valid:
            status_flags.append("dynamic_sample_invalid")
        if bool(self.config.dry_run):
            status_flags.extend(["dry_run", "synthetic_capture"])
        session.add_sample(
            ExperimentTimeseriesSample(
                monotonic_time_s=float(sync.host_time_s),
                wall_time_utc=("dry_run" if bool(self.config.dry_run) else _utc_iso()),
                phase="dynamic_capture",
                step_index=int(sync.sample_index),
                sample_index=int(sync.sample_index),
                tracker_frame_id=(
                    int(sync.tracker.frame_number)
                    if sync.tracker is not None and sync.tracker.frame_number is not None
                    else None
                ),
                status_flags=status_flags,
                backend_health=(
                    {"capture_mode": "synthetic_dry_run"}
                    if bool(self.config.dry_run)
                    else {"capture_mode": "live_hardware"}
                ),
                extra=extra,
            )
        )

    def _record_run_metrics(
        self,
        *,
        session: ExperimentSession,
        elapsed_s: float,
        transform_chain_summary: dict[str, Any] | None,
    ) -> None:
        # Re-evaluate thesis eligibility now that we know runtime state.
        eligibility_reasons = list(self._thesis_eligibility_reasons)
        if bool(self.config.dry_run):
            eligibility_reasons.append("dry_run")
        if bool(session.context.settings.runtime.mock_mode):
            eligibility_reasons.append("mock_mode")
        if self._failure_code:
            eligibility_reasons.append(f"failure:{self._failure_code}")
        self._thesis_eligible = not eligibility_reasons
        metrics = summarize_dynamic_run(
            config=self.config,
            elapsed_s=float(elapsed_s),
            sync_samples=self._sync_samples,
            commands=self._commands,
            trackers=self._trackers,
            servo_telemetry=self._servo_telemetry,
            failure_code=self._failure_code,
            thesis_eligible=bool(self._thesis_eligible),
            eligibility_reasons=eligibility_reasons,
            transform_chain_summary=transform_chain_summary,
        )
        session.metrics.update(metrics)
        if not self._thesis_eligible:
            session.metrics.setdefault("debug_or_synthetic", True)
            session.metrics["not_thesis_evidence"] = True
            session.metrics["not_thesis_evidence_reasons"] = sorted(
                set(session.metrics.get("not_thesis_evidence_reasons", []) or []) | set(eligibility_reasons)
            )
        else:
            session.metrics.setdefault("debug_or_synthetic", False)
        if self._failure_code:
            session.metrics["failure_reason"] = str(self._failure_code)
            session.metrics["stop_reason"] = str(self._failure_code)
        # Force the runner status to "failed" if we hit a safety event.
        if self._failure_code:
            requirements = dict(session.metrics.get("summary_requirements", {}) or {})
            requirements["force_status"] = "failed"
            session.metrics["summary_requirements"] = requirements


def _utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _build_text_summary(
    *,
    metrics: dict[str, Any],
    config: DynamicModelingDatasetConfig,
    sync_samples: list[SynchronizedSample],
) -> list[str]:
    lines = [
        "Dynamic Modeling Dataset Summary",
        "================================",
        f"Duration requested:   {metrics.get('duration_requested_s', 0.0):.2f} s",
        f"Duration completed:   {metrics.get('duration_completed_s', 0.0):.2f} s",
        f"Target sample rate:   {metrics.get('target_sample_rate_hz', 0.0):.2f} Hz",
        f"Achieved sample rate: {metrics.get('achieved_sample_rate_hz', 0.0):.2f} Hz",
        f"Sync sample rows:     {metrics.get('dynamic_sample_row_count', 0)}",
        f"Valid sample rows:    {metrics.get('valid_sample_count', 0)} ({metrics.get('valid_sample_ratio', 0.0):.2%})",
        f"Command events:       {metrics.get('command_event_count', 0)}",
        f"Tracker frames:       {metrics.get('tracker_unique_frame_count', 0)} unique "
        f"({metrics.get('tracker_unique_frame_rate_hz', 0.0):.1f} Hz)",
        f"Servo telemetry:      {metrics.get('servo_telemetry_count', 0)} samples "
        f"({metrics.get('servo_telemetry_rate_hz', 0.0):.1f} Hz)",
        f"Tracker age p50/p95:  {_fmt(metrics.get('tracker_age_ms_median'))} / {_fmt(metrics.get('tracker_age_ms_p95'))} ms",
        f"Servo age p50/p95:    {_fmt(metrics.get('servo_age_ms_median'))} / {_fmt(metrics.get('servo_age_ms_p95'))} ms",
        f"Servo->tracker p50/p95: {_fmt(metrics.get('servo_to_tracker_offset_ms_median'))} / "
        f"{_fmt(metrics.get('servo_to_tracker_offset_ms_p95'))} ms",
        f"Max commanded tick delta: {metrics.get('max_commanded_tick_delta', 0)}",
        f"Max measured tick delta:  {metrics.get('max_measured_tick_delta', 0)}",
        f"Random seed: {metrics.get('random_seed', 0)}",
        f"Pair-axis convention: {metrics.get('pair_axis_convention', '')}",
        f"Thesis eligible: {bool(metrics.get('thesis_eligible'))}",
    ]
    failure_code = metrics.get("failure_code")
    if failure_code:
        lines.append(f"Failure code: {failure_code}")
    reasons = list(metrics.get("thesis_eligibility_reasons", []) or [])
    if reasons:
        lines.append("Thesis eligibility reasons:")
        for reason in reasons:
            lines.append(f"  - {reason}")
    lines.append("")
    lines.append(
        "This run is the dynamic-modeling continuous-trajectory dataset; it is independent of "
        "the quasi-static collect_pose_command_dataset workflow."
    )
    _ = config, sync_samples
    return lines


# Public registration helper, kept here so callers (builtins.register_builtin_experiments)
# can opt in without importing the experiment class directly.


def register_dynamic_modeling_dataset(registry) -> None:
    """Register the dynamic modeling dataset experiment."""
    registry.register(
        name=DynamicModelingDatasetExperiment.name,
        title="Dynamic Modeling Dataset",
        description=DynamicModelingDatasetExperiment.description,
        category="dataset",
        tags=["Dynamic", "Modeling", "Dataset", "Tracking", "Servo"],
        default_config_path="config/experiment_dynamic_modeling_dataset.example.yaml",
        factory=DynamicModelingDatasetExperiment.from_dict,
    )

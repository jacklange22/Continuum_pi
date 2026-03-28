"""Canonical command schedule models and generators."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from itertools import product
import json
import random
from typing import Iterable

from continuum_robot.experiments.experiment_models import ExperimentPoint


@dataclass
class CommandScheduleConfig:
    """Config for generated command schedules."""

    kind: str = "sweep"
    dimensions: int = 4
    amplitude_cm: float = 0.2
    steps_per_axis: int = 5
    repeats: int = 1
    settle_time_s: float | None = None
    seed: int = 0
    babble_count: int = 20
    trajectory_points_cm: list[list[float]] = field(default_factory=list)
    bounds_cm: list[list[float]] | None = None

    @classmethod
    def from_dict(cls, payload: dict | None) -> "CommandScheduleConfig":
        """Construct schedule config from a dictionary payload."""
        payload = dict(payload or {})
        return cls(
            kind=str(payload.get("kind", "sweep")),
            dimensions=int(payload.get("dimensions", 4)),
            amplitude_cm=float(payload.get("amplitude_cm", 0.2)),
            steps_per_axis=int(payload.get("steps_per_axis", 5)),
            repeats=int(payload.get("repeats", 1)),
            settle_time_s=(
                float(payload["settle_time_s"])
                if payload.get("settle_time_s") not in (None, "")
                else None
            ),
            seed=int(payload.get("seed", 0)),
            babble_count=int(payload.get("babble_count", 20)),
            trajectory_points_cm=[
                [float(value) for value in point]
                for point in payload.get("trajectory_points_cm", []) or []
            ],
            bounds_cm=[
                [float(value) for value in pair]
                for pair in payload.get("bounds_cm", []) or []
            ]
            or None,
        )

    def to_dict(self) -> dict:
        """Return the config as a JSON-serializable dictionary."""
        return asdict(self)


def generate_command_schedule(config: CommandScheduleConfig) -> list[ExperimentPoint]:
    """Generate a deterministic command schedule from config."""
    kind = str(config.kind).strip().lower()
    if config.dimensions <= 0:
        raise ValueError("Schedule dimensions must be positive")
    if config.repeats <= 0:
        raise ValueError("Schedule repeats must be positive")
    bounds = _bounds_for(config)
    if kind == "sweep":
        raw_points = _generate_sweep_points(bounds=bounds, steps_per_axis=config.steps_per_axis)
    elif kind == "grid":
        raw_points = _generate_grid_points(bounds=bounds, steps_per_axis=config.steps_per_axis)
    elif kind == "trajectory":
        raw_points = _generate_trajectory_points(config)
    elif kind == "babble":
        raw_points = _generate_babble_points(config=config, bounds=bounds)
    else:
        raise ValueError(f"Unsupported command schedule kind: {config.kind!r}")
    points: list[ExperimentPoint] = []
    for index, point in enumerate(raw_points):
        points.append(
            ExperimentPoint(
                index=index,
                tendon_displacement_cm=[float(value) for value in point],
                settle_time_s=config.settle_time_s,
                repeat=config.repeats,
                label=f"{kind}_{index:04d}",
            )
        )
    return points


def command_schedule_checksum(points: Iterable[ExperimentPoint]) -> str:
    """Return a stable JSON checksum-like string for one generated schedule."""
    payload = [
        {
            "index": point.index,
            "dl_cm": [round(float(value), 8) for value in point.tendon_displacement_cm],
            "repeat": point.repeat,
            "settle_time_s": point.settle_time_s,
            "label": point.label,
        }
        for point in points
    ]
    return json.dumps(payload, sort_keys=True, separators=(",", ":"))


def _bounds_for(config: CommandScheduleConfig) -> list[tuple[float, float]]:
    if config.bounds_cm is not None:
        if len(config.bounds_cm) != config.dimensions:
            raise ValueError("bounds_cm length must match schedule dimensions")
        bounds: list[tuple[float, float]] = []
        for pair in config.bounds_cm:
            if len(pair) != 2:
                raise ValueError("Each bounds_cm entry must contain [min, max]")
            low, high = float(pair[0]), float(pair[1])
            if low > high:
                raise ValueError("Schedule bounds must be ordered as [min, max]")
            bounds.append((low, high))
        return bounds
    amplitude = abs(float(config.amplitude_cm))
    return [(-amplitude, amplitude) for _ in range(config.dimensions)]


def _axis_values(low: float, high: float, steps_per_axis: int) -> list[float]:
    if steps_per_axis <= 0:
        raise ValueError("steps_per_axis must be positive")
    if steps_per_axis == 1:
        return [float(low)]
    return [
        float(low + ((high - low) * step_index / float(steps_per_axis - 1)))
        for step_index in range(steps_per_axis)
    ]


def _generate_sweep_points(*, bounds: list[tuple[float, float]], steps_per_axis: int) -> list[list[float]]:
    dimensions = len(bounds)
    points: list[list[float]] = []
    for dimension_index, (low, high) in enumerate(bounds):
        for value in _axis_values(low, high, steps_per_axis):
            point = [0.0] * dimensions
            point[dimension_index] = float(value)
            points.append(point)
    return points


def _generate_grid_points(*, bounds: list[tuple[float, float]], steps_per_axis: int) -> list[list[float]]:
    axis_values = [_axis_values(low, high, steps_per_axis) for low, high in bounds]
    return [[float(value) for value in point] for point in product(*axis_values)]


def _generate_trajectory_points(config: CommandScheduleConfig) -> list[list[float]]:
    if not config.trajectory_points_cm:
        raise ValueError("trajectory_points_cm is required for trajectory schedules")
    points: list[list[float]] = []
    for point in config.trajectory_points_cm:
        if len(point) != config.dimensions:
            raise ValueError("Each trajectory point must match schedule dimensions")
        points.append([float(value) for value in point])
    return points


def _generate_babble_points(
    *,
    config: CommandScheduleConfig,
    bounds: list[tuple[float, float]],
) -> list[list[float]]:
    if config.babble_count <= 0:
        raise ValueError("babble_count must be positive")
    rng = random.Random(config.seed)
    points: list[list[float]] = []
    for _ in range(config.babble_count):
        point = [float(rng.uniform(low, high)) for low, high in bounds]
        points.append(point)
    return points


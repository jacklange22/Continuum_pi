"""Experiment input models."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class ExperimentPoint:
    """One command point in an experiment file.

    Fields:
    - index: point index in sequence
    - tendon_displacement_cm: displacement vector
    - settle_time_s: optional per-point settle override
    - repeat: optional number of repeated acquisitions
    """

    index: int
    tendon_displacement_cm: list[float]
    settle_time_s: float | None = None
    repeat: int = 1

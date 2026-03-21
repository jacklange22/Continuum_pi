"""Experiment CSV loader with explicit supported format."""

from __future__ import annotations

import csv
from pathlib import Path

from continuum_robot.experiments.experiment_models import ExperimentPoint


class ExperimentLoader:
    """Load experiment points from CSV files.

    Required columns:
    - index
    - one or more displacement columns prefixed with dl_

    Optional columns:
    - settle_time_s
    - repeat
    """

    def load_csv(self, path: Path) -> list[ExperimentPoint]:
        rows: list[ExperimentPoint] = []
        with path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            if reader.fieldnames is None:
                raise ValueError("CSV header is required")
            dl_cols = [c for c in reader.fieldnames if c.startswith("dl_")]
            if not dl_cols:
                raise ValueError("At least one dl_* column is required")

            for raw in reader:
                index = int(raw["index"])
                disp = [float(raw[c]) for c in dl_cols]
                settle = raw.get("settle_time_s")
                repeat = raw.get("repeat")
                rows.append(
                    ExperimentPoint(
                        index=index,
                        tendon_displacement_cm=disp,
                        settle_time_s=float(settle) if settle not in (None, "") else None,
                        repeat=int(repeat) if repeat not in (None, "") else 1,
                    )
                )
        return rows

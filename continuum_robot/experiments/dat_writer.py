"""Legacy `.dat` writer retained only for compatibility.

Prefer the canonical dataset writer in `continuum_robot.experiments.dataset_io`.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path


class DatRunWriter:
    """Write a repeatability-style `.dat` output file."""

    def __init__(self, output_dir: Path) -> None:
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def write_run(
        self,
        num_cables: int,
        rows: list[dict],
        filename_stem: str | None = None,
    ) -> Path:
        """Write one .dat file and return its path."""
        now = datetime.now()
        stem = filename_stem or f"data_{now:%Y_%m_%d_%H_%M_%S}"
        path = self.output_dir / f"{stem}.dat"

        with path.open("w", encoding="utf-8") as handle:
            handle.write(f"DATE: {now.year}-{now.month}-{now.day}\n")
            handle.write(f"TIME: {now.hour:02d}-{now.minute:02d}-{now.second:02d}\n")
            handle.write(f"NUM_CABLES: {num_cables}\n")
            handle.write("num_coils: 1\n")
            handle.write(f"NUM_MEASUREMENTS: {len(rows)}\n")
            handle.write("---\n")
            for row in rows:
                handle.write(self._row_to_line(row) + "\n")
        return path

    @staticmethod
    def _row_to_line(row: dict) -> str:
        """Serialize one row into CSV-like .dat body line."""
        values = [
            row.get("timestamp_utc", ""),
            row.get("index", 0),
            row.get("repeat_index", 0),
            *row.get("commanded_displacement_cm", []),
            *row.get("commanded_goal_ticks", []),
            *row.get("servo_position_ticks", []),
            *row.get("servo_current_ma", []),
            *row.get("servo_voltage_mv", []),
            *row.get("tool_0A_translation_mm", []),
            *row.get("tool_0B_translation_mm", []),
            *row.get("tip_position_xyz", []),
            *row.get("tip_tangent_xyz", []),
        ]
        return ",".join(str(v) for v in values)

"""Run output writer for one .dat file per experiment run."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path


class DatRunWriter:
    """Write a repeatability-style .dat output file."""

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
            row.get("index", 0),
            *row.get("commanded_displacement_cm", []),
            *row.get("tip_position_xyz", []),
            *row.get("tip_tangent_xyz", []),
        ]
        return ",".join(str(v) for v in values)

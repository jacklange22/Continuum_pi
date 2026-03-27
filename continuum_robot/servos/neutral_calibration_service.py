"""Manual neutral calibration workflow.

This service is intentionally separate from startup pretension validation.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
import json


class NeutralCalibrationService:
    """Stores and retrieves manually calibrated neutral setpoints."""

    def __init__(self, path: Path | None = None) -> None:
        project_root = Path(__file__).resolve().parents[2]
        self.path = path or (project_root / "data" / "calibrations" / "neutral_setpoints.json")
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def save_neutral_setpoints(self, setpoints_by_id: dict[int, int]) -> None:
        if self.path.exists():
            stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
            archive_path = self.path.with_name(f"{self.path.stem}_{stamp}{self.path.suffix}")
            archive_path.write_text(self.path.read_text(encoding="utf-8"), encoding="utf-8")
        payload = {str(k): v for k, v in setpoints_by_id.items()}
        self.path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    def load_neutral_setpoints(self) -> dict[int, int]:
        if not self.path.exists():
            return {}
        payload = json.loads(self.path.read_text(encoding="utf-8"))
        return {int(k): int(v) for k, v in payload.items()}

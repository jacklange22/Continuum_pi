"""Registration persistence with full record payload."""

from __future__ import annotations

from dataclasses import dataclass, asdict
from datetime import datetime, timezone
import json
from pathlib import Path


@dataclass
class RegistrationRecord:
    """Persisted registration output.

    Includes transforms plus capture and validation details required by spec.
    """

    timestamp_utc: str
    landmark_labels: list[str]
    raw_captured_landmarks_robot_xyz: dict[str, list[list[float]]]
    averaged_landmarks_robot_xyz: dict[str, list[float]]
    residuals_robot_xyz_mm: dict[str, list[float]]
    fre_mm: float
    T_robot_aurora: list[list[float]]
    T_coil_tip: list[list[float]]
    config_used: dict


class RegistrationRepository:
    """Save/load registration records."""

    def __init__(self, root_dir: Path | None = None) -> None:
        self.root_dir = root_dir or Path("/Users/jacklange/Continuum/pi_code/data/registrations")
        self.root_dir.mkdir(parents=True, exist_ok=True)

    def save_record(self, record: RegistrationRecord) -> Path:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        path = self.root_dir / f"registration_{stamp}.json"
        path.write_text(json.dumps(asdict(record), indent=2), encoding="utf-8")

        latest = self.root_dir / "latest_registration.json"
        latest.write_text(json.dumps(asdict(record), indent=2), encoding="utf-8")
        return path

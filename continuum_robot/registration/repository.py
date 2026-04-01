"""Registration persistence with full record payload."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any
import numpy as np


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
    T_aurora_2_tip: list[list[float]] | None = None
    measurement_tool_id: str | None = None
    coil_tool_id: str | None = None
    raw_measurement_tool_samples_by_label: dict[str, list[dict[str, Any]]] = field(default_factory=dict)
    raw_coil_samples_by_label: dict[str, list[dict[str, Any]]] = field(default_factory=dict)
    truth_points_in_sw_by_label: dict[str, list[float]] = field(default_factory=dict)
    group_by_label: dict[str, str] = field(default_factory=dict)
    validation_metrics: dict[str, Any] = field(default_factory=dict)
    capture_tip_provenance: dict[str, Any] = field(default_factory=dict)
    live_pose_tip_transform: dict[str, Any] = field(default_factory=dict)


class RegistrationRepository:
    """Save/load registration records."""

    def __init__(self, root_dir: Path | None = None) -> None:
        project_root = Path(__file__).resolve().parents[2]
        self.root_dir = root_dir or (project_root / "data" / "registrations")
        self.root_dir.mkdir(parents=True, exist_ok=True)

    def save_record(self, record: RegistrationRecord) -> Path:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        path = self.root_dir / f"registration_{stamp}.json"
        payload = self._to_payload(record)
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

        latest = self.root_dir / "latest_registration.json"
        latest.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return path

    @staticmethod
    def _to_payload(record: RegistrationRecord) -> dict:
        payload = asdict(record)
        T_robot_aurora = np.asarray(record.T_robot_aurora, dtype=float)
        if T_robot_aurora.shape == (4, 4):
            payload["T_aurora_2_model"] = T_robot_aurora.tolist()
        T_coil_tip = np.asarray(record.T_coil_tip, dtype=float)
        if T_coil_tip.shape == (4, 4):
            # Legacy compatibility for reference-style tooling.
            payload["T_tip_2_coil"] = np.linalg.inv(T_coil_tip).tolist()
        if record.T_aurora_2_tip is not None:
            T_aurora_2_tip = np.asarray(record.T_aurora_2_tip, dtype=float)
            if T_aurora_2_tip.shape == (4, 4):
                payload["T_aurora_2_tip"] = T_aurora_2_tip.tolist()
        # The existing field names are historical; add precise aliases as well.
        payload["raw_captured_landmarks_aurora_xyz"] = payload["raw_captured_landmarks_robot_xyz"]
        payload["averaged_landmarks_aurora_xyz"] = payload["averaged_landmarks_robot_xyz"]
        return payload

"""Persistence for runtime 0A hat-based tip calibration artifacts."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any

import numpy as np


@dataclass
class RuntimeTipCalibrationRecord:
    """Saved runtime tip calibration artifact."""

    timestamp_utc: str
    calibration_kind: str
    measurement_tool_id: str
    coil_tool_id: str
    setup_id: str | None
    truth_points_in_sw_by_label: dict[str, list[float]]
    truth_points_in_tip_by_label: dict[str, list[float]]
    raw_captured_hat_points_aurora_xyz_by_label: dict[str, list[list[float]]]
    averaged_hat_points_aurora_xyz_by_label: dict[str, list[float]]
    raw_coil_samples: list[dict[str, Any]]
    residuals_tip_xyz_mm_by_label: dict[str, list[float]]
    fit_rmse_mm: float
    T_tip_aurora: list[list[float]]
    T_aurora_tip: list[list[float]]
    T_aurora_coil_avg: list[list[float]]
    T_coil_tip: list[list[float]]
    config_used: dict[str, Any]
    hat_geometry: dict[str, Any] = field(default_factory=dict)
    validation_metrics: dict[str, Any] = field(default_factory=dict)


class RuntimeTipCalibrationRepository:
    """Save/load runtime tip calibration artifacts."""

    def __init__(self, latest_path: Path | None = None) -> None:
        project_root = Path(__file__).resolve().parents[2]
        self.latest_path = latest_path or (
            project_root / "data" / "calibrations" / "runtime_tip" / "latest_runtime_tip_calibration.json"
        )
        self.root_dir = self.latest_path.parent
        self.root_dir.mkdir(parents=True, exist_ok=True)

    def save_record(self, record: RuntimeTipCalibrationRecord) -> Path:
        stamp = _timestamp_stamp(record.timestamp_utc)
        path = self.root_dir / f"runtime_tip_calibration_{stamp}.json"
        payload = self._to_payload(record)
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        self.latest_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return path

    def load_latest_payload(self) -> dict[str, Any] | None:
        if not self.latest_path.exists():
            return None
        return self.load_payload(self.latest_path)

    def list_saved_records(self, *, limit: int | None = None) -> list[Path]:
        paths = sorted(self.root_dir.glob("runtime_tip_calibration_*.json"), reverse=True)
        if limit is None:
            return paths
        return paths[: max(0, int(limit))]

    @staticmethod
    def load_payload(path: Path) -> dict[str, Any]:
        return json.loads(path.read_text(encoding="utf-8"))

    @staticmethod
    def _to_payload(record: RuntimeTipCalibrationRecord) -> dict[str, Any]:
        payload = asdict(record)
        T_coil_tip = np.asarray(record.T_coil_tip, dtype=float)
        if T_coil_tip.shape == (4, 4):
            payload["T_tip_2_coil"] = np.linalg.inv(T_coil_tip).tolist()
        return payload


def _timestamp_stamp(timestamp_utc: str | None) -> str:
    if timestamp_utc:
        try:
            parsed = datetime.fromisoformat(str(timestamp_utc).replace("Z", "+00:00"))
        except ValueError:
            cleaned = "".join(character for character in str(timestamp_utc) if character.isdigit())
            if cleaned:
                return cleaned
        else:
            return parsed.astimezone(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")

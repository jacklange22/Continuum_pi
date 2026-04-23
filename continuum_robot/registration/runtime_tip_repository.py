"""Persistence for runtime 0A hat-based tip calibration artifacts."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
from pathlib import Path
from typing import Any

import numpy as np

from continuum_robot.experiments.dataset_io import canonical_timestamped_path


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
            project_root / "data" / "runtime_tip_calibration" / "latest_runtime_tip_calibration.json"
        )
        self.root_dir = self.latest_path.parent
        self.root_dir.mkdir(parents=True, exist_ok=True)
        self.quick_latest_path = self.root_dir / "latest_quick_4_point_runtime_tip.json"

    def save_record(
        self,
        record: RuntimeTipCalibrationRecord,
        *,
        mark_as_latest: bool = True,
        alias_path: Path | None = None,
    ) -> Path:
        prefix = (
            "runtime_tip_quick_4_point"
            if str(record.calibration_kind).strip().lower() == "runtime_tip_calibration_quick_4_point"
            else "runtime_tip_calibration"
        )
        path = canonical_timestamped_path(
            self.root_dir,
            prefix,
            timestamp_utc=record.timestamp_utc,
            extension=".json",
        )
        payload = self._to_payload(record)
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        if alias_path is not None:
            alias_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        elif mark_as_latest:
            self.latest_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return path

    def load_latest_payload(self) -> dict[str, Any] | None:
        if not self.latest_path.exists():
            return None
        return self.load_payload(self.latest_path)

    def load_latest_quick_payload(self) -> dict[str, Any] | None:
        if not self.quick_latest_path.exists():
            return None
        return self.load_payload(self.quick_latest_path)

    def list_saved_records(self, *, limit: int | None = None) -> list[Path]:
        paths = sorted(
            (
                path
                for path in self.root_dir.glob("*.json")
                if path.name not in {"latest_runtime_tip_calibration.json", "latest_quick_4_point_runtime_tip.json"}
                and _is_runtime_tip_record_name(path.name)
            ),
            reverse=True,
        )
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
def _is_runtime_tip_record_name(name: str) -> bool:
    if name.startswith("runtime_tip_calibration_") and name.endswith(".json"):
        return True
    if name.startswith("runtime_tip_quick_4_point_") and name.endswith(".json"):
        return True
    return name.endswith("_runtime_tip_calibration.json") or name.endswith("_runtime_tip_quick_4_point.json")

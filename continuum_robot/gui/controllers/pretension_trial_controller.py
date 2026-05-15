"""One-click pretension-trial controller used by the Servos tab.

This is the replacement for the deleted Pretension tab. It owns:

- The "Run Pretension Trial" button flow: load the saved
  ``config/experiment_pretension_validation.example.yaml`` (or the user-
  edited version), launch the ``pretension_validation`` experiment with
  the segment's expected servo IDs, return the run output dir.
- The "Record Manual Baseline" flow: read the current servo + tracker
  state once, append it to a sidecar JSON. The experiment consumes that
  file via ``manual_baseline_record_path``.

Tuning still lives on the Experiments tab (the existing pretension page).
This controller drives the same algorithm with the saved config so the
operator can pretension in one click after tuning.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
import json
import math
from pathlib import Path
from typing import Any

import yaml


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class PretensionTrialState:
    """Snapshot of trial state for the GUI to render."""

    is_running: bool = False
    last_status: str = "Pretension trial idle."
    last_error: str | None = None
    last_run_output_dir: Path | None = None
    last_run_summary: dict[str, Any] = field(default_factory=dict)
    last_run_accepted: bool | None = None
    last_run_comparison_report: dict[str, Any] = field(default_factory=dict)
    manual_baseline_count: int = 0
    manual_baseline_path: Path | None = None
    manual_baseline_records: list[dict[str, Any]] = field(default_factory=list)


class PretensionTrialController:
    """Coordinates one-click pretension runs + manual-baseline capture.

    Construction-injected dependencies:
    - ``servo_service``: provides bus, telemetry, calibration access.
    - ``tracking_service``: optional; if present, manual baseline records
      include tip XY pose.
    - ``experiment_runner``: ``ExperimentRunner`` that owns experiment
      registry. The trial calls ``run_experiment("pretension_validation", ...)``.
    - ``settings``: needed for the active-segment servo IDs and the example
      pretension config path.
    - ``project_root``: where to anchor relative paths.

    Manual baselines live at ``data/diagnostics/pretension_manual_baselines.json``
    (created on first capture). Each capture appends one record. The
    experiment reads the file via ``manual_baseline_record_path``.
    """

    DEFAULT_CONFIG_RELATIVE_PATH = "config/experiment_pretension_validation.example.yaml"
    DEFAULT_MANUAL_BASELINE_RELATIVE_PATH = "data/diagnostics/pretension_manual_baselines.json"

    def __init__(
        self,
        *,
        servo_service,
        tracking_service,
        experiment_runner,
        settings,
        project_root: Path,
        config_path: str | None = None,
        manual_baseline_path: str | None = None,
    ) -> None:
        self.servo_service = servo_service
        self.tracking_service = tracking_service
        self.experiment_runner = experiment_runner
        self.settings = settings
        self.project_root = Path(project_root)
        self._config_rel_path = str(config_path or self.DEFAULT_CONFIG_RELATIVE_PATH)
        self._manual_baseline_rel_path = str(
            manual_baseline_path or self.DEFAULT_MANUAL_BASELINE_RELATIVE_PATH
        )
        self.state = PretensionTrialState(
            manual_baseline_path=self._resolve_manual_baseline_path(),
        )
        self._reload_manual_baseline_records_from_disk()

    # --- query helpers ---------------------------------------------------

    @property
    def manual_baseline_path(self) -> Path:
        return self._resolve_manual_baseline_path()

    def _resolve_manual_baseline_path(self) -> Path:
        path = Path(self._manual_baseline_rel_path)
        if path.is_absolute():
            return path
        return self.project_root / path

    def _resolve_config_path(self) -> Path:
        path = Path(self._config_rel_path)
        if path.is_absolute():
            return path
        return self.project_root / path

    def _active_segment_servo_ids(self) -> list[int]:
        operating_context = self.settings.robot.operating_context()
        return [int(value) for value in (operating_context.expected_servo_ids or [])]

    def _reload_manual_baseline_records_from_disk(self) -> None:
        path = self._resolve_manual_baseline_path()
        if not path.exists():
            self.state.manual_baseline_records = []
            self.state.manual_baseline_count = 0
            return
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            self.state.manual_baseline_records = []
            self.state.manual_baseline_count = 0
            self.state.last_error = (
                f"Could not parse existing manual baseline file at {path}. "
                "Delete it or fix the JSON before recording new baselines."
            )
            return
        records = payload if isinstance(payload, list) else payload.get("records") or []
        if not isinstance(records, list):
            records = []
        self.state.manual_baseline_records = [dict(item) for item in records if isinstance(item, dict)]
        self.state.manual_baseline_count = int(len(self.state.manual_baseline_records))

    # --- workflows -------------------------------------------------------

    def record_manual_baseline(self, *, note: str = "") -> PretensionTrialState:
        """Capture one manual-baseline record (positions + currents + tip XY).

        Reads telemetry once, optionally reads the tracker tip pose, and
        appends a record to ``manual_baseline_path``. The operator hand-
        tensions before each call; this just snapshots the current state.
        """
        servo_ids = self._active_segment_servo_ids()
        if not servo_ids:
            raise RuntimeError(
                "Cannot record manual baseline: no active-segment servo IDs configured."
            )
        if not getattr(self.servo_service, "is_connected", False):
            raise RuntimeError(
                "Cannot record manual baseline: the servo bus is not connected. "
                "Connect OpenRB and discover servos first."
            )
        telemetry = self.servo_service.read_live_telemetry(servo_ids)
        positions: dict[int, int] = {}
        currents: dict[int, float] = {}
        missing: list[int] = []
        for servo_id in servo_ids:
            entry = telemetry.get(int(servo_id))
            if entry is None or entry.present_position is None or entry.present_current_ma is None:
                missing.append(int(servo_id))
                continue
            positions[int(servo_id)] = int(entry.present_position)
            currents[int(servo_id)] = float(entry.present_current_ma)
        if missing:
            raise RuntimeError(
                f"Manual baseline capture incomplete: missing telemetry for servo IDs {missing}."
            )

        tip_xy: list[float] | None = None
        tip_xyz: list[float] | None = None
        if self.tracking_service is not None:
            snapshot_reader = getattr(self.tracking_service, "peek_snapshot", None)
            snapshot = snapshot_reader() if callable(snapshot_reader) else self.tracking_service.get_snapshot()
            matrix = getattr(snapshot, "T_robot_tip", None) if snapshot is not None else None
            if matrix is not None:
                try:
                    tip_xyz = [float(matrix[0][3]), float(matrix[1][3]), float(matrix[2][3])]
                    tip_xy = [float(matrix[0][3]), float(matrix[1][3])]
                except Exception:
                    tip_xy = None
                    tip_xyz = None

        record = {
            "index": int(self.state.manual_baseline_count),
            "timestamp_utc": _utc_now_iso(),
            "source": "servos_tab_manual_capture",
            "note": str(note or ""),
            "servo_ids": list(servo_ids),
            "positions_by_servo": positions,
            "currents_ma_by_servo": currents,
            "tip_xy_mm": tip_xy,
            "tip_xyz_mm": tip_xyz,
        }
        self.state.manual_baseline_records = list(self.state.manual_baseline_records) + [record]
        self.state.manual_baseline_count = int(len(self.state.manual_baseline_records))

        path = self._resolve_manual_baseline_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps({"records": self.state.manual_baseline_records}, indent=2),
            encoding="utf-8",
        )
        self.state.manual_baseline_path = path
        spread_ma = _max_minus_min(currents.values())
        self.state.last_error = None
        self.state.last_status = (
            f"Manual baseline {self.state.manual_baseline_count} captured. "
            f"Current spread across servos: {spread_ma:.2f} mA. "
            f"Saved to {path}."
        )
        return self.state

    def clear_manual_baselines(self) -> PretensionTrialState:
        """Empty the manual-baseline records file. No-op if it does not exist."""
        path = self._resolve_manual_baseline_path()
        if path.exists():
            path.write_text(json.dumps({"records": []}, indent=2), encoding="utf-8")
        self.state.manual_baseline_records = []
        self.state.manual_baseline_count = 0
        self.state.last_error = None
        self.state.last_status = "Manual baseline records cleared."
        return self.state

    def run_pretension_trial(self) -> PretensionTrialState:
        """Run the pretension experiment with the saved config + active segment.

        Loads the example config YAML, fills in the active segment's servo IDs
        when ``servo_ids`` is empty in the config, points
        ``manual_baseline_record_path`` at the sidecar file if records exist,
        and calls ``experiment_runner.run_experiment``. Synchronous; returns
        when the run is done.
        """
        if self.state.is_running:
            raise RuntimeError("A pretension trial is already running.")
        if not getattr(self.servo_service, "is_connected", False):
            raise RuntimeError(
                "Cannot run pretension trial: the servo bus is not connected."
            )
        config_path = self._resolve_config_path()
        config_payload = self._load_config_payload(config_path)
        # Active segment IDs override empty servo_ids in the config.
        if not config_payload.get("servo_ids"):
            config_payload["servo_ids"] = self._active_segment_servo_ids()
        # Wire the manual baselines path so the experiment uses them for
        # comparison if any records exist.
        if self.state.manual_baseline_count > 0:
            config_payload["manual_baseline_record_path"] = str(
                self._resolve_manual_baseline_path().relative_to(self.project_root)
                if self._resolve_manual_baseline_path().is_absolute()
                and self._resolve_manual_baseline_path().is_relative_to(self.project_root)
                else self._resolve_manual_baseline_path()
            )
            # If manual records are pre-recorded, we do NOT want the inline
            # capture phase to also run; force it off.
            config_payload["manual_baseline_capture_count"] = 0

        self.state.is_running = True
        self.state.last_status = (
            f"Pretension trial running ({len(config_payload.get('servo_ids') or [])} servos, "
            f"{int(config_payload.get('repeat_runs', 1))} repeats)."
        )
        try:
            result = self.experiment_runner.run_experiment(
                "pretension_validation",
                config=config_payload,
                operator_notes="servos_tab_pretension_trial",
            )
        except Exception as exc:
            self.state.is_running = False
            self.state.last_error = str(exc)
            self.state.last_status = f"Pretension trial errored: {exc}"
            raise
        self.state.is_running = False
        self.state.last_run_output_dir = result.paths.output_dir
        self.state.last_run_summary = dict(result.summary.experiment_metrics or {})
        self.state.last_run_accepted = bool(result.success)
        self.state.last_run_comparison_report = dict(
            self.state.last_run_summary.get("pretension_comparison_report") or {}
        )
        if result.success:
            accepted_runs = int(self.state.last_run_summary.get("accepted_run_count", 0) or 0)
            repeat_runs = int(self.state.last_run_summary.get("run_count", 0) or 0)
            tip_summary = (
                self.state.last_run_comparison_report.get("algorithm_population_summary", {}).get(
                    "tip_xy_error_to_target_mm", {}
                )
                if isinstance(self.state.last_run_comparison_report, dict)
                else {}
            )
            mean_tip_err = tip_summary.get("mean") if isinstance(tip_summary, dict) else None
            self.state.last_error = None
            self.state.last_status = (
                f"Pretension trial complete: {accepted_runs}/{repeat_runs} accepted. "
                + (
                    f"Mean tip XY error: {float(mean_tip_err):.3f} mm. "
                    if isinstance(mean_tip_err, (int, float))
                    else ""
                )
                + f"Run folder: {result.paths.output_dir}."
            )
        else:
            self.state.last_error = result.message
            self.state.last_status = f"Pretension trial failed: {result.message}"
        return self.state

    def _load_config_payload(self, config_path: Path) -> dict[str, Any]:
        if config_path.exists():
            try:
                payload = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
            except Exception as exc:
                raise RuntimeError(
                    f"Could not parse pretension config at {config_path}: {exc}"
                ) from exc
            if not isinstance(payload, dict):
                raise RuntimeError(
                    f"Pretension config at {config_path} must be a YAML mapping; got {type(payload).__name__}."
                )
            return dict(payload)
        # No saved config; use the experiment's built-in defaults.
        return {"mode": "single_segment_staged", "staged_strategy": "conservative_startup"}


def _max_minus_min(values) -> float:
    items = [float(v) for v in values if v is not None]
    if not items:
        return 0.0
    return float(max(items) - min(items))

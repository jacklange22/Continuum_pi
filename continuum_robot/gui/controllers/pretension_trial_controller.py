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

    def verify_tightening_signs(
        self,
        *,
        probe_ticks: int = 5,
        settle_s: float = 0.4,
    ) -> dict[str, Any]:
        """Jog each active-segment servo +probe_ticks, then -probe_ticks, while
        snapshotting position + current at every step. The operator visually
        confirms that decreasing-tick == tighten == higher current.

        The algorithm assumes ``tightening = decreasing tick`` for every active
        servo. This is hardcoded in the pretension stepping loop and in the
        ``tightening_rotation_by_servo`` map in the robot config. If any servo
        is physically wired or wound in reverse, the algorithm will RELEASE
        that tendon while tightening the other three. This sign-test produces
        the per-servo evidence the operator needs before the first trial run.

        Returns a structured report with the per-servo delta-current
        observations and a verdict for each servo:

        - ``"consistent"``: decreasing tick raised current (and increasing
          tick lowered it) by ``min_current_swing_ma`` or more in BOTH
          directions. Tightening sign matches the algorithm's assumption.
        - ``"inverted"``: the sign was the opposite. Tightening would
          actually RELEASE that servo. STOP and fix the wiring/spool before
          running pretension.
        - ``"low_response"``: motion produced less than
          ``min_current_swing_ma`` of swing in either direction. The probe
          was too small or the spine was slack; rerun with a larger
          ``probe_ticks`` once the spine is taut enough to register force.
        - ``"missing_telemetry"``: a telemetry read failed during the probe.

        Returns:
            ``{"servo_id": int, "start_tick": int, "current_at_start_ma": float,
               "after_decrease_tick_current_ma": float,
               "after_increase_tick_current_ma": float,
               "delta_decrease_ma": float, "delta_increase_ma": float,
               "verdict": str, "note": str}`` per servo, plus an
            ``overall_verdict`` summary and ``all_consistent: bool``.

        Does not touch the production registration session or the manual-
        baseline file. Does not torque-off afterwards. Each servo returns to
        its starting tick at the end of its individual probe.
        """
        servo_ids = self._active_segment_servo_ids()
        if not servo_ids:
            raise RuntimeError(
                "Cannot run sign-test: no active-segment servo IDs configured."
            )
        if not getattr(self.servo_service, "is_connected", False):
            raise RuntimeError(
                "Cannot run sign-test: the servo bus is not connected."
            )
        probe = max(1, int(probe_ticks))
        settle = max(0.05, float(settle_s))
        min_current_swing_ma = 2.0  # Below the typical noise floor + safety margin.

        per_servo: list[dict[str, Any]] = []
        all_consistent = True
        any_inverted = False
        for servo_id in servo_ids:
            sid = int(servo_id)
            # Read starting telemetry.
            telemetry = self.servo_service.read_live_telemetry([sid])
            entry = telemetry.get(sid)
            if entry is None or entry.present_position is None or entry.present_current_ma is None:
                per_servo.append(
                    {
                        "servo_id": sid,
                        "start_tick": None,
                        "current_at_start_ma": None,
                        "after_decrease_tick_current_ma": None,
                        "after_increase_tick_current_ma": None,
                        "delta_decrease_ma": None,
                        "delta_increase_ma": None,
                        "verdict": "missing_telemetry",
                        "note": "Could not read present_position / present_current_ma before probe.",
                    }
                )
                all_consistent = False
                continue
            start_tick = int(entry.present_position)
            start_current = float(entry.present_current_ma)
            # Step 1: decrease tick by probe_ticks. Should TIGHTEN.
            target_decrease = start_tick - probe
            try:
                self.servo_service._write_goal_positions({sid: int(target_decrease)})
            except Exception as exc:
                per_servo.append(
                    {
                        "servo_id": sid,
                        "start_tick": start_tick,
                        "current_at_start_ma": start_current,
                        "after_decrease_tick_current_ma": None,
                        "after_increase_tick_current_ma": None,
                        "delta_decrease_ma": None,
                        "delta_increase_ma": None,
                        "verdict": "missing_telemetry",
                        "note": f"Goal-write to tick {target_decrease} failed: {exc}",
                    }
                )
                all_consistent = False
                continue
            _sleep = getattr(self.servo_service, "_sleep_fn", None) or __import__("time").sleep
            _sleep(settle)
            after_decrease_telemetry = self.servo_service.read_live_telemetry([sid])
            after_decrease_entry = after_decrease_telemetry.get(sid)
            after_decrease_current = (
                float(after_decrease_entry.present_current_ma)
                if after_decrease_entry is not None and after_decrease_entry.present_current_ma is not None
                else None
            )
            # Step 2: return to start, then increase tick by probe_ticks. Should RELEASE.
            self.servo_service._write_goal_positions({sid: int(start_tick)})
            _sleep(settle)
            target_increase = start_tick + probe
            self.servo_service._write_goal_positions({sid: int(target_increase)})
            _sleep(settle)
            after_increase_telemetry = self.servo_service.read_live_telemetry([sid])
            after_increase_entry = after_increase_telemetry.get(sid)
            after_increase_current = (
                float(after_increase_entry.present_current_ma)
                if after_increase_entry is not None and after_increase_entry.present_current_ma is not None
                else None
            )
            # Always restore to start tick.
            self.servo_service._write_goal_positions({sid: int(start_tick)})
            _sleep(settle)

            delta_dec = (
                after_decrease_current - start_current
                if after_decrease_current is not None
                else None
            )
            delta_inc = (
                after_increase_current - start_current
                if after_increase_current is not None
                else None
            )
            verdict = "consistent"
            note = ""
            if delta_dec is None or delta_inc is None:
                verdict = "missing_telemetry"
                note = "Could not read current after one of the probe steps."
                all_consistent = False
            elif (
                abs(delta_dec) < min_current_swing_ma
                and abs(delta_inc) < min_current_swing_ma
            ):
                verdict = "low_response"
                note = (
                    f"Both directions produced |delta_current| < {min_current_swing_ma} mA. "
                    "Probe too small or spine too slack to register tension; increase "
                    "probe_ticks once the spine has some preload."
                )
                all_consistent = False
            elif delta_dec > 0 and delta_inc <= 0:
                # Algorithm-consistent: decreasing tick raised current.
                verdict = "consistent"
                note = "Decreasing tick raised current (tighten) as expected."
            elif delta_dec < 0 and delta_inc >= 0:
                verdict = "inverted"
                note = (
                    "Decreasing tick LOWERED current. This servo is wired or wound "
                    "OPPOSITE the algorithm's assumption. The pretension routine "
                    "would RELEASE this tendon while tightening the others. "
                    "STOP and fix the wiring or spool direction before running pretension."
                )
                all_consistent = False
                any_inverted = True
            else:
                # Mixed / ambiguous (e.g. both went up, or both went down).
                verdict = "low_response"
                note = (
                    f"Ambiguous current swing: delta_decrease={delta_dec:+.2f} mA, "
                    f"delta_increase={delta_inc:+.2f} mA. Expected opposite signs."
                )
                all_consistent = False
            per_servo.append(
                {
                    "servo_id": sid,
                    "start_tick": start_tick,
                    "current_at_start_ma": start_current,
                    "after_decrease_tick_current_ma": after_decrease_current,
                    "after_increase_tick_current_ma": after_increase_current,
                    "delta_decrease_ma": delta_dec,
                    "delta_increase_ma": delta_inc,
                    "verdict": verdict,
                    "note": note,
                }
            )

        overall_verdict = (
            "STOP — at least one servo is wired or wound opposite the algorithm. "
            "Fix the physical setup before running pretension."
            if any_inverted
            else (
                "OK — every active servo's tightening sign matches the algorithm."
                if all_consistent
                else "REVIEW — one or more servos returned low-response or missing telemetry."
            )
        )
        report = {
            "schema_version": "1.0",
            "generated_at_utc": _utc_now_iso(),
            "probe_ticks": int(probe),
            "settle_s": float(settle),
            "min_current_swing_ma": float(min_current_swing_ma),
            "servo_ids": list(servo_ids),
            "per_servo": per_servo,
            "all_consistent": bool(all_consistent),
            "any_inverted": bool(any_inverted),
            "overall_verdict": overall_verdict,
        }
        self.state.last_error = None
        self.state.last_status = f"Tightening-sign verification: {overall_verdict}"
        return report

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

    def run_one_rig_consistency_proof(self, *, repeat_runs: int = 5) -> PretensionTrialState:
        """Run the pretension experiment for ``repeat_runs`` repeats and grade
        the result with the consistency verdict (high/medium/low/failed).

        Thin wrapper around ``run_pretension_trial`` that just forces the
        repeat count. The experiment's own consistency-verdict logic
        (``PretensionValidationExperiment._compute_consistency_verdict``)
        produces the headline grade, written to ``pretension_quality_summary.json``
        and surfaced into ``state.last_run_summary["consistency_verdict"]``.
        """
        return self.run_pretension_trial(repeat_runs_override=int(repeat_runs))

    def run_pretension_trial(self, *, repeat_runs_override: int | None = None) -> PretensionTrialState:
        """Run the pretension experiment with the saved config + active segment.

        Loads the example config YAML, fills in the active segment's servo IDs
        when ``servo_ids`` is empty in the config, points
        ``manual_baseline_record_path`` at the sidecar file if records exist,
        and calls ``experiment_runner.run_experiment``. Synchronous; returns
        when the run is done.

        Args:
            repeat_runs_override: If set, replaces ``repeat_runs`` in the
                loaded config payload. Used by the one-rig consistency proof
                button on the Servos tab.
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
        if repeat_runs_override is not None:
            config_payload["repeat_runs"] = max(1, int(repeat_runs_override))
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

    def latest_pretension_report_path(self) -> Path | None:
        """Path to the last pretension run's report directory (or ``None`` if
        no run has completed in this session). Used by the Servos tab "Open
        Latest Pretension Report" button."""
        path = getattr(self.state, "last_run_output_dir", None)
        if path is None:
            return None
        return Path(path)

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

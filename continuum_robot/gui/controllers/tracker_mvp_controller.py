"""Focused tracker-first MVP controller for tracker, pivot, and registration bring-up."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
import json
from pathlib import Path
import threading
import time

import numpy as np

from continuum_robot.config.settings import Settings
from continuum_robot.experiments.pivot_utils import write_tip_vector_file
from continuum_robot.hardware.serial_ports import SerialPortInfo, discover_serial_ports
from continuum_robot.tracking.benchmarking import TrackerBenchmarkThresholds
from continuum_robot.tracking.diagnostics import build_tracking_diagnostics_report
from continuum_robot.tracking.transforms import quat_wxyz_to_rotmat


@dataclass
class WorkflowStepState:
    """One operator-facing workflow step."""

    index: int
    label: str
    status: str
    message: str


@dataclass
class PivotCollectionSample:
    """One live 0B sample collected for pivot review."""

    frame_number: int | None
    captured_at_utc: str | None
    quaternion_wxyz: tuple[float, float, float, float]
    translation_mm: tuple[float, float, float]


@dataclass
class TrackerMvpViewState:
    """UI-facing tracker MVP workflow state."""

    tracker_port: str
    available_ports: list[SerialPortInfo] = field(default_factory=list)
    tracker_connected: bool = False
    tracker_healthy: bool = False
    tracker_operational: bool = False
    tracker_verdict: str = "not_validated"
    validation_passed: bool = False
    backend_name: str = ""
    backend_identity: str = ""
    canonical_state: str = "disconnected"
    connection_state: str = "disconnected"
    unique_frames_observed: int = 0
    effective_frame_rate_hz: float | None = None
    tracker_data_age_s: float | None = None
    raw_live_tool_ids: list[str] = field(default_factory=list)
    normalized_live_tool_ids: list[str] = field(default_factory=list)
    tool_0a_detected: bool = False
    tool_0a_visible: bool = False
    tool_0a_status: str = "missing"
    tool_0b_detected: bool = False
    tool_0b_visible: bool = False
    tool_0b_status: str = "missing"
    transforms_valid: bool = False
    validation_report_path: str = ""
    validation_summary: str = "Run tracker validation after connecting."
    validation_lines: list[str] = field(default_factory=list)
    pivot_status: str = "not_run"
    pivot_summary: str = "Run pivot calibration after tracker validation and 0B visibility."
    pivot_collection_active: bool = False
    pivot_live_tool_status: str = "not_collecting"
    pivot_live_sample_count: int = 0
    pivot_motion_span_deg: float | None = None
    pivot_motion_ready: bool = False
    pivot_can_start: bool = False
    pivot_can_stop: bool = False
    pivot_can_solve: bool = False
    pivot_can_accept: bool = False
    pivot_pending_accept: bool = False
    pivot_run_path: str = ""
    pivot_capture_dataset_path: str = ""
    pivot_tip_path: str = ""
    pivot_tip_exists: bool = False
    pivot_pending_tip_path: str = ""
    pivot_tip_preview: str = "No tip file loaded."
    pivot_rmse_mm: float | None = None
    pivot_sample_count_total: int = 0
    pivot_sample_count_used: int = 0
    pivot_sample_count_rejected: int = 0
    pivot_input_format: str = ""
    pivot_input_usable_rows: int = 0
    pivot_input_rejected_row_count: int = 0
    pivot_input_rejected_rows: list[str] = field(default_factory=list)
    measurement_point_ready: bool = False
    measurement_point_source: str = ""
    measurement_point_message: str = ""
    selected_landmarks: list[str] = field(default_factory=list)
    registration_selection_ready: bool = False
    registration_capture_complete: bool = False
    registration_solved: bool = False
    registration_saved: bool = False
    registration_ready: bool = False
    registration_blockers: list[str] = field(default_factory=list)
    registration_fre_mm: float | None = None
    latest_registration_path: str = ""
    latest_registration_status: str = "No accepted registration saved."
    live_pose_ready: bool = False
    live_tip_status: str = "missing_registration"
    live_tip_position_mm: tuple[float, float, float] | None = None
    transform_summary_lines: list[str] = field(default_factory=list)
    workflow_steps: list[WorkflowStepState] = field(default_factory=list)
    status_message: str = "Connect tracker to begin."
    last_error: str | None = None


class TrackerMvpController:
    """Focused operator controller for tracker validation, pivot calibration, and registration readiness."""

    PIVOT_TARGET_SAMPLES = 80
    PIVOT_MIN_SAMPLES = 12
    PIVOT_SAMPLE_PERIOD_S = 0.02
    PIVOT_STD_DEV_THRESHOLD = 3.0
    PIVOT_MIN_MOTION_SPAN_DEG = 25.0

    def __init__(
        self,
        *,
        tracking_service,
        registration_service,
        registration_controller,
        experiment_runner,
        settings: Settings,
        project_root: Path,
    ) -> None:
        self.tracking_service = tracking_service
        self.registration_service = registration_service
        self.registration_controller = registration_controller
        self.experiment_runner = experiment_runner
        self.settings = settings
        self.project_root = Path(project_root)
        self.thresholds = TrackerBenchmarkThresholds(
            min_effective_fps=float(settings.serial.tracker_min_effective_fps),
            max_stale_interval_s=float(settings.serial.tracker_max_stale_interval_s),
            max_consecutive_missing_frames=int(settings.serial.tracker_max_consecutive_missing_frames),
            require_valid_transforms=bool(settings.serial.tracker_require_valid_transforms),
        )
        self._validation_passed = False
        self._validation_operational = False
        self._last_validation_report_path: Path | None = None
        self._last_validation_report = None
        self._last_pivot_run_path: Path | None = None
        self._last_pivot_metrics: dict[str, object] = {}
        self._pivot_collection_active = False
        self._pivot_collection_status = "not_run"
        self._pivot_live_tool_status = "not_collecting"
        self._pivot_live_tool_tracked = False
        self._pivot_samples: list[PivotCollectionSample] = []
        self._pivot_last_sample_key: tuple[int | None, str | None] | None = None
        self._pivot_motion_span_deg: float | None = None
        self._pivot_pending_accept = False
        self._pivot_capture_dataset_path: Path | None = None
        self._pivot_pending_tip_path: Path | None = None
        self._pivot_thread: threading.Thread | None = None
        self._pivot_stop_event = threading.Event()
        self._pivot_lock = threading.Lock()
        self.state = TrackerMvpViewState(tracker_port=str(settings.serial.aurora_port))
        self.rescan_ports()
        self.refresh()

    def rescan_ports(self) -> TrackerMvpViewState:
        ports = discover_serial_ports()
        if self.settings.runtime.mock_mode:
            ports.append(SerialPortInfo(device="/dev/mock-aurora", description="Mock Aurora port"))
        deduped = {port.device: port for port in ports}
        self.state.available_ports = sorted(deduped.values(), key=lambda item: item.device)
        return self.refresh()

    def set_tracker_port(self, port: str) -> None:
        resolved = str(port).strip()
        self.state.tracker_port = resolved
        self.tracking_service.set_port(resolved)
        self._validation_passed = False
        self._validation_operational = False

    def connect_tracker(self) -> None:
        try:
            if not self.settings.runtime.mock_mode and not self.state.tracker_port:
                raise RuntimeError("Tracker port is empty. Select the Aurora device before connecting.")
            self.tracking_service.start(self.state.tracker_port or None)
        except Exception as exc:
            self.refresh()
            self.state.last_error = str(exc)
            self.state.status_message = f"Tracker connect failed: {exc}"
            raise
        self.refresh()
        self.state.last_error = None
        self.state.status_message = "Tracker connection requested."

    def disconnect_tracker(self) -> None:
        self._stop_pivot_collection(silent=True)
        try:
            self.tracking_service.stop()
            self._validation_passed = False
            self._validation_operational = False
        except Exception as exc:
            self.refresh()
            self.state.last_error = str(exc)
            self.state.status_message = f"Tracker disconnect failed: {exc}"
            raise
        self.refresh()
        self.state.last_error = None
        self.state.status_message = "Tracker disconnected."

    def validate_tracker(self) -> Path:
        try:
            report = build_tracking_diagnostics_report(
                self.tracking_service,
                duration_s=1.5,
                sample_period_s=0.05,
                wait_for_first_frame_s=2.0,
                thresholds=self.thresholds,
                required_tool_ids=(
                    self.settings.registration.coil_tool_id,
                    self.settings.registration.capture_tool_id,
                ),
            )
            report_dir = self.project_root / "data" / "tracker_validations"
            report_dir.mkdir(parents=True, exist_ok=True)
            stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
            report_path = report_dir / f"{stamp}_tracker_validation.json"
            report_path.write_text(json.dumps(report.to_dict(), indent=2), encoding="utf-8")
            self._validation_passed = bool(
                report.tracker_ready
                and self._tool_is_visible(self.settings.registration.coil_tool_id)
                and self._tool_is_visible(self.settings.registration.capture_tool_id)
            )
            self._validation_operational = bool(
                getattr(report, "tracker_operational", report.tracker_ready)
                and self._tool_is_visible(self.settings.registration.coil_tool_id)
                and self._tool_is_visible(self.settings.registration.capture_tool_id)
            )
            self._last_validation_report_path = report_path
            self._last_validation_report = report
        except Exception as exc:
            self._validation_passed = False
            self._validation_operational = False
            self._last_validation_report = None
            self.refresh()
            self.state.last_error = str(exc)
            self.state.status_message = f"Tracker validation failed: {exc}"
            raise
        self.refresh()
        self.state.last_error = None
        self.state.status_message = (
            f"Tracker validation {self.state.tracker_verdict.replace('_', ' ')}. Saved report to {report_path.name}."
        )
        return report_path

    def run_pivot_calibration(self) -> Path:
        """Compatibility one-shot pivot path retained outside the guided tracker workflow."""
        self._require_pivot_start_ready()
        tip_output = self._registration_tip_output_path()
        result = self.experiment_runner.run_experiment(
            "pivot_calibration",
            config={
                "tool_id": self.settings.registration.capture_tool_id,
                "dry_run": False,
                "sample_count": self.PIVOT_TARGET_SAMPLES,
                "sample_period_s": self.PIVOT_SAMPLE_PERIOD_S,
                "std_dev_threshold": self.PIVOT_STD_DEV_THRESHOLD,
                "min_samples": self.PIVOT_MIN_SAMPLES,
                "output_tip_file": str(tip_output),
            },
            operator_notes="tracker_mvp_gui_legacy_one_shot",
            output_dir=self._pivot_run_output_root(),
        )
        if not result.success:
            self.refresh()
            self.state.last_error = result.message
            self.state.status_message = f"Pivot calibration failed: {result.message}"
            raise RuntimeError(result.message)

        self._record_pivot_result(
            result=result,
            accepted_tip_path=tip_output,
            capture_dataset_path=None,
            pending_tip_path=None,
            pending_accept=False,
        )
        self.registration_service.refresh_measurement_point_geometry()
        self.refresh()
        self.state.last_error = None
        self.state.status_message = (
            f"Pivot calibration saved tip file to {tip_output}. "
            f"RMSE={float(self._last_pivot_metrics.get('rmse_mm', 0.0)):.3f} mm."
        )
        return result.paths.output_dir

    def start_pivot_collection(self) -> None:
        self._require_pivot_start_ready()
        with self._pivot_lock:
            self._pivot_collection_active = True
            self._pivot_collection_status = "collecting"
            self._pivot_live_tool_status = "waiting_for_0B_tracking"
            self._pivot_live_tool_tracked = False
            self._pivot_samples = []
            self._pivot_last_sample_key = None
            self._pivot_motion_span_deg = None
            self._pivot_pending_accept = False
            self._pivot_capture_dataset_path = None
            self._pivot_pending_tip_path = None
            self._last_pivot_run_path = None
            self._last_pivot_metrics = {}
            self._pivot_stop_event = threading.Event()
            self._pivot_thread = threading.Thread(
                target=self._collect_pivot_samples_loop,
                name="tracker-mvp-pivot-collector",
                daemon=True,
            )
            self._pivot_thread.start()
        self.refresh()
        self.state.last_error = None
        self.state.status_message = (
            "Pivot collection started. Move 0B through a wide range of orientations, then stop and solve."
        )

    def stop_pivot_collection(self) -> None:
        if not self._pivot_collection_active:
            raise RuntimeError("Pivot collection is not running.")
        sample_count = self._stop_pivot_collection(silent=False)
        self.refresh()
        self.state.last_error = None
        self.state.status_message = (
            f"Pivot collection stopped with {sample_count} sample(s). Review motion span, then solve."
        )

    def solve_pivot_collection(self) -> Path:
        samples = self._pivot_samples_for_solve()
        if len(samples) < self.PIVOT_MIN_SAMPLES:
            raise RuntimeError(
                f"Pivot solve requires at least {self.PIVOT_MIN_SAMPLES} samples; collected {len(samples)}."
            )
        stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        capture_dataset_path = self._write_pivot_capture_dataset(samples=samples, stamp=stamp)
        pending_tip_path = self.project_root / "data" / "tip_cals" / "staged" / f"{stamp}_generated_penprobe_tip.csv"
        result = self.experiment_runner.run_experiment(
            "pivot_calibration",
            config={
                "tool_id": self.settings.registration.capture_tool_id,
                "input_path": str(capture_dataset_path),
                "dry_run": False,
                "std_dev_threshold": self.PIVOT_STD_DEV_THRESHOLD,
                "min_samples": self.PIVOT_MIN_SAMPLES,
                "output_tip_file": str(pending_tip_path),
            },
            operator_notes="tracker_mvp_gui_review",
            output_dir=self._pivot_run_output_root(),
            output_dir_name=f"{stamp}_pivot_calibration_review",
        )
        if not result.success:
            self._record_pivot_result(
                result=result,
                accepted_tip_path=self._registration_tip_output_path(),
                capture_dataset_path=capture_dataset_path,
                pending_tip_path=pending_tip_path,
                pending_accept=False,
                status_override="solve_failed",
            )
            self.refresh()
            self.state.last_error = result.message
            self.state.status_message = f"Pivot calibration solve failed: {result.message}"
            raise RuntimeError(result.message)

        self._record_pivot_result(
            result=result,
            accepted_tip_path=self._registration_tip_output_path(),
            capture_dataset_path=capture_dataset_path,
            pending_tip_path=pending_tip_path,
            pending_accept=True,
            status_override="review_ready",
        )
        self.refresh()
        self.state.last_error = None
        self.state.status_message = (
            f"Pivot solved. Review RMSE and sample counts, then accept the staged tip file {pending_tip_path.name}."
        )
        return result.paths.output_dir

    def accept_pivot_tip_file(self) -> Path:
        pending_tip_path = self._pivot_pending_tip_path
        if pending_tip_path is None or not pending_tip_path.exists():
            raise RuntimeError("No staged pivot tip file is pending acceptance.")
        vector = np.asarray(np.loadtxt(pending_tip_path, delimiter=","), dtype=float).reshape(-1)
        if vector.shape != (3,):
            raise RuntimeError(f"Staged tip file is invalid: {pending_tip_path}")
        output_path = write_tip_vector_file(self._registration_tip_output_path(), vector.tolist())
        with self._pivot_lock:
            self._pivot_pending_accept = False
            self._pivot_pending_tip_path = pending_tip_path
            self._pivot_collection_status = "accepted"
            self._last_pivot_metrics["status"] = "accepted"
            self._last_pivot_metrics["accepted_tip_file"] = str(output_path)
        self.registration_service.refresh_measurement_point_geometry()
        self.refresh()
        self.state.last_error = None
        self.state.status_message = f"Accepted pivot tip file saved to {output_path}."
        return output_path

    def reset_pivot_workflow(self) -> None:
        self._stop_pivot_collection(silent=True)
        with self._pivot_lock:
            self._pivot_collection_status = "not_run"
            self._pivot_live_tool_status = "not_collecting"
            self._pivot_live_tool_tracked = False
            self._pivot_samples = []
            self._pivot_last_sample_key = None
            self._pivot_motion_span_deg = None
            self._pivot_pending_accept = False
            self._pivot_capture_dataset_path = None
            self._pivot_pending_tip_path = None
            self._last_pivot_run_path = None
            self._last_pivot_metrics = {}
        self.refresh()
        self.state.last_error = None
        self.state.status_message = "Pivot workflow reset. Start collection again when 0B is ready."

    def refresh(self) -> TrackerMvpViewState:
        snapshot = self.tracking_service.get_snapshot()
        registration_snapshot = self.registration_service.get_snapshot()
        measurement_status = self.registration_service.get_measurement_point_status(refresh=True)
        capture_tool_id = self.settings.registration.capture_tool_id
        coil_tool_id = self.settings.registration.coil_tool_id
        tool_0a = snapshot.tools.get(coil_tool_id)
        tool_0b = snapshot.tools.get(capture_tool_id)
        tool_0a_detected = bool(
            coil_tool_id in snapshot.normalized_live_tool_ids
            or (tool_0a is not None and tool_0a.tracking_state != "unknown")
        )
        tool_0b_detected = bool(
            capture_tool_id in snapshot.normalized_live_tool_ids
            or (tool_0b is not None and tool_0b.tracking_state != "unknown")
        )

        tracker_connected = bool(
            snapshot.backend_connected
            or snapshot.canonical_state in {"mock", "streaming_healthy", "streaming_degraded"}
            or snapshot.connection_state in {"tracking", "starting", "connecting"}
        )
        tracker_healthy = bool(
            tracker_connected
            and snapshot.unique_frames_observed > 0
            and not snapshot.tracker_data_stale
            and not snapshot.last_error
        )
        tool_0a_visible = bool(tool_0a is not None and tool_0a.tracking_state == "tracked")
        tool_0b_visible = bool(tool_0b is not None and tool_0b.tracking_state == "tracked")
        transforms_valid = bool(
            tool_0a_detected
            and tool_0b_detected
            and tool_0b_visible
            and tool_0a_visible
            and tool_0b is not None
            and tool_0a is not None
            and tool_0b.tracking_state != "invalid"
            and tool_0a.tracking_state != "invalid"
        )
        latest_registration_status = (
            "Solved and pending save."
            if registration_snapshot.pending_accept
            else (
                f"Saved registration FRE={registration_snapshot.fre_mm:.3f} mm."
                if registration_snapshot.latest_accepted_path and registration_snapshot.fre_mm is not None
                else (
                    "Accepted registration is saved."
                    if registration_snapshot.latest_accepted_path
                    else "No accepted registration saved."
                )
            )
        )
        live_tip_status = self._render_live_pose_status(snapshot)
        live_pose_ready = bool(
            snapshot.tip_pose_status == "ok"
            and snapshot.T_robot_tip is not None
            and not snapshot.tracker_data_stale
        )
        live_tip_position = None
        if live_pose_ready and snapshot.T_robot_tip is not None:
            live_tip_position = (
                float(snapshot.T_robot_tip[0][3]),
                float(snapshot.T_robot_tip[1][3]),
                float(snapshot.T_robot_tip[2][3]),
            )

        accepted_tip_path = (
            Path(measurement_status["path"])
            if measurement_status.get("path")
            else self._registration_tip_output_path()
        )
        accepted_tip_exists = accepted_tip_path.exists()

        with self._pivot_lock:
            pivot_collection_active = self._pivot_collection_active
            pivot_collection_status = self._pivot_collection_status
            pivot_live_tool_status = self._pivot_live_tool_status
            pivot_live_tool_tracked = self._pivot_live_tool_tracked
            pivot_live_sample_count = len(self._pivot_samples)
            pivot_motion_span_deg = self._pivot_motion_span_deg
            pivot_pending_accept = self._pivot_pending_accept
            pivot_capture_dataset_path = str(self._pivot_capture_dataset_path or "")
            pivot_pending_tip_path = str(self._pivot_pending_tip_path or "")

        pivot_motion_ready = bool(
            pivot_motion_span_deg is not None and pivot_motion_span_deg >= self.PIVOT_MIN_MOTION_SPAN_DEG
        )
        display_tip_path = Path(pivot_pending_tip_path) if pivot_pending_accept and pivot_pending_tip_path else accepted_tip_path
        pivot_tip_preview = self._tip_file_preview(display_tip_path)

        self.state.tracker_port = snapshot.port or self.state.tracker_port
        self.state.tracker_connected = tracker_connected
        self.state.tracker_healthy = tracker_healthy
        report = self._last_validation_report
        self.state.tracker_operational = bool(
            self._validation_operational
            if report is not None
            else (tracker_healthy and transforms_valid)
        )
        self.state.tracker_verdict = (
            str(getattr(report, "tracker_verdict", "passed" if self.state.validation_passed else "not_validated"))
            if report is not None
            else ("passed" if self.state.validation_passed else "not_validated")
        )
        self.state.validation_passed = bool(self._validation_passed and tracker_healthy and transforms_valid)
        self.state.backend_name = snapshot.selected_backend_name or snapshot.configured_backend_name
        self.state.backend_identity = snapshot.backend_identity
        self.state.canonical_state = snapshot.canonical_state
        self.state.connection_state = snapshot.connection_state
        self.state.unique_frames_observed = snapshot.unique_frames_observed
        self.state.effective_frame_rate_hz = snapshot.effective_frame_rate_hz
        self.state.tracker_data_age_s = snapshot.tracker_data_age_s
        self.state.raw_live_tool_ids = list(snapshot.raw_live_tool_ids)
        self.state.normalized_live_tool_ids = list(snapshot.normalized_live_tool_ids)
        self.state.tool_0a_detected = tool_0a_detected
        self.state.tool_0a_visible = tool_0a_visible
        self.state.tool_0a_status = tool_0a.status if tool_0a is not None else "missing"
        self.state.tool_0b_detected = tool_0b_detected
        self.state.tool_0b_visible = tool_0b_visible
        self.state.tool_0b_status = tool_0b.status if tool_0b is not None else "missing"
        self.state.transforms_valid = transforms_valid
        self.state.validation_report_path = str(self._last_validation_report_path or "")
        self.state.validation_summary = self._validation_summary(snapshot)
        self.state.validation_lines = self._validation_lines(snapshot)
        self.state.pivot_status = pivot_collection_status
        self.state.pivot_collection_active = pivot_collection_active
        self.state.pivot_live_tool_status = (
            "0B tracked"
            if pivot_live_tool_tracked and pivot_collection_active
            else pivot_live_tool_status
        )
        self.state.pivot_live_sample_count = pivot_live_sample_count
        self.state.pivot_motion_span_deg = pivot_motion_span_deg
        self.state.pivot_motion_ready = pivot_motion_ready
        self.state.pivot_pending_accept = pivot_pending_accept
        self.state.pivot_run_path = str(self._last_pivot_run_path or "")
        self.state.pivot_capture_dataset_path = pivot_capture_dataset_path
        self.state.pivot_tip_path = str(accepted_tip_path)
        self.state.pivot_tip_exists = accepted_tip_exists
        self.state.pivot_pending_tip_path = pivot_pending_tip_path
        self.state.pivot_tip_preview = pivot_tip_preview
        self.state.pivot_rmse_mm = self._as_float(self._last_pivot_metrics.get("rmse_mm"))
        self.state.pivot_sample_count_total = int(self._last_pivot_metrics.get("sample_count_total", 0) or 0)
        self.state.pivot_sample_count_used = int(self._last_pivot_metrics.get("sample_count_used", 0) or 0)
        self.state.pivot_sample_count_rejected = int(self._last_pivot_metrics.get("sample_count_rejected", 0) or 0)
        self.state.pivot_input_format = str(self._last_pivot_metrics.get("pivot_input_format", "") or "")
        self.state.pivot_input_usable_rows = int(self._last_pivot_metrics.get("pivot_input_usable_rows", 0) or 0)
        rejected_rows = self._last_pivot_metrics.get("pivot_input_rejected_rows", []) or []
        self.state.pivot_input_rejected_row_count = int(self._last_pivot_metrics.get("pivot_input_rejected_row_count", 0) or 0)
        self.state.pivot_input_rejected_rows = [
            f"row {item.get('row')}: {item.get('reason')}"
            for item in rejected_rows
            if isinstance(item, dict)
        ]
        self.state.pivot_can_start = bool(self._pivot_start_blockers(snapshot) == [])
        self.state.pivot_can_stop = pivot_collection_active
        self.state.pivot_can_solve = bool(
            (not pivot_collection_active)
            and pivot_live_sample_count >= self.PIVOT_MIN_SAMPLES
            and pivot_collection_status in {"collecting_stopped", "collecting", "not_run", "solve_failed"}
            and pivot_live_sample_count > 0
        )
        self.state.pivot_can_accept = bool(pivot_pending_accept and Path(pivot_pending_tip_path).exists())
        self.state.pivot_summary = self._pivot_summary(
            measurement_status=measurement_status,
            sample_count=pivot_live_sample_count,
            motion_span_deg=pivot_motion_span_deg,
            pivot_collection_active=pivot_collection_active,
            pivot_pending_accept=pivot_pending_accept,
        )
        self.state.measurement_point_ready = bool(measurement_status["ready"])
        self.state.measurement_point_source = str(measurement_status["source"])
        self.state.measurement_point_message = str(measurement_status["message"])
        self.state.selected_landmarks = list(self.registration_controller.state.selected_model_labels)
        self.state.registration_selection_ready = self.registration_controller.selection_is_ready()
        self.state.registration_capture_complete = bool(
            registration_snapshot.health.state in {"ready_to_solve", "solved", "accepted"}
            or registration_snapshot.pending_accept
        )
        self.state.registration_solved = bool(
            registration_snapshot.pending_accept or registration_snapshot.health.state in {"solved", "accepted"}
        )
        self.state.registration_saved = bool(registration_snapshot.latest_accepted_path)
        self.state.registration_fre_mm = registration_snapshot.fre_mm
        self.state.latest_registration_path = str(registration_snapshot.latest_accepted_path or "")
        self.state.latest_registration_status = latest_registration_status
        self.state.live_pose_ready = live_pose_ready
        self.state.live_tip_status = live_tip_status
        self.state.live_tip_position_mm = live_tip_position
        latest_registration_summary = self.registration_service.get_latest_registration_artifact_summary()
        self.state.transform_summary_lines = self._transform_summary(
            measurement_status,
            latest_registration_summary=latest_registration_summary,
        )
        self.state.registration_blockers = self._registration_blockers(measurement_status)
        self.state.registration_ready = not self.state.registration_blockers and self.registration_controller.can_begin_session()
        self.state.workflow_steps = self._build_workflow_steps()
        if not self.state.last_error:
            self.state.status_message = self._default_status_message()
        return self.state

    def _build_workflow_steps(self) -> list[WorkflowStepState]:
        steps: list[WorkflowStepState] = []

        def _step(index: int, label: str, complete: bool, message: str, *, blocked: bool = False) -> None:
            if complete:
                status = "complete"
            elif blocked:
                status = "blocked"
            else:
                status = "ready"
            steps.append(WorkflowStepState(index=index, label=label, status=status, message=message))

        _step(1, "Connect tracker", self.state.tracker_connected, f"Port: {self.state.tracker_port or 'unset'}")
        _step(
            2,
            "Validate tracker health",
            self.state.validation_passed or self.state.tracker_operational,
            self.state.validation_summary,
            blocked=not self.state.tracker_connected,
        )
        _step(
            3,
            "Confirm tool visibility / IDs",
            self.state.tool_0a_detected and self.state.tool_0b_detected,
            f"0A={self.state.tool_0a_status}; 0B={self.state.tool_0b_status}; normalized IDs: {self.state.normalized_live_tool_ids}",
            blocked=not self.state.tracker_connected,
        )
        _step(
            4,
            "Confirm transforms valid",
            self.state.transforms_valid,
            self._transform_stage_message(),
            blocked=not (self.state.tool_0a_detected and self.state.tool_0b_detected),
        )
        pivot_review_ready = self.state.pivot_status in {"review_ready", "accepted", "success"}
        _step(
            5,
            "Run pivot calibration on 0B",
            pivot_review_ready,
            self.state.pivot_summary,
            blocked=not self.state.pivot_can_start and not self.state.pivot_collection_active and not pivot_review_ready,
        )
        _step(
            6,
            "Save tip file",
            self.state.pivot_tip_exists and self.state.measurement_point_ready and not self.state.pivot_pending_accept,
            (
                f"Accepted tip file: {self.state.pivot_tip_path}"
                if self.state.pivot_tip_exists
                else "Accept the staged pivot tip file before registration."
            ),
            blocked=self.state.pivot_pending_accept or not self.state.pivot_tip_exists,
        )
        _step(
            7,
            "Select 4 registration landmarks",
            self.state.registration_selection_ready,
            f"Selected: {self.state.selected_landmarks}",
            blocked=bool(self.state.registration_blockers and not self.state.registration_selection_ready),
        )
        _step(
            8,
            "Capture registration samples",
            self.state.registration_capture_complete,
            self.registration_controller.state.status_message,
            blocked=not self.state.registration_ready and not self.registration_controller.state.active,
        )
        _step(
            9,
            "Solve registration",
            self.state.registration_solved,
            (
                self.registration_controller.state.status_message
                if self.state.registration_solved
                else self.registration_controller.solve_readiness_message()
            ),
            blocked=not self.state.registration_capture_complete,
        )
        _step(
            10,
            "Save accepted registration",
            self.state.registration_saved,
            self.state.latest_registration_path or "Save the solved registration to persist the artifact.",
            blocked=not self.state.registration_solved,
        )
        _step(
            11,
            "Confirm live robot-frame pose availability",
            self.state.live_pose_ready,
            f"Tip pose status: {self.state.live_tip_status}",
            blocked=not self.state.registration_saved,
        )
        return steps

    def _default_status_message(self) -> str:
        for step in self.state.workflow_steps:
            if step.status != "complete":
                return f"Step {step.index}: {step.label}. {step.message}"
        return "Tracker-first MVP workflow is complete. Live robot-frame pose is available."

    def _registration_tip_output_path(self) -> Path:
        raw = str(self.settings.registration.penprobe_file or "data/tip_cals/generated_penprobe_tip.csv")
        path = Path(raw)
        return path if path.is_absolute() else self.project_root / path

    def _pivot_runtime_root(self) -> Path:
        raw = str(self.settings.experiment.output_dir or "data/experiments")
        path = Path(raw)
        root = path if path.is_absolute() else self.project_root / path
        return root / "pivot"

    def _pivot_run_output_root(self) -> Path:
        return self._pivot_runtime_root() / "runs"

    def _pivot_capture_output_root(self) -> Path:
        return self._pivot_runtime_root() / "captures"

    def _tool_is_visible(self, tool_id: str) -> bool:
        snapshot = self.tracking_service.get_snapshot()
        tool = snapshot.tools.get(tool_id)
        return bool(tool is not None and tool.tracking_state == "tracked")

    @staticmethod
    def _as_float(value) -> float | None:
        return None if value in (None, "") else float(value)

    @staticmethod
    def _render_live_pose_status(snapshot) -> str:
        if snapshot.tip_pose_status == "ok" and snapshot.tracker_data_stale:
            if snapshot.tracker_data_age_s is not None:
                return f"stale_tracker_data ({snapshot.tracker_data_age_s:.3f} s)"
            return "stale_tracker_data"
        return snapshot.tip_pose_status

    def _tip_file_preview(self, path: Path) -> str:
        if not path.exists():
            return "No tip file saved yet."
        try:
            raw = path.read_text(encoding="utf-8").strip()
        except Exception as exc:
            return f"Could not read tip file: {exc}"
        return raw or "Tip file is empty."

    def _validation_summary(self, snapshot) -> str:
        if self._last_validation_report_path is None:
            return "No tracker validation report saved yet."
        report = self._last_validation_report
        if self.state.validation_passed:
            return (
                f"Validation passed. backend={snapshot.selected_backend_name or snapshot.backend_identity}, "
                f"fps={snapshot.effective_frame_rate_hz}, transforms_valid={self.state.transforms_valid}, "
                f"0A tracked={self.state.tool_0a_visible}, 0B tracked={self.state.tool_0b_visible}."
            )
        if report is not None and getattr(report, "tracker_verdict", "") == "operational_with_warning":
            return str(getattr(report, "tracker_verdict_message", "Tracker is operational with warning."))
        if report is not None:
            return (
                f"Validation failed. startup_state={getattr(report, 'startup_state', 'unknown')}, "
                f"warmup_invalid={getattr(report, 'warmup_invalid_frame_count_by_tool', {})}, "
                f"first_valid_frame_latency_s={getattr(report, 'first_valid_frame_latency_s', None)}, "
                f"0A={self.state.tool_0a_status}, 0B={self.state.tool_0b_status}."
            )
        return (
            f"Validation failed. backend={snapshot.selected_backend_name or snapshot.backend_identity}, "
            f"state={snapshot.canonical_state}, transforms_valid={self.state.transforms_valid}, "
            f"0A={self.state.tool_0a_status}, 0B={self.state.tool_0b_status}."
        )

    def _validation_lines(self, snapshot) -> list[str]:
        lines = [
            f"backend={snapshot.selected_backend_name or snapshot.backend_identity}",
            f"tracker_port={snapshot.port}",
            f"state={snapshot.canonical_state} ({snapshot.connection_state})",
            f"frames={snapshot.unique_frames_observed}, fps={snapshot.effective_frame_rate_hz}",
            f"freshness_s={snapshot.tracker_data_age_s}",
            f"tool_ids raw={snapshot.raw_live_tool_ids} normalized={snapshot.normalized_live_tool_ids}",
            f"0A={self.state.tool_0a_status}, 0B={self.state.tool_0b_status}",
        ]
        report = self._last_validation_report
        if report is not None:
            lines.extend(
                [
                    f"tracker_verdict={getattr(report, 'tracker_verdict', 'unknown')}",
                    f"startup_state={getattr(report, 'startup_state', 'unknown')}",
                    "first_frame_latency_s="
                    f"{getattr(report, 'first_frame_latency_s', None)}, "
                    f"first_valid_frame_latency_s={getattr(report, 'first_valid_frame_latency_s', None)}",
                    f"warmup_invalid={getattr(report, 'warmup_invalid_frame_count_by_tool', {})}",
                    f"warmup_nonfinite={getattr(report, 'warmup_nonfinite_invalid_frame_count_by_tool', {})}",
                ]
            )
        lines.extend(self._transform_debug_lines(snapshot))
        if self._last_validation_report_path is not None:
            lines.append(f"report={self._last_validation_report_path}")
        if snapshot.last_error:
            lines.append(f"last_error={snapshot.last_error}")
        return lines

    def _pivot_summary(
        self,
        *,
        measurement_status: dict[str, object],
        sample_count: int,
        motion_span_deg: float | None,
        pivot_collection_active: bool,
        pivot_pending_accept: bool,
    ) -> str:
        motion_text = (
            f"motion_span={motion_span_deg:.1f} deg"
            if motion_span_deg is not None
            else "motion_span=collect more poses"
        )
        format_text = ""
        if self.state.pivot_input_format:
            format_text = (
                f" format={self.state.pivot_input_format}, usable_0B_rows={self.state.pivot_input_usable_rows}, "
                f"rejected_rows={self.state.pivot_input_rejected_row_count}."
            )
        if pivot_collection_active:
            return (
                f"Collecting 0B pivot samples: samples={sample_count}, {motion_text}, "
                "stop when the tool has been swept through a wide range of orientations."
            )
        if pivot_pending_accept:
            return (
                f"Pivot solved. RMSE={self._last_pivot_metrics.get('rmse_mm')} mm, "
                f"used={self._last_pivot_metrics.get('sample_count_used', 0)}, "
                f"rejected={self._last_pivot_metrics.get('sample_count_rejected', 0)}.{format_text} "
                "Accept the staged tip file to unblock registration."
            )
        if self.state.pivot_status == "solve_failed" and self._last_pivot_metrics:
            rejected_preview = "; ".join(self.state.pivot_input_rejected_rows[:3])
            suffix = f" Rejected: {rejected_preview}" if rejected_preview else ""
            return (
                f"Pivot solve failed.{format_text} "
                f"Check dataset {self.state.pivot_capture_dataset_path or 'path unavailable'} and retry.{suffix}"
            )
        if self._last_pivot_metrics:
            return (
                f"Accepted pivot result. RMSE={self._last_pivot_metrics.get('rmse_mm')} mm, "
                f"used={self._last_pivot_metrics.get('sample_count_used', 0)}, "
                f"rejected={self._last_pivot_metrics.get('sample_count_rejected', 0)}.{format_text} "
                f"tip_file={measurement_status.get('path') or self.state.pivot_tip_path}"
            )
        if sample_count > 0:
            return f"Collection stopped with {sample_count} sample(s) and {motion_text}. Solve to review the fit."
        if measurement_status.get("ready"):
            return f"Using existing tip file: {measurement_status.get('path') or self.state.pivot_tip_path}"
        return "No successful pivot calibration has been recorded in this session."

    def _transform_summary(
        self,
        measurement_status: dict[str, object],
        *,
        latest_registration_summary: dict[str, object],
    ) -> list[str]:
        capture_tip_vector = measurement_status.get("tip_vector_mm")
        capture_tip_text = (
            ", ".join(f"{float(value):.3f}" for value in capture_tip_vector)
            if isinstance(capture_tip_vector, list) and len(capture_tip_vector) == 3
            else "unknown"
        )
        lines = [
            "Registration capture uses T_tool0B_tip during sample capture, before point averaging and before solving T_robot_aurora.",
            f"Current 0B capture tip source: {measurement_status.get('source')} | path={measurement_status.get('path') or 'none'} | vector_mm=[{capture_tip_text}]",
            "Saved registration defines T_robot_aurora, which maps Aurora tracker coordinates into the robot/body frame.",
            "Live robot-frame pose uses T_robot_tip = T_robot_aurora @ T_aurora_coil @ T_coil_tip for tool 0A.",
        ]
        capture_tip_provenance = dict(latest_registration_summary.get("capture_tip_provenance", {}) or {})
        if capture_tip_provenance:
            lines.append(
                "Saved registration capture tip provenance: "
                f"path={capture_tip_provenance.get('path') or 'none'}, "
                f"sha256={capture_tip_provenance.get('file_sha256') or 'none'}, "
                f"applied_before_solving={capture_tip_provenance.get('offset_applied_before_solving')}"
            )
        live_pose_tip_transform = dict(latest_registration_summary.get("live_pose_tip_transform", {}) or {})
        if live_pose_tip_transform:
            lines.append(
                "Saved registration live-pose tip transform: "
                f"source={live_pose_tip_transform.get('source')}, "
                f"assumption={live_pose_tip_transform.get('assumption') or 'none'}"
            )
        elif self.state.latest_registration_path:
            lines.append("Saved registration is missing explicit tip provenance metadata.")
        return lines

    def _transform_debug_lines(self, snapshot) -> list[str]:
        debug_root = dict(snapshot.backend_details or {}).get("ndi_transform_debug", {})
        debug_tools = dict(debug_root.get("tool_transform_debug", {}) or {})
        lines: list[str] = []
        for tool_id in (self.settings.registration.coil_tool_id, self.settings.registration.capture_tool_id):
            tool = snapshot.tools.get(tool_id)
            debug_entry = dict(debug_tools.get(tool_id, {}) or {})
            if tool is None or not debug_entry:
                continue
            failure_stage = debug_entry.get("failure_stage") or "ok"
            det = debug_entry.get("rotation_determinant")
            det_text = f", det={det:.6f}" if isinstance(det, (int, float)) else ""
            reason = str(debug_entry.get("invalid_reason") or tool.status)
            payload = str(debug_entry.get("parse_mode") or debug_entry.get("raw_payload_summary") or "unknown_payload")
            lines.append(f"{tool_id}_transform={tool.tracking_state} stage={failure_stage} payload={payload}{det_text}")
            lines.append(f"{tool_id}_reason={reason}")
        return lines

    def _transform_stage_message(self) -> str:
        if self.state.transforms_valid:
            return "0A and 0B transforms are valid, rigid, and ready for pivot calibration."
        details: list[str] = []
        for tool_id, status in (
            (self.settings.registration.coil_tool_id, self.state.tool_0a_status),
            (self.settings.registration.capture_tool_id, self.state.tool_0b_status),
        ):
            if "invalid_transform" in str(status):
                details.append(f"{tool_id}: {status}")
        if details:
            return "; ".join(details)
        return (
            f"0A={self.state.tool_0a_status}, 0B={self.state.tool_0b_status}. "
            "Transforms must be tracked and rigid-valid before pivot calibration."
        )

    def _registration_blockers(self, measurement_status: dict[str, object]) -> list[str]:
        blockers: list[str] = []
        if not self.state.tracker_operational:
            blockers.append("Tracker validation must at least be operational before registration.")
        if not self.state.transforms_valid:
            blockers.append("0A and 0B must both stay tracked with valid rigid transforms.")
        if self.state.pivot_collection_active:
            blockers.append("Stop pivot collection before beginning registration.")
        if self.state.pivot_pending_accept:
            blockers.append("Accept or reset the staged pivot tip file before registration.")
        if not self.state.measurement_point_ready:
            blockers.append(str(measurement_status.get("message") or "Pen-probe tip file is not ready."))
        if not self.state.registration_selection_ready:
            blockers.append("Select four unique model landmarks before starting registration.")
        if self.registration_controller.state.pending_accept:
            blockers.append("Save or restart the currently solved registration before beginning a new one.")
        return blockers

    def _pivot_start_blockers(self, snapshot) -> list[str]:
        blockers: list[str] = []
        if not self._validation_operational:
            blockers.append("Run tracker validation until it is at least operational before pivot calibration.")
        if snapshot.connection_state in {"disconnected", "stopped", "error"} and snapshot.canonical_state not in {"mock"}:
            blockers.append("Tracker is not connected.")
        if not self._tool_is_visible(self.settings.registration.capture_tool_id):
            blockers.append("Tool 0B must be visible and tracked before pivot calibration.")
        if not self._tool_is_visible(self.settings.registration.coil_tool_id):
            blockers.append("Tool 0A must be visible and tracked before pivot calibration.")
        tool_0a = snapshot.tools.get(self.settings.registration.coil_tool_id)
        tool_0b = snapshot.tools.get(self.settings.registration.capture_tool_id)
        if tool_0a is not None and tool_0a.tracking_state == "invalid":
            blockers.append("Tool 0A transform is invalid.")
        if tool_0b is not None and tool_0b.tracking_state == "invalid":
            blockers.append("Tool 0B transform is invalid.")
        return blockers

    def _require_pivot_start_ready(self) -> None:
        blockers = self._pivot_start_blockers(self.tracking_service.get_snapshot())
        if blockers:
            raise RuntimeError(blockers[0])
        if self._pivot_collection_active:
            raise RuntimeError("Pivot collection is already running.")

    def _stop_pivot_collection(self, *, silent: bool) -> int:
        thread = self._pivot_thread
        self._pivot_stop_event.set()
        if thread is not None and thread.is_alive():
            thread.join(timeout=1.0)
        with self._pivot_lock:
            self._pivot_thread = None
            self._pivot_collection_active = False
            sample_count = len(self._pivot_samples)
            if sample_count > 0 and self._pivot_collection_status == "collecting":
                self._pivot_collection_status = "collecting_stopped"
            elif self._pivot_collection_status == "collecting":
                self._pivot_collection_status = "not_run"
            if not silent and sample_count == 0:
                self._pivot_live_tool_status = "No tracked 0B samples were collected."
            return sample_count

    def _collect_pivot_samples_loop(self) -> None:
        while not self._pivot_stop_event.is_set():
            tool = self.tracking_service.get_latest_tool(self.settings.registration.capture_tool_id)
            if tool is None:
                with self._pivot_lock:
                    self._pivot_live_tool_tracked = False
                    self._pivot_live_tool_status = "Tool 0B snapshot unavailable."
                time.sleep(self.PIVOT_SAMPLE_PERIOD_S)
                continue
            if tool.tracking_state != "tracked":
                with self._pivot_lock:
                    self._pivot_live_tool_tracked = False
                    self._pivot_live_tool_status = str(tool.status or tool.tracking_state)
                time.sleep(self.PIVOT_SAMPLE_PERIOD_S)
                continue
            if tool.quaternion_wxyz is None or tool.translation_mm is None:
                with self._pivot_lock:
                    self._pivot_live_tool_tracked = False
                    self._pivot_live_tool_status = "Tool 0B is tracked but missing pose data."
                time.sleep(self.PIVOT_SAMPLE_PERIOD_S)
                continue
            if tool.frame_number is None and tool.last_update_utc is None:
                with self._pivot_lock:
                    self._pivot_live_tool_tracked = False
                    self._pivot_live_tool_status = "Tool 0B is tracked but missing a frame key."
                time.sleep(self.PIVOT_SAMPLE_PERIOD_S)
                continue

            sample_key = (tool.frame_number, tool.last_update_utc)
            with self._pivot_lock:
                self._pivot_live_tool_tracked = True
                self._pivot_live_tool_status = "0B tracked"
                if sample_key == self._pivot_last_sample_key:
                    pass
                else:
                    self._pivot_last_sample_key = sample_key
                    self._pivot_samples.append(
                        PivotCollectionSample(
                            frame_number=tool.frame_number,
                            captured_at_utc=tool.last_update_utc,
                            quaternion_wxyz=tuple(float(v) for v in tool.quaternion_wxyz),
                            translation_mm=tuple(float(v) for v in tool.translation_mm),
                        )
                    )
                    self._pivot_motion_span_deg = self._compute_motion_span_deg(self._pivot_samples)
            time.sleep(self.PIVOT_SAMPLE_PERIOD_S)

    def _pivot_samples_for_solve(self) -> list[PivotCollectionSample]:
        if self._pivot_collection_active:
            raise RuntimeError("Stop pivot collection before solving the pivot calibration.")
        with self._pivot_lock:
            return list(self._pivot_samples)

    def _write_pivot_capture_dataset(self, *, samples: list[PivotCollectionSample], stamp: str) -> Path:
        output_dir = self._pivot_capture_output_root()
        output_dir.mkdir(parents=True, exist_ok=True)
        path = output_dir / f"{stamp}_pivot_0B_samples.csv"
        lines = ["tool_id,qw,qx,qy,qz,x,y,z"]
        for sample in samples:
            row = [
                self.settings.registration.capture_tool_id,
                *[f"{float(value):.9f}" for value in sample.quaternion_wxyz],
                *[f"{float(value):.9f}" for value in sample.translation_mm],
            ]
            lines.append(",".join(row))
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return path

    def _record_pivot_result(
        self,
        *,
        result,
        accepted_tip_path: Path,
        capture_dataset_path: Path | None,
        pending_tip_path: Path | None,
        pending_accept: bool,
        status_override: str | None = None,
    ) -> None:
        metrics = result.summary.experiment_metrics if isinstance(result.summary.experiment_metrics, dict) else {}
        with self._pivot_lock:
            self._last_pivot_run_path = result.paths.output_dir
            self._last_pivot_metrics = dict(metrics)
            self._pivot_capture_dataset_path = capture_dataset_path
            self._pivot_pending_tip_path = pending_tip_path
            self._pivot_pending_accept = pending_accept
            self._pivot_collection_status = (
                str(status_override)
                if status_override is not None
                else ("review_ready" if pending_accept else "accepted")
            )
            self._last_pivot_metrics["accepted_tip_file"] = str(accepted_tip_path)

    @staticmethod
    def _compute_motion_span_deg(samples: list[PivotCollectionSample]) -> float | None:
        if len(samples) < 2:
            return None
        axes: list[np.ndarray] = []
        for sample in samples:
            rotation = quat_wxyz_to_rotmat(sample.quaternion_wxyz)
            axes.append(np.asarray(rotation[:, 2], dtype=float))
        axis_matrix = np.vstack(axes)
        dot_products = np.clip(axis_matrix @ axis_matrix.T, -1.0, 1.0)
        min_dot = float(np.min(dot_products))
        return float(np.degrees(np.arccos(min_dot)))

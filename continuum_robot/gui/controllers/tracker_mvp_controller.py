"""Focused tracker-first MVP controller for tracker, pivot, and registration bring-up."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
import json
from pathlib import Path

from continuum_robot.config.settings import Settings
from continuum_robot.hardware.serial_ports import SerialPortInfo, discover_serial_ports
from continuum_robot.tracking.benchmarking import TrackerBenchmarkThresholds
from continuum_robot.tracking.diagnostics import build_tracking_diagnostics_report


@dataclass
class WorkflowStepState:
    """One operator-facing workflow step."""

    index: int
    label: str
    status: str
    message: str


@dataclass
class TrackerMvpViewState:
    """UI-facing tracker MVP workflow state."""

    tracker_port: str
    available_ports: list[SerialPortInfo] = field(default_factory=list)
    tracker_connected: bool = False
    tracker_healthy: bool = False
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
    pivot_run_path: str = ""
    pivot_tip_path: str = ""
    pivot_tip_exists: bool = False
    pivot_tip_preview: str = "No tip file loaded."
    pivot_rmse_mm: float | None = None
    pivot_sample_count_total: int = 0
    pivot_sample_count_used: int = 0
    pivot_sample_count_rejected: int = 0
    measurement_point_ready: bool = False
    measurement_point_source: str = ""
    measurement_point_message: str = ""
    selected_landmarks: list[str] = field(default_factory=list)
    registration_selection_ready: bool = False
    registration_capture_complete: bool = False
    registration_solved: bool = False
    registration_saved: bool = False
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
        self._last_validation_report_path: Path | None = None
        self._last_validation_report = None
        self._last_pivot_run_path: Path | None = None
        self._last_pivot_metrics: dict[str, object] = {}
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
        try:
            self.tracking_service.stop()
            self._validation_passed = False
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
            self._last_validation_report_path = report_path
            self._last_validation_report = report
        except Exception as exc:
            self._validation_passed = False
            self._last_validation_report = None
            self.refresh()
            self.state.last_error = str(exc)
            self.state.status_message = f"Tracker validation failed: {exc}"
            raise
        self.refresh()
        self.state.last_error = None
        self.state.status_message = (
            f"Tracker validation {'passed' if self._validation_passed else 'failed'}. Saved report to {report_path.name}."
        )
        return report_path

    def run_pivot_calibration(self) -> Path:
        snapshot = self.tracking_service.get_snapshot()
        if not self._validation_passed:
            raise RuntimeError("Run tracker validation successfully before pivot calibration.")
        if snapshot.connection_state in {"disconnected", "stopped", "error"} and snapshot.canonical_state not in {"mock"}:
            raise RuntimeError("Tracker is not connected. Connect and validate tracker before pivot calibration.")
        if not self._tool_is_visible(self.settings.registration.capture_tool_id):
            raise RuntimeError("Tool 0B must be visible and tracked before pivot calibration.")

        tip_output = self._registration_tip_output_path()
        result = self.experiment_runner.run_experiment(
            "pivot_calibration",
            config={
                "tool_id": self.settings.registration.capture_tool_id,
                "dry_run": False,
                "sample_count": 80,
                "sample_period_s": 0.02,
                "std_dev_threshold": 3.0,
                "min_samples": 12,
                "output_tip_file": str(tip_output),
            },
            operator_notes="tracker_mvp_gui",
        )
        if not result.success:
            self.refresh()
            self.state.last_error = result.message
            self.state.status_message = f"Pivot calibration failed: {result.message}"
            raise RuntimeError(result.message)

        metrics = result.summary.experiment_metrics if isinstance(result.summary.experiment_metrics, dict) else {}
        self._last_pivot_run_path = result.paths.output_dir
        self._last_pivot_metrics = dict(metrics)
        self.registration_service.refresh_measurement_point_geometry()
        self.refresh()
        self.state.last_error = None
        self.state.status_message = (
            f"Pivot calibration saved tip file to {tip_output}. "
            f"RMSE={float(metrics.get('rmse_mm', 0.0)):.3f} mm."
        )
        return result.paths.output_dir

    def refresh(self) -> TrackerMvpViewState:
        snapshot = self.tracking_service.get_snapshot()
        registration_snapshot = self.registration_service.get_snapshot()
        measurement_status = self.registration_service.get_measurement_point_status(refresh=True)
        capture_tool_id = self.settings.registration.capture_tool_id
        coil_tool_id = self.settings.registration.coil_tool_id
        tool_0a = snapshot.tools.get(coil_tool_id)
        tool_0b = snapshot.tools.get(capture_tool_id)
        tool_0a_detected = bool(coil_tool_id in snapshot.normalized_live_tool_ids or (tool_0a is not None and tool_0a.tracking_state != "unknown"))
        tool_0b_detected = bool(capture_tool_id in snapshot.normalized_live_tool_ids or (tool_0b is not None and tool_0b.tracking_state != "unknown"))

        tracker_connected = bool(
            snapshot.backend_connected
            or snapshot.canonical_state in {"mock", "streaming_healthy", "streaming_degraded"}
            or snapshot.connection_state in {"tracking", "starting", "connecting"}
        )
        tracker_healthy = bool(
            tracker_connected
            and snapshot.unique_frames_observed > 0
            and not snapshot.tracker_data_stale
            and (snapshot.effective_frame_rate_hz is None or snapshot.effective_frame_rate_hz >= self.thresholds.min_effective_fps)
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
            f"Saved registration FRE={registration_snapshot.fre_mm:.3f} mm."
            if registration_snapshot.latest_accepted_path and registration_snapshot.fre_mm is not None
            else (
                "Accepted registration is saved."
                if registration_snapshot.latest_accepted_path
                else "No accepted registration saved."
            )
        )
        live_pose_ready = bool(snapshot.tip_pose_status == "ok" and snapshot.T_robot_tip is not None)
        live_tip_position = None
        if live_pose_ready and snapshot.T_robot_tip is not None:
            live_tip_position = (
                float(snapshot.T_robot_tip[0][3]),
                float(snapshot.T_robot_tip[1][3]),
                float(snapshot.T_robot_tip[2][3]),
            )

        pivot_tip_path = Path(measurement_status["path"]) if measurement_status.get("path") else self._registration_tip_output_path()
        pivot_tip_exists = pivot_tip_path.exists()
        pivot_tip_preview = self._tip_file_preview(pivot_tip_path)

        self.state.tracker_port = snapshot.port or self.state.tracker_port
        self.state.tracker_connected = tracker_connected
        self.state.tracker_healthy = tracker_healthy
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
        self.state.pivot_status = str(self._last_pivot_metrics.get("status", "not_run"))
        self.state.pivot_run_path = str(self._last_pivot_run_path or "")
        self.state.pivot_tip_path = str(pivot_tip_path)
        self.state.pivot_tip_exists = pivot_tip_exists
        self.state.pivot_tip_preview = pivot_tip_preview
        self.state.pivot_rmse_mm = self._as_float(self._last_pivot_metrics.get("rmse_mm"))
        self.state.pivot_sample_count_total = int(self._last_pivot_metrics.get("sample_count_total", 0) or 0)
        self.state.pivot_sample_count_used = int(self._last_pivot_metrics.get("sample_count_used", 0) or 0)
        self.state.pivot_sample_count_rejected = int(self._last_pivot_metrics.get("sample_count_rejected", 0) or 0)
        self.state.pivot_summary = self._pivot_summary(measurement_status, pivot_tip_exists)
        self.state.measurement_point_ready = bool(measurement_status["ready"])
        self.state.measurement_point_source = str(measurement_status["source"])
        self.state.measurement_point_message = str(measurement_status["message"])
        self.state.selected_landmarks = list(self.registration_controller.state.selected_model_labels)
        self.state.registration_selection_ready = self.registration_controller.selection_is_ready()
        self.state.registration_capture_complete = bool(
            registration_snapshot.health.state in {"ready_to_solve", "solved", "accepted"}
            or registration_snapshot.pending_accept
        )
        self.state.registration_solved = bool(registration_snapshot.pending_accept or registration_snapshot.health.state in {"solved", "accepted"})
        self.state.registration_saved = bool(registration_snapshot.latest_accepted_path)
        self.state.registration_fre_mm = registration_snapshot.fre_mm
        self.state.latest_registration_path = str(registration_snapshot.latest_accepted_path or "")
        self.state.latest_registration_status = latest_registration_status
        self.state.live_pose_ready = live_pose_ready
        self.state.live_tip_status = snapshot.tip_pose_status
        self.state.live_tip_position_mm = live_tip_position
        self.state.transform_summary_lines = self._transform_summary(measurement_status)
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
        _step(2, "Validate tracker health", self.state.validation_passed, self.state.validation_summary, blocked=not self.state.tracker_connected)
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
        pivot_ready = bool(self.state.pivot_status == "success" or (self.state.measurement_point_ready and self.state.pivot_tip_exists))
        _step(5, "Run pivot calibration on 0B", pivot_ready, self.state.pivot_summary, blocked=not self.state.validation_passed or not self.state.transforms_valid or not self.state.tool_0b_visible)
        _step(6, "Save tip file", self.state.pivot_tip_exists and self.state.measurement_point_ready, f"Tip file: {self.state.pivot_tip_path or 'missing'}", blocked=not self.state.tool_0b_visible)
        _step(7, "Select 4 registration landmarks", self.state.registration_selection_ready, f"Selected: {self.state.selected_landmarks}", blocked=not self.state.measurement_point_ready)
        _step(8, "Capture registration samples", self.state.registration_capture_complete, self.registration_controller.state.status_message, blocked=not self.state.registration_selection_ready or not self.state.measurement_point_ready)
        _step(9, "Solve registration", self.state.registration_solved, self.state.latest_registration_status if self.state.registration_saved else "Solve once all four selected points are complete.", blocked=not self.state.registration_capture_complete)
        _step(10, "Save accepted registration", self.state.registration_saved, self.state.latest_registration_path or "Save the solved registration to persist the artifact.", blocked=not self.state.registration_solved)
        _step(11, "Confirm live robot-frame pose availability", self.state.live_pose_ready, f"Tip pose status: {self.state.live_tip_status}", blocked=not self.state.registration_saved)
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

    def _tool_is_visible(self, tool_id: str) -> bool:
        snapshot = self.tracking_service.get_snapshot()
        tool = snapshot.tools.get(tool_id)
        return bool(tool is not None and tool.tracking_state == "tracked")

    @staticmethod
    def _as_float(value) -> float | None:
        return None if value in (None, "") else float(value)

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

    def _pivot_summary(self, measurement_status: dict[str, object], tip_exists: bool) -> str:
        if self._last_pivot_metrics:
            return (
                f"RMSE={self._last_pivot_metrics.get('rmse_mm')} mm, "
                f"used={self._last_pivot_metrics.get('sample_count_used', 0)}, "
                f"rejected={self._last_pivot_metrics.get('sample_count_rejected', 0)}, "
                f"tip_file={self.state.pivot_tip_path or measurement_status.get('path')}"
            )
        if tip_exists and measurement_status.get("ready"):
            return f"Using existing tip file: {measurement_status.get('path') or self.state.pivot_tip_path}"
        return "No successful pivot calibration has been recorded in this session."

    def _transform_summary(self, measurement_status: dict[str, object]) -> list[str]:
        return [
            "Tip file -> defines T_tool0B_tip, the pen-tip offset used while capturing registration landmarks.",
            "Saved registration -> defines T_robot_aurora, which maps Aurora tracker coordinates into the robot/body frame.",
            "Live robot-frame pose -> requires saved registration plus live 0A tracking: T_robot_tip = T_robot_aurora @ T_aurora_coil @ T_coil_tip.",
            f"Current tip geometry source: {measurement_status.get('source')}",
        ]

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

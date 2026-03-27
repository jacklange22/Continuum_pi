"""Experiment tab controller for file loading and run execution."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import threading


@dataclass
class ExperimentViewState:
    """UI-facing experiment workflow state."""

    loaded_file: str = ""
    point_count: int = 0
    status_message: str = "No experiment loaded."
    last_error: str | None = None
    run_active: bool = False
    progress_current: int = 0
    progress_total: int = 0
    last_output_path: str | None = None
    prerequisites_ok: bool = False
    prerequisite_message: str = "Neutral calibration and registration are required."


class ExperimentController:
    """Owns experiment loading, execution, and run logging actions."""

    def __init__(
        self,
        experiment_loader,
        experiment_runner,
        registration_path: Path,
        servo_service,
        tracker_manager,
    ) -> None:
        self.experiment_loader = experiment_loader
        self.experiment_runner = experiment_runner
        self.registration_path = registration_path
        self.servo_service = servo_service
        self.tracker_manager = tracker_manager
        self.points = []
        self.state = ExperimentViewState()
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._lock = threading.Lock()
        self.refresh_prerequisites()

    def load_file(self, path: Path) -> None:
        try:
            self.points = self.experiment_loader.load_csv(path)
            self.state.loaded_file = str(path)
            self.state.point_count = len(self.points)
            self.state.status_message = f"Loaded {len(self.points)} experiment point(s)."
            self.state.last_error = None
        except Exception as exc:
            self.state.last_error = str(exc)
            self.state.status_message = f"Experiment load failed: {exc}"
            raise
        finally:
            self.refresh_prerequisites()

    def refresh_prerequisites(self) -> ExperimentViewState:
        has_neutral = bool(self.servo_service.load_neutral_setpoints())
        has_registration = self.registration_path.exists()
        tracker_state = self.tracker_manager.get_state_snapshot()
        tracker_connected = tracker_state.connection_state == "tracking"
        tool_0a = tracker_state.tools.get("0A")
        tool_0a_valid = bool(tool_0a and tool_0a.valid)
        servo_connected = self.servo_service.is_connected
        self.state.prerequisites_ok = (
            has_neutral
            and has_registration
            and bool(self.points)
            and servo_connected
            and tracker_connected
            and tool_0a_valid
        )
        missing = []
        if not self.points:
            missing.append("experiment file")
        if not has_neutral:
            missing.append("neutral calibration")
        if not has_registration:
            missing.append("registration")
        if not servo_connected:
            missing.append("OpenRB/DYNAMIXEL connection")
        if not tracker_connected:
            missing.append("tracker connection")
        elif not tool_0a_valid:
            missing.append("valid tool 0A sample")
        self.state.prerequisite_message = (
            "Ready to run."
            if self.state.prerequisites_ok
            else f"Missing prerequisites: {', '.join(missing)}."
        )
        return self.state

    def run(self) -> None:
        self.refresh_prerequisites()
        if not self.state.prerequisites_ok:
            raise RuntimeError(self.state.prerequisite_message)
        if self.state.run_active:
            raise RuntimeError("Experiment is already running")

        self._stop_event.clear()
        self.state.run_active = True
        self.state.progress_current = 0
        self.state.progress_total = sum(max(1, point.repeat) for point in self.points)
        self.state.status_message = "Experiment running."
        self.state.last_error = None

        def _worker() -> None:
            try:
                summary = self.experiment_runner.run(
                    self.points,
                    progress_callback=self._on_progress,
                    stop_requested=self._stop_event.is_set,
                )
                with self._lock:
                    self.state.last_output_path = str(summary.output_path)
                    self.state.status_message = summary.message
                    self.state.last_error = None
            except Exception as exc:
                with self._lock:
                    self.state.last_error = str(exc)
                    self.state.status_message = f"Experiment failed: {exc}"
            finally:
                with self._lock:
                    self.state.run_active = False
                self.refresh_prerequisites()

        self._thread = threading.Thread(target=_worker, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        self.state.status_message = "Stop requested."

    def _on_progress(self, current: int, total: int, _row: dict) -> None:
        with self._lock:
            self.state.progress_current = current
            self.state.progress_total = total
            self.state.status_message = f"Collected {current}/{total} row(s)."

    def shutdown(self, timeout_s: float = 2.0) -> None:
        self._stop_event.set()
        thread = self._thread
        if thread is not None and thread.is_alive():
            thread.join(timeout=timeout_s)

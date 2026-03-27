"""Lifecycle manager for the C++ tracker_bridge process and socket stream."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import copy
import logging
import subprocess
import threading
import time

from continuum_robot.tracking.tracker_protocol import (
    TrackerStatusMessage,
    TrackerTransformMessage,
    parse_tracker_json_line,
)
from continuum_robot.tracking.tracker_socket_client import TrackerSocketClient


@dataclass
class TrackerToolState:
    """Latest known state for one tracked tool."""

    tool_id: str
    frame_number: int
    valid: bool
    status: str
    quaternion: tuple[float, float, float, float]
    translation_mm: tuple[float, float, float]
    quality: float | None
    timestamp: str


@dataclass
class TrackerRuntimeState:
    """Shared runtime state for GUI/diagnostics polling."""

    connection_state: str = "disconnected"
    socket_connected: bool = False
    bridge_running: bool = False
    latest_frame_number: int | None = None
    latest_timestamp: str | None = None
    last_status_message: str = ""
    last_error: str | None = None
    tools: dict[str, TrackerToolState] = field(default_factory=dict)


class TrackerServiceManager:
    """Starts tracker_bridge and consumes tracker JSON events from Unix socket."""

    def __init__(
        self,
        bridge_executable: Path,
        socket_path: Path,
        aurora_port: str,
        poll_ms: int = 20,
    ) -> None:
        self.bridge_executable = bridge_executable
        self.socket_path = socket_path
        self.aurora_port = aurora_port
        self.poll_ms = poll_ms

        self._process: subprocess.Popen | None = None
        self._client = TrackerSocketClient(socket_path)
        self._stop_event = threading.Event()
        self._lock = threading.Lock()
        self._receiver_thread: threading.Thread | None = None
        self._stdout_thread: threading.Thread | None = None
        self._stderr_thread: threading.Thread | None = None
        self._state = TrackerRuntimeState()

    def start(self) -> None:
        """Launch bridge process and begin receiving tracker events."""
        if self.is_alive():
            return

        self._stop_event.clear()
        self._set_state(connection_state="starting", last_error=None, last_status_message="Starting tracker bridge")
        try:
            self._launch_bridge_process()
        except Exception as exc:
            self._set_state(
                bridge_running=False,
                socket_connected=False,
                connection_state="error",
                last_error=str(exc),
                last_status_message=f"Tracker start failed: {exc}",
            )
            raise

        self._receiver_thread = threading.Thread(target=self._receiver_loop, daemon=True)
        self._receiver_thread.start()

        if self._process is not None:
            self._stdout_thread = threading.Thread(
                target=self._pipe_log_loop,
                args=(self._process.stdout, logging.INFO, "tracker_bridge"),
                daemon=True,
            )
            self._stdout_thread.start()
            self._stderr_thread = threading.Thread(
                target=self._pipe_log_loop,
                args=(self._process.stderr, logging.WARNING, "tracker_bridge"),
                daemon=True,
            )
            self._stderr_thread.start()

    def stop(self, timeout_s: float = 3.0) -> None:
        """Stop socket receiver and bridge process."""
        self._stop_event.set()
        self._client.close()

        if self._receiver_thread is not None:
            self._receiver_thread.join(timeout=timeout_s)
            self._receiver_thread = None
        if self._stdout_thread is not None:
            self._stdout_thread.join(timeout=timeout_s)
            self._stdout_thread = None
        if self._stderr_thread is not None:
            self._stderr_thread.join(timeout=timeout_s)
            self._stderr_thread = None

        if self._process is not None:
            if self._process.poll() is None:
                self._process.terminate()
                try:
                    self._process.wait(timeout=timeout_s)
                except subprocess.TimeoutExpired:
                    self._process.kill()
                    self._process.wait(timeout=timeout_s)
            self._process = None

        self._set_state(
            bridge_running=False,
            socket_connected=False,
            connection_state="disconnected",
            last_status_message="Tracker disconnected",
        )

    def is_alive(self) -> bool:
        return self._process is not None and self._process.poll() is None

    def get_state_snapshot(self) -> TrackerRuntimeState:
        with self._lock:
            return copy.deepcopy(self._state)

    def get_latest_tool(self, tool_id: str) -> TrackerToolState | None:
        with self._lock:
            tool = self._state.tools.get(tool_id)
            return copy.deepcopy(tool) if tool is not None else None

    def _launch_bridge_process(self) -> None:
        if not self.aurora_port:
            raise RuntimeError("Aurora port is empty. Configure aurora_port before starting tracker_bridge.")
        if not self.bridge_executable.exists():
            raise FileNotFoundError(f"tracker_bridge executable not found: {self.bridge_executable}")

        cmd = [
            str(self.bridge_executable),
            "--aurora-port",
            self.aurora_port,
            "--socket-path",
            str(self.socket_path),
            "--poll-ms",
            str(self.poll_ms),
        ]

        self._process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
        )
        self._set_state(bridge_running=True)

    def _receiver_loop(self) -> None:
        while not self._stop_event.is_set():
            if self._process is not None and self._process.poll() is not None and not self._client.is_connected:
                self._set_state(
                    bridge_running=False,
                    socket_connected=False,
                    connection_state="error",
                    last_error=f"tracker_bridge exited with code {self._process.returncode}",
                )
                return

            try:
                if not self._client.is_connected:
                    self._set_state(connection_state="connecting", last_status_message="Connecting to tracker socket")
                    self._client.connect(timeout_s=1.0)
                    self._set_state(socket_connected=True, connection_state="connected")

                line = self._client.read_line(timeout_s=0.5)
                if line is None:
                    continue

                msg = parse_tracker_json_line(line)
                if isinstance(msg, TrackerStatusMessage):
                    self._apply_status(msg)
                elif isinstance(msg, TrackerTransformMessage):
                    self._apply_transform(msg)
            except Exception as exc:
                self._client.close()
                self._set_state(
                    socket_connected=False,
                    connection_state="reconnecting",
                    last_error=str(exc),
                )
                time.sleep(0.25)

    def _apply_status(self, msg: TrackerStatusMessage) -> None:
        updates: dict = {
            "latest_timestamp": msg.timestamp,
            "last_status_message": msg.message,
        }
        if msg.level == "error":
            updates["connection_state"] = "error"
            updates["last_error"] = msg.message
        elif msg.state in {"connecting", "initialized", "tracking_started"}:
            updates["connection_state"] = msg.state
        elif msg.state == "tracking_stopped":
            updates["connection_state"] = "stopped"
        self._set_state(**updates)

    def _apply_transform(self, msg: TrackerTransformMessage) -> None:
        tool_state = TrackerToolState(
            tool_id=msg.tool_id,
            frame_number=msg.frame_number,
            valid=msg.valid,
            status=msg.status,
            quaternion=msg.quaternion,
            translation_mm=msg.translation_mm,
            quality=msg.quality,
            timestamp=msg.timestamp,
        )

        with self._lock:
            self._state.tools[msg.tool_id] = tool_state
            self._state.latest_frame_number = msg.frame_number
            self._state.latest_timestamp = msg.timestamp
            self._state.last_error = None
            if self._state.connection_state not in {"error", "disconnected"}:
                self._state.connection_state = "tracking"

    def _set_state(self, **updates) -> None:
        with self._lock:
            for key, value in updates.items():
                setattr(self._state, key, value)

    @staticmethod
    def _pipe_log_loop(pipe, level: int, prefix: str) -> None:
        if pipe is None:
            return
        for line in pipe:
            text = line.strip()
            if text:
                logging.log(level, "%s: %s", prefix, text)

"""Python-native Aurora backend built on scikit-surgerynditracker.

This backend owns the live NDITracker lifecycle and exposes the same lightweight
runtime interface used by the rest of the app:

- ``start()``
- ``stop()``
- ``is_alive()``
- ``get_state_snapshot()``
- ``get_latest_tool(tool_id)``

The exact ``NDITracker`` settings contract is version-dependent and is not fully
specified anywhere in this repository. The backend therefore keeps the base
configuration explicit, allows override settings, and surfaces configuration
errors directly instead of guessing silently.
"""

from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timezone
import threading
import time
from typing import Any, Callable

import numpy as np

from continuum_robot.tracking.tracker_service_manager import TrackerRuntimeState, TrackerToolState
from continuum_robot.tracking.transforms import assert_rigid_transform_matrix, rotmat_to_quat_wxyz
from continuum_robot.utils.time_utils import utc_now_iso


def load_ndi_tracker_class():
    """Import and return the NDITracker class with a clear error on failure."""
    candidates = [
        ("sksurgerynditracker.nditracker", "NDITracker"),
        ("sksurgerynditracker", "NDITracker"),
    ]
    last_exc: Exception | None = None
    for module_name, class_name in candidates:
        try:
            module = __import__(module_name, fromlist=[class_name])
            return getattr(module, class_name)
        except Exception as exc:  # pragma: no cover - exercised in tests via injection
            last_exc = exc
    raise RuntimeError(
        "Could not import NDITracker. Install the Python dependency for "
        "scikit-surgerynditracker and verify the import path for your local "
        f"version. Last import error: {last_exc}"
    )


def normalize_tool_id(raw_handle: Any) -> str:
    """Normalize a library handle/port identifier into the app's string form."""
    if raw_handle is None:
        return ""
    if isinstance(raw_handle, bytes):
        text = raw_handle.decode("ascii", errors="replace")
    else:
        text = str(raw_handle)
    text = text.replace("\x00", "").strip().upper()
    if text in {"0A", "0B"}:
        return text
    if text.endswith("0A"):
        return "0A"
    if text.endswith("0B"):
        return "0B"
    return text


def _coerce_list(value: Any) -> list[Any]:
    if isinstance(value, (list, tuple)):
        return list(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    return [value]


def _expand_list(value: Any, target_len: int) -> list[Any]:
    values = _coerce_list(value)
    if target_len <= 0:
        return []
    if len(values) == target_len:
        return values
    if len(values) == 1:
        return values * target_len
    if len(values) < target_len:
        return values + [None] * (target_len - len(values))
    return values[:target_len]


def _timestamp_to_iso(raw_timestamp: Any, fallback_utc: str) -> str:
    if raw_timestamp in (None, ""):
        return fallback_utc
    if isinstance(raw_timestamp, str):
        return raw_timestamp
    if isinstance(raw_timestamp, (int, float)):
        try:
            return datetime.fromtimestamp(float(raw_timestamp), tz=timezone.utc).isoformat(timespec="milliseconds").replace(
                "+00:00", "Z"
            )
        except Exception:
            return fallback_utc
    return fallback_utc


def coerce_transform_matrix(raw_transform: Any, *, tool_id: str) -> np.ndarray:
    """Convert one tracker-library transform payload into a strict 4x4 matrix."""
    matrix = np.asarray(raw_transform, dtype=float)
    if matrix.shape == (16,):
        matrix = matrix.reshape(4, 4)
    elif matrix.shape == (1, 16):
        matrix = matrix.reshape(4, 4)
    elif matrix.shape == (4, 4):
        matrix = matrix.copy()
    else:
        raise ValueError(f"T_aurora_{tool_id} must be 4x4 or length-16, got shape {matrix.shape}")
    assert_rigid_transform_matrix(matrix, f"T_aurora_{tool_id}")
    return matrix


class TrackerBackendNDI:
    """Live Aurora backend backed by ``scikit-surgerynditracker.NDITracker``."""

    backend_identity = "ndi_tracker_python"

    def __init__(
        self,
        aurora_port: str,
        *,
        tracker_type: str = "aurora",
        poll_interval_ms: int = 20,
        reconnect_delay_s: float = 1.0,
        ports_to_probe: list[Any] | None = None,
        settings_overrides: dict[str, Any] | None = None,
        tracker_factory: Callable[[dict[str, Any]], Any] | None = None,
        expected_tool_ids: tuple[str, ...] = ("0A", "0B"),
    ) -> None:
        self.aurora_port = aurora_port
        self.tracker_type = str(tracker_type)
        self.poll_interval_ms = max(1, int(poll_interval_ms))
        self.reconnect_delay_s = max(0.1, float(reconnect_delay_s))
        self.ports_to_probe = list(ports_to_probe or [])
        self.settings_overrides = dict(settings_overrides or {})
        self.expected_tool_ids = tuple(str(v).upper() for v in expected_tool_ids)
        self._tracker_factory = tracker_factory

        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._tracker = None
        self._synthetic_frame_number = 0
        self._state = TrackerRuntimeState(
            connection_state="disconnected",
            backend_running=False,
            backend_connected=False,
            bridge_running=False,
            socket_connected=False,
            latest_frame_number=None,
            latest_timestamp=None,
            last_status_message="NDI tracker backend idle",
            last_error=None,
            tools={},
        )

    def start(self) -> None:
        if self.is_alive():
            return
        self._stop_event.clear()
        with self._lock:
            self._state.connection_state = "starting"
            self._state.backend_running = True
            self._state.backend_connected = False
            self._state.bridge_running = False
            self._state.socket_connected = False
            self._state.last_error = None
            self._state.last_status_message = "Starting Python NDI tracker backend"
        self._thread = threading.Thread(target=self._run_loop, daemon=True)
        self._thread.start()

    def stop(self, timeout_s: float = 3.0) -> None:
        self._stop_event.set()
        thread = self._thread
        if thread is not None:
            thread.join(timeout=timeout_s)
            self._thread = None
        self._cleanup_tracker()
        with self._lock:
            self._state.connection_state = "disconnected"
            self._state.backend_running = False
            self._state.backend_connected = False
            self._state.bridge_running = False
            self._state.socket_connected = False
            self._state.last_status_message = "NDI tracker backend stopped"

    def is_alive(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def get_state_snapshot(self) -> TrackerRuntimeState:
        with self._lock:
            tools = {tool_id: replace(tool) for tool_id, tool in self._state.tools.items()}
            return TrackerRuntimeState(
                connection_state=self._state.connection_state,
                backend_running=self._state.backend_running,
                backend_connected=self._state.backend_connected,
                socket_connected=self._state.socket_connected,
                bridge_running=self._state.bridge_running,
                latest_frame_number=self._state.latest_frame_number,
                latest_timestamp=self._state.latest_timestamp,
                last_status_message=self._state.last_status_message,
                last_error=self._state.last_error,
                tools=tools,
            )

    def get_latest_tool(self, tool_id: str) -> TrackerToolState | None:
        with self._lock:
            tool = self._state.tools.get(str(tool_id).upper())
            return replace(tool) if tool is not None else None

    def _run_loop(self) -> None:
        while not self._stop_event.is_set():
            tracker = None
            try:
                self._set_state(
                    connection_state="connecting",
                    backend_running=True,
                    backend_connected=False,
                    last_error=None,
                    last_status_message=f"Connecting to Aurora on {self.aurora_port}",
                )
                tracker = self._create_tracker()
                self._tracker = tracker
                self._start_tracker(tracker)
                self._set_state(
                    connection_state="tracking",
                    backend_running=True,
                    backend_connected=True,
                    last_status_message="NDI tracker streaming live tool data",
                )

                while not self._stop_event.is_set():
                    observed_at_utc = utc_now_iso()
                    frame_payload = self._get_frame(tracker)
                    self._apply_frame_payload(frame_payload, observed_at_utc=observed_at_utc)
                    time.sleep(self.poll_interval_ms / 1000.0)
            except Exception as exc:
                if self._stop_event.is_set():
                    break
                self._set_state(
                    connection_state="reconnecting",
                    backend_running=True,
                    backend_connected=False,
                    last_error=str(exc),
                    last_status_message=f"NDI tracker error: {exc}",
                )
                self._cleanup_tracker()
                time.sleep(self.reconnect_delay_s)
            finally:
                if tracker is not None:
                    self._stop_tracker(tracker)
                    self._close_tracker(tracker)
                self._tracker = None

    def _create_tracker(self):
        if not self.aurora_port:
            raise RuntimeError("Aurora port is empty. Configure aurora_port before starting the NDI tracker backend.")
        tracker_class = self._tracker_factory or load_ndi_tracker_class()
        settings = self._build_tracker_settings()
        try:
            return tracker_class(settings)
        except Exception as exc:
            raise RuntimeError(
                "Failed to construct NDITracker with settings "
                f"{settings!r}: {exc}"
            ) from exc

    def _build_tracker_settings(self) -> dict[str, Any]:
        settings: dict[str, Any] = {
            "tracker type": self.tracker_type,
            "serial port": self.aurora_port,
        }
        if self.ports_to_probe:
            settings["ports to probe"] = list(self.ports_to_probe)
        settings.update(self.settings_overrides)
        return settings

    @staticmethod
    def _start_tracker(tracker: Any) -> None:
        start_fn = getattr(tracker, "start_tracking", None)
        if callable(start_fn):
            start_fn()
            return
        raise RuntimeError("NDITracker object does not expose start_tracking()")

    @staticmethod
    def _stop_tracker(tracker: Any) -> None:
        stop_fn = getattr(tracker, "stop_tracking", None)
        if callable(stop_fn):
            try:
                stop_fn()
            except Exception:
                return

    @staticmethod
    def _close_tracker(tracker: Any) -> None:
        close_fn = getattr(tracker, "close", None)
        if callable(close_fn):
            try:
                close_fn()
            except Exception:
                return

    def _cleanup_tracker(self) -> None:
        tracker = self._tracker
        self._tracker = None
        if tracker is None:
            return
        self._stop_tracker(tracker)
        self._close_tracker(tracker)

    @staticmethod
    def _get_frame(tracker: Any):
        get_frame_fn = getattr(tracker, "get_frame", None)
        if not callable(get_frame_fn):
            raise RuntimeError("NDITracker object does not expose get_frame()")
        return get_frame_fn()

    def _apply_frame_payload(self, frame_payload: Any, *, observed_at_utc: str) -> None:
        tools, latest_frame = self._parse_frame_payload(frame_payload, observed_at_utc=observed_at_utc)
        with self._lock:
            self._state.tools = tools
            self._state.latest_timestamp = observed_at_utc
            self._state.last_error = None
            if latest_frame is not None:
                self._state.latest_frame_number = latest_frame

    def _parse_frame_payload(
        self,
        frame_payload: Any,
        *,
        observed_at_utc: str,
    ) -> tuple[dict[str, TrackerToolState], int | None]:
        if not isinstance(frame_payload, (list, tuple)) or len(frame_payload) < 5:
            raise RuntimeError(
                "NDITracker.get_frame() returned an unsupported payload shape. "
                "Expected a sequence containing port handles, timestamps, frame "
                f"numbers, transforms, and quality values; got {type(frame_payload).__name__}: {frame_payload!r}"
            )

        raw_handles = _coerce_list(frame_payload[0])
        raw_timestamps = _expand_list(frame_payload[1], len(raw_handles))
        raw_frame_numbers = _expand_list(frame_payload[2], len(raw_handles))
        raw_tracking = _expand_list(frame_payload[3], len(raw_handles))
        raw_quality = _expand_list(frame_payload[4], len(raw_handles))

        tools: dict[str, TrackerToolState] = {}
        observed_frames: list[int] = []
        for index, raw_handle in enumerate(raw_handles):
            tool_id = normalize_tool_id(raw_handle)
            if not tool_id:
                continue

            frame_number = self._coerce_frame_number(raw_frame_numbers[index])
            if frame_number is not None:
                observed_frames.append(frame_number)

            quality = self._coerce_quality(raw_quality[index])
            timestamp = _timestamp_to_iso(raw_timestamps[index], observed_at_utc)
            raw_transform = raw_tracking[index]

            if raw_transform is None:
                tools[tool_id] = TrackerToolState(
                    tool_id=tool_id,
                    frame_number=frame_number,
                    valid=False,
                    validity_known=True,
                    status="missing",
                    quaternion=None,
                    translation_mm=None,
                    quality=quality,
                    timestamp=timestamp,
                )
                continue

            try:
                T_aurora_tool = coerce_transform_matrix(raw_transform, tool_id=tool_id)
                quaternion = rotmat_to_quat_wxyz(T_aurora_tool[0:3, 0:3])
                translation = tuple(float(v) for v in T_aurora_tool[0:3, 3])
                tools[tool_id] = TrackerToolState(
                    tool_id=tool_id,
                    frame_number=frame_number,
                    valid=None,
                    validity_known=False,
                    status="tracked",
                    quaternion=quaternion,
                    translation_mm=translation,
                    quality=quality,
                    timestamp=timestamp,
                )
            except Exception as exc:
                tools[tool_id] = TrackerToolState(
                    tool_id=tool_id,
                    frame_number=frame_number,
                    valid=False,
                    validity_known=True,
                    status=f"invalid_transform: {exc}",
                    quaternion=None,
                    translation_mm=None,
                    quality=quality,
                    timestamp=timestamp,
                )

        latest_frame = max(observed_frames) if observed_frames else None
        if latest_frame is None and tools:
            self._synthetic_frame_number += 1
            latest_frame = self._synthetic_frame_number
            for tool in tools.values():
                tool.frame_number = latest_frame
        return tools, latest_frame

    @staticmethod
    def _coerce_frame_number(raw_frame_number: Any) -> int | None:
        if raw_frame_number in (None, ""):
            return None
        try:
            return int(raw_frame_number)
        except Exception:
            return None

    @staticmethod
    def _coerce_quality(raw_quality: Any) -> float | None:
        if raw_quality in (None, ""):
            return None
        try:
            return float(raw_quality)
        except Exception:
            return None

    def _set_state(self, **updates) -> None:
        with self._lock:
            if "backend_running" in updates and "bridge_running" not in updates:
                updates["bridge_running"] = bool(updates["backend_running"])
            if "backend_connected" in updates and "socket_connected" not in updates:
                updates["socket_connected"] = bool(updates["backend_connected"])
            for key, value in updates.items():
                setattr(self._state, key, value)

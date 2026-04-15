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
import re
import sys
import threading
import time
from typing import Any, Callable

import numpy as np

from continuum_robot.tracking.runtime_models import TrackerRuntimeState, TrackerToolState
from continuum_robot.tracking.transforms import assert_rigid_transform_matrix, make_transform_A_B, rotmat_to_quat_wxyz
from continuum_robot.utils.time_utils import utc_now_iso

DEFAULT_AURORA_TOOL_ID_ALIASES = {
    # Observed on the Raspberry Pi Aurora session. scikit-surgerynditracker
    # reports these raw handle ids while the rest of the app expects the
    # historical runtime roles 0A/0B.
    "10": "0A",
    "11": "0B",
}


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


def _stringify_raw_handle(raw_handle: Any) -> str:
    """Convert a tracker-library handle/object into a stable debug string."""
    if raw_handle is None:
        return ""
    if isinstance(raw_handle, bytes):
        return raw_handle.decode("ascii", errors="replace")
    if isinstance(raw_handle, np.ndarray):
        return np.array2string(raw_handle, separator=",")
    return str(raw_handle)


def _handle_candidates(raw_handle: Any) -> list[str]:
    """Return normalized candidate strings for handle-to-role matching.

    The library contract is not fully pinned down in this repo. In practice,
    the handle may arrive as a byte string, a short text token, or some other
    object whose string form still contains useful information. We therefore
    try a few stable canonicalizations before giving up and treating the handle
    as unmapped.
    """
    text = _stringify_raw_handle(raw_handle).replace("\x00", " ").strip().upper()
    if not text:
        return []
    candidates: list[str] = [text]
    collapsed = re.sub(r"[^A-Z0-9]+", "", text)
    if collapsed and collapsed not in candidates:
        candidates.append(collapsed)
    for token in re.findall(r"[A-Z0-9]+", text):
        if token not in candidates:
            candidates.append(token)
    return candidates


def normalize_tool_id(
    raw_handle: Any,
    *,
    tool_id_aliases: dict[str, str] | None = None,
) -> tuple[str, str, bool]:
    """Normalize one live tracker handle into an app tool id.

    Returns ``(normalized_tool_id, raw_handle_text, mapped_to_runtime_role)``.
    If the returned id is not one of the expected runtime roles, the caller
    should treat it as an observed-but-unmapped live tool until the operator
    supplies an explicit alias in config.
    """
    raw_text = _stringify_raw_handle(raw_handle).replace("\x00", " ").strip()
    alias_map = dict(DEFAULT_AURORA_TOOL_ID_ALIASES)
    alias_map.update({str(key).upper(): str(value).upper() for key, value in (tool_id_aliases or {}).items()})
    candidates = _handle_candidates(raw_handle)
    for candidate in candidates:
        alias = alias_map.get(candidate)
        if alias in {"0A", "0B"}:
            return alias, raw_text, True
    for candidate in candidates:
        if candidate in {"0A", "0B"}:
            return candidate, raw_text, True
        if candidate.endswith("0A"):
            return "0A", raw_text, True
        if candidate.endswith("0B"):
            return "0B", raw_text, True
    if candidates:
        return candidates[0], raw_text, False
    return "", raw_text, False


def _coerce_list(value: Any) -> list[Any]:
    if isinstance(value, (list, tuple)):
        return list(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    return [value]


def _is_scalar_sequence(value: Any) -> bool:
    if isinstance(value, np.ndarray):
        array = np.asarray(value)
        if array.ndim != 1:
            return False
        try:
            np.asarray(value, dtype=float)
            return True
        except Exception:
            return False
    if not isinstance(value, (list, tuple)):
        return False
    for item in value:
        if isinstance(item, (list, tuple, dict, np.ndarray)):
            return False
        try:
            float(item)
        except Exception:
            return False
    return True


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


def _split_tracking_values(value: Any) -> list[Any]:
    """Split tracker transform payloads into one item per observed tool."""
    if isinstance(value, np.ndarray):
        array = np.asarray(value)
        if array.shape in {(4, 4), (16,), (1, 16), (7,), (1, 7), (8,), (1, 8)}:
            return [array]
        if array.ndim == 3 and array.shape[-2:] == (4, 4):
            return [array[index] for index in range(array.shape[0])]
        if array.ndim == 2 and array.shape[1] in {7, 8, 16}:
            return [array[index] for index in range(array.shape[0])]
        return [array]
    if isinstance(value, (list, tuple)):
        if _is_scalar_sequence(value):
            return [list(value)]
        if len(value) == 4 and all(isinstance(item, (list, tuple, np.ndarray)) for item in value):
            matrix = np.asarray(value, dtype=float)
            if matrix.shape == (4, 4):
                return [matrix]
        return list(value)
    return [value]


def _tracking_state_rank(status: str) -> int:
    normalized = (status or "").strip().lower()
    if normalized.startswith("tracked") or normalized in {"visible", "ok"}:
        return 3
    if normalized.startswith("invalid"):
        return 2
    if normalized == "missing":
        return 1
    return 0


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


def _coerce_vector(value: Any, length: int, *, field_name: str) -> tuple[float, ...]:
    array = np.asarray(value, dtype=float)
    if array.shape == (length,):
        return tuple(float(v) for v in array)
    if array.shape == (1, length):
        return tuple(float(v) for v in array.reshape(length))
    raise ValueError(f"{field_name} must have shape ({length},), got {array.shape}")


def _quality_from_value(raw_quality: Any, *, fallback: float | None = None) -> float | None:
    if raw_quality in (None, ""):
        return fallback
    try:
        return float(raw_quality)
    except Exception:
        return fallback


def _read_mapping_or_attr(container: Any, *names: str) -> Any:
    if isinstance(container, dict):
        for name in names:
            if name in container:
                return container[name]
    for name in names:
        if hasattr(container, name):
            return getattr(container, name)
    return None


def _payload_summary(raw_transform: Any) -> str:
    if raw_transform is None:
        return "none"
    if isinstance(raw_transform, np.ndarray):
        return f"ndarray{tuple(raw_transform.shape)}"
    if isinstance(raw_transform, dict):
        keys = sorted(str(key) for key in raw_transform.keys())[:6]
        return f"dict(keys={keys})"
    if isinstance(raw_transform, (list, tuple)):
        if _is_scalar_sequence(raw_transform):
            return f"{type(raw_transform).__name__}[len={len(raw_transform)}]"
        return f"{type(raw_transform).__name__}[len={len(raw_transform)}]"
    return type(raw_transform).__name__


def _reshape_transform_matrix(raw_transform: Any, *, tool_id: str) -> np.ndarray:
    matrix = np.asarray(raw_transform, dtype=float)
    if matrix.shape == (16,):
        matrix = matrix.reshape(4, 4)
    elif matrix.shape == (1, 16):
        matrix = matrix.reshape(4, 4)
    elif matrix.shape == (4, 4):
        matrix = matrix.copy()
    else:
        raise ValueError(f"T_aurora_{tool_id} must be 4x4 or length-16, got shape {matrix.shape}")
    return matrix


def _serialize_debug_value(value: Any) -> Any:
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, np.ndarray):
        return np.asarray(value).tolist()
    if isinstance(value, (list, tuple)):
        return [_serialize_debug_value(item) for item in list(value)]
    if isinstance(value, dict):
        return {str(key): _serialize_debug_value(item) for key, item in value.items()}

    fields = {}
    for name in (
        "transform",
        "matrix",
        "T",
        "tracking",
        "pose",
        "quaternion",
        "quaternion_wxyz",
        "translation",
        "translation_mm",
        "position",
        "position_mm",
        "quality",
        "error",
        "tracking_error",
        "q0",
        "qw",
        "qx",
        "qy",
        "qz",
        "tx",
        "ty",
        "tz",
        "x",
        "y",
        "z",
    ):
        if hasattr(value, name):
            fields[name] = _serialize_debug_value(getattr(value, name))
    if fields:
        return fields
    return repr(value)


def _matrix_debug_payload(matrix: np.ndarray | None) -> dict[str, Any]:
    if matrix is None:
        return {
            "matrix_before_validation": None,
            "rotation_block": None,
            "rotation_gram": None,
            "rotation_determinant": None,
        }
    R = np.asarray(matrix[0:3, 0:3], dtype=float)
    return {
        "matrix_before_validation": np.asarray(matrix, dtype=float).tolist(),
        "rotation_block": R.tolist(),
        "rotation_gram": (R.T @ R).tolist(),
        "rotation_determinant": float(np.linalg.det(R)),
    }


def _debug_matrix_from_raw_payload(raw_transform: Any, *, tool_id: str) -> np.ndarray | None:
    try:
        if raw_transform is None:
            return None
        if isinstance(raw_transform, np.ndarray):
            array = np.asarray(raw_transform, dtype=float)
            if array.shape in {(4, 4), (16,), (1, 16)}:
                return _reshape_transform_matrix(array, tool_id=tool_id)
            flat = array.reshape(-1)
            if flat.size in {7, 8}:
                quat = tuple(float(v) for v in flat[0:4])
                translation = tuple(float(v) for v in flat[4:7])
                return make_transform_A_B(quat, translation)
            return None
        if _is_scalar_sequence(raw_transform):
            flat = np.asarray(raw_transform, dtype=float).reshape(-1)
            if flat.size in {7, 8}:
                quat = tuple(float(v) for v in flat[0:4])
                translation = tuple(float(v) for v in flat[4:7])
                return make_transform_A_B(quat, translation)
            if flat.size == 16:
                return _reshape_transform_matrix(flat, tool_id=tool_id)
            return None
        named = _extract_pose_from_named_fields(raw_transform, tool_id=tool_id)
        if named is not None:
            return named[0]
    except Exception:
        return None
    return None


def _failure_stage_from_exception(exc: Exception) -> str:
    message = str(exc).lower()
    if "missing quaternion/translation" in message or "unsupported tracking payload" in message:
        return "backend_payload"
    if "must be 4x4" in message or "unsupported pose-vector length" in message or "must have shape" in message:
        return "conversion"
    if "not orthonormal" in message or "determinant must" in message or "last row is not" in message:
        return "validation"
    if "quaternion" in message or "translation" in message:
        return "conversion"
    return "conversion"


def _normalize_timing_tool_status(tool: TrackerToolState) -> str:
    normalized = str(getattr(tool, "status", "") or "").strip().lower()
    if normalized.startswith("tracked") or normalized in {"visible", "ok"}:
        return "tracked"
    if normalized.startswith("invalid"):
        return "invalid"
    if normalized == "missing":
        return "missing"
    return "unknown"


def _matrix_from_quat_translation(
    quat_wxyz: tuple[float, float, float, float],
    translation_mm: tuple[float, float, float],
    *,
    tool_id: str,
) -> tuple[np.ndarray, tuple[float, float, float, float], tuple[float, float, float]]:
    T_aurora_tool = make_transform_A_B(quat_wxyz, translation_mm)
    assert_rigid_transform_matrix(T_aurora_tool, f"T_aurora_{tool_id}")
    quaternion = rotmat_to_quat_wxyz(T_aurora_tool[0:3, 0:3])
    translation = tuple(float(v) for v in T_aurora_tool[0:3, 3])
    return T_aurora_tool, quaternion, translation


def _extract_pose_from_numeric_sequence(
    raw_transform: Any,
    *,
    tool_id: str,
) -> tuple[np.ndarray, tuple[float, float, float, float], tuple[float, float, float], str]:
    values = np.asarray(raw_transform, dtype=float).reshape(-1)
    if values.size == 7:
        quat = tuple(float(v) for v in values[0:4])
        translation = tuple(float(v) for v in values[4:7])
        matrix, quaternion, translation = _matrix_from_quat_translation(quat, translation, tool_id=tool_id)
        return matrix, quaternion, translation, "pose_vector_wxyz_xyz"
    if values.size == 8:
        quat = tuple(float(v) for v in values[0:4])
        translation = tuple(float(v) for v in values[4:7])
        matrix, quaternion, translation = _matrix_from_quat_translation(quat, translation, tool_id=tool_id)
        return matrix, quaternion, translation, "pose_vector_wxyz_xyzq"
    if values.size == 16:
        matrix = coerce_transform_matrix(values, tool_id=tool_id)
        quaternion = rotmat_to_quat_wxyz(matrix[0:3, 0:3])
        translation = tuple(float(v) for v in matrix[0:3, 3])
        return matrix, quaternion, translation, "matrix_vector_16"
    raise ValueError(f"unsupported pose-vector length {values.size}")


def _extract_pose_from_named_fields(
    raw_transform: Any,
    *,
    tool_id: str,
) -> tuple[np.ndarray, tuple[float, float, float, float], tuple[float, float, float], str] | None:
    nested_matrix = _read_mapping_or_attr(raw_transform, "transform", "matrix", "T", "tracking", "pose")
    if nested_matrix is not None and nested_matrix is not raw_transform:
        try:
            matrix = coerce_transform_matrix(nested_matrix, tool_id=tool_id)
            quaternion = rotmat_to_quat_wxyz(matrix[0:3, 0:3])
            translation = tuple(float(v) for v in matrix[0:3, 3])
            return matrix, quaternion, translation, "named_matrix_field"
        except Exception:
            pass

    quat_value = _read_mapping_or_attr(
        raw_transform,
        "quaternion_wxyz",
        "quaternion",
        "quat",
        "orientation",
        "rotation",
    )
    translation_value = _read_mapping_or_attr(
        raw_transform,
        "translation_mm",
        "translation_xyz",
        "translation",
        "position_mm",
        "position",
    )
    if quat_value is not None or translation_value is not None:
        if quat_value is None or translation_value is None:
            raise ValueError("missing quaternion/translation fields")
        quat = _coerce_vector(quat_value, 4, field_name="quaternion")
        translation = _coerce_vector(translation_value, 3, field_name="translation")
        matrix, quaternion, translation = _matrix_from_quat_translation(quat, translation, tool_id=tool_id)
        return matrix, quaternion, translation, "named_quaternion_translation"

    qx = _read_mapping_or_attr(raw_transform, "qx")
    qy = _read_mapping_or_attr(raw_transform, "qy")
    qz = _read_mapping_or_attr(raw_transform, "qz")
    qw = _read_mapping_or_attr(raw_transform, "q0", "qw")
    tx = _read_mapping_or_attr(raw_transform, "tx", "x")
    ty = _read_mapping_or_attr(raw_transform, "ty", "y")
    tz = _read_mapping_or_attr(raw_transform, "tz", "z")
    if all(value is not None for value in (qx, qy, qz, qw, tx, ty, tz)):
        quat = (float(qw), float(qx), float(qy), float(qz))
        translation = (float(tx), float(ty), float(tz))
        matrix, quaternion, translation = _matrix_from_quat_translation(quat, translation, tool_id=tool_id)
        return matrix, quaternion, translation, "scalar_pose_fields"

    return None


def _extract_transform_payload(
    raw_transform: Any,
    *,
    tool_id: str,
    fallback_quality: float | None,
) -> tuple[np.ndarray, tuple[float, float, float, float], tuple[float, float, float], float | None, str]:
    summary = _payload_summary(raw_transform)

    named = _extract_pose_from_named_fields(raw_transform, tool_id=tool_id)
    if named is not None:
        matrix, quaternion, translation, parse_mode = named
        payload_quality = _quality_from_value(
            _read_mapping_or_attr(raw_transform, "quality", "error", "tracking_error"),
            fallback=fallback_quality,
        )
        return matrix, quaternion, translation, payload_quality, f"{summary}:{parse_mode}"

    if isinstance(raw_transform, np.ndarray):
        matrix_or_pose = np.asarray(raw_transform)
        if matrix_or_pose.shape in {(4, 4), (16,), (1, 16)}:
            matrix = coerce_transform_matrix(matrix_or_pose, tool_id=tool_id)
            quaternion = rotmat_to_quat_wxyz(matrix[0:3, 0:3])
            translation = tuple(float(v) for v in matrix[0:3, 3])
            return matrix, quaternion, translation, fallback_quality, f"{summary}:matrix"
        matrix, quaternion, translation, parse_mode = _extract_pose_from_numeric_sequence(
            matrix_or_pose,
            tool_id=tool_id,
        )
        payload_quality = fallback_quality
        if matrix_or_pose.reshape(-1).size == 8:
            payload_quality = _quality_from_value(matrix_or_pose.reshape(-1)[7], fallback=fallback_quality)
        return matrix, quaternion, translation, payload_quality, f"{summary}:{parse_mode}"

    if _is_scalar_sequence(raw_transform):
        matrix, quaternion, translation, parse_mode = _extract_pose_from_numeric_sequence(
            raw_transform,
            tool_id=tool_id,
        )
        flat = np.asarray(raw_transform, dtype=float).reshape(-1)
        payload_quality = fallback_quality
        if flat.size == 8:
            payload_quality = _quality_from_value(flat[7], fallback=fallback_quality)
        return matrix, quaternion, translation, payload_quality, f"{summary}:{parse_mode}"

    raise ValueError(f"unsupported tracking payload {summary}")


def coerce_transform_matrix(raw_transform: Any, *, tool_id: str) -> np.ndarray:
    """Convert one tracker-library transform payload into a strict 4x4 matrix."""
    matrix = _reshape_transform_matrix(raw_transform, tool_id=tool_id)
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
        tool_id_aliases: dict[str, str] | None = None,
        debug_frames_to_log: int = 12,
        tracker_factory: Callable[[dict[str, Any]], Any] | None = None,
        expected_tool_ids: tuple[str, ...] = ("0A", "0B"),
    ) -> None:
        self.aurora_port = aurora_port
        self.tracker_type = str(tracker_type)
        self.poll_interval_ms = max(1, int(poll_interval_ms))
        self.reconnect_delay_s = max(0.1, float(reconnect_delay_s))
        self.ports_to_probe = list(ports_to_probe or [])
        self.settings_overrides = dict(settings_overrides or {})
        self.tool_id_aliases = {str(key).upper(): str(value).upper() for key, value in (tool_id_aliases or {}).items()}
        self.debug_frames_to_log = max(0, int(debug_frames_to_log))
        self.expected_tool_ids = tuple(str(v).upper() for v in expected_tool_ids)
        self._tracker_factory = tracker_factory

        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._tracker = None
        self._synthetic_frame_number = 0
        self._frames_received_total = 0
        self._remaining_debug_frames = self.debug_frames_to_log
        self._last_debug_signature: tuple[Any, ...] | None = None
        self._tracker_settings_snapshot: dict[str, Any] = {}
        self._last_frame_debug: dict[str, Any] = {}
        self._timing_listeners: set[Callable[[dict[str, Any]], None]] = set()
        self._state = TrackerRuntimeState(
            connection_state="disconnected",
            canonical_state="disconnected",
            backend_identity=self.backend_identity,
            backend_running=False,
            backend_connected=False,
            bridge_running=False,
            socket_connected=False,
            backend_frame_counter=0,
            latest_frame_number=None,
            latest_timestamp=None,
            last_status_message="NDI tracker backend idle",
            last_error=None,
            raw_tool_ids=[],
            normalized_tool_ids=[],
            tool_id_mapping={},
            runtime_role_mappings={},
            unmapped_tool_ids=[],
            backend_details={},
            tools={},
        )

    def register_timing_listener(self, listener: Callable[[dict[str, Any]], None]) -> None:
        """Subscribe to per-frame backend timing records."""
        with self._lock:
            self._timing_listeners.add(listener)

    def unregister_timing_listener(self, listener: Callable[[dict[str, Any]], None]) -> None:
        """Remove a previously registered timing listener."""
        with self._lock:
            self._timing_listeners.discard(listener)

    def start(self) -> None:
        if self.is_alive():
            return
        self._stop_event.clear()
        self._frames_received_total = 0
        self._synthetic_frame_number = 0
        self._remaining_debug_frames = self.debug_frames_to_log
        self._last_debug_signature = None
        with self._lock:
            self._state.connection_state = "starting"
            self._state.canonical_state = "connecting"
            self._state.backend_running = True
            self._state.backend_connected = False
            self._state.bridge_running = False
            self._state.socket_connected = False
            self._state.backend_frame_counter = 0
            self._state.last_error = None
            self._state.last_status_message = "Starting Python NDI tracker backend"
            self._state.raw_tool_ids = []
            self._state.normalized_tool_ids = []
            self._state.tool_id_mapping = {}
            self._state.runtime_role_mappings = {}
            self._state.unmapped_tool_ids = []
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
            self._state.canonical_state = "disconnected"
            self._state.backend_running = False
            self._state.backend_connected = False
            self._state.bridge_running = False
            self._state.socket_connected = False
            self._state.backend_frame_counter = self._frames_received_total
            self._state.last_status_message = "NDI tracker backend stopped"

    def is_alive(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def get_state_snapshot(self) -> TrackerRuntimeState:
        with self._lock:
            tools = {tool_id: replace(tool) for tool_id, tool in self._state.tools.items()}
            return TrackerRuntimeState(
                connection_state=self._state.connection_state,
                canonical_state=self._state.canonical_state,
                backend_identity=self.backend_identity,
                backend_running=self._state.backend_running,
                backend_connected=self._state.backend_connected,
                socket_connected=self._state.socket_connected,
                bridge_running=self._state.bridge_running,
                backend_frame_counter=self._state.backend_frame_counter,
                latest_frame_number=self._state.latest_frame_number,
                latest_timestamp=self._state.latest_timestamp,
                last_status_message=self._state.last_status_message,
                last_error=self._state.last_error,
                raw_tool_ids=list(self._state.raw_tool_ids),
                normalized_tool_ids=list(self._state.normalized_tool_ids),
                tool_id_mapping=dict(self._state.tool_id_mapping),
                runtime_role_mappings=dict(self._state.runtime_role_mappings),
                unmapped_tool_ids=list(self._state.unmapped_tool_ids),
                backend_details=self._build_backend_details(),
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
                    sample_start_ns = time.monotonic_ns()
                    observed_at_utc = utc_now_iso()
                    backend_call_end_ns: int | None = None
                    parse_complete_ns: int | None = None
                    state_commit_complete_ns: int | None = None
                    latest_frame: int | None = None
                    frame_number_source = "missing"
                    tools: dict[str, TrackerToolState] = {}
                    debug: dict[str, Any] = {}
                    raw_payload_available = False
                    parsed_payload_available = False
                    output_committed = False
                    is_new_frame: bool | None = None
                    is_duplicate_frame: bool | None = None
                    error_stage: str | None = None
                    error_message: str | None = None
                    try:
                        frame_payload = self._get_frame(tracker)
                        backend_call_end_ns = time.monotonic_ns()
                        observed_at_utc = utc_now_iso()
                        raw_payload_available = True
                        tools, latest_frame, frame_number_source, debug = self._parse_frame_payload_with_source(
                            frame_payload,
                            observed_at_utc=observed_at_utc,
                        )
                        parse_complete_ns = time.monotonic_ns()
                        parsed_payload_available = True
                        is_new_frame, is_duplicate_frame = self._classify_frame_number(latest_frame)
                        self._commit_frame_payload(
                            tools=tools,
                            latest_frame=latest_frame,
                            debug=debug,
                            observed_at_utc=observed_at_utc,
                        )
                        state_commit_complete_ns = time.monotonic_ns()
                        output_committed = True
                    except Exception as exc:
                        failure_ns = time.monotonic_ns()
                        backend_call_end_ns = backend_call_end_ns or failure_ns
                        state_commit_complete_ns = state_commit_complete_ns or failure_ns
                        error_stage = (
                            "get_frame"
                            if not raw_payload_available
                            else ("parse" if not parsed_payload_available else "commit")
                        )
                        error_message = str(exc)
                        self._emit_timing_record(
                            self._build_timing_record(
                                sample_start_ns=sample_start_ns,
                                backend_call_end_ns=backend_call_end_ns,
                                parse_complete_ns=parse_complete_ns,
                                state_commit_complete_ns=state_commit_complete_ns,
                                observed_at_utc=observed_at_utc,
                                latest_frame=latest_frame,
                                frame_number_source=frame_number_source,
                                tools=tools,
                                debug=debug,
                                raw_payload_available=raw_payload_available,
                                parsed_payload_available=parsed_payload_available,
                                output_committed=output_committed,
                                is_new_frame=is_new_frame,
                                is_duplicate_frame=is_duplicate_frame,
                                error_stage=error_stage,
                                error_message=error_message,
                            )
                        )
                        raise
                    self._emit_timing_record(
                        self._build_timing_record(
                            sample_start_ns=sample_start_ns,
                            backend_call_end_ns=backend_call_end_ns or time.monotonic_ns(),
                            parse_complete_ns=parse_complete_ns,
                            state_commit_complete_ns=state_commit_complete_ns or time.monotonic_ns(),
                            observed_at_utc=observed_at_utc,
                            latest_frame=latest_frame,
                            frame_number_source=frame_number_source,
                            tools=tools,
                            debug=debug,
                            raw_payload_available=raw_payload_available,
                            parsed_payload_available=parsed_payload_available,
                            output_committed=output_committed,
                            is_new_frame=is_new_frame,
                            is_duplicate_frame=is_duplicate_frame,
                            error_stage=error_stage,
                            error_message=error_message,
                        )
                    )
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
        self._tracker_settings_snapshot = dict(settings)
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
        if "use quaternions" not in self.settings_overrides:
            settings["use quaternions"] = True
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
        tools, latest_frame, _frame_number_source, debug = self._parse_frame_payload_with_source(
            frame_payload,
            observed_at_utc=observed_at_utc,
        )
        self._commit_frame_payload(
            tools=tools,
            latest_frame=latest_frame,
            debug=debug,
            observed_at_utc=observed_at_utc,
        )

    def _commit_frame_payload(
        self,
        *,
        tools: dict[str, TrackerToolState],
        latest_frame: int | None,
        debug: dict[str, Any],
        observed_at_utc: str,
    ) -> None:
        self._frames_received_total += 1
        self._maybe_log_debug_frame(debug, latest_frame=latest_frame)
        with self._lock:
            self._last_frame_debug = dict(debug)
            self._state.tools = tools
            self._state.backend_frame_counter = self._frames_received_total
            self._state.latest_timestamp = observed_at_utc
            self._state.last_error = None
            self._state.raw_tool_ids = list(debug["raw_tool_ids"])
            self._state.normalized_tool_ids = list(debug["normalized_tool_ids"])
            self._state.tool_id_mapping = dict(debug["tool_id_mapping"])
            self._state.runtime_role_mappings = dict(debug["runtime_role_mappings"])
            self._state.unmapped_tool_ids = list(debug["unmapped_tool_ids"])
            self._state.backend_details = self._build_backend_details()
            if latest_frame is not None:
                self._state.latest_frame_number = latest_frame

    def _parse_frame_payload(
        self,
        frame_payload: Any,
        *,
        observed_at_utc: str,
    ) -> tuple[dict[str, TrackerToolState], int | None, dict[str, Any]]:
        tools, latest_frame, _frame_number_source, debug = self._parse_frame_payload_with_source(
            frame_payload,
            observed_at_utc=observed_at_utc,
        )
        return tools, latest_frame, debug

    def _parse_frame_payload_with_source(
        self,
        frame_payload: Any,
        *,
        observed_at_utc: str,
    ) -> tuple[dict[str, TrackerToolState], int | None, str, dict[str, Any]]:
        if not isinstance(frame_payload, (list, tuple)) or len(frame_payload) < 5:
            raise RuntimeError(
                "NDITracker.get_frame() returned an unsupported payload shape. "
                "Expected a sequence containing port handles, timestamps, frame "
                f"numbers, transforms, and quality values; got {type(frame_payload).__name__}: {frame_payload!r}"
            )

        raw_handles = _coerce_list(frame_payload[0])
        tracking_values = _split_tracking_values(frame_payload[3])
        sample_count = max(
            len(raw_handles),
            len(tracking_values),
            len(_coerce_list(frame_payload[4])),
            0,
        )
        if len(raw_handles) < sample_count:
            raw_handles = raw_handles + [f"tool[{index}]" for index in range(len(raw_handles), sample_count)]
        raw_timestamps = _expand_list(frame_payload[1], sample_count)
        raw_frame_numbers = _expand_list(frame_payload[2], sample_count)
        raw_tracking = _expand_list(tracking_values, sample_count)
        raw_quality = _expand_list(frame_payload[4], sample_count)

        tools: dict[str, TrackerToolState] = {}
        observed_frames: list[int] = []
        raw_tool_ids: list[str] = []
        normalized_tool_ids: list[str] = []
        tool_id_mapping: dict[str, str] = {}
        runtime_role_mappings: dict[str, str] = {}
        unmapped_tool_ids: list[str] = []
        tool_payload_summaries: dict[str, str] = {}
        tool_transform_debug: dict[str, dict[str, Any]] = {}

        for index in range(sample_count):
            raw_handle = raw_handles[index]
            tool_id, raw_text, mapped_to_runtime_role = normalize_tool_id(
                raw_handle,
                tool_id_aliases=self.tool_id_aliases,
            )
            if not tool_id:
                continue
            raw_tool_ids.append(raw_text)
            normalized_tool_ids.append(tool_id)
            tool_id_mapping[raw_text] = tool_id
            if mapped_to_runtime_role and tool_id in self.expected_tool_ids and tool_id not in runtime_role_mappings:
                runtime_role_mappings[tool_id] = raw_text
            if not mapped_to_runtime_role:
                unmapped_tool_ids.append(raw_text)

            frame_number = self._coerce_frame_number(raw_frame_numbers[index])
            if frame_number is not None:
                observed_frames.append(frame_number)

            quality = self._coerce_quality(raw_quality[index])
            timestamp = _timestamp_to_iso(raw_timestamps[index], observed_at_utc)
            raw_transform = raw_tracking[index]
            debug_entry = {
                "raw_handle": raw_text,
                "normalized_tool_id": tool_id,
                "frame_number": frame_number,
                "timestamp": timestamp,
                "raw_payload_summary": _payload_summary(raw_transform),
                "raw_transform": _serialize_debug_value(raw_transform),
                "quality": quality,
            }

            if raw_transform is None:
                tool_payload_summaries[raw_text or tool_id] = "none:missing"
                debug_entry.update(
                    {
                        "parse_mode": "missing",
                        "classified_state": "missing",
                        "failure_stage": "backend_payload",
                        "invalid_reason": "raw transform missing from backend payload",
                        **_matrix_debug_payload(None),
                    }
                )
                candidate = TrackerToolState(
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
            else:
                try:
                    _T_aurora_tool, quaternion, translation, quality, payload_summary = _extract_transform_payload(
                        raw_transform,
                        tool_id=tool_id,
                        fallback_quality=quality,
                    )
                    tool_payload_summaries[raw_text or tool_id] = payload_summary
                    debug_entry.update(
                        {
                            "parse_mode": payload_summary,
                            "classified_state": "tracked",
                            "failure_stage": None,
                            "invalid_reason": None,
                            **_matrix_debug_payload(_T_aurora_tool),
                        }
                    )
                    candidate = TrackerToolState(
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
                    tool_payload_summaries[raw_text or tool_id] = f"{_payload_summary(raw_transform)}:error={exc}"
                    debug_entry.update(
                        {
                            "parse_mode": f"{_payload_summary(raw_transform)}:error",
                            "classified_state": "invalid",
                            "failure_stage": _failure_stage_from_exception(exc),
                            "invalid_reason": str(exc),
                            **_matrix_debug_payload(_debug_matrix_from_raw_payload(raw_transform, tool_id=tool_id)),
                        }
                    )
                    candidate = TrackerToolState(
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

            existing = tools.get(tool_id)
            if existing is None or _tracking_state_rank(candidate.status) >= _tracking_state_rank(existing.status):
                tools[tool_id] = candidate
                tool_transform_debug[tool_id] = debug_entry

        latest_frame_source = "missing"
        latest_frame = max(observed_frames) if observed_frames else self._coerce_frame_number(frame_payload[2])
        if latest_frame is not None:
            latest_frame_source = "device"
        if latest_frame is None and tools:
            self._synthetic_frame_number += 1
            latest_frame = self._synthetic_frame_number
            latest_frame_source = "synthetic"
            for tool in tools.values():
                if tool.frame_number is None:
                    tool.frame_number = latest_frame

        debug = {
            "raw_tool_ids": list(dict.fromkeys(raw_tool_ids)),
            "normalized_tool_ids": list(dict.fromkeys(normalized_tool_ids)),
            "tool_id_mapping": dict(tool_id_mapping),
            "runtime_role_mappings": dict(runtime_role_mappings),
            "unmapped_tool_ids": list(dict.fromkeys(unmapped_tool_ids)),
            "tool_payload_summaries": dict(tool_payload_summaries),
            "tool_transform_debug": dict(tool_transform_debug),
        }
        return tools, latest_frame, latest_frame_source, debug

    def _classify_frame_number(self, latest_frame: int | None) -> tuple[bool | None, bool | None]:
        if latest_frame is None:
            return None, None
        with self._lock:
            previous_frame = self._state.latest_frame_number
        if previous_frame is None:
            return True, False
        if int(latest_frame) == int(previous_frame):
            return False, True
        return True, False

    def _build_timing_record(
        self,
        *,
        sample_start_ns: int,
        backend_call_end_ns: int,
        parse_complete_ns: int | None,
        state_commit_complete_ns: int,
        observed_at_utc: str,
        latest_frame: int | None,
        frame_number_source: str,
        tools: dict[str, TrackerToolState],
        debug: dict[str, Any],
        raw_payload_available: bool,
        parsed_payload_available: bool,
        output_committed: bool,
        is_new_frame: bool | None,
        is_duplicate_frame: bool | None,
        error_stage: str | None,
        error_message: str | None,
    ) -> dict[str, Any]:
        tool_validity = {
            str(tool_id): _normalize_timing_tool_status(tool)
            for tool_id, tool in sorted(tools.items())
        }
        tool_pose_payload = {
            str(tool_id): {
                "tracking_state": str(tool.status or "unknown"),
                "frame_number": int(tool.frame_number) if tool.frame_number is not None else None,
                "translation_mm": (
                    [float(value) for value in tool.translation_mm]
                    if tool.translation_mm is not None
                    else None
                ),
                "quaternion_wxyz": (
                    [float(value) for value in tool.quaternion]
                    if tool.quaternion is not None
                    else None
                ),
            }
            for tool_id, tool in sorted(tools.items())
        }
        valid_transform_count = sum(1 for status in tool_validity.values() if status == "tracked")
        return {
            "sample_start_monotonic_ns": int(sample_start_ns),
            "backend_call_start_ns": int(sample_start_ns),
            "backend_call_end_ns": int(backend_call_end_ns),
            "parse_complete_ns": int(parse_complete_ns) if parse_complete_ns is not None else None,
            "state_commit_complete_ns": int(state_commit_complete_ns),
            "sample_commit_monotonic_ns": int(state_commit_complete_ns),
            "observed_at_utc": str(observed_at_utc),
            "backend_identity": self.backend_identity,
            "requested_tool_ids": list(self.expected_tool_ids),
            "frame_number": int(latest_frame) if latest_frame is not None else None,
            "frame_number_source": str(frame_number_source),
            "is_new_frame": is_new_frame,
            "is_duplicate_frame": is_duplicate_frame,
            "raw_payload_available": bool(raw_payload_available),
            "parsed_payload_available": bool(parsed_payload_available),
            "output_committed": bool(output_committed),
            "error_flag": bool(error_message),
            "error_stage": str(error_stage) if error_stage else None,
            "error_message": str(error_message) if error_message else None,
            "raw_tool_ids": list(debug.get("raw_tool_ids", [])),
            "normalized_tool_ids": list(debug.get("normalized_tool_ids", [])),
            "runtime_role_mappings": dict(debug.get("runtime_role_mappings", {})),
            "tools_visible": sorted(tool_validity),
            "tool_validity": tool_validity,
            "tool_pose_payload": tool_pose_payload,
            "valid_transform_count": int(valid_transform_count),
            "total_cycle_ms": float(max(0, state_commit_complete_ns - sample_start_ns)) / 1_000_000.0,
            "backend_call_ms": float(max(0, backend_call_end_ns - sample_start_ns)) / 1_000_000.0,
            "parse_ms": (
                float(max(0, parse_complete_ns - backend_call_end_ns)) / 1_000_000.0
                if parse_complete_ns is not None
                else None
            ),
            "state_commit_ms": float(
                max(0, state_commit_complete_ns - (parse_complete_ns or backend_call_end_ns))
            )
            / 1_000_000.0,
        }

    def _emit_timing_record(self, record: dict[str, Any]) -> None:
        with self._lock:
            listeners = list(self._timing_listeners)
        for listener in listeners:
            try:
                listener(dict(record))
            except Exception:
                continue

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

    def _maybe_log_debug_frame(self, debug: dict[str, Any], *, latest_frame: int | None) -> None:
        if self._remaining_debug_frames <= 0:
            return
        signature = (
            tuple(debug.get("raw_tool_ids", [])),
            tuple(debug.get("normalized_tool_ids", [])),
            tuple(sorted((debug.get("tool_id_mapping", {}) or {}).items())),
            tuple(sorted((debug.get("runtime_role_mappings", {}) or {}).items())),
            tuple(debug.get("unmapped_tool_ids", [])),
            tuple(sorted((debug.get("tool_payload_summaries", {}) or {}).items())),
        )
        if signature == self._last_debug_signature:
            return
        self._last_debug_signature = signature
        self._remaining_debug_frames -= 1
        print(
            "[ndi_backend] "
            f"frame={latest_frame} raw_tool_ids={debug.get('raw_tool_ids', [])} "
            f"normalized_tool_ids={debug.get('normalized_tool_ids', [])} "
            f"runtime_role_mappings={debug.get('runtime_role_mappings', {})} "
            f"unmapped={debug.get('unmapped_tool_ids', [])} "
            f"payloads={debug.get('tool_payload_summaries', {})}",
            file=sys.stderr,
            flush=True,
        )

    def _set_state(self, **updates) -> None:
        with self._lock:
            if "backend_running" in updates and "bridge_running" not in updates:
                updates["bridge_running"] = bool(updates["backend_running"])
            if "backend_connected" in updates and "socket_connected" not in updates:
                updates["socket_connected"] = bool(updates["backend_connected"])
            if "connection_state" in updates and "canonical_state" not in updates:
                updates["canonical_state"] = self._canonical_state_for_connection(str(updates["connection_state"]))
            for key, value in updates.items():
                setattr(self._state, key, value)

    def _build_backend_details(self) -> dict[str, Any]:
        return {
            "tracker_type": self.tracker_type,
            "aurora_port": self.aurora_port,
            "poll_interval_ms": self.poll_interval_ms,
            "tool_aliases": dict(self.tool_id_aliases),
            "expected_tool_ids": list(self.expected_tool_ids),
            "tracker_settings": dict(self._tracker_settings_snapshot),
            "ndi_transform_debug": dict(self._last_frame_debug),
        }

    @staticmethod
    def _canonical_state_for_connection(connection_state: str) -> str:
        if connection_state in {"starting", "connecting", "reconnecting"}:
            return "connecting"
        if connection_state in {"tracking"}:
            return "streaming_healthy"
        if connection_state in {"error"}:
            return "error"
        return "disconnected"

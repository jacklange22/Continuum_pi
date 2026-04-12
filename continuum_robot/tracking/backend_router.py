"""Canonical live-tracking backend router.

This module owns deterministic backend selection, capability checks, fallback
policy, and backend-level diagnostic metadata. The rest of the app should talk
to TrackingService, which in turn talks to this router as its single live
backend seam.
"""

from __future__ import annotations

import copy
import os
from pathlib import Path
from typing import Any, Callable

from continuum_robot.tracking.mock_tracker_manager import MockTrackerManager
from continuum_robot.tracking.ndi_backend import TrackerBackendNDI, load_ndi_tracker_class
from continuum_robot.tracking.runtime_models import TrackerRuntimeState, TrackerToolState
from continuum_robot.tracking.tracker_service_manager import TrackerServiceManager


class TrackingBackendRouter:
    """Select and run one live tracking backend with explicit fallback behavior."""

    backend_identity = "tracking_backend_router"

    def __init__(
        self,
        *,
        mock_mode: bool,
        preferred_backend: str,
        fallback_backend: str | None,
        fallback_enabled: bool,
        aurora_port: str,
        tracker_type: str,
        poll_interval_ms: int,
        reconnect_delay_s: float,
        ports_to_probe: list[str],
        settings_overrides: dict[str, Any],
        tool_id_aliases: dict[str, str],
        bridge_executable: Path,
        socket_path: Path,
        expected_tool_ids: tuple[str, ...] = ("0A", "0B"),
        backend_factories: dict[str, Callable[[], Any]] | None = None,
    ) -> None:
        self.mock_mode = bool(mock_mode)
        self.preferred_backend = str(preferred_backend).strip().lower() or "ndi"
        self.fallback_backend = (
            str(fallback_backend).strip().lower() if fallback_backend not in (None, "") else None
        )
        self.fallback_enabled = bool(fallback_enabled)
        self.aurora_port = aurora_port
        self.tracker_type = str(tracker_type)
        self.poll_interval_ms = max(1, int(poll_interval_ms))
        self.reconnect_delay_s = max(0.1, float(reconnect_delay_s))
        self.ports_to_probe = list(ports_to_probe)
        self.settings_overrides = dict(settings_overrides)
        self.tool_id_aliases = {str(key): str(value) for key, value in tool_id_aliases.items()}
        self.bridge_executable = Path(bridge_executable)
        self.socket_path = Path(socket_path)
        self.expected_tool_ids = tuple(str(tool_id).upper() for tool_id in expected_tool_ids)
        self._backend_factories = dict(backend_factories or {})

        self._backend = None
        self._state = TrackerRuntimeState(
            connection_state="disconnected",
            canonical_state="mock" if self.mock_mode else "disconnected",
            backend_identity=self.backend_identity,
            configured_backend_name=self.preferred_backend,
            selected_backend_name="mock" if self.mock_mode else "",
            fallback_backend_name=self.fallback_backend,
            fallback_used=False,
            last_status_message="Tracking backend router idle",
            backend_details=self._build_backend_details("mock" if self.mock_mode else None),
        )

    def apply_runtime_overrides(
        self,
        *,
        tracker_port: str | None = None,
        poll_ms: int | None = None,
        socket_path: Path | None = None,
        bridge_executable: Path | None = None,
    ) -> None:
        """Apply runtime overrides used by CLI diagnostics and ad hoc bring-up."""
        if tracker_port is not None:
            self.aurora_port = str(tracker_port)
        if poll_ms is not None:
            self.poll_interval_ms = max(1, int(poll_ms))
        if socket_path is not None:
            self.socket_path = Path(socket_path)
        if bridge_executable is not None:
            self.bridge_executable = Path(bridge_executable)
        if self._backend is not None:
            if hasattr(self._backend, "aurora_port"):
                self._backend.aurora_port = self.aurora_port
            if hasattr(self._backend, "poll_interval_ms") and poll_ms is not None:
                self._backend.poll_interval_ms = self.poll_interval_ms
            if hasattr(self._backend, "poll_ms") and poll_ms is not None:
                self._backend.poll_ms = self.poll_interval_ms
            if hasattr(self._backend, "socket_path") and socket_path is not None:
                self._backend.socket_path = self.socket_path
            if hasattr(self._backend, "bridge_executable") and bridge_executable is not None:
                self._backend.bridge_executable = self.bridge_executable

    def probe_capabilities(self) -> dict[str, dict[str, Any]]:
        """Return backend availability checks without starting hardware tracking."""
        backend_names: list[str] = []
        for name in (self.preferred_backend, self.fallback_backend):
            if name and name not in backend_names:
                backend_names.append(name)
        if self.mock_mode and "mock" not in backend_names:
            backend_names.insert(0, "mock")
        report = {name: self._capability_for(name) for name in backend_names}
        self._state.capability_report = copy.deepcopy(report)
        return report

    def start(self) -> None:
        """Select, start, and remember one backend using deterministic fallback rules."""
        if self.is_alive():
            return

        startup_messages = [
            f"Configured preferred tracking backend: {self.preferred_backend}",
            f"Configured fallback backend: {self.fallback_backend or 'none'}",
            f"Fallback enabled: {self.fallback_enabled}",
        ]

        if self.mock_mode:
            self._state.capability_report = {"mock": self._capability_for("mock")}
            startup_messages.append("Mock mode is enabled; selecting mock tracking backend.")
            self._select_backend("mock", startup_messages=startup_messages)
            self._backend.start()
            return

        if self.preferred_backend == "disabled":
            startup_messages.append("Tracking backend is disabled by config.")
            self._state.startup_messages = list(startup_messages)
            self._state.connection_state = "disabled"
            self._state.canonical_state = "disabled"
            self._state.selected_backend_name = "disabled"
            self._state.last_status_message = "Tracking backend disabled by config"
            self._state.backend_details = self._build_backend_details("disabled")
            return

        capability_report = self.probe_capabilities()
        attempt_order = [self.preferred_backend]
        if self.fallback_enabled and self.fallback_backend and self.fallback_backend != self.preferred_backend:
            attempt_order.append(self.fallback_backend)

        selection_errors: list[str] = []
        for index, backend_name in enumerate(attempt_order):
            capability = capability_report.get(backend_name, {})
            if capability.get("available"):
                fallback_used = index > 0
                if fallback_used:
                    startup_messages.append(f"Falling back from {self.preferred_backend} to {backend_name}.")
                else:
                    startup_messages.append(f"Selected backend {backend_name}.")
                self._select_backend(
                    backend_name,
                    fallback_used=fallback_used,
                    startup_messages=startup_messages,
                )
                try:
                    self._backend.start()
                    return
                except Exception as exc:
                    message = f"Tracking backend {backend_name} failed to start: {exc}"
                    startup_messages.append(message)
                    selection_errors.append(f"{backend_name}: start failed: {exc}")
                    self._teardown_failed_backend()
                    continue
            reason = capability.get("reason", "unknown backend availability failure")
            startup_messages.append(f"Backend {backend_name} unavailable: {reason}")
            selection_errors.append(f"{backend_name}: {reason}")

        message = "Tracking backend selection failed. " + "; ".join(selection_errors)
        self._state.startup_messages = list(startup_messages)
        self._state.connection_state = "error"
        self._state.canonical_state = "error"
        self._state.last_error = message
        self._state.error_messages = [message]
        self._state.last_status_message = message
        self._state.backend_details = self._build_backend_details(None)
        raise RuntimeError(message)

    def stop(self, timeout_s: float = 3.0) -> None:
        """Stop the currently selected backend."""
        backend = self._backend
        if backend is not None and hasattr(backend, "stop"):
            backend.stop(timeout_s=timeout_s)
        self._backend = None
        if self.mock_mode:
            self._state.connection_state = "disconnected"
            self._state.canonical_state = "mock"
            self._state.last_status_message = "Mock tracking backend stopped"
        elif self._state.selected_backend_name == "disabled":
            self._state.connection_state = "disabled"
            self._state.canonical_state = "disabled"
        else:
            self._state.connection_state = "disconnected"
            self._state.canonical_state = "disconnected"
            self._state.last_status_message = "Tracking backend router stopped"
        self._state.backend_details = self._build_backend_details(self._state.selected_backend_name or None)

    def is_alive(self) -> bool:
        """Return whether the selected backend is alive."""
        return bool(self._backend is not None and getattr(self._backend, "is_alive", lambda: False)())

    def get_state_snapshot(self) -> TrackerRuntimeState:
        """Return the merged router plus selected-backend runtime state."""
        state = copy.deepcopy(self._state)
        backend = self._backend
        if backend is None:
            return state

        backend_state = backend.get_state_snapshot()
        state.backend_identity = (
            getattr(backend_state, "backend_identity", None)
            or getattr(backend, "backend_identity", "")
            or state.backend_identity
        )
        state.connection_state = backend_state.connection_state
        state.backend_running = backend_state.backend_running
        state.backend_connected = backend_state.backend_connected
        state.socket_connected = backend_state.socket_connected
        state.bridge_running = backend_state.bridge_running
        state.backend_frame_counter = backend_state.backend_frame_counter
        state.latest_frame_number = backend_state.latest_frame_number
        state.latest_timestamp = backend_state.latest_timestamp
        state.last_status_message = backend_state.last_status_message or state.last_status_message
        state.last_error = backend_state.last_error
        state.raw_tool_ids = list(backend_state.raw_tool_ids)
        state.normalized_tool_ids = list(backend_state.normalized_tool_ids)
        state.tool_id_mapping = dict(backend_state.tool_id_mapping)
        state.runtime_role_mappings = dict(backend_state.runtime_role_mappings)
        state.unmapped_tool_ids = list(backend_state.unmapped_tool_ids)
        state.tools = copy.deepcopy(backend_state.tools)
        state.backend_details = dict(getattr(backend_state, "backend_details", {}))
        state.warning_messages = self._derive_warning_messages(state)
        state.error_messages = [state.last_error] if state.last_error else []
        state.canonical_state = self._derive_canonical_state(state)
        state.backend_details = {
            **self._build_backend_details(state.selected_backend_name or None),
            **dict(getattr(backend_state, "backend_details", {})),
        }
        return state

    def get_latest_tool(self, tool_id: str) -> TrackerToolState | None:
        """Return one tool state from the selected backend."""
        if self._backend is None:
            return None
        tool = self._backend.get_latest_tool(tool_id)
        return copy.deepcopy(tool) if tool is not None else None

    def register_timing_listener(self, listener) -> None:
        """Forward a backend timing listener to the currently selected backend."""
        backend = self._backend
        if backend is None:
            raise RuntimeError("Tracking backend is not started.")
        register = getattr(backend, "register_timing_listener", None)
        if not callable(register):
            raise RuntimeError(
                f"Selected tracking backend {self._state.selected_backend_name or 'unknown'} "
                "does not expose backend timing listeners."
            )
        register(listener)

    def unregister_timing_listener(self, listener) -> None:
        """Remove a backend timing listener when the selected backend supports it."""
        backend = self._backend
        if backend is None:
            return
        unregister = getattr(backend, "unregister_timing_listener", None)
        if callable(unregister):
            unregister(listener)

    def _select_backend(
        self,
        backend_name: str,
        *,
        fallback_used: bool = False,
        startup_messages: list[str] | None = None,
    ) -> None:
        self._backend = self._create_backend(backend_name)
        self._state.configured_backend_name = self.preferred_backend
        self._state.selected_backend_name = backend_name
        self._state.fallback_backend_name = self.fallback_backend
        self._state.fallback_used = fallback_used
        self._state.startup_messages = list(startup_messages or [])
        self._state.connection_state = "connecting" if backend_name != "mock" else "tracking"
        self._state.canonical_state = "connecting" if backend_name != "mock" else "mock"
        self._state.last_error = None
        self._state.error_messages = []
        self._state.backend_identity = getattr(self._backend, "backend_identity", backend_name)
        self._state.last_status_message = f"Selected tracking backend: {backend_name}"
        self._state.backend_details = self._build_backend_details(backend_name)

    def _create_backend(self, backend_name: str):
        factory = self._backend_factories.get(backend_name)
        if factory is not None:
            return factory()
        if backend_name == "mock":
            return MockTrackerManager(poll_hz=max(1, int(round(1000.0 / self.poll_interval_ms))))
        if backend_name == "ndi":
            return TrackerBackendNDI(
                aurora_port=self.aurora_port,
                tracker_type=self.tracker_type,
                poll_interval_ms=self.poll_interval_ms,
                reconnect_delay_s=self.reconnect_delay_s,
                ports_to_probe=self.ports_to_probe,
                settings_overrides=self.settings_overrides,
                tool_id_aliases=self.tool_id_aliases,
                expected_tool_ids=self.expected_tool_ids,
            )
        if backend_name == "bridge":
            return TrackerServiceManager(
                bridge_executable=self.bridge_executable,
                socket_path=self.socket_path,
                aurora_port=self.aurora_port,
                poll_ms=self.poll_interval_ms,
            )
        raise ValueError(f"Unsupported tracking backend {backend_name!r}")

    def _capability_for(self, backend_name: str) -> dict[str, Any]:
        if backend_name == "mock":
            return {
                "backend": "mock",
                "available": True,
                "code": "ok",
                "reason": "Mock tracking backend is always available.",
                "details": self._build_backend_details("mock"),
            }
        if backend_name == "disabled":
            return {
                "backend": "disabled",
                "available": False,
                "code": "disabled",
                "reason": "Tracking backend disabled by config.",
                "details": self._build_backend_details("disabled"),
            }

        port_check = self._check_serial_port()
        if not port_check["available"]:
            return {
                "backend": backend_name,
                "available": False,
                "code": str(port_check.get("code", "serial_preflight_failed")),
                "reason": port_check["reason"],
                "details": port_check["details"],
            }

        if backend_name == "ndi":
            try:
                load_ndi_tracker_class()
            except Exception as exc:
                return {
                    "backend": "ndi",
                    "available": False,
                    "code": "backend_import_failure",
                    "reason": f"Python NDI backend import failed: {exc}",
                    "details": self._build_backend_details("ndi"),
                }
            return {
                "backend": "ndi",
                "available": True,
                "code": "ok",
                "reason": "Python NDI backend import and serial preflight passed.",
                "details": self._build_backend_details("ndi"),
            }

        if backend_name == "bridge":
            if not self.bridge_executable.exists():
                return {
                    "backend": "bridge",
                    "available": False,
                    "code": "bridge_executable_missing",
                    "reason": f"Bridge executable missing: {self.bridge_executable}",
                    "details": self._build_backend_details("bridge"),
                }
            if not self.bridge_executable.is_file():
                return {
                    "backend": "bridge",
                    "available": False,
                    "code": "bridge_executable_not_file",
                    "reason": f"Bridge path is not a file: {self.bridge_executable}",
                    "details": self._build_backend_details("bridge"),
                }
            if not os.access(self.bridge_executable, os.X_OK):
                return {
                    "backend": "bridge",
                    "available": False,
                    "code": "bridge_executable_not_executable",
                    "reason": f"Bridge executable is not executable: {self.bridge_executable}",
                    "details": self._build_backend_details("bridge"),
                }
            return {
                "backend": "bridge",
                "available": True,
                "code": "ok",
                "reason": "Bridge executable and serial preflight passed.",
                "details": self._build_backend_details("bridge"),
            }

        return {
            "backend": backend_name,
            "available": False,
            "code": "unsupported_backend",
            "reason": f"Unsupported tracking backend {backend_name!r}",
            "details": self._build_backend_details(backend_name),
        }

    def _check_serial_port(self) -> dict[str, Any]:
        if not self.aurora_port:
            return {
                "available": False,
                "code": "serial_port_not_configured",
                "reason": "Aurora port is not configured.",
                "details": self._build_backend_details(None),
            }
        port_path = Path(self.aurora_port)
        if not port_path.exists():
            return {
                "available": False,
                "code": "serial_port_missing",
                "reason": f"Serial port missing: {self.aurora_port}",
                "details": self._build_backend_details(None),
            }
        if not os.access(port_path, os.R_OK | os.W_OK):
            return {
                "available": False,
                "code": "permission_denied",
                "reason": f"Permission denied for serial port: {self.aurora_port}",
                "details": self._build_backend_details(None),
            }
        return {
            "available": True,
            "code": "ok",
            "reason": "Serial port exists and is accessible.",
            "details": self._build_backend_details(None),
        }

    def _build_backend_details(self, backend_name: str | None) -> dict[str, Any]:
        details: dict[str, Any] = {
            "preferred_backend": self.preferred_backend,
            "selected_backend_name": backend_name,
            "fallback_backend": self.fallback_backend,
            "fallback_enabled": self.fallback_enabled,
            "aurora_port": self.aurora_port,
            "poll_interval_ms": self.poll_interval_ms,
            "expected_tool_ids": list(self.expected_tool_ids),
        }
        if backend_name == "mock":
            details["mock_mode"] = self.mock_mode
            return details
        if backend_name == "ndi":
            details["tracker_type"] = self.tracker_type
            details["tool_aliases"] = dict(self.tool_id_aliases)
            details["tool_configuration_check"] = (
                "aliases_configured" if self.tool_id_aliases else "no_aliases_configured"
            )
            return details
        if backend_name == "bridge":
            details["bridge_executable"] = str(self.bridge_executable)
            details["socket_path"] = str(self.socket_path)
            return details
        return details

    def _teardown_failed_backend(self) -> None:
        backend = self._backend
        self._backend = None
        if backend is not None and hasattr(backend, "stop"):
            try:
                backend.stop(timeout_s=1.0)
            except Exception:
                pass

    def _derive_canonical_state(self, state: TrackerRuntimeState) -> str:
        if state.selected_backend_name == "disabled" or state.connection_state == "disabled":
            return "disabled"
        if state.selected_backend_name == "mock":
            return "mock"
        if state.connection_state in {"starting", "connecting", "connected", "initialized", "tracking_started", "reconnecting"}:
            return "connecting"
        if state.connection_state in {"error"}:
            return "error"
        if state.connection_state in {"disconnected", "stopped"}:
            return "disconnected"
        if state.backend_frame_counter <= 0 and state.latest_frame_number is None:
            return "streaming_degraded" if state.backend_connected else "connecting"
        return "streaming_degraded" if self._derive_warning_messages(state) else "streaming_healthy"

    def _derive_warning_messages(self, state: TrackerRuntimeState) -> list[str]:
        warnings: list[str] = []
        if state.backend_connected and state.backend_frame_counter <= 0 and state.latest_frame_number is None:
            warnings.append("tracker_connected_but_no_frames")
        if state.backend_connected and not state.raw_tool_ids:
            warnings.append("no_live_tools_visible")
        if state.unmapped_tool_ids:
            warnings.append("unmapped_live_tool_ids")
        missing_runtime_roles = [
            tool_id
            for tool_id in self.expected_tool_ids
            if tool_id not in state.runtime_role_mappings and tool_id not in state.tools
        ]
        if missing_runtime_roles:
            warnings.append(f"missing_expected_tools:{','.join(missing_runtime_roles)}")
        invalid_tools = sorted(
            tool_id
            for tool_id, tool in state.tools.items()
            if tool.status.startswith("invalid") or tool.valid is False
        )
        if invalid_tools:
            warnings.append(f"invalid_transforms:{','.join(invalid_tools)}")
        return warnings

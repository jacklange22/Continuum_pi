"""Registration tab controller backed by the shared registration service."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from continuum_robot.config.schemas import RegistrationWorkflowConfig


@dataclass
class RegistrationViewState:
    """UI-facing registration workflow state."""

    active: bool = False
    capture_tool_id: str = "0B"
    coil_tool_id: str = "0A"
    landmark_labels: list[str] = field(default_factory=list)
    current_label: str | None = None
    captures_per_landmark: int = 0
    captured_counts: dict[str, int] = field(default_factory=dict)
    raw_samples_by_label: dict[str, list[list[float]]] = field(default_factory=dict)
    latest_sample_by_label: dict[str, list[float]] = field(default_factory=dict)
    truth_points_in_sw_by_label: dict[str, list[float]] = field(default_factory=dict)
    last_error: str | None = None
    last_result_path: str | None = None
    pending_accept: bool = False
    overwrite_target_path: str | None = None
    overwrite_required: bool = False
    fre_mm: float | None = None
    max_residual_mm: float | None = None
    residuals_by_label: dict[str, list[float]] = field(default_factory=dict)
    validation_metrics: dict[str, object] = field(default_factory=dict)
    latest_validation_summary: dict[str, object] = field(default_factory=dict)
    T_robot_aurora: list[list[float]] | None = None
    capture_geometry_status: str = "coil origin"
    current_tracked_xyz_mm: list[float] | None = None
    current_tracked_frame_id: int | None = None
    current_tracking_status: str = "waiting_for_tracking"
    completed_labels: list[str] = field(default_factory=list)
    accepted_registration_valid: bool = False
    trust_state: str = "missing"
    trust_message: str = "No accepted registration saved."
    runtime_tip_mode: str = "latest_accepted"
    runtime_tip_trust_level: str = "missing"
    runtime_tip_mode_message: str = "No runtime tip mode selected."
    runtime_tip_mode_options: list[str] = field(
        default_factory=lambda: ["latest_accepted", "quick_4_point", "coil_as_tip"]
    )
    live_chain_state: str = "missing_registration"
    live_chain_message: str = "Live robot-frame pose is unavailable until registration is saved."
    comparison_message: str = "Repeat registration runs to build comparison data."
    validation_lines: list[str] = field(default_factory=list)
    registration_mode: str = "simple"
    available_model_points_by_label: dict[str, list[float]] = field(default_factory=dict)
    available_model_labels: list[str] = field(default_factory=list)
    selectable_model_labels: list[str] = field(default_factory=list)
    model_point_display_labels: dict[str, str] = field(default_factory=dict)
    model_point_enabled: dict[str, bool] = field(default_factory=dict)
    selected_model_labels: list[str] = field(default_factory=list)
    selection_editable: bool = True
    averaged_points_by_label: dict[str, list[float]] = field(default_factory=dict)
    result_status: str = "Not solved"
    status_message: str = "Registration idle."


@dataclass
class RegistrationActionResult:
    """Small controller return payload after one registration save."""

    output_path: Path
    payload: dict


class RegistrationController:
    """Owns guided 4-point capture, solve, and save actions."""

    REQUIRED_SELECTION_COUNT = 4

    def __init__(
        self,
        registration_service,
        registration_config: RegistrationWorkflowConfig,
    ) -> None:
        self.registration_service = registration_service
        self.config = registration_config
        initial = self.registration_service.get_snapshot()
        self.state = RegistrationViewState(
            capture_tool_id=initial.capture_tool_id,
            coil_tool_id=registration_config.coil_tool_id,
            landmark_labels=list(initial.labels),
            captures_per_landmark=initial.captures_per_landmark,
            captured_counts={label: 0 for label in initial.labels},
            current_label=initial.current_label,
            capture_geometry_status=self._capture_geometry_status(registration_config),
        )
        self._selected_model_labels: list[str] = []
        self._apply_snapshot(initial)
        self.refresh()

    def refresh(self) -> RegistrationViewState:
        snapshot = self.registration_service.get_snapshot()
        self._apply_snapshot(snapshot)
        live_point = self.registration_service.peek_current_measurement_point()
        trust_summary = self.registration_service.get_registration_trust_summary()
        self.state.current_tracked_xyz_mm = live_point.get("point_xyz_mm")
        self.state.current_tracked_frame_id = live_point.get("frame_number")
        self.state.current_tracking_status = str(live_point.get("status", "unknown"))
        self.state.trust_state = str(trust_summary.get("trust_state", "missing"))
        self.state.trust_message = str(trust_summary.get("trust_message", "No accepted registration saved."))
        tracking_snapshot = self.registration_service.tracking_service.get_snapshot()
        self.state.runtime_tip_mode = str(getattr(tracking_snapshot, "runtime_tip_mode", "latest_accepted") or "latest_accepted")
        self.state.runtime_tip_trust_level = str(getattr(tracking_snapshot, "runtime_tip_trust_level", "missing") or "missing")
        self.state.runtime_tip_mode_message = str(
            getattr(tracking_snapshot, "runtime_tip_mode_message", "No runtime tip mode selected.") or "No runtime tip mode selected."
        )
        self.state.live_chain_state = str(trust_summary.get("live_chain_state", "missing_registration"))
        self.state.live_chain_message = str(
            trust_summary.get(
                "live_chain_message",
                "Live robot-frame pose is unavailable until an accepted registration is saved.",
            )
        )
        self.state.comparison_message = str(
            trust_summary.get("comparison_message", "Repeat registration runs to build comparison data.")
        )
        self.state.validation_lines = [
            str(line)
            for line in trust_summary.get("detail_lines", [])
            if str(line).strip()
        ]
        if self.state.pending_accept and self.state.fre_mm is not None:
            pending_lines = [
                f"Pending solve FRE / RMSE: {self.state.fre_mm:.3f} mm",
            ]
            if self.state.max_residual_mm is not None:
                pending_lines.append(f"Pending solve max residual: {self.state.max_residual_mm:.3f} mm")
            worst_label = self.state.validation_metrics.get("worst_landmark_label")
            worst_residual = self.state.validation_metrics.get("worst_landmark_residual_mm")
            if worst_label and worst_residual is not None:
                pending_lines.append(f"Pending solve worst landmark: {worst_label} = {float(worst_residual):.3f} mm")
            pending_lines.append("Save the accepted registration to add this run to the repeated-validation history.")
            self.state.validation_lines = [*pending_lines, *self.state.validation_lines]
        self.state.accepted_registration_valid = bool(
            trust_summary.get("accepted_exists")
            and self.state.trust_state != "invalid"
        )
        return self.state

    def set_runtime_tip_mode(self, mode: str) -> None:
        try:
            self.registration_service.tracking_service.set_runtime_tip_mode(str(mode))
            self.state.last_error = None
            self.state.status_message = (
                f"Runtime tip mode set to {str(mode).replace('_', ' ')}."
            )
            self.refresh()
        except Exception as exc:
            self.state.last_error = str(exc)
            self.state.status_message = f"Runtime tip mode update failed: {exc}"
            raise

    def set_selected_model_point(self, slot_index: int, label: str) -> None:
        if not self.state.selection_editable:
            raise RuntimeError("Model-point selection is fixed in the current registration mode.")
        if self.state.active or self.state.pending_accept:
            raise RuntimeError("Restart the registration session before changing model points.")
        if slot_index < 0 or slot_index >= self.REQUIRED_SELECTION_COUNT:
            raise IndexError(f"Selection slot {slot_index} is out of range.")
        if label not in self.state.selectable_model_labels:
            raise ValueError(f"Unknown or disabled model point: {label}")
        proposed = list(self.state.selected_model_labels or self._selected_model_labels)
        if len(proposed) < self.REQUIRED_SELECTION_COUNT:
            proposed = self._fill_selected_labels(proposed, self.state.selectable_model_labels)
        proposed[slot_index] = str(label)
        duplicates = self._duplicate_labels(proposed)
        if duplicates:
            raise RuntimeError(
                f"Each registration slot must use a unique model point. Duplicate selection: {', '.join(duplicates)}"
            )
        self._selected_model_labels = proposed
        self.state.selected_model_labels = list(proposed)
        self.state.last_error = None
        self.state.status_message = f"Selected model points: {', '.join(self._selected_model_labels)}."

    def toggle_selected_model_point(self, label: str) -> None:
        if not self.state.selection_editable:
            raise RuntimeError("Model-point selection is fixed in the current registration mode.")
        if self.state.active or self.state.pending_accept:
            raise RuntimeError("Restart the registration session before changing model points.")
        if label not in self.state.selectable_model_labels:
            raise ValueError(f"Model point {label} is not enabled for registration.")

        selected = self._filter_selected_labels(
            self.state.selected_model_labels or self._selected_model_labels,
            self.state.available_model_labels,
        )
        if label in selected:
            selected = [item for item in selected if item != label]
        else:
            if len(selected) >= self.REQUIRED_SELECTION_COUNT:
                raise RuntimeError("Only four model points can be selected. Deselect one point before adding another.")
            selected.append(label)

        self._selected_model_labels = list(selected)
        self.state.selected_model_labels = list(selected)
        self.state.last_error = None
        if self.state.selected_model_labels:
            self.state.status_message = f"Selected model points: {', '.join(self.state.selected_model_labels)}."
        else:
            self.state.status_message = "Choose four model points before starting registration."

    def begin_session(self, capture_tool_id: str | None = None) -> None:
        try:
            blockers = self.begin_session_blockers()
            if blockers:
                raise RuntimeError(blockers[0])
            if capture_tool_id is not None and capture_tool_id != self.config.capture_tool_id:
                raise RuntimeError(
                    f"Registration controller is configured for capture tool {self.config.capture_tool_id}; "
                    f"override {capture_tool_id} is not supported from the GUI controller"
                )
            if self.state.selection_editable:
                labels = list(self.state.selected_model_labels)
                nominal = {
                    label: list(self.state.available_model_points_by_label[label])
                    for label in labels
                }
                snapshot = self.registration_service.begin_session(
                    labels=labels,
                    nominal_landmarks_robot_xyz_mm=nominal,
                )
            else:
                snapshot = self.registration_service.begin_session()
            self._apply_snapshot(snapshot)
            self.state.last_error = None
            self.state.status_message = (
                f"Registration started for {', '.join(self.state.selected_model_labels)}. "
                "Capture one or more samples for the active point, then mark it complete."
            )
            self.refresh()
        except Exception as exc:
            self.state.last_error = str(exc)
            self.state.status_message = f"Registration start failed: {exc}"
            raise

    def capture_current_label_sample(self) -> list[float]:
        if self.state.current_label is None:
            raise RuntimeError("No current registration point is selected")
        return self.capture_label_sample(self.state.current_label)

    def capture_label_sample(self, label: str) -> list[float]:
        try:
            # All GUI capture actions must flow through RegistrationService so
            # the active session snapshot and persisted record stay consistent.
            sample = self.registration_service.capture_sample(label)
            snapshot = self.registration_service.get_snapshot()
            self._apply_snapshot(snapshot)
            self.state.last_error = None
            count = snapshot.captured_counts.get(label, 0)
            self.state.status_message = (
                f"Captured sample {count} for {label}. "
                "Capture more if needed, then mark the point complete."
            )
            self.refresh()
            return sample
        except Exception as exc:
            self.state.last_error = str(exc)
            self.state.status_message = f"Capture failed: {exc}"
            raise

    def complete_current_label(self) -> None:
        try:
            previous = self.state.current_label
            next_label = self.registration_service.complete_landmark()
            self._apply_snapshot(self.registration_service.get_snapshot())
            if next_label is None:
                self.state.status_message = (
                    f"{previous} marked complete. All four points are ready. Solve the registration next."
                )
            else:
                self.state.status_message = f"{previous} marked complete. Continue with {next_label}."
            self.refresh()
        except Exception as exc:
            self.state.last_error = str(exc)
            self.state.status_message = f"Could not complete point: {exc}"
            raise

    def solve_session(self) -> dict:
        blockers = self.solve_blockers()
        if blockers:
            raise RuntimeError(f"Registration is not ready to solve. {self.solve_readiness_message()}")
        try:
            payload = self.registration_service.solve_registration()
            self._apply_snapshot(self.registration_service.get_snapshot())
            samples_used = self.total_samples_captured()
            max_residual_text = (
                f" Max residual = {self.state.max_residual_mm:.3f} mm."
                if self.state.max_residual_mm is not None
                else ""
            )
            self.state.status_message = (
                f"Registration solved. RMSE/FRE = {self.state.fre_mm:.3f} mm using {samples_used} captured samples. "
                f"{max_residual_text} Review the result, landmark residuals, and trust summary, then save the accepted registration."
            )
            self.refresh()
            return payload
        except Exception as exc:
            self.state.last_error = str(exc)
            self.state.status_message = f"Registration solve failed: {exc}"
            raise

    def finish_session(self) -> RegistrationActionResult:
        """Compatibility helper for the older solve-and-save GUI tests."""
        self.solve_session()
        return self.save_registration(confirm_overwrite=True)

    def save_registration(self, *, confirm_overwrite: bool = False) -> RegistrationActionResult:
        overwrite_target = self._overwrite_target_path()
        if overwrite_target.exists() and not confirm_overwrite:
            self.refresh()
            self.state.status_message = (
                f"Saving will overwrite {overwrite_target.name}. Confirm overwrite to continue."
            )
            raise RuntimeError("Registration save requires explicit overwrite confirmation.")
        try:
            payload = self.registration_service.get_snapshot().pending_record or {}
            output_path = self.registration_service.accept_registration()
            snapshot = self.registration_service.get_snapshot()
            self._apply_snapshot(snapshot)
            self.state.status_message = f"Registration saved to {output_path.name}."
            self.refresh()
            return RegistrationActionResult(output_path=output_path, payload=payload)
        except Exception as exc:
            self.state.last_error = str(exc)
            self.state.status_message = f"Registration save failed: {exc}"
            raise

    def retry_session(self) -> None:
        if self.state.selection_editable:
            self.begin_session()
            self.state.status_message = (
                f"Registration restarted with {', '.join(self.state.selected_model_labels)}."
            )
        else:
            snapshot = self.registration_service.retry_session()
            self._apply_snapshot(snapshot)
            self.state.status_message = "Registration restarted."
            self.refresh()

    def load_latest_result(self) -> None:
        payload = self.registration_service.load_latest_accepted()
        if payload is not None:
            self._apply_snapshot(self.registration_service.get_snapshot())
            self.state.status_message = "Loaded the latest accepted registration."
            self.state.last_error = None
        else:
            self.state.last_error = None
            self.state.status_message = "No accepted registration file was found."
        self.refresh()

    def is_ready_to_complete_current(self) -> bool:
        if self.state.current_label is None:
            return False
        return self.state.captured_counts.get(self.state.current_label, 0) >= self.state.captures_per_landmark

    def can_begin_session(self) -> bool:
        return not self.begin_session_blockers()

    def selection_is_ready(self) -> bool:
        if not self.state.selection_editable:
            return bool(self.state.landmark_labels)
        labels = list(self.state.selected_model_labels)
        return (
            len(labels) == self.REQUIRED_SELECTION_COUNT
            and len(set(labels)) == self.REQUIRED_SELECTION_COUNT
            and all(label in self.state.selectable_model_labels for label in labels)
        )

    def is_ready_to_solve(self) -> bool:
        return not self.solve_blockers()

    def begin_session_blockers(self) -> list[str]:
        blockers: list[str] = []
        if self.state.active:
            blockers.append("Registration session is already active.")
        if self.state.pending_accept:
            blockers.append("Save or restart the solved registration before beginning a new one.")
        if self.state.selection_editable and not self.selection_is_ready():
            blockers.append("Choose four unique model points before starting registration.")
        return blockers

    def begin_session_readiness_message(self) -> str:
        blockers = self.begin_session_blockers()
        return blockers[0] if len(blockers) == 1 else "; ".join(blockers) if blockers else "Ready to begin registration."

    def solve_blockers(self) -> list[str]:
        blockers: list[str] = []
        if not self.state.active:
            blockers.append("Begin the registration session first.")
        if not self.selection_is_ready():
            blockers.append("Choose four unique model points before solving registration.")
        incomplete = [
            f"{label} ({self.state.captured_counts.get(label, 0)}/{self.state.captures_per_landmark})"
            for label in self.state.landmark_labels
            if self.state.captured_counts.get(label, 0) < self.state.captures_per_landmark
        ]
        if incomplete:
            blockers.append("Incomplete captures: " + ", ".join(incomplete))
        return blockers

    def solve_readiness_message(self) -> str:
        blockers = self.solve_blockers()
        if not blockers:
            return "Registration is ready to solve."
        return "; ".join(blockers)

    def total_samples_captured(self) -> int:
        return sum(len(samples) for samples in self.state.raw_samples_by_label.values())

    def _overwrite_target_path(self) -> Path:
        return self.registration_service.repository.root_dir / "latest_registration.json"

    def _apply_snapshot(self, snapshot) -> None:
        self.state.active = snapshot.active
        self.state.capture_tool_id = snapshot.capture_tool_id
        self.state.coil_tool_id = self.config.coil_tool_id
        self.state.landmark_labels = list(snapshot.labels)
        self.state.current_label = snapshot.current_label
        self.state.captures_per_landmark = snapshot.captures_per_landmark
        self.state.captured_counts = dict(snapshot.captured_counts)
        self.state.raw_samples_by_label = {
            label: [list(sample) for sample in samples]
            for label, samples in snapshot.raw_points_by_label.items()
        }
        self.state.averaged_points_by_label = {
            label: list(point)
            for label, point in snapshot.averaged_points_by_label.items()
        }
        self.state.latest_sample_by_label = {
            label: list(samples[-1])
            for label, samples in self.state.raw_samples_by_label.items()
            if samples
        }
        self.state.truth_points_in_sw_by_label = dict(
            snapshot.truth_points_in_sw_by_label or snapshot.nominal_landmarks_robot_xyz_mm
        )
        self.state.fre_mm = snapshot.fre_mm
        max_residual = snapshot.validation_metrics.get("max_residual_mm")
        self.state.max_residual_mm = float(max_residual) if max_residual is not None else None
        self.state.residuals_by_label = dict(snapshot.residuals_by_label)
        self.state.validation_metrics = dict(snapshot.validation_metrics)
        self.state.latest_validation_summary = dict(snapshot.latest_validation_summary)
        self.state.T_robot_aurora = snapshot.T_robot_aurora
        self.state.pending_accept = bool(snapshot.pending_accept)
        self.state.last_result_path = snapshot.accepted_output_path or snapshot.latest_accepted_path
        self.state.last_error = snapshot.health.last_error
        overwrite_target = self._overwrite_target_path()
        self.state.overwrite_target_path = str(overwrite_target)
        self.state.overwrite_required = bool(snapshot.pending_accept and overwrite_target.exists())
        self.state.completed_labels = self._completed_labels(snapshot)
        self.state.registration_mode = str(snapshot.health.details.get("registration_mode", "simple"))
        self.state.selection_editable = self.state.registration_mode == "simple"
        available_points, display_labels, enabled_lookup, ordered_labels = self._derive_available_model_points(snapshot)
        self.state.available_model_points_by_label = available_points
        self.state.available_model_labels = list(ordered_labels)
        self.state.selectable_model_labels = [
            label for label in ordered_labels if enabled_lookup.get(label, True)
        ]
        self.state.model_point_display_labels = dict(display_labels)
        self.state.model_point_enabled = dict(enabled_lookup)
        if snapshot.active or snapshot.pending_accept or snapshot.averaged_points_by_label:
            self._selected_model_labels = self._filter_selected_labels(snapshot.labels, ordered_labels)
        elif self._selected_model_labels:
            self._selected_model_labels = self._filter_selected_labels(self._selected_model_labels, ordered_labels)
        else:
            self._selected_model_labels = self._fill_selected_labels(
                self._filter_selected_labels(self.config.landmark_labels, self.state.selectable_model_labels),
                self.state.selectable_model_labels,
            )
        self.state.selected_model_labels = list(self._selected_model_labels)
        self.state.result_status = self._result_status(snapshot)

    @staticmethod
    def _completed_labels(snapshot) -> list[str]:
        completed: list[str] = []
        labels = list(snapshot.labels)
        for index, label in enumerate(labels):
            count = snapshot.captured_counts.get(label, 0)
            if count < snapshot.captures_per_landmark:
                continue
            if snapshot.current_label is None or index < snapshot.current_landmark_index:
                completed.append(label)
        return completed

    @staticmethod
    def _capture_geometry_status(config: RegistrationWorkflowConfig) -> str:
        if config.capture_tool_tip_transform is not None:
            return "Explicit 4x4 pen-tip transform"
        if config.penprobe_file:
            return "Pen-probe tip file"
        return "Coil origin / no explicit tip offset"

    def _derive_available_model_points(
        self,
        snapshot,
    ) -> tuple[dict[str, list[float]], dict[str, str], dict[str, bool], list[str]]:
        grouped = dict(snapshot.group_by_label or {})
        truth_points = dict(snapshot.truth_points_in_sw_by_label or {})
        if grouped:
            labels = [
                label
                for label in snapshot.labels
                if grouped.get(label) == "model" and label in truth_points
            ]
            extra = [
                label
                for label, group in grouped.items()
                if group == "model" and label not in labels and label in truth_points
            ]
            ordered = [*labels, *extra]
            return (
                {
                    label: list(truth_points[label])
                    for label in ordered
                },
                {label: label for label in ordered},
                {label: True for label in ordered},
                ordered,
            )
        points: dict[str, list[float]] = {}
        display_labels: dict[str, str] = {}
        enabled_lookup: dict[str, bool] = {}
        ordered: list[str] = []
        if self.config.candidate_landmarks:
            for landmark in self.config.candidate_landmarks:
                ordered.append(landmark.id)
                points[landmark.id] = list(landmark.xyz_mm)
                display_labels[landmark.id] = landmark.display_label or landmark.id
                enabled_lookup[landmark.id] = bool(landmark.enabled)
        else:
            nominal_from_config = dict(self.config.nominal_landmarks_robot_xyz_mm or {})
            for label in self.config.landmark_labels:
                if label in nominal_from_config and label not in ordered:
                    ordered.append(label)
            for label in nominal_from_config:
                if label not in ordered:
                    ordered.append(label)
            for label in ordered:
                points[label] = list(nominal_from_config[label])
                display_labels[label] = label
                enabled_lookup[label] = True
        if snapshot.nominal_landmarks_robot_xyz_mm:
            for label, point in snapshot.nominal_landmarks_robot_xyz_mm.items():
                if label not in points:
                    ordered.append(str(label))
                    display_labels[str(label)] = str(label)
                    enabled_lookup[str(label)] = True
                points[str(label)] = list(point)
        if not ordered:
            ordered = list(points.keys())
        return points, display_labels, enabled_lookup, ordered

    def _fill_selected_labels(self, labels: list[str], selectable_labels: list[str]) -> list[str]:
        ordered: list[str] = []
        seen: set[str] = set()
        for label in labels:
            if label in selectable_labels and label not in seen:
                ordered.append(label)
                seen.add(label)
        for label in selectable_labels:
            if label not in seen:
                ordered.append(label)
                seen.add(label)
            if len(ordered) >= self.REQUIRED_SELECTION_COUNT:
                break
        return ordered[: self.REQUIRED_SELECTION_COUNT]

    @staticmethod
    def _filter_selected_labels(labels: list[str], available_labels: list[str]) -> list[str]:
        ordered: list[str] = []
        seen: set[str] = set()
        for label in labels:
            if label in available_labels and label not in seen:
                ordered.append(label)
                seen.add(label)
        return ordered

    @staticmethod
    def _duplicate_labels(labels: list[str]) -> list[str]:
        seen: set[str] = set()
        duplicates: list[str] = []
        for label in labels:
            if label in seen and label not in duplicates:
                duplicates.append(label)
            seen.add(label)
        return duplicates

    def _result_status(self, snapshot) -> str:
        if snapshot.pending_accept:
            return "Solved - review and save"
        if snapshot.latest_accepted_path:
            max_residual = snapshot.validation_metrics.get("max_residual_mm")
            if snapshot.fre_mm is not None and self.config.max_fre_mm is not None and snapshot.fre_mm > self.config.max_fre_mm:
                return "Accepted - FRE above limit"
            if max_residual is not None and self.config.max_fre_mm is not None and float(max_residual) > self.config.max_fre_mm:
                return "Accepted - inspect landmark residuals"
            return "Accepted"
        if snapshot.health.last_error:
            return "Invalid"
        return "Not solved"

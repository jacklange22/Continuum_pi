"""Manual two-segment startup validation experiment.

This workflow records staged all-8 telemetry snapshots for a stacked
two-segment robot. It does not command servo motion and does not implement
automatic two-segment pretension.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
import csv
import json
from pathlib import Path
from typing import Any

from continuum_robot.experiments.framework import BaseExperiment, ExperimentHardwareRequirements, ExperimentSession
from continuum_robot.experiments.plotting import color, create_figure, save_figure, style_axes
from continuum_robot.experiments.schemas import ExperimentTimeseriesSample
from continuum_robot.two_segment import TwoSegmentPoseObservation, build_two_segment_foundation_metadata


STARTUP_SCHEMA_VERSION = "manual_two_segment_startup_v2"
EXPERIMENT_NAME = "two_segment_startup_validation"
# Stage order is structured by physical role: bottom (proximal) is pretensioned
# first, then top (distal), then the bottom is rechecked because top tendons pass
# through the bottom segment and can disturb its tension.
STAGE_ORDER = [
    "baseline",
    "bottom_pretensioned",
    "top_pretensioned",
    "bottom_recheck",
    "final_accept",
]
# Backwards-compat alias map: older configs/workflow files used segment_a/b
# stage names. Either form is accepted on input; outputs use bottom/top.
STAGE_ALIASES = {
    "segment_a_pretensioned": "bottom_pretensioned",
    "segment_b_pretensioned": "top_pretensioned",
    "segment_a_recheck": "bottom_recheck",
}
STAGE_LABELS = {
    "baseline": "Baseline",
    "bottom_pretensioned": "Bottom (Proximal) Pretensioned",
    "top_pretensioned": "Top (Distal) Pretensioned",
    "bottom_recheck": "Bottom Recheck",
    "final_accept": "Final Accept",
}


def normalize_stage_name(stage: str) -> str:
    """Resolve legacy A/B stage names to the canonical bottom/top names."""
    raw = str(stage or "").strip()
    if raw in STAGE_ALIASES:
        return STAGE_ALIASES[raw]
    return raw
DUAL_SEGMENT_BLOCK_MESSAGE = (
    "Two-segment startup validation requires operating_mode=dual_segment. "
    "This workflow records manual all-8 startup stages; it does not run automatic two-segment pretension."
)


@dataclass
class TwoSegmentStartupValidationConfig:
    """Configuration for manual staged two-segment startup validation."""

    allow_missing_current: bool = True
    capture_tracker_snapshot: bool = True
    final_accept: bool = True
    capture_stage: str = ""
    workflow_state_path: str = ""
    reset_workflow_state: bool = False
    stage_notes: dict[str, str] = field(default_factory=dict)
    run_trust_mode: str = "servo_only_startup_validation"

    @classmethod
    def from_dict(cls, payload: dict[str, Any] | None) -> "TwoSegmentStartupValidationConfig":
        payload = dict(payload or {})
        return cls(
            allow_missing_current=bool(payload.get("allow_missing_current", True)),
            capture_tracker_snapshot=bool(payload.get("capture_tracker_snapshot", True)),
            final_accept=bool(payload.get("final_accept", True)),
            capture_stage=str(payload.get("capture_stage", "") or ""),
            workflow_state_path=str(payload.get("workflow_state_path", "") or ""),
            reset_workflow_state=bool(payload.get("reset_workflow_state", False)),
            stage_notes={
                str(key): str(value)
                for key, value in dict(payload.get("stage_notes", {}) or {}).items()
            },
            run_trust_mode=str(payload.get("run_trust_mode", "servo_only_startup_validation") or "servo_only_startup_validation"),
        )


class TwoSegmentStartupValidationExperiment(BaseExperiment):
    """Structured manual all-8 startup validation for dual_segment mode."""

    name = EXPERIMENT_NAME
    description = (
        "Capture staged manual all-8 startup snapshots for Segment A/proximal and Segment B/distal. "
        "This does not command motion or run automatic two-segment pretension."
    )
    hardware_requirements = ExperimentHardwareRequirements(
        tracking_required=False,
        servo_required=True,
        registration_required=False,
        mock_compatible=True,
    )

    def __init__(self, config: TwoSegmentStartupValidationConfig) -> None:
        super().__init__(config)
        self.config: TwoSegmentStartupValidationConfig = config

    @classmethod
    def from_dict(cls, payload: dict[str, Any] | None = None) -> "TwoSegmentStartupValidationExperiment":
        return cls(TwoSegmentStartupValidationConfig.from_dict(payload))

    def precheck(self, session: ExperimentSession) -> None:
        context = session.context.settings.robot.operating_context()
        if context.operating_mode != "dual_segment":
            raise RuntimeError(DUAL_SEGMENT_BLOCK_MESSAGE)
        expected_ids = [int(value) for value in context.expected_servo_ids]
        commanded_ids = [int(value) for value in context.commanded_servo_ids]
        if expected_ids != [1, 2, 3, 4, 5, 6, 7, 8] or commanded_ids != expected_ids:
            raise RuntimeError(
                "Two-segment startup validation requires all 8 expected/commanded servo IDs [1,2,3,4,5,6,7,8]; "
                f"resolved expected={expected_ids}, commanded={commanded_ids}."
            )
        if session.context.servo_service is None or not bool(getattr(session.context.servo_service, "is_connected", False)):
            raise RuntimeError("Two-segment startup validation requires a connected ServoService.")
        segments = dict(context.metadata().get("segments", {}) or {})
        for key, expected in {"segment_a": [1, 2, 3, 4], "segment_b": [5, 6, 7, 8]}.items():
            data = dict(segments.get(key, {}) or {})
            if [int(value) for value in data.get("servo_ids", [])] != expected:
                raise RuntimeError(f"{key} servo IDs must be {expected}; resolved {data.get('servo_ids')}.")
        assembly_issues = list(getattr(context, "physical_assembly_issues", []) or [])
        if assembly_issues:
            raise RuntimeError(
                "Two-segment startup validation requires a valid bottom/top physical assembly. "
                f"Issues: {'; '.join(assembly_issues)}"
            )
        session.set_stage("precheck", "passed", "dual_segment all-8 manual startup validation is ready.")

    def _role_for_stage(self, stage: str, *, context) -> dict[str, Any]:
        assembly = dict(context.metadata().get("physical_assembly") or {})
        bottom_key = str(assembly.get("bottom_segment_key") or "")
        top_key = str(assembly.get("top_segment_key") or "")
        bottom_ids = [int(value) for value in list(assembly.get("bottom_servo_ids") or [])]
        top_ids = [int(value) for value in list(assembly.get("top_servo_ids") or [])]
        role: str | None
        targeted_ids: list[int]
        if stage == "bottom_pretensioned" or stage == "bottom_recheck":
            role = "proximal"
            targeted_segment_key = bottom_key
            targeted_ids = list(bottom_ids)
        elif stage == "top_pretensioned":
            role = "distal"
            targeted_segment_key = top_key
            targeted_ids = list(top_ids)
        else:
            role = None
            targeted_segment_key = ""
            targeted_ids = []
        return {
            "stage_role": role or "",
            "targeted_segment_key": targeted_segment_key,
            "targeted_servo_ids": targeted_ids,
            "bottom_segment_key": bottom_key,
            "top_segment_key": top_key,
            "bottom_servo_ids": list(bottom_ids),
            "top_servo_ids": list(top_ids),
            "physical_assembly": dict(assembly),
        }

    def execute(self, session: ExperimentSession) -> None:
        context = session.context.settings.robot.operating_context()
        self._sync_calibration_context(session)
        workflow_state_path = self._workflow_state_path(session)
        stage_snapshots = self._load_workflow_snapshots(workflow_state_path)
        stages_to_capture = self._stages_to_capture()
        if len(stages_to_capture) == 1:
            self._require_previous_stages(stage=stages_to_capture[0], snapshots=stage_snapshots)
        baseline_currents = self._baseline_currents_from_snapshots(stage_snapshots)
        if self.config.reset_workflow_state or stages_to_capture == list(STAGE_ORDER) or stages_to_capture == ["baseline"]:
            stage_snapshots = []
            baseline_currents = {}
        for capture_index, stage in enumerate(stages_to_capture):
            session.raise_if_stop_requested()
            index = STAGE_ORDER.index(stage)
            snapshot = self._capture_stage(
                session=session,
                stage=stage,
                step_index=index,
                baseline_currents=baseline_currents,
            )
            if stage == "baseline":
                baseline_currents = {
                    int(servo_id): data.get("signed_raw_current_ma")
                    for servo_id, data in dict(snapshot["servos"]).items()
                }
            stage_snapshots = _replace_stage_snapshot(stage_snapshots, snapshot)
            self._save_workflow_snapshots(workflow_state_path, stage_snapshots)
            session.add_sample(_sample_from_stage(session=session, stage_snapshot=snapshot, step_index=index))
            session.update_progress(capture_index + 1, len(stages_to_capture), {"stage": stage})
        artifact_path = ""
        all_required_captured = all(stage in {str(snapshot.get("stage")) for snapshot in stage_snapshots} for stage in STAGE_ORDER)
        if self.config.final_accept and "final_accept" in stages_to_capture and all_required_captured:
            artifact_path = str(self._save_final_startup_artifact(session=session, stage_snapshots=stage_snapshots))
        elif self.config.final_accept and "final_accept" in stages_to_capture and not all_required_captured:
            missing = [stage for stage in STAGE_ORDER if stage not in {str(snapshot.get("stage")) for snapshot in stage_snapshots}]
            raise RuntimeError(f"Final accept requires all prior startup stages; missing {missing}.")
        tracker_available = any(bool(stage.get("tracker", {}).get("available")) for stage in stage_snapshots)
        missing_current_count = sum(
            1
            for stage in stage_snapshots
            for servo in dict(stage.get("servos", {}) or {}).values()
            if dict(servo).get("signed_raw_current_ma") is None
        )
        if missing_current_count:
            session.add_warning(
                "One or more servo current estimates were unavailable. Positions were captured; current/load proxy rows are partial."
            )
        session.set_metric("startup_schema_version", STARTUP_SCHEMA_VERSION)
        session.set_metric("startup_type", "manual_two_segment_startup")
        session.set_metric("operating_mode", context.operating_mode)
        session.set_metric("stage_order", list(STAGE_ORDER))
        session.set_metric("physical_assembly", self._physical_assembly_metadata(context=context))
        session.set_metric("bottom_segment_key", context.bottom_segment_key)
        session.set_metric("top_segment_key", context.top_segment_key)
        session.set_metric("bottom_servo_ids", list(context.bottom_servo_ids or []))
        session.set_metric("top_servo_ids", list(context.top_servo_ids or []))
        captured_stage_names = {str(stage.get("stage")) for stage in stage_snapshots}
        session.set_metric("stage_completion", {stage: stage in captured_stage_names for stage in STAGE_ORDER})
        session.set_metric("stage_snapshots", stage_snapshots)
        session.set_metric("captured_this_run", list(stages_to_capture))
        session.set_metric("workflow_state_path", str(workflow_state_path))
        final_snapshot = next((stage for stage in stage_snapshots if str(stage.get("stage")) == "final_accept"), {})
        session.set_metric(
            "final_positions_by_servo",
            {
                str(servo_id): dict(state or {}).get("position_tick")
                for servo_id, state in dict(final_snapshot.get("servos", {}) or {}).items()
            },
        )
        session.set_metric("final_startup_artifact_path", artifact_path)
        session.set_metric("final_accepted", bool(self.config.final_accept and artifact_path))
        session.set_metric("tracker_available", bool(tracker_available))
        session.set_metric("current_label", "servo-reported current estimate; not tendon force")
        session.set_metric("load_proxy_label", "baseline-subtracted absolute servo-reported current estimate")
        session.set_metric("two_segment_foundation", build_two_segment_foundation_metadata(context))
        session.set_metric("automatic_two_segment_pretension_validated", False)
        session.set_metric("two_segment_control_validated", False)
        session.set_metric("valid_for_model_training", False)
        session.set_metric("valid_for_thesis_repeatability", False)
        session.set_metric("run_trust_mode", self.config.run_trust_mode)

    def write_outputs(self, session: ExperimentSession, paths, summary) -> None:
        write_two_segment_startup_outputs(
            output_dir=paths.output_dir,
            metrics=dict(summary.experiment_metrics or {}),
        )

    def _capture_stage(
        self,
        *,
        session: ExperimentSession,
        stage: str,
        step_index: int,
        baseline_currents: dict[int, int | None],
    ) -> dict[str, Any]:
        context = session.context.settings.robot.operating_context()
        expected_ids = [int(value) for value in context.expected_servo_ids]
        telemetry = session.context.servo_service.read_live_telemetry(expected_ids)
        servos: dict[str, dict[str, Any]] = {}
        missing_positions: list[int] = []
        missing_currents: list[int] = []
        for servo_id in expected_ids:
            item = telemetry.get(int(servo_id))
            if item is None or item.present_position is None:
                missing_positions.append(int(servo_id))
                position = None
            else:
                position = int(item.present_position)
            current = item.present_current_ma if item is not None else None
            if current is None:
                missing_currents.append(int(servo_id))
            baseline_current = baseline_currents.get(int(servo_id))
            load_proxy = (
                abs(float(current) - float(baseline_current))
                if current is not None and baseline_current is not None
                else None
            )
            servos[str(servo_id)] = {
                "servo_id": int(servo_id),
                "position_tick": position,
                "signed_raw_current_ma": int(current) if current is not None else None,
                "load_proxy_ma": float(load_proxy) if load_proxy is not None else None,
                "voltage_mv": int(item.present_voltage_mv) if item is not None and item.present_voltage_mv is not None else None,
                "temperature_c": int(item.present_temperature_c) if item is not None and item.present_temperature_c is not None else None,
                "hardware_error": item.hardware_error if item is not None else "missing telemetry",
                "telemetry_error": item.telemetry_error if item is not None else "missing telemetry",
                "telemetry_stale": not bool(session.context.servo_service.telemetry_is_fresh(item)) if item is not None else True,
                "telemetry_age_s": session.context.servo_service.telemetry_age_s(item) if item is not None else None,
            }
        if missing_positions:
            raise RuntimeError(
                f"Stage {stage} cannot be captured because position ticks are missing for servo IDs {missing_positions}."
            )
        if missing_currents and not self.config.allow_missing_current:
            raise RuntimeError(
                f"Stage {stage} cannot be captured because current estimates are missing for servo IDs {missing_currents}."
            )
        if missing_currents:
            session.add_warning(
                f"Stage {stage} captured without current estimates for servo IDs {missing_currents}."
            )
        role_info = self._role_for_stage(stage, context=context)
        return {
            "startup_schema_version": STARTUP_SCHEMA_VERSION,
            "stage": stage,
            "stage_label": STAGE_LABELS.get(stage, stage),
            "stage_index": int(step_index),
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "operator_note": str(self.config.stage_notes.get(stage, "")),
            "operating_mode": context.operating_mode,
            "expected_servo_ids": list(expected_ids),
            "commanded_servo_ids": [int(value) for value in context.commanded_servo_ids],
            "segment_order": list(context.segment_order),
            "segments": dict(context.metadata().get("segments", {}) or {}),
            "physical_assembly": dict(role_info.get("physical_assembly") or {}),
            "stage_role": role_info.get("stage_role"),
            "targeted_segment_key": role_info.get("targeted_segment_key"),
            "targeted_servo_ids": list(role_info.get("targeted_servo_ids") or []),
            "bottom_segment_key": role_info.get("bottom_segment_key"),
            "top_segment_key": role_info.get("top_segment_key"),
            "bottom_servo_ids": list(role_info.get("bottom_servo_ids") or []),
            "top_servo_ids": list(role_info.get("top_servo_ids") or []),
            "servos": servos,
            "current_label": "servo-reported current estimate; not tendon force",
            "load_proxy_label": "baseline-subtracted absolute servo-reported current estimate",
            "tracker": self._tracker_snapshot(session),
        }

    def _tracker_snapshot(self, session: ExperimentSession) -> dict[str, Any]:
        if not self.config.capture_tracker_snapshot or session.context.tracking_service is None:
            return {"available": False, "reason": "tracker_not_requested_or_missing"}
        try:
            reader = getattr(session.context.tracking_service, "peek_snapshot", None)
            snapshot = reader() if callable(reader) else session.context.tracking_service.get_snapshot()
        except Exception as exc:
            return {"available": False, "reason": str(exc)}
        tools: dict[str, Any] = {}
        for tool_id, tool in dict(getattr(snapshot, "tools", {}) or {}).items():
            tools[str(tool_id)] = {
                "translation_mm": list(getattr(tool, "translation_mm", []) or []),
                "quaternion": list(getattr(tool, "quaternion", []) or []),
                "quality": getattr(tool, "quality", None),
            }
        return {
            "available": True,
            "canonical_state": getattr(snapshot, "canonical_state", ""),
            "registration_state": getattr(snapshot, "registration_state", ""),
            "runtime_tip_mode": getattr(snapshot, "runtime_tip_mode", ""),
            "runtime_tip_trust_level": getattr(snapshot, "runtime_tip_trust_level", ""),
            "tip_pose_status": getattr(snapshot, "tip_pose_status", ""),
            "tools": tools,
            "tool_ids_seen": sorted(tools),
        }

    def _save_final_startup_artifact(
        self,
        *,
        session: ExperimentSession,
        stage_snapshots: list[dict[str, Any]],
    ) -> Path:
        final_stage = dict(stage_snapshots[-1])
        states_by_servo: dict[int, dict[str, Any]] = {}
        for raw_id, raw_state in dict(final_stage.get("servos", {}) or {}).items():
            servo_id = int(raw_id)
            state = dict(raw_state or {})
            states_by_servo[servo_id] = {
                "servo_id": servo_id,
                "measured_position_tick": state.get("position_tick"),
                "measured_current_ma": state.get("signed_raw_current_ma"),
                "manual_two_segment_startup": True,
                "startup_schema_version": STARTUP_SCHEMA_VERSION,
                "startup_type": "manual_two_segment_startup",
                "stage_order": list(STAGE_ORDER),
                "stage_snapshots": stage_snapshots,
                "automatic_two_segment_pretension_validated": False,
                "two_segment_control_validated": False,
            }
        neutral_calibration = session.context.servo_service.neutral_calibration
        neutral_calibration.save_manual_pretension_state(
            states_by_servo=states_by_servo,
            note="manual two-segment startup validation final accept",
            accepted=True,
        )
        artifact = neutral_calibration.load_calibration_artifact()
        context = session.context.settings.robot.operating_context()
        artifact.robot.update(
            {
                "startup_schema_version": STARTUP_SCHEMA_VERSION,
                "startup_type": "manual_two_segment_startup",
                "stage_order": list(STAGE_ORDER),
                "stage_snapshots": stage_snapshots,
                "segment_order": list(context.segment_order),
                "segments": dict(context.metadata().get("segments", {}) or {}),
                "physical_assembly": self._physical_assembly_metadata(context=context),
                "bottom_segment_key": context.bottom_segment_key,
                "top_segment_key": context.top_segment_key,
                "bottom_servo_ids": list(context.bottom_servo_ids or []),
                "top_servo_ids": list(context.top_servo_ids or []),
                "final_positions_by_servo": {
                    str(servo_id): states_by_servo[int(servo_id)]["measured_position_tick"]
                    for servo_id in sorted(states_by_servo)
                },
                "manual_not_automatic": True,
                "tracker_available": any(bool(stage.get("tracker", {}).get("available")) for stage in stage_snapshots),
                "automatic_two_segment_pretension_validated": False,
                "two_segment_control_validated": False,
            }
        )
        neutral_calibration.save_calibration_artifact(artifact)
        return Path(neutral_calibration.path)

    def _physical_assembly_metadata(self, *, context) -> dict[str, Any]:
        return dict(context.metadata().get("physical_assembly") or {})

    def _sync_calibration_context(self, session: ExperimentSession) -> None:
        neutral_calibration = getattr(session.context.servo_service, "neutral_calibration", None)
        context = getattr(neutral_calibration, "context", None)
        if context is None:
            return
        robot = session.context.settings.robot
        resolved = robot.operating_context()
        context.robot_mode = str(resolved.operating_mode)
        context.robot_config_name = str(session.context.settings.runtime.robot_config)
        context.servo_ids = list(robot.servo_ids)
        context.tendon_to_servo = list(robot.tendon_to_servo)
        context.expected_servo_ids = list(resolved.expected_servo_ids)
        context.commanded_servo_ids = list(resolved.commanded_servo_ids)
        context.segments = robot.segment_metadata()
        context.segment_order = robot.segment_order()
        context.mode_profile = resolved.mode_profile
        context.mode_capabilities = dict(resolved.mode_capabilities)
        context.mode_notes = list(resolved.mode_notes)

    def _stages_to_capture(self) -> list[str]:
        stage = normalize_stage_name(self.config.capture_stage)
        if not stage:
            return list(STAGE_ORDER)
        if stage not in STAGE_ORDER:
            raise RuntimeError(
                f"Unknown two-segment startup stage '{self.config.capture_stage}'. "
                f"Expected one of {STAGE_ORDER} or legacy aliases {sorted(STAGE_ALIASES)}."
            )
        return [stage]

    def _workflow_state_path(self, session: ExperimentSession) -> Path:
        configured = str(self.config.workflow_state_path or "").strip()
        if configured:
            path = Path(configured)
            if not path.is_absolute():
                path = session.context.project_root / path
            return path
        return session.context.project_root / "data" / "two_segment_startup_validation" / "current_workflow_state.json"

    def _load_workflow_snapshots(self, path: Path) -> list[dict[str, Any]]:
        if not path.exists():
            return []
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return []
        raw_stages = payload.get("stage_snapshots") if isinstance(payload, dict) else []
        snapshots = [dict(stage) for stage in list(raw_stages or []) if isinstance(stage, dict)]
        return sorted(snapshots, key=lambda stage: STAGE_ORDER.index(str(stage.get("stage"))) if str(stage.get("stage")) in STAGE_ORDER else 999)

    def _save_workflow_snapshots(self, path: Path, snapshots: list[dict[str, Any]]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "startup_schema_version": STARTUP_SCHEMA_VERSION,
            "experiment_name": EXPERIMENT_NAME,
            "stage_order": list(STAGE_ORDER),
            "stage_completion": {str(stage.get("stage")): True for stage in snapshots},
            "stage_snapshots": snapshots,
            "updated_at_utc": datetime.now(timezone.utc).isoformat(),
        }
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    def _baseline_currents_from_snapshots(self, snapshots: list[dict[str, Any]]) -> dict[int, int | None]:
        for snapshot in snapshots:
            if str(snapshot.get("stage")) != "baseline":
                continue
            return {
                int(servo_id): dict(data or {}).get("signed_raw_current_ma")
                for servo_id, data in dict(snapshot.get("servos", {}) or {}).items()
            }
        return {}

    def _require_previous_stages(self, *, stage: str, snapshots: list[dict[str, Any]]) -> None:
        stage_index = STAGE_ORDER.index(stage)
        captured = {str(snapshot.get("stage")) for snapshot in snapshots}
        missing = [candidate for candidate in STAGE_ORDER[:stage_index] if candidate not in captured]
        if missing:
            raise RuntimeError(f"Stage {stage} requires earlier startup stage(s) {missing} to be captured first.")


def _sample_from_stage(*, session: ExperimentSession, stage_snapshot: dict[str, Any], step_index: int) -> ExperimentTimeseriesSample:
    tracker = dict(stage_snapshot.get("tracker", {}) or {})
    pose = TwoSegmentPoseObservation(
        frame="robot",
        tool_poses=dict(tracker.get("tools", {}) or {}),
        tool_roles={"0A": "current_runtime_tip_or_distal_tool"},
        status=str(tracker.get("tip_pose_status") or ("tracker_available" if tracker.get("available") else "tracker_unavailable")),
    ).to_dict()
    commanded = {
        str(servo_id): dict(state).get("position_tick")
        for servo_id, state in dict(stage_snapshot.get("servos", {}) or {}).items()
    }
    return ExperimentTimeseriesSample(
        monotonic_time_s=session.elapsed_s(),
        wall_time_utc=str(stage_snapshot.get("timestamp_utc")),
        phase=str(stage_snapshot.get("stage")),
        step_index=int(step_index),
        sample_index=0,
        commanded_motor_values=commanded,
        commanded_cable_deltas_cm=[0.0] * 8,
        tracker_frame_id=None,
        tool_ids_seen=list(tracker.get("tool_ids_seen", []) or []),
        transform_validity={"tracker": "available" if tracker.get("available") else "unavailable"},
        pose_in_tracker_frame=dict(tracker.get("tools", {}) or {}),
        pose_in_robot_frame={},
        two_segment_pose=pose,
        status_flags=["manual_capture", "no_servo_motion_commanded"],
        backend_health={
            "operating_mode": stage_snapshot.get("operating_mode"),
            "tracker_available": bool(tracker.get("available")),
        },
        extra={"two_segment_startup_stage": stage_snapshot},
    )


def _replace_stage_snapshot(snapshots: list[dict[str, Any]], snapshot: dict[str, Any]) -> list[dict[str, Any]]:
    stage = str(snapshot.get("stage"))
    if stage not in STAGE_ORDER:
        return list(snapshots) + [snapshot]
    target_index = STAGE_ORDER.index(stage)
    retained = [
        dict(existing)
        for existing in snapshots
        if str(existing.get("stage")) in STAGE_ORDER and STAGE_ORDER.index(str(existing.get("stage"))) < target_index
    ]
    retained.append(snapshot)
    return sorted(retained, key=lambda item: STAGE_ORDER.index(str(item.get("stage"))))


def write_two_segment_startup_outputs(*, output_dir: Path, metrics: dict[str, Any]) -> dict[str, Path]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    stages = [dict(stage) for stage in list(metrics.get("stage_snapshots", []) or [])]
    paths = {
        "metrics_csv": output_dir / "metrics.csv",
        "summary_text": output_dir / "two_segment_startup_summary.txt",
        "positions_report": output_dir / "two_segment_startup_positions_report.png",
        "load_proxy_report": output_dir / "two_segment_startup_load_proxy_report.png",
        "stage_drift_report": output_dir / "two_segment_startup_stage_drift_report.png",
        "startup_artifact_metadata": output_dir / "two_segment_startup_artifact_metadata.json",
    }
    _write_metrics_csv(paths["metrics_csv"], stages)
    _write_summary_text(paths["summary_text"], metrics, stages)
    _write_positions_report(paths["positions_report"], stages)
    _write_load_proxy_report(paths["load_proxy_report"], stages)
    _write_stage_drift_report(paths["stage_drift_report"], stages)
    raw_artifact_path = str(metrics.get("final_startup_artifact_path") or "").strip()
    artifact_path = Path(raw_artifact_path) if raw_artifact_path else None
    if artifact_path is not None and artifact_path.exists() and artifact_path.is_file():
        payload = json.loads(artifact_path.read_text(encoding="utf-8"))
        paths["startup_artifact_metadata"].write_text(json.dumps(payload, indent=2), encoding="utf-8")
    else:
        paths["startup_artifact_metadata"].write_text(json.dumps({"artifact_path": raw_artifact_path, "exists": False}, indent=2), encoding="utf-8")
    return paths


def _write_metrics_csv(path: Path, stages: list[dict[str, Any]]) -> None:
    fields = [
        "stage",
        "stage_index",
        "servo_id",
        "segment",
        "position_tick",
        "signed_raw_current_ma",
        "load_proxy_ma",
        "voltage_mv",
        "temperature_c",
        "hardware_error",
        "telemetry_stale",
    ]
    with Path(path).open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for stage in stages:
            segments = dict(stage.get("segments", {}) or {})
            segment_by_servo = {
                int(servo_id): str(key)
                for key, segment in segments.items()
                for servo_id in list(dict(segment or {}).get("servo_ids", []) or [])
            }
            for servo_id, servo in sorted(dict(stage.get("servos", {}) or {}).items(), key=lambda item: int(item[0])):
                data = dict(servo or {})
                writer.writerow(
                    {
                        "stage": stage.get("stage"),
                        "stage_index": stage.get("stage_index"),
                        "servo_id": int(servo_id),
                        "segment": segment_by_servo.get(int(servo_id), ""),
                        "position_tick": data.get("position_tick"),
                        "signed_raw_current_ma": data.get("signed_raw_current_ma"),
                        "load_proxy_ma": data.get("load_proxy_ma"),
                        "voltage_mv": data.get("voltage_mv"),
                        "temperature_c": data.get("temperature_c"),
                        "hardware_error": data.get("hardware_error"),
                        "telemetry_stale": data.get("telemetry_stale"),
                    }
                )


def _write_summary_text(path: Path, metrics: dict[str, Any], stages: list[dict[str, Any]]) -> None:
    lines = [
        "Two-Segment Startup Validation",
        f"startup_schema_version: {metrics.get('startup_schema_version', STARTUP_SCHEMA_VERSION)}",
        f"startup_type: {metrics.get('startup_type', 'manual_two_segment_startup')}",
        f"final_accepted: {metrics.get('final_accepted')}",
        f"final_startup_artifact_path: {metrics.get('final_startup_artifact_path') or 'not written'}",
        f"tracker_available: {metrics.get('tracker_available')}",
        "automatic_two_segment_pretension_validated: false",
        "two_segment_control_validated: false",
        "",
        "Stages:",
    ]
    for stage in stages:
        lines.append(f"- {stage.get('stage')}: {stage.get('timestamp_utc')}")
    lines.append("")
    lines.append("Current is servo-reported current estimate and is not tendon force.")
    lines.append("Load proxy is absolute baseline-subtracted servo-reported current estimate.")
    Path(path).write_text("\n".join(lines).strip() + "\n", encoding="utf-8")


def _write_positions_report(path: Path, stages: list[dict[str, Any]]) -> None:
    if not stages:
        _write_empty_report(path, "Two-Segment Startup Positions", "No startup stages were captured.")
        return
    baseline = _servo_values(stages[0], "position_tick") if stages else {}
    final = _servo_values(stages[-1], "position_tick") if stages else {}
    servo_ids = sorted(set(baseline) | set(final))
    fig, ax = create_figure(size="wide")
    x = list(range(len(servo_ids)))
    width = 0.36
    ax.bar([value - width / 2 for value in x], [baseline.get(sid, 0) for sid in servo_ids], width=width, label="baseline", color=color("baseline"))
    ax.bar([value + width / 2 for value in x], [final.get(sid, 0) for sid in servo_ids], width=width, label="final", color=color("accepted"))
    ax.set_xticks(x)
    ax.set_xticklabels([str(sid) for sid in servo_ids])
    style_axes(ax, title="Two-Segment Startup Positions", xlabel="Servo ID", ylabel="Position tick")
    ax.legend()
    _shade_segment_split(ax)
    save_figure(fig, path)


def _write_load_proxy_report(path: Path, stages: list[dict[str, Any]]) -> None:
    if not stages:
        _write_empty_report(path, "Startup Load Proxy By Stage", "No startup stages were captured.")
        return
    fig, ax = create_figure(size="wide")
    for servo_id in range(1, 9):
        values = [_servo_values(stage, "load_proxy_ma").get(servo_id) for stage in stages]
        ax.plot(range(len(stages)), [0.0 if value is None else float(value) for value in values], marker="o", label=f"S{servo_id}")
    ax.set_xticks(range(len(stages)))
    ax.set_xticklabels([str(stage.get("stage", "")).replace("_", "\n") for stage in stages])
    style_axes(ax, title="Startup Load Proxy By Stage", xlabel="Stage", ylabel="Load proxy (mA)")
    ax.legend(ncol=4, fontsize=7)
    save_figure(fig, path)


def _write_stage_drift_report(path: Path, stages: list[dict[str, Any]]) -> None:
    if not stages:
        _write_empty_report(path, "Position Drift From Baseline", "No startup stages were captured.")
        return
    if len(stages) < 2:
        _write_empty_report(path, "Position Drift From Baseline", "Capture at least two stages to show drift from baseline.")
        return
    baseline = _servo_values(stages[0], "position_tick") if stages else {}
    fig, ax = create_figure(size="wide")
    for stage in stages[1:]:
        values = _servo_values(stage, "position_tick")
        ax.plot(
            sorted(baseline),
            [float(values.get(servo_id, baseline[servo_id])) - float(baseline[servo_id]) for servo_id in sorted(baseline)],
            marker="o",
            label=str(stage.get("stage")),
        )
    style_axes(ax, title="Position Drift From Baseline", xlabel="Servo ID", ylabel="Delta ticks")
    ax.set_xticks(sorted(baseline))
    ax.legend()
    _shade_segment_split(ax)
    save_figure(fig, path)


def _write_empty_report(path: Path, title: str, message: str) -> None:
    fig, ax = create_figure(size="wide")
    ax.text(0.5, 0.5, message, ha="center", va="center", transform=ax.transAxes)
    ax.set_xticks([])
    ax.set_yticks([])
    style_axes(ax, title=title, xlabel="", ylabel="", grid=False)
    save_figure(fig, path)


def _servo_values(stage: dict[str, Any], field: str) -> dict[int, Any]:
    return {
        int(servo_id): dict(data or {}).get(field)
        for servo_id, data in dict(stage.get("servos", {}) or {}).items()
        if dict(data or {}).get(field) is not None
    }


def _shade_segment_split(ax) -> None:
    ax.axvline(3.5, color="#94a3b8", linestyle="--", linewidth=1.0)
    ax.text(1.5, 0.98, "Segment A", transform=ax.get_xaxis_transform(), ha="center", va="top", fontsize=8)
    ax.text(5.5, 0.98, "Segment B", transform=ax.get_xaxis_transform(), ha="center", va="top", fontsize=8)

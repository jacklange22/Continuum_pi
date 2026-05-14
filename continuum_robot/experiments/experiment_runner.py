"""Canonical experiment runner with compatibility helpers."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
import logging
from pathlib import Path
import subprocess
import time
from typing import Any, Callable
from uuid import uuid4

from continuum_robot.experiments.builtins import register_builtin_experiments
from continuum_robot.experiments.dataset_io import (
    ExperimentDatasetLoader,
    ExperimentDatasetWriter,
    canonical_experiment_output_root,
    default_experiment_output_dir_name,
)
from continuum_robot.experiments.experiment_models import ExperimentPoint
from continuum_robot.experiments.framework import ExperimentContext, ExperimentSession
from continuum_robot.experiments.plotting import set_figure_output_quality
from continuum_robot.experiments.registry import ExperimentRegistry
from continuum_robot.experiments.schemas import (
    ExperimentMetadata,
    ExperimentRunResult,
    ExperimentSummary,
)
from continuum_robot.experiments.transform_chain_outputs import write_transform_chain_outputs
from continuum_robot.experiments.validation import STATUS_PARTIAL_SUCCESS, STATUS_SUCCESS, classify_summary_status
from continuum_robot.data.run_management import ensure_run_review
from continuum_robot.servos.segment_readiness import evaluate_selected_segment_readiness
from continuum_robot.servos.sign_mapping_check import ServoMappingCheckRepository
from continuum_robot.tracking.runtime_tip_policy import evaluate_runtime_tip_trust
from continuum_robot.two_segment import build_two_segment_foundation_metadata


LOG = logging.getLogger(__name__)


@dataclass
class ExperimentRunSummary:
    """Compatibility summary returned by the legacy GUI/controller path."""

    output_path: Path
    rows_written: int
    message: str


class ExperimentRunner:
    """Run registered experiments and write canonical datasets."""

    SCHEMA_VERSION = "1.0"

    def __init__(
        self,
        *,
        project_root: Path,
        settings,
        tracking_service,
        servo_service,
        output_dir: Path,
        registration_path: Path,
        experiment_registry: ExperimentRegistry | None = None,
        dataset_writer: ExperimentDatasetWriter | None = None,
        dataset_loader: ExperimentDatasetLoader | None = None,
        default_settle_time_s: float = 0.0,
        sleep_fn=time.sleep,
        monotonic_fn=time.monotonic,
    ) -> None:
        self.project_root = Path(project_root)
        self.settings = settings
        self.tracking_service = tracking_service
        self.servo_service = servo_service
        self.output_dir = Path(output_dir)
        self.registration_path = Path(registration_path)
        self.default_settle_time_s = float(default_settle_time_s)
        self.sleep_fn = sleep_fn
        self.monotonic_fn = monotonic_fn
        self.registry = experiment_registry or ExperimentRegistry()
        if experiment_registry is None:
            register_builtin_experiments(self.registry)
        self.dataset_writer = dataset_writer or ExperimentDatasetWriter(self.output_dir)
        self.dataset_loader = dataset_loader or ExperimentDatasetLoader()
        self.penprobe_live_gui_hz: Callable[..., float | None] | None = None

    def available_experiments(self) -> list:
        """Return registered experiment descriptors."""
        return self.registry.list_descriptors()

    def load_dataset(self, path: Path):
        """Load one canonical experiment dataset bundle."""
        resolved = self._resolve_dataset_path(Path(path))
        return self.dataset_loader.load_dataset(resolved)

    def run_experiment(
        self,
        experiment_name: str,
        *,
        config: dict[str, Any] | None = None,
        operator_notes: str = "",
        output_dir: Path | None = None,
        output_dir_name: str | None = None,
        progress_callback=None,
        stop_requested=None,
        sample_callback=None,
    ) -> ExperimentRunResult:
        """Run one registered experiment and write a canonical dataset."""
        experiment = self.registry.create(experiment_name, config or {})
        LOG.info(
            "Experiment run start | name=%s | output_root=%s",
            experiment_name,
            str(output_dir) if output_dir is not None else str(self.output_dir),
        )
        resolved_output_root = self._resolve_output_root(output_dir)
        run_id = uuid4().hex[:12]
        metadata = self._build_metadata(
            experiment_name=experiment.name,
            run_id=run_id,
            config_used=experiment.config_dict(),
            operator_notes=operator_notes,
        )
        root = canonical_experiment_output_root(resolved_output_root, experiment.name)
        root.mkdir(parents=True, exist_ok=True)
        resolved_folder_name = output_dir_name or default_experiment_output_dir_name(root, experiment.name)
        run_output_dir = root / resolved_folder_name
        run_output_dir.mkdir(parents=True, exist_ok=True)
        session = ExperimentSession(
            context=ExperimentContext(
                project_root=self.project_root,
                settings=self.settings,
                tracking_service=self.tracking_service,
                servo_service=self.servo_service,
                registration_path=self.registration_path,
                output_root=resolved_output_root,
                run_output_dir=run_output_dir,
                monotonic_fn=self.monotonic_fn,
                sleep_fn=self.sleep_fn,
                penprobe_live_gui_hz=self.penprobe_live_gui_hz,
            ),
            metadata=metadata,
            stop_requested=stop_requested,
            progress_callback=progress_callback,
            sample_callback=sample_callback,
        )
        success = False
        stage_name = "setup"
        message = ""
        try:
            experiment.setup(session)
            session.set_stage("setup", "passed")
            stage_name = "precheck"
            experiment.precheck(session)
            session.set_stage("precheck", "passed")
            stage_name = "execute"
            experiment.execute(session)
            session.set_stage("execute", "passed")
            success = True
            message = f"Completed experiment {experiment.name}."
        except Exception as exc:
            LOG.exception(
                "Experiment stage failed | name=%s | run_id=%s | stage=%s | error=%s",
                experiment.name,
                run_id,
                stage_name,
                exc,
            )
            session.add_error(str(exc))
            session.set_stage(stage_name, "failed", str(exc))
            if stage_name == "setup":
                session.set_stage("precheck", "skipped")
                session.set_stage("execute", "skipped")
            elif stage_name == "precheck":
                session.set_stage("execute", "skipped")
            message = f"Experiment {experiment.name} failed: {exc}"
        finally:
            try:
                experiment.finalize(session)
                if session.stage_pass_fail.get("finalize") != "failed":
                    session.set_stage("finalize", "passed")
            except Exception as exc:
                LOG.exception(
                    "Experiment finalize failed | name=%s | run_id=%s | error=%s",
                    experiment.name,
                    run_id,
                    exc,
                )
                session.add_error(f"Finalize failed: {exc}")
                session.set_stage("finalize", "failed", str(exc))
                message = f"{message} Finalize failed: {exc}".strip()
                success = False

        try:
            session.metrics.update(experiment.summarize(session))
        except Exception as exc:
            session.add_error(f"Summary generation failed: {exc}")
            success = False
        summary = self._build_summary(
            experiment_name=experiment.name,
            run_id=run_id,
            session=session,
            success=bool(success and not session.error_messages),
        )
        final_success = bool(summary.success)
        paths = self.dataset_writer.write_dataset(
            metadata,
            session.samples,
            summary,
            output_root=resolved_output_root,
            output_dir_name=resolved_folder_name,
        )
        self._write_default_review(paths.output_dir, metadata=metadata)
        try:
            set_figure_output_quality(getattr(self.settings.runtime, "figure_output_quality", "production"))
            experiment.write_outputs(session, paths, summary)
        except Exception as exc:
            LOG.exception(
                "Experiment write_outputs failed | name=%s | run_id=%s | output_dir=%s | error=%s",
                experiment.name,
                run_id,
                paths.output_dir,
                exc,
            )
            if message:
                message = f"{message} Additional outputs failed: {exc}"
            else:
                message = f"Additional outputs failed: {exc}"
        try:
            tracking_snapshot = None
            if self.tracking_service is not None:
                snapshot_reader = getattr(self.tracking_service, "peek_snapshot", None)
                tracking_snapshot = snapshot_reader() if callable(snapshot_reader) else self.tracking_service.get_snapshot()
            if tracking_snapshot is not None:
                write_transform_chain_outputs(
                    output_dir=paths.output_dir,
                    snapshot=tracking_snapshot,
                    workflow=experiment.name,
                    allow_lower_trust=bool((experiment.config_dict() or {}).get("allow_lower_trust_runtime_tip", False)),
                    provenance_note=f"experiment_run:{experiment.name}:{run_id}",
                )
        except Exception as exc:
            LOG.exception(
                "Transform-chain outputs failed | name=%s | run_id=%s | output_dir=%s | error=%s",
                experiment.name,
                run_id,
                paths.output_dir,
                exc,
            )
            session.add_warning(f"Transform-chain summary outputs failed: {exc}")
        if final_success:
            message = message or f"Completed experiment {experiment.name}."
        elif not message:
            message = f"Experiment {experiment.name} failed."
        LOG.info(
            "Experiment run finish | name=%s | run_id=%s | success=%s | sample_count=%s | output_dir=%s | message=%s",
            experiment.name,
            run_id,
            final_success,
            len(session.samples),
            paths.output_dir,
            message,
        )
        return ExperimentRunResult(
            experiment_name=experiment.name,
            run_id=run_id,
            success=final_success,
            message=message,
            paths=paths,
            metadata=metadata,
            summary=summary,
            sample_count=len(session.samples),
        )

    def _resolve_output_root(self, output_dir: Path | None) -> Path:
        root = Path(output_dir) if output_dir is not None else self.output_dir
        if bool(getattr(self.settings.runtime, "mock_mode", False)):
            try:
                data_experiments = self.project_root / "data" / "experiments"
                relative = root.resolve().relative_to(data_experiments.resolve())
            except ValueError:
                if root.name == "experiments":
                    return root.parent / "mock_experiments"
                return root
            return self.project_root / "data" / "mock_experiments" / relative
        return root

    def _write_default_review(self, run_dir: Path, *, metadata: ExperimentMetadata) -> None:
        mock_mode = bool((metadata.provenance_info or {}).get("mock_mode", False))
        ensure_run_review(
            run_dir,
            status="debug",
            intended_use="mock" if mock_mode else "debug",
            include_in_evidence_index=False,
        )

    def run(
        self,
        points: list[ExperimentPoint],
        progress_callback=None,
        stop_requested=None,
    ) -> ExperimentRunSummary:
        """Compatibility wrapper that routes CSV-loaded points through the canonical dataset experiment."""
        result = self.run_experiment(
            "collect_pose_command_dataset",
            config=self._config_from_points(points),
            progress_callback=progress_callback,
            stop_requested=stop_requested,
        )
        if not result.success:
            raise RuntimeError(result.message)
        return ExperimentRunSummary(
            output_path=result.paths.output_dir,
            rows_written=result.summary.sample_counts.get("total", result.sample_count),
            message=result.message,
        )

    def _resolve_dataset_path(self, path: Path) -> Path:
        candidate = Path(path)
        if candidate.exists():
            return candidate
        legacy_root = self.project_root / "runs"
        try:
            relative = candidate.relative_to(legacy_root)
        except ValueError:
            if not candidate.is_absolute() and candidate.parts and candidate.parts[0] == "runs":
                relative = Path(*candidate.parts[1:])
            else:
                return candidate
        experiments_root = self.project_root / "data" / "experiments"
        if not experiments_root.exists():
            return candidate
        matches = []
        for experiment_root in experiments_root.iterdir():
            if not experiment_root.is_dir():
                continue
            migrated = experiment_root / relative
            if migrated.exists():
                matches.append(migrated)
        return matches[0] if len(matches) == 1 else candidate

    def _build_metadata(
        self,
        *,
        experiment_name: str,
        run_id: str,
        config_used: dict[str, Any],
        operator_notes: str,
    ) -> ExperimentMetadata:
        if self.tracking_service is None:
            tracking_snapshot = None
        else:
            snapshot_reader = getattr(self.tracking_service, "peek_snapshot", None)
            if callable(snapshot_reader):
                tracking_snapshot = snapshot_reader()
            else:
                tracking_snapshot = self.tracking_service.get_snapshot()
        pretension_source_info: dict[str, Any] = {}
        calibration_summary_for_metadata = None
        if self.servo_service is not None:
            try:
                calibration_summary = self.servo_service.get_calibration_summary()
                calibration_summary_for_metadata = calibration_summary
                configured_servo_ids = [int(value) for value in self.settings.robot.expected_servo_ids()]
                if configured_servo_ids:
                    source_summary = calibration_summary.pretension_source_summary(configured_servo_ids)
                    pretension_source_info = {
                        "source_type": source_summary.source_type,
                        "accepted": bool(source_summary.accepted),
                        "usable": bool(source_summary.usable),
                        "message": source_summary.message,
                        "updated_at_utc": source_summary.updated_at_utc,
                        "note": source_summary.note,
                    }
            except Exception as exc:
                pretension_source_info = {"error": str(exc)}
        operating_context = self.settings.robot.operating_context()
        selected_segment_readiness_info: dict[str, Any] = {}
        sign_mapping_info: dict[str, Any] = {}
        if self.servo_service is not None and operating_context.operating_mode == "single_segment":
            try:
                if calibration_summary_for_metadata is None:
                    calibration_summary_for_metadata = self.servo_service.get_calibration_summary()
                expected_ids = [int(value) for value in operating_context.expected_servo_ids]
                ownership = self.servo_service.bus_ownership_status()
                runtime_snapshot = self.servo_service.build_cached_runtime_servo_snapshot(
                    expected_ids,
                    selected_servo_id=(
                        int(operating_context.selected_servo_id)
                        if operating_context.selected_servo_id is not None
                        else None
                    ),
                )
                sign_mapping = _latest_sign_mapping_summary(
                    project_root=self.project_root,
                    active_segment_key=str(operating_context.active_segment_key or ""),
                    expected_servo_ids=expected_ids,
                )
                readiness = evaluate_selected_segment_readiness(
                    operating_mode=operating_context.operating_mode,
                    active_segment_key=operating_context.active_segment_key,
                    active_segment_label=operating_context.active_segment_label,
                    expected_servo_ids=expected_ids,
                    calibration_summary=calibration_summary_for_metadata,
                    mock_mode=bool(self.settings.runtime.mock_mode),
                    servo_connected=bool(getattr(self.servo_service, "is_connected", False)),
                    runtime_snapshot=runtime_snapshot,
                    sign_mapping=sign_mapping,
                )
                selected_segment_readiness_info = readiness.to_dict()
                selected_segment_readiness_info["metadata_readiness_source"] = "cached_telemetry_used"
                selected_segment_readiness_info["cached_telemetry_used"] = True
                selected_segment_readiness_info["live_scan_unavailable_due_to_experiment_ownership"] = bool(
                    ownership.active
                )
                if sign_mapping is not None:
                    sign_mapping_info = {
                        "exists": bool(sign_mapping.exists),
                        "compatible": bool(sign_mapping.compatible),
                        "confirmed": bool(sign_mapping.confirmed),
                        "path": sign_mapping.path,
                        "timestamp_utc": sign_mapping.timestamp_utc,
                        "message": sign_mapping.message,
                    }
            except Exception as exc:
                selected_segment_readiness_info = {"error": str(exc)}
        two_segment_foundation = build_two_segment_foundation_metadata(operating_context)
        backend_info = {
            "mock_mode": bool(self.settings.runtime.mock_mode),
            "hardware_profile": self.settings.runtime.robot_config,
            "tracking_backend_configured": getattr(self.settings.serial, "tracker_backend", ""),
            "tracking_backend_selected": (
                tracking_snapshot.selected_backend_name if tracking_snapshot is not None else ""
            ),
            "tracking_backend_identity": (
                tracking_snapshot.backend_identity if tracking_snapshot is not None else ""
            ),
            "tracking_state": tracking_snapshot.canonical_state if tracking_snapshot is not None else "disabled",
            "servo_connected": bool(getattr(self.servo_service, "is_connected", False)),
            "pretension_source": pretension_source_info,
            "selected_segment_readiness": selected_segment_readiness_info,
            "servo_sign_mapping_check": sign_mapping_info,
            "robot_mode": self.settings.robot.operating_mode(),
            "operating_context": operating_context.metadata(),
            "two_segment_foundation": two_segment_foundation,
            "active_segment": {
                "key": self.settings.robot.active_segment_key(),
                "label": self.settings.robot.active_segment_label(),
                "servo_ids": self.settings.robot.active_segment_servo_ids(),
                "pairs": self.settings.robot.active_segment_pairs(),
            },
            "configured_servo_ids": list(self.settings.robot.servo_ids),
        }
        LOG.info(
            "Experiment metadata context | name=%s | run_id=%s | tracking_backend=%s | runtime_tip_mode=%s | pretension_source=%s",
            experiment_name,
            run_id,
            backend_info.get("tracking_backend_selected", ""),
            (
                tracking_snapshot.runtime_tip_mode
                if tracking_snapshot is not None
                else "latest_accepted"
            ),
            pretension_source_info.get("source_type", "unknown"),
        )
        try:
            runtime_tip_policy = (
                evaluate_runtime_tip_trust(
                    snapshot=tracking_snapshot,
                    workflow=experiment_name,
                    allow_lower_trust=bool(config_used.get("allow_lower_trust_runtime_tip", False)),
                ).to_dict()
                if tracking_snapshot is not None
                else evaluate_runtime_tip_trust(
                    workflow=experiment_name,
                    allow_lower_trust=bool(config_used.get("allow_lower_trust_runtime_tip", False)),
                ).to_dict()
            )
        except ValueError as exc:
            runtime_tip_policy = evaluate_runtime_tip_trust(
                snapshot=tracking_snapshot,
                workflow=None,
                allow_lower_trust=bool(config_used.get("allow_lower_trust_runtime_tip", False)),
            ).to_dict()
            runtime_tip_policy.setdefault("warnings", []).append(str(exc))
        registration_info = {
            "path": str(self.registration_path),
            "exists": self.registration_path.exists(),
            "tracking_registration_state": (
                tracking_snapshot.registration_state if tracking_snapshot is not None else "missing_registration"
            ),
            "tip_pose_status": tracking_snapshot.tip_pose_status if tracking_snapshot is not None else "missing_registration",
            "runtime_tip_mode": (
                tracking_snapshot.runtime_tip_mode if tracking_snapshot is not None else "latest_accepted"
            ),
            "runtime_tip_trust_level": (
                tracking_snapshot.runtime_tip_trust_level if tracking_snapshot is not None else "missing"
            ),
            "runtime_tip_mode_message": (
                tracking_snapshot.runtime_tip_mode_message if tracking_snapshot is not None else ""
            ),
            "runtime_tip_calibration_state": (
                tracking_snapshot.runtime_tip_calibration_state
                if tracking_snapshot is not None
                else "missing_runtime_tip_calibration"
            ),
            "runtime_tip_selected_artifact_kind": (
                tracking_snapshot.runtime_tip_selected_artifact_kind if tracking_snapshot is not None else None
            ),
            "runtime_tip_selected_artifact_path": (
                tracking_snapshot.runtime_tip_selected_artifact_path if tracking_snapshot is not None else None
            ),
            "runtime_tip_policy": runtime_tip_policy,
            "thesis_trusted_runtime_tip": bool(runtime_tip_policy.get("thesis_trusted", False)),
        }
        mock_mode = bool(self.settings.runtime.mock_mode)
        config_trust_mode = str(config_used.get("run_trust_mode", "") or "").strip().lower()
        if not config_trust_mode:
            config_trust_mode = "thesis_trusted"
        if mock_mode:
            config_trust_mode = "mock"
        tracker_state = str(backend_info.get("tracking_state", "") or "")
        tracker_connected = tracker_state not in {"", "disabled", "disconnected", "missing", "none"}
        lower_trust_modes = {"servo_only", "current_only", "lower_trust", "debug_only", "debug", "mock"}
        data_quality_warnings: list[str] = []
        if mock_mode:
            data_quality_warnings.append("mock_mode")
        if config_trust_mode in lower_trust_modes:
            data_quality_warnings.append(f"run_trust_mode={config_trust_mode}")
        if config_used.get("allow_no_tracker_test_run") and not tracker_connected:
            data_quality_warnings.append("no_tracker_test_run")
        if not bool(runtime_tip_policy.get("thesis_trusted", False)):
            data_quality_warnings.append("runtime_tip_not_thesis_trusted")
        experiment_lower = str(experiment_name).strip().lower()
        valid_for_model_training = bool(
            experiment_lower == "collect_pose_command_dataset"
            and not mock_mode
            and config_trust_mode not in lower_trust_modes
            and not bool(config_used.get("allow_no_tracker_test_run") and not tracker_connected)
            and bool(runtime_tip_policy.get("allowed_for_workflow", runtime_tip_policy.get("thesis_trusted", False)))
        )
        valid_for_thesis_repeatability = bool(
            experiment_lower == "single_segment_repeatability"
            and not mock_mode
            and config_trust_mode not in lower_trust_modes
            and bool(runtime_tip_policy.get("thesis_trusted", False))
        )
        if experiment_lower not in {"collect_pose_command_dataset", "single_segment_repeatability"}:
            valid_for_thesis_repeatability = bool(
                not mock_mode and config_trust_mode not in lower_trust_modes and bool(runtime_tip_policy.get("thesis_trusted", False))
            )
        provenance_info = {
            "hardware_profile": self.settings.runtime.robot_config,
            "operating_mode": operating_context.operating_mode,
            "expected_servo_ids": list(operating_context.expected_servo_ids),
            "commanded_servo_ids": list(operating_context.commanded_servo_ids),
            "active_segment": operating_context.metadata().get("active_segment"),
            "segments": dict(operating_context.metadata().get("segments") or {}),
            "segment_order": list(operating_context.metadata().get("segment_order") or []),
            "two_segment_foundation": two_segment_foundation,
            "mirror_mapping": dict(operating_context.metadata().get("mirror_pairs") or {}),
            "tracker_backend": backend_info.get("tracking_backend_selected") or backend_info.get("tracking_backend_configured"),
            "registration_artifact": str(self.registration_path),
            "registration_status": registration_info.get("tracking_registration_state"),
            "runtime_tip_mode": registration_info.get("runtime_tip_mode"),
            "runtime_tip_trust_status": registration_info.get("runtime_tip_trust_level"),
            "startup_pretension_artifact": pretension_source_info,
            "selected_segment_readiness": selected_segment_readiness_info,
            "servo_sign_mapping_check": sign_mapping_info,
            "mock_mode": bool(self.settings.runtime.mock_mode),
        }
        trust_info = {
            "run_trust_mode": config_trust_mode,
            "valid_for_model_training": bool(valid_for_model_training),
            "valid_for_thesis_repeatability": bool(valid_for_thesis_repeatability),
            "valid_for_two_segment_model_training": False if mock_mode else bool(config_used.get("valid_for_two_segment_model_training", False)),
            "include_in_evidence_index": False,
            "data_quality_warnings": sorted(set(data_quality_warnings)),
            "runtime_tip_policy": runtime_tip_policy,
            "success_does_not_imply_thesis_validity": True,
        }
        return ExperimentMetadata(
            schema_version=self.SCHEMA_VERSION,
            experiment_name=experiment_name,
            run_id=run_id,
            timestamp_utc=datetime.now(timezone.utc).isoformat(),
            git_commit=self._git_commit(),
            backend_info=backend_info,
            registration_info=registration_info,
            config_used=config_used,
            operator_notes=operator_notes,
            provenance_info=provenance_info,
            trust_info=trust_info,
        )

    def _build_summary(
        self,
        *,
        experiment_name: str,
        run_id: str,
        session: ExperimentSession,
        success: bool,
    ) -> ExperimentSummary:
        phase_counts = Counter(sample.phase for sample in session.samples)
        sample_counts = {"total": len(session.samples)}
        for phase, count in sorted(phase_counts.items()):
            sample_counts[f"phase::{phase}"] = int(count)
        dropped_frames = 0
        invalid_transforms = 0
        for sample in session.samples:
            if sample.tracker_frame_id is None or "tracker_data_stale" in sample.status_flags:
                dropped_frames += 1
            if any(state == "invalid" for state in sample.transform_validity.values()):
                invalid_transforms += 1
        metrics = dict(session.metrics)
        metrics.setdefault("run_provenance", dict(session.metadata.provenance_info or {}))
        metrics.setdefault("run_trust", dict(session.metadata.trust_info or {}))
        metrics.setdefault("run_trust_mode", (session.metadata.trust_info or {}).get("run_trust_mode", "thesis_trusted"))
        metrics.setdefault(
            "valid_for_model_training",
            bool((session.metadata.trust_info or {}).get("valid_for_model_training", False)),
        )
        metrics.setdefault(
            "valid_for_thesis_repeatability",
            bool((session.metadata.trust_info or {}).get("valid_for_thesis_repeatability", False)),
        )
        metrics.setdefault(
            "valid_for_two_segment_model_training",
            bool((session.metadata.trust_info or {}).get("valid_for_two_segment_model_training", False)),
        )
        metrics.setdefault(
            "mock_mode",
            bool((session.metadata.provenance_info or {}).get("mock_mode", False)),
        )
        metrics.setdefault(
            "include_in_evidence_index",
            bool((session.metadata.trust_info or {}).get("include_in_evidence_index", False)),
        )
        metrics.setdefault(
            "data_quality_warnings",
            list((session.metadata.trust_info or {}).get("data_quality_warnings", [])),
        )
        if (session.metadata.trust_info or {}).get("run_trust_mode") == "mock":
            warnings = set(_as_string_list(metrics.get("data_quality_warnings")))
            warnings.update(_as_string_list((session.metadata.trust_info or {}).get("data_quality_warnings", [])))
            warnings.add("mock_mode")
            metrics["run_trust"] = dict(session.metadata.trust_info or {})
            metrics["run_trust_mode"] = "mock"
            metrics["valid_for_model_training"] = False
            metrics["valid_for_thesis_repeatability"] = False
            metrics["valid_for_two_segment_model_training"] = False
            metrics["mock_mode"] = True
            metrics["include_in_evidence_index"] = False
            metrics["data_quality_warnings"] = sorted(warnings)
        if session.error_messages:
            metrics.setdefault("failure_reason", str(session.error_messages[-1]))
            metrics.setdefault("stop_reason", str(session.error_messages[-1]))
        if session.stage_messages:
            metrics["stage_messages"] = dict(session.stage_messages)
        requirements = dict(metrics.get("summary_requirements", {}))
        status = str(
            requirements.get(
                "force_status",
                classify_summary_status(
                    sample_count=len(session.samples),
                    invalid_transform_count=invalid_transforms,
                    min_sample_count=int(requirements.get("min_sample_count", 1)),
                    require_registration=bool(requirements.get("require_registration", False)),
                    registration_available=bool(requirements.get("registration_available", False)),
                    allow_partial_missing_registration=bool(
                        requirements.get("allow_partial_missing_registration", False)
                    ),
                    require_tip_calibration=bool(requirements.get("require_tip_calibration", False)),
                    tip_calibration_available=bool(requirements.get("tip_calibration_available", False)),
                    allow_partial_missing_tip_cal=bool(requirements.get("allow_partial_missing_tip_cal", False)),
                    invalid_transforms_are_fatal=bool(
                        requirements.get("invalid_transforms_are_fatal", True)
                    ),
                ),
            )
        )
        if not success and status in {STATUS_SUCCESS, STATUS_PARTIAL_SUCCESS}:
            status = "failed"
        summary_success = bool(success and status in {STATUS_SUCCESS, STATUS_PARTIAL_SUCCESS})
        return ExperimentSummary(
            schema_version=self.SCHEMA_VERSION,
            experiment_name=experiment_name,
            run_id=run_id,
            status=status,
            success=summary_success,
            sample_counts=sample_counts,
            dropped_frames=dropped_frames,
            invalid_transforms=invalid_transforms,
            stage_pass_fail=dict(session.stage_pass_fail),
            experiment_metrics=metrics,
            warning_messages=list(session.warning_messages),
            error_messages=list(session.error_messages),
        )

    def _config_from_points(self, points: list[ExperimentPoint]) -> dict[str, Any]:
        if not points:
            raise RuntimeError("No experiment points are loaded.")
        dry_run = bool(self.settings.runtime.mock_mode or not self.servo_service.is_connected)
        return {
            "dry_run": dry_run,
            # Legacy CSV/schedule replay is retained as a lower-trust compatibility path that routes
            # through the canonical Motor Babble dataset collector without demanding thesis-grade state.
            "require_robot_frame_tip": False,
            "allow_lower_trust_runtime_tip": True,
            "allow_lower_trust_pretension": True,
            "sample_count_per_point": 1,
            "settle_time_s": self.default_settle_time_s,
            "command_points": [
                {
                    "index": int(point.index),
                    "tendon_displacement_cm": [float(value) for value in point.tendon_displacement_cm],
                    "settle_time_s": point.settle_time_s,
                    "repeat": int(point.repeat),
                    "label": point.label,
                }
                for point in self._expand_points(points)
            ],
            "command_schedule": {
                "kind": "trajectory",
                "dimensions": len(points[0].tendon_displacement_cm),
            },
        }

    @staticmethod
    def _expand_points(points: list[ExperimentPoint]) -> list[ExperimentPoint]:
        expanded: list[ExperimentPoint] = []
        running_index = 0
        for point in points:
            for repeat_index in range(max(1, int(point.repeat))):
                expanded.append(
                    ExperimentPoint(
                        index=running_index,
                        tendon_displacement_cm=[float(value) for value in point.tendon_displacement_cm],
                        settle_time_s=point.settle_time_s,
                        repeat=1,
                        label=point.label or f"csv_point_{point.index:04d}_r{repeat_index}",
                    )
                )
                running_index += 1
        return expanded

    def _git_commit(self) -> str | None:
        try:
            completed = subprocess.run(
                ["git", "-C", str(self.project_root), "rev-parse", "HEAD"],
                check=True,
                capture_output=True,
                text=True,
            )
        except Exception:
            return None
        commit = completed.stdout.strip()
        return commit or None


def _as_string_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, (list, tuple, set)):
        return [str(item) for item in value if str(item)]
    if isinstance(value, str) and value.strip():
        return [value.strip()]
    return []


def _latest_sign_mapping_summary(
    *,
    project_root: Path,
    active_segment_key: str,
    expected_servo_ids: list[int],
):
    try:
        repo = ServoMappingCheckRepository(project_root / "data" / "calibration" / "servo_mapping_checks")
        return repo.latest_for_segment(
            active_segment_key=str(active_segment_key or ""),
            expected_servo_ids=[int(value) for value in expected_servo_ids],
        )
    except Exception:
        return None

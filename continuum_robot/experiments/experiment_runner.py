"""Canonical experiment runner with compatibility helpers."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
import subprocess
import time
from typing import Any
from uuid import uuid4

from continuum_robot.experiments.builtins import register_builtin_experiments
from continuum_robot.experiments.dataset_io import ExperimentDatasetLoader, ExperimentDatasetWriter
from continuum_robot.experiments.experiment_models import ExperimentPoint
from continuum_robot.experiments.framework import ExperimentContext, ExperimentSession
from continuum_robot.experiments.registry import ExperimentRegistry
from continuum_robot.experiments.schemas import (
    ExperimentMetadata,
    ExperimentRunResult,
    ExperimentSummary,
)
from continuum_robot.experiments.validation import STATUS_PARTIAL_SUCCESS, STATUS_SUCCESS, classify_summary_status


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

    def available_experiments(self) -> list:
        """Return registered experiment descriptors."""
        return self.registry.list_descriptors()

    def load_dataset(self, path: Path):
        """Load one canonical experiment dataset bundle."""
        return self.dataset_loader.load_dataset(Path(path))

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
        run_id = uuid4().hex[:12]
        metadata = self._build_metadata(
            experiment_name=experiment.name,
            run_id=run_id,
            config_used=experiment.config_dict(),
            operator_notes=operator_notes,
        )
        session = ExperimentSession(
            context=ExperimentContext(
                project_root=self.project_root,
                settings=self.settings,
                tracking_service=self.tracking_service,
                servo_service=self.servo_service,
                registration_path=self.registration_path,
                output_root=Path(output_dir) if output_dir is not None else self.output_dir,
                monotonic_fn=self.monotonic_fn,
                sleep_fn=self.sleep_fn,
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
            output_root=Path(output_dir) if output_dir is not None else None,
            output_dir_name=output_dir_name,
        )
        if final_success:
            message = message or f"Completed experiment {experiment.name}."
        elif not message:
            message = f"Experiment {experiment.name} failed."
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

    def _build_metadata(
        self,
        *,
        experiment_name: str,
        run_id: str,
        config_used: dict[str, Any],
        operator_notes: str,
    ) -> ExperimentMetadata:
        tracking_snapshot = self.tracking_service.get_snapshot() if self.tracking_service is not None else None
        backend_info = {
            "mock_mode": bool(self.settings.runtime.mock_mode),
            "tracking_backend_configured": getattr(self.settings.serial, "tracker_backend", ""),
            "tracking_backend_selected": (
                tracking_snapshot.selected_backend_name if tracking_snapshot is not None else ""
            ),
            "tracking_backend_identity": (
                tracking_snapshot.backend_identity if tracking_snapshot is not None else ""
            ),
            "tracking_state": tracking_snapshot.canonical_state if tracking_snapshot is not None else "disabled",
            "servo_connected": bool(getattr(self.servo_service, "is_connected", False)),
        }
        registration_info = {
            "path": str(self.registration_path),
            "exists": self.registration_path.exists(),
            "tracking_registration_state": (
                tracking_snapshot.registration_state if tracking_snapshot is not None else "missing_registration"
            ),
            "tip_pose_status": tracking_snapshot.tip_pose_status if tracking_snapshot is not None else "missing_registration",
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
        return {
            "dry_run": bool(self.settings.runtime.mock_mode or not self.servo_service.is_connected),
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

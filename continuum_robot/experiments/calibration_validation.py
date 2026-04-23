"""Offline repeated-run validation experiments for calibration artifacts."""

from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np

from continuum_robot.experiments.dataset_io import ExperimentDatasetLoader
from continuum_robot.experiments.framework import BaseExperiment, ExperimentHardwareRequirements, ExperimentSession
from continuum_robot.registration.legacy_compat import average_quaternions
from continuum_robot.registration.repository import RegistrationRepository
from continuum_robot.registration.validation import compute_transform_delta
from continuum_robot.tracking.transforms import rotmat_to_quat_wxyz, quat_wxyz_to_rotmat


@dataclass
class RegistrationValidationConfig:
    """Config for offline repeated-registration analysis."""

    run_paths: list[str] = field(default_factory=list)

    @classmethod
    def from_dict(cls, payload: dict[str, Any] | None) -> "RegistrationValidationConfig":
        payload = dict(payload or {})
        return cls(run_paths=[str(value) for value in (payload.get("run_paths") or []) if str(value).strip()])


@dataclass
class PivotValidationConfig:
    """Config for offline repeated-pivot analysis."""

    run_paths: list[str] = field(default_factory=list)

    @classmethod
    def from_dict(cls, payload: dict[str, Any] | None) -> "PivotValidationConfig":
        payload = dict(payload or {})
        return cls(run_paths=[str(value) for value in (payload.get("run_paths") or []) if str(value).strip()])


@dataclass
class ValidationRunCandidate:
    """One selectable source artifact for an offline validation page."""

    path: str
    timestamp_utc: str
    label: str
    status: str = ""
    metric_primary: str = ""
    metric_secondary: str = ""
    experiment_name: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


class RegistrationValidationExperiment(BaseExperiment):
    """Analyze repeated registration solves as one thesis-oriented validation run."""

    name = "registration_validation"
    description = "Analyze repeated saved registration solves for transform spread, FRE distribution, and downstream frame stability."
    hardware_requirements = ExperimentHardwareRequirements(mock_compatible=True)

    def __init__(self, config: RegistrationValidationConfig) -> None:
        super().__init__(config)

    @classmethod
    def from_dict(cls, payload: dict[str, Any] | None = None) -> "RegistrationValidationExperiment":
        return cls(config=RegistrationValidationConfig.from_dict(payload))

    def precheck(self, session: ExperimentSession) -> None:
        if len(self.config.run_paths) < 2:
            raise RuntimeError("Select at least 2 saved registration runs for validation.")
        missing = [raw for raw in self.config.run_paths if not _resolve_source_path(session, raw).exists()]
        if missing:
            raise RuntimeError(f"Selected registration runs are missing: {', '.join(missing[:3])}")

    def execute(self, session: ExperimentSession) -> None:
        metrics, warnings = analyze_registration_runs(
            self.config.run_paths,
            project_root=session.context.project_root,
        )
        if int(metrics.get("valid_run_count", 0) or 0) < 2:
            raise RuntimeError("Registration validation requires at least 2 valid saved runs.")
        metrics["summary_requirements"] = {"force_status": "success"}
        session.metrics.update(metrics)
        for message in warnings:
            session.add_warning(message)

    def write_outputs(self, session: ExperimentSession, paths, summary) -> None:
        from continuum_robot.experiments.calibration_validation_outputs import write_registration_validation_outputs

        write_registration_validation_outputs(
            output_dir=paths.output_dir,
            metadata=session.metadata,
            summary=summary,
        )


class PivotValidationExperiment(BaseExperiment):
    """Analyze repeated pivot-calibration runs as one thesis-oriented validation run."""

    name = "pivot_validation"
    description = "Analyze repeated saved pivot calibrations for tip-offset spread, per-axis variation, and solve-quality stability."
    hardware_requirements = ExperimentHardwareRequirements(mock_compatible=True)

    def __init__(self, config: PivotValidationConfig) -> None:
        super().__init__(config)

    @classmethod
    def from_dict(cls, payload: dict[str, Any] | None = None) -> "PivotValidationExperiment":
        return cls(config=PivotValidationConfig.from_dict(payload))

    def precheck(self, session: ExperimentSession) -> None:
        if len(self.config.run_paths) < 2:
            raise RuntimeError("Select at least 2 saved pivot calibration runs for validation.")
        missing = [raw for raw in self.config.run_paths if not _resolve_source_path(session, raw).exists()]
        if missing:
            raise RuntimeError(f"Selected pivot runs are missing: {', '.join(missing[:3])}")

    def execute(self, session: ExperimentSession) -> None:
        metrics, warnings = analyze_pivot_runs(
            self.config.run_paths,
            project_root=session.context.project_root,
        )
        if int(metrics.get("valid_run_count", 0) or 0) < 2:
            raise RuntimeError("Pivot validation requires at least 2 valid saved runs.")
        metrics["summary_requirements"] = {"force_status": "success"}
        session.metrics.update(metrics)
        for message in warnings:
            session.add_warning(message)

    def write_outputs(self, session: ExperimentSession, paths, summary) -> None:
        from continuum_robot.experiments.calibration_validation_outputs import write_pivot_validation_outputs

        write_pivot_validation_outputs(
            output_dir=paths.output_dir,
            metadata=session.metadata,
            summary=summary,
        )


def register_calibration_validation_experiments(registry) -> None:
    """Register offline calibration validation experiments."""
    registry.register(
        name=RegistrationValidationExperiment.name,
        title="Registration Validation",
        description=RegistrationValidationExperiment.description,
        category="analysis",
        tags=["Registration", "Validation", "Repeatability"],
        default_config_path="config/experiment_registration_validation.example.yaml",
        factory=RegistrationValidationExperiment.from_dict,
    )
    registry.register(
        name=PivotValidationExperiment.name,
        title="Pivot Validation",
        description=PivotValidationExperiment.description,
        category="analysis",
        tags=["Pivot", "Validation", "Repeatability"],
        default_config_path="config/experiment_pivot_validation.example.yaml",
        factory=PivotValidationExperiment.from_dict,
    )


def list_registration_validation_candidates(project_root: Path, *, limit: int | None = None) -> list[ValidationRunCandidate]:
    """List saved registration records for the validation page."""
    repository = RegistrationRepository(root_dir=Path(project_root) / "data" / "registrations")
    candidates: list[ValidationRunCandidate] = []
    for path in repository.list_saved_records(limit=limit):
        try:
            payload = repository.load_payload(path)
        except Exception:
            continue
        validation_metrics = dict(payload.get("validation_metrics", {}) or {})
        fre_mm = validation_metrics.get("overall_fre_mm")
        if fre_mm is None:
            fre_mm = payload.get("fre_mm")
        landmark_count = len(payload.get("landmark_labels") or [])
        candidates.append(
            ValidationRunCandidate(
                path=_display_path(path, project_root=project_root),
                timestamp_utc=str(payload.get("timestamp_utc", "")),
                label=path.name,
                status="saved",
                metric_primary=(f"FRE {float(fre_mm):.3f} mm" if fre_mm is not None else "FRE n/a"),
                metric_secondary=f"{landmark_count} landmarks",
                metadata={
                    "measurement_tool_id": payload.get("measurement_tool_id"),
                    "landmark_count": landmark_count,
                    "fre_mm": float(fre_mm) if fre_mm is not None else None,
                },
            )
        )
    return candidates


def list_pivot_validation_candidates(project_root: Path, *, limit: int | None = None) -> list[ValidationRunCandidate]:
    """List saved canonical pivot-calibration runs for the validation page."""
    loader = ExperimentDatasetLoader()
    root = Path(project_root) / "data" / "pivot_calibration"
    if not root.exists():
        return []
    run_dirs = sorted(
        (
            path
            for path in root.iterdir()
            if path.is_dir() and (path / "metadata.json").exists() and (path / "summary.json").exists()
        ),
        key=lambda item: item.name,
        reverse=True,
    )
    if limit is not None:
        run_dirs = run_dirs[: max(0, int(limit))]
    candidates: list[ValidationRunCandidate] = []
    for run_dir in run_dirs:
        try:
            bundle = loader.load_dataset(run_dir)
        except Exception:
            continue
        if bundle.metadata.experiment_name != "pivot_calibration":
            continue
        metrics = bundle.summary.experiment_metrics if isinstance(bundle.summary.experiment_metrics, dict) else {}
        candidates.append(
            ValidationRunCandidate(
                path=_display_path(run_dir, project_root=project_root),
                timestamp_utc=str(bundle.metadata.timestamp_utc),
                label=run_dir.name,
                status=str(bundle.summary.status),
                metric_primary=(
                    f"RMSE {float(metrics.get('rmse_mm')):.3f} mm"
                    if metrics.get("rmse_mm") is not None
                    else "RMSE n/a"
                ),
                metric_secondary=(
                    f"{int(metrics.get('sample_count_used', 0) or 0)} used / "
                    f"{int(metrics.get('sample_count_rejected', 0) or 0)} rejected"
                ),
                experiment_name=bundle.metadata.experiment_name,
                metadata={
                    "tool_id": metrics.get("pivot_input_tool_id", bundle.metadata.config_used.get("tool_id")),
                    "rmse_mm": float(metrics.get("rmse_mm")) if metrics.get("rmse_mm") is not None else None,
                    "sample_count_used": int(metrics.get("sample_count_used", 0) or 0),
                    "sample_count_rejected": int(metrics.get("sample_count_rejected", 0) or 0),
                },
            )
        )
    return candidates


def analyze_registration_runs(run_paths: Iterable[str], *, project_root: Path) -> tuple[dict[str, Any], list[str]]:
    """Compute repeated-run registration spread metrics from saved records."""
    warnings: list[str] = []
    valid_rows: list[dict[str, Any]] = []
    invalid_rows: list[dict[str, Any]] = []
    for raw_path in run_paths:
        try:
            row = _load_registration_row(_resolve_path(project_root, raw_path))
        except Exception as exc:
            invalid_rows.append({"path": str(raw_path), "reason": str(exc)})
            continue
        valid_rows.append(row)
    if len(valid_rows) < 2:
        return _base_validation_metrics("registration_validation", run_paths, valid_rows, invalid_rows), _warning_messages(invalid_rows)

    consensus_transform = _consensus_transform([row["T_robot_aurora_array"] for row in valid_rows])
    consensus_origin = np.linalg.inv(consensus_transform)[0:3, 3]
    fre_values: list[float] = []
    translation_deltas: list[float] = []
    rotation_deltas: list[float] = []
    origin_distances: list[float] = []
    landmark_counts: list[float] = []
    pairwise_translation: list[float] = []
    pairwise_rotation: list[float] = []

    for index, row in enumerate(valid_rows):
        delta = compute_transform_delta(consensus_transform, row["T_robot_aurora_array"])
        translation_delta = float(delta["translation_delta_mm"])
        rotation_delta = float(delta["rotation_delta_deg"])
        origin_distance = float(np.linalg.norm(row["robot_origin_in_aurora_mm_array"] - consensus_origin))
        row["translation_delta_to_consensus_mm"] = translation_delta
        row["rotation_delta_to_consensus_deg"] = rotation_delta
        row["origin_distance_to_consensus_mm"] = origin_distance
        row["run_index"] = index
        fre_values.append(float(row["fre_mm"]))
        translation_deltas.append(translation_delta)
        rotation_deltas.append(rotation_delta)
        origin_distances.append(origin_distance)
        landmark_counts.append(float(row["landmark_count"]))

    for left in range(len(valid_rows)):
        for right in range(left + 1, len(valid_rows)):
            delta = compute_transform_delta(
                valid_rows[left]["T_robot_aurora_array"],
                valid_rows[right]["T_robot_aurora_array"],
            )
            pairwise_translation.append(float(delta["translation_delta_mm"]))
            pairwise_rotation.append(float(delta["rotation_delta_deg"]))

    origin_points = np.vstack([row["robot_origin_in_aurora_mm_array"] for row in valid_rows])
    metrics = _base_validation_metrics("registration_validation", run_paths, valid_rows, invalid_rows)
    metrics.update(
        {
            "consensus_transform_T_robot_aurora": consensus_transform.tolist(),
            "consensus_robot_origin_in_aurora_mm": _vector3(consensus_origin),
            "per_run_rows": [
                {
                    key: value
                    for key, value in row.items()
                    if not key.endswith("_array")
                }
                for row in valid_rows
            ],
            "fre_summary_mm": _summarize_scalars(fre_values),
            "translation_delta_to_consensus_summary_mm": _summarize_scalars(translation_deltas),
            "rotation_delta_to_consensus_summary_deg": _summarize_scalars(rotation_deltas),
            "origin_distance_to_consensus_summary_mm": _summarize_scalars(origin_distances),
            "pairwise_translation_delta_summary_mm": _summarize_scalars(pairwise_translation),
            "pairwise_rotation_delta_summary_deg": _summarize_scalars(pairwise_rotation),
            "landmark_count_summary": _summarize_scalars(landmark_counts),
            "robot_origin_spread_mm": {
                "std_x_mm": float(origin_points[:, 0].std(ddof=0)),
                "std_y_mm": float(origin_points[:, 1].std(ddof=0)),
                "std_z_mm": float(origin_points[:, 2].std(ddof=0)),
                "mean_distance_mm": float(np.mean(origin_distances)),
                "rms_distance_mm": float(np.sqrt(np.mean(np.square(origin_distances)))),
                "max_distance_mm": float(np.max(origin_distances)),
                "span_x_mm": float(origin_points[:, 0].max() - origin_points[:, 0].min()),
                "span_y_mm": float(origin_points[:, 1].max() - origin_points[:, 1].min()),
                "span_z_mm": float(origin_points[:, 2].max() - origin_points[:, 2].min()),
            },
        }
    )
    return metrics, _warning_messages(invalid_rows)


def analyze_pivot_runs(run_paths: Iterable[str], *, project_root: Path) -> tuple[dict[str, Any], list[str]]:
    """Compute repeated-run pivot spread metrics from saved canonical runs."""
    warnings: list[str] = []
    valid_rows: list[dict[str, Any]] = []
    invalid_rows: list[dict[str, Any]] = []
    loader = ExperimentDatasetLoader()
    for raw_path in run_paths:
        try:
            row = _load_pivot_row(loader, _resolve_path(project_root, raw_path))
        except Exception as exc:
            invalid_rows.append({"path": str(raw_path), "reason": str(exc)})
            continue
        valid_rows.append(row)
    if len(valid_rows) < 2:
        return _base_validation_metrics("pivot_validation", run_paths, valid_rows, invalid_rows), _warning_messages(invalid_rows)

    tip_vectors = np.vstack([row["tip_vector_local_mm_array"] for row in valid_rows])
    consensus_tip = tip_vectors.mean(axis=0)
    norms: list[float] = []
    rmse_values: list[float] = []
    distances: list[float] = []
    sample_used: list[float] = []
    sample_rejected: list[float] = []
    for index, row in enumerate(valid_rows):
        tip_vector = row["tip_vector_local_mm_array"]
        tip_norm = float(np.linalg.norm(tip_vector))
        distance = float(np.linalg.norm(tip_vector - consensus_tip))
        row["tip_norm_mm"] = tip_norm
        row["distance_to_consensus_mm"] = distance
        row["run_index"] = index
        norms.append(tip_norm)
        distances.append(distance)
        if row["rmse_mm"] is not None:
            rmse_values.append(float(row["rmse_mm"]))
        sample_used.append(float(row["sample_count_used"]))
        sample_rejected.append(float(row["sample_count_rejected"]))

    metrics = _base_validation_metrics("pivot_validation", run_paths, valid_rows, invalid_rows)
    metrics.update(
        {
            "consensus_tip_vector_local_mm": _vector3(consensus_tip),
            "per_run_rows": [
                {
                    key: value
                    for key, value in row.items()
                    if not key.endswith("_array")
                }
                for row in valid_rows
            ],
            "tip_norm_summary_mm": _summarize_scalars(norms),
            "rmse_summary_mm": _summarize_scalars(rmse_values),
            "distance_to_consensus_summary_mm": _summarize_scalars(distances),
            "sample_count_used_summary": _summarize_scalars(sample_used),
            "sample_count_rejected_summary": _summarize_scalars(sample_rejected),
            "per_axis_summary_mm": {
                axis: _summarize_scalars([float(row["tip_vector_local_mm"][axis_index]) for row in valid_rows])
                for axis_index, axis in enumerate(("x", "y", "z"))
            },
        }
    )
    return metrics, _warning_messages(invalid_rows)


def _load_registration_row(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    transform = np.asarray(payload.get("T_robot_aurora") or payload.get("T_aurora_2_model"), dtype=float)
    if transform.shape != (4, 4):
        raise ValueError("Registration record is missing a valid T_robot_aurora transform.")
    validation_metrics = dict(payload.get("validation_metrics", {}) or {})
    fre_mm = validation_metrics.get("overall_fre_mm")
    if fre_mm is None:
        fre_mm = payload.get("fre_mm")
    if fre_mm is None:
        raise ValueError("Registration record is missing FRE.")
    origin = np.linalg.inv(transform)[0:3, 3]
    landmark_labels = [str(value) for value in (payload.get("landmark_labels") or [])]
    return {
        "path": str(path),
        "label": path.name,
        "timestamp_utc": str(payload.get("timestamp_utc", "")),
        "fre_mm": float(fre_mm),
        "landmark_count": len(landmark_labels),
        "landmark_labels": landmark_labels,
        "measurement_tool_id": str(payload.get("measurement_tool_id") or "unknown"),
        "coil_tool_id": str(payload.get("coil_tool_id") or "unknown"),
        "robot_origin_in_aurora_mm": _vector3(origin),
        "T_robot_aurora": transform.tolist(),
        "T_robot_aurora_array": transform,
        "robot_origin_in_aurora_mm_array": origin,
    }


def _load_pivot_row(loader: ExperimentDatasetLoader, path: Path) -> dict[str, Any]:
    bundle = loader.load_dataset(path)
    if bundle.metadata.experiment_name != "pivot_calibration":
        raise ValueError(f"Expected pivot_calibration run, found {bundle.metadata.experiment_name!r}.")
    metrics = bundle.summary.experiment_metrics if isinstance(bundle.summary.experiment_metrics, dict) else {}
    tip_vector = np.asarray(metrics.get("tip_vector_local_mm") or [], dtype=float)
    if tip_vector.shape != (3,):
        raise ValueError("Pivot run is missing tip_vector_local_mm.")
    rmse = metrics.get("rmse_mm")
    return {
        "path": str(bundle.paths.output_dir),
        "label": bundle.paths.output_dir.name,
        "timestamp_utc": str(bundle.metadata.timestamp_utc),
        "status": str(bundle.summary.status),
        "tool_id": str(metrics.get("pivot_input_tool_id") or bundle.metadata.config_used.get("tool_id") or "0B"),
        "tip_vector_local_mm": _vector3(tip_vector),
        "tip_vector_local_mm_array": tip_vector,
        "rmse_mm": float(rmse) if rmse is not None else None,
        "sample_count_total": int(metrics.get("sample_count_total", 0) or 0),
        "sample_count_used": int(metrics.get("sample_count_used", 0) or 0),
        "sample_count_rejected": int(metrics.get("sample_count_rejected", 0) or 0),
        "input_format": str(metrics.get("pivot_input_format") or "unknown"),
    }


def _consensus_transform(transforms: list[np.ndarray]) -> np.ndarray:
    translations = np.asarray([transform[0:3, 3] for transform in transforms], dtype=float)
    quaternions = np.asarray([rotmat_to_quat_wxyz(transform[0:3, 0:3]) for transform in transforms], dtype=float)
    mean_quaternion, _ = average_quaternions(quaternions, method="sign_aligned_mean")
    consensus = np.eye(4, dtype=float)
    consensus[0:3, 0:3] = quat_wxyz_to_rotmat(tuple(float(value) for value in mean_quaternion))
    consensus[0:3, 3] = translations.mean(axis=0)
    return consensus


def _base_validation_metrics(
    experiment_name: str,
    run_paths: Iterable[str],
    valid_rows: list[dict[str, Any]],
    invalid_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    selected_paths = [str(value) for value in run_paths]
    return {
        "experiment_name": experiment_name,
        "selected_run_count": len(selected_paths),
        "valid_run_count": len(valid_rows),
        "invalid_run_count": len(invalid_rows),
        "selected_run_paths": selected_paths,
        "valid_run_paths": [str(row["path"]) for row in valid_rows],
        "invalid_runs": invalid_rows,
    }


def _warning_messages(invalid_rows: list[dict[str, Any]]) -> list[str]:
    return [
        f"Skipped {Path(str(row.get('path', 'unknown'))).name}: {row.get('reason', 'invalid source artifact')}"
        for row in invalid_rows
    ]


def _summarize_scalars(values: Iterable[float]) -> dict[str, Any]:
    array = np.asarray([float(value) for value in values], dtype=float)
    if array.size == 0:
        return {"count": 0, "mean": None, "median": None, "std": None, "min": None, "max": None}
    return {
        "count": int(array.size),
        "mean": float(array.mean()),
        "median": float(np.median(array)),
        "std": float(array.std(ddof=0)),
        "min": float(array.min()),
        "max": float(array.max()),
    }


def _vector3(values: np.ndarray | Iterable[float]) -> list[float]:
    array = np.asarray(list(values) if not isinstance(values, np.ndarray) else values, dtype=float).reshape(3)
    return [float(value) for value in array.tolist()]


def _resolve_path(project_root: Path, raw_path: str | Path) -> Path:
    candidate = Path(raw_path)
    if candidate.is_absolute():
        return candidate
    return Path(project_root) / candidate


def _display_path(path: Path, *, project_root: Path) -> str:
    try:
        return str(path.relative_to(project_root))
    except ValueError:
        return str(path)


def _resolve_source_path(session: ExperimentSession, raw_path: str | Path) -> Path:
    return _resolve_path(session.context.project_root, raw_path)

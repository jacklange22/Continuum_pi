"""Run review, validation, archive, and thesis-evidence helpers."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
from typing import Any

from continuum_robot.data.validate_run_bundle import validate_run_folder
from continuum_robot.experiments.dataset_io import sanitize_output_name


REVIEW_FILENAME = "run_review.json"
REVIEW_STATUSES = {"keep", "thesis_candidate", "advisor_share", "debug", "garbage", "archived"}
PROTECTED_REVIEW_STATUSES = {"keep", "thesis_candidate", "advisor_share"}


@dataclass(frozen=True)
class RunReview:
    """Local operator review state stored as a sidecar in the run folder."""

    review_status: str = "debug"
    notes: str = ""
    created_at: str = ""
    reviewed_at: str = ""
    reviewed_by: str = ""
    intended_use: str = ""
    include_in_evidence_index: bool = False

    def to_dict(self) -> dict[str, Any]:
        payload = {
            "schema_version": "1.0",
            "review_status": self.review_status,
            "notes": self.notes,
            "created_at": self.created_at,
            "reviewed_at": self.reviewed_at,
            "reviewer": self.reviewed_by,
            "include_in_evidence_index": self.include_in_evidence_index,
            "intended_use": self.intended_use,
        }
        if self.reviewed_by:
            payload["reviewed_by"] = self.reviewed_by
        return payload


@dataclass(frozen=True)
class ManagedRunSummary:
    """Small, display-oriented summary of one experiment run folder."""

    run_dir: Path
    experiment_name: str
    run_id: str
    timestamp_label: str
    validation_status: str
    validation_issues: list[str] = field(default_factory=list)
    success: Any = None
    status: str = "unknown"
    stop_or_failure_reason: str = ""
    run_trust_mode: str = "unknown"
    valid_for_model_training: Any = None
    valid_for_two_segment_model_training: Any = None
    valid_for_thesis_repeatability: Any = None
    data_quality_warnings: list[str] = field(default_factory=list)
    operating_mode: str = "unknown"
    active_segment: str = ""
    two_segment_summary: str = ""
    two_segment_pose_summary: str = ""
    two_segment_modeling_summary: str = ""
    hardware_profile: str = "unknown"
    runtime_tip_mode: str = ""
    runtime_tip_trust_status: str = ""
    registration_summary: str = ""
    pretension_startup_summary: str = ""
    report_figures: list[str] = field(default_factory=list)
    metrics_files: list[str] = field(default_factory=list)
    samples_present: bool = False
    samples_size_bytes: int = 0
    total_size_bytes: int = 0
    mock_mode: bool | None = None
    root_location: str = ""
    has_run_review: bool = False
    review: RunReview = field(default_factory=RunReview)


@dataclass(frozen=True)
class MoveRunResult:
    """Result of moving a run into archive or trash."""

    source_path: Path
    destination_path: Path
    action: str


def discover_experiment_run_dirs(project_root: Path, *, experiment_name: str | None = None) -> list[Path]:
    """Return canonical experiment run directories, newest first."""
    return discover_run_dirs(project_root, root_name="experiments", experiment_name=experiment_name)


def discover_run_dirs(project_root: Path, *, root_name: str, experiment_name: str | None = None) -> list[Path]:
    """Return run directories under one data root, newest first."""
    root = Path(project_root)
    experiments_root = root / "data" / str(root_name)
    if not experiments_root.exists():
        return []
    runs: list[Path] = []
    if experiment_name:
        experiment_roots = [experiments_root / sanitize_output_name(experiment_name)]
    else:
        experiment_roots = [path for path in experiments_root.iterdir() if path.is_dir()]
    for experiment_root in experiment_roots:
        if not experiment_root.exists() or not experiment_root.is_dir():
            continue
        for candidate in experiment_root.iterdir():
            if _looks_like_run_dir(candidate):
                runs.append(candidate)
    return sorted(runs, key=lambda path: (path.stat().st_mtime, path.name), reverse=True)


def load_run_review(run_dir: Path) -> RunReview:
    """Load the local review sidecar, returning a default debug review if absent."""
    path = Path(run_dir) / REVIEW_FILENAME
    if not path.exists():
        return RunReview()
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return RunReview(notes="Unreadable run_review.json")
    if not isinstance(payload, dict):
        return RunReview(notes="Invalid run_review.json")
    status = str(payload.get("review_status") or "debug")
    if status not in REVIEW_STATUSES:
        status = "debug"
    return RunReview(
        review_status=status,
        notes=str(payload.get("notes") or ""),
        created_at=str(payload.get("created_at") or ""),
        reviewed_at=str(payload.get("reviewed_at") or ""),
        reviewed_by=str(payload.get("reviewed_by") or payload.get("reviewer") or ""),
        intended_use=str(payload.get("intended_use") or ""),
        include_in_evidence_index=bool(payload.get("include_in_evidence_index", False)),
    )


def ensure_run_review(
    run_dir: Path,
    *,
    status: str = "debug",
    intended_use: str = "debug",
    include_in_evidence_index: bool = False,
) -> RunReview:
    """Create a default run_review.json if one does not already exist."""
    run_dir = Path(run_dir)
    path = run_dir / REVIEW_FILENAME
    if path.exists():
        return load_run_review(run_dir)
    review = RunReview(
        review_status=status if status in REVIEW_STATUSES else "debug",
        notes="",
        created_at=_utc_now(),
        reviewed_at="",
        reviewed_by="",
        intended_use=str(intended_use or "debug"),
        include_in_evidence_index=bool(include_in_evidence_index),
    )
    path.write_text(json.dumps(review.to_dict(), indent=2), encoding="utf-8")
    return review


def write_run_review(
    run_dir: Path,
    *,
    status: str,
    notes: str = "",
    reviewed_by: str = "",
    include_in_evidence_index: bool | None = None,
    intended_use: str | None = None,
) -> RunReview:
    """Create or update run_review.json without touching raw experiment outputs."""
    run_dir = Path(run_dir)
    if not run_dir.exists() or not run_dir.is_dir():
        raise FileNotFoundError(f"Run directory does not exist: {run_dir}")
    status = str(status)
    if status not in REVIEW_STATUSES:
        raise ValueError(f"Unknown review status '{status}'. Expected one of: {', '.join(sorted(REVIEW_STATUSES))}")
    previous = load_run_review(run_dir)
    review = RunReview(
        review_status=status,
        notes=str(notes if notes != "" else previous.notes),
        created_at=previous.created_at or _utc_now(),
        reviewed_at=_utc_now(),
        reviewed_by=str(reviewed_by or previous.reviewed_by),
        intended_use=str(
            intended_use
            if intended_use not in (None, "")
            else (previous.intended_use or _default_intended_use_for_status(status))
        ),
        include_in_evidence_index=(
            bool(include_in_evidence_index)
            if include_in_evidence_index is not None
            else (status in {"thesis_candidate", "advisor_share"})
        ),
    )
    (run_dir / REVIEW_FILENAME).write_text(json.dumps(review.to_dict(), indent=2), encoding="utf-8")
    return review


def summarize_run(run_dir: Path) -> ManagedRunSummary:
    """Summarize one run for the Data tab and thesis evidence index."""
    run_dir = Path(run_dir)
    validation = validate_run_folder(run_dir)
    metadata = _read_json(run_dir / "metadata.json")
    summary = _read_json(run_dir / "summary.json")
    metrics = _as_dict(summary.get("experiment_metrics"))
    metadata_trust = _as_dict(metadata.get("trust_info"))
    metadata_provenance = _as_dict(metadata.get("provenance_info"))
    run_trust = _as_dict(metrics.get("run_trust"))
    run_provenance = _as_dict(metrics.get("run_provenance"))
    experiment_name = str(
        metadata.get("experiment_name")
        or summary.get("experiment_name")
        or validation.experiment_name
        or _infer_experiment_name(run_dir)
    )
    run_id = str(metadata.get("run_id") or summary.get("run_id") or run_dir.name)
    runtime_tip = _first_dict(
        run_provenance.get("runtime_tip_calibration"),
        metadata_provenance.get("runtime_tip_calibration"),
        run_provenance.get("runtime_tip"),
        metadata_provenance.get("runtime_tip"),
    )
    registration = _first_dict(
        run_provenance.get("registration"),
        metadata_provenance.get("registration"),
        metadata.get("registration_info"),
    )
    active_segment = _format_segment(
        _first_value(
            run_provenance.get("active_segment"),
            metadata_provenance.get("active_segment"),
            _nested(run_provenance, "operating_context", "active_segment"),
        )
    )
    two_segment_foundation = _first_dict(
        run_provenance.get("two_segment_foundation"),
        metadata_provenance.get("two_segment_foundation"),
        _nested(run_provenance, "operating_context", "two_segment_foundation"),
    )
    samples_path = run_dir / "samples.jsonl"
    review_path = run_dir / REVIEW_FILENAME
    review = load_run_review(run_dir)
    root_location = _root_location(run_dir)
    mock_mode = _first_value(
        metadata_provenance.get("mock_mode"),
        run_provenance.get("mock_mode"),
        _nested(metadata, "backend_info", "mock_mode"),
    )
    return ManagedRunSummary(
        run_dir=run_dir,
        experiment_name=sanitize_output_name(experiment_name),
        run_id=run_id,
        timestamp_label=_timestamp_label(run_dir, metadata, summary),
        validation_status=validation.status,
        validation_issues=[f"{issue.level}: {issue.message}" for issue in validation.issues],
        success=summary.get("success"),
        status=str(summary.get("status") or metadata.get("status") or "unknown"),
        stop_or_failure_reason=str(
            _first_value(
                metrics.get("stop_reason"),
                metrics.get("failure_reason"),
                summary.get("stop_reason"),
                summary.get("failure_reason"),
                "",
            )
            or ""
        ),
        run_trust_mode=str(
            _first_value(
                metrics.get("run_trust_mode"),
                run_trust.get("run_trust_mode"),
                metadata_trust.get("run_trust_mode"),
                "unknown",
            )
        ),
        valid_for_model_training=_first_value(
            metrics.get("valid_for_model_training"),
            run_trust.get("valid_for_model_training"),
            metadata_trust.get("valid_for_model_training"),
        ),
        valid_for_two_segment_model_training=_first_value(
            metrics.get("valid_for_two_segment_model_training"),
            run_trust.get("valid_for_two_segment_model_training"),
            metadata_trust.get("valid_for_two_segment_model_training"),
        ),
        valid_for_thesis_repeatability=_first_value(
            metrics.get("valid_for_thesis_repeatability"),
            run_trust.get("valid_for_thesis_repeatability"),
            metadata_trust.get("valid_for_thesis_repeatability"),
        ),
        data_quality_warnings=_as_string_list(
            _first_value(metrics.get("data_quality_warnings"), metadata_trust.get("data_quality_warnings"), [])
        ),
        operating_mode=str(
            _first_value(run_provenance.get("operating_mode"), metadata_provenance.get("operating_mode"), "unknown")
        ),
        active_segment=active_segment,
        two_segment_summary=_format_two_segment_foundation(two_segment_foundation),
        two_segment_pose_summary=_format_two_segment_pose_summary(metrics),
        two_segment_modeling_summary=_format_two_segment_modeling_summary(metrics),
        hardware_profile=str(
            _first_value(run_provenance.get("hardware_profile"), metadata_provenance.get("hardware_profile"), "unknown")
        ),
        runtime_tip_mode=str(_first_value(runtime_tip.get("mode"), metrics.get("runtime_tip_mode_used"), "")),
        runtime_tip_trust_status=str(_first_value(runtime_tip.get("trust_status"), runtime_tip.get("trust"), "")),
        registration_summary=_format_registration(registration),
        pretension_startup_summary=_format_startup_summary(metrics, metadata_provenance, run_provenance),
        report_figures=_relative_files(
            run_dir,
            lambda path: path.suffix.lower() == ".png"
            and ("_report" in path.name or path.name.startswith("thesis_")),
        ),
        metrics_files=_relative_files(run_dir, lambda path: path.suffix.lower() == ".csv"),
        samples_present=samples_path.exists(),
        samples_size_bytes=samples_path.stat().st_size if samples_path.exists() else 0,
        total_size_bytes=_dir_size_bytes(run_dir),
        mock_mode=(None if mock_mode is None else bool(mock_mode)),
        root_location=root_location,
        has_run_review=review_path.exists(),
        review=review,
    )


def detail_pairs_for_run(summary: ManagedRunSummary, *, project_root: Path | None = None) -> list[tuple[str, str]]:
    """Return human-readable detail rows for one run summary."""
    path = summary.run_dir
    if project_root is not None:
        try:
            path_text = str(summary.run_dir.relative_to(project_root))
        except ValueError:
            path_text = str(summary.run_dir)
    else:
        path_text = str(path)
    pairs = [
        ("Experiment", summary.experiment_name),
        ("Run ID", summary.run_id),
        ("Run Path", path_text),
        ("Validation", summary.validation_status),
        ("Success / Status", f"{summary.success} / {summary.status}"),
        ("Stop / Failure", summary.stop_or_failure_reason or "n/a"),
        ("Trust Mode", summary.run_trust_mode),
        ("Model Training Valid", _display_value(summary.valid_for_model_training)),
        ("Two-Segment Model Training Valid", _display_value(summary.valid_for_two_segment_model_training)),
        ("Thesis Repeatability Valid", _display_value(summary.valid_for_thesis_repeatability)),
        ("Warnings", "; ".join(summary.data_quality_warnings) or "none"),
        ("Operating Mode", summary.operating_mode),
        ("Active Segment", summary.active_segment or "n/a"),
        ("Two-Segment Foundation", summary.two_segment_summary or "n/a"),
        ("Two-Segment Pose Roles", summary.two_segment_pose_summary or "n/a"),
        ("Two-Segment Modeling", summary.two_segment_modeling_summary or "n/a"),
        ("Hardware Profile", summary.hardware_profile),
        ("Runtime Tip", _join_nonempty([summary.runtime_tip_mode, summary.runtime_tip_trust_status]) or "n/a"),
        ("Registration", summary.registration_summary or "n/a"),
        ("Startup / Pretension", summary.pretension_startup_summary or "n/a"),
        ("Report Figures", _count_and_names(summary.report_figures)),
        ("Metrics Files", _count_and_names(summary.metrics_files)),
        ("Samples", f"{'present' if summary.samples_present else 'absent'} ({summary.samples_size_bytes} bytes)"),
        ("Run Size", f"{summary.total_size_bytes} bytes"),
        ("Root", summary.root_location or "n/a"),
        ("Mock Mode", _display_value(summary.mock_mode)),
        ("Run Review Sidecar", "present" if summary.has_run_review else "missing"),
        ("Review Status", summary.review.review_status),
        ("Evidence Index", str(summary.review.include_in_evidence_index)),
        ("Intended Use", summary.review.intended_use or "n/a"),
        ("Review Notes", summary.review.notes or "n/a"),
    ]
    failure_context = _read_json(path / "failure_context.json")
    if failure_context:
        pairs.append(("Failure Context", str(failure_context.get("failure_reason") or failure_context.get("failure_category") or "present")))
    quality = _read_json(path / "dataset_quality_summary.json")
    if quality:
        pairs.append(("Dataset Quality", str(quality.get("recommendation") or "present")))
        pairs.append(("Recovered Packet Errors", str(quality.get("recovered_packet_error_count", 0))))
    long_run = _read_json(path / "long_run_health.json")
    if long_run:
        long_run_pieces = [
            f"stop={long_run.get('stop_reason') or 'n/a'}",
            f"cycles={long_run.get('cycles_completed', 0)}",
            f"accepted={long_run.get('accepted_samples', 0)}",
            f"rejected={long_run.get('rejected_samples', 0)}",
            f"cmd_fail={long_run.get('command_failures', 0)}",
            f"transport_fail={long_run.get('transport_failures', 0)}",
        ]
        if long_run.get("continue_until_valid_samples"):
            long_run_pieces.append(f"target={long_run.get('target_valid_sample_count')}")
        pairs.append(("Long-Run Health", "; ".join(long_run_pieces)))
    summary_payload = _read_json(path / "summary.json")
    metrics_payload = summary_payload.get("experiment_metrics") if isinstance(summary_payload, dict) else {}
    current_load = (metrics_payload or {}).get("current_load_summary") if isinstance(metrics_payload, dict) else None
    if isinstance(current_load, dict) and current_load:
        pieces = []
        sustained = list(current_load.get("sustained_jam_servo_ids") or [])
        if sustained:
            pieces.append(f"sustained_servos={sustained}")
        if current_load.get("any_hard_observed"):
            pieces.append("hard_threshold_observed")
        elif current_load.get("any_warning_observed"):
            pieces.append("warning_threshold_observed")
        per_servo = current_load.get("per_servo") or {}
        peak_servo = None
        peak_value = -1.0
        for servo_id, data in dict(per_servo).items():
            try:
                value = float(dict(data.get("load_proxy") or {}).get("max_ma") or 0.0)
            except (TypeError, ValueError):
                value = 0.0
            if value > peak_value:
                peak_value = value
                peak_servo = servo_id
        if peak_servo is not None and peak_value > 0.0:
            pieces.append(f"peak_load_proxy_ma={peak_value:.0f}@servo{peak_servo}")
        if pieces:
            pairs.append(("Current/Load Summary", "; ".join(pieces)))
    tracker_freshness = (metrics_payload or {}).get("tracker_freshness_summary") if isinstance(metrics_payload, dict) else None
    if isinstance(tracker_freshness, dict) and tracker_freshness:
        pieces = []
        if tracker_freshness.get("any_stale_observed"):
            pieces.append(f"stale_samples={tracker_freshness.get('samples_stale', 0)}")
        max_age = tracker_freshness.get("max_freshness_s")
        if isinstance(max_age, (int, float)):
            pieces.append(f"max_age_s={float(max_age):.3f}")
        p95_age = tracker_freshness.get("p95_freshness_s")
        if isinstance(p95_age, (int, float)):
            pieces.append(f"p95_age_s={float(p95_age):.3f}")
        if pieces:
            pairs.append(("Tracker Freshness", "; ".join(pieces)))
    scatter = _read_json(path / "two_segment_repeatability_scatter_metrics.json")
    if scatter:
        scatter_pieces = []
        distal = scatter.get("aggregate_distal_rms_mm")
        intermediate = scatter.get("aggregate_intermediate_rms_mm")
        if distal is not None:
            scatter_pieces.append(f"distal_rms_mm={distal:.3f}")
        if intermediate is not None:
            scatter_pieces.append(f"intermediate_rms_mm={intermediate:.3f}")
        if scatter_pieces:
            pairs.append(("Two-Segment Repeatability Scatter", "; ".join(scatter_pieces)))
    return pairs


def archive_run(run_dir: Path, *, project_root: Path, force: bool = False) -> MoveRunResult:
    """Move a run into data/experiments_archived without permanently deleting it."""
    return _move_run(run_dir, project_root=project_root, action="archive", root_name="experiments_archived", force=force)


def trash_run(run_dir: Path, *, project_root: Path, force: bool = False) -> MoveRunResult:
    """Move a run into data/trash without permanently deleting it."""
    return _move_run(run_dir, project_root=project_root, action="trash", root_name="trash", force=force)


def _move_run(run_dir: Path, *, project_root: Path, action: str, root_name: str, force: bool) -> MoveRunResult:
    run_dir = Path(run_dir).resolve()
    project_root = Path(project_root).resolve()
    if not run_dir.exists() or not run_dir.is_dir():
        raise FileNotFoundError(f"Run directory does not exist: {run_dir}")
    try:
        run_dir.relative_to(project_root)
    except ValueError as exc:
        raise ValueError(f"Refusing to move run outside project root: {run_dir}") from exc
    review = load_run_review(run_dir)
    if review.review_status in PROTECTED_REVIEW_STATUSES and not force:
        raise ValueError(
            f"Run is marked {review.review_status}; use force after explicit confirmation to {action} it."
        )
    summary = summarize_run(run_dir)
    destination_root = project_root / "data" / root_name / summary.experiment_name
    destination = _collision_safe_path(destination_root / run_dir.name)
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(run_dir), str(destination))
    if action == "archive":
        write_run_review(destination, status="archived", notes=review.notes, reviewed_by=review.reviewed_by)
    elif action == "trash":
        write_run_review(destination, status="garbage", notes=review.notes, reviewed_by=review.reviewed_by, include_in_evidence_index=False)
    return MoveRunResult(source_path=run_dir, destination_path=destination, action=action)


def _looks_like_run_dir(path: Path) -> bool:
    return path.is_dir() and ((path / "summary.json").exists() or (path / "metadata.json").exists())


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def _as_dict(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, dict) else {}


def _first_dict(*values: Any) -> dict[str, Any]:
    for value in values:
        if isinstance(value, dict) and value:
            return value
    return {}


def _first_value(*values: Any) -> Any:
    for value in values:
        if value not in (None, "", {}):
            return value
    return None


def _nested(payload: dict[str, Any], *keys: str) -> Any:
    current: Any = payload
    for key in keys:
        if not isinstance(current, dict):
            return None
        current = current.get(key)
    return current


def _as_string_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(item) for item in value if str(item)]
    if isinstance(value, tuple):
        return [str(item) for item in value if str(item)]
    if isinstance(value, str) and value.strip():
        return [value.strip()]
    return []


def _dir_size_bytes(path: Path) -> int:
    total = 0
    if not Path(path).exists():
        return 0
    for candidate in Path(path).rglob("*"):
        if candidate.is_file():
            try:
                total += candidate.stat().st_size
            except OSError:
                continue
    return total


def _root_location(run_dir: Path) -> str:
    parts = list(Path(run_dir).parts)
    for index, part in enumerate(parts[:-1]):
        if part == "data" and index + 1 < len(parts):
            return f"data/{parts[index + 1]}"
    return ""


def _default_intended_use_for_status(status: str) -> str:
    if status == "thesis_candidate":
        return "thesis"
    if status == "advisor_share":
        return "advisor"
    if status == "garbage":
        return "debug"
    if status == "keep":
        return "thesis"
    if status == "archived":
        return "debug"
    return "debug"


def _format_segment(value: Any) -> str:
    if isinstance(value, dict):
        key = value.get("key") or value.get("segment_key") or value.get("name")
        label = value.get("label")
        servo_ids = value.get("servo_ids")
        pieces = [str(part) for part in (label, key) if part]
        if servo_ids:
            pieces.append(str(servo_ids))
        return " ".join(pieces)
    return str(value or "")


def _format_two_segment_foundation(value: dict[str, Any]) -> str:
    if not value:
        return ""
    segments = _as_dict(value.get("segments"))
    segment_order = value.get("segment_order") if isinstance(value.get("segment_order"), list) else []
    pieces = [
        f"available={bool(value.get('available'))}",
        str(value.get("semantic_scope") or ""),
    ]
    ordered_segments: list[str] = []
    for key in segment_order:
        data = _as_dict(segments.get(str(key)))
        if not data:
            continue
        label = data.get("segment_label") or data.get("label") or key
        role = data.get("segment_role") or ""
        ids = data.get("servo_ids") or []
        ordered_segments.append(" ".join(str(part) for part in (label, role, ids) if part))
    if ordered_segments:
        pieces.append("; ".join(ordered_segments))
    bottom_top = _as_dict(value.get("bottom_top"))
    if bottom_top:
        bottom_label = bottom_top.get("bottom_segment_label") or bottom_top.get("bottom_segment_key") or ""
        top_label = bottom_top.get("top_segment_label") or bottom_top.get("top_segment_key") or ""
        bottom_ids = bottom_top.get("bottom_servo_ids") or []
        top_ids = bottom_top.get("top_servo_ids") or []
        if bottom_label or top_label:
            pieces.append(f"Bottom={bottom_label}{bottom_ids}, Top={top_label}{top_ids}")
    raw_operator_notes = str(_as_dict(value.get("physical_assembly")).get("notes") or "").strip()
    if raw_operator_notes:
        # Display in a single line; newlines / tabs would break the Data-tab row.
        # The full raw text remains available in the underlying JSON.
        flattened = " ".join(raw_operator_notes.split())
        if len(flattened) > 200:
            flattened = flattened[:197] + "..."
        pieces.append(f"operator_notes={flattened}")
    assembly_issues = list(value.get("physical_assembly_issues") or [])
    if assembly_issues:
        pieces.append("assembly_issues=[" + "; ".join(str(item) for item in assembly_issues) + "]")
    return _join_nonempty(pieces)


def _format_two_segment_pose_summary(metrics: dict[str, Any]) -> str:
    if str(metrics.get("dataset_type") or "") != "two_segment_collect_pose_command_dataset":
        return ""
    pose = _as_dict(metrics.get("pose_label_summary"))
    role_config = _as_dict(metrics.get("two_segment_tracking_role_config"))
    required = [
        role
        for role, data in role_config.items()
        if bool(_as_dict(data).get("required_for_two_segment_model_training", False))
    ]
    return _join_nonempty(
        [
            f"available={pose.get('available_roles', [])}",
            f"missing_required={pose.get('missing_required_roles', required)}",
            f"distal_only={bool(metrics.get('distal_only'))}",
            f"includes_intermediate={bool(metrics.get('includes_intermediate_pose'))}",
        ]
    )


def _format_two_segment_modeling_summary(metrics: dict[str, Any]) -> str:
    if str(metrics.get("dataset_type") or "") != "two_segment_modeling":
        return ""
    models = _as_dict(metrics.get("models"))
    completed = [
        key
        for key, value in models.items()
        if str(_as_dict(value).get("status") or "") == "completed"
    ]
    unavailable = [
        key
        for key, value in models.items()
        if str(_as_dict(value).get("status") or "") == "unavailable"
    ]
    return _join_nonempty(
        [
            f"samples={metrics.get('accepted_sample_count')}",
            f"split={metrics.get('split_method')}",
            f"completed={completed}",
            f"unavailable={unavailable}",
            f"label_mode={metrics.get('label_mode')}",
            f"includes_intermediate_label={bool(metrics.get('includes_intermediate_label'))}",
            f"orientation_available={bool(metrics.get('orientation_available'))}",
            f"physics={metrics.get('physics_model_status', {})}",
        ]
    )


def _format_registration(value: dict[str, Any]) -> str:
    if not value:
        return ""
    candidates = [
        value.get("artifact_path"),
        value.get("registration_path"),
        value.get("record_id"),
        value.get("status"),
    ]
    fre = _first_value(value.get("fre_mm"), value.get("registration_fre_mm"))
    text = _join_nonempty(str(item) for item in candidates if item)
    if fre is not None:
        text = _join_nonempty([text, f"FRE {fre} mm"])
    return text


def _format_startup_summary(metrics: dict[str, Any], metadata_provenance: dict[str, Any], run_provenance: dict[str, Any]) -> str:
    if str(metrics.get("startup_type") or "") == "manual_two_segment_startup":
        stages = metrics.get("stage_completion") if isinstance(metrics.get("stage_completion"), dict) else {}
        completed = sum(1 for value in stages.values() if bool(value))
        path = str(metrics.get("final_startup_artifact_path") or "")
        accepted = "accepted" if metrics.get("final_accepted") else "not accepted"
        return _join_nonempty(
            [
                "manual_two_segment_startup",
                f"{completed}/5 stages",
                accepted,
                path,
            ]
        )
    startup = _first_dict(
        metrics.get("startup_reference"),
        metrics.get("startup_artifact"),
        metadata_provenance.get("startup_reference"),
        metadata_provenance.get("pretension_artifact"),
        run_provenance.get("startup_reference"),
        run_provenance.get("pretension_artifact"),
    )
    if startup:
        return _join_nonempty(
            [
                str(startup.get("source") or startup.get("startup_source") or ""),
                str(startup.get("artifact_path") or startup.get("path") or ""),
                str(startup.get("status") or ""),
            ]
        )
    source = _first_value(metrics.get("startup_source"), metrics.get("pretension_source"))
    return str(source or "")


def _relative_files(run_dir: Path, predicate) -> list[str]:
    files: list[str] = []
    for path in sorted(run_dir.rglob("*")):
        if path.is_file() and predicate(path):
            try:
                files.append(str(path.relative_to(run_dir)))
            except ValueError:
                files.append(str(path))
    return files


def _timestamp_label(run_dir: Path, metadata: dict[str, Any], summary: dict[str, Any]) -> str:
    timestamp = str(metadata.get("timestamp_utc") or summary.get("timestamp_utc") or "")
    if timestamp:
        return timestamp
    parts = run_dir.name.split("_", 2)
    if len(parts) >= 2 and parts[0].isdigit() and parts[1].isdigit():
        return f"{parts[0]}_{parts[1]}"
    return run_dir.name


def _infer_experiment_name(run_dir: Path) -> str:
    if run_dir.parent.name and run_dir.parent.name != "experiments":
        return sanitize_output_name(run_dir.parent.name)
    parts = run_dir.name.split("_", 2)
    return sanitize_output_name(parts[2] if len(parts) >= 3 else run_dir.name)


def _display_value(value: Any) -> str:
    return "unknown" if value is None else str(value)


def _join_nonempty(values) -> str:
    return " | ".join(str(value) for value in values if str(value).strip())


def _count_and_names(values: list[str]) -> str:
    if not values:
        return "0"
    preview = ", ".join(values[:4])
    if len(values) > 4:
        preview += f", ... (+{len(values) - 4})"
    return f"{len(values)}: {preview}"


def _collision_safe_path(path: Path) -> Path:
    candidate = Path(path)
    if not candidate.exists():
        return candidate
    suffix = 1
    while True:
        next_candidate = candidate.with_name(f"{candidate.name}_{suffix:02d}")
        if not next_candidate.exists():
            return next_candidate
        suffix += 1


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()

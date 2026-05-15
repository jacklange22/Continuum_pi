"""Dataset loading for two-segment modeling analysis.

This module consumes saved ``two_segment_collect_pose_command_dataset`` runs.
It does not command hardware, train live controllers, or alter tracking math.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
import json
from pathlib import Path
from typing import Any

import numpy as np


DATASET_EXPERIMENT_NAME = "two_segment_collect_pose_command_dataset"
MODEL_SCHEMA_VERSION = "two_segment_modeling_dataset_v1"
SERVO_ID_ORDER = [1, 2, 3, 4, 5, 6, 7, 8]


@dataclass(frozen=True)
class RejectedSample:
    """One sample that was intentionally not used for modeling."""

    run_dir: Path
    sample_index: int
    reason: str
    payload: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class TwoSegmentModelingSample:
    """One accepted sample with canonical feature and label data."""

    run_dir: Path
    run_name: str
    sample_index: int
    timestamp_utc: str
    feature_mm: np.ndarray
    distal_position_mm: np.ndarray
    distal_tangent: np.ndarray | None
    intermediate_position_mm: np.ndarray | None
    intermediate_tangent: np.ndarray | None
    segment_command: dict[str, list[float]]
    ordered_servo_ids: list[int]
    segment_order: list[str]
    segment_metadata: dict[str, Any]
    command_schema_version: str
    command_units_source: str
    startup_artifact_path: str
    startup_artifact_hash: str
    run_trust_mode: str
    distal_only: bool
    includes_intermediate_pose: bool
    pose_role_summary: dict[str, Any]
    source_payload: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class TwoSegmentModelingDataset:
    """Accepted/rejected samples and run-level provenance."""

    samples: list[TwoSegmentModelingSample]
    rejected_samples: list[RejectedSample]
    run_provenance: list[dict[str, Any]]
    allow_lower_trust: bool = False

    @property
    def accepted_count(self) -> int:
        return len(self.samples)

    @property
    def rejected_count(self) -> int:
        return len(self.rejected_samples)

    @property
    def run_count(self) -> int:
        return len({str(sample.run_dir) for sample in self.samples} | {str(item.run_dir) for item in self.rejected_samples})

    @property
    def orientation_available(self) -> bool:
        return bool(self.samples) and all(sample.distal_tangent is not None for sample in self.samples)

    @property
    def two_coil_xyz_available(self) -> bool:
        return bool(self.samples) and all(sample.intermediate_position_mm is not None for sample in self.samples)

    @property
    def two_coil_orientation_available(self) -> bool:
        return bool(self.samples) and all(
            sample.intermediate_tangent is not None and sample.distal_tangent is not None
            for sample in self.samples
        )

    @property
    def includes_intermediate_pose(self) -> bool:
        return any(sample.includes_intermediate_pose for sample in self.samples)

    @property
    def distal_only(self) -> bool:
        return bool(self.samples) and all(sample.distal_only for sample in self.samples)

    def rejection_counts(self) -> dict[str, int]:
        return dict(Counter(item.reason for item in self.rejected_samples))


def load_two_segment_modeling_dataset(
    run_dirs: list[Path] | tuple[Path, ...],
    *,
    allow_lower_trust: bool = False,
) -> TwoSegmentModelingDataset:
    """Load one or more two-segment dataset runs for offline modeling."""

    accepted: list[TwoSegmentModelingSample] = []
    rejected: list[RejectedSample] = []
    provenance: list[dict[str, Any]] = []
    for raw_run_dir in run_dirs:
        run_dir = Path(raw_run_dir)
        if run_dir.is_file():
            run_dir = run_dir.parent
        summary = _read_json(run_dir / "summary.json")
        metadata = _read_json(run_dir / "metadata.json")
        role_provenance = _read_json(run_dir / "two_segment_tracking_role_provenance.json")
        config_snapshot = (run_dir / "config_snapshot.yaml").read_text(encoding="utf-8") if (run_dir / "config_snapshot.yaml").exists() else ""
        metrics = summary.get("experiment_metrics") if isinstance(summary.get("experiment_metrics"), dict) else {}
        provenance.append(
            {
                "run_dir": str(run_dir),
                "run_name": run_dir.name,
                "experiment_name": metadata.get("experiment_name") or summary.get("experiment_name") or run_dir.parent.name,
                "summary": summary,
                "metadata": metadata,
                "config_snapshot_present": bool(config_snapshot),
                "tracking_role_provenance": role_provenance,
                "startup_artifact_provenance": metrics.get("startup_artifact_provenance", {}),
                "pose_label_summary": metrics.get("pose_label_summary", {}),
            }
        )
        samples_path = run_dir / "samples.jsonl"
        if not samples_path.exists():
            rejected.append(RejectedSample(run_dir=run_dir, sample_index=-1, reason="samples_jsonl_missing"))
            continue
        for index, sample in enumerate(_iter_jsonl(samples_path)):
            reason = _rejection_reason(sample, metrics=metrics, allow_lower_trust=allow_lower_trust)
            if reason is not None:
                rejected.append(RejectedSample(run_dir=run_dir, sample_index=index, reason=reason, payload=_thin_payload(sample)))
                continue
            try:
                accepted.append(_accepted_sample(run_dir=run_dir, sample_index=index, sample=sample))
            except ValueError as exc:
                rejected.append(RejectedSample(run_dir=run_dir, sample_index=index, reason=str(exc), payload=_thin_payload(sample)))
    return TwoSegmentModelingDataset(
        samples=accepted,
        rejected_samples=rejected,
        run_provenance=provenance,
        allow_lower_trust=bool(allow_lower_trust),
    )


def _accepted_sample(*, run_dir: Path, sample_index: int, sample: dict[str, Any]) -> TwoSegmentModelingSample:
    command_record = _as_dict(sample.get("two_segment_command"))
    extra = _as_dict(sample.get("extra"))
    feature_mm, source_units = extract_feature_mm(sample)
    position_mm = extract_distal_position_mm(sample)
    if position_mm is None:
        raise ValueError("distal_tip_robot_frame_pose_missing")
    tangent = extract_distal_tangent(sample)
    intermediate_position = extract_role_position_mm(sample, "intermediate_segment")
    intermediate_tangent = extract_role_tangent(sample, "intermediate_segment")
    startup = _as_dict(extra.get("startup_artifact_provenance"))
    segment_order = _segment_order(sample)
    return TwoSegmentModelingSample(
        run_dir=run_dir,
        run_name=run_dir.name,
        sample_index=int(sample_index),
        timestamp_utc=str(sample.get("wall_time_utc") or ""),
        feature_mm=feature_mm,
        distal_position_mm=position_mm,
        distal_tangent=tangent,
        intermediate_position_mm=intermediate_position,
        intermediate_tangent=intermediate_tangent,
        segment_command=_segment_command_mm(command_record),
        ordered_servo_ids=[int(value) for value in list(extra.get("commanded_servo_ids") or command_record.get("commanded_servo_ids") or SERVO_ID_ORDER)],
        segment_order=segment_order,
        segment_metadata=_segment_metadata(sample, segment_order=segment_order),
        command_schema_version=str(extra.get("command_schema_version") or command_record.get("schema_version") or ""),
        command_units_source=source_units,
        startup_artifact_path=str(startup.get("artifact_path") or ""),
        startup_artifact_hash=str(startup.get("artifact_sha256") or startup.get("sha256") or ""),
        run_trust_mode=str(extra.get("run_trust_mode") or ""),
        distal_only=bool(extra.get("distal_only", False)),
        includes_intermediate_pose=bool(extra.get("includes_intermediate_pose", False)),
        pose_role_summary={
            "available_pose_roles": list(extra.get("available_pose_roles") or []),
            "missing_pose_roles": list(extra.get("missing_pose_roles") or []),
            "missing_required_pose_roles": list(extra.get("missing_required_pose_roles") or []),
            "pose_trust_by_role": _as_dict(extra.get("pose_trust_by_role")),
            "tracker_freshness_by_role": _as_dict(extra.get("tracker_freshness_by_role")),
        },
        source_payload=_thin_payload(sample),
    )


def extract_feature_mm(sample: dict[str, Any]) -> tuple[np.ndarray, str]:
    """Return canonical 8-tendon displacement features in mm."""

    command_record = _as_dict(sample.get("two_segment_command"))
    extra = _as_dict(sample.get("extra"))
    units = str(extra.get("command_units") or command_record.get("units") or "").strip().lower()
    flat = command_record.get("flat_command_cm")
    source_units = units or "cm"
    if not isinstance(flat, list):
        flat = extra.get("ordered_8_displacements_cm")
        source_units = "cm"
    if not isinstance(flat, list) or len(flat) != 8:
        raise ValueError("ordered_8_tendon_displacements_missing")
    values = np.asarray([float(value) for value in flat], dtype=float)
    if source_units in {"cm", "centimeter", "centimeters"}:
        return values * 10.0, "cm"
    if source_units in {"mm", "millimeter", "millimeters"}:
        return values, "mm"
    raise ValueError("tendon_displacement_units_unknown")


def extract_distal_position_mm(sample: dict[str, Any]) -> np.ndarray | None:
    """Extract distal-tip XYZ in robot frame without changing transform math."""

    point = extract_role_position_mm(sample, "distal_tip")
    if point is not None:
        return point
    pose = _as_dict(_nested(sample, "two_segment_pose", "distal_tip_pose"))
    if str(_nested(sample, "two_segment_pose", "frame") or "").lower() == "robot":
        point = _translation_from_matrix(pose.get("T_robot_tip"))
        if point is not None:
            return point
        translation = pose.get("translation_mm")
        if isinstance(translation, list) and len(translation) >= 3:
            return np.asarray([float(translation[0]), float(translation[1]), float(translation[2])], dtype=float)
    return None


def extract_distal_tangent(sample: dict[str, Any]) -> np.ndarray | None:
    """Extract explicit tangent/orientation labels when present.

    The scaffold intentionally does not synthesize tangent labels from missing
    fields. If future datasets store a calibrated tangent vector, it is used.
    """

    return extract_role_tangent(sample, "distal_tip")


def extract_role_position_mm(sample: dict[str, Any], role: str) -> np.ndarray | None:
    """Extract a named role XYZ label in robot frame when explicitly stored."""

    role_pose = _as_dict(_nested(sample, "pose_in_robot_frame", "roles", role))
    for matrix_key in ("T_robot_tip", "T_robot_tool", "T_robot_intermediate", "T_robot_role"):
        point = _translation_from_matrix(role_pose.get(matrix_key))
        if point is not None:
            return point
    translation = role_pose.get("translation_mm")
    if isinstance(translation, list) and len(translation) >= 3:
        return np.asarray([float(translation[0]), float(translation[1]), float(translation[2])], dtype=float)
    if role != "distal_tip":
        pose = _as_dict(_nested(sample, "two_segment_pose", "role_observations", role))
        if str(_nested(sample, "two_segment_pose", "frame") or "").lower() == "robot":
            translation = pose.get("translation_mm")
            if isinstance(translation, list) and len(translation) >= 3:
                return np.asarray([float(translation[0]), float(translation[1]), float(translation[2])], dtype=float)
    return None


def extract_role_tangent(sample: dict[str, Any], role: str) -> np.ndarray | None:
    """Extract explicit tangent/orientation labels for a named role."""

    candidates = [
        _nested(sample, "pose_in_robot_frame", "roles", role, "tangent_xyz"),
        _nested(sample, "pose_in_robot_frame", "roles", role, "orientation_tangent_xyz"),
    ]
    if role == "distal_tip":
        candidates.extend(
            [
                _nested(sample, "two_segment_pose", "distal_tip_pose", "tangent_xyz"),
                _nested(sample, "two_segment_pose", "distal_tip_pose", "orientation_tangent_xyz"),
            ]
        )
    else:
        candidates.extend(
            [
                _nested(sample, "two_segment_pose", "role_observations", role, "tangent_xyz"),
                _nested(sample, "two_segment_pose", "role_observations", role, "orientation_tangent_xyz"),
            ]
        )
    for value in candidates:
        if isinstance(value, list) and len(value) >= 3:
            vector = np.asarray([float(value[0]), float(value[1]), float(value[2])], dtype=float)
            norm = float(np.linalg.norm(vector))
            if norm > 0.0:
                return vector / norm
    return None


def _segment_order(sample: dict[str, Any]) -> list[str]:
    command_record = _as_dict(sample.get("two_segment_command"))
    extra = _as_dict(sample.get("extra"))
    candidates = [
        extra.get("segment_order"),
        command_record.get("segment_order"),
        _nested(sample, "metadata", "segment_order"),
    ]
    for value in candidates:
        if isinstance(value, list) and value:
            return [str(item) for item in value]
    return ["segment_a", "segment_b"]


def _segment_metadata(sample: dict[str, Any], *, segment_order: list[str]) -> dict[str, Any]:
    extra = _as_dict(sample.get("extra"))
    segments = _as_dict(extra.get("segments") or extra.get("segment_definitions"))
    fallback = {
        "segment_a": {
            "label": "Segment A",
            "role": "proximal",
            "role_aliases": ["proximal", "lower", "base", "bottom"],
            "servo_ids": [1, 2, 3, 4],
            "pair_mapping": [[1, 3], [2, 4]],
            "segment_order_index": segment_order.index("segment_a") if "segment_a" in segment_order else 0,
        },
        "segment_b": {
            "label": "Segment B",
            "role": "distal",
            "role_aliases": ["distal", "upper", "top"],
            "servo_ids": [5, 6, 7, 8],
            "pair_mapping": [[5, 7], [6, 8]],
            "segment_order_index": segment_order.index("segment_b") if "segment_b" in segment_order else 1,
        },
    }
    merged: dict[str, Any] = {}
    for segment in segment_order:
        raw = _as_dict(segments.get(segment))
        base = dict(fallback.get(segment, {"label": segment, "role": "", "servo_ids": [], "pair_mapping": []}))
        base.update(raw)
        base["segment_order_index"] = int(base.get("segment_order_index", segment_order.index(segment)))
        base["servo_ids"] = [int(value) for value in list(base.get("servo_ids") or [])]
        base["pair_mapping"] = [[int(v) for v in list(pair)] for pair in list(base.get("pair_mapping") or []) if isinstance(pair, list)]
        merged[segment] = base
    for segment, payload in fallback.items():
        if segment not in merged:
            merged[segment] = payload
    return merged


def _rejection_reason(sample: dict[str, Any], *, metrics: dict[str, Any], allow_lower_trust: bool) -> str | None:
    extra = _as_dict(sample.get("extra"))
    if str(extra.get("record_kind") or "") != "two_segment_dataset_capture":
        return "not_two_segment_dataset_capture"
    if not bool(extra.get("command_success", True)):
        return "command_failed"
    if not bool(extra.get("capture_accepted", False)):
        return "capture_not_accepted"
    if not allow_lower_trust and not bool(extra.get("valid_for_two_segment_model_training", False)):
        return "sample_not_two_segment_model_training_valid"
    if not allow_lower_trust and str(extra.get("run_trust_mode") or "").lower() in {"servo_only", "dry_run", "debug", "lower_trust"}:
        return "servo_only_or_lower_trust"
    startup = _as_dict(extra.get("startup_artifact_provenance") or metrics.get("startup_artifact_provenance"))
    if not allow_lower_trust and not bool(startup.get("accepted_all_8_startup")):
        return "accepted_all8_startup_missing"
    if allow_lower_trust and str(extra.get("run_trust_mode") or "").lower() in {"servo_only", "dry_run"}:
        return "servo_only_or_dry_run_samples_not_trainable"
    try:
        extract_feature_mm(sample)
    except ValueError as exc:
        return str(exc)
    if extract_distal_position_mm(sample) is None:
        return "distal_tip_robot_frame_pose_missing"
    if not allow_lower_trust and "distal_tip" in list(extra.get("missing_required_pose_roles") or []):
        return "distal_tip_required_role_missing"
    if not allow_lower_trust:
        missing = _missing_measured_servo_positions(extra)
        if missing:
            return f"measured_servo_position_missing:{','.join(str(x) for x in missing)}"
    return None


def _missing_measured_servo_positions(extra: dict[str, Any]) -> list[int]:
    """Return servo IDs in 1..8 with a None/missing measured position_tick."""
    feedback = _as_dict(extra.get("measured_servo_feedback"))
    if not feedback:
        # Older runs that did not record measured feedback at all are flagged
        # as missing-all so they get rejected from trusted modeling exports.
        return list(SERVO_ID_ORDER)
    missing: list[int] = []
    for servo_id in SERVO_ID_ORDER:
        data = _as_dict(feedback.get(str(int(servo_id))))
        if data.get("position_tick") in (None, ""):
            missing.append(int(servo_id))
    return missing


def _segment_command_mm(command_record: dict[str, Any]) -> dict[str, list[float]]:
    segments = _as_dict(command_record.get("segments"))
    units = str(command_record.get("units") or "cm").lower()
    factor = 10.0 if units in {"cm", "centimeter", "centimeters"} else 1.0
    return {
        "segment_a": [float(value) * factor for value in list(segments.get("segment_a") or [])],
        "segment_b": [float(value) * factor for value in list(segments.get("segment_b") or [])],
    }


def _translation_from_matrix(value: Any) -> np.ndarray | None:
    if not isinstance(value, list) or len(value) < 3:
        return None
    try:
        return np.asarray([float(value[0][3]), float(value[1][3]), float(value[2][3])], dtype=float)
    except Exception:
        return None


def _iter_jsonl(path: Path):
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            raw = line.strip()
            if not raw:
                continue
            payload = json.loads(raw)
            if isinstance(payload, dict):
                yield payload


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


def _nested(payload: dict[str, Any], *keys: str) -> Any:
    current: Any = payload
    for key in keys:
        if not isinstance(current, dict):
            return None
        current = current.get(key)
    return current


def _thin_payload(sample: dict[str, Any]) -> dict[str, Any]:
    extra = _as_dict(sample.get("extra"))
    return {
        "wall_time_utc": sample.get("wall_time_utc"),
        "phase": sample.get("phase"),
        "step_index": sample.get("step_index"),
        "sample_index": sample.get("sample_index"),
        "two_segment_command": sample.get("two_segment_command"),
        "two_segment_pose": sample.get("two_segment_pose"),
        "extra": {
            "record_kind": extra.get("record_kind"),
            "run_trust_mode": extra.get("run_trust_mode"),
            "capture_accepted": extra.get("capture_accepted"),
            "command_success": extra.get("command_success"),
            "valid_for_two_segment_model_training": extra.get("valid_for_two_segment_model_training"),
            "data_quality_warnings": extra.get("data_quality_warnings"),
        },
    }

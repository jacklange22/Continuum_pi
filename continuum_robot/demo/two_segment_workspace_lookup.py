"""Offline workspace lookup map builder for the two-segment penprobe demo.

The map is a downsampled set of (distal_xyz_robot_mm, all_8_goal_ticks)
points harvested from accepted `two_segment_collect_pose_command_dataset`
runs. At runtime, the penprobe demo finds the nearest map point to a live
0B target and commands the associated all-8 servo goals.

This module is DEMO-ONLY. It does not implement closed-loop control,
inverse kinematics, or thesis-valid modeling. The artifacts carry an
explicit `demo_only_artifact: true` flag and are blocked from the
`valid_for_model_training` chain.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, field
import json
import math
from pathlib import Path
import sys
from typing import Any

import numpy as np


MAP_SCHEMA_VERSION = "two_segment_workspace_lookup_map_v1"
SUPPORTED_INTERPOLATION_MODES = ("nearest", "inverse_distance")
SUPPORTED_LABEL_MODES = ("distal_xyz",)
COMMANDED_SERVO_IDS_DEFAULT = [1, 2, 3, 4, 5, 6, 7, 8]


@dataclass(frozen=True)
class LookupMapPoint:
    """One distal-XYZ-to-all-8-goal-ticks entry in the lookup map."""

    map_index: int
    distal_xyz_robot_mm: list[float]
    all_8_goal_ticks: dict[str, int]
    commanded_servo_ids: list[int]
    bottom_top_assignment: dict[str, Any]
    source_run_id: str
    source_sample_index: int
    run_trust_mode: str
    accepted_all_8_startup: bool
    quality_proxy: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "map_index": int(self.map_index),
            "distal_xyz_robot_mm": [float(v) for v in self.distal_xyz_robot_mm],
            "all_8_goal_ticks": {str(k): int(v) for k, v in self.all_8_goal_ticks.items()},
            "commanded_servo_ids": list(self.commanded_servo_ids),
            "bottom_top_assignment": dict(self.bottom_top_assignment),
            "source_run_id": str(self.source_run_id),
            "source_sample_index": int(self.source_sample_index),
            "run_trust_mode": str(self.run_trust_mode),
            "accepted_all_8_startup": bool(self.accepted_all_8_startup),
            "quality_proxy": dict(self.quality_proxy),
        }


@dataclass(frozen=True)
class RejectedMapCandidate:
    """One sample that was scanned but excluded from the map."""

    source_run_id: str
    source_sample_index: int
    reason: str


@dataclass
class LookupMapBuildConfig:
    """Operator config for the map builder."""

    label_mode: str = "distal_xyz"
    allow_lower_trust: bool = False
    max_points: int | None = None
    voxel_size_mm: float | None = 6.0
    knn: int = 1
    interpolation: str = "nearest"
    output_dir: Path | None = None
    figure_quality: str = "production"

    def __post_init__(self) -> None:
        if self.label_mode not in SUPPORTED_LABEL_MODES:
            raise ValueError(
                f"Unsupported label_mode {self.label_mode!r}; expected one of {SUPPORTED_LABEL_MODES}."
            )
        if self.interpolation not in SUPPORTED_INTERPOLATION_MODES:
            raise ValueError(
                f"Unsupported interpolation {self.interpolation!r}; expected one of {SUPPORTED_INTERPOLATION_MODES}."
            )
        if int(self.knn) < 1:
            raise ValueError("knn must be >= 1")


@dataclass(frozen=True)
class LookupMapBuildResult:
    """Outcome of a map build, including artifacts and rejection summary."""

    output_dir: Path
    map_points: list[LookupMapPoint]
    rejected: list[RejectedMapCandidate]
    metadata: dict[str, Any]
    artifact_paths: dict[str, Path]

    @property
    def accepted_count(self) -> int:
        return len(self.map_points)

    @property
    def rejected_count(self) -> int:
        return len(self.rejected)


# ---------------------------------------------------------------------------
# Sample iteration + filtering
# ---------------------------------------------------------------------------


def _iter_jsonl(path: Path):
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            raw = line.strip()
            if not raw:
                continue
            try:
                payload = json.loads(raw)
            except json.JSONDecodeError:
                continue
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


def _extract_distal_xyz_robot_mm(sample: dict[str, Any]) -> np.ndarray | None:
    """Pull the distal-tip XYZ in robot base frame. Mirrors the modeling loader."""
    roles_pose = _as_dict(_nested(sample, "pose_in_robot_frame", "roles", "distal_tip"))
    translation = roles_pose.get("translation_mm")
    if isinstance(translation, list) and len(translation) >= 3:
        return np.asarray([float(translation[0]), float(translation[1]), float(translation[2])], dtype=float)
    for matrix_key in ("T_robot_tip", "T_robot_tool", "T_robot_role"):
        matrix = roles_pose.get(matrix_key)
        if isinstance(matrix, list) and len(matrix) >= 3:
            try:
                return np.asarray(
                    [float(matrix[0][3]), float(matrix[1][3]), float(matrix[2][3])], dtype=float
                )
            except Exception:
                continue
    pose = _as_dict(_nested(sample, "two_segment_pose", "distal_tip_pose"))
    if str(_nested(sample, "two_segment_pose", "frame") or "").lower() == "robot":
        translation = pose.get("translation_mm")
        if isinstance(translation, list) and len(translation) >= 3:
            return np.asarray([float(translation[0]), float(translation[1]), float(translation[2])], dtype=float)
        matrix = pose.get("T_robot_tip")
        if isinstance(matrix, list) and len(matrix) >= 3:
            try:
                return np.asarray([float(matrix[0][3]), float(matrix[1][3]), float(matrix[2][3])], dtype=float)
            except Exception:
                pass
    return None


def _extract_all_8_goal_ticks(sample: dict[str, Any]) -> dict[int, int] | None:
    """Return the 8 goal ticks the bus actually saw for this sample."""
    extra = _as_dict(sample.get("extra"))
    raw = _as_dict(extra.get("goal_ticks_by_servo"))
    if not raw:
        commanded = _as_dict(sample.get("commanded_motor_values"))
        raw = commanded
    if not raw:
        return None
    ticks: dict[int, int] = {}
    for key, value in raw.items():
        try:
            ticks[int(key)] = int(value)
        except (TypeError, ValueError):
            return None
    expected = sorted({int(v) for v in (extra.get("commanded_servo_ids") or COMMANDED_SERVO_IDS_DEFAULT)})
    if set(ticks) != set(expected):
        return None
    return ticks


def _build_quality_proxy(sample: dict[str, Any]) -> dict[str, Any]:
    """Tiny per-sample quality summary for picking voxel representatives."""
    extra = _as_dict(sample.get("extra"))
    feedback = _as_dict(extra.get("measured_servo_feedback"))
    loads = []
    for entry in feedback.values():
        load = _as_dict(entry).get("load_proxy_ma")
        if load is not None:
            try:
                loads.append(float(load))
            except (TypeError, ValueError):
                continue
    quality: dict[str, Any] = {
        "max_load_proxy_ma": float(max(loads)) if loads else None,
        "mean_load_proxy_ma": float(sum(loads) / len(loads)) if loads else None,
        "missing_measured_servo_ids": list(extra.get("missing_measured_servo_ids") or []),
        "tracker_freshness_s": _as_dict(extra.get("backend_health")).get("freshness_s"),
        "top_routing_compensation_applied": bool(
            _as_dict(extra.get("top_routing_compensation")).get("applied", False)
        ),
    }
    return quality


def _accept_sample(
    sample: dict[str, Any],
    *,
    run_metrics: dict[str, Any],
    allow_lower_trust: bool,
) -> tuple[dict[str, Any] | None, str | None]:
    """Filter logic. Returns ``(candidate, None)`` on accept, ``(None, reason)`` on reject.

    ``candidate`` is a dict with the fields needed to build a LookupMapPoint.
    """
    extra = _as_dict(sample.get("extra"))
    if str(extra.get("record_kind") or "") != "two_segment_dataset_capture":
        return None, "not_two_segment_dataset_capture"
    if not bool(extra.get("capture_accepted", False)):
        return None, "capture_not_accepted"
    if not bool(extra.get("command_success", True)):
        return None, "command_failed"
    run_trust_mode = str(extra.get("run_trust_mode") or "").lower()
    if not allow_lower_trust and run_trust_mode in {"servo_only", "dry_run", "debug", "lower_trust"}:
        return None, "servo_only_or_lower_trust"
    startup = _as_dict(extra.get("startup_artifact_provenance") or run_metrics.get("startup_artifact_provenance"))
    accepted_startup = bool(startup.get("accepted_all_8_startup"))
    if not allow_lower_trust and not accepted_startup:
        return None, "accepted_all8_startup_missing"
    distal_xyz = _extract_distal_xyz_robot_mm(sample)
    if distal_xyz is None:
        return None, "distal_tip_robot_frame_pose_missing"
    ticks = _extract_all_8_goal_ticks(sample)
    if ticks is None:
        return None, "all_8_goal_ticks_missing_or_incomplete"
    missing = list(extra.get("missing_measured_servo_ids") or [])
    if not allow_lower_trust and missing:
        return None, f"measured_servo_position_missing:{','.join(str(x) for x in missing)}"
    assembly = _as_dict(extra.get("physical_assembly") or run_metrics.get("physical_assembly"))
    candidate = {
        "distal_xyz": distal_xyz,
        "ticks": ticks,
        "assembly": assembly,
        "run_trust_mode": run_trust_mode or "unknown",
        "accepted_all_8_startup": accepted_startup,
        "quality": _build_quality_proxy(sample),
    }
    return candidate, None


# ---------------------------------------------------------------------------
# Downselection
# ---------------------------------------------------------------------------


def voxel_downsample(
    candidates: list[dict[str, Any]],
    *,
    voxel_size_mm: float,
) -> list[dict[str, Any]]:
    """Keep one representative candidate per voxel. Voxels indexed by XYZ floor(/size).

    Representative is chosen by smallest `max_load_proxy_ma` (lower current =
    cleaner sample). When no load proxy is available, the first-encountered
    candidate wins.
    """
    if voxel_size_mm <= 0 or not candidates:
        return list(candidates)
    by_voxel: dict[tuple[int, int, int], dict[str, Any]] = {}

    def _score(cand: dict[str, Any]) -> float:
        load = cand.get("quality", {}).get("max_load_proxy_ma")
        return float("inf") if load is None else float(load)

    for cand in candidates:
        xyz = np.asarray(cand["distal_xyz"], dtype=float)
        voxel = tuple(int(math.floor(v / float(voxel_size_mm))) for v in xyz.tolist())  # type: ignore[arg-type]
        prev = by_voxel.get(voxel)
        if prev is None or _score(cand) < _score(prev):
            by_voxel[voxel] = cand
    return list(by_voxel.values())


def farthest_point_sampling(
    candidates: list[dict[str, Any]],
    *,
    max_points: int,
    rng_seed: int = 0,
) -> list[dict[str, Any]]:
    """Greedy farthest-point downsample to roughly cover the workspace."""
    if max_points <= 0 or not candidates or len(candidates) <= max_points:
        return list(candidates)
    rng = np.random.default_rng(int(rng_seed))
    positions = np.asarray([cand["distal_xyz"] for cand in candidates], dtype=float)
    n = positions.shape[0]
    selected_indices: list[int] = [int(rng.integers(n))]
    distances = np.linalg.norm(positions - positions[selected_indices[0]], axis=1)
    while len(selected_indices) < int(max_points):
        next_idx = int(np.argmax(distances))
        if distances[next_idx] <= 0.0:
            break
        selected_indices.append(next_idx)
        new_d = np.linalg.norm(positions - positions[next_idx], axis=1)
        distances = np.minimum(distances, new_d)
    return [candidates[i] for i in selected_indices]


# ---------------------------------------------------------------------------
# Lookup queries (used by both build-side preview and runtime controller)
# ---------------------------------------------------------------------------


def nearest_lookup(
    target_xyz_robot_mm: np.ndarray | list[float],
    *,
    map_points: list[LookupMapPoint] | list[dict[str, Any]],
    knn: int = 1,
    interpolation: str = "nearest",
) -> dict[str, Any]:
    """Return the nearest (or KNN/IDW-interpolated) command for a target.

    ``map_points`` may be either a list of ``LookupMapPoint`` instances or
    dicts in the on-disk shape (``map_points`` from the map JSON). Returns a
    decision dict with ``selected_point``, ``nearest_distance_mm``,
    ``all_8_goal_ticks``, ``interpolation_mode``, ``knn_used``, and a
    ``source_sample_ids`` audit list.
    """
    if not map_points:
        raise ValueError("Cannot run nearest lookup on an empty map.")
    if interpolation not in SUPPORTED_INTERPOLATION_MODES:
        raise ValueError(f"Unsupported interpolation {interpolation!r}.")
    target = np.asarray(target_xyz_robot_mm, dtype=float)
    if target.shape != (3,):
        raise ValueError(f"target_xyz_robot_mm must be a 3-vector, got shape {target.shape}.")

    def _as_dict_point(point: Any) -> dict[str, Any]:
        if hasattr(point, "to_dict"):
            return point.to_dict()
        if isinstance(point, dict):
            return point
        raise TypeError(f"Unexpected map point type {type(point)!r}")

    dict_points = [_as_dict_point(p) for p in map_points]
    positions = np.asarray(
        [[float(v) for v in p["distal_xyz_robot_mm"]] for p in dict_points], dtype=float
    )
    distances = np.linalg.norm(positions - target, axis=1)
    order = np.argsort(distances)
    knn_clamped = max(1, min(int(knn), len(dict_points)))
    selected_indices = order[:knn_clamped].tolist()
    nearest_index = int(selected_indices[0])
    nearest_point = dict_points[nearest_index]
    nearest_distance = float(distances[nearest_index])
    selected_xyz = [float(v) for v in nearest_point["distal_xyz_robot_mm"]]
    all_8 = {str(k): int(v) for k, v in nearest_point["all_8_goal_ticks"].items()}
    source_ids = [
        f"{dict_points[i].get('source_run_id', 'unknown')}#{int(dict_points[i].get('source_sample_index', -1))}"
        for i in selected_indices
    ]
    if interpolation == "inverse_distance" and knn_clamped > 1:
        # Inverse-distance weighted average of goal ticks across the knn nearest.
        weights = []
        ticks_per_servo: dict[str, list[float]] = {}
        for idx in selected_indices:
            d = float(distances[idx])
            w = 1.0 / max(d, 1e-6)
            weights.append(w)
            for sid, tick in dict_points[idx]["all_8_goal_ticks"].items():
                ticks_per_servo.setdefault(str(sid), []).append(float(tick))
        weights_arr = np.asarray(weights, dtype=float)
        weights_norm = weights_arr / float(weights_arr.sum())
        interpolated = {
            sid: int(round(float(np.dot(np.asarray(values, dtype=float), weights_norm))))
            for sid, values in ticks_per_servo.items()
        }
        all_8 = interpolated
        selected_xyz = [
            float(np.dot(positions[selected_indices, axis], weights_norm)) for axis in range(3)
        ]
        nearest_point = {**nearest_point, "interpolated": True}
    return {
        "target_xyz_robot_mm": [float(v) for v in target.tolist()],
        "selected_point_xyz_robot_mm": [float(v) for v in selected_xyz],
        "nearest_distance_mm": nearest_distance,
        "selected_map_index": int(nearest_point.get("map_index", nearest_index)),
        "all_8_goal_ticks": all_8,
        "interpolation_mode": str(interpolation),
        "knn_used": int(knn_clamped),
        "source_sample_ids": source_ids,
    }


def workspace_bounds_mm(
    map_points: list[LookupMapPoint] | list[dict[str, Any]],
) -> dict[str, Any]:
    """Bounding box + radius statistics for the mapped workspace."""
    if not map_points:
        return {"min_xyz_mm": None, "max_xyz_mm": None, "centroid_xyz_mm": None, "max_radius_mm": None}
    positions = np.asarray(
        [
            [float(v) for v in (p.distal_xyz_robot_mm if hasattr(p, "distal_xyz_robot_mm") else p["distal_xyz_robot_mm"])]
            for p in map_points
        ],
        dtype=float,
    )
    centroid = positions.mean(axis=0)
    radii = np.linalg.norm(positions - centroid, axis=1)
    return {
        "min_xyz_mm": [float(v) for v in positions.min(axis=0).tolist()],
        "max_xyz_mm": [float(v) for v in positions.max(axis=0).tolist()],
        "centroid_xyz_mm": [float(v) for v in centroid.tolist()],
        "max_radius_mm": float(radii.max()),
    }


# ---------------------------------------------------------------------------
# Map build orchestration
# ---------------------------------------------------------------------------


def _resolve_assembly(candidates: list[dict[str, Any]]) -> dict[str, Any]:
    """Take the assembly block from the first candidate; verify all match.

    The map is locked to one bottom/top assignment; runs that disagree are
    not valid as a single map.
    """
    if not candidates:
        return {}
    first = dict(candidates[0].get("assembly") or {})
    bottom = str(first.get("bottom_segment_key") or "")
    top = str(first.get("top_segment_key") or "")
    for cand in candidates[1:]:
        assembly = dict(cand.get("assembly") or {})
        if str(assembly.get("bottom_segment_key") or "") != bottom or str(assembly.get("top_segment_key") or "") != top:
            return {
                "bottom_segment_key": bottom,
                "top_segment_key": top,
                "mixed_assemblies_detected": True,
            }
    return {
        "bottom_segment_key": bottom,
        "top_segment_key": top,
        "bottom_servo_ids": [int(v) for v in list(first.get("bottom_servo_ids") or [])],
        "top_servo_ids": [int(v) for v in list(first.get("top_servo_ids") or [])],
        "mixed_assemblies_detected": False,
    }


def build_workspace_lookup_map(
    run_dirs: list[Path],
    *,
    config: LookupMapBuildConfig | None = None,
) -> LookupMapBuildResult:
    """Build a lookup map artifact from one or more dataset runs."""
    config = config or LookupMapBuildConfig()
    candidates: list[dict[str, Any]] = []
    rejected: list[RejectedMapCandidate] = []
    run_provenance: list[dict[str, Any]] = []
    for run_dir in run_dirs:
        run_dir = Path(run_dir)
        samples_path = run_dir / "samples.jsonl"
        summary_payload = _read_json(run_dir / "summary.json")
        run_metrics = _as_dict(summary_payload.get("experiment_metrics"))
        run_id = str(summary_payload.get("run_id") or run_dir.name)
        if not samples_path.exists():
            rejected.append(
                RejectedMapCandidate(source_run_id=run_id, source_sample_index=-1, reason="samples_jsonl_missing")
            )
            continue
        accepted_in_run = 0
        rejected_in_run = 0
        for sample_index, sample in enumerate(_iter_jsonl(samples_path)):
            candidate, reason = _accept_sample(
                sample, run_metrics=run_metrics, allow_lower_trust=bool(config.allow_lower_trust)
            )
            if candidate is None:
                rejected.append(
                    RejectedMapCandidate(
                        source_run_id=run_id,
                        source_sample_index=int(sample_index),
                        reason=str(reason or "unknown"),
                    )
                )
                rejected_in_run += 1
                continue
            candidate["source_run_id"] = run_id
            candidate["source_run_dir"] = str(run_dir)
            candidate["source_sample_index"] = int(sample_index)
            candidates.append(candidate)
            accepted_in_run += 1
        run_provenance.append(
            {
                "run_id": run_id,
                "run_dir": str(run_dir),
                "scanned_samples": accepted_in_run + rejected_in_run,
                "accepted_samples": accepted_in_run,
                "rejected_samples": rejected_in_run,
            }
        )

    pre_downsample_count = len(candidates)
    if config.voxel_size_mm and float(config.voxel_size_mm) > 0:
        candidates = voxel_downsample(candidates, voxel_size_mm=float(config.voxel_size_mm))
    if config.max_points is not None and int(config.max_points) > 0:
        candidates = farthest_point_sampling(candidates, max_points=int(config.max_points))

    assembly = _resolve_assembly(candidates)
    map_points = [
        LookupMapPoint(
            map_index=int(index),
            distal_xyz_robot_mm=[float(v) for v in cand["distal_xyz"].tolist()],
            all_8_goal_ticks={str(k): int(v) for k, v in cand["ticks"].items()},
            commanded_servo_ids=sorted(int(k) for k in cand["ticks"].keys()),
            bottom_top_assignment=dict(cand.get("assembly") or {}),
            source_run_id=str(cand.get("source_run_id") or ""),
            source_sample_index=int(cand.get("source_sample_index", -1)),
            run_trust_mode=str(cand.get("run_trust_mode") or "unknown"),
            accepted_all_8_startup=bool(cand.get("accepted_all_8_startup", False)),
            quality_proxy=dict(cand.get("quality") or {}),
        )
        for index, cand in enumerate(candidates)
    ]

    bounds = workspace_bounds_mm(map_points)
    metadata: dict[str, Any] = {
        "schema_version": MAP_SCHEMA_VERSION,
        "label_mode": config.label_mode,
        "interpolation_supported": list(SUPPORTED_INTERPOLATION_MODES),
        "interpolation_default": config.interpolation,
        "knn": int(config.knn),
        "voxel_size_mm": float(config.voxel_size_mm) if config.voxel_size_mm else None,
        "max_points_requested": int(config.max_points) if config.max_points else None,
        "allow_lower_trust": bool(config.allow_lower_trust),
        "demo_only_artifact": True,
        "not_model_training_valid": True,
        "not_thesis_repeatability_valid": True,
        "not_closed_loop_validated": True,
        "pre_downsample_candidate_count": int(pre_downsample_count),
        "map_point_count": int(len(map_points)),
        "rejected_count": int(len(rejected)),
        "rejection_reason_counts": _count_reasons(rejected),
        "run_provenance": run_provenance,
        "bottom_top_assignment": assembly,
        "commanded_servo_ids": (map_points[0].commanded_servo_ids if map_points else COMMANDED_SERVO_IDS_DEFAULT),
        "workspace_bounds_mm": bounds,
        "command_units": "ticks (DYNAMIXEL goal_position register)",
        "label_units": "millimetres, robot base frame",
    }
    git_commit = _try_resolve_git_commit()
    if git_commit:
        metadata["git_commit"] = git_commit

    output_dir = Path(config.output_dir or _default_output_dir(run_dirs))
    output_dir.mkdir(parents=True, exist_ok=True)
    artifact_paths = _write_artifacts(output_dir, map_points, metadata, rejected)

    return LookupMapBuildResult(
        output_dir=output_dir,
        map_points=map_points,
        rejected=rejected,
        metadata=metadata,
        artifact_paths=artifact_paths,
    )


def _count_reasons(rejected: list[RejectedMapCandidate]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for item in rejected:
        counts[item.reason] = counts.get(item.reason, 0) + 1
    return counts


def _try_resolve_git_commit() -> str | None:
    try:
        import subprocess

        result = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, timeout=2.0
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except Exception:
        return None
    return None


def _default_output_dir(run_dirs: list[Path]) -> Path:
    if not run_dirs:
        return Path("data/experiments/two_segment_workspace_lookup_maps/unspecified")
    root = Path(run_dirs[0]).resolve()
    # Step up from the run dir into a sibling "two_segment_workspace_lookup_maps/<run>" folder.
    parent = root.parent.parent if root.parent.name == "two_segment_collect_pose_command_dataset" else root.parent
    from datetime import datetime, timezone

    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    return parent / "two_segment_workspace_lookup_maps" / f"{stamp}_workspace_lookup_map"


def _write_artifacts(
    output_dir: Path,
    map_points: list[LookupMapPoint],
    metadata: dict[str, Any],
    rejected: list[RejectedMapCandidate],
) -> dict[str, Path]:
    map_path = output_dir / "two_segment_workspace_lookup_map.json"
    points_csv_path = output_dir / "two_segment_workspace_lookup_points.csv"
    summary_path = output_dir / "two_segment_workspace_lookup_summary.txt"
    quality_path = output_dir / "two_segment_workspace_lookup_quality.json"

    map_payload = {
        "schema_version": MAP_SCHEMA_VERSION,
        "metadata": metadata,
        "map_points": [point.to_dict() for point in map_points],
        "rejected_candidates": [asdict(item) for item in rejected],
    }
    map_path.write_text(json.dumps(map_payload, indent=2, sort_keys=False), encoding="utf-8")
    _write_points_csv(points_csv_path, map_points)
    _write_summary_text(summary_path, metadata)
    quality_path.write_text(
        json.dumps(
            {
                "schema_version": "two_segment_workspace_lookup_quality_v1",
                "demo_only_artifact": True,
                "rejection_reason_counts": metadata.get("rejection_reason_counts", {}),
                "workspace_bounds_mm": metadata.get("workspace_bounds_mm"),
                "map_point_count": int(len(map_points)),
                "pre_downsample_candidate_count": int(metadata.get("pre_downsample_candidate_count", 0)),
                "voxel_size_mm": metadata.get("voxel_size_mm"),
                "max_points_requested": metadata.get("max_points_requested"),
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    paths: dict[str, Path] = {
        "map_json": map_path,
        "points_csv": points_csv_path,
        "summary_text": summary_path,
        "quality_json": quality_path,
    }

    # Figures (best-effort; if matplotlib is missing, skip silently — the map
    # itself remains valid and the runtime controller does not require figures).
    try:
        from continuum_robot.experiments.plotting import create_figure, save_figure, style_axes

        fig_xy_path = output_dir / "workspace_lookup_xy_report.png"
        _write_xy_figure(fig_xy_path, map_points, metadata, create_figure, save_figure, style_axes)
        paths["xy_figure"] = fig_xy_path

        fig_density_path = output_dir / "workspace_lookup_source_density_report.png"
        _write_source_density_figure(
            fig_density_path, map_points, metadata, create_figure, save_figure, style_axes
        )
        paths["density_figure"] = fig_density_path

        fig_cmd_path = output_dir / "workspace_lookup_command_coverage_report.png"
        _write_command_coverage_figure(
            fig_cmd_path, map_points, metadata, create_figure, save_figure, style_axes
        )
        paths["command_coverage_figure"] = fig_cmd_path
    except Exception:
        # Figures are nice-to-have; never block the map build on plotting deps.
        pass

    return paths


def _write_points_csv(path: Path, map_points: list[LookupMapPoint]) -> None:
    import csv

    fieldnames = [
        "map_index",
        "x_mm",
        "y_mm",
        "z_mm",
        "source_run_id",
        "source_sample_index",
        "run_trust_mode",
        "accepted_all_8_startup",
        "max_load_proxy_ma",
    ] + [f"goal_tick_servo_{i}" for i in COMMANDED_SERVO_IDS_DEFAULT]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for point in map_points:
            row = {
                "map_index": int(point.map_index),
                "x_mm": float(point.distal_xyz_robot_mm[0]),
                "y_mm": float(point.distal_xyz_robot_mm[1]),
                "z_mm": float(point.distal_xyz_robot_mm[2]),
                "source_run_id": point.source_run_id,
                "source_sample_index": int(point.source_sample_index),
                "run_trust_mode": point.run_trust_mode,
                "accepted_all_8_startup": bool(point.accepted_all_8_startup),
                "max_load_proxy_ma": (point.quality_proxy or {}).get("max_load_proxy_ma"),
            }
            for sid in COMMANDED_SERVO_IDS_DEFAULT:
                row[f"goal_tick_servo_{sid}"] = point.all_8_goal_ticks.get(str(sid))
            writer.writerow(row)


def _write_summary_text(path: Path, metadata: dict[str, Any]) -> None:
    bounds = metadata.get("workspace_bounds_mm") or {}
    lines = [
        "Two-Segment Workspace Lookup Map (DEMO ONLY)",
        f"schema_version: {metadata['schema_version']}",
        f"demo_only_artifact: {metadata['demo_only_artifact']}",
        f"not_closed_loop_validated: {metadata['not_closed_loop_validated']}",
        f"label_mode: {metadata['label_mode']}",
        f"map_point_count: {metadata['map_point_count']}",
        f"pre_downsample_candidate_count: {metadata['pre_downsample_candidate_count']}",
        f"rejected_count: {metadata['rejected_count']}",
        f"voxel_size_mm: {metadata['voxel_size_mm']}",
        f"max_points_requested: {metadata['max_points_requested']}",
        f"allow_lower_trust: {metadata['allow_lower_trust']}",
        "",
        "Bottom/Top Assembly:",
        f"  bottom_segment_key: {metadata['bottom_top_assignment'].get('bottom_segment_key')}",
        f"  top_segment_key:    {metadata['bottom_top_assignment'].get('top_segment_key')}",
        f"  mixed_assemblies_detected: {metadata['bottom_top_assignment'].get('mixed_assemblies_detected', False)}",
        "",
        "Workspace bounds (robot base frame, mm):",
        f"  min: {bounds.get('min_xyz_mm')}",
        f"  max: {bounds.get('max_xyz_mm')}",
        f"  centroid: {bounds.get('centroid_xyz_mm')}",
        f"  max_radius_from_centroid_mm: {bounds.get('max_radius_mm')}",
        "",
        "Rejection breakdown:",
    ]
    for reason, count in sorted(metadata.get("rejection_reason_counts", {}).items()):
        lines.append(f"  {reason}: {count}")
    lines.extend(
        [
            "",
            "This artifact is for the two-segment penprobe lookup demo. It is",
            "NOT a closed-loop controller, NOT a calibrated workspace, and NOT",
            "valid for thesis or model-training evaluation.",
        ]
    )
    path.write_text("\n".join(lines).strip() + "\n", encoding="utf-8")


def _write_xy_figure(path: Path, map_points, metadata, create_figure, save_figure, style_axes) -> None:
    if not map_points:
        fig, ax = create_figure(size="square")
        ax.text(0.5, 0.5, "No map points.", ha="center", va="center", transform=ax.transAxes)
        style_axes(ax, title="Workspace Lookup XY (DEMO)", xlabel="X (mm)", ylabel="Y (mm)")
        save_figure(fig, path)
        return
    fig, ax = create_figure(size="square")
    xs = [float(p.distal_xyz_robot_mm[0]) for p in map_points]
    ys = [float(p.distal_xyz_robot_mm[1]) for p in map_points]
    ax.scatter(xs, ys, s=14)
    style_axes(ax, title="Workspace Lookup XY (DEMO ONLY)", xlabel="X (mm)", ylabel="Y (mm)")
    ax.set_aspect("equal", adjustable="datalim")
    save_figure(fig, path)


def _write_source_density_figure(path: Path, map_points, metadata, create_figure, save_figure, style_axes) -> None:
    fig, ax = create_figure(size="wide")
    if not map_points:
        ax.text(0.5, 0.5, "No map points.", ha="center", va="center", transform=ax.transAxes)
        style_axes(ax, title="Source Sample Density (DEMO)", xlabel="Run", ylabel="Map Points")
        save_figure(fig, path)
        return
    counts: dict[str, int] = {}
    for point in map_points:
        counts[point.source_run_id] = counts.get(point.source_run_id, 0) + 1
    keys = sorted(counts)
    ax.bar(range(len(keys)), [counts[k] for k in keys])
    ax.set_xticks(range(len(keys)))
    ax.set_xticklabels(keys, rotation=45, ha="right")
    style_axes(ax, title="Source Run Contribution (DEMO ONLY)", xlabel="Source run", ylabel="Map points")
    save_figure(fig, path)


def _write_command_coverage_figure(path: Path, map_points, metadata, create_figure, save_figure, style_axes) -> None:
    fig, ax = create_figure(size="wide")
    if not map_points:
        ax.text(0.5, 0.5, "No map points.", ha="center", va="center", transform=ax.transAxes)
        style_axes(ax, title="Command Coverage (DEMO)", xlabel="Servo ID", ylabel="Tick range")
        save_figure(fig, path)
        return
    servo_ids = COMMANDED_SERVO_IDS_DEFAULT
    mins = []
    maxs = []
    means = []
    for sid in servo_ids:
        values = [int(p.all_8_goal_ticks.get(str(sid), 0)) for p in map_points]
        mins.append(min(values))
        maxs.append(max(values))
        means.append(sum(values) / len(values))
    width = 0.7
    x = np.arange(len(servo_ids))
    ax.bar(x, [maxs[i] - mins[i] for i in range(len(servo_ids))], bottom=mins, width=width, alpha=0.4, label="min..max")
    ax.scatter(x, means, label="mean", marker="o")
    ax.set_xticks(x)
    ax.set_xticklabels([str(s) for s in servo_ids])
    style_axes(
        ax,
        title="Goal-Tick Coverage Across Map Points (DEMO ONLY)",
        xlabel="Servo ID",
        ylabel="Goal tick",
    )
    ax.legend(loc="upper right", fontsize=8)
    save_figure(fig, path)


# ---------------------------------------------------------------------------
# Loader for runtime use
# ---------------------------------------------------------------------------


def load_workspace_lookup_map(path: Path) -> dict[str, Any]:
    """Load a previously written map artifact and return the parsed payload."""
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Workspace lookup map at {path} is not a JSON object.")
    if payload.get("schema_version") != MAP_SCHEMA_VERSION:
        raise ValueError(
            f"Workspace lookup map schema mismatch: got {payload.get('schema_version')!r}, "
            f"expected {MAP_SCHEMA_VERSION!r}."
        )
    return payload


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _resolve_run_dirs(
    *, runs: list[str] | None, latest: bool, project_root: Path
) -> list[Path]:
    if latest:
        from continuum_robot.data.export_run_bundle import find_latest_run

        return [
            find_latest_run(
                project_root=project_root,
                experiment_name="two_segment_collect_pose_command_dataset",
            )
        ]
    if not runs:
        raise SystemExit("Pass at least one --run-dir or use --latest.")
    return [Path(value).resolve() for value in runs]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Build a DEMO-ONLY two-segment workspace lookup map from collected dataset runs.",
    )
    parser.add_argument(
        "--run-dir", action="append", default=[], help="One or more two-segment dataset run directories."
    )
    parser.add_argument("--latest", action="store_true", help="Use the latest two-segment dataset run.")
    parser.add_argument("--project-root", default=".", help="Project root.")
    parser.add_argument("--label", default="distal_xyz", choices=list(SUPPORTED_LABEL_MODES))
    parser.add_argument("--allow-lower-trust", action="store_true")
    parser.add_argument("--max-points", type=int, default=None)
    parser.add_argument("--voxel-size-mm", type=float, default=6.0)
    parser.add_argument("--knn", type=int, default=1)
    parser.add_argument(
        "--interpolation", default="nearest", choices=list(SUPPORTED_INTERPOLATION_MODES)
    )
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--figure-quality", default="production", choices=["low", "medium", "production"])
    args = parser.parse_args(argv)
    project_root = Path(args.project_root).resolve()
    run_dirs = _resolve_run_dirs(runs=args.run_dir, latest=bool(args.latest), project_root=project_root)
    config = LookupMapBuildConfig(
        label_mode=str(args.label),
        allow_lower_trust=bool(args.allow_lower_trust),
        max_points=args.max_points,
        voxel_size_mm=float(args.voxel_size_mm) if args.voxel_size_mm is not None else None,
        knn=int(args.knn),
        interpolation=str(args.interpolation),
        output_dir=Path(args.output_dir).resolve() if args.output_dir else None,
        figure_quality=str(args.figure_quality),
    )
    result = build_workspace_lookup_map(run_dirs, config=config)
    print(f"Workspace lookup map: {result.output_dir}")
    print(f"  map points : {result.accepted_count}")
    print(f"  rejected   : {result.rejected_count}")
    print(f"  bounds (mm): {result.metadata['workspace_bounds_mm']}")
    print("  artifact   :")
    for key, path in sorted(result.artifact_paths.items()):
        print(f"    {key}: {path}")
    if result.accepted_count == 0:
        print(
            "WARNING: no accepted points — pass --allow-lower-trust if testing on servo-only/dry-run data.",
            file=sys.stderr,
        )
        return 2
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())

"""Shared sample construction helpers for experiment runs."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

import numpy as np

from continuum_robot.experiments.framework import ExperimentSession
from continuum_robot.experiments.schemas import ExperimentTimeseriesSample
from continuum_robot.tracking.transforms import rotmat_to_quat_wxyz


def sample_from_tracking_snapshot(
    session: ExperimentSession,
    *,
    snapshot,
    phase: str,
    step_index: int,
    sample_index: int,
    commanded_cable_deltas_cm: list[float],
    commanded_motor_values: dict[str, int | float | None],
    status_flags: list[str] | None = None,
    extra: dict[str, Any] | None = None,
    cycle_index: int | None = None,
    target_index: int | None = None,
    revisit_index: int | None = None,
    approach_index: int | None = None,
    override_tracker_position_mm: list[float] | None = None,
    override_tracker_quaternion_wxyz: list[float] | None = None,
    override_robot_position_mm: list[float] | None = None,
    tracker_tool_id: str = "0A",
) -> ExperimentTimeseriesSample:
    """Build one canonical sample from a tracking snapshot."""
    tool_ids_seen = list(snapshot.normalized_live_tool_ids)
    transform_validity = {
        tool_id: tool.tracking_state
        for tool_id, tool in snapshot.tools.items()
    }
    pose_in_tracker_frame: dict[str, Any] = {}
    for tool_id, tool in snapshot.tools.items():
        translation = list(tool.translation_mm) if tool.translation_mm is not None else None
        quaternion = list(tool.quaternion_wxyz) if tool.quaternion_wxyz is not None else None
        tangent_xyz = None
        if tool_id == tracker_tool_id and override_tracker_position_mm is not None:
            translation = [float(value) for value in override_tracker_position_mm]
        if tool_id == tracker_tool_id and override_tracker_quaternion_wxyz is not None:
            quaternion = [float(value) for value in override_tracker_quaternion_wxyz]
        if isinstance(quaternion, list) and len(quaternion) == 4:
            tangent_xyz = _quat_to_tangent_xyz(quaternion)
        pose_in_tracker_frame[tool_id] = {
            "tracking_state": tool.tracking_state,
            "translation_mm": translation,
            "quaternion_wxyz": quaternion,
            "tangent_xyz": tangent_xyz,
            "frame_number": tool.frame_number,
        }
    pose_in_robot_frame: dict[str, Any] = {}
    if snapshot.T_robot_tip is not None:
        matrix = np.asarray(snapshot.T_robot_tip, dtype=float)
        translation_mm = [
            float(matrix[0][3]),
            float(matrix[1][3]),
            float(matrix[2][3]),
        ]
        if override_robot_position_mm is not None:
            translation_mm = [float(value) for value in override_robot_position_mm]
        pose_in_robot_frame["tip"] = {
            "matrix": snapshot.T_robot_tip,
            "translation_mm": translation_mm,
            "quaternion_wxyz": list(rotmat_to_quat_wxyz(matrix[0:3, 0:3])),
            "tangent_xyz": [float(matrix[0][2]), float(matrix[1][2]), float(matrix[2][2])],
        }
    elif override_robot_position_mm is not None:
        pose_in_robot_frame["tip"] = {
            "translation_mm": [float(value) for value in override_robot_position_mm],
        }
    flags = list(status_flags or [])
    flags.extend(str(flag) for flag in snapshot.tracker_faults)
    flags.extend(str(flag) for flag in snapshot.pipeline_faults)
    if snapshot.tracker_data_stale:
        flags.append("tracker_data_stale")
    if snapshot.last_error:
        flags.append("tracker_error")
    return ExperimentTimeseriesSample(
        monotonic_time_s=session.elapsed_s(),
        wall_time_utc=datetime.now(timezone.utc).isoformat(),
        phase=phase,
        step_index=int(step_index),
        sample_index=int(sample_index),
        cycle_index=None if cycle_index is None else int(cycle_index),
        target_index=None if target_index is None else int(target_index),
        revisit_index=None if revisit_index is None else int(revisit_index),
        approach_index=None if approach_index is None else int(approach_index),
        commanded_motor_values={str(key): value for key, value in commanded_motor_values.items()},
        commanded_cable_deltas_cm=[float(value) for value in commanded_cable_deltas_cm],
        tracker_frame_id=snapshot.last_frame_number,
        tool_ids_seen=tool_ids_seen,
        transform_validity=transform_validity,
        pose_in_tracker_frame=pose_in_tracker_frame,
        pose_in_robot_frame=pose_in_robot_frame,
        freshness_s=snapshot.tracker_data_age_s,
        latency_s=snapshot.tracker_data_age_s,
        status_flags=sorted(set(flags)),
        backend_health={
            "canonical_state": snapshot.canonical_state,
            "selected_backend_name": snapshot.selected_backend_name,
            "backend_identity": snapshot.backend_identity,
            "effective_frame_rate_hz": snapshot.effective_frame_rate_hz,
        },
        extra=dict(extra or {}),
    )


def _quat_to_tangent_xyz(quaternion_wxyz: list[float]) -> list[float] | None:
    if len(quaternion_wxyz) != 4:
        return None
    quat = np.asarray(quaternion_wxyz, dtype=float)
    norm = float(np.linalg.norm(quat))
    if norm <= 1e-12:
        return None
    w, x, y, z = quat / norm
    tangent = np.asarray(
        [
            2.0 * (x * z + y * w),
            2.0 * (y * z - x * w),
            1.0 - 2.0 * (x * x + y * y),
        ],
        dtype=float,
    )
    return [float(value) for value in tangent]

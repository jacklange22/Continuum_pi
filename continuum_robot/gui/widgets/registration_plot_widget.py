"""Registration preview widget with display-only progressive alignment."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping, Sequence

import numpy as np

from continuum_robot.gui.widgets.lightweight_scene_3d_widget import (
    LightweightScene3DWidget,
    SceneAnnotation3D,
    SceneAxes3D,
    SceneLine3D,
    SceneModel3D,
    ScenePoint3D,
)
from continuum_robot.registration.rigid_solver import RigidRegistrationSolver


@dataclass
class RegistrationPreviewAlignment:
    """Display-only preview alignment result."""

    mode: str = "no_preview"
    message: str = "Complete one point to start the visualization-only preview alignment."
    underconstrained: bool = False
    transform: np.ndarray | None = None
    aligned_points_by_label: dict[str, tuple[float, float, float]] = field(default_factory=dict)
    residual_vectors_by_label: dict[str, tuple[float, float, float]] = field(default_factory=dict)
    residual_norms_by_label: dict[str, float] = field(default_factory=dict)
    fitted_labels: list[str] = field(default_factory=list)


class RegistrationPlotWidget(LightweightScene3DWidget):
    """Draw truth landmarks, captured points, and preview-aligned overlays."""

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._preview = RegistrationPreviewAlignment()
        self.setMinimumHeight(220)

    def set_data(
        self,
        nominal: Mapping[str, Sequence[float]],
        captured: Mapping[str, Sequence[Sequence[float]]],
        *,
        averaged_points: Mapping[str, Sequence[float]] | None = None,
        completed_labels: Sequence[str] | None = None,
        selected_label: str | None = None,
        current_point: Sequence[float] | None = None,
        solved_transform: Sequence[Sequence[float]] | None = None,
        solved_residuals_by_label: Mapping[str, Sequence[float]] | None = None,
    ) -> None:
        truth_points = {
            str(label): _as_xyz(point)
            for label, point in nominal.items()
            if point is not None and len(point) >= 3
        }
        raw_samples = {
            str(label): [_as_xyz(sample) for sample in samples if sample is not None and len(sample) >= 3]
            for label, samples in captured.items()
        }
        ordered_labels = [str(label) for label in truth_points]
        completed = [str(label) for label in (completed_labels or [])]
        median_points = compute_median_capture_points(raw_samples)
        capture_centers = (
            {
                str(label): _as_xyz(point)
                for label, point in (averaged_points or {}).items()
                if point is not None and len(point) >= 3
            }
            if solved_transform is not None
            else median_points
        )
        solved_transform_matrix = _as_transform(solved_transform)
        self._preview = compute_registration_preview_alignment(
            truth_points_by_label=truth_points,
            captured_points_by_label=capture_centers,
            ordered_labels=completed or ordered_labels,
            solved_transform=solved_transform_matrix,
            solved_residuals_by_label=solved_residuals_by_label,
        )
        scene = build_registration_preview_scene(
            truth_points_by_label=truth_points,
            raw_samples_by_label=raw_samples,
            capture_points_by_label=capture_centers,
            preview=self._preview,
            ordered_labels=ordered_labels,
            completed_labels=completed,
            selected_label=selected_label,
            current_point=current_point,
        )
        self.set_scene(scene)

    @property
    def preview(self) -> RegistrationPreviewAlignment:
        return self._preview


def compute_median_capture_points(
    raw_samples_by_label: Mapping[str, Sequence[Sequence[float]]],
) -> dict[str, tuple[float, float, float]]:
    medians: dict[str, tuple[float, float, float]] = {}
    for label, samples in raw_samples_by_label.items():
        if not samples:
            continue
        arr = np.asarray(samples, dtype=float)
        if arr.ndim != 2 or arr.shape[1] < 3:
            continue
        median = np.median(arr[:, 0:3], axis=0)
        medians[str(label)] = tuple(float(value) for value in median.tolist())
    return medians


def compute_registration_preview_alignment(
    *,
    truth_points_by_label: Mapping[str, Sequence[float]],
    captured_points_by_label: Mapping[str, Sequence[float]],
    ordered_labels: Sequence[str],
    solved_transform: np.ndarray | None = None,
    solved_residuals_by_label: Mapping[str, Sequence[float]] | None = None,
) -> RegistrationPreviewAlignment:
    truth = {str(label): _as_xyz(point) for label, point in truth_points_by_label.items()}
    captured = {str(label): _as_xyz(point) for label, point in captured_points_by_label.items()}
    eligible_labels = [str(label) for label in ordered_labels if str(label) in truth and str(label) in captured]
    if solved_transform is not None:
        aligned = _apply_transform_to_points(solved_transform, captured)
        residual_vectors = {
            label: (
                tuple(float(value) for value in solved_residuals_by_label[label])
                if solved_residuals_by_label is not None and label in solved_residuals_by_label
                else tuple(float(value) for value in (np.asarray(truth[label]) - np.asarray(aligned[label])).tolist())
            )
            for label in eligible_labels
        }
        residual_norms = {
            label: float(np.linalg.norm(np.asarray(vector, dtype=float)))
            for label, vector in residual_vectors.items()
        }
        return RegistrationPreviewAlignment(
            mode="solved_fit",
            message="Solved overlay. This view uses the actual rigid registration transform and does not change the saved result.",
            underconstrained=False,
            transform=np.asarray(solved_transform, dtype=float),
            aligned_points_by_label={label: aligned[label] for label in eligible_labels},
            residual_vectors_by_label=residual_vectors,
            residual_norms_by_label=residual_norms,
            fitted_labels=list(eligible_labels),
        )
    if not eligible_labels:
        return RegistrationPreviewAlignment()
    if len(eligible_labels) == 1:
        label = eligible_labels[0]
        translation = np.asarray(truth[label], dtype=float) - np.asarray(captured[label], dtype=float)
        transform = np.eye(4, dtype=float)
        transform[0:3, 3] = translation
        aligned = _apply_transform_to_points(transform, captured)
        return RegistrationPreviewAlignment(
            mode="translation_preview",
            message="1-point preview. Display-only translation alignment anchors the completed point to its target.",
            underconstrained=True,
            transform=transform,
            aligned_points_by_label=aligned,
            residual_vectors_by_label={label: (0.0, 0.0, 0.0)},
            residual_norms_by_label={label: 0.0},
            fitted_labels=[label],
        )
    if len(eligible_labels) == 2:
        transform = _solve_two_point_preview_transform(
            captured_a=captured[eligible_labels[0]],
            captured_b=captured[eligible_labels[1]],
            truth_a=truth[eligible_labels[0]],
            truth_b=truth[eligible_labels[1]],
        )
        aligned = _apply_transform_to_points(transform, captured)
        residual_vectors = {
            label: tuple(float(value) for value in (np.asarray(truth[label]) - np.asarray(aligned[label])).tolist())
            for label in eligible_labels
        }
        residual_norms = {
            label: float(np.linalg.norm(np.asarray(vector, dtype=float)))
            for label, vector in residual_vectors.items()
        }
        return RegistrationPreviewAlignment(
            mode="segment_preview",
            message="2-point preview. Display-only segment alignment is underconstrained, so the remaining twist is chosen with a minimal-rotation heuristic.",
            underconstrained=True,
            transform=transform,
            aligned_points_by_label=aligned,
            residual_vectors_by_label=residual_vectors,
            residual_norms_by_label=residual_norms,
            fitted_labels=list(eligible_labels),
        )

    truth_arr = np.asarray([truth[label] for label in eligible_labels], dtype=float)
    captured_arr = np.asarray([captured[label] for label in eligible_labels], dtype=float)
    truth_rank = int(np.linalg.matrix_rank(truth_arr - truth_arr.mean(axis=0)))
    captured_rank = int(np.linalg.matrix_rank(captured_arr - captured_arr.mean(axis=0)))
    if min(truth_rank, captured_rank) < 2:
        return compute_registration_preview_alignment(
            truth_points_by_label=truth,
            captured_points_by_label=captured,
            ordered_labels=eligible_labels[:2],
        )
    solver = RigidRegistrationSolver()
    transform = solver.solve_T_robot_aurora(captured_arr, truth_arr)
    aligned = _apply_transform_to_points(transform, captured)
    residual_vectors = {
        label: tuple(float(value) for value in (np.asarray(truth[label]) - np.asarray(aligned[label])).tolist())
        for label in eligible_labels
    }
    residual_norms = {
        label: float(np.linalg.norm(np.asarray(vector, dtype=float)))
        for label, vector in residual_vectors.items()
    }
    return RegistrationPreviewAlignment(
        mode="rigid_fit_preview",
        message="3+ point preview. Display-only rigid fit is available because the completed points are no longer underconstrained.",
        underconstrained=False,
        transform=transform,
        aligned_points_by_label=aligned,
        residual_vectors_by_label=residual_vectors,
        residual_norms_by_label=residual_norms,
        fitted_labels=list(eligible_labels),
    )


def build_registration_preview_scene(
    *,
    truth_points_by_label: Mapping[str, Vec3Like],
    raw_samples_by_label: Mapping[str, Sequence[Vec3Like]],
    capture_points_by_label: Mapping[str, Vec3Like],
    preview: RegistrationPreviewAlignment,
    ordered_labels: Sequence[str],
    completed_labels: Sequence[str],
    selected_label: str | None,
    current_point: Sequence[float] | None,
) -> SceneModel3D:
    truth = {str(label): _as_xyz(point) for label, point in truth_points_by_label.items()}
    capture_centers = {str(label): _as_xyz(point) for label, point in capture_points_by_label.items()}
    completed = set(str(label) for label in completed_labels)
    points: list[ScenePoint3D] = [ScenePoint3D(key="robot_origin", xyz=(0.0, 0.0, 0.0), color_hex="#cbd5e1", shape="circle", radius_px=3.5)]
    lines: list[SceneLine3D] = []
    annotations: list[SceneAnnotation3D] = []
    axes = [SceneAxes3D(key="robot_frame", origin_xyz=(0.0, 0.0, 0.0), label="Robot")]
    overlay_lines = [preview.message]

    current_status = "Selected point: none"
    if selected_label:
        if selected_label in completed:
            current_status = f"Selected point: {selected_label} | complete"
        elif selected_label in capture_centers:
            current_status = f"Selected point: {selected_label} | in progress"
        else:
            current_status = f"Selected point: {selected_label} | not started"
    overlay_lines.append(current_status)
    overlay_lines.append("Preview only. Solver inputs, solve path, and saved registration outputs are unchanged.")
    overlay_lines.append(
        "Display centers use captured medians before solve and actual solved centroids after solve."
        if preview.mode == "solved_fit"
        else "Display centers use captured medians for preview alignment."
    )
    if preview.underconstrained:
        overlay_lines.append("Underconstrained preview. Use it for trust only, not as a solved registration result.")

    for label in ordered_labels:
        if label not in truth:
            continue
        outline = "#0f766e" if label == selected_label else "#ffffff"
        complete = label in completed or label in preview.fitted_labels
        points.append(
            ScenePoint3D(
                key=f"truth_{label}",
                xyz=truth[label],
                color_hex="#2563eb" if complete else "#cbd5e1",
                label=label,
                radius_px=6.0 if label == selected_label else 5.0,
                outline_hex=outline,
                shape="square",
            )
        )

    if preview.transform is not None:
        transformed_samples = _transform_sample_clouds(preview.transform, raw_samples_by_label)
        for label, samples in transformed_samples.items():
            for index, sample in enumerate(samples[:20]):
                points.append(
                    ScenePoint3D(
                        key=f"sample_{label}_{index}",
                        xyz=sample,
                        color_hex="#fbbf24",
                        radius_px=2.3,
                        shape="circle",
                    )
                )
        for label, point in preview.aligned_points_by_label.items():
            points.append(
                ScenePoint3D(
                    key=f"aligned_{label}",
                    xyz=point,
                    color_hex="#16a34a",
                    label=f"{label} aligned",
                    radius_px=5.0,
                    outline_hex="#ffffff",
                    shape="circle",
                )
            )
            if label in truth:
                lines.append(
                    SceneLine3D(
                        start_xyz=point,
                        end_xyz=truth[label],
                        color_hex="#dc2626",
                        width_px=1.4,
                    )
                )
                residual_mm = preview.residual_norms_by_label.get(label)
                if residual_mm is not None:
                    annotations.append(
                        SceneAnnotation3D(
                            xyz=point,
                            text=f"{label}: {float(residual_mm):.3f} mm",
                            color_hex="#7f1d1d",
                        )
                    )
        if current_point is not None and len(current_point) >= 3:
            current_xyz = _transform_point(preview.transform, _as_xyz(current_point))
            points.append(
                ScenePoint3D(
                    key="current_live_point",
                    xyz=current_xyz,
                    color_hex="#0f766e",
                    label="live",
                    radius_px=4.6,
                    outline_hex="#ffffff",
                    shape="diamond",
                )
            )
    elif current_point is not None and len(current_point) >= 3:
        overlay_lines.append("Live measurement point is hidden until at least one completed point defines a display frame.")

    return SceneModel3D(
        frame_label="robot",
        overlay_lines=tuple(overlay_lines),
        points=tuple(points),
        lines=tuple(lines),
        axes=tuple(axes),
        annotations=tuple(annotations),
    )


def _solve_two_point_preview_transform(
    *,
    captured_a: Sequence[float],
    captured_b: Sequence[float],
    truth_a: Sequence[float],
    truth_b: Sequence[float],
) -> np.ndarray:
    src_a = np.asarray(captured_a, dtype=float)
    src_b = np.asarray(captured_b, dtype=float)
    dst_a = np.asarray(truth_a, dtype=float)
    dst_b = np.asarray(truth_b, dtype=float)
    src_dir = src_b - src_a
    dst_dir = dst_b - dst_a
    if np.linalg.norm(src_dir) <= 1e-9 or np.linalg.norm(dst_dir) <= 1e-9:
        transform = np.eye(4, dtype=float)
        transform[0:3, 3] = dst_a - src_a
        return transform
    rotation = _minimal_rotation_matrix(src_dir, dst_dir)
    transform = np.eye(4, dtype=float)
    transform[0:3, 0:3] = rotation
    transform[0:3, 3] = dst_a - (rotation @ src_a)
    return transform


def _minimal_rotation_matrix(source_vector: Sequence[float], target_vector: Sequence[float]) -> np.ndarray:
    source = np.asarray(source_vector, dtype=float)
    target = np.asarray(target_vector, dtype=float)
    source /= np.linalg.norm(source)
    target /= np.linalg.norm(target)
    cross = np.cross(source, target)
    dot = float(np.clip(np.dot(source, target), -1.0, 1.0))
    norm_cross = float(np.linalg.norm(cross))
    if norm_cross <= 1e-9:
        if dot > 0.0:
            return np.eye(3, dtype=float)
        axis = _orthogonal_axis(source)
        return _axis_angle_rotation(axis, np.pi)
    skew = np.asarray(
        [
            [0.0, -cross[2], cross[1]],
            [cross[2], 0.0, -cross[0]],
            [-cross[1], cross[0], 0.0],
        ],
        dtype=float,
    )
    return np.eye(3, dtype=float) + skew + (skew @ skew) * ((1.0 - dot) / (norm_cross ** 2))


def _orthogonal_axis(vector: np.ndarray) -> np.ndarray:
    candidate = np.asarray([1.0, 0.0, 0.0], dtype=float)
    if abs(float(np.dot(candidate, vector))) > 0.9:
        candidate = np.asarray([0.0, 1.0, 0.0], dtype=float)
    axis = np.cross(vector, candidate)
    axis /= np.linalg.norm(axis)
    return axis


def _axis_angle_rotation(axis: np.ndarray, angle_rad: float) -> np.ndarray:
    x, y, z = (float(value) for value in axis.tolist())
    c = float(np.cos(angle_rad))
    s = float(np.sin(angle_rad))
    one_c = 1.0 - c
    return np.asarray(
        [
            [c + x * x * one_c, x * y * one_c - z * s, x * z * one_c + y * s],
            [y * x * one_c + z * s, c + y * y * one_c, y * z * one_c - x * s],
            [z * x * one_c - y * s, z * y * one_c + x * s, c + z * z * one_c],
        ],
        dtype=float,
    )


def _apply_transform_to_points(
    transform: np.ndarray,
    points_by_label: Mapping[str, Sequence[float]],
) -> dict[str, tuple[float, float, float]]:
    output: dict[str, tuple[float, float, float]] = {}
    for label, point in points_by_label.items():
        output[str(label)] = _transform_point(transform, _as_xyz(point))
    return output


def _transform_sample_clouds(
    transform: np.ndarray,
    raw_samples_by_label: Mapping[str, Sequence[Sequence[float]]],
) -> dict[str, list[tuple[float, float, float]]]:
    return {
        str(label): [_transform_point(transform, _as_xyz(sample)) for sample in samples]
        for label, samples in raw_samples_by_label.items()
        if samples
    }


def _transform_point(transform: np.ndarray, point: Sequence[float]) -> tuple[float, float, float]:
    arr = np.asarray(point, dtype=float)
    mapped = (transform[0:3, 0:3] @ arr) + transform[0:3, 3]
    return tuple(float(value) for value in mapped.tolist())


def _as_transform(transform) -> np.ndarray | None:
    if transform is None:
        return None
    arr = np.asarray(transform, dtype=float)
    if arr.shape != (4, 4):
        return None
    return arr


def _as_xyz(point: Sequence[float]) -> tuple[float, float, float]:
    return (float(point[0]), float(point[1]), float(point[2]))


Vec3Like = Sequence[float]

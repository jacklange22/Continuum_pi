"""Legacy-compatible registration helpers with explicit validation."""

from __future__ import annotations

from dataclasses import dataclass
import csv
from pathlib import Path
import re
from typing import Any

import numpy as np

from continuum_robot.registration.rigid_solver import RigidRegistrationSolver
from continuum_robot.registration.validation import compute_fre_mm
from continuum_robot.tracking.transforms import assert_transform_matrix, quat_wxyz_to_rotmat


_TOOL_ID_RE = re.compile(r"^[0-9A-Za-z]{2}$")


@dataclass
class RegistrationAssetPaths:
    """Protected registration asset file paths."""

    model_points_file: Path
    tip_points_file: Path
    T_sw_2_model_file: Path
    T_sw_2_tip_file: Path
    penprobe_file: Path


@dataclass
class RegistrationAssets:
    """Loaded protected registration assets."""

    paths: RegistrationAssetPaths
    model_truth_in_sw: np.ndarray
    tip_truth_in_sw: np.ndarray
    T_sw_2_model: np.ndarray
    T_sw_2_tip: np.ndarray
    T_measurement_point: np.ndarray
    model_labels: list[str]
    tip_labels: list[str]


@dataclass
class AuroraPoseSample:
    """One Aurora tool pose sample from a CSV or live tracker."""

    tool_id: str
    quaternion_wxyz: tuple[float, float, float, float]
    translation_mm: tuple[float, float, float]
    quality: float | None = None
    source_row: int | None = None
    source_token: str | None = None


@dataclass
class LegacyRegistrationResult:
    """Full legacy-compatible registration solve output."""

    measurement_tool_id: str
    coil_tool_id: str
    repetitions: int
    ordered_labels: list[str]
    group_by_label: dict[str, str]
    truth_points_in_sw_by_label: dict[str, list[float]]
    raw_points_by_label: dict[str, list[list[float]]]
    raw_measurement_tool_samples_by_label: dict[str, list[dict[str, Any]]]
    raw_coil_samples_by_label: dict[str, list[dict[str, Any]]]
    averaged_points_by_label: dict[str, list[float]]
    measured_points_in_aurora: np.ndarray
    model_truth_in_model: np.ndarray
    tip_truth_in_tip: np.ndarray
    T_aurora_2_model: np.ndarray
    T_aurora_2_tip: np.ndarray
    T_tip_2_coil: np.ndarray
    T_coil_tip: np.ndarray
    T_coil_2_aurora_mean: np.ndarray
    averaged_coil_quaternion_wxyz: np.ndarray
    averaged_coil_translation_mm: np.ndarray
    residuals_by_label: dict[str, list[float]]
    validation_metrics: dict[str, Any]


def load_registration_assets(
    paths: RegistrationAssetPaths,
    *,
    measurement_point_transform: list[list[float]] | None = None,
) -> RegistrationAssets:
    """Load and validate legacy registration assets from protected files."""
    model_truth_in_sw = _load_points_3xN(paths.model_points_file)
    tip_truth_in_sw = _load_points_3xN(paths.tip_points_file)
    T_sw_2_model = _load_transform_4x4(paths.T_sw_2_model_file)
    T_sw_2_tip = _load_transform_4x4(paths.T_sw_2_tip_file)

    if measurement_point_transform is not None:
        T_measurement_point = np.asarray(measurement_point_transform, dtype=float)
        assert_transform_matrix(T_measurement_point, "measurement_point_transform")
        _validate_rotation_matrix(T_measurement_point[0:3, 0:3], "measurement_point_transform")
    else:
        penprobe = _load_vector3(paths.penprobe_file)
        T_measurement_point = np.eye(4, dtype=float)
        T_measurement_point[0:3, 3] = penprobe

    model_labels = [f"M{idx + 1:02d}" for idx in range(model_truth_in_sw.shape[1])]
    tip_labels = [f"T{idx + 1:02d}" for idx in range(tip_truth_in_sw.shape[1])]
    return RegistrationAssets(
        paths=paths,
        model_truth_in_sw=model_truth_in_sw,
        tip_truth_in_sw=tip_truth_in_sw,
        T_sw_2_model=T_sw_2_model,
        T_sw_2_tip=T_sw_2_tip,
        T_measurement_point=T_measurement_point,
        model_labels=model_labels,
        tip_labels=tip_labels,
    )


def parse_aurora_csv(path: Path) -> dict[str, list[AuroraPoseSample]]:
    """Parse legacy Aurora CSV rows into ordered pose samples by tool id."""
    if not path.exists():
        raise FileNotFoundError(f"Missing Aurora registration CSV: {path}")

    output: dict[str, list[AuroraPoseSample]] = {}
    parsed_rows = 0
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle)
        for row_index, row in enumerate(reader, start=1):
            row = [item.strip() for item in row if item.strip() != ""]
            if not row:
                continue

            tool_index = _find_tool_index(row)
            if tool_index is None:
                if row_index == 1:
                    # Allow one header row, but only if it is clearly non-numeric.
                    continue
                raise ValueError(f"Could not identify tool id in Aurora CSV row {row_index}: {row}")

            tool_id = row[tool_index]
            numeric_tail = row[tool_index + 1 :]
            if len(numeric_tail) < 7:
                raise ValueError(
                    f"Aurora CSV row {row_index} has too few numeric fields after tool id {tool_id}: {row}"
                )

            try:
                quat = tuple(float(value) for value in numeric_tail[0:4])
                trans = tuple(float(value) for value in numeric_tail[4:7])
                extras = [float(value) for value in numeric_tail[7:]]
            except ValueError as exc:
                raise ValueError(f"Aurora CSV row {row_index} contains non-float numeric fields: {row}") from exc

            # Legacy files often duplicate translation columns before the final error.
            if len(extras) >= 3 and np.allclose(np.asarray(extras[0:3], dtype=float), np.asarray(trans), atol=1e-6):
                extras = extras[3:]

            quality = float(extras[-1]) if extras else None
            sample = AuroraPoseSample(
                tool_id=tool_id,
                quaternion_wxyz=quat,
                translation_mm=trans,
                quality=quality,
                source_row=row_index,
                source_token=",".join(row[0:tool_index]),
            )
            output.setdefault(tool_id, []).append(sample)
            parsed_rows += 1

    if parsed_rows == 0:
        raise ValueError(f"Aurora CSV contains no pose rows: {path}")
    return output


def expand_points_by_repetition(points_3xN: np.ndarray, repetitions: int) -> np.ndarray:
    """Repeat each point contiguously to match legacy registration ordering."""
    points = _as_points_3xN(points_3xN, "points_3xN")
    reps = int(repetitions)
    if reps <= 0:
        raise ValueError("repetitions must be >= 1")

    expanded = np.zeros((3, points.shape[1] * reps), dtype=float)
    for rep_index in range(reps):
        for point_index in range(points.shape[1]):
            expanded[:, point_index * reps + rep_index] = points[:, point_index]
    return expanded


def solve_registration_from_tool_samples(
    *,
    assets: RegistrationAssets,
    measurement_tool_samples: list[AuroraPoseSample],
    coil_tool_samples: list[AuroraPoseSample],
    repetitions: int,
    measurement_tool_id: str,
    coil_tool_id: str,
    solver: RigidRegistrationSolver,
    quaternion_average_method: str = "sign_aligned_mean",
    model_tre_reference_radius_mm: float = 5.0,
    tip_tre_reference_radius_mm: float = 3.0,
) -> LegacyRegistrationResult:
    """Solve the legacy-compatible registration pipeline from ordered tool samples."""
    repetitions = int(repetitions)
    if repetitions <= 0:
        raise ValueError("repetitions must be >= 1")
    if not measurement_tool_samples:
        raise ValueError(f"No measurement-tool samples were provided for {measurement_tool_id}")
    if not coil_tool_samples:
        raise ValueError(f"No coil-tool samples were provided for {coil_tool_id}")

    model_truth_in_model_base = apply_transform_to_points(assets.T_sw_2_model, assets.model_truth_in_sw)
    tip_truth_in_tip_base = apply_transform_to_points(assets.T_sw_2_tip, assets.tip_truth_in_sw)
    model_truth_in_model = expand_points_by_repetition(model_truth_in_model_base, repetitions)
    tip_truth_in_tip = expand_points_by_repetition(tip_truth_in_tip_base, repetitions)

    expected_model_count = model_truth_in_model.shape[1]
    expected_tip_count = tip_truth_in_tip.shape[1]
    expected_total_count = expected_model_count + expected_tip_count

    if len(measurement_tool_samples) != expected_total_count:
        raise ValueError(
            "Measurement-tool sample count does not match truth-point expansion: "
            f"got {len(measurement_tool_samples)}, expected {expected_total_count} "
            f"= ({assets.model_truth_in_sw.shape[1]} + {assets.tip_truth_in_sw.shape[1]}) * {repetitions}"
        )
    if len(coil_tool_samples) != expected_total_count:
        raise ValueError(
            "Coil-tool sample count does not match measurement count: "
            f"got {len(coil_tool_samples)}, expected {expected_total_count}"
        )

    measured_points_in_aurora = measurement_points_from_tool_samples(
        measurement_tool_samples,
        assets.T_measurement_point,
    )

    model_meas_in_aurora = measured_points_in_aurora[:, :expected_model_count]
    tip_meas_in_aurora = measured_points_in_aurora[:, expected_model_count : expected_total_count]

    model_alignment = solver.solve_alignment(model_meas_in_aurora, model_truth_in_model)
    tip_alignment = solver.solve_alignment(tip_meas_in_aurora, tip_truth_in_tip)

    T_coil_2_aurora_mean, averaged_quaternion, averaged_translation, averaging_details = average_tool_samples_as_transform(
        coil_tool_samples,
        method=quaternion_average_method,
    )
    T_tip_2_coil = np.linalg.inv(T_coil_2_aurora_mean) @ np.linalg.inv(tip_alignment["transform"])
    T_coil_tip = np.linalg.inv(T_tip_2_coil)

    ordered_labels = [*assets.model_labels, *assets.tip_labels]
    group_by_label = {label: "model" for label in assets.model_labels} | {
        label: "tip" for label in assets.tip_labels
    }
    truth_points_in_sw_by_label = {
        label: assets.model_truth_in_sw[:, idx].astype(float).tolist()
        for idx, label in enumerate(assets.model_labels)
    }
    truth_points_in_sw_by_label.update(
        {
            label: assets.tip_truth_in_sw[:, idx].astype(float).tolist()
            for idx, label in enumerate(assets.tip_labels)
        }
    )

    raw_points_by_label, averaged_points_by_label = _group_points_by_label(
        ordered_labels=ordered_labels,
        group_sizes={"model": assets.model_truth_in_sw.shape[1], "tip": assets.tip_truth_in_sw.shape[1]},
        repetitions=repetitions,
        measured_points_in_aurora=measured_points_in_aurora,
    )
    raw_measurement_tool_samples_by_label = _group_tool_samples_by_label(
        ordered_labels=ordered_labels,
        samples=measurement_tool_samples,
        repetitions=repetitions,
    )
    raw_coil_samples_by_label = _group_tool_samples_by_label(
        ordered_labels=ordered_labels,
        samples=coil_tool_samples,
        repetitions=repetitions,
    )

    residuals_by_label = _mean_residuals_by_label(
        model_labels=assets.model_labels,
        tip_labels=assets.tip_labels,
        repetitions=repetitions,
        model_residuals=model_alignment["residuals"],
        tip_residuals=tip_alignment["residuals"],
    )

    combined_residuals = [*model_alignment["residuals"].T.tolist(), *tip_alignment["residuals"].T.tolist()]
    overall_fre_mm = float(compute_fre_mm(combined_residuals))
    validation_metrics = {
        "model_fre_mm": float(model_alignment["rmse_mm"]),
        "tip_fre_mm": float(tip_alignment["rmse_mm"]),
        "overall_fre_mm": overall_fre_mm,
        "model_fle_mm": _legacy_fle_estimate_mm(float(model_alignment["rmse_mm"]), expected_model_count),
        "tip_fle_mm": _legacy_fle_estimate_mm(float(tip_alignment["rmse_mm"]), expected_tip_count),
        "legacy_tre_mm": {
            "aurora_2_model": _legacy_tre_estimate_mm(
                float(model_alignment["rmse_mm"]),
                expected_model_count,
                model_truth_in_model,
                reference_radius_mm=model_tre_reference_radius_mm,
                divisor=3.0,
            ),
            "aurora_2_tip": _legacy_tre_estimate_mm(
                float(tip_alignment["rmse_mm"]),
                expected_tip_count,
                tip_truth_in_tip,
                reference_radius_mm=tip_tre_reference_radius_mm,
                divisor=2.0,
            ),
        },
        "quaternion_average_method": quaternion_average_method,
        "quaternion_average_details": averaging_details,
        "model_residuals_repeated_mm": model_alignment["residuals"].T.tolist(),
        "tip_residuals_repeated_mm": tip_alignment["residuals"].T.tolist(),
        "expected_total_measurement_count": expected_total_count,
    }
    legacy_tre = validation_metrics["legacy_tre_mm"]
    if legacy_tre["aurora_2_model"] is not None and legacy_tre["aurora_2_tip"] is not None:
        legacy_tre["tip_2_model"] = float(
            np.sqrt(legacy_tre["aurora_2_model"] ** 2 + legacy_tre["aurora_2_tip"] ** 2)
        )
    else:
        legacy_tre["tip_2_model"] = None

    return LegacyRegistrationResult(
        measurement_tool_id=measurement_tool_id,
        coil_tool_id=coil_tool_id,
        repetitions=repetitions,
        ordered_labels=ordered_labels,
        group_by_label=group_by_label,
        truth_points_in_sw_by_label=truth_points_in_sw_by_label,
        raw_points_by_label=raw_points_by_label,
        raw_measurement_tool_samples_by_label=raw_measurement_tool_samples_by_label,
        raw_coil_samples_by_label=raw_coil_samples_by_label,
        averaged_points_by_label=averaged_points_by_label,
        measured_points_in_aurora=measured_points_in_aurora,
        model_truth_in_model=model_truth_in_model,
        tip_truth_in_tip=tip_truth_in_tip,
        T_aurora_2_model=model_alignment["transform"],
        T_aurora_2_tip=tip_alignment["transform"],
        T_tip_2_coil=T_tip_2_coil,
        T_coil_tip=T_coil_tip,
        T_coil_2_aurora_mean=T_coil_2_aurora_mean,
        averaged_coil_quaternion_wxyz=averaged_quaternion,
        averaged_coil_translation_mm=averaged_translation,
        residuals_by_label=residuals_by_label,
        validation_metrics=validation_metrics,
    )


def measurement_points_from_tool_samples(
    samples: list[AuroraPoseSample],
    T_measurement_point: np.ndarray,
) -> np.ndarray:
    """Convert tracked measurement-tool poses into measured Aurora points."""
    assert_transform_matrix(T_measurement_point, "T_measurement_point")
    points = np.zeros((3, len(samples)), dtype=float)
    for idx, sample in enumerate(samples):
        R = quat_wxyz_to_rotmat(sample.quaternion_wxyz)
        T_aurora_measurement = np.eye(4, dtype=float)
        T_aurora_measurement[0:3, 0:3] = R
        T_aurora_measurement[0:3, 3] = np.asarray(sample.translation_mm, dtype=float)
        T_aurora_point = T_aurora_measurement @ T_measurement_point
        points[:, idx] = T_aurora_point[0:3, 3]
    return points


def average_tool_samples_as_transform(
    samples: list[AuroraPoseSample],
    *,
    method: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    """Average a tool pose sequence into one homogeneous transform."""
    if not samples:
        raise ValueError("Cannot average an empty tool-sample sequence")

    quats = np.asarray([sample.quaternion_wxyz for sample in samples], dtype=float)
    trans = np.asarray([sample.translation_mm for sample in samples], dtype=float)
    averaged_quaternion, details = average_quaternions(quats, method=method)
    averaged_translation = trans.mean(axis=0)

    T = np.eye(4, dtype=float)
    T[0:3, 0:3] = quat_wxyz_to_rotmat(tuple(float(v) for v in averaged_quaternion))
    T[0:3, 3] = averaged_translation
    return T, averaged_quaternion, averaged_translation, details


def average_quaternions(quaternions_n4: np.ndarray, *, method: str) -> tuple[np.ndarray, dict[str, Any]]:
    """Average quaternions with explicit method control."""
    quats = np.asarray(quaternions_n4, dtype=float)
    if quats.ndim != 2 or quats.shape[1] != 4:
        raise ValueError("quaternions_n4 must have shape (N, 4)")
    if not np.isfinite(quats).all():
        raise ValueError("quaternions_n4 contains non-finite values")
    if quats.shape[0] == 0:
        raise ValueError("quaternions_n4 must not be empty")

    normalized = quats / np.linalg.norm(quats, axis=1, keepdims=True)
    if method == "legacy_component_mean":
        mean_quat = normalized.mean(axis=0)
        mean_quat /= np.linalg.norm(mean_quat)
        return mean_quat, {"sign_flips_applied": 0}

    if method == "sign_aligned_mean":
        aligned = normalized.copy()
        reference = aligned[0]
        sign_flips = 0
        for idx in range(1, aligned.shape[0]):
            if float(np.dot(reference, aligned[idx])) < 0.0:
                aligned[idx] *= -1.0
                sign_flips += 1
        mean_quat = aligned.mean(axis=0)
        mean_quat /= np.linalg.norm(mean_quat)
        return mean_quat, {"sign_flips_applied": sign_flips}

    raise ValueError(
        "Unsupported quaternion average method. Expected one of "
        "{'legacy_component_mean', 'sign_aligned_mean'}"
    )


def apply_transform_to_points(T_A_B: np.ndarray, points_3xN: np.ndarray) -> np.ndarray:
    """Apply one homogeneous transform to a 3xN point matrix."""
    assert_transform_matrix(T_A_B, "T_A_B")
    _validate_rotation_matrix(T_A_B[0:3, 0:3], "T_A_B")
    points = _as_points_3xN(points_3xN, "points_3xN")
    homogeneous = np.vstack([points, np.ones((1, points.shape[1]), dtype=float)])
    return (T_A_B @ homogeneous)[0:3, :]


def _group_points_by_label(
    *,
    ordered_labels: list[str],
    group_sizes: dict[str, int],
    repetitions: int,
    measured_points_in_aurora: np.ndarray,
) -> tuple[dict[str, list[list[float]]], dict[str, list[float]]]:
    raw_points_by_label: dict[str, list[list[float]]] = {}
    averaged_points_by_label: dict[str, list[float]] = {}
    for index, label in enumerate(ordered_labels):
        start = index * repetitions
        end = start + repetitions
        samples = measured_points_in_aurora[:, start:end].T.tolist()
        raw_points_by_label[label] = [[float(v) for v in sample] for sample in samples]
        averaged_points_by_label[label] = np.asarray(samples, dtype=float).mean(axis=0).tolist()
    return raw_points_by_label, averaged_points_by_label


def _group_tool_samples_by_label(
    *,
    ordered_labels: list[str],
    samples: list[AuroraPoseSample],
    repetitions: int,
) -> dict[str, list[dict[str, Any]]]:
    output: dict[str, list[dict[str, Any]]] = {}
    for index, label in enumerate(ordered_labels):
        start = index * repetitions
        end = start + repetitions
        output[label] = [
            {
                "tool_id": sample.tool_id,
                "quaternion_wxyz": list(sample.quaternion_wxyz),
                "translation_mm": list(sample.translation_mm),
                "quality": sample.quality,
                "source_row": sample.source_row,
                "source_token": sample.source_token,
            }
            for sample in samples[start:end]
        ]
    return output


def _mean_residuals_by_label(
    *,
    model_labels: list[str],
    tip_labels: list[str],
    repetitions: int,
    model_residuals: np.ndarray,
    tip_residuals: np.ndarray,
) -> dict[str, list[float]]:
    output: dict[str, list[float]] = {}
    for index, label in enumerate(model_labels):
        start = index * repetitions
        end = start + repetitions
        output[label] = np.asarray(model_residuals[:, start:end], dtype=float).mean(axis=1).tolist()
    for index, label in enumerate(tip_labels):
        start = index * repetitions
        end = start + repetitions
        output[label] = np.asarray(tip_residuals[:, start:end], dtype=float).mean(axis=1).tolist()
    return output


def _legacy_fle_estimate_mm(fre_mm: float, sample_count: int) -> float | None:
    if sample_count <= 2:
        return None
    return float(np.sqrt((fre_mm**2) * (1.0 - 2.0 / sample_count)))


def _legacy_tre_estimate_mm(
    fre_mm: float,
    sample_count: int,
    truth_points_3xN: np.ndarray,
    *,
    reference_radius_mm: float,
    divisor: float,
) -> float | None:
    fle_mm = _legacy_fle_estimate_mm(fre_mm, sample_count)
    if fle_mm is None:
        return None

    x_sq_mean = float(np.mean(np.square(truth_points_3xN[0, :])))
    y_sq_mean = float(np.mean(np.square(truth_points_3xN[1, :])))
    if x_sq_mean <= 1e-12 or y_sq_mean <= 1e-12:
        return None

    factor = (1.0 / sample_count) + ((reference_radius_mm**2) / divisor) * (
        (1.0 / x_sq_mean) + (1.0 / y_sq_mean)
    )
    return float(fle_mm * factor)


def _load_points_3xN(path: Path) -> np.ndarray:
    arr = _load_numeric_file(path)
    if arr.ndim == 1:
        raise ValueError(f"Expected a 3xN point matrix in {path}, got shape {arr.shape}")
    if arr.shape[0] != 3:
        raise ValueError(f"Expected first dimension 3 for point matrix {path}, got shape {arr.shape}")
    if arr.shape[1] < 1:
        raise ValueError(f"Point matrix {path} must contain at least one point")
    return arr


def _load_vector3(path: Path) -> np.ndarray:
    arr = _load_numeric_file(path)
    arr = np.asarray(arr, dtype=float).reshape(-1)
    if arr.shape != (3,):
        raise ValueError(f"Expected a 3-vector in {path}, got shape {arr.shape}")
    return arr


def _load_transform_4x4(path: Path) -> np.ndarray:
    arr = _load_numeric_file(path)
    if arr.shape != (4, 4):
        raise ValueError(f"Expected a 4x4 transform in {path}, got shape {arr.shape}")
    assert_transform_matrix(arr, str(path))
    _validate_rotation_matrix(arr[0:3, 0:3], str(path))
    return arr


def _load_numeric_file(path: Path) -> np.ndarray:
    if not path.exists():
        raise FileNotFoundError(f"Missing registration asset file: {path}")
    try:
        arr = np.loadtxt(path, delimiter=",")
    except Exception as exc:
        raise ValueError(f"Failed to parse numeric registration asset file: {path}") from exc
    arr = np.asarray(arr, dtype=float)
    if not np.isfinite(arr).all():
        raise ValueError(f"Registration asset file contains non-finite values: {path}")
    return arr


def _as_points_3xN(points: np.ndarray, name: str) -> np.ndarray:
    arr = np.asarray(points, dtype=float)
    if arr.ndim != 2:
        raise ValueError(f"{name} must be a 2D array")
    if arr.shape[0] == 3:
        return arr
    if arr.shape[1] == 3:
        return arr.T
    raise ValueError(f"{name} must have shape (3, N) or (N, 3)")


def _validate_rotation_matrix(R: np.ndarray, name: str) -> None:
    if R.shape != (3, 3):
        raise ValueError(f"{name} rotation block must have shape (3, 3)")
    if not np.allclose(R.T @ R, np.eye(3), atol=1e-5):
        raise ValueError(f"{name} rotation block is not orthonormal")
    det = float(np.linalg.det(R))
    if not np.isclose(det, 1.0, atol=1e-5):
        raise ValueError(f"{name} rotation block determinant must be +1, got {det:.6f}")


def _find_tool_index(row: list[str]) -> int | None:
    for index, value in enumerate(row):
        if _TOOL_ID_RE.fullmatch(value):
            return index
    return None

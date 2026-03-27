"""Registration validation, comparison, and runtime sanity helpers."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
import time
from typing import Any

import numpy as np

from continuum_robot.hardware.mock_aurora_client import MockAuroraClient
from continuum_robot.registration.legacy_compat import (
    AuroraPoseSample,
    LegacyRegistrationResult,
    RegistrationAssets,
    parse_aurora_csv,
    solve_registration_from_tool_samples,
)
from continuum_robot.registration.rigid_solver import RigidRegistrationSolver
from continuum_robot.services.tracking_service import TrackingService
from continuum_robot.tracking.tip_pose_service import TipPoseService
from continuum_robot.tracking.transforms import assert_transform_matrix, make_transform_A_B
from continuum_robot.utils.time_utils import utc_now_iso


_REQUIRED_COMPARISON_TRANSFORMS = ("T_aurora_2_model", "T_tip_2_coil")
_OPTIONAL_COMPARISON_TRANSFORMS = ("T_aurora_2_tip", "T_coil_tip")


@dataclass
class RegistrationValidationReport:
    """Serializable registration validation report."""

    generated_at_utc: str
    source_kind: str
    source_path: str
    measurement_tool_id: str
    coil_tool_id: str
    repetition_count: int
    ordered_labels: list[str]
    point_counts: dict[str, int]
    captured_counts_by_label: dict[str, int]
    transforms: dict[str, list[list[float]]]
    validation_metrics: dict[str, Any]
    asset_paths: dict[str, str]


@dataclass
class TransformComparison:
    """One transform comparison outcome."""

    name: str
    required: bool
    compared: bool
    within_tolerance: bool
    translation_difference_mm: float | None
    rotation_difference_deg: float | None
    left_present: bool
    right_present: bool
    note: str = ""


@dataclass
class RegistrationComparisonReport:
    """Registration output comparison summary."""

    generated_at_utc: str
    left_source: str
    right_source: str
    translation_tolerance_mm: float
    rotation_tolerance_deg: float
    fre_tolerance_mm: float
    tool_role_match: bool | None
    tool_role_note: str
    transform_comparisons: list[TransformComparison]
    metric_differences_mm: dict[str, float | None]
    passed: bool


@dataclass
class RuntimeSanityReport:
    """Runtime tip-pose sanity-check report."""

    generated_at_utc: str
    source_kind: str
    registration_path: str
    expected_runtime_coil_tool_id: str
    stored_coil_tool_id: str | None
    passed: bool
    status: str
    registration_state: str
    tip_pose_status: str
    connection_state: str
    packets_received_count: int | None
    last_frame_number: int | None
    tracking_faults: list[str]
    T_robot_tip: list[list[float]] | None
    tip_translation_mm: list[float] | None
    last_error: str | None


@dataclass
class RegistrationOutputSet:
    """Normalized transform/metric bundle for compatibility comparison."""

    source_path: str
    source_kind: str
    transforms: dict[str, np.ndarray]
    validation_metrics: dict[str, Any]
    measurement_tool_id: str | None
    coil_tool_id: str | None


def run_registration_validation_from_csv(
    *,
    registration_csv: Path,
    assets: RegistrationAssets,
    solver: RigidRegistrationSolver,
    measurement_tool_id: str,
    coil_tool_id: str,
    quaternion_average_method: str,
    model_tre_reference_radius_mm: float,
    tip_tre_reference_radius_mm: float,
) -> RegistrationValidationReport:
    """Solve registration from a legacy Aurora CSV and return a validation report."""
    transforms = parse_aurora_csv(registration_csv)
    if measurement_tool_id not in transforms:
        raise ValueError(f"Aurora CSV {registration_csv} is missing measurement tool {measurement_tool_id}")
    if coil_tool_id not in transforms:
        raise ValueError(f"Aurora CSV {registration_csv} is missing coil tool {coil_tool_id}")

    measurement_samples = transforms[measurement_tool_id]
    coil_samples = transforms[coil_tool_id]
    repetitions = _expected_repetition_count(assets, measurement_samples)
    result = solve_registration_from_tool_samples(
        assets=assets,
        measurement_tool_samples=measurement_samples,
        coil_tool_samples=coil_samples,
        repetitions=repetitions,
        measurement_tool_id=measurement_tool_id,
        coil_tool_id=coil_tool_id,
        solver=solver,
        quaternion_average_method=quaternion_average_method,
        model_tre_reference_radius_mm=model_tre_reference_radius_mm,
        tip_tre_reference_radius_mm=tip_tre_reference_radius_mm,
    )
    return _build_validation_report(result=result, source_kind="aurora_csv", source_path=registration_csv, assets=assets)


def run_registration_validation_from_session_json(
    *,
    session_json: Path,
    assets: RegistrationAssets,
    solver: RigidRegistrationSolver,
    measurement_tool_id: str | None,
    coil_tool_id: str | None,
    quaternion_average_method: str,
    model_tre_reference_radius_mm: float,
    tip_tre_reference_radius_mm: float,
) -> RegistrationValidationReport:
    """Solve registration from a saved registration/session JSON artifact."""
    ordered_labels = [*assets.model_labels, *assets.tip_labels]
    measurement_samples, coil_samples, repetitions, resolved_measurement_tool_id, resolved_coil_tool_id, _counts = (
        load_pose_samples_from_saved_session(
            session_json,
            measurement_tool_id=measurement_tool_id,
            coil_tool_id=coil_tool_id,
            expected_ordered_labels=ordered_labels,
        )
    )
    result = solve_registration_from_tool_samples(
        assets=assets,
        measurement_tool_samples=measurement_samples,
        coil_tool_samples=coil_samples,
        repetitions=repetitions,
        measurement_tool_id=resolved_measurement_tool_id,
        coil_tool_id=resolved_coil_tool_id,
        solver=solver,
        quaternion_average_method=quaternion_average_method,
        model_tre_reference_radius_mm=model_tre_reference_radius_mm,
        tip_tre_reference_radius_mm=tip_tre_reference_radius_mm,
    )
    return _build_validation_report(
        result=result,
        source_kind="saved_session_json",
        source_path=session_json,
        assets=assets,
    )


def load_pose_samples_from_saved_session(
    session_json: Path,
    *,
    measurement_tool_id: str | None,
    coil_tool_id: str | None,
    expected_ordered_labels: list[str],
) -> tuple[list[AuroraPoseSample], list[AuroraPoseSample], int, str, str, dict[str, int]]:
    """Load grouped raw tool samples from a saved registration/session JSON artifact."""
    payload = json.loads(session_json.read_text(encoding="utf-8"))
    grouped_measurement = payload.get("raw_measurement_tool_samples_by_label")
    grouped_coil = payload.get("raw_coil_samples_by_label")
    if not isinstance(grouped_measurement, dict) or not grouped_measurement:
        raise ValueError(
            f"Saved session {session_json} is missing raw_measurement_tool_samples_by_label; "
            "use a registration/session JSON produced by the new backend"
        )
    if not isinstance(grouped_coil, dict) or not grouped_coil:
        raise ValueError(
            f"Saved session {session_json} is missing raw_coil_samples_by_label; "
            "cannot rerun legacy-compatible registration"
        )

    ordered_labels = payload.get("landmark_labels")
    if ordered_labels is None:
        ordered_labels = list(grouped_measurement.keys())
    ordered_labels = [str(label) for label in ordered_labels]
    if ordered_labels != list(expected_ordered_labels):
        raise ValueError(
            "Saved session label ordering does not match the configured protected assets. "
            f"Got {ordered_labels}, expected {expected_ordered_labels}"
        )

    config_used = payload.get("config_used", {}) if isinstance(payload.get("config_used"), dict) else {}
    recorded_measurement_tool_id = (
        measurement_tool_id
        or payload.get("measurement_tool_id")
        or config_used.get("measurement_tool_id")
        or config_used.get("capture_tool_id")
    )
    recorded_coil_tool_id = coil_tool_id or payload.get("coil_tool_id") or config_used.get("coil_tool_id")
    if not recorded_measurement_tool_id:
        raise ValueError(f"Saved session {session_json} does not record a measurement tool id")
    if not recorded_coil_tool_id:
        raise ValueError(f"Saved session {session_json} does not record a coil tool id")

    counts_by_label: dict[str, int] = {}
    repetition_count: int | None = None
    measurement_samples: list[AuroraPoseSample] = []
    coil_samples: list[AuroraPoseSample] = []
    for label in ordered_labels:
        measurement_raw = grouped_measurement.get(label)
        coil_raw = grouped_coil.get(label)
        if not isinstance(measurement_raw, list) or not isinstance(coil_raw, list):
            raise ValueError(f"Saved session {session_json} has malformed grouped samples for label {label}")
        if len(measurement_raw) != len(coil_raw):
            raise ValueError(
                f"Saved session {session_json} has mismatched measurement/coil counts for label {label}: "
                f"{len(measurement_raw)} vs {len(coil_raw)}"
            )
        if repetition_count is None:
            repetition_count = len(measurement_raw)
        elif len(measurement_raw) != repetition_count:
            raise ValueError(
                f"Saved session {session_json} has repetition-count mismatch at label {label}: "
                f"expected {repetition_count}, got {len(measurement_raw)}"
            )
        counts_by_label[label] = len(measurement_raw)
        for raw in measurement_raw:
            sample = _pose_sample_from_raw(raw, label=label, role_name="measurement tool")
            if sample.tool_id != recorded_measurement_tool_id:
                raise ValueError(
                    f"Tool-role mismatch for label {label}: measurement sample records tool {sample.tool_id}, "
                    f"expected {recorded_measurement_tool_id}"
                )
            measurement_samples.append(sample)
        for raw in coil_raw:
            sample = _pose_sample_from_raw(raw, label=label, role_name="coil tool")
            if sample.tool_id != recorded_coil_tool_id:
                raise ValueError(
                    f"Tool-role mismatch for label {label}: coil sample records tool {sample.tool_id}, "
                    f"expected {recorded_coil_tool_id}"
                )
            coil_samples.append(sample)

    if repetition_count is None or repetition_count <= 0:
        raise ValueError(f"Saved session {session_json} does not contain any capture repetitions")

    return (
        measurement_samples,
        coil_samples,
        repetition_count,
        str(recorded_measurement_tool_id),
        str(recorded_coil_tool_id),
        counts_by_label,
    )


def save_validation_report(report: RegistrationValidationReport, path: Path) -> Path:
    """Serialize a validation report to JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(asdict(report), indent=2), encoding="utf-8")
    return path


def load_registration_output(path: Path) -> RegistrationOutputSet:
    """Load transforms and metrics from a validation report, registration JSON, or legacy transform dir."""
    if path.is_dir():
        return _load_transform_directory(path)

    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload.get("transforms"), dict):
        transforms = _load_transforms_from_mapping(payload["transforms"], source_path=path)
        return RegistrationOutputSet(
            source_path=str(path),
            source_kind="validation_report_json",
            transforms=transforms,
            validation_metrics=dict(payload.get("validation_metrics", {})),
            measurement_tool_id=str(payload.get("measurement_tool_id")) if payload.get("measurement_tool_id") else None,
            coil_tool_id=str(payload.get("coil_tool_id")) if payload.get("coil_tool_id") else None,
        )

    config_used = payload.get("config_used", {}) if isinstance(payload.get("config_used"), dict) else {}
    transforms = _load_transforms_from_mapping(payload, source_path=path)
    measurement_tool_id = payload.get("measurement_tool_id") or config_used.get("measurement_tool_id") or config_used.get(
        "capture_tool_id"
    )
    coil_tool_id = payload.get("coil_tool_id") or config_used.get("coil_tool_id")
    return RegistrationOutputSet(
        source_path=str(path),
        source_kind="registration_json",
        transforms=transforms,
        validation_metrics=dict(payload.get("validation_metrics", {})),
        measurement_tool_id=str(measurement_tool_id) if measurement_tool_id else None,
        coil_tool_id=str(coil_tool_id) if coil_tool_id else None,
    )


def compare_registration_outputs(
    left: RegistrationOutputSet,
    right: RegistrationOutputSet,
    *,
    translation_tolerance_mm: float = 0.25,
    rotation_tolerance_deg: float = 0.25,
    fre_tolerance_mm: float = 0.05,
) -> RegistrationComparisonReport:
    """Compare registration outputs numerically and return a pass/fail report."""
    comparisons: list[TransformComparison] = []
    passed = True
    for name in [*_REQUIRED_COMPARISON_TRANSFORMS, *_OPTIONAL_COMPARISON_TRANSFORMS]:
        required = name in _REQUIRED_COMPARISON_TRANSFORMS
        left_T = left.transforms.get(name)
        right_T = right.transforms.get(name)
        if left_T is None or right_T is None:
            comparisons.append(
                TransformComparison(
                    name=name,
                    required=required,
                    compared=False,
                    within_tolerance=not required,
                    translation_difference_mm=None,
                    rotation_difference_deg=None,
                    left_present=left_T is not None,
                    right_present=right_T is not None,
                    note="not present on both sides",
                )
            )
            if required:
                passed = False
            continue

        translation_diff, rotation_diff = _transform_difference(left_T, right_T)
        within = translation_diff <= translation_tolerance_mm and rotation_diff <= rotation_tolerance_deg
        comparisons.append(
            TransformComparison(
                name=name,
                required=required,
                compared=True,
                within_tolerance=within,
                translation_difference_mm=translation_diff,
                rotation_difference_deg=rotation_diff,
                left_present=True,
                right_present=True,
            )
        )
        if required and not within:
            passed = False

    metric_differences_mm: dict[str, float | None] = {}
    for metric_name in ("overall_fre_mm", "model_fre_mm", "tip_fre_mm"):
        left_value = left.validation_metrics.get(metric_name)
        right_value = right.validation_metrics.get(metric_name)
        if left_value is None or right_value is None:
            metric_differences_mm[metric_name] = None
            continue
        diff = abs(float(left_value) - float(right_value))
        metric_differences_mm[metric_name] = diff
        if diff > fre_tolerance_mm:
            passed = False

    tool_role_match = None
    tool_role_note = "tool roles unavailable on one or both sides"
    if (
        left.measurement_tool_id is not None
        and right.measurement_tool_id is not None
        and left.coil_tool_id is not None
        and right.coil_tool_id is not None
    ):
        tool_role_match = (
            left.measurement_tool_id == right.measurement_tool_id and left.coil_tool_id == right.coil_tool_id
        )
        if tool_role_match:
            tool_role_note = "tool-role assignment matches"
        else:
            tool_role_note = (
                f"left uses measurement={left.measurement_tool_id}, coil={left.coil_tool_id}; "
                f"right uses measurement={right.measurement_tool_id}, coil={right.coil_tool_id}"
            )
            passed = False

    return RegistrationComparisonReport(
        generated_at_utc=utc_now_iso(),
        left_source=left.source_path,
        right_source=right.source_path,
        translation_tolerance_mm=float(translation_tolerance_mm),
        rotation_tolerance_deg=float(rotation_tolerance_deg),
        fre_tolerance_mm=float(fre_tolerance_mm),
        tool_role_match=tool_role_match,
        tool_role_note=tool_role_note,
        transform_comparisons=comparisons,
        metric_differences_mm=metric_differences_mm,
        passed=passed,
    )


def evaluate_runtime_sanity_from_capture(
    *,
    registration_path: Path,
    capture_path: Path,
    expected_runtime_coil_tool_id: str = "0A",
) -> RuntimeSanityReport:
    """Replay a packet capture through the runtime tip-pose path and summarize the outcome."""
    try:
        payload = json.loads(registration_path.read_text(encoding="utf-8"))
        _ = TipPoseService.from_registration_file(registration_path)
    except Exception as exc:
        return _runtime_failure_report(
            source_kind="replay_capture",
            registration_path=registration_path,
            expected_runtime_coil_tool_id=expected_runtime_coil_tool_id,
            stored_coil_tool_id=None,
            registration_state="invalid_registration",
            tip_pose_status="invalid_registration",
            connection_state="replay",
            last_error=str(exc),
        )

    stored_coil_tool_id = _resolve_registration_coil_tool_id(payload)
    if stored_coil_tool_id is not None and stored_coil_tool_id != expected_runtime_coil_tool_id:
        return _runtime_failure_report(
            source_kind="replay_capture",
            registration_path=registration_path,
            expected_runtime_coil_tool_id=expected_runtime_coil_tool_id,
            stored_coil_tool_id=stored_coil_tool_id,
            registration_state="loaded",
            tip_pose_status="role_mismatch",
            connection_state="replay",
            last_error=(
                f"Registration expects coil tool {stored_coil_tool_id}, "
                f"but runtime sanity check expects {expected_runtime_coil_tool_id}"
            ),
        )

    try:
        service = TrackingService(
            MockAuroraClient(),
            port="/dev/null",
            registration_path=registration_path,
            config_source="runtime_sanity_replay",
        )
        snapshot = service.replay_capture(capture_path)
    except Exception as exc:
        return _runtime_failure_report(
            source_kind="replay_capture",
            registration_path=registration_path,
            expected_runtime_coil_tool_id=expected_runtime_coil_tool_id,
            stored_coil_tool_id=stored_coil_tool_id,
            registration_state="loaded",
            tip_pose_status="runtime_error",
            connection_state="replay",
            last_error=str(exc),
        )

    tip_translation_mm = None
    if snapshot.T_robot_tip is not None:
        tip_translation_mm = [float(snapshot.T_robot_tip[row][3]) for row in range(3)]

    passed = snapshot.registration_state == "loaded" and snapshot.tip_pose_status == "ok" and snapshot.T_robot_tip is not None
    status = (
        "Runtime tip pose is valid from replayed 0A data"
        if passed
        else f"Runtime tip pose check failed: registration={snapshot.registration_state}, tip={snapshot.tip_pose_status}"
    )
    return RuntimeSanityReport(
        generated_at_utc=utc_now_iso(),
        source_kind="replay_capture",
        registration_path=str(registration_path),
        expected_runtime_coil_tool_id=expected_runtime_coil_tool_id,
        stored_coil_tool_id=stored_coil_tool_id,
        passed=passed,
        status=status,
        registration_state=snapshot.registration_state,
        tip_pose_status=snapshot.tip_pose_status,
        connection_state=snapshot.connection_state,
        packets_received_count=snapshot.packets_received_count,
        last_frame_number=snapshot.last_frame_number,
        tracking_faults=list(snapshot.faults),
        T_robot_tip=snapshot.T_robot_tip,
        tip_translation_mm=tip_translation_mm,
        last_error=snapshot.health.last_error,
    )


def evaluate_runtime_sanity_live(
    *,
    tracking_service,
    registration_path: Path,
    expected_runtime_coil_tool_id: str = "0A",
    frames: int = 3,
    timeout_s: float = 5.0,
) -> RuntimeSanityReport:
    """Compute T_robot_tip from live TrackingService data and summarize the result."""
    if not registration_path.exists():
        return _runtime_failure_report(
            source_kind="live_tracker",
            registration_path=registration_path,
            expected_runtime_coil_tool_id=expected_runtime_coil_tool_id,
            stored_coil_tool_id=None,
            registration_state="missing_registration",
            tip_pose_status="missing_registration",
            connection_state="not_started",
            last_error=(
                f"Missing registration file: {registration_path}. "
                "Run the registration workflow first to create latest_registration.json, "
                "or use scripts/run_diagnostics.py and scripts/run_tracker_benchmark.py "
                "to validate tracker connectivity without registration."
            ),
        )
    try:
        payload = json.loads(registration_path.read_text(encoding="utf-8"))
        tip_service = TipPoseService.from_registration_file(registration_path)
    except Exception as exc:
        return _runtime_failure_report(
            source_kind="live_tracker",
            registration_path=registration_path,
            expected_runtime_coil_tool_id=expected_runtime_coil_tool_id,
            stored_coil_tool_id=None,
            registration_state="invalid_registration",
            tip_pose_status="invalid_registration",
            connection_state="disconnected",
            last_error=str(exc),
        )

    stored_coil_tool_id = _resolve_registration_coil_tool_id(payload)
    if stored_coil_tool_id is not None and stored_coil_tool_id != expected_runtime_coil_tool_id:
        return _runtime_failure_report(
            source_kind="live_tracker",
            registration_path=registration_path,
            expected_runtime_coil_tool_id=expected_runtime_coil_tool_id,
            stored_coil_tool_id=stored_coil_tool_id,
            registration_state="loaded",
            tip_pose_status="role_mismatch",
            connection_state="disconnected",
            last_error=(
                f"Registration expects coil tool {stored_coil_tool_id}, "
                f"but runtime sanity check expects {expected_runtime_coil_tool_id}"
            ),
        )

    observed_frames: set[int] = set()
    last_snapshot = tracking_service.get_snapshot()
    last_error = None
    last_tool_snapshot = None
    started_here = last_snapshot.connection_state in {"disconnected", "stopped", "error"}
    try:
        if started_here:
            tracking_service.start()
        deadline = time.monotonic() + max(0.1, float(timeout_s))
        target_frames = max(1, int(frames))
        while time.monotonic() < deadline:
            snapshot = tracking_service.get_snapshot()
            last_snapshot = snapshot
            if snapshot.last_error:
                last_error = snapshot.last_error
            tool = snapshot.tools.get(expected_runtime_coil_tool_id)
            if (
                tool is not None
                and tool.tracking_state == "tracked"
                and tool.frame_number is not None
                and tool.frame_number not in observed_frames
            ):
                observed_frames.add(int(tool.frame_number))
                last_tool_snapshot = tool
                if len(observed_frames) >= target_frames:
                    break
            time.sleep(0.05)
    except Exception as exc:
        return _runtime_failure_report(
            source_kind="live_tracker",
            registration_path=registration_path,
            expected_runtime_coil_tool_id=expected_runtime_coil_tool_id,
            stored_coil_tool_id=stored_coil_tool_id,
            registration_state="loaded",
            tip_pose_status="runtime_error",
            connection_state=last_snapshot.connection_state,
            last_error=str(exc),
        )
    finally:
        if started_here:
            tracking_service.stop()

    if last_tool_snapshot is None:
        return _runtime_failure_report(
            source_kind="live_tracker",
            registration_path=registration_path,
            expected_runtime_coil_tool_id=expected_runtime_coil_tool_id,
            stored_coil_tool_id=stored_coil_tool_id,
            registration_state="loaded",
            tip_pose_status="missing_0A",
            connection_state=last_snapshot.connection_state,
            last_error=last_error or f"No valid {expected_runtime_coil_tool_id} sample observed before timeout",
        )

    try:
        T_aurora_coil = make_transform_A_B(
            tuple(float(v) for v in last_tool_snapshot.quaternion_wxyz),
            tuple(float(v) for v in last_tool_snapshot.translation_mm),
        )
        T_robot_tip = tip_service.compute_T_robot_tip(
            T_robot_aurora=tip_service.inputs.T_robot_aurora,
            T_aurora_coil=T_aurora_coil,
            T_coil_tip=tip_service.inputs.T_coil_tip,
        )
    except Exception as exc:
        return _runtime_failure_report(
            source_kind="live_tracker",
            registration_path=registration_path,
            expected_runtime_coil_tool_id=expected_runtime_coil_tool_id,
            stored_coil_tool_id=stored_coil_tool_id,
            registration_state="loaded",
            tip_pose_status="invalid_transform_chain",
            connection_state=last_snapshot.connection_state,
            last_error=str(exc),
        )

    tip_translation_mm = [float(T_robot_tip[row, 3]) for row in range(3)]
    return RuntimeSanityReport(
        generated_at_utc=utc_now_iso(),
        source_kind="live_tracker",
        registration_path=str(registration_path),
        expected_runtime_coil_tool_id=expected_runtime_coil_tool_id,
        stored_coil_tool_id=stored_coil_tool_id,
        passed=True,
        status=f"Runtime tip pose is valid from live {expected_runtime_coil_tool_id} data",
        registration_state="loaded",
        tip_pose_status="ok",
        connection_state=last_snapshot.connection_state,
        packets_received_count=len(observed_frames),
        last_frame_number=int(last_tool_snapshot.frame_number)
        if last_tool_snapshot.frame_number is not None
        else None,
        tracking_faults=list(last_snapshot.faults),
        T_robot_tip=T_robot_tip.tolist(),
        tip_translation_mm=tip_translation_mm,
        last_error=last_error,
    )


def _build_validation_report(
    *,
    result: LegacyRegistrationResult,
    source_kind: str,
    source_path: Path,
    assets: RegistrationAssets,
) -> RegistrationValidationReport:
    point_counts = {
        "model_truth_points": int(assets.model_truth_in_sw.shape[1]),
        "tip_truth_points": int(assets.tip_truth_in_sw.shape[1]),
        "total_truth_points": int(assets.model_truth_in_sw.shape[1] + assets.tip_truth_in_sw.shape[1]),
        "measurement_samples": int(result.measured_points_in_aurora.shape[1]),
        "coil_samples": int(sum(len(samples) for samples in result.raw_coil_samples_by_label.values())),
    }
    captured_counts_by_label = {label: len(points) for label, points in result.raw_points_by_label.items()}
    transforms = {
        "T_aurora_2_model": result.T_aurora_2_model.tolist(),
        "T_aurora_2_tip": result.T_aurora_2_tip.tolist(),
        "T_tip_2_coil": result.T_tip_2_coil.tolist(),
        "T_coil_tip": result.T_coil_tip.tolist(),
    }
    return RegistrationValidationReport(
        generated_at_utc=utc_now_iso(),
        source_kind=source_kind,
        source_path=str(source_path),
        measurement_tool_id=result.measurement_tool_id,
        coil_tool_id=result.coil_tool_id,
        repetition_count=result.repetitions,
        ordered_labels=list(result.ordered_labels),
        point_counts=point_counts,
        captured_counts_by_label=captured_counts_by_label,
        transforms=transforms,
        validation_metrics=json.loads(json.dumps(result.validation_metrics)),
        asset_paths={
            "model_points_file": str(assets.paths.model_points_file),
            "tip_points_file": str(assets.paths.tip_points_file),
            "T_sw_2_model_file": str(assets.paths.T_sw_2_model_file),
            "T_sw_2_tip_file": str(assets.paths.T_sw_2_tip_file),
            "penprobe_file": str(assets.paths.penprobe_file),
        },
    )


def _expected_repetition_count(assets: RegistrationAssets, measurement_samples: list[AuroraPoseSample]) -> int:
    truth_point_count = assets.model_truth_in_sw.shape[1] + assets.tip_truth_in_sw.shape[1]
    total_measurements = len(measurement_samples)
    if total_measurements <= 0:
        raise ValueError("No measurement samples were provided")
    if truth_point_count <= 0:
        raise ValueError("Configured registration assets contain no truth points")
    if total_measurements % truth_point_count != 0:
        raise ValueError(
            "Measurement sample count does not divide evenly into configured truth points: "
            f"{total_measurements} vs {truth_point_count}"
        )
    return total_measurements // truth_point_count


def _pose_sample_from_raw(raw: dict[str, Any], *, label: str, role_name: str) -> AuroraPoseSample:
    if not isinstance(raw, dict):
        raise ValueError(f"Saved session contains malformed {role_name} sample for label {label}")
    quat = raw.get("quaternion_wxyz")
    translation = raw.get("translation_mm")
    if quat is None or translation is None:
        raise ValueError(f"Saved session contains an incomplete {role_name} sample for label {label}")
    if len(quat) != 4 or len(translation) != 3:
        raise ValueError(f"Saved session contains a wrong-shaped {role_name} sample for label {label}")
    return AuroraPoseSample(
        tool_id=str(raw.get("tool_id", "")),
        quaternion_wxyz=tuple(float(v) for v in quat),
        translation_mm=tuple(float(v) for v in translation),
        quality=float(raw["quality"]) if raw.get("quality") is not None else None,
        source_row=int(raw["source_row"]) if raw.get("source_row") is not None else None,
        source_token=str(raw.get("source_token", "")),
    )


def _load_transform_directory(path: Path) -> RegistrationOutputSet:
    transforms: dict[str, np.ndarray] = {}
    T_aurora_2_model_path = path / "T_aurora_2_model"
    T_tip_2_coil_path = path / "T_tip_2_coil"
    if not T_aurora_2_model_path.exists():
        raise FileNotFoundError(f"Legacy transform directory {path} is missing T_aurora_2_model")
    if not T_tip_2_coil_path.exists():
        raise FileNotFoundError(f"Legacy transform directory {path} is missing T_tip_2_coil")

    transforms["T_aurora_2_model"] = _load_transform_matrix(T_aurora_2_model_path, "T_aurora_2_model")
    transforms["T_tip_2_coil"] = _load_transform_matrix(T_tip_2_coil_path, "T_tip_2_coil")
    transforms["T_coil_tip"] = np.linalg.inv(transforms["T_tip_2_coil"])

    for optional_name in ("T_aurora_2_tip", "T_coil_tip"):
        optional_path = path / optional_name
        if optional_path.exists():
            transforms[optional_name] = _load_transform_matrix(optional_path, optional_name)

    return RegistrationOutputSet(
        source_path=str(path),
        source_kind="legacy_transform_dir",
        transforms=transforms,
        validation_metrics={},
        measurement_tool_id=None,
        coil_tool_id=None,
    )


def _load_transforms_from_mapping(mapping: dict[str, Any], *, source_path: Path) -> dict[str, np.ndarray]:
    transforms: dict[str, np.ndarray] = {}
    if "T_robot_aurora" in mapping:
        transforms["T_aurora_2_model"] = _matrix_from_any(mapping["T_robot_aurora"], "T_robot_aurora", source_path)
    elif "T_aurora_2_model" in mapping:
        transforms["T_aurora_2_model"] = _matrix_from_any(mapping["T_aurora_2_model"], "T_aurora_2_model", source_path)

    if "T_aurora_2_tip" in mapping:
        transforms["T_aurora_2_tip"] = _matrix_from_any(mapping["T_aurora_2_tip"], "T_aurora_2_tip", source_path)
    if "T_tip_2_coil" in mapping:
        transforms["T_tip_2_coil"] = _matrix_from_any(mapping["T_tip_2_coil"], "T_tip_2_coil", source_path)
    if "T_coil_tip" in mapping:
        transforms["T_coil_tip"] = _matrix_from_any(mapping["T_coil_tip"], "T_coil_tip", source_path)
    elif "T_tip_2_coil" in transforms:
        transforms["T_coil_tip"] = np.linalg.inv(transforms["T_tip_2_coil"])
    if "T_tip_2_coil" not in transforms and "T_coil_tip" in transforms:
        transforms["T_tip_2_coil"] = np.linalg.inv(transforms["T_coil_tip"])
    return transforms


def _matrix_from_any(raw: Any, name: str, source_path: Path) -> np.ndarray:
    matrix = np.asarray(raw, dtype=float)
    if matrix.shape != (4, 4):
        raise ValueError(f"{name} in {source_path} must have shape (4, 4)")
    assert_transform_matrix(matrix, name)
    return matrix


def _load_transform_matrix(path: Path, name: str) -> np.ndarray:
    matrix = np.loadtxt(path, delimiter=",")
    if matrix.shape != (4, 4):
        raise ValueError(f"{name} at {path} must have shape (4, 4)")
    assert_transform_matrix(matrix, name)
    return matrix


def _transform_difference(left_T: np.ndarray, right_T: np.ndarray) -> tuple[float, float]:
    delta = np.linalg.inv(left_T) @ right_T
    translation_mm = float(np.linalg.norm(delta[0:3, 3]))
    rotation_deg = _rotation_angle_deg(delta[0:3, 0:3])
    return translation_mm, rotation_deg


def _rotation_angle_deg(R: np.ndarray) -> float:
    value = float((np.trace(R) - 1.0) / 2.0)
    value = min(1.0, max(-1.0, value))
    return float(np.degrees(np.arccos(value)))


def _resolve_registration_coil_tool_id(payload: dict[str, Any]) -> str | None:
    config_used = payload.get("config_used", {}) if isinstance(payload.get("config_used"), dict) else {}
    coil_tool_id = payload.get("coil_tool_id") or config_used.get("coil_tool_id")
    return str(coil_tool_id) if coil_tool_id else None


def _runtime_failure_report(
    *,
    source_kind: str,
    registration_path: Path,
    expected_runtime_coil_tool_id: str,
    stored_coil_tool_id: str | None,
    registration_state: str,
    tip_pose_status: str,
    connection_state: str,
    last_error: str,
) -> RuntimeSanityReport:
    return RuntimeSanityReport(
        generated_at_utc=utc_now_iso(),
        source_kind=source_kind,
        registration_path=str(registration_path),
        expected_runtime_coil_tool_id=expected_runtime_coil_tool_id,
        stored_coil_tool_id=stored_coil_tool_id,
        passed=False,
        status=f"Runtime sanity check failed: {last_error}",
        registration_state=registration_state,
        tip_pose_status=tip_pose_status,
        connection_state=connection_state,
        packets_received_count=None,
        last_frame_number=None,
        tracking_faults=[],
        T_robot_tip=None,
        tip_translation_mm=None,
        last_error=last_error,
    )

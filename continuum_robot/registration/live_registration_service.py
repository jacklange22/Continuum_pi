"""Legacy-compatible registration workflow backed by raw tracker-manager poses.

This module is retained for CSV/session validation utilities and legacy-style
workflows. The main app runtime path uses `RegistrationService`, which consumes
the shared `TrackingService` cache.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import json
from pathlib import Path

import numpy as np

from continuum_robot.registration.capture_session import RegistrationSession
from continuum_robot.registration.legacy_compat import (
    AuroraPoseSample,
    LegacyRegistrationResult,
    RegistrationAssetPaths,
    load_registration_assets,
    parse_aurora_csv,
    solve_registration_from_tool_samples,
)
from continuum_robot.registration.repository import RegistrationRecord, RegistrationRepository
from continuum_robot.registration.rigid_solver import RigidRegistrationSolver
from continuum_robot.registration.validation import compute_fre_mm
from continuum_robot.tracking.transforms import compose_T_A_C, make_transform_A_B
from continuum_robot.tracking.legacy_bridge.tracker_service_manager import TrackerServiceManager


@dataclass
class RegistrationResult:
    """Result payload after completing one registration run."""

    record: RegistrationRecord
    output_path: Path


class LiveRegistrationService:
    """Collect repeated registration captures and solve strict, legacy-style outputs."""

    def __init__(
        self,
        tracker_manager: TrackerServiceManager,
        repository: RegistrationRepository,
        solver: RigidRegistrationSolver,
        capture_tool_tip_transform: list[list[float]] | None = None,
        *,
        asset_paths: RegistrationAssetPaths | None = None,
        measurement_tool_id: str = "0B",
        coil_tool_id: str = "0A",
        quaternion_average_method: str = "sign_aligned_mean",
        model_tre_reference_radius_mm: float = 5.0,
        tip_tre_reference_radius_mm: float = 3.0,
    ) -> None:
        self.tracker_manager = tracker_manager
        self.repository = repository
        self.solver = solver
        self.session: RegistrationSession | None = None
        self.nominal_landmarks_robot_xyz_mm: dict[str, list[float]] = {}
        self.capture_tool_id = measurement_tool_id
        self.coil_tool_id = coil_tool_id
        self.capture_tool_tip_transform = self._coerce_transform(capture_tool_tip_transform)
        self.quaternion_average_method = quaternion_average_method
        self.model_tre_reference_radius_mm = float(model_tre_reference_radius_mm)
        self.tip_tre_reference_radius_mm = float(tip_tre_reference_radius_mm)
        self.asset_paths = asset_paths
        self.assets = (
            load_registration_assets(asset_paths, measurement_point_transform=capture_tool_tip_transform)
            if asset_paths is not None
            else None
        )

    def begin_session(
        self,
        labels: list[str] | None = None,
        captures_per_landmark: int | None = None,
        nominal_landmarks_robot_xyz_mm: dict[str, list[float]] | None = None,
        capture_tool_id: str | None = None,
        capture_tool_tip_transform: list[list[float]] | None = None,
        coil_tool_id: str | None = None,
    ) -> RegistrationSession:
        """Start a registration capture session."""
        if capture_tool_tip_transform is not None:
            self.capture_tool_tip_transform = self._coerce_transform(capture_tool_tip_transform)
            if self.asset_paths is not None:
                self.assets = load_registration_assets(
                    self.asset_paths,
                    measurement_point_transform=capture_tool_tip_transform,
                )
        if capture_tool_id is not None:
            self.capture_tool_id = capture_tool_id
        if coil_tool_id is not None:
            self.coil_tool_id = coil_tool_id

        if self.assets is not None and nominal_landmarks_robot_xyz_mm is None:
            labels = list(labels or [*self.assets.model_labels, *self.assets.tip_labels])
            captures = int(captures_per_landmark or 5)
            group_by_label = {label: "model" for label in self.assets.model_labels} | {
                label: "tip" for label in self.assets.tip_labels
            }
            truth_points_in_sw_by_label = {
                label: self.assets.model_truth_in_sw[:, idx].astype(float).tolist()
                for idx, label in enumerate(self.assets.model_labels)
            }
            truth_points_in_sw_by_label.update(
                {
                    label: self.assets.tip_truth_in_sw[:, idx].astype(float).tolist()
                    for idx, label in enumerate(self.assets.tip_labels)
                }
            )
            self.session = RegistrationSession(
                labels=labels,
                captures_per_landmark=captures,
                raw_points_by_label={label: [] for label in labels},
                raw_measurement_tool_samples_by_label={label: [] for label in labels},
                raw_coil_samples_by_label={label: [] for label in labels},
                group_by_label=group_by_label,
                truth_points_in_sw_by_label=truth_points_in_sw_by_label,
                measurement_tool_id=self.capture_tool_id,
                coil_tool_id=self.coil_tool_id,
            )
            self.nominal_landmarks_robot_xyz_mm = {}
            return self.session

        resolved_labels = list(labels or [])
        self.session = RegistrationSession(
            labels=resolved_labels,
            captures_per_landmark=int(captures_per_landmark or 1),
            raw_points_by_label={label: [] for label in resolved_labels},
            raw_measurement_tool_samples_by_label={label: [] for label in resolved_labels},
            raw_coil_samples_by_label={label: [] for label in resolved_labels},
            measurement_tool_id=self.capture_tool_id,
            coil_tool_id=self.coil_tool_id,
        )
        self.nominal_landmarks_robot_xyz_mm = nominal_landmarks_robot_xyz_mm or {}
        return self.session

    def capture_current_sample(self, label: str) -> list[float]:
        """Capture current tracker sample for one landmark label."""
        if self.session is None:
            raise RuntimeError("Registration session has not been started")
        if label not in self.session.raw_points_by_label:
            raise ValueError(f"Unknown registration label: {label}")

        measurement_tool = self.tracker_manager.get_latest_tool(self.session.measurement_tool_id)
        if measurement_tool is None:
            raise RuntimeError(f"No tracker sample available for tool {self.session.measurement_tool_id}")
        if not measurement_tool.valid:
            raise RuntimeError(
                f"Latest sample for {self.session.measurement_tool_id} is not valid ({measurement_tool.status})"
            )

        measurement_pose = self._tracker_tool_to_pose_sample(measurement_tool)
        point_sample = self._measurement_point_from_pose(measurement_tool.quaternion, measurement_tool.translation_mm)

        self.session.raw_points_by_label[label].append(point_sample)
        self.session.raw_measurement_tool_samples_by_label[label].append(self._pose_sample_to_dict(measurement_pose))
        if self.assets is not None:
            coil_tool = self.tracker_manager.get_latest_tool(self.session.coil_tool_id)
            if coil_tool is None:
                raise RuntimeError(f"No tracker sample available for coil tool {self.session.coil_tool_id}")
            if not coil_tool.valid:
                raise RuntimeError(f"Latest sample for {self.session.coil_tool_id} is not valid ({coil_tool.status})")
            coil_pose = self._tracker_tool_to_pose_sample(coil_tool)
            self.session.raw_coil_samples_by_label[label].append(self._pose_sample_to_dict(coil_pose))
        return point_sample

    def complete_registration(self, config_used: dict, max_fre_mm: float | None = None) -> RegistrationResult:
        """Solve and persist registration using current session samples."""
        if self.session is None:
            raise RuntimeError("Registration session has not been started")

        if self.assets is not None and self.session.truth_points_in_sw_by_label:
            result = self._complete_legacy_compatible_registration(config_used=config_used, max_fre_mm=max_fre_mm)
            self.session = None
            return result

        result = self._complete_simple_registration(config_used=config_used, max_fre_mm=max_fre_mm)
        self.session = None
        return result

    def complete_registration_from_csv(
        self,
        registration_csv: Path,
        *,
        config_used: dict | None = None,
        max_fre_mm: float | None = None,
        measurement_tool_id: str | None = None,
        coil_tool_id: str | None = None,
    ) -> RegistrationResult:
        """Run the legacy-compatible registration solve directly from a saved Aurora CSV."""
        if self.assets is None:
            raise RuntimeError("Legacy-compatible registration requires configured asset paths")

        transforms = parse_aurora_csv(registration_csv)
        measurement_id = measurement_tool_id or self.capture_tool_id
        coil_id = coil_tool_id or self.coil_tool_id
        if measurement_id not in transforms:
            raise RuntimeError(f"Aurora CSV {registration_csv} is missing measurement tool {measurement_id}")
        if coil_id not in transforms:
            raise RuntimeError(f"Aurora CSV {registration_csv} is missing coil tool {coil_id}")

        result = solve_registration_from_tool_samples(
            assets=self.assets,
            measurement_tool_samples=transforms[measurement_id],
            coil_tool_samples=transforms[coil_id],
            repetitions=self._expected_repetitions_from_csv(transforms[measurement_id]),
            measurement_tool_id=measurement_id,
            coil_tool_id=coil_id,
            solver=self.solver,
            quaternion_average_method=self.quaternion_average_method,
            model_tre_reference_radius_mm=self.model_tre_reference_radius_mm,
            tip_tre_reference_radius_mm=self.tip_tre_reference_radius_mm,
        )
        self._validate_fre_limits(result.validation_metrics, max_fre_mm)
        record = self._record_from_legacy_result(
            result=result,
            config_used=(config_used or {}) | {
                "registration_csv": str(registration_csv),
            },
        )
        output_path = self.repository.save_record(record)
        return RegistrationResult(record=record, output_path=output_path)

    def _complete_legacy_compatible_registration(self, *, config_used: dict, max_fre_mm: float | None) -> RegistrationResult:
        if self.session is None or self.assets is None:
            raise RuntimeError("Legacy-compatible registration session is not active")

        measurement_samples = self._flatten_pose_samples(self.session.raw_measurement_tool_samples_by_label, self.session.labels)
        coil_samples = self._flatten_pose_samples(self.session.raw_coil_samples_by_label, self.session.labels)
        self._validate_session_counts(self.session)

        result = solve_registration_from_tool_samples(
            assets=self.assets,
            measurement_tool_samples=measurement_samples,
            coil_tool_samples=coil_samples,
            repetitions=self.session.captures_per_landmark,
            measurement_tool_id=self.session.measurement_tool_id,
            coil_tool_id=self.session.coil_tool_id,
            solver=self.solver,
            quaternion_average_method=self.quaternion_average_method,
            model_tre_reference_radius_mm=self.model_tre_reference_radius_mm,
            tip_tre_reference_radius_mm=self.tip_tre_reference_radius_mm,
        )
        self._validate_fre_limits(result.validation_metrics, max_fre_mm)
        record = self._record_from_legacy_result(result=result, config_used=config_used)
        output_path = self.repository.save_record(record)
        return RegistrationResult(record=record, output_path=output_path)

    def _record_from_legacy_result(self, *, result: LegacyRegistrationResult, config_used: dict) -> RegistrationRecord:
        return RegistrationRecord(
            timestamp_utc=datetime.now(timezone.utc).isoformat(),
            landmark_labels=result.ordered_labels,
            raw_captured_landmarks_robot_xyz=result.raw_points_by_label,
            averaged_landmarks_robot_xyz=result.averaged_points_by_label,
            residuals_robot_xyz_mm=result.residuals_by_label,
            fre_mm=float(result.validation_metrics["overall_fre_mm"]),
            T_robot_aurora=result.T_aurora_2_model.tolist(),
            T_coil_tip=result.T_coil_tip.tolist(),
            T_aurora_2_tip=result.T_aurora_2_tip.tolist(),
            measurement_tool_id=result.measurement_tool_id,
            coil_tool_id=result.coil_tool_id,
            raw_measurement_tool_samples_by_label=result.raw_measurement_tool_samples_by_label,
            raw_coil_samples_by_label=result.raw_coil_samples_by_label,
            truth_points_in_sw_by_label=result.truth_points_in_sw_by_label,
            group_by_label=result.group_by_label,
            validation_metrics=result.validation_metrics,
            config_used=config_used
            | {
                "measurement_tool_id": result.measurement_tool_id,
                "coil_tool_id": result.coil_tool_id,
                "captures_per_landmark": result.repetitions,
                "quaternion_average_method": self.quaternion_average_method,
                "registration_mode": "legacy_compatible",
                "model_points_file": str(self.assets.paths.model_points_file) if self.assets is not None else None,
                "tip_points_file": str(self.assets.paths.tip_points_file) if self.assets is not None else None,
                "T_sw_2_model_file": str(self.assets.paths.T_sw_2_model_file) if self.assets is not None else None,
                "T_sw_2_tip_file": str(self.assets.paths.T_sw_2_tip_file) if self.assets is not None else None,
                "penprobe_file": str(self.assets.paths.penprobe_file) if self.assets is not None else None,
            },
        )

    def _complete_simple_registration(self, *, config_used: dict, max_fre_mm: float | None) -> RegistrationResult:
        if self.session is None:
            raise RuntimeError("Registration session has not been started")

        labels = self.session.labels
        captures = self.session.raw_points_by_label

        for label in labels:
            points = captures.get(label, [])
            if len(points) < self.session.captures_per_landmark:
                raise RuntimeError(
                    f"Label {label} has {len(points)} capture(s); requires {self.session.captures_per_landmark}"
                )
            if label not in self.nominal_landmarks_robot_xyz_mm:
                raise RuntimeError(f"Missing nominal landmark for label {label}")

        averaged_measured_aurora: dict[str, list[float]] = {}
        for label in labels:
            points = np.asarray(captures[label], dtype=float)
            averaged_measured_aurora[label] = points.mean(axis=0).tolist()

        measured = np.asarray([averaged_measured_aurora[label] for label in labels], dtype=float)
        truth = np.asarray([self.nominal_landmarks_robot_xyz_mm[label] for label in labels], dtype=float)

        T_robot_aurora = self.solver.solve_T_robot_aurora(measured, truth)
        transformed = self.solver.apply_transform(T_robot_aurora, measured)
        residuals = truth - transformed

        residuals_by_label = {
            label: residuals[idx, :].tolist()
            for idx, label in enumerate(labels)
        }
        fre_mm = float(compute_fre_mm(list(residuals_by_label.values())))
        if max_fre_mm is not None and fre_mm > max_fre_mm:
            raise RuntimeError(f"Registration FRE {fre_mm:.3f} mm exceeds limit {max_fre_mm:.3f} mm")

        T_coil_tip = self._load_existing_T_coil_tip()
        record = RegistrationRecord(
            timestamp_utc=datetime.now(timezone.utc).isoformat(),
            landmark_labels=labels,
            raw_captured_landmarks_robot_xyz=captures,
            averaged_landmarks_robot_xyz=averaged_measured_aurora,
            residuals_robot_xyz_mm=residuals_by_label,
            fre_mm=float(fre_mm),
            T_robot_aurora=T_robot_aurora.tolist(),
            T_coil_tip=T_coil_tip.tolist(),
            measurement_tool_id=self.capture_tool_id,
            coil_tool_id=self.coil_tool_id,
            raw_measurement_tool_samples_by_label=self.session.raw_measurement_tool_samples_by_label,
            raw_coil_samples_by_label=self.session.raw_coil_samples_by_label,
            validation_metrics={"overall_fre_mm": fre_mm, "registration_mode": "simple"},
            config_used=config_used | {"registration_mode": "simple"},
        )

        output_path = self.repository.save_record(record)
        return RegistrationResult(record=record, output_path=output_path)

    def _measurement_point_from_pose(
        self,
        quat_wxyz: tuple[float, float, float, float],
        translation_mm: tuple[float, float, float],
    ) -> list[float]:
        T_aurora_tool = make_transform_A_B(quat_wxyz, translation_mm)
        if self.assets is not None:
            T_aurora_point = compose_T_A_C(T_aurora_tool, self.assets.T_measurement_point)
        else:
            T_aurora_point = compose_T_A_C(T_aurora_tool, self.capture_tool_tip_transform)
        return [float(v) for v in T_aurora_point[0:3, 3]]

    def _load_existing_T_coil_tip(self) -> np.ndarray:
        latest = self.repository.root_dir / "latest_registration.json"
        if not latest.exists():
            return np.eye(4)

        try:
            payload = json.loads(latest.read_text(encoding="utf-8"))
            if "T_coil_tip" in payload:
                return np.asarray(payload["T_coil_tip"], dtype=float)
            if "T_tip_2_coil" in payload:
                return np.linalg.inv(np.asarray(payload["T_tip_2_coil"], dtype=float))
        except Exception:
            return np.eye(4)

        return np.eye(4)

    def _validate_session_counts(self, session: RegistrationSession) -> None:
        for label in session.labels:
            count = len(session.raw_points_by_label.get(label, []))
            if count != session.captures_per_landmark:
                raise RuntimeError(
                    f"Label {label} has {count} capture(s); expected exactly {session.captures_per_landmark}"
                )
            measurement_count = len(session.raw_measurement_tool_samples_by_label.get(label, []))
            if measurement_count != session.captures_per_landmark:
                raise RuntimeError(
                    f"Label {label} has {measurement_count} measurement-tool pose(s); "
                    f"expected exactly {session.captures_per_landmark}"
                )
            coil_count = len(session.raw_coil_samples_by_label.get(label, []))
            if coil_count != session.captures_per_landmark:
                raise RuntimeError(
                    f"Label {label} has {coil_count} coil-tool pose(s); expected exactly {session.captures_per_landmark}"
                )

    def _expected_repetitions_from_csv(self, measurement_samples: list[AuroraPoseSample]) -> int:
        if self.assets is None:
            raise RuntimeError("Legacy-compatible CSV registration requires configured asset paths")
        total_measurements = len(measurement_samples)
        if total_measurements <= 0:
            raise ValueError("Aurora CSV does not contain any measurement-tool samples")
        truth_point_count = self.assets.model_truth_in_sw.shape[1] + self.assets.tip_truth_in_sw.shape[1]
        if truth_point_count <= 0:
            raise RuntimeError("Configured registration assets do not contain any truth points")
        if total_measurements % truth_point_count != 0:
            raise ValueError(
                "Measurement-tool sample count does not divide evenly into the configured truth-point count "
                f"({truth_point_count}). Got {total_measurements} samples"
            )
        return total_measurements // truth_point_count

    @staticmethod
    def _validate_fre_limits(validation_metrics: dict, max_fre_mm: float | None) -> None:
        if max_fre_mm is None:
            return
        offending = {
            key: float(validation_metrics[key])
            for key in ("model_fre_mm", "tip_fre_mm", "overall_fre_mm")
            if key in validation_metrics and validation_metrics[key] is not None and float(validation_metrics[key]) > max_fre_mm
        }
        if offending:
            rendered = ", ".join(f"{key}={value:.3f}" for key, value in offending.items())
            raise RuntimeError(f"Registration FRE exceeds limit {max_fre_mm:.3f} mm: {rendered}")

    @staticmethod
    def _tracker_tool_to_pose_sample(tool) -> AuroraPoseSample:
        return AuroraPoseSample(
            tool_id=tool.tool_id,
            quaternion_wxyz=tuple(float(v) for v in tool.quaternion),
            translation_mm=tuple(float(v) for v in tool.translation_mm),
            quality=float(tool.quality) if tool.quality is not None else None,
            source_row=int(tool.frame_number) if tool.frame_number is not None else None,
            source_token=str(getattr(tool, "timestamp", "")),
        )

    @staticmethod
    def _pose_sample_to_dict(sample: AuroraPoseSample) -> dict:
        return {
            "tool_id": sample.tool_id,
            "quaternion_wxyz": list(sample.quaternion_wxyz),
            "translation_mm": list(sample.translation_mm),
            "quality": sample.quality,
            "source_row": sample.source_row,
            "source_token": sample.source_token,
        }

    @staticmethod
    def _flatten_pose_samples(
        grouped_samples: dict[str, list[dict]],
        ordered_labels: list[str],
    ) -> list[AuroraPoseSample]:
        output: list[AuroraPoseSample] = []
        for label in ordered_labels:
            for raw in grouped_samples.get(label, []):
                output.append(
                    AuroraPoseSample(
                        tool_id=str(raw["tool_id"]),
                        quaternion_wxyz=tuple(float(v) for v in raw["quaternion_wxyz"]),
                        translation_mm=tuple(float(v) for v in raw["translation_mm"]),
                        quality=float(raw["quality"]) if raw.get("quality") is not None else None,
                        source_row=int(raw["source_row"]) if raw.get("source_row") is not None else None,
                        source_token=str(raw.get("source_token", "")),
                    )
                )
        return output

    @staticmethod
    def _coerce_transform(transform: list[list[float]] | None) -> np.ndarray:
        if transform is None:
            return np.eye(4)
        matrix = np.asarray(transform, dtype=float)
        if matrix.shape != (4, 4):
            raise ValueError("capture_tool_tip_transform must be 4x4")
        return matrix

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from continuum_robot.data_management import (
    apply_migration,
    delete_managed_items,
    discover_managed_data,
    preview_migration,
)
from continuum_robot.experiments.dataset_io import (
    ExperimentDatasetWriter,
    canonical_timestamped_name,
)
from continuum_robot.experiments.schemas import ExperimentMetadata, ExperimentSummary, ExperimentTimeseriesSample
from continuum_robot.gui.controllers.data_management_controller import DataManagementController
from continuum_robot.registration.repository import RegistrationRecord, RegistrationRepository
from continuum_robot.registration.runtime_tip_repository import (
    RuntimeTipCalibrationRecord,
    RuntimeTipCalibrationRepository,
)
from continuum_robot.servos.neutral_calibration_service import (
    NeutralCalibrationService,
    ServoCalibrationContext,
)


def test_canonical_timestamped_name_uses_suffix_only_on_collision(tmp_path: Path) -> None:
    first = canonical_timestamped_name(tmp_path, "registration_validation", timestamp_utc="2026-04-22T15:30:11Z")
    (tmp_path / first).mkdir()
    second = canonical_timestamped_name(tmp_path, "registration_validation", timestamp_utc="2026-04-22T15:30:11Z")

    assert first == "20260422_153011_registration_validation"
    assert second == "20260422_153011_registration_validation_01"


def test_registration_and_runtime_tip_repositories_use_canonical_timestamp_names(tmp_path: Path) -> None:
    registration_repo = RegistrationRepository(root_dir=tmp_path / "data" / "registrations")
    registration_path = registration_repo.save_record(
        RegistrationRecord(
            timestamp_utc="2026-04-22T16:00:00Z",
            landmark_labels=["P1", "P2", "P3", "P4"],
            raw_captured_landmarks_robot_xyz={},
            averaged_landmarks_robot_xyz={},
            residuals_robot_xyz_mm={},
            fre_mm=0.25,
            T_robot_aurora=np.eye(4).tolist(),
            T_coil_tip=np.eye(4).tolist(),
            config_used={"registration_mode": "simple"},
        )
    )

    runtime_tip_repo = RuntimeTipCalibrationRepository(
        latest_path=tmp_path / "data" / "runtime_tip_calibration" / "latest_runtime_tip_calibration.json"
    )
    quick_path = runtime_tip_repo.save_record(
        RuntimeTipCalibrationRecord(
            timestamp_utc="2026-04-22T16:05:00Z",
            calibration_kind="runtime_tip_calibration_quick_4_point",
            measurement_tool_id="0B",
            coil_tool_id="0A",
            setup_id="bench-a",
            truth_points_in_sw_by_label={"T01": [0.0, 0.0, 0.0]},
            truth_points_in_tip_by_label={"T01": [0.0, 0.0, 0.0]},
            raw_captured_hat_points_aurora_xyz_by_label={"T01": [[0.0, 0.0, 0.0]]},
            averaged_hat_points_aurora_xyz_by_label={"T01": [0.0, 0.0, 0.0]},
            raw_coil_samples=[],
            residuals_tip_xyz_mm_by_label={"T01": [0.0, 0.0, 0.0]},
            fit_rmse_mm=0.0,
            T_tip_aurora=np.eye(4).tolist(),
            T_aurora_tip=np.eye(4).tolist(),
            T_aurora_coil_avg=np.eye(4).tolist(),
            T_coil_tip=np.eye(4).tolist(),
            config_used={},
        ),
        alias_path=runtime_tip_repo.quick_latest_path,
    )

    assert registration_path.name == "20260422_160000_registration.json"
    assert quick_path.name == "20260422_160500_runtime_tip_quick_4_point.json"


def test_neutral_calibration_archives_use_canonical_timestamp_names(tmp_path: Path) -> None:
    service = NeutralCalibrationService(
        path=tmp_path / "config" / "neutral_setpoints.json",
        context=ServoCalibrationContext(
            robot_mode="4-servo",
            robot_config_name="robot_4servo.yaml",
            servo_ids=[1],
            tendon_to_servo=[1],
            position_min_offset_ticks=-50,
            position_max_offset_ticks=50,
            default_pretension_current_threshold_ma=220,
            tightening_rotation_by_servo={1: "cw"},
        ),
    )

    service.save_neutral_setpoints({1: 100})
    service.save_neutral_setpoints({1: 200})

    archives = sorted((tmp_path / "config").glob("*.json"))
    archive_names = [path.name for path in archives if path.name != "neutral_setpoints.json"]
    assert len(archive_names) == 1
    assert archive_names[0].endswith("_neutral_setpoints.json")


def test_discover_managed_data_covers_all_categories_and_protects_active_aliases(tmp_path: Path) -> None:
    registration_repo = RegistrationRepository(root_dir=tmp_path / "data" / "registrations")
    registration_repo.save_record(
        RegistrationRecord(
            timestamp_utc="2026-04-22T16:00:00Z",
            landmark_labels=["P1", "P2", "P3", "P4"],
            raw_captured_landmarks_robot_xyz={},
            averaged_landmarks_robot_xyz={},
            residuals_robot_xyz_mm={},
            fre_mm=0.25,
            T_robot_aurora=np.eye(4).tolist(),
            T_coil_tip=np.eye(4).tolist(),
            config_used={"registration_mode": "simple"},
        )
    )
    runtime_tip_repo = RuntimeTipCalibrationRepository(
        latest_path=tmp_path / "data" / "runtime_tip_calibration" / "latest_runtime_tip_calibration.json"
    )
    runtime_tip_repo.save_record(
        RuntimeTipCalibrationRecord(
            timestamp_utc="2026-04-22T16:05:00Z",
            calibration_kind="runtime_tip_calibration_hat",
            measurement_tool_id="0B",
            coil_tool_id="0A",
            setup_id="bench-a",
            truth_points_in_sw_by_label={"T01": [0.0, 0.0, 0.0]},
            truth_points_in_tip_by_label={"T01": [0.0, 0.0, 0.0]},
            raw_captured_hat_points_aurora_xyz_by_label={"T01": [[0.0, 0.0, 0.0]]},
            averaged_hat_points_aurora_xyz_by_label={"T01": [0.0, 0.0, 0.0]},
            raw_coil_samples=[],
            residuals_tip_xyz_mm_by_label={"T01": [0.0, 0.0, 0.0]},
            fit_rmse_mm=0.0,
            T_tip_aurora=np.eye(4).tolist(),
            T_aurora_tip=np.eye(4).tolist(),
            T_aurora_coil_avg=np.eye(4).tolist(),
            T_coil_tip=np.eye(4).tolist(),
            config_used={},
        )
    )
    neutral = NeutralCalibrationService(
        path=tmp_path / "config" / "neutral_setpoints.json",
        context=ServoCalibrationContext(
            robot_mode="4-servo",
            robot_config_name="robot_4servo.yaml",
            servo_ids=[1],
            tendon_to_servo=[1],
            position_min_offset_ticks=-50,
            position_max_offset_ticks=50,
            default_pretension_current_threshold_ma=220,
            tightening_rotation_by_servo={1: "cw"},
        ),
    )
    neutral.save_neutral_setpoints({1: 150})

    writer = ExperimentDatasetWriter(output_root=tmp_path / "data" / "experiments")
    writer.write_dataset(
        ExperimentMetadata(
            schema_version="1.0",
            experiment_name="registration_validation",
            run_id="registration-validation-1",
            timestamp_utc="2026-04-22T16:10:00Z",
            git_commit="deadbeef",
            backend_info={},
            registration_info={},
            config_used={},
        ),
        [
            ExperimentTimeseriesSample(
                monotonic_time_s=0.0,
                wall_time_utc="2026-04-22T16:10:00Z",
                phase="summary",
                step_index=0,
                sample_index=0,
            )
        ],
        ExperimentSummary(
            schema_version="1.0",
            experiment_name="registration_validation",
            run_id="registration-validation-1",
            success=True,
            sample_counts={"total": 1},
            dropped_frames=0,
            invalid_transforms=0,
            stage_pass_fail={"analysis": "pass"},
            experiment_metrics={"sample_count": 1},
        ),
    )

    artifact_dir = tmp_path / "data" / "models" / "ann" / "20260422_161512_ann_training"
    artifact_dir.mkdir(parents=True)
    (artifact_dir / "training_metadata.json").write_text(
        json.dumps(
            {
                "created_at_utc": "2026-04-22T16:15:12Z",
                "status": "completed",
                "dataset": {"run_name": "collect_pose_command_dataset"},
                "backend": {"selected_backend": "mps"},
                "training": {"epochs_completed": 12, "best_validation_loss": 0.123},
                "files": {"model_path": str(artifact_dir / "model.pt")},
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    (artifact_dir / "model.pt").write_text("stub", encoding="utf-8")

    modeling_result_dir = tmp_path / "data" / "modeling_results" / "20260422_161700_workspace_coverage"
    modeling_result_dir.mkdir(parents=True)
    (modeling_result_dir / "summary.json").write_text(json.dumps({"status": "success"}, indent=2), encoding="utf-8")
    (modeling_result_dir / "evaluation_metadata.json").write_text(
        json.dumps(
            {
                "created_at_utc": "2026-04-22T16:17:00Z",
                "dataset": {"run_name": "workspace_coverage"},
                "evaluation_scope_used": "full_dataset",
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    diag_dir = tmp_path / "data" / "diagnostics" / "servo_telemetry" / "20260422_161900_servo_telemetry_live_4servos"
    diag_dir.mkdir(parents=True)
    (diag_dir / "summary.json").write_text(
        json.dumps({"metadata": {"created_at_utc": "2026-04-22T16:19:00Z", "baudrate": 1000000}, "results": [{"profile_name": "live"}]}, indent=2),
        encoding="utf-8",
    )

    items = discover_managed_data(tmp_path)
    categories = {item.category_key for item in items}

    assert {"calibration", "experiments", "modeling", "diagnostics"} <= categories
    latest_registration = next(item for item in items if item.readable_name == "Latest Accepted Registration")
    neutral_active = next(item for item in items if item.readable_name == "Active Servo Calibration")
    ann_artifact = next(item for item in items if item.item_type == "ann_artifact")
    diagnostics_item = next(item for item in items if item.item_type == "servo_telemetry")

    assert latest_registration.deletable is False
    assert neutral_active.deletable is False
    assert ann_artifact.details.startswith("collect_pose_command_dataset")
    assert "1000000 baud" in diagnostics_item.details


def test_discover_managed_data_normalizes_legacy_display_and_paths(tmp_path: Path) -> None:
    legacy_path = tmp_path / "data" / "registrations" / "registration_20260422T160000_000001Z.json"
    legacy_path.parent.mkdir(parents=True)
    legacy_path.write_text(
        json.dumps(
            {
                "timestamp_utc": "2026-04-22T16:00:00Z",
                "fre_mm": 0.25,
                "T_robot_aurora": np.eye(4).tolist(),
                "T_coil_tip": np.eye(4).tolist(),
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    item = next(entry for entry in discover_managed_data(tmp_path) if entry.path == legacy_path)

    assert item.readable_name == "Registration"
    assert item.timestamp_label == "20260422_160000"
    assert item.original_name == "registration_20260422T160000_000001Z.json"
    assert item.original_path.endswith("registration_20260422T160000_000001Z.json")
    assert item.legacy_naming is True
    assert item.canonical_name == "20260422_160000_registration.json"


def test_data_management_controller_filters_and_deletes_selected_items(tmp_path: Path) -> None:
    repo = RegistrationRepository(root_dir=tmp_path / "data" / "registrations")
    record_path = repo.save_record(
        RegistrationRecord(
            timestamp_utc="2026-04-22T16:00:00Z",
            landmark_labels=["P1", "P2", "P3", "P4"],
            raw_captured_landmarks_robot_xyz={},
            averaged_landmarks_robot_xyz={},
            residuals_robot_xyz_mm={},
            fre_mm=0.25,
            T_robot_aurora=np.eye(4).tolist(),
            T_coil_tip=np.eye(4).tolist(),
            config_used={"registration_mode": "simple"},
        )
    )
    controller = DataManagementController(project_root=tmp_path)

    state = controller.refresh()
    assert any(item.path == record_path for item in state.items)

    controller.set_category_filter("calibration")
    controller.set_search_text("registration")
    state = controller.refresh()
    assert all(item.category_key == "calibration" for item in state.filtered_items)

    controller.set_selected_paths([str(record_path)])
    deleted = controller.delete_selected()
    state = controller.refresh()

    assert deleted == [record_path]
    assert record_path.exists() is False
    assert all(item.path != record_path for item in state.items)


def test_preview_migration_detects_legacy_candidates_and_writes_ledger(tmp_path: Path) -> None:
    legacy_path = tmp_path / "data" / "registrations" / "registration_20260422T160000_000001Z.json"
    legacy_path.parent.mkdir(parents=True)
    legacy_path.write_text(
        json.dumps(
            {
                "timestamp_utc": "2026-04-22T16:00:00Z",
                "fre_mm": 0.25,
                "T_robot_aurora": np.eye(4).tolist(),
                "T_coil_tip": np.eye(4).tolist(),
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    items = discover_managed_data(tmp_path)
    report = preview_migration(tmp_path, items)

    assert report.candidate_count == 1
    assert report.manifest_path is not None and report.manifest_path.exists()
    assert report.summary_path is not None and report.summary_path.exists()
    candidate = report.actionable_entries[0]
    assert candidate.source_path == legacy_path
    assert candidate.target_path == tmp_path / "data" / "registrations" / "20260422_160000_registration.json"


def test_apply_migration_moves_legacy_artifact_and_post_migration_discovery_is_clean(tmp_path: Path) -> None:
    legacy_path = tmp_path / "data" / "experiments" / "pivot_validation" / "pivot_validation_20260422T161000Z"
    legacy_path.mkdir(parents=True)
    (legacy_path / "metadata.json").write_text(
        json.dumps(
            {
                "schema_version": "1.0",
                "experiment_name": "pivot_validation",
                "run_id": "pivot-validation-1",
                "timestamp_utc": "2026-04-22T16:10:00Z",
                "git_commit": "deadbeef",
                "backend_info": {},
                "registration_info": {},
                "config_used": {},
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    (legacy_path / "summary.json").write_text(
        json.dumps(
            {
                "schema_version": "1.0",
                "experiment_name": "pivot_validation",
                "run_id": "pivot-validation-1",
                "success": True,
                "sample_counts": {"total": 1},
                "dropped_frames": 0,
                "invalid_transforms": 0,
                "stage_pass_fail": {"analysis": "pass"},
                "status": "success",
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    report = apply_migration(tmp_path, discover_managed_data(tmp_path))
    migrated_path = tmp_path / "data" / "experiments" / "pivot_validation" / "20260422_161000_pivot_validation"

    assert report.applied_count == 1
    assert migrated_path.exists() is True
    assert legacy_path.exists() is False
    migrated_item = next(entry for entry in discover_managed_data(tmp_path) if entry.path == migrated_path)
    assert migrated_item.is_legacy is False


def test_preview_migration_skips_protected_aliases(tmp_path: Path) -> None:
    repo = RegistrationRepository(root_dir=tmp_path / "data" / "registrations")
    repo.save_record(
        RegistrationRecord(
            timestamp_utc="2026-04-22T16:00:00Z",
            landmark_labels=["P1", "P2", "P3", "P4"],
            raw_captured_landmarks_robot_xyz={},
            averaged_landmarks_robot_xyz={},
            residuals_robot_xyz_mm={},
            fre_mm=0.25,
            T_robot_aurora=np.eye(4).tolist(),
            T_coil_tip=np.eye(4).tolist(),
            config_used={"registration_mode": "simple"},
        )
    )

    report = preview_migration(tmp_path, discover_managed_data(tmp_path))

    assert any("Protected" in entry.reason or "protected" in entry.reason.lower() for entry in report.entries)
    assert all(entry.status != "candidate" for entry in report.entries if entry.source_path.name == "latest_registration.json")


def test_delete_managed_items_rejects_protected_aliases(tmp_path: Path) -> None:
    repo = RegistrationRepository(root_dir=tmp_path / "data" / "registrations")
    repo.save_record(
        RegistrationRecord(
            timestamp_utc="2026-04-22T16:00:00Z",
            landmark_labels=["P1", "P2", "P3", "P4"],
            raw_captured_landmarks_robot_xyz={},
            averaged_landmarks_robot_xyz={},
            residuals_robot_xyz_mm={},
            fre_mm=0.25,
            T_robot_aurora=np.eye(4).tolist(),
            T_coil_tip=np.eye(4).tolist(),
            config_used={"registration_mode": "simple"},
        )
    )

    latest_item = next(item for item in discover_managed_data(tmp_path) if item.readable_name == "Latest Accepted Registration")

    try:
        delete_managed_items(tmp_path, [latest_item])
    except ValueError as exc:
        assert "Active calibration alias" in str(exc)
    else:
        raise AssertionError("Expected protected alias deletion to be rejected.")

import json
from pathlib import Path

import numpy as np

from continuum_robot.registration.legacy_compat import RegistrationAssetPaths, load_registration_assets
from continuum_robot.registration.live_registration_service import LiveRegistrationService
from continuum_robot.registration.repository import RegistrationRepository
from continuum_robot.registration.rigid_solver import RigidRegistrationSolver
from continuum_robot.registration.validation_tools import (
    compare_registration_outputs,
    evaluate_runtime_sanity_from_capture,
    evaluate_runtime_sanity_live,
    load_pose_samples_from_saved_session,
    load_registration_output,
    run_registration_validation_from_csv,
    save_validation_report,
)
from tests.fixtures.aurora_samples import build_valid_transform_frame
from tests.test_live_registration_service import _FakeTrackerManager


def _asset_paths() -> RegistrationAssetPaths:
    repo_root = Path(__file__).resolve().parents[1]
    return RegistrationAssetPaths(
        model_points_file=repo_root / "tools" / "12_model_registration_points_in_sw",
        tip_points_file=repo_root / "tools" / "all_tip_registration_points_in_sw",
        T_sw_2_model_file=repo_root / "tools" / "T_sw_2_model",
        T_sw_2_tip_file=repo_root / "tools" / "T_sw_2_tip",
        penprobe_file=repo_root / "tools" / "penprobe_08_09_24c",
    )


def test_compare_registration_outputs_matches_validation_report_to_saved_registration(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    assets = load_registration_assets(_asset_paths())
    service = LiveRegistrationService(
        tracker_manager=_FakeTrackerManager([]),
        repository=RegistrationRepository(root_dir=tmp_path / "registrations"),
        solver=RigidRegistrationSolver(),
        asset_paths=_asset_paths(),
        measurement_tool_id="0A",
        coil_tool_id="0B",
    )
    registration_result = service.complete_registration_from_csv(
        repo_root / "references" / "RegistrationPoints.csv",
        config_used={"test": True},
    )
    validation_report = run_registration_validation_from_csv(
        registration_csv=repo_root / "references" / "RegistrationPoints.csv",
        assets=assets,
        solver=RigidRegistrationSolver(),
        measurement_tool_id="0A",
        coil_tool_id="0B",
        quaternion_average_method="sign_aligned_mean",
        model_tre_reference_radius_mm=5.0,
        tip_tre_reference_radius_mm=3.0,
    )
    report_path = save_validation_report(validation_report, tmp_path / "validation_report.json")

    comparison = compare_registration_outputs(
        load_registration_output(report_path),
        load_registration_output(registration_result.output_path),
        translation_tolerance_mm=1e-9,
        rotation_tolerance_deg=1e-9,
        fre_tolerance_mm=1e-9,
    )

    assert comparison.passed is True
    assert all(item.within_tolerance for item in comparison.transform_comparisons if item.compared)


def test_load_pose_samples_from_saved_session_detects_tool_role_mismatch(tmp_path: Path) -> None:
    session_path = tmp_path / "session.json"
    session_path.write_text(
        json.dumps(
            {
                "landmark_labels": ["M01"],
                "measurement_tool_id": "0A",
                "coil_tool_id": "0B",
                "raw_measurement_tool_samples_by_label": {
                    "M01": [
                        {
                            "tool_id": "0A",
                            "quaternion_wxyz": [1.0, 0.0, 0.0, 0.0],
                            "translation_mm": [0.0, 0.0, 0.0],
                        }
                    ]
                },
                "raw_coil_samples_by_label": {
                    "M01": [
                        {
                            "tool_id": "0B",
                            "quaternion_wxyz": [1.0, 0.0, 0.0, 0.0],
                            "translation_mm": [0.0, 0.0, 0.0],
                        }
                    ]
                },
            }
        ),
        encoding="utf-8",
    )

    try:
        load_pose_samples_from_saved_session(
            session_path,
            measurement_tool_id="0B",
            coil_tool_id="0B",
            expected_ordered_labels=["M01"],
        )
    except ValueError as exc:
        assert "Tool-role mismatch" in str(exc)
    else:
        raise AssertionError("Expected tool-role mismatch to raise ValueError")


def test_load_pose_samples_from_saved_session_detects_repetition_mismatch(tmp_path: Path) -> None:
    session_path = tmp_path / "session.json"
    session_path.write_text(
        json.dumps(
            {
                "landmark_labels": ["M01", "M02"],
                "measurement_tool_id": "0A",
                "coil_tool_id": "0B",
                "raw_measurement_tool_samples_by_label": {
                    "M01": [
                        {
                            "tool_id": "0A",
                            "quaternion_wxyz": [1.0, 0.0, 0.0, 0.0],
                            "translation_mm": [0.0, 0.0, 0.0],
                        },
                        {
                            "tool_id": "0A",
                            "quaternion_wxyz": [1.0, 0.0, 0.0, 0.0],
                            "translation_mm": [0.0, 0.0, 0.0],
                        },
                    ],
                    "M02": [
                        {
                            "tool_id": "0A",
                            "quaternion_wxyz": [1.0, 0.0, 0.0, 0.0],
                            "translation_mm": [1.0, 0.0, 0.0],
                        }
                    ],
                },
                "raw_coil_samples_by_label": {
                    "M01": [
                        {
                            "tool_id": "0B",
                            "quaternion_wxyz": [1.0, 0.0, 0.0, 0.0],
                            "translation_mm": [0.0, 0.0, 0.0],
                        },
                        {
                            "tool_id": "0B",
                            "quaternion_wxyz": [1.0, 0.0, 0.0, 0.0],
                            "translation_mm": [0.0, 0.0, 0.0],
                        },
                    ],
                    "M02": [
                        {
                            "tool_id": "0B",
                            "quaternion_wxyz": [1.0, 0.0, 0.0, 0.0],
                            "translation_mm": [0.0, 0.0, 0.0],
                        }
                    ],
                },
            }
        ),
        encoding="utf-8",
    )

    try:
        load_pose_samples_from_saved_session(
            session_path,
            measurement_tool_id="0A",
            coil_tool_id="0B",
            expected_ordered_labels=["M01", "M02"],
        )
    except ValueError as exc:
        assert "repetition-count mismatch" in str(exc)
    else:
        raise AssertionError("Expected repetition mismatch to raise ValueError")


def test_runtime_sanity_reports_invalid_registration_transform(tmp_path: Path) -> None:
    registration_path = tmp_path / "bad_registration.json"
    registration_path.write_text(
        json.dumps(
            {
                "T_robot_aurora": np.eye(4).tolist(),
                "T_coil_tip": [[1.0, 0.0], [0.0, 1.0]],
                "coil_tool_id": "0A",
            }
        ),
        encoding="utf-8",
    )
    capture_path = tmp_path / "capture.jsonl"
    capture_path.write_text("", encoding="utf-8")

    report = evaluate_runtime_sanity_from_capture(
        registration_path=registration_path,
        capture_path=capture_path,
        expected_runtime_coil_tool_id="0A",
    )

    assert report.passed is False
    assert report.registration_state == "invalid_registration"
    assert report.tip_pose_status == "invalid_registration"


def test_runtime_sanity_live_reports_missing_registration_file(tmp_path: Path) -> None:
    class _UnusedTrackingService:
        def start(self) -> None:  # pragma: no cover - should not be reached
            raise AssertionError("tracking should not start when registration is missing")

    missing_path = tmp_path / "missing_registration.json"
    report = evaluate_runtime_sanity_live(
        tracking_service=_UnusedTrackingService(),
        registration_path=missing_path,
        expected_runtime_coil_tool_id="0A",
    )

    assert report.passed is False
    assert report.registration_state == "missing_registration"
    assert report.tip_pose_status == "missing_registration"
    assert "Missing registration file" in (report.last_error or "")

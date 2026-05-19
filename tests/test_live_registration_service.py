from __future__ import annotations
from pathlib import Path

import numpy as np

from continuum_robot.registration.live_registration_service import LiveRegistrationService
from continuum_robot.registration.repository import RegistrationRepository
from continuum_robot.registration.rigid_solver import RigidRegistrationSolver
from continuum_robot.tracking.runtime_models import TrackerToolState


class _FakeTrackerManager:
    def __init__(self, samples: list[TrackerToolState]) -> None:
        self._samples = list(samples)

    def get_latest_tool(self, tool_id: str) -> TrackerToolState | None:
        if not self._samples:
            return None
        sample = self._samples.pop(0)
        if sample.tool_id != tool_id:
            return None
        return sample


def _sample(tool_id: str, xyz: tuple[float, float, float], frame: int) -> TrackerToolState:
    return TrackerToolState(
        tool_id=tool_id,
        frame_number=frame,
        valid=True,
        status="tracked",
        quaternion=(1.0, 0.0, 0.0, 0.0),
        translation_mm=xyz,
        quality=0.1,
        timestamp="2026-01-01T00:00:00.000Z",
    )


def test_live_registration_service_saves_registration(tmp_path: Path) -> None:
    labels = ["L1", "L2", "L3"]
    nominal = {
        "L1": [0.0, 0.0, 0.0],
        "L2": [10.0, 0.0, 0.0],
        "L3": [0.0, 10.0, 0.0],
    }

    samples = [
        _sample("0A", (0.0, 0.0, 0.0), 1),
        _sample("0A", (0.0, 0.0, 0.0), 2),
        _sample("0A", (10.0, 0.0, 0.0), 3),
        _sample("0A", (10.0, 0.0, 0.0), 4),
        _sample("0A", (0.0, 10.0, 0.0), 5),
        _sample("0A", (0.0, 10.0, 0.0), 6),
    ]

    service = LiveRegistrationService(
        tracker_manager=_FakeTrackerManager(samples),
        repository=RegistrationRepository(root_dir=tmp_path),
        solver=RigidRegistrationSolver(),
    )
    service.begin_session(
        labels=labels,
        captures_per_landmark=2,
        nominal_landmarks_robot_xyz_mm=nominal,
        capture_tool_id="0A",
    )

    service.capture_current_sample("L1")
    service.capture_current_sample("L1")
    service.capture_current_sample("L2")
    service.capture_current_sample("L2")
    service.capture_current_sample("L3")
    service.capture_current_sample("L3")

    result = service.complete_registration(config_used={"test": True})
    assert result.output_path.exists()
    assert np.allclose(np.asarray(result.record.T_robot_aurora), np.eye(4), atol=1e-6)

    latest = tmp_path / "latest_registration.json"
    assert latest.exists()


def test_live_registration_service_applies_capture_tool_tip_transform(tmp_path: Path) -> None:
    samples = [_sample("0B", (1.0, 2.0, 3.0), 1)]
    service = LiveRegistrationService(
        tracker_manager=_FakeTrackerManager(samples),
        repository=RegistrationRepository(root_dir=tmp_path),
        solver=RigidRegistrationSolver(),
    )
    service.begin_session(
        labels=["L1"],
        captures_per_landmark=1,
        nominal_landmarks_robot_xyz_mm={"L1": [11.0, 2.0, 3.0]},
        capture_tool_id="0B",
        capture_tool_tip_transform=[
            [1.0, 0.0, 0.0, 10.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
    )

    sample = service.capture_current_sample("L1")

    assert sample == [11.0, 2.0, 3.0]


def _square_grid_labels_and_truth() -> tuple[list[str], dict[str, list[float]]]:
    """4 outer + 4 inner landmarks on a plane, big enough to absorb one outlier."""
    truth = {
        "L1": [0.0, 35.0, -5.0],
        "L2": [-35.0, 0.0, -5.0],
        "L3": [0.0, -35.0, -5.0],
        "L4": [35.0, 0.0, -5.0],
        "L5": [0.0, 25.0, -5.0],
        "L6": [-25.0, 0.0, -5.0],
        "L7": [0.0, -25.0, -5.0],
        "L8": [25.0, 0.0, -5.0],
    }
    return list(truth.keys()), truth


def _capture_session_with_one_outlier(
    tmp_path: Path,
    *,
    ransac_outlier_rejection_enabled: bool,
    ransac_inlier_threshold_mm: float = 1.0,
) -> tuple[LiveRegistrationService, list[str]]:
    """Captures sit on truth except L5, which is offset by 5 mm in +X."""
    labels, truth = _square_grid_labels_and_truth()
    samples = []
    for idx, label in enumerate(labels, start=1):
        xyz = list(truth[label])
        if label == "L5":
            xyz[0] += 5.0
        samples.append(_sample("0A", tuple(xyz), idx))
    service = LiveRegistrationService(
        tracker_manager=_FakeTrackerManager(samples),
        repository=RegistrationRepository(root_dir=tmp_path),
        solver=RigidRegistrationSolver(),
        ransac_outlier_rejection_enabled=ransac_outlier_rejection_enabled,
        ransac_inlier_threshold_mm=ransac_inlier_threshold_mm,
        ransac_seed=7,
    )
    service.begin_session(
        labels=labels,
        captures_per_landmark=1,
        nominal_landmarks_robot_xyz_mm=truth,
        capture_tool_id="0A",
    )
    for label in labels:
        service.capture_current_sample(label)
    return service, labels


def test_live_registration_simple_path_is_poisoned_by_outlier(tmp_path: Path) -> None:
    """Without RANSAC, the bad L5 capture inflates FRE far above the noise floor."""
    service, _labels = _capture_session_with_one_outlier(
        tmp_path, ransac_outlier_rejection_enabled=False,
    )
    result = service.complete_registration(config_used={"test": True}, max_fre_mm=None)
    assert result.record.validation_metrics["registration_mode"] == "simple"
    assert result.record.fre_mm > 1.0  # outlier drags the whole fit


def test_live_registration_ransac_path_rejects_outlier(tmp_path: Path) -> None:
    """With RANSAC on, the bad L5 capture is identified and excluded from FRE."""
    service, _labels = _capture_session_with_one_outlier(
        tmp_path,
        ransac_outlier_rejection_enabled=True,
        ransac_inlier_threshold_mm=1.0,
    )
    result = service.complete_registration(config_used={"test": True}, max_fre_mm=None)
    metrics = result.record.validation_metrics
    assert metrics["registration_mode"] == "simple_ransac"
    ransac = metrics["ransac"]
    assert ransac["enabled"] is True
    assert ransac["converged"] is True
    assert "L5" in ransac["rejected_labels"]
    # FRE is computed over inliers only, so it should be near machine zero
    # because the remaining 7 landmarks sit exactly on truth.
    assert result.record.fre_mm < 0.01
    # Raw per-label residual for L5 is still recorded for the audit trail.
    l5_residual = result.record.residuals_robot_xyz_mm["L5"]
    assert abs(l5_residual[0]) > 1.0  # the +5 mm X offset shows in the residual

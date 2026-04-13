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

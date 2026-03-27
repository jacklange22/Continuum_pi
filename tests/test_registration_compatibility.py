import json
from pathlib import Path

import numpy as np

from continuum_robot.registration.repository import RegistrationRecord, RegistrationRepository
from continuum_robot.tracking.tip_pose_service import TipPoseService


def _identity() -> list[list[float]]:
    return np.eye(4).tolist()


def test_repository_save_record_writes_legacy_tip_key(tmp_path: Path) -> None:
    repo = RegistrationRepository(root_dir=tmp_path)
    record = RegistrationRecord(
        timestamp_utc="2026-01-01T00:00:00Z",
        landmark_labels=["A"],
        raw_captured_landmarks_robot_xyz={"A": [[0.0, 0.0, 0.0]]},
        averaged_landmarks_robot_xyz={"A": [0.0, 0.0, 0.0]},
        residuals_robot_xyz_mm={"A": [0.0, 0.0, 0.0]},
        fre_mm=0.0,
        T_robot_aurora=_identity(),
        T_coil_tip=_identity(),
        config_used={},
    )

    path = repo.save_record(record)
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert "T_aurora_2_model" in payload
    assert "T_coil_tip" in payload
    assert "T_tip_2_coil" in payload
    assert np.allclose(np.asarray(payload["T_tip_2_coil"], dtype=float), np.eye(4))


def test_tip_pose_service_loads_legacy_tip_key_by_inverting(tmp_path: Path) -> None:
    T_robot_aurora = np.eye(4)
    T_robot_aurora[0, 3] = 1.0
    T_tip_2_coil = np.eye(4)
    T_tip_2_coil[0, 3] = 2.0

    payload = {
        "T_robot_aurora": T_robot_aurora.tolist(),
        "T_tip_2_coil": T_tip_2_coil.tolist(),
    }
    path = tmp_path / "legacy.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    service = TipPoseService.from_registration_file(path)
    assert np.allclose(service.inputs.T_robot_aurora, T_robot_aurora)
    assert np.allclose(service.inputs.T_coil_tip, np.linalg.inv(T_tip_2_coil))


def test_tip_pose_service_loads_legacy_aurora_model_key(tmp_path: Path) -> None:
    T_aurora_2_model = np.eye(4)
    T_aurora_2_model[1, 3] = 3.0
    payload = {
        "T_aurora_2_model": T_aurora_2_model.tolist(),
        "T_coil_tip": np.eye(4).tolist(),
    }
    path = tmp_path / "legacy_model.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    service = TipPoseService.from_registration_file(path)
    assert np.allclose(service.inputs.T_robot_aurora, T_aurora_2_model)

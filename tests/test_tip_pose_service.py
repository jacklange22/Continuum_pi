import json
from pathlib import Path

import numpy as np

from continuum_robot.tracking.tip_pose_service import TipPoseService
from continuum_robot.tracking.tool_models import AuroraToolMeasurement


def test_tip_pose_service_computes_robot_tip_from_0A(tmp_path: Path) -> None:
    reg_file = tmp_path / "latest_registration.json"
    reg_file.write_text(
        json.dumps(
            {
                "T_robot_aurora": [[1, 0, 0, 1], [0, 1, 0, 2], [0, 0, 1, 3], [0, 0, 0, 1]],
                "T_coil_tip": [[1, 0, 0, 0.5], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]],
            }
        ),
        encoding="utf-8",
    )

    service = TipPoseService.from_registration_file(reg_file)
    tool = AuroraToolMeasurement(
        tool_id="0A",
        quat_wxyz=(1.0, 0.0, 0.0, 0.0),
        translation_xyz=(10.0, 0.0, 0.0),
        quality=0.1,
        status_byte=0,
        valid=True,
        status_text="ok",
    )
    T_robot_tip = service.compute_T_robot_tip_from_0A(tool)
    assert np.allclose(T_robot_tip[0:3, 3], np.array([11.5, 2.0, 3.0]))


def test_tip_pose_service_missing_registration_file_raises(tmp_path: Path) -> None:
    missing = tmp_path / "missing.json"
    try:
        TipPoseService.from_registration_file(missing)
    except FileNotFoundError:
        return
    raise AssertionError("Expected FileNotFoundError")

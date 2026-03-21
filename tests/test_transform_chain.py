import numpy as np

from continuum_robot.tracking.tip_pose_service import TipPoseService


def test_transform_chain_identity_case() -> None:
    T_robot_aurora = np.eye(4)
    T_aurora_coil = np.eye(4)
    T_coil_tip = np.eye(4)
    T_robot_tip = TipPoseService.compute_T_robot_tip(
        T_robot_aurora=T_robot_aurora,
        T_aurora_coil=T_aurora_coil,
        T_coil_tip=T_coil_tip,
    )
    assert np.allclose(T_robot_tip, np.eye(4))


def test_transform_chain_translation_composition() -> None:
    T_robot_aurora = np.eye(4)
    T_robot_aurora[0, 3] = 1.0
    T_aurora_coil = np.eye(4)
    T_aurora_coil[1, 3] = 2.0
    T_coil_tip = np.eye(4)
    T_coil_tip[2, 3] = 3.0

    T_robot_tip = TipPoseService.compute_T_robot_tip(
        T_robot_aurora=T_robot_aurora,
        T_aurora_coil=T_aurora_coil,
        T_coil_tip=T_coil_tip,
    )
    assert np.allclose(T_robot_tip[0:3, 3], np.array([1.0, 2.0, 3.0]))

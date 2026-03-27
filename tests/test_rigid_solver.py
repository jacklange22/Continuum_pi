import numpy as np

from continuum_robot.registration.rigid_solver import RigidRegistrationSolver


def _make_transform(rotation_rad: float, translation_xyz: tuple[float, float, float]) -> np.ndarray:
    c = np.cos(rotation_rad)
    s = np.sin(rotation_rad)
    T = np.eye(4)
    T[0:3, 0:3] = np.array(
        [
            [c, -s, 0.0],
            [s, c, 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    T[0:3, 3] = np.array(translation_xyz, dtype=float)
    return T


def _apply_transform(T: np.ndarray, points_n3: np.ndarray) -> np.ndarray:
    return (T[0:3, 0:3] @ points_n3.T).T + T[0:3, 3]


def test_solver_recovers_known_transform() -> None:
    solver = RigidRegistrationSolver()
    T_truth = _make_transform(np.deg2rad(20.0), (5.0, -3.0, 2.0))
    measured_in_aurora = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 2.0, 0.0],
            [1.0, 2.0, 3.0],
        ]
    )
    truth_in_robot = _apply_transform(T_truth, measured_in_aurora)

    T_est = solver.solve_T_robot_aurora(measured_in_aurora, truth_in_robot)
    assert np.allclose(T_est, T_truth, atol=1e-9)


def test_solver_handles_reflection_case_without_reflection_output() -> None:
    solver = RigidRegistrationSolver()
    measured_in_aurora = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    truth_reflected = measured_in_aurora.copy()
    truth_reflected[:, 0] *= -1.0

    T_est = solver.solve_T_robot_aurora(measured_in_aurora, truth_reflected)
    det = np.linalg.det(T_est[0:3, 0:3])
    assert det > 0.0

"""Rigid registration solver."""

import numpy as np


class RigidRegistrationSolver:
    """Solve best-fit rigid transform with SVD and reflection correction."""

    def solve_T_robot_aurora(self, measured_in_aurora: np.ndarray, truth_in_robot: np.ndarray) -> np.ndarray:
        """Return best-fit ``T_robot_aurora``.

        Supports points arranged as ``(3, N)`` or ``(N, 3)``.
        """
        src = self._as_n_by_3(measured_in_aurora, "measured_in_aurora")
        dst = self._as_n_by_3(truth_in_robot, "truth_in_robot")
        if src.shape != dst.shape:
            raise ValueError("measured_in_aurora and truth_in_robot must have identical point shape")
        if src.shape[0] < 3:
            raise ValueError("At least 3 point correspondences are required")

        src_centroid = src.mean(axis=0)
        dst_centroid = dst.mean(axis=0)
        src_centered = src - src_centroid
        dst_centered = dst - dst_centroid

        H = src_centered.T @ dst_centered
        U, _S, Vt = np.linalg.svd(H)
        R = Vt.T @ U.T

        # Enforce proper rotation when SVD returns a reflection.
        if np.linalg.det(R) < 0:
            Vt[-1, :] *= -1.0
            R = Vt.T @ U.T

        t = dst_centroid - (R @ src_centroid)

        T_robot_aurora = np.eye(4, dtype=float)
        T_robot_aurora[0:3, 0:3] = R
        T_robot_aurora[0:3, 3] = t
        return T_robot_aurora

    @staticmethod
    def _as_n_by_3(points: np.ndarray, name: str) -> np.ndarray:
        arr = np.asarray(points, dtype=float)
        if arr.ndim != 2:
            raise ValueError(f"{name} must be a 2D array")
        if arr.shape[1] == 3:
            return arr
        if arr.shape[0] == 3:
            return arr.T
        raise ValueError(f"{name} must have shape (N, 3) or (3, N)")

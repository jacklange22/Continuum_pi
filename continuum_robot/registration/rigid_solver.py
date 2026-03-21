"""Rigid registration solver scaffold."""

import numpy as np


class RigidRegistrationSolver:
    """Placeholder for SVD-based rigid alignment implementation."""

    def solve_T_robot_aurora(self, measured_in_aurora: np.ndarray, truth_in_robot: np.ndarray) -> np.ndarray:
        """Return best-fit T_robot_aurora.

        Scaffold returns identity.
        """
        _ = (measured_in_aurora, truth_in_robot)
        return np.eye(4)

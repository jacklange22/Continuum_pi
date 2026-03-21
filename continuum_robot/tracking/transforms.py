"""Transform math utilities with strict frame naming.

Convention:
- ``T_A_B`` transforms coordinates from frame B into frame A.
- Composition rule: ``T_A_C = T_A_B @ T_B_C``.
"""

from __future__ import annotations

import numpy as np


def quat_wxyz_to_rotmat(quat_wxyz: tuple[float, float, float, float]) -> np.ndarray:
    """Convert quaternion ``(w, x, y, z)`` to a 3x3 rotation matrix."""
    q = np.asarray(quat_wxyz, dtype=float)
    if q.shape != (4,):
        raise ValueError("Quaternion must have shape (4,)")
    if not np.isfinite(q).all():
        raise ValueError("Quaternion contains non-finite values")

    norm = np.linalg.norm(q)
    if norm <= 1e-12:
        raise ValueError("Quaternion norm is zero")

    w, x, y, z = q / norm
    return np.array(
        [
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
            [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
            [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
        ],
        dtype=float,
    )


def make_transform_A_B(
    quat_wxyz: tuple[float, float, float, float],
    translation_xyz: tuple[float, float, float],
) -> np.ndarray:
    """Build homogeneous transform ``T_A_B`` from rotation and translation."""
    t = np.asarray(translation_xyz, dtype=float)
    if t.shape != (3,):
        raise ValueError("Translation must have shape (3,)")
    if not np.isfinite(t).all():
        raise ValueError("Translation contains non-finite values")

    T_A_B = np.eye(4, dtype=float)
    T_A_B[0:3, 0:3] = quat_wxyz_to_rotmat(quat_wxyz)
    T_A_B[0:3, 3] = t
    return T_A_B


def assert_transform_matrix(T_A_B: np.ndarray, name: str) -> None:
    """Validate that ``T_A_B`` is a finite 4x4 homogeneous matrix."""
    if T_A_B.shape != (4, 4):
        raise ValueError(f"{name} must have shape (4, 4)")
    if not np.isfinite(T_A_B).all():
        raise ValueError(f"{name} contains non-finite values")
    if not np.allclose(T_A_B[3, :], np.array([0.0, 0.0, 0.0, 1.0]), atol=1e-8):
        raise ValueError(f"{name} last row is not [0, 0, 0, 1]")


def compose_T_A_C(T_A_B: np.ndarray, T_B_C: np.ndarray) -> np.ndarray:
    """Compose transforms using ``T_A_C = T_A_B @ T_B_C``."""
    assert_transform_matrix(T_A_B, "T_A_B")
    assert_transform_matrix(T_B_C, "T_B_C")
    return T_A_B @ T_B_C

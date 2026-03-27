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


def rotmat_to_quat_wxyz(R_A_B: np.ndarray) -> tuple[float, float, float, float]:
    """Convert a proper 3x3 rotation matrix to a quaternion ``(w, x, y, z)``."""
    assert_rotation_matrix(R_A_B, "R_A_B")
    trace = float(R_A_B[0, 0] + R_A_B[1, 1] + R_A_B[2, 2])

    if trace > 0.0:
        s = np.sqrt(trace + 1.0) * 2.0
        qw = 0.25 * s
        qx = (R_A_B[2, 1] - R_A_B[1, 2]) / s
        qy = (R_A_B[0, 2] - R_A_B[2, 0]) / s
        qz = (R_A_B[1, 0] - R_A_B[0, 1]) / s
    elif R_A_B[0, 0] > R_A_B[1, 1] and R_A_B[0, 0] > R_A_B[2, 2]:
        s = np.sqrt(1.0 + R_A_B[0, 0] - R_A_B[1, 1] - R_A_B[2, 2]) * 2.0
        qw = (R_A_B[2, 1] - R_A_B[1, 2]) / s
        qx = 0.25 * s
        qy = (R_A_B[0, 1] + R_A_B[1, 0]) / s
        qz = (R_A_B[0, 2] + R_A_B[2, 0]) / s
    elif R_A_B[1, 1] > R_A_B[2, 2]:
        s = np.sqrt(1.0 + R_A_B[1, 1] - R_A_B[0, 0] - R_A_B[2, 2]) * 2.0
        qw = (R_A_B[0, 2] - R_A_B[2, 0]) / s
        qx = (R_A_B[0, 1] + R_A_B[1, 0]) / s
        qy = 0.25 * s
        qz = (R_A_B[1, 2] + R_A_B[2, 1]) / s
    else:
        s = np.sqrt(1.0 + R_A_B[2, 2] - R_A_B[0, 0] - R_A_B[1, 1]) * 2.0
        qw = (R_A_B[1, 0] - R_A_B[0, 1]) / s
        qx = (R_A_B[0, 2] + R_A_B[2, 0]) / s
        qy = (R_A_B[1, 2] + R_A_B[2, 1]) / s
        qz = 0.25 * s

    quat = np.asarray([qw, qx, qy, qz], dtype=float)
    quat /= np.linalg.norm(quat)
    if quat[0] < 0.0:
        quat *= -1.0
    return tuple(float(v) for v in quat)


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


def assert_rotation_matrix(R_A_B: np.ndarray, name: str, *, atol: float = 1e-5) -> None:
    """Validate that ``R_A_B`` is a proper rotation matrix."""
    if R_A_B.shape != (3, 3):
        raise ValueError(f"{name} must have shape (3, 3)")
    if not np.isfinite(R_A_B).all():
        raise ValueError(f"{name} contains non-finite values")
    if not np.allclose(R_A_B.T @ R_A_B, np.eye(3), atol=atol):
        raise ValueError(f"{name} is not orthonormal")
    determinant = float(np.linalg.det(R_A_B))
    if not np.isclose(determinant, 1.0, atol=atol):
        raise ValueError(f"{name} determinant must be +1, got {determinant:.6f}")


def assert_rigid_transform_matrix(T_A_B: np.ndarray, name: str, *, atol: float = 1e-5) -> None:
    """Validate that ``T_A_B`` is a finite rigid transform with a proper rotation."""
    assert_transform_matrix(T_A_B, name)
    assert_rotation_matrix(T_A_B[0:3, 0:3], f"{name}[0:3,0:3]", atol=atol)


def compose_T_A_C(T_A_B: np.ndarray, T_B_C: np.ndarray) -> np.ndarray:
    """Compose transforms using ``T_A_C = T_A_B @ T_B_C``."""
    assert_rigid_transform_matrix(T_A_B, "T_A_B")
    assert_rigid_transform_matrix(T_B_C, "T_B_C")
    return T_A_B @ T_B_C

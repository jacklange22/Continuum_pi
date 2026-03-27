import numpy as np
from scipy.linalg import expm, logm, det
from math import sqrt
from typing import List, Tuple


def quat_2_dcm(quat: np.ndarray) -> np.ndarray:
    """
    Converts a quaternion (or many quaternions) to a DCM (rotation matrix)

    Args:
        quat: 4 x N array containing quaternions

    Returns:
        The corresponding DCMs with dimensions N x 3 x 3 (if N = 1, then dimensions are squeezed to 3 x 3)
    """
    assert quat.shape[0] == 4

    if len(quat.shape) == 1:
        quat = np.expand_dims(quat, axis=1)

    num_arrays = quat.shape[1]

    out = np.zeros((num_arrays, 3, 3))

    for i in range(num_arrays):
        q = quat[:, i] / np.linalg.norm(quat[:, i])

        out[i, 0, 0] = q[0] ** 2 + q[1] ** 2 - q[2] ** 2 - q[3] ** 2
        out[i, 0, 1] = 2 * (q[1] * q[2] - q[0] * q[3])
        out[i, 0, 2] = 2 * (q[1] * q[3] + q[0] * q[2])
        out[i, 1, 0] = 2 * (q[1] * q[2] + q[0] * q[3])
        out[i, 1, 1] = q[0] ** 2 - q[1] ** 2 + q[2] ** 2 - q[3] ** 2
        out[i, 1, 2] = 2 * (q[2] * q[3] - q[0] * q[1])
        out[i, 2, 0] = 2 * (q[1] * q[3] - q[0] * q[2])
        out[i, 2, 1] = 2 * (q[2] * q[3] + q[0] * q[1])
        out[i, 2, 2] = q[0] ** 2 - q[1] ** 2 - q[2] ** 2 + q[3] ** 2

    return np.squeeze(out, axis=0)


def dcm_2_quat(dcm: np.ndarray) -> np.ndarray:
    """
    Converts a DCM or DCMs to the corresponding quaternion

    Args:
        dcm: N x 3 x 3 (3 x 3 if N == 1) array containing DCMs

    Returns:
        A 4 x N array of quaternions
    """

    if len(dcm.shape) == 2:
        dcm = dcm[np.newaxis, :, :]

    n = dcm.shape[0]
    quaternions = np.zeros((4, n))

    for i in range(n):
        m = dcm[i, :, :]
        tr = m[0, 0] + m[1, 1] + m[2, 2]

        if tr > 0:
            S = np.sqrt(tr + 1.0) * 2
            qw = 0.25 * S
            qx = (m[2, 1] - m[1, 2]) / S
            qy = (m[0, 2] - m[2, 0]) / S
            qz = (m[1, 0] - m[0, 1]) / S
        elif (m[0, 0] > m[1, 1]) and (m[0, 0] > m[2, 2]):
            S = np.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2
            qw = (m[2, 1] - m[1, 2]) / S
            qx = 0.25 * S
            qy = (m[0, 1] + m[1, 0]) / S
            qz = (m[0, 2] + m[2, 0]) / S
        elif m[1, 1] > m[2, 2]:
            S = np.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2
            qw = (m[0, 2] - m[2, 0]) / S
            qx = (m[0, 1] + m[1, 0]) / S
            qy = 0.25 * S
            qz = (m[1, 2] + m[2, 1]) / S
        else:
            S = np.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2
            qw = (m[1, 0] - m[0, 1]) / S
            qx = (m[0, 2] + m[2, 0]) / S
            qy = (m[1, 2] + m[2, 1]) / S
            qz = 0.25 * S

        quaternions[:, i] = [qw, qx, qy, qz]

    if n == 1:
        return quaternions.flatten()

    return quaternions


def skew(vec: np.ndarray) -> np.ndarray:
    """
    Generates a skew symmetric matrix (used for tang map)

    Args:
        vec: A vector of dimension 3

    Returns:
        A skew symmetric 3 x 3 matrix
    """
    assert len(vec) == 3
    out = np.zeros((3, 3))

    out[0, 1] = -vec[2]
    out[0, 2] = vec[1]
    out[1, 0] = vec[2]
    out[1, 2] = -vec[0]
    out[2, 0] = -vec[1]
    out[2, 1] = vec[0]

    return out


def Tmult(T: np.ndarray, x: np.ndarray) -> np.ndarray:
    """
    Applies a transformation matrix T to (x, y, z) coordinates

    Args:
        T: The 4 x 4 transformation matrix
        x: A 3 x N array of the points to be transformed

    Returns:
        The 3 x N array of transformed points
    """
    if len(x.shape) == 1:
        x = x[:, np.newaxis]

    n = x.shape[1]
    x = np.concatenate((x, np.ones((1, n))), axis=0)

    y = T @ x

    return y[0:3, :]


def rmse(x: np.ndarray, y: np.ndarray) -> float:
    """
    Calculates the RMS error between two ordered sets of points x and y.

    Args:
        x: A 3 x N array of points
        y: A 3 x N array of points

    Returns:
        The RMS error between the two points
    """

    assert x.shape == y.shape and x.shape[0] == 3

    e = np.linalg.norm(x - y, axis=0)

    return np.sqrt((e**2).mean())


def rigid_align_svd(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """
    Uses SVD rigid registration to produce a transformation matrix that goes
    from x to y.

    Args:
        x: The points in the original frame
        y: The points in the transformed frame

    Returns:
        The points in x transformed into the y frame
        The transformation matrix from x to y
        The rmse error between the transformed x points in y and the y points
    """

    assert x.shape == y.shape and x.shape[0] == 3

    x_bar = x.mean(axis=1).reshape((3, 1))
    y_bar = y.mean(axis=1).reshape((3, 1))

    x_prime = x - x_bar
    y_prime = y - y_bar

    Cxy = x_prime @ np.transpose(y_prime)

    U, _, Vt = np.linalg.svd(Cxy, full_matrices=False)

    R = np.transpose(Vt) @ np.transpose(U)

    R[2, 0:3] *= np.linalg.det(R)

    t = (y_bar - R @ x_bar).flatten()

    T = np.identity(4)
    T[0:3, 0:3] = R
    T[0:3, 3] = t

    x_in_y = Tmult(T, x)

    return x_in_y, T, rmse(x_in_y, y)


def penprobe_transform(
    penprobe: np.ndarray,
    aurora_transforms: List[Tuple[np.ndarray, np.ndarray, float]],
) -> np.ndarray:
    """
    Performs penprobe transform from coil to tip position.

    Args:
        penprobe: A vector containing the position of the pen tip relative to the coil
        aurora_transforms: The transforms containing coil positions, quaternions, and rmse

    Returns:
        The pen tip positions for all aurora transforms
    """

    # Make sure penprobe is in right shape
    penprobe = penprobe.reshape((3, 1))

    n = len(aurora_transforms)

    # Go through all transforms and find tip positions
    tip_pos = np.nan * np.zeros((3, n))
    for i in range(n):
        R = quat_2_dcm(aurora_transforms[i][0])
        tip_pos[:, i] = (
            R @ penprobe + aurora_transforms[i][1].reshape((3, 1))
        ).flatten()

    return tip_pos


def tang_2_dcm(tang: np.ndarray) -> np.ndarray:
    """
    Converts from tang representation to DCM

    Args:
        tang: The tangent space representation of the orientation

    Returns:
        The corresponding DCM
    """
    assert len(tang) == 3
    return expm(skew(tang))


def dcm_2_tang(dcm: np.ndarray) -> np.ndarray:
    """
    Converts from DCM to tangent space representation

    Args:
        dcm: A DCM representing an orientation

    Returns:
        The corresponding tangent space representation
    """
    tang_skew = logm(dcm)
    return np.array([-tang_skew[1, 2], tang_skew[0, 2], -tang_skew[0, 1]])


def rotx(theta: float) -> np.ndarray:
    """
    Passive rotation matrix about the x-axis.

    Args:
        theta: The rotation angle in radians

    Returns:
        The corresponding rotation matrix
    """
    return np.array(
        [
            [1, 0, 0],
            [0, np.cos(theta), np.sin(theta)],
            [0, -np.sin(theta), np.cos(theta)],
        ]
    )


def roty(theta: float) -> np.ndarray:
    """
    Passive rotation matrix about the y-axis.

    Args:
        theta: The rotation angle in radians

    Returns:
        The corresponding rotation matrix
    """
    return np.array(
        [
            [np.cos(theta), 0, -np.sin(theta)],
            [0, 1, 0],
            [np.sin(theta), 0, np.cos(theta)],
        ]
    )


def rotz(theta: float) -> np.ndarray:
    """
    Passive rotation matrix about the z-axis.

    Args:
        theta: The rotation angle in radians

    Returns:
        The corresponding rotation matrix
    """
    return np.array(
        [
            [np.cos(theta), np.sin(theta), 0],
            [-np.sin(theta), np.cos(theta), 0],
            [0, 0, 1],
        ]
    )

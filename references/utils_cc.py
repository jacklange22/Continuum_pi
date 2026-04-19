import numpy as np
from math import sin, cos, pi, sqrt, atan2
from typing import List, Tuple
import matplotlib.pyplot as plt


def webster_2_camarillo_params(
    webster_params: List[Tuple[float, float, float]],
    initial_segment_lengths: List[float],
) -> np.ndarray:
    assert len(webster_params) == len(initial_segment_lengths)
    camarillo_params = np.zeros(3 * len(webster_params))

    for index, values in enumerate(zip(webster_params, initial_segment_lengths)):
        l, kappa, phi = values[0]
        seg_length = values[1]
        kappa_x = kappa * cos(phi)
        kappa_y = kappa * sin(phi)
        strain = l / seg_length - 1

        camarillo_params[3 * index] = kappa_x
        camarillo_params[3 * index + 1] = kappa_y
        camarillo_params[3 * index + 2] = strain

    return camarillo_params.reshape((-1, 1))


def camarillo_2_webster_params(
    camarillo_params: np.ndarray, segment_lengths: List[float]
) -> np.ndarray:
    """
    Converts camarillo parameters (kappa_x, kappa_y, epsilon_a) to webster parameters
    (l, kappa, phi)

    Args:
        camarillo_params: A length 3n numpy array containing the camarillo
        parameters (kappa_x, kappa_y, epsilon_a) for n segments where kappa_x and
        kappa_y are the curvature in the x and y axes, and epsilon_a is the axial
        strain (where positive epsilon_a indicates compression of the spine)
        segment_lengths: The length of each segment

    Returns:
        A length 3n numpy array containing the webster parameters (l, kappa, phi)
        for n segments.
    """
    assert len(camarillo_params) % 3 == 0
    num_segments = len(camarillo_params) // 3
    assert num_segments == len(segment_lengths)

    webster_params = []

    for i in range(num_segments):

        kappa_x = camarillo_params[3 * i].item()
        kappa_y = camarillo_params[3 * i + 1].item()
        axial_strain = camarillo_params[3 * i + 2].item()
        kappa = sqrt(kappa_x**2 + kappa_y**2)
        phi = atan2(kappa_y, kappa_x)

        l = (1 - axial_strain) * segment_lengths[i]

        webster_params.extend((l, kappa, phi))

    return np.array(webster_params)


def mike_2_webster_params(mike_params: np.ndarray) -> np.ndarray:
    """
    Converts Mike parameters (l, theta, phi) to webster parameters (l, kappa, phi)

    Args:
        mike_params: A length 3n numpy array containing the mike parameters
        (l, theta, phi) for n segments.

    Returns:
        A length 3n numpy array containing the webster parameters (l, kappa, phi)
        for n segments.
    """

    assert len(mike_params) % 3 == 0
    num_segments = len(mike_params) // 3

    webster_params = mike_params

    for i in range(num_segments):
        webster_params[3 * i + 1] = mike_params[3 * i + 1] / mike_params[3 * i]

    return webster_params


def dh_param_2_transform(param: Tuple[float, ...]) -> np.ndarray:
    assert len(param) == 4
    theta = param[0]
    d = param[1]
    r = param[2]
    alpha = param[3]

    Z = np.array(
        [
            [cos(theta), -sin(theta), 0, 0],
            [sin(theta), cos(theta), 0, 0],
            [0, 0, 1, d],
            [0, 0, 0, 1],
        ]
    )

    X = np.array(
        [
            [1, 0, 0, r],
            [0, cos(alpha), -sin(alpha), 0],
            [0, sin(alpha), cos(alpha), 0],
            [0, 0, 0, 1],
        ]
    )

    return np.matmul(Z, X)


def get_dh_params(param_tuple: Tuple[float, ...]) -> List[Tuple[float, ...]]:
    assert len(param_tuple) == 3
    l = param_tuple[0]
    kappa = param_tuple[1]
    phi = param_tuple[2]

    if kappa != 0:
        return [
            (phi, 0, 0, -pi / 2),
            (kappa * l / 2, 0, 0, pi / 2),
            (0, 2 / kappa * sin(kappa * l / 2), 0, -pi / 2),
            (kappa * l / 2, 0, 0, pi / 2),
            (-phi, 0, 0, 0),
        ]

    return [(0, l, 0, 0)]


def calculate_transforms(webster_params: np.ndarray) -> List[np.ndarray]:
    T = np.eye(4, dtype=float)
    segment_transforms = []
    assert len(webster_params) % 3 == 0
    num_segments = len(webster_params) // 3

    for i in range(num_segments):
        dh_params = get_dh_params(webster_params[3 * i : 3 * (i + 1)])

        for dh_param in dh_params:
            new_T = dh_param_2_transform(dh_param)
            T = np.matmul(T, new_T)

        segment_transforms.append(np.copy(T))

    return segment_transforms


def calculate_transform(webster_params: np.ndarray) -> np.ndarray:

    T = np.eye(4, dtype=float)

    dh_params = get_dh_params(webster_params)

    for dh_param in dh_params:
        new_T = dh_param_2_transform(dh_param)
        T = np.matmul(T, new_T)

    return T


def plot_robot(
    webster_params: np.ndarray,
    points_per_segment: float = 32,
    ax: plt.axes = None,
) -> plt.axes:

    assert len(webster_params) % 3 == 0
    num_segments = len(webster_params) // 3
    points = np.zeros((3, num_segments * points_per_segment + 1))
    T_base = np.identity(4)

    for segment in range(num_segments):
        segment_sample_lengths = np.linspace(
            0, webster_params[3 * segment], points_per_segment + 1
        )
        segment_sample_lengths = segment_sample_lengths[1:]

        kappa = webster_params[3 * segment + 1]
        phi = webster_params[3 * segment + 2]

        T_point = np.identity(4)

        for i in range(points_per_segment):
            point_params = np.array([segment_sample_lengths[i], kappa, phi])
            T_point = calculate_transform(point_params)

            T_point_in_base = np.matmul(T_base, T_point)

            points[:, 1 + segment * points_per_segment + i] = T_point_in_base[0:3, 3]

        T_base = np.matmul(T_base, T_point)
    if not ax:
        ax = plt.figure().add_subplot(projection="3d")
    ax.plot(points[0, :], points[1, :], points[2, :])
    plt.xlim([-32, 32])
    plt.ylim([-32, 32])
    ax.set_zlim([0, 64])

    return ax

from math import sin, cos, pi, sqrt, atan2
from typing import List, Tuple
import numpy as np


class MikeModel:
    """
    A constant curvature model based on Mike's original model, but adapted to use
    arbitrary cable positions.  Currently limited to one segment.

    Attributes:
        num_cables: The number of cables the system has
        cable_positions: A list of tuples (x, y) containing the locations of the
        cables
        segment_length: The length of the segment

    Methods:
        forward: Takes input cable displacements and outputs the corresponding Mike
        parameters
        inverse: Takes input Mike parameters and outputs the corresponding cable
        displacements
    """

    def __init__(
        self,
        num_cables: int,
        cable_positions: List[Tuple[float, ...]],
        segment_length: float,
    ):
        assert num_cables == len(cable_positions)
        self.num_cables = num_cables
        self.cable_positions = cable_positions
        self.segment_length = segment_length

    def forward(self, dls: np.ndarray) -> np.ndarray:
        assert len(dls) == self.num_cables

        phi_numerator = (
            dls[0] * self.cable_positions[1][0] - dls[1] * self.cable_positions[0][0]
        )
        phi_denomenator = (
            dls[1] * self.cable_positions[0][1] - dls[0] * self.cable_positions[1][1]
        )

        phi = atan2(phi_numerator, phi_denomenator)

        theta_numerator = (dls[:2] ** 2).sum()
        theta_denomenator = (
            self.cable_positions[0][0] * cos(phi)
            + self.cable_positions[0][1] * sin(phi)
        ) ** 2 + (
            self.cable_positions[1][0] * cos(phi)
            + self.cable_positions[1][1] * sin(phi)
        ) ** 2

        if theta_denomenator != 0:
            theta = sqrt(theta_numerator / theta_denomenator)
        else:
            theta = 0

        return np.array([self.segment_length, theta, phi])

    def inverse(self, mike_params: np.ndarray):

        if len(mike_params) == 2:
            theta = mike_params[0]
            phi = mike_params[1]

        elif len(mike_params) == 3:
            theta = mike_params[1]
            phi = mike_params[2]

        else:
            raise Exception("Size of input parameters is invalid")

        dls = np.array(
            [
                -(
                    self.cable_positions[i][0] * cos(phi)
                    + self.cable_positions[i][1] * sin(phi)
                )
                * theta
                for i in range(len(self.cable_positions))
            ]
        )

        return dls


def one_seg_forward_kinematics(
    delta_la: float,
    delta_lb: float,
    pos_a: Tuple[float, ...],
    pos_b: Tuple[float, ...],
    segment_length: float,
    other_segment_positions: List[Tuple[float, ...]] = None,
) -> Tuple[Tuple[float, ...], List[float]]:
    phi_numerator = delta_la * pos_b[0] - delta_lb * pos_a[0]
    phi_denomenator = delta_lb * pos_a[1] - delta_la * pos_b[1]

    phi = atan2(phi_numerator, phi_denomenator)

    theta_numerator = delta_la**2 + delta_lb**2
    theta_denomerator = (pos_a[0] * cos(phi) + pos_a[1] * sin(phi)) ** 2 + (
        pos_b[0] * cos(phi) + pos_b[1] * sin(phi)
    ) ** 2

    if theta_denomerator != 0:
        theta = sqrt(theta_numerator / theta_denomerator)
    else:
        theta = 0

    kappa = theta / segment_length

    if other_segment_positions:
        other_delta_ls = []
        for pos in other_segment_positions:
            delta_l = (-pos[0] * cos(phi) - pos[1] * sin(phi)) * theta
            other_delta_ls.append(delta_l)

        return (segment_length, kappa, phi), other_delta_ls

    return (segment_length, kappa, phi), None


def multi_seg_forward_kinematics(
    delta_las: List[float],
    delta_lbs: List[float],
    pos_as: List[Tuple[float, ...]],
    pos_bs: List[Tuple[float, ...]],
    segment_lengths: List[float],
    other_cable_positions: List[List[Tuple[float, ...]]] = None,
) -> List[Tuple[Tuple[float, ...], List[float]]]:
    assert len(delta_las) == len(delta_lbs) == len(pos_as), "num segments"
    assert len(pos_as) == len(pos_bs) == len(segment_lengths), "num segments"

    if other_cable_positions:
        assert len(segment_lengths) == len(other_cable_positions), "num segments"
    cumulative_delta_la = 0
    cumulative_delta_lb = 0
    segment_params = []
    if other_cable_positions:
        cumulative_other_deltas = [0] * len(other_cable_positions)
        all_other_dls = [[0] * len(other_cable_positions[0])] * len(
            other_cable_positions
        )
        print(all_other_dls)
        for i in range(len(segment_lengths)):
            param, other_dls = one_seg_forward_kinematics(
                delta_las[i] - cumulative_delta_la,
                delta_lbs[i] - cumulative_delta_lb,
                pos_as[i],
                pos_bs[i],
                segment_lengths[i],
                other_cable_positions[i],
            )
            segment_params.append(param)
            cumulative_delta_la += delta_las[i]
            cumulative_delta_lb += delta_lbs[i]


def one_seg_inverse_kinematics(
    seg_params: Tuple[float, ...], cable_positions: List[Tuple[float, ...]]
) -> List[float]:
    seg_len = seg_params[0]
    kappa = seg_params[1]
    phi = seg_params[2]

    theta = seg_len * kappa

    return [
        -(cable_positions[i][0] * cos(phi) + cable_positions[i][1] * sin(phi)) * theta
        for i in range(len(cable_positions))
    ]

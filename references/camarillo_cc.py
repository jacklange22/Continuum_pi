import numpy as np
from scipy.optimize import minimize, Bounds, LinearConstraint
from math import sin, cos, pi, sqrt, atan2
from typing import List, Tuple
from utils_cc import *


class CamarilloModel:

    def __init__(
        self,
        cable_positions: List[Tuple[Tuple[float, ...], ...]],
        segment_stiffness_vals: List[Tuple[float, ...]],
        cable_stiffness_vals: List[Tuple[float, ...]],
        segment_lengths: List[float],
        additional_cable_length: float,
    ):
        assert (
            len(cable_positions)
            == len(segment_stiffness_vals)
            == len(cable_stiffness_vals)
            == len(segment_lengths)
        )
        self.num_segments = len(cable_positions)
        self.cables_per_segment = [-1] * self.num_segments

        # Ensure number of cables per segment is consistent, segment_stiffness_vals is correct shape, and cable_positions is correct shape
        for segment_num in range(self.num_segments):
            assert len(cable_positions[segment_num]) == len(
                cable_stiffness_vals[segment_num]
            ), "Number of cables in segment {} is inconsistent".format(segment_num)
            assert (
                len(segment_stiffness_vals[segment_num]) == 2
            ), "Length of segment_stiffness_vals in segment {} is not 2".format(
                segment_num
            )

            num_cables = len(cable_positions[segment_num])
            self.cables_per_segment[segment_num] = num_cables

            for cable_num in range(num_cables):
                assert (
                    len(cable_positions[segment_num][cable_num]) == 2
                ), "len(cable_positions) in segment {} at cable {} != 2".format(
                    segment_num, cable_num
                )
        self.num_cables = sum(self.cables_per_segment)
        self.cable_positions = cable_positions
        self.segment_stiffness_vals = segment_stiffness_vals
        self.cable_stiffness_vals = cable_stiffness_vals
        self.segment_lengths = segment_lengths
        self.additional_cable_length = additional_cable_length

        self.update_matrices()

    def update_matrices(self):
        D, K_m_inv, L_0, L_t, K_t_inv = self.__get_camarillo_matrices()
        self.C_m = np.matmul(
            np.matmul(np.matmul(np.transpose(D), L_0), K_m_inv), D
        ) + np.matmul(L_t, K_t_inv)

        self.C_m_inv = np.linalg.inv(self.C_m)
        self.C_m_inv_trans = np.transpose(self.C_m_inv)

        self.A = np.matmul(np.matmul(K_m_inv, D), self.C_m_inv)
        self.A_inv = np.linalg.pinv(self.A)

    def forward(self, dls: np.ndarray, handle_slack: bool = False) -> np.ndarray:
        assert len(dls) == self.num_cables
        dls = -dls

        if handle_slack:

            def min_fcn(delta, dls):
                return delta @ self.C_m_inv @ (delta + dls)

            def min_der(delta, dls):
                return self.C_m_inv @ (dls + delta) + self.C_m_inv_trans @ delta

            def min_hess(delta):
                return self.C_m_inv + self.C_m_inv_trans

            linear_constraint = LinearConstraint(
                self.C_m_inv, -self.C_m_inv @ dls, np.full(dls.shape, np.inf)
            )
            bounds = Bounds(np.zeros(dls.shape), np.full(dls.shape, np.inf))

            results = minimize(
                lambda x: min_fcn(x, dls),
                np.zeros(dls.shape),
                method="trust-constr",
                jac=lambda x: min_der(x, dls),
                hess=min_hess,
                constraints=[linear_constraint],
                bounds=bounds,
                options={"gtol": 1e-9},
            )

            return self.A @ (dls + results.x)

        else:
            return self.A @ dls

    def inverse(self, camarillo_params: np.ndarray) -> np.ndarray:
        assert len(camarillo_params) / 3 == self.num_segments

        return -(self.A_inv @ camarillo_params)

    def __get_camarillo_matrices(self):

        # Initialization of matrices and lists for equation
        K_m_inv_diag_list = [-1] * 3 * self.num_segments
        D_matrix_list = []
        L_0_diag_list = [-1] * 3 * self.num_segments
        L_t_diag_list = [-1] * sum(self.cables_per_segment)
        K_t_inv_diag_list = [-1] * sum(self.cables_per_segment)

        # Loop through segments
        for segment_num in range(self.num_segments):
            # Set stiffness from inputted stiffness values
            K_m_inv_diag_list[3 * segment_num : 3 * segment_num + 3] = [
                1 / self.segment_stiffness_vals[segment_num][1],
                1 / self.segment_stiffness_vals[segment_num][1],
                1 / self.segment_stiffness_vals[segment_num][0],
            ]

            # Form D matrix for segment
            row_1 = np.array(
                [position[0] for position in self.cable_positions[segment_num]],
                dtype=float,
            ).reshape((1, -1))
            row_2 = np.array(
                [position[1] for position in self.cable_positions[segment_num]],
                dtype=float,
            ).reshape((1, -1))
            row_3 = np.ones((1, self.cables_per_segment[segment_num]), dtype=float)

            D = np.concatenate([row_1, row_2, row_3], axis=0)

            D_matrix_list.append(D)

            # Form L0 diagonal matrix list
            L_0_diag_list[3 * segment_num : 3 * segment_num + 3] = [
                self.segment_lengths[segment_num]
            ] * 3

            # Form Lt diagonal matrix list
            cables_so_far = sum(self.cables_per_segment[:segment_num])
            num_cables = self.cables_per_segment[segment_num]
            L_t_diag_list[cables_so_far : cables_so_far + num_cables] = [
                self.additional_cable_length
                + sum(self.segment_lengths[: segment_num + 1])
            ] * num_cables

            # Form Kt^-1 diagonal matrix list
            K_t_inv_diag_list[cables_so_far : cables_so_far + num_cables] = [
                1 / x for x in self.cable_stiffness_vals[segment_num]
            ]

        # Form stiffness matrix
        K_m_inv = np.diag(np.array(K_m_inv_diag_list, dtype=float))

        # Form L0 matrix
        L_0 = np.diag(np.array(L_0_diag_list, dtype=float))

        # Form Lt matrix
        L_t = np.diag(np.array(L_t_diag_list, dtype=float))

        # Form Kt^-1 matrix
        K_t_inv = np.diag(np.array(K_t_inv_diag_list, dtype=float))

        # Form D matrix
        D = np.zeros((3 * self.num_segments, sum(self.cables_per_segment)), dtype=float)
        for i in range(self.num_segments):
            for segment_num in range(i, self.num_segments):
                cables_so_far = sum(self.cables_per_segment[:segment_num])
                num_cables = self.cables_per_segment[segment_num]
                D[3 * i : 3 * i + 3, cables_so_far : cables_so_far + num_cables] = (
                    D_matrix_list[segment_num]
                )

        return D, K_m_inv, L_0, L_t, K_t_inv


def get_camarillo_matrices(
    cable_positions: List[Tuple[Tuple[float, ...], ...]],
    segment_stiffness_vals: List[Tuple[float, ...]],
    cable_stiffness_vals: List[Tuple[float, ...]],
    segment_lengths: List[float],
    additional_cable_length: float,
    cables_per_segment: List[int],
):
    num_segments = len(cables_per_segment)

    # Initialization of matrices and lists for equation
    K_m_inv_diag_list = [-1] * 3 * num_segments
    D_matrix_list = []
    L_0_diag_list = [-1] * 3 * num_segments
    L_t_diag_list = [-1] * sum(cables_per_segment)
    K_t_inv_diag_list = [-1] * sum(cables_per_segment)

    # Loop through segments
    for segment_num in range(num_segments):
        # Set stiffness from inputted stiffness values
        K_m_inv_diag_list[3 * segment_num : 3 * segment_num + 3] = [
            1 / segment_stiffness_vals[segment_num][1],
            1 / segment_stiffness_vals[segment_num][1],
            1 / segment_stiffness_vals[segment_num][0],
        ]

        # Form D matrix for segment
        row_1 = np.array(
            [position[0] for position in cable_positions[segment_num]],
            dtype=float,
        ).reshape((1, -1))
        row_2 = np.array(
            [position[1] for position in cable_positions[segment_num]],
            dtype=float,
        ).reshape((1, -1))
        row_3 = np.ones((1, cables_per_segment[segment_num]), dtype=float)

        D = np.concatenate([row_1, row_2, row_3], axis=0)

        D_matrix_list.append(D)

        # Form L0 diagonal matrix list
        L_0_diag_list[3 * segment_num : 3 * segment_num + 3] = [
            segment_lengths[segment_num]
        ] * 3

        # Form Lt diagonal matrix list
        cables_so_far = sum(cables_per_segment[:segment_num])
        num_cables = cables_per_segment[segment_num]
        L_t_diag_list[cables_so_far : cables_so_far + num_cables] = [
            additional_cable_length + sum(segment_lengths[: segment_num + 1])
        ] * num_cables

        # Form Kt^-1 diagonal matrix list
        K_t_inv_diag_list[cables_so_far : cables_so_far + num_cables] = [
            1 / x for x in cable_stiffness_vals[segment_num]
        ]

    # Form stiffness matrix
    K_m_inv = np.diag(np.array(K_m_inv_diag_list, dtype=float))

    # Form L0 matrix
    L_0 = np.diag(np.array(L_0_diag_list, dtype=float))

    # Form Lt matrix
    L_t = np.diag(np.array(L_t_diag_list, dtype=float))

    # Form Kt^-1 matrix
    K_t_inv = np.diag(np.array(K_t_inv_diag_list, dtype=float))

    # Form D matrix
    D = np.zeros((3 * num_segments, sum(cables_per_segment)), dtype=float)
    for i in range(num_segments):
        for segment_num in range(i, num_segments):
            cables_so_far = sum(cables_per_segment[:segment_num])
            num_cables = cables_per_segment[segment_num]
            D[3 * i : 3 * i + 3, cables_so_far : cables_so_far + num_cables] = (
                D_matrix_list[segment_num]
            )

    return D, K_m_inv, L_0, L_t, K_t_inv


def no_slack_model(
    delta_lengths: List[Tuple[float, ...]],
    cable_positions: List[Tuple[Tuple[float, ...], ...]],
    segment_stiffness_vals: List[Tuple[float, ...]],
    cable_stiffness_vals: List[Tuple[float, ...]],
    segment_lengths: List[float],
    additional_cable_length: float = 0,
):
    # Ensure number of segments is consistent
    assert (
        len(delta_lengths)
        == len(cable_positions)
        == len(segment_stiffness_vals)
        == len(cable_stiffness_vals)
        == len(segment_lengths)
    )
    num_segments = len(delta_lengths)

    cables_per_segment = [-1] * num_segments

    # Ensure number of cables per segment is consistent, segment_stiffness_vals is correct shape, and cable_positions is correct shape
    for segment_num in range(num_segments):
        assert (
            len(delta_lengths[segment_num])
            == len(cable_positions[segment_num])
            == len(cable_stiffness_vals[segment_num])
        ), "Number of cables in segment {} is inconsistent".format(segment_num)
        assert (
            len(segment_stiffness_vals[segment_num]) == 2
        ), "Length of segment_stiffness_vals in segment {} is not 2".format(segment_num)

        num_cables = len(delta_lengths[segment_num])
        cables_per_segment[segment_num] = num_cables

        for cable_num in range(num_cables):
            assert (
                len(cable_positions[segment_num][cable_num]) == 2
            ), "len(cable_positions) in segment {} at cable {} != 2".format(
                segment_num, cable_num
            )

    # Convert delta_lengths to np array to form y
    y = np.concatenate(
        [-np.array(x, dtype=float).reshape((-1, 1)) for x in delta_lengths],
        axis=0,
    )

    # Form matrices used for forward and inverse kinematics
    D, K_m_inv, L_0, L_t, K_t_inv = get_camarillo_matrices(
        cable_positions,
        segment_stiffness_vals,
        cable_stiffness_vals,
        segment_lengths,
        additional_cable_length,
        cables_per_segment,
    )

    # Form C_m and C_m^-1
    C_m = np.matmul(np.matmul(np.matmul(np.transpose(D), L_0), K_m_inv), D) + np.matmul(
        L_t, K_t_inv
    )

    C_m_inv = np.linalg.inv(C_m)

    # q = Ay
    A = np.matmul(np.matmul(K_m_inv, D), C_m_inv)

    q = np.matmul(A, y)

    return q


def slack_model(
    delta_lengths: List[Tuple[float, ...]],
    cable_positions: List[Tuple[Tuple[float, ...], ...]],
    segment_stiffness_vals: List[Tuple[float, ...]],
    cable_stiffness_vals: List[Tuple[float, ...]],
    segment_lengths: List[float],
    additional_cable_length: float = 0,
):
    # Ensure number of segments is consistent
    assert (
        len(delta_lengths)
        == len(cable_positions)
        == len(segment_stiffness_vals)
        == len(cable_stiffness_vals)
        == len(segment_lengths)
    )
    num_segments = len(delta_lengths)

    cables_per_segment = [-1] * num_segments

    # Ensure number of cables per segment is consistent, segment_stiffness_vals is correct shape, and cable_positions is correct shape
    for segment_num in range(num_segments):
        assert (
            len(delta_lengths[segment_num])
            == len(cable_positions[segment_num])
            == len(cable_stiffness_vals[segment_num])
        ), "Number of cables in segment {} is inconsistent".format(segment_num)
        assert (
            len(segment_stiffness_vals[segment_num]) == 2
        ), "Length of segment_stiffness_vals in segment {} is not 2".format(segment_num)

        num_cables = len(delta_lengths[segment_num])
        cables_per_segment[segment_num] = num_cables

        for cable_num in range(num_cables):
            assert (
                len(cable_positions[segment_num][cable_num]) == 2
            ), "Length of cable_positions in segment {} at cable {} is not 2".format(
                segment_num, cable_num
            )

    # Convert delta_lengths to np array to form y
    y_hat = np.concatenate([np.array(x, dtype=float) for x in delta_lengths], axis=0)

    # Form matrices used for forward and inverse kinematics
    D, K_m_inv, L_0, L_t, K_t_inv = get_camarillo_matrices(
        cable_positions,
        segment_stiffness_vals,
        cable_stiffness_vals,
        segment_lengths,
        additional_cable_length,
        cables_per_segment,
    )

    # Form C_m and C_m^-1
    C_m = np.matmul(np.matmul(np.matmul(np.transpose(D), L_0), K_m_inv), D) + np.matmul(
        L_t, K_t_inv
    )
    C_m_inv = np.linalg.inv(C_m)

    # Functions for slack minimization calculation
    def f(delta: np.ndarray) -> float:
        out: float = np.matmul(np.matmul(np.transpose(delta), C_m_inv), (y_hat + delta))
        return out

    def grad_f(delta: np.ndarray, out: np.ndarray) -> None:
        out[()] = np.matmul(np.transpose(C_m_inv) + C_m_inv, delta) + np.matmul(
            C_m_inv, y_hat
        )

    def g(delta: np.ndarray, out: np.ndarray) -> np.ndarray:
        out[()] = np.matmul(C_m_inv, (y_hat + delta))
        return out

    jacobian_non_zero = np.nonzero(C_m_inv)

    def jac_g(delta: np.ndarray, out: np.ndarray) -> np.ndarray:
        out = C_m_inv[C_m_inv != 0]
        return out

    hessian_matrix = np.transpose(C_m_inv) + C_m_inv
    hessian_non_zero = np.nonzero(hessian_matrix)

    def h(
        delta: np.ndarray,
        lagrange: np.ndarray,
        obj_factor: float,
        out: np.ndarray,
    ) -> np.ndarray:
        out[()] = obj_factor * hessian_matrix[hessian_matrix != 0]

    # Set up slack estimation optimization problem
    nlp = ipyopt.Problem(
        n=len(y_hat),
        x_l=np.zeros(len(y_hat)),
        x_u=np.ones(len(y_hat)) * np.inf,
        m=len(y_hat),
        g_l=np.zeros(len(y_hat)),
        g_u=np.ones(len(y_hat)) * np.inf,
        sparsity_indices_jac_g=jacobian_non_zero,
        sparsity_indices_h=hessian_non_zero,
        eval_f=f,
        eval_grad_f=grad_f,
        eval_g=g,
        eval_jac_g=jac_g,
        eval_h=h,
    )

    delta, obj, status = nlp.solve(x0=np.ones(len(y_hat)))

    y = y_hat + delta
    # y = y_hat
    # q = Ay
    A = np.matmul(np.matmul(K_m_inv, D), C_m_inv)

    q = np.matmul(A, y)

    return q

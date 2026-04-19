#!/bin/python3
import numpy as np
import scipy.optimize as opt
import torch
import matplotlib.pyplot as plt
from typing import List, Tuple
import ANN
import camarillo_cc
import kinematics
import utils_cc
import utils_data

trajectory = np.loadtxt("output/nathan_trajectory.dat", delimiter=",")
trajectory_tensor = torch.tensor(trajectory)
num_points = trajectory.shape[1]
num_closed_loop_steps = 11
step_size = 1


model = ANN.Model(
    input_dim=4, output_dim=6, hidden_layers=[32, 32], loss=ANN.PositionLoss()
)

model.load("../model_learning/models/real_05_12_2024a/2024_05_12_19_56_49.pt")
model.model.eval()

array = np.array([-12, -12, -12, -12])

tensor = torch.tensor(array)
output = model(tensor)
# tensor([0, 0, 64, 0, 0, 0])
output_pos = output[:3]


def loss_fcn(
    u: np.ndarray,
    model: ANN.Model,
    y_star: torch.tensor,
    u_0: torch.tensor,
    weighting: Tuple[float, float],
) -> Tuple[float, np.ndarray]:
    """
    Finds the value of the loss function L(y) and its gradient.

    Args:
        u: The cable displacements fed into the model
        model: The learned forward kinematics model
        y_star: The desired tip position
        u_0: The cable displacements at the previous timestep
        weighting: Contains the diagonal value of Q and R

    Returns:
        The value of the loss function and its gradient
    """

    # Convert numpy array to tensor (requires_grad is used to extract gradient)
    u_tensor = torch.tensor(u, requires_grad=True)

    # Evaluate model and extract position
    model.zero_grad()
    y_hat = model(dl_tensor)[0:3]

    # Calculate error and change in cable length
    e = y_hat - y_star
    delta_u = u_tensor - u_0

    # Calculate loss function
    loss = torch.dot(e, e) * weighting[0] + torch.dot(delta_u, delta_u) * weighting[1]

    # Obtain gradient of loss function based on inputs
    loss.backward()
    grad = u_tensor.grad.numpy()

    # Return loss function and its gradient
    return (loss.item(), grad)


open_loop_dls = np.zeros((4, num_points))

for i in range(num_points):
    print(f"{i+1} of {num_points}")
    dl_0 = 0
    if i == 0:
        dl_0 = torch.tensor(np.array([0.0] * 4))
    else:
        dl_0 = torch.tensor(open_loop_dls[:, i - 1].flatten())

    x_star = trajectory_tensor[:, i].flatten()

    optim_func = lambda x: loss_fcn(x, model, x_star, dl_0, (50.0, 1.0))

    result = opt.minimize(
        optim_func,
        dl_0.numpy(),
        method="L-BFGS-B",
        jac=True,
        bounds=[(-12, 12), (-12, 12), (-12, 12), (-12, 12)],
    )
    open_loop_dls[:, i] = result["x"]

np.savetxt("output/nathan_cable_trajectory.dat", open_loop_dls, delimiter=",")

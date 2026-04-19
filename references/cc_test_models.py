#!/bin/python3
import numpy as np
import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import scipy.optimize as opt
import ANN
import mike_cc
import camarillo_cc
import utils_data
import utils_cc

test_file = "training_data/meas_2024_05_12_18_47_41.dat"

dataset = ANN.Dataset()
dataset.load_from_file(test_file)
dataset.clean()

num_points = len(dataset)

model = ANN.Model(
    input_dim=4, output_dim=6, hidden_layers=[32, 32], loss=ANN.PositionLoss()
)
model.load("../model_learning/models/real_05_12_2024a/2024_05_12_19_56_49.pt")
model.model.eval()

camarillo_stiffness = np.loadtxt("../../tools/camarillo_stiffness", delimiter=",")
ka, kb, kt = camarillo_stiffness[0], camarillo_stiffness[1], camarillo_stiffness[2]
cable_positions = [
    ((6, 0), (0, 6), (-6, 0), (0, -6)),
]
segment_stiffness_vals = [(ka, kb)]
cable_stiffness_vals = [(kt, kt, kt, kt)]
segment_lengths = [64]

mike_pos = np.nan * np.zeros((3, num_points))
camarillo_pos = np.nan * np.zeros((3, num_points))
learned_pos = np.nan * np.zeros((3, num_points))
true_pos = np.nan * np.zeros((3, num_points))

for i, data in enumerate(dataset):
    input, output = data[0], data[1]

    true_pos[:, i] = output[:3].numpy()

    learned_pos[:, i] = model(input)[:3].detach().numpy()

    mike_out, _ = mike_cc.one_seg_forward_kinematics(
        input[0].item(),
        input[1].item(),
        cable_positions[0][0],
        cable_positions[0][1],
        segment_lengths[0],
    )

    T_mike = utils_cc.calculate_transform(mike_out)
    mike_pos[:, i] = T_mike[0:3, 3]

    camarillo_params = camarillo_cc.no_slack_model(
        [tuple(input.numpy().tolist())],
        cable_positions,
        segment_stiffness_vals,
        cable_stiffness_vals,
        segment_lengths,
    )

    webster_params = utils_cc.camarillo_2_webster_params(
        camarillo_params, segment_lengths
    )

    T_camarillo = utils_cc.calculate_transforms(webster_params)[0]

    camarillo_pos[:, i] = T_camarillo[0:3, 3]


def optim_func(x):
    error = np.linalg.norm(
        mike_pos
        - true_pos
        + np.concatenate((x.reshape((-1, 1)), np.array([[0]])), axis=0),
        axis=0,
    )
    return error.mean()


result = opt.minimize(optim_func, np.array([0, 0]), method="BFGS")
true_pos -= np.concatenate((result["x"].reshape((-1, 1)), np.array([[0]])), axis=0)
learned_pos -= np.concatenate((result["x"].reshape((-1, 1)), np.array([[0]])), axis=0)

mike_error = np.linalg.norm(mike_pos - true_pos, axis=0)
camarillo_error = np.linalg.norm(camarillo_pos - true_pos, axis=0)
learned_error = np.linalg.norm(learned_pos - true_pos, axis=0)

mike_rmse = np.sqrt((mike_error**2).mean())
camarillo_rmse = np.sqrt((camarillo_error**2).mean())
learned_rmse = np.sqrt((learned_error**2).mean())

ax = plt.figure().add_subplot(projection="3d")
ax.plot(true_pos[0, :], true_pos[1, :], true_pos[2, :], "o")
ax.plot(mike_pos[0, :], mike_pos[1, :], mike_pos[2, :], "x")
ax.plot(camarillo_pos[0, :], camarillo_pos[1, :], camarillo_pos[2, :], "+")
ax.plot(learned_pos[0, :], learned_pos[1, :], learned_pos[2, :], "1")
plt.xlim((-45, 45))
plt.ylim((-45, 45))
ax.set_zlim((0, 70))

plt.figure()
plt.plot(true_pos[0, :], true_pos[1, :], "o", label="Measured")
plt.plot(
    mike_pos[0, :],
    mike_pos[1, :],
    "x",
    label=f"Kinematic CC Model (RMSE: {mike_rmse:.3f} mm)",
)
plt.plot(
    camarillo_pos[0, :],
    camarillo_pos[1, :],
    "+",
    label=f"Camarillo CC Model (RMSE: {camarillo_rmse:.3f} mm)",
)
plt.plot(
    learned_pos[0, :],
    learned_pos[1, :],
    "1",
    label=f"Learned Model (RMSE: {learned_rmse:.3f} mm)",
)
plt.title("Test Set Measured and Modeled Positions")
plt.xlabel("x (mm)")
plt.ylabel("y (mm)")
plt.xlim((-45, 45))
plt.ylim((-45, 45))
plt.legend(loc="lower left")

plt.figure()
plt.hist(mike_error, 50)
plt.xlabel("Position Error (mm)")
plt.ylabel("Count")
plt.title(f"Kinematic CC Model Error (Mean: {mike_error.mean():.3f} mm)")

plt.figure()
plt.hist(camarillo_error, 50)
plt.xlabel("Position Error (mm)")
plt.ylabel("Count")
plt.title(f"Statics CC Model Error (Mean: {camarillo_error.mean():.3f} mm)")

plt.figure()
plt.hist(learned_error, 50)
plt.xlabel("Position Error (mm)")
plt.ylabel("Count")
plt.title(f"Learned Model Error (Mean: {learned_error.mean():.3f} mm)")
plt.show()

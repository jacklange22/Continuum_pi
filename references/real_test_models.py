#!/bin/python3
import numpy as np
import torch
from torch.utils.data import DataLoader
import os
import glob
import ANN
import utils_data

test_file = "./test_data/kinematic_2024_07_30_11_30_29.dat"

dataset = ANN.Dataset()
dataset.load_from_file(test_file)
dataset.clean()

model_files = sorted(glob.glob("models/real_07_17_2024/*.pt"))

model = ANN.Model(
    input_dim=4, output_dim=6, hidden_layers=[32, 32], loss=ANN.PoseLoss()
)


def test(model, model_files, dataset):

    position_losses = []
    orientation_losses = []
    counter = 0
    for file in model_files:
        counter += 1
        model.load(file)
        print(f"Model {counter} position")
        model.loss = ANN.PositionLoss()
        position_losses.append(np.array(model.test_dataset(dataset)).reshape((1, -1)))
        print(f"Model {counter} orientation")
        model.loss = ANN.OrientationLoss()
        orientation_losses.append(
            np.array(model.test_dataset(dataset)).reshape((1, -1))
        )

    return np.concatenate(position_losses, axis=0), np.concatenate(
        orientation_losses, axis=0
    )


position_loss, orientation_loss = test(model, model_files, dataset)
np.savetxt("output/real_pos_test_loss.dat", position_loss, delimiter=",")
np.savetxt(
    "output/real_tang_test_loss.dat",
    orientation_loss,
    delimiter=",",
)

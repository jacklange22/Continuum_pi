#!/bin/python3
import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from collections import OrderedDict
from typing import List, Tuple
import os
import numpy as np
import datetime
import ANN
import utils_cc
import utils_data
import kinematics

TRAINING_ITERATIONS = 10
previous_meas = 1

data_filename = "training_data/kinematic_2024_05_09_20_11_13.dat"

container = utils_data.DataContainer()
container.file_import(data_filename)
cable_deltas, pos, tang = container.to_numpy()
num_cables = container.num_cables
num_meas = container.num_measurements

new_cable_deltas = np.zeros(
    ((1 + previous_meas) * num_cables, num_meas - previous_meas)
)

new_cable_deltas[:, 0] = np.transpose(cable_deltas[:, previous_meas::-1]).flatten()

for i in range(previous_meas + 1, num_meas):
    new_cable_deltas[:, i - previous_meas] = np.transpose(
        cable_deltas[:, i : (i - previous_meas - 1) : -1]
    ).flatten()

pos = pos[:, previous_meas:]
tang = tang[:, previous_meas:]

container.from_raw_data(
    container.date,
    container.time,
    (1 + previous_meas) * num_cables,
    num_meas - previous_meas,
    new_cable_deltas,
    pos,
    tang,
)

dataset = ANN.Dataset()
dataset.load_from_DataContainer(container)
dataset.clean()


def train(training_iterations):
    alphabet = [""] + [x for x in "abcdefghifjklmnopqrstuvwxyz"]

    now = datetime.datetime.now()
    save_dir = f"hysteresis_{now.month:02n}_{now.day:02n}_{now.year}"
    counter = 0
    while os.path.isdir("models/" + save_dir + alphabet[counter]):
        counter += 1
    save_dir += alphabet[counter]
    os.mkdir(f"models/{save_dir}")
    os.mkdir(f"output/{save_dir}")

    for iteration in range(training_iterations):
        print(f"ITERATION {iteration + 1}")

        # Training
        split_datasets = torch.utils.data.random_split(dataset, [0.75, 0.25])
        train_dataloader = DataLoader(split_datasets[0], batch_size=64)
        validation_dataloader = DataLoader(split_datasets[1], batch_size=64)

        model = ANN.Model(
            input_dim=(1 + previous_meas) * num_cables,
            output_dim=6,
            hidden_layers=[32, 32],
            loss=ANN.PoseLoss(),
            save_path=f"models/{save_dir}/{iteration}.pt",
        )
        train_loss, validation_loss = model.train(
            train_dataloader,
            validation_dataloader,
            checkpoints=True,
            save_model=True,
        )

        # Loss data restructuring
        train_loss = np.array(train_loss)
        validation_loss = np.array(validation_loss)

        # Save loss as csv
        file_num = 0
        while os.path.exists(
            os.path.join(
                os.path.dirname(os.path.realpath("__file__")),
                f"output/{save_dir}/train_loss_{file_num}.dat",
            )
        ):
            file_num += 1

        np.savetxt(
            f"output/{save_dir}/train_loss_{file_num}.dat",
            train_loss,
            delimiter=",",
        )
        np.savetxt(
            f"output/{save_dir}/validation_loss_{file_num}.dat",
            validation_loss,
            delimiter=",",
        )


train(TRAINING_ITERATIONS)

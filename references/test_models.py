#!/bin/python3
import numpy as np
import torch
from torch.utils.data import DataLoader
import os
import glob
import ANN
import utils_data

clean_one_seg_test_file = "training_data/clean_1_seg_2024_05_05_14_24_57.dat"
noisy_one_seg_test_file = "training_data/noisy_1_seg_2024_05_05_14_24_57.dat"
clean_two_seg_test_file = "training_data/clean_2_seg_2024_05_05_14_24_58.dat"
noisy_two_seg_test_file = "training_data/noisy_2_seg_2024_05_05_14_24_58.dat"

clean_one_seg_dataset = ANN.Dataset()
clean_one_seg_dataset.load_from_file(clean_one_seg_test_file)
noisy_one_seg_dataset = ANN.Dataset()
noisy_one_seg_dataset.load_from_file(noisy_one_seg_test_file)
clean_two_seg_dataset = ANN.Dataset()
clean_two_seg_dataset.load_from_file(clean_two_seg_test_file)
noisy_two_seg_dataset = ANN.Dataset()
noisy_two_seg_dataset.load_from_file(noisy_two_seg_test_file)

clean_one_seg_model_files = glob.glob("models/run_05_09_2024a/*_clean_one_seg.pt")
noisy_one_seg_model_files = glob.glob("models/run_05_09_2024a/*_noisy_one_seg.pt")
clean_two_seg_model_files = glob.glob("models/run_05_09_2024a/*_clean_two_seg.pt")
noisy_two_seg_model_files = glob.glob("models/run_05_09_2024a/*_noisy_two_seg.pt")

one_seg_model = ANN.Model(
    input_dim=4, output_dim=6, hidden_layers=[32, 32], loss=ANN.PoseLoss()
)
two_seg_model = ANN.Model(
    input_dim=8, output_dim=6, hidden_layers=[32, 32], loss=ANN.PoseLoss()
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


clean_clean_one_seg_position_loss, clean_clean_one_seg_orientation_loss = test(
    one_seg_model, clean_one_seg_model_files, clean_one_seg_dataset
)
noisy_clean_one_seg_position_loss, noisy_clean_one_seg_orientation_loss = test(
    one_seg_model, noisy_one_seg_model_files, clean_one_seg_dataset
)
clean_clean_two_seg_position_loss, clean_clean_two_seg_orientation_loss = test(
    two_seg_model, clean_two_seg_model_files, clean_two_seg_dataset
)
noisy_clean_two_seg_position_loss, noisy_clean_two_seg_orientation_loss = test(
    two_seg_model, noisy_two_seg_model_files, clean_two_seg_dataset
)

np.savetxt(
    "output/c_c_1_pos_test_loss.dat", clean_clean_one_seg_position_loss, delimiter=","
)
np.savetxt(
    "output/c_c_1_tang_test_loss.dat",
    clean_clean_one_seg_orientation_loss,
    delimiter=",",
)
np.savetxt(
    "output/n_c_1_pos_test_loss.dat", noisy_clean_one_seg_position_loss, delimiter=","
)
np.savetxt(
    "output/n_c_1_tang_test_loss.dat",
    noisy_clean_one_seg_orientation_loss,
    delimiter=",",
)

np.savetxt(
    "output/c_c_2_pos_test_loss.dat", clean_clean_two_seg_position_loss, delimiter=","
)
np.savetxt(
    "output/c_c_2_tang_test_loss.dat",
    clean_clean_two_seg_orientation_loss,
    delimiter=",",
)
np.savetxt(
    "output/n_c_2_pos_test_loss.dat", noisy_clean_two_seg_position_loss, delimiter=","
)
np.savetxt(
    "output/n_c_2_tang_test_loss.dat",
    noisy_clean_two_seg_orientation_loss,
    delimiter=",",
)

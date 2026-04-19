#!/bin/python3
from pathlib import Path
from glob import glob
import numpy as np
import torch
import matplotlib.pyplot as plt
from camarillo_cc import CamarilloModel
from mike_cc import MikeModel
from utils_cc import (
    mike_2_webster_params,
    camarillo_2_webster_params,
    calculate_transform,
)
from ANN import Model, Dataset
from multi_input import MultiInputModel, MultiInputDataset
from utils_data import DataContainer

# Data import
test_file = "./test_data/kinematic_2024_07_29_17_06_11.dat"

test_container = DataContainer()
test_container.file_import(test_file)

cable_deltas, true_pos, true_tang = test_container.to_numpy()

standard_dataset = Dataset()
standard_dataset.load_from_DataContainer(test_container)

multi_input_dataset = MultiInputDataset(1)
multi_input_dataset.load_from_DataContainer(test_container)

# Model setup
cable_positions = [(4.0, 0.0), (0.0, 4.0), (-4.0, 0.0), (0.0, -4.0)]
mike_model = MikeModel(4, cable_positions, 64.0)

camarillo_stiffness = np.loadtxt(Path("../../tools/camarillo_stiffness"), delimiter=",")
ka, kb, kt = *camarillo_stiffness
segment_stiffness_vals = [(ka, kb)]
cable_stiffness_vals = [(kt, kt, kt, kt)]
segment_lengths = [64]
camarillo_model = CamarilloModel(
    cable_positions,
    segment_stiffness_vals,
    cable_stiffness_vals,
    segment_lengths,
    50,
)

standard_model = Model(4, 6, [32, 32])
multi_input_model = MultiInputModel(4, 1, 1, [32, 32])

standard_models_dir = Path("./models/standard/")
multi_input_models_dir = Path("./models/multi_input/")

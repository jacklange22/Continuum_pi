#!/bin/python3
import numpy as np
import time
from pathlib import Path
from continuum_arduino import ContinuumArduino
from continuum_aurora import ContinuumAurora
from kinematics import dcm_2_tang
from utils_data import DataContainer

# Script parameters
wait_time = 4

# Setup continuum robot
T_aurora_2_model = np.loadtxt(Path("../../tools/T_aurora_2_model"), delimiter=",")
T_tip_2_coil = np.loadtxt(Path("../../tools/T_tip_2_coil"), delimiter=",")

aurora = ContinuumAurora(T_aurora_2_model, T_tip_2_coil)
arduino = ContinuumArduino()

# Define parameters to determine cable displacements
cable_length = np.array([0.0] + [6.0] * 8 + [12.0] * 8)
cable_angle = np.concatenate([np.array([0.0]), np.arange(16) * 4 * np.pi / 16])

cable_deltas = np.zeros((4, 17))

for i in range(17):
    cable_deltas[:, i] = -cable_length[i] * np.array(
        [
            np.cos(cable_angle[i]),
            np.sin(cable_angle[i]),
            -np.cos(cable_angle[i]),
            -np.sin(cable_angle[i]),
        ]
    )

# Calculate important script parameters
num_points = len(cable_angle)
num_measurements = 2 * num_points * (num_points - 1)

# Allocate arrays for data collection
num_range = np.array(list(range(num_points)))
pos = np.nan * np.zeros((3, num_measurements))
tang = np.nan * np.zeros((3, num_measurements))
ordered_deltas = np.zeros((4, num_measurements))

# Data collection
print("i", "j", "#", "idx")
for i in range(num_points):
    # Get cable deltas for current point, and randomize order of other points
    desired_delta = cable_deltas[:, i]
    other_deltas = cable_deltas[:, np.random.permutation(num_range[num_range != i])]

    # Go through each other point, go to it, then to the desired point
    for j in range(num_points - 1):
        idx = 2 * (num_points - 1) * i + 2 * j
        print(i, j, 0, idx)

        # Move to other point
        arduino.write_dls(other_deltas[:, j])
        time.sleep(wait_time)

        # Collect data
        transforms = aurora.get_aurora_transforms(["0A"])
        T = aurora.get_T_tip_2_model(transforms["0A"])
        ordered_deltas[:, idx] = other_deltas[:, j]
        pos[:, idx] = T[0:3, 3]
        tang[:, idx] = dcm_2_tang(T[0:3, 0:3])

        # Move to desired point
        idx = 2 * (num_points - 1) * i + 2 * j + 1
        print(i, j, 1, idx)
        arduino.write_dls(desired_delta)
        time.sleep(wait_time)

        # Collect data
        transforms = aurora.get_aurora_transforms(["0A"])
        T = aurora.get_T_tip_2_model(transforms["0A"])
        ordered_deltas[:, idx] = desired_delta
        pos[:, idx] = T[0:3, 3]
        tang[:, idx] = dcm_2_tang(T[0:3, 0:3])


# Zero spine
arduino.write_dls(np.zeros(4))

# Output data
container = DataContainer()
container.prefix = "output/data"
container.set_date_and_time()
container.from_raw_data(
    container.date, container.time, 4, num_measurements, ordered_deltas, pos, tang
)
container.file_export()

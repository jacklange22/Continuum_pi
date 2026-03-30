#!/bin/python3
import numpy as np
import time
from continuum_aurora import ContinuumAurora
from pathlib import Path

T_aurora_2_model = np.loadtxt(Path("../../tools/T_aurora_2_model"), delimiter=",")
T_tip_2_coil = np.loadtxt(Path("../../tools/T_tip_2_coil"), delimiter=",")

aurora = ContinuumAurora(T_aurora_2_model, T_tip_2_coil, serial_port_name="COM5")

counter = 0
start = time.time_ns()
while counter < 40:
    transforms = aurora.get_aurora_transforms(["0A", "0B"])
    t = time.time_ns()
    print(
        counter,
        "|",
        (t - start) / 1e9,
        "\n",
        aurora.get_T_tip_2_model(transforms["0B"]),
    )
    counter += 1
    start = t

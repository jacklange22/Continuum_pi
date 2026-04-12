#!/bin/python3
import numpy as np
import matplotlib.pyplot as plt

# Parameters
step_size = 0.1

# Load data
data_1 = np.loadtxt("displacements_1.dat", delimiter=",")
data_5 = np.loadtxt("displacements_5.dat", delimiter=",")

# After passing the threshold, tension stops going, so the data array will be full of -1.  Find the actual data
data_1_lens = [0] * 4
data_5_lens = [0] * 4

for i in range(4):
    data_1_row = data_1[i, :]
    data_1_lens[i] = len(data_1_row[data_1_row != -1])

    data_5_row = data_5[i, :]
    data_5_lens[i] = len(data_5_row[data_5_row != -1])

# Plotting
fig, ax = plt.subplots(2, 2, sharex=True, sharey=True, figsize=[12, 9])
plt.suptitle(
    "Tip vs Cable Displacement During Tensioning", fontsize=19, fontweight="bold"
)
curves = []
for i in range(2):
    for j in range(2):
        idx = 2 * i + j
        # Extract for legend
        if idx == 0:
            curves.append(
                ax[i, j].scatter(
                    step_size * np.array(list(range(data_5_lens[idx]))),
                    data_5[idx, 0 : data_5_lens[idx]],
                )
            )

            curves.append(
                ax[i, j].scatter(
                    step_size * np.array(list(range(data_1_lens[idx]))),
                    data_1[idx, 0 : data_1_lens[idx]],
                )
            )
        # Plot normally
        else:
            ax[i, j].scatter(
                step_size * np.array(list(range(data_5_lens[idx]))),
                data_5[idx, 0 : data_5_lens[idx]],
            )

            ax[i, j].scatter(
                step_size * np.array(list(range(data_1_lens[idx]))),
                data_1[idx, 0 : data_1_lens[idx]],
            )
        ax[i, j].plot([0, 6], [5, 5])
        ax[i, j].plot([0, 6], [1, 1])
        ax[i, j].set_title(f"Motor {idx+1}", fontsize=15)

labels = ["Threshold = 5mm", "Threshold = 1mm"]
fig.legend(curves, labels, loc="upper right", fontsize=14)
fig.add_subplot(1, 1, 1, frame_on=False)
plt.tick_params(labelcolor="none", bottom=False, left=False)
plt.xlabel("Cable Displacement (mm)", fontsize=15)
plt.ylabel("Tip Displacement (mm)", fontsize=15)
plt.savefig("tension_displacement.png")
plt.show()

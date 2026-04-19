#!/bin/python3
import numpy as np
import matplotlib.pyplot as plt
import utils_data
from glob import glob

num_iterations = 10
folder = "./output/real_07_17_2024/"
# File inputs
# Datacontainer
container = utils_data.DataContainer()
container.file_import("./training_data/kinematic_2024_07_16_21_33_12.dat")
container.clean()
_, pos, _ = container.to_numpy()

# Training loss vs epoch
train_loss_files = sorted(glob(f"{folder}real_train*"))
train_loss = np.concatenate(
    [
        np.loadtxt(train_loss_files[i], delimiter=",").reshape((1, -1))
        for i in range(num_iterations)
    ],
    axis=0,
)

# Validation loss vs epoch
validation_loss_files = sorted(glob(f"{folder}real_validation*"))
validation_loss = np.concatenate(
    [
        np.loadtxt(validation_loss_files[i], delimiter=",").reshape((1, -1))
        for i in range(num_iterations)
    ],
    axis=0,
)

# Test loss vs model
pos_loss = np.loadtxt(f"{folder}real_pos_test_loss.dat", delimiter=",")
tang_loss = np.loadtxt(f"{folder}real_tang_test_loss.dat", delimiter=",")

avg_pos_loss = pos_loss.mean(axis=0)
avg_tang_loss = tang_loss.mean(axis=0)

# Means and stdevs
avg_train_loss = train_loss.mean(axis=0)
std_train_loss = np.std(train_loss, axis=0, ddof=1)

avg_validation_loss = validation_loss.mean(axis=0)
std_validation_loss = np.std(validation_loss, axis=0, ddof=1)


# Plotting
plt.figure()
plt.scatter(pos[0, :], pos[1, :], alpha=0.3)
plt.title("Training Data Position Measurements")
plt.xlabel("x (mm)")
plt.ylabel("y (mm)")

plt.figure()
x = np.arange(1, len(avg_train_loss) + 1)
plt.semilogy(x, avg_train_loss, label="Training Loss")
plt.fill_between(
    x,
    avg_train_loss - std_train_loss,
    avg_train_loss + std_train_loss,
    alpha=0.3,
    label="_clean one segment std",
)
plt.semilogy(x, avg_validation_loss, label="Validation Loss")
plt.fill_between(
    x,
    avg_validation_loss - std_validation_loss,
    avg_validation_loss + std_validation_loss,
    alpha=0.3,
    label="_clean one segment std",
)
plt.title("Average Batch Loss vs Epoch")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.ylim((1, 100))

plt.figure()
for i in range(0, 10, 2):
    plt.hist(pos_loss[i, :], 100, alpha=0.3, label=f"Model {i+1}")
plt.title("Position Error of Learned Model")
plt.xlabel("Norm of Error (mm)")
plt.ylabel("Count")
plt.legend(loc="upper right")

plt.figure()
for i in range(0, 10, 2):
    plt.hist(
        180 / np.pi * np.sqrt(tang_loss[i, :]),
        100,
        alpha=0.3,
        label=f"Model {i+1}",
    )
plt.title("Orientation Error of Learned Model")
plt.xlabel("Norm of Error (degrees)")
plt.ylabel("Count")
plt.legend(loc="upper right")

plt.figure()
plt.boxplot(np.transpose(pos_loss), showfliers=False)
plt.title("Position Error vs Model")
plt.xlabel("Model")
plt.ylabel("Position Error (mm)")

plt.figure()
plt.boxplot(np.transpose(tang_loss) * 180 / np.pi, showfliers=False)
plt.title("Orientation Error vs Model")
plt.xlabel("Model")
plt.ylabel("Orientation Error (degrees)")

plt.figure()
plt.hist(pos_loss[6, :], 50)
plt.title(f"Position Error of Best Model (Mean: {avg_pos_loss[6]:.3f} mm)")
plt.xlabel("Position Error (mm)")
plt.ylabel("Count")

plt.figure()
plt.hist(180 / np.pi * tang_loss[6, :], 50)
plt.title(
    f"Orientation Error of Best Model (Mean: ${180/np.pi*avg_tang_loss[6]:.3f}^\circ$)"
)
plt.xlabel("Orientation Error (degrees)")
plt.ylabel("Count")
plt.show()

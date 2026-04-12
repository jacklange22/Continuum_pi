#!/bin/python3
import numpy as np
import serial
import time
import continuum_arduino
import continuum_aurora
import kinematics

""" Tensioning
Cameron Wolfe 04/06/24

Tensions cables via incrementally tensioning them, and measuring displacement
from starting position.  If displacement exceeds a threshold, then the cables
are considered tensioned.  Repeats for each cable.
"""

# Parameters
distance_threshold = 2  # mm - Movement level to consider a cable "tightened"
max_delta = 15  # mm - Cannot pull a cable more than 15 mm from its starting length
step_size = 0.1  # mm - How many mm we pull the cable each step
num_motors = 4
servo_min = 1221
servo_max = 2813

# Load tools and setpoints
T_aurora_2_model = np.loadtxt("../../tools/T_aurora_2_model", delimiter=",")
T_tip_2_coil = np.loadtxt("../../tools/T_tip_2_coil", delimiter=",")

motor_setpoints = [int((servo_min + servo_max) / 2)] * num_motors

# Assuming that both pen and probe are plugged in, but it will work if just the probe is plugged in
probe_list = ["0A", "0B"]

# Open serial connections to arduino and aurora
arduino = continuum_arduino.init_arduino()
aurora = continuum_aurora.init_aurora()

# Go to setpoints
continuum_arduino.write_motor_vals(arduino, motor_setpoints)
time.sleep(0.5)

# Loosen cables
loose_length = [5, 5, 5, 5]
starting_motor_vals = continuum_arduino.one_seg_dl_2_motor_vals(
    loose_length, motor_setpoints
)
continuum_arduino.write_motor_vals(arduino, starting_motor_vals)
time.sleep(0.5)

# Get initial transform from aurora
starting_transforms = continuum_aurora.get_aurora_transforms(aurora, probe_list)

# Determine if both pen and coil are plugged in or not
coil = "0A"
if len(starting_transforms.keys()) == 2:
    coil = "0B"

# Prepare for tensioning
deltas = [0.0] * 4
displacements = -np.ones((4, int(max_delta / step_size) + 1))

# Loop through each motor
for i in range(4):

    # Go to loose position
    continuum_arduino.write_motor_vals(arduino, starting_motor_vals)
    time.sleep(2)

    # Get reference (starting) transform
    starting_transforms = continuum_aurora.get_aurora_transforms(aurora, probe_list)
    T_starting_tip_2_model = continuum_aurora.get_T_tip_2_model(
        starting_transforms[coil], T_aurora_2_model, T_tip_2_coil
    )
    T_model_2_starting_tip = np.linalg.inv(T_starting_tip_2_model)

    # Begin tensioning
    tensioning = True
    dl = [0.0] * 4
    meas_counter = 0
    while tensioning:

        # Reset dl each time
        dl[i] = 0

        # Get aurora transform and convert to transformation matrix
        transforms = continuum_aurora.get_aurora_transforms(aurora, probe_list)
        T_new_tip_2_model = continuum_aurora.get_T_tip_2_model(
            transforms[coil], T_aurora_2_model, T_tip_2_coil
        )

        # Find transformation matrix from current position to reference (starting) position
        T_new_tip_2_starting_tip = np.matmul(T_model_2_starting_tip, T_new_tip_2_model)

        # Get distance between old and current tip position
        dist = np.linalg.norm(T_new_tip_2_starting_tip[0:3, 3])
        displacements[i, meas_counter] = dist

        print_deltas = [f"{x:.2f}" for x in deltas]
        print(f"{i} [" + ", ".join(print_deltas) + f"] : {dist:.5f}")

        # Either the robot cable is tensioned, or something is fishy
        if dist > distance_threshold or abs(deltas[i]) > max_delta:
            tensioning = False

            if abs(deltas[i]) > max_delta:
                print(f"Cable {i+1} tightened by 15.  Something is off")

        # Not tensioned yet
        else:
            # Tighten a little further
            deltas[i] -= step_size
            dl[i] = deltas[i]
            meas_counter += 1
            motor_vals = continuum_arduino.one_seg_dl_2_motor_vals(
                dl, starting_motor_vals
            )
            continuum_arduino.write_motor_vals(arduino, motor_vals)
            time.sleep(0.5)

# Reset to loose position
continuum_arduino.write_motor_vals(arduino, starting_motor_vals)
time.sleep(1)

# Using deltas, calculate new "tensioned" motor setpoints
final_motor_vals = continuum_arduino.one_seg_dl_2_motor_vals(
    deltas, starting_motor_vals
)

# Go to the new setpoints
continuum_arduino.write_motor_vals(arduino, final_motor_vals)

# Output setpoints
file = open("../../tools/motor_setpoints", "w")
for i in range(4):
    if i < 3:
        file.write(str(final_motor_vals[i]) + ",")
    else:
        file.write(str(final_motor_vals[i]))
file.close()

# Output displacement data
np.savetxt("displacements.dat", displacements, delimiter=",")

# Get one final transform and display it
transforms = continuum_aurora.get_aurora_transforms(aurora, probe_list)
T_tip_2_model = continuum_aurora.get_T_tip_2_model(
    transforms[coil], T_aurora_2_model, T_tip_2_coil
)
print(T_tip_2_model)

# Close serial connections
arduino.close()
aurora.close()

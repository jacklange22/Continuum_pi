#!/bin/python3
import numpy as np
import serial
import time
from pathlib import Path
from continuum_arduino import ContinuumArduino
from continuum_aurora import ContinuumAurora
import kinematics

step_size = 0.1
loose_length = 1

arduino = ContinuumArduino(serial_port_name="COM3")

user_input = input("Would you like to start with existing setpoints? [Y/n] ")

if user_input == "n":
    arduino.reset_motor_setpoints()

deltas = np.zeros(arduino.num_motors, dtype=float)

state = 0
current_motor = 0

while state >= 0:
    arduino.write_dls(deltas)
    if state == 0:
        user_input = input(
            f"Input the motor you would like to tension (0:{arduino.num_motors-1}) or (e)xit "
        )

        if user_input.isdigit():
            selected_motor = int(user_input)
            if 0 <= selected_motor < arduino.num_motors:
                current_motor = selected_motor
                state = 1
        elif user_input == "e" or user_input == "exit":
            state = -1
        else:
            print("Invalid input: " + user_input)

    elif state == 1:
        user_input = input(
            "Current deltas: ["
            + ", ".join([f"{x:.1f}" for x in deltas.tolist()])
            + "] | (T)ighten, (l)oosen, or (e)xit? "
        )

        if user_input == "t" or user_input == "tighten":
            deltas[current_motor] -= step_size

        elif user_input == "l" or user_input == "loosen":
            deltas[current_motor] += step_size

        elif user_input == "e" or user_input == "exit":
            state = 0

        else:
            print("Invalid input: " + user_input)

print("Deltas:", deltas)
current_setpoints = arduino.dls_2_cmds(deltas)

user_input = input("Would you like to save? [y/N] ")

if user_input == "y":
    with open(Path(__file__).parent.parent.parent.joinpath("tools", "motor_setpoints"), "w") as f:
        for i in range(arduino.num_motors):
            if i != arduino.num_motors - 1:
                f.write(f"{current_setpoints[i]},")
            else:
                f.write(f"{current_setpoints[i]}")

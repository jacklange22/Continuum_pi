import numpy as np
import serial
import struct
import math
import time
from pathlib import Path
from typing import List


class ContinuumArduino:
    """
    ContinuumArduino class handles all serial communication with Arduino.
    """

    def __init__(
        self,
        num_motors: int = 4,
        setpoint_filename: str = Path("../../tools/motor_setpoints"),
        wheel_radii: np.ndarray = np.array([15, 15, 15, 15], dtype=float),
        oscillator_frequency_kHz: int = 25_000,
        servo_frequency: int = 324,
        serial_port_name: str = "Com3",
        timeout: float = 1.0,
        testing: bool = False,
    ):
        """
        Opens serial port, and initializes Arduino to receive motor commands.

        Args:
            num_motors: The number of motors in the system
            setpoint_filename: Location of motor setpoint text file
            wheel_radii: Radii of pulley wheels
            oscillator_frequency_kHz: Clock frequency of PWM driver in kHz
            servo_frequency: Frequency of PWM signal sent to servos in Hz
            serial_port_name: The name of the serial port
            timeout: Time in seconds for Serial communication timeout
            testing: Are we testing the system using a virtual serial port?

        Returns:
            A ContinuumArduino object with an active serial port.
        """

        # Servo parameters
        self.servo_min = 1221
        self.servo_max = 2813

        # Motor details
        assert 0 < num_motors <= 16
        self.num_motors = num_motors
        if setpoint_filename:
            self.load_motor_setpoints(setpoint_filename)
        else:
            self.motor_setpoints = np.full(num_motors, int((self.servo_min + self.servo_max) // 2))
        self.wheel_radii = wheel_radii

        # PWM driver details
        assert 23_000 <= oscillator_frequency_kHz <= 27_000
        assert 40 <= servo_frequency <= 1600
        self.oscillator_frequency = oscillator_frequency_kHz
        self.servo_frequency = servo_frequency

        # Transmission flags
        self.transmission_flags = {}
        self.transmission_flags["DLE"] = 0x10
        self.transmission_flags["STX"] = 0x02
        self.transmission_flags["ETX"] = 0x03
        self.transmission_flags["ACK"] = 0x06
        self.transmission_flags["ERR"] = 0x15

        # Packet type flags
        self.packet_flags = {}
        self.packet_flags["CMD"] = 0x04  # Motor command
        self.packet_flags["COM"] = 0x05  # Communication (ACK/ ERR)
        self.packet_flags["NUM"] = 0x07  # Number of motors command
        self.packet_flags["OHZ"] = 0x08  # Oscillator frequency of PWM driver
        self.packet_flags["SHZ"] = 0x09  # Servo (PWM) frequency of PWM driver

        # Transmission error flags
        self.error_flags = {}
        self.error_flags[0x01] = "Length does not match"
        self.error_flags[0x02] = "CRC does not match"
        self.error_flags[0x03] = "Unknown or incorrect packet flag"
        self.error_flags[0x04] = (
            "System has not been initialized (number of servos and frequencies set)"
        )
        self.error_flags[0x05] = "Number of motors does not match command sent"
        self.error_flags[0x06] = "Parameter update packet length is incorrect"
        self.error_flags[0x07] = "Invalid parameter value sent"
        self.error_flags[0x08] = "Invalid communication packet"

        # Check dimensions
        assert len(self.motor_setpoints) == self.num_motors == len(self.wheel_radii)

        # Initialize Arduino
        self.timeout = timeout
        self.arduino = serial.Serial(serial_port_name, 115200, timeout=timeout)
        time.sleep(2)
        # Initialize system if not in testing mode
        if not testing:
            num_motors_success = self.__write_num_motors()
            oscillator_freq_success = self.__write_oscillator_frequency()
            servo_freq_success = self.__write_servo_frequency()

            print(f"Number of motors initialized correctly: {num_motors_success}")
            print(
                f"Oscillator frequency initialized correctly: {oscillator_freq_success}"
            )
            print(f"Servo frequency initialized correctly: {servo_freq_success}")

    def dls_2_cmds(self, dls: np.ndarray) -> List[int]:
        """
        Converts changes in cable lengths to commands to send to the Aruino.

        Args:
            dls: Changes in cable lengths

        Returns:
            A list of PWM values for each motor (the commands to be sent)
        """
        cmds = (
            (
                self.motor_setpoints
                + (self.servo_max - self.servo_min)
                / (2 * np.pi / 3)
                * dls
                / self.wheel_radii
            )
            .round()
            .astype(int)
        ).tolist()

        return cmds

    def write_dls(self, dls: np.ndarray, attempts: int = 5) -> bool:
        """
        Converts cable length deltas (dls) to motor commands, then sends the
        commands over serial to the Arduino, waiting for an acknowledgement.

        Args:
            dls: Changes in cable length from 0 point
            attempts: Number of attempts before deciding transmission is unsuccessful

        Returns:
            bool: Was transmission successful?
        """
        if len(dls) != self.num_motors:
            print("Dimension of dls does not match num_motors")
            return False

        # Map cable displacements to motor commands
        cmds = self.dls_2_cmds(dls)

        packet = self.__build_packet("CMD", cmds)

        # Send packet to Arduino
        success = False
        attempt = 0
        while not success and attempt < attempts:
            self.__transmit_packet(packet)
            success = self.__get_response()
            attempt += 1

        return success

    def update_num_motors(
        self,
        num_motors: int,
        motor_setpoints: np.ndarray,
        wheel_radii: np.ndarray,
        attempts: int = 5,
    ):
        """
        Allows the user to change the number of motors that the system can control.

        Args:
            num_motors: The desired number of motors
            motor_setpoints: The servo setpoints representing 0 cable displacement
            wheel_radii: The radii of the cable pulleys
            attempts: Number of times to try to send the command

        Returns:
            Was transmission successful?
        """
        if not (num_motors == len(motor_setpoints) == len(wheel_radii)):
            print(
                "Number of motors must share same size as motor setpoints and wheel radii"
            )
            return False

        old_num_motors = self.num_motors
        old_motor_setpoints = self.motor_setpoints
        old_wheel_radii = self.wheel_radii

        self.num_motors = num_motors
        self.motor_setpoints = motor_setpoints
        self.wheel_radii = wheel_radii

        # Send packet to Arduino
        success = self.__write_num_motors(attempts)

        if not success:
            self.num_motors = old_num_motors
            self.motor_setpoints = old_motor_setpoints
            self.wheel_radii = old_wheel_radii

        return success

    def update_oscillator_frequency(self, freq_kHz: int, attempts: int = 5) -> bool:
        """
        Update the oscillator frequency of the Adafruit PWM driver after ensuring it is value.

        Args:
            freq_kHz: The desired frequency in kHz
            attempts: The number of attempts to make the transmission

        Returns:
            Was the transmission successful?
        """
        if 23_000 <= freq_kHz <= 27_000:
            old_oscillator_frequency = self.oscillator_frequency
            self.oscillator_frequency = freq_kHz
            success = self.__write_oscillator_frequency(attempts)
            if not success:
                self.oscillator_frequency = old_oscillator_frequency

            return success

        else:
            print("Invalid frequency given")
            return False

    def update_servo_frequency(self, freq: int, attempts: int = 5) -> bool:
        """
        Update the servo (PWM) frequency of the Adafruit PWM driver.

        Args:
            freq: The desired frequency in Hz
            attempts: The number of attempts to make the transmission

        Returns:
            Was the transmission successful?
        """

        if 40 <= freq <= 1600:
            old_servo_frequency = self.servo_frequency
            self.servo_frequency = freq
            success = self.__write_servo_frequency(attempts)

            if not success:
                self.servo_frequency = old_servo_frequency

            return success

        else:
            print("Invalid servo frequency given")
            return False

    def update_motor_setpoints(self, setpoints: np.ndarray):
        """
        Updates the motor setpoints of the system.

        Args:
            setpoints: An array containing the new setpoints

        Raises:
            AssertionError: If number of motor setpoints does not match the number
            of motors.
        """

        assert (
            len(setpoints) == self.num_motors
        ), "Setpoints do not match number of motors"

        self.motor_setpoints = setpoints.astype(int)

    def reset_motor_setpoints(self):
        """
        Resets the motor setpoints to the middle of the servo range
        """
        self.motor_setpoints = (
            np.ones(self.num_motors, dtype=int) * (self.servo_max - self.servo_min) // 2
        )

    def update_wheel_radii(self, wheel_radii: np.ndarray):
        """
        Updates the wheel radii of the system.

        Args:
            wheel_radii: An array containing the new wheel radii

        Raises:
            AssertionError: If number of wheel radii does not match the number of
            motors
        """

        assert (
            len(wheel_radii) == self.num_motors
        ), "Number of radii does not match number of motors"

        self.wheel_radii = wheel_radii.astype(float)

    def load_motor_setpoints(
        self,
        filename: str,
    ) -> np.ndarray:
        """
        Loads motor setpoints from a file.

        Args:
            filename: The name of the file to read from

        Returns:
            A numpy array containing the setpoints
        """
        self.motor_setpoints = np.loadtxt(Path(filename), delimiter=",", dtype=int)


    def crc_add_bytes(self, CRC: int, payload: bytearray) -> int:
        """
        Calculates a CRC checksum given a bytearray.

        Args:
            CRC: Starting CRC value (Set to 0 for transmission)
            payload: The byte array for which we are generating the CRC

        Returns:
            The CRC of the payload
        """

        CRC = CRC & 0xFF  # Ensure CRC is treated as uint8
        for byte in payload:
            for bit_num in range(8, 0, -1):
                thisBit = (byte >> (bit_num - 1)) & 1
                doInvert = (thisBit ^ ((CRC & 128) >> 7)) == 1
                CRC = (CRC << 1) & 0xFF  # Ensure the result is treated as uint8
                if doInvert:
                    CRC = CRC ^ 7
        return CRC

    def __build_payload(self, packet_flag: str, data: List[int]) -> bytearray:
        """
        Builds a payload (non transmission/ crc components of packet).

        Args:
            packet_flag: A packet flag denoting what type of packet is to be sent
            data: The data that is to be sent with the packet

        Returns:
            Payload bytearray which can be put into a packet
        """
        assert packet_flag in self.packet_flags

        payload = bytearray([self.packet_flags[packet_flag]])
        payload.append(2 * len(data))

        for datum in data:
            payload.extend(struct.pack("H", datum))

        return payload

    def __build_packet(self, packet_flag: str, data: List[int]) -> bytearray:
        """
        Builds a packet (bytearray) that can then be transmitted.

        Args:
            packet_flag: A packet flag denoting what type of packet is to be sent
            data: The data that is to be sent with the packet

        Returns:
            bytearray: The packet with all transmission bytes, a payload, and a crc, which can be transmitted to the Arduino
        """

        payload = self.__build_payload(packet_flag, data)
        crc = self.crc_add_bytes(0, payload)

        # Start packet
        packet = bytearray(
            [self.transmission_flags["DLE"], self.transmission_flags["STX"]]
        )

        # Build packet from payload
        for byte in payload:
            packet.append(byte)
            # Stuff DLE
            if byte == self.transmission_flags["DLE"]:
                packet.append(self.transmission_flags["DLE"])

        # Add CRC (and stuff DLE if need be)
        packet.append(crc)
        if crc == self.transmission_flags["DLE"]:
            packet.append(self.transmission_flags["DLE"])

        # End packet
        packet.extend([self.transmission_flags["DLE"], self.transmission_flags["ETX"]])

        return packet

    def __transmit_packet(self, packet: bytearray):
        """
        Transmit a packet to the arduino.

        Args:
            packet: The packet to be sent (including all transmission bytes and CRC)
        """
        # Flush and write packet
        self.arduino.flush()
        self.arduino.write(packet)

    def __get_response(self) -> bool:
        """
        Listen for response from Arduino to determine if communication was successful.

        Returns:
            Was communication successful (received ACK)?
        """
        start_time = time.time()
        buffer = bytearray()
        reading = False
        dle_high = False

        while not reading and (time.time() - start_time) < self.timeout:
            buffer += self.arduino.read()

            if not dle_high:
                if buffer and buffer[-1] == self.transmission_flags["DLE"]:
                    dle_high = True
            else:
                if buffer and buffer[-1] == self.transmission_flags["STX"]:
                    reading = True
                    buffer = bytearray()
                dle_high = False

        if not reading:
            print("Arduino serial timeout: no start sequence found")
            return False

        while reading and (time.time() - start_time) < self.timeout:
            buffer += self.arduino.read()

            if not dle_high:
                if buffer and buffer[-1] == self.transmission_flags["DLE"]:
                    dle_high = True
            else:
                if buffer and buffer[-1] == self.transmission_flags["ETX"]:
                    reading = False
                dle_high = False

        if reading:
            print("Arduino serial timeout: no end sequence found")
            return False

        # Extract payload and crc
        payload = buffer[:-3]
        crc = buffer[-3]

        # Expecting a communication type flag
        if payload[0] != self.packet_flags["COM"]:
            print("Non communication packet flag received from Arduino")
            return False

        # Unstuff DLEs
        unstuffed_payload = bytearray()
        dle = False
        for byte in payload:
            if dle:
                if byte == self.transmission_flags["DLE"]:
                    unstuffed_payload.append(byte)
                else:
                    unstuffed_payload.extend(
                        bytearray([self.transmission_flags["DLE"], byte])
                    )
                dle = False
            else:
                if byte == self.transmission_flags["DLE"]:
                    dle = True
                else:
                    unstuffed_payload.append(byte)
        # Check packet length match
        expected_length = unstuffed_payload[1]
        data = unstuffed_payload[2:]

        if len(data) != expected_length:
            print(
                "Data received from Arduino does not match the length it specified in the packet"
            )
            return False

        # Check packet length for communication packet specifically
        if expected_length < 2 or expected_length > 3:
            print(
                f"Invalid length ({expected_length}) for ACK/ ERR packet from Arduino"
            )
            return False

        # See if CRC matches what we expect (this should be zero)
        if crc != self.crc_add_bytes(0, unstuffed_payload):
            print("Arduino calculated CRC does not match our CRC (receiving packet).")
            return False

        # At this point, structure of packet is verified
        # Received ACK: successful transmission
        if data[:2] == bytearray(
            [self.transmission_flags["DLE"], self.transmission_flags["ACK"]]
        ):
            return True
        # Received ERR: unsuccessful transmission
        elif data[:2] == bytearray(
            [self.transmission_flags["DLE"], self.transmission_flags["ERR"]]
        ):
            print("Arduino sent ERR with the following error code:")
            print(f"{data[2]}:", self.error_flags[data[2]])
        # Received unexpected packet
        else:
            print("Arduino sent unexpected packet in response")

        return False

    def __write_num_motors(self, attempts: int = 5) -> bool:
        """
        Transmit command for Arduino to update the number of motors it controls.

        Args:
            attempts: How many times the system will attempt to send the packet.

        Returns:
            Was the transmission successful?
        """

        packet = self.__build_packet("NUM", [self.num_motors])

        success = False
        attempt = 0
        while not success and attempt < attempts:
            self.__transmit_packet(packet)
            success = self.__get_response()
            attempt += 1

        return success

    def __write_oscillator_frequency(self, attempts: int = 5) -> bool:
        """
        Transmit command for Arduino to update the PWM driver oscillator frequency.

        Args:
            attempts: How many times the system will attempt to send the packet.

        Returns:
            Was the transmission successful?
        """

        packet = self.__build_packet("OHZ", [self.oscillator_frequency])

        success = False
        attempt = 0
        while not success and attempt < attempts:
            self.__transmit_packet(packet)
            success = self.__get_response()
            attempt += 1

        return success

    def __write_servo_frequency(self, attempts: int = 5) -> bool:
        """
        Transmit command for Arduino to update the servo (PWM) frequency.

        Args:
            attempts: How many times the system will attempt to send the packet.

        Returns:
            Was the transmission successful?
        """
        packet = self.__build_packet("SHZ", [self.servo_frequency])

        success = False
        attempt = 0
        while not success and attempt < attempts:
            self.__transmit_packet(packet)
            success = self.__get_response()
            attempt += 1

        return success


"""
Following functions not in class are deprecated, but are included to maintain
compatibility with older scripts.
"""


# Initialize Arduino serial port
def init_arduino():
    return serial.Serial("COM3", 115200, timeout=1)


# write motor values (commands)
def write_motor_vals(
    device: serial.Serial, motor_vals: List[int], error_delay: float = 0.5
) -> bool:
    # Important bytes for transmission
    DLE = 0x10
    STX = 0x02
    ETX = 0x03
    ACK = 0x06
    ERR = 0x15

    # Ensure correct baud rate
    if device.baudrate != 115200:
        print("Incorrect baud rate: " + str(device.baudrate))
        return False

    num_motors = len(motor_vals)

    # Form bytearray to send to arduino
    packet = bytearray()
    packet.append(DLE)
    packet.append(STX)
    packet.extend(struct.pack("B", 2 * num_motors))
    for val in motor_vals:
        packed_val = struct.pack("H", val)

        dle_idx = packed_val.find(DLE)
        # DLE in bytes, stuff it
        if dle_idx == 0:
            packet.append(DLE)
            packet.extend(packed_val)
        elif dle_idx == 1:
            packet.extend(packed_val)
            packet.append(DLE)
        # No DLE in bytes
        else:
            packet.extend(packed_val)

    packet.append(DLE)
    packet.append(ETX)

    # Try transmitting byte array up to 5 times
    success = False
    errors = 0
    while not success and errors < 5:
        device.write(packet)

        output = device.read(7)

        if len(output) == 7:
            if output[4] == ACK:
                success = True
            elif output[4] == ERR:
                print("ERR received")
                time.sleep(error_delay)
                errors += 1
            else:
                print("Neither ACK or ERR received")
                time.sleep(error_delay)
                errors += 1

        else:
            if len(output) == 0:
                print("Timeout")
                time.sleep(error_delay)
                errors += 1
            else:
                print("Transmission error, received incorrect length response")
                time.sleep(error_delay)
                errors += 1

    return success


def one_seg_dl_2_motor_vals(dls: List[float], setpoints: List[int]) -> List[int]:

    assert len(dls) == len(setpoints) == 4

    servo_min = 1221
    servo_max = 2813
    wheel_radius = 15

    l_2_cmd = (servo_max - servo_min) / (2 * math.pi / 3 * wheel_radius)

    return [int(setpoints[i] + dls[i] * l_2_cmd) for i in range(4)]


def load_motor_setpoints(filename: str = "../tools/motor_setpoints") -> List[int]:
    file = open(filename, "r")
    values = file.readline()
    return [int(x) for x in values.split(",")]


def crc_add_bytes(CRC: int, payload: bytearray) -> int:
    """
    Calculates a CRC checksum given a bytearray.

    Args:
        CRC: Starting CRC value (Set to 0 for transmission)
        payload: The byte array for which we are generating the CRC

    Returns:
        The CRC of the payload
    """

    CRC = CRC & 0xFF  # Ensure CRC is treated as uint8
    for byte in payload:
        for bit_num in range(8, 0, -1):
            thisBit = (byte >> (bit_num - 1)) & 1
            doInvert = (thisBit ^ ((CRC & 128) >> 7)) == 1
            CRC = (CRC << 1) & 0xFF  # Ensure the result is treated as uint8
            if doInvert:
                CRC = CRC ^ 7
    return CRC

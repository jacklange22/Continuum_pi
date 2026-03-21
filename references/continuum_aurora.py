import numpy as np
import time
import serial
import struct
from kinematics import quat_2_dcm
from typing import List, Dict


class ContinuumAurora:
    """
    Handles all communication with Aurora tracker and creation of tip transforms.

    Attributes:
        T_aurora_2_model (np.ndarray): Transformation matrix from Aurora frame to Model frame
        T_tip_2_coil (np.ndarray): Transformation matrix from Tip frame to Coil frame
        serial_port (serial.Serial): The serial port to communicate with the Aurora
    """

    def __init__(
        self,
        T_aurora_2_model: np.ndarray,
        T_tip_2_coil: np.ndarray,
        serial_port_name: str = "COM5",
        timeout: float = 1,
    ):
        """
        Creates a ContinuumAurora object, opening the serial port

        Args:
            T_aurora_2_model: Transformation from Aurora to model frame, from rigid_registration
            T_tip_2_coil: Transformation from Tip to Coil frame, from rigid_registration
            serial_port_name: The location of the serial port
            timeout: Seconds before timeout of serial port
        """

        self.T_aurora_2_model = T_aurora_2_model
        self.T_tip_2_coil = T_tip_2_coil

        self.serial_port = serial.Serial(
            serial_port_name,
            115200,
            bytesize=serial.EIGHTBITS,
            parity=serial.PARITY_NONE,
            stopbits=serial.STOPBITS_ONE,
            timeout=timeout,
            xonxoff=False,
        )

    def get_aurora_transforms(
        self,
        probe_list: List[str],
        timeout: float = 1,
        attempts: int = 5,
    ) -> Dict[str, List[float]]:
        """
        Gets the raw transforms of the probes in probe_list from the Aurora.

        Args:
            probe_list: The serial numbers of the probes desired 0A, 0B, etc.
            timeout: Seconds before timeout
            attempts: How many attempts will be made before giving up

        Returns:
            A dictionary containing the raw transform for each probe provided, where
            a raw transform is comprised of a quaternion and a position.

            Example:
                aurora.get_aurora_transforms(["0A", "0B"]) = {
                    '0A': [q0, q1, q2, q3, t0, t1, t2],
                    '0B': [q0, q1, q2, q3, t0, t1, t2],
                }
        """
        output = {}
        counter = 0
        while not output and counter < attempts:
            self.__request_aurora_packet(probe_list)
            pkt = self.__read_aurora_packet(timeout)
            if pkt:
                try:
                    output = self.__parse_aurora_transforms(pkt)
                except:
                    print("Error with packet parsing, requesting another")
            counter += 1

        return output

    def read_aurora_transforms(
        self, probe_list: List[str], timeout: float = 1, attempts: int = 5
    ) -> Dict[str, List[float]]:
        """
        Reads the raw transforms of the probes in probe_list from an Aurora data stream.

        Use this function if track_server is used with no flags (a stream of data is
        being collected).

        Args:
            probe_list: The serial numbers of the probes desired 0A, 0B, etc.
            timeout: Seconds before timeout
            attempts: How many attempts will be made before giving up

        Returns:
            A dictionary containing the raw transform for each probe provided, where
            a raw transform is comprised of a quaternion and a position.

            Example:
                aurora.read_aurora_transforms(["0A", "0B"]) = {
                    '0A': [q0, q1, q2, q3, t0, t1, t2],
                    '0B': [q0, q1, q2, q3, t0, t1, t2],
                }
        """
        output = {}
        counter = 0
        while not output and counter < attempts:
            pkt = self.__read_aurora_packet(timeout)
            if pkt:
                try:
                    output = self.__parse_aurora_transforms(pkt)
                except:
                    print("Error with packet parsing, trying again")
            counter += 1

        return output

    def get_T_tip_2_model(
        self,
        raw_aurora_transform: List[float],
    ) -> np.ndarray:
        """
        Converts a raw aurora transform into the tip transform.

        Args:
            raw_aurora_transform: The values in the dictionary returned from
            get_aurora_transforms (Example: [q0, q1, q2, q3, t0, t1, t2])

        Returns:
            The transformation matrix from the tip of the robot to its base (the
            location of the Model frame).
        """

        T_coil_2_aurora = self.__aurora_transform_2_T(raw_aurora_transform)

        return np.matmul(
            self.T_aurora_2_model, np.matmul(T_coil_2_aurora, self.T_tip_2_coil)
        )

    def crc_add_bytes(self, CRC: int, payload: bytearray):
        """
        Calculates a CRC-8 checksum given a bytearray.

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

    def __request_aurora_packet(self, probe_list: List[str]):
        """
        Sends packet to Aurora requesting the transforms of the probes in probe_list

        Args:
            probe_list: A list of the probes desired (Example: ["0A", "0B"])
        """
        # Definitions
        DLE_BYTE = 0x10
        STX_BYTE = 0x02
        ETX_BYTE = 0x03
        GET_PROBE_TFORM = 0x13

        # Assemble payload
        payload = bytearray()
        payload.append(GET_PROBE_TFORM)  # packet type
        payload.append(len(probe_list))  # sent as uint8_t

        # Add port handle string characters
        for probe_sn in probe_list:
            payload_insert = bytearray([0x00, 0x00, 0x00, 0x00])
            this_porthandle_string = bytearray(probe_sn, "utf-8")
            assert (
                len(this_porthandle_string) == 2
            ), "Port handle string length must be 2!"
            payload_insert[: len(this_porthandle_string)] = this_porthandle_string
            payload.extend(payload_insert)

        # Compute CRC
        CRC_BYTE = self.crc_add_bytes(0, payload)
        payload.append(CRC_BYTE)
        if CRC_BYTE == DLE_BYTE:
            payload.append(CRC_BYTE)

        # Assemble message with DLE stuffing
        msg = bytearray([DLE_BYTE, STX_BYTE])  # first two bytes not stuffed
        for byte in payload:
            msg.append(byte)
            if byte == DLE_BYTE:  # handle DLE stuffing
                msg.append(DLE_BYTE)
        msg.extend([DLE_BYTE, ETX_BYTE])

        # Write message
        self.serial_port.flush()
        self.serial_port.write(msg)

    def __read_aurora_packet(self, timeout: float = 1) -> bytearray:
        """
        After requesting a packet from the Aurora, this gets the response.

        This can also be used to read from a stream of aurora packets (when
        track_server is in normal mode).

        Args:
            Timeout: Seconds before timeout

        Returns:
            A bytearray containing the full aurora packet including start/ end bytes
        """
        # Initial setup
        DLE_BYTE = 0x10
        STX_BYTE = 0x02
        ETX_BYTE = 0x03
        PKT_TYPE_TRANSFORM_DATA = 0x01
        message_max_size = 300
        buffer = bytearray()
        reading = False
        dle_high = False

        # Record the start time
        start_time = time.time()
        # Keep reading data until we find the start pattern or timeout
        while not reading and (time.time() - start_time) < timeout:
            buffer += self.serial_port.read()
            if not dle_high:
                if buffer and buffer[-1] == DLE_BYTE:
                    dle_high = True
            else:
                if buffer and buffer[-1] == STX_BYTE:
                    reading = True
                dle_high = False

        # Exit if start pattern not found within timeout
        if not reading:
            print("Aurora timeout: start pattern not found.")
            return None

        # Now, we are looking for the end pattern within the timeout
        buffer = bytearray([DLE_BYTE, STX_BYTE])
        while reading and (time.time() - start_time) < timeout:
            buffer += self.serial_port.read()

            if dle_high:
                if buffer[-1] == ETX_BYTE:
                    reading = False
                dle_high = False

            else:
                if buffer[-1] == DLE_BYTE:
                    dle_high = True

        if reading:
            print("Aurora timeout: end pattern not found.")
            return None

        return buffer

    def __unstuff_dle(self, packet: bytearray) -> bytearray:
        """
        Unstuffs (removes duplicate) DLEs from packet.

        All non-flag DLEs (any time a DLE byte is in the data) are "stuffed" with
        an additional DLE byte to avoid accidentally reading a transmission flag.
        This routine removes these duplicate DLE bytes, as we know we have read
        the start and end of the packet.

        Args:
            packet: A sequence of bytes

        Returns:
            The unstuffed packet

        Example:
            With 0x10 as DLE, [0x10, 0x02, 0x10, 0x10, 0x05, 0x10, 0x03] becomes
            [0x10, 0x02, 0x10, 0x05, 0x10, 0x03].
        """
        DLE = 0x10
        output = b""

        dle_flag = False
        for i in range(len(packet)):
            if packet[i] == DLE:
                if not dle_flag:
                    dle_flag = True
                else:
                    dle_flag = False
                    output += int.to_bytes(packet[i], 1, "little")
            else:
                if dle_flag:
                    output += int.to_bytes(packet[i - 1], 1, "little")
                    dle_flag = False
                output += int.to_bytes(packet[i], 1, "little")

        return output

    def __parse_aurora_transforms(self, packet: bytearray) -> Dict[str, List[float]]:
        """
        Parses Aurora transform data from a given packet.

        This method processes a packet containing Aurora transform data, removes
        stuffed DLE (Data Link Escape) characters, validates the presence of a
        transform, and then extracts transform data for each tool. The extracted
        data is organized into a dictionary where the keys are tool IDs and the
        values are lists of transform data.

        Args:
            packet: The byte array packet containing the Aurora transform data.

        Returns:
            A dictionary where each key is a tool ID (str) and the corresponding
            value is a list of floats representing the transform data for that tool.

        Raises:
            AssertionError: If the packet is not a transform (indicated by a flag).
        """
        # Get rid of stuffed DLEs
        packet = self.__unstuff_dle(packet)

        # Ensure we got a transform (can add error handling later)
        assert packet[2] == 0x01

        # Parse data packet
        num_transforms = packet[3]
        frame = int.from_bytes(packet[4:8], "little")
        transform_dict = {}

        # Go through all tools and get transform data
        for i in range(num_transforms):
            data_bytes = packet[8 + 36 * i : 8 + 36 * (i + 1) + 1]
            tool_id = data_bytes[0:2].decode("utf-8")
            transform_data = [0.0] * 8
            for j in range(8):
                transform_data_bytes = data_bytes[4 + 4 * j : 4 + 4 * (j + 1)]
                transform_data[j] = struct.unpack("f", transform_data_bytes)[0]

            transform_dict[tool_id] = transform_data

        return transform_dict

    def __aurora_transform_2_T(self, raw_aurora_transform: List[float]) -> np.ndarray:
        """
        Converts a raw aurora transform (q, t) into a transformation matrix.

        Args:
            raw_aurora_transform: A list of floats containing the quaternion and
            position of a given probe

        Returns:
            The corresponding transformation matrix
        """

        T = np.identity(4)

        q = np.array(raw_aurora_transform[0:4])
        t = np.array(raw_aurora_transform[4:-1])

        R = quat_2_dcm(q)

        T[0:3, 0:3] = R
        T[0:3, 3] = t

        return T


"""
The following functions are deprecated, but are kept for old scripts
"""


def init_aurora():
    return serial.Serial(
        "COM5",
        115200,
        bytesize=serial.EIGHTBITS,
        parity=serial.PARITY_NONE,
        stopbits=serial.STOPBITS_ONE,
        timeout=1,
        xonxoff=False,
    )


def crc_add_bytes(CRC: int, byte_array):
    CRC = CRC & 0xFF  # Ensure CRC is treated as uint8
    for byte in byte_array:
        for bit_num in range(8, 0, -1):
            thisBit = (byte >> (bit_num - 1)) & 1
            doInvert = (thisBit ^ ((CRC & 128) >> 7)) == 1
            CRC = (CRC << 1) & 0xFF  # Ensure the result is treated as uint8
            if doInvert:
                CRC = CRC ^ 7
    return CRC


def request_aurora_packet(serial_port: serial.Serial, probe_sn_array: List[str]):
    # Definitions
    DLE_BYTE = 0x10
    STX_BYTE = 0x02
    ETX_BYTE = 0x03
    GET_PROBE_TFORM = 0x13

    # Assemble payload
    payload = bytearray()
    payload.append(GET_PROBE_TFORM)  # packet type
    payload.append(len(probe_sn_array))  # sent as uint8_t

    # Add port handle string characters
    for probe_sn in probe_sn_array:
        payload_insert = bytearray([0x00, 0x00, 0x00, 0x00])
        this_porthandle_string = bytearray(probe_sn, "utf-8")
        assert len(this_porthandle_string) == 2, "Port handle string length must be 2!"
        payload_insert[: len(this_porthandle_string)] = this_porthandle_string
        payload.extend(payload_insert)

    # Compute CRC
    CRC_BYTE = crc_add_bytes(0, payload)
    payload.append(CRC_BYTE)  # CRC byte IS stuffed... (TODO: check this)

    # Assemble message with DLE stuffing
    msg = bytearray([DLE_BYTE, STX_BYTE])  # first two bytes not stuffed
    for byte in payload:
        msg.append(byte)
        if byte == DLE_BYTE:  # handle DLE stuffing
            msg.append(DLE_BYTE)
    msg.extend([DLE_BYTE, ETX_BYTE])

    # Write message
    serial_port.write(msg)


def get_aurora_packet(serial_port, timeout):
    # Initial setup
    DLE_BYTE = 0x10
    STX_BYTE = 0x02
    ETX_BYTE = 0x03
    PKT_TYPE_TRANSFORM_DATA = 0x01
    message_max_size = 300
    new_serial_data = bytearray()
    start_idx = None

    # Record the start time
    start_time = time.time()

    # Flush serial port
    serial_port.reset_input_buffer()

    # Keep reading data until we find the start pattern or timeout
    while start_idx is None and (time.time() - start_time) < timeout:
        if serial_port.in_waiting > 0:
            data = serial_port.read(serial_port.in_waiting)
            new_serial_data.extend(data)

            # Search for the start of the message
            start_idx = new_serial_data.find(bytearray([DLE_BYTE, STX_BYTE]))

            # If found, trim the data
            if start_idx != -1:
                new_serial_data = new_serial_data[start_idx:]

    # Exit if start pattern not found within timeout
    if start_idx is None:
        print("Timeout or start pattern not found.")
        return None

    # Now, we are looking for the end pattern within the timeout
    pkt = None
    while pkt is None and (time.time() - start_time) < timeout:
        end_idx = new_serial_data.find(bytearray([DLE_BYTE, ETX_BYTE]))
        if end_idx != -1:
            # Extract the message, considering DLE stuffing
            message = bytearray()
            i = 0
            while i < end_idx + 2:  # Include ETX_BYTE
                message.append(new_serial_data[i])
                # Skip the stuffed DLE byte
                if (
                    new_serial_data[i] == DLE_BYTE
                    and i + 1 < len(new_serial_data)
                    and new_serial_data[i + 1] == DLE_BYTE
                ):
                    i += 1
                i += 1

            # Remove the processed message from buffer
            new_serial_data = new_serial_data[end_idx + 2 :]
            pkt = message  # Placeholder, replace with actual parsing result

        # Read more data if needed
        if serial_port.in_waiting > 0:
            data = serial_port.read(serial_port.in_waiting)
            new_serial_data.extend(data)

    if (time.time() - start_time) >= timeout:
        print("Serial read timeout!")

    return


def unstuff_dle(pkt: bytearray) -> bytearray:
    DLE = 0x10
    output = b""

    dle_flag = False
    for i in range(len(pkt)):
        if pkt[i] == DLE:
            if not dle_flag:
                dle_flag = True
            else:
                dle_flag = False
                output += int.to_bytes(pkt[i], 1, "little")
        else:
            if dle_flag:
                output += int.to_bytes(pkt[i - 1], 1, "little")
                dle_flag = False
            output += int.to_bytes(pkt[i], 1, "little")

    return output


def parse_aurora_transforms(pkt: bytearray) -> Dict[str, List[float]]:

    # Get rid of stuffed DLE's
    pkt = unstuff_dle(pkt)

    # Ensure we got a transform (can add error handling later)
    assert pkt[2] == 0x01

    # Parse data packet
    num_transforms = pkt[3]
    frame = int.from_bytes(pkt[4:8], "little")
    transform_dict = {}

    # Go through all tools and get transform data
    for i in range(num_transforms):
        data_bytes = pkt[8 + 36 * i : 8 + 36 * (i + 1) + 1]
        tool_id = data_bytes[0:2].decode("utf-8")
        transform_data = [0.0] * 8
        for j in range(8):
            transform_data_bytes = data_bytes[4 + 4 * j : 4 + 4 * (j + 1)]
            transform_data[j] = struct.unpack("f", transform_data_bytes)[0]

        transform_dict[tool_id] = transform_data

    return transform_dict


def get_aurora_transforms(
    serial_port: serial.Serial,
    probe_list: List[str],
    timeout: float = 1,
    attempts: int = 5,
) -> Dict[str, List[float]]:
    output = {}
    counter = 0
    while not output and counter < attempts:
        request_aurora_packet(serial_port, probe_list)
        pkt = get_aurora_packet(serial_port, timeout)
        if pkt:
            try:
                output = parse_aurora_transforms(pkt)
            except:
                print("Error with packet parsing, requesting another")
        serial_port.flush()
        counter += 1

    return output


def _aurora_transform_2_T(aurora_transform: List[float]) -> np.ndarray:
    T = np.identity(4)

    q = np.array(aurora_transform[0:4])
    t = np.array(aurora_transform[4:-1])

    R = quat_2_dcm(q)

    T[0:3, 0:3] = R
    T[0:3, 3] = t

    return T


def get_T_tip_2_model(
    aurora_transform: List[float],
    T_aurora_2_model: np.ndarray,
    T_tip_2_coil: np.ndarray,
) -> np.ndarray:
    T_coil_2_aurora = _aurora_transform_2_T(aurora_transform)

    return np.matmul(T_aurora_2_model, np.matmul(T_coil_2_aurora, T_tip_2_coil))


def __parse_aurora_transforms(self, pkt: bytearray) -> Dict[str, List[float]]:
    # Get rid of stuffed DLEs
    pkt = self.__unstuff_dle(pkt)

    # Ensure we got a transform (can add error handling later)
    assert pkt[2] == 0x01

    # Parse data packet
    num_transforms = pkt[3]
    frame = int.from_bytes(pkt[4:8], "little")
    transform_dict = {}

    # Go through all tools and get transform data
    for i in range(num_transforms):
        data_bytes = pkt[8 + 36 * i : 8 + 36 * (i + 1) + 1]
        tool_id = data_bytes[0:2].decode("utf-8")
        transform_data = [0.0] * 8
        for j in range(8):
            transform_data_bytes = data_bytes[4 + 4 * j : 4 + 4 * (j + 1)]
            transform_data[j] = struct.unpack("f", transform_data_bytes)[0]

        transform_dict[tool_id] = transform_data

    return transform_dict

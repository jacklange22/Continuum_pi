import datetime
import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Tuple, List, Dict
import matplotlib.pyplot as plt


@dataclass
class DataContainer:
    date: Tuple[int, int, int] = None
    time: Tuple[int, int, int] = None
    num_cables: int = None
    num_measurements: int = None
    inputs: List[np.ndarray] = field(default_factory=list)
    outputs: List[np.ndarray] = field(default_factory=list)
    num_auroras: int = 1
    prefix: str = "data"

    def file_export(self, filename: str = None):
        if not filename:
            filename = (
                self.prefix
                + f"_{self.date[0]:02n}_{self.date[1]:02n}_{self.date[2]:02n}_{self.time[0]:02n}_{self.time[1]:02n}_{self.time[2]:02n}.dat"
            )

        with open(filename, "w") as file:
            file.write(f"DATE: {self.date[0]}-{self.date[1]}-{self.date[2]}\n")
            file.write(f"TIME: {self.time[0]}-{self.time[1]}-{self.time[2]}\n")
            file.write(f"NUM_CABLES: {self.num_cables}\n")
            file.write(f"NUM_AURORAS: {self.num_auroras}\n")
            file.write(f"NUM_MEASUREMENTS: {self.num_measurements}\n")
            file.write("---\n")

            counter = 0
            for input, output in zip(self.inputs, self.outputs):
                file.write(f"{counter},")

                for input_val in input:
                    file.write(f"{input_val},")

                for i in range(len(output)):
                    if i < len(output) - 1:
                        file.write(f"{output[i]},")
                    else:
                        file.write(f"{output[i]}\n")

                counter += 1

    def file_import(self, filename: str):
        with open(filename, "r") as file:
            date_line = file.readline()
            date_list = date_line.split(":")
            assert date_list[0] == "DATE"

            self.date = tuple([int(x) for x in date_list[1].split("-")])

            time_line = file.readline()
            time_list = time_line.split(":")
            assert time_list[0] == "TIME"

            self.time = tuple([int(x) for x in time_list[1].split("-")])

            num_cables_line = file.readline()
            self.num_cables = int(num_cables_line.split(":")[1])

            num_auroras_line = file.readline()
            num_auroras_list = num_auroras_line.split(":")
            self.num_auroras = int(num_auroras_list[1])

            num_measurements_line = file.readline()
            num_measurements_list = num_measurements_line.split(":")
            assert num_measurements_list[0] == "NUM_MEASUREMENTS"

            self.num_measurements = int(num_measurements_list[1])

            spacer = file.readline()
            assert spacer.strip() == "---"

            num_outputs = 6 * self.num_auroras

            while line := file.readline():
                row = line.split(",")
                self.inputs.append(
                    np.array(
                        [float(x) for x in row[1 : self.num_cables + 1]],
                        dtype=float,
                    )
                )
                self.outputs.append(
                    np.array(
                        [float(x) for x in row[self.num_cables + 1 :]],
                        dtype=float,
                    )
                )

            assert len(self.inputs) == len(self.outputs) == self.num_measurements

    def from_raw_data(
        self,
        date: Tuple[int, int, int],
        time: Tuple[int, int, int],
        num_cables: int,
        num_measurements: int,
        cable_deltas: np.ndarray,
        positions: np.ndarray,
        orientations: np.ndarray,
        num_auroras: int = 1,
    ):
        assert cable_deltas.shape == (num_cables, num_measurements)
        assert positions.shape == (3 * num_auroras, num_measurements)
        assert orientations.shape == (3 * num_auroras, num_measurements)

        self.date = date
        self.time = time
        self.num_cables = num_cables
        self.num_measurements = num_measurements
        self.num_auroras = num_auroras

        self.inputs = []
        self.outputs = []

        input_size = num_cables
        output_size = num_auroras * 6

        for i in range(num_measurements):
            self.inputs.append(cable_deltas[:, i].flatten())
            self.outputs.append(
                np.concatenate([positions[:, i], orientations[:, i]], axis=0).flatten()
            )

    def set_date_and_time(self):
        now = datetime.datetime.now()

        self.date = (now.year, now.month, now.day)
        self.time = (now.hour, now.minute, now.second)

    def to_numpy(self):
        cable_deltas = np.concatenate([x.reshape((-1, 1)) for x in self.inputs], axis=1)

        pos = np.concatenate([x[:3].reshape((-1, 1)) for x in self.outputs], axis=1)
        tang = np.concatenate([x[3:].reshape((-1, 1)) for x in self.outputs], axis=1)

        return cable_deltas, pos, tang

    def clean(self, pos_threshold=128, tang_threshold=np.pi):
        bad_indices = []
        for i in range(self.num_measurements):
            has_nan = np.isnan(self.inputs[i]).any() or np.isnan(self.outputs[i]).any()
            bad_pos = (np.abs(self.outputs[i][:3]) > pos_threshold).any()
            bad_tang = (np.abs(self.outputs[i][3:]) > tang_threshold).any()
            if has_nan or bad_pos or bad_tang:
                bad_indices.append(i)

        for idx in reversed(sorted(bad_indices)):
            self.inputs.pop(idx)
            self.outputs.pop(idx)
            self.num_measurements -= 1


def parse_aurora_csv(
    filename: str,
) -> Dict[str, List[Tuple[np.ndarray, np.ndarray, float]]]:

    df = pd.read_csv(filename, header=None)

    probes = pd.unique(df.iloc[:, 2])

    output = {}
    for probe in probes:
        output[probe] = []
        probe_df = df[df[2] == probe]
        qs = np.transpose(probe_df.iloc[:, 3:7].to_numpy())
        ts = np.transpose(probe_df.iloc[:, 7:10].to_numpy())
        rms = probe_df.iloc[:, 13].to_numpy()

        for i in range(qs.shape[1]):
            output[probe].append((qs[:, i], ts[:, i], rms[i].item()))

    return output

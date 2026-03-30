#!/bin/python3
import time
import numpy as np
import matplotlib.pyplot as plt
import continuum_aurora

NUM_SAMPLES = 500

aurora = continuum_aurora.init_aurora()

times = np.zeros(NUM_SAMPLES + 1)

for i in range(NUM_SAMPLES):
    times[i] = time.perf_counter_ns()
    trans = continuum_aurora.get_aurora_transforms(aurora, ["0A"])

times[-1] = time.perf_counter_ns()

deltas = (times[1:] - times[:-1]) / 1e6

average_time = deltas.mean()
std = np.std(deltas, ddof=1)

plt.figure(1, figsize=[12,9])
plt.hist(deltas, bins=100)
plt.title("Time to Request Aurora Packet", fontsize=19, fontweight='bold')
plt.xlabel("Time (ms)", fontsize=15)
plt.ylabel("Count", fontsize=15)
plt.legend([f"Time (mean = {average_time:.2f}, std = {std:.2f})"], fontsize=14)
plt.show()

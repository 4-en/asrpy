# testing artifact subspace reconstruction (ASR) for real-time EEG data

# https://patentimages.storage.googleapis.com/c3/6b/dc/7dadfae33c0062/US20160113587A1.pdf
# https://sccn.ucsd.edu/githubwiki/files/asr-final-export.pdf
# https://github.com/DiGyt/asrpy
# https://github.com/sccn/clean_rawdata
# https://pmc.ncbi.nlm.nih.gov/articles/PMC4710679/
# https://github.com/nbara/python-meegkit

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import csv
from asrpy import ASR


data_path = "data/raw_eeg_14c.csv"

def load_csv_eeg(file_path: str, columns=14) -> np.ndarray:
    """Load EEG data from a csv file
    
    Parameters
    ----------
    file_path : str
        Path to the file with EEG data

    Returns
    -------
    np.ndarray
        EEG data
    """
    
    data = []
    with open(file_path, 'r') as f:
        reader = csv.reader(f)
        for i, row in enumerate(reader):
            if i == 0:
                continue
            row_data = row[:columns]
            if len(row_data) < columns:
                # print(f"Row {i} has less than {columns} columns")
                continue
            data.append(row_data)
            
    return np.array(data, dtype=float)

data = load_csv_eeg(data_path)
print(f"Data shape: {data.shape}")
data = data.T

from live_digital_filter import MultiChannelSOSFilter

bandpass_filter = MultiChannelSOSFilter([0.1, 10], 100, "bandpass", 14)

filtered_data = bandpass_filter(data)
print(f"Filtered data shape: {filtered_data.shape}")

import matplotlib.pyplot as plt
def compare_samples(data: list[np.ndarray], labels:list[str]|None=None, channel=0, length=1000, start=0):
    """Plots two samples of data for comparison"""
    plt.figure(figsize=(15, 5))
    for i, sample in enumerate(data):
        if labels is not None:
            label = labels[i]
        else:
            label = f"Sample {i}"
        plt.plot(sample[channel, start:start+length], label=label)
    plt.legend()
    plt.show()

# compare_sample(data, filtered_data)


class _StreamSimulatorIterator:
    def __init__(self, stream_simulator):
        self._stream_simulator = stream_simulator
        self._index = 0
    
    def __iter__(self):
        return self
    
    def __next__(self):
        if self._index >= len(self._stream_simulator):
            raise StopIteration
        chunk = self._stream_simulator.get_chunk(self._index)
        self._index += 1
        return chunk
        
    

class StreamSimulator:
    """Simulate a stream of data to test real-time processing"""
    def __init__(self, data, chunk_size=1):
        self.data = data
        self.chunk_size = chunk_size

    def get_chunk(self, index):
        if index >= len(self):
            raise ValueError("Index out of range")

        chunk = self.data[:, index * self.chunk_size: index * self.chunk_size + self.chunk_size]

        return chunk
    
    def __iter__(self):
        return _StreamSimulatorIterator(self)
    
    def __len__(self):
        return self.data.shape[1] // self.chunk_size
    

# setup the stream simulator
import tqdm
stream_simulator = StreamSimulator(filtered_data, 10)
    
    
from asrpy import asr_calibrate, asr_process, RTASR

# test ASR on offline data first
M, T = asr_calibrate(filtered_data[:, :6000], 100)
asr_data = asr_process(filtered_data, 100, M, T, stepsize=10)

        

rtasr = RTASR(100, 14, 0.25, True, 60)
#rtasr._M = M
#rtasr._T = T
#rtasr._was_calibrated = True

rtasr_data = np.zeros((14, 1))

for chunk in tqdm.tqdm(stream_simulator):
    processed_data = rtasr.process(chunk)
    if processed_data is not None:
        rtasr_data = np.concatenate((rtasr_data, processed_data), axis=1)

rtasr_data = rtasr_data[:, 1:]
print(f"Output data shape: {rtasr_data.shape}")

lookahead = 0.25
roll = int(lookahead * 100)
rtasr_data = np.roll(rtasr_data, -roll, axis=1)
asr_data = np.roll(asr_data, -roll, axis=1)

compare_samples([filtered_data, asr_data, rtasr_data], ["No ASR", "ASR Offline", "ASR Online"], start=100*153, length=1000)

carry = rtasr._carry
print(f"Carry shape: {carry.shape}")
for i, sample in enumerate(carry[0]):
    if i % 10 == 0:
        print(f"Sample {i}: {sample}")


def measure_outliers(data: np.ndarray, threshold: float = 3) -> np.ndarray:
    """Measure outliers in the data using the threshold method"""
    mean = np.mean(data, axis=1)
    std = np.std(data, axis=1)
    z_scores = np.abs((data - mean[:, None]) / std[:, None])
    outliers = z_scores > threshold
    return np.sum(outliers, axis=1)

def print_signal_info(data: np.ndarray, name=None):
    if name is not None:
        print(f"Signal: {name}")
    print(f"Shape: {data.shape}")
    # print(f"Mean: {np.mean(data, axis=1)}")
    # print(f"Std: {np.std(data, axis=1)}")
    print(f"Outliers: {measure_outliers(data)}")

print_signal_info(data[:,10000:], "Original")
print_signal_info(filtered_data[:,10000:], "Filtered")
print_signal_info(asr_data[:,10000:], "ASR")
print_signal_info(rtasr_data[:,10000:], "RTASR")

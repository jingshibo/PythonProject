##
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.interpolate import PchipInterpolator
import os
import datetime
import csv
import json


## plot the pulses of imu and emg for alignment
def plotImuData(imu_data, start_index, end_index):
    # if you mean the third column by position, use iloc
    imu_sync = imu_data.iloc[:, 2]

    # extract the portion you want to plot
    segment = imu_sync.iloc[start_index:end_index]

    # create figure + axis
    fig, ax = plt.subplots(figsize=(8, 4))

    ax.plot(range(len(segment)), segment, label="imu_sync")
    ax.set(title="imu_sync", ylabel="pulse")
    ax.tick_params(labelbottom=True)
    ax.legend(loc="upper right")

    plt.tight_layout()
    plt.show()

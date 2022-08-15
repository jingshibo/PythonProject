##
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import json

## plot split line in order to seperate the gait cycle
def plotSplitLine(left_insole_dataframe, right_insole_dataframe, emg_dataframe, start_index, end_index,
                  left_force_baseline, right_force_baseline):
    left_total_force = left_insole_dataframe.iloc[:, 192]
    right_total_force = right_insole_dataframe.iloc[:, 192]
    emg_data = emg_dataframe.sum(1)
    left_length = len(left_total_force.iloc[start_index:end_index])
    right_length = len(right_total_force.iloc[start_index:end_index])
    emg_length = len(emg_data.iloc[start_index:end_index])

    # plot emg and insole force
    fig, axes = plt.subplots(nrows=3, ncols=1, sharex=True)
    axes[0].plot(range(left_length), left_total_force.iloc[start_index:end_index], label="Left Insole Force")
    axes[1].plot(range(right_length), right_total_force.iloc[start_index:end_index], label="Right Insole Force")
    axes[2].plot(range(emg_length), emg_data.iloc[start_index:end_index], label="Emg Signal")

    # add force baseline
    axes[0].plot(range(left_length), left_length * [left_force_baseline], label="Left Force Baseline")
    axes[1].plot(range(right_length), right_length * [right_force_baseline], label="Right Force Baseline")

    # find intersection point's x value
    left_x = np.array(range(left_length))
    right_x = np.array(range(right_length))
    left_force = left_total_force.iloc[start_index:end_index].to_numpy()
    right_force = right_total_force.iloc[start_index:end_index].to_numpy()

    left_baseline = np.full(left_length, left_force_baseline)
    right_baseline = np.full(right_length, right_force_baseline)
    left_cross_idx = np.argwhere(np.diff(np.sign(left_force - left_baseline))).flatten()
    right_cross_idx = np.argwhere(np.diff(np.sign(right_force - right_baseline))).flatten()

    # plot intersection point's x value
    axes[0].plot(left_x[left_cross_idx], left_baseline[left_cross_idx], 'r')  # plot intersection points
    for i, x_value in enumerate(left_cross_idx):  # annotate intersection points
        axes[0].annotate(x_value, (left_x[left_cross_idx[i]], left_baseline[left_cross_idx[i]]), fontsize=10)
    axes[1].plot(right_x[right_cross_idx], right_baseline[right_cross_idx], 'r')  # plot intersection points
    for i, x_value in enumerate(right_cross_idx):  # annotate intersection points
        axes[1].annotate(x_value, (right_x[right_cross_idx[i]], right_baseline[right_cross_idx[i]]), fontsize=10)

    # plot parameters
    axes[0].set(title="Left Insole Force", ylabel="force(kg)")
    axes[1].set(title="Right Insole Force", ylabel="force(kg)")
    axes[2].set(title="Emg Signal", xlabel="Sample Number", ylabel="Emg Value")

    axes[0].tick_params(labelbottom=True)  # show x-axis ticklabels
    axes[1].tick_params(labelbottom=True)

    axes[0].legend(loc="upper right")
    axes[1].legend(loc="upper right")
    axes[2].legend(loc="upper right")

def saveSplitData(subject, split_data):
    data_dir = 'D:\Data\Insole_Emg'
    split_file = f'subject_{subject}\subject_{subject}_split.json'
    split_path = os.path.join(data_dir, split_file)

    with open(split_path, 'w') as json_file:
        json.dump(split_data, json_file, indent=8)

def readSplitData(subject):
    data_dir = 'D:\Data\Insole_Emg'
    split_file = f'subject_{subject}\subject_{subject}_split.json'
    split_path = os.path.join(data_dir, split_file)

    with open(split_path) as json_file:
        split_data = json.load(json_file)
    return split_data
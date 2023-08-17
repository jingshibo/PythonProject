##
import matplotlib.pyplot as plt
import numpy as np
import os
import json


## plot split line in order to seperate the gait cycle
# plot the left and right insole data, select an appropriate baseline to help identify the gait events
def plotSplitLine(left_insole_dataframe, right_insole_dataframe, emg_dataframe, start_index, end_index, left_force_baseline,
        right_force_baseline, emg1_missing_indicator=None, emg2_missing_indicator=None):
    left_total_force = left_insole_dataframe.iloc[:, 192]  # extract total force column
    right_total_force = right_insole_dataframe.iloc[:, 192]
    left_length = len(left_total_force.iloc[start_index:end_index])
    right_length = len(right_total_force.iloc[start_index:end_index])

    if emg_dataframe.shape[1] >= 64 and emg_dataframe.shape[1] < 128:  # if one emg device
        emg_data = emg_dataframe.iloc[:, 0:65].sum(axis=1)
        emg_data_2 = emg_data
    elif emg_dataframe.shape[1] >= 128 and emg_dataframe.shape[1] < 192:  # if two emg device
        emg_data = emg_dataframe.iloc[:, 0:65].sum(axis=1)
        emg_data_2 = emg_dataframe.iloc[:, 65:130].sum(axis=1)
    emg_length = len(emg_data.iloc[start_index:end_index])

    # plot emg and insole force
    fig, axes = plt.subplots(nrows=4, ncols=1, sharex=True)
    axes[0].plot(range(right_length), right_total_force.iloc[start_index:end_index], label="Right Insole Force")
    axes[1].plot(range(left_length), left_total_force.iloc[start_index:end_index], label="Left Insole Force")
    axes[2].plot(range(emg_length), emg_data.iloc[start_index:end_index], label="Emg 1 Signal")
    axes[3].plot(range(emg_length), emg_data_2.iloc[start_index:end_index], label="Emg 2 Signal")

    # add missing emg data indicator to both insole plots (the values are set to 1 at the place where the emg data lost)
    if emg1_missing_indicator is not None and emg2_missing_indicator is not None:
        axes[0].plot(range(emg_length), emg1_missing_indicator.iloc[start_index:end_index], label="emg1 missing data")
        axes[0].plot(range(emg_length), emg2_missing_indicator.iloc[start_index:end_index], label="emg2 missing data")
        axes[1].plot(range(emg_length), emg1_missing_indicator.iloc[start_index:end_index], label="emg1 missing data")
        axes[1].plot(range(emg_length), emg2_missing_indicator.iloc[start_index:end_index], label="emg2 missing data")

    # add force baseline
    axes[0].plot(range(right_length), right_length * [right_force_baseline], label="Right Force Baseline")
    axes[1].plot(range(left_length), left_length * [left_force_baseline], label="Left Force Baseline")

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
    axes[0].plot(right_x[right_cross_idx], right_baseline[right_cross_idx], 'r')  # plot intersection points
    for i, x_value in enumerate(right_cross_idx):  # annotate intersection points
        axes[0].annotate(x_value, (right_x[right_cross_idx[i]], right_baseline[right_cross_idx[i]]), fontsize=10)
    axes[1].plot(left_x[left_cross_idx], left_baseline[left_cross_idx], 'r')  # plot intersection points
    for i, x_value in enumerate(left_cross_idx):  # annotate intersection points
        axes[1].annotate(x_value, (left_x[left_cross_idx[i]], left_baseline[left_cross_idx[i]]), fontsize=10)

    # plot parameters
    axes[0].set(title="Right Insole Force", ylabel="force(kg)")
    axes[1].set(title="Left Insole Force", ylabel="force(kg)")
    axes[2].set(title="Emg 1 Signal", ylabel="Emg Value")
    axes[3].set(title="Emg 2 Signal", xlabel="Sample Number", ylabel="Emg Value")

    axes[0].tick_params(labelbottom=True)  # show x-axis ticklabels
    axes[1].tick_params(labelbottom=True)
    axes[2].tick_params(labelbottom=True)

    axes[0].legend(loc="upper right")
    axes[1].legend(loc="upper right")
    axes[2].legend(loc="upper right")
    axes[3].legend(loc="upper right")

    return left_cross_idx, right_cross_idx


# save the split parameters into a json file
def saveSplitParameters(subject, split_data, project='Insole_Emg'):
    data_dir = f'D:\Data\{project}'
    split_file = f'subject_{subject}\subject_{subject}_split_parameters.json'
    split_path = os.path.join(data_dir, split_file)

    with open(split_path, 'w') as json_file:
        json.dump(split_data, json_file, indent=8)


# read the split parameters from a json file
def readSplitParameters(subject, project='Insole_Emg'):
    data_dir = f'D:\Data\{project}'
    split_file = f'subject_{subject}\subject_{subject}_split_parameters.json'
    split_path = os.path.join(data_dir, split_file)

    with open(split_path) as json_file:
        split_data = json.load(json_file)
    return split_data

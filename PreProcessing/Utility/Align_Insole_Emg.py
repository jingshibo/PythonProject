##
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal

## align EMG and insoles
def alignInsoleEmg(raw_emg_data, left_insole_aligned, right_insole_aligned):
    # get the average beginning and ending insole timestamp
    left_insole_aligned[0] = pd.to_datetime(left_insole_aligned[0], format='%Y-%m-%d_%H:%M:%S.%f')
    right_insole_aligned[0] = pd.to_datetime(right_insole_aligned[0], format='%Y-%m-%d_%H:%M:%S.%f')
    insole_begin_timestamp = min(left_insole_aligned.iloc[0, 0], right_insole_aligned.iloc[0, 0]) # select earliest one
    insole_end_timestamp = min((left_insole_aligned[0].iloc[-1], right_insole_aligned[0].iloc[-1]))

    # only keep data between the beginning and ending index
    raw_emg_data[0] = pd.to_datetime(raw_emg_data[0], format='%Y-%m-%d_%H:%M:%S.%f')
    emg_start_index = (pd.to_datetime(raw_emg_data[0]) - insole_begin_timestamp).abs().idxmin() # obtain the closet beginning timestamp
    emg_end_index = (pd.to_datetime(raw_emg_data[0]) - insole_end_timestamp).abs().idxmin() # obtain the closet ending timestamp
    emg_aligned = raw_emg_data.iloc[emg_start_index:emg_end_index+1, :].reset_index(drop=True)
    return emg_aligned

## filter EMG signal
def filterEmg(emg_aligned, notch = True, quality_factor = 10):
    sos = signal.butter(4, [20, 400], fs=2000, btype="bandpass", output='sos')
    emg_bandpass_filtered = signal.sosfiltfilt(sos, emg_aligned.iloc[:, 3:67], axis=0)
    emg_filtered = emg_bandpass_filtered
    if notch:
        b, a = signal.iircomb(50, quality_factor, fs=2000, ftype='notch')
        emg_notch_filtered = signal.filtfilt(b, a, pd.DataFrame(emg_bandpass_filtered), axis=0)
        emg_filtered = emg_notch_filtered
    return pd.DataFrame(emg_filtered)

## plot insole and emg data
def plotInsoleEmg(emg_dataframe, left_insole_dataframe, right_insole_dataframe, start_index, end_index):
    left_total_force = left_insole_dataframe.iloc[:, 192] # extract total force column
    right_total_force = right_insole_dataframe.iloc[:, 192]
    emg_data = emg_dataframe.sum(1)

    # plot
    fig, axes = plt.subplots(nrows=3, ncols=1, sharex=True)
    axes[0].plot(range(len(left_total_force.iloc[start_index:end_index])), left_total_force.iloc[start_index:end_index],
                 label="Left Insole Force")
    axes[1].plot(range(len(right_total_force.iloc[start_index:end_index])),
                 right_total_force.iloc[start_index:end_index], label="Right Insole Force")
    axes[2].plot(range(len(emg_data.iloc[start_index:end_index])), emg_data.iloc[start_index:end_index],
                 label="Emg Signal")

    axes[0].set(title="Left Insole Force", ylabel="force(kg)")
    axes[1].set(title="Right Insole Force", ylabel="force(kg)")
    axes[2].set(title="Emg Signal", xlabel="Sample Number", ylabel="Emg Value")

    axes[0].tick_params(labelbottom=True)  # show x-axis ticklabels
    axes[1].tick_params(labelbottom=True)

    axes[0].legend(loc="upper right")
    axes[1].legend(loc="upper right")
    axes[2].legend(loc="upper right")

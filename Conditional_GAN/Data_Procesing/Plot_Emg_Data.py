'''
    plot average emg data (channel mean, repetition mean, all dataset mean) and frequency spectrum
'''


##
import copy
from scipy.fft import fft
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import math


## calculate the mean value of emg data to get average value of all channels/all repetitions/all channel and repetitions.
def averageEmgValues(emg_values):
    # calculate the mean value of all channels for each repetition
    emg_1_repetition_list = {}
    emg_2_repetition_list = {}
    # Loop through each key in the original dictionary
    for gait_event_label, gait_event_emg in emg_values.items():
        result_shape = (gait_event_emg[0].shape[0], len(gait_event_emg))
        # Initialize arrays to store the column-wise averages for the first and second parts for the current key
        result_for_key_part1 = np.zeros(result_shape)
        result_for_key_part2 = np.zeros(result_shape)
        # Loop through each list (of shape (1800, 130) in your case)
        for i, array in enumerate(gait_event_emg):
            # In this demonstration, I'll split the 130 columns into two
            array_part1, array_part2 = np.split(array, 2, axis=1)
            # Calculate the column-wise average for each part
            column_avg_part1 = np.mean(array_part1, axis=1)
            column_avg_part2 = np.mean(array_part2, axis=1)
            # Store the column-wise average in the corresponding position of the result_for_key_part arrays
            result_for_key_part1[:, i] = column_avg_part1
            result_for_key_part2[:, i] = column_avg_part2  # Store the (1800, 60) ndarrays in the result_dict for the current key and part
        emg_1_repetition_list[gait_event_label] = result_for_key_part1
        emg_2_repetition_list[gait_event_label] = result_for_key_part2

    # calculate the mean value of all repetitions for each channel
    emg_1_channel_list = {}
    emg_2_channel_list = {}
    # Loop through each key in the original dictionary
    for gait_event_label, gait_event_emg in emg_values.items():
        # Convert the list of arrays for each key to a 3D NumPy array and Calculate the mean along the first axis
        average_array = np.mean(np.stack(gait_event_emg), axis=0)
        # Store the average array in the dictionary with the corresponding key
        array_part1, array_part2 = np.split(average_array, 2, axis=1)
        # Store the split arrays in the dictionary with the corresponding key
        emg_1_channel_list[gait_event_label] = array_part1
        emg_2_channel_list[gait_event_label] = array_part2

    #  calculate the mean value of all data at each timepoint for a gait event
    emg_1_event_mean = {}  # emg device 1: tibialis
    emg_2_event_mean = {}  # emg device 2: rectus
    # Loop through each key in the original dictionary
    for gait_event_label, gait_event_emg in emg_values.items():
        emg_1_event_mean[gait_event_label] = np.mean(emg_1_channel_list[gait_event_label], axis=1)
        emg_2_event_mean[gait_event_label] = np.mean(emg_2_channel_list[gait_event_label], axis=1)

    # store all mean values in a dict
    emg_mean_values = {'emg_1_repetition_list': emg_1_repetition_list, 'emg_2_repetition_list': emg_2_repetition_list,
        'emg_1_channel_list': emg_1_channel_list, 'emg_2_channel_list': emg_2_channel_list,
        'emg_1_event_mean': emg_1_event_mean, 'emg_2_event_mean': emg_2_event_mean}
    return emg_mean_values


## plot the time series value of muliple columns from a locomotion mode
def plotAverageValue(emg_data, mode, num_columns=30, layout=None, title=None, ylim=None):
    # Define the plotting parameters
    if layout is None:
        vertical = int(math.ceil(math.sqrt(num_columns)))
        horizontal = int(math.ceil(num_columns / vertical))
    else:
        horizontal, vertical = layout
    # Plot using a single line of code
    (pd.DataFrame(emg_data[mode])).iloc[:, :num_columns].plot(subplots=True, layout=(horizontal, vertical),
        title=title, figsize=(15, 10), ylim=ylim)


##  Plotting multiple average values in a single plots
def plotMultipleModeValues(emg_list, title=None, ylim=(0, 0.5)):
    plt.figure(figsize=(10, 6))
    for label, arr in emg_list.items():
        plt.plot(arr, label=label)

    plt.title(title)
    plt.xlabel('Index')
    plt.ylabel('Value')
    plt.ylim(*ylim)
    plt.legend()
    plt.grid(True)
    plt.show(block=False)


## calculate and plot the PSD of mean emg values
def plotPsd(emg_data, mode, num_columns=30, layout=None, title=None):
    # compute the frequency spectrum of a 2D array
    def compute_frequency_spectrum(time_series_data, fs):
        N = time_series_data.shape[0]  # Number of data points
        freq = np.fft.fftfreq(N, 1/fs)  # Frequency bins
        fft_values = fft(time_series_data, axis=0)  # FFT along the time axis
        psd_values = np.abs(fft_values)**2 / N  # PSD values
        return freq, psd_values
    # plot the frequency spectrum of a 2D array
    def plot_frequency_spectrum(freq, magnitude, label):
        plt.plot(freq, magnitude)
        plt.title(label)
        plt.ylim(0, 0.1)
        plt.xlabel('Frequency (Hz)')
        plt.ylabel('Magnitude')
        plt.grid(True)

    if emg_data[mode].ndim == 1:  # draw only one plot using the single vector
        # Apply a Hanning window and compute PSD
        window = np.hanning(emg_data[mode].shape[0])
        windowed_data = emg_data[mode] * window
        freq, magnitude = compute_frequency_spectrum(windowed_data, 1000)
        # Plotting
        plt.figure(figsize=(14, 6))
        plot_frequency_spectrum(freq, magnitude, title)
        plt.tight_layout()
        plt.show(block=False)
    elif emg_data[mode].ndim == 2:  # draw multiple subplots for selected columns
        # Apply a Hanning window and compute PSD
        window = np.hanning(emg_data[mode].shape[0])
        windowed_real_data = emg_data[mode] * window[:, None]
        freq, magnitude = compute_frequency_spectrum(windowed_real_data[:, ], 1000)
        # Define the plotting parameters
        if layout is None:
            vertical = int(math.ceil(math.sqrt(num_columns)))
            horizontal = int(math.ceil(num_columns / vertical))
        else:
            horizontal, vertical = layout
        # plot
        fig, axes = plt.subplots(horizontal, vertical, figsize=(18, 15))
        fig.suptitle(title)
        # Flatten the 2D axes array to make it easier to iterate
        flattened_axes = axes.flatten()
        # Loop through each selected column
        for column in range(num_columns):
            # Select the subplot
            plt.sca(flattened_axes[column])
            # Plotting using the modified function
            label = f'{column + 1}'
            plot_frequency_spectrum(freq, magnitude[:, column], label)
        plt.tight_layout()
        plt.show(block=False)









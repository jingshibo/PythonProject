import copy
from scipy.fft import fft
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


## calculate the mean value of all repetitions for each channel and the mean value of all data from each transition event
def averageRepetitionAndEventValues(emg_values):
    emg_1_channel_list = {}
    emg_2_channel_list = {}
    emg_1_event_mean = {}  # emg device 1: tibialis
    emg_2_event_mean = {}  # emg device 2: rectus
    # sum the emg data of all channels and samples up for the same gait event
    for gait_event_label, gait_event_emg in emg_values.items():
        emg_1_channel_list[f"{gait_event_label}_data"] = [np.sum(emg_per_repetition[:, 0:65], axis=1) / 65 for emg_per_repetition in
            gait_event_emg]  # average the emg values of all channels
        emg_2_channel_list[f"{gait_event_label}_data"] = [np.sum(emg_per_repetition[:, 65:130], axis=1) / 65 for emg_per_repetition in
            gait_event_emg]  # average the emg values of all channels
        emg_1_event_mean[f"{gait_event_label}_data"] = np.add.reduce(emg_1_channel_list[f"{gait_event_label}_data"]) / len(
            emg_1_channel_list[f"{gait_event_label}_data"])  # average all repetitions for the same gait event
        emg_2_event_mean[f"{gait_event_label}_data"] = np.add.reduce(emg_2_channel_list[f"{gait_event_label}_data"]) / len(
            emg_2_channel_list[f"{gait_event_label}_data"])  # average all repetitions for the same gait event
    return emg_1_channel_list, emg_2_channel_list, emg_1_event_mean, emg_2_event_mean


## calculate the mean value of all channels for each repetition
def averageChannelValues(emg_value):
    emg_1_repetition_list = {}
    emg_2_repetition_list = {}
    # Loop through each key in the original dictionary
    for gait_event_label, gait_event_emg in emg_value.items():
        result_shape = (gait_event_emg[0].shape[0], len(gait_event_emg))
        # Initialize arrays to store the column-wise averages for the first and second parts for the current key
        result_for_key_part1 = np.zeros(result_shape)
        result_for_key_part2 = np.zeros(result_shape)
        # Loop through each list (of shape (1800, 130) in your case)
        for i, array in enumerate(gait_event_emg):
            # In this demonstration, I'll split the 130 columns into two
            array_part1, array_part2 = np.split(array, [65], axis=1)
            # Calculate the column-wise average for each part
            column_avg_part1 = np.mean(array_part1, axis=1)
            column_avg_part2 = np.mean(array_part2, axis=1)
            # Store the column-wise average in the corresponding position of the result_for_key_part arrays
            result_for_key_part1[:, i] = column_avg_part1
            result_for_key_part2[:, i] = column_avg_part2  # Store the (1800, 60) ndarrays in the result_dict for the current key and part
        emg_1_repetition_list[gait_event_label] = result_for_key_part1
        emg_2_repetition_list[gait_event_label] = result_for_key_part2
    return emg_1_repetition_list, emg_2_repetition_list


## plot the average value of all channels from selected repetitions in a mode
def plotAverageChannel(emg_data, mode, title=None, ylim=None):
    # Define the plotting parameters
    start_index = 0
    end_index = 30   # select 30 repetitions for plotting
    horizontal = 6
    vertical = 5
    # Plot using a single line of code
    (pd.DataFrame(emg_data[mode])).iloc[:, start_index:end_index].plot(subplots=True, layout=(horizontal, vertical),
        title=title, figsize=(15, 10), ylim=ylim)


## plot the PSD value of
def plotPsd(fake_data, real_data, mode):

    # compute the frequency spectrum of a 2D array
    def compute_frequency_spectrum(data, fs):
        N = data.shape[0]  # Number of data points
        freq = np.fft.fftfreq(N, 1/fs)  # Frequency bins
        fft_values = fft(data, axis=0)  # FFT along the time axis
        psd_values = np.abs(fft_values)**2 / N  # PSD values
        return freq, psd_values

    # plot the frequency spectrum of a 2D array
    def plot_frequency_spectrum(freq, magnitude, title):
        plt.plot(freq, magnitude)
        plt.title(title)
        plt.ylim(0, 0.1)
        plt.xlabel('Frequency (Hz)')
        plt.ylabel('Magnitude')
        plt.grid(True)

    # Compute the frequency spectrum for the first 2D array in the data list
    _, _, fake_1_event_mean, fake_2_event_mean = averageRepetitionAndEventValues(fake_data)
    _, _, real_1_event_mean, real_2_event_mean = averageRepetitionAndEventValues(real_data)
    fs = 1000  # Sampling frequency (Hz)

    # Apply a Hanning window to the first 2D array in the data list
    window = np.hanning(fake_1_event_mean[mode].shape)
    windowed_filtered_data = fake_1_event_mean[mode] * window[:, None]
    freq_filtered, magnitude_filtered = compute_frequency_spectrum(windowed_filtered_data, fs)

    window = np.hanning(real_1_event_mean[mode].shape[0])
    windowed_real_data = real_1_event_mean[mode] * window[:, None]
    freq_real, magnitude_real = compute_frequency_spectrum(windowed_real_data, fs)

    # Plotting
    plt.figure(figsize=(14, 6))
    plt.subplot(1, 2, 1)
    plot_frequency_spectrum(freq_filtered, magnitude_filtered[:, 0], 'filtered data (mean)')
    # Plot frequency spectrum of windowed data
    plt.subplot(1, 2, 2)
    plot_frequency_spectrum(freq_real, magnitude_real[:, 0], 'real data (mean)')
    plt.tight_layout()
    plt.show()










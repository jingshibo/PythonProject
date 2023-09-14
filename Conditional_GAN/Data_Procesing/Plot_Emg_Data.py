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
def averageEmgValues(emg_values, split=True):
    emg_repetition_list = {}
    emg_channel_list = {}
    emg_event_mean = {}

    for gait_event_label, gait_event_emg in emg_values.items():
        num_columns = gait_event_emg[0].shape[1]
        if split:  # calculate each electrode grid separately
            num_parts = num_columns // 65
        else:  # calculate all electrode grids as a whole
            num_parts = 1

        for i, array in enumerate(gait_event_emg):
            split_arrays = np.split(array, num_parts, axis=1)
            for j, part in enumerate(split_arrays):
                part_key = f'grid_{j + 1}'
                # Update emg_repetition_list
                if part_key not in emg_repetition_list:
                    emg_repetition_list[part_key] = {}
                if gait_event_label not in emg_repetition_list[part_key]:
                    result_shape = (array.shape[0], len(gait_event_emg))
                    emg_repetition_list[part_key][gait_event_label] = np.zeros(result_shape)
                emg_repetition_list[part_key][gait_event_label][:, i] = np.mean(part, axis=1)

        # Update emg_channel_list and emg_event_mean
        average_array = np.mean(np.stack(gait_event_emg), axis=0)
        split_avg_arrays = np.split(average_array, num_parts, axis=1)

        for i, part in enumerate(split_avg_arrays):
            part_key = f'grid_{i + 1}'
            if part_key not in emg_channel_list:
                emg_channel_list[part_key] = {}
            emg_channel_list[part_key][gait_event_label] = part
            if part_key not in emg_event_mean:
                emg_event_mean[part_key] = {}
            emg_event_mean[part_key][gait_event_label] = np.mean(part, axis=1)

    emg_mean_values = {'emg_repetition_list': emg_repetition_list, 'emg_channel_list': emg_channel_list, 'emg_event_mean': emg_event_mean}

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
def plotPsd(emg_data, mode, num_columns=30, layout=None, title=None, ylim=None):
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
        plt.ylim(*ylim)
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


##  plot average emg values from multiple locomotion modes in a single plot for comparison
def plotMultipleEventMeanValues(fake_data, real_data, modes, title=None, ylim=(0, 0.5), grid='grid_1'):
    mean_emg_to_plot = {f'fake_{modes[2]}': fake_data['emg_event_mean'][grid][modes[2]],
        f'real_{modes[2]}': real_data['emg_event_mean'][grid][modes[2]],
        f'fake_{modes[0]}': fake_data['emg_event_mean'][grid][modes[0]],
        f'fake_{modes[1]}': fake_data['emg_event_mean'][grid][modes[1]]}
    plotMultipleModeValues(mean_emg_to_plot, title=title, ylim=ylim)


## plot multiple repetition values of each locomotion mode in subplots for comparison
def plotMultipleRepetitionValues(fake_data, real_data, reference, modes, ylim=(0, 1), num_columns=30, grid='grid_1'):
    plotAverageValue(fake_data['emg_repetition_list'][grid], modes[2], num_columns=num_columns, title=f'fake_repeat_{modes[2]}', ylim=ylim)
    plotAverageValue(real_data['emg_repetition_list'][grid], modes[2], num_columns=num_columns, title=f'real_repeat_{modes[2]}', ylim=ylim)
    plotAverageValue(fake_data['emg_repetition_list'][grid], modes[0], num_columns=num_columns, title=f'real_repeat_{modes[0]}', ylim=ylim)
    # plotAverageValue(fake_data['emg_repetition_list'][grid], modes[1], num_columns=num_columns, title=f'real_repeat_{modes[1]}', ylim=ylim)
    plotAverageValue(reference['emg_repetition_list'][grid], modes[2], num_columns=num_columns, title=f'reference_repeat_{modes[2]}', ylim=ylim)


## plot multiple repetition values of each locomotion mode in subplots for comparison
def plotMultipleChannelValues(fake_data, real_data, reference, modes, ylim=(0, 1), num_columns=30, grid='grid_1'):
    plotAverageValue(fake_data['emg_channel_list'][grid], modes[2], num_columns=num_columns, title=f'fake_channel_{modes[2]}', ylim=ylim)
    plotAverageValue(real_data['emg_channel_list'][grid], modes[2], num_columns=num_columns, title=f'real_channel_{modes[2]}', ylim=ylim)
    plotAverageValue(fake_data['emg_channel_list'][grid], modes[0], num_columns=num_columns, title=f'real_channel_{modes[0]}', ylim=ylim)
    # plotAverageValue(fake_data['emg_channel_list'][grid], modes[1], num_columns=num_columns, title=f'real_channel_{modes[1]}', ylim=ylim)
    plotAverageValue(reference['emg_channel_list'][grid], modes[2], num_columns=num_columns, title=f'reference_{modes[2]}', ylim=ylim)


## plot the average psd of each locomotion mode for comparison
def plotMutipleEventPsdMeanValues(fake_data, real_data, reference, modes, ylim=(0, 1), grid='grid_1'):
    plotPsd(fake_data['emg_event_mean'][grid],  modes[2], title=f'fake_{modes[2]}', ylim=ylim)
    plotPsd(real_data['emg_event_mean'][grid],  modes[2], title=f'real_{modes[2]}', ylim=ylim)
    plotPsd(fake_data['emg_event_mean'][grid],  modes[0], title=f'real_{modes[0]}', ylim=ylim)
    # plotPsd(fake_data['emg_event_mean'][grid],  modes[1], title=f'real_{modes[1]}', ylim=ylim)
    plotPsd(reference['emg_event_mean'][grid],  modes[2], title=f'reference_{modes[2]}', ylim=ylim)
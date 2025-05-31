import numpy as np
import matplotlib.pyplot as plt
import math
import random



def plot_time_series_samples(data, time_start=0, time_end='end', key_label=None, num_samples=None, y_limit=None, random_sampling=False, stata='mean'):
    """
    Plot random samples' average across channels and their overall average.

    Parameters:
        data (list of np.ndarray): List of samples, each of shape (time, channels)
        num_samples (int): Number of random samples to plot
        y_limit (tuple): y-axis limit for all subplots (default: (0, 0.8))
        key_label (str): Optional label for plot titles (e.g., the EMG key)
    """

    total_available = len(data)
    if num_samples > total_available:
        num_samples = total_available
    if time_end == 'end':
        time_end = data[0].shape[0]

    if random_sampling:     # Randomly select samples
        random_indices = random.sample(range(total_available), num_samples)
    else:
        random_indices = list(range(num_samples))

    samples = [data[i][time_start: time_end, :] for i in random_indices]
    avg_curves = [np.mean(sample, axis=1) for sample in samples]

    if stata == 'max':
        overall_avg = np.max(np.stack(avg_curves, axis=0), axis=0)
    elif stata == 'mean':
        overall_avg = np.mean(np.stack(avg_curves, axis=0), axis=0)
    elif stata == 'median':
        overall_avg = np.median(np.stack(avg_curves, axis=0), axis=0)

    # Subplot config
    total_plots = num_samples + 1
    cols = 5
    rows = math.ceil(total_plots / cols)
    fig, axes = plt.subplots(nrows=rows, ncols=cols, figsize=(cols * 4, rows * 2.8))
    axes = axes.flatten()

    # Plot each sample
    for i in range(num_samples):
        ax = axes[i]
        ax.plot(avg_curves[i])
        ax.set_title(f'Sample {random_indices[i]}')
        ax.set_ylim(*y_limit)
        ax.set_xlabel('Time Step')
        ax.set_ylabel('Avg Value')
        ax.grid(True)

    # Plot overall average
    ax = axes[num_samples]
    ax.plot(overall_avg, color='black')
    ax.set_title(f'Average of {num_samples} Samples {key_label}')
    ax.set_ylim(*y_limit)
    ax.set_xlabel('Time Step')
    ax.set_ylabel('Avg Value')
    ax.grid(True)

    # Hide unused axes
    for j in range(total_plots, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    plt.show()


def plot_heatmaps_samples(data, time_start=0, time_end='end', key_label=None, num_samples=None, y_limit=None, random_sampling=False, cmap='viridis', stata='mean'):
    """
    Plot random heatmaps and their overall average from EMG data.

    Parameters:
        data (list of np.ndarray): Each item shape (T, C)
        num_samples (int): Number of random samples to plot
        cmap (str): Matplotlib colormap name
        key_label (str): Optional label for subplot titles
    """

    total_available = len(data)
    if num_samples > total_available:
        num_samples = total_available
    if time_end == 'end':
        time_end = data[0].shape[0]

    if random_sampling:     # Randomly select samples
        random_indices = random.sample(range(total_available), num_samples)
    else:
        random_indices = list(range(num_samples))

    samples = [data[i][time_start: time_end, :] for i in random_indices]
    # Compute average matrix: shape (T, C)
    stacked = np.stack(samples, axis=0)  # (N, T, C)
    if stata == 'mean':
        avg_matrix = np.mean(stacked, axis=0)
    elif stata == 'max':
        avg_matrix = np.max(stacked, axis=0)
    elif stata == 'median':
        avg_matrix = np.median(np.stack(stacked, axis=0), axis=0)

    # Layout for N + 1 subplots
    total_plots = num_samples + 1
    cols = 5
    rows = math.ceil(total_plots / cols)

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 4, rows * 3))
    axes = axes.flatten()

    # Extract vmin/vmax
    vmin, vmax = y_limit if y_limit else (None, None)

    # Plot each sample as a heatmap
    for i in range(num_samples):
        ax = axes[i]
        im = ax.imshow(samples[i].T, aspect='auto', origin='lower', cmap=cmap, vmin=vmin, vmax=vmax)
        ax.set_title(f'Sample {random_indices[i]}')
        ax.set_xlabel('Time')
        ax.set_ylabel('Channel')

    # Plot the average heatmap
    ax = axes[num_samples]
    im = ax.imshow(avg_matrix.T, aspect='auto', origin='lower', cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_title(f'Average of {num_samples} Samples {key_label}')
    ax.set_xlabel('Time')
    ax.set_ylabel('Channel')

    # Optional: add colorbar to the average plot only
    fig.colorbar(im, ax=axes[num_samples], orientation='vertical', fraction=0.046, pad=0.04)

    # Hide any unused axes
    for j in range(total_plots, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    plt.show()


def plot_time_series_and_heatmaps(data, time_start=0, time_end='end', key_label=None, num_samples=None, y_limit=None, random_sampling=False, cmap='viridis'):
    """
    Plots a 3-row x N-column layout of:
    - Row 1: individual time-series (avg across channels)
    - Row 2: individual heatmaps (channels x time)
    - Row 3: left column = avg time-series, middle column = avg heatmap

    Parameters:
        data (dict): Dict with keys like 'emg_LWLW' and list of samples (shape: T x C)
        key_label (str): Key to select from the data dictionary
        num_samples (int): Number of random samples to plot
        y_limit (tuple): (vmin, vmax) for y-axis and heatmap color range
        cmap (str): Colormap for heatmaps
    """
    total_available = len(data)
    if num_samples > total_available:
        num_samples = total_available
    if time_end == 'end':
        time_end = data[0].shape[0]

    if random_sampling:     # Randomly select samples
        random_indices = random.sample(range(total_available), num_samples)
    else:
        random_indices = list(range(num_samples))
    samples = [data[i][time_start: time_end, :] for i in random_indices]

    # Compute average sample
    stacked = np.stack(samples, axis=0)       # (N, T, C)
    avg_curve = np.mean(stacked, axis=(0, 2)) # (T,)
    avg_heatmap = np.mean(stacked, axis=0).T  # (C, T)
    T, C = samples[0].shape  # T = time steps, C = channels

    # Set up 3-row x num_samples-column grid
    fig, axes = plt.subplots(nrows=3, ncols=num_samples, figsize=(num_samples * 4, 9), constrained_layout=True)

    for i in range(num_samples):
        sample = samples[i]
        curve = np.mean(sample, axis=1)
        heatmap_data = sample.T

        # Row 1: Time-series
        ax_ts = axes[0, i]
        ax_ts.plot(curve)
        ax_ts.set_xlim(0, T)
        ax_ts.set_ylim(*y_limit)
        ax_ts.set_title(f'Sample {random_indices[i]}')
        ax_ts.set_xlabel('Time Step')
        ax_ts.set_ylabel('Avg Val')
        ax_ts.ticklabel_format(useOffset=False)
        ax_ts.grid(True)

        # Row 2: Heatmap (aligned)
        ax_hm = axes[1, i]
        im = ax_hm.imshow(heatmap_data, aspect='auto', origin='lower', cmap=cmap,
                          extent=[0, T, 0, C],
                          vmin=y_limit[0], vmax=y_limit[1])
        ax_hm.set_xlim(0, T)
        ax_hm.set_xlabel('Time')
        ax_hm.set_ylabel('Channel')

    # Row 3, Column 0: Average Time-series
    ax_avg_ts = axes[2, 0]
    ax_avg_ts.plot(avg_curve, color='black')
    ax_avg_ts.set_xlim(0, T)
    ax_avg_ts.set_ylim(*y_limit)
    ax_avg_ts.set_title('Average Time-Series')
    ax_avg_ts.set_xlabel('Time Step')
    ax_avg_ts.set_ylabel('Avg Val')
    ax_avg_ts.ticklabel_format(useOffset=False)
    ax_avg_ts.grid(True)

    # Row 3, Column 1: Average Heatmap (aligned)
    ax_avg_hm = axes[2, 1]
    im = ax_avg_hm.imshow(avg_heatmap, aspect='auto', origin='lower', cmap=cmap,
                          extent=[0, T, 0, C],
                          vmin=y_limit[0], vmax=y_limit[1])
    ax_avg_hm.set_xlim(0, T)
    ax_avg_hm.set_title('Average Heatmap')
    ax_avg_hm.set_xlabel('Time')
    ax_avg_hm.set_ylabel('Channel')

    # Optional: hide unused axes in row 3
    for j in range(2, num_samples):
        fig.delaxes(axes[2, j])

    # Shared colorbar
    fig.colorbar(im, ax=axes[2, 1], orientation='vertical', fraction=0.05, pad=0.02).set_label('Channel Value')

    fig.suptitle(f'Random Samples and Average (Bottom Left) {key_label}', fontsize=16)
    plt.show()


def plot_overlap_sample_all_modes(old_emg_central):
    # The 7 EMG data keys
    keys = ['emg_LWLW', 'emg_LWSA', 'emg_SASA', 'emg_SDSD', 'emg_SDLW', 'emg_LWSD', 'emg_SALW']
    # Setup subplots
    fig, axes = plt.subplots(nrows=4, ncols=2, figsize=(18, 14))
    axes = axes.flatten()
    for idx, key in enumerate(keys):
        ax = axes[idx]
        data_list = old_emg_central[key]

        for i in range(min(30, len(data_list))):  # limit to 30 samples
            sample = data_list[i]  # Shape: (T, 65)
            avg_signal = np.mean(sample, axis=1)  # Mean across channels
            ax.plot(avg_signal, alpha=0.6, label=f'Sample {i + 1}')

        ax.set_title(f'{key}')
        ax.set_xlabel('Time Step')
        ax.set_ylabel('Avg Channel Value')
        ax.set_ylim(0, 0.8)  # Set y-axis limit
        ax.grid(True)
    # Hide unused subplot if only using 7
    for j in range(len(keys), len(axes)):
        fig.delaxes(axes[j])
    plt.tight_layout()
    plt.show()

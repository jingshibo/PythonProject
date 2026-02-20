import numpy as np
import matplotlib.pyplot as plt
import math
import random
from scipy.signal import welch
from matplotlib.gridspec import GridSpec

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
    return overall_avg


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
    return avg_matrix.T


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


def plot_overlap_sample_all_modes(old_emg_central, y_limit=(0, 0.4)):
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
        ax.set_ylim(*y_limit)  # Set y-axis limit
        ax.grid(True)
    # Hide unused subplot if only using 7
    for j in range(len(keys), len(axes)):
        fig.delaxes(axes[j])
    plt.tight_layout()
    plt.show()


def plot_sample_fft(data, transition_type, num_samples=5, fs=1000):
    """
    Plot PSDs of averaged time-series data for a given transition type.

    Parameters:
    - data_dict: your full data (e.g., selected_fake_data)
    - transition_type: key like 'emg_LWSA'
    - num_samples: how many samples to plot
    - fs: sampling frequency (Hz) for PSD, adjust based on your data
    """


    # Limit number of samples
    num_samples = min(num_samples, len(data))
    N = data[0].shape[0]  # 1200 time steps

    plt.figure(figsize=(15, 4 * num_samples))
    for i in range(num_samples):
        sample = data[i]  # shape: (1200, 65)
        avg_signal = np.mean(sample, axis=1)  # shape: (1200,)

        # Perform FFT
        fft_result = np.fft.rfft(avg_signal)
        freqs = np.fft.rfftfreq(N, d=1/fs)
        # ✅ Normalize magnitude
        magnitude = np.abs(fft_result) / N
        magnitude[1:-1] *= 2  # double non-DC and non-Nyquist
        magnitude_db = 20 * np.log10(magnitude + 1e-10)  # Avoid log(0)

        # Plot
        plt.subplot(num_samples, 1, i + 1)
        plt.plot(freqs, magnitude_db)
        plt.title(f"Sample {i} - FFT Magnitude Spectrum {transition_type}")
        plt.xlabel("Frequency (Hz)")
        plt.ylabel("Magnitude")

    plt.tight_layout()
    plt.show()


## Plot the old, new, and synthetic HDsEMG values
def plot_summary_2x3(
    old_avg_image, new_avg_image, gen_avg_image,
    old_avg_curve, new_avg_curve, gen_avg_curve,
    column_titles=("Old Data", "New Data", "Synthetic Data"),
    cmap="viridis",
    heatmap_vlim=(0, 0.4),
    curve_ylim=(0, 0.4),
    font_size=18
):

    plt.rcParams.update({
        "font.size": font_size,
        "axes.titlesize": font_size,
        "axes.labelsize": font_size,
        "xtick.labelsize": font_size - 2,
        "ytick.labelsize": font_size - 2
    })

    fig, axes = plt.subplots(2, 3, figsize=(18, 8), constrained_layout=True)

    # -------- Column titles (top row only) --------
    for j, title in enumerate(column_titles):
        axes[0, j].set_title(title, fontsize=font_size + 2, pad=15)

    # -------- Row 1: Heatmaps --------
    heatmaps = [old_avg_image, new_avg_image, gen_avg_image]
    vmin, vmax = heatmap_vlim if heatmap_vlim else (None, None)

    last_im = None
    for j, hm in enumerate(heatmaps):
        ax = axes[0, j]
        last_im = ax.imshow(
            hm, aspect="auto", origin="lower",
            cmap=cmap, vmin=vmin, vmax=vmax
        )

        # Y-label only on left
        if j == 0:
            ax.set_ylabel("Channel")
        else:
            ax.set_ylabel("")
            ax.set_yticklabels([])

        # Remove X labels on heatmaps
        ax.set_xlabel("")
        ax.set_xticklabels([])

    # Shared colorbar
    fig.colorbar(last_im, ax=axes[0, :],
                 orientation="vertical",
                 fraction=0.02, pad=0.02)

    # -------- Row 2: Curves --------
    curves = [old_avg_curve, new_avg_curve, gen_avg_curve]

    for j, cv in enumerate(curves):
        ax = axes[1, j]
        ax.plot(cv, linewidth=2)
        ax.grid(True)

        # Y-label only on left
        if j == 0:
            ax.set_ylabel("Avg Value")
        else:
            ax.set_ylabel("")
            ax.set_yticklabels([])

        # Keep ticks for all bottom plots
        T = len(cv)
        ax.set_xticks(np.arange(0, T+1, 300))

        # X-label only on bottom-left
        if j == 1:
            ax.set_xlabel("Time Step")
        else:
            ax.set_xlabel("")

        if curve_ylim:
            ax.set_ylim(*curve_ylim)

    plt.show()
    return fig, axes


##
def plot_emg_3x2(
    old_avg_image, new_avg_image, gen_avg_image,
    old_avg_curve, new_avg_curve, gen_avg_curve,
    row_labels=("Old Data", "New Data", "Synthetic Data"),
    col_labels=("Heatmap", "Mean Curve"),
    cmap="jet",
    heatmap_vlim=(0, 0.4),
    curve_ylim=(0, 0.4),
    xtick_step=200,
    font_size=12,
    row_label_size=None,
    title_size=None,
    line_width=1.4,
    show_channel_ticks=(0, 20, 40, 60),
    save_path=None,
    dpi=600,
    # --- NEW: control each subplot box aspect (height/width) ---
    heatmap_box_aspect=0.32,   # height/width for each heatmap axis
    curve_box_aspect=0.32,     # height/width for each curve axis
):
    """
    Publication-style 3x2 figure:
      rows: Old/New/Synthetic
      col1: heatmap (C x T)
      col2: mean curve (T,)
    No colorbar.

    `*_box_aspect` controls each subplot's height/width ratio (per-axis),
    independent of figure size.
    """

    heatmaps = [old_avg_image, new_avg_image, gen_avg_image]
    curves   = [old_avg_curve, new_avg_curve, gen_avg_curve]

    # --- shared limits ---
    if heatmap_vlim is None:
        vmin = min(np.nanmin(h) for h in heatmaps)
        vmax = max(np.nanmax(h) for h in heatmaps)
    else:
        vmin, vmax = heatmap_vlim

    if curve_ylim is None:
        ymin = min(np.nanmin(c) for c in curves)
        ymax = max(np.nanmax(c) for c in curves)
        curve_ylim = (ymin, ymax)

    row_label_size = row_label_size or (font_size + 1)
    title_size = title_size or (font_size + 1)

    plt.rcParams.update({
        "font.size": font_size,
        "axes.titlesize": title_size,
        "axes.labelsize": font_size,
        "xtick.labelsize": font_size - 1,
        "ytick.labelsize": font_size - 1,
        "axes.spines.top": False,
        "axes.spines.right": False,
    })

    # --- layout (keep as you like; subplot shape is controlled by set_box_aspect) ---
    fig = plt.figure(figsize=(10.2, 5.6))
    gs = GridSpec(
        3, 2, figure=fig,
        width_ratios=[1.25, 1.0],
        wspace=0.2, hspace=0.18
    )
    ax_h = [fig.add_subplot(gs[i, 0]) for i in range(3)]
    ax_c = [fig.add_subplot(gs[i, 1]) for i in range(3)]

    # Column headers
    ax_h[0].set_title(col_labels[0], pad=6)
    ax_c[0].set_title(col_labels[1], pad=6)

    # ---------------- heatmaps ----------------
    for i in range(3):
        ax = ax_h[i]
        hm = heatmaps[i]
        T = hm.shape[1]  # time length

        ax.imshow(hm, aspect="auto", origin="lower", cmap=cmap, vmin=vmin, vmax=vmax, interpolation="nearest")

        # NEW: force identical x-limits for ALL heatmaps (align right borders)
        ax.set_xlim(0, T + 10)

        # NEW: force per-subplot height/width ratio (if you're using it)
        ax.set_box_aspect(heatmap_box_aspect)

        if show_channel_ticks is not None:
            ax.set_yticks(list(show_channel_ticks))

        # y-label only on middle heatmap
        if i == 1:
            ax.set_ylabel("Channel")
        else:
            ax.set_ylabel("")

        # x-ticks/label only on bottom heatmap (but xlim already aligned for all)
        if i < 2:
            ax.tick_params(axis="x", labelbottom=False)
        else:
            ax.set_xlabel("Time Step")
            ticks = list(np.arange(0, T + 1, xtick_step))
            if ticks[-1] != T:
                ticks.append(T)
            ax.set_xticks(ticks)

        # row label (left margin) - NOT bold
        ax.text(-0.16, 0.5, row_labels[i], transform=ax.transAxes, rotation=90, va="center", ha="right", fontsize=row_label_size)

    # ---------------- curves ----------------
    for i in range(3):
        ax = ax_c[i]
        cv = curves[i]
        T = len(cv)

        ax.plot(cv, linewidth=line_width)
        ax.set_ylim(*curve_ylim)

        # NEW: force per-subplot height/width ratio
        ax.set_box_aspect(curve_box_aspect)

        # show the endpoint tick (e.g., 1200) and avoid overlap with right spine
        ax.set_xlim(0, T + 10)

        ticks = list(np.arange(0, T + 1, xtick_step))
        if ticks[-1] != T:
            ticks.append(T)
        ax.set_xticks(ticks)

        # subtle y-grid only
        ax.grid(True, axis="y", alpha=0.18, linewidth=0.8)
        ax.grid(False, axis="x")

        # restore full rectangular border for curves
        for spine in ax.spines.values():
            spine.set_visible(True)

        # y-label only middle (tick labels still visible for all)
        if i == 1:
            ax.set_ylabel("Amplitude")
        else:
            ax.set_ylabel("")

        # only bottom curve shows x tick labels + xlabel
        if i < 2:
            ax.set_xlabel("")
            ax.tick_params(axis="x", labelbottom=False)
        else:
            ax.set_xlabel("Time step")
            ax.tick_params(axis="x", labelbottom=True)

    # margins
    fig.subplots_adjust(left=0.15, right=0.80, top=0.92, bottom=0.12)

    if save_path is not None:
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight")

    plt.show()
    return fig, (ax_h, ax_c)
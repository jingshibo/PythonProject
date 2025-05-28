

##
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import OneHotEncoder
import numpy as np
from scipy.signal import correlate, correlation_lags, butter, filtfilt
from scipy.ndimage import uniform_filter1d # Kept for alternative smoothing



## build classification datasets and extract the relevent modes for data generation
def extractGanTrainingData(modes_generation, old_emg_normalized, new_emg_normalized, time_range):
    # select only the central part data for training
    def slice_center_time(emg_dict, selected_range):
        new_emg_dict = {}
        for label, samples in emg_dict.items():
            new_emg_dict[label] = []
            for sample in samples:
                # total_length = sample.shape[0]
                # start = total_length // 2 + selected_range[0]
                # end = total_length // 2 + selected_range[1]
                start = selected_range[0]
                end = selected_range[1]
                new_emg_dict[label].append(sample[start:end, :])  # Slice time axis
        return new_emg_dict
    old_emg_central = slice_center_time(old_emg_normalized, time_range)
    new_emg_central = slice_center_time(new_emg_normalized, time_range)

    # build gan generation dataset
    train_gan_data = {}
    data_keys = ['gen_data_1', 'gen_data_2', 'disc_data']  # The order in the list is critical, corresponding to the locomotion modes

    for transition_type, modes in modes_generation.items():
        # Initialize transition_type key in real_emg and train_gan_data dictionaries
        train_gan_data[transition_type] = {'gen_data_1': None, 'gen_data_2': None, 'disc_data': None}
        for idx, mode in enumerate(modes):
            # Assign values using the new structure
            train_gan_data[transition_type][data_keys[idx]] = old_emg_central[mode]

    return old_emg_central, new_emg_central, train_gan_data


# build the paired x and y dataset
def buildClassifyDataset(emg_central):
    # build classification dataset
    all_labels = sorted(set(emg_central.keys()))
    label_map = {label: idx for idx, label in enumerate(all_labels)}

    def classifier_dataset(emg_dict, label_map, all_labels):
        data = []
        labels_categorical = []

        for label in all_labels:
            if label in emg_dict:
                for sample in emg_dict[label]:
                    data.append(sample[np.newaxis, :, :])  # (1, 2000, 65)
                    labels_categorical.append(label_map[label])

        data = np.array(data)
        labels_categorical = np.array(labels_categorical).reshape(-1, 1)

        encoder = OneHotEncoder(sparse_output=False)
        labels_onehot = encoder.fit_transform(labels_categorical)

        return {'data_x': data, 'int_y': labels_categorical.flatten(), 'onehot_y': labels_onehot, 'label_map': label_map}

    classify_emg = classifier_dataset(emg_central, label_map, all_labels)

    return classify_emg


## build cross validation datset for classification
def crossValidationSet(fold_number, classify_emg_data):
    # StratifiedKFold ensures that each fold has the same class distribution as the full dataset.
    skf = StratifiedKFold(n_splits=fold_number, shuffle=True, random_state=42)
    X = classify_emg_data['data_x']
    y = classify_emg_data['int_y']

    # only index
    cross_validation_indices = []
    for train_idx, val_idx in skf.split(X, y):
        cross_validation_indices.append({'train': train_idx, 'val': val_idx})

    # fold data
    cross_validation_dataset = []
    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        fold_data = {'train_x': X[train_idx], 'train_y': y[train_idx], 'val_x': X[val_idx], 'val_y': y[val_idx]}
        cross_validation_dataset.append(fold_data)

        print(f"Fold {fold_idx + 1}:")
        unique_train, counts_train = np.unique(fold_data['train_y'], return_counts=True)
        unique_val, counts_val = np.unique(fold_data['val_y'], return_counts=True)
        print(f"  Train class counts: {dict(zip(unique_train, counts_train))}")
        print(f"  Val class counts:   {dict(zip(unique_val, counts_val))}")

    return cross_validation_indices, cross_validation_dataset


def align_by_cross_correlation(emg_data, max_lag=100, channel_weights=None, num_iterations=2, initial_reference_method='first',
        verbose=False, smoothing_method='butterworth', butter_cutoff_freq=None, butter_filter_order=4, sampling_rate=1000,
        ma_smoothing_window_size=0):
    """
    Align time-series by maximizing cross-correlation, with iterative reference refinement
    and optional Butterworth low-pass filtering or moving average for smoothing.

    Args:
        data_list: List of NumPy arrays, each of shape (Time, Channel).
                   All arrays must have the same number of time steps and channels.
        max_lag: Maximum shift allowed in either direction (in time steps).
        channel_weights: Optional. 1D array of weights for each channel when creating
                         the 1D signal for cross-correlation. If None, channels are averaged.
        num_iterations: Optional. Number of iterations to refine the alignment and reference.
        initial_reference_method: How to choose the initial reference.
            'first': Use the first signal in data_list.
            'average': Use the average of all initial signals.
            'highest_energy': Use the signal with the highest energy.
        verbose: If True, print iteration and alignment info.
        smoothing_method: 'butterworth', 'moving_average', or 'none'.
        butter_cutoff_freq: Cutoff frequency in Hz for Butterworth filter.
                            Required if smoothing_method='butterworth'.
        butter_filter_order: Order for Butterworth filter.
        sampling_rate: Sampling rate of the signals in Hz.
                       Required if smoothing_method='butterworth'.
        ma_smoothing_window_size: Size of the moving average window.
                                  Used if smoothing_method='moving_average'.

    Returns:
        aligned_list: List of aligned time-series arrays.
        final_lags: List of the final applied lags for each time-series.
    """
    emg_list_aligned = {}
    emg_cumulative_lags = {}
    for data_mode, data_list in emg_data.items():
        if not data_list:
            return [], []

        # --- Input Validation ---
        num_samples = len(data_list)
        time_steps, num_channels = data_list[0].shape
        # ... (rest of the input validation from previous version for shapes, channel_weights) ...
        for i, ts in enumerate(data_list): # Basic shape check
            if ts.shape[0] != time_steps or ts.shape[1] != num_channels:
                raise ValueError("All time-series must have the same dimensions.")

        if smoothing_method == 'butterworth':
            if sampling_rate is None:
                raise ValueError("sampling_rate must be provided when using Butterworth filter.")
            if butter_cutoff_freq is None:
                raise ValueError("butter_cutoff_freq must be provided when using Butterworth filter.")
            if butter_cutoff_freq <= 0:
                raise ValueError("butter_cutoff_freq must be positive.")
            nyquist_freq = 0.5 * sampling_rate
            if butter_cutoff_freq >= nyquist_freq:
                raise ValueError(f"Butterworth cutoff frequency ({butter_cutoff_freq} Hz) "
                                 f"must be less than the Nyquist frequency ({nyquist_freq} Hz).")


        # --- Helper function to get 1D signal for correlation ---
        def get_1d_signal(ts_2d):
            if channel_weights is not None:
                s = np.sum(ts_2d * np.array(channel_weights).reshape(1, -1), axis=1)
            else:
                s = np.mean(ts_2d, axis=1)

            if smoothing_method == 'butterworth':
                # Design the Butterworth filter
                nyquist = 0.5 * sampling_rate
                normal_cutoff = butter_cutoff_freq / nyquist
                b, a = butter(butter_filter_order, normal_cutoff, btype='low', analog=False)
                # Apply the filter (zero-phase)
                s_filtered = filtfilt(b, a, s)
                return s_filtered
            elif smoothing_method == 'moving_average' and ma_smoothing_window_size > 1:
                s_padded = np.pad(s, (ma_smoothing_window_size//2, ma_smoothing_window_size//2), mode='reflect')
                s_smoothed = uniform_filter1d(s_padded, size=ma_smoothing_window_size)
                return s_smoothed[ma_smoothing_window_size//2 : - (ma_smoothing_window_size//2)]
            else: # 'none' or invalid MA window
                return s

        # --- Iterative Alignment (largely unchanged from previous version) ---
        current_data = [ts.copy() for ts in data_list]
        cumulative_lags = np.zeros(num_samples, dtype=int)

        for iteration in range(num_iterations):
            if verbose:
                print(f"--- Iteration {iteration + 1}/{num_iterations} ---")

            # 1. Determine/Update Reference Signal
            if iteration == 0:
                if initial_reference_method == 'first':
                    reference_1d = get_1d_signal(current_data[0])
                elif initial_reference_method == 'highest_energy':
                    energies = [np.sum(get_1d_signal(ts)**2) for ts in current_data]
                    ref_idx = np.argmax(energies)
                    reference_1d = get_1d_signal(current_data[ref_idx])
                    if verbose: print(f"Initial reference: Sample {ref_idx} (highest energy)")
                else: # 'average' or default
                    reference_1d = np.mean([get_1d_signal(ts) for ts in current_data], axis=0)
            else:
                reference_1d = np.mean([get_1d_signal(ts) for ts in current_data], axis=0)

            if verbose and iteration > 0 : print("Updated reference with current average.")

            # 2. Align each signal to the current reference
            newly_aligned_data = []
            current_iter_lags = np.zeros(num_samples, dtype=int)

            for i, ts_2d in enumerate(current_data):
                signal_1d = get_1d_signal(ts_2d)
                xcorr = correlate(signal_1d, reference_1d, mode='full', method='auto')
                lags = correlation_lags(len(signal_1d), len(reference_1d), mode='full')

                valid_indices = np.where((lags >= -max_lag) & (lags <= max_lag))[0]
                if not valid_indices.size: # No valid lags found, shouldn't happen with reasonable max_lag
                    best_lag_for_xcorr = 0
                else:
                    best_lag_for_xcorr_idx = valid_indices[np.argmax(xcorr[valid_indices])]
                    best_lag_for_xcorr = lags[best_lag_for_xcorr_idx]

                shift_to_apply = -best_lag_for_xcorr
                current_iter_lags[i] = shift_to_apply

                shifted_ts_2d = np.roll(ts_2d, shift=shift_to_apply, axis=0)
                if shift_to_apply > 0:
                    shifted_ts_2d[:shift_to_apply, :] = 0
                elif shift_to_apply < 0:
                    shifted_ts_2d[shift_to_apply:, :] = 0
                newly_aligned_data.append(shifted_ts_2d)

            current_data = newly_aligned_data
            cumulative_lags += current_iter_lags

            if verbose:
                print(f"Shifts applied in this iteration: {current_iter_lags}")
                print(f"Cumulative shifts: {cumulative_lags}")
            if np.all(current_iter_lags == 0) and iteration > 0:
                if verbose: print("Converged: No shifts applied in the last iteration.")
                break

        # Final alignment based on cumulative lags applied to original data
        aligned_list_final = []
        for i, original_ts in enumerate(data_list):
            final_shift = cumulative_lags[i]
            shifted_ts_2d = np.roll(original_ts, shift=final_shift, axis=0)
            if final_shift > 0:
                shifted_ts_2d[:final_shift, :] = 0
            elif final_shift < 0:
                shifted_ts_2d[final_shift:, :] = 0
            aligned_list_final.append(shifted_ts_2d)

        emg_list_aligned[data_mode] = aligned_list_final
        emg_cumulative_lags[data_mode] = cumulative_lags.tolist()
    return emg_list_aligned, emg_cumulative_lags

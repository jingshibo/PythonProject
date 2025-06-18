

##
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import OneHotEncoder
import numpy as np
from scipy.signal import correlate, correlation_lags, butter, filtfilt
from scipy.ndimage import uniform_filter1d  # Kept for alternative smoothing
import random
from cGAN_Transformer.Models import Transformer_GAN_Testing
from Conditional_GAN.Data_Procesing import Dtw_Similarity


## build classification datasets and extract the relevent modes for data generation
def extractGanTrainingData(modes_generation, old_emg_normalized, new_emg_normalized, time_range):
    # select only the central part data for training
    def slice_center_time(emg_dict, selected_range):
        new_emg_dict = {}
        for label, samples in emg_dict.items():
            new_emg_dict[label] = []
            for sample in samples:
                total_length = sample.shape[0]
                start = total_length // 2 + selected_range[0]
                end = total_length // 2 + selected_range[1]
                new_emg_dict[label].append(sample[start:end, :])  # Slice time axis
        return new_emg_dict
    old_emg_central = slice_center_time(old_emg_normalized, time_range)
    new_emg_central = slice_center_time(new_emg_normalized, time_range)

    # build gan generation dataset
    data_keys = ['gen_data_1', 'gen_data_2', 'disc_data']  # The order in the list is critical, corresponding to the locomotion modes

    old_gan_data = {}
    new_gan_data = {}
    for transition_type, modes in modes_generation.items():
        # Initialize transition_type key in real_emg and train_gan_data dictionaries
        old_gan_data[transition_type] = {'gen_data_1': None, 'gen_data_2': None, 'disc_data': None}
        new_gan_data[transition_type] = {'gen_data_1': None, 'gen_data_2': None, 'disc_data': None}
        for idx, mode in enumerate(modes):
            # Assign values using the new structure
            old_gan_data[transition_type][data_keys[idx]] = old_emg_central[mode]
            new_gan_data[transition_type][data_keys[idx]] = new_emg_central[mode]

    return old_emg_central, new_emg_central, old_gan_data, new_gan_data


## build the paired x and y dataset
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


## build a cross validation dataset with generated data incorporated into the training set
def build_cv_dataset_with_augmented_data(original_emg_data, augmented_emg_data, modes_generation, n_splits=5, n_real_steady_state=50,
        n_synthetic_transition=50, n_real_transition=5, random_sampling=True):
    # Step 1: Set seed for reproducibility
    if not random_sampling:
        random.seed(5)
        np.random.seed(5)

    # Step 2: Flatten original data
    all_keys = sorted(original_emg_data.keys())
    label_map = {key: i for i, key in enumerate(all_keys)}
    X_all = [sample[np.newaxis, :, :] for key in all_keys for sample in original_emg_data[key]]
    y_all = [label_map[key] for key in all_keys for _ in original_emg_data[key]]
    X_all = np.array(X_all).astype(np.float32)
    y_all = np.array(y_all).astype(np.int64)
    augmented_emg_dict = {key: [arr[np.newaxis, :, :] if arr.ndim == 2 else arr for arr in list_of_arrays] for key, list_of_arrays in
        augmented_emg_data.items()}

    # Step 3: Cross Validation data
    kf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    encoder = OneHotEncoder(sparse_output=False)
    encoder.fit(y_all.reshape(-1, 1))  # fit once
    original_folds = []
    replaced_folds = []

    for train_idx, test_idx in kf.split(X_all, y_all):
        # Split original data
        X_train_orig, y_train_orig = X_all[train_idx], y_all[train_idx]
        X_test, y_test = X_all[test_idx], y_all[test_idx]

        y_train_onehot_orig = encoder.transform(y_train_orig.reshape(-1, 1))
        y_test_onehot = encoder.transform(y_test.reshape(-1, 1))

        # 🔹 ORIGINAL FOLDS: keep all real samples per class
        original_folds.append({'X_train': X_train_orig, 'y_train_int': y_train_orig, 'y_train_onehot': y_train_onehot_orig, 'X_test': X_test,
                'y_test_int': y_test, 'y_test_onehot': y_test_onehot, 'label_map': label_map})

        # 🔹 REPLACED FOLDS: N real + N generated if available; else just N real
        replaced_X_train, replaced_y_train = [], []

        train_data_by_label = {}
        for i, x in enumerate(X_train_orig):
            label = y_train_orig[i]
            train_data_by_label.setdefault(label, []).append(x)

        for label, samples in train_data_by_label.items():
            key = all_keys[label]
            samples = np.array(samples)
            if key in list(modes_generation.keys()):
                # N real transition samples
                real_indices = np.random.choice(len(samples), size=min(n_real_transition, len(samples)), replace=False)
                replaced_X_train.extend(samples[real_indices])
                replaced_y_train.extend([label] * len(real_indices))

                # N generated samples
                gen_samples = augmented_emg_dict[key]
                gen_indices = np.random.choice(len(gen_samples), size=min(n_synthetic_transition, len(gen_samples)), replace=False)
                replaced_X_train.extend([gen_samples[i] for i in gen_indices])
                replaced_y_train.extend([label] * len(gen_indices))
            else:
                # Only N real samples (no generated data available)
                real_indices = np.random.choice(len(samples), size=min(n_real_steady_state, len(samples)), replace=False)
                replaced_X_train.extend(samples[real_indices])
                replaced_y_train.extend([label] * len(real_indices))

        y_train_onehot_replaced = encoder.transform(np.array(replaced_y_train).reshape(-1, 1))

        replaced_folds.append(
            {'X_train': np.array(replaced_X_train), 'y_train_int': np.array(replaced_y_train), 'y_train_onehot': y_train_onehot_replaced,
                'X_test': np.array(X_test), 'y_test_int': y_test, 'y_test_onehot': y_test_onehot, 'label_map': label_map})

    return replaced_folds, original_folds


## build a cross validation dataset with data generation based on available new real data
def build_cv_dataset_with_noisy_data(original_emg_data, modes_generation, snr=25, n_splits=5, n_real_steady_state=50, n_synthetic_transition=50,
        n_real_transition=5, random_sampling=True):
    # Step 1: Set seed for reproducibility
    if not random_sampling:
        random.seed(5)
        np.random.seed(5)

    # Step 2: Flatten original data
    all_keys = sorted(original_emg_data.keys())
    label_map = {key: i for i, key in enumerate(all_keys)}
    X_all = [sample[np.newaxis, :, :] for key in all_keys for sample in original_emg_data[key]]
    y_all = [label_map[key] for key in all_keys for _ in original_emg_data[key]]
    X_all = np.array(X_all).astype(np.float32)
    y_all = np.array(y_all).astype(np.int64)

    # Step 3: Cross Validation data
    kf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    encoder = OneHotEncoder(sparse_output=False)
    encoder.fit(y_all.reshape(-1, 1))  # fit once
    original_folds = []
    replaced_folds = []

    for train_idx, test_idx in kf.split(X_all, y_all):
        # Split original data
        X_train_orig, y_train_orig = X_all[train_idx], y_all[train_idx]
        X_test, y_test = X_all[test_idx], y_all[test_idx]
        y_train_onehot_orig = encoder.transform(y_train_orig.reshape(-1, 1))
        y_test_onehot = encoder.transform(y_test.reshape(-1, 1))

        # 🔹 ORIGINAL FOLDS: keep all real samples per class
        original_folds.append({'X_train': X_train_orig, 'y_train_int': y_train_orig, 'y_train_onehot': y_train_onehot_orig, 'X_test': X_test,
                'y_test_int': y_test, 'y_test_onehot': y_test_onehot, 'label_map': label_map})

        # 🔹 REPLACED FOLDS: N real + N generated if available; else just N real
        replaced_train_dict = {}

        train_data_by_label = {}
        for i, x in enumerate(X_train_orig):
            label = y_train_orig[i]
            train_data_by_label.setdefault(label, []).append(x)

        for label, samples in train_data_by_label.items():
            key = all_keys[label]
            samples = np.array(samples)

            # Initialize an empty list for this class in our new dictionary
            replaced_train_dict[key] = []
            if key in list(modes_generation.keys()):
                # N real transition samples
                real_indices = np.random.choice(len(samples), size=min(n_real_transition, len(samples)), replace=False)
                # Add the selected real samples to the list for this key
                replaced_train_dict[key].extend(list(samples[real_indices]))
                # generate noisy data for augmentation
                gen_samples = generateNoiseData(list(samples[real_indices]), n_synthetic_transition, snr)
                # Add the selected generated samples to the same list
                replaced_train_dict[key].extend(gen_samples)
            else:
                # Only N real samples (no generated data available)
                real_indices = np.random.choice(len(samples), size=min(n_real_steady_state, len(samples)), replace=False)
                # Add the selected real samples to the list for this key
                replaced_train_dict[key].extend(list(samples[real_indices]))

        # To maintain compatibility with the rest of the function (like one-hot encoding),
        # we can now flatten this dictionary back into X and y arrays.
        replaced_X_train = [sample for key in sorted(replaced_train_dict.keys()) for sample in replaced_train_dict[key]]
        replaced_y_train = [label_map[key] for key in sorted(replaced_train_dict.keys()) for _ in replaced_train_dict[key]]

        y_train_onehot_replaced = encoder.transform(np.array(replaced_y_train).reshape(-1, 1))

        replaced_folds.append({# We now add our new dictionary to the output for easy access
            'train_data_dict': replaced_train_dict, 'X_train': np.array(replaced_X_train), 'y_train_int': np.array(replaced_y_train),
            'y_train_onehot': y_train_onehot_replaced, 'X_test': np.array(X_test), 'y_test_int': y_test, 'y_test_onehot': y_test_onehot,
            'label_map': label_map})

    return replaced_folds, original_folds


# generate fake data by adding noise to real emg transition data
def generateNoiseData(real_transition_data, num_sample_to_generate, snr=25):
    '''
    :param real_transition_data: selected real data used for noisy sample generation
    :param num_sample_to_generate:  the number of noisy samples to generate for each locomotion mode
    :param snr:  the amplitude of noise to add based on signal-to-noise ratio. None means simply copying without adding noise.
    :return: real emg data with certain modes replaced by generated noisy data
    '''

    if not real_transition_data:  # if no new data are available for the transition modes
        raise Exception('No real transition data available!')

    # add noise to reference data
    def generate_noisy_sample(real_data, snr_db):  # add Gaussian Noise with signal-to-noise ratio (SNR) of 25
        # If snr is None, simply return a copy of the original data.
        if snr_db is None:
            return real_data.copy()
        else:
            # Calculate the variance of the signal
            signal_var = np.var(real_data)
            # Convert SNR from dB scale to linear scale
            snr_linear = 10 ** (snr_db / 10)
            # Calculate the required noise variance
            noise_variance = signal_var / snr_linear
            # Generate the noise with the calculated standard deviation
            noise = np.random.normal(0, np.sqrt(noise_variance), real_data.shape)
            return real_data + noise.astype(np.float32)

    # Generate noisy samples for each numpy array in reference_data
    noisy_data = [generate_noisy_sample(array, snr) for array in real_transition_data for _ in
        range(int(num_sample_to_generate / len(real_transition_data)))]

    return noisy_data


## build a cross validation dataset with data generation based on available new real data
def build_cv_dataset_for_model_updating(original_emg_data, gan_model, modes_generation, training_parameters, n_splits=5,
        n_real_steady_state=50, n_synthetic_transition=50, n_real_transition=5, random_sampling=True):
    # Step 1: Set seed for reproducibility
    if not random_sampling:
        random.seed(5)
        np.random.seed(5)

    # Step 2: Flatten original data
    all_keys = sorted(original_emg_data.keys())
    label_map = {key: i for i, key in enumerate(all_keys)}
    X_all = [sample[np.newaxis, :, :] for key in all_keys for sample in original_emg_data[key]]
    y_all = [label_map[key] for key in all_keys for _ in original_emg_data[key]]
    X_all = np.array(X_all).astype(np.float32)
    y_all = np.array(y_all).astype(np.int64)

    # Step 3: Cross Validation data
    kf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    encoder = OneHotEncoder(sparse_output=False)
    encoder.fit(y_all.reshape(-1, 1))  # fit once
    original_folds = []
    replaced_folds = []

    for train_idx, test_idx in kf.split(X_all, y_all):
        # Split original data
        X_train_orig, y_train_orig = X_all[train_idx], y_all[train_idx]
        X_test, y_test = X_all[test_idx], y_all[test_idx]
        y_train_onehot_orig = encoder.transform(y_train_orig.reshape(-1, 1))
        y_test_onehot = encoder.transform(y_test.reshape(-1, 1))

        # 🔹 ORIGINAL FOLDS: keep all real samples per class
        original_folds.append({'X_train': X_train_orig, 'y_train_int': y_train_orig, 'y_train_onehot': y_train_onehot_orig, 'X_test': X_test,
                'y_test_int': y_test, 'y_test_onehot': y_test_onehot, 'label_map': label_map})

        # 🔹 REPLACED FOLDS: N real + N generated if available; else just N real
        replaced_train_dict = {}

        train_data_by_label = {}
        for i, x in enumerate(X_train_orig):
            label = y_train_orig[i]
            train_data_by_label.setdefault(label, []).append(x)

        for label, samples in train_data_by_label.items():
            key = all_keys[label]
            samples = np.array(samples)

            # Initialize an empty list for this class in our new dictionary
            replaced_train_dict[key] = []

            if key in list(modes_generation.keys()):
                # N real transition samples
                real_indices = np.random.choice(len(samples), size=min(n_real_transition, len(samples)), replace=False)
                # Add the selected real samples to the list for this key
                replaced_train_dict[key].extend(list(samples[real_indices]))
            else:
                # Only N real samples (no generated data available)
                real_indices = np.random.choice(len(samples), size=min(n_real_steady_state, len(samples)), replace=False)
                # Add the selected real samples to the list for this key
                replaced_train_dict[key].extend(list(samples[real_indices]))

        # generate transition data
        gan_test_data = buildGanTestData(modes_generation, replaced_train_dict)
        generated_transition_data = Transformer_GAN_Testing.generateTransitionData(gan_model['gen'], gan_test_data,
            training_parameters['transition_encoding'], training_parameters['num_window_per_transition'],
            training_parameters['window_length'], training_parameters['window_increment'], training_parameters['window_shift'],
            number_to_generate=3600, batch_size=30)
        time_0_fake_data, _ = Transformer_GAN_Testing.returnDataForPlotting(generated_transition_data, ordered_sample_number=50)
        generated_data = {key: [arr[np.newaxis, :, :] for arr in list_of_arrays] for key, list_of_arrays in time_0_fake_data.items()}

        # augment dataset using generated transition data
        for label, samples in train_data_by_label.items():
            key = all_keys[label]
            if key in list(modes_generation.keys()):
                # N generated samples
                gen_samples = generated_data[key]
                gen_indices = np.random.choice(len(gen_samples), size=min(n_synthetic_transition, len(gen_samples)), replace=False)
                # Add the selected generated samples to the same list
                replaced_train_dict[key].extend([gen_samples[i] for i in gen_indices])

        # To maintain compatibility with the rest of the function (like one-hot encoding),
        # we can now flatten this dictionary back into X and y arrays.
        replaced_X_train = [sample for key in sorted(replaced_train_dict.keys()) for sample in replaced_train_dict[key]]
        replaced_y_train = [label_map[key] for key in sorted(replaced_train_dict.keys()) for _ in replaced_train_dict[key]]

        y_train_onehot_replaced = encoder.transform(np.array(replaced_y_train).reshape(-1, 1))

        replaced_folds.append({# We now add our new dictionary to the output for easy access
            'train_data_dict': replaced_train_dict, 'X_train': np.array(replaced_X_train), 'y_train_int': np.array(replaced_y_train),
            'y_train_onehot': y_train_onehot_replaced, 'X_test': np.array(X_test), 'y_test_int': y_test, 'y_test_onehot': y_test_onehot,
            'label_map': label_map})

    return replaced_folds, original_folds


## generate fake data using only selected real data
def buildGanTestData(modes_generation, emg_data_dict):
    # build gan generation dataset
    data_keys = ['gen_data_1', 'gen_data_2', 'disc_data']  # The order in the list is critical, corresponding to the locomotion modes

    # add an axis to the front if only two dim
    new_emg_dict = {key: [arr.squeeze(0) if arr.ndim == 3 else arr for arr in list_of_arrays] for key, list_of_arrays in emg_data_dict.items()}

    gan_test_data = {}
    for transition_type, modes in modes_generation.items():
        # Initialize transition_type key in real_emg and train_gan_data dictionaries
        gan_test_data[transition_type] = {'gen_data_1': None, 'gen_data_2': None, 'disc_data': None}
        for idx, mode in enumerate(modes):
            # Assign values using the new structure
            gan_test_data[transition_type][data_keys[idx]] = new_emg_dict[mode]

    return gan_test_data


##
def align_by_cross_correlation(emg_data, max_lag=100, channel_weights=None, num_iterations=2, initial_reference_method='average',
        verbose=False, smoothing_method='butterworth', butter_cutoff_freq=None, butter_filter_order=4, sampling_rate=1000,
        ma_smoothing_window_size=0):
    """
    Align time-series EMG data by maximizing cross-correlation, with iterative reference refinement
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



##
def align_new_emg_to_old_emg(new_emg_aligned, old_emg_aligned, fs=1000, lp_cutoff=20, max_lag=50):
    """
    Aligns EMG samples by cross-correlating with a reference mean waveform.

    This function takes new EMG trials, finds the optimal time-shift to align them
    with a template created from old trials, and applies that shift to the
    original, unfiltered data.

    Parameters:
    - new_emg_aligned (dict): A dictionary where keys are trial identifiers and
      values are lists of new EMG samples. Each sample is a (time, channels)
      NumPy array, e.g., (1300, 65).
    - old_emg_aligned (dict): A dictionary with the same structure containing
      reference EMG samples used to build the alignment template.
    - fs (int): Sampling frequency in Hz.
    - lp_cutoff (int): Low-pass filter cutoff frequency in Hz for robust alignment.
    - max_lag (int): Maximum allowable time shift (in samples) for alignment.

    Returns:
    - dict: A dictionary with the same keys, but with the new EMG samples
      time-aligned to the reference.
    """

    def butter_lowpass_filter(data, cutoff, fs, order=4):
        """
        Apply a zero-phase low-pass Butterworth filter to a 1D signal.
        """
        nyquist = 0.5 * fs
        normal_cutoff = cutoff / nyquist
        b, a = butter(order, normal_cutoff, btype='low', analog=False)
        # Use filtfilt for zero-phase filtering
        return filtfilt(b, a, data)

    def shift_array(arr, lag, fill_value=0.0):
        """
        Shifts a 2D array along its first axis (time) by a given lag.

        This function correctly interprets the lag from scipy's cross-correlation:
        - A positive lag indicates the signal is delayed, so we shift it LEFT.
        - A negative lag indicates the signal is advanced, so we shift it RIGHT.

        Parameters:
        - arr (np.ndarray): The 2D array to shift, with shape (time, channels).
        - lag (int): The integer lag to apply.
        - fill_value (float): The value to use for padding.

        Returns:
        - np.ndarray: The shifted array with the same shape as the input.
        """
        shifted_arr = np.full_like(arr, fill_value)

        if lag > 0:  # Positive lag -> Shift LEFT (advance in time)
            shifted_arr[:-lag, :] = arr[lag:, :]
        elif lag < 0:  # Negative lag -> Shift RIGHT (delay in time)
            shift = abs(lag)
            shifted_arr[shift:, :] = arr[:-shift, :]
        else:  # No lag
            shifted_arr = arr.copy()

        return shifted_arr

    aligned_emg = {}
    for key in new_emg_aligned.keys():
        # --- Improvement 1: Robustness ---
        # Skip if there are no corresponding reference samples to create a template
        if key not in old_emg_aligned or not old_emg_aligned[key]:
            print(f"Warning: No reference samples found for key '{key}'. Skipping alignment.")
            continue

        # --- Improvement 2: Efficiency ---
        # Step 1: Compute reference mean efficiently using NumPy Stacks all (1300, 65) arrays into (N, 1300, 65), then averages over
        # trials (axis 0) and channels (axis 2) in one step.
        reference_samples = np.stack(old_emg_aligned[key])
        reference_mean = np.mean(reference_samples, axis=(0, 2))  # Shape: (1300,)

        # Step 2: Low-pass filter the reference for robust correlation
        ref_filtered = butter_lowpass_filter(reference_mean, cutoff=lp_cutoff, fs=fs)

        aligned_samples = []
        for sample in new_emg_aligned[key]:
            # Step 3: Compute and filter the mean of the current sample
            sample_mean = np.mean(sample, axis=1)
            sample_filtered = butter_lowpass_filter(sample_mean, cutoff=lp_cutoff, fs=fs)

            # --- Improvement 3: Clarity ---
            # Step 4: Compute constrained cross-correlation more cleanly
            correlation = correlate(sample_filtered, ref_filtered, mode='full', method='auto')
            lags = correlation_lags(len(sample_filtered), len(ref_filtered), mode="full")

            # Constrain to the max_lag window
            lag_mask = (lags >= -max_lag) & (lags <= max_lag)
            constrained_corr = correlation[lag_mask]
            constrained_lags = lags[lag_mask]

            # Find the lag that maximizes the correlation
            if len(constrained_corr) == 0:
                # This can happen if the signal length is smaller than max_lag
                best_lag = 0
            else:
                best_lag = constrained_lags[np.argmax(constrained_corr)]

            # --- Improvement 4: Correctness and Modularity ---
            # Step 5: Apply the lag to the original, unfiltered signal using a robust helper function
            shifted_sample = shift_array(sample, best_lag)
            aligned_samples.append(shifted_sample)

        aligned_emg[key] = aligned_samples

    return aligned_emg

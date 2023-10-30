'''
    load original data for cGAN model training
'''

##
from Transition_Prediction.Pre_Processing import Preprocessing
from Transition_Prediction.Models.Utility_Functions import Data_Preparation
from Cycle_GAN.Functions import Data_Processing
from Conditional_GAN.Data_Procesing import Process_Fake_Data
from scipy.ndimage import gaussian_filter
import numpy as np
import copy
import cv2


## load raw data and filter them
def readFilterEmgData(data_source, window_parameters, lower_limit=20, higher_limit=400, envelope_cutoff=10, envelope=False,
        project='Insole_Emg', selected_grid='all', include_standing=True):
    split_parameters = Preprocessing.readSplitParameters(data_source['subject'], data_source['version'], project=project)
    emg_filtered_data = Preprocessing.labelFilteredData(data_source['subject'], data_source['modes'],
        data_source['sessions'], data_source['version'], split_parameters, project=project,
        start_position=-int(window_parameters['start_before_toeoff_ms'] * (2 / window_parameters['sample_rate'])),
        end_position=int(window_parameters['endtime_after_toeoff_ms'] * (2 / window_parameters['sample_rate'])), lower_limit=lower_limit,
        higher_limit=higher_limit, envelope_cutoff=envelope_cutoff, envelope=envelope, notchEMG=False, median_filtering=True, reordering=True)
    emg_preprocessed = Data_Preparation.removeSomeSamples(emg_filtered_data, is_down_sampling=window_parameters['down_sampling'],
        standing=include_standing)
    for key in emg_preprocessed.keys():    # Convert all float64 arrays to float32
        emg_preprocessed[key] = [arr.astype('float32') for arr in emg_preprocessed[key]]

    # select certain grids
    if selected_grid == 'all':
        emg_grid_selected = emg_preprocessed
    elif selected_grid == 'grid_1':
        emg_grid_selected = {key: [arr[:, :65] for arr in value] for key, value in emg_preprocessed.items()}
    elif selected_grid == 'grid_2' and emg_preprocessed['emg_LWLW'][0].shape[1] > 65:
        emg_grid_selected = {key: [arr[:, 65:130] for arr in value] for key, value in emg_preprocessed.items()}
    else:
        raise Exception('selected wrong grids!')

    return emg_grid_selected


## normalize and reshape them to be images, then spatial filtering the images if needed
def normalizeFilterEmgData(old_emg_preprocessed, new_emg_preprocessed, limit, normalize='(0,1)', spatial_filter=True, kernel=(1, 1),
        axes=(1, 2), radius=None):
    old_emg_normalized = Data_Processing.normalizeEmgData(old_emg_preprocessed, range_limit=limit, normalize=normalize)
    new_emg_normalized = Data_Processing.normalizeEmgData(new_emg_preprocessed, range_limit=limit, normalize=normalize)
    num_samples = old_emg_normalized['emg_LWLW'][0].shape[0]
    old_emg_reshaped = {k: [np.transpose(np.reshape(arr, newshape=(num_samples, 13, -1, 1), order='F'), (0, 3, 1, 2)).
        astype(np.float32) for arr in v] for k, v in old_emg_normalized.items()}
    new_emg_reshaped = {k: [np.transpose(np.reshape(arr, newshape=(num_samples, 13, -1, 1), order='F'), (0, 3, 1, 2)).
        astype(np.float32) for arr in v] for k, v in new_emg_normalized.items()}
    if spatial_filter:
        old_emg_reshaped = spatialFilterEmgData(old_emg_reshaped, kernel=kernel, axes=axes, radius=radius)
        new_emg_reshaped = spatialFilterEmgData(new_emg_reshaped, kernel=kernel, axes=axes, radius=radius)
        old_emg_normalized = {k: [np.reshape(np.transpose(arr, (0, 2, 3, 1)), newshape=(num_samples, -1), order='F') for arr in v]
        for k, v in old_emg_reshaped.items()}
        new_emg_normalized = {k: [np.reshape(np.transpose(arr, (0, 2, 3, 1)), newshape=(num_samples, -1), order='F') for arr in v]
        for k, v in new_emg_reshaped.items()}
    return old_emg_normalized, new_emg_normalized, old_emg_reshaped, new_emg_reshaped


## spatial filtering images
def spatialFilterEmgData(emg_reshaped, kernel=(1, 1), axes=(1, 2), radius=None):
    '''
    Performs Gaussian spatial filtering on EMG data.
    emg_reshaped: The EMG data reshaped for spatial filtering. each array should be (num_samples, num_channels, height, width).
    sigma (int or float, default=1): The standard deviation of the Gaussian kernel. Defines the filter's spread.
    axes (tuple, default=(2, 3)): The dimensions along which filtering will occur, typically the spatial dimensions.
    radius (int, default=4): The truncation radius for the filter.
    '''
    filtered_emg_reshaped = copy.deepcopy(emg_reshaped)
    for locomotion_mode, locomotion_data in emg_reshaped.items():
        # Stack together all images in the list of the given locomotion mode
        combined_images = np.vstack(locomotion_data)
        # Reshape images to (850,13,5)
        squeezed_images = combined_images.squeeze(axis=1)
        # Apply gaussian filter (kernel size = sigma * truncate(default:4))
        filtered_images = gaussian_filter(squeezed_images, sigma=kernel, mode='reflect', axes=axes, radius=radius)  # filter two dims one by one
        # filtered_images = cv2.blur(squeezed_images, kernel)
        # Reshape the filtered images back to original shape
        unsqueezed_images = filtered_images[:, np.newaxis, :, :]
        # Splitting the combined image array back into a list of 60 images
        split_images_list = [img for img in np.array_split(unsqueezed_images, len(locomotion_data), axis=0)]
        filtered_emg_reshaped[locomotion_mode] = split_images_list
    return filtered_emg_reshaped


## spatial filtering images
def spatialFilterBlendingFactors(gen_results, kernel=(1, 1), axes=(2, 3), radius=None):
    '''
    Performs Gaussian spatial filtering on EMG data.
    gen_results: The blending factors for spatial filtering. each array should be (num_samples, num_channels, height, width).
    sigma (int or float, default=1): The standard deviation of the Gaussian kernel. Defines the filter's spread.
    axes (tuple, default=(2, 3)): The dimensions along which filtering will occur, typically the spatial dimensions.
    radius (int, default=4): The truncation radius for the filter.
    '''
    filtered_gen_results = copy.deepcopy(gen_results)
    for locomotion_mode, locomotion_data in gen_results.items():
        blending_factors = locomotion_data['model_results']
        # Sort keys based on the timepoint value
        sorted_keys = sorted(blending_factors.keys(), key=lambda x: int(x.split('_')[1]))
        # Stack arrays from sorted keys
        combined_images = np.vstack([blending_factors[key] for key in sorted_keys])
        # Apply gaussian filter (kernel size = sigma * truncate(default:4))
        filtered_images = gaussian_filter(combined_images, sigma=kernel, mode='reflect', axes=axes, radius=radius)  # filter two dims one by one
        # Splitting the combined image array back into a list of 60 images
        for idx, key in enumerate(sorted_keys):
            filtered_gen_results[locomotion_mode]['model_results'][key] = filtered_images[idx, :, :, :].reshape(1, 2, 13, 5)
    return filtered_gen_results


## extract the relevent modes for data generation and separate them by timepoint_interval
def extractSeparateEmgData(modes_generation, old_emg_reshaped, new_emg_reshaped, time_interval, length, output_list=False):
    extracted_emg = {}
    train_gan_data = {}
    data_keys = ['gen_data_1', 'gen_data_2', 'disc_data']  # The order in the list is critical, corresponding to the locomotion modes

    for transition_type, modes in modes_generation.items():
        # Initialize transition_type key in real_emg and train_gan_data dictionaries
        extracted_emg[transition_type] = {'old': {}, 'new': {}}
        train_gan_data[transition_type] = {'gen_data_1': None, 'gen_data_2': None, 'disc_data': None}

        for idx, mode in enumerate(modes):
            # Assign values using the new structure
            extracted_emg[transition_type]['old'][mode] = np.vstack(old_emg_reshaped[mode])
            extracted_emg[transition_type]['new'][mode] = np.vstack(new_emg_reshaped[mode])

            train_gan_data[transition_type][data_keys[idx]] = Process_Fake_Data.separateByInterval(
                extracted_emg[transition_type]['old'][mode], timepoint_interval=time_interval, length=length, output_list=output_list)

    return extracted_emg, train_gan_data


## get window parameters for gan and classify model training
def returnWindowParameters(start_before_toeoff_ms, endtime_after_toeoff_ms, feature_window_ms, predict_window_ms):
    down_sampling = True
    start_before_toeoff_ms = start_before_toeoff_ms
    endtime_after_toeoff_ms = endtime_after_toeoff_ms
    feature_window_ms = feature_window_ms
    predict_window_ms = predict_window_ms
    sample_rate = 1 if down_sampling is True else 2
    predict_window_size = predict_window_ms * sample_rate
    feature_window_size = feature_window_ms * sample_rate
    predict_window_increment_ms = 20
    feature_window_increment_ms = 20
    predict_window_shift_unit = int(predict_window_increment_ms / feature_window_increment_ms)
    predict_using_window_number = int((predict_window_size - feature_window_size) / (feature_window_increment_ms * sample_rate)) + 1
    predict_window_per_repetition = int(
        (endtime_after_toeoff_ms + start_before_toeoff_ms - predict_window_ms) / predict_window_increment_ms) + 1

    window_parameters = {'down_sampling': down_sampling, 'start_before_toeoff_ms': start_before_toeoff_ms,
        'endtime_after_toeoff_ms': endtime_after_toeoff_ms, 'feature_window_ms': feature_window_ms, 'predict_window_ms': predict_window_ms,
        'sample_rate': sample_rate, 'predict_window_size': predict_window_size, 'feature_window_size': feature_window_size,
        'predict_window_increment_ms': predict_window_increment_ms, 'feature_window_increment_ms': feature_window_increment_ms,
        'predict_window_shift_unit': predict_window_shift_unit, 'predict_using_window_number': predict_using_window_number,
        'predict_window_per_repetition': predict_window_per_repetition}

    return window_parameters



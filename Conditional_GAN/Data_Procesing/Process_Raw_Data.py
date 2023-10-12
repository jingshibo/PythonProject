'''
    load original data for cGAN model training
'''

##
from Transition_Prediction.Pre_Processing import Preprocessing
from Transition_Prediction.Models.Utility_Functions import Data_Preparation
from Cycle_GAN.Functions import Data_Processing
from Conditional_GAN.Data_Procesing import Process_Fake_Data
import numpy as np


## load raw data and filter them
def readFilterEmgData(data_source, window_parameters, lower_limit=20, higher_limit=400, envelope_cutoff=10, envelope=False,
        project='Insole_Emg', selected_grid='all'):
    split_parameters = Preprocessing.readSplitParameters(data_source['subject'], data_source['version'], project=project)
    emg_filtered_data = Preprocessing.labelFilteredData(data_source['subject'], data_source['modes'],
        data_source['sessions'], data_source['version'], split_parameters, project=project,
        start_position=-int(window_parameters['start_before_toeoff_ms'] * (2 / window_parameters['sample_rate'])),
        end_position=int(window_parameters['endtime_after_toeoff_ms'] * (2 / window_parameters['sample_rate'])), lower_limit=lower_limit,
        higher_limit=higher_limit, envelope_cutoff=envelope_cutoff, envelope=envelope, notchEMG=False, median_filtering=True, reordering=True)
    emg_preprocessed = Data_Preparation.removeSomeSamples(emg_filtered_data, is_down_sampling=window_parameters['down_sampling'])
    for key in emg_preprocessed.keys():    # Convert all float64 arrays to float32
        emg_preprocessed[key] = [arr.astype('float32') for arr in emg_preprocessed[key]]

    # select certain grids
    if selected_grid == 'all':
        emg_grid_selected = emg_preprocessed
    elif selected_grid == 'grid_1':
        emg_grid_selected = {key: [arr[:, :65] for arr in value] for key, value in emg_preprocessed.items()}
    elif selected_grid == 'grid_2':
        emg_grid_selected = {key: [arr[:, 65:130] for arr in value] for key, value in emg_preprocessed.items()}
    else:
        raise Exception('selected wrong grids!')

    return emg_grid_selected


## normalize filtered data and reshape them to be images
def normalizeReshapeEmgData(old_emg_preprocessed, new_emg_preprocessed, limit, normalize='(0,1)'):
    old_emg_normalized = Data_Processing.normalizeEmgData(old_emg_preprocessed, range_limit=limit, normalize=normalize)
    new_emg_normalized = Data_Processing.normalizeEmgData(new_emg_preprocessed, range_limit=limit, normalize=normalize)
    num_samples = old_emg_normalized['emg_LWLW'][0].shape[0]
    old_emg_reshaped = {k: [np.transpose(np.reshape(arr, newshape=(num_samples, 13, -1, 1), order='F'), (0, 3, 1, 2)).astype(np.float32) for arr in v]
        for k, v in old_emg_normalized.items()}
    new_emg_reshaped = {k: [np.transpose(np.reshape(arr, newshape=(num_samples, 13, -1, 1), order='F'), (0, 3, 1, 2)).astype(np.float32) for arr in v]
        for k, v in new_emg_normalized.items()}

    return old_emg_normalized, new_emg_normalized, old_emg_reshaped, new_emg_reshaped


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
def returnWindowParameters(start_before_toeoff_ms, endtime_after_toeoff_ms, feature_window_ms):
    down_sampling = True
    start_before_toeoff_ms = start_before_toeoff_ms
    endtime_after_toeoff_ms = endtime_after_toeoff_ms
    feature_window_ms = feature_window_ms
    predict_window_ms = start_before_toeoff_ms
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



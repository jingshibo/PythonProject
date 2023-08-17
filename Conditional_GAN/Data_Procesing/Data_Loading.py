##
from Transition_Prediction.Pre_Processing import Preprocessing
from Transition_Prediction.Models.Utility_Functions import Data_Preparation
from Cycle_GAN.Functions import Data_Processing
from Conditional_GAN.Data_Procesing import cGAN_Processing
import numpy as np


##
def readFilterEmgData(data_source, window_parameters, lower_limit=20, higher_limit=400):
    split_parameters = Preprocessing.readSplitParameters(data_source['subject'], data_source['version'])
    emg_filtered_data = Preprocessing.labelFilteredData(data_source['subject'], data_source['modes'],
        data_source['sessions'], data_source['version'], split_parameters,
        start_position=-int(window_parameters['start_before_toeoff_ms'] * (2 / window_parameters['sample_rate'])),
        end_position=int(window_parameters['endtime_after_toeoff_ms'] * (2 / window_parameters['sample_rate'])), lower_limit=lower_limit,
        higher_limit=higher_limit, notchEMG=False, reordering=True, median_filtering=True)
    old_emg_preprocessed = Data_Preparation.removeSomeSamples(emg_filtered_data, is_down_sampling=window_parameters['down_sampling'])

    return old_emg_preprocessed


def normalizeReshapeEmgData(old_emg_preprocessed, new_emg_preprocessed, limit):
    old_emg_normalized = Data_Processing.normalizeEmgData(old_emg_preprocessed, range_limit=limit)
    new_emg_normalized = Data_Processing.normalizeEmgData(new_emg_preprocessed, range_limit=limit)
    old_emg_reshaped = {k: [np.transpose(np.reshape(arr, newshape=(-1, 13, 10, 1), order='F'), (0, 3, 1, 2)).astype(np.float32) for arr in v]
        for k, v in old_emg_normalized.items()}
    new_emg_reshaped = {k: [np.transpose(np.reshape(arr, newshape=(-1, 13, 10, 1), order='F'), (0, 3, 1, 2)).astype(np.float32) for arr in v]
        for k, v in new_emg_normalized.items()}

    return old_emg_normalized, new_emg_normalized, old_emg_reshaped, new_emg_reshaped


def extractSeparateEmgData(modes, old_emg_reshaped, new_emg_reshaped, time_interval, length, output_list=False):
    real_emg = {'old': {}, 'new': {}}
    train_gan_data = {'gen_data_1': None, 'gen_data_2': None, 'disc_data': None}
    data_keys = list(train_gan_data.keys())

    for idx, mode in enumerate(modes):
        real_emg['old'][mode] = np.vstack(old_emg_reshaped[mode])
        real_emg['new'][mode] = np.vstack(new_emg_reshaped[mode])

        train_gan_data[data_keys[idx]] = cGAN_Processing.separateByTimeInterval(real_emg['old'][mode], timepoint_interval=time_interval,
            length=length, output_list=output_list)

    return real_emg, train_gan_data


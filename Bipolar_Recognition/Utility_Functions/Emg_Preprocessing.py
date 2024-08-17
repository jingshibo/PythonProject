import os
import json
import pandas as pd
import numpy as np
from Transition_Prediction.RawData.Utility_Functions import Upsampling_Filtering, Insole_Data_Splition
from Bipolar_Recognition.Utility_Functions import Emg_Preprocessing
from Transition_Prediction.Pre_Processing.Utility_Functions import Data_Reshaping


## read all sensor data after alignment from a csc file
def readAlignedData(subject, mode, project='HDsEMG_Recognition'):
    data_dir = f'D:\Data\\{project}\subject_{subject}\\aligned_data'
    data_file = f'emg_subject_{subject}_{mode}_aligned.csv'
    emg_path = os.path.join(data_dir, data_file)

    emg_aligned = pd.read_csv(emg_path)
    emg_aligned.columns = emg_aligned.columns.astype(int)

    return emg_aligned


## preprocess EMG data, such as filtering and reordering
def preprocessEmgData(subject, modes):
    emg_preprocessed = {}
    for mode, name in modes.items():
        # read data
        split_parameters = Insole_Data_Splition.readSplitParameters(subject, project='HDsEMG_Recognition')
        emg_aligned = Emg_Preprocessing.readAlignedData(subject, mode, project='HDsEMG_Recognition')

        # extract data
        emg_stable = []
        for param_list in split_parameters[mode]:
            emg_stable.append(emg_aligned.iloc[param_list[0]: param_list[1], :])  # select required columns
        emg_extract = pd.concat(emg_stable, axis=0)

        # preprocess data
        emg_filtered = Upsampling_Filtering.filterEmg(emg_extract, lower_limit=20, higher_limit=400, median_filtering=True)
        emg_filtered = Data_Reshaping.insertElectrode(emg_filtered)
        emg_preprocessed[name] = Data_Reshaping.reorderElectrodes(emg_filtered)

    return emg_preprocessed


## separate data into windows
def createWindows(data, window_size, increment):
    windows_data = []
    for start in range(0, len(data) - window_size + 1, increment):
        end = start + window_size
        window = data.iloc[start:end]
        if len(window) == window_size:  # drop the last one with the size smaller than the window size
            windows_data.append(window.to_numpy())
    return windows_data


## save calculated emg features
def saveFeatures(subject, emg_features, feature_set):
    data_dir = f'D:\Data\HDsEMG_Recognition\subject_{subject}\\features'

    # save emg features
    emg_feature_file = f'subject_{subject}_emg_feature_set_{feature_set}.json'
    emg_feature_path = os.path.join(data_dir, emg_feature_file)

    with open(emg_feature_path, 'w') as json_file:
        json.dump(emg_features, json_file, indent=8)


## read calculated emg features
def readFeatures(subject, feature_set):
    data_dir = f'D:\Data\HDsEMG_Recognition\subject_{subject}\\features'

    # load emg features
    emg_feature_file = f'subject_{subject}_emg_feature_set_{feature_set}.json'
    emg_feature_path = os.path.join(data_dir, emg_feature_file)

    with open(emg_feature_path) as json_file:
        emg_features = json.load(json_file)

    return emg_features


## save interpolated emg features
def saveInterpFeatures(subject, emg_features, feature_set):
    data_dir = f'D:\Data\HDsEMG_Recognition\subject_{subject}\\features'

    # save emg features
    emg_feature_file = f'subject_{subject}_emg_interp_feature_set_{feature_set}.npz'
    emg_feature_path = os.path.join(data_dir, emg_feature_file)

    np.savez_compressed(emg_feature_path, **emg_features)

## read calculated emg and imu features
def readInterpFeatures(subject, feature_set):
    data_dir = f'D:\Data\HDsEMG_Recognition\subject_{subject}\\features'

    # load emg features
    emg_feature_file = f'subject_{subject}_emg_interp_feature_set_{feature_set}.npz'
    emg_feature_path = os.path.join(data_dir, emg_feature_file)

    # Load the .npz file
    loaded_data = np.load(emg_feature_path)
    # Convert the loaded .npz file back to a dictionary
    emg_features = {key: loaded_data[key] for key in loaded_data.files}

    return emg_features


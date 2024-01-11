##
from Bipolar_EMG.Utility_Functions import Emg_Imu_Preprocessing
from Transition_Prediction.RawData.Utility_Functions import Upsampling_Filtering, Insole_Data_Splition
from Transition_Prediction.Pre_Processing.Utility_Functions import Feature_Calculation
from scipy import signal
import numpy as np
import pandas as pd


## extract data
subject = 'Number1'

emg_filtered = {}
imu_filtered = {}
modes = {'standing': 'SS', 'level': 'LW', 'upstairs': 'SA', 'downstairs': 'SD', 'upslope': 'RA', 'downslope': 'RD'}
for mode, name in modes.items():
    # read data
    split_parameters = Insole_Data_Splition.readSplitParameters(subject, project='Bipolar_Data')
    imu_aligned, emg_aligned = Emg_Imu_Preprocessing.readAlignedData(subject, mode, project='Bipolar_Data')
    imu_upsampled = Emg_Imu_Preprocessing.upsampleImuEqualToEmg(imu_aligned, emg_aligned)

    # extract data
    emg_stable = []
    imu_stable = []
    for param_list in split_parameters[mode]:
        emg_stable.append(emg_aligned.iloc[param_list[0]: param_list[1], 1:-1])  # select required columns
        imu_stable.append(imu_upsampled.iloc[param_list[0]: param_list[1], 1:])  # select required columns
    emg_extract = pd.concat(emg_stable, axis=0)
    imu_extract = pd.concat(imu_stable, axis=0)

    # filter data
    emg_filtered[name] = Upsampling_Filtering.filterEmg(emg_extract, lower_limit=20, higher_limit=400, median_filtering=False)
    sos = signal.butter(4, [45], fs=2000, btype="lowpass", output='sos')
    imu_filtered[name] = pd.DataFrame(signal.sosfiltfilt(sos, imu_extract, axis=0))  # only filter the measurements


## reorganize by sliding windows
window_size = 512  # 256ms
window_increment = 64  # 32ms

# separate data into windows
def createWindows(data, window_size, increment):
    windows_data = []
    for start in range(0, len(data) - window_size + 1, increment):
        end = start + window_size
        window = data.iloc[start:end]
        if len(window) == window_size:  # drop the last one with the size smaller than the window size
            windows_data.append(window.to_numpy())
    return windows_data

emg_windowed = {}
imu_windowed = {}
for mode, name in modes.items():
    emg_windowed[name] = createWindows(emg_filtered[name], window_size, window_increment)
    imu_windowed[name] = createWindows(imu_filtered[name], window_size, window_increment)


## calculate features for each window data
emg_features = {}
imu_features = {}
for mode, name in modes.items():
    emg_feature_list = []
    for emg_window_data in emg_windowed[name]:
        emg_feature = Feature_Calculation.calcuEmgFeatures(emg_window_data)
        emg_feature_list.append(emg_feature)
    emg_features[name] = np.vstack(emg_feature_list).tolist()  # convert numpy to list for dict storage

    imu_feature_list = []
    for imu_window_data in imu_windowed[name]:
        imu_feature = Feature_Calculation.calcuImuFeatures(imu_window_data)
        imu_feature_list.append(imu_feature)
    imu_features[name] = np.vstack(imu_feature_list).tolist()  # convert numpy to list for dict storage


## save features
feature_set = 0  # there may be multiple sets of features to be calculated for comparison
Emg_Imu_Preprocessing.saveFeatures(subject, emg_features, imu_features, feature_set)


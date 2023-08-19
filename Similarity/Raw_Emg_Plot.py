'''
plot the filtered emg data from each channel at each gait event
'''


## modules
from Transition_Prediction.Pre_Processing.Utility_Functions import Data_Reshaping, Feature_Storage
from Transition_Prediction.Models.Utility_Functions import Data_Preparation
from Transition_Prediction.Pre_Processing import Preprocessing
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import copy

## input emg labelled series data
subject = 'Number4'
version = 0  # the data from which experiment version to process
modes = ['up_down', 'down_up']
up_down_session = [0, 1, 2, 3, 4]
down_up_session = [0, 1, 2, 5, 6]
sessions = [up_down_session, down_up_session]

# labelled emg series data
split_parameters = Preprocessing.readSplitParameters(subject, version)
combined_emg_labelled = Preprocessing.labelFilteredData(subject, modes, sessions, version, split_parameters, start_position=-1000,
    end_position=800, lower_limit=20, higher_limit=400, envelope_cutoff=10, notchEMG=False, median_filtering=True, reordering=True,
    envelope=False)
emg_preprocessed = Data_Preparation.removeSomeSamples(combined_emg_labelled)


## organize all summed emg data
emg_1_mean_channels = {}
emg_2_mean_channels = {}
emg_1_mean_events = {}  # emg device 1: tibialis
emg_2_mean_events = {}  # emg device 2: rectus
# sum the emg data of all channels and samples up for the same gait event
for gait_event_label, gait_event_emg in emg_preprocessed.items():
    emg_1_mean_channels[f"{gait_event_label}_data"] = [np.sum(emg_per_repetition[:, 0:65], axis=1) / 65 for emg_per_repetition in
        gait_event_emg]  # average the emg values of all channels
    emg_1_mean_events[f"{gait_event_label}_data"] = np.add.reduce(emg_1_mean_channels[f"{gait_event_label}_data"]) / len\
            (emg_1_mean_channels[f"{gait_event_label}_data"])  # average all repetitions for the same gait event
    emg_1_mean_channels[f"{gait_event_label}_data"].insert(0, emg_1_mean_events[f"{gait_event_label}_data"])  # insert the mean event value in front of the dataset
    emg_2_mean_channels[f"{gait_event_label}_data"] = [np.sum(emg_per_repetition[:, 65:130], axis=1) / 65 for emg_per_repetition in
        gait_event_emg]  # average the emg values of all channels
    emg_2_mean_events[f"{gait_event_label}_data"] = np.add.reduce(emg_2_mean_channels[f"{gait_event_label}_data"]) / len\
            (emg_2_mean_channels[f"{gait_event_label}_data"])  # average all repetitions for the same gait event
    emg_2_mean_channels[f"{gait_event_label}_data"].insert(0, emg_2_mean_events[f"{gait_event_label}_data"])


## organize all summed absolute emg value
emg_1_mean_channels = {}
emg_2_mean_channels = {}
emg_1_mean_events = {}  # emg device 1: tibialis
emg_2_mean_events = {}  # emg device 2: rectus
# sum the emg data of all channels and samples up for the same gait event
for gait_event_label, gait_event_emg in emg_preprocessed.items():
    emg_1_mean_channels[f"{gait_event_label}_data"] = [np.sum(np.abs(emg_per_repetition[:, 0:65]), axis=1) / 65 for emg_per_repetition in
        gait_event_emg]  # average the emg values of all channels
    emg_1_mean_events[f"{gait_event_label}_data"] = np.add.reduce(emg_1_mean_channels[f"{gait_event_label}_data"]) / len\
            (emg_1_mean_channels[f"{gait_event_label}_data"])  # average all repetitions for the same gait event
    emg_1_mean_channels[f"{gait_event_label}_data"].insert(0, emg_1_mean_events[f"{gait_event_label}_data"])  # insert the mean event value in front of the dataset
    emg_2_mean_channels[f"{gait_event_label}_data"] = [np.sum(np.abs(emg_per_repetition[:, 65:130]), axis=1) / 65 for emg_per_repetition in
        gait_event_emg]  # average the emg values of all channels
    emg_2_mean_events[f"{gait_event_label}_data"] = np.add.reduce(emg_2_mean_channels[f"{gait_event_label}_data"]) / len\
            (emg_2_mean_channels[f"{gait_event_label}_data"])  # average all repetitions for the same gait event
    emg_2_mean_channels[f"{gait_event_label}_data"].insert(0, emg_2_mean_events[f"{gait_event_label}_data"])


##  plot summed series emg data
left_number = 000
right_number = 1800
emg_1_value = copy.deepcopy(emg_1_mean_events)
for event, value in emg_1_mean_events.items():
    emg_1_value[event] = value[left_number:right_number]
emg_2_value = copy.deepcopy(emg_2_mean_events)
for event, value in emg_2_mean_events.items():
    emg_2_value[event] = value[left_number:right_number]
pd.DataFrame(emg_1_value).plot(subplots=True, layout=(4, 4), title="EMG 1")
pd.DataFrame(emg_2_value).plot(subplots=True, layout=(4, 4), title="EMG 2")


##  plot summed series emg data by group
emd_data = emg_2_value
x = list(range(left_number-1000, right_number-1000))
plt.figure()
plt.plot(x, emd_data["emg_LWLW_data"], x, emd_data["emg_LWSA_data"], x, emd_data["emg_SASA_data"], x, emd_data["emg_SALW_data"])
plt.legend(['emg_LWLW_data', 'emg_LWSA_data', 'emg_SASA_data', 'emg_SALW_data'])
plt.title("LWSA")
plt.figure()
plt.plot(x, emd_data["emg_LWLW_data"], x, emd_data["emg_LWSD_data"], x, emd_data["emg_SDSD_data"], x, emd_data["emg_SDLW_data"])
plt.legend(['emg_LWLW_data', 'emg_LWSD_data', 'emg_SDSD_data', 'emg_SDLW_data'])
plt.title("LWSD")


##  plot summed series emg data by group
emd_data = emg_2_value
x = list(range(left_number-1000, right_number-1000))
plt.figure()
plt.plot(x, emd_data["emg_LWLW_data"], x, emd_data["emg_LWSA_data"], x, emd_data["emg_LWSD_data"], x, emd_data["emg_LWSS_data"])
plt.legend(['emg_LWLW_data', 'emg_LWSA_data', 'emg_LWSD_data', 'emg_LWSS_data'])
plt.title("LW")
plt.figure()
plt.plot(x, emd_data["emg_SASA_data"], x, emd_data["emg_SALW_data"], x, emd_data["emg_SASS_data"])
plt.legend(['emg_SASA_data', 'emg_SALW_data', 'emg_SASS_data'])
plt.title("SA")
plt.figure()
plt.plot(x, emd_data["emg_SDSD_data"], x, emd_data["emg_SDLW_data"], x, emd_data["emg_SDSS_data"])
plt.legend(['emg_SDSD_data', 'emg_SDLW_data', 'emg_SDSS_data'])
plt.title("SD")
plt.figure()
plt.plot(x, emd_data["emg_SSLW_data"], x, emd_data["emg_SSSA_data"], x, emd_data["emg_SSSD_data"])
plt.legend(['emg_SSLW_data', 'emg_SSSA_data', 'emg_SSSD_data'])
plt.title("SS")





## organize channel summed emg data
emd_data = emg_1_mean_channels
start_index = 0  # note: the first image is the mean value for the entire dataset of the gait event (emg_1_mean_events)
end_index = 49
horizontal = 7
vertical = 7

## transit from LW
(pd.DataFrame(emd_data["emg_LWLW_data"])).T.iloc[:, start_index:end_index].plot(subplots=True, layout=(horizontal, vertical), title="LWLW")
(pd.DataFrame(emd_data["emg_LWSA_data"])).T.iloc[:, start_index:end_index].plot(subplots=True, layout=(horizontal, vertical), title="LWSA")
(pd.DataFrame(emd_data["emg_LWSD_data"])).T.iloc[:, start_index:end_index].plot(subplots=True, layout=(horizontal, vertical), title="LWSD")
(pd.DataFrame(emd_data["emg_LWSS_data"])).T.iloc[:, start_index:end_index].plot(subplots=True, layout=(horizontal, vertical), title="LWSS")
## transit from SA
(pd.DataFrame(emd_data["emg_SASA_data"])).T.iloc[:, start_index:end_index].plot(subplots=True, layout=(horizontal, vertical), title="SASA")
(pd.DataFrame(emd_data["emg_SALW_data"])).T.iloc[:, start_index:end_index].plot(subplots=True, layout=(horizontal, vertical), title="SALW")
(pd.DataFrame(emd_data["emg_SASS_data"])).T.iloc[:, start_index:end_index].plot(subplots=True, layout=(horizontal, vertical), title="SASS")
## transit from SD
(pd.DataFrame(emd_data["emg_SDSD_data"])).T.iloc[:, start_index:end_index].plot(subplots=True, layout=(horizontal, vertical), title="SDSD")
(pd.DataFrame(emd_data["emg_SDLW_data"])).T.iloc[:, start_index:end_index].plot(subplots=True, layout=(horizontal, vertical), title="SDLW")
(pd.DataFrame(emd_data["emg_SDSS_data"])).T.iloc[:, start_index:end_index].plot(subplots=True, layout=(horizontal, vertical), title="SDSS")
## transit from SS
(pd.DataFrame(emd_data["emg_SSLW_data"])).T.iloc[:, start_index:end_index].plot(subplots=True, layout=(horizontal, vertical), title="SSLW")
(pd.DataFrame(emd_data["emg_SSSA_data"])).T.iloc[:, start_index:end_index].plot(subplots=True, layout=(horizontal, vertical), title="SSSA")
(pd.DataFrame(emd_data["emg_SSSD_data"])).T.iloc[:, start_index:end_index].plot(subplots=True, layout=(horizontal, vertical), title="SSSD")
## no transition
(pd.DataFrame(emd_data["emg_LWLW_data"])).T.iloc[:, start_index:end_index].plot(subplots=True, layout=(horizontal, vertical), title="LWLW")
(pd.DataFrame(emd_data["emg_SASA_data"])).T.iloc[:, start_index:end_index].plot(subplots=True, layout=(horizontal, vertical), title="SASA")
(pd.DataFrame(emd_data["emg_SDSD_data"])).T.iloc[:, start_index:end_index].plot(subplots=True, layout=(horizontal, vertical), title="SDSD")


## input emg feature data
subject = "Shibo"
version = 1  # which experiment data to process
feature_set = 0  # which feature set to use
emg_feature_data = Feature_Storage.readEmgFeatures(subject, version, feature_set)
# if you need to use CNN model, you need to reshape the data
emg_feature_reshaped = Data_Reshaping.reshapeEmgFeatures(emg_feature_data)
# reduce the sample number of certain modes to balance the dataset
emg_feature_sample_reduced = copy.deepcopy(emg_feature_data)
emg_feature_sample_reduced['emg_LW_features'] = emg_feature_sample_reduced['emg_LW_features'][
int(emg_feature_data['emg_LW_features'].shape[0] / 4): int(emg_feature_data['emg_LW_features'].shape[0] * 3 / 4), :]






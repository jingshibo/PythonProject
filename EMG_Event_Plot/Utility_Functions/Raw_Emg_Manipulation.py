## import
import numpy as np
import os
import pickle
import matplotlib.pyplot as plt
import copy


## average raw emg data
def calcuAverageEvent(emg_preprocessed):
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
    emg_event_mean = {'emg_1_mean_events': emg_1_mean_events, 'emg_2_mean_events': emg_2_mean_events}

    return emg_event_mean


## save averaged data
def saveAverageEvent(subject, version, result_set, emg_event_mean):
    data_dir = f'D:\Data\Insole_Emg\subject_{subject}\Experiment_{version}\event_value'
    mean_event_file = f'subject_{subject}_Experiment_{version}_emg_mean_set_{result_set}.pickle'
    mean_event_path = os.path.join(data_dir, mean_event_file)

    # Save the dictionary
    with open(mean_event_path, "wb") as f:
        pickle.dump(emg_event_mean, f)


## load averaged data
def loadAverageEvent(subject, version, result_set):
    data_dir = f'D:\Data\Insole_Emg\subject_{subject}\Experiment_{version}\event_value'
    mean_event_file = f'subject_{subject}_Experiment_{version}_emg_mean_set_{result_set}.pickle'
    mean_event_path = os.path.join(data_dir, mean_event_file)

    # Load the dictionary from the .npz file
    with open(mean_event_path, "rb") as f:
        loaded_dict = pickle.load(f)

    return loaded_dict


##  plot averaged series emg data
def plotMeanEvent(emg_mean_value, left_number=100, right_number=1800):
    # slice emg data from certain positions
    emd_data = copy.deepcopy(emg_mean_value)
    for event, value in emg_mean_value.items():
        emd_data[event] = value[left_number:right_number]
    # pd.DataFrame(emg_1_value).plot(subplots=True, layout=(4, 4), title="EMG 1")

    #  plot averaged emg data by group
    x = list(range(left_number-1000, right_number-1000))
    plt.figure()
    plt.plot(x, emd_data["emg_LWLW_data"], x, emd_data["emg_LWSA_data"], x, emd_data["emg_LWSD_data"], x, emd_data["emg_LWSS_data"])
    plt.legend(['LW_LW', 'LW_SA', 'LW_SD', 'LW_SS'])
    plt.title("LW")
    plt.figure()
    plt.plot(x, emd_data["emg_SASA_data"], x, emd_data["emg_SALW_data"], x, emd_data["emg_SASS_data"])
    plt.legend(['SA_SA', 'SA_LW', 'SA_SS'])
    plt.title("SA")
    plt.figure()
    plt.plot(x, emd_data["emg_SDSD_data"], x, emd_data["emg_SDLW_data"], x, emd_data["emg_SDSS_data"])
    plt.legend(['SD_SD', 'SD_LW', 'SD_SS'])
    plt.title("SD")
    plt.figure()
    plt.plot(x, emd_data["emg_SSLW_data"], x, emd_data["emg_SSSA_data"], x, emd_data["emg_SSSD_data"])
    plt.legend(['SS_LW', 'SS_SA', 'SS_SD'])
    plt.title("SS")
## import modules
import pandas as pd
import numpy as np
import datetime
import matplotlib.pyplot as plt

from Processing import Extract_Label_Features
from dtaidistance import dtw  # this method seems to work worse than Accelerated DTW package
from dtaidistance import dtw_visualisation as dtwvis

## input emg labelled series data
subject = 'Shibo'
version = 1  # the data from which experiment version to process
modes = ['up_down', 'down_up']
up_down_session = [10, 11, 12, 13, 14, 15, 16, 17, 18, 19]
down_up_session = [10, 11, 12, 13, 19, 24, 25, 26, 27, 28, 20]
sessions = [up_down_session, down_up_session]

# labelled emg series data
split_data = Extract_Label_Features.readSplitData(subject, version)
# obtain the envelope of emg data
combined_emg_labelled = Extract_Label_Features.labelSensorData(subject, modes, sessions, version, split_data, envelope=True)


## Calcualte the mean value of each gait event
emg_1_mean_channels = {}
emg_2_mean_channels = {}
emg_1_mean_events = {}  # emg device 1: tibialis
emg_2_mean_events = {}  # emg device 2: rectus
# sum the emg data of all channels and samples up for the same gait event
for gait_event_label, gait_event_emg in combined_emg_labelled.items():
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
# plot summed series emg data
pd.DataFrame(emg_1_mean_events).plot(subplots=True, layout=(4, 4))
pd.DataFrame(emg_2_mean_events).plot(subplots=True, layout=(4, 4))

## Calculate the DTW distance between emg sequences
def calcuDtwDistance(input_data):
    dtw_distance = []
    warp_paths = []
    dtw_results = {}
    gait_group = [["LWLW", "LWSA", "LWSD", "LWSS"], ["SALW", "SASA", "SASS"], ["SDLW", "SDSD", "SDSS"], ["SSLW", "SSSA", "SSSD"]]  # 4 groups
    now = datetime.datetime.now()
    for gait_reference_label, gait_reference_data in input_data.items():  # the first element in gait_reference_data is the reference
        for gait_event_label, gait_event_data in input_data.items():  # the second-end elements in gait_event_data is the data
            # only comparing the emg sequences within one of the 4 groups
            if (any(x in gait_reference_label for x in gait_group[0]) and any(x in gait_event_label for x in gait_group[0]) or any(
                x in gait_reference_label for x in gait_group[1]) and any(x in gait_event_label for x in gait_group[1]) or any(
                x in gait_reference_label for x in gait_group[2]) and any(x in gait_event_label for x in gait_group[2]) or any(
                x in gait_reference_label for x in gait_group[3]) and any(x in gait_event_label for x in gait_group[3])):
                for i in range(1, len(gait_event_data)):  # comparing the emg data to a reference sequence
                    distance, paths = dtw.warping_paths_fast(gait_reference_data[0], gait_event_data[i])
                    best_path = dtw.best_path(paths)
                    dtw_distance.append(distance)
                    warp_paths.append(best_path)
                dtw_results[f"reference_{gait_reference_label}_{gait_event_label}"] = (dtw_distance, warp_paths)
                dtw_distance = []
                warp_paths = []
                print("reference:", gait_reference_label, ", data:", gait_event_label)
    print(datetime.datetime.now() - now)
    return dtw_results  # including distance and warp path

emg_1_dtw_distance = calcuDtwDistance(emg_1_mean_channels)
emg_2_dtw_distance = calcuDtwDistance(emg_2_mean_channels)


## save dtw data
subject = 'Shibo'
version = 0   # the data from which experiment version to process
from Similarity.Utility_Functions import Dtw_Storage

emg_dtw_results = {"emg_1_dtw_results": emg_1_dtw_distance, "emg_2_dtw_results": emg_2_dtw_distance}
Dtw_Storage.saveEmgDtw(subject, emg_dtw_results, version)

##
emg_1_average_value = {}
for dtw_label, dtw_data in emg_1_dtw_distance.items():
    mean_value = sum(emg_1_dtw_distance[dtw_label][0]) / len(emg_1_dtw_distance[dtw_label][0])
    emg_1_average_value[f'{dtw_label}'] = mean_value
emg_2_average_value = {}
for dtw_label, dtw_data in emg_2_dtw_distance.items():
    mean_value = sum(emg_2_dtw_distance[dtw_label][0]) / len(emg_2_dtw_distance[dtw_label][0])
    emg_2_average_value[f'{dtw_label}'] = mean_value


## plot warp path between two emg sequences
# signal_1 = input_data['emg_LWSA_data'][0]
# signal_2 = input_data['emg_LWSA_data'][50]
# # plot the two sequences
# fig = plt.figure()
# ax = plt.axes()
# # Remove the border and axes ticks
# fig.patch.set_visible(False)
# ax.axis('off')
# # plot warp path connecting the mapping points
# for [map_x, map_y] in dtw_results['reference_emg_LWSA_data_emg_SSLW_data'][1][49]:
#     ax.plot([map_x, map_y], [signal_1[map_x], signal_2[map_y]], linewidth=4)
# # figure setting
# ax.plot(signal_1, '-ro', label='x', linewidth=2, markersize=5, markerfacecolor='lightcoral', markeredgecolor='lightcoral')
# ax.plot(signal_2, '-bo', label='y', linewidth=2, markersize=5, markerfacecolor='skyblue', markeredgecolor='skyblue')
# ax.set_title("DTW Distance", fontsize=28, fontweight="bold")
# plt.legend()


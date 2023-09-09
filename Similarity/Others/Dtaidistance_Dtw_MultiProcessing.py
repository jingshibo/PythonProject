## import modules
import numpy as np
import datetime
import concurrent.futures

from Transition_Prediction.Pre_Processing import Preprocessing
from dtaidistance import dtw


## input emg labelled series data
def readEmgData(subject, version, modes, sessions):
    # labelled emg series data
    split_data = Preprocessing.readSplitParameters(subject, version)
    combined_emg_labelled = Preprocessing.labelFilteredData(subject, modes, sessions, version, split_data, envelope=True)
    combined_emg_labelled.pop('emg_LW', None)
    combined_emg_labelled.pop('emg_SD', None)
    combined_emg_labelled.pop('emg_SA', None)
    return combined_emg_labelled

## Calcualte the mean value of each gait event
def calculateEmgMean(combined_emg_labelled):
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
    return emg_1_mean_channels, emg_2_mean_channels


## using dtw package to calculate DTW distance
def calculateDtwDistance(gait_event_data, gait_event_label):
    dtw_result = {}
    dtw_distance = []
    warp_path = []
    for i in range(1, len(gait_event_data)):
        distance, paths = dtw.warping_paths_fast(gait_event_data[0], gait_event_data[i])
        best_path = dtw.best_path(paths)
        dtw_distance.append(distance)
        warp_path.append(best_path)
        dtw_result[f"{gait_event_label}_dtw"] = (dtw_distance, warp_path)
    return dtw_result  # including distance and warp path

# calculate emg features using multiprocessing. there is balance of CPU number, not more is better as numpy auto parallel to some extent
if __name__ == '__main__':
    # basic information
    subject = 'Shibo'
    version = 1  # the data from which experiment version to process
    modes = ['up_down', 'down_up']
    up_down_session = [10, 11, 12, 13, 14, 15, 16, 17, 18, 19]
    down_up_session = [10, 11, 12, 13, 19, 24, 25, 26, 27, 28, 20]
    sessions = [up_down_session, down_up_session]

    # calculate mean value
    combined_emg_labelled = readEmgData(subject, version, modes, sessions)
    emg_1_mean_channels, emg_2_mean_channels = calculateEmgMean(combined_emg_labelled)

    # calculate DTW distance
    input_data = emg_1_mean_channels
    now = datetime.datetime.now()
    dtw_results = []
    with concurrent.futures.ProcessPoolExecutor(max_workers=5) as executor:
        futures = [executor.submit(calculateDtwDistance, gait_event_data, gait_event_label) for gait_event_label, gait_event_data in
            input_data.items()]  # parallel calculate DTW distance in multiple gait events
        for future in concurrent.futures.as_completed(futures):
            dtw_results.append(future.result())

    # reorganize and label the calculated features
    self_compare_dtw = {}
    for gait_event_dtw in dtw_results:
        gait_event_label = list(gait_event_dtw.keys())[0]
        self_compare_dtw[gait_event_label] = gait_event_dtw[gait_event_label]
    print(datetime.datetime.now() - now)

# # plot the two sequences and connect the mapping points
dtwvis.plot_warpingpaths(signal_1, signal_2, paths, best_path)
fig = plt.figure()
ax = plt.axes()
# Remove the border and axes ticks
fig.patch.set_visible(False)
ax.axis('off')
for [map_x, map_y] in best_path:
    ax.plot([map_x, map_y], [signal_1[map_x], signal_2[map_y]], linewidth=4)
ax.plot(signal_1, '-ro', label='x', linewidth=2, markersize=5, markerfacecolor='lightcoral', markeredgecolor='lightcoral')
ax.plot(signal_2, '-bo', label='y', linewidth=2, markersize=5, markerfacecolor='skyblue', markeredgecolor='skyblue')
ax.set_title("DTW Distance", fontsize=28, fontweight="bold")
plt.legend()
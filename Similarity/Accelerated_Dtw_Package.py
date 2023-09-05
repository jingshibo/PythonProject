'''
using accelerated dtw package for DTW distance calculation
'''


## import modules
import numpy as np
import datetime
import concurrent.futures
from Transition_Prediction.Pre_Processing import Preprocessing
from dtw import accelerated_dtw
from Similarity.Utility_Functions import Dtw_Storage


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
def calcuDtwDistance(input_data, gait_reference_label, gait_reference_data):
    dtw_result = {}
    gait_group = [["LWLW", "LWSA", "LWSD", "LWSS"], ["SALW", "SASA", "SASS"], ["SDLW", "SDSD", "SDSS"], ["SSLW", "SSSA", "SSSD"]]  # 4 groups
    now = datetime.datetime.now()
    for gait_event_label, gait_event_data in input_data.items():  # the second-end elements in gait_event_data is the data
        # find and compare the emg sequences only within one of the 4 groups
        dtw_distance = []
        warp_paths = []
        if (any(x in gait_reference_label for x in gait_group[0]) and any(x in gait_event_label for x in gait_group[0]) or any(
            x in gait_reference_label for x in gait_group[1]) and any(x in gait_event_label for x in gait_group[1]) or any(
            x in gait_reference_label for x in gait_group[2]) and any(x in gait_event_label for x in gait_group[2]) or any(
            x in gait_reference_label for x in gait_group[3]) and any(x in gait_event_label for x in gait_group[3])):
            for i in range(1, len(gait_event_data)):  # comparing the emg data to a reference sequence (the first element in gait_reference_data)
                distance, cost_matrix, accumulated_cost_matrix, path = accelerated_dtw(gait_reference_data[0], gait_event_data[i], dist='euclidean')
                best_path = list(zip(path[0], path[1]))
                dtw_distance.append(distance)
                warp_paths.append(best_path)
            dtw_result[f"reference_{gait_reference_label}_{gait_event_label}"] = (dtw_distance, warp_paths)
            print("reference:", gait_reference_label, ", data:", gait_event_label)
    print("reference:", gait_reference_label, datetime.datetime.now() - now)
    return dtw_result  # including distance and warp path


## parallel computing DTW values for each event reference
def parallelCalculateDtw(input_data):
    # calculate DTW distance
    now = datetime.datetime.now()
    dtw_results = []
    print("start:", datetime.datetime.now())
    with concurrent.futures.ProcessPoolExecutor(max_workers=8) as executor:
        futures = [executor.submit(calcuDtwDistance, input_data, gait_reference_label, gait_reference_data) for
            gait_reference_label, gait_reference_data in input_data.items()]  # parallel calculate DTW distance with different reference value
        for future in concurrent.futures.as_completed(futures):
            dtw_results.append(future.result())
    print(datetime.datetime.now() - now)
    # reorganize and label the calculated features
    compare_dtw = {}
    for dtw_dict in dtw_results:
        for gait_event_label, gait_event_dtw in dtw_dict.items():
            compare_dtw[gait_event_label] = gait_event_dtw
    return compare_dtw

##
if __name__ == '__main__':
    # basic information
    subject = 'Shibo'
    version = 1  # process the data from experiment version 1
    modes = ['up_down', 'down_up']
    up_down_session = [10, 11, 12, 13, 14, 15, 16, 17, 18, 19]
    down_up_session = [10, 11, 12, 13, 19, 24, 25, 26, 27, 28, 20]
    sessions = [up_down_session, down_up_session]

    # calculate mean value
    combined_emg_labelled = readEmgData(subject, version, modes, sessions)
    emg_1_mean_channels, emg_2_mean_channels = calculateEmgMean(combined_emg_labelled)  # the fisrt element for each key is the mean value of the event

    # calculate dtw value
    emg_1_dtw_results = parallelCalculateDtw(emg_1_mean_channels)
    emg_2_dtw_results = parallelCalculateDtw(emg_2_mean_channels)
    emg_dtw_results = {"emg_1_dtw_results": emg_1_dtw_results, "emg_2_dtw_results": emg_2_dtw_results}

    # store dtw values
    Dtw_Storage.saveEmgDtw(subject, emg_dtw_results, version)

    # read dtw values
    # emg_1_dtw_data = Dtw_Storage.readEmgDtw(subject, version, 'emg_1_dtw_results')
    # emg_2_dtw_data = Dtw_Storage.readEmgDtw(subject,  version, 'emg_2_dtw_results')

## calculate average values
# emg_1_average_value = {}
# for dtw_label, dtw_data in emg_1_dtw_results.items():
#     mean_value = sum(emg_1_dtw_results[dtw_label][0]) / len(emg_1_dtw_results[dtw_label][0])
#     emg_1_average_value[f'{dtw_label}'] = mean_value
# emg_2_average_value = {}
# for dtw_label, dtw_data in emg_2_dtw_results.items():
#     mean_value = sum(emg_2_dtw_results[dtw_label][0]) / len(emg_2_dtw_results[dtw_label][0])
#     emg_2_average_value[f'{dtw_label}'] = mean_value



## plotting the cost matrix and warp path of two sequences
# signal_1 = emg_1_mean_channels['emg_LWSA_data'][0]
# signal_2 = emg_1_mean_channels['emg_LWSA_data'][50]
# # plot the two sequences and the paths connecting the mapping points
# fig, ax = plt.subplots(figsize=(140, 100))
# # Remove the border and axes ticks
# fig.patch.set_visible(False)
# ax.axis('off')
#
# a = copy.deepcopy(self_compare_dtw)
# for [point_x, point_y] in a['emg_LWSA_data_dtw'][1][49]:
#     ax.plot([point_x, point_y], [signal_1[point_x], signal_2[point_y]], linewidth=4)
# ax.plot(signal_1, '-ro', label='x', linewidth=2, markersize=5, markerfacecolor='lightcoral', markeredgecolor='lightcoral')
# ax.plot(signal_2, '-bo', label='y', linewidth=2, markersize=5, markerfacecolor='skyblue', markeredgecolor='skyblue')
# ax.set_title("DTW Distance", fontsize=28, fontweight="bold")
# plt.legend()



## using dtw package to calculate DTW distance
# def calculateDtwDistance(gait_event_data, gait_event_label):
#     dtw_result = {}
#     dtw_distance = []
#     warp_path = []
#     for i in range(1, len(gait_event_data)):
#         distance, cost_matrix, accumulated_cost_matrix, path = accelerated_dtw(gait_event_data[0], gait_event_data[i], dist='euclidean')
#         path_pair = list(zip(path[0], path[1]))
#         dtw_distance.append(distance)
#         warp_path.append(path_pair)
#         dtw_result[f"{gait_event_label}_dtw"] = (dtw_distance, warp_path)
#     return dtw_result  # including distance and warp path
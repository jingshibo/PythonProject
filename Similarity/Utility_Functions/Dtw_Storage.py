import os
import json
import numpy as np

## save emg features into json file
def saveEmgDtw(subject, emg_dtw_results, version):
    for emg_index, dtw_data in emg_dtw_results.items():  # save two emg grid data seperately
        # store dtw values
        data_dir = f'D:\Data\Insole_Emg\subject_{subject}\Experiment_{version}\dtw_similarity'
        dtw_file = f'subject_{subject}_Experiment_{version}_{emg_index}.json'
        dtw_path = os.path.join(data_dir, dtw_file)

        # convert all numpy data to list, otherwise it cannot be saved as json
        emg_dtw_json = {}
        for gait_event_label, gait_event_dtw in dtw_data.items():
            emg_dtw_json[gait_event_label] = gait_event_dtw[0]  # only save distance data, abandon warping path data

        # save the lists in a dict to a json file
        with open(dtw_path, 'w') as json_file:
            json.dump(emg_dtw_json, json_file, indent=8)

## read emg features from json file
def readEmgDtw(subject, version, emg_index):
    data_dir = f'D:\Data\Insole_Emg\subject_{subject}\Experiment_{version}\dtw_similarity'
    dtw_file = f'subject_{subject}_Experiment_{version}_{emg_index}.json'
    dtw_path = os.path.join(data_dir, dtw_file)

    # read json file
    with open(dtw_path) as json_file:
        emg_dtw_json = json.load(json_file)

    # return emg features as numpy in a dict
    emg_dtw_data = {}
    for gait_event_label, gait_event_emg in emg_dtw_json.items():
        emg_dtw_data[gait_event_label] = np.asarray(gait_event_emg)
    return emg_dtw_data

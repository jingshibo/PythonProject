import os
import json
import numpy as np

## save emg features into json file
def saveEmgFeatures(subject, emg_features, version, feature_set):
    data_dir = f'D:\Data\Insole_Emg\subject_{subject}\Experiment_{version}\extracted_features'
    feature_file = f'subject_{subject}_Experiment_{version}_emg_feature_set_{feature_set}.json'
    feature_path = os.path.join(data_dir, feature_file)

    # convert all numpy data to list, otherwise it cannot be saved as json
    emg_feature_json = {}
    for gait_event_label, gait_event_emg in emg_features.items():
        emg_feature_json[gait_event_label] = gait_event_emg.tolist()

    # save the lists in a dict to a json file
    with open(feature_path, 'w') as json_file:
        json.dump(emg_feature_json, json_file, indent=8)

## read emg features from json file
def readEmgFeatures(subject, version, feature_set):
    data_dir = f'D:\Data\Insole_Emg\subject_{subject}\Experiment_{version}\extracted_features'
    feature_file = f'subject_{subject}_Experiment_{version}_emg_feature_set_{feature_set}.json'
    feature_path = os.path.join(data_dir, feature_file)

    # read json file
    with open(feature_path) as json_file:
        emg_feature_json = json.load(json_file)

    # return emg features as numpy in a dict
    emg_feature_data = {}
    for gait_event_label, gait_event_emg in emg_feature_json.items():
        emg_feature_data[gait_event_label] = np.asarray(gait_event_emg)
    return emg_feature_data

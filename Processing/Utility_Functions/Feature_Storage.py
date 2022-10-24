import os
import json
import numpy as np

## save emg features into json file
def saveEmgFeatures(subject, emg_features, version, feature_set):
    data_dir = f'D:\Data\Insole_Emg\subject_{subject}\Experiment_{version}\extracted_features'
    feature_file = f'subject_{subject}_Experiment_{version}_emg_feature_set_{feature_set}.json'
    feature_path = os.path.join(data_dir, feature_file)

    with open(feature_path, 'w') as json_file:
        json.dump(emg_features, json_file, indent=8)

## read emg features from json file
def readEmgFeatures(subject, version, feature_set):
    data_dir = f'D:\Data\Insole_Emg\subject_{subject}\Experiment_{version}\extracted_features'
    feature_file = f'subject_{subject}_Experiment_{version}_emg_feature_set_{feature_set}.json'
    feature_path = os.path.join(data_dir, feature_file)

    # read json file
    with open(feature_path) as json_file:
        emg_feature_json = json.load(json_file)

    return emg_feature_json

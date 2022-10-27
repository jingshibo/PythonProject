## import modules
import datetime
import copy
import random
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix
from Processing.Utility_Functions import Feature_Storage, Data_Reshaping
from Models.Utility_Functions import Confusion_Matrix


## load emg feature data
subject = "Shibo"
version = 1  # which experiment data to process
feature_set = 1  # which feature set to use
emg_features = Feature_Storage.readEmgFeatures(subject, version, feature_set)
# reorganize the data structure
repetition_data = []
for gait_event_label, gait_event_features in emg_features.items():
    for repetition_label, repetition_features in gait_event_features.items():
        repetition_data.append(np.array(repetition_features))  # convert 2d list into numpy
    emg_features[gait_event_label] = repetition_data  # convert repetitions from dict into list
    repetition_data = []
# if you want to use CNN model, you need to reshape the data
emg_feature_reshaped = Data_Reshaping.reshapeEmgFeatures(emg_features)


## abandon samples from some modes
emg_feature_data = copy.deepcopy(emg_features)
emg_feature_data['emg_LWLW_features'] = emg_feature_data['emg_LWLW_features'][
int(len(emg_feature_data['emg_LWLW_features']) / 4): int(len(emg_feature_data['emg_LWLW_features']) * 3 / 4)]
emg_feature_data.pop('emg_LW_features', None)
emg_feature_data.pop('emg_SD_features', None)
emg_feature_data.pop('emg_SA_features', None)
#  class name to labels  # according to the alphabetical order
class_all = {'emg_LWLW_features': 0, 'emg_LWSA_features': 1, 'emg_LWSD_features': 2, 'emg_LWSS_features': 3, 'emg_LW_features': 4,
    'emg_SALW_features': 5, 'emg_SASA_features': 6, 'emg_SASS_features': 7, 'emg_SA_features': 8, 'emg_SDLW_features': 9,
    'emg_SDSD_features': 10, 'emg_SDSS_features': 11, 'emg_SD_features': 12, 'emg_SSLW_features': 13, 'emg_SSSA_features': 14,
    'emg_SSSD_features': 15}
class_reduced = {'emg_LWLW_features': 0, 'emg_LWSA_features': 1, 'emg_LWSD_features': 2, 'emg_LWSS_features': 3, 'emg_SALW_features': 4,
    'emg_SASA_features': 5, 'emg_SASS_features': 6, 'emg_SDLW_features': 7, 'emg_SDSD_features': 8, 'emg_SDSS_features': 9,
    'emg_SSLW_features': 10, 'emg_SSSA_features': 11, 'emg_SSSD_features': 12}

class_number = len(emg_feature_data.keys())  # number of classes
window_per_repetition = emg_feature_data['emg_LWLW_features'][0].shape[0]  # how many windows there are for each event repetition


## create k-fold cross validation groups
fold = 5  # 5-fold cross validation
cross_validation_groups = {}  # 5 groups of cross validation set
for i in range(fold):
    train_set = {}  # store train set of all gait events for each group
    test_set = {}  # store test set of all gait events for each group
    for gait_event_label, gait_event_features in copy.deepcopy(emg_feature_data).items():
        # shuffle the list (important)
        random.shuffle(gait_event_features)
        test_set[gait_event_label] = gait_event_features[int(len(gait_event_features) * i / fold): int(len(gait_event_features) * (i+1) / fold)]
        del gait_event_features[int(len(gait_event_features) * i / fold):  int(len(gait_event_features) * (i+1) / fold)]  # remove test set from original set
        train_set[gait_event_label] = gait_event_features
    cross_validation_groups[f"group_{i}"] = {"train_set": train_set, "test_set": test_set}  # a pair of training and test set for one group


## divide into 4 transition groups
transition_grouped = copy.deepcopy(cross_validation_groups)
for group_number, group_value in cross_validation_groups.items():
    for set_type, set_value in group_value.items():
        transition_LW = {}
        transition_SA = {}
        transition_SD = {}
        transition_SS = {}
        transition_group = [["LWLW", "LWSA", "LWSD", "LWSS"], ["SALW", "SASA", "SASS"], ["SDLW", "SDSD", "SDSS"],
            ["SSLW", "SSSA", "SSSD"]]  # 4 groups
        for transition_type, transition_value in set_value.items():
            if any(x in transition_type for x in transition_group[0]):
                transition_LW[transition_type] = transition_value
            elif any(x in transition_type for x in transition_group[1]):
                transition_SA[transition_type] = transition_value
            elif any(x in transition_type for x in transition_group[2]):
                transition_SD[transition_type] = transition_value
            elif any(x in transition_type for x in transition_group[3]):
                transition_SS[transition_type] = transition_value
            transition_grouped[group_number][set_type].pop(transition_type)
        transition_grouped[group_number][set_type]["transition_LW"] = transition_LW
        transition_grouped[group_number][set_type]["transition_SA"] = transition_SA
        transition_grouped[group_number][set_type]["transition_SD"] = transition_SD
        transition_grouped[group_number][set_type]["transition_SS"] = transition_SS


## combine data of all gait events into a single dataset
classification_groups = copy.deepcopy(transition_grouped)
for group_number, group_value in transition_grouped.items():
    for set_type, set_value in group_value.items():
        for transition_type, transition_value in set_value.items():
            # combine the data set within the transition group
            feature_x = []
            feature_y = []
            for gait_event_label, gait_event_features in transition_value.items():
                feature_x.extend(np.concatenate(gait_event_features))
                feature_y.extend([gait_event_label] * len(gait_event_features) * window_per_repetition)
                classification_groups[group_number][set_type][transition_type].pop(gait_event_label)
            # one-hot encode categories (according to the alphabetical order)
            int_y = LabelEncoder().fit_transform(feature_y)
            onehot_y = tf.keras.utils.to_categorical(int_y)
            classification_groups[group_number][set_type][transition_type]['feature_x'] = np.array(feature_x)
            classification_groups[group_number][set_type][transition_type]['feature_int_y'] = int_y
            classification_groups[group_number][set_type][transition_type]['feature_onehot_y'] = onehot_y


## normalize dataset
normalized_groups = copy.deepcopy(classification_groups)
for group_number, group_value in classification_groups.items():
    for set_type, set_value in group_value.items():
        if set_type == 'train_set':
            for transition_type, transition_value in set_value.items():
                normalized_groups[group_number][set_type][transition_type]['feature_norm_x'] = (transition_value['feature_x'] - np.mean(
                    transition_value['feature_x'], axis=0)) / np.std(transition_value['feature_x'], axis=0)
        elif set_type == 'test_set':
            for transition_type, transition_value in set_value.items():
                train_x = normalized_groups[group_number]['train_set'][transition_type]['feature_x']
                normalized_groups[group_number][set_type][transition_type]['feature_norm_x'] = (transition_value['feature_x'] - np.mean(
                    train_x, axis=0)) / np.std(train_x, axis=0)


## shuffle training set
shuffled_groups = copy.deepcopy(normalized_groups)
for group_number, group_value in shuffled_groups.items():
    for set_type, set_value in group_value.items():
        if set_type == 'train_set':
            for transition_type, transition_value in set_value.items():
                data_number = len(transition_value['feature_x'])
                # Shuffles the indices
                idx = np.arange(data_number)
                np.random.shuffle(idx)
                train_idx = idx[: int(data_number)]
                # shuffle the data
                transition_value['feature_x'], transition_value['feature_norm_x'], transition_value['feature_int_y'], transition_value[
                    'feature_onehot_y'] = transition_value['feature_x'][train_idx, :], transition_value['feature_norm_x'][train_idx, :], \
                    transition_value['feature_int_y'][train_idx], transition_value['feature_onehot_y'][train_idx, :]
